from collections import deque
from pathlib import Path
import os
import logging

import numpy as np
from datasets import Dataset
from transformers import PreTrainedTokenizerBase

logger = logging.getLogger(__name__)


class OffsetBasedWordSplitter:
    """Aligns tokenizations at word (segment) boundaries using character offsets.

    Uses character-level offsets from tokenizers to determine word (segment) boundaries
    and assign consistent word (segment) IDs across teacher and student tokenizations.

    Args:
        teacher_tokenizer: Original model's tokenizer.
        student_tokenizer: Target tokenizer.
    """
    def __init__(
        self,
        teacher_tokenizer: PreTrainedTokenizerBase,
        student_tokenizer: PreTrainedTokenizerBase,
    ) -> None:
        self.teacher_tokenizer = teacher_tokenizer
        self.student_tokenizer = student_tokenizer

    def get_word_ids(
        self,
        teacher_offsets: list[tuple[int, int]],
        student_offsets: list[tuple[int, int]],
        teacher_special_tokens_mask: list[bool],
        student_special_tokens_mask: list[bool],
    ) -> tuple[np.ndarray, np.ndarray]:
        """Assign word (segment) IDs to tokens based on character offsets.

        Args:
            teacher_offsets: Character offsets for teacher tokens.
            student_offsets: Character offsets for student tokens.
            teacher_special_tokens_mask: Flags for special tokens in teacher.
            student_special_tokens_mask: Flags for special tokens in student.

        Returns:
            Tuple of (teacher_word_ids, student_word_ids) where -100 indicates
            special tokens and other values are word (segment) indices.
        """

        teacher_word_ids = []
        student_word_ids = []

        teacher_offsets = deque(teacher_offsets)
        student_offsets = deque(student_offsets)

        teacher_special = deque(teacher_special_tokens_mask)
        student_special = deque(student_special_tokens_mask)

        current_end = -1
        current_word_id = -1
        while True:
            if not teacher_offsets:
                for special in student_special: 
                    student_word_ids.append(-100 if special else current_word_id)
                break

            if not student_offsets:
                for special in teacher_special: 
                    teacher_word_ids.append(-100 if special else current_word_id)
                break

            tstart, tstop = teacher_offsets[0]
            sstart, sstop = student_offsets[0]

            # special tokens
            if teacher_special[0]:
                teacher_word_ids.append(-100)
                teacher_offsets.popleft()
                teacher_special.popleft()
                continue
            if student_special[0]:
                student_word_ids.append(-100)
                student_offsets.popleft()
                student_special.popleft()
                continue

            # same word continues
            if tstart < current_end:
                teacher_word_ids.append(current_word_id)
                teacher_offsets.popleft()
                teacher_special.popleft()
                current_end = max(current_end, tstop)
            elif sstart < current_end:
                student_word_ids.append(current_word_id)
                student_offsets.popleft()
                student_special.popleft()
                current_end = max(current_end, sstop)

            # new word
            else:
                current_word_id += 1
                teacher_word_ids.append(current_word_id)
                student_word_ids.append(current_word_id)
                teacher_offsets.popleft()
                student_offsets.popleft()
                teacher_special.popleft()
                student_special.popleft()
                current_end = max(current_end, tstop, sstop)
        
        return np.array(teacher_word_ids), np.array(student_word_ids)


def _tokenize_function(
    examples: dict[str, list],
    teacher_tokenizer: PreTrainedTokenizerBase,
    student_tokenizer: PreTrainedTokenizerBase,
    text_column_name: str,
    word_splitter: OffsetBasedWordSplitter,
    max_length: int,
) -> dict:
    teacher_output = teacher_tokenizer(
        examples[text_column_name],
        return_special_tokens_mask=True,
        return_offsets_mapping=True,
    )

    student_output = student_tokenizer(
        examples[text_column_name],
        return_special_tokens_mask=True,
        return_offsets_mapping=True,
    )

    outputs = {
        'teacher_input_ids': [],
        'teacher_word_ids': [],
        'teacher_attention_mask': [],
        'student_input_ids': [],
        'student_word_ids': [],
        'student_attention_mask': [],
    }

    # splitting into multiple examples if the input is too long
    for doc_idx in range(len(examples[text_column_name])):
        teacher_input_ids = teacher_output['input_ids'][doc_idx]
        teacher_attention_mask = teacher_output['attention_mask'][doc_idx]

        student_input_ids = student_output['input_ids'][doc_idx]
        student_attention_mask = student_output['attention_mask'][doc_idx]

        teacher_word_ids, student_word_ids = word_splitter.get_word_ids(
            teacher_offsets=teacher_output['offset_mapping'][doc_idx],
            student_offsets=student_output['offset_mapping'][doc_idx],
            teacher_special_tokens_mask=teacher_output['special_tokens_mask'][doc_idx],
            student_special_tokens_mask=student_output['special_tokens_mask'][doc_idx],
        )

        if len(teacher_input_ids) > max_length or len(student_input_ids) > max_length:
            max_teacher_word_id = max(teacher_word_ids) \
                if len(teacher_word_ids) <= max_length else max(teacher_word_ids[:max_length+1]) - 1
            max_student_word_id = max(student_word_ids) \
                if len(student_word_ids) <= max_length else max(student_word_ids[:max_length+1]) - 1

            max_common_word_id = min(max_teacher_word_id, max_student_word_id)

            teacher_cut_idx = np.searchsorted(teacher_word_ids, max_common_word_id)
            student_cut_idx = np.searchsorted(student_word_ids, max_common_word_id)

            # cutting potential special tokens should not harm decoder-only models
            # FIXME: what do we do with encoder-only models?
            teacher_input_ids = teacher_input_ids[:teacher_cut_idx]
            teacher_attention_mask = teacher_attention_mask[:teacher_cut_idx]
            teacher_word_ids = teacher_word_ids[:teacher_cut_idx]

            student_input_ids = student_input_ids[:student_cut_idx]
            student_attention_mask = student_attention_mask[:student_cut_idx]
            student_word_ids = student_word_ids[:student_cut_idx]

        outputs['teacher_input_ids'].append(teacher_input_ids)
        outputs['teacher_attention_mask'].append(teacher_attention_mask)
        outputs['teacher_word_ids'].append(teacher_word_ids)

        outputs['student_input_ids'].append(student_input_ids)
        outputs['student_attention_mask'].append(student_attention_mask)
        outputs['student_word_ids'].append(student_word_ids)

        if not len(teacher_word_ids) or not len(student_word_ids) or max(teacher_word_ids) != max(student_word_ids):
            logger.debug(f'doc_idx={doc_idx}')
            logger.debug(repr(examples[text_column_name][doc_idx]))
            continue

    return outputs


def prepare_dataset(
    dataset: Dataset,
    teacher_tokenizer: PreTrainedTokenizerBase,
    student_tokenizer: PreTrainedTokenizerBase,
    max_length: int,
    dataset_output_dir: str | Path,
    text_column_name: str = 'text',
    force_reprocess: bool = False,
    batch_size: int = 1000,
    num_proc: int | None = None,
) -> str:
    """
    Prepare dataset by tokenizing with teacher and student tokenizers.
    
    Args:
        dataset: The dataset to process
        teacher_tokenizer: The teacher tokenizer
        student_tokenizer: The student tokenizer
        max_length: Maximum sequence length
        dataset_output_dir: Base output directory for the dataset
        text_column_name: Name of the column in the dataset containing the text.
        force_reprocess: If True, ignore the dataset and reprocess the dataset
        batch_size: Batch size for dataset processing
        num_proc: Number of processes for parallel processing
        
    Returns:
        Path to the prepared dataset
    """

    dataset_name = dataset.info.dataset_name.replace('/', '_')
    teacher_tokenizer_name = teacher_tokenizer.name_or_path.replace('/', '_')
    student_tokenizer_name = student_tokenizer.name_or_path.replace('/', '_')

    dataset_dirname = f'{dataset_name}__{teacher_tokenizer_name}__{student_tokenizer_name}__{max_length}'
    dataset_output_dir = Path(dataset_output_dir) / dataset_dirname
    if dataset_output_dir.exists() and not force_reprocess:
        logger.info(f"Loading cached dataset from: {dataset_output_dir}")
        return str(dataset_output_dir)
    else:
        dataset_output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Processing dataset and saving to: {dataset_output_dir}")
    word_splitter = OffsetBasedWordSplitter(teacher_tokenizer, student_tokenizer)
    dataset = dataset.map(
        _tokenize_function,
        batched=True,
        batch_size=batch_size,
        num_proc=num_proc or os.cpu_count(),
        fn_kwargs={
            'teacher_tokenizer': teacher_tokenizer,
            'student_tokenizer': student_tokenizer,
            'text_column_name': text_column_name,
            'word_splitter': word_splitter,
            'max_length': max_length,
        }
    )
    
    dataset.save_to_disk(dataset_output_dir)
    logger.info(f"Dataset processed and saved to: {dataset_output_dir}")

    return str(dataset_output_dir)
