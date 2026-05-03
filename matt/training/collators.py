from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass

import torch

from transformers import PreTrainedTokenizerBase
from transformers.data.data_collator import (
    DataCollatorMixin,
    pad_without_fast_tokenizer_warning,
)


@dataclass
class DataCollatorForAIM(DataCollatorMixin):
    """
    Data collator used for Attention Influence Modeling. Inputs are dynamically padded to the maximum
    length of a batch if they are not all of the same length.

    Args:
        teacher_tokenizer ([`PreTrainedTokenizer`] or [`PreTrainedTokenizerFast`]):
            The tokenizer used for encoding the teacher data.
        student_tokenizer ([`PreTrainedTokenizer`] or [`PreTrainedTokenizerFast`]):
            The tokenizer used for encoding the student data.
        pad_to_multiple_of (`int`, *optional*):
            If set will pad the sequence to a multiple of the provided value.
        return_tensors (`str`):
            The type of Tensor to return. Allowable values are "np", "pt" and "tf".
    """

    teacher_tokenizer: PreTrainedTokenizerBase
    student_tokenizer: PreTrainedTokenizerBase
    pad_to_multiple_of: Optional[int] = None
    return_tensors: str = 'pt'

    def __post_init__(self):
        pass

    def torch_call(self, examples: List[Union[List[int], Any, Dict[str, Any]]]) -> Dict[str, Any]:
        teacher_examples = [{
            'input_ids': example['teacher_input_ids'],
            'attention_mask': example['teacher_attention_mask'],
        } for example in examples]

        teacher_batch = pad_without_fast_tokenizer_warning(
            self.teacher_tokenizer,
            teacher_examples,
            return_tensors=self.return_tensors,
            pad_to_multiple_of=self.pad_to_multiple_of,
            padding_side='right',
        )

        student_examples = [{
            'input_ids': example['student_input_ids'],
            'attention_mask': example['student_attention_mask'],
        } for example in examples]

        student_batch = pad_without_fast_tokenizer_warning(
            self.student_tokenizer,
            student_examples,
            return_tensors=self.return_tensors,
            pad_to_multiple_of=self.pad_to_multiple_of,
            padding_side='right',
        )

        student_labels = student_batch["input_ids"].clone()
        if self.student_tokenizer.pad_token_id is not None:
            student_labels[student_labels == self.student_tokenizer.pad_token_id] = -100

        teacher_batch_length = teacher_batch['input_ids'].size(1)
        student_batch_length = student_batch['input_ids'].size(1)

        batch = {
            'teacher_input_ids': teacher_batch['input_ids'],
            'teacher_attention_mask': teacher_batch['attention_mask'].bool(),
            'teacher_word_ids': torch.tensor([
                example['teacher_word_ids'] \
                    + [-100] * (teacher_batch_length - len(example['teacher_word_ids']))
                for example in examples
            ], dtype=torch.int64),
            'student_input_ids': student_batch['input_ids'],
            'student_attention_mask': student_batch['attention_mask'].bool(),
            'student_word_ids': torch.tensor([
                example['student_word_ids'] \
                    + [-100] * (student_batch_length - len(example['student_word_ids']))
                for example in examples
            ], dtype=torch.int64),
            'student_labels': student_labels,
        }

        return batch

    def tf_call(self, examples: List[Union[List[int], Any, Dict[str, Any]]]) -> Dict[str, Any]:
        raise NotImplementedError('TensorFlow is not yet supported for this collator.')

    def numpy_call(self, examples: List[Union[List[int], Any, Dict[str, Any]]]) -> Dict[str, Any]:
        raise NotImplementedError('NumPy is not yet supported for this collator.')
