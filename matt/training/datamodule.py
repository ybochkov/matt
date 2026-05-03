from torch.utils.data import DataLoader
import lightning.pytorch as pl

from transformers import AutoTokenizer
from datasets import load_from_disk, Dataset

from ..config import MATTConfig
from .collators import DataCollatorForAIM


def load_prepared_dataset(path: str) -> Dataset:
    """
    Loads a prepared dataset from a file.

    We expect the dataset to contain the following columns:
      - "teacher_input_ids": list of input ids
      - "teacher_attention_mask": list of attention masks
      - "teacher_word_ids": list of word ids (of the same length as "teacher_input_ids")
      - "student_input_ids": list of input ids
      - "student_attention_mask": list of attention masks
      - "student_word_ids": list of word ids (of the same length as "student_input_ids")

    Args:
        path (str): The path to the prepared dataset file.

    Returns:
        Dataset: The prepared dataset.
    """
    return load_from_disk(path)


class AIMDataModule(pl.LightningDataModule):
    def __init__(
        self,
        config: MATTConfig,
        dataset_path: str,
        teacher_tokenizer_path: str,
        student_tokenizer_path: str,
    ) -> None:
        super().__init__()
        self.config = config
        self.dataset_path = dataset_path

        self.teacher_tokenizer_path = teacher_tokenizer_path
        self.student_tokenizer_path = student_tokenizer_path

        self.train_dataset: Dataset | None = None
        self.collator: DataCollatorForAIM | None = None

    def prepare_data(self) -> None:
        pass

    def setup(self, stage: str | None = None) -> None:
        self.train_dataset = load_prepared_dataset(self.dataset_path)

        teacher_tokenizer = AutoTokenizer.from_pretrained(self.teacher_tokenizer_path)
        student_tokenizer = AutoTokenizer.from_pretrained(self.student_tokenizer_path)

        # if pad token is not set, set it to eos token
        if teacher_tokenizer.pad_token_id is None:
            teacher_tokenizer.pad_token_id = teacher_tokenizer.eos_token_id
        if student_tokenizer.pad_token_id is None:
            student_tokenizer.pad_token_id = student_tokenizer.eos_token_id

        self.collator = DataCollatorForAIM(
            teacher_tokenizer=teacher_tokenizer,
            student_tokenizer=student_tokenizer,
        )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.config.batch_size,
            num_workers=self.config.num_workers,
            collate_fn=self.collator,
            shuffle=True,
        )
