from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union
import logging
import tqdm

from datasets import Dataset
from transformers import PreTrainedModel, PreTrainedTokenizerBase

import torch
import lightning.pytorch as pl

from deepfocus.focus import get_overlapping_tokens
from matt.modeling.base import PreTrainedModelForAIM

from .config import MATTConfig
from .preparation import prepare_dataset
from .training.callbacks import PeakVRAMMonitorCallback, FLOPSMonitorCallback, TotalTrainingTimeCallback
from .training.datamodule import AIMDataModule
from .training.aim import AttentionInfluenceModelingTask

logger = logging.getLogger(__name__)


@dataclass
class MATTOutput:
    """Learned embeddings returned by MATT.train().

    Attributes:
        input_embeddings: Learned input embedding weights,
            shape (student_vocab_size, embedding_dim). Always present.
        output_embeddings: Learned LM head weights,
            shape (student_vocab_size, embedding_dim). Only set when
            training was run with with_ntp=True; None otherwise.
    """
    input_embeddings: torch.Tensor
    output_embeddings: Optional[torch.Tensor]

    def apply_to(self, model: PreTrainedModel) -> None:
        """Copy the learned embeddings into a full pretrained model in-place.

        Call this after freeing the training models from GPU memory and loading
        the full model:

            result = matt.train(teacher_model, student_model)
            del teacher_model, student_model
            torch.cuda.empty_cache()
            full_model = AutoModelForCausalLM.from_pretrained(...)
            result.apply_to(full_model)

        Args:
            model: Full pretrained model whose embeddings will be replaced.
        """
        model.get_input_embeddings().weight.data.copy_(self.input_embeddings)
        if self.output_embeddings is not None:
            model.get_output_embeddings().weight.data.copy_(self.output_embeddings)


class MATT:
    """Model-Aware Tokenizer Transfer using Attention Influence Modeling.

    MATT is a self-distillation method designed to adapt a pre-trained language model to a new tokenizer.
    It utilizes the rich signal encoded in the attention layers, and by distilling how each word (regardless of the tokenization)
    influences the next token's representation, we can effectively train a model to work with a new tokenizer.

    At the same time, distillation based on the attention layers allows us to train only the input embeddings.
    This is enough for models with tied embeddings, but requires an additional Next Token Prediction (NTP) loss component
    for models with separate input embeddings and LM head.

    Args:
        config: Configuration for training and logging.
        teacher_tokenizer: Tokenizer used by the original (teacher) model.
        student_tokenizer: New tokenizer to adapt the model to.

    Attributes:
        config: Training configuration.
        teacher_tokenizer: Original model's tokenizer.
        student_tokenizer: Target tokenizer for adaptation.
        dataset_path: Path to preprocessed dataset (set after prepare_dataset).

    Example:
        >>> from transformers import AutoTokenizer
        >>> from matt import MATT, MATTConfig
        >>>
        >>> teacher_tok = AutoTokenizer.from_pretrained("google/gemma-2-2b")
        >>> student_tok = AutoTokenizer.from_pretrained("meta-llama/Llama-3-8B")
        >>>
        >>> matt = MATT(
        ...     config=MATTConfig(max_steps=10000, batch_size=8),
        ...     teacher_tokenizer=teacher_tok,
        ...     student_tokenizer=student_tok
        ... )
    """
    def __init__(
        self,
        config: MATTConfig,
        teacher_tokenizer: PreTrainedTokenizerBase,
        student_tokenizer: PreTrainedTokenizerBase
    ) -> None:
        self.config = config

        self.teacher_tokenizer = teacher_tokenizer
        self.student_tokenizer = student_tokenizer

        self.dataset_path: str | None = None

    def prepare_dataset(
        self,
        dataset: Dataset,
        max_length: int,
        dataset_output_dir: str | Path = None,
        text_column_name: str = 'text',
        force_reprocess: bool = False,
        batch_size: int = 1000,
        num_proc: int | None = None,
    ) -> None:
        """Preprocess dataset for AIM training.

        Tokenizes texts with both teacher and student tokenizers, aligns them at
        the word level, and caches results to disk for efficient training.

        Args:
            dataset: HuggingFace dataset with a 'text' column.
            max_length: Maximum sequence length (longer sequences are truncated).
            dataset_output_dir: Directory for cached dataset. If None, uses
                config.dataset_output_dir.
            text_column_name: Name of the column in the dataset containing the text.
            force_reprocess: If True, reprocess even if cache exists.
            batch_size: Batch size for dataset processing.
            num_proc: Number of processes for parallel processing. If None,
                uses all CPU cores.

        Note:
            Sets self.dataset_path to the cache directory. This path is required
            before calling train().
        """
        self.dataset_path = prepare_dataset(
            dataset=dataset,
            teacher_tokenizer=self.teacher_tokenizer,
            student_tokenizer=self.student_tokenizer,
            max_length=max_length,
            dataset_output_dir=dataset_output_dir or self.config.dataset_output_dir,
            text_column_name=text_column_name,
            force_reprocess=force_reprocess,
            batch_size=batch_size,
            num_proc=num_proc,
        )

    def prepare_frozen_embeddings_mask(self) -> torch.BoolTensor:
        """Create mask identifying overlapping tokens between tokenizers.

        Uses deepfocus to find tokens that have exact matches between teacher
        and student vocabularies. These tokens can have their embeddings frozen
        during training for efficiency.

        Returns:
            Boolean tensor of shape (student_vocab_size,) where True indicates
            tokens that overlap with the teacher tokenizer and can be frozen.

        Note:
            This is optional. Pass the result to student model's from_pretrained
            via frozen_embeddings_mask parameter to enable partial freezing.
        """
        overlapping_tokens, _ = get_overlapping_tokens(
            source_tokenizer=self.teacher_tokenizer,
            target_tokenizer=self.student_tokenizer,
            match_symbols=False,
            exact_match_all=True,
            fuzzy_match_all=False,
        )

        # using .vocab_size returns incorrect value
        vocab_size = len(self.student_tokenizer)
        original_embeddings_mask = torch.zeros(
            vocab_size,
            dtype=torch.bool,
        )
        for token, overlap in tqdm.tqdm(list(overlapping_tokens.items()),
                                        desc="Getting original embeddings mask"):
            if overlap.target.id >= vocab_size:
                logger.warning(f'Student token id {overlap.target.id} is greater than student tokenizer vocab size: {overlap.target}')
                continue
            original_embeddings_mask[overlap.target.id] = True

        # Log overlap statistics
        num_overlapping = original_embeddings_mask.sum().item()
        overlap_ratio = num_overlapping / vocab_size
        logger.info(f"Token overlap: {num_overlapping}/{vocab_size} ({overlap_ratio:.2%})")

        return original_embeddings_mask

    def train(
        self,
        teacher_model: PreTrainedModelForAIM,
        student_model: PreTrainedModelForAIM,
        **kwargs
    ) -> MATTOutput:
        """Train student embeddings via Attention Influence Modeling.

        Runs PyTorch Lightning training to learn embeddings for the student
        tokenizer by matching word-level attention influence patterns from the
        teacher model.

        Args:
            teacher_model: Truncated teacher model (created via XXXForAIM.from_pretrained).
            student_model: Student model with trainable embeddings. Must be created
                with with_ntp=True when config.with_ntp=True.
            **kwargs: Additional arguments passed to PyTorch Lightning Trainer.

        Returns:
            MATTOutput with learned input (and optionally output) embeddings.
            Call result.apply_to(full_model) to inject them after freeing GPU memory.

        Raises:
            RuntimeError: If prepare_dataset() was not called first.
            TypeError: If teacher_model or student_model are not PreTrainedModelForAIM instances.
            ValueError: If config.with_ntp and student_model.with_ntp are inconsistent.
        """
        # Validate dataset is prepared
        if self.dataset_path is None:
            raise RuntimeError(
                "No dataset prepared. Call prepare_dataset() before train(). "
                "Example: matt.prepare_dataset(dataset, max_length=512)"
            )

        # Validate model types
        if not isinstance(teacher_model, PreTrainedModelForAIM):
            raise TypeError(
                f"teacher_model must be a PreTrainedModelForAIM instance, "
                f"got {type(teacher_model).__name__}. "
                f"Use an existing adapter or create one by inheriting from PreTrainedModelForAIM."
            )
        if not isinstance(student_model, PreTrainedModelForAIM):
            raise TypeError(
                f"student_model must be a PreTrainedModelForAIM instance, "
                f"got {type(student_model).__name__}. "
                f"Use an existing adapter or create one by inheriting from PreTrainedModelForAIM."
            )

        # Validate NTP consistency
        if self.config.with_ntp and not student_model.with_ntp:
            raise ValueError(
                "config.with_ntp=True but student_model was created with with_ntp=False. "
                "Pass with_ntp=True to the student model's from_pretrained() call."
            )
        if not self.config.with_ntp and student_model.with_ntp:
            raise ValueError(
                "student_model was created with with_ntp=True but config.with_ntp=False. "
                "Set with_ntp=True in MATTConfig to enable NTP loss."
            )

        datamodule = AIMDataModule(
            config=self.config,
            dataset_path=self.dataset_path,
            teacher_tokenizer_path=self.teacher_tokenizer.name_or_path,
            student_tokenizer_path=self.student_tokenizer.name_or_path,
        )

        task = AttentionInfluenceModelingTask(
            config=self.config,
            teacher_model=teacher_model,
            student_model=student_model,
        )

        wandb_logger = None
        if self.config.use_wandb:
            try:
                from lightning.pytorch.loggers import WandbLogger
            except ImportError as e:
                raise ImportError(
                    "W&B logging requested but not available. "
                    "Install the extra: 'pip install matt[wandb]'"
                ) from e
            wandb_logger = WandbLogger(
                project=self.config.wandb_project,
                save_dir=self.config.wandb_output_dir,
            )

        peak_vram_monitor = PeakVRAMMonitorCallback()
        flops_monitor = FLOPSMonitorCallback()
        total_training_time_monitor = TotalTrainingTimeCallback()

        teacher_model_type = getattr(
            teacher_model.config,
            'model_type',
            teacher_model.__class__.__name__,
        )
        student_model_type = getattr(
            student_model.config,
            'model_type',
            student_model.__class__.__name__,
        )
        filename = f'{teacher_model_type}_{student_model_type}_step{{step}}_loss{{loss:.3f}}'
        
        checkpoint_callback = pl.callbacks.ModelCheckpoint(
            dirpath=self.config.checkpoint_output_dir,
            filename=filename,
            save_last=True,
            save_top_k=self.config.save_top_k,
            auto_insert_metric_name=False,
            every_n_train_steps=self.config.save_every_n_steps,
        )
        learning_rate_monitor = pl.callbacks.LearningRateMonitor(logging_interval='step')

        # TODO: add grad_norm logging 
        # https://lightning.ai/docs/pytorch/stable/debug/debugging_intermediate.html#look-out-for-exploding-gradients
        trainer = pl.Trainer(
            logger=wandb_logger,
            callbacks=[
                checkpoint_callback,
                learning_rate_monitor,
                peak_vram_monitor,
                flops_monitor,
                total_training_time_monitor,
            ],
            max_epochs=self.config.max_epochs,
            max_steps=self.config.max_steps,
            log_every_n_steps=self.config.log_every_n_steps,
            accelerator=self.config.accelerator,
            devices=self.config.devices,
            strategy=self.config.strategy,
            precision=self.config.precision,
            gradient_clip_val=self.config.gradient_clip_val,
            gradient_clip_algorithm=self.config.gradient_clip_algorithm,
            accumulate_grad_batches=self.config.accumulate_grad_batches,
            enable_model_summary=True,
            **kwargs,
        )
        trainer.fit(task, datamodule=datamodule)

        embeddings_module = student_model.get_input_embeddings() \
            if student_model.frozen_embeddings_mask is None \
            else student_model.get_input_embeddings().to_embeddings()

        return MATTOutput(
            input_embeddings=embeddings_module.weight.data.clone().detach().cpu(),
            output_embeddings=student_model.get_output_embeddings().weight.data.clone().detach().cpu()
                if self.config.with_ntp else None,
        )

