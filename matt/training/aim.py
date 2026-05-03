from dataclasses import asdict
import logging

import torch
import torch.nn as nn

import lightning.pytorch as pl

from transformers import get_scheduler

from ..config import MATTConfig
from ..modeling.base import PreTrainedModelForAIM
from .aim_impl import get_aim_impl

logger = logging.getLogger(__name__)


def _freeze_parameters(module: nn.Module) -> None:
    for param in module.parameters():
        param.requires_grad = False


def _unfreeze_parameters(module: nn.Module) -> None:
    for param in module.parameters():
        param.requires_grad = True


# copied from `src/transformers/models/gemma3/modeling_gemma3.py`
def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


class AttentionInfluenceModelingTask(pl.LightningModule):
    def __init__(
        self,
        config: MATTConfig,
        teacher_model: PreTrainedModelForAIM,
        student_model: PreTrainedModelForAIM,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(asdict(config))

        self.config = config

        self.teacher_model = teacher_model
        self.student_model = student_model

        num_attention_heads = self.teacher_model.config.num_attention_heads
        num_key_value_heads = getattr(
            self.teacher_model.config,
            'num_key_value_heads',
            num_attention_heads,
        )
        self.num_key_value_groups = num_attention_heads // num_key_value_heads

        self.aim_impl = get_aim_impl(config.aim_impl)

        if config.loss == "mse":
            self.loss_fn = nn.MSELoss()
        elif config.loss == "cosine":
            self.loss_fn = nn.CosineEmbeddingLoss()
        else:
            raise ValueError(f"Unknown loss: {config.loss}")

        _freeze_parameters(self.teacher_model)
        _freeze_parameters(self.student_model)

        _unfreeze_parameters(self.student_model.get_input_embeddings())
        if config.with_ntp:
            _unfreeze_parameters(self.student_model.get_output_embeddings())
        elif self.student_model.frozen_embeddings_mask is not None:
            self.student_model.get_input_embeddings().freeze_all()
            self.student_model.get_input_embeddings().unfreeze_active()

    def _create_dummy_loss_with_grad(self) -> torch.Tensor:
        """
        Create a dummy loss tensor that has a gradient computation graph
        but doesn't affect training. This is used when all tokens in a batch
        are frozen (no new tokens to learn from).
        
        Returns:
            A scalar tensor with grad_fn that evaluates to 0.0
        """
        # Get a parameter from the student model that requires grad
        # This ensures we have a proper gradient computation graph
        student_embeddings = self.student_model.get_input_embeddings()

        # if using PartlyFrozenEmbeddings
        if self.student_model.frozen_embeddings_mask is not None:
            dummy_loss = 0.0 * student_embeddings.active_embeddings.weight.sum()
        else:
            dummy_loss = 0.0 * student_embeddings.weight.sum()

        if self.config.with_ntp:
            dummy_loss = dummy_loss + 0.0 * self.student_model.get_output_embeddings().weight.sum()

        return dummy_loss

    def common_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        teacher_input_ids = batch["teacher_input_ids"]
        teacher_attention_mask = batch["teacher_attention_mask"]
        teacher_word_ids = batch["teacher_word_ids"]

        student_input_ids = batch["student_input_ids"]
        student_attention_mask = batch["student_attention_mask"]
        student_word_ids = batch["student_word_ids"]
        student_labels = batch["student_labels"]

        teacher_output = self.teacher_model(
            input_ids=teacher_input_ids,
            attention_mask=teacher_attention_mask,
        )
        student_output = self.student_model(
            input_ids=student_input_ids,
            attention_mask=student_attention_mask,
            ntp_labels=student_labels,
        )

        if self.config.with_ntp:
            ntp_loss = student_output.ntp_loss
        else:
            ntp_loss = self._create_dummy_loss_with_grad()

        teacher_word_states, student_word_states = self.aim_impl(
            teacher_attn_weights=teacher_output.attentions,
            teacher_value_states=repeat_kv(
                teacher_output.value_states,
                self.num_key_value_groups,
            ),
            teacher_word_ids=teacher_word_ids,
            student_attn_weights=student_output.attentions,
            student_value_states=repeat_kv(
                student_output.value_states,
                self.num_key_value_groups,
            ),
            student_word_ids=student_word_ids,
        )

        # Handle edge cases where no gradients are available
        if teacher_word_states.size(0) != student_word_states.size(0):
            logger.error(
                f'Word states size mismatch at batch_idx={batch_idx}: '
                f'teacher_word_states.size(0)={teacher_word_states.size(0)}, '
                f'student_word_states.size(0)={student_word_states.size(0)}'
            )
            logger.debug(f'teacher_word_ids: {teacher_word_ids}')
            logger.debug(f'student_word_ids: {student_word_ids}')
            logger.debug(f'teacher_input_ids: {teacher_input_ids}')
            logger.debug(f'student_input_ids: {student_input_ids}')
            return self._create_dummy_loss_with_grad()

        if self.config.loss == "mse":
            aim_loss = self.loss_fn(teacher_word_states, student_word_states)
        elif self.config.loss == "cosine":
            target = torch.ones(teacher_word_states.size(0), device=teacher_word_states.device)
            aim_loss = self.loss_fn(teacher_word_states, student_word_states, target)
        else:
            raise ValueError(f"Unknown loss: {self.config.loss}")

        # Handle edge cases where no gradients are available (all frozen tokens in batch)
        if not aim_loss.requires_grad:
            return self._create_dummy_loss_with_grad()

        self.log('aim_loss', aim_loss)

        if self.config.with_ntp:
            # Dynamic scaling keeps AIM and NTP losses at equal magnitude.
            # Reference: https://github.com/konstantinjdobler/token-distillation/blob/main/paper/token_distillation.py
            # `.item()` detaches from the computation graph intentionally.
            scaling_factor = aim_loss.item() / ntp_loss.item()
            scaled_ntp_loss = scaling_factor * ntp_loss
            loss = aim_loss + scaled_ntp_loss

            self.log('ntp_loss', ntp_loss)
            self.log('scaled_ntp_loss', scaled_ntp_loss)
            self.log('ntp_scaling_factor', scaling_factor)
        else:
            loss = aim_loss

        self.log('loss', loss)

        return loss

    def training_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        return self.common_step(batch, batch_idx)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.config.lr,
        )

        if self.config.num_warmup_steps > 0 or self.config.num_decay_steps > 0:
            scheduler = get_scheduler(
                'warmup_stable_decay',
                optimizer=optimizer,
                num_warmup_steps=self.config.num_warmup_steps,
                num_training_steps=self.trainer.estimated_stepping_batches,
                scheduler_specific_kwargs=dict(
                    num_decay_steps=self.config.num_decay_steps,
                ),
            )
            return {
                'optimizer': optimizer,
                'lr_scheduler': {
                    'scheduler': scheduler,
                    'interval': 'step',
                },
            }

        return optimizer
