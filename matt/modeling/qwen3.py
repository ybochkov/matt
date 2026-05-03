from typing import Iterable

import torch
import torch.nn as nn

from transformers.models.qwen3.modeling_qwen3 import (
    Qwen3ForCausalLM,
    Qwen3Model,
)

from .embeddings import PartlyFrozenEmbeddings
from .base import PreTrainedModelForAIM


class Qwen3ForAIM(PreTrainedModelForAIM):
    """Qwen 3 model adapter for Attention Influence Modeling.

    Adapts Qwen 3 models for AIM-based tokenizer transfer training.
    """

    def get_base_model(self, model: Qwen3ForCausalLM) -> Qwen3Model:
        return model.model

    def get_layers(self) -> Iterable[nn.Module]:
        return self.model.layers

    def set_layers(self, layers: Iterable[nn.Module]) -> None:
        self.model.layers = nn.ModuleList(layers)

    def create_partly_frozen_embeddings(self, frozen_mask: torch.Tensor) -> nn.Embedding:
        return PartlyFrozenEmbeddings(
            embeddings=self.model.embed_tokens,
            frozen_mask=frozen_mask,
        )
