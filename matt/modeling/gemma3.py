from typing import Iterable

import torch
import torch.nn as nn

from transformers.models.gemma3.modeling_gemma3 import (
    Gemma3ForCausalLM,
    Gemma3TextModel,
)

from .embeddings import PartlyFrozenEmbeddings
from .base import PreTrainedModelForAIM


class Gemma3TextScaledPartlyFrozenWordEmbedding(PartlyFrozenEmbeddings):
    def __init__(self, embeddings: nn.Embedding, frozen_mask: torch.Tensor, embed_scale: float = 1.0) -> None:
        super().__init__(embeddings, frozen_mask)
        self.register_buffer("embed_scale", torch.tensor(embed_scale), persistent=False)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return super().forward(input_ids) * self.embed_scale.to(self.frozen_embeddings.weight.dtype)


class Gemma3ForAIM(PreTrainedModelForAIM):
    """Gemma 3 model adapter for Attention Influence Modeling.

    Adapts Gemma 3 models for AIM-based tokenizer transfer training.
    Handles Gemma's scaled embeddings correctly.
    """

    def get_base_model(self, model: Gemma3ForCausalLM) -> Gemma3TextModel:
        return model.language_model

    def get_layers(self) -> Iterable[nn.Module]:
        return self.model.layers

    def set_layers(self, layers: Iterable[nn.Module]) -> None:
        self.model.layers = nn.ModuleList(layers)

    def create_partly_frozen_embeddings(self, frozen_mask: torch.Tensor) -> PartlyFrozenEmbeddings:
        return Gemma3TextScaledPartlyFrozenWordEmbedding(
            embeddings=self.model.embed_tokens,
            frozen_mask=frozen_mask,
            embed_scale=self.model.embed_tokens.embed_scale.item(),
        )
