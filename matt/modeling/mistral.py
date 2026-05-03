from typing import Iterable

import torch
import torch.nn as nn

from transformers.models.mistral.modeling_mistral import (
    MistralForCausalLM,
    MistralModel,
)

from .embeddings import PartlyFrozenEmbeddings
from .base import PreTrainedModelForAIM


class MistralForAIM(PreTrainedModelForAIM):
    """Mistral for Attention Influence Modeling (AIM)."""

    def get_base_model(self, model: MistralForCausalLM) -> MistralModel:
        return model.model

    def get_layers(self) -> Iterable[nn.Module]:
        return self.model.layers

    def set_layers(self, layers: Iterable[nn.Module]) -> None:
        self.model.layers = nn.ModuleList(layers)

    def create_partly_frozen_embeddings(self, frozen_mask: torch.Tensor) -> PartlyFrozenEmbeddings:
        return PartlyFrozenEmbeddings(
            embeddings=self.model.embed_tokens,
            frozen_mask=frozen_mask,
        )
