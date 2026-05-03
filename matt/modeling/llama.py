from typing import Iterable

import torch
import torch.nn as nn

from transformers.models.llama.modeling_llama import (
    LlamaForCausalLM,
    LlamaModel,
)

from .embeddings import PartlyFrozenEmbeddings
from .base import PreTrainedModelForAIM


class LlamaForAIM(PreTrainedModelForAIM):
    """Llama model adapter for Attention Influence Modeling.

    Adapts Llama models for AIM-based tokenizer transfer training.
    """

    def get_base_model(self, model: LlamaForCausalLM) -> LlamaModel:
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
