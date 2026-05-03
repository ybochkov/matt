from typing import Iterable

import torch
import torch.nn as nn

from transformers.models.gpt_neox.modeling_gpt_neox import (
    GPTNeoXForCausalLM,
    GPTNeoXModel,
)

from .embeddings import PartlyFrozenEmbeddings
from .base import PreTrainedModelForAIM


class GPTNeoXForAIM(PreTrainedModelForAIM):
    """GPT-NeoX model adapter for Attention Influence Modeling.

    Adapts GPT-NeoX models (e.g., Pythia) for AIM-based
    tokenizer transfer training.
    """
    
    def get_base_model(self, model: GPTNeoXForCausalLM) -> GPTNeoXModel:
        return model.gpt_neox

    def get_layers(self) -> Iterable[nn.Module]:
        return self.model.layers
    
    def set_layers(self, layers: Iterable[nn.Module]) -> None:
        self.model.layers = nn.ModuleList(layers)
    
    def create_partly_frozen_embeddings(self, frozen_mask: torch.Tensor) -> nn.Embedding:
        return PartlyFrozenEmbeddings(
            embeddings=self.model.embed_in,
            frozen_mask=frozen_mask,
        )
