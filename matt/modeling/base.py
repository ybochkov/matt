from typing import Optional, Iterable
from dataclasses import dataclass

import torch
import torch.nn as nn

from transformers.modeling_outputs import ModelOutput
from transformers.cache_utils import DynamicCache
from transformers import PreTrainedModel, AutoModelForCausalLM


@dataclass
class AIMOutput(ModelOutput):
    """Output from AIM-adapted models containing attention and value states.

    Attributes:
        aim_loss: Optional AIM loss value (computed externally in the training task).
        ntp_loss: Optional NTP loss value (only present when with_ntp=True).
        attentions: Attention weights from the last included layer.
        value_states: Value states from the last included layer's key-value cache.
    """
    aim_loss: Optional[torch.Tensor] = None
    ntp_loss: Optional[torch.Tensor] = None
    attentions: Optional[tuple[torch.FloatTensor, ...]] = None
    value_states: Optional[tuple[torch.FloatTensor, ...]] = None


class PreTrainedModelForAIM(nn.Module):
    """Base class for adapting transformer models to AIM training.

    Wraps a pretrained model to extract attention patterns and value states
    needed for Attention Influence Modeling. Truncates the model to only the
    first n_layers for efficiency during embedding training (unless with_ntp=True,
    in which case the full model is kept to enable next-token prediction loss).

    Args:
        model: Full pretrained transformer model.
        n_layers: Number of layers to keep (typically 35-50% of the total).
        frozen_embeddings_mask: Optional mask for partial embedding freezing.
            Not compatible with with_ntp=True.
        with_ntp: If True, keep the full model (all layers) and compute NTP loss
            alongside AIM loss. Required for models with untied input/output
            embeddings (e.g. Qwen3).
    """
    def __init__(
        self,
        model: PreTrainedModel,
        n_layers: int = 1,
        frozen_embeddings_mask: Optional[torch.BoolTensor] = None,
        with_ntp: bool = False,
    ) -> None:
        super().__init__()

        self.n_layers = n_layers
        self.frozen_embeddings_mask = frozen_embeddings_mask
        self.with_ntp = with_ntp

        if with_ntp:
            # Keep full CausalLM model to produce NTP loss via labels
            self.model = model
        else:
            self.model = self.get_base_model(model)
            layers = self.get_layers()[:n_layers]
            self.set_layers(layers)

        self.config = model.config.get_text_config()

        if frozen_embeddings_mask is not None:
            if with_ntp:
                raise NotImplementedError("frozen_embeddings_mask is not compatible with with_ntp=True")
            partly_frozen_embeddings = self.create_partly_frozen_embeddings(frozen_embeddings_mask)
            self.model.set_input_embeddings(partly_frozen_embeddings)

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        n_layers: int,
        frozen_embeddings_mask: Optional[torch.BoolTensor] = None,
        with_ntp: bool = False,
        **kwargs,
    ):
        """Load a pretrained model and adapt it for AIM training.

        Args:
            pretrained_model_name_or_path: HuggingFace model ID or local path.
            n_layers: Number of transformer layers to keep.
            frozen_embeddings_mask: Optional mask for partially frozen embeddings.
                Not compatible with with_ntp=True.
            with_ntp: If True, keep full model and enable NTP loss computation.
            **kwargs: Additional arguments for model loading (e.g. device_map).

        Returns:
            Adapted model instance ready for AIM training.
        """
        whole_model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path,
            torch_dtype=torch.bfloat16,
            # IMPORTANT: Must use eager attention to extract attention weights and value states.
            # Flash Attention and other optimized implementations don't return attention weights,
            # which are required for AIM computation.
            attn_implementation='eager',
            **kwargs,
        )
        return cls(
            whole_model,
            n_layers=n_layers,
            frozen_embeddings_mask=frozen_embeddings_mask,
            with_ntp=with_ntp,
        )

    def get_base_model(self, model: PreTrainedModel) -> PreTrainedModel:
        raise NotImplementedError("Override this method for your architecture.")

    def get_layers(self) -> Iterable[nn.Module]:
        raise NotImplementedError("Override this method for your architecture.")

    def set_layers(self, layers: Iterable[nn.Module]) -> None:
        raise NotImplementedError("Override this method for your architecture.")

    def get_input_embeddings(self) -> nn.Embedding:
        return self.model.get_input_embeddings()

    def set_input_embeddings(self, embeddings: nn.Embedding) -> None:
        self.model.set_input_embeddings(embeddings)

    def get_output_embeddings(self) -> nn.Linear:
        if not self.with_ntp:
            raise ValueError("Output embeddings are only available when with_ntp=True")
        return self.model.get_output_embeddings()

    def set_output_embeddings(self, embeddings: nn.Linear) -> None:
        if not self.with_ntp:
            raise ValueError("Output embeddings are only available when with_ntp=True")
        self.model.set_output_embeddings(embeddings)

    def create_partly_frozen_embeddings(self, frozen_mask: torch.Tensor) -> nn.Embedding:
        raise NotImplementedError("Override this method for your architecture.")

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        ntp_labels: Optional[torch.Tensor] = None,
    ) -> AIMOutput:

        past_key_values = DynamicCache()

        model_kwargs = dict(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            output_attentions=True,
            use_cache=True,
        )

        if self.with_ntp:
            model_kwargs['labels'] = ntp_labels

        output = self.model(**model_kwargs)

        attn_weights = output.attentions[self.n_layers - 1]
        value_states = past_key_values[self.n_layers - 1][1]

        return AIMOutput(
            ntp_loss=output.loss if self.with_ntp else None,
            attentions=attn_weights,
            value_states=value_states,
        )
