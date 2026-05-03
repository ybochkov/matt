import torch
import torch.nn as nn


class PartlyFrozenEmbeddings(nn.Module):
    """Embedding layer with selective freezing for transfer learning.

    Splits embeddings into frozen (overlapping with teacher) and active
    (new tokens to learn) components for memory and compute efficiency.

    Args:
        embeddings: Original embedding layer to split.
        frozen_mask: Boolean tensor indicating which tokens to freeze.

    Attributes:
        vocab_size: Total vocabulary size.
        embedding_dim: Embedding dimensionality.
        frozen_embeddings: Embeddings for overlapping tokens (frozen).
        active_embeddings: Embeddings for new tokens (trainable).
    """
    def __init__(self, embeddings: nn.Embedding, frozen_mask: torch.Tensor) -> None:
        super().__init__()
        device = embeddings.weight.device
        frozen_mask = frozen_mask.to(device)

        self.vocab_size = embeddings.num_embeddings
        self.embedding_dim = embeddings.embedding_dim

        num_frozen = frozen_mask.sum().item()
        num_active = self.vocab_size - num_frozen

        self.frozen_embeddings = nn.Embedding(
            num_embeddings=num_frozen,
            embedding_dim=self.embedding_dim,
            padding_idx=embeddings.padding_idx,
            _weight=embeddings.weight.data[frozen_mask],
            device=device,
        )
        self.frozen_embeddings.weight.requires_grad = False

        self.active_embeddings = nn.Embedding(
            num_embeddings=num_active,
            embedding_dim=self.embedding_dim,
            padding_idx=embeddings.padding_idx,
            _weight=embeddings.weight.data[~frozen_mask],
            device=device,
        )

        self.register_buffer("frozen_mask", frozen_mask, persistent=True)
        self.register_buffer("active_mask", ~frozen_mask, persistent=True)

        frozen_indices = torch.where(frozen_mask)[0]
        active_indices = torch.where(~frozen_mask)[0]
        
        original_to_frozen = torch.full((self.vocab_size,), -1, dtype=torch.long, device=device)
        original_to_active = torch.full((self.vocab_size,), -1, dtype=torch.long, device=device)
        
        original_to_frozen[frozen_indices] = torch.arange(len(frozen_indices), device=device)
        original_to_active[active_indices] = torch.arange(len(active_indices), device=device)

        self.register_buffer("original_to_frozen", original_to_frozen, persistent=True)
        self.register_buffer("original_to_active", original_to_active, persistent=True)

    def freeze_all(self) -> None:
        self.frozen_embeddings.weight.requires_grad = False
        self.active_embeddings.weight.requires_grad = False

    def unfreeze_active(self) -> None:
        self.frozen_embeddings.weight.requires_grad = False
        self.active_embeddings.weight.requires_grad = True

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        input_frozen_mask = self.frozen_mask[input_ids]
        input_active_mask = ~input_frozen_mask

        output = torch.zeros(
            input_ids.shape + (self.frozen_embeddings.embedding_dim,), 
            device=input_ids.device, 
            dtype=self.frozen_embeddings.weight.dtype
        )
        
        if input_frozen_mask.any():
            frozen_input_ids = input_ids[input_frozen_mask]
            frozen_new_ids = self.original_to_frozen[frozen_input_ids]
            frozen_embeddings = self.frozen_embeddings(frozen_new_ids)
            output[input_frozen_mask] = frozen_embeddings
        
        if input_active_mask.any():
            active_input_ids = input_ids[input_active_mask]
            active_new_ids = self.original_to_active[active_input_ids]
            active_embeddings = self.active_embeddings(active_new_ids)
            output[input_active_mask] = active_embeddings

        return output

    def to_embeddings(self) -> nn.Embedding:
        """Convert back to a standard nn.Embedding layer.

        Returns:
            Standard embedding layer with combined frozen and active weights.
        """
        embeddings = nn.Embedding(
            self.vocab_size,
            self.embedding_dim,
            device=self.frozen_embeddings.weight.device,
        )
        
        with torch.no_grad():
            indices = torch.arange(self.vocab_size, device=self.frozen_embeddings.weight.device)
            embeddings.weight.data.copy_(self.forward(indices).detach().clone())

        return embeddings

    # def state_dict(self, *args, **kwargs):
    #     embeddings = self.to_embeddings()
    #     return embeddings.state_dict(*args, **kwargs)
