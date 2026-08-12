import unittest

import torch
import torch.nn as nn

from matt.modeling.embeddings import PartlyFrozenEmbeddings
from matt.modeling.gemma3 import Gemma3TextScaledPartlyFrozenWordEmbedding


class PartlyFrozenEmbeddingsTest(unittest.TestCase):
    def test_to_embeddings_reconstructs_original_weight_and_metadata(self) -> None:
        embeddings = nn.Embedding(6, 3, padding_idx=0, dtype=torch.float64)
        original_weight = embeddings.weight.detach().clone()
        frozen_mask = torch.tensor([True, False, True, False, False, True])

        exported = PartlyFrozenEmbeddings(
            embeddings,
            frozen_mask,
        ).to_embeddings()

        torch.testing.assert_close(exported.weight, original_weight)
        self.assertEqual(exported.padding_idx, embeddings.padding_idx)
        self.assertEqual(exported.weight.dtype, embeddings.weight.dtype)

    def test_gemma_export_does_not_persist_runtime_embedding_scale(self) -> None:
        embeddings = nn.Embedding(6, 3)
        original_weight = embeddings.weight.detach().clone()
        frozen_mask = torch.tensor([True, False, True, False, False, True])
        embed_scale = 5.0
        partly_frozen = Gemma3TextScaledPartlyFrozenWordEmbedding(
            embeddings,
            frozen_mask,
            embed_scale=embed_scale,
        )
        token_ids = torch.arange(embeddings.num_embeddings)

        # Gemma needs scaled values while running the model.
        torch.testing.assert_close(
            partly_frozen(token_ids),
            original_weight * embed_scale,
        )

        exported = partly_frozen.to_embeddings()

        # The checkpoint must contain raw weights because Gemma will apply its
        # runtime scale again after the exported embedding table is installed.
        torch.testing.assert_close(exported.weight, original_weight)
        torch.testing.assert_close(
            exported(token_ids) * embed_scale,
            partly_frozen(token_ids),
        )


if __name__ == "__main__":
    unittest.main()
