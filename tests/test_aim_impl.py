import unittest

import torch

from matt.training.aim_impl import get_aim_states, get_aim_star_states


class AIMStatesTest(unittest.TestCase):
    def test_aim_states_keep_same_word_ids_separate_across_batch_rows(self) -> None:
        word_ids = torch.tensor([[0], [0]])
        attn_weights = torch.ones(2, 1, 1, 1)
        value_states = torch.tensor([[[[1.0]]], [[[2.0]]]])

        actual = get_aim_states(word_ids, attn_weights, value_states)

        torch.testing.assert_close(actual, torch.tensor([[1.0], [2.0]]))

    def test_aim_states_keep_same_word_ids_separate_across_heads(self) -> None:
        word_ids = torch.tensor([[0]])
        attn_weights = torch.ones(1, 2, 1, 1)
        value_states = torch.tensor([[[[1.0]], [[2.0]]]])

        actual = get_aim_states(word_ids, attn_weights, value_states)

        torch.testing.assert_close(actual, torch.tensor([[1.0], [2.0]]))

    def test_aim_star_states_keep_same_word_ids_separate_across_batch_rows(self) -> None:
        word_ids = torch.tensor([[0], [0]])
        attn_weights = torch.ones(2, 1, 1, 1)
        value_states = torch.tensor([[[[1.0]]], [[[2.0]]]])

        actual = get_aim_star_states(word_ids, attn_weights, value_states)

        torch.testing.assert_close(actual, torch.tensor([[1.0], [2.0]]))

    def test_aim_star_states_keep_same_word_ids_separate_across_heads(self) -> None:
        word_ids = torch.tensor([[0]])
        attn_weights = torch.ones(1, 2, 1, 1)
        value_states = torch.tensor([[[[1.0]], [[2.0]]]])

        actual = get_aim_star_states(word_ids, attn_weights, value_states)

        torch.testing.assert_close(actual, torch.tensor([[1.0], [2.0]]))


if __name__ == "__main__":
    unittest.main()
