"""
Implementations of the Attention Influence Modeling task.

#############################################
#############################################

Each implementation takes the follows the following schema:

Args:
    teacher_word_ids: (batch_size, seq_len)
    teacher_attn_weights: (batch_size, num_heads, seq_len, seq_len)
    teacher_value_states: (batch_size, num_heads, seq_len, hidden_size)
    student_word_ids: (batch_size, seq_len)
    student_attn_weights: (batch_size, num_heads, seq_len, seq_len)
    student_value_states: (batch_size, num_heads, seq_len, hidden_size)

Returns:
    tuple(
        teacher_word_states: (batch_size * num_heads * num_words, hidden_size),
        student_word_states: (batch_size * num_heads * num_words, hidden_size),
    )

#############################################
#####                                   #####
#####     Available implementations     #####
#####                                   #####
#############################################

AIM:

    We model the influence of each word on every other word,
    but only when the last word is fully present (all its tokens are present).

Teacher:

    0 [a]
    1 [b][c]
    2 ---------
    2 ------------
    2 [d][e][g][g][g]
    3 [h][i][j][j][j][k]
    4 ---------------------
    4 [l][m][n][n][n][o][p][p]
    5 ------------------------------
    5 [q][r][s][s][s][t][u][u][v][v][w]
       0  1  2  2  2  3  4  4  5  5


Student:

    0 [a]
    1 [b][c]
    2 [d][e][g]
    3 [h][i][j][k]
    4 [l][m][n][o][p]
    5 ------------------
    5 [q][r][s][t][u][v][w]
       0  1  2  3  4  5  5


#############################################
#############################################

AIM Star:

    We model the influence of each word on the last token of the word.

Teacher:

    0 [a]
    1 [b][b]
    2 ---------
    2 ------------
    2 [c][c][c][c][c]
    3 [d][d][d][d][d][d]
    4 ---------------------
    4 [e][e][e][e][e][e][e][e]
    5 ------------------------------
    5 [f][f][f][f][f][f][f][f][f][f][f]
       0  1  2  2  2  3  4  4  5  5


Student:

    0 [a]
    1 [b][b]
    2 [c][c][c]
    3 [d][d][d][d]
    4 [e][e][e][e][e]
    5 ------------------
    5 [f][f][f][f][f][f][f]
       0  1  2  3  4  5  5

"""

from typing import Callable

import torch


AIMImplType = Callable[
    [
        torch.FloatTensor,
        torch.FloatTensor,
        torch.LongTensor,
        torch.FloatTensor,
        torch.FloatTensor,
        torch.LongTensor,
    ],
    tuple[torch.FloatTensor, torch.FloatTensor],
]


def last_token_mask(word_ids: torch.Tensor) -> torch.Tensor:
    """Return a mask for the final token of each non-special segment."""
    if word_ids.ndim != 2:
        raise ValueError(f"word_ids must be two-dimensional, got {tuple(word_ids.shape)}")

    next_word_ids = torch.full_like(word_ids, -100)
    next_word_ids[:, :-1] = word_ids[:, 1:]
    return (word_ids != -100) & (word_ids != next_word_ids)


def causal_word_pair_ids(
    word_ids: torch.Tensor,
    num_heads: int,
) -> tuple[torch.Tensor, int]:
    """Build flattened AIM pair IDs without crossing batch/head boundaries.

    The returned tensor has the same flattened order as
    ``attn_weights.unsqueeze(-1) * value_states.unsqueeze(-3)``.

    Word ids restart in every sequence, so pair ids must be derived per
    (batch row, head) slice. An earlier implementation compared neighbours
    with ``roll()`` and numbered pairs with a global ``cumsum()`` over the
    flattened batch: whenever the word ids touching a row boundary (or the
    wrap-around from the last token back to the first) happened to be equal,
    word segments from different rows or heads were merged into one pair row
    or dropped altogether. Teacher and student AIM states are built
    independently, so any merged or dropped row shifts every following row
    and breaks their one-to-one correspondence in the loss. Keeping the
    row/head structure until the very end makes such collisions impossible.
    """
    if word_ids.ndim != 2:
        raise ValueError(f"word_ids must be two-dimensional, got {tuple(word_ids.shape)}")
    if num_heads <= 0:
        raise ValueError(f"num_heads must be positive, got {num_heads}")

    device = word_ids.device
    _, seq_len = word_ids.shape
    repeated_word_ids = word_ids.repeat_interleave(num_heads, dim=0)
    valid_query_mask = last_token_mask(repeated_word_ids)

    causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=device, dtype=torch.bool))
    causal_word_ids = (
        repeated_word_ids
        .unsqueeze(-2)
        .expand(-1, seq_len, -1)
        .masked_fill(~causal_mask, -100)
        .masked_fill(~valid_query_mask.unsqueeze(-1), -100)
    )

    previous_word_ids = torch.full_like(causal_word_ids, -100)
    previous_word_ids[..., 1:] = causal_word_ids[..., :-1]
    pair_start_mask = (causal_word_ids != -100) & (causal_word_ids != previous_word_ids)

    pair_counts = pair_start_mask.sum(dim=-1)
    flat_pair_counts = pair_counts.reshape(-1)
    pair_offsets = (flat_pair_counts.cumsum(0) - flat_pair_counts).reshape_as(pair_counts)
    pair_ids = pair_start_mask.cumsum(dim=-1) - 1
    pair_ids = pair_ids + pair_offsets.unsqueeze(-1)
    pair_ids = pair_ids.masked_fill(causal_word_ids == -100, -100)

    return pair_ids.reshape(-1), int(flat_pair_counts.sum().item())


def aim_impl(
    teacher_attn_weights: torch.Tensor,
    teacher_value_states: torch.Tensor,
    teacher_word_ids: torch.Tensor,
    student_attn_weights: torch.Tensor,
    student_value_states: torch.Tensor,
    student_word_ids: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Args:
        teacher_word_ids: (batch_size, seq_len)
        teacher_attn_weights: (batch_size, num_heads, seq_len, seq_len)
        teacher_value_states: (batch_size, num_heads, seq_len, hidden_size)
        student_word_ids: (batch_size, seq_len)
        student_attn_weights: (batch_size, num_heads, seq_len, seq_len)
        student_value_states: (batch_size, num_heads, seq_len, hidden_size)

    Returns:
        tuple(
            teacher_word_states: (batch_size * num_heads * num_words, hidden_size),
            student_word_states: (batch_size * num_heads * num_words, hidden_size),
        )
    """
    teacher_word_states = get_aim_states(
        word_ids=teacher_word_ids,
        attn_weights=teacher_attn_weights,
        value_states=teacher_value_states,
    )

    student_word_states = get_aim_states(
        word_ids=student_word_ids,
        attn_weights=student_attn_weights,
        value_states=student_value_states,
    )
    return teacher_word_states, student_word_states


def get_aim_states(
    word_ids: torch.LongTensor,
    attn_weights: torch.FloatTensor,
    value_states: torch.FloatTensor,
) -> torch.Tensor:
    """
    Args:
        word_ids: (batch_size, seq_len)
        attn_weights: (batch_size, num_heads, seq_len, seq_len)
        value_states: (batch_size, num_heads, seq_len, hidden_size)

    Returns:
        (num_pairs, hidden_size)
    """

    num_heads = attn_weights.size(1)
    hidden_size = value_states.size(-1)

    # Pair ids are assigned per (batch row, head) slice so that identical
    # word ids in neighbouring rows or heads never share a pair row.
    full_word_ids, num_pairs = causal_word_pair_ids(word_ids, num_heads)
    valid_word_ids_mask = full_word_ids != -100
    valid_word_ids = full_word_ids[valid_word_ids_mask]

    # (batch_size * num_heads * valid_rows * valid_cols, hidden_size)
    attv = attn_weights.unsqueeze(-1) * value_states.unsqueeze(-3)
    attv = attv.reshape(-1, hidden_size)[valid_word_ids_mask, :]

    if num_pairs == 0:
        return torch.empty(
            0,
            hidden_size,
            device=word_ids.device,
            dtype=attv.dtype,
        )

    # (num_pairs, hidden_size)
    attv = torch.zeros(
        num_pairs,
        hidden_size,
        device=word_ids.device,
        dtype=attv.dtype,
    ).scatter_reduce_(
        dim=0,
        index=valid_word_ids.unsqueeze(-1).expand(-1, hidden_size),
        src=attv,
        reduce='sum',
        include_self=False,
    )

    return attv


def aim_star_impl(
    teacher_attn_weights: torch.Tensor,
    teacher_value_states: torch.Tensor,
    teacher_word_ids: torch.Tensor,
    student_attn_weights: torch.Tensor,
    student_value_states: torch.Tensor,
    student_word_ids: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Args:
        teacher_word_ids: (batch_size, seq_len)
        teacher_attn_weights: (batch_size, num_heads, seq_len, seq_len)
        teacher_value_states: (batch_size, num_heads, seq_len, hidden_size)
        student_word_ids: (batch_size, seq_len)
        student_attn_weights: (batch_size, num_heads, seq_len, seq_len)
        student_value_states: (batch_size, num_heads, seq_len, hidden_size)

    Returns:
        tuple(
            teacher_word_states: (batch_size * num_heads * num_words, hidden_size),
            student_word_states: (batch_size * num_heads * num_words, hidden_size),
        )
    """
    teacher_word_states = get_aim_star_states(
        word_ids=teacher_word_ids,
        attn_weights=teacher_attn_weights,
        value_states=teacher_value_states,
    )

    student_word_states = get_aim_star_states(
        word_ids=student_word_ids,
        attn_weights=student_attn_weights,
        value_states=student_value_states,
    )
    return teacher_word_states, student_word_states


def get_aim_star_states(
    word_ids: torch.LongTensor,
    attn_weights: torch.FloatTensor,
    value_states: torch.FloatTensor,
) -> torch.Tensor:
    """
    Args:
        word_ids: (batch_size, seq_len)
        attn_weights: (batch_size, num_heads, seq_len, seq_len)
        value_states: (batch_size, num_heads, seq_len, hidden_size)

    Returns:
        (num_pairs, hidden_size)
    """

    num_heads = attn_weights.size(1)
    hidden_size = value_states.size(-1)

    # (batch_size * num_heads, seq_len) -> (batch_size * num_heads * seq_len)
    rep_word_ids = word_ids.repeat_interleave(num_heads, dim=0)
    valid_word_mask = last_token_mask(rep_word_ids)

    # (batch_size * num_heads * valid_rows, hidden_size)
    word_states = (
        torch.matmul(attn_weights, value_states)
        .reshape(-1, hidden_size)[valid_word_mask.reshape(-1)]
    )

    return word_states


def get_aim_impl(name: str) -> AIMImplType:
    if name == 'aim':
        return aim_impl
    elif name == 'aim_star':
        return aim_star_impl
    else:
        raise ValueError(f'Invalid implementation: {name}')
