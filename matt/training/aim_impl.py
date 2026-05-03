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

    device = word_ids.device

    batch_size, seq_len = word_ids.shape
    num_heads = attn_weights.size(1)
    hidden_size = value_states.size(-1)

    # (batch_size * num_heads, seq_len) -> (batch_size * num_heads * seq_len)
    rep_word_ids = word_ids.repeat_interleave(num_heads, dim=0)
    rep_word_ids_flat = rep_word_ids.view(-1)
    valid_word_mask = torch.logical_and(
        rep_word_ids_flat != -100,  # no padding or special tokens
        rep_word_ids_flat != rep_word_ids_flat.roll(-1),  # last token of the word
    )

    causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=device, dtype=torch.bool))
    rep_full_word_ids = (
        # (batch_size * num_heads, seq_len)
        rep_word_ids
            .unsqueeze(-2)
            # (batch_size * num_heads, seq_len, seq_len)
            .repeat(1, seq_len, 1)
            .masked_fill(~causal_mask, -100)
            # (batch_size * num_heads * seq_len, seq_len)
            .view(-1, seq_len)
            .masked_fill(~valid_word_mask.unsqueeze(-1), -100)
            # (batch_size * num_heads * seq_len * seq_len)
            .view(-1)
    )

    # (batch_size * num_heads * seq_len * seq_len)
    full_word_ids = torch.where(
        rep_full_word_ids != -100,
        # mask of where the words change, cumsum to get word ids across the batch
        torch.logical_and(
            rep_full_word_ids != -100,
            rep_full_word_ids != rep_full_word_ids.roll(1),
        ).cumsum(0) - 1,  # -1 because we want to start from 0
        -100,
    )
    valid_word_ids_mask = full_word_ids != -100
    valid_word_ids = full_word_ids[valid_word_ids_mask]

    # (batch_size * num_heads * valid_rows * valid_cols, hidden_size)
    attv = attn_weights.unsqueeze(-1) * value_states.unsqueeze(-3)
    attv = attv.view(-1, hidden_size)[valid_word_ids_mask, :]

    num_pairs = valid_word_ids.max() + 1

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
    rep_word_ids_flat = rep_word_ids.view(-1)
    valid_word_mask = torch.logical_and(
        rep_word_ids_flat != -100,  # no padding or special tokens
        rep_word_ids_flat != rep_word_ids_flat.roll(-1),  # last token of the word
    )

    # (batch_size * num_heads * valid_rows, hidden_size)
    word_states = (
        torch.matmul(attn_weights, value_states)
        .view(-1, hidden_size)[valid_word_mask]
    )

    return word_states


def get_aim_impl(name: str) -> AIMImplType:
    if name == 'aim':
        return aim_impl
    elif name == 'aim_star':
        return aim_star_impl
    else:
        raise ValueError(f'Invalid implementation: {name}')
