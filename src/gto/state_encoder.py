from typing import Iterable, Sequence

import torch

from .cards import indices_to_mask
from .constants import (
    ACTION_TYPE_BITS,
    BETTING_FEATURES,
    CARD_FEATURES,
    HISTORY_ACTIONS,
    HISTORY_FEATURES,
    HISTORY_FEATURES_PER_ACTION,
    NUM_CARDS,
    STATE_DIM,
)
from .types import ActionEvent, GameState


CARD_OFFSET = 0
HOLE_OFFSET = CARD_OFFSET
BOARD_OFFSET = HOLE_OFFSET + NUM_CARDS
OPP_RANGE_OFFSET = BOARD_OFFSET + NUM_CARDS

BETTING_OFFSET = CARD_FEATURES
HISTORY_OFFSET = BETTING_OFFSET + BETTING_FEATURES


class StateEncoder:
    def __init__(self, *, dtype: torch.dtype = torch.float32, device: torch.device | str | None = None) -> None:
        self.dtype = dtype
        self.device = device

    def encode(self, game_state: GameState) -> torch.Tensor:
        state = torch.zeros(STATE_DIM, dtype=self.dtype, device=self.device)

        hole_mask = indices_to_mask(game_state.player_hole_cards, dtype=self.dtype, device=self.device)
        board_mask = indices_to_mask(game_state.board_cards, dtype=self.dtype, device=self.device)

        if game_state.opponent_range_mask is not None:
            opp_range = game_state.opponent_range_mask.to(dtype=self.dtype, device=self.device)
            if opp_range.shape != (NUM_CARDS,):
                raise ValueError("opponent_range_mask must have shape (52,)")
        else:
            dead_mask = torch.clamp(hole_mask + board_mask, 0.0, 1.0)
            opp_range = 1.0 - dead_mask

        state[HOLE_OFFSET:BOARD_OFFSET] = hole_mask
        state[BOARD_OFFSET:OPP_RANGE_OFFSET] = board_mask
        state[OPP_RANGE_OFFSET:BETTING_OFFSET] = opp_range

        stack_norm = max(float(game_state.initial_effective_stack), 1e-6)
        state[BETTING_OFFSET + 0] = _norm(game_state.pot_size, stack_norm)
        state[BETTING_OFFSET + 1] = _norm(game_state.player_stack, stack_norm)
        state[BETTING_OFFSET + 2] = _norm(game_state.opponent_stack, stack_norm)
        state[BETTING_OFFSET + 3] = _norm(game_state.amount_to_call, stack_norm)

        street = int(game_state.street)
        if not (0 <= street <= 3):
            raise ValueError(f"Street must be in [0, 3], got {street}")
        state[BETTING_OFFSET + 4 + street] = 1.0

        history = _encode_history(game_state.action_history, dtype=self.dtype, device=self.device)
        state[HISTORY_OFFSET:HISTORY_OFFSET + HISTORY_FEATURES] = history

        return state

    def encode_batch(self, game_states: Sequence[GameState]) -> torch.Tensor:
        if not game_states:
            return torch.empty((0, STATE_DIM), dtype=self.dtype, device=self.device)
        return torch.stack([self.encode(s) for s in game_states], dim=0)


def _encode_history(
    action_history: Iterable[ActionEvent], *, dtype: torch.dtype, device: torch.device | str | None
) -> torch.Tensor:
    vec = torch.zeros(HISTORY_FEATURES, dtype=dtype, device=device)
    actions = list(action_history)[-HISTORY_ACTIONS:]
    pad = HISTORY_ACTIONS - len(actions)

    for i, event in enumerate(actions):
        base = (pad + i) * HISTORY_FEATURES_PER_ACTION
        player_id = int(event.player_id)
        action_type = int(event.action_type)

        if player_id not in (0, 1):
            raise ValueError(f"player_id must be 0 or 1, got {player_id}")
        if not (0 <= action_type < ACTION_TYPE_BITS):
            raise ValueError(f"action_type must be in [0, {ACTION_TYPE_BITS - 1}], got {action_type}")

        vec[base] = float(player_id)
        vec[base + 1 + action_type] = 1.0
        vec[base + 6] = _clip01(float(event.bet_fraction))

    return vec


def _clip01(v: float) -> float:
    return min(max(v, 0.0), 1.0)


def _norm(v: float, denom: float) -> float:
    return _clip01(float(v) / denom)
