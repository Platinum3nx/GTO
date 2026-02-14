from dataclasses import dataclass, field
from typing import Sequence

import torch


@dataclass(slots=True)
class ActionEvent:
    player_id: int
    action_type: int
    bet_fraction: float


@dataclass(slots=True)
class GameState:
    player_hole_cards: Sequence[int]
    board_cards: Sequence[int]
    pot_size: float
    player_stack: float
    opponent_stack: float
    amount_to_call: float
    street: int
    action_history: list[ActionEvent] = field(default_factory=list)
    initial_effective_stack: float = 100.0
    opponent_range_mask: torch.Tensor | None = None
