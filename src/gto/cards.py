from typing import Iterable

import torch

from .constants import NUM_CARDS

RANKS = "23456789TJQKA"
SUITS = "cdhs"


def card_to_index(card: str) -> int:
    """Convert human card string (e.g. 'Ah') to a 0..51 index."""
    if len(card) != 2:
        raise ValueError(f"Invalid card: {card}")
    rank, suit = card[0].upper(), card[1].lower()
    if rank not in RANKS or suit not in SUITS:
        raise ValueError(f"Invalid card: {card}")
    return RANKS.index(rank) * 4 + SUITS.index(suit)


def index_to_card(index: int) -> str:
    if not (0 <= index < NUM_CARDS):
        raise ValueError(f"Card index out of range: {index}")
    return f"{RANKS[index // 4]}{SUITS[index % 4]}"


def parse_card_sequence(cards: str) -> list[int]:
    """
    Parse a compact or spaced card sequence into card indices.

    Examples:
    - "Kh8s2c" -> [idx(Kh), idx(8s), idx(2c)]
    - "Kh 8s 2c" -> [idx(Kh), idx(8s), idx(2c)]
    """
    compact = cards.replace(" ", "").replace(",", "")
    if len(compact) == 0 or len(compact) % 2 != 0:
        raise ValueError(f"Invalid card sequence: {cards}")

    out = [card_to_index(compact[i : i + 2]) for i in range(0, len(compact), 2)]
    if len(out) != len(set(out)):
        raise ValueError(f"Duplicate cards in sequence: {cards}")
    return out


def indices_to_mask(
    cards: Iterable[int],
    *,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    mask = torch.zeros(NUM_CARDS, dtype=dtype, device=device)
    card_idx = torch.as_tensor(list(cards), dtype=torch.long, device=device)
    if card_idx.numel() == 0:
        return mask
    if torch.any(card_idx < 0) or torch.any(card_idx >= NUM_CARDS):
        raise ValueError("Card index out of range")
    mask[card_idx] = 1.0
    return mask
