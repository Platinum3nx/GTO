import math
from typing import ClassVar

import torch
from torch.nn import functional as F

from gto.constants import NUM_CARDS

LUT5_SIZE = math.comb(NUM_CARDS, 5)


class TensorLUTShowdownEvaluator:
    """
    GPU-resident batched showdown evaluator.

    - Builds a 5-card lookup table indexed by combinadic rank C(52,5).
    - Evaluates 7-card hold'em hands by taking max score across 21 five-card subsets.
    - Keeps LUT and gather operations on the active torch device.
    """

    _cpu_lut5: ClassVar[torch.Tensor | None] = None
    _device_cache: ClassVar[dict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor]]] = {}

    def __init__(self, *, device: torch.device | str = "cpu") -> None:
        self.device = torch.device(device)
        key = str(self.device)

        if TensorLUTShowdownEvaluator._cpu_lut5 is None:
            TensorLUTShowdownEvaluator._cpu_lut5 = _build_lut5_cpu()

        if key not in TensorLUTShowdownEvaluator._device_cache:
            lut5 = TensorLUTShowdownEvaluator._cpu_lut5.to(self.device, non_blocking=True)
            choose5 = torch.combinations(torch.arange(7, dtype=torch.long), r=5).to(self.device)
            binom = _build_binom_table(device=self.device)
            TensorLUTShowdownEvaluator._device_cache[key] = (lut5, choose5, binom)

        self._lut5, self._choose5, self._binom = TensorLUTShowdownEvaluator._device_cache[key]

    def score_7card(self, cards7: torch.Tensor) -> torch.Tensor:
        """Return exact hand strength score for each 7-card hand, larger is better."""
        if cards7.ndim != 2 or cards7.shape[1] != 7:
            raise ValueError("cards7 must have shape [B, 7]")

        cards7 = cards7.to(self.device, dtype=torch.long)
        subsets = cards7[:, self._choose5]  # [B, 21, 5]
        subsets = torch.sort(subsets, dim=-1).values

        idx = _combinadic_rank_5(subsets.reshape(-1, 5), self._binom)
        scores = self._lut5[idx].view(cards7.shape[0], -1)
        return scores.max(dim=1).values

    def compare(self, hole0: torch.Tensor, hole1: torch.Tensor, board5: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return 7-card scores for each player for batched heads-up showdowns."""
        if hole0.shape != hole1.shape or hole0.ndim != 2 or hole0.shape[1] != 2:
            raise ValueError("hole tensors must both have shape [B, 2]")
        if board5.ndim != 2 or board5.shape[1] != 5:
            raise ValueError("board5 must have shape [B, 5]")

        cards0 = torch.cat((hole0, board5), dim=1)
        cards1 = torch.cat((hole1, board5), dim=1)
        return self.score_7card(cards0), self.score_7card(cards1)


def _build_lut5_cpu() -> torch.Tensor:
    deck = torch.arange(NUM_CARDS, dtype=torch.long)
    combos = torch.combinations(deck, r=5)  # [2,598,960, 5]

    scores = _score_five_card_hands(combos)
    binom = _build_binom_table(device=torch.device("cpu"))
    idx = _combinadic_rank_5(combos, binom)

    lut = torch.empty((LUT5_SIZE,), dtype=torch.int32)
    lut[idx] = scores
    return lut


def _build_binom_table(*, device: torch.device) -> torch.Tensor:
    table = torch.zeros((NUM_CARDS + 1, 6), dtype=torch.long, device=device)
    for n in range(NUM_CARDS + 1):
        table[n, 0] = 1
        for k in range(1, min(5, n) + 1):
            table[n, k] = math.comb(n, k)
    return table


def _combinadic_rank_5(cards5: torch.Tensor, binom: torch.Tensor) -> torch.Tensor:
    cards5 = cards5.to(dtype=torch.long)
    if cards5.ndim != 2 or cards5.shape[1] != 5:
        raise ValueError("cards5 must have shape [N, 5]")

    c0 = cards5[:, 0]
    c1 = cards5[:, 1]
    c2 = cards5[:, 2]
    c3 = cards5[:, 3]
    c4 = cards5[:, 4]

    return (
        binom[c0, 1]
        + binom[c1, 2]
        + binom[c2, 3]
        + binom[c3, 4]
        + binom[c4, 5]
    )


def _score_five_card_hands(cards5: torch.Tensor) -> torch.Tensor:
    """Exact 5-card high-hand ordering score. Higher is stronger."""
    cards5 = cards5.to(dtype=torch.long)
    ranks = cards5 // 4
    suits = cards5 % 4

    n = cards5.shape[0]
    rank_counts = F.one_hot(ranks, num_classes=13).sum(dim=1)
    rank_faces = torch.arange(2, 15, dtype=torch.long).unsqueeze(0).expand(n, -1)

    is_flush = (suits == suits[:, :1]).all(dim=1)
    straight_high = _straight_high(rank_counts > 0)
    is_straight = straight_high > 0

    pair_mask = rank_counts == 2
    trip_mask = rank_counts == 3
    quad_mask = rank_counts == 4
    single_mask = rank_counts == 1

    pair_count = pair_mask.sum(dim=1)
    has_trip = trip_mask.any(dim=1)
    has_quad = quad_mask.any(dim=1)

    is_straight_flush = is_straight & is_flush
    is_four = has_quad
    is_full_house = has_trip & (pair_count == 1)
    is_three = has_trip & ~is_full_house
    is_two_pair = pair_count == 2
    is_pair = (pair_count == 1) & ~has_trip
    is_high = ~(is_straight_flush | is_four | is_full_house | is_flush | is_straight | is_three | is_two_pair | is_pair)

    d1 = torch.zeros((n,), dtype=torch.long)
    d2 = torch.zeros((n,), dtype=torch.long)
    d3 = torch.zeros((n,), dtype=torch.long)
    d4 = torch.zeros((n,), dtype=torch.long)
    d5 = torch.zeros((n,), dtype=torch.long)
    category = torch.zeros((n,), dtype=torch.long)

    # Straight flush
    category[is_straight_flush] = 8
    d1[is_straight_flush] = straight_high[is_straight_flush]

    # Four of a kind
    quad_rank = _topk_masked(rank_faces, quad_mask, k=1)[:, 0]
    quad_kicker = _topk_masked(rank_faces, single_mask, k=1)[:, 0]
    category[is_four] = 7
    d1[is_four] = quad_rank[is_four]
    d2[is_four] = quad_kicker[is_four]

    # Full house
    trip_rank = _topk_masked(rank_faces, trip_mask, k=1)[:, 0]
    full_pair_rank = _topk_masked(rank_faces, pair_mask, k=1)[:, 0]
    category[is_full_house] = 6
    d1[is_full_house] = trip_rank[is_full_house]
    d2[is_full_house] = full_pair_rank[is_full_house]

    # Flush
    sorted_faces = torch.sort(ranks + 2, dim=1, descending=True).values
    category[is_flush & ~is_straight_flush] = 5
    d1[is_flush & ~is_straight_flush] = sorted_faces[is_flush & ~is_straight_flush, 0]
    d2[is_flush & ~is_straight_flush] = sorted_faces[is_flush & ~is_straight_flush, 1]
    d3[is_flush & ~is_straight_flush] = sorted_faces[is_flush & ~is_straight_flush, 2]
    d4[is_flush & ~is_straight_flush] = sorted_faces[is_flush & ~is_straight_flush, 3]
    d5[is_flush & ~is_straight_flush] = sorted_faces[is_flush & ~is_straight_flush, 4]

    # Straight
    straight_only = is_straight & ~is_straight_flush
    category[straight_only] = 4
    d1[straight_only] = straight_high[straight_only]

    # Three of a kind
    three_kickers = _topk_masked(rank_faces, single_mask, k=2)
    category[is_three] = 3
    d1[is_three] = trip_rank[is_three]
    d2[is_three] = three_kickers[is_three, 0]
    d3[is_three] = three_kickers[is_three, 1]

    # Two pair
    pair_top2 = _topk_masked(rank_faces, pair_mask, k=2)
    two_pair_kicker = _topk_masked(rank_faces, single_mask, k=1)[:, 0]
    category[is_two_pair] = 2
    d1[is_two_pair] = pair_top2[is_two_pair, 0]
    d2[is_two_pair] = pair_top2[is_two_pair, 1]
    d3[is_two_pair] = two_pair_kicker[is_two_pair]

    # One pair
    pair_rank = _topk_masked(rank_faces, pair_mask, k=1)[:, 0]
    pair_kickers = _topk_masked(rank_faces, single_mask, k=3)
    category[is_pair] = 1
    d1[is_pair] = pair_rank[is_pair]
    d2[is_pair] = pair_kickers[is_pair, 0]
    d3[is_pair] = pair_kickers[is_pair, 1]
    d4[is_pair] = pair_kickers[is_pair, 2]

    # High card
    category[is_high] = 0
    d1[is_high] = sorted_faces[is_high, 0]
    d2[is_high] = sorted_faces[is_high, 1]
    d3[is_high] = sorted_faces[is_high, 2]
    d4[is_high] = sorted_faces[is_high, 3]
    d5[is_high] = sorted_faces[is_high, 4]

    base = 15
    tiebreak = d1 * (base**4) + d2 * (base**3) + d3 * (base**2) + d4 * base + d5
    score = category * (base**5) + tiebreak
    return score.to(torch.int32)


def _topk_masked(values: torch.Tensor, mask: torch.Tensor, *, k: int) -> torch.Tensor:
    masked = torch.where(mask, values, torch.zeros_like(values))
    return torch.topk(masked, k=k, dim=1).values


def _straight_high(rank_present: torch.Tensor) -> torch.Tensor:
    """Return straight high-card face value in [5..14], or 0 if not straight."""
    n = rank_present.shape[0]
    out = torch.zeros((n,), dtype=torch.long)

    for high in range(12, 3, -1):
        seq = rank_present[:, high - 4 : high + 1]
        hit = seq.all(dim=1) & (out == 0)
        out[hit] = high + 2

    wheel = (
        rank_present[:, 12]
        & rank_present[:, 0]
        & rank_present[:, 1]
        & rank_present[:, 2]
        & rank_present[:, 3]
    )
    out[wheel & (out == 0)] = 5
    return out
