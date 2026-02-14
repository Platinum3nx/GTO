import torch
from torch.nn import functional as F

from gto.eval import TensorLUTShowdownEvaluator
from gto.constants import (
    ActionBucket,
    HISTORY_ACTIONS,
    HISTORY_FEATURES_PER_ACTION,
    N_ACTIONS,
    NUM_CARDS,
    STATE_DIM,
    Street,
)


class VectorHUNLEnv:
    """
    Vectorized heads-up no-limit hold'em flop-subgame simulator.

    This environment is intentionally vectorized for high-throughput GPU simulation.
    Showdowns are resolved by an exact 7-card evaluator backed by a tensorized LUT.
    """

    def __init__(
        self,
        batch_size: int,
        *,
        stack_bb: float = 100.0,
        small_blind: float = 0.5,
        big_blind: float = 1.0,
        initial_pot_bb: float = 6.0,
        dtype: torch.dtype = torch.float32,
        device: torch.device | str = "cpu",
        exact_showdown: bool = True,
        showdown_evaluator: TensorLUTShowdownEvaluator | None = None,
    ) -> None:
        self.batch_size = batch_size
        self.stack_bb = float(stack_bb)
        self.small_blind = float(small_blind)
        self.big_blind = float(big_blind)
        self.initial_pot_bb = float(initial_pot_bb)
        self.dtype = dtype
        self.device = torch.device(device)
        self.exact_showdown = exact_showdown
        self.showdown_evaluator = (
            showdown_evaluator
            if showdown_evaluator is not None
            else (TensorLUTShowdownEvaluator(device=self.device) if exact_showdown else None)
        )

        self.reset(batch_size=batch_size)

    def reset(self, *, batch_size: int | None = None) -> torch.Tensor:
        if batch_size is not None and batch_size != self.batch_size:
            self.batch_size = batch_size

        b = self.batch_size
        self.initial_stack = torch.full((b,), self.stack_bb, dtype=self.dtype, device=self.device)
        self.stacks = torch.full((b, 2), self.stack_bb, dtype=self.dtype, device=self.device)
        self.pot = torch.full((b,), self.initial_pot_bb, dtype=self.dtype, device=self.device)
        self.to_call = torch.zeros((b,), dtype=self.dtype, device=self.device)
        self.street = torch.full((b,), int(Street.FLOP), dtype=torch.long, device=self.device)
        self.current_player = torch.zeros((b,), dtype=torch.long, device=self.device)
        self.checks_in_round = torch.zeros((b,), dtype=torch.long, device=self.device)
        self.done = torch.zeros((b,), dtype=torch.bool, device=self.device)
        self.terminal_utility = torch.zeros((b,), dtype=self.dtype, device=self.device)
        self.history = torch.zeros(
            (b, HISTORY_ACTIONS, HISTORY_FEATURES_PER_ACTION), dtype=self.dtype, device=self.device
        )

        self._deal_cards()
        return self.get_state_tensor()

    def _deal_cards(self) -> None:
        perm = torch.rand((self.batch_size, NUM_CARDS), device=self.device).argsort(dim=1)
        self.hole_cards = torch.stack((perm[:, 0:2], perm[:, 2:4]), dim=1)
        self.board_cards = perm[:, 4:9]

    def legal_action_mask(self) -> torch.Tensor:
        b = self.batch_size
        mask = torch.zeros((b, N_ACTIONS), dtype=torch.bool, device=self.device)
        active = ~self.done
        cur_stack = self._current_stack()
        facing_bet = self.to_call > 0

        mask[:, ActionBucket.CHECK_CALL] = active
        mask[:, ActionBucket.FOLD] = active & facing_bet

        can_put_more = active & (cur_stack > 0)
        mask[:, ActionBucket.BET_33] = can_put_more
        mask[:, ActionBucket.BET_75] = can_put_more
        mask[:, ActionBucket.ALL_IN] = can_put_more
        return mask

    def step(self, actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if actions.shape != (self.batch_size,):
            raise ValueError(f"actions must have shape ({self.batch_size},)")

        actions = actions.to(device=self.device, dtype=torch.long).clamp(0, N_ACTIONS - 1)
        active = ~self.done
        if not active.any():
            return self.get_state_tensor(), self.done.clone(), self.terminal_utility.clone()

        legal = self.legal_action_mask()
        batch = torch.arange(self.batch_size, device=self.device)
        fallback = torch.full_like(actions, int(ActionBucket.CHECK_CALL))
        actions = torch.where(legal[batch, actions], actions, fallback)

        self._append_history(actions, active)

        cur = self.current_player
        opp = 1 - cur

        cur_stack = self.stacks[batch, cur]
        to_call = self.to_call.clone()

        round_closed = torch.zeros_like(active)
        next_to_call = torch.zeros_like(self.to_call)

        fold_mask = active & (actions == int(ActionBucket.FOLD)) & (to_call > 0)
        self._award_pot(fold_mask, opp)

        check_call_mask = active & (actions == int(ActionBucket.CHECK_CALL))
        facing_mask = check_call_mask & (to_call > 0)
        check_mask = check_call_mask & (to_call == 0)

        call_amt = torch.minimum(cur_stack, to_call)
        self._move_chips_from_actor(facing_mask, call_amt)
        round_closed = round_closed | facing_mask

        self.checks_in_round[check_mask] += 1
        round_closed = round_closed | (check_mask & (self.checks_in_round >= 2))

        bet_mask = active & (actions >= int(ActionBucket.BET_33))
        bet_frac = torch.zeros((self.batch_size,), dtype=self.dtype, device=self.device)
        bet_frac[actions == int(ActionBucket.BET_33)] = 0.33
        bet_frac[actions == int(ActionBucket.BET_75)] = 0.75
        bet_frac[actions == int(ActionBucket.ALL_IN)] = 1.0

        facing_for_bet = bet_mask & (to_call > 0)
        call_then_raise = torch.minimum(self.stacks[batch, cur], to_call)
        self._move_chips_from_actor(facing_for_bet, call_then_raise)

        cur_stack_after_call = self.stacks[batch, cur]
        target_raise = bet_frac * self.pot
        target_raise = torch.where(actions == int(ActionBucket.ALL_IN), cur_stack_after_call, target_raise)
        raise_amt = torch.minimum(cur_stack_after_call, target_raise)
        raise_amt = torch.where(bet_mask, raise_amt, torch.zeros_like(raise_amt))

        self._move_chips_from_actor(bet_mask, raise_amt)
        next_to_call = torch.where(bet_mask, raise_amt, next_to_call)

        self.checks_in_round[bet_mask] = 0
        round_closed = round_closed | (facing_for_bet & (raise_amt <= 0))

        new_done = self.done.clone()
        new_done[fold_mask] = True

        unresolved = active & ~new_done
        switch_mask = unresolved & ~round_closed
        self.current_player[switch_mask] = opp[switch_mask]
        self.to_call[switch_mask] = next_to_call[switch_mask]

        close_mask = unresolved & round_closed
        self.to_call[close_mask] = 0.0
        self.checks_in_round[close_mask] = 0

        river_close = close_mask & (self.street == int(Street.RIVER))
        advance = close_mask & ~river_close

        self.street[advance] += 1
        self.current_player[advance] = 0

        self._run_showdown(river_close)
        new_done[river_close] = True

        all_in_and_matched = unresolved & ~new_done & (self.to_call <= 0) & (self.stacks.min(dim=1).values <= 0)
        if all_in_and_matched.any():
            self.street[all_in_and_matched] = int(Street.RIVER)
            self._run_showdown(all_in_and_matched)
            new_done[all_in_and_matched] = True

        self.done = new_done
        self.terminal_utility = torch.where(
            self.done,
            self.stacks[:, 0] - self.initial_stack,
            self.terminal_utility,
        )

        return self.get_state_tensor(), self.done.clone(), self.terminal_utility.clone()

    def get_state_tensor(self) -> torch.Tensor:
        b = self.batch_size
        batch = torch.arange(b, device=self.device)

        acting_hole = self.hole_cards[batch, self.current_player]

        hole_mask = torch.zeros((b, NUM_CARDS), dtype=self.dtype, device=self.device)
        hole_mask.scatter_(1, acting_hole, 1.0)

        reveal_n = self.street + 2
        board_mask = torch.zeros((b, NUM_CARDS), dtype=self.dtype, device=self.device)
        positions = torch.arange(5, device=self.device).unsqueeze(0).expand(b, 5)
        revealed = positions < reveal_n.unsqueeze(1)

        for pos in range(5):
            pos_mask = revealed[:, pos]
            if pos_mask.any():
                idx = self.board_cards[pos_mask, pos]
                board_mask[pos_mask, idx] = 1.0

        dead_mask = torch.clamp(hole_mask + board_mask, 0.0, 1.0)
        opp_range_mask = 1.0 - dead_mask

        cur_stack = self._current_stack()
        opp_stack = self._opponent_stack()

        betting = torch.zeros((b, 8), dtype=self.dtype, device=self.device)
        denom = self.initial_stack.clamp_min(1e-6)
        betting[:, 0] = (self.pot / denom).clamp(0.0, 1.0)
        betting[:, 1] = (cur_stack / denom).clamp(0.0, 1.0)
        betting[:, 2] = (opp_stack / denom).clamp(0.0, 1.0)
        betting[:, 3] = (self.to_call / denom).clamp(0.0, 1.0)
        betting[:, 4:8] = F.one_hot(self.street.clamp(0, 3), num_classes=4).to(self.dtype)

        history = self.history.reshape(b, -1)

        state = torch.cat((hole_mask, board_mask, opp_range_mask, betting, history), dim=1)
        if state.shape[1] != STATE_DIM:
            raise RuntimeError(f"State dim mismatch: got {state.shape[1]}, expected {STATE_DIM}")
        return state

    def _append_history(self, actions: torch.Tensor, active: torch.Tensor) -> None:
        if active.any():
            self.history[active] = torch.roll(self.history[active], shifts=-1, dims=1)

        feat = torch.zeros(
            (self.batch_size, HISTORY_FEATURES_PER_ACTION), dtype=self.dtype, device=self.device
        )
        feat[:, 0] = self.current_player.to(self.dtype)
        feat[:, 1:6] = F.one_hot(actions.clamp(0, 4), num_classes=5).to(self.dtype)

        bet_frac = torch.zeros((self.batch_size,), dtype=self.dtype, device=self.device)
        bet_frac[actions == int(ActionBucket.BET_33)] = 0.33
        bet_frac[actions == int(ActionBucket.BET_75)] = 0.75
        bet_frac[actions == int(ActionBucket.ALL_IN)] = 1.0
        feat[:, 6] = bet_frac

        last = self.history[:, -1]
        last[active] = feat[active]
        self.history[:, -1] = last

    def _move_chips_from_actor(self, mask: torch.Tensor, amount: torch.Tensor) -> None:
        if not mask.any():
            return
        batch = torch.arange(self.batch_size, device=self.device)
        cur = self.current_player

        pay = torch.where(mask, amount, torch.zeros_like(amount)).clamp_min(0.0)
        pay = torch.minimum(pay, self.stacks[batch, cur])

        self.stacks[batch, cur] = self.stacks[batch, cur] - pay
        self.pot = self.pot + pay

    def _award_pot(self, mask: torch.Tensor, winner: torch.Tensor) -> None:
        if not mask.any():
            return
        idx = torch.where(mask)[0]
        win = winner[idx]
        self.stacks[idx, win] = self.stacks[idx, win] + self.pot[idx]
        self.pot[idx] = 0.0

    def _run_showdown(self, mask: torch.Tensor) -> None:
        if not mask.any():
            return

        if self.showdown_evaluator is not None:
            s0, s1 = self.showdown_evaluator.compare(
                self.hole_cards[:, 0],
                self.hole_cards[:, 1],
                self.board_cards,
            )
        else:
            s0 = _approx_hand_strength(self.hole_cards[:, 0], self.board_cards)
            s1 = _approx_hand_strength(self.hole_cards[:, 1], self.board_cards)

        p0_win = mask & (s0 > s1)
        p1_win = mask & (s1 > s0)
        tie = mask & (s0 == s1)

        if p0_win.any():
            idx = torch.where(p0_win)[0]
            self.stacks[idx, 0] = self.stacks[idx, 0] + self.pot[idx]
            self.pot[idx] = 0.0

        if p1_win.any():
            idx = torch.where(p1_win)[0]
            self.stacks[idx, 1] = self.stacks[idx, 1] + self.pot[idx]
            self.pot[idx] = 0.0

        if tie.any():
            idx = torch.where(tie)[0]
            split = self.pot[idx] * 0.5
            self.stacks[idx, 0] = self.stacks[idx, 0] + split
            self.stacks[idx, 1] = self.stacks[idx, 1] + split
            self.pot[idx] = 0.0

    def _current_stack(self) -> torch.Tensor:
        batch = torch.arange(self.batch_size, device=self.device)
        return self.stacks[batch, self.current_player]

    def _opponent_stack(self) -> torch.Tensor:
        batch = torch.arange(self.batch_size, device=self.device)
        return self.stacks[batch, 1 - self.current_player]


def _approx_hand_strength(hole_cards: torch.Tensor, board_cards: torch.Tensor) -> torch.Tensor:
    cards = torch.cat((hole_cards, board_cards), dim=1)
    ranks = cards // 4
    suits = cards % 4

    top5 = torch.topk(ranks, k=5, dim=1).values.sum(dim=1).to(torch.float32)

    rank_counts = F.one_hot(ranks, num_classes=13).sum(dim=1)
    suit_counts = F.one_hot(suits, num_classes=4).sum(dim=1)

    pair_bonus = (rank_counts >= 2).sum(dim=1) * 4
    trips_bonus = (rank_counts >= 3).sum(dim=1) * 10
    quads_bonus = (rank_counts >= 4).sum(dim=1) * 20
    flush_bonus = (suit_counts.max(dim=1).values >= 5).to(torch.float32) * 8

    return top5 + pair_bonus + trips_bonus + quads_bonus + flush_bonus
