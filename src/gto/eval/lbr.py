from dataclasses import dataclass

import torch

from gto.cfr import sample_actions
from gto.constants import ActionBucket, N_ACTIONS
from gto.env import VectorHUNLEnv
from gto.models import PolicyNetwork


@dataclass(slots=True)
class LBRConfig:
    eval_games: int = 128
    max_steps_per_hand: int = 16
    rollouts_per_action: int = 2


class LocalBestResponseEvaluator:
    """
    Local Best Response (LBR) exploitability proxy.

    For each exploiter decision, run one-step local action search by estimating the
    continuation value of each legal action via Monte Carlo rollouts against the
    policy network.
    """

    def __init__(
        self,
        cfg: LBRConfig,
        *,
        device: torch.device | str,
        stack_bb: float,
        small_blind: float,
        big_blind: float,
        exact_showdown: bool,
        fixed_flop_cards: list[int] | None = None,
        showdown_evaluator=None,
    ) -> None:
        self.cfg = cfg
        self.device = torch.device(device)

        self.eval_env = VectorHUNLEnv(
            batch_size=cfg.eval_games,
            stack_bb=stack_bb,
            small_blind=small_blind,
            big_blind=big_blind,
            device=self.device,
            exact_showdown=exact_showdown,
            fixed_flop_cards=fixed_flop_cards,
            showdown_evaluator=showdown_evaluator,
        )
        self.probe_env = VectorHUNLEnv(
            batch_size=1,
            stack_bb=stack_bb,
            small_blind=small_blind,
            big_blind=big_blind,
            device=self.device,
            exact_showdown=exact_showdown,
            fixed_flop_cards=fixed_flop_cards,
            showdown_evaluator=showdown_evaluator,
        )

    @torch.no_grad()
    def estimate(self, policy_net: PolicyNetwork) -> dict[str, float]:
        was_training = policy_net.training
        policy_net.eval()

        lbr_p0 = self._run_exploiter(policy_net, exploiter=0)
        lbr_p1 = self._run_exploiter(policy_net, exploiter=1)

        if was_training:
            policy_net.train()

        return {
            "lbr_p0": lbr_p0,
            "lbr_p1": lbr_p1,
            "lbr_exploitability": 0.5 * (lbr_p0 + lbr_p1),
        }

    def _run_exploiter(self, policy_net: PolicyNetwork, *, exploiter: int) -> float:
        self.eval_env.reset(batch_size=self.cfg.eval_games)
        batch_size = self.cfg.eval_games

        for _ in range(self.cfg.max_steps_per_hand):
            active = ~self.eval_env.done
            if not active.any():
                break

            players = self.eval_env.current_player
            actions = torch.full(
                (batch_size,),
                int(ActionBucket.CHECK_CALL),
                dtype=torch.long,
                device=self.device,
            )

            hero_idx = torch.where(active & (players != exploiter))[0]
            if hero_idx.numel() > 0:
                state = self.eval_env.get_state_tensor()[hero_idx]
                legal = self.eval_env.legal_action_mask()[hero_idx]
                probs = policy_net.action_probs(state, legal)
                sampled = sample_actions(probs)
                actions[hero_idx] = sampled

            exploiter_idx = torch.where(active & (players == exploiter))[0]
            for idx in exploiter_idx.tolist():
                game_state = self.eval_env.export_game(idx)
                actions[idx] = self._best_local_action(policy_net, game_state, exploiter)

            self.eval_env.step(actions)

        util = self.eval_env.terminal_utility
        if exploiter == 1:
            util = -util
        return float(util.mean().item())

    def _best_local_action(
        self,
        policy_net: PolicyNetwork,
        game_state: dict[str, torch.Tensor],
        exploiter: int,
    ) -> int:
        self.probe_env.import_game(game_state)
        legal = self.probe_env.legal_action_mask()[0]

        best_action = int(ActionBucket.CHECK_CALL)
        best_value = float("-inf")

        for action in range(N_ACTIONS):
            if not bool(legal[action].item()):
                continue

            v = self._estimate_action_value(policy_net, game_state, exploiter, action)
            if v > best_value:
                best_value = v
                best_action = action

        return best_action

    def _estimate_action_value(
        self,
        policy_net: PolicyNetwork,
        game_state: dict[str, torch.Tensor],
        exploiter: int,
        action: int,
    ) -> float:
        total = 0.0

        for _ in range(self.cfg.rollouts_per_action):
            self.probe_env.import_game(game_state)
            self.probe_env.step(torch.tensor([action], device=self.device, dtype=torch.long))

            for _ in range(self.cfg.max_steps_per_hand):
                if bool(self.probe_env.done[0].item()):
                    break

                state = self.probe_env.get_state_tensor()
                legal = self.probe_env.legal_action_mask()
                probs = policy_net.action_probs(state, legal)
                sampled = sample_actions(probs)
                self.probe_env.step(sampled)

            u0 = float(self.probe_env.terminal_utility[0].item())
            total += u0 if exploiter == 0 else -u0

        return total / max(self.cfg.rollouts_per_action, 1)
