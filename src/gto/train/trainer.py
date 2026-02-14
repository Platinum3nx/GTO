from dataclasses import dataclass

import torch
from torch.nn import functional as F

from gto.cfr import regret_matching_plus, sample_actions
from gto.constants import N_ACTIONS, STATE_DIM
from gto.env import VectorHUNLEnv
from gto.eval.lbr import LBRConfig, LocalBestResponseEvaluator
from gto.models import AdvantageNetwork, PolicyNetwork, masked_kl_policy_loss
from gto.replay import ReservoirBuffer


@dataclass(slots=True)
class DeepCFRConfig:
    iterations: int = 200
    trajectories_per_iter: int = 1024
    parallel_games: int = 1024
    max_steps_per_trajectory: int = 16
    update_every_iter: int = 1
    updates_per_cycle: int = 2
    batch_size: int = 4096
    buffer_capacity: int = 1_000_000
    learning_rate: float = 3e-4
    weight_decay: float = 1e-5
    device: str = "cuda"
    precision: str = "bf16"  # one of: fp32, fp16, bf16
    stack_bb: float = 100.0
    small_blind: float = 0.5
    big_blind: float = 1.0
    exact_showdown: bool = True
    lbr_enabled: bool = True
    lbr_eval_every_iters: int = 5
    lbr_eval_games: int = 128
    lbr_rollouts_per_action: int = 2


class DeepCFRTrainer:
    def __init__(self, cfg: DeepCFRConfig) -> None:
        self.cfg = cfg
        self.device = self._resolve_device(cfg.device)

        self.env = VectorHUNLEnv(
            batch_size=cfg.parallel_games,
            stack_bb=cfg.stack_bb,
            small_blind=cfg.small_blind,
            big_blind=cfg.big_blind,
            device=self.device,
            exact_showdown=cfg.exact_showdown,
        )

        self.adv_net = AdvantageNetwork().to(self.device)
        self.policy_net = PolicyNetwork().to(self.device)

        self.adv_opt = torch.optim.AdamW(
            self.adv_net.parameters(),
            lr=cfg.learning_rate,
            weight_decay=cfg.weight_decay,
        )
        self.policy_opt = torch.optim.AdamW(
            self.policy_net.parameters(),
            lr=cfg.learning_rate,
            weight_decay=cfg.weight_decay,
        )

        self.adv_buffer = ReservoirBuffer(
            cfg.buffer_capacity,
            state_dim=STATE_DIM,
            target_dim=N_ACTIONS,
            device="cpu",
        )
        self.strategy_buffer = ReservoirBuffer(
            cfg.buffer_capacity,
            state_dim=STATE_DIM,
            target_dim=N_ACTIONS,
            device="cpu",
        )

        self.scaler = torch.cuda.amp.GradScaler(enabled=(cfg.precision == "fp16" and self.device.type == "cuda"))
        self.lbr_evaluator = (
            LocalBestResponseEvaluator(
                LBRConfig(
                    eval_games=cfg.lbr_eval_games,
                    max_steps_per_hand=cfg.max_steps_per_trajectory,
                    rollouts_per_action=cfg.lbr_rollouts_per_action,
                ),
                device=self.device,
                stack_bb=cfg.stack_bb,
                small_blind=cfg.small_blind,
                big_blind=cfg.big_blind,
                exact_showdown=cfg.exact_showdown,
                showdown_evaluator=self.env.showdown_evaluator,
            )
            if cfg.lbr_enabled
            else None
        )

    def train(self) -> list[dict[str, float]]:
        history: list[dict[str, float]] = []

        for it in range(1, self.cfg.iterations + 1):
            self.collect_trajectories(self.cfg.trajectories_per_iter)

            metrics = {"iter": float(it)}
            if it % self.cfg.update_every_iter == 0:
                for _ in range(self.cfg.updates_per_cycle):
                    step_metrics = self.train_step()
                    for k, v in step_metrics.items():
                        metrics[k] = metrics.get(k, 0.0) + v / self.cfg.updates_per_cycle
            if self.cfg.lbr_enabled and self.cfg.lbr_eval_every_iters > 0 and it % self.cfg.lbr_eval_every_iters == 0:
                metrics.update(self.estimate_lbr())

            history.append(metrics)

        return history

    def collect_trajectories(self, n_trajectories: int) -> None:
        remaining = n_trajectories
        while remaining > 0:
            batch = min(self.cfg.parallel_games, remaining)
            self._collect_episode_batch(batch)
            remaining -= batch

    def _collect_episode_batch(self, batch_size: int) -> None:
        self.env.reset(batch_size=batch_size)

        step_records: list[dict[str, torch.Tensor]] = []
        p0_prefix = torch.ones((batch_size,), dtype=torch.float32, device=self.device)
        p1_prefix = torch.ones((batch_size,), dtype=torch.float32, device=self.device)
        q_reach = torch.ones((batch_size,), dtype=torch.float32, device=self.device)

        for _ in range(self.cfg.max_steps_per_trajectory):
            active = ~self.env.done
            if not active.any():
                break

            state = self.env.get_state_tensor()
            legal = self.env.legal_action_mask()
            players = self.env.current_player.clone()

            with torch.no_grad(), self._autocast_ctx():
                advantages = self.adv_net(state)
            policy = regret_matching_plus(advantages, legal)
            if (~active).any():
                policy[~active] = 0.0
                policy[~active, 1] = 1.0
            action = sample_actions(policy)

            chosen_prob = policy.gather(1, action.unsqueeze(1)).squeeze(1).clamp_min(1e-8)

            step_records.append(
                {
                    "state": state.detach().cpu(),
                    "legal": legal.detach().cpu(),
                    "policy": policy.detach().cpu(),
                    "action": action.detach().cpu(),
                    "player": players.detach().cpu(),
                    "active": active.detach().cpu(),
                    "p0_prefix": p0_prefix.detach().cpu(),
                    "p1_prefix": p1_prefix.detach().cpu(),
                    "chosen_prob": chosen_prob.detach().cpu(),
                }
            )

            q_reach = q_reach * torch.where(active, chosen_prob, torch.ones_like(chosen_prob))
            p0_prefix = p0_prefix * torch.where(active & (players == 0), chosen_prob, torch.ones_like(chosen_prob))
            p1_prefix = p1_prefix * torch.where(active & (players == 1), chosen_prob, torch.ones_like(chosen_prob))

            self.env.step(action)

        terminal_u0 = self.env.terminal_utility.detach().cpu().to(torch.float32)
        q_total = q_reach.detach().cpu().clamp_min(1e-8)

        for rec in reversed(step_records):
            active = rec["active"]
            if not active.any():
                continue

            state = rec["state"][active]
            legal = rec["legal"][active]
            policy = rec["policy"][active]
            action = rec["action"][active]
            player = rec["player"][active]
            p0_pre = rec["p0_prefix"][active]
            p1_pre = rec["p1_prefix"][active]
            chosen_prob = rec["chosen_prob"][active].clamp_min(1e-8)

            u0 = terminal_u0[active]
            q = q_total[active]

            u_actor = torch.where(player == 0, u0, -u0)
            opp_prefix = torch.where(player == 0, p1_pre, p0_pre)

            # Outcome-sampling MCCFR estimator for sampled regrets at visited infosets.
            weight = u_actor * opp_prefix / q

            regrets = torch.where(
                legal,
                -weight.unsqueeze(1),
                torch.zeros_like(policy),
            )
            idx = torch.arange(action.shape[0])
            regrets[idx, action] = weight * (1.0 / chosen_prob - 1.0)

            self.adv_buffer.add_batch(state, regrets)
            self.strategy_buffer.add_batch(state, policy)

    def train_step(self) -> dict[str, float]:
        if len(self.adv_buffer) < self.cfg.batch_size or len(self.strategy_buffer) < self.cfg.batch_size:
            return {"adv_loss": 0.0, "policy_loss": 0.0}

        adv_states, adv_targets = self.adv_buffer.sample(self.cfg.batch_size, out_device=self.device)
        strat_states, strat_targets = self.strategy_buffer.sample(self.cfg.batch_size, out_device=self.device)

        self.adv_opt.zero_grad(set_to_none=True)
        with self._autocast_ctx():
            adv_pred = self.adv_net(adv_states)
            adv_loss = F.mse_loss(adv_pred, adv_targets)
        self._backward_and_step(adv_loss, self.adv_opt)

        self.policy_opt.zero_grad(set_to_none=True)
        with self._autocast_ctx():
            logits = self.policy_net(strat_states)
            policy_loss = masked_kl_policy_loss(logits, strat_targets)
        self._backward_and_step(policy_loss, self.policy_opt)

        return {
            "adv_loss": float(adv_loss.detach().cpu()),
            "policy_loss": float(policy_loss.detach().cpu()),
        }

    def estimate_lbr(self) -> dict[str, float]:
        if self.lbr_evaluator is None:
            return {}
        return self.lbr_evaluator.estimate(self.policy_net)

    def _backward_and_step(self, loss: torch.Tensor, opt: torch.optim.Optimizer) -> None:
        if self.scaler.is_enabled():
            self.scaler.scale(loss).backward()
            self.scaler.step(opt)
            self.scaler.update()
        else:
            loss.backward()
            opt.step()

    def _resolve_device(self, configured: str) -> torch.device:
        if configured == "cuda" and not torch.cuda.is_available():
            return torch.device("cpu")
        return torch.device(configured)

    def _autocast_ctx(self):
        if self.device.type != "cuda" or self.cfg.precision == "fp32":
            return torch.autocast(device_type=self.device.type, enabled=False)
        if self.cfg.precision == "bf16":
            return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if self.cfg.precision == "fp16":
            return torch.autocast(device_type="cuda", dtype=torch.float16)
        raise ValueError(f"Unsupported precision mode: {self.cfg.precision}")
