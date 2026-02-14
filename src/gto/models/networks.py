import torch
from torch import nn
from torch.nn import functional as F

from gto.constants import N_ACTIONS, STATE_DIM


class _MLPTrunk(nn.Module):
    def __init__(self, input_dim: int = STATE_DIM, hidden_dim: int = 512, hidden_layers: int = 4) -> None:
        super().__init__()
        layers: list[nn.Module] = [nn.Linear(input_dim, hidden_dim), nn.LeakyReLU()]
        for _ in range(hidden_layers):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.LeakyReLU()])
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class AdvantageNetwork(nn.Module):
    def __init__(self, state_dim: int = STATE_DIM, n_actions: int = N_ACTIONS) -> None:
        super().__init__()
        self.trunk = _MLPTrunk(input_dim=state_dim)
        self.head = nn.Linear(512, n_actions)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.head(self.trunk(state))


class PolicyNetwork(nn.Module):
    def __init__(self, state_dim: int = STATE_DIM, n_actions: int = N_ACTIONS) -> None:
        super().__init__()
        self.trunk = _MLPTrunk(input_dim=state_dim)
        self.head = nn.Linear(512, n_actions)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.head(self.trunk(state))

    def action_probs(self, state: torch.Tensor, legal_action_mask: torch.Tensor | None = None) -> torch.Tensor:
        logits = self.forward(state)
        if legal_action_mask is not None:
            logits = logits.masked_fill(~legal_action_mask.bool(), float("-inf"))
        return torch.softmax(logits, dim=-1)


def masked_kl_policy_loss(
    logits: torch.Tensor,
    target_probs: torch.Tensor,
    legal_action_mask: torch.Tensor | None = None,
    eps: float = 1e-8,
) -> torch.Tensor:
    """KL(P_target || P_model) for CFR soft targets."""
    if legal_action_mask is not None:
        mask = legal_action_mask.bool()
        logits = logits.masked_fill(~mask, -1e9)
        target_probs = target_probs * mask

    target_probs = target_probs.clamp_min(0.0)
    target_probs = target_probs / target_probs.sum(dim=-1, keepdim=True).clamp_min(eps)
    log_probs = F.log_softmax(logits, dim=-1)
    return F.kl_div(log_probs, target_probs, reduction="batchmean")
