import torch


def regret_matching_plus(
    advantages: torch.Tensor,
    legal_action_mask: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Regret Matching+ policy from predicted cumulative regrets."""
    legal = legal_action_mask.to(dtype=advantages.dtype)
    pos = torch.relu(advantages) * legal
    denom = pos.sum(dim=-1, keepdim=True)

    uniform = legal / legal.sum(dim=-1, keepdim=True).clamp_min(eps)
    policy = torch.where(denom > eps, pos / denom.clamp_min(eps), uniform)
    return policy


def regret_matching(advantages: torch.Tensor, legal_action_mask: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Backward-compatible alias."""
    return regret_matching_plus(advantages, legal_action_mask, eps=eps)


def sample_actions(action_probs: torch.Tensor) -> torch.Tensor:
    """Sample one discrete action per row from a probability matrix."""
    return torch.multinomial(action_probs, num_samples=1).squeeze(1)
