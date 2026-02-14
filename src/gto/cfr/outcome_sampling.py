import torch


def regret_matching(advantages: torch.Tensor, legal_action_mask: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Convert predicted advantages to a behavior policy via regret matching."""
    legal = legal_action_mask.to(dtype=advantages.dtype)
    pos = torch.relu(advantages) * legal
    denom = pos.sum(dim=-1, keepdim=True)

    uniform = legal / legal.sum(dim=-1, keepdim=True).clamp_min(eps)
    policy = torch.where(denom > eps, pos / denom.clamp_min(eps), uniform)
    return policy


def sample_actions(action_probs: torch.Tensor) -> torch.Tensor:
    """Sample one discrete action per row from a probability matrix."""
    return torch.multinomial(action_probs, num_samples=1).squeeze(1)
