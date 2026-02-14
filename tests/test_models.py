import torch

from gto.constants import N_ACTIONS, STATE_DIM
from gto.models import AdvantageNetwork, PolicyNetwork, masked_kl_policy_loss


def test_network_shapes() -> None:
    x = torch.randn(16, STATE_DIM)
    adv = AdvantageNetwork()
    pol = PolicyNetwork()

    adv_out = adv(x)
    pol_out = pol(x)

    assert adv_out.shape == (16, N_ACTIONS)
    assert pol_out.shape == (16, N_ACTIONS)


def test_kl_policy_loss_runs() -> None:
    logits = torch.randn(8, N_ACTIONS)
    target = torch.softmax(torch.randn(8, N_ACTIONS), dim=-1)
    loss = masked_kl_policy_loss(logits, target)
    assert torch.isfinite(loss)
    assert loss.item() >= 0.0
