import torch

from gto.replay import ReservoirBuffer


def test_reservoir_buffer_add_and_sample() -> None:
    buf = ReservoirBuffer(capacity=100, state_dim=199, target_dim=5)

    states = torch.randn(32, 199)
    targets = torch.randn(32, 5)
    buf.add_batch(states, targets)

    assert len(buf) == 32
    s, t = buf.sample(16)
    assert s.shape == (16, 199)
    assert t.shape == (16, 5)
