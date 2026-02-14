import torch

from gto.constants import N_ACTIONS, STATE_DIM
from gto.env import VectorHUNLEnv


def test_env_reset_and_step_shapes() -> None:
    env = VectorHUNLEnv(batch_size=32, device="cpu", exact_showdown=False)
    state = env.reset()

    assert state.shape == (32, STATE_DIM)

    legal = env.legal_action_mask()
    assert legal.shape == (32, N_ACTIONS)

    actions = torch.argmax(legal.to(torch.int64), dim=1)
    next_state, done, reward = env.step(actions)

    assert next_state.shape == (32, STATE_DIM)
    assert done.shape == (32,)
    assert reward.shape == (32,)
