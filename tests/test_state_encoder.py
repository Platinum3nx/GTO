import torch

from gto.constants import STATE_DIM
from gto.state_encoder import StateEncoder
from gto.types import ActionEvent, GameState


def test_state_encoder_shape_and_history_one_hot() -> None:
    state = GameState(
        player_hole_cards=[0, 12],
        board_cards=[20, 21, 22],
        pot_size=12.0,
        player_stack=88.0,
        opponent_stack=88.0,
        amount_to_call=4.0,
        street=1,
        action_history=[
            ActionEvent(player_id=0, action_type=1, bet_fraction=0.0),
            ActionEvent(player_id=1, action_type=3, bet_fraction=0.75),
        ],
        initial_effective_stack=100.0,
    )

    enc = StateEncoder(dtype=torch.float32)
    vec = enc.encode(state)

    assert vec.shape == (STATE_DIM,)
    assert torch.isclose(vec.sum(), vec.sum())  # no NaNs

    history = vec[164:199].reshape(5, 7)
    assert torch.all(history[:3] == 0)
    assert history[3, 0].item() == 0.0
    assert history[3, 2].item() == 1.0  # action_type=1 one-hot offset
    assert history[4, 0].item() == 1.0
    assert history[4, 4].item() == 1.0  # action_type=3 one-hot offset
