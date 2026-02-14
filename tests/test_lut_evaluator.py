import torch

from gto.eval import TensorLUTShowdownEvaluator


def test_lut_evaluator_orders_clear_showdown() -> None:
    ev = TensorLUTShowdownEvaluator(device="cpu")

    # Board: Ah Kh Qh Jh 2c
    board = torch.tensor([[50, 46, 42, 38, 0]], dtype=torch.long)
    # P0: Th 9h -> royal flush
    p0 = torch.tensor([[34, 30]], dtype=torch.long)
    # P1: As Ad -> one pair
    p1 = torch.tensor([[51, 49]], dtype=torch.long)

    s0, s1 = ev.compare(p0, p1, board)
    assert int(s0.item()) > int(s1.item())


def test_lut_evaluator_handles_tie() -> None:
    ev = TensorLUTShowdownEvaluator(device="cpu")

    # Board: 2c 3d 4h 5s 9c
    board = torch.tensor([[0, 5, 10, 15, 28]], dtype=torch.long)
    # Both players make wheel straight with an ace
    p0 = torch.tensor([[48, 25]], dtype=torch.long)  # As, 8d
    p1 = torch.tensor([[49, 22]], dtype=torch.long)  # Ad, 7h

    s0, s1 = ev.compare(p0, p1, board)
    assert int(s0.item()) == int(s1.item())
