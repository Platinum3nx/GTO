import random

import torch


class ReservoirBuffer:
    """Fixed-size reservoir sampling buffer for Deep CFR targets."""

    def __init__(
        self,
        capacity: int,
        state_dim: int,
        target_dim: int,
        *,
        dtype: torch.dtype = torch.float32,
        device: torch.device | str = "cpu",
    ) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self.capacity = capacity
        self.state_dim = state_dim
        self.target_dim = target_dim
        self.dtype = dtype
        self.device = torch.device(device)

        self.states = torch.zeros((capacity, state_dim), dtype=dtype, device=self.device)
        self.targets = torch.zeros((capacity, target_dim), dtype=dtype, device=self.device)

        self._size = 0
        self._seen = 0

    def __len__(self) -> int:
        return self._size

    @property
    def seen(self) -> int:
        return self._seen

    def add(self, state: torch.Tensor, target: torch.Tensor) -> None:
        state = state.detach().to(device=self.device, dtype=self.dtype).view(-1)
        target = target.detach().to(device=self.device, dtype=self.dtype).view(-1)

        if state.numel() != self.state_dim:
            raise ValueError(f"state must have {self.state_dim} features")
        if target.numel() != self.target_dim:
            raise ValueError(f"target must have {self.target_dim} features")

        index = self._reservoir_index()
        if index is None:
            return
        self.states[index] = state
        self.targets[index] = target

    def add_batch(self, states: torch.Tensor, targets: torch.Tensor) -> None:
        if states.ndim != 2 or states.shape[1] != self.state_dim:
            raise ValueError(f"states must have shape [B, {self.state_dim}]")
        if targets.ndim != 2 or targets.shape[1] != self.target_dim:
            raise ValueError(f"targets must have shape [B, {self.target_dim}]")

        for i in range(states.shape[0]):
            self.add(states[i], targets[i])

    def sample(
        self,
        batch_size: int,
        *,
        out_device: torch.device | str | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self._size == 0:
            raise RuntimeError("Cannot sample from an empty buffer")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")

        idx = torch.randint(0, self._size, (batch_size,), device=self.device)
        states = self.states[idx]
        targets = self.targets[idx]

        if out_device is not None:
            d = torch.device(out_device)
            states = states.to(d, non_blocking=True)
            targets = targets.to(d, non_blocking=True)

        return states, targets

    def _reservoir_index(self) -> int | None:
        self._seen += 1

        if self._size < self.capacity:
            idx = self._size
            self._size += 1
            return idx

        candidate = random.randint(0, self._seen - 1)
        if candidate < self.capacity:
            return candidate
        return None
