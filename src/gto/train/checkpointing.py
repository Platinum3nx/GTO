from __future__ import annotations

import os
import shutil
from pathlib import Path

import torch


class CheckpointManager:
    def __init__(self, checkpoint_dir: str | Path, *, prefix: str = "deepcfr", keep_last: int = 5) -> None:
        self.checkpoint_dir = Path(checkpoint_dir)
        self.prefix = prefix
        self.keep_last = max(int(keep_last), 0)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def save(
        self,
        payload: dict[str, object],
        *,
        iteration: int,
        trajectories: int,
        reason: str,
    ) -> Path:
        name = f"{self.prefix}_iter{iteration:06d}_traj{trajectories:012d}_{reason}.pt"
        dst = self.checkpoint_dir / name
        tmp = self.checkpoint_dir / f".{name}.tmp"

        torch.save(payload, tmp)
        os.replace(tmp, dst)

        latest = self.checkpoint_dir / "latest.pt"
        self._update_latest_pointer(latest, dst)

        self._prune_old_checkpoints()
        return dst

    def load(self, path: str | Path) -> dict[str, object]:
        src = Path(path)
        if not src.exists():
            raise FileNotFoundError(f"Checkpoint file not found: {src}")
        return torch.load(src, map_location="cpu")

    def latest_path(self) -> Path | None:
        latest = self.checkpoint_dir / "latest.pt"
        if latest.exists():
            return latest

        candidates = sorted(self.checkpoint_dir.glob(f"{self.prefix}_iter*_traj*_*.pt"))
        if not candidates:
            return None
        return candidates[-1]

    def _prune_old_checkpoints(self) -> None:
        if self.keep_last <= 0:
            return

        ckpts = sorted(self.checkpoint_dir.glob(f"{self.prefix}_iter*_traj*_*.pt"))
        if len(ckpts) <= self.keep_last:
            return

        for old in ckpts[: len(ckpts) - self.keep_last]:
            old.unlink(missing_ok=True)

    def _update_latest_pointer(self, latest: Path, dst: Path) -> None:
        if latest.exists() or latest.is_symlink():
            latest.unlink()

        try:
            latest.symlink_to(dst.name)
        except OSError:
            shutil.copy2(dst, latest)
