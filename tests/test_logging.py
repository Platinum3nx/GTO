from pathlib import Path

import pytest

from gto.train import DeepCFRConfig, DeepCFRTrainer


def test_tensorboard_event_file_created(tmp_path: Path, monkeypatch) -> None:
    pytest.importorskip("tensorboard")
    monkeypatch.setenv("SLURM_JOB_ID", "unit-test-job")
    run_root = tmp_path / "runs"

    cfg = DeepCFRConfig(
        iterations=1,
        trajectories_per_iter=8,
        parallel_games=8,
        max_steps_per_trajectory=4,
        batch_size=8,
        buffer_capacity=128,
        updates_per_cycle=1,
        device="cpu",
        precision="fp32",
        exact_showdown=False,
        lbr_enabled=True,
        lbr_eval_every_iters=1,
        lbr_eval_games=4,
        lbr_rollouts_per_action=1,
        handle_sigusr1=False,
        logging_enabled=True,
        run_dir_root=str(run_root),
        log_flush_every_iters=1,
    )

    trainer = DeepCFRTrainer(cfg)
    trainer.train()
    trainer.close()

    run_dir = run_root / "unit-test-job"
    assert run_dir.exists()
    event_files = list(run_dir.glob("events.out.tfevents.*"))
    assert event_files
