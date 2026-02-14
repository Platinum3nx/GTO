from pathlib import Path

from gto.train import DeepCFRConfig, DeepCFRTrainer


def test_periodic_checkpoint_and_resume_round_trip(tmp_path: Path) -> None:
    checkpoint_dir = tmp_path / "checkpoints"

    cfg = DeepCFRConfig(
        iterations=1,
        trajectories_per_iter=8,
        parallel_games=8,
        max_steps_per_trajectory=6,
        batch_size=8,
        updates_per_cycle=1,
        device="cpu",
        precision="fp32",
        exact_showdown=False,
        lbr_enabled=False,
        checkpoint_dir=str(checkpoint_dir),
        checkpoint_every_trajectories=4,
        handle_sigusr1=False,
    )
    trainer = DeepCFRTrainer(cfg)
    trainer.train()

    latest = checkpoint_dir / "latest.pt"
    assert latest.exists()

    manual = trainer.save_checkpoint(reason="manual")
    assert manual.exists()

    cfg_resume = DeepCFRConfig(
        iterations=1,
        trajectories_per_iter=8,
        parallel_games=8,
        max_steps_per_trajectory=6,
        batch_size=8,
        updates_per_cycle=1,
        device="cpu",
        precision="fp32",
        exact_showdown=False,
        lbr_enabled=False,
        checkpoint_dir=str(checkpoint_dir),
        checkpoint_every_trajectories=4,
        resume_from=str(manual),
        handle_sigusr1=False,
    )
    resumed = DeepCFRTrainer(cfg_resume)

    assert resumed.total_trajectories == trainer.total_trajectories
    assert len(resumed.adv_buffer) == len(trainer.adv_buffer)
    assert len(resumed.strategy_buffer) == len(trainer.strategy_buffer)


def test_preemption_flag_interrupts_collection(tmp_path: Path) -> None:
    cfg = DeepCFRConfig(
        iterations=1,
        trajectories_per_iter=16,
        parallel_games=8,
        max_steps_per_trajectory=4,
        batch_size=8,
        updates_per_cycle=1,
        device="cpu",
        precision="fp32",
        exact_showdown=False,
        lbr_enabled=False,
        checkpoint_dir=str(tmp_path / "checkpoints"),
        checkpoint_every_trajectories=100,
        handle_sigusr1=False,
    )
    trainer = DeepCFRTrainer(cfg)
    trainer._handle_sigusr1(10, None)

    interrupted = trainer.collect_trajectories(16)
    assert interrupted
