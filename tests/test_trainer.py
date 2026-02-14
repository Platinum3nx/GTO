from gto.train import DeepCFRConfig, DeepCFRTrainer


def test_trainer_collects_outcome_sampling_targets() -> None:
    cfg = DeepCFRConfig(
        iterations=1,
        trajectories_per_iter=32,
        parallel_games=16,
        max_steps_per_trajectory=8,
        batch_size=16,
        updates_per_cycle=1,
        device="cpu",
        precision="fp32",
        exact_showdown=False,
    )
    trainer = DeepCFRTrainer(cfg)

    trainer.collect_trajectories(cfg.trajectories_per_iter)

    assert len(trainer.adv_buffer) > 0
    assert len(trainer.strategy_buffer) > 0

    metrics = trainer.train_step()
    assert "adv_loss" in metrics
    assert "policy_loss" in metrics
