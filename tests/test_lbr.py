from gto.train import DeepCFRConfig, DeepCFRTrainer


def test_lbr_callback_metrics_exist() -> None:
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
        lbr_enabled=True,
        lbr_eval_every_iters=1,
        lbr_eval_games=4,
        lbr_rollouts_per_action=1,
    )
    trainer = DeepCFRTrainer(cfg)

    metrics = trainer.estimate_lbr()

    assert "lbr_p0" in metrics
    assert "lbr_p1" in metrics
    assert "lbr_exploitability" in metrics
