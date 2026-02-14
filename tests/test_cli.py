from gto.train.trainer import build_arg_parser, config_from_args


def test_cli_parser_maps_required_arguments() -> None:
    parser = build_arg_parser()
    args = parser.parse_args(
        [
            "--subgame_board",
            "Kh8s2c",
            "--learning_rate",
            "0.0005",
            "--batch_size",
            "256",
            "--iterations",
            "12",
            "--checkpoint_dir",
            "checkpoints/test",
            "--resume_from",
            "checkpoints/latest.pt",
        ]
    )

    cfg = config_from_args(args)

    assert cfg.subgame_board == "Kh8s2c"
    assert cfg.learning_rate == 0.0005
    assert cfg.batch_size == 256
    assert cfg.iterations == 12
    assert cfg.checkpoint_dir == "checkpoints/test"
    assert cfg.resume_from == "checkpoints/latest.pt"
