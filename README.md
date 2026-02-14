# GTO

Flop and post-flop HUNL subgame solver scaffold using Deep CFR.

## Current Architecture
- Input state size: `199`
- Action abstraction: `5` buckets
  - Fold
  - Check/Call
  - Bet 33% Pot
  - Bet 75% Pot
  - All-In
- Networks: Advantage and Policy MLPs (`199 -> 512 -> ... -> 512 -> 5`)
- Policy loss: KL divergence with soft CFR strategy targets
- Traversal: RM+ policy generation + outcome-sampling counterfactual regret backup
- Replay: 1M-capacity reservoir buffers for advantage and strategy targets
- Environment: vectorized flop-start HUNL subgame simulator API
- Showdown: exact 7-card batched evaluator via tensorized 5-card LUT (`C(52,5)` index)
- Evaluation: Local Best Response (LBR) callback metric (`lbr_exploitability`)
- Precision target: bf16 on A100 (CUDA)
- Checkpointing: periodic trajectory-based saves + SIGUSR1 preemption-triggered final save
- Logging: TensorBoard scalars written to `runs/<SLURM_JOB_ID|timestamp>/`

## Install
```bash
pip install -e .
pip install -e .[dev]
```

## Run Tests
```bash
pytest
```

## CLI
Run training with configurable cluster-friendly arguments:

```bash
python -m gto.train.trainer \
  --subgame_board Kh8s2c \
  --learning_rate 3e-4 \
  --batch_size 4096 \
  --iterations 200 \
  --checkpoint_dir checkpoints \
  --resume_from checkpoints/latest.pt
```

## Notes
- This repository currently provides a training scaffold and interfaces.
- LUT build can take time on first initialization, then reuses cached tensors per process/device.
- Checkpoints include model weights, optimizer states, and both reservoir buffers.
