# GTO ♠️

> Deep CFR solver for Heads-Up No-Limit Texas Hold'em subgames — learn Nash Equilibrium strategies from scratch via self-play.

## Architecture

```mermaid
flowchart TB
    subgraph ENV["🃏 Vectorized HUNL Environment"]
        direction LR
        DEAL["Deal Flop +<br>Hole Cards"]
        SIM["1024 Parallel<br>Games (CUDA)"]
        SD["7-Card Showdown<br>LUT Evaluator"]
        DEAL --> SIM --> SD
    end

    subgraph ENCODE["📐 State Encoder (199 Features)"]
        direction LR
        CARDS["Card Encoding<br>156 features"]
        BET["Betting State<br>8 features"]
        HIST["Action History<br>35 features"]
    end

    subgraph NETS["🧠 Neural Networks (PyTorch)"]
        direction LR
        ADV["Advantage Net<br>199→512x4→5<br>MSE Loss"]
        POL["Policy Net<br>199→512x4→5<br>KL Loss"]
    end

    subgraph CFR["♻️ MCCFR Training Loop"]
        direction TB
        RM["Regret Matching+<br>Policy"]
        SAMPLE["Outcome Sampling<br>Tree Traversal"]
        REGRET["Counterfactual<br>Regret Backup"]
        RM --> SAMPLE --> REGRET
    end

    subgraph BUFFERS["💾 Reservoir Buffers (1M)"]
        direction LR
        ABUF["Advantage Buffer<br>(state, regrets)"]
        SBUF["Strategy Buffer<br>(state, policy)"]
    end

    subgraph EVAL["📊 Evaluation & Infra"]
        direction LR
        LBR["LBR Metric<br>(Exploitability)"]
        CKPT["Checkpointing<br>Periodic+SIGUSR1"]
        TB["TensorBoard<br>Loss+LBR"]
    end

    ENV -->|"game state"| ENCODE
    ENCODE -->|"tensor"| NETS
    ADV -->|"advantages"| CFR
    CFR -->|"regrets"| ABUF
    CFR -->|"policy"| SBUF
    ABUF -->|"batch"| ADV
    SBUF -->|"batch"| POL
    POL -.->|"evaluate"| LBR
    NETS -.->|"weights"| CKPT
    NETS -.->|"logs"| TB

    style ENV fill:#1a1a2e,stroke:#e94560,color:#eee
    style ENCODE fill:#16213e,stroke:#0f3460,color:#eee
    style NETS fill:#0f3460,stroke:#533483,color:#eee
    style CFR fill:#533483,stroke:#e94560,color:#eee
    style BUFFERS fill:#1a1a2e,stroke:#0f3460,color:#eee
    style EVAL fill:#16213e,stroke:#533483,color:#eee
```

## Key Specs

| Component | Detail |
|---|---|
| **State Vector** | `199` features (156 cards + 8 betting + 35 history) |
| **Action Space** | Fold · Check/Call · Bet 33% · Bet 75% · All-In |
| **Networks** | 4-hidden-layer MLPs, 512 units, LeakyReLU |
| **Losses** | MSE (advantage regrets) · KL divergence (policy targets) |
| **Traversal** | Outcome-sampling MCCFR with Regret Matching+ |
| **Replay** | 1M-capacity reservoir sampling buffers |
| **Showdown** | Exact 7-card batched evaluator via C(52,5) LUT |
| **Evaluation** | Local Best Response exploitability estimate |
| **Precision** | bf16 mixed-precision on A100 (CUDA) |
| **Infra** | SIGUSR1-aware checkpointing · TensorBoard logging · SLURM-ready |

## Install
```bash
pip install -e .
pip install -e .[dev]
```

## Run Tests
```bash
pytest
```

## Training
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
