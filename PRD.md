# Deep CFR Backend Architecture & State Schema

## 1. Scope & Game Setup
- Variant: Heads-up No-Limit Texas Hold'em (HUNL) subgame solving.
- Training start point: Flop-only states (preflop ranges are provided externally).
- Stack/blinds: 100bb effective stacks, 0.5 / 1.0 blinds.
- Action abstraction (`N_actions=5`):
  - Fold
  - Check/Call
  - Bet 33% Pot
  - Bet 75% Pot
  - All-In

## 2. State Vector Representation (Tensor Schema)
The environment must encode the game state into a flat 1D tensor for the neural networks. All inputs must be normalized.

### A. Card Encoding (156 features)
Use a flat 1D tensor representing a standard 52-card deck.
- Player Hole Cards (52 features): 1 for present, 0 for absent.
- Board Cards (52 features): 1 for present, 0 for absent (zeros for unrevealed cards).
- Opponent Range Mask (52 features): card-removal mask, 1 if card is still mathematically possible, 0 if blocked/dead.

### B. Betting & Pot State (8 features)
Normalize all financial values by the initial effective stack.
- Current Pot Size (1 feature)
- Player Stack (1 feature)
- Opponent Stack (1 feature)
- Amount to Call (1 feature)
- Street Indicator (4 features): one-hot for Preflop, Flop, Turn, River.

### C. Action History Sequence (35 features)
Use a fixed-length vector for the last 5 actions.
- For each action, encode `(player_id, action_type_one_hot, bet_size_fraction)`.
- `player_id`: 1 feature (0 or 1)
- `action_type_one_hot`: 5 features (Fold, Check/Call, Bet 33%, Bet 75%, All-In)
- `bet_size_fraction`: 1 feature
- Total per action: 7 features
- Total history block: `5 * 7 = 35` features
- Pad with zeros if fewer than 5 actions have occurred.

Total State Vector Size: `156 + 8 + 35 = 199` features.

## 3. Neural Network Architectures (PyTorch)

### A. Advantage Network (Regret)
- Input: Linear layer (`199 -> 512`)
- Hidden Layers: `4x` Linear layers (`512 -> 512`) with LeakyReLU activations.
- Output: Linear layer (`512 -> N_actions`).
- Loss Function: Mean Squared Error (MSE) between output and sampled counterfactual regret.

### B. Policy Network (Average Strategy)
- Input: Linear layer (`199 -> 512`)
- Hidden Layers: `4x` Linear layers (`512 -> 512`) with LeakyReLU activations.
- Output: Linear layer (`512 -> N_actions`) with softmax applied at inference/sampling.
- Loss Function: KL Divergence against CFR soft targets (average strategy distribution).

## 4. Training Loop (MCCFR with Outcome Sampling)
1. Initialize Advantage Buffer (size 1M) and Strategy Buffer (size 1M) for reservoir sampling.
2. For iteration `1..N`:
   - Generate trajectories (self-play) from flop states using Advantage Network + regret matching.
   - Compute terminal utilities (showdown/fold outcomes).
   - Traverse sampled paths backward and compute counterfactual regrets.
   - Store `(State_Vector, Action_Regrets)` in Advantage Buffer.
   - Store `(State_Vector, Action_Probabilities)` in Strategy Buffer.
3. Every `K` trajectories:
   - Sample batch (e.g., 4096) from Advantage Buffer and update Advantage Network.
   - Sample batch from Strategy Buffer and update Policy Network.

## 5. Compute Constraints & Optimizations
- Device target: CUDA (A100 on SLURM).
- Precision target: bf16 first (Ampere-native), with mixed precision training.
- Parallelization: vectorized `step()` and rollout logic with PyTorch tensors to support 1024+ parallel games.
- CPU loops across per-game simulation paths are a bottleneck and should be avoided on hot paths.

## 6. Evaluation Metrics
Primary success criteria:
- Local Best Response (LBR) exploitability estimate (distance from Nash behavior).
- Training loss convergence for both advantage and policy networks.
