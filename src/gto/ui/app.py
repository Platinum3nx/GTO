import logging
import os
from contextlib import asynccontextmanager
from typing import List, Dict, Optional

import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from gto.constants import ActionBucket, N_ACTIONS
from gto.env import VectorHUNLEnv
from gto.models import PolicyNetwork
from gto.cards import parse_card_sequence, card_to_index, index_to_card

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Global State ---
MODEL: Optional[PolicyNetwork] = None
DEVICE = "cpu"  # Inference on CPU is fine for single requests

# --- Pydantic Models ---
class StrategyRequest(BaseModel):
    board: str  # e.g., "Ks 8h 2c" (Flo p)
    pot: float = 6.0
    stack: float = 100.0  # Effective stack per player
    history: List[str] = []  # e.g., ["CHECK", "BET_33"]
    hole_cards: Optional[str] = None  # e.g., "As Kd" - if provided, returns stats for this hand

class StrategyResponse(BaseModel):
    strategy: Dict[str, float]  # Action -> Probability
    legal_actions: List[str]
    ev: Optional[float] = None
    street: str
    grid: Optional[Dict[str, Dict[str, float]]] = None # Hand -> Action -> Prob

# --- Action Mapping ---
ACTION_MAP = {
    "FOLD": ActionBucket.FOLD,
    "CHECK": ActionBucket.CHECK_CALL,
    "CALL": ActionBucket.CHECK_CALL,
    "BET_33": ActionBucket.BET_33,
    "BET_75": ActionBucket.BET_75,
    "ALL_IN": ActionBucket.ALL_IN,
}
REVERSE_ACTION_MAP = {v: k for k, v in ACTION_MAP.items()}
# Handle aliases (CHECK/CALL often map to same bucket)
REVERSE_ACTION_MAP[ActionBucket.CHECK_CALL] = "CHECK/CALL" 

# --- Lifespan & Startup ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load model on startup
    global MODEL
    try:
        model_path = get_latest_checkpoint()
        if model_path:
            logger.info(f"Loading model from {model_path}...")
            # Initialize minimal policy network
            # Note: We assume standard 199 input dim from constants
            net = PolicyNetwork()
            checkpoint = torch.load(model_path, map_location=DEVICE)
            
            # Checkpoint structure usually has "policy_model" key
            state_dict = checkpoint.get("policy_model", checkpoint)
            net.load_state_dict(state_dict)
            net.eval()
            MODEL = net
            logger.info("Model loaded successfully.")
        else:
            logger.warning("No checkpoints found! API will define model but inference will fail.")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
    
    yield
    # Cleanup if needed

def get_latest_checkpoint(checkpoint_dir: str = "checkpoints") -> str | None:
    if not os.path.isdir(checkpoint_dir):
        return None
    files = [os.path.join(checkpoint_dir, f) for f in os.listdir(checkpoint_dir) if f.endswith(".pt")]
    if not files:
        return None
    # Sort by modification time (or name could be better if numbered)
    return max(files, key=os.path.getmtime)


# --- Application ---
app = FastAPI(lifespan=lifespan)

# Normalize API actions to Enum
def parse_action(action_str: str) -> int:
    norm = action_str.upper().replace(" ", "_")
    if norm in ACTION_MAP:
        return int(ACTION_MAP[norm])
    # Try exact match if user sends "CHECK_CALL"
    if norm == "CHECK_CALL": return int(ActionBucket.CHECK_CALL)
    raise ValueError(f"Unknown action: {action_str}")

    # If hole_cards provided, run single inference.
    # If not, run batch inference for all 169 canonical hands (13x13 grid).
    if req.hole_cards:
        batch_size = 1
        hands_to_solve = [req.hole_cards]
    else:
        # Generate 169 canonical hands
        batch_size = 169
        hands_to_solve = _generate_canonical_169()

    # 1. Initialize Environment
    try:
        flop_indices = parse_card_sequence(req.board)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    env = VectorHUNLEnv(
        batch_size=batch_size,
        device=DEVICE,
        stack_bb=req.stack,
        initial_pot_bb=req.pot,
        fixed_flop_cards=flop_indices,
        small_blind=0.5,
        big_blind=1.0
    )

    # 2. Set Hole Cards
    for i, hand_str in enumerate(hands_to_solve):
        try:
            cards = parse_card_sequence(hand_str)
            env.hole_cards[i, 0] = cards[0]
            env.hole_cards[i, 1] = cards[1]
        except ValueError:
            pass # Should not happen with canonical gen

    # 3. Replay History
    for act_str in req.history:
        try:
            a_idx = parse_action(act_str)
            # Apply action to ALL batch elements
            # This assumes the history is the same for the entire range (which is true for GTO analysis)
            action_tensor = torch.full((batch_size,), a_idx, dtype=torch.long, device=DEVICE)
            env.step(action_tensor)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid history action '{act_str}': {e}")
        
        if env.done.any():
             raise HTTPException(status_code=400, detail="Game ended during history replay for some hands")

    # 4. Inference
    state = env.get_state_tensor()
    legal_mask = env.legal_action_mask()

    with torch.no_grad():
        probs = MODEL.action_probs(state, legal_action_mask=legal_mask)
    
    probs_np = probs.cpu().numpy()
    legal_np = legal_mask.cpu().numpy().astype(bool)

    # 5. Format Output
    # If batch, return dict mapping Hand -> Strategy
    if batch_size > 1:
        grid_data = {}
        for i, hand in enumerate(hands_to_solve):
            # Convert generic strict hand (AsAh) to friendly (AA)
            friendly_hand = _canonical_to_friendly(hand)
            
            strat = {}
            for act_idx in range(N_ACTIONS):
                if legal_np[i, act_idx]:
                    name = REVERSE_ACTION_MAP.get(act_idx, f"ACT_{act_idx}")
                    strat[name] = float(probs_np[i, act_idx])
            grid_data[friendly_hand] = strat
        
        return StrategyResponse(
            strategy={"ALL": 0.0}, # Dummy for schema compatibility or use a new field
            grid=grid_data,
            legal_actions=list(REVERSE_ACTION_MAP.values()), # All possible
            street="FLOP" # Simplified
        )
    else:
        # Single hand logic
        metrics = {}
        legal_actions_list = []
        for i in range(N_ACTIONS):
            if legal_np[0, i]:
                name = REVERSE_ACTION_MAP.get(i, f"ACTION_{i}")
                metrics[name] = float(probs_np[0, i])
                legal_actions_list.append(name)
        
        return StrategyResponse(
            strategy=metrics,
            legal_actions=legal_actions_list,
            street="FLOP" # Todo: get real street
        )

def _canonical_to_friendly(hand: str) -> str:
    # AsAh -> AA
    # AsKs -> AKs
    # AsKh -> AKo
    if len(hand) != 4: return hand
    r1, s1, r2, s2 = hand[0], hand[1], hand[2], hand[3]
    if r1 == r2: return r1 + r2
    if s1 == s2: return r1 + r2 + "s"
    return r1 + r2 + "o"

def _generate_canonical_169() -> List[str]:
    """Generate generic representations: 'AA', 'AKs', 'AKo'."""
    # Note: parse_card_sequence expects real cards like 'AsAh'.
    # We need to pick CONCRETE suits for the canonical forms to run through the net.
    # Pairs: AsAh (Suits don't matter much for pre-flop/flop generic unless flush draw)
    # Suited: AsKs
    # Offsuit: AsKh
    ranks = "AKQJT98765432"
    hands = []
    
    # We construct the 13x13 grid in row-major order (AA, AKs... then KAs, KK...)
    # Actually typical grid is:
    # AA  AKs AQs ...
    # AKo KK  KQs ...
    # AQo KQo QQ ...
    
    # We will just return a list and let frontend map it.
    # But wait, the frontend sends "AsKd".
    # If we return "AsKs", "AsKh", "AsAh" that works.
    
    for r1 in ranks:
        for r2 in ranks:
            if r1 == r2:
                # Pair: AsAh
                hands.append(f"{r1}s{r2}h")
            elif ranks.index(r1) < ranks.index(r2):
                # Suited (Upper Triangle): AsKs
                hands.append(f"{r1}s{r2}s")
            else:
                # Offsuit (Lower Triangle): AsKh
                hands.append(f"{r1}s{r2}h")
    return hands

# --- Static Files ---
# Mount the frontend
app.mount("/", StaticFiles(directory="src/gto/ui/static", html=True), name="static")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
