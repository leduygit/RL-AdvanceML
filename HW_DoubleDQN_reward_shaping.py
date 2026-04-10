import random
import re
from collections import deque, namedtuple

import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import pyspiel

print("PyTorch version:", torch.__version__)
print("OpenSpiel version:", pyspiel.__version__ if hasattr(pyspiel, "__version__") else "unknown")
print("CUDA available:", torch.cuda.is_available())
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", DEVICE)

game = pyspiel.load_game("2048")
state = game.new_initial_state()

game_type = game.get_type()
short_name = game_type.short_name if hasattr(game_type, "short_name") else "2048"

print("Registered game:", short_name)
print("Num distinct actions:", game.num_distinct_actions())
print("Observation tensor shape:", game.observation_tensor_shape())
print("Observation tensor size:", game.observation_tensor_size())
print("Max chance outcomes:", game.max_chance_outcomes())
print("Max game length:", game.max_game_length())
print("Min / Max utility:", game.min_utility(), game.max_utility())
print()
print("Initial state is chance node:", state.is_chance_node())
print("Initial state string:")
print(state)

# def extract_obs(state, player_id=0):
#     """Return log2-scaled observation vector."""
#     for fn_name, args in [
#         ("observation_tensor", (player_id,)),
#         ("observation_tensor", tuple()),
#         ("information_state_tensor", (player_id,)),
#         ("information_state_tensor", tuple()),
#     ]:
#         fn = getattr(state, fn_name, None)
#         if fn is None:
#             continue
#         try:
#             obs = fn(*args)
#             obs = np.asarray(obs, dtype=np.float32).reshape(-1)

#             # 🔥 Convert to log2 scale
#             obs = np.log2(obs + 1.0)

#             return obs
#         except TypeError:
#             pass
#     raise RuntimeError("Could not extract an observation tensor from state.")

def extract_obs(state, player_id=0):
    board = parse_board_numbers(state)
    if board is None:
        raise RuntimeError("Failed to parse board.")

    board = board.astype(np.float32)

    # log2 transform (0 stays 0)
    obs = np.log2(board + 1.0)

    return obs.reshape(-1)


def legal_actions(state, player_id=0) -> list[int]:
    """Return legal actions for the current player state."""
    try:
        return list(state.legal_actions(player_id))
    except TypeError:
        return list(state.legal_actions())


def sample_chance_action(state, rng):
    outcomes = state.chance_outcomes()  # list of (action, prob)
    actions, probs = zip(*outcomes)
    idx = rng.choice(len(actions), p=np.asarray(probs, dtype=np.float64))
    return actions[idx]


def auto_resolve_chance_nodes(state, rng):
    """Mutate state until it is no longer a chance node."""
    while state.is_chance_node() and not state.is_terminal():
        a = sample_chance_action(state, rng)
        state.apply_action(a)
    return state


def state_return(state, player_id=0):
    vals = state.returns()
    return float(vals[player_id]) if len(vals) > player_id else 0.0


def state_reward(state, player_id=0):
    vals = state.rewards()
    return float(vals[player_id]) if len(vals) > player_id else 0.0


def parse_board_numbers(state):
    """Best-effort text parser for showing the board as a 4x4 integer array."""
    txt = str(state)
    nums = [int(x) for x in re.findall(r"\d+", txt)]
    if len(nums) >= 16:
        nums = nums[-16:]
        return np.array(nums, dtype=np.int64).reshape(4, 4)
    return None


def empty_tile_count(board):
    return int(np.sum(board == 0))


def max_tile_in_target_corner(board, target_corner="top_left"):
    n = board.shape[0]
    corner_lookup = {
        "top_left": (0, 0),
        "top_right": (0, n - 1),
        "bottom_left": (n - 1, 0),
        "bottom_right": (n - 1, n - 1),
    }
    i, j = corner_lookup[target_corner]
    return board[i, j] == np.max(board)


def monotonicity_score_toward_corner(board, target_corner="top_left"):
    oriented = board.astype(np.float32)
    if target_corner == "top_right":
        oriented = np.fliplr(oriented)
    elif target_corner == "bottom_left":
        oriented = np.flipud(oriented)
    elif target_corner == "bottom_right":
        oriented = np.flipud(np.fliplr(oriented))

    # Use log2 scale so each doubling has consistent spacing.
    log_board = np.log2(np.maximum(oriented, 1.0))

    row_non_increasing = (log_board[:, :-1] >= log_board[:, 1:]).mean()
    col_non_increasing = (log_board[:-1, :] >= log_board[1:, :]).mean()
    return float(0.5 * (row_non_increasing + col_non_increasing))


# Quick sanity check
test_state = game.new_initial_state()
auto_resolve_chance_nodes(test_state, np.random.default_rng(0))
print("Observation shape after resolving initial chance:", extract_obs(test_state).shape)
print("Legal actions:", legal_actions(test_state))
print("Board (best effort):")
print(parse_board_numbers(test_state))
print()
print(test_state)


class OpenSpiel2048Env:
    def __init__(
        self,
        seed=42,
        target_corner="top_left",
        score_weight=1.0,
        empty_weight=0.05,
        monotonicity_weight=0.5,
        corner_weight=1.0,
    ):
        if target_corner not in {"top_left", "top_right", "bottom_left", "bottom_right"}:
            raise ValueError("target_corner must be one of top_left, top_right, bottom_left, bottom_right")

        self.game = pyspiel.load_game("2048")
        self.player_id = 0
        self.num_actions = self.game.num_distinct_actions()
        self.obs_dim = self.game.observation_tensor_size()
        self.rng = np.random.default_rng(seed)
        self.state = None
        self.target_corner = target_corner
        self.score_weight = float(score_weight)
        self.empty_weight = float(empty_weight)
        self.monotonicity_weight = float(monotonicity_weight)
        self.corner_weight = float(corner_weight)

    def reset(self, seed=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.state = self.game.new_initial_state()
        auto_resolve_chance_nodes(self.state, self.rng)
        return extract_obs(self.state, self.player_id)

    def step(self, action):
        if self.state is None:
            raise RuntimeError("Call reset() before step().")
        if self.state.is_terminal():
            raise RuntimeError("Episode already ended. Call reset().")

        legal = legal_actions(self.state, self.player_id)
        if action not in legal:
            raise ValueError(f"Illegal action {action}. Legal actions: {legal}")

        prev_return = state_return(self.state, self.player_id)
        prev_board = parse_board_numbers(self.state)

        self.state.apply_action(int(action))
        auto_resolve_chance_nodes(self.state, self.rng)

        next_obs = extract_obs(self.state, self.player_id) if not self.state.is_terminal() else np.zeros(self.obs_dim, dtype=np.float32)
        new_return = state_return(self.state, self.player_id)

        board = parse_board_numbers(self.state)
        base_reward = new_return - prev_return

        if board is None:
            empty_bonus = 0.0
            monotonicity_bonus = 0.0
            corner_bonus = 0.0
        else:
            empty_bonus = self.empty_weight * empty_tile_count(board)
            monotonicity_bonus = self.monotonicity_weight * monotonicity_score_toward_corner(
                board, self.target_corner
            )
            corner_bonus = self.corner_weight if max_tile_in_target_corner(board, self.target_corner) else 0.0

        reward = (
            self.score_weight * base_reward
            + empty_bonus
            + monotonicity_bonus
            + corner_bonus
        )

        prev_empty = empty_tile_count(prev_board) if prev_board is not None else 0
        next_empty = empty_tile_count(board) if board is not None else 0
        done = self.state.is_terminal()
        info = {
            "legal_actions": legal_actions(self.state, self.player_id) if not done else [],
            "state_return": new_return,
            "state_reward_raw": state_reward(self.state, self.player_id),
            "board": board,
            "state_text": str(self.state),
            "reward_components": {
                "base_score_delta": float(base_reward),
                "empty_bonus": float(empty_bonus),
                "monotonicity_bonus": float(monotonicity_bonus),
                "corner_bonus": float(corner_bonus),
            },
            "empty_tiles_before": int(prev_empty),
            "empty_tiles_after": int(next_empty),
        }
        return next_obs, float(reward), done, info

    def legal_actions(self):
        if self.state is None or self.state.is_terminal():
            return []
        return legal_actions(self.state, self.player_id)

    def render(self):
        if self.state is None:
            print("<env not reset>")
        else:
            print(self.state)
            

# Demo: random rollout
env = OpenSpiel2048Env(seed=123)
obs = env.reset()

total_reward = 0.0
steps = 0
done = False

while not done and steps < 20:
    a = random.choice(env.legal_actions())
    obs, reward, done, info = env.step(a)
    total_reward += reward
    steps += 1

print("Random steps:", steps)
print("Partial return:", total_reward)
print("Legal actions now:", env.legal_actions())
env.render()


Transition = namedtuple("Transition", ["obs", "action", "reward", "next_obs", "done", "legal_mask", "next_legal_mask"])

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def __len__(self):
        return len(self.buffer)

    def add(self, *args):
        self.buffer.append(Transition(*args))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        return Transition(*zip(*batch))


def make_legal_mask(num_actions, legal_actions_list):
    mask = np.zeros(num_actions, dtype=np.float32)
    mask[legal_actions_list] = 1.0
    return mask


class QNetwork(nn.Module):
    def __init__(self, obs_dim, num_actions, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_actions),
        )

    def forward(self, x):
        return self.net(x)


@torch.no_grad()
def masked_greedy_action(q_net, obs, legal_actions_list, num_actions, epsilon=0.0, device=DEVICE):
    if random.random() < epsilon:
        return random.choice(legal_actions_list)

    obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
    q = q_net(obs_t).squeeze(0)

    legal_mask = torch.zeros(num_actions, dtype=torch.bool, device=device)
    legal_mask[legal_actions_list] = True

    q_masked = q.masked_fill(~legal_mask, -1e9)
    action = int(torch.argmax(q_masked).item())
    return action


# Hyperparameters
SEED = 7
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

NUM_EPISODES = 5000          # increase to 800+ for stronger results
BUFFER_SIZE = 50_000
BATCH_SIZE = 128
GAMMA = 0.99
# LR = 1e-3
LR = 5e-4
TARGET_SYNC_EVERY = 250
LEARN_START = 1_000
LEARN_EVERY = 4
EPS_START = 1.0
EPS_END = 0.05
EPS_DECAY_STEPS = 20_000
MAX_STEPS_PER_EPISODE = 5_000
GRAD_CLIP = 10.0

REWARD_SHAPING_CONFIG = {
    "target_corner": "top_left",
    "score_weight": 1.0,
    "empty_weight": 0.05,
    "monotonicity_weight": 0.5,
    "corner_weight": 1.0,
}

train_env = OpenSpiel2048Env(seed=SEED, **REWARD_SHAPING_CONFIG)

obs_dim = train_env.obs_dim
num_actions = train_env.num_actions

q_net = QNetwork(obs_dim, num_actions).to(DEVICE)
target_net = QNetwork(obs_dim, num_actions).to(DEVICE)
target_net.load_state_dict(q_net.state_dict())
target_net.eval()

optimizer = optim.Adam(q_net.parameters(), lr=LR)
replay = ReplayBuffer(BUFFER_SIZE)

print("obs_dim =", obs_dim)
print("num_actions =", num_actions)
print("reward_shaping =", REWARD_SHAPING_CONFIG)


def epsilon_by_step(step):
    frac = min(1.0, step / EPS_DECAY_STEPS)
    return EPS_START + frac * (EPS_END - EPS_START)


def dqn_update(batch):
    obs = torch.tensor(np.asarray(batch.obs), dtype=torch.float32, device=DEVICE)
    actions = torch.tensor(batch.action, dtype=torch.int64, device=DEVICE).unsqueeze(1)
    rewards = torch.tensor(batch.reward, dtype=torch.float32, device=DEVICE)
    next_obs = torch.tensor(np.asarray(batch.next_obs), dtype=torch.float32, device=DEVICE)
    dones = torch.tensor(batch.done, dtype=torch.float32, device=DEVICE)

    next_legal_mask = torch.tensor(np.asarray(batch.next_legal_mask), dtype=torch.bool, device=DEVICE)

    q_values = q_net(obs)
    q_sa = q_values.gather(1, actions).squeeze(1)

    with torch.no_grad():
        # 🔥 Step 1: select best action using ONLINE network
        next_q_online = q_net(next_obs)
        next_q_online = next_q_online.masked_fill(~next_legal_mask, -1e9)
        next_actions = torch.argmax(next_q_online, dim=1, keepdim=True)

        # 🔥 Step 2: evaluate using TARGET network
        next_q_target = target_net(next_obs)
        next_q_target = next_q_target.gather(1, next_actions).squeeze(1)

        # zero for terminal states
        next_q_target = torch.where(dones > 0.5, torch.zeros_like(next_q_target), next_q_target)

        target = rewards + GAMMA * next_q_target

    loss = F.mse_loss(q_sa, target)

    optimizer.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(q_net.parameters(), GRAD_CLIP)
    optimizer.step()

    return float(loss.item())


episode_returns = []
episode_lengths = []
loss_history = []
eval_returns = []

global_step = 0

for episode in tqdm(range(1, NUM_EPISODES + 1), desc="Training (shaped reward)"):
    obs = train_env.reset(seed=SEED + episode)
    done = False
    ep_return = 0.0
    ep_len = 0

    while not done and ep_len < MAX_STEPS_PER_EPISODE:
        eps = epsilon_by_step(global_step)
        legal = train_env.legal_actions()
        legal_mask = make_legal_mask(num_actions, legal)

        action = masked_greedy_action(
            q_net=q_net,
            obs=obs,
            legal_actions_list=legal,
            num_actions=num_actions,
            epsilon=eps,
            device=DEVICE,
        )

        next_obs, reward, done, info = train_env.step(action)
        next_legal = info["legal_actions"] if not done else []
        next_legal_mask = make_legal_mask(num_actions, next_legal)

        replay.add(obs, action, reward, next_obs, done, legal_mask, next_legal_mask)

        obs = next_obs
        ep_return += reward
        ep_len += 1
        global_step += 1

        if len(replay) >= LEARN_START and global_step % LEARN_EVERY == 0:
            batch = replay.sample(BATCH_SIZE)
            loss = dqn_update(batch)
            loss_history.append(loss)

        if global_step % TARGET_SYNC_EVERY == 0:
            target_net.load_state_dict(q_net.state_dict())

    episode_returns.append(ep_return)
    episode_lengths.append(ep_len)

    if episode % 20 == 0:
        eval_env = OpenSpiel2048Env(seed=1000 + episode, **REWARD_SHAPING_CONFIG)
        obs_eval = eval_env.reset(seed=2000 + episode)
        done_eval = False
        ret_eval = 0.0
        steps_eval = 0
        while not done_eval and steps_eval < MAX_STEPS_PER_EPISODE:
            legal = eval_env.legal_actions()
            action = masked_greedy_action(q_net, obs_eval, legal, num_actions, epsilon=0.0, device=DEVICE)
            obs_eval, reward, done_eval, info_eval = eval_env.step(action)
            ret_eval += reward
            steps_eval += 1
        eval_returns.append((episode, ret_eval))

print("Training complete.")


def moving_average(x, w=20):
    if len(x) < w:
        return np.asarray(x)
    return np.convolve(x, np.ones(w)/w, mode="valid")

plt.figure(figsize=(12, 4))
plt.subplot(1, 3, 1)
plt.plot(episode_returns, alpha=0.35, label="episode return")
ma = moving_average(episode_returns, 20)
plt.plot(range(len(ma)), ma, label="moving avg (20)")
plt.title("Training return")
plt.xlabel("Episode")
plt.ylabel("Return")
plt.legend()

plt.subplot(1, 3, 2)
plt.plot(episode_lengths, alpha=0.5)
plt.title("Episode length")
plt.xlabel("Episode")
plt.ylabel("Steps")

plt.subplot(1, 3, 3)
plt.plot(loss_history, alpha=0.8)
plt.title("DQN loss")
plt.xlabel("Update step")
plt.ylabel("MSE loss")

plt.tight_layout()
plt.show()

if eval_returns:
    eval_eps, eval_vals = zip(*eval_returns)
    plt.figure(figsize=(6,4))
    plt.plot(eval_eps, eval_vals, marker="o")
    plt.title("Greedy evaluation return")
    plt.xlabel("Episode")
    plt.ylabel("Return")
    plt.show()
    

eval_env = OpenSpiel2048Env(seed=999, **REWARD_SHAPING_CONFIG)
obs = eval_env.reset(seed=999)
done = False
greedy_return = 0.0
rollout = []

while not done and len(rollout) < MAX_STEPS_PER_EPISODE:
    legal = eval_env.legal_actions()
    action = masked_greedy_action(q_net, obs, legal, num_actions, epsilon=0.0, device=DEVICE)
    next_obs, reward, done, info = eval_env.step(action)
    rollout.append({
        "action": action,
        "reward": reward,
        "legal_actions": legal,
        "board": info["board"],
        "state_text": info["state_text"],
    })
    obs = next_obs
    greedy_return += reward

print("Greedy evaluation return:", greedy_return)
print("Rollout length:", len(rollout))
print()
eval_env.render()


# Show a few last boards from the greedy rollout
n_show = min(5, len(rollout))
for i, step_info in enumerate(rollout[-n_show:], start=len(rollout)-n_show+1):
    print("=" * 60)
    print(f"Step {i} | action={step_info['action']} | reward={step_info['reward']:.1f}")
    if step_info["board"] is not None:
        print(step_info["board"])
    print(step_info["state_text"])
    

checkpoint_path = "dqn_openspiel_2048_shaped.pt"
torch.save(
    {
        "model_state_dict": q_net.state_dict(),
        "target_state_dict": target_net.state_dict(),
        "obs_dim": obs_dim,
        "num_actions": num_actions,
        "episode_returns": episode_returns,
        "episode_lengths": episode_lengths,
        "loss_history": loss_history,
    },
    checkpoint_path,
)
print("Saved checkpoint to:", checkpoint_path)