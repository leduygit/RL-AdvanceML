import json
import random
import re
from collections import deque, namedtuple
from pathlib import Path

import numpy as np
import optuna
import pyspiel
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm.auto import tqdm

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Runtime configuration constants.
# These values control how the Optuna TPE search is executed.
# NUM_EPISODES:
#   Number of training episodes inside each trial.
#   Higher values make each trial more reliable but slower.
#
# EVAL_EVERY:
#   Evaluate the current policy every N training episodes.
#   This metric is used for progress reporting and pruning decisions.
#
# EVAL_EPISODES:
#   Number of greedy evaluation episodes per evaluation call.
#   Higher values reduce variance in trial scores.
#
# MAX_STEPS_PER_EPISODE:
#   Max environment steps allowed in both training and evaluation episodes.
#
# SEED:
#   Base seed for reproducibility. Each trial derives its own seed from this.
#
# STUDY_NAME:
#   Optuna study identifier. Reused when persistent storage is enabled.
#
# STORAGE:
#   Optional Optuna storage backend URL (for example: "sqlite:///optuna.db").
#   Keep as None to use in-memory study only.
#
# OUTPUT_JSON:
#   File path to save best-trial summary (best value, params, and attrs).
N_TRIALS = 20
NUM_EPISODES = 5000
EVAL_EVERY = 20
EVAL_EPISODES = 3
MAX_STEPS_PER_EPISODE = 2000
SEED = 7
STUDY_NAME = "dqn_2048_tpe"
STORAGE = None
OUTPUT_JSON = "optuna_best_params.json"

Transition = namedtuple(
    "Transition",
    ["obs", "action", "reward", "next_obs", "done", "legal_mask", "next_legal_mask"],
)


def parse_board_numbers(state):
    txt = str(state)
    nums = [int(x) for x in re.findall(r"\d+", txt)]
    if len(nums) >= 16:
        nums = nums[-16:]
        return np.array(nums, dtype=np.int64).reshape(4, 4)
    return None


def extract_obs(state, player_id=0):
    board = parse_board_numbers(state)
    if board is None:
        raise RuntimeError("Failed to parse board.")
    board = board.astype(np.float32)
    return np.log2(board + 1.0).reshape(-1)


def legal_actions(state, player_id=0):
    try:
        return list(state.legal_actions(player_id))
    except TypeError:
        return list(state.legal_actions())


def sample_chance_action(state, rng):
    outcomes = state.chance_outcomes()
    actions, probs = zip(*outcomes)
    idx = rng.choice(len(actions), p=np.asarray(probs, dtype=np.float64))
    return actions[idx]


def auto_resolve_chance_nodes(state, rng):
    while state.is_chance_node() and not state.is_terminal():
        state.apply_action(sample_chance_action(state, rng))
    return state


def state_return(state, player_id=0):
    vals = state.returns()
    return float(vals[player_id]) if len(vals) > player_id else 0.0


class OpenSpiel2048Env:
    def __init__(self, seed=42):
        self.game = pyspiel.load_game("2048")
        self.player_id = 0
        self.num_actions = self.game.num_distinct_actions()
        self.obs_dim = self.game.observation_tensor_size()
        self.rng = np.random.default_rng(seed)
        self.state = None

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
        self.state.apply_action(int(action))
        auto_resolve_chance_nodes(self.state, self.rng)

        done = self.state.is_terminal()
        next_obs = extract_obs(self.state, self.player_id) if not done else np.zeros(self.obs_dim, dtype=np.float32)
        new_return = state_return(self.state, self.player_id)
        reward = new_return - prev_return

        info = {
            "legal_actions": legal_actions(self.state, self.player_id) if not done else [],
            "state_return": new_return,
        }
        return next_obs, float(reward), done, info

    def legal_actions(self):
        if self.state is None or self.state.is_terminal():
            return []
        return legal_actions(self.state, self.player_id)


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
    return int(torch.argmax(q_masked).item())


def epsilon_by_step(step, eps_start, eps_end, eps_decay_steps):
    frac = min(1.0, step / eps_decay_steps)
    return eps_start + frac * (eps_end - eps_start)


def evaluate_greedy(q_net, num_actions, max_steps_per_episode, seed, n_eval_episodes):
    eval_returns = []
    eval_lengths = []
    for i in range(n_eval_episodes):
        env = OpenSpiel2048Env(seed=seed + i)
        obs = env.reset(seed=seed + 10_000 + i)
        done = False
        ep_return = 0.0
        ep_len = 0

        while not done and ep_len < max_steps_per_episode:
            legal = env.legal_actions()
            action = masked_greedy_action(q_net, obs, legal, num_actions, epsilon=0.0, device=DEVICE)
            obs, reward, done, _ = env.step(action)
            ep_return += reward
            ep_len += 1

        eval_returns.append(ep_return)
        eval_lengths.append(ep_len)

    return float(np.mean(eval_returns)), float(np.mean(eval_lengths))


def train_and_eval(trial):
    hidden_dim = trial.suggest_categorical("hidden_dim", [128, 256, 384])
    gamma = trial.suggest_float("gamma", 0.97, 0.999)
    lr = trial.suggest_float("lr", 1e-5, 5e-3, log=True)
    batch_size = trial.suggest_categorical("batch_size", [64, 128, 256])
    buffer_size = trial.suggest_categorical("buffer_size", [20_000, 50_000, 100_000])
    target_sync_every = trial.suggest_categorical("target_sync_every", [100, 250, 500])
    learn_every = trial.suggest_categorical("learn_every", [1, 2, 4])
    eps_end = trial.suggest_float("eps_end", 0.01, 0.15)
    eps_decay_steps = trial.suggest_int("eps_decay_steps", 5_000, 60_000, step=5_000)
    grad_clip = trial.suggest_float("grad_clip", 1.0, 20.0)

    seed = SEED + trial.number * 997
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    train_env = OpenSpiel2048Env(seed=seed)
    obs_dim = train_env.obs_dim
    num_actions = train_env.num_actions

    q_net = QNetwork(obs_dim, num_actions, hidden_dim=hidden_dim).to(DEVICE)
    target_net = QNetwork(obs_dim, num_actions, hidden_dim=hidden_dim).to(DEVICE)
    target_net.load_state_dict(q_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(q_net.parameters(), lr=lr)
    replay = ReplayBuffer(buffer_size)

    global_step = 0
    learn_start = max(batch_size * 8, 500)

    for episode in range(1, NUM_EPISODES + 1):
        obs = train_env.reset(seed=seed + episode)
        done = False
        ep_len = 0

        while not done and ep_len < MAX_STEPS_PER_EPISODE:
            eps = epsilon_by_step(global_step, 1.0, eps_end, eps_decay_steps)
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
            ep_len += 1
            global_step += 1

            if len(replay) >= learn_start and global_step % learn_every == 0:
                batch = replay.sample(batch_size)

                obs_b = torch.tensor(np.asarray(batch.obs), dtype=torch.float32, device=DEVICE)
                actions_b = torch.tensor(batch.action, dtype=torch.int64, device=DEVICE).unsqueeze(1)
                rewards_b = torch.tensor(batch.reward, dtype=torch.float32, device=DEVICE)
                next_obs_b = torch.tensor(np.asarray(batch.next_obs), dtype=torch.float32, device=DEVICE)
                dones_b = torch.tensor(batch.done, dtype=torch.float32, device=DEVICE)
                next_legal_mask_b = torch.tensor(np.asarray(batch.next_legal_mask), dtype=torch.bool, device=DEVICE)

                q_values = q_net(obs_b)
                q_sa = q_values.gather(1, actions_b).squeeze(1)

                with torch.no_grad():
                    # Double DQN: online net selects, target net evaluates.
                    next_q_online = q_net(next_obs_b).masked_fill(~next_legal_mask_b, -1e9)
                    next_actions = torch.argmax(next_q_online, dim=1, keepdim=True)
                    next_q_target = target_net(next_obs_b).gather(1, next_actions).squeeze(1)
                    next_q_target = torch.where(dones_b > 0.5, torch.zeros_like(next_q_target), next_q_target)
                    target = rewards_b + gamma * next_q_target

                loss = F.mse_loss(q_sa, target)
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(q_net.parameters(), grad_clip)
                optimizer.step()

            if global_step % target_sync_every == 0:
                target_net.load_state_dict(q_net.state_dict())

        # Report intermediate value for pruning.
        if episode % max(1, EVAL_EVERY) == 0:
            eval_ret, eval_len = evaluate_greedy(
                q_net=q_net,
                num_actions=num_actions,
                max_steps_per_episode=MAX_STEPS_PER_EPISODE,
                seed=seed,
                n_eval_episodes=EVAL_EPISODES,
            )
            trial.report(eval_ret, step=episode)
            trial.set_user_attr("last_eval_len", eval_len)
            if trial.should_prune():
                raise optuna.TrialPruned()

    final_eval_return, final_eval_length = evaluate_greedy(
        q_net=q_net,
        num_actions=num_actions,
        max_steps_per_episode=MAX_STEPS_PER_EPISODE,
        seed=seed,
        n_eval_episodes=EVAL_EPISODES,
    )
    trial.set_user_attr("final_eval_length", final_eval_length)
    return final_eval_return


def main():
    sampler = optuna.samplers.TPESampler(seed=SEED, multivariate=True)
    pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=max(1, EVAL_EVERY))

    study = optuna.create_study(
        study_name=STUDY_NAME,
        direction="maximize",
        sampler=sampler,
        pruner=pruner,
        storage=STORAGE,
        load_if_exists=True,
    )

    study.optimize(train_and_eval, n_trials=N_TRIALS, show_progress_bar=True)

    best = study.best_trial
    summary = {
        "study_name": study.study_name,
        "n_trials_total": len(study.trials),
        "best_trial_number": best.number,
        "best_value": float(best.value),
        "best_params": best.params,
        "best_user_attrs": best.user_attrs,
    }

    output_path = Path(OUTPUT_JSON)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Device: {DEVICE}")
    print(f"Best trial: {best.number}")
    print(f"Best eval return: {best.value:.3f}")
    print(f"Best params: {best.params}")
    print(f"Saved best summary: {output_path}")


if __name__ == "__main__":
    main()
