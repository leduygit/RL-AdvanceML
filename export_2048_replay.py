import argparse
import json
import random
import re
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

import pyspiel


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def parse_board_numbers(state):
    txt = str(state)
    nums = [int(x) for x in re.findall(r"\d+", txt)]
    if len(nums) >= 16:
        nums = nums[-16:]
        return np.array(nums, dtype=np.int64).reshape(4, 4)
    return None


def extract_tensor_obs(state, player_id=0):
    for fn_name, args in [
        ("observation_tensor", (player_id,)),
        ("observation_tensor", tuple()),
        ("information_state_tensor", (player_id,)),
        ("information_state_tensor", tuple()),
    ]:
        fn = getattr(state, fn_name, None)
        if fn is None:
            continue
        try:
            obs = fn(*args)
            return np.asarray(obs, dtype=np.float32).reshape(-1)
        except TypeError:
            continue
    raise RuntimeError("Could not extract observation tensor from state.")


def extract_log2_board_obs(state):
    board = parse_board_numbers(state)
    if board is None:
        raise RuntimeError("Failed to parse 4x4 board from state.")
    return np.log2(board.astype(np.float32) + 1.0).reshape(-1)


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
    def __init__(self, seed=42, obs_mode="auto", expected_obs_dim=None):
        self.game = pyspiel.load_game("2048")
        self.player_id = 0
        self.num_actions = self.game.num_distinct_actions()
        self.rng = np.random.default_rng(seed)
        self.state = None
        self.obs_mode = self._resolve_obs_mode(obs_mode, expected_obs_dim)
        self.obs_dim = 16 if self.obs_mode == "log2_board" else self.game.observation_tensor_size()

    def _resolve_obs_mode(self, obs_mode, expected_obs_dim):
        if obs_mode in {"tensor", "log2_board"}:
            return obs_mode
        if expected_obs_dim == 16:
            return "log2_board"
        return "tensor"

    def extract_obs(self):
        if self.obs_mode == "log2_board":
            return extract_log2_board_obs(self.state)
        return extract_tensor_obs(self.state, self.player_id)

    def reset(self, seed=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.state = self.game.new_initial_state()
        auto_resolve_chance_nodes(self.state, self.rng)
        return self.extract_obs()

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
        next_obs = self.extract_obs() if not done else np.zeros(self.obs_dim, dtype=np.float32)
        new_return = state_return(self.state, self.player_id)
        reward = new_return - prev_return

        info = {
            "board": parse_board_numbers(self.state),
            "state_text": str(self.state),
            "legal_actions": legal_actions(self.state, self.player_id) if not done else [],
            "score": new_return,
        }
        return next_obs, float(reward), done, info

    def current_board(self):
        board = parse_board_numbers(self.state)
        if board is None:
            raise RuntimeError("Failed to parse current board from state.")
        return board


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
def greedy_policy_step(q_net, obs, legal_actions_list, num_actions, device=DEVICE):
    obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
    q_values = q_net(obs_t).squeeze(0)
    legal_mask = torch.zeros(num_actions, dtype=torch.bool, device=device)
    legal_mask[legal_actions_list] = True
    q_masked = q_values.masked_fill(~legal_mask, -1e9)
    action = int(torch.argmax(q_masked).item())

    q_values_cpu = q_values.detach().cpu().numpy()
    legal_q = {int(a): float(q_values_cpu[a]) for a in legal_actions_list}
    return action, q_values_cpu, legal_q


def action_to_label(state, action):
    try:
        return state.action_to_string(state.current_player(), int(action))
    except Exception:
        pass

    fallback = {
        0: "up",
        1: "right",
        2: "down",
        3: "left",
    }
    return fallback.get(int(action), f"action_{action}")


def load_checkpoint(path):
    checkpoint = torch.load(path, map_location=DEVICE)
    obs_dim = int(checkpoint["obs_dim"])
    num_actions = int(checkpoint["num_actions"])
    model = QNetwork(obs_dim, num_actions).to(DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return checkpoint, model


def rollout_episode(model, checkpoint, seed, max_steps, obs_mode):
    env = OpenSpiel2048Env(seed=seed, obs_mode=obs_mode, expected_obs_dim=int(checkpoint["obs_dim"]))
    obs = env.reset(seed=seed)
    board = env.current_board()
    initial_legal = legal_actions(env.state, env.player_id)

    steps = [
        {
            "step": 0,
            "board": board.tolist(),
            "score": 0,
            "action": None,
            "action_id": None,
            "reward": 0.0,
            "done": False,
            "legal_actions": initial_legal,
            "legal_action_labels": [action_to_label(env.state, a) for a in initial_legal],
            "state_text": str(env.state),
        }
    ]

    total_reward = 0.0
    step_index = 0

    while not env.state.is_terminal() and step_index < max_steps:
        legal = legal_actions(env.state, env.player_id)
        state_before_action = env.state.clone()
        action, q_values, legal_q = greedy_policy_step(model, obs, legal, env.num_actions, device=DEVICE)
        action_label = action_to_label(state_before_action, action)
        legal_action_labels = [action_to_label(state_before_action, a) for a in legal]

        next_obs, reward, done, info = env.step(action)
        total_reward += reward
        step_index += 1

        steps.append(
            {
                "step": step_index,
                "board": info["board"].tolist() if info["board"] is not None else None,
                "score": info["score"],
                "action": action_label,
                "action_id": action,
                "reward": reward,
                "done": done,
                "legal_actions": legal,
                "legal_action_labels": legal_action_labels,
                "q_values": [float(x) for x in q_values.tolist()],
                "legal_q_values": {str(k): v for k, v in legal_q.items()},
                "state_text": info["state_text"],
            }
        )

        obs = next_obs

    final_board = steps[-1]["board"]
    max_tile = max(max(row) for row in final_board) if final_board else 0
    return {
        "meta": {
            "game": "2048",
            "board_size": 4,
            "agent": Path(str(checkpoint.get("checkpoint_path", "checkpoint"))).stem,
            "episode_name": f"Model rollout seed {seed}",
            "checkpoint": str(checkpoint.get("checkpoint_path", "")),
            "obs_mode": env.obs_mode,
            "device": str(DEVICE),
            "total_steps": len(steps) - 1,
            "final_score": total_reward,
            "max_tile": int(max_tile),
        },
        "steps": steps,
    }


def main():
    parser = argparse.ArgumentParser(description="Export a 2048 model rollout to JSON for the replay UI.")
    parser.add_argument("--checkpoint", required=True, help="Path to a saved .pt checkpoint.")
    parser.add_argument("--output", default=str(Path("UI") / "model-rollout.json"), help="Where to write the JSON replay.")
    parser.add_argument("--seed", type=int, default=999, help="RNG seed for the environment rollout.")
    parser.add_argument("--max-steps", type=int, default=5000, help="Maximum number of agent decisions to export.")
    parser.add_argument(
        "--obs-mode",
        choices=["auto", "tensor", "log2_board"],
        default="auto",
        help="Observation preprocessing mode. Use auto to infer from checkpoint obs_dim.",
    )
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint)
    output_path = Path(args.output)

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint, model = load_checkpoint(checkpoint_path)
    checkpoint["checkpoint_path"] = str(checkpoint_path)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    replay = rollout_episode(
        model=model,
        checkpoint=checkpoint,
        seed=args.seed,
        max_steps=args.max_steps,
        obs_mode=args.obs_mode,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(replay, indent=2), encoding="utf-8")

    print(f"Checkpoint: {checkpoint_path}")
    print(f"Output JSON: {output_path}")
    print(f"Observation mode: {replay['meta']['obs_mode']}")
    print(f"Steps exported: {replay['meta']['total_steps']}")
    print(f"Final score: {replay['meta']['final_score']}")
    print(f"Max tile: {replay['meta']['max_tile']}")


if __name__ == "__main__":
    main()
