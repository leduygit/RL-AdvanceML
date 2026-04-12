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

BOARD_SIZE = 4
DEFAULT_MAX_ONEHOT_EXPONENT = 15


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


def extract_onehot_board_obs(state, max_onehot_exponent=DEFAULT_MAX_ONEHOT_EXPONENT):
    board = parse_board_numbers(state)
    if board is None:
        raise RuntimeError("Failed to parse 4x4 board from state.")

    obs_channels = int(max_onehot_exponent) + 1
    board = board.astype(np.int64)
    exponents = np.zeros_like(board, dtype=np.int64)
    non_zero = board > 0
    exponents[non_zero] = np.log2(board[non_zero]).astype(np.int64)
    exponents = np.clip(exponents, 0, int(max_onehot_exponent))

    one_hot = np.eye(obs_channels, dtype=np.float32)[exponents]
    return np.moveaxis(one_hot, -1, 0)


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
    def __init__(self, seed=42, obs_mode="auto", expected_obs_dim=None, max_onehot_exponent=DEFAULT_MAX_ONEHOT_EXPONENT):
        self.game = pyspiel.load_game("2048")
        self.player_id = 0
        self.num_actions = self.game.num_distinct_actions()
        self.rng = np.random.default_rng(seed)
        self.state = None
        self.max_onehot_exponent = int(max_onehot_exponent)
        self.obs_mode = self._resolve_obs_mode(obs_mode, expected_obs_dim)

        if self.obs_mode == "log2_board":
            self.obs_shape = (16,)
        elif self.obs_mode == "onehot_board":
            self.obs_shape = (self.max_onehot_exponent + 1, BOARD_SIZE, BOARD_SIZE)
        else:
            self.obs_shape = (self.game.observation_tensor_size(),)

        self.obs_dim = int(np.prod(self.obs_shape))

    def _resolve_obs_mode(self, obs_mode, expected_obs_dim):
        if obs_mode in {"tensor", "log2_board", "onehot_board"}:
            return obs_mode
        if expected_obs_dim == 16:
            return "log2_board"
        if expected_obs_dim and expected_obs_dim % (BOARD_SIZE * BOARD_SIZE) == 0:
            channels = expected_obs_dim // (BOARD_SIZE * BOARD_SIZE)
            if channels >= 2:
                return "onehot_board"
        return "tensor"

    def extract_obs(self):
        if self.obs_mode == "log2_board":
            return extract_log2_board_obs(self.state)
        if self.obs_mode == "onehot_board":
            return extract_onehot_board_obs(self.state, self.max_onehot_exponent)
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
        next_obs = self.extract_obs() if not done else np.zeros(self.obs_shape, dtype=np.float32)
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


class DuelingQNetwork(nn.Module):
    def __init__(self, in_channels, num_actions, hidden_dim=512):
        super().__init__()

        flat_dim = in_channels * BOARD_SIZE * BOARD_SIZE

        self.feature_layer = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat_dim, hidden_dim),
            nn.ReLU(),
        )

        self.value_stream = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, num_actions),
        )

    def forward(self, x):
        features = self.feature_layer(x)
        values = self.value_stream(features)
        advantages = self.advantage_stream(features)
        return values + (advantages - advantages.mean(dim=1, keepdim=True))


class QNetworkCNN(nn.Module):
    def __init__(
        self,
        in_channels,
        num_actions,
        hidden_dim=256,
        conv1_out=64,
        conv2_out=128,
        conv1_kernel_size=2,
        conv2_kernel_size=2,
        conv1_padding=0,
        conv2_padding=0,
    ):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(
                in_channels,
                conv1_out,
                kernel_size=conv1_kernel_size,
                stride=1,
                padding=conv1_padding,
            ),
            nn.ReLU(),
            nn.Conv2d(
                conv1_out,
                conv2_out,
                kernel_size=conv2_kernel_size,
                stride=1,
                padding=conv2_padding,
            ),
            nn.ReLU(),
        )

        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, BOARD_SIZE, BOARD_SIZE)
            flat_dim = self.features(dummy).reshape(1, -1).shape[1]

        self.head = nn.Sequential(
            nn.Linear(flat_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_actions),
        )

    def forward(self, x):
        z = self.features(x)
        z = z.reshape(z.size(0), -1)
        return self.head(z)


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


def infer_cnn_arch_from_state_dict(state_dict):
    conv1_w = state_dict.get("features.0.weight")
    conv2_w = state_dict.get("features.2.weight")
    head0_w = state_dict.get("head.0.weight")

    if conv1_w is None or conv2_w is None or head0_w is None:
        raise KeyError("CNN checkpoint is missing expected keys under features/head blocks.")

    conv1_out, in_channels, k1_h, k1_w = conv1_w.shape
    conv2_out, conv2_in, k2_h, k2_w = conv2_w.shape
    if conv2_in != conv1_out:
        raise ValueError("Incompatible CNN checkpoint: features.0 and features.2 channel sizes do not align.")

    hidden_dim = int(head0_w.shape[0])
    expected_flat_dim = int(head0_w.shape[1])

    # Try small paddings and pick one that reproduces the saved head input size.
    for p1 in range(0, 4):
        out1_h = BOARD_SIZE + 2 * p1 - int(k1_h) + 1
        out1_w = BOARD_SIZE + 2 * p1 - int(k1_w) + 1
        if out1_h <= 0 or out1_w <= 0:
            continue
        for p2 in range(0, 4):
            out2_h = out1_h + 2 * p2 - int(k2_h) + 1
            out2_w = out1_w + 2 * p2 - int(k2_w) + 1
            if out2_h <= 0 or out2_w <= 0:
                continue
            flat_dim = int(conv2_out) * out2_h * out2_w
            if flat_dim == expected_flat_dim:
                return {
                    "in_channels": int(in_channels),
                    "conv1_out": int(conv1_out),
                    "conv2_out": int(conv2_out),
                    "conv1_kernel_size": (int(k1_h), int(k1_w)),
                    "conv2_kernel_size": (int(k2_h), int(k2_w)),
                    "conv1_padding": p1,
                    "conv2_padding": p2,
                    "hidden_dim": hidden_dim,
                }

    raise ValueError(
        "Could not infer CNN paddings from checkpoint shapes. "
        "Please ensure the checkpoint matches the expected two-conv architecture."
    )


def load_checkpoint(path):
    checkpoint = torch.load(path, map_location=DEVICE)
    obs_dim = int(checkpoint["obs_dim"])
    num_actions = int(checkpoint["num_actions"])

    state_dict = checkpoint["model_state_dict"]
    uses_cnn = any(k.startswith("features.") or k.startswith("head.") for k in state_dict.keys())
    uses_dueling = any(k.startswith("feature_layer.") or k.startswith("value_stream.") or k.startswith("advantage_stream.") for k in state_dict.keys())

    if uses_cnn:
        arch = infer_cnn_arch_from_state_dict(state_dict)
        model = QNetworkCNN(
            in_channels=arch["in_channels"],
            num_actions=num_actions,
            hidden_dim=arch["hidden_dim"],
            conv1_out=arch["conv1_out"],
            conv2_out=arch["conv2_out"],
            conv1_kernel_size=arch["conv1_kernel_size"],
            conv2_kernel_size=arch["conv2_kernel_size"],
            conv1_padding=arch["conv1_padding"],
            conv2_padding=arch["conv2_padding"],
        ).to(DEVICE)
    elif uses_dueling:
        obs_shape = checkpoint.get("obs_shape")
        if obs_shape is not None and len(obs_shape) == 3:
            in_channels = int(obs_shape[0])
        else:
            in_channels = max(1, obs_dim // (BOARD_SIZE * BOARD_SIZE))

        hidden_dim = 512
        if "feature_layer.1.weight" in state_dict:
            hidden_dim = int(state_dict["feature_layer.1.weight"].shape[0])

        model = DuelingQNetwork(
            in_channels=in_channels,
            num_actions=num_actions,
            hidden_dim=hidden_dim,
        ).to(DEVICE)
    else:
        model = QNetwork(obs_dim, num_actions).to(DEVICE)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return checkpoint, model


def rollout_episode(model, checkpoint, seed, max_steps, obs_mode):
    env = OpenSpiel2048Env(
        seed=seed,
        obs_mode=obs_mode,
        expected_obs_dim=int(checkpoint["obs_dim"]),
        max_onehot_exponent=int(checkpoint.get("max_onehot_exponent", DEFAULT_MAX_ONEHOT_EXPONENT)),
    )
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
        choices=["auto", "tensor", "log2_board", "onehot_board"],
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
