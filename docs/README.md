# Training and JSON Export Guide

This document explains what the project scripts are doing, from training a model to exporting a replay JSON for the UI.

## 1) What each file does

- `HW_DoubleDQN.py`
  - Trains a Double DQN agent on OpenSpiel 2048.
  - Saves a checkpoint file named `dqn_openspiel_2048.pt`.
- `export_2048_replay.py`
  - Loads a saved checkpoint.
  - Runs one greedy episode in the 2048 environment.
  - Exports the full episode as JSON (default: `UI/model-rollout.json`).
- `UI/index.html` + `UI/app.js`
  - Visualizes a replay JSON step-by-step.
  - On startup, it loads a built-in sample episode unless you choose your own JSON file.

## 2) Environment setup

Use the project virtual environment via `uv`.

```bash
uv sync
```

Important package note:
- Python import is `pyspiel`
- Package name in dependencies is `open-spiel`

Run scripts with `uv run` so they use the project environment:

```bash
uv run HW_DoubleDQN.py
uv run export_2048_replay.py --checkpoint dqn_openspiel_2048.pt --output UI/model-rollout.json
```

## 3) How training works (`HW_DoubleDQN.py`)

### Observation and state handling

- The script reads a 4x4 board from OpenSpiel state text.
- It applies `log2(board + 1)` and flattens to a vector for the network input.
- Chance nodes (tile spawn events) are auto-resolved using RNG before/after actions.

### Action space

- Uses legal actions from OpenSpiel only.
- Action selection during training is epsilon-greedy:
  - random action with probability `epsilon`
  - otherwise argmax Q among legal actions

### Replay buffer

- Stores transitions: `(obs, action, reward, next_obs, done, legal_mask, next_legal_mask)`
- Samples mini-batches once replay size reaches `LEARN_START`.

### Double DQN update

The target is computed in two steps:

1. Choose next action using the online network.
2. Evaluate that action with the target network.

This is the core Double DQN idea that reduces overestimation compared to vanilla DQN.

### Training loop summary

For each episode:

1. Reset environment.
2. Roll out until terminal or max steps.
3. Store transitions in replay buffer.
4. Periodically update Q-network from sampled batches.
5. Periodically sync target network from online network.

### Outputs from training

At the end, the script saves:

- `model_state_dict`
- `target_state_dict`
- `obs_dim`
- `num_actions`
- `episode_returns`
- `episode_lengths`
- `loss_history`

to `dqn_openspiel_2048.pt`.

## 4) How export works (`export_2048_replay.py`)

### What happens internally

1. Load checkpoint and rebuild `QNetwork` with checkpoint dimensions.
2. Create OpenSpiel 2048 env.
3. Infer observation mode:
   - `auto` picks `log2_board` if checkpoint `obs_dim == 16`
   - else `tensor`
4. Reset env with seed.
5. Roll out one episode with greedy policy (`argmax` over legal actions).
6. Stop when:
   - game is terminal, or
   - `step_index == max_steps`
7. Save replay JSON.

### CLI options

```bash
uv run export_2048_replay.py \
  --checkpoint dqn_openspiel_2048.pt \
  --output UI/model-rollout.json \
  --seed 123 \
  --max-steps 5000 \
  --obs-mode auto
```

- `--checkpoint`: required checkpoint path.
- `--output`: output JSON path.
- `--seed`: controls environment randomness.
- `--max-steps`: upper bound on decisions exported.
- `--obs-mode`: `auto`, `tensor`, or `log2_board`.

## 5) JSON structure written by exporter

The JSON has:

- `meta`:
  - game info, checkpoint path, obs mode, total steps, final score, max tile
- `steps`:
  - one entry per timestep (including initial state at step 0)
  - board, action/action_id, reward, done, legal actions, q-values, and state text

## 6) Why you might see only 6 steps in UI

This is usually a UI behavior, not model behavior.

- `UI/app.js` includes a hardcoded sample episode with 6 steps.
- UI loads that sample by default on startup.

To view your real model rollout:

1. Open `UI/index.html`
2. Click **Choose replay JSON**
3. Select `UI/model-rollout.json`

## 7) Quick end-to-end command flow

```bash
# 1) install deps in project environment
uv sync

# 2) train and save checkpoint
uv run HW_DoubleDQN.py

# 3) export one replay from the trained checkpoint
uv run export_2048_replay.py --checkpoint dqn_openspiel_2048.pt --output UI/model-rollout.json --seed 123 --max-steps 5000

# 4) open UI (macOS)
open UI/index.html
```

Then load `UI/model-rollout.json` in the file picker inside the UI.
