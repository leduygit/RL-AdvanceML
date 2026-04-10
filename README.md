# 2048 RL Agent + Replay Visualizer

This project contains:

- `main.ipynb`: baseline DQN training script for OpenSpiel 2048
- `HW_DoubleDQN.py`: upgraded Double DQN variant
- `export_2048_replay.py`: inference/export script that runs a trained checkpoint and writes a JSON replay
- `UI/`: static replay viewer for loading JSON episodes and visualizing them step by step

## Project Flow

The intended workflow is:

1. Train a model with `main.ipynb` or `HW_DoubleDQN.py`
2. Save the checkpoint as a `.pt` file
3. Run `export_2048_replay.py` to generate a JSON rollout
4. Open `UI/index.html`
5. Load the generated JSON file into the UI

## Requirements

You need a Python environment with at least:

- `torch`
- `numpy`
- `matplotlib`
- `tqdm`
- `pyspiel` from OpenSpiel

The UI is plain HTML/CSS/JS and does not require npm.

## Training

Train the Double DQN version:

```powershell
python HW_DoubleDQN.py
```

The script save a checkpoint named:

```text
dqn_openspiel_2048.pt
```

## Export A Replay JSON

Generate a rollout from a trained checkpoint:

```powershell
python export_2048_replay.py --checkpoint dqn_openspiel_2048.pt --output "UI/model-rollout.json"
```

Useful options:

- `--seed 999`: controls the environment randomness during rollout
- `--max-steps 5000`: caps the rollout length
- `--obs-mode auto`: automatically chooses the observation preprocessing

Example:

```powershell
python export_2048_replay.py --checkpoint dqn_openspiel_2048.pt --output "UI/model-rollout.json" --seed 123 --max-steps 1000
```

Important on Windows:

- prefer `"UI/model-rollout.json"` instead of `UI\model-rollout.json`
- quoting the output path avoids accidental path issues

## Open The UI

Open:

```text
UI/index.html
```

The UI will load a built-in sample episode on startup. After that you can:

- click `Choose replay JSON`
- select your exported file such as `UI/model-rollout.json`
- use `Play`, `Prev`, `Next`, `Restart`, and the timeline slider to inspect the episode

## JSON Format

The viewer expects snapshot-based JSON:

```json
{
  "meta": {
    "game": "2048",
    "board_size": 4,
    "agent": "dqn_openspiel_2048",
    "episode_name": "Model rollout seed 999"
  },
  "steps": [
    {
      "step": 0,
      "board": [[0, 0, 2, 0], [0, 0, 0, 0], [0, 0, 0, 0], [2, 0, 0, 0]],
      "score": 0,
      "action": null,
      "reward": 0,
      "done": false
    }
  ]
}
```

The exporter can include extra fields such as:

- `action_id`
- `legal_actions`
- `legal_action_labels`
- `q_values`
- `legal_q_values`
- `state_text`

The UI ignores extra fields safely.

## Notes

- `main.ipynb` and `HW_DoubleDQN.py` train immediately when run; they are not structured as importable modules.
- `export_2048_replay.py` is intentionally standalone so inference/export does not trigger training.
- The UI animation is snapshot-based. It gives smooth replay, but it does not reconstruct exact 2048 merge lineage.
