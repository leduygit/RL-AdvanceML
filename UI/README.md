# 2048 Replay Visualizer

Static UI for loading a 2048 episode JSON file and replaying it step by step.

## Files

- `index.html`: app shell
- `styles.css`: dashboard styling
- `app.js`: JSON validation, rendering, and playback controls
- `sample-episode.json`: example replay file

## Expected JSON

```json
{
  "meta": {
    "game": "2048",
    "board_size": 4,
    "agent": "dqn_v1",
    "episode_name": "Episode 001"
  },
  "steps": [
    {
      "step": 0,
      "board": [[0, 2, 0, 0], [0, 0, 0, 0], [0, 0, 2, 0], [0, 0, 0, 0]],
      "score": 0,
      "action": null,
      "reward": 0,
      "done": false
    }
  ]
}
```

## Run

Open `index.html` in a browser, or serve the folder with any static server.
