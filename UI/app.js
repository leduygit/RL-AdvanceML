const TILE_COLORS = {
  0: { bg: "rgba(255, 248, 238, 0.35)", text: "transparent" },
  2: { bg: "#f5ede3", text: "#5b5148" },
  4: { bg: "#f0e1c8", text: "#5b5148" },
  8: { bg: "#f4b26d", text: "#fff8f0" },
  16: { bg: "#ef9861", text: "#fff8f0" },
  32: { bg: "#ea7e54", text: "#fff8f0" },
  64: { bg: "#df6241", text: "#fff8f0" },
  128: { bg: "#d9b558", text: "#fff8f0" },
  256: { bg: "#cfa348", text: "#fff8f0" },
  512: { bg: "#bd8b3b", text: "#fff8f0" },
  1024: { bg: "#a46d2d", text: "#fff8f0" },
  2048: { bg: "#7b4b1f", text: "#fff8f0" }
};

const elements = {
  boardGrid: document.getElementById("boardGrid"),
  animationLayer: document.getElementById("animationLayer"),
  episodeTitle: document.getElementById("episodeTitle"),
  statusPill: document.getElementById("statusPill"),
  scoreValue: document.getElementById("scoreValue"),
  maxTileValue: document.getElementById("maxTileValue"),
  stepValue: document.getElementById("stepValue"),
  timelineLabel: document.getElementById("timelineLabel"),
  timelineSlider: document.getElementById("timelineSlider"),
  fileInput: document.getElementById("fileInput"),
  loadSampleButton: document.getElementById("loadSampleButton"),
  reloadButton: document.getElementById("reloadButton"),
  fileName: document.getElementById("fileName"),
  agentName: document.getElementById("agentName"),
  errorBox: document.getElementById("errorBox"),
  prevButton: document.getElementById("prevButton"),
  playPauseButton: document.getElementById("playPauseButton"),
  nextButton: document.getElementById("nextButton"),
  restartButton: document.getElementById("restartButton"),
  speedSlider: document.getElementById("speedSlider"),
  speedLabel: document.getElementById("speedLabel"),
  actionValue: document.getElementById("actionValue"),
  rewardValue: document.getElementById("rewardValue"),
  doneValue: document.getElementById("doneValue"),
  boardSizeValue: document.getElementById("boardSizeValue"),
  stepJsonView: document.getElementById("stepJsonView")
};

const state = {
  episode: null,
  currentStepIndex: 0,
  playbackTimer: null,
  playbackSpeed: 1,
  lastLoadedText: "",
  lastSourceName: "None"
};

function createEmptyBoard(size = 4) {
  return Array.from({ length: size }, () => Array.from({ length: size }, () => 0));
}

function getBoardSize(episode) {
  return episode?.meta?.board_size ?? episode?.steps?.[0]?.board?.length ?? 4;
}

function normalizeBoard(board, size) {
  if (!Array.isArray(board)) {
    throw new Error("Each step must include a board array.");
  }

  if (board.length !== size) {
    throw new Error(`Board row count must match board_size (${size}).`);
  }

  return board.map((row, rowIndex) => {
    if (!Array.isArray(row) || row.length !== size) {
      throw new Error(`Board row ${rowIndex} must contain ${size} columns.`);
    }

    return row.map((value) => {
      if (!Number.isInteger(value) || value < 0) {
        throw new Error("Board values must be non-negative integers.");
      }
      return value;
    });
  });
}

function normalizeEpisode(rawEpisode) {
  if (!rawEpisode || typeof rawEpisode !== "object") {
    throw new Error("JSON must be an object with meta and steps.");
  }

  if (!Array.isArray(rawEpisode.steps) || rawEpisode.steps.length === 0) {
    throw new Error("Episode must include a non-empty steps array.");
  }

  const size = rawEpisode.meta?.board_size ?? rawEpisode.steps[0]?.board?.length ?? 4;
  if (!Number.isInteger(size) || size <= 0) {
    throw new Error("meta.board_size must be a positive integer.");
  }

  const normalizedSteps = rawEpisode.steps.map((step, index) => {
    if (!step || typeof step !== "object") {
      throw new Error(`Step ${index} must be an object.`);
    }

    const board = normalizeBoard(step.board, size);
    const score = Number(step.score ?? 0);
    const reward = Number(step.reward ?? 0);
    const done = Boolean(step.done);
    const action = step.action ?? null;
    const stepNumber = Number.isInteger(step.step) ? step.step : index;

    return {
      step: stepNumber,
      board,
      score,
      reward,
      done,
      action,
      info: step.info ?? null
    };
  });

  return {
    meta: {
      game: rawEpisode.meta?.game ?? "2048",
      board_size: size,
      agent: rawEpisode.meta?.agent ?? "Unknown",
      episode_name: rawEpisode.meta?.episode_name ?? "Replay session"
    },
    steps: normalizedSteps
  };
}

function getCurrentStep() {
  return state.episode?.steps?.[state.currentStepIndex] ?? null;
}

function getMaxTile(board) {
  return Math.max(...board.flat());
}

function getTilePalette(value) {
  if (TILE_COLORS[value]) {
    return TILE_COLORS[value];
  }

  return { bg: "#5e3716", text: "#fff8f0" };
}

function clearAnimations() {
  elements.animationLayer.innerHTML = "";
  elements.boardGrid.classList.remove(
    "board-swipe-up",
    "board-swipe-right",
    "board-swipe-down",
    "board-swipe-left"
  );
  elements.boardGrid.querySelectorAll(".merge-pulse, .spawn-pop").forEach((tile) => {
    tile.classList.remove("merge-pulse", "spawn-pop");
  });
}

function pulseMergedTiles(previousBoard, board) {
  const previousCounts = new Map();
  previousBoard.flat().forEach((value) => {
    if (value === 0) {
      return;
    }
    previousCounts.set(value, (previousCounts.get(value) ?? 0) + 1);
  });

  const currentCounts = new Map();
  board.flat().forEach((value) => {
    if (value === 0) {
      return;
    }
    currentCounts.set(value, (currentCounts.get(value) ?? 0) + 1);
  });

  board.forEach((row, rowIndex) => {
    row.forEach((value, colIndex) => {
      if (value < 4) {
        return;
      }

      const priorHalves = previousCounts.get(value / 2) ?? 0;
      const priorSame = previousCounts.get(value) ?? 0;
      const currentSame = currentCounts.get(value) ?? 0;

      if (priorHalves >= 2 && currentSame > priorSame) {
        const tile = elements.boardGrid.children[rowIndex * board.length + colIndex];
        tile?.classList.add("merge-pulse");
      }
    });
  });
}

function normalizeActionLabel(action) {
  if (!action) {
    return null;
  }

  const value = String(action).trim().toLowerCase();
  if (value.includes("up")) {
    return "up";
  }
  if (value.includes("right")) {
    return "right";
  }
  if (value.includes("down")) {
    return "down";
  }
  if (value.includes("left")) {
    return "left";
  }
  return null;
}

function pulseSpawnedTiles(previousBoard, board) {
  const previousCounts = new Map();
  previousBoard.flat().forEach((value) => {
    if (value === 0) {
      return;
    }
    previousCounts.set(value, (previousCounts.get(value) ?? 0) + 1);
  });

  const consumed = new Map();
  board.forEach((row, rowIndex) => {
    row.forEach((value, colIndex) => {
      if (value === 0) {
        return;
      }

      const used = consumed.get(value) ?? 0;
      const available = previousCounts.get(value) ?? 0;
      const tile = elements.boardGrid.children[rowIndex * board.length + colIndex];

      if (used >= available) {
        tile?.classList.add("spawn-pop");
      } else {
        consumed.set(value, used + 1);
      }
    });
  });
}

function animateBoardTransition(previousStep, board) {
  clearAnimations();

  if (!previousStep || previousStep.board.length !== board.length) {
    return;
  }

  const action = normalizeActionLabel(previousStep.action);
  if (action) {
    elements.boardGrid.classList.add(`board-swipe-${action}`);
    window.setTimeout(() => {
      elements.boardGrid.classList.remove(`board-swipe-${action}`);
    }, 240);
  }

  pulseMergedTiles(previousStep.board, board);
  pulseSpawnedTiles(previousStep.board, board);
}

const SAMPLE_EPISODE = {
  meta: {
    game: "2048",
    board_size: 4,
    agent: "demo_policy_v1",
    episode_name: "Greedy Merge Demo"
  },
  steps: [
    {
      step: 0,
      board: [[0, 2, 0, 0], [0, 0, 0, 0], [0, 0, 2, 0], [0, 0, 0, 0]],
      score: 0,
      action: null,
      reward: 0,
      done: false
    },
    {
      step: 1,
      board: [[2, 0, 0, 0], [0, 0, 0, 0], [2, 0, 0, 0], [0, 0, 2, 0]],
      score: 0,
      action: "left",
      reward: 0,
      done: false
    },
    {
      step: 2,
      board: [[4, 2, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [2, 0, 0, 0]],
      score: 4,
      action: "up",
      reward: 4,
      done: false
    },
    {
      step: 3,
      board: [[4, 2, 0, 0], [2, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 2]],
      score: 4,
      action: "down",
      reward: 0,
      done: false
    },
    {
      step: 4,
      board: [[0, 0, 4, 2], [0, 0, 0, 2], [0, 0, 0, 0], [0, 0, 0, 2]],
      score: 4,
      action: "right",
      reward: 0,
      done: false
    },
    {
      step: 5,
      board: [[0, 0, 4, 4], [0, 0, 0, 2], [0, 0, 0, 0], [0, 0, 0, 0]],
      score: 8,
      action: "up",
      reward: 4,
      done: false
    },
    {
      step: 6,
      board: [[0, 0, 0, 8], [0, 0, 0, 2], [0, 0, 0, 0], [0, 0, 0, 0]],
      score: 16,
      action: "right",
      reward: 8,
      done: true
    }
  ]
};

function renderBoard(board, previousBoard = null) {
  const size = board.length;
  elements.boardGrid.style.gridTemplateColumns = `repeat(${size}, minmax(0, 1fr))`;
  elements.boardGrid.innerHTML = "";

  board.forEach((row, rowIndex) => {
    row.forEach((value, colIndex) => {
      const tile = document.createElement("div");
      const palette = getTilePalette(value);
      const previousValue = previousBoard?.[rowIndex]?.[colIndex] ?? value;

      tile.className = "tile";
      tile.dataset.empty = String(value === 0);
      tile.dataset.highlight = String(value !== 0 && value !== previousValue);
      tile.style.background = palette.bg;
      tile.style.color = palette.text;
      tile.textContent = value === 0 ? "" : String(value);

      if (value >= 1024) {
        tile.style.fontSize = "1.9rem";
      } else if (value >= 128) {
        tile.style.fontSize = "2.2rem";
      }

      elements.boardGrid.appendChild(tile);
    });
  });
}

function renderStep() {
  const step = getCurrentStep();
  if (!step) {
    clearAnimations();
    renderBoard(createEmptyBoard());
    elements.scoreValue.textContent = "0";
    elements.maxTileValue.textContent = "0";
    elements.stepValue.textContent = "0 / 0";
    elements.timelineLabel.textContent = "Step 0";
    elements.actionValue.textContent = "-";
    elements.rewardValue.textContent = "0";
    elements.doneValue.textContent = "idle";
    elements.stepJsonView.textContent = "{}";
    return;
  }

  const previousStep = state.episode.steps[state.currentStepIndex - 1] ?? null;
  renderBoard(step.board);
  animateBoardTransition(previousStep, step.board);
  elements.scoreValue.textContent = String(step.score);
  elements.maxTileValue.textContent = String(getMaxTile(step.board));
  elements.stepValue.textContent = `${state.currentStepIndex} / ${state.episode.steps.length - 1}`;
  elements.timelineLabel.textContent = `Step ${step.step}`;
  elements.actionValue.textContent = step.action ?? "start";
  elements.rewardValue.textContent = String(step.reward);
  elements.doneValue.textContent = step.done ? "finished" : "running";
  elements.stepJsonView.textContent = JSON.stringify(step, null, 2);
  elements.timelineSlider.value = String(state.currentStepIndex);

  const status = step.done ? "Episode complete" : "Loaded";
  elements.statusPill.textContent = status;
}

function stopPlayback() {
  if (state.playbackTimer) {
    clearInterval(state.playbackTimer);
    state.playbackTimer = null;
  }
  elements.playPauseButton.textContent = "Play";
}

function startPlayback() {
  if (!state.episode || state.episode.steps.length <= 1) {
    return;
  }

  stopPlayback();
  elements.playPauseButton.textContent = "Pause";
  const intervalMs = 1000 / state.playbackSpeed;

  state.playbackTimer = setInterval(() => {
    if (state.currentStepIndex >= state.episode.steps.length - 1) {
      stopPlayback();
      return;
    }

    state.currentStepIndex += 1;
    renderStep();
  }, intervalMs);
}

function togglePlayback() {
  if (state.playbackTimer) {
    stopPlayback();
  } else {
    startPlayback();
  }
}

function showError(message) {
  elements.errorBox.textContent = message;
  elements.errorBox.classList.remove("hidden");
}

function clearError() {
  elements.errorBox.textContent = "";
  elements.errorBox.classList.add("hidden");
}

function updateEpisodeMeta(sourceName) {
  if (!state.episode) {
    elements.episodeTitle.textContent = "No episode loaded";
    elements.fileName.textContent = "None";
    elements.agentName.textContent = "Unknown";
    elements.boardSizeValue.textContent = "4x4";
    elements.statusPill.textContent = "Idle";
    return;
  }

  elements.episodeTitle.textContent = state.episode.meta.episode_name;
  elements.fileName.textContent = sourceName;
  elements.agentName.textContent = state.episode.meta.agent;
  elements.boardSizeValue.textContent = `${state.episode.meta.board_size}x${state.episode.meta.board_size}`;
  elements.timelineSlider.max = String(state.episode.steps.length - 1);
}

function loadEpisodeFromText(text, sourceName) {
  stopPlayback();
  clearError();

  let parsed;
  try {
    parsed = JSON.parse(text);
  } catch (error) {
    showError(`Invalid JSON: ${error.message}`);
    return;
  }

  try {
    state.episode = normalizeEpisode(parsed);
  } catch (error) {
    showError(error.message);
    return;
  }

  state.lastLoadedText = text;
  state.lastSourceName = sourceName;
  state.currentStepIndex = 0;
  updateEpisodeMeta(sourceName);
  renderStep();
}

function loadSampleEpisode() {
  clearError();
  loadEpisodeFromText(JSON.stringify(SAMPLE_EPISODE), "sample-episode.json");
}

function bindEvents() {
  elements.fileInput.addEventListener("change", async (event) => {
    const [file] = event.target.files ?? [];
    if (!file) {
      return;
    }

    const text = await file.text();
    loadEpisodeFromText(text, file.name);
  });

  elements.loadSampleButton.addEventListener("click", () => {
    loadSampleEpisode();
  });

  elements.reloadButton.addEventListener("click", () => {
    if (!state.lastLoadedText) {
      showError("No episode has been loaded yet.");
      return;
    }

    loadEpisodeFromText(state.lastLoadedText, state.lastSourceName);
  });

  elements.prevButton.addEventListener("click", () => {
    stopPlayback();
    if (!state.episode) {
      return;
    }

    state.currentStepIndex = Math.max(0, state.currentStepIndex - 1);
    renderStep();
  });

  elements.nextButton.addEventListener("click", () => {
    stopPlayback();
    if (!state.episode) {
      return;
    }

    state.currentStepIndex = Math.min(state.episode.steps.length - 1, state.currentStepIndex + 1);
    renderStep();
  });

  elements.restartButton.addEventListener("click", () => {
    stopPlayback();
    state.currentStepIndex = 0;
    renderStep();
  });

  elements.playPauseButton.addEventListener("click", () => {
    togglePlayback();
  });

  elements.timelineSlider.addEventListener("input", (event) => {
    stopPlayback();
    if (!state.episode) {
      return;
    }

    state.currentStepIndex = Number(event.target.value);
    renderStep();
  });

  elements.speedSlider.addEventListener("input", (event) => {
    state.playbackSpeed = Number(event.target.value);
    elements.speedLabel.textContent = `${state.playbackSpeed.toFixed(2).replace(/\.00$/, "")}x`;

    if (state.playbackTimer) {
      startPlayback();
    }
  });
}

function initialize() {
  bindEvents();
  updateEpisodeMeta("None");
  renderStep();
  elements.speedLabel.textContent = "1x";
  loadSampleEpisode();
}

initialize();
