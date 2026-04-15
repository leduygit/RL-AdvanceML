import argparse
import csv
import json
from pathlib import Path

from tqdm.auto import tqdm

from export_2048_replay import load_checkpoint, rollout_episode


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Sweep multiple rollout seeds for one or many checkpoints, "
            "then report best-seed metrics and summary averages."
        )
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Path to a saved .pt checkpoint. If omitted, sweep all .pt files in the current directory.",
    )
    parser.add_argument("--seed-start", type=int, default=0, help="Start seed (inclusive).")
    parser.add_argument("--seed-end", type=int, default=5000, help="End seed (inclusive).")
    parser.add_argument("--max-steps", type=int, default=5000, help="Maximum number of steps per rollout.")
    parser.add_argument(
        "--obs-mode",
        choices=["auto", "tensor", "log2_board", "onehot_board"],
        default="auto",
        help="Observation preprocessing mode. Use auto to infer from checkpoint obs_dim.",
    )
    parser.add_argument(
        "--output-best-json",
        default=None,
        help="Optional path to save the replay JSON of the best seed.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="How many top seeds to print in the summary table.",
    )
    parser.add_argument(
        "--output-csv",
        default="seed_sweep_summary.csv",
        help="Path to save CSV summary. One row per checkpoint.",
    )
    args = parser.parse_args()

    if args.seed_end < args.seed_start:
        raise ValueError("seed-end must be >= seed-start")

    if args.checkpoint:
        checkpoint_paths = [Path(args.checkpoint)]
    else:
        checkpoint_paths = sorted(Path.cwd().glob("*.pt"))

    if not checkpoint_paths:
        raise FileNotFoundError("No .pt files found in current directory.")

    missing = [p for p in checkpoint_paths if not p.exists()]
    if missing:
        missing_display = ", ".join(str(p) for p in missing)
        raise FileNotFoundError(f"Checkpoint not found: {missing_display}")

    summary_rows = []

    for checkpoint_path in checkpoint_paths:
        print(f"\n=== Checkpoint: {checkpoint_path} ===")
        checkpoint, model = load_checkpoint(checkpoint_path)
        checkpoint["checkpoint_path"] = str(checkpoint_path)

        results = []
        best_replay = None
        best_entry = None

        seed_range = range(args.seed_start, args.seed_end + 1)
        for seed in tqdm(seed_range, desc=f"Sweeping seeds ({checkpoint_path.name})"):
            replay = rollout_episode(
                model=model,
                checkpoint=checkpoint,
                seed=seed,
                max_steps=args.max_steps,
                obs_mode=args.obs_mode,
            )

            total_steps = int(replay["meta"]["total_steps"])
            final_score = float(replay["meta"]["final_score"])
            max_tile = int(replay["meta"]["max_tile"])

            entry = {
                "seed": seed,
                "steps": total_steps,
                "score": final_score,
                "max_tile": max_tile,
            }
            results.append(entry)

            if best_entry is None:
                best_entry = entry
                best_replay = replay
            else:
                # Primary sort: steps desc, secondary: score desc, tertiary: smaller seed
                is_better = (
                    entry["steps"] > best_entry["steps"]
                    or (
                        entry["steps"] == best_entry["steps"]
                        and entry["score"] > best_entry["score"]
                    )
                    or (
                        entry["steps"] == best_entry["steps"]
                        and entry["score"] == best_entry["score"]
                        and entry["seed"] < best_entry["seed"]
                    )
                )
                if is_better:
                    best_entry = entry
                    best_replay = replay

        results_sorted = sorted(
            results,
            key=lambda x: (-x["steps"], -x["score"], x["seed"]),
        )

        top_k = max(1, min(args.top_k, len(results_sorted)))
        top_rows = results_sorted[:top_k]

        avg_steps_all = sum(row["steps"] for row in results) / len(results)
        avg_score_all = sum(row["score"] for row in results) / len(results)
        avg_max_tile_all = sum(row["max_tile"] for row in results) / len(results)

        avg_steps_topk = sum(row["steps"] for row in top_rows) / len(top_rows)
        avg_score_topk = sum(row["score"] for row in top_rows) / len(top_rows)
        avg_max_tile_topk = sum(row["max_tile"] for row in top_rows) / len(top_rows)

        summary_rows.append(
            {
                "checkpoint": checkpoint_path.name,
                "seed_start": args.seed_start,
                "seed_end": args.seed_end,
                "max_steps": args.max_steps,
                "top_k": top_k,
                "best_seed": best_entry["seed"],
                "best_steps": best_entry["steps"],
                "best_score": f"{best_entry['score']:.4f}",
                "best_max_tile": best_entry["max_tile"],
                "avg_steps_all": f"{avg_steps_all:.4f}",
                "avg_score_all": f"{avg_score_all:.4f}",
                "avg_max_tile_all": f"{avg_max_tile_all:.4f}",
                "avg_steps_topk": f"{avg_steps_topk:.4f}",
                "avg_score_topk": f"{avg_score_topk:.4f}",
                "avg_max_tile_topk": f"{avg_max_tile_topk:.4f}",
            }
        )

        print("All-seed averages:")
        print(f"Average steps: {avg_steps_all:.2f}")
        print(f"Average score: {avg_score_all:.2f}")
        print(f"Average max tile: {avg_max_tile_all:.2f}")

        print("Best seed summary:")
        print(f"Seed: {best_entry['seed']}")
        print(f"Steps: {best_entry['steps']}")
        print(f"Final score: {best_entry['score']:.1f}")
        print(f"Max tile: {best_entry['max_tile']}")

        if args.output_best_json and len(checkpoint_paths) == 1:
            output_path = Path(args.output_best_json)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(json.dumps(best_replay, indent=2), encoding="utf-8")
            print(f"Best replay JSON saved to: {output_path}")

        print(f"Top {top_k} seeds (by steps, then score):")
        print(f"{'rank':>4}  {'seed':>8}  {'steps':>8}  {'score':>12}  {'max_tile':>8}")
        for idx, row in enumerate(top_rows, start=1):
            print(
                f"{idx:>4}  {row['seed']:>8}  {row['steps']:>8}  {row['score']:>12.1f}  {row['max_tile']:>8}"
            )

    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "checkpoint",
        "seed_start",
        "seed_end",
        "max_steps",
        "top_k",
        "best_seed",
        "best_steps",
        "best_score",
        "best_max_tile",
        "avg_steps_all",
        "avg_score_all",
        "avg_max_tile_all",
        "avg_steps_topk",
        "avg_score_topk",
        "avg_max_tile_topk",
    ]

    with output_csv.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)

    print(f"\nCSV summary saved to: {output_csv}")


if __name__ == "__main__":
    main()
