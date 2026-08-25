#!/usr/bin/env python3
"""Aggregate Harbor job results into a table: mean reward +/- stderr across repeats."""
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path


def load(jobs_dir: Path):
    """Yield (model, dataset, mean_reward, n_trials, n_errors) per job."""
    for result in sorted(jobs_dir.glob("*/result.json")):
        name = result.parent.name
        parts = name.split("__")
        if len(parts) < 3:
            continue
        model, dataset = parts[0], parts[1]
        try:
            data = json.loads(result.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        for eval_stats in (data.get("stats", {}).get("evals") or {}).values():
            metrics = eval_stats.get("metrics") or [{}]
            yield (
                model,
                dataset,
                metrics[0].get("mean"),
                eval_stats.get("n_trials", 0),
                eval_stats.get("n_errors", 0),
            )


def main() -> int:
    jobs_dir = Path(sys.argv[1] if len(sys.argv) > 1 else "jobs")
    if not jobs_dir.is_dir():
        print(f"no such directory: {jobs_dir}", file=sys.stderr)
        return 1

    runs = defaultdict(list)
    errors = defaultdict(int)
    for model, dataset, mean, _n_trials, n_errors in load(jobs_dir):
        if mean is not None:
            runs[(model, dataset)].append(mean)
        errors[(model, dataset)] += n_errors

    if not runs:
        print(f"no results under {jobs_dir}")
        return 0

    datasets = sorted({d for _, d in runs})
    models = sorted({m for m, _ in runs})
    width = max(len(m) for m in models) + 2

    header = "model".ljust(width) + "".join(d.ljust(22) for d in datasets)
    print(header)
    print("-" * len(header))
    for model in models:
        row = model.ljust(width)
        for dataset in datasets:
            values = runs.get((model, dataset))
            if not values:
                row += "-".ljust(22)
                continue
            mean = statistics.mean(values)
            # stderr is undefined for a single repeat
            if len(values) > 1:
                stderr = statistics.stdev(values) / len(values) ** 0.5
                cell = f"{mean * 100:.1f} +/- {stderr * 100:.1f} (n={len(values)})"
            else:
                cell = f"{mean * 100:.1f} (n=1)"
            row += cell.ljust(22)
        print(row)

    flagged = {k: v for k, v in errors.items() if v}
    if flagged:
        print("\nWARNING: errored trials are excluded from the mean, not scored zero.")
        for (model, dataset), count in sorted(flagged.items()):
            print(f"  {model} / {dataset}: {count} errored trial(s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
