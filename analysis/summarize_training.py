#!/usr/bin/env python3
"""Collect checkpoint learning curves and compute trapezoidal accuracy AUC."""

import argparse
import json
from pathlib import Path


def trapezoid_auc(points):
    if len(points) < 2:
        return None
    area = 0.0
    for (left_x, left_y), (right_x, right_y) in zip(points, points[1:]):
        area += (right_x - left_x) * (left_y + right_y) / 2
    return area / (points[-1][0] - points[0][0])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dirs", nargs="+")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    runs = []
    for run_dir_text in args.run_dirs:
        run_dir = Path(run_dir_text)
        state_path = run_dir / "trainer_state.json"
        if not state_path.exists():
            checkpoints = sorted(run_dir.glob("checkpoint-*/trainer_state.json"))
            if not checkpoints:
                raise FileNotFoundError(f"No trainer_state.json below {run_dir}")
            state_path = checkpoints[-1]
        state = json.loads(state_path.read_text(encoding="utf-8"))
        history = state.get("log_history", [])
        curve = sorted({
            (int(row["step"]), float(row.get("eval_accuracy", row.get("accuracy"))))
            for row in history
            if "step" in row and ("eval_accuracy" in row or "accuracy" in row)
        })
        latest = {}
        for row in history:
            latest.update({key: value for key, value in row.items() if isinstance(value, (int, float))})
        runs.append({
            "run_dir": str(run_dir),
            "curve": curve,
            "accuracy_auc": trapezoid_auc(curve),
            "latest_metrics": latest,
        })
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(runs, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(runs, indent=2))


if __name__ == "__main__":
    main()
