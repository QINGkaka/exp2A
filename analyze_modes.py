#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


def resample(values: np.ndarray, points: int) -> np.ndarray:
    if len(values) == 0:
        return np.zeros((points, 3), dtype=float)
    source = np.linspace(0, 1, len(values))
    target = np.linspace(0, 1, points)
    return np.stack([np.interp(target, source, values[:, i]) for i in range(values.shape[1])], axis=1)


def feature(trace: dict, points: int) -> tuple[np.ndarray, dict]:
    steps = trace["steps"]
    left = np.asarray([row["left_endpose"][:3] for row in steps], dtype=float)
    right = np.asarray([row["right_endpose"][:3] for row in steps], dtype=float)
    lg = np.asarray([row["left_gripper"] for row in steps], dtype=float)
    rg = np.asarray([row["right_gripper"] for row in steps], dtype=float)
    if len(left):
        left = left - left[0]
        right = right - right[0]
    lr = resample(left, points)
    rr = resample(right, points)
    lpath = float(np.linalg.norm(np.diff(left, axis=0), axis=1).sum()) if len(left) > 1 else 0.0
    rpath = float(np.linalg.norm(np.diff(right, axis=0), axis=1).sum()) if len(right) > 1 else 0.0
    lclose = int(np.argmin(lg)) if len(lg) else 0
    rclose = int(np.argmin(rg)) if len(rg) else 0
    lclose_pose = left[lclose] if len(left) else np.zeros(3)
    rclose_pose = right[rclose] if len(right) else np.zeros(3)
    approach_window = max(1, min(8, len(left) - 1))
    lapproach = left[approach_window] - left[0] if len(left) > 1 else np.zeros(3)
    rapproach = right[approach_window] - right[0] if len(right) > 1 else np.zeros(3)
    contact_names = Counter()
    first_contact = len(steps)
    first_contact_position = np.zeros(3)
    for index, row in enumerate(steps):
        for contact in row.get("contacts", []):
            bodies = tuple(contact.get("bodies", []))
            if index < first_contact:
                first_contact = index
                points_at_contact = contact.get("points", [])
                if points_at_contact:
                    first_contact_position = np.asarray(points_at_contact[0], dtype=float)
            contact_names["|".join(bodies)] += 1
    scalars = np.array([
        len(steps), lpath, rpath, lclose / max(len(steps), 1), rclose / max(len(steps), 1),
        first_contact / max(len(steps), 1), float(lpath >= rpath), float(rpath > lpath),
    ])
    vector = np.concatenate((lr.ravel(), rr.ravel(), lclose_pose, rclose_pose,
                             lapproach, rapproach, first_contact_position, scalars))
    meta = {
        "steps": len(steps), "left_path_length": lpath, "right_path_length": rpath,
        "first_contact_fraction": scalars[5], "dominant_arm": "left" if lpath >= rpath else "right",
        "top_contact": contact_names.most_common(1)[0][0] if contact_names else "",
    }
    return vector, meta


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--clusters", type=int, default=6)
    parser.add_argument("--pca-components", type=int, default=12)
    parser.add_argument("--resample-points", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    traces = [json.loads(path.read_text(encoding="utf-8")) for path in sorted(args.run_dir.glob("trajectories/**/*.trace.json"))]
    if not traces:
        raise SystemExit(f"no traces under {args.run_dir}")
    output = args.run_dir / "analysis"
    output.mkdir(parents=True, exist_ok=True)
    assignment_rows = []
    mode_rows = []

    for task in sorted({row["task"] for row in traces}):
        subset = [row for row in traces if row["task"] == task]
        vectors, metadata = zip(*(feature(row, args.resample_points) for row in subset))
        matrix = StandardScaler().fit_transform(np.stack(vectors))
        components = min(args.pca_components, matrix.shape[0] - 1, matrix.shape[1])
        embedded = PCA(n_components=max(1, components), random_state=args.seed).fit_transform(matrix)
        clusters = min(args.clusters, len(subset))
        labels = KMeans(n_clusters=clusters, random_state=args.seed, n_init=20).fit_predict(embedded)
        silhouette = silhouette_score(embedded, labels) if 1 < clusters < len(subset) else float("nan")

        for trace, meta, label in zip(subset, metadata, labels):
            assignment_rows.append({
                "task": task, "method": trace["method"], "state_index": trace["state_index"],
                "rollout_index": trace["rollout_index"], "seed": trace["seed"],
                "policy_sample_seed": trace.get("policy_sample_seed"),
                "success": int(trace["success"]), "mode": int(label), **meta,
            })
        for label in range(clusters):
            members = [(trace, meta) for trace, meta, value in zip(subset, metadata, labels) if value == label]
            total = len(members)
            for method in ("no_wm", "wm"):
                method_members = [trace for trace, _ in members if trace["method"] == method]
                method_total = sum(trace["method"] == method for trace in subset)
                successes = sum(bool(trace["success"]) for trace in method_members)
                mode_rows.append({
                    "task": task, "mode": label, "method": method,
                    "count": len(method_members), "method_total": method_total,
                    "mode_probability": len(method_members) / method_total if method_total else 0,
                    "successes": successes,
                    "mode_success_rate": successes / len(method_members) if method_members else "",
                    "pooled_mode_count": total, "silhouette": silhouette,
                })

    for name, rows in (("assignments.csv", assignment_rows), ("mode_statistics.csv", mode_rows)):
        with (output / name).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader(); writer.writerows(rows)

    lines = ["# Experiment 2A Mode Summary", "",
             "Clustering used trajectory features only; model source and success labels were joined afterward.", ""]
    for task in sorted({row["task"] for row in mode_rows}):
        lines += [f"## {task}", "", "| Mode | No-WM probability | WM probability | Pooled success rate |", "|---:|---:|---:|---:|"]
        task_rows = [row for row in mode_rows if row["task"] == task]
        for mode in sorted({row["mode"] for row in task_rows}):
            pair = {row["method"]: row for row in task_rows if row["mode"] == mode}
            assigned = [row for row in assignment_rows if row["task"] == task and row["mode"] == mode]
            success = sum(row["success"] for row in assigned) / len(assigned)
            lines.append(f"| {mode} | {pair['no_wm']['mode_probability']:.3f} | {pair['wm']['mode_probability']:.3f} | {success:.3f} |")
        lines.append("")
    (output / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {output / 'summary.md'}")


if __name__ == "__main__":
    main()
