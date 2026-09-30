#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score, silhouette_score
from sklearn.preprocessing import StandardScaler


def resample(values: np.ndarray, points: int) -> np.ndarray:
    if len(values) == 0:
        return np.zeros((points, 3), dtype=float)
    source = np.linspace(0, 1, len(values))
    target = np.linspace(0, 1, points)
    return np.stack([np.interp(target, source, values[:, i]) for i in range(values.shape[1])], axis=1)


def normalized(vector: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vector)
    return vector / norm if norm > 1e-8 else np.zeros_like(vector)


def relative_quaternion(value: np.ndarray, origin: np.ndarray) -> np.ndarray:
    # SAPIEN uses wxyz quaternions. Canonicalizing w removes q/-q ambiguity.
    w1, x1, y1, z1 = value
    w2, x2, y2, z2 = origin * np.array([1.0, -1.0, -1.0, -1.0])
    result = np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ])
    result = normalized(result)
    return -result if result[0] < 0 else result


def first_close(values: np.ndarray) -> tuple[int, bool]:
    matches = np.flatnonzero(values <= 0.2)
    return (int(matches[0]), True) if len(matches) else (max(0, len(values) - 1), False)


def feature(trace: dict, points: int) -> tuple[np.ndarray, dict]:
    steps = trace["steps"]
    left = np.asarray([row["left_endpose"] for row in steps], dtype=float)
    right = np.asarray([row["right_endpose"] for row in steps], dtype=float)
    lg = np.asarray([row["left_gripper"] for row in steps], dtype=float)
    rg = np.asarray([row["right_gripper"] for row in steps], dtype=float)
    left_pos, right_pos = left[:, :3], right[:, :3]
    left_rel = left_pos - left_pos[0]
    right_rel = right_pos - right_pos[0]
    lr = resample(left_rel, points)
    rr = resample(right_rel, points)
    lpath = float(np.linalg.norm(np.diff(left_pos, axis=0), axis=1).sum()) if len(left) > 1 else 0.0
    rpath = float(np.linalg.norm(np.diff(right_pos, axis=0), axis=1).sum()) if len(right) > 1 else 0.0
    lclose, lclosed = first_close(lg)
    rclose, rclosed = first_close(rg)
    lclose_pose = np.concatenate((left_rel[lclose], relative_quaternion(left[lclose, 3:7], left[0, 3:7])))
    rclose_pose = np.concatenate((right_rel[rclose], relative_quaternion(right[rclose, 3:7], right[0, 3:7])))
    contact_names = Counter()
    first_contact = len(steps)
    first_contact_position = np.zeros(3)
    first_contact_arm = ""
    for index, row in enumerate(steps):
        for contact in row.get("contacts", []):
            if index < first_contact:
                first_contact = index
                first_contact_arm = contact.get("arm", "")
                points_at_contact = contact.get("points", [])
                if points_at_contact:
                    ee = left_pos[index] if first_contact_arm == "left" else right_pos[index]
                    first_contact_position = np.asarray(points_at_contact[0], dtype=float) - ee
            contact_names[contact.get("object", "")] += 1
    lanchor = min(lclose, first_contact) if first_contact_arm in ("", "left") else lclose
    ranchor = min(rclose, first_contact) if first_contact_arm in ("", "right") else rclose
    lapproach = normalized(left_pos[lanchor] - left_pos[max(0, lanchor - 8)])
    rapproach = normalized(right_pos[ranchor] - right_pos[max(0, ranchor - 8)])
    if first_contact_arm:
        dominant_arm = first_contact_arm
    elif lclosed != rclosed:
        dominant_arm = "left" if lclosed else "right"
    else:
        dominant_arm = "left" if lclose <= rclose else "right"
    scalars = np.array([
        lpath, rpath, lclose / max(len(steps) - 1, 1), rclose / max(len(steps) - 1, 1),
        first_contact / max(len(steps), 1), float(lclosed), float(rclosed),
        float(dominant_arm == "left"), float(dominant_arm == "right"),
    ])
    vector = np.concatenate((lr.ravel(), rr.ravel(), lclose_pose, rclose_pose,
                             lapproach, rapproach, first_contact_position, scalars))
    meta = {
        "steps": len(steps), "left_path_length": lpath, "right_path_length": rpath,
        "left_close_fraction": scalars[2], "right_close_fraction": scalars[3],
        "first_contact_fraction": scalars[4], "dominant_arm": dominant_arm,
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
    parser.add_argument("--allow-legacy-traces", action="store_true")
    args = parser.parse_args()

    traces = [json.loads(path.read_text(encoding="utf-8")) for path in sorted(args.run_dir.glob("trajectories/**/*.trace.json"))]
    if not traces:
        raise SystemExit(f"no traces under {args.run_dir}")
    legacy = [row for row in traces if int(row.get("schema_version", 1)) < 2]
    if legacy and not args.allow_legacy_traces:
        raise SystemExit(
            f"{len(legacy)} legacy traces lack reliable gripper contacts; rerun collection "
            "or pass --allow-legacy-traces only for debugging"
        )
    output = args.run_dir / "analysis"
    output.mkdir(parents=True, exist_ok=True)
    assignment_rows = []
    mode_rows = []
    state_mode_rows = []
    diagnostic_rows = []

    for task in sorted({row["task"] for row in traces}):
        subset = [row for row in traces if row["task"] == task]
        methods = {row["method"] for row in subset}
        if methods != {"no_wm", "wm"}:
            raise SystemExit(f"{task}: analysis requires both no_wm and wm, found {sorted(methods)}")
        for state in sorted({row["state_index"] for row in subset}):
            state_rows = [row for row in subset if row["state_index"] == state]
            counts = Counter(row["method"] for row in state_rows)
            if counts["no_wm"] != counts["wm"]:
                raise SystemExit(f"{task} state {state}: unbalanced methods {dict(counts)}")
            paired = {}
            for row in state_rows:
                key = row["rollout_index"]
                paired.setdefault(key, set()).add(row.get("policy_sample_seed"))
            mismatched = [rollout for rollout, seeds in paired.items() if len(seeds) != 1]
            if mismatched and not args.allow_legacy_traces:
                raise SystemExit(f"{task} state {state}: policy seeds differ for rollouts {mismatched[:5]}")
        vectors, metadata = zip(*(feature(row, args.resample_points) for row in subset))
        matrix = StandardScaler().fit_transform(np.stack(vectors))
        components = min(args.pca_components, matrix.shape[0] - 1, matrix.shape[1])
        embedded = PCA(n_components=max(1, components), random_state=args.seed).fit_transform(matrix)
        max_clusters = min(args.clusters, len(subset) - 1)
        candidates = []
        for clusters in range(2, max_clusters + 1):
            candidate_labels = KMeans(n_clusters=clusters, random_state=args.seed, n_init=20).fit_predict(embedded)
            score = silhouette_score(embedded, candidate_labels)
            candidates.append((score, clusters, candidate_labels))
        if candidates:
            silhouette, clusters, labels = max(candidates, key=lambda row: row[0])
        else:
            clusters, silhouette = 1, float("nan")
            labels = np.zeros(len(subset), dtype=int)
        repeat_labels = KMeans(n_clusters=clusters, random_state=args.seed + 1, n_init=20).fit_predict(embedded)
        stability = adjusted_rand_score(labels, repeat_labels)
        state_ami = adjusted_mutual_info_score([row["state_index"] for row in subset], labels)
        diagnostic_rows.append({
            "task": task, "selected_clusters": clusters, "silhouette": silhouette,
            "restart_adjusted_rand": stability, "state_adjusted_mutual_info": state_ami,
            "trace_count": len(subset), "contact_trace_rate": sum(any(s.get("contacts") for s in row["steps"]) for row in subset) / len(subset),
        })

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
        for state in sorted({row["state_index"] for row in subset}):
            for method in ("no_wm", "wm"):
                indices = [i for i, row in enumerate(subset) if row["state_index"] == state and row["method"] == method]
                for label in range(clusters):
                    count = sum(labels[i] == label for i in indices)
                    state_mode_rows.append({
                        "task": task, "state_index": state, "method": method, "mode": label,
                        "count": count, "total": len(indices),
                        "mode_probability": count / len(indices) if indices else "",
                    })

    for name, rows in (("assignments.csv", assignment_rows), ("mode_statistics.csv", mode_rows),
                       ("state_mode_statistics.csv", state_mode_rows), ("cluster_diagnostics.csv", diagnostic_rows)):
        with (output / name).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader(); writer.writerows(rows)

    lines = ["# Experiment 2A Mode Summary", "",
             "Clustering used trajectory features only; model source and success labels were joined afterward.", ""]
    for task in sorted({row["task"] for row in mode_rows}):
        diagnostic = next(row for row in diagnostic_rows if row["task"] == task)
        lines += [f"## {task}", "", "| Mode | No-WM probability | WM probability | Pooled success rate |", "|---:|---:|---:|---:|"]
        task_rows = [row for row in mode_rows if row["task"] == task]
        for mode in sorted({row["mode"] for row in task_rows}):
            pair = {row["method"]: row for row in task_rows if row["mode"] == mode}
            assigned = [row for row in assignment_rows if row["task"] == task and row["mode"] == mode]
            success = sum(row["success"] for row in assigned) / len(assigned)
            lines.append(f"| {mode} | {pair['no_wm']['mode_probability']:.3f} | {pair['wm']['mode_probability']:.3f} | {success:.3f} |")
        lines += ["", (
            f"Diagnostics: K={diagnostic['selected_clusters']}, silhouette={diagnostic['silhouette']:.3f}, "
            f"restart ARI={diagnostic['restart_adjusted_rand']:.3f}, "
            f"state AMI={diagnostic['state_adjusted_mutual_info']:.3f}, "
            f"contact coverage={diagnostic['contact_trace_rate']:.1%}."
        ), ""]
    (output / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {output / 'summary.md'}")


if __name__ == "__main__":
    main()
