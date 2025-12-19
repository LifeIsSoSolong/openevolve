"""
Generate per-step evolution snapshots using plot_evolution_trace.py utilities.

Usage:
    python plot_step.py --trace ../../bean01/outputs/evolution_trace.jsonl --out ./frames_step --format png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Any, Tuple

from plot_evolution_trace import extract_score, draw_figure


def build_from_records(records: List[Dict[str, Any]]) -> Tuple[Dict, List[Dict], List[Tuple[int, float | None]]]:
    """Mimic load_trace but from in-memory records list."""
    nodes: Dict[str, Dict[str, Any]] = {}
    edges: List[Dict[str, Any]] = []
    per_iter_scores: Dict[int, List[float]] = {}

    def update_node(node_id, metrics, iteration, island_id):
        score = extract_score(metrics)
        if node_id not in nodes:
            nodes[node_id] = {
                "score": score,
                "metrics": metrics if score is not None else None,
                "first_iter": iteration,
                "island_id": island_id,
            }
        else:
            info = nodes[node_id]
            if iteration < info["first_iter"]:
                info["first_iter"] = iteration
            if info["island_id"] is None and island_id is not None:
                info["island_id"] = island_id
            if score is not None and (info["score"] is None or score > info["score"]):
                info["score"] = score
                info["metrics"] = metrics

    for obj in records:
        iteration = obj.get("iteration")
        if iteration is None:
            continue
        iteration = int(iteration)

        generation = obj.get("generation")
        island_id = obj.get("island_id")

        parent_id = obj.get("parent_id")
        child_id = obj.get("child_id")

        parent_metrics = obj.get("parent_metrics") or {}
        child_metrics = obj.get("child_metrics") or {}

        if parent_id is not None:
            update_node(parent_id, parent_metrics, iteration, island_id)
        if child_id is not None:
            update_node(child_id, child_metrics, iteration, island_id)

        if parent_id is not None and child_id is not None:
            edges.append(
                {
                    "parent": parent_id,
                    "child": child_id,
                    "iteration": iteration,
                    "generation": generation,
                    "island_id": island_id,
                }
            )

        for metrics in (parent_metrics, child_metrics):
            score = extract_score(metrics)
            if score is not None:
                per_iter_scores.setdefault(iteration, []).append(score)

    iteration_best: List[Tuple[int, float | None]] = []
    best_so_far = None
    for it in sorted(per_iter_scores.keys()):
        scores = per_iter_scores[it]
        if scores:
            iter_best = max(scores)
            if best_so_far is None or iter_best > best_so_far:
                best_so_far = iter_best
        iteration_best.append((it, best_so_far))

    return nodes, edges, iteration_best


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot per-step evolution graphs.")
    parser.add_argument("--trace", required=True, help="Path to evolution_trace.jsonl")
    parser.add_argument("--out", default="./frames_step", help="Output directory for frames")
    parser.add_argument("--format", default="png", help="Image format (png/svg/pdf...)")
    args = parser.parse_args()

    trace_path = Path(args.trace)
    if not trace_path.exists():
        raise FileNotFoundError(trace_path)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    records: List[Dict[str, Any]] = []
    max_iter = 0
    with trace_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if "iteration" in obj and obj["iteration"] is not None:
                max_iter = max(max_iter, int(obj["iteration"]))
            records.append(obj)

    print(f"Loaded {len(records)} records, max iteration = {max_iter}")

    # 预先计算全局坐标范围
    all_scores_nonzero = [
        extract_score(r.get("parent_metrics") or {}) for r in records if extract_score(r.get("parent_metrics") or {}) not in (None, 0)
    ] + [
        extract_score(r.get("child_metrics") or {}) for r in records if extract_score(r.get("child_metrics") or {}) not in (None, 0)
    ]
    all_scores_nonzero = [s for s in all_scores_nonzero if s is not None and s > 0]
    if all_scores_nonzero:
        min_s = min(all_scores_nonzero)
        max_s = max(all_scores_nonzero)
        span = max_s - min_s
        margin = span * 0.08 if span > 0 else 0.5
        ylim = (min_s - margin, max_s + margin)
    else:
        ylim = (0, 1)

    xlim = (-1.5, max_iter + 0.5)

    for step in range(1, max_iter + 1):
        subset = [r for r in records if r.get("iteration") is not None and int(r["iteration"]) <= step]
        nodes, edges, iteration_best = build_from_records(subset)
        out_path = out_dir / f"step-{step}.{args.format}"
        draw_figure(nodes, edges, iteration_best, out_path, xlim=xlim, ylim=ylim)
        print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
