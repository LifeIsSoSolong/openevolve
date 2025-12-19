#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Plot evolution trace from an OpenEvolve-style evolution_trace.jsonl file.

Usage:
    python plot_evolution_trace.py --trace evolution_trace.jsonl --out evolution.png
    python plot_evolution_trace.py --trace evolution_trace.jsonl --out evolution.svg
    python plot_evolution_trace.py --trace evolution_trace.jsonl --out evolution.pdf
    python plot_evolution_trace.py --trace evolution_trace.jsonl --out evolution.dot
"""

import argparse
import json
import collections
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from matplotlib.ticker import MaxNLocator


def extract_score(metrics: dict):
    """Safely extract combined_score from a metrics dict, ignoring error cases."""
    if not isinstance(metrics, dict):
        return None
    # 即便有 error，也尝试读取 combined_score 为数值
    cs = metrics.get("combined_score")
    try:
        return float(cs)
    except Exception:
        return None


def load_trace(trace_path: Path):
    """
    读取 evolution_trace.jsonl，构建节点、边和每一代的 best score 轨迹。

    Returns:
        nodes: dict[node_id] = {
            "score": float | None,
            "metrics": dict | None,
            "first_iter": int,
            "island_id": int | None,
        }
        edges: list of {
            "parent": str,
            "child": str,
            "iteration": int,
            "generation": int | None,
            "island_id": int | None,
        }
        iteration_best: list of (iteration:int, best_score:float|None)
    """
    nodes = {}
    edges = []
    per_iter_scores = collections.defaultdict(list)

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
            # 记录最早出现的迭代
            if iteration < info["first_iter"]:
                info["first_iter"] = iteration
            # 补充 island_id
            if info["island_id"] is None and island_id is not None:
                info["island_id"] = island_id
            # 用更好的分数更新
            if score is not None:
                if info["score"] is None or score > info["score"]:
                    info["score"] = score
                    info["metrics"] = metrics

    with trace_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)

            iteration = obj.get("iteration")
            if iteration is not None:
                iteration = int(iteration)

            generation = obj.get("generation")
            island_id = obj.get("island_id")

            parent_id = obj.get("parent_id")
            child_id = obj.get("child_id")

            parent_metrics = obj.get("parent_metrics") or {}
            child_metrics = obj.get("child_metrics") or {}

            # 更新节点信息
            if parent_id is not None and iteration is not None:
                update_node(parent_id, parent_metrics, iteration, island_id)
            if child_id is not None and iteration is not None:
                update_node(child_id, child_metrics, iteration, island_id)

            # 记录边
            if parent_id is not None and child_id is not None and iteration is not None:
                edges.append(
                    {
                        "parent": parent_id,
                        "child": child_id,
                        "iteration": iteration,
                        "generation": generation,
                        "island_id": island_id,
                    }
                )

            # 用 parent/child 的有效分数更新 per_iter_scores
            if iteration is not None:
                for metrics in (parent_metrics, child_metrics):
                    score = extract_score(metrics)
                    if score is not None:
                        per_iter_scores[iteration].append(score)

    # 计算每一迭代的“到目前为止的最佳分数”
    iteration_best = []
    best_so_far = None
    for it in sorted(per_iter_scores.keys()):
        scores = per_iter_scores[it]
        if scores:
            iter_best = max(scores)
            if best_so_far is None or iter_best > best_so_far:
                best_so_far = iter_best
        iteration_best.append((it, best_so_far))

    return nodes, edges, iteration_best


def draw_figure(nodes, edges, iteration_best, out_path: Path, xlim=None, ylim=None):
    """
    画出左侧 evolution tree（y 轴 = score，0 分节点单独左侧） + 右侧 best score 曲线。
    """
    if not nodes:
        raise RuntimeError("No nodes parsed from trace file.")

    # 收集 score
    scores_nonzero = [
        info["score"] for info in nodes.values() if info["score"] is not None and info["score"] > 0
    ]
    zero_nodes = [nid for nid, info in nodes.items() if info["score"] is not None and info["score"] == 0]

    if not scores_nonzero and not zero_nodes:
        raise RuntimeError("No valid 'combined_score' found in trace.")

    min_s = min(scores_nonzero) if scores_nonzero else 0.0
    max_s = max(scores_nonzero) if scores_nonzero else 1.0
    span = max_s - min_s if max_s is not None else 0.0

    # 归一化 island_id：无 island 的记为 -1
    for info in nodes.values():
        if info["island_id"] is None:
            info["island_id"] = -1

    # 给不同 island 分配更好看的离散颜色
    island_ids = sorted(set(info["island_id"] for info in nodes.values()))
    num_islands = len(island_ids)
    if num_islands <= 8:
        base_cmap = cm.get_cmap("Set2", 8)   # 柔和又有区分度
    else:
        base_cmap = cm.get_cmap("tab20", 20)  # 岛多时用更大的盘子

    island_to_color = {
        isl: base_cmap(i % base_cmap.N) for i, isl in enumerate(island_ids)
    }

    node_x = []
    node_y = []
    node_colors = []
    node_sizes = []

    coords = {}  # nid -> (x, y)

    # 非零节点正常按 score 绘制
    for nid, info in nodes.items():
        s = info["score"]
        if s is None or s == 0 or info["first_iter"] is None:
            continue
        x = info["first_iter"]
        y = s
        coords[nid] = (x, y)
        node_x.append(x)
        node_y.append(y)

        color = island_to_color[info["island_id"]]
        if span > 1e-12:
            t = (s - min_s) / span
            size = 60.0 + 220.0 * t
        else:
            size = 160.0

        node_colors.append(color)
        node_sizes.append(size)

    # 0 分节点统一放在左侧竖排
    if zero_nodes:
        zero_x = -1  # 固定放在 x = -1
        if scores_nonzero:
            y_positions = np.linspace(min_s, max_s, len(zero_nodes) + 2)[1:-1]
        else:
            y_positions = range(len(zero_nodes))
        for nid, y in zip(zero_nodes, y_positions):
            coords[nid] = (zero_x, y)
            node_x.append(zero_x)
            node_y.append(y)
            node_colors.append("#cccccc")
            node_sizes.append(20.0)

    # 准备迭代最优曲线
    iters = [it for it, _ in iteration_best]
    best_scores = [bs for _, bs in iteration_best]

    # 画图
    fig = plt.figure(figsize=(14, 7), dpi=120)
    gs = fig.add_gridspec(1, 2, width_ratios=[2.0, 1.0])

    # 左：演化图（score vs iteration，0 分节点在左侧）
    ax_tree = fig.add_subplot(gs[0, 0])

    # 构造用于绘制的边列表（可添加虚拟根）
    edges_plot = list(edges)

    # 找到没有父节点的起点（in-degree==0），连向虚拟根
    child_ids = {e["child"] for e in edges}
    root_nodes = [nid for nid in coords if nid not in child_ids]
    has_virtual_root = False
    if root_nodes:
        root_score = nodes[root_nodes[0]]["score"]
        if root_score is None:
            root_score = min_s
        root_id = "__virtual_root__"
        coords[root_id] = (0, root_score)
        node_x.append(0)
        node_y.append(root_score)
        node_colors.append("#444444")
        node_sizes.append(220.0)
        has_virtual_root = True
        for rn in root_nodes:
            edges_plot.append({"parent": root_id, "child": rn})

    # 先画边，再画点
    for e in edges_plot:
        p = e["parent"]
        c = e["child"]
        if p not in coords or c not in coords:
            # 如果 parent 或 child 没有有效 score，就不画这条边
            continue
        x0, y0 = coords[p]
        x1, y1 = coords[c]
        ax_tree.annotate(
            "",
            xy=(x1, y1),
            xytext=(x0, y0),
            arrowprops=dict(
                arrowstyle="->",
                color="#1565C0",  # 更醒目的蓝色
                lw=1.2,
                alpha=0.7,
                shrinkA=2,
                shrinkB=2,
            ),
        )

    sc = ax_tree.scatter(
        node_x,
        node_y,
        s=node_sizes,
        c=node_colors,
        edgecolors="black",
        linewidths=0.3,
        alpha=0.9,
    )

    ax_tree.set_xlabel("Iteration (first appearance)")
    ax_tree.set_ylabel("Combined Score")
    ax_tree.set_title("Evolution Tree (Arrows: parent -> child)")
    ax_tree.grid(True, linestyle="--", alpha=0.25)
    # 横轴强制整数刻度
    ax_tree.xaxis.set_major_locator(MaxNLocator(integer=True))
    # y/x 轴范围
    if ylim is not None:
        ax_tree.set_ylim(*ylim)
    else:
        if span > 0:
            margin = span * 0.08
        else:
            margin = 0.5
        ax_tree.set_ylim(min_s - margin, max_s + margin)
    if xlim is not None:
        ax_tree.set_xlim(*xlim)

    # 为 island 做个 legend
    handles = []
    labels = []
    for isl in island_ids:
        handles.append(
            plt.Line2D(
                [0],
                [0],
                marker="o",
                linestyle="",
                markerfacecolor=island_to_color[isl],
                markeredgecolor="black",
                markersize=6,
            )
        )
        labels.append(f"island {isl}")

    if zero_nodes:
        handles.append(
            plt.Line2D(
                [0],
                [0],
                marker="o",
                linestyle="",
                markerfacecolor="#cccccc",
                markeredgecolor="black",
                markersize=5,
            )
        )
        labels.append("score=0")

    if has_virtual_root:
        handles.append(
            plt.Line2D(
                [0],
                [0],
                marker="o",
                linestyle="",
                markerfacecolor="#444444",
                markeredgecolor="black",
                markersize=7,
            )
        )
        labels.append("root")
    ax_tree.legend(handles, labels, title="Islands", loc="best", fontsize=8)

    # 右：best score 曲线
    ax_curve = fig.add_subplot(gs[0, 1])
    if iters:
        xy = sorted(
            [(it, bs) for it, bs in zip(iters, best_scores) if bs is not None],
            key=lambda x: x[0],
        )
        if xy:
            sorted_iters = [x for x, _ in xy]
            sorted_best = [y for _, y in xy]
            ax_curve.plot(
                sorted_iters,
                sorted_best,
                marker="o",
                linewidth=1.8,
                markersize=4,
                color="#FF7043",  # 柔和的橙色
            )
        ax_curve.set_xlabel("Iteration")
        ax_curve.set_ylabel("Best Combined Score (so far)")
        ax_curve.set_title("Best Fitness Over Iterations")
        ax_curve.grid(True, linestyle="--", alpha=0.3)
        # 横轴强制整数刻度
        ax_curve.xaxis.set_major_locator(MaxNLocator(integer=True))
    else:
        ax_curve.text(
            0.5,
            0.5,
            "No score information",
            ha="center",
            va="center",
            transform=ax_curve.transAxes,
        )
        ax_curve.axis("off")

    fig.suptitle("Evolution Trace Visualization", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0.02, 1, 0.96])

    fig.savefig(str(out_path), bbox_inches="tight")
    plt.close(fig)


def write_dot(nodes, edges, out_path: Path):
    """
    输出 Graphviz DOT 文件（节点 label 包含 score），方便用 dot/neato 再作图。
    """
    if not nodes:
        raise RuntimeError("No nodes parsed from trace file.")

    scores = [info["score"] for info in nodes.values() if info["score"] is not None]
    has_scores = bool(scores)
    if has_scores:
        min_s, max_s = min(scores), max(scores)
        span = max_s - min_s if max_s is not None else 0.0

    def score_color(score):
        if not has_scores or score is None:
            return "#B0B0B0"
        if span > 1e-12:
            t = (score - min_s) / span
        else:
            t = 0.7
        rgba = cm.plasma(t)
        r, g, b = [int(255 * v) for v in rgba[:3]]
        return f"#{r:02X}{g:02X}{b:02X}"

    with out_path.open("w", encoding="utf-8") as f:
        f.write("digraph evolution {\n")
        f.write("  rankdir=LR;\n")
        f.write('  node [shape=circle, style=filled, fontname="Helvetica"];\n')

        for nid, info in nodes.items():
            short_id = nid.split("-")[0]
            s = info["score"]
            if s is None:
                label = short_id
            else:
                label = f"{short_id}\\n{float(s):.4f}"
            color = score_color(s)
            f.write(f'  "{nid}" [label="{label}", fillcolor="{color}"];\n')

        for e in edges:
            p = e["parent"]
            c = e["child"]
            f.write(f'  "{p}" -> "{c}";\n')

        f.write("}\n")


def main():
    parser = argparse.ArgumentParser(description="Plot evolution trace graph.")
    parser.add_argument("--trace", required=True, help="Path to evolution_trace.jsonl")
    parser.add_argument(
        "--out",
        required=True,
        help="Output graph file (png/svg/pdf/dot)",
    )

    args = parser.parse_args()
    trace_path = Path(args.trace)
    out_path = Path(args.out)

    if not trace_path.is_file():
        raise FileNotFoundError(f"Trace file not found: {trace_path}")

    nodes, edges, iteration_best = load_trace(trace_path)

    ext = out_path.suffix.lower()
    if ext in {".dot", ".gv"}:
        write_dot(nodes, edges, out_path)
    else:
        draw_figure(nodes, edges, iteration_best, out_path)


if __name__ == "__main__":
    main()
