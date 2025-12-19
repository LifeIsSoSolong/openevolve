# plotAlphaEvolveTree 简要使用说明

目录下有两类绘图脚本，输入均为 `evolution_trace.jsonl`（OpenEvolve 的轨迹文件）：

## 1. 单图模式：`plot_evolution_trace.py`
- 功能：读取一份 `evolution_trace.jsonl`，画出整体演化图（左侧 parent→child 箭头，右侧 best score 曲线）。
- 输入：`--trace <path/to/evolution_trace.jsonl>`  
- 输出：`--out <out.png/svg/pdf/dot>`  
- 示例：
  ```bash
  python plot_evolution_trace.py --trace ../../bean01/outputs/evolution_trace.jsonl --out evolution.png
  ```

## 2. 逐步模式：`plot_step.py`
- 功能：基于同一份 `evolution_trace.jsonl`，按迭代逐步生成帧（step-1…step-n），方便连播动画。绘图时使用全局坐标范围，避免帧间跳动。
- 输入：`--trace <path/to/evolution_trace.jsonl>`  
- 输出目录：`--out <frames_dir>`（默认 `./frames_step`）  
- 输出格式：`--format <png/svg/pdf...>`（默认 `png`）  
- 示例：
  ```bash
  python plot_step.py --trace ../../bean01/outputs/evolution_trace.jsonl --out ./frames_step --format png
  ```
  生成的帧文件形如 `frames_step/step-1.png` … `step-n.png`。

## 其他说明
- 0 分节点会被放在 x=-1 竖排显示；虚拟根节点在 x=0，汇聚所有无父节点的起点。
- 依赖：`matplotlib`、`numpy`（无需系统 graphviz）。如缺失请先安装：
  ```bash
  pip install matplotlib numpy
  ```
