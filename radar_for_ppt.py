#!/usr/bin/env python3
"""
draw_radar.py - 绘制三张雷达图：
1. ASan 数据图（Result/Potential/Baseline）
2. CRISP 数据图（Result/Potential/Baseline）
3. 纯说明图（维度为 bugtype，标注各层级数值，底部水平颜色图例）

使用项目虚拟环境运行:
    .venv/bin/python draw_radar.py --prefix radar
"""

import argparse
import math

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# ---------------------------------------------------------------------------
# 配色与样式常量
# ---------------------------------------------------------------------------
COLORS = {
    "blue": "#1f77b4",
    "orange": "#ff7f0e",
    "gray": "#7f7f7f",
    "light_gray": "#cccccc",
}

DIMENSION_LABELS = [
    "Double-free",
    "Non-Linear OOBA",
    "Linear OOBA",
    "Type Confusion OOBA",
    "Use-after-*",
]
DATA = {
    "ASan": [1, 0.3889, 0.8889, 0.8, 0.5],
    "CRISP": [1, 0.5, 1, 1, 0.75],
    "Baseline": [0, 0.25, 0.3778, 0.40, 0],
}


def get_pentagon_coords(R: float = 0.3):
    """计算正五边形顶点坐标（第一个顶点朝下）。"""
    N = 5
    angles = [2 * math.pi * i / N - math.pi / 2 for i in range(N)]
    axis_x = [R * math.cos(a) for a in angles]
    axis_y = [R * math.sin(a) for a in angles]
    axis_x_closed = axis_x + [axis_x[0]]
    axis_y_closed = axis_y + [axis_y[0]]
    return axis_x, axis_y, axis_x_closed, axis_y_closed


def draw_radar_grid(ax, axis_x, axis_y, axis_x_closed, axis_y_closed, labels, description = False):
    """绘制正五边形背景、网格、轴线与维度标签。"""
    # 最外层浅灰背景
    ax.fill(axis_x_closed, axis_y_closed, color="#cccccc", alpha=0.10)

    # 同心五边形网格
    grid_levels = [0.2, 0.4, 0.6, 0.8, 1.0]
    for level in grid_levels:
        gx = [x * level for x in axis_x_closed]
        gy = [y * level for y in axis_y_closed]
        ax.plot(gx, gy, color=COLORS["light_gray"], linewidth=0.8)

    # 轴线
    for i in range(5):
        ax.plot([0, axis_x[i]], [0, axis_y[i]], color=COLORS["light_gray"], linewidth=0.8)

    # # 维度标签
    # if description:
    label_offset = 1.18
    for i in range(5):
        lx = axis_x[i] * label_offset
        ly = axis_y[i] * label_offset
        if i == 1:  # Non-Linear OOBA (right-bottom)
            lx = axis_x[i] * 1.55
            ly = axis_y[i] * 1
            ax.text(lx, ly, "Non-Linear\nOOBA", fontsize=10, ha="center", va="center")
        elif i == 4:  # Use-after-* (left-bottom)
            lx = axis_x[i] * 1.55
            ly = axis_y[i] * 0.75
            ax.text(lx, ly, labels[i], fontsize=10, ha="center", va="center")
        elif i == 0:  # Double-free (top)
            lx = axis_x[i] * 1.35
            ly = axis_y[i] * 1.1
            ax.text(lx, ly, labels[i], fontsize=10, ha="center", va="center")
        elif i == 2:  # Linear OOBA (right-top)
            lx = axis_x[i] * 1.35
            ly = axis_y[i] * 1.15
            ax.text(lx, ly, labels[i], fontsize=10, ha="center", va="center")
        elif i == 3:  # Type Confusion OOBA (upper-left)
            lx = axis_x[i] * 1.7
            ly = axis_y[i] * 1
            ax.text(lx, ly, 'Type Confusion\n OOBA', fontsize=10, ha="center", va="center")
        else:
            ax.text(lx, ly, labels[i], fontsize=10, ha="center", va="center")

    return grid_levels


def draw_data_radar(title, data_dict, color_map, output_path):
    """绘制带数据的雷达图（图 1 / 图 2）。"""
    axis_x, axis_y, axis_x_closed, axis_y_closed = get_pentagon_coords()

    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    draw_radar_grid(ax, axis_x, axis_y, axis_x_closed, axis_y_closed, DIMENSION_LABELS)

    # 绘制数据折线与填充
    for name, values in data_dict.items():
        color = color_map[name]
        px = [values[i] * axis_x[i] for i in range(5)]
        py = [values[i] * axis_y[i] for i in range(5)]
        px_closed = px + [px[0]]
        py_closed = py + [py[0]]

        ax.plot(px_closed, py_closed, color=color, linewidth=1, label=name)
        alpha = 0.15 if name == "Baseline" else 0.10
        ax.fill(px_closed, py_closed, color=color, alpha=alpha)

    # 底部标题
    fig.subplots_adjust(bottom=0.12)
    fig.text(0.5, 0.02, title, ha="center", fontsize=14)

    legend_elements = [
        Line2D([0], [0], color=COLORS["orange"], lw=1.5, label="ASan"),
        Line2D([0], [0], color=COLORS["blue"], lw=1.5, label="ASan with CRISP"),
        Line2D([0], [0], color=COLORS["gray"], lw=1.5, label="Baseline"),
    ]
    ax.legend(
        handles=legend_elements,
        ncol=1,
        loc="lower left",
        bbox_to_anchor=(0.2, 0.33),
        fontsize='xx-small',
        frameon=False,
    )
    ax.set_aspect("equal")
    ax.set_xlim(-1.05, 1.05)
    ax.set_ylim(-1.05, 1.05)
    ax.axis("off")
    base = output_path.rsplit(".", 1)[0]
    for ext in ["png",  "pdf"]:
        path = f"{base}.{ext}"
        fig.savefig(path, dpi=500, bbox_inches="tight")
        print(f"Saved {path}")

def main():
    parser = argparse.ArgumentParser(description="Draw three radar charts.")
    parser.add_argument("--prefix", default="radar", help="Output file prefix")
    args = parser.parse_args()

    color_map = {
        "ASan": COLORS["orange"],
        "CRISP": COLORS["blue"],
        "Baseline": COLORS["gray"],
    }

    # 示例数据（5 维对应 DIMENSION_LABELS）
    

    draw_data_radar("", DATA, color_map, f"{args.prefix}.png")


if __name__ == "__main__":
    main()
