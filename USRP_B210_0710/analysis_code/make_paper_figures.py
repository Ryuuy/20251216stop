#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
论文用图：复用 analyze_caf.py 里的缓存 CAF matrix，只挑指定 range 的几张
micro-Doppler 帧图，不走 GIF 拼接。不改 analyze_caf.py 本身，只是调用它的函数。
去掉顶部 title，放大坐标数字/轴标签/colorbar/Range 标注字号。
"""

import os
import numpy as np
import matplotlib.pyplot as plt

import analyze_caf as ac

OUT_DIR = "paper_figures"

# (experiment 文件夹名, [目标 range(m), ...])
TARGETS = {
    "experiment_30MHz_static_20260710_155243": [21.25, 13.75],
    "experiment_30MHz_static_20260710_161256": [12.50, 71.25],
}

# 字号（比原脚本大很多，方便放进 paper）
FS_AXIS_LABEL = 26
FS_TICK = 22
FS_CBAR_LABEL = 24
FS_CBAR_TICK = 20
FS_RANGE_TEXT = 24


def make_frame(data, delay_value, base_name, step_m, vmin, vmax, out_dir):
    Sxx, delay_axis, t_axis, freq_axis = data["Sxx"], data["delay_axis"], data["t_axis"], data["freq_axis"]
    di = int(np.argmin(np.abs(delay_axis - delay_value)))
    r_m = delay_axis[di] * step_m
    Sxx_db = 10 * np.log10(np.abs(Sxx[di]).T + 1e-10)

    fig, ax = plt.subplots(figsize=(12, 6))
    mesh = ax.pcolormesh(t_axis, freq_axis, Sxx_db, shading="auto", cmap="jet", vmin=vmin, vmax=vmax)
    ax.set_ylabel("Doppler Frequency (Hz)", fontsize=FS_AXIS_LABEL)
    ax.set_xlabel("Time (s)", fontsize=FS_AXIS_LABEL)
    ax.tick_params(axis="both", labelsize=FS_TICK)
    ax.text(0.98, 0.98, f"Range: ~{r_m:.2f} m", transform=ax.transAxes, fontsize=FS_RANGE_TEXT, fontweight="bold",
            verticalalignment="top", horizontalalignment="right",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="yellow", alpha=0.9))
    cbar = plt.colorbar(mesh, ax=ax)
    cbar.set_label("Relative Power (dB)", fontsize=FS_CBAR_LABEL)
    cbar.ax.tick_params(labelsize=FS_CBAR_TICK)
    plt.tight_layout()

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"caf_matrix_delay{delay_axis[di]:.2f}_{base_name}.png")
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"已保存: {out_path} (delay={delay_axis[di]:.2f} samples, range~{r_m:.2f} m)")
    return out_path


def main():
    for exp_folder, ranges in TARGETS.items():
        matrix_path = ac.get_caf_matrix_path(exp_folder)
        if not os.path.exists(matrix_path):
            print(f"找不到缓存 matrix: {matrix_path}，跳过 {exp_folder}")
            continue
        data = ac.load_caf_matrix(matrix_path)
        fs = data["meta"]["sample_rate_hz"]
        _, step_m = ac.range_step_for_sample_rate(fs)

        # 固定色标范围 -32~-10 dB（论文里两张图统一色标，便于对比）
        vmin, vmax = -32, -10

        base_name = exp_folder
        for r in ranges:
            delay_value = r / step_m
            make_frame(data, delay_value, base_name, step_m, vmin, vmax, OUT_DIR)


if __name__ == "__main__":
    main()
