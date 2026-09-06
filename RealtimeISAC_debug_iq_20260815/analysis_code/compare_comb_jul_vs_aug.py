#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""算法 vs 数据：把**同一组算法**跑在 7月10日 的采集 和 8月15日 的采集上，
看 ±50 / ±100 / ±200 / ±300 Hz 那几条多普勒线到底是"算法造的"还是"数据里就有的"。

数据（格式不同，脚本统一转成 4×int16 交织的临时 .bin 再喂同一套代码）：
  - 7月10日: experiment_30MHz_static_20260710_*/2ch_iq_data.bin  (interleaved_sc16, 30MHz)
  - 8月15日: RealtimeISAC/debug/iq_2026081 5_*.npz               (rt_iqdump 存的 (2,N*2) int16, 10MHz)
    （不是室外那份 fc32；fc32 在 RealtimeISAC/reference/，本脚本没碰）

对每份数据跑 3 个算法，全用 step=0.02s / window=0.2s：
  1. oldCAF      : analyze_caf 风格 —— 连续流 conj(ch0)·ch1 -> 链式抗混叠 decimate ->
                   0.2s Blackman 滑动 STFT。**无 TDD 门控 / 无逐帧归一化 / 无去 DC**。
  2. RT-noDC     : RealtimeISAC 链路，去掉逐 step 去 DC（= 目前 rt_dsp 改完的样子）。
                   TDD 门控 + 每帧一个 H + /‖ch0‖‖ch1‖ 归一化 + 0.2s ring FFT。
  3. RT-perstepDC: 同上，但保留原来那段"每 step 更新一次的 EMA 去 DC"（改之前的 rt_dsp）。

判读：
  - 某条线在 RT-perstepDC 有、RT-noDC 没有、oldCAF 也没有  -> 算法伪影（逐 step 去 DC）
  - 某条线在 oldCAF / RT-noDC 都有                          -> 数据里就有（TX 调制）
  - 某条线只在 8月数据出现、7月数据没有                     -> 那次实验的 TX 行为，不是算法

    python compare_comb_jul_vs_aug.py                       # 默认 155243 vs iq_20260815_175024
    python compare_comb_jul_vs_aug.py --jul <exp文件夹> --aug <iq_npz> --sec 0.5
"""

import argparse
import os
import sys
import tempfile

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import spectrogram, windows

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

HERE = os.path.dirname(os.path.abspath(__file__))
RT_DEBUG = os.path.join(HERE, "RealtimeISAC", "debug")
sys.path.insert(0, os.path.join(HERE, "RealtimeISAC", "realtime"))
sys.path.insert(0, RT_DEBUG)

import analyze_caf as ac                       # noqa: E402
from rt_config import RtConfig                 # noqa: E402
from ablate_doppler import run_chain           # noqa: E402

TARGETS = (50, 100, 150, 200, 250, 300, 350, 400)
SCRATCH = os.environ.get("TEMP", tempfile.gettempdir())


def comb_excess(P, freqs):
    from numpy.lib.stride_tricks import sliding_window_view
    Pdb = 10 * np.log10(P + 1e-20)
    k = 31
    base = np.median(sliding_window_view(np.pad(Pdb, k // 2, mode="edge"), k), axis=-1)
    ex = Pdb - base
    out = []
    for f0 in TARGETS:
        best = -9.0
        for s in (1, -1):
            i = int(np.argmin(np.abs(freqs - s * f0)))
            j = slice(max(0, i - 1), i + 2)
            best = max(best, float(ex[j].max()))
        out.append(best)
    return out, ex


def write_4int16_bin(ch0_re, ch0_im, ch1_re, ch1_im, path):
    """4×int16 交织 = experiment 的 2ch_iq_data.bin 格式。"""
    n = len(ch0_re)
    out = np.empty(n * 4, dtype=np.int16)
    out[0::4] = ch0_re
    out[1::4] = ch0_im
    out[2::4] = ch1_re
    out[3::4] = ch1_im
    out.tofile(path)
    return n


def prep_july(exp_folder, sec, scratch):
    import json
    params = json.load(open(os.path.join(exp_folder, "acquisition_parameters.json")))
    fs = float(params["sample_rate_hz"])
    SC16 = np.dtype([("re", np.int16), ("im", np.int16)])
    mm = np.memmap(os.path.join(exp_folder, "2ch_iq_data.bin"), dtype=SC16, mode="r")
    N = min(int(fs * sec), mm.shape[0] // 2)
    ch0 = mm[0:2 * N:2]
    ch1 = mm[1:2 * N:2]
    path = os.path.join(scratch, f"_cmp_july_{os.path.basename(exp_folder)}.bin")
    write_4int16_bin(np.ascontiguousarray(ch0["re"]), np.ascontiguousarray(ch0["im"]),
                     np.ascontiguousarray(ch1["re"]), np.ascontiguousarray(ch1["im"]), path)
    return path, fs, N, os.path.basename(exp_folder)


def prep_aug(iq_npz, sec, scratch):
    d = np.load(iq_npz)
    iq = d["iq"]
    fs = float(d["fs"])
    N = min(int(fs * sec), iq.shape[1] // 2)
    path = os.path.join(scratch, f"_cmp_aug_{os.path.splitext(os.path.basename(iq_npz))[0]}.bin")
    write_4int16_bin(iq[0, 0:2 * N:2], iq[0, 1:2 * N:2],
                     iq[1, 0:2 * N:2], iq[1, 1:2 * N:2], path)
    return path, fs, N, os.path.splitext(os.path.basename(iq_npz))[0]


def oldcaf_spectrum(bin_path, fs, N, step, tw, fmax):
    """analyze_caf 风格连续 CAF：全速率 conj(ch0)·ch1 -> 链式抗混叠 decimate -> STFT。"""
    raw = np.memmap(bin_path, dtype=np.int16, mode="r").reshape(N, 4)
    c0 = (raw[:, 0].astype(np.float64) + 1j * raw[:, 1]) / 32767.0
    c1 = (raw[:, 2].astype(np.float64) + 1j * raw[:, 3]) / 32767.0
    prod = np.conj(c0) * c1
    q = max(1, int(round(fs / 8000.0)))
    stages = ac._decimate_chain_factors(q)
    prod_d = ac.chained_decimate(prod, stages)
    fsd = fs / q
    nps = int(fsd * tw)
    nov = int(fsd * (tw - step))
    f, t, S = spectrogram(prod_d, fs=fsd, window=windows.blackman(nps), nperseg=nps,
                          noverlap=nov, return_onesided=False, mode="complex", detrend=False)
    f = np.fft.fftshift(f)
    S = np.fft.fftshift(S, axes=0)
    m = (f >= -fmax) & (f <= fmax)
    f = f[m]
    P = (np.abs(S[m]) ** 2).mean(axis=1)
    return f, P, S.shape[1]


def rt_spectrum(bin_path, fs, N, step, tw, fmax, dc):
    """RealtimeISAC 链路(ablate_doppler.run_chain)：dc='none' 或 'ema'(逐step)。"""
    cfg = RtConfig(sample_rate=fs, step_sec=step, window_sec=tw)
    n_steps = max(1, N // cfg.n_step)
    spec, freqs, sync = run_chain(bin_path, cfg, n_steps, norm="coeff", gate="on",
                                  dc=dc, win="blackman")
    rows = np.any(spec != 0, axis=1)
    S = spec[rows] if rows.any() else spec
    m = (freqs >= -fmax) & (freqs <= fmax)
    P = (np.abs(S[:, m]) ** 2).mean(axis=0)
    return freqs[m], P, int(rows.sum()), sync


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jul", default="experiment_30MHz_static_20260710_155243")
    ap.add_argument("--aug", default=os.path.join(RT_DEBUG, "iq_20260815_175024.npz"))
    ap.add_argument("--sec", type=float, default=0.5, help="两边各取多少秒 (默认 0.5 = 8月dump的全长)")
    ap.add_argument("--step", type=float, default=0.02)
    ap.add_argument("--tw", type=float, default=0.2)
    ap.add_argument("--fmax", type=float, default=500.0)
    a = ap.parse_args()

    jbin, jfs, jN, jname = prep_july(a.jul, a.sec, SCRATCH)
    abin, afs, aN, aname = prep_aug(a.aug, a.sec, SCRATCH)
    print(f"7月: {jname}  fs={jfs/1e6:.0f}MHz  取 {jN/jfs*1e3:.0f}ms")
    print(f"8月: {aname}  fs={afs/1e6:.0f}MHz  取 {aN/afs*1e3:.0f}ms")
    print(f"参数: step={a.step}s  window={a.tw}s  多普勒 ±{a.fmax:.0f}Hz\n")

    results = {}  # (dataset, algo) -> (freqs, P, excess_list)
    panels = []
    for dsname, binp, fs, Nn in ((f"Jul10  {jname[-6:]}", jbin, jfs, jN),
                                 (f"Aug15  {aname[-6:]}", abin, afs, aN)):
        for algo in ("oldCAF", "RT-noDC", "RT-perstepDC"):
            if algo == "oldCAF":
                f, P, nfr = oldcaf_spectrum(binp, fs, Nn, a.step, a.tw, a.fmax)
                info = f"{nfr} win"
            else:
                dc = "none" if algo == "RT-noDC" else "ema"
                f, P, nfr, sync = rt_spectrum(binp, fs, Nn, a.step, a.tw, a.fmax, dc)
                info = f"{nfr} win, TDD {'lock' if sync.locked else 'unlock'}"
            ex_list, ex_curve = comb_excess(P, f)
            results[(dsname, algo)] = (f, P, ex_list)
            panels.append((dsname, algo, f, ex_curve, ex_list, info))

    # ---- 表 ----
    hdr = f"{'数据':<20}{'算法':<14}" + "".join(f"{t:>6}" for t in TARGETS)
    print(hdr)
    print("-" * len(hdr))
    for (ds, algo), (f, P, ex) in results.items():
        print(f"{ds:<20}{algo:<14}" + "".join(f"{v:>+6.1f}" for v in ex))
    print("\n(数字 = 相对 31 点滑动中值本底 dB。>+5 = 有那条线)")

    # ---- 图：6 张时间平均谱 ----
    fig, axes = plt.subplots(2, 3, figsize=(17, 8), sharex=True)
    for ax, (ds, algo, f, ex, exl, info) in zip(axes.flat, panels):
        ax.plot(f, ex, lw=0.8)
        for h in range(-int(a.fmax // 50) * 50, int(a.fmax) + 1, 50):
            ax.axvline(h, color="tab:red", lw=0.5, alpha=0.3)
        ax.axhline(5, color="k", ls=":", lw=0.8)
        ax.set_xlim(-a.fmax, a.fmax)
        ax.set_ylim(-3, max(16, max(exl) + 3))
        ax.set_title(f"{ds}\n{algo}  ({info})", fontsize=9)
        ax.grid(alpha=0.3)
        for f0, v in zip(TARGETS, exl):
            if v > 4:
                ax.annotate(f"{f0}:{v:+.0f}", (f0, v), fontsize=7, ha="center", color="tab:red")
    for ax in axes[-1]:
        ax.set_xlabel("Doppler (Hz)")
    for ax in axes[:, 0]:
        ax.set_ylabel("dB above local floor")
    fig.suptitle(f"Comb: algorithm vs data  (July10 30MHz  vs  Aug15 10MHz,  {a.sec*1e3:.0f}ms each, "
                 f"step={a.step}s win={a.tw}s)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out = os.path.join(RT_DEBUG, "compare_comb_jul_vs_aug.png")
    fig.savefig(out, dpi=120, bbox_inches="tight")
    print(f"\n🖼  {out}")

    for p in (jbin, abin):
        try:
            os.remove(p)
        except OSError:
            pass


if __name__ == "__main__":
    main()
