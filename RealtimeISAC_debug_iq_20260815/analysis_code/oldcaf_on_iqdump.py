#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""把**老 CAF 算法**（analyze_caf 风格：连续流 conj(ch0)·ch1 -> 抗混叠降采样 ->
0.2s Blackman 滑动 STFT，**不做 TDD 门控 / 不做逐帧归一化 / 不做时域去 DC**）
跑在一个 rt_iqdump.py 存下的原始 IQ npz 上（比如 debug/iq_20260815_175024.npz，
0.5s @ 10MHz，locked）。

目的：去掉 rt_dsp 的逐 step 去 DC 之后 50Hz 那条梳没了，但 **200Hz 还在** ——
用老 CAF（本来就没有那段去 DC）跑一遍看 200Hz 是不是也在。在 → 200Hz 是真实的
（TX 5ms 超帧），跟 50Hz 那条 DC 伪影是两回事，该单独 notch 掉。

    python oldcaf_on_iqdump.py RealtimeISAC/debug/iq_20260815_175024.npz
    python oldcaf_on_iqdump.py <npz> --step 0.02 --tw 0.2 --fmax 500
"""

import argparse
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import spectrogram, windows

import analyze_caf as ac


def comb_excess(P, freqs, targets=(50, 100, 150, 200, 250, 300, 350, 400)):
    from numpy.lib.stride_tricks import sliding_window_view
    Pdb = 10 * np.log10(P + 1e-20)
    k = 31
    base = np.median(sliding_window_view(np.pad(Pdb, k // 2, mode="edge"), k), axis=-1)
    ex = Pdb - base
    out = []
    for f0 in targets:
        best = -9.0
        for s in (1, -1):
            i = int(np.argmin(np.abs(freqs - s * f0)))
            j = slice(max(0, i - 1), i + 2)
            best = max(best, float(ex[j].max()))
        out.append((f0, best))
    return out, ex


def pdp_delay(ch0, ch1, fs, kmax=6, win_sec=0.05):
    n = min(int(fs * win_sec), len(ch0))
    a, b = ch0[:n], ch1[:n]
    best_d, best_v = 0, -1.0
    for d in range(-kmax, kmax + 1):
        if d >= 0:
            v = abs(np.sum(np.conj(a[:n - d]) * b[d:n]))
        else:
            v = abs(np.sum(np.conj(a[-d:n]) * b[:n + d]))
        if v > best_v:
            best_v, best_d = v, d
    return best_d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("npz", help="rt_iqdump.py 存的 iq_*.npz")
    ap.add_argument("--step", type=float, default=0.02, help="STFT 步进秒 (默认 0.02)")
    ap.add_argument("--tw", type=float, default=0.2, help="STFT 窗秒 (默认 0.2)")
    ap.add_argument("--fmax", type=float, default=500.0, help="多普勒范围 ±fmax Hz")
    ap.add_argument("--decim", type=int, default=16, help="老 fast_caf 风格的跳采样倍数 (默认 16)")
    ap.add_argument("--out", type=str, default=None)
    a = ap.parse_args()

    d = np.load(a.npz)
    iq = d["iq"]                       # (2, N*2) int16 交织
    fs = float(d["fs"])
    ch0 = (iq[0, 0::2].astype(np.float64) + 1j * iq[0, 1::2].astype(np.float64)) / 32767.0
    ch1 = (iq[1, 0::2].astype(np.float64) + 1j * iq[1, 1::2].astype(np.float64)) / 32767.0
    n = min(len(ch0), len(ch1))
    ch0, ch1 = ch0[:n], ch1[:n]
    dur = n / fs
    print(f"=== 老 CAF on {os.path.basename(a.npz)} ===")
    print(f"{n} 样点/通道  fs={fs/1e6:.0f}MHz  时长 {dur*1e3:.0f}ms  "
          f"locked={bool(d['locked'])} contrast={float(d['contrast']):.3f}  "
          f"phase={int(d['phase'])} n_int={int(d['n_int'])} n_period={int(d['n_period'])}")

    # ch0/ch1 相对延迟对齐（老 fast_caf 会先跑 PDP）
    dly = pdp_delay(ch0, ch1, fs)
    if dly >= 0:
        c0, c1 = ch0[:n - dly], ch1[dly:n]
    else:
        c0, c1 = ch0[-dly:n], ch1[:n + dly]
    print(f"PDP 对齐延迟 = {dly} 样点")

    # ---------- 变体 A：老 fast_caf_spectrogram 风格（跳采样 decim + spectrogram） ----------
    dec = a.decim
    c0d, c1d = c0[::dec], c1[::dec]
    m = min(len(c0d), len(c1d))
    prodA = np.conj(c0d[:m]) * c1d[:m]           # 连续流，无门控 / 无归一化 / 无去 DC
    fsA = fs / dec
    npsA = int(fsA * a.tw)
    novA = int(fsA * (a.tw - a.step))
    fA, tA, SA = spectrogram(prodA, fs=fsA, window=windows.blackman(npsA),
                             nperseg=npsA, noverlap=novA, return_onesided=False, mode="complex",
                             detrend=False)
    fA = np.fft.fftshift(fA); SA = np.fft.fftshift(SA, axes=0)
    mA = (fA >= -a.fmax) & (fA <= a.fmax)
    fA, SA = fA[mA], SA[mA]
    PA = (np.abs(SA) ** 2).mean(axis=1)
    rowsA, exA = comb_excess(PA, fA)

    # ---------- 变体 B：compute_caf_matrix 风格（全速率乘积 -> 链式抗混叠 decimate） ----------
    q = max(1, int(round(fs / 8000.0)))
    stages = ac._decimate_chain_factors(q)
    prodB_full = np.conj(c0) * c1
    prodB = ac.chained_decimate(prodB_full, stages)
    fsB = fs / q
    npsB = int(fsB * a.tw)
    novB = int(fsB * (a.tw - a.step))
    fB, tB, SB = spectrogram(prodB, fs=fsB, window=windows.blackman(npsB),
                             nperseg=npsB, noverlap=novB, return_onesided=False, mode="complex",
                             detrend=False)
    fB = np.fft.fftshift(fB); SB = np.fft.fftshift(SB, axes=0)
    mB = (fB >= -a.fmax) & (fB <= a.fmax)
    fB, SB = fB[mB], SB[mB]
    PB = (np.abs(SB) ** 2).mean(axis=1)
    rowsB, exB = comb_excess(PB, fB)

    print(f"\nSTFT: step={a.step}s window={a.tw}s  ->  "
          f"A: {SA.shape[1]} 帧 @ {fsA:.0f}Hz(decim{dec})   "
          f"B: {SB.shape[1]} 帧 @ {fsB:.0f}Hz(链={stages})")
    print(f"\n{'目标Hz':>7} | {'A 跳采样+spectrogram':>22} | {'B 抗混叠decimate':>20}")
    print("-" * 56)
    for (f0, va), (_, vb) in zip(rowsA, rowsB):
        print(f"{f0:>7} | {va:>+21.1f} | {vb:>+19.1f}")
    print("\n(高出 31 点滑动中值本底 dB；老 CAF 无去 DC / 无归一化 / 无门控)")

    # ---------- 出图 ----------
    fig, axes = plt.subplots(2, 2, figsize=(15, 9))
    for ax, (t, f, S, ttl) in zip(
            [axes[0, 0], axes[0, 1]],
            [(tA, fA, SA, f"A: old fast_caf style (stride-decimate x{dec} + spectrogram)"),
             (tB, fB, SB, f"B: compute_caf_matrix style (full-rate product + anti-alias decimate {stages})")]):
        Sdb = 10 * np.log10(np.abs(S) ** 2 + 1e-20)
        vmin, vmax = np.percentile(Sdb, 5), np.percentile(Sdb, 98)
        ax.pcolormesh(t, f, Sdb, shading="auto", cmap="jet", vmin=vmin, vmax=vmax)
        for h in range(-int(a.fmax // 50) * 50, int(a.fmax) + 1, 50):
            ax.axhline(h, color="w", lw=0.3, alpha=0.25)
        ax.set_title(ttl, fontsize=10)
        ax.set_xlabel("Time (s)"); ax.set_ylabel("Doppler (Hz)")
    for ax, (f, ex, rows, ttl) in zip(
            [axes[1, 0], axes[1, 1]],
            [(fA, exA, rowsA, "A: time-averaged spectrum (rel. local floor)"),
             (fB, exB, rowsB, "B: time-averaged spectrum (rel. local floor)")]):
        ax.plot(f, ex, lw=0.8)
        for h in range(-int(a.fmax // 50) * 50, int(a.fmax) + 1, 50):
            ax.axvline(h, color="tab:red", lw=0.5, alpha=0.3)
        ax.axhline(8, color="k", ls=":", lw=0.8)
        ax.set_xlim(-a.fmax, a.fmax); ax.set_ylim(-3, max(16, max(v for _, v in rows) + 3))
        ax.set_xlabel("Doppler (Hz)"); ax.set_ylabel("dB above floor"); ax.set_title(ttl)
        ax.grid(alpha=0.3)
        for f0, v in rows:
            if v > 3:
                ax.annotate(f"{f0}:{v:+.0f}", (f0, v), fontsize=7, ha="center", color="tab:red")
    fig.suptitle(f"OLD CAF (no DC removal) on {os.path.basename(a.npz)}  "
                 f"step={a.step}s window={a.tw}s  full {dur*1e3:.0f}ms", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out = a.out or os.path.join(os.path.dirname(os.path.abspath(a.npz)),
                                f"oldcaf_{os.path.splitext(os.path.basename(a.npz))[0]}.png")
    fig.savefig(out, dpi=120, bbox_inches="tight")
    print(f"\n🖼  {out}")


if __name__ == "__main__":
    main()
