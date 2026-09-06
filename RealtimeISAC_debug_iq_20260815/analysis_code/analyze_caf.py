#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CAF 分析脚本 (USRP B210 双通道数据) —— 从 validate_capture_and_caf.py 拆出，只保留分析部分。
数据采集已完成，本脚本只做「找最新/指定 experiment 文件夹 -> 验证 -> 出图/GIF」，不再采数。
- Ref=ch0(Tx), Sensing=ch1。两通道 |IQ|>1.4 检查；Tw=0.4s, step=0.02s。
- PDP: delay -3~+3，选最强的 delay 做 CAF。相位: spectrogram mode='complex'，去旋转 exp(-j*2*pi*f*t)，折叠 0 单独、+f×(-f)。
"""

import os
import glob
import re
import json
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import spectrogram, windows, resample_poly, decimate
from scipy.fft import fft as sp_fft

# 内存优化：memmap 打开后，饱和检查(check_iq_magnitude_abort_memmap)分块扫描原始 sc16 整数，
# PDP(compute_pdp_delay_memmap)只读 0.1s 小窗口，CAF(build_decimated_channel)先按 decim=16
# 跳采样切片再转 complex64——全程不把整段采集数据以 complex64/128 形式摊在内存里，
# 10s 数据的峰值内存从原来的 ~4GB 降到几百 MB。

# sc16 存盘格式：每个复数样点 = 2 个 int16 (re, im)，4 字节。定标系数须与采集端
# (USRPSaveData2Channel.py 的 SC16_SCALE) 一致，否则幅度/能量算出来是错的。
SC16_DTYPE = np.dtype([('re', np.int16), ('im', np.int16)])
SC16_SCALE = 32767.0

MAGNITUDE_ABORT_THRESHOLD = 1.4
MAGNITUDE_ABORT_COUNT = 100
TW_SEC = 0.2
STEP_SEC = 0.04       # 时间步长，改这一个值就行，下面 margin/降采样链/warmup 都会跟着自动重算
DECIM = 16
PDP_WINDOW_SEC = 0.1
PDP_DELAY_NEG = 4   # PDP 范围 -3..+3，选最强的 delay
PDP_DELAY_POS = 5
# GIF 用：delay 从 -3 扫到 +3，每帧 0.5s，距离标注
DELAY_MIN_GIF = 12
DELAY_MAX_GIF = 18
GIF_FRAME_DURATION_SEC = 0.5
# 与采集时用的 ACQUISITION_DURATION_SEC 对应，手动改这里以匹配实际采集时长。
ACQUISITION_DURATION_SEC = 20.0

C_LIGHT = 3e8  # m/s
# max_duration_sec=None 时默认只读 60 秒，避免整盘加载崩溃
DEFAULT_MAX_DURATION_SEC = 30.0

# ---- 全采样率 CAF matrix 参数（不做 DECIM=16 跳采样，相关/乘积用全带宽数据，
# delay 分辨率不受影响；多普勒方向用"流式链式 decimate"把相关乘积滤波+降采样到一个刚好够
# 覆盖 freq_range 的低速率再做 FFT——比直接对全采样率整窗做 FFT 快得多，也比"整窗重新降采样"快
# 得多(只处理每个 step 新增的数据+一段边界余量，不重新处理整个 Tw 窗口)。
# 下面几个参数改了之后，margin/降采样链/warmup 长度等都是按公式自动重算的，不需要改函数内部逻辑）----
MAX_DELAY_RAW = 18            # delay 范围: 0..18 个原始(未降采样)采样点，单位跟旧 GIF 的 delay_bin 一致
DELAY_INTERP_FACTOR = 4       # ch1 插值倍数：delay 网格间隔变成 1/interp_factor 个原始采样点，
                               # 只是让 CAF 沿 delay 方向的曲线更平滑，不代表真实分辨率变细(仍由带宽决定)。
                               # 以后想改成 2 倍插值，直接改这个常数就行。
FILTER_EDGE_MARGIN_RAW = 200  # ch1 分数延迟插值(resample_poly)用的边界余量(原始采样点数)，
                               # 跟每个 hop 的 step_samples 比起来很小，对耗时影响可忽略
DOPPLER_TARGET_FS_HZ = 8000.0  # 多普勒方向流式降采样的目标速率：要求远大于 2*600=1200Hz，
                               # 留够抗混叠滤波器的过渡带余量；同时要求 sample_rate/此值 能被拆成
                               # 若干个 <=10 的链式小阶段(见 _decimate_chain_factors)，
                               # 20/25/30MHz 采样率下实测都能拆得比较干净
DOPPLER_DECIMATE_MAX_STAGE = 10   # 链式 decimate 每一级最大降采样因子
DOPPLER_MARGIN_SAFETY_FACTOR = 2.0  # 边界余量相对理论估算值的安全倍数(用合成信号验证过，
                                     # 2 倍能让流式结果和"整段一次性算"完全一致，见开发记录)
CAF_MATRIX_OUT_DIR = "validate_caf_output"


def range_step_for_sample_rate(sample_rate_hz):
    """delay(单位: 未降采样的原始 sample)每差 1 对应的距离 = c / (2*sample_rate)（往返）。
    之前是硬编码 RANGE_FIRST_BIN_MAX_M=2.5 / RANGE_STEP_M=5.0（正好是 30MHz 时的值），
    换成 25MHz 采样率而没改这两个数，GIF 上标的距离会是错的，所以改成跟着实际采样率算。"""
    step_m = C_LIGHT / (2.0 * sample_rate_hz)
    first_bin_max_m = step_m / 2.0
    return first_bin_max_m, step_m


def read_2ch_iq_memmap(filename: str):
    """memmap 读双通道 IQ (sc16 存盘：每复数样点 4 字节 int16 I/Q)，不整盘加载。"""
    try:
        file_size = os.path.getsize(filename)
        total_complex = file_size // 4
        total_per_ch = total_complex // 2
        data_memmap = np.memmap(filename, dtype=SC16_DTYPE, mode='r', shape=(total_complex,))
        ch0_memmap = data_memmap[0::2]
        ch1_memmap = data_memmap[1::2]
        return ch0_memmap, ch1_memmap, total_per_ch
    except Exception as e:
        print(f"Error creating memory map for {filename}: {e}")
        return None, None, 0


def sc16_to_complex64(sc16_arr):
    """sc16 structured array (re,im int16) -> complex64，按 UHD 定标 (/32767) 还原，
    使幅度/能量与旧 fc32 存盘等价（满量程仍是 ±1.0），下游 PDP/CAF/能量检查逻辑不用改。
    注意：不能写成 re.astype(f32) + 1j*im.astype(f32) —— python 的 1j 是 complex128，
    这个乘法会把整个数组隐式升级成 complex128（16 字节/点，是 complex64 的 2 倍），
    10 秒数据在这一步就会多出好几个 G 的临时数组。改成直接对 complex64 的
    real/imag 视图赋值，全程只有 complex64 大小的分配。"""
    out = np.empty(sc16_arr.shape, dtype=np.complex64)
    out.real = sc16_arr['re']
    out.imag = sc16_arr['im']
    out /= SC16_SCALE
    return out


def check_data_format(params, filename):
    """防止误把旧的 fc32 (interleaved_complex64) 文件当 sc16 解析——那样幅度/能量会完全算错。"""
    data_format = params.get("data_format") if params else None
    if data_format is None:
        print(f"警告: {filename} 缺少 acquisition_parameters.json 中的 data_format 字段，"
              f"无法确认是否为 sc16 格式，按 sc16 继续解析（旧数据请用旧脚本处理）。")
        return True
    if data_format != "interleaved_sc16":
        print(f"错误: {filename} 的 data_format={data_format!r}，不是本脚本预期的 interleaved_sc16。"
              f"这大概率是旧的 fc32 数据，用 sc16 方式解析会得到完全错误的幅度/能量，已中止。")
        return False
    return True


def find_latest_experiment_folder():
    experiment_folders = glob.glob("experiment_*")
    if not experiment_folders:
        return None
    pattern = r'experiment_.*_(\d{8})_(\d{6})'
    folders_with_timestamp = []
    for folder in experiment_folders:
        match = re.search(pattern, folder)
        if match:
            date_str, time_str = match.group(1), match.group(2)
            try:
                ts = datetime.strptime(f"{date_str}_{time_str}", "%Y%m%d_%H%M%S")
                folders_with_timestamp.append((ts, folder))
            except ValueError:
                folders_with_timestamp.append((datetime.fromtimestamp(os.path.getctime(folder)), folder))
        else:
            folders_with_timestamp.append((datetime.fromtimestamp(os.path.getctime(folder)), folder))
    return max(folders_with_timestamp, key=lambda x: x[0])[1]


def load_acquisition_parameters(folder_path):
    path = os.path.join(folder_path, "acquisition_parameters.json")
    if not os.path.exists(path):
        return None
    with open(path, "r") as f:
        return json.load(f)


def compute_pdp_delay_memmap(ch0_memmap, ch1_memmap, n_total, sample_rate_hz,
                              delay_neg=PDP_DELAY_NEG, delay_pos=PDP_DELAY_POS, window_sec=PDP_WINDOW_SEC):
    """PDP 只需要开头 window_sec(默认 0.1s) 的一小段原始采样率数据，直接从 memmap 切这一小段
    再转 complex64，不需要（也不应该）把整段采集数据转出来才能算 PDP。"""
    K_neg, K_pos = delay_neg, delay_pos
    delay_axis = np.arange(-K_neg, K_pos + 1, dtype=np.int32)
    n_pdp = min(int(sample_rate_hz * window_sec), n_total)
    n_need = max(K_neg, K_pos) + 1
    if n_pdp <= n_need:
        return 0, np.zeros(len(delay_axis), dtype=np.float64), delay_axis
    ch0_seg = sc16_to_complex64(ch0_memmap[:n_pdp])
    ch1_seg = sc16_to_complex64(ch1_memmap[:n_pdp])
    pdp_amp = np.zeros(len(delay_axis), dtype=np.float64)
    for i, d in enumerate(delay_axis):
        if d >= 0:
            pdp_amp[i] = np.abs(np.sum(np.conj(ch0_seg[: n_pdp - d]) * ch1_seg[d:n_pdp]))
        else:
            pdp_amp[i] = np.abs(np.sum(np.conj(ch0_seg[-d:n_pdp]) * ch1_seg[: n_pdp + d]))
    delay_max = int(delay_axis[np.argmax(pdp_amp)])
    return delay_max, pdp_amp, delay_axis


MAGNITUDE_CHECK_CHUNK = 5_000_000  # 每块样点数，分块扫描原始 sc16，避免整段转 complex64


def check_iq_magnitude_abort_memmap(ch0_memmap, ch1_memmap, n_total,
                                     threshold=MAGNITUDE_ABORT_THRESHOLD, max_count=MAGNITUDE_ABORT_COUNT,
                                     chunk_size=MAGNITUDE_CHECK_CHUNK):
    """饱和检查只需要计数超阈值的点数，不需要把整段数据转成 complex64 再求 abs()。
    直接在 int16 的 re/im 上用整数比较 (re^2+im^2 > (threshold*32767)^2)，分块扫描，
    峰值内存只有一个 chunk 大小，而不是整个采集时长。"""
    thr2 = (threshold * SC16_SCALE) ** 2
    n0 = n1 = 0
    for start in range(0, n_total, chunk_size):
        end = min(start + chunk_size, n_total)
        re0 = ch0_memmap['re'][start:end].astype(np.int64)
        im0 = ch0_memmap['im'][start:end].astype(np.int64)
        n0 += int(np.sum(re0 * re0 + im0 * im0 > thr2))
        re1 = ch1_memmap['re'][start:end].astype(np.int64)
        im1 = ch1_memmap['im'][start:end].astype(np.int64)
        n1 += int(np.sum(re1 * re1 + im1 * im1 > thr2))
        if n0 + n1 > max_count:
            break
    return (n0 + n1) <= max_count, n0, n1


def build_decimated_channel(ch_memmap, offset, n_total, decim):
    """先在原始 sc16 memmap 上按 decim 跳采样切片，再转 complex64——
    切片后数组只有 n_total/decim 个点，比先转 complex64 整段再切片小 decim 倍(默认 16x)。
    这是本文件里能把 10s/4GB 降到几百 MB 的关键一步。"""
    sc16_slice = ch_memmap[offset:n_total:decim]
    return sc16_to_complex64(sc16_slice)


def fast_caf_spectrogram(ch0_memmap, ch1_memmap, n_total, sample_rate_hz, Tw=TW_SEC, step=STEP_SEC, decim=DECIM, freq_range=(-600, 600),
                         use_pdp=True, pdp_delay_neg=PDP_DELAY_NEG, pdp_delay_pos=PDP_DELAY_POS, pdp_window_sec=PDP_WINDOW_SEC):
    """ch0_memmap/ch1_memmap 是原始 sc16 memmap（未转 complex、未截断），n_total 是两通道
    共同可用的样点数。PDP 只读一小段窗口；CAF 数据在 build_decimated_channel 里先 decim
    再转 complex64，全程不会把整段采集数据以 complex 形式摊开在内存里。"""
    pdp_amp, pdp_delay_axis = None, None
    if use_pdp:
        delay_max, pdp_amp, pdp_delay_axis = compute_pdp_delay_memmap(ch0_memmap, ch1_memmap, n_total, sample_rate_hz, delay_neg=pdp_delay_neg, delay_pos=pdp_delay_pos, window_sec=pdp_window_sec)
    else:
        delay_max = 0
    if delay_max >= 0:
        ch0_ds = build_decimated_channel(ch0_memmap, 0, n_total, decim)
        ch1_ds = build_decimated_channel(ch1_memmap, delay_max, n_total, decim)
    else:
        ch0_ds = build_decimated_channel(ch0_memmap, -delay_max, n_total, decim)
        ch1_ds = build_decimated_channel(ch1_memmap, 0, n_total, decim)
    n_ds = min(len(ch0_ds), len(ch1_ds))
    ch0_ds, ch1_ds = ch0_ds[:n_ds], ch1_ds[:n_ds]
    effective_fs = sample_rate_hz / decim
    prod_stream = np.conj(ch0_ds) * ch1_ds
    nperseg = int(effective_fs * Tw)
    noverlap = int(effective_fs * (Tw - step))
    win = windows.blackman(nperseg)
    f, t, Sxx = spectrogram(prod_stream, fs=effective_fs, window=win, nperseg=nperseg, noverlap=noverlap, return_onesided=False, mode="complex")
    f = np.fft.fftshift(f)
    Sxx = np.fft.fftshift(Sxx, axes=0)
    Sxx_db = 10 * np.log10(np.abs(Sxx) + 1e-10)
    fmin, fmax = freq_range
    mask = (f >= fmin) & (f <= fmax)
    f_axis = f[mask]
    Sxx_db = Sxx_db[mask, :]
    Sxx_complex = Sxx[mask, :]
    return f_axis, t, Sxx_db, Sxx_complex, effective_fs, delay_max, pdp_amp, pdp_delay_axis


def fast_caf_spectrogram_at_delay(ch0_memmap, ch1_memmap, delay_bin, n_total, sample_rate_hz, Tw=TW_SEC, step=STEP_SEC, decim=DECIM, freq_range=(-600, 600)):
    """固定 delay_bin 的 CAF spectrogram，返回 f_axis, t, Sxx_db。ch0_memmap/ch1_memmap 是原始
    sc16 memmap，这里直接按 delay_bin 偏移 + decim 跳采样切片再转 complex64，不会先把整段
    数据转成 complex64。"""
    if delay_bin >= 0:
        ch0_ds = build_decimated_channel(ch0_memmap, 0, n_total, decim)
        ch1_ds = build_decimated_channel(ch1_memmap, delay_bin, n_total, decim)
    else:
        ch0_ds = build_decimated_channel(ch0_memmap, -delay_bin, n_total, decim)
        ch1_ds = build_decimated_channel(ch1_memmap, 0, n_total, decim)
    n_ds = min(len(ch0_ds), len(ch1_ds))
    ch0_ds, ch1_ds = ch0_ds[:n_ds], ch1_ds[:n_ds]
    effective_fs = sample_rate_hz / decim
    prod_stream = np.conj(ch0_ds) * ch1_ds
    nperseg = int(effective_fs * Tw)
    noverlap = int(effective_fs * (Tw - step))
    win = windows.blackman(nperseg)
    f, t, Sxx = spectrogram(prod_stream, fs=effective_fs, window=win, nperseg=nperseg, noverlap=noverlap, return_onesided=False, mode="complex")
    f = np.fft.fftshift(f)
    Sxx = np.fft.fftshift(Sxx, axes=0)
    Sxx_db = 10 * np.log10(np.abs(Sxx) + 1e-10)
    fmin, fmax = freq_range
    mask = (f >= fmin) & (f <= fmax)
    f_axis = f[mask]
    Sxx_db = Sxx_db[mask, :]
    return f_axis, t, Sxx_db


def range_label_for_delay(delay_bin, first_bin_max_m, step_m):
    """Delay 对应距离 (m)：delay 0 -> 0~half_step, 1 -> half_step~half_step+step, ...
    first_bin_max_m/step_m 请用 range_step_for_sample_rate(sample_rate_hz) 算，不要手填常数。"""
    if delay_bin <= 0:
        return 0.0, first_bin_max_m
    r_min = first_bin_max_m + (delay_bin - 1) * step_m
    r_max = first_bin_max_m + delay_bin * step_m
    return r_min, r_max


def build_delay_axis_raw(max_delay_raw=MAX_DELAY_RAW, interp_factor=DELAY_INTERP_FACTOR):
    """delay 轴，单位=原始(未降采样)采样点，从 0 到 max_delay_raw，按 1/interp_factor 一档。"""
    n_steps = int(round(max_delay_raw * interp_factor))
    return np.arange(n_steps + 1, dtype=np.float64) / interp_factor


def _decimate_chain_factors(q_total, max_stage=DOPPLER_DECIMATE_MAX_STAGE):
    """把总降采样因子 q_total 拆成若干个 <=max_stage 的阶段，链式调用 decimate。
    单级因子太大时 FIR 阶数(近似 20*q+1)会很大、边界瞬态也跟着变长，拆成多级小阶段后
    每级都短很多，链式总耗时明显更低(实测过 q=500/2500/4000 等场景)。"""
    remaining = int(round(q_total))
    stages = []
    for f in range(int(max_stage), 1, -1):
        while remaining % f == 0:
            stages.append(f)
            remaining //= f
    if remaining > 1:
        stages.append(remaining)
    return stages


def _decimate_margin_raw(stages, safety_factor=DOPPLER_MARGIN_SAFETY_FACTOR):
    """估算链式 decimate 需要多少个原始采样点的边界余量，才能让【因果】FIR 滤波器的启动瞬态
    完全落在会被丢弃的余量区间内，不污染保留下来的新数据。scipy.signal.decimate(ftype='fir')
    默认阶数公式约为 20*q+1；按这个估算再乘安全系数。用合成信号(白噪声+已知弱音调)验证过：
    q_total=500 时理论估算 12311，实测最小安全余量在 8000~16000 之间，乘 2 倍安全系数后
    (这里默认 DOPPLER_MARGIN_SAFETY_FACTOR=2.0)跟"整段一次性 decimate"完全对得上(误差为0)。"""
    margin = 0.0
    upstream = 1
    for q in stages:
        taps = 20 * q + 1
        margin += taps * upstream
        upstream *= q
    return int(np.ceil(margin * safety_factor))


def chained_decimate(x, stages):
    """链式降采样：真正的抗混叠滤波(FIR + 因果 lfilter)，不是跳采样。"""
    y = x
    for f in stages:
        if f > 1:
            y = decimate(y, f, ftype='fir', zero_phase=False)
    return y


def compute_caf_matrix(experiment_folder, out_dir=CAF_MATRIX_OUT_DIR, max_duration_sec=None,
                        Tw=TW_SEC, step=STEP_SEC, freq_range=(-600, 600),
                        max_delay_raw=MAX_DELAY_RAW, interp_factor=DELAY_INTERP_FACTOR,
                        doppler_target_fs_hz=DOPPLER_TARGET_FS_HZ, skip_magnitude_check=False):
    """
    流式处理，按 step 逐 hop 推进，不是每个 hop 重新处理整个 Tw 窗口：

    - delay: ch0/ch1 相关(乘积)用全采样率数据，delay 分辨率不受影响；ch1 做 interp_factor 倍插值
      得到分数采样点延迟(0..max_delay_raw，间隔 1/interp_factor)。
    - 多普勒: 相关乘积(prod)按 doppler_target_fs_hz 目标速率做【链式 decimate】(真正的抗混叠滤波，
      不是跳采样) —— 每个 hop 只处理新增的 step_samples 数据 + 一段边界余量(margin，用来让 FIR
      滤波器的启动瞬态落在会丢弃的区间内，不是为了别的)，把降采样结果接到一个小的滑动缓冲区
      (长度 nperseg_dec)里，缓冲区满了就对它做一次多普勒 FFT、裁剪到 freq_range。
      降采样只是换一种更快的方式得到跟"整窗做 FFT 再裁剪"同样的±freq_range 窄带结果，
      不会引入之前 16 跳采样(在相关之前、没有滤波)那种折叠噪声的问题。
    - 内存占用只跟 step/margin/nperseg_dec 有关，跟总采集时长无关。
    - step、interp_factor 改了之后，margin/降采样链/warmup 长度都是按公式自动重算的，
      不需要改这个函数内部逻辑。
    """
    if not os.path.isdir(experiment_folder):
        print(f"文件夹不存在: {experiment_folder}")
        return None
    data_file = os.path.join(experiment_folder, "2ch_iq_data.bin")
    if not os.path.exists(data_file):
        print(f"数据文件不存在: {data_file}")
        return None
    params = load_acquisition_parameters(experiment_folder)
    if not check_data_format(params, data_file):
        return None
    sample_rate_hz = params["sample_rate_hz"] if params and "sample_rate_hz" in params else 20e6
    use_sec = max_duration_sec if max_duration_sec is not None else DEFAULT_MAX_DURATION_SEC

    ch0_memmap, ch1_memmap, total_per_ch = read_2ch_iq_memmap(data_file)
    if ch0_memmap is None:
        print("内存映射创建失败")
        return None
    n_total = min(total_per_ch, int(sample_rate_hz * use_sec))
    if n_total < 2:
        return None

    ok, n0, n1 = check_iq_magnitude_abort_memmap(ch0_memmap, ch1_memmap, n_total)
    if not skip_magnitude_check and not ok:
        print(f"警告: 幅度超出 |IQ|>{MAGNITUDE_ABORT_THRESHOLD} 的点数超过阈值 "
              f"(ch0={n0}, ch1={n1})，可能存在削波/增益过高/定标错位，结果仅供参考。")

    U = int(interp_factor)
    max_delay_raw = int(max_delay_raw)
    step_samples = int(round(sample_rate_hz * step))
    if step_samples < 1:
        print(f"step={step}s 对应的 step_samples<1，太小了")
        return None

    q_total_dop = max(1, int(round(sample_rate_hz / doppler_target_fs_hz)))
    stages = _decimate_chain_factors(q_total_dop)
    actual_decimated_fs = sample_rate_hz / q_total_dop
    nperseg_dec = int(round(Tw * actual_decimated_fs))
    if nperseg_dec < 8:
        print(f"警告: nperseg_dec={nperseg_dec} 太小(Tw={Tw}s, 降采样后速率={actual_decimated_fs:.1f}Hz)，"
              f"多普勒频率分辨率会很差，建议调大 Tw 或调大 doppler_target_fs_hz。")
    keep_count = max(1, int(round(step_samples / q_total_dop)))
    margin_dop_raw = _decimate_margin_raw(stages)

    # 约定跟原脚本(PDP/旧GIF)一致: prod[j] ~ conj(ch0[t_j]) * ch1[t_j + d]，d 越大表示反射比
    # 直达径晚到(ch1(t)=ch0(t-d))。所以 ch1 需要往"未来"多读 max_delay_raw 个原始采样点
    # (tail 方向)，而不是往过去多读——这个 tail 余量跟 decimate 的 margin_dop_raw(在 front)
    # 是两件独立的事，别搞混。
    front_raw = margin_dop_raw + FILTER_EDGE_MARGIN_RAW
    tail_raw = max_delay_raw + FILTER_EDGE_MARGIN_RAW

    delay_axis = build_delay_axis_raw(max_delay_raw, U)
    n_delay = len(delay_axis)
    delay_k = np.round(delay_axis * U).astype(np.int64)
    base = FILTER_EDGE_MARGIN_RAW * U

    freq_full = np.fft.fftshift(np.fft.fftfreq(nperseg_dec, d=1.0 / actual_decimated_fs))
    fmin, fmax = freq_range
    freq_mask = (freq_full >= fmin) & (freq_full <= fmax)
    freq_axis = freq_full[freq_mask]
    n_freq = int(np.sum(freq_mask))
    win_dec = windows.blackman(nperseg_dec).astype(np.float32)

    ws0 = front_raw
    if ws0 + step_samples + tail_raw > n_total:
        print("数据过短，连一个 hop 都不够(margin 比数据还长)")
        return None
    n_hops = 1 + (n_total - tail_raw - step_samples - ws0) // step_samples
    warmup_hops = int(np.ceil(nperseg_dec / keep_count))
    if n_hops < warmup_hops:
        print(f"数据过短，凑不满一个完整的 Tw 窗口(需要至少 {warmup_hops} 个 hop 累积到 "
              f"nperseg_dec={nperseg_dec} 个降采样点，但总共只有 {n_hops} 个 hop)")
        return None
    n_win = n_hops - warmup_hops + 1

    print(f"CAF matrix(流式): sample_rate={sample_rate_hz/1e6:.2f}MHz, step={step}s(={step_samples}样点/hop), "
          f"Tw={Tw}s, 多普勒降采样链={stages}(共{q_total_dop}倍 -> {actual_decimated_fs:.1f}Hz), "
          f"nperseg_dec={nperseg_dec}, 边界余量={margin_dop_raw}样点(margin/step={margin_dop_raw/step_samples:.2f})",
          flush=True)
    print(f"delay 0..{max_delay_raw} (x{U} 插值, 共 {n_delay} 档) x {n_win} 个时间窗口 x {n_freq} 个多普勒频点, "
          f"Sxx 约 {n_delay * n_win * n_freq * 8 / (1024*1024):.1f} MB, 共 {n_hops} 个 hop", flush=True)

    Sxx = np.zeros((n_delay, n_win, n_freq), dtype=np.complex64)
    t_axis = np.zeros(n_win, dtype=np.float64)
    dec_buf = np.zeros((n_delay, nperseg_dec), dtype=np.complex64)
    filled = 0
    out_idx = 0
    len_prod = margin_dop_raw + step_samples

    t_print_every = max(1, n_hops // 20)
    for hop in range(n_hops):
        ws = ws0 + hop * step_samples
        ch0_chunk = sc16_to_complex64(ch0_memmap[ws - margin_dop_raw: ws + step_samples])
        ch1_chunk_raw = sc16_to_complex64(ch1_memmap[ws - front_raw: ws + step_samples + tail_raw])
        ch1_up = resample_poly(ch1_chunk_raw, U, 1).astype(np.complex64)

        new_batch = np.empty((n_delay, keep_count), dtype=np.complex64)
        for ki in range(n_delay):
            k = int(delay_k[ki])
            start_idx = base + k
            ch1_shift = ch1_up[start_idx: start_idx + len_prod * U: U]
            prod = np.conj(ch0_chunk) * ch1_shift
            dec = chained_decimate(prod, stages)
            new_batch[ki, :] = dec[-keep_count:]
        del ch0_chunk, ch1_chunk_raw, ch1_up

        dec_buf[:, :-keep_count] = dec_buf[:, keep_count:]
        dec_buf[:, -keep_count:] = new_batch
        filled = min(nperseg_dec, filled + keep_count)

        if filled >= nperseg_dec:
            spec = np.fft.fftshift(sp_fft(dec_buf * win_dec[np.newaxis, :], axis=-1, workers=-1), axes=-1)
            Sxx[:, out_idx, :] = spec[:, freq_mask]
            t_axis[out_idx] = (ws + step_samples) / sample_rate_hz - Tw / 2.0
            out_idx += 1

        if (hop + 1) % t_print_every == 0 or hop == n_hops - 1:
            print(f"  hop {hop + 1}/{n_hops} 完成 (已出 {out_idx}/{n_win} 个时间窗口)", flush=True)

    os.makedirs(out_dir, exist_ok=True)
    base_name = os.path.basename(os.path.normpath(experiment_folder))
    matrix_path = os.path.join(out_dir, f"caf_matrix_{base_name}.npz")
    meta = dict(sample_rate_hz=sample_rate_hz, Tw=Tw, step=step, freq_range=list(freq_range),
                max_delay_raw=max_delay_raw, interp_factor=interp_factor, n_total=n_total,
                doppler_target_fs_hz=doppler_target_fs_hz, actual_decimated_fs_hz=actual_decimated_fs,
                doppler_decimate_stages=stages, nperseg_dec=nperseg_dec,
                experiment_folder=experiment_folder)
    np.savez_compressed(matrix_path, Sxx=Sxx, delay_axis=delay_axis, t_axis=t_axis,
                         freq_axis=freq_axis, meta=json.dumps(meta))
    print(f"CAF matrix 已保存: {matrix_path}", flush=True)
    return matrix_path


def get_caf_matrix_path(experiment_folder, out_dir=CAF_MATRIX_OUT_DIR):
    base_name = os.path.basename(os.path.normpath(experiment_folder))
    return os.path.join(out_dir, f"caf_matrix_{base_name}.npz")


def load_caf_matrix(matrix_path):
    npz = np.load(matrix_path, allow_pickle=False)
    meta = json.loads(str(npz["meta"]))
    return {
        "Sxx": npz["Sxx"],              # complex64, shape (n_delay, n_win, n_freq)
        "delay_axis": npz["delay_axis"],  # 原始采样点单位，0..max_delay_raw，间隔 1/interp_factor
        "t_axis": npz["t_axis"],
        "freq_axis": npz["freq_axis"],
        "meta": meta,
    }


def get_or_build_caf_matrix(experiment_folder, out_dir=CAF_MATRIX_OUT_DIR, force_rebuild=False, **kwargs):
    """主入口：检测到缓存的 matrix 就直接加载，不重新计算；没有就算好存下来再返回。"""
    matrix_path = get_caf_matrix_path(experiment_folder, out_dir)
    if not force_rebuild and os.path.exists(matrix_path):
        print(f"发现已缓存的 CAF matrix: {matrix_path}，直接加载，不重新计算。", flush=True)
        return load_caf_matrix(matrix_path)
    print(f"未找到缓存 matrix ({matrix_path})，开始计算(全采样率，不做16降采样)...", flush=True)
    computed_path = compute_caf_matrix(experiment_folder, out_dir=out_dir, **kwargs)
    if computed_path is None:
        return None
    return load_caf_matrix(computed_path)


def plot_caf_matrix_slice(data, delay_value, out_dir=CAF_MATRIX_OUT_DIR, base_name="caf",
                           time_range=None, freq_range=None):
    """从缓存的 matrix 里挑一个 delay(最接近 delay_value 的档)，画 时间x多普勒 幅度图，不重新计算。"""
    Sxx, delay_axis, t_axis, freq_axis = data["Sxx"], data["delay_axis"], data["t_axis"], data["freq_axis"]
    fs = data["meta"].get("sample_rate_hz")
    di = int(np.argmin(np.abs(delay_axis - delay_value)))
    t_mask = np.ones_like(t_axis, dtype=bool) if time_range is None else (t_axis >= time_range[0]) & (t_axis <= time_range[1])
    f_mask = np.ones_like(freq_axis, dtype=bool) if freq_range is None else (freq_axis >= freq_range[0]) & (freq_axis <= freq_range[1])

    S = Sxx[di][t_mask][:, f_mask]
    Sxx_db = 10 * np.log10(np.abs(S).T + 1e-10)
    t_sel, f_sel = t_axis[t_mask], freq_axis[f_mask]
    vmin, vmax = np.percentile(Sxx_db, 5), np.percentile(Sxx_db, 95)
    _, step_m = range_step_for_sample_rate(fs)
    r_m = delay_axis[di] * step_m

    fig, ax = plt.subplots(figsize=(12, 6))
    mesh = ax.pcolormesh(t_sel, f_sel, Sxx_db, shading="auto", cmap="jet", vmin=vmin, vmax=vmax)
    ax.set_ylabel("Doppler Frequency (Hz)")
    ax.set_xlabel("Time (s)")
    ax.set_title(f"CAF amplitude — {base_name}  |  delay={delay_axis[di]:.2f} samples (~{r_m:.2f} m)")
    plt.colorbar(mesh, ax=ax, label="Relative Power (dB)")
    plt.tight_layout()
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"caf_matrix_slice_d{delay_axis[di]:.2f}_{base_name}.png")
    plt.savefig(out_path, bbox_inches="tight")
    plt.show()
    plt.close(fig)
    return out_path


def plot_caf_matrix_gif(data, delay_min=None, delay_max=None, out_dir=CAF_MATRIX_OUT_DIR, base_name="caf",
                         gif_frame_duration_sec=GIF_FRAME_DURATION_SEC, gif_name=None, vmin=None, vmax=None):
    """从缓存的 matrix 里按 delay 逐档出图拼 GIF，数据都已经算好，只做切片+画图，不重新计算。
    vmin/vmax 不传时按 5/95 百分位自动算，传了就用固定色标(方便跟其他图对比/复现)。"""
    try:
        from PIL import Image
    except ImportError:
        print("需要 Pillow (PIL) 才能导出 GIF，请安装: pip install Pillow")
        return None
    Sxx, delay_axis, t_axis, freq_axis = data["Sxx"], data["delay_axis"], data["t_axis"], data["freq_axis"]
    fs = data["meta"].get("sample_rate_hz")
    dmin = delay_axis[0] if delay_min is None else delay_min
    dmax = delay_axis[-1] if delay_max is None else delay_max
    idxs = np.where((delay_axis >= dmin) & (delay_axis <= dmax))[0]
    if len(idxs) == 0:
        print("所选 delay 范围内没有数据")
        return None

    if vmin is None or vmax is None:
        Sxx_db_all = 10 * np.log10(np.abs(Sxx[idxs]) + 1e-10)
        v_lo, v_hi = np.percentile(Sxx_db_all, 5), np.percentile(Sxx_db_all, 95)
        vmin = v_lo if vmin is None else vmin
        vmax = v_hi if vmax is None else vmax
    _, step_m = range_step_for_sample_rate(fs)

    os.makedirs(out_dir, exist_ok=True)
    temp_pngs = []
    for di in idxs:
        Sxx_db = 10 * np.log10(np.abs(Sxx[di]).T + 1e-10)
        r_m = delay_axis[di] * step_m
        fig, ax = plt.subplots(figsize=(12, 6))
        mesh = ax.pcolormesh(t_axis, freq_axis, Sxx_db, shading="auto", cmap="jet", vmin=vmin, vmax=vmax)
        ax.set_ylabel("Doppler Frequency (Hz)")
        ax.set_xlabel("Time (s)")
        ax.set_title(f"CAF amplitude — {base_name}  |  delay={delay_axis[di]:.2f} samples")
        ax.text(0.98, 0.98, f"Range: ~{r_m:.2f} m", transform=ax.transAxes, fontsize=16, fontweight="bold",
                verticalalignment="top", horizontalalignment="right",
                bbox=dict(boxstyle="round,pad=0.5", facecolor="yellow", alpha=0.9))
        plt.colorbar(mesh, ax=ax, label="Relative Power (dB)")
        plt.tight_layout()
        tmp_path = os.path.join(out_dir, f"_gif_frame_matrix_d{delay_axis[di]:.2f}_{base_name}.png")
        plt.savefig(tmp_path, format="png", dpi=100, bbox_inches="tight")
        plt.close(fig)
        temp_pngs.append(tmp_path)

    frames_pil = [Image.open(p).convert("RGB") for p in temp_pngs]
    out_name = gif_name or f"caf_matrix_delay{dmin:.2f}_to_{dmax:.2f}_{base_name}.gif"
    out_path = os.path.join(out_dir, out_name)
    frames_pil[0].save(out_path, save_all=True, append_images=frames_pil[1:],
                        duration=int(gif_frame_duration_sec * 1000), loop=0)
    for p in temp_pngs:
        try:
            os.remove(p)
        except OSError:
            pass
    print(f"GIF 已保存: {out_path}（{len(frames_pil)} 帧）")
    return out_path


def render_range_doppler_mp4(data, out_path, fps=None, freq_range=None, range_max_m=None,
                              vmin=None, vmax=None, cmap="jet", dpi=110, figsize=(9, 6)):
    """从缓存的 CAF matrix 直接出 mp4：每一帧代表一个时刻(横轴=多普勒频率Hz，纵轴=距离m)，
    不再是"每帧一个delay"那种。播放速度按 1:1 真实时间(fps = 1/step)，不额外加速/减速——
    step 秒的数据对应 1/step 帧，放出来正好是原始录制时长。
    数据都已经在缓存的 matrix 里算好，这里只做切片/取 dB/渲染，不重新跑 CAF。"""
    import matplotlib.animation as animation

    Sxx, delay_axis, t_axis, freq_axis = data["Sxx"], data["delay_axis"], data["t_axis"], data["freq_axis"]
    meta = data["meta"]
    sample_rate_hz = meta["sample_rate_hz"]
    step = meta["step"]

    _, step_m = range_step_for_sample_rate(sample_rate_hz)
    range_axis = delay_axis * step_m  # delay(原始采样点) -> 距离(米)

    if fps is None:
        fps = 1.0 / step  # 1:1 真实时间播放

    fmask = np.ones_like(freq_axis, dtype=bool) if freq_range is None else \
        (freq_axis >= freq_range[0]) & (freq_axis <= freq_range[1])
    rmask = np.ones_like(range_axis, dtype=bool) if range_max_m is None else (range_axis <= range_max_m)
    f_sel = freq_axis[fmask]
    r_sel = range_axis[rmask]
    if len(f_sel) < 2 or len(r_sel) < 2:
        print("freq_range/range_max_m 筛选后剩下的点太少，检查一下参数")
        return None

    # (n_delay, n_win, n_freq) -> 按 range/freq 裁剪，Sxx[:, i, :] 天然就是 (range行, freq列)，
    # 正好是 imshow 要的形状，不用转置。
    power_db = 10 * np.log10(np.abs(Sxx[rmask][:, :, fmask]) ** 2 + 1e-20).astype(np.float32)
    n_win = power_db.shape[1]

    if vmin is None or vmax is None:
        v_lo, v_hi = np.percentile(power_db, [5, 95])
        vmin = v_lo if vmin is None else vmin
        vmax = v_hi if vmax is None else vmax

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(power_db[:, 0, :], origin="lower", aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax,
                   extent=[f_sel[0], f_sel[-1], r_sel[0], r_sel[-1]])
    ax.set_xlabel("Doppler Frequency (Hz)")
    ax.set_ylabel("Range (m)")
    title = ax.set_title(f"t = {t_axis[0]:.2f} s")
    fig.colorbar(im, ax=ax, label="Relative Power (dB)")
    plt.tight_layout()

    def update(i):
        im.set_data(power_db[:, i, :])
        title.set_text(f"t = {t_axis[i]:.2f} s")
        return [im, title]

    print(f"渲染 range-Doppler mp4: {n_win} 帧, fps={fps:.3f}(=1/step, 1:1 真实时间), "
          f"时长≈{n_win / fps:.2f}s, 距离范围{r_sel[0]:.1f}~{r_sel[-1]:.1f}m, "
          f"多普勒范围{f_sel[0]:.0f}~{f_sel[-1]:.0f}Hz", flush=True)
    anim = animation.FuncAnimation(fig, update, frames=n_win, blit=False)
    writer = animation.FFMpegWriter(fps=fps, codec="libx264", extra_args=["-pix_fmt", "yuv420p"])
    anim.save(out_path, writer=writer, dpi=dpi)
    plt.close(fig)
    print(f"mp4 已保存: {out_path}", flush=True)
    return out_path


def run_validation(experiment_folder, Tw=TW_SEC, step=STEP_SEC, freq_range=(-600, 600), skip_magnitude_check=False,
                   max_duration_sec=None, use_pdp=True, show_phase=True):
    if experiment_folder is None:
        experiment_folder = find_latest_experiment_folder()
        if experiment_folder is None:
            print("未找到任何 experiment_* 文件夹")
            return None
    if not os.path.isdir(experiment_folder):
        print(f"文件夹不存在: {experiment_folder}")
        return None
    data_file = os.path.join(experiment_folder, "2ch_iq_data.bin")
    if not os.path.exists(data_file):
        print(f"数据文件不存在: {data_file}")
        return None
    params = load_acquisition_parameters(experiment_folder)
    if not check_data_format(params, data_file):
        return None
    sample_rate_hz = params["sample_rate_hz"] if params and "sample_rate_hz" in params else 20e6
    use_sec = max_duration_sec if max_duration_sec is not None else DEFAULT_MAX_DURATION_SEC
    print(f"验证: 打开 memmap，加载前 {use_sec}s 数据...", flush=True)
    ch0_memmap, ch1_memmap, total_per_ch = read_2ch_iq_memmap(data_file)
    if ch0_memmap is None:
        print("内存映射创建失败")
        return None
    n_total = min(total_per_ch, int(sample_rate_hz * use_sec))
    if n_total < 2:
        return None
    print(f"验证: {n_total} 样点/通道，按 decim={DECIM} 跳采样处理 (峰值内存约 "
          f"{n_total*8*2/DECIM/(1024*1024):.0f} MB，而不是整段 {n_total*8*2/(1024*1024):.0f} MB)...", flush=True)
    effective_fs = sample_rate_hz / DECIM
    nperseg = int(effective_fs * Tw)
    if n_total < nperseg * DECIM:
        print("数据过短，无法做 CAF")
        return None
    ok, n0, n1 = check_iq_magnitude_abort_memmap(ch0_memmap, ch1_memmap, n_total)
    if not skip_magnitude_check and not ok:
        print(f"警告: 幅度超出 |IQ|>{MAGNITUDE_ABORT_THRESHOLD} 的点数超过阈值 "
              f"(ch0={n0}, ch1={n1})，可能存在削波/增益过高/定标错位，结果仅供参考。"
              f"仍继续计算并出图，请自行核实。")
    f_axis, t_axis, Sxx_db, Sxx_complex, eff_fs, delay_max, pdp_amp, pdp_delay_axis = fast_caf_spectrogram(
        ch0_memmap, ch1_memmap, n_total, sample_rate_hz, Tw=Tw, step=step, freq_range=freq_range, use_pdp=use_pdp)
    if use_pdp:
        print(f"PDP delay = {delay_max} (范围 {PDP_DELAY_NEG}..+{PDP_DELAY_POS})")
    print(f"CAF 形状: {Sxx_db.shape}, 有效采样率 = {eff_fs/1e6:.2f} MHz")

    out_dir = "validate_caf_output"
    os.makedirs(out_dir, exist_ok=True)
    base_name = os.path.basename(experiment_folder)

    if use_pdp and pdp_amp is not None and pdp_delay_axis is not None and np.any(pdp_amp > 0):
        pdp_db = 10 * np.log10(pdp_amp + 1e-12)
        fig_pdp, ax_pdp = plt.subplots(figsize=(8, 5))
        ax_pdp.bar(pdp_delay_axis, pdp_db, color="steelblue", edgecolor="navy", alpha=0.8, width=0.7)
        ax_pdp.axvline(delay_max, color="red", linestyle="--", linewidth=1.5, label=f"delay={delay_max}")
        ax_pdp.set_xlabel("Delay (bin)")
        ax_pdp.set_ylabel("PDP amplitude (dB)")
        ax_pdp.set_title(f"PDP — {base_name}")
        ax_pdp.legend()
        ax_pdp.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"pdp_{base_name}.png"), bbox_inches="tight")
        plt.show()
        plt.close(fig_pdp)

    vmin, vmax = np.percentile(Sxx_db, 5), np.percentile(Sxx_db, 95)

    if show_phase:
        use_phase_derotation = True
        if use_phase_derotation:
            phase_derotation = np.exp(-1j * 2 * np.pi * f_axis.reshape(-1, 1) * t_axis.reshape(1, -1))
            Sxx_derot = Sxx_complex * phase_derotation
        else:
            Sxx_derot = Sxx_complex
        phase_plot_deg = np.rad2deg(np.angle(Sxx_derot))

        pos_mask = f_axis > 0
        neg_mask = f_axis < 0
        zero_mask = np.abs(f_axis) < 1e-6
        has_fused = False
        if np.any(pos_mask) and np.any(neg_mask):
            min_len = min(np.sum(pos_mask), np.sum(neg_mask))
            freqs_pos = f_axis[pos_mask][:min_len]
            freqs_neg = f_axis[neg_mask][:min_len]
            S_pos = Sxx_derot[pos_mask, :][:min_len, :]
            S_neg = Sxx_derot[neg_mask, :][:min_len, :]
            S_neg_flipped = np.flipud(S_neg)
            S_fused = S_pos * S_neg_flipped
            phase_fused_deg = np.rad2deg(np.angle(S_fused))
            if np.any(zero_mask):
                S_0 = Sxx_derot[zero_mask, :][:1, :]
                phase_fused_deg = np.vstack([np.rad2deg(np.angle(S_0)), phase_fused_deg])
                freq_axis_folded = np.concatenate([[0.0], freqs_pos])
            else:
                freq_axis_folded = freqs_pos
            has_fused = True

        n_rows = 3 if has_fused else 2
        fig, axes = plt.subplots(n_rows, 1, figsize=(12, 4 * n_rows), sharex=True)
        ax_amp, ax_phase = axes[0], axes[1]
        mesh_amp = ax_amp.pcolormesh(t_axis, f_axis, Sxx_db, shading="auto", cmap="jet", vmin=vmin, vmax=vmax)
        ax_amp.set_ylabel("Doppler Frequency (Hz)")
        ax_amp.set_title(f"CAF amplitude — {base_name} (Tw={Tw}s, step={step}s, delay={delay_max})")
        fig.colorbar(mesh_amp, ax=ax_amp, label="Relative Power (dB)")
        mesh_phase = ax_phase.pcolormesh(t_axis, f_axis, phase_plot_deg, shading="auto", cmap="hsv", vmin=-180, vmax=180)
        ax_phase.set_ylabel("Doppler Frequency (Hz)")
        ax_phase.set_xlabel("Time (s)" if not has_fused else None)
        ax_phase.set_title("Doppler phase")
        fig.colorbar(mesh_phase, ax=ax_phase, label="Phase (deg)")
        if has_fused:
            ax_fused = axes[2]
            mesh_fused = ax_fused.pcolormesh(t_axis, freq_axis_folded, phase_fused_deg, shading="auto", cmap="hsv", vmin=-180, vmax=180)
            ax_fused.set_ylabel("Doppler Frequency (Hz)")
            ax_fused.set_xlabel("Time (s)")
            ax_fused.set_title("Doppler phase (folded: 0 alone, +f x -f)")
            fig.colorbar(mesh_fused, ax=ax_fused, label="Phase (deg)")
        plt.tight_layout()
        out_path = os.path.join(out_dir, f"caf_amp_phase_{base_name}.png")
    else:
        fig, ax_amp = plt.subplots(figsize=(12, 6))
        mesh_amp = ax_amp.pcolormesh(t_axis, f_axis, Sxx_db, shading="auto", cmap="jet", vmin=vmin, vmax=vmax)
        ax_amp.set_ylabel("Doppler Frequency (Hz)")
        ax_amp.set_xlabel("Time (s)")
        ax_amp.set_title(f"CAF amplitude — {base_name} (Tw={Tw}s, step={step}s, delay={delay_max})")
        fig.colorbar(mesh_amp, ax=ax_amp, label="Relative Power (dB)")
        plt.tight_layout()
        out_path = os.path.join(out_dir, f"caf_{base_name}.png")
    plt.savefig(out_path, bbox_inches="tight")
    print(f"谱图已保存: {out_path}")
    plt.show()
    plt.close(fig)
    return f_axis, t_axis, Sxx_db


def run_validation_gif(
    experiment_folder,
    delay_min=DELAY_MIN_GIF,
    delay_max=DELAY_MAX_GIF,
    Tw=TW_SEC,
    step=STEP_SEC,
    freq_range=(-600, 600),
    skip_magnitude_check=False,
    max_duration_sec=10.0,
    gif_frame_duration_sec=GIF_FRAME_DURATION_SEC,
    first_bin_max_m=None,
    step_m=None,
    out_dir="validate_caf_output",
    gif_name=None,
):
    """
    多 delay（delay_min 到 delay_max，默认 -3..+3）各出一张 CAF 幅度谱图，每张显眼处标 delay 对应距离，
    存成临时 PNG 后立刻关图释内存；最后再读这些图拼成 GIF，拼完删临时文件。
    默认只读前 10 秒数据，避免大文件加载过久卡死（可传 max_duration_sec=30 等）。
    """
    print("GIF: 开始 run_validation_gif...", flush=True)
    try:
        from PIL import Image
    except ImportError:
        print("需要 Pillow (PIL) 才能导出 GIF，请安装: pip install Pillow")
        return None
    if experiment_folder is None:
        experiment_folder = find_latest_experiment_folder()
    if experiment_folder is None:
        print("未找到任何 experiment_* 文件夹")
        return None
    if not os.path.isdir(experiment_folder):
        print(f"文件夹不存在: {experiment_folder}")
        return None
    data_file = os.path.join(experiment_folder, "2ch_iq_data.bin")
    if not os.path.exists(data_file):
        print(f"数据文件不存在: {data_file}")
        return None
    params = load_acquisition_parameters(experiment_folder)
    if not check_data_format(params, data_file):
        return None
    sample_rate_hz = params["sample_rate_hz"] if params and "sample_rate_hz" in params else 20e6
    if step_m is None or first_bin_max_m is None:
        auto_first_bin_max_m, auto_step_m = range_step_for_sample_rate(sample_rate_hz)
        if step_m is None:
            step_m = auto_step_m
        if first_bin_max_m is None:
            first_bin_max_m = auto_first_bin_max_m
    print(f"GIF: 打开 memmap {data_file} ...", flush=True)
    ch0_memmap, ch1_memmap, total_per_ch = read_2ch_iq_memmap(data_file)
    if ch0_memmap is None:
        print("内存映射创建失败")
        return None
    n_total = min(total_per_ch, int(sample_rate_hz * max_duration_sec))
    if n_total < 2:
        return None
    n_mb_full = n_total * 8 * 2 / (1024 * 1024)  # 整段转 complex64 会占用的大小（两通道）——之前就是卡在这里
    n_mb_ds = n_mb_full / DECIM
    print(f"GIF: {n_total} 样点/通道，按 decim={DECIM} 跳采样逐 delay 处理 "
          f"(每次峰值内存约 {n_mb_ds:.0f} MB，而不是整段 {n_mb_full:.0f} MB)...", flush=True)
    effective_fs = sample_rate_hz / DECIM
    nperseg = int(effective_fs * Tw)
    if n_total < nperseg * DECIM:
        print("数据过短，无法做 CAF")
        return None
    ok, n0, n1 = check_iq_magnitude_abort_memmap(ch0_memmap, ch1_memmap, n_total)
    if not skip_magnitude_check and not ok:
        print(f"警告: 幅度超出 |IQ|>{MAGNITUDE_ABORT_THRESHOLD} 的点数超过阈值 "
              f"(ch0={n0}, ch1={n1})，可能存在削波/增益过高/定标错位，结果仅供参考。"
              f"仍继续计算并出图，请自行核实。")

    os.makedirs(out_dir, exist_ok=True)
    base_name = os.path.basename(experiment_folder)
    vmin, vmax = None, None
    temp_pngs = []

    for d in range(delay_min, delay_max + 1):
        f_axis, t_axis, Sxx_db = fast_caf_spectrogram_at_delay(
            ch0_memmap, ch1_memmap, d, n_total, sample_rate_hz, Tw=Tw, step=step, decim=DECIM, freq_range=freq_range
        )
        if vmin is None:
            vmin, vmax = np.percentile(Sxx_db, 5), np.percentile(Sxx_db, 95)
        r_min, r_max = range_label_for_delay(d, first_bin_max_m=first_bin_max_m, step_m=step_m)
        fig, ax = plt.subplots(figsize=(12, 6))
        mesh = ax.pcolormesh(t_axis, f_axis, Sxx_db, shading="auto", cmap="jet", vmin=vmin, vmax=vmax)
        ax.set_ylabel("Doppler Frequency (Hz)")
        ax.set_xlabel("Time (s)")
        ax.set_title(f"CAF amplitude — {base_name}  |  delay={d}")
        text = f"Range: {r_min:.1f} - {r_max:.1f} m"
        ax.text(0.98, 0.98, text, transform=ax.transAxes, fontsize=16, fontweight="bold",
                verticalalignment="top", horizontalalignment="right",
                bbox=dict(boxstyle="round,pad=0.5", facecolor="yellow", alpha=0.9))
        plt.colorbar(mesh, ax=ax, label="Relative Power (dB)")
        plt.tight_layout()
        tmp_path = os.path.join(out_dir, f"_gif_frame_d{d}_{base_name}.png")
        plt.savefig(tmp_path, format="png", dpi=100, bbox_inches="tight")
        plt.close(fig)
        temp_pngs.append(tmp_path)

    if not temp_pngs:
        print("没有可用的帧")
        return None
    frames_pil = [Image.open(p).convert("RGB") for p in temp_pngs]
    out_name = gif_name or f"caf_delay{delay_min}_to_{delay_max}_{base_name}.gif"
    out_path = os.path.join(out_dir, out_name)
    frames_pil[0].save(
        out_path,
        save_all=True,
        append_images=frames_pil[1:],
        duration=int(gif_frame_duration_sec * 1000),
        loop=0,
    )
    for p in temp_pngs:
        try:
            os.remove(p)
        except OSError:
            pass
    print(f"GIF 已保存: {out_path}（{len(frames_pil)} 帧，每帧 {gif_frame_duration_sec}s）")
    return out_path


def main():
    print("=== analyze_caf 启动（仅分析，不采数；全采样率 CAF matrix，带缓存） ===", flush=True)
    # 与 doppler_3d_time_delay.py 一致：可指定文件夹，None 则用「最新」experiment_*
    experiment_folder = None  # 或写死如 "experiment_30MHz_static_20260710_161817"
    if experiment_folder is None:
        print("查找最新 experiment_* 文件夹...", flush=True)
        experiment_folder = find_latest_experiment_folder()
    if experiment_folder is None:
        print("未找到任何 experiment_* 文件夹")
        return
    print(f"使用实验文件夹: {experiment_folder}", flush=True)

    # 大流程：检测到缓存的 matrix(caf_matrix_<experiment>.npz) 就直接加载，不重新计算；
    # 没有就用全采样率(不降采样)逐窗口算好，delay 0..MAX_DELAY_RAW、4倍插值、存下来。
    data = get_or_build_caf_matrix(
        experiment_folder,
        max_duration_sec=ACQUISITION_DURATION_SEC,
        Tw=TW_SEC, step=STEP_SEC, freq_range=(-600, 600),
        max_delay_raw=MAX_DELAY_RAW, interp_factor=DELAY_INTERP_FACTOR,
    )
    if data is None:
        return

    base_name = os.path.basename(os.path.normpath(experiment_folder))
    plot_caf_matrix_gif(data, delay_min=0, delay_max=MAX_DELAY_RAW, base_name=base_name,
                         gif_frame_duration_sec=GIF_FRAME_DURATION_SEC)


if __name__ == "__main__":
    main()
