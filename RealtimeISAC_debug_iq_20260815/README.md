# USRP B210 0815 capture (RealtimeISAC pipeline dump)

`iq_20260815_175024.npz` — a raw IQ snapshot saved by the RealtimeISAC real-time pipeline's `rt_iqdump` tool on 2026-08-15 17:50:24, at `fs=10MHz` (different from the 0710 captures' 30MHz). Small enough (40MB) to commit directly, no splitting needed.

## Contents (`np.load` keys)

| key | dtype/shape | meaning |
|---|---|---|
| `iq` | `int16`, `(2, 10000000)` | 2 channels × interleaved (re,im) int16, i.e. 5,000,000 complex samples/channel at 10MHz ≈ 0.5s. Same sc16 convention as the 0710 USRP data. |
| `fs` | `float64` scalar | sample rate (Hz) |
| `center_freq` | `float64` scalar | RF center frequency (Hz) |
| `gain` | `float64` scalar | SDR gain (dB) |
| `phase` | `int64` scalar | TDD superframe sync phase offset (samples), from `rt_sync`/`TddSync` |
| `n_period` | `int64` scalar | TDD superframe period (samples) |
| `n_int` | `int64` scalar | integration length per Doppler frame (samples) |
| `locked` | `bool` scalar | whether TDD sync achieved lock when this was captured |
| `contrast` | `float64` scalar | sync quality metric (envelope contrast) |

Load with:
```python
import numpy as np
d = np.load("iq_20260815_175024.npz")
iq, fs = d["iq"], float(d["fs"])
```

## Analysis code (`analysis_code/`)

- `oldcaf_on_iqdump.py` — runs the old CAF algorithm (no TDD gating, no per-frame normalization, no DC removal) directly on this npz:
  ```
  python oldcaf_on_iqdump.py iq_20260815_175024.npz
  python oldcaf_on_iqdump.py iq_20260815_175024.npz --step 0.02 --tw 0.2 --fmax 500
  ```
- `compare_comb_jul_vs_aug.py` — cross-checks whether the ±50/100/200/300Hz Doppler lines are real TX behavior or algorithm artifacts, by running 3 algorithm variants (oldCAF / RT-noDC / RT-perstepDC) on **both** this 0815 capture and a 0710 USRP experiment (converts both to a common interleaved-int16 temp `.bin` first):
  ```
  python compare_comb_jul_vs_aug.py   # default: 0710/155243 vs iq_20260815_175024
  python compare_comb_jul_vs_aug.py --jul <experiment_folder> --aug <npz_path> --sec 0.5
  ```
  Needs the `USRP_B210_0710/raw_iq_155243/` data (reassembled) alongside for its default `--jul` argument.
- `analyze_caf.py` — the core CAF/FFT module both scripts import (`import analyze_caf as ac`); same file as in `USRP_B210_0710/analysis_code/`.
