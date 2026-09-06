# SDR Doppler datasets

Non-cooperative passive mmWave/sub-6GHz ISAC Doppler sensing captures, from two SDR platforms.

## Layout

```
Channel0/, Channel1/          LimeSDR capture "20251216stop" (this repo's original data)
USRP_B210_0710/                USRP B210 captures, 2026-07-10
RealtimeISAC_debug_iq_20260815/  USRP B210 capture, 2026-08-15 (from the RealtimeISAC real-time pipeline)
```

## LimeSDR data (`Channel0/`, `Channel1/`)

One `.bin` file per frame, e.g. `center1895frame1234.bin` — raw `int16` little-endian (`<i2`), one file per channel per frame index. Load and concatenate a frame range:

```python
import numpy as np
from pathlib import Path
ROOT = Path(".")
c0 = np.concatenate([np.fromfile(ROOT/"Channel0"/f"center1895frame{n}.bin", dtype="<i2") for n in range(a, b+1)])
c1 = np.concatenate([np.fromfile(ROOT/"Channel1"/f"center1895frame{n}.bin", dtype="<i2") for n in range(a, b+1)])
```
`center1895` = 1895 MHz center frequency. See `USRP_B210_0710/analysis_code/` scripts (e.g. `limesdr_doppler.py` pattern) for how these frames are decimated/CAF'd.

## USRP B210 data (`USRP_B210_0710/`, `RealtimeISAC_debug_iq_20260815/`)

See the `README.md` inside each folder — different capture format (single `2ch_iq_data.bin` vs `rt_iqdump` `.npz`), documented separately.

## Cross-dataset comparison

`RealtimeISAC_debug_iq_20260815/analysis_code/compare_comb_jul_vs_aug.py` runs the same 3 algorithms (oldCAF / RT-noDC / RT-perstepDC) on both the 2026-07-10 USRP capture and the 2026-08-15 npz, to separate real TX-side spectral lines from algorithm artifacts (see its docstring). Default: `155243` vs `iq_20260815_175024`.

Note: an older USRP outdoor capture (`experiment_30MHz_static_20260227_150212`, fc32 format) is referenced in the codebase's regression-test script (`RealtimeISAC/reference/validate_capture_and_caf_fc32test.py`) but the raw file is no longer on this machine — ask if you need it, it may exist elsewhere.
