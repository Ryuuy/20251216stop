# USRP B210 0710 data

Static-scene captures from 2026-07-10, USRP B210 (`sample_rate=30MHz`, `center_freq=1.89GHz`, gain=30dB, 2 channels, `serial=321D889`). Companion to the LimeSDR data in this repo's `Channel0/`/`Channel1/`.

## Data format

Each `raw_iq_*/` folder holds one experiment's raw `2ch_iq_data.bin`: interleaved `sc16` (16-bit signed int, little-endian) — `ch0[0](re,im), ch1[0](re,im), ch0[1], ch1[1], ...` (see that folder's `acquisition_parameters.json` for the exact field, `sc16_scale=32767.0` to normalize to ±1.0). To avoid GitHub's 100MB file limit, the `.bin` is split into 90MB parts (`2ch_iq_data.bin.part00`, `part01`, ...) — reassemble with:
```
cat 2ch_iq_data.bin.part* > 2ch_iq_data.bin
```
(all parts verified byte-identical to the original via md5 before splitting.)

## Contents

- `raw_iq_155243/` — raw capture from `experiment_30MHz_static_20260710_155243` (4.5GB before splitting). The dataset most reused across this project's analysis scripts.
- `raw_iq_161256/` — raw capture from `experiment_30MHz_static_20260710_161256` (4.5GB before splitting). Also has a rendered CAF micro-Doppler GIF in the original project.
- `paper_figures/` — processed micro-Doppler CAF figures, generated from `155243` and `161256` by `analysis_code/make_paper_figures.py`.
- `analysis_code/` — `analyze_caf.py` (core CAF/FFT Doppler analysis; see `get_caf_matrix_path()` for how it caches results per experiment folder) and `make_paper_figures.py` (produces the PNGs/PDFs in `paper_figures/`).

## Running the code

```
cd analysis_code
python make_paper_figures.py   # expects ../raw_iq_155243 and ../raw_iq_161256 reassembled as
                                # experiment_30MHz_static_20260710_155243/2ch_iq_data.bin, etc.
                                # (edit OUT_DIR / TARGETS at the top of the script for other ranges)
```
`analyze_caf.py` can also be imported directly (`import analyze_caf as ac`) — see `ac.get_caf_matrix_path(experiment_folder)` and the CAF computation it wraps.
