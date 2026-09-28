# STEP2b_denoise — DeepCAD-RT self-supervised denoising (BU SCC)

Self-supervised denoising of Femtonics 2P **volumetric** calcium stacks to raise
SNR for every downstream step (activity-statistics mask, NMF cell split,
skeleton labeling, and eventual real-time 3D visualization). DeepCAD-RT needs
**no clean ground truth** — it trains directly on the noisy movie.

Everything cluster-side lives under `/projectnb/devorlab/daria/` only. Nothing
outside daria's own tree was created, modified, or deleted.

---

## TL;DR — current state (verified 2026-09-19)

* **Env built & working:** `/projectnb/devorlab/daria/envs/deepcad` (5.5 G, py3.10, torch 2.2.2+cu121).
* **Training data uploaded:** `…/Femtonics/deepcad/data/{run7_clean.tif, run4_clean.tif}` (both 4D, byte-for-byte size match verified).
* **Training job `7648318` COMPLETED, exit 0** on an L40S (scc-505) in **118 s wallclock** (2 h cap never approached).
* **Model saved:** `…/Femtonics/deepcad/models/planes_202609191434/E_10_Iter_1008.pth` (+ E_01…E_09, `para.yaml`, and matching `.onnx` under `models/onnx/…`).
* Run inference later with: `qsub -v DENOISE_MODEL=planes_202609191434 deepcad_infer.qsub`

> An earlier submission `7648310` **failed instantly** on a shell syntax error
> (an inline `#comment` glued to a `${VAR:-default}` with no space, so bash choked
> on a `(` in the comment). Fixed; both scripts now pass `bash -n`. No GPU compute
> was consumed by the failed job.

---

## 1. Cluster resources (verified, not assumed)

| Thing | Value | How checked |
|---|---|---|
| SGE project (`-P`) | `devorlab` | `qconf -sprjl` |
| GPU request | `-l gpus=1` | `qconf -sc` → `gpus` INT, JOB-consumable |
| GPU capability filter | `-l gpu_c=7.0` (min compute capability) | `qconf -sc` → `gpu_compute_capability` (shortcut `gpu_c`); job landed on L40S cc 8.9, consistent with "≥" semantics |
| CUDA modules present | 10.x–13.2, default 12.8 (also 11.8/12.2/12.5) | `module avail cuda` |
| miniconda module | `miniconda/24.5.0` | `module avail miniconda` |
| GPU pool (cc≥7.0) | V100(16G), A100, A40/A6000/L40S(48G), H200 | `qhost -F gpu_type,gpu_c,gpu_memory` |

`-pe omp 4` requests 4 CPU cores (RAM headroom + numpy/torch intra-op; DeepCAD's
dataloader is `num_workers=0`). `h_rt=02:00:00` is the **hard 2 GPU-hour cap** — SGE kills at it.

---

## 2. Environment — what was installed and where

Location: **`/projectnb/devorlab/daria/envs/deepcad`** (conda prefix env, Python 3.10.21).
Build script kept for provenance: `…/Femtonics/deepcad/build_env.sh` (logs: `build.log`, `build2.log`).

Built on the **login node**, niced (`nice -n 19`), in the background — package
download/unpack is I/O-bound, not heavy compute. **Key gotcha solved:** the login
node's `/tmp` is only **2 G** and `$HOME` is near its 10 G quota, so pip's default
temp/cache blew up with `[Errno 28] No space left on device` mid-torch-download.
Fix: `export TMPDIR=/projectnb/devorlab/daria/tmp` + `PIP_NO_CACHE_DIR=1`
(/projectnb has PB free). **If you rebuild, keep that TMPDIR override.**

Install recipe (tensorflow-free — see note):

```bash
module load miniconda/24.5.0
ENV=/projectnb/devorlab/daria/envs/deepcad
conda create -y -p "$ENV" python=3.10
export TMPDIR=/projectnb/devorlab/daria/tmp; mkdir -p "$TMPDIR"
"$ENV/bin/pip" install --no-cache-dir torch==2.2.2 torchvision==0.17.2 \
    --index-url https://download.pytorch.org/whl/cu121
"$ENV/bin/pip" install --no-cache-dir tifffile scikit-image opencv-python-headless \
    matplotlib pyyaml scipy six tqdm packaging gdown onnx
"$ENV/bin/pip" install --no-cache-dir --no-deps csbdeep      # utils.normalize only; no TF
"$ENV/bin/pip" install --no-cache-dir --no-deps deepcad==1.2.0
"$ENV/bin/pip" install --no-cache-dir "numpy<2"              # MUST: torch 2.2.2 needs numpy<2
```

Installed versions (verified importable + torch↔numpy roundtrip OK, GPU forward + ONNX export OK):
`torch 2.2.2+cu121`, `torchvision 0.17.2+cu121`, `numpy 1.26.4`, `onnx 1.23.0`,
plus tifffile / scikit-image / opencv-python-headless / matplotlib / pyyaml / scipy / csbdeep / deepcad 1.2.0.

Activate for interactive use:
```bash
module load miniconda/24.5.0
conda activate /projectnb/devorlab/daria/envs/deepcad   # or just call $ENV/bin/python
```

### Why these specific choices (each is a real decision, not a default)
* **`deepcad==1.2.0`, NOT latest.** The PyPI **`deepcad 1.3.0` sdist is broken** —
  `top_level.txt` is empty and it ships **zero `.py` modules** (only packaging
  metadata), so `pip install deepcad` (latest) gives you the dependencies but no
  importable `deepcad`. 1.2.0 (and ≤1.2.0) ship the real 10-module package.
* **tensorflow avoided.** `deepcad`'s only `csbdeep` use is
  `from csbdeep.utils import normalize` (in `movie_display.py`). Verified that
  `csbdeep/__init__` and `csbdeep/utils/__init__` do **not** import tensorflow, so
  `csbdeep --no-deps` works and we skip a ~500 MB+ TF install entirely.
* **`numpy<2`.** torch 2.2.2 predates numpy 2 ABI; with numpy 2.2.6 you get
  `Failed to initialize NumPy: _ARRAY_API not found` and **all torch↔numpy
  conversion silently breaks** (DeepCAD does this constantly). Downgrading to
  numpy 1.26.4 fixed the roundtrip. `opencv-python-headless 5.0.0.93` and
  `ml-dtypes` *declare* `numpy>=2` (pip prints a conflict warning) but both
  import and run fine on 1.26.4 — verified empirically.
* **torch cu121.** Bundles its own CUDA 12.1 runtime, so we **do not** `module
  load cuda` (avoids libcudart clashes). cu121 covers sm_70…sm_90, i.e. every
  `gpu_c≥7.0` card including H200. Only the node's NVIDIA driver (610.57.04 on scc-505) is used.

---

## 3. Data & the volumetric→per-plane decision

Uploaded (rsync, sizes verified equal to local):

| file | shape (T,Z,Y,X) | dtype |
|---|---|---|
| `data/run7_clean.tif` | (1431, 21, 19, 336) | uint16 |
| `data/run4_clean.tif` | (1821, 21, 19, 264) | uint16 |

**DeepCAD-RT trains on temporal 3D stacks `(T, H, W)`** — its preprocessing reads
`shape[0]=T, shape[1]=Y, shape[2]=X`. Our data is 4D `(T,Z,Y,X)`. **Decision: split
each volume into its Z planes and denoise each plane's `(T,Y,X)` movie
independently** (DeepCAD's standard practice for volumetric data). The train/infer
qsubs do this reslice automatically into `data/planes/` — here **42 planes**
(21 per run), e.g. `run7_clean_z00.tif = (1431, 19, 336)`.

> **Implication (UNVERIFIED benefit/cost):** z-planes are denoised **independently**,
> so **axial (Z) correlations are not exploited**. With ~0.8 µm near-isotropic
> voxels there *is* real axial structure a 3D-over-volume model could use. Per-plane
> is the fast, robust, well-trodden starting point; a volumetric variant is future work.

---

## 4. The training job (`deepcad_train.qsub`)

Sizing for **smallest-sensible under the 2 h cap** (all env-overridable via `qsub -v`):

| param | value | note |
|---|---|---|
| `N_EPOCHS` | 10 | |
| `PATCH_T` | 128 | network sees 2·128=256 frames/sample |
| `PATCH_X` | 160 | ≤ min X (run4=264) |
| `PATCH_Y` | 16 | Y=19 thin slab; 16 ≤ 19, ÷8 clean |
| `TRAIN_SIZE` | 1000 | ≈ patches/epoch (actual: 1008) |
| `SELECT_IMG` | 1000 | first 1000 frames/plane used |
| `FMAP` | 16 | DeepCAD memory-minimal default |
| `LR` | 1e-4 | raised from the 1e-5 class default so a short run actually learns |

**Estimate vs actual:** predicted ~10 080 iters ⇒ tens of minutes on a V100; the job
actually finished in **118 s wallclock** on an L40S (much faster card), peak vmem
11.2 G. Comfortably inside the 2 h cap — you can safely scale up epochs/frames later.

Outputs (verified present):
* `models/planes_202609191434/E_01…E_10_Iter_1008.pth` + `para.yaml`
* `models/onnx/planes_202609191434/E_10_Iter_1008_Patch_160_16_128.onnx` (+ E_01…E_09)
  — the ONNX files are the on-ramp to DeepCAD-RT's TensorRT **real-time** path for
  the "cells firing" visualization you eventually want.

Submit again / retrain (example: more epochs, all frames):
```bash
cd /projectnb/devorlab/daria/Femtonics/deepcad
qsub -v N_EPOCHS=30,SELECT_IMG=1431 deepcad_train.qsub
```

---

## 5. Monitor a job

```bash
qstat -u daria                 # state: qw=queued, r=running
qstat -j <JOBID>               # resources, exec host
qacct -j <JOBID> | grep -E 'exit_status|failed|ru_wallclock|maxvmem'   # after it finishes
tail -f /projectnb/devorlab/daria/Femtonics/deepcad/logs/deepcad_train_<JOBID>.log
```
Progress looks like `[Epoch 10/10] [Batch 1008/1008] [Total loss: …]` then
`Training finished. All models saved to disk.` and `exit=0`.

---

## 6. Run inference when you want denoised stacks (`deepcad_infer.qsub`)

Uses the **last** `.pth` in the chosen model folder, denoises each z-plane, and
**reassembles the planes back into a 4D `(T,Z,Y,X)` tiff**.

```bash
cd /projectnb/devorlab/daria/Femtonics/deepcad
# default: denoise the two stacks already in data/
qsub -v DENOISE_MODEL=planes_202609191434 deepcad_infer.qsub
# or a specific 4D stack (or a directory of them):
qsub -v DENOISE_MODEL=planes_202609191434,INPUT_TIF=/projectnb/devorlab/daria/…/foo_clean.tif deepcad_infer.qsub
```
Result: `results/<run>_denoised.tif` (4D, same dtype/shape as input). `TEST_DATASIZE`
defaults to 2048 so **all** frames are denoised (DeepCAD's default 400 would truncate).
Patch params default to the training values and should match them.

Then point the downstream mask / NMF / skeleton steps at `*_denoised.tif` instead of
`*_clean.tif` and compare mask recall / NMF block separation at the higher SNR.

---

## 7. UNVERIFIED assumptions & caveats (read before trusting output)

1. **Denoising *quality* is NOT benchmarked.** The run proves the pipeline trains
   end-to-end and emits a model with finite, non-diverging loss. It does **not**
   prove SNR actually improved on your data. Validate by running inference and
   comparing (e.g. mask recall vs the 92 % hand-trace baseline, NMF subtree blocks)
   on held-out runs before relying on it downstream.
2. **`LR=1e-4` and 10 epochs / 1000 frames are minimal, untuned.** A larger run
   (`N_EPOCHS=20–30`, `SELECT_IMG=full`, maybe lower LR) will likely denoise better.
3. **Per-plane independence** discards axial (Z) correlation (see §3).
4. **`PATCH_Y=16 < Y=19` in *training*** ⇒ rows 16–18 are never inside a training
   patch (only 1 Y position fits). The conv net still applies to them, and
   *inference* covers the full Y via DeepCAD's edge-snapped last patch — but the top
   3 rows were effectively unsupervised.
5. **GPU selection relies on init order.** DeepCAD hard-sets
   `CUDA_VISIBLE_DEVICES='0'`; SCC does **not** cgroup-isolate GPUs (all 4 are
   visible; only `CUDA_VISIBLE_DEVICES` is set per job). The driver initializes CUDA
   on the SGE-allocated GPU *before* DeepCAD's override (our `[gpu]` check touches
   `torch.cuda` first), so the override is a no-op and the correct GPU is used —
   confirmed on job 7648318 (device reported = L40S, exit 0). If a future DeepCAD
   version reorders this, pin the device explicitly.
6. **pip metadata conflicts** (opencv 5.0.0.93, ml-dtypes want numpy≥2) are cosmetic
   here — runtime verified on numpy 1.26.4 — but a naive `pip install --upgrade`
   could resurrect the numpy-2 break. Keep `numpy<2` pinned.
7. **Inference reassembly** assumes every plane output has equal T and a parseable
   `_z##_` index in its filename (both hold for the reslice this pipeline produces).
   The reassembly path itself has **not yet been executed** on real inference output
   — run §6 once and eyeball a `*_denoised.tif` before batch-trusting it.
8. **Env built on the login node.** Compute-node outbound internet was not tested;
   if you ever must build in a batch job, verify pip can reach PyPI from the node.
9. **`gpu_c` "≥" semantics** inferred from the job landing on an L40S (cc 8.9) when
   asked for 7.0; consistent, but the exact SGE relation wasn't independently confirmed.

---

## 8. File map

```
code/STEP2b_denoise/                       (LOCAL — the 3 deliverables)
  deepcad_train.qsub   deepcad_infer.qsub   README.md

/projectnb/devorlab/daria/                 (CLUSTER — daria's tree only)
  envs/deepcad/                            conda env (py3.10, torch 2.2.2+cu121)
  Femtonics/deepcad/
    build_env.sh  build.log  build2.log    env build provenance
    deepcad_train.qsub  deepcad_infer.qsub uploaded copies (what qsub runs)
    data/  run7_clean.tif  run4_clean.tif  uploaded 4D stacks
    data/planes/  *_z##.tif                42 per-plane 3D stacks (auto-generated)
    models/planes_202609191434/  *.pth  para.yaml
    models/onnx/planes_202609191434/  *.onnx
    results/                               inference output lands here
    logs/  deepcad_train_<JOBID>.log       job stdout+stderr
```
