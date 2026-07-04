# ml-repro — ML Reproduction Monorepo

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> 🇰🇷 **[한국어 README](README.ko.md)**

Reproducing recent ML breakthroughs (DreamerV3, OccWorld, …) at small scale on a gaming PC (**RTX 3060, 8 GB**) or Colab, redesigned toward SLAM / urban / spatial-intelligence tasks. The goal is **learning the principles, not matching SOTA**.

## Purpose
Take recent ML/DL/RL breakthroughs (e.g. **DreamerV3**, **OccWorld**) and reproduce their **core principles** at reduced scale on consumer hardware or Colab, redesigned toward SLAM / urban / spatial-intelligence problems. This is a study / reproduction repo — **not** an attempt to match state-of-the-art numbers.

## Core principle — do not fabricate results
Only values from actual runs (`metrics.json`) are reported. Anything unverified or not run is stated explicitly as a **limitation**, never presented as if achieved. Datasets, labels, and checkpoints are **never committed** (blocked by `.gitignore`).

## Status (honest)
| Task | Topic | Status |
|---|---|---|
| **01 — OccWorld / DreamerV3** | 3D occupancy world model | Phase 0·1 **done** (baselines measured) · Phase 2 (OccWorld inference) **blocked (external)** · Phase 3 **published** |
| 02 — STGNN | Urban traffic spatio-temporal forecasting | **Planned** (not started) |
| 03 — Cross-Embodiment Nav | Cross-embodiment navigation | **Planned** (not started) |
| 04 — PointMamba / SLAM | Mamba-based LiDAR / point cloud | **Planned** (not started) |

## Task 01 — measured results (real, baselines only)
Real nuScenes **v1.0-mini**, protocol past 2s → future 3s @ 2 Hz, CPU. **Validation split = 2 scenes / 63 windows** (⚠️ small sample — do not over-generalize).

**Occupancy — copy-last baseline (= paper "Copy&Paste" definition), real Occ3D gts** (camera-mask applied, free class 17 excluded):

| Metric (%) | @1s | @2s | @3s |
|---|---|---|---|
| future mIoU | 10.75 | 6.40 | 5.10 |
| binary IoU | 27.21 | 21.11 | 18.35 |

**Ego trajectory — L2 (m), cumulative average:**

| Model | L2@1s | L2@2s | L2@3s |
|---|---|---|---|
| copy-last (persistence) | 3.89 | 6.46 | 9.00 |
| linear-extrapolation (constant velocity) | 0.45 | 1.10 | 2.01 |
| OccWorld (pretrained inference) | — | — | — (blocked) |

> These are **our** measurements, not paper numbers. Constant-velocity extrapolation clearly beats persistence for a moving ego vehicle, as expected.

## Phase 2 — why OccWorld inference is blocked (external factor)
OccWorld's pretrained weights and temporal pkl are distributed only via the authors' **Tsinghua cloud (Seafile) links**, which are **currently inactive** ("share link not found" for both browser and CLI; a real token behaves identically to a bogus one). We cannot fabricate the model's numbers, so the **OccWorld-vs-baseline row is left blank**. See [`ISSUE_phase2_blocked.md`](tasks/task-01-occworld-spatial/ISSUE_phase2_blocked.md). Work resumes if the links are restored or a mirror is found.

## Reproduce (Task 01 baselines)
```bash
# 1) data acquisition guide (manual; nuScenes/Occ3D need registration — no auto-download)
bash tasks/task-01-occworld-spatial/scripts/download_data.sh --subset mini
# 2) build mini temporal infos (stdlib only) + baseline evals (real gts)
python tasks/task-01-occworld-spatial/scripts/build_mini_infos.py
python tasks/task-01-occworld-spatial/scripts/eval_ego_baseline.py
python tasks/task-01-occworld-spatial/scripts/eval_occ_baseline.py
```
Details: [`PROJECT_GUIDELINE.md`](PROJECT_GUIDELINE.md), task card [`tasks/task-01-occworld-spatial/README.md`](tasks/task-01-occworld-spatial/README.md).

## Papers
- **Anchor:** Hafner et al. (2025). *Mastering diverse control tasks through world models.* **Nature 640, 647–653.** DOI [10.1038/s41586-025-08744-2](https://doi.org/10.1038/s41586-025-08744-2). (DreamerV3)
- **Bridge:** Zheng et al. (2024). *OccWorld.* **ECCV 2024.** arXiv [2311.16038](https://arxiv.org/abs/2311.16038).

## Data sources & license
- **Code (this repo):** **MIT License** — see [`LICENSE`](LICENSE). Applies to our own scripts, metrics, and configs.
- **Data is NOT covered by MIT and is NOT included here:**
  - **nuScenes** — **non-commercial (CC BY-NC-SA 4.0)**, registration required, **redistribution prohibited**.
  - **Occ3D-nuScenes** — bound by the nuScenes Terms of Use (non-commercial).
  - You must **register and obtain the data yourself** under its original license. This repository contains **no raw data, labels, or model weights** (blocked by `.gitignore`).

## Contact / Author
- **Author:** urbsn4i-sw (GitHub)
- **Email:** urban4i.sw@gmail.com
- Questions / reproduction issues via GitHub Issues or the email above. Non-commercial study / reproduction repository.
