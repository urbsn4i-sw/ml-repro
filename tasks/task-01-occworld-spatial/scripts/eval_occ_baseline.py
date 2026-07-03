#!/usr/bin/env python
"""실 Occ3D-nuScenes gts 기반 점유 기준선 평가 — copy-last (실측, 더미 아님).

프로토콜: 과거 2s(4 keyframe) → 미래 3s(6 keyframe) 슬라이딩 윈도우.
기준선: copy-last(persistence) = 마지막 관측 점유 프레임을 미래로 복사 (= 논문 Copy&Paste 정의).
지표: 미래 mIoU(semantic, free=17 제외) · 이진 점유 IoU. Occ3D 공식 규약대로 mask_camera=1 복셀만.

집계 방식(논문 비교 가능): 지평별로 '전역 confusion matrix'를 모든 윈도우에 누적한 뒤
IoU 를 한 번 계산(프레임별 mIoU 평균이 아니라 데이터셋 누적 방식).

⚠️ 논문 Copy&Paste 수치(mIoU 1s=14.91 등)는 metrics.json 에 reference_only 로만 기입하고
   우리 결과로 쓰지 않는다(재현성 규칙: 값을 지어내지 않음).
"""
from __future__ import annotations

import argparse
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

_HERE = Path(__file__).resolve()
_TASK_DIR = _HERE.parents[1]
_REPO_ROOT = _HERE.parents[3]
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_TASK_DIR / "src"))

from common.seeding import set_seed          # noqa: E402
from common import metrics as M               # noqa: E402
import nusc_infos as NI                       # noqa: E402

# 논문(OccWorld, Zheng+ 2024) Copy&Paste 기준선 — 참조용(우리 결과 아님).
PAPER_COPY_PASTE_REFERENCE = {
    "source": "OccWorld (Zheng+ 2024, arXiv:2311.16038) Table — Copy&Paste baseline",
    "mIoU_percent": {"1s": 14.91, "2s": 10.54, "3s": 8.11},
    "IoU_percent": {"1s": 24.47, "2s": 19.77, "3s": 17.14},
    "note": "reference_only — 우리 실측값과 별개. 데이터 subset(mini)·평가범위 차이로 직접 비교 불가.",
}


def _git_hash() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(_REPO_ROOT)).decode().strip()
    except Exception:
        return "unknown"


def _load_yaml(path: Path):
    import yaml
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def load_scene_frames(gts_dir: Path, scene: str, tokens, use_mask: bool):
    """씬의 keyframe 순서대로 (semantics, mask) 배열을 로드. 누락 프레임은 None."""
    import numpy as np
    sem_list, mask_list = [], []
    for tok in tokens:
        npz = gts_dir / scene / tok / "labels.npz"
        if not npz.is_file():
            return None, None, tok  # 누락 → 이 씬 스킵 신호
        d = np.load(npz)
        sem_list.append(d["semantics"])
        mask_list.append(d["mask_camera"].astype(bool) if use_mask else None)
    return sem_list, mask_list, None


def eval_split(scenes_info, gts_dir, split, hist, fc, num_classes, free_index, use_mask):
    """copy-last 점유 기준선의 지평별 mIoU/IoU (전역 confusion 누적)."""
    import numpy as np
    win = hist + fc
    C = num_classes
    cm = [np.zeros((C, C), dtype=np.int64) for _ in range(fc)]     # 지평별 confusion
    occ_inter = np.zeros(fc, dtype=np.int64)                        # 이진 IoU 누적
    occ_union = np.zeros(fc, dtype=np.int64)
    n_windows = 0
    used_scenes, skipped = [], []

    for name, s in sorted(scenes_info["scenes"].items()):
        if s["split"] != split:
            continue
        tokens = s["sample_tokens"]
        sem, mask, missing = load_scene_frames(gts_dir, name, tokens, use_mask)
        if sem is None:
            skipped.append((name, missing))
            continue
        if len(sem) < win:
            continue
        used_scenes.append(name)
        for start in range(0, len(sem) - win + 1):
            last_obs = sem[start + hist - 1]           # copy-last: 마지막 관측 프레임
            pred = last_obs.reshape(-1).astype(np.int64)
            for j in range(fc):
                gt = sem[start + hist + j].reshape(-1).astype(np.int64)
                if use_mask:
                    mk = mask[start + hist + j].reshape(-1)
                    p, g = pred[mk], gt[mk]
                else:
                    p, g = pred, gt
                if g.size == 0:
                    continue
                cm[j] += np.bincount(g * C + p, minlength=C * C).reshape(C, C)
                po, go = p != free_index, g != free_index
                occ_inter[j] += int(np.count_nonzero(po & go))
                occ_union[j] += int(np.count_nonzero(po | go))
            n_windows += 1

    # 지평별 IoU 산출
    def miou_from_cm(m):
        inter = np.diag(m).astype(np.float64)
        union = m.sum(1) + m.sum(0) - inter
        iou = np.full(C, np.nan)
        nz = union > 0
        iou[nz] = inter[nz] / union[nz]
        iou[free_index] = np.nan  # free 제외
        return float(np.nanmean(iou)) if np.any(~np.isnan(iou)) else float("nan"), iou

    miou_ph, iou_ph = [], []
    for j in range(fc):
        mi, _ = miou_from_cm(cm[j])
        miou_ph.append(mi)
        iou_ph.append(float(occ_inter[j] / occ_union[j]) if occ_union[j] else float("nan"))

    # @1s/2s/3s = 그 시각의 프레임(2Hz → step 2/4/6 = index 1/3/5)
    def at(seq):
        out = {}
        for sec, idx in [("1s", 1), ("2s", 3), ("3s", 5)]:
            if idx < fc:
                out[sec] = seq[idx]
        return out

    return {
        "split": split,
        "num_scenes": len(used_scenes),
        "scenes": used_scenes,
        "skipped_scenes": skipped,
        "num_windows": n_windows,
        "mIoU": {"per_horizon": miou_ph, "at": at(miou_ph),
                 "mean": float(np.nanmean(miou_ph)) if n_windows else float("nan")},
        "IoU": {"per_horizon": iou_ph, "at": at(iou_ph),
                "mean": float(np.nanmean(iou_ph)) if n_windows else float("nan")},
    }


def main(argv=None) -> int:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except (AttributeError, ValueError):
            pass

    p = argparse.ArgumentParser(description="실 gts 점유 기준선(copy-last) 평가")
    p.add_argument("--config", type=Path, default=_TASK_DIR / "config" / "base.yaml")
    p.add_argument("--nusc-root", type=Path, default=_TASK_DIR / "data" / "nuscenes")
    p.add_argument("--gts-dir", type=Path, default=_TASK_DIR / "data" / "nuscenes" / "gts")
    p.add_argument("--version", default="v1.0-mini")
    p.add_argument("--out-root", type=Path, default=_TASK_DIR / "results")
    p.add_argument("--run-id", default=None)
    args = p.parse_args(argv)

    if not args.gts_dir.is_dir():
        sys.exit(f"[occ] 오류: gts 디렉토리 없음: {args.gts_dir}")

    cfg = _load_yaml(args.config)
    seed = set_seed(int(cfg.get("seed", 42)))
    hz = int(cfg["temporal"]["frame_hz"])
    hist = int(cfg["temporal"]["history_seconds"] * hz)
    fc = int(cfg["temporal"]["forecast_seconds"] * hz)
    num_classes = int(cfg["metrics"]["num_classes"])
    free_index = int(cfg["metrics"]["free_class_index"])
    use_mask = bool(cfg["metrics"].get("use_camera_mask", True))

    infos = NI.build_scene_trajectories(args.nusc_root, args.version)
    results = {s: eval_split(infos, args.gts_dir, s, hist, fc, num_classes, free_index, use_mask)
               for s in ("val", "train")}

    now = datetime.now(timezone.utc)
    run_id = args.run_id or ("occ-copylast-mini-" + now.strftime("%Y%m%dT%H%M%SZ"))
    out_dir = args.out_root / run_id

    metrics = {
        "real": True,
        "task": "task-01-occworld-spatial",
        "eval": "occupancy_copy_last_baseline",
        "note": "실 Occ3D-nuScenes gts(mini 10씬). copy-last=논문 Copy&Paste 정의.",
        "protocol": {
            "history_frames": hist, "forecast_frames": fc, "frame_hz": hz,
            "num_classes": num_classes, "free_index": free_index,
            "camera_mask_applied": use_mask,
            "miou_class_set": "0..16 (free=17 제외; 'others'=0 포함)",
            "aggregation": "지평별 전역 confusion 누적 후 IoU (프레임 평균 아님)",
            "at_time_convention": "mIoU/IoU @Xs = 해당 시각 프레임(2Hz step); ego-L2 는 누적평균(별도)",
        },
        "reproducibility": {
            "seed": seed, "git_commit": _git_hash(), "timestamp_utc": now.isoformat(),
            "hardware": {"platform": platform.platform(), "python": platform.python_version(),
                         "gpu": "N/A (기준선은 CPU만)"},
            "dataset": {"version": args.version, "counts": infos["counts"]},
        },
        "annotations_json_split_crosscheck": "미확인/한계 (annotations.json 미취득; devkit 공식 mini split 사용)",
        "paper_reference_only": PAPER_COPY_PASTE_REFERENCE,
        "results": results,
    }
    out_path = M.save_metrics(metrics, out_dir / "metrics.json")

    # 요약 md (커밋 허용)
    def pct(x):
        return "nan" if x != x else f"{100*x:.2f}"
    lines = [f"# 점유 기준선 copy-last (실 Occ3D-nuScenes mini) — {run_id}", "",
             f"- 프로토콜: 과거 {hist}f→미래 {fc}f @{hz}Hz, mask_camera={use_mask}, free={free_index} 제외",
             f"- 집계: 지평별 전역 confusion 누적. git `{metrics['reproducibility']['git_commit'][:10]}` seed {seed}",
             "- ⚠️ 논문 Copy&Paste 값은 reference_only(우리 결과 아님). annotations.json 교차확인은 미확인/한계.", ""]
    for split in ("val", "train"):
        r = results[split]
        lines += [f"## {split} (씬 {r['num_scenes']}, 윈도우 {r['num_windows']})", "",
                  "| 지표(%) | @1s | @2s | @3s | 6지평평균 |", "|---|---|---|---|---|",
                  f"| mIoU | {pct(r['mIoU']['at'].get('1s',float('nan')))} | {pct(r['mIoU']['at'].get('2s',float('nan')))} "
                  f"| {pct(r['mIoU']['at'].get('3s',float('nan')))} | {pct(r['mIoU']['mean'])} |",
                  f"| IoU  | {pct(r['IoU']['at'].get('1s',float('nan')))} | {pct(r['IoU']['at'].get('2s',float('nan')))} "
                  f"| {pct(r['IoU']['at'].get('3s',float('nan')))} | {pct(r['IoU']['mean'])} |", ""]
    (out_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")

    print(f"[occ] run_id={run_id} → {out_path}")
    for split in ("val", "train"):
        r = results[split]
        print(f"  [{split}] 씬 {r['num_scenes']} 윈도우 {r['num_windows']}"
              + (f" (스킵 {r['skipped_scenes']})" if r["skipped_scenes"] else ""))
        mi, io = r["mIoU"]["at"], r["IoU"]["at"]
        print(f"    mIoU%  @1s={pct(mi.get('1s',float('nan')))} @2s={pct(mi.get('2s',float('nan')))} "
              f"@3s={pct(mi.get('3s',float('nan')))} mean={pct(r['mIoU']['mean'])}")
        print(f"    IoU%   @1s={pct(io.get('1s',float('nan')))} @2s={pct(io.get('2s',float('nan')))} "
              f"@3s={pct(io.get('3s',float('nan')))} mean={pct(r['IoU']['mean'])}")
    print("  (논문 Copy&Paste 값은 reference_only, 비교 불가)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
