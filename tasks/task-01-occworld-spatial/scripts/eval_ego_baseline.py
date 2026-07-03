#!/usr/bin/env python
"""실데이터 ego-L2 기준선 평가 — nuScenes v1.0-mini (실측, 더미 아님).

프로토콜: 과거 2s(4 keyframe @2Hz) → 미래 3s(6 keyframe) 슬라이딩 윈도우.
기준선: copy-last(persistence) / linear-extrapolation. ego 위치는 global translation.
지표: ego L2 (per-step, 그리고 1s/2s/3s 까지 누적평균). split(train/val) 별 집계.

⚠️ 점유(mIoU/IoU/충돌률)는 Occ3D-nuScenes gts 가 있어야 계산 가능 → 여기서는 계산하지 않고
   'blocked_on_occ3d' 로 명시(값을 지어내지 않음).

결과는 results/<run_id>/metrics.json + summary.md 로 저장(재현성: config·git·하드웨어 기록).
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
import baselines as B                         # noqa: E402
import nusc_infos as NI                       # noqa: E402


def _git_hash() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=str(_REPO_ROOT)
        ).decode().strip()
    except Exception:
        return "unknown"


def _load_yaml(path: Path) -> dict:
    import yaml
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def eval_split(scenes: dict, split: str, hist: int, fc: int):
    """한 split 의 모든 윈도우에 대해 두 기준선의 ego-L2 집계."""
    import numpy as np
    win = hist + fc
    per_step = {"copy_last": [], "linear_extrapolation": []}
    n_windows = 0
    used_scenes = []
    for name, s in sorted(scenes.items()):
        if s["split"] != split:
            continue
        xy = np.asarray(s["ego_xy"], dtype=np.float64)
        if xy.shape[0] < win:
            continue
        used_scenes.append(name)
        for start in range(0, xy.shape[0] - win + 1):
            history = xy[start:start + hist]
            gt = xy[start + hist:start + win]
            pred_c = B.copy_last_trajectory(history, fc)
            pred_l = B.linear_extrapolation_trajectory(history, fc)
            per_step["copy_last"].append(M.ego_l2(pred_c, gt)["per_step"])
            per_step["linear_extrapolation"].append(M.ego_l2(pred_l, gt)["per_step"])
            n_windows += 1

    def summarize(rows):
        if not rows:
            return None
        arr = np.asarray(rows, dtype=np.float64)  # (W, fc)
        mean_ps = arr.mean(axis=0)                # 지평별 평균
        # 누적평균 @1s/2s/3s (2Hz → 2/4/6 스텝)
        cum = {}
        for sec, k in [("1s", 2), ("2s", 4), ("3s", 6)]:
            if k <= fc:
                cum[sec] = float(mean_ps[:k].mean())
        return {
            "per_step_mean": [float(x) for x in mean_ps],
            "avg_up_to": cum,
            "overall_mean": float(mean_ps.mean()),
        }

    return {
        "split": split,
        "num_scenes": len(used_scenes),
        "scenes": used_scenes,
        "num_windows": n_windows,
        "copy_last": summarize(per_step["copy_last"]),
        "linear_extrapolation": summarize(per_step["linear_extrapolation"]),
    }


def main(argv=None) -> int:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except (AttributeError, ValueError):
            pass

    p = argparse.ArgumentParser(description="실 ego-L2 기준선 평가 (nuScenes mini)")
    p.add_argument("--config", type=Path, default=_TASK_DIR / "config" / "base.yaml")
    p.add_argument("--nusc-root", type=Path, default=_TASK_DIR / "data" / "nuscenes")
    p.add_argument("--version", default="v1.0-mini")
    p.add_argument("--out-root", type=Path, default=_TASK_DIR / "results")
    p.add_argument("--run-id", default=None)
    args = p.parse_args(argv)

    cfg = _load_yaml(args.config)
    seed = set_seed(int(cfg.get("seed", 42)))
    hz = int(cfg["temporal"]["frame_hz"])
    hist = int(cfg["temporal"]["history_seconds"] * hz)   # 2s*2 = 4
    fc = int(cfg["temporal"]["forecast_seconds"] * hz)    # 3s*2 = 6

    infos = NI.build_scene_trajectories(args.nusc_root, args.version)
    scenes = infos["scenes"]

    results = {s: eval_split(scenes, s, hist, fc) for s in ("val", "train")}

    now = datetime.now(timezone.utc)
    run_id = args.run_id or ("ego-l2-mini-" + now.strftime("%Y%m%dT%H%M%SZ"))
    out_dir = args.out_root / run_id

    metrics = {
        "real": True,
        "task": "task-01-occworld-spatial",
        "eval": "ego_l2_baselines",
        "note": "실 nuScenes v1.0-mini ego 궤적 기반. 점유 지표는 Occ3D gts 부재로 미계산.",
        "protocol": {
            "history_frames": hist, "forecast_frames": fc, "frame_hz": hz,
            "history_seconds": cfg["temporal"]["history_seconds"],
            "forecast_seconds": cfg["temporal"]["forecast_seconds"],
        },
        "occupancy_metrics": "blocked_on_occ3d",
        "reproducibility": {
            "seed": seed,
            "git_commit": _git_hash(),
            "timestamp_utc": now.isoformat(),
            "hardware": {
                "platform": platform.platform(),
                "processor": platform.processor(),
                "python": platform.python_version(),
                "gpu": "N/A (기준선은 CPU만 사용)",
            },
            "dataset": {"version": args.version, "counts": infos["counts"]},
            "config_used": {"temporal": cfg["temporal"], "baselines": cfg["baselines"]},
        },
        "results": results,
    }
    out_path = M.save_metrics(metrics, out_dir / "metrics.json")

    # 사람이 읽는 요약(.md 는 커밋 허용)
    def fmt(split_res, model):
        r = split_res[model]
        if not r:
            return f"| {model} | (윈도우 없음) | | | |"
        a = r["avg_up_to"]
        return (f"| {model} | {a.get('1s', float('nan')):.3f} | {a.get('2s', float('nan')):.3f} "
                f"| {a.get('3s', float('nan')):.3f} | {r['overall_mean']:.3f} |")

    lines = [
        f"# ego-L2 기준선 (실 nuScenes v1.0-mini) — {run_id}",
        "",
        f"- 프로토콜: 과거 {cfg['temporal']['history_seconds']}s({hist}f) → 미래 "
        f"{cfg['temporal']['forecast_seconds']}s({fc}f) @ {hz}Hz, 슬라이딩 윈도우",
        f"- git: `{metrics['reproducibility']['git_commit'][:10]}` · "
        f"seed {seed} · {platform.platform()}",
        "- ⚠️ 점유 지표(mIoU/IoU/충돌률)는 Occ3D gts 부재로 **미계산**(blocked_on_occ3d).",
        "",
    ]
    for split in ("val", "train"):
        sr = results[split]
        lines += [
            f"## {split} (씬 {sr['num_scenes']}, 윈도우 {sr['num_windows']})",
            "",
            "| 기준선 | L2@1s | L2@2s | L2@3s | 전체평균 | (m) |",
            "|---|---|---|---|---|---|",
            fmt(sr, "copy_last") + " |",
            fmt(sr, "linear_extrapolation") + " |",
            "",
        ]
    (out_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")

    # 콘솔 요약
    print(f"[eval] run_id={run_id}  → {out_path}")
    for split in ("val", "train"):
        sr = results[split]
        print(f"  [{split}] 씬 {sr['num_scenes']} 윈도우 {sr['num_windows']}")
        for model in ("copy_last", "linear_extrapolation"):
            r = sr[model]
            if r:
                a = r["avg_up_to"]
                print(f"    {model:22s} L2@1s={a.get('1s', float('nan')):.3f} "
                      f"@2s={a.get('2s', float('nan')):.3f} @3s={a.get('3s', float('nan')):.3f} "
                      f"overall={r['overall_mean']:.3f} m")
    print("  점유 지표: blocked_on_occ3d (Occ3D-nuScenes gts 취득 후 계산)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
