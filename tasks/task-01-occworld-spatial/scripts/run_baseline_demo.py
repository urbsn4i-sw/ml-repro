#!/usr/bin/env python
"""smoke용 더미 파이프라인: 기준선 → 지표 계산이 끝까지 도는지 검증.

⚠️ 이 스크립트는 **합성(dummy) 텐서**만 사용한다. 실제 nuScenes/Occ3D 데이터도,
   OccWorld 가중치도 필요/사용하지 않는다. 출력 수치는 파이프라인 동작 확인용이며
   **성능 보고가 아니다**(metrics.json 의 synthetic_dummy=true 로 표시).

하는 일
  1) set_seed 로 시드 고정
  2) 합성 점유 시퀀스·ego 궤적 생성(지평이 길수록 gt 가 더 달라지게 하여 지표가 반응하는지 확인)
  3) copy-last / linear-extrapolation 기준선 예측
  4) miou_per_horizon · ego_l2 · collision_rate · rollout_divergence 계산
  5) metrics.json 저장(--out) + 요약 출력
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_HERE = Path(__file__).resolve()
_TASK_DIR = _HERE.parents[1]
_REPO_ROOT = _HERE.parents[3]
sys.path.insert(0, str(_REPO_ROOT))          # common.*
sys.path.insert(0, str(_TASK_DIR / "src"))   # baselines

from common.seeding import set_seed          # noqa: E402
from common import metrics as M               # noqa: E402
import baselines as B                         # noqa: E402


def build_dummy(seed: int, t_in: int, horizon: int, grid, num_classes: int):
    """합성 점유 시퀀스·궤적 생성. 지평↑ → gt 변형↑ (지표 반응 확인용)."""
    import numpy as np
    rng = np.random.default_rng(seed)

    # --- 점유: 관측 history + '진짜 미래'(합성) ---
    occ_hist = rng.integers(0, num_classes, size=(t_in, *grid))
    last = occ_hist[-1]
    gt_future = np.empty((horizon, *grid), dtype=last.dtype)
    n_vox = int(np.prod(grid))
    for t in range(horizon):
        frame = last.copy()
        # 지평이 길수록 더 많은 복셀을 무작위로 바꿔 '멀어지는 미래'를 모사
        frac = min(1.0, 0.15 * (t + 1))
        n_change = int(frac * n_vox)
        if n_change:
            flat = frame.reshape(-1)
            idx = rng.choice(n_vox, size=n_change, replace=False)
            flat[idx] = rng.integers(0, num_classes, size=n_change)
        gt_future[t] = frame

    # --- ego 궤적: 등속 history + 곡선형 '진짜 미래' ---
    v = np.array([1.0, 0.5])
    p0 = np.array([0.0, 0.0])
    hist_xy = np.stack([p0 + i * v for i in range(t_in)], axis=0)
    last_xy = hist_xy[-1]
    gt_xy = np.stack(
        [last_xy + (k + 1) * v + np.array([0.0, 0.3 * (k + 1) ** 2]) for k in range(horizon)],
        axis=0,
    )  # 2차 항으로 곡선 → 등속 외삽 오차가 지평따라 증가
    return occ_hist, gt_future, hist_xy, gt_xy


def to_grid_cells(traj_xy, grid):
    """연속 좌표를 정수 복셀 인덱스로(충돌률 데모용). z=0 평면 가정."""
    import numpy as np
    xy = np.asarray(traj_xy)
    cells = np.rint(xy).astype(int)
    ndim = len(grid)
    if ndim > 2:  # z 축(그리고 그 이상)은 0 으로 채움
        pad = np.zeros((cells.shape[0], ndim - 2), dtype=int)
        cells = np.concatenate([cells, pad], axis=1)
    return np.clip(cells, 0, np.array(grid) - 1)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="기준선→지표 더미 파이프라인(smoke)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--t-in", type=int, default=3)
    p.add_argument("--horizon", type=int, default=4)
    p.add_argument("--grid", type=int, nargs=3, default=[6, 6, 2], metavar=("X", "Y", "Z"))
    p.add_argument("--num-classes", type=int, default=4)
    p.add_argument("--out", type=Path, required=True, help="metrics.json 저장 디렉토리")
    args = p.parse_args(argv)

    # Windows 콘솔(cp949 등)에서도 한글/기호 출력이 깨지지 않게 UTF-8 강제.
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")  # py3.7+
        except (AttributeError, ValueError):
            pass

    seed = set_seed(args.seed)
    grid = tuple(args.grid)

    occ_hist, gt_future, hist_xy, gt_xy = build_dummy(
        seed, args.t_in, args.horizon, grid, args.num_classes
    )

    # --- 기준선 예측 ---
    pred_occ = B.copy_last_occupancy(occ_hist, args.horizon)
    pred_xy_copy = B.copy_last_trajectory(hist_xy, args.horizon)
    pred_xy_lin = B.linear_extrapolation_trajectory(hist_xy, args.horizon)

    # --- 지표 계산 (모두 합성 텐서에 대한 실제 계산값) ---
    occ_miou = M.miou_per_horizon(
        pred_occ, gt_future, args.num_classes, exclude=[0]  # class 0 = free 제외
    )
    l2_copy = M.ego_l2(pred_xy_copy, gt_xy)
    l2_lin = M.ego_l2(pred_xy_lin, gt_xy)
    ego_cells = to_grid_cells(pred_xy_lin, grid)
    collision = M.collision_rate(ego_cells, pred_occ, free_index=0)
    # 롤아웃 발산: (a) 선형 기준선 지평별 L2, (b) 1 - 지평별 mIoU
    div_l2 = M.rollout_divergence(l2_lin["per_step"])
    div_occ = M.rollout_divergence([1.0 - x for x in occ_miou["per_horizon"]])

    metrics = {
        "synthetic_dummy": True,
        "note": "합성 텐서 기반 파이프라인 검증. 성능 보고 아님(실데이터/가중치 미사용).",
        "config": {
            "seed": seed,
            "t_in": args.t_in,
            "horizon": args.horizon,
            "grid": list(grid),
            "num_classes": args.num_classes,
        },
        "occupancy_miou": occ_miou,
        "ego_l2": {"copy_last": l2_copy, "linear_extrapolation": l2_lin},
        "collision_rate": collision,
        "rollout_divergence": {"ego_l2_linear": div_l2, "occ_1_minus_miou": div_occ},
    }

    out_path = M.save_metrics(metrics, Path(args.out) / "metrics.json")

    # --- 요약 출력 ---
    print("[SMOKE/DUMMY] 합성 텐서 파이프라인 — 성능 보고 아님")
    print(f"  seed={seed} t_in={args.t_in} horizon={args.horizon} grid={grid} classes={args.num_classes}")
    print(f"  occ mIoU(지평별) = {[round(x, 3) for x in occ_miou['per_horizon']]}  mean={occ_miou['mean']:.3f}")
    print(f"  ego L2 copy-last   누적평균 = {[round(x, 3) for x in l2_copy['cumulative_mean']]}")
    print(f"  ego L2 linear-ext  누적평균 = {[round(x, 3) for x in l2_lin['cumulative_mean']]}")
    print(f"  collision rate = {collision['rate']}  (evaluable={collision['evaluable']}, oob={collision['out_of_bounds']})")
    print(f"  rollout divergence(L2): slope={div_l2['slope']:.3f}  monotonic_frac={div_l2['monotonic_frac']}")
    print(f"  → metrics.json 저장: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
