"""Task 01 (OccWorld) 표준 지표 구현.

과제별 표준 지표(PROJECT_GUIDELINE.md §8): **미래 mIoU · ego L2 · 충돌률 · 롤아웃 안정성**.

설계 원칙
  - numpy 외 무거운 의존성 없음. numpy는 함수 내부 지연 임포트(python 규칙:
    Phase 0/유틸 환경에서 import 만으로 깨지지 않게).
  - 결과를 지어내지 않는다: 지표는 입력 배열에서 계산된 값만 반환. 데이터가 없으면
    호출자가 계산을 하지 않으면 그만이며, 여기서 임의 기본값을 만들지 않는다.
  - py3.8 호환(향후 Phase 2 OccWorld 환경과 동일 코드 재사용).

용어/형상 규약
  - 점유 시퀀스(occupancy): 정수 semantic 라벨 복셀 격자. 형상 (T, *spatial),
    T=미래 지평(horizon) 스텝 수, *spatial 예: (X, Y, Z).
  - ego 궤적(trajectory): 형상 (T, D), D∈{2,3} (BEV면 2, 3D면 3).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence


def _np():
    """numpy 지연 임포트 (미설치 환경에서 모듈 import 는 되게)."""
    import numpy as np  # noqa: PLC0415
    return np


# ---------------------------------------------------------------------
# 저장 유틸 (커밋 허용 산출물)
# ---------------------------------------------------------------------
def save_metrics(metrics: dict[str, Any], out_path: str | Path) -> Path:
    """지표 dict를 results/<run>/metrics.json 으로 저장."""
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    return out


# ---------------------------------------------------------------------
# 1) 미래 mIoU (점유 semantic 예측)
# ---------------------------------------------------------------------
def per_class_iou(
    pred: Any,
    gt: Any,
    num_classes: int,
    ignore_index: Optional[int] = None,
):
    """클래스별 IoU 배열(shape=(num_classes,)) 반환.

    union==0(해당 클래스가 pred·gt 어디에도 없음)인 클래스는 NaN.
    ignore_index 라벨은 gt 기준으로 계산에서 제외.
    """
    np = _np()
    pred = np.asarray(pred).reshape(-1)
    gt = np.asarray(gt).reshape(-1)
    if pred.shape != gt.shape:
        raise ValueError(f"pred/gt 원소 수 불일치: {pred.shape} vs {gt.shape}")
    if ignore_index is not None:
        keep = gt != ignore_index
        pred, gt = pred[keep], gt[keep]

    ious = np.full(num_classes, np.nan, dtype=np.float64)
    for c in range(num_classes):
        p = pred == c
        g = gt == c
        union = int(np.count_nonzero(p | g))
        if union == 0:
            continue  # 해당 클래스 부재 → NaN 유지(평균에서 제외)
        inter = int(np.count_nonzero(p & g))
        ious[c] = inter / union
    return ious


def mean_iou(
    pred: Any,
    gt: Any,
    num_classes: int,
    ignore_index: Optional[int] = None,
    exclude: Optional[Iterable[int]] = None,
) -> float:
    """단일 프레임 mIoU (등장한 클래스에 대한 평균).

    exclude: 평균에서 뺄 클래스 id(예: free/empty 클래스). NaN(부재 클래스)은 자동 제외.
    유효 클래스가 하나도 없으면 NaN 반환(값을 지어내지 않음).
    """
    np = _np()
    ious = per_class_iou(pred, gt, num_classes, ignore_index=ignore_index)
    if exclude is not None:
        for c in exclude:
            if 0 <= c < num_classes:
                ious[c] = np.nan
    if not np.any(~np.isnan(ious)):
        return float("nan")
    return float(np.nanmean(ious))


def miou_per_horizon(
    pred_seq: Any,
    gt_seq: Any,
    num_classes: int,
    ignore_index: Optional[int] = None,
    exclude: Optional[Iterable[int]] = None,
) -> dict[str, Any]:
    """지평(horizon)별 mIoU. pred_seq/gt_seq 형상 (T, *spatial).

    반환: {"per_horizon": [mIoU_t1, ...], "mean": 전체 지평 평균(NaN 무시)}.
    """
    np = _np()
    pred_seq = np.asarray(pred_seq)
    gt_seq = np.asarray(gt_seq)
    if pred_seq.shape != gt_seq.shape:
        raise ValueError(f"시퀀스 형상 불일치: {pred_seq.shape} vs {gt_seq.shape}")
    if pred_seq.ndim < 2:
        raise ValueError("pred_seq/gt_seq 는 (T, *spatial) 형상이어야 함")

    per_h = [
        mean_iou(pred_seq[t], gt_seq[t], num_classes, ignore_index, exclude)
        for t in range(pred_seq.shape[0])
    ]
    arr = np.asarray(per_h, dtype=np.float64)
    mean = float(np.nanmean(arr)) if np.any(~np.isnan(arr)) else float("nan")
    return {"per_horizon": per_h, "mean": mean}


# ---------------------------------------------------------------------
# 2) ego L2 (궤적 오차)
# ---------------------------------------------------------------------
def ego_l2(pred_traj: Any, gt_traj: Any) -> dict[str, Any]:
    """지평별 ego 위치 L2 오차(단위: 입력 좌표 단위, 보통 m).

    형상 (T, D). 반환:
      - per_step:        각 지평의 L2 [d_1, ..., d_T]
      - mean:            전체 지평 평균
      - cumulative_mean: 1..k 지평 평균 [avg(d_1), avg(d_1:2), ...]
                         (OccWorld가 1s/2s/3s로 보고하는 누적평균 방식과 동일)
    """
    np = _np()
    pred = np.asarray(pred_traj, dtype=np.float64)
    gt = np.asarray(gt_traj, dtype=np.float64)
    if pred.shape != gt.shape:
        raise ValueError(f"궤적 형상 불일치: {pred.shape} vs {gt.shape}")
    if pred.ndim != 2:
        raise ValueError("궤적은 (T, D) 형상이어야 함")

    d = np.linalg.norm(pred - gt, axis=1)  # (T,)
    t = np.arange(1, d.shape[0] + 1)
    cumulative = np.cumsum(d) / t
    return {
        "per_step": d.tolist(),
        "mean": float(d.mean()) if d.size else float("nan"),
        "cumulative_mean": cumulative.tolist(),
    }


# ---------------------------------------------------------------------
# 3) 충돌률 (예측 ego 위치가 점유 복셀에 들어가는 비율)
# ---------------------------------------------------------------------
def collision_rate(
    ego_cells: Any,
    occupancy_seq: Any,
    free_index: int = 0,
    ignore_index: Optional[int] = None,
) -> dict[str, Any]:
    """예측 ego 셀이 '점유(비어있지 않음)' 복셀에 위치하는 스텝의 비율.

    입력
      - ego_cells:     (T, ndim) 정수 복셀 인덱스 (occupancy_seq[t] 격자 좌표계).
      - occupancy_seq: (T, *spatial) 정수 semantic 라벨.
    규칙
      - 셀 라벨이 free_index/ignore_index 가 아니면 충돌 1회.
      - 격자 범위를 벗어난 셀은 '판정 불가'로 분모에서 제외(값 지어내지 않음).
    반환: {"rate": 충돌/판정가능, "collisions": n, "evaluable": m, "out_of_bounds": k}
    """
    np = _np()
    ego = np.asarray(ego_cells)
    occ = np.asarray(occupancy_seq)
    T = occ.shape[0]
    if ego.shape[0] != T:
        raise ValueError(f"스텝 수 불일치: ego {ego.shape[0]} vs occ {T}")
    spatial = occ.shape[1:]
    if ego.ndim != 2 or ego.shape[1] != len(spatial):
        raise ValueError(f"ego_cells 는 (T, {len(spatial)}) 정수 인덱스여야 함")

    collisions = 0
    evaluable = 0
    out_of_bounds = 0
    for t in range(T):
        idx = tuple(int(v) for v in ego[t])
        in_bounds = all(0 <= idx[k] < spatial[k] for k in range(len(spatial)))
        if not in_bounds:
            out_of_bounds += 1
            continue
        label = int(occ[t][idx])
        if ignore_index is not None and label == ignore_index:
            continue  # 판정 불가(무시 라벨)
        evaluable += 1
        if label != free_index:
            collisions += 1

    rate = float(collisions / evaluable) if evaluable else float("nan")
    return {
        "rate": rate,
        "collisions": collisions,
        "evaluable": evaluable,
        "out_of_bounds": out_of_bounds,
    }


# ---------------------------------------------------------------------
# 4) 롤아웃 안정성 / 발산 지표
# ---------------------------------------------------------------------
def rollout_divergence(error_seq: Sequence[float]) -> dict[str, Any]:
    """지평이 길어질수록 오차가 얼마나 커지는지(발산)를 정량화.

    입력: 지평별 '오차' 수열(예: 지평별 ego L2, 또는 1-mIoU). 클수록 나쁨.
    반환
      - slope:           오차 vs 지평 index 의 최소제곱 기울기(스텝당 증가량)
      - final_over_first: error[-1]/error[0] (첫 스텝 대비 마지막 배율)
      - monotonic_frac:  인접 스텝에서 오차가 증가한 비율(롤아웃 불안정 지표)
    NaN 은 계산에서 제외. 유효 점 <2 면 해당 지표는 NaN.
    """
    np = _np()
    e = np.asarray(list(error_seq), dtype=np.float64)
    valid = ~np.isnan(e)
    ev = e[valid]

    if ev.size >= 2:
        x = np.arange(ev.size, dtype=np.float64)
        slope = float(np.polyfit(x, ev, 1)[0])
        diffs = np.diff(ev)
        monotonic_frac = float(np.count_nonzero(diffs > 0) / diffs.size)
        first = ev[0]
        final_over_first = float(ev[-1] / first) if first != 0 else float("inf")
    else:
        slope = float("nan")
        monotonic_frac = float("nan")
        final_over_first = float("nan")

    return {
        "slope": slope,
        "final_over_first": final_over_first,
        "monotonic_frac": monotonic_frac,
    }
