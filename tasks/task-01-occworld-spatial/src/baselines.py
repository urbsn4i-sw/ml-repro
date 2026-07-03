"""Task 01 기준선(baseline) — 학습 없는 참조 예측기.

OccWorld(학습형 월드모델) 대비 '이점이 있는가'를 재기 위한 단순 기준선.
어떤 학습·가중치도 필요 없고, 과거 관측만으로 미래를 예측한다.

포함 기준선
  1. copy-last (persistence): 마지막 관측을 미래로 그대로 복사.
     - 점유(occupancy)와 궤적(trajectory) 둘 다 제공.
  2. linear-extrapolation: 최근 관측으로 등속(constant-velocity) 외삽(궤적).
     - 점유는 라벨의 선형 외삽이 의미 없으므로 궤적에만 적용.

형상 규약(common.metrics 와 동일)
  - 점유 history: (T_in, *spatial) 정수 semantic 라벨.
  - 궤적 history: (T_in, D) 실수 좌표, D∈{2,3}.
  - 예측 반환:   점유 (horizon, *spatial) / 궤적 (horizon, D).

numpy 는 지연 임포트(python 규칙: 미설치 환경에서도 import 가능).
"""
from __future__ import annotations

from typing import Any


def _np():
    import numpy as np  # noqa: PLC0415
    return np


# ---------------------------------------------------------------------
# copy-last (persistence)
# ---------------------------------------------------------------------
def copy_last_occupancy(history: Any, horizon: int):
    """마지막 관측 점유 프레임을 horizon 스텝 동안 복사. → (horizon, *spatial)."""
    np = _np()
    hist = np.asarray(history)
    if hist.ndim < 2:
        raise ValueError("history 는 (T_in, *spatial) 형상이어야 함")
    if horizon < 1:
        raise ValueError("horizon 은 1 이상")
    last = hist[-1]
    return np.stack([last for _ in range(horizon)], axis=0)


def copy_last_trajectory(history_xy: Any, horizon: int):
    """마지막 ego 위치를 horizon 스텝 동안 유지(정지 가정). → (horizon, D)."""
    np = _np()
    hist = np.asarray(history_xy, dtype=np.float64)
    if hist.ndim != 2:
        raise ValueError("history_xy 는 (T_in, D) 형상이어야 함")
    if horizon < 1:
        raise ValueError("horizon 은 1 이상")
    last = hist[-1]
    return np.stack([last for _ in range(horizon)], axis=0)


# ---------------------------------------------------------------------
# linear extrapolation (constant velocity) — 궤적 전용
# ---------------------------------------------------------------------
def linear_extrapolation_trajectory(history_xy: Any, horizon: int, use_last_k: int = 2):
    """최근 use_last_k 점의 평균 속도로 등속 외삽. → (horizon, D).

    관측점이 1개뿐이면 속도를 추정할 수 없어 copy-last 로 폴백(값을 지어내지 않음).
    """
    np = _np()
    hist = np.asarray(history_xy, dtype=np.float64)
    if hist.ndim != 2:
        raise ValueError("history_xy 는 (T_in, D) 형상이어야 함")
    if horizon < 1:
        raise ValueError("horizon 은 1 이상")

    if hist.shape[0] < 2:
        return copy_last_trajectory(hist, horizon)  # 속도 추정 불가 → 폴백

    k = min(max(2, use_last_k), hist.shape[0])
    recent = hist[-k:]
    velocity = np.diff(recent, axis=0).mean(axis=0)  # 스텝당 평균 변위
    last = hist[-1]
    steps = np.arange(1, horizon + 1, dtype=np.float64)[:, None]
    return last[None, :] + steps * velocity[None, :]
