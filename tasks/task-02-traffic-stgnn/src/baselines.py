"""Task 02 (교통 STGNN) 단순 기준선 — 파이프라인 검증 + 비교 기준.

기준선 정의 (DCRNN / Graph WaveNet 논문 관례)
  - **copy-last (persistence):** 마지막 관측 프레임을 미래 전 지평에 복사. 교통은 자기상관이
    높아 단기(15분)에서 의외로 강한 기준선.
  - **Historical Average (HA):** 관측 history 평균을 미래 전 지평에 사용(단순형). 논문의 HA 는
    보통 '요일×시간대' 계절 평균을 쓰지만, 여기서는 학습/스모크 단계용 단순형을 제공하고
    계절형(seasonal)은 Phase 1 에서 데이터가 있을 때 확장한다.

설계 원칙: numpy 만 사용, 지연 임포트, 값 지어내지 않음(입력에서 계산된 값만).
형상 규약: history (T_in, N[, C]), 반환 (horizon, N[, C]).
"""
from __future__ import annotations

from typing import Any


def _np():
    import numpy as np  # noqa: PLC0415
    return np


def copy_last(history: Any, horizon: int) -> Any:
    """마지막 관측 프레임을 horizon 만큼 복사. 반환 (horizon, *history.shape[1:])."""
    np = _np()
    hist = np.asarray(history, dtype=np.float64)
    if hist.ndim < 1 or hist.shape[0] < 1:
        raise ValueError("history 는 (T_in, ...) 이고 T_in>=1 이어야 함")
    last = hist[-1]
    return np.repeat(last[None, ...], horizon, axis=0)


def historical_average(history: Any, horizon: int, null_val: float = 0.0) -> Any:
    """관측 history 의 (결측 제외) 평균을 미래 전 지평에 사용. 반환 (horizon, *).

    null_val 위치는 평균에서 제외한다(교통 결측=0 관례). 어떤 노드에서 유효 관측이
    하나도 없으면 그 노드는 NaN(값을 지어내지 않음).
    """
    np = _np()
    hist = np.asarray(history, dtype=np.float64)
    if hist.ndim < 1 or hist.shape[0] < 1:
        raise ValueError("history 는 (T_in, ...) 이고 T_in>=1 이어야 함")
    if null_val is None or (isinstance(null_val, float) and np.isnan(null_val)):
        mask = ~np.isnan(hist) if null_val is not None else np.ones_like(hist, dtype=bool)
    else:
        mask = hist != null_val
    summed = np.where(mask, hist, 0.0).sum(axis=0)
    counts = mask.sum(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = np.where(counts > 0, summed / counts, np.nan)
    return np.repeat(mean[None, ...], horizon, axis=0)
