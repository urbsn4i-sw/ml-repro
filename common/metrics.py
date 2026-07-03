"""공통 지표 유틸 (스켈레톤).

Phase 0에서는 인터페이스만 확정한다. 실제 계산 로직은 Phase 1(평가 골격)에서
과제별 표준 지표에 맞춰 채운다. (PROJECT_GUIDELINE.md §8)

Task 01 (OccWorld) 표준 지표: 미래 mIoU, ego L2, 충돌률, 롤아웃 안정성.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def save_metrics(metrics: dict[str, Any], out_path: str | Path) -> Path:
    """지표 dict를 results/<run>/metrics.json 으로 저장 (커밋 허용 산출물)."""
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    return out


def mean_iou(*args: Any, **kwargs: Any) -> float:
    """미래 점유 mIoU. Phase 1에서 구현."""
    raise NotImplementedError("Phase 1(평가 골격)에서 구현 예정")


def ego_l2(*args: Any, **kwargs: Any) -> float:
    """ego trajectory L2 오차(m). Phase 1에서 구현."""
    raise NotImplementedError("Phase 1(평가 골격)에서 구현 예정")
