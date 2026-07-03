#!/usr/bin/env bash
# smoke.sh — Task 01 파이프라인 스모크런 (더미 입력 기준선→지표)
# =====================================================================
# 목적: 실제 데이터/가중치 없이, 소량 합성 텐서 1회로
#       "기준선 예측 → 지표 계산 → metrics.json 저장" 파이프라인이 끝까지 도는지만 확인.
# ⚠️ nuScenes/Occ3D/OccWorld 가중치는 받지도, 쓰지도 않는다. 성능 보고가 아니다.
#
# 실제 mini 데이터 기반 스모크런/평가는 데이터·(Phase 2)가중치 준비 후 별도로 붙인다.
# =====================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${PYTHON:-python}"

# 출력은 커밋하지 않는 임시 디렉토리로(결과가 results/ 에 남지 않게).
OUT_DIR="$(mktemp -d 2>/dev/null || echo "${TMPDIR:-/tmp}/occworld_smoke_$$")"
mkdir -p "${OUT_DIR}"
trap 'rm -rf "${OUT_DIR}"' EXIT

echo "[smoke] 더미 기준선→지표 파이프라인 실행 (out=${OUT_DIR})"
"${PY}" "${SCRIPT_DIR}/run_baseline_demo.py" \
  --seed 42 --t-in 3 --horizon 4 --grid 6 6 2 --num-classes 4 \
  --out "${OUT_DIR}"

# 산출물 존재 검증
if [[ -f "${OUT_DIR}/metrics.json" ]]; then
  echo "[smoke] OK: metrics.json 생성 확인. 파이프라인 정상."
else
  echo "[smoke] FAIL: metrics.json 이 없습니다." >&2
  exit 1
fi
