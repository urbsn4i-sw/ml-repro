#!/usr/bin/env bash
# download_data.sh — Task 01 (OccWorld) 데이터 취득 골격
# =====================================================================
# ⚠️ Phase 0: 이 스크립트는 "골격"이다. 지금은 아무것도 내려받지 않는다.
#    실제 취득은 Phase 1(mini 데이터)에서 사용자 승인 후 실행한다.
#
# 데이터 라이선스 (반드시 준수):
#   - nuScenes: 비상업 라이선스 + 계정 등록 필요. https://nuscenes.org
#   - Occ3D-nuScenes: nuScenes 약관 종속 + 별도 배포. 등록/동의 후 사용.
#   → 자동 다운로드 불가한 부분은 "수동 취득 안내"만 출력한다(임의 URL 실행 금지).
#
# 기본 경로: --subset mini (nuScenes v1.0-mini + 해당 Occ3D gts)
# 데이터는 data/ 아래에 두며 .gitignore로 커밋 차단된다.
# =====================================================================
set -euo pipefail

SUBSET="mini"
DATA_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/data"

usage() {
  cat <<EOF
사용법: bash download_data.sh [--subset mini|trainval] [--data-root PATH]

  --subset     mini(기본) | trainval
  --data-root  데이터 저장 위치 (기본: tasks/task-01-*/data)

Phase 0에서는 실제 다운로드를 하지 않고 취득 절차만 안내한다.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --subset) SUBSET="$2"; shift 2 ;;
    --data-root) DATA_ROOT="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "알 수 없는 인자: $1"; usage; exit 1 ;;
  esac
done

echo "[download] subset=${SUBSET}  data_root=${DATA_ROOT}"
mkdir -p "${DATA_ROOT}"

# --- TODO(Phase 1): 아래 단계를 실제 취득 로직으로 채운다 ---
# 1) nuScenes v1.0-mini (metadata + samples/sweeps)
#    - 사전 등록 필요. 수동 다운로드 후 ${DATA_ROOT}/nuscenes 에 배치하도록 안내.
# 2) Occ3D-nuScenes gts (mini에 해당하는 씬만)
#    - 약관 동의 후 취득. ${DATA_ROOT}/nuscenes/gts 에 배치.
# 3) mini info pkl 생성 (기본: "직접 생성")
#    - create_data 도구로 nuscenes_infos_train/val.pkl 생성 시도.
#    - 생성 절차와 "몇 개 씬이 잡히는지"를 results/ 로그에 남긴다.
#    - 실패 시 대안: 공개 val pkl 을 mini 씬으로 필터링.
#    - 둘 다 실패하면 "한계"로 명시 (README 한계 섹션).

cat <<'NOTE'
[download] Phase 0 골격입니다. 실제 취득은 미실행 상태.
  → nuScenes/Occ3D는 등록·약관 동의가 필요하므로 자동 다운로드하지 않습니다.
  → Phase 1 진입(사용자 승인) 후 위 TODO 단계를 채워 실행합니다.
NOTE
