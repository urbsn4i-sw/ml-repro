#!/usr/bin/env bash
# download_data.sh — Task 01 (OccWorld) 데이터 취득 안내 + 배치 검증
# =====================================================================
# ⚠️ 이 스크립트는 "무엇도 자동으로 내려받지 않는다".
#    nuScenes/Occ3D는 계정 등록 + 이용약관(ToU) 동의가 필요하므로,
#    - (1) 수동 취득 절차/약관 문구를 출력하고
#    - (2) 로컬에 이미 배치된 파일이 있으면 "존재 여부만" 읽기전용으로 점검한다.
#    임의 URL 실행·자동 wget/curl 다운로드는 금지(CLAUDE.md 절대규칙 2, data-and-hub 규칙).
#
# 기본 경로: --subset mini (nuScenes v1.0-mini + 해당 Occ3D-nuScenes gts)
# 데이터는 DATA_ROOT(기본 tasks/task-01-*/data) 아래에 두며 .gitignore로 커밋 차단된다.
# =====================================================================
set -euo pipefail

SUBSET="mini"
DATA_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/data"

usage() {
  cat <<EOF
사용법: bash download_data.sh [--subset mini|trainval] [--data-root PATH]

  --subset     mini(기본) | trainval
  --data-root  데이터 저장 위치 (기본: tasks/task-01-*/data)

이 스크립트는 자동 다운로드를 하지 않는다. 취득 절차 안내 + 배치 검증만 수행한다.
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

# ---------------------------------------------------------------------
# 1) 라이선스 / 이용약관(ToU) 고지 — 취득 전 반드시 동의 필요
# ---------------------------------------------------------------------
print_tou() {
  cat <<'TOU'
────────────────────────────────────────────────────────────────────
[라이선스·이용약관 고지 — 반드시 읽고 동의 후 취득할 것]

■ nuScenes (Motional)
  - 라이선스: 비상업(Non-Commercial) 연구/학습용. 상업적 사용 금지.
  - 취득: https://www.nuscenes.org 계정 등록 → 로그인 → Terms of Use 동의 후 다운로드.
  - 본 과제는 "v1.0-mini"(약 4GB, 10개 씬)만 사용한다.

■ Occ3D-nuScenes (Tsinghua MARS Lab / CVPR2023 Occupancy 계열)
  - nuScenes 원본 약관에 종속되는 파생 점유(occupancy) 라벨.
  - 취득: Occ3D 공식 저장소(github.com/Tsinghua-MARS-Lab/Occ3D)의 안내에 따라
    nuScenes 계정/약관 동의 상태에서 배포 링크로 gts를 내려받는다.
  - 비상업·연구용. 재배포 금지.

※ 위 데이터·라벨·체크포인트는 이 저장소에 절대 커밋하지 않는다(.gitignore로 차단).
※ 아래 링크는 "공식 배포처로 이동해 직접 로그인·동의"하라는 안내이며,
   이 스크립트가 대신 다운로드하지 않는다.
────────────────────────────────────────────────────────────────────
TOU
}

# ---------------------------------------------------------------------
# 2) 기대 디렉토리 레이아웃 (수동 배치 목표) — mini 기준
# ---------------------------------------------------------------------
print_layout() {
  cat <<EOF
[기대 레이아웃] (수동 취득 후 아래처럼 배치)
  ${DATA_ROOT}/
  └─ nuscenes/
     ├─ v1.0-mini/              # nuScenes mini 메타데이터(json 테이블)
     ├─ samples/                # 키프레임 센서 데이터
     ├─ sweeps/                 # 중간 프레임(비키프레임)
     ├─ maps/                   # 지도
     └─ gts/                    # Occ3D-nuScenes 점유 GT (mini 씬에 해당)
  └─ nuscenes_infos_train.pkl   # (Phase 1에서 생성) temporal info, mini
  └─ nuscenes_infos_val.pkl     # (Phase 1에서 생성) temporal info, mini
EOF
}

# ---------------------------------------------------------------------
# 3) 배치 검증 (읽기 전용) — 있으면 OK, 없으면 안내만. 다운로드는 안 함.
# ---------------------------------------------------------------------
check_present() {
  local nusc="${DATA_ROOT}/nuscenes"
  echo "[check] 로컬 배치 상태 점검 (읽기 전용):"
  local ok=0
  for sub in v1.0-mini samples sweeps maps gts; do
    if [[ -d "${nusc}/${sub}" ]]; then
      echo "  ✓ ${nusc}/${sub}"
    else
      echo "  x 없음: ${nusc}/${sub}  (수동 취득·배치 필요)"
      ok=1
    fi
  done
  return "${ok}"
}

# ---------------------------------------------------------------------
# 4) mini temporal info pkl 생성 전략 (문서화만 — 지금은 실행하지 않음)
# ---------------------------------------------------------------------
# OccWorld는 씬 단위 시계열 정보(nuscenes_infos_*_temporal_*.pkl)를 입력으로 쓴다.
# mini 서브셋용 pkl은 아래 "우선순위 순서"로 만든다. Phase 1의 실제 실행 단계에서
# 사용자 승인 후 아래 함수 본문을 채워 실행하며, 각 시도 결과(성공/실패, 잡힌 씬 수)를
# results/<run_id>/ 로그에 기록한다. (재현성 규칙: 실패는 "한계/미확인"으로 명시)
#
#   [A] 직접 생성 (기본)
#       - nuscenes-devkit + (OccWorld/ mmdet3d) create_data 계열 도구로
#         v1.0-mini에서 train/val temporal pkl을 생성.
#       - 검증: "몇 개 씬이 잡히는지" 출력(mini는 train 8 / val 2 씬이 기대치).
#   [B] 실패 시: 공개 val pkl 필터
#       - 배포된 full val temporal pkl을 받아, mini의 scene token 집합으로 필터링해
#         mini 전용 pkl을 만든다. (scene token은 v1.0-mini/scene.json에서 획득)
#   [C] 둘 다 실패 시: 한계 명시
#       - README "한계/미확인"에 사유(도구 버전 불일치 등)와 함께 남긴다.
#
# 아래는 자리표시자일 뿐, Phase 0/현재 시점에서는 호출하지 않는다.
generate_mini_infos() {
  echo "[infos] mini temporal pkl 생성은 아직 실행하지 않습니다."
  echo "        → 전략 A(직접 생성) → B(val pkl 필터) → C(한계) 순서로,"
  echo "          Phase 1 실제 실행 단계에서 사용자 승인 후 진행합니다."
  echo "        생성물(*.pkl)은 대용량이므로 .gitignore로 커밋 차단됩니다."
}

# ---------------------------------------------------------------------
# 메인
# ---------------------------------------------------------------------
print_tou
echo ""
print_layout
echo ""
if check_present; then
  echo "[check] 필수 디렉토리가 모두 존재합니다. (pkl 생성 단계로 진행 가능)"
else
  echo "[check] 일부/전부 미배치. 위 약관 동의 후 공식 배포처에서 수동 취득·배치하세요."
fi
echo ""
generate_mini_infos

cat <<'NOTE'

[download] 요약: 이 스크립트는 자동 다운로드를 수행하지 않았습니다.
  → 데이터/라벨은 공식 배포처에서 로그인·약관 동의 후 직접 취득해 DATA_ROOT에 배치하세요.
  → temporal pkl 생성(A→B→C)은 Phase 1 실행 단계에서 승인 후 진행합니다.
NOTE
