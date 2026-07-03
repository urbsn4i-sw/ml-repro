# ml-repro — ML 재현 과제 모노레포

ML/DL/RL 최근 돌파 사례를 **SLAM·도시·공간지능 방향으로 재설계한 과제(01~04)** 를
게이밍PC/Colab 수준에서 **축소 재현·공개**하는 저장소. 목적은 SOTA 재현이 아니라 **원리 재현·학습**.

- 마스터 지침: [`PROJECT_GUIDELINE.md`](PROJECT_GUIDELINE.md)
- 에이전트 규약: [`CLAUDE.md`](CLAUDE.md) · [`AGENTS.md`](AGENTS.md) · [`.claude/rules/`](.claude/rules)

## 과제 목록
| 과제 | 방향 | 상태 |
|---|---|---|
| [task-01 OccWorld](tasks/task-01-occworld-spatial) | 3D 점유 월드모델 | **Phase 0 (스캐폴딩)** |
| task-02 STGNN | 도시 교통 시공간 예측 | 미착수 |
| task-03 Cross-Embodiment Nav | 교차 임바디먼트 내비 | 미착수 |
| task-04 PointMamba/SLAM | Mamba 기반 LiDAR·점군 | 미착수 |

## 저장소 구조 (Phase 0 스캐폴딩)
```
ml-repro/
├─ common/                # 공통 유틸 (seeding · metrics · hub[HF 화이트리스트])
├─ environment.yml        # 핀 버전 후보 (설치는 Phase 2 실측)
├─ .gitignore             # 데이터·라벨·체크포인트·비밀 전 경로 차단
├─ .claude/rules/         # python · reproducibility · data-and-hub
└─ tasks/
   └─ task-01-occworld-spatial/
      ├─ README.md        # 12항목 과제 카드 + DoD
      ├─ config/ src/ notebooks/ results/
      └─ scripts/         # download_data.sh · smoke.sh (골격)
```

## 원칙 (요약)
- 결과를 지어내지 않는다 — 실제 실행값만 보고, 미검증은 "한계"로 명시.
- 대용량·비밀 커밋 금지 — 데이터·가중치·`.env`·토큰은 어떤 경로로도 커밋하지 않음.
- 재현성 우선 — 시드 고정 · config 기반 · 스모크런 먼저.
- 논문·라이선스 인용 유지.

자세한 내용은 [`PROJECT_GUIDELINE.md`](PROJECT_GUIDELINE.md) 참조.
