# ml-repro — ML 재현 과제 모노레포

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> 🇬🇧 **[English README](README.md)**

최근 ML 돌파(DreamerV3, OccWorld 등)를 게이밍PC(**RTX 3060, 8GB**)나 Colab 수준에서 SLAM·도시·공간지능 방향으로 재설계하여 **축소 재현**한다. 목적은 SOTA 재현이 아니라 **원리 재현·학습**이다.

## 프로젝트 목적
최근 ML/DL/RL 돌파 사례(예: **DreamerV3**, **OccWorld**)를 소비자용 하드웨어나 Colab에서 **원리 중심으로 축소 재현**하고, SLAM·도시·공간지능 문제로 재설계한다. SOTA 수치 재현이 목적이 **아니라** 학습·재현이 목적이다.

## 핵심 원칙 — 결과를 지어내지 않는다
실제 실행값(`metrics.json`)만 보고한다. 미검증·미실행은 **한계**로 명시하며 달성한 것처럼 쓰지 않는다. 데이터셋·라벨·체크포인트는 **절대 커밋하지 않는다**(`.gitignore` 차단).

## 진행 상태 (정직하게)
| 과제 | 주제 | 상태 |
|---|---|---|
| **01 — OccWorld / DreamerV3** | 3D 점유 월드모델 | Phase 0·1 **완료**(기준선 실측) · Phase 2(OccWorld 추론) **보류(외부요인)** · Phase 3 **공개 완료** |
| 02 — STGNN | 도시 교통 시공간 예측 | **계획됨**(미착수) |
| 03 — Cross-Embodiment Nav | 교차 임바디먼트 내비 | **계획됨**(미착수) |
| 04 — PointMamba / SLAM | Mamba 기반 LiDAR·점군 | **계획됨**(미착수) |

## 과제 01 — 실측 결과 (실제값, 기준선만)
실 nuScenes **v1.0-mini**, 프로토콜 과거 2s → 미래 3s @ 2Hz, CPU. **검증 split = 2씬 / 63윈도우** (⚠️ 표본이 작아 일반화 주의).

**점유 — copy-last 기준선(= 논문 "Copy&Paste" 정의), 실 Occ3D gts** (camera-mask 적용, free 클래스 17 제외):

| 지표 (%) | @1s | @2s | @3s |
|---|---|---|---|
| 미래 mIoU | 10.75 | 6.40 | 5.10 |
| 이진 IoU | 27.21 | 21.11 | 18.35 |

**ego 궤적 — L2 (m), 누적평균:**

| 모델 | L2@1s | L2@2s | L2@3s |
|---|---|---|---|
| copy-last (persistence) | 3.89 | 6.46 | 9.00 |
| linear-extrapolation (등속) | 0.45 | 1.10 | 2.01 |
| OccWorld (사전학습 추론) | — | — | — (보류) |

> 위 수치는 **우리 실측값**이며 논문 수치가 아니다. 움직이는 차량에 대해 등속 외삽이 persistence를 크게 앞서는 것은 예상되는 결과다.

## Phase 2 — OccWorld 추론이 보류된 이유 (외부요인)
OccWorld 사전학습 가중치·temporal pkl은 저자 **Tsinghua cloud(Seafile) 링크**로만 배포되는데, 이 링크가 **현재 비활성**이다(브라우저·CLI 모두 "share link not found"; 실제 토큰이 가짜 토큰과 동일하게 반응). 모델 수치를 지어낼 수 없으므로 **OccWorld↔기준선 대비 행은 공란으로 둔다.** [`ISSUE_phase2_blocked.md`](tasks/task-01-occworld-spatial/ISSUE_phase2_blocked.md) 참조. 링크 복구 또는 미러 확보 시 재개한다.

## 재현 방법 (과제 01 기준선)
```bash
# 1) 데이터 취득 안내(수동; nuScenes/Occ3D는 등록 필요 — 자동 다운로드 없음)
bash tasks/task-01-occworld-spatial/scripts/download_data.sh --subset mini
# 2) mini temporal info 생성(무설치) + 기준선 평가(실 gts)
python tasks/task-01-occworld-spatial/scripts/build_mini_infos.py
python tasks/task-01-occworld-spatial/scripts/eval_ego_baseline.py
python tasks/task-01-occworld-spatial/scripts/eval_occ_baseline.py
```
상세: [`PROJECT_GUIDELINE.md`](PROJECT_GUIDELINE.md), 과제 카드 [`tasks/task-01-occworld-spatial/README.md`](tasks/task-01-occworld-spatial/README.md).

## 논문
- **앵커:** Hafner et al. (2025). *Mastering diverse control tasks through world models.* **Nature 640, 647–653.** DOI [10.1038/s41586-025-08744-2](https://doi.org/10.1038/s41586-025-08744-2). (DreamerV3)
- **브리지:** Zheng et al. (2024). *OccWorld.* **ECCV 2024.** arXiv [2311.16038](https://arxiv.org/abs/2311.16038).

## 데이터 출처·라이선스
- **코드(이 저장소):** **MIT License** — [`LICENSE`](LICENSE) 참조. 우리가 작성한 스크립트·지표·config에 적용.
- **데이터는 MIT 적용 대상이 아니며, 이 저장소에 포함되지 않는다:**
  - **nuScenes** — **비상업(CC BY-NC-SA 4.0)**, 계정 등록 필요, **재배포 금지**.
  - **Occ3D-nuScenes** — nuScenes 이용약관(비상업) 종속.
  - 데이터는 원 라이선스에 따라 **사용자가 직접 등록·취득**해야 한다. 이 저장소에는 **원데이터·라벨·모델 가중치가 없다**(`.gitignore` 차단).

## Contact / Author
- **Author:** urbsn4i-sw (GitHub)
- **Email:** urban4i.sw@gmail.com
- 문의·재현 이슈는 GitHub Issues 또는 위 이메일로. 학습·재현 목적의 비상업 저장소입니다.
