# Task 01 — 3D 점유 월드모델 (OccWorld) · Spatial Intelligence

> 상태: **Phase 0 (스캐폴딩)**. 데이터·가중치 미취득. 아래 결과 표는 실측 후에만 채운다.

## 과제 카드 (12항목)
- **과제명 / 방향:** 과제01 — 3D 점유 월드모델 · Spatial Intelligence
- **1. 문제 설명:** 자율주행 씬을 3D 점유(occupancy) 격자로 표현하고, 과거 관측으로 **미래 점유와 ego 궤적**을 예측하는 월드모델을 축소 재현한다. OccWorld는 점유를 VQ-VAE로 토큰화하고 GPT류 시공간 트랜스포머로 미래를 자기회귀 예측한다.
- **2. 왜 어려운가:**
  - 3D 점유 데이터(Occ3D)와 nuScenes는 **비상업 라이선스·등록 필요**, 용량이 크다.
  - `mmcv/mmdet3d/spconv` 스택은 **CUDA·PyTorch·컴파일 버전 궁합이 극도로 예민**(sm_86, py3.8).
  - VRAM 제약(3060 8GB)으로 원본 배치·해상도 그대로는 OOM.
- **3. 관련 과제:** 과제04(PointMamba/점군), 자율주행 BEV/Occ 인식.
- **4. 핵심 질문(RQ):** 게이밍PC/Colab 예산에서 OccWorld의 **미래 점유 예측 원리**를 mini 데이터로 재현하고, 단순 기준선(copy-last) 대비 이점을 관측할 수 있는가?
- **5. 예시 시나리오:** nuScenes mini 씬에서 t 시점까지의 점유를 입력받아 t+1..t+N 점유와 ego 위치를 예측.
- **6. 실험 구조:** 1) mini 데이터·info pkl 준비 → 2) copy-last 기준선 → 3) OccWorld 사전학습 가중치 추론 → 4) mIoU/L2 평가 → 5) (여력 시) 축소 파인튜닝.
- **7. 실험 설계(표):**

  | 구성요소 | 축소 재현 설정 |
  |---|---|
  | 목표 태스크 | 미래 3D 점유 예측 (+ ego L2) |
  | 데이터 | nuScenes v1.0-mini + Occ3D gts (subset mini) |
  | 모델 | OccWorld (VQ-VAE + 시공간 트랜스포머), 사전학습 가중치 추론 우선 |
  | 지표 | 미래 mIoU, ego L2, 충돌률, 롤아웃 안정성 |
  | 비교군(기준선) | copy-last (마지막 관측 점유를 미래로 복사) |
  | 절제(ablation) | (여력 시) 예측 horizon 길이, 토큰화 유무 |
- **8. 학습자가 배울 수 있는 점:** 점유 토큰화(VQ-VAE), 자기회귀 시공간 예측, mmdet3d 계열 환경 구축의 현실적 난점, 컴퓨트 예산 내 축소 재현 방법.
- **9. 상세 설명:** §"재현 방법" 참조. 범위 축소는 (a) mini 데이터, (b) 사전학습 추론 우선, (c) 배치·해상도 축소, (d) 안 되면 Colab 이관으로 달성.
- **10. 최초 프롬프트:** "OccWorld를 nuScenes mini로 축소 재현. 먼저 스캐폴딩·기준선·평가 골격(Phase 0~1)을 로컬에서, 사전학습 추론(Phase 2)은 로컬 smoke 후 실패 시 Colab 이관. 데이터·가중치는 커밋 금지."
- **11. 난이도:** 전반 **상**. ① 데이터 취득: **중~상**(등록·라이선스·용량). ② 게이밍PC 구동: **상**(mmdet3d/spconv 빌드 + 8GB VRAM; Colab fallback 상정).
- **12. 태그:** #occupancy #world-model #nuscenes #occ3d #vqvae #autoregressive #mmdet3d

## 대표 논문·라이선스
> 방침: **서지정보(링크·DOI)만 기록**한다. PDF는 저장소에 넣지 않는다.
> (OccWorld=arXiv 기본 라이선스로 재배포 불확실 / DreamerV3=CC BY 4.0 이나 통일성 위해 둘 다 링크만.)

- **앵커 논문:** Hafner, D., Pasukonis, J., Ba, J., & Lillicrap, T. (2025). *Mastering diverse control
  tasks through world models*. **Nature, 640, 647–653.** DOI [10.1038/s41586-025-08744-2](https://doi.org/10.1038/s41586-025-08744-2).
  SCI(E) **O**. (DreamerV3 — world-model 원리의 앵커)
- **브리지 논문:** Zheng, W., Chen, W., Huang, Y., Zhang, B., Duan, Y., & Lu, J. (2024).
  *OccWorld: Learning a 3D Occupancy World Model for Autonomous Driving*. **ECCV 2024.**
  arXiv [2311.16038](https://arxiv.org/abs/2311.16038). 학회 논문(SCI(E) 해당없음).
- **데이터 라이선스:** nuScenes — 비상업·계정 등록 필요. Occ3D-nuScenes — nuScenes 약관 종속.
- **코드 라이선스:** OccWorld 원 구현 라이선스(리포 확인 후 명시) — Phase 2에서 확정.

### 논문에서 확정한 설정 (config/문서 반영, 성능수치는 *참조용*)
- **입력/예측 지평:** 과거 **2s** 관측 → 미래 **3s** 예측(OccWorld 표준 프로토콜).
- **표준 지표:** 미래 **mIoU · IoU**, ego **L2**, **충돌률(collision rate)**.
- **기준선 정의:** 우리의 `copy-last`(persistence) = OccWorld 논문의 **Copy&Paste** 기준선과 동일 정의
  (마지막 관측 점유를 미래로 복사).
- **토큰화(VQ-VAE):** 코드북 **512**, 임베딩 차원 **128**, 공간 다운샘플 **×4**.
- ⚠️ 논문 보고 성능수치는 **참조용**일 뿐, 우리 결과 표에 절대 옮겨 적지 않는다(재현성 규칙).

## 재현 방법 (실제 명령) — Phase 진행에 따라 채움
```bash
# 0) 환경 (Phase 2에서 실측 설치)
conda env create -f ../../environment.yml

# 1) 데이터(소규모) — Phase 1
bash scripts/download_data.sh --subset mini

# 2) 스모크런 — Phase 1 (구현됨: 합성 텐서로 기준선→지표 파이프라인 검증, 실데이터 불필요)
bash scripts/smoke.sh

# 3) 기준선 → 사전학습 추론 — Phase 1~2
#    기준선 구현됨: src/baselines.py (copy-last / linear-extrapolation)
#    지표 구현됨:   common/metrics.py (미래 mIoU · ego L2 · 충돌률 · 롤아웃 발산)
# python scripts/infer_occworld.py --config config/occworld.yaml --ckpt <hf_or_local>   # Phase 2
```

## 실행 계획 (하이브리드 경로)
- **Phase 0 (스캐폴딩):** 로컬 3060. ✅ 완료.
- **Phase 1 (mini 데이터·기준선·평가 골격):** 로컬 3060. ✅ 완료(mini info는 무설치 직접 생성 성공, 기준선·점유 지표 실측).
- **Phase 2 (OccWorld 사전학습 추론):** ⚠️ *부분 완료 / 미확보* — 코드 조사·재개 준비는 끝났으나 저자 가중치·pkl 링크 비활성으로 추론 미확보(아래 「Phase 2」 절). 링크 복구 시 WSL2 smoke → 실패/8GB OOM 시 Colab(T4/L4 16GB) 이관.

## 결과 (실제 실행값만 기입) — Phase 1 기준선, 실 nuScenes v1.0-mini
과거 2s(4f) → 미래 3s(6f) @2Hz, CPU. **val 2씬/63윈도우**(주 평가), train 8씬/251윈도우는 참고.

**(a) 점유 예측 — copy-last(=논문 Copy&Paste 정의), 실 Occ3D gts** (mask_camera 적용, free=17 제외, %)

| split | 지표 | @1s | @2s | @3s | 6지평평균 |
|---|---|---|---|---|---|
| **val** | 미래 mIoU | 10.75 | 6.40 | 5.10 | 8.76 |
| **val** | 이진 IoU | 27.21 | 21.11 | 18.35 | 24.13 |
| train | 미래 mIoU | 22.95 | 19.01 | 17.39 | 21.17 |
| train | 이진 IoU | 40.40 | 34.84 | 32.37 | 37.41 |
| *(참조용)* | *논문 Copy&Paste mIoU* | *14.91* | *10.54* | *8.11* | *—* |
| *(참조용)* | *논문 Copy&Paste IoU* | *24.47* | *19.77* | *17.14* | *—* |

> ⚠️ 논문 수치는 `reference_only` — full nuScenes val(150씬) 기준이라 mini(2씬)와 **직접 비교 불가**. 우리 결과로 쓰지 않음.

**(b) ego 궤적 예측 — L2 누적평균(m), val**

| 모델 | L2@1s | L2@2s | L2@3s |
|---|---|---|---|
| copy-last (persistence) | 3.887 | 6.460 | 9.002 |
| linear-extrapolation (등속) | 0.454 | 1.100 | 2.006 |
| OccWorld (추론) | — | — | — (Phase 2) |

- 재현: `python scripts/build_mini_infos.py && python scripts/eval_ego_baseline.py && python scripts/eval_occ_baseline.py`
  → `results/<run_id>/{metrics.json,summary.md}`에 git hash·하드웨어·seed·train split 기록.
- ⚠️ val은 2씬(63윈도우)로 표본이 작아 일반화 주의. mIoU 클래스셋은 0..16(free=17 제외, 'others'=0 포함).

## Phase 1 — ✅ 공식 종료 (2026-07-03)
> **최종 상태:** 실 nuScenes v1.0-mini에서 **ego-L2 + 점유 mIoU/IoU 기준선 실측 완료**. 충돌률은 예측 궤적↔점유 결합 프로토콜 확정 후로 유보, OccWorld 모델 대비는 Phase 2로 이월.

- 구현 완료: 지표(`common/metrics.py`), 기준선(`src/baselines.py`), 더미 스모크(`smoke.sh`), 취득 안내(`download_data.sh`).
- **실 nuScenes v1.0-mini 취득·검증 완료**: 10씬(train 8/val 2, 공식 mini split), 404 키프레임.
  `.tgz` 로컬 배치 → `data/nuscenes/`(gitignore) 추출 → `build_mini_infos.py`로 temporal info 생성(전략 [A] 성공, 무설치).
- **실데이터 ego-L2 기준선 실측 완료**. 등속 외삽이 persistence보다 크게 우수.
- **실 Occ3D gts 점유 기준선(copy-last) 실측 완료**: 10씬 gts 선택 취득, mask_camera 적용 미래 mIoU/IoU.
- 아직 미실행/미확인: OccWorld 사전학습 추론(Phase 2), 충돌률(예측 궤적×점유 결합 정의 확정 후).

## 한계 / 미확인
- **점유 기준선은 copy-last 1종**만 실측 — OccWorld 모델 대비는 아직 없음(Phase 2 추론 후).
- **충돌률 미계산**: 구현은 있으나(metrics.collision_rate) 예측 ego 궤적↔점유 결합 프로토콜 확정 후 산출 예정.
- **annotations.json split 교차확인 미확인/한계**: 파일 미취득. devkit 공식 mini split(train8/val2)로 진행(gts 샘플 수 일치로 정합성만 확인).
- ⚠️ val 2씬(63윈도우) 표본 작음. 논문 수치는 full val 기준이라 직접 비교 불가(reference_only).
- OccWorld 가중치·py3.8/mmdet3d/spconv 환경 미설치. 로컬 빌드 성공 여부는 Phase 2에서 실측.
- smoke.sh 수치는 합성 더미이며 성능 보고가 아니다(`synthetic_dummy=true`, 실 결과와 별개).

## 완료 정의 (DoD) 체크 — *Phase 1 종료 시점*
**Phase 1 범위 내 — 완료**
- [x] `smoke.sh` 통과 + 문서화된 단일 명령으로 재현 (합성 더미 파이프라인 검증)
- [x] 기준선 대비 지표 표(results/, 실제 값) — ego-L2 + 점유 mIoU/IoU 실측(copy-last/linear, val·train)
- [x] 시드 고정 + 환경 파일 존재 — `seeding.py` + `environment.yml`(핀) + `config/base.yaml`
- [x] 데이터·가중치 git 미포함 + download 스크립트 + 라이선스 명시 — `.gitignore` + `download_data.sh` + 라이선스 절
- [x] 앵커/브리지 논문 인용(DOI·SCI(E) 여부) — Hafner+ 2025(DOI, SCI(E) O) / Zheng+ 2024(arXiv, 학회)
- [x] 실패·미확인 정직하게 기술 — 「한계/미확인」·「Phase 1」 섹션
- [x] 비밀키·대용량 파일 커밋 없음 — 각 커밋 전 금지 확장자 스캔 통과, PDF·데이터·gts·npz 미포함

**미완 — Phase 2 / 후속으로 이월** (Phase 1 범위 밖, 정직하게 미달)
- [ ] 충돌률 실측 — *유보*: 구현은 있으나(`metrics.collision_rate`) 예측 궤적↔점유 결합 프로토콜 확정 후
- [ ] OccWorld 사전학습 추론 + 모델↔기준선 대비 — *미확보(외부요인)*: 저자 Tsinghua cloud 가중치·pkl 링크 비활성으로 취득 불가 → [ISSUE_phase2_blocked.md](ISSUE_phase2_blocked.md)
- [ ] Colab/게이밍PC 구동 가능성 + 실제 하드웨어(GPU/VRAM) 기록 — *Phase 2*(환경 구축·추론에서 실측)

> 요약: Phase 1 범위(스캐폴딩·mini 데이터·기준선·평가)는 **모두 충족하여 종료**. OccWorld 추론·충돌률·하드웨어 구동은 Phase 2/후속으로 이월(현재 정직하게 미달로 표기).

## Phase 2 — ⚠️ 부분 완료 / OccWorld 추론 **미확보(외부요인 한계)** (2026-07-04)
> **결론:** OccWorld 사전학습 추론은 **취득 불가로 미확보**. 저자 사전학습 가중치·temporal pkl이 올라간 **Tsinghua cloud Seafile 공유 링크가 현재 비활성**(브라우저·CLI 모두 "링크 없음"). 우리가 값을 지어낼 수 없으므로 **모델↔기준선 대비 지표는 공란 유지**. → 이슈 [ISSUE_phase2_blocked.md](ISSUE_phase2_blocked.md).

**확보 완료 — Phase 2 준비 상태(링크 복구 시 즉시 재개 가능):**
- **2-A 코드 조사 완료**: OccWorld 원 리포 구조 파악. eval(`eval_metric_stp3.py` + `config/occworld.py`)이 요구하는 데이터·모델·축소 지점 확정.
- **temporal pkl 재생성 경로 확정 = 전략 B(필터)**: 원 리포에 pkl **생성 스크립트 없음**(배포 전용). 따라서 배포된 full `val` pkl을 우리 mini 씬(scene-0103·0916)으로 **필터링**하는 방법이 정공법. pkl 구조(`data['infos']`=scene→프레임 리스트, 각 프레임 token·ego pose·`gt_ego_fut_trajs`·`pose_mode`·`cams`)까지 파악.
- **eval 데이터 요구사항 확정**: `data/nuscenes/gts/{scene}/{token}/labels.npz`(✅ 보유) + val temporal pkl(미확보) + 사전학습 ckpt(미확보).
- **8GB OOM 예상 지점 확정**: VQVAE 인코더가 12프레임×200²×128ch 처리 시 활성화 ≈2.5GB→배수로 8GB 초과 가능. 완화책(eval 프레임 수 축소·no_grad·청크). 실패 시 Colab(T4/L4 16GB) 이관 — 사전 합의.
- **의존성 리스크 확정**: eval도 `mmdet3d`/`mmcv`/`mmengine` 필수(dataset.py 모듈 로드 시 import). sm_86/py3.8 빌드가 관문.

**재개 조건:** Tsinghua 링크 복구 **또는** 대체 미러로 가중치·pkl 확보 시 → 2-B2(pkl 필터) → 2-D(환경 설치) → 2-E(smoke) 순으로 진행.
