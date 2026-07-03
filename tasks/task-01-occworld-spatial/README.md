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
- **Phase 0 (스캐폴딩):** 로컬 3060. ← *현재 단계*
- **Phase 1 (mini 데이터·기준선·평가 골격):** 로컬 3060. mini info pkl은 "직접 생성" 기본, 실패 시 val pkl 필터, 그래도 안 되면 한계 명시.
- **Phase 2 (OccWorld 사전학습 추론):** 로컬 WSL2 smoke 우선 → mmcv/mmdet3d/spconv 빌드 실패 또는 8GB OOM 시 **Colab(T4/L4 16GB) 이관**. 이관 지점·사유 기록.

## 결과 (실제 실행값만 기입)
**ego L2 기준선 — 실 nuScenes v1.0-mini, val 2씬/63윈도우** (과거 2s→미래 3s, CPU, 누적평균 m)

| 모델 | 미래 mIoU | L2@1s | L2@2s | L2@3s | 하드웨어 | 비고 |
|---|---|---|---|---|---|---|
| copy-last (persistence=논문 Copy&Paste) | (Occ3D 대기) | 3.887 | 6.460 | 9.002 | CPU (Win11) | 실측 |
| linear-extrapolation (등속) | (Occ3D 대기) | 0.454 | 1.100 | 2.006 | CPU (Win11) | 실측 |
| OccWorld (추론) | — | — | — | — | — | Phase 2 미실행 |

- 재현: `python scripts/build_mini_infos.py && python scripts/eval_ego_baseline.py`
  → `results/<run_id>/metrics.json`(+`summary.md`)에 git hash·하드웨어·seed·train split 포함.
- 점유 mIoU/IoU/충돌률은 **Occ3D-nuScenes gts 부재로 미계산**(`blocked_on_occ3d`).
- ⚠️ val은 2씬(63윈도우)로 표본이 작아 일반화 주의.

## Phase 1 진행 현황 (2026-07 기준)
- 구현 완료: 지표(`common/metrics.py`), 기준선(`src/baselines.py`), 더미 스모크(`smoke.sh`), 취득 안내(`download_data.sh`).
- **실 nuScenes v1.0-mini 취득·검증 완료**: 10씬(train 8/val 2, 공식 mini split), 404 키프레임.
  `.tgz` 로컬 배치 → `data/nuscenes/`(gitignore) 추출 → `build_mini_infos.py`로 temporal info 생성(전략 [A] 성공, 무설치).
- **실데이터 ego-L2 기준선 실측 완료**(위 결과 표). 등속 외삽이 persistence보다 크게 우수.
- 아직 미실행/미확인: **Occ3D-nuScenes gts 취득**(점유 지표), OccWorld 추론(Phase 2).

## 한계 / 미확인
- **점유 지표(mIoU/IoU/충돌률) 없음**: v1.0-mini.tgz에는 센서/메타만 있고 Occ3D 점유 gts가 없어 미계산(`blocked_on_occ3d`).
- ego-L2는 **기준선 2종만**의 실측이며 OccWorld 대비는 아직 없음(Phase 2 추론 후).
- OccWorld 가중치·py3.8/mmdet3d/spconv 환경 미설치. mmdet3d/spconv 로컬 빌드 성공 여부는 Phase 2에서 실측.
- smoke.sh 수치는 합성 더미이며 성능 보고가 아니다(`metrics.json`의 `synthetic_dummy=true`, 실 결과와 별개).

## 완료 정의 (DoD) 체크 — *Phase 1 종료 시점*
- [x] `smoke.sh` 통과 + 문서화된 단일 명령으로 재현 — **단, 합성 더미 파이프라인 검증**(실데이터 아님)
- [~] 기준선 대비 지표 표(results/, 실제 값) — **부분: ego-L2 실측 완료**(2 기준선, val/train), 점유 지표는 Occ3D 대기
- [x] 시드 고정 + 환경 파일 존재 — `common/seeding.py`(set_seed) + `environment.yml`(핀, 미설치) + `config/base.yaml`
- [x] 데이터·가중치 git 미포함 + download 스크립트 + 라이선스 명시 — `.gitignore` 차단 + `download_data.sh` + 위 라이선스 절
- [x] 앵커/브리지 논문 인용(DOI·SCI(E) 여부) — Hafner+ 2025(DOI, SCI(E) O) / Zheng+ 2024(arXiv, 학회)
- [ ] Colab/게이밍PC 구동 가능성 + 실제 하드웨어 기록 — **미달: Phase 2(환경 구축·추론)에서 실측**
- [x] 실패·미확인 정직하게 기술 — 「Phase 1 진행 현황」·「한계/미확인」 섹션
- [x] 비밀키·대용량 파일 커밋 없음 — 각 커밋 전 금지 확장자 스캔 통과, PDF·데이터 미포함

> 요약: **환경·재현·라이선스·문서·안전** 항목은 Phase 1에서 충족. **실데이터 지표 표**와
> **하드웨어 구동 기록**은 데이터 취득·Phase 2 이후에 채운다(현재는 정직하게 미달로 표기).
