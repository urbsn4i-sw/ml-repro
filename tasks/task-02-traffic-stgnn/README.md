# Task 02 — 도시 교통 시공간 예측 (STGNN) · Spatiotemporal Forecasting

> 상태: **Phase 1 진행 — 데이터 취득 + 기준선 실측 완료**. METR-LA 실데이터로 copy-last·seasonal-HA
> 실측(아래 결과 표). STGNN 실제 학습은 **Phase 2** 로 이월. 데이터·가중치는 커밋하지 않는다(.gitignore).

## 과제 카드 (12항목)
- **과제명 / 방향:** 과제02 — 도시 교통 시공간 예측 · 그래프 신경망(STGNN)
- **1. 문제 설명:** 도로망 위 수백 개 센서의 과거 속도 시계열로 **미래 15/30/60분 속도**를 예측한다.
  핵심은 **공간(도로망 그래프) × 시간(시계열)** 의존성을 함께 모델링하는 것. 앵커인 **GraphCast**가
  지구를 격자→그래프로 보고 GNN 으로 다음 상태를 자기회귀 롤아웃하듯, 교통도 센서를 노드로 하는
  그래프 위에서 미래를 예측한다(도시 규모 축소판).
- **2. 왜 어려운가:**
  - 공간 의존이 **비유클리드**(도로망 연결)라 격자 CNN 이 안 맞고 그래프가 필요하다.
  - 다단계(multi-step) 예측에서 **오차가 지평 따라 누적**된다(15분은 쉬워도 60분은 급격히 어려움).
  - 인접행렬을 어떻게 정의/학습하느냐(고정 도로망 vs 데이터로 학습)가 성능을 좌우한다.
  - 결측(센서 고장=0)이 섞여 있어 **마스크드 지표**가 필수.
- **3. 관련 과제:** 과제01(시공간 롤아웃 예측·월드모델), GraphCast 류 격자 물리예측.
- **4. 핵심 질문(RQ):**
  - **RQ1** 도로망 **고정 인접행렬 vs 학습(adaptive) 인접행렬**은 예측 정확도에 어떤 차이를 주는가?
  - **RQ2** 다단계 예측에서 **오차가 지평(15→30→60분) 따라 어떻게 누적**되는가?
  - **RQ3** GraphCast 의 **격자→그래프→롤아웃** 패러다임이 교통 STGNN 에 어떻게 개념 대응되는가?
- **5. 예시 시나리오:** METR-LA 207센서에서 과거 60분(12스텝) 속도를 입력받아 미래 60분(12스텝)을
  예측하고, 15/30/60분 지평의 MAE/RMSE/MAPE 를 기준선(HA·copy-last) 대비 비교.
- **6. 실험 구조:** 1) mini/실데이터 준비 → 2) 기준선(HA·copy-last) → 3) 소형 STGNN 학습 →
  4) horizon 3/6/12 평가 → 5) 절제(인접행렬 모드) → 6) 오차 누적 분석.
- **7. 실험 설계(표):**

  | 구성요소 | 축소 재현 설정 |
  |---|---|
  | 목표 태스크 | 다단계 교통 속도 예측 (h=3/6/12 = 15/30/60분) |
  | 데이터 | METR-LA(207) / PEMS-BAY(325), 5분 간격, 7:1:2 시간분할 |
  | 모델 | 소형 STGNN (Graph WaveNet-lite: 확산 그래프 conv + 게이트드 시간 conv) |
  | 지표 | MAE / RMSE / MAPE (masked, null=0), 지평별 |
  | 비교군(기준선) | Historical Average, copy-last(persistence) |
  | **절제 1 (RQ1)** | 인접행렬: `fixed` vs `learned` vs `hybrid` vs `identity`(그래프 제거) |
  | **절제 2 (RQ2)** | 지평별 오차 누적: 스텝별 MAE 발산(slope·final/first) 관찰 |
  | **절제 3 (RQ3)** | GraphCast 격자→그래프→롤아웃 ↔ 교통 STGNN 개념 대응 정리(아래 표) |
- **8. 학습자가 배울 수 있는 점:** 그래프 신경망의 공간 모델링, 확산 합성곱(DCRNN)·adaptive
  인접행렬(Graph WaveNet), 다단계 예측 오차 누적, 마스크드 지표, GraphCast 류 격자-그래프 예측의
  공통 원리, 게이밍 PC 에서의 **실제 학습**(과제01의 추론중심과 대비).
- **9. 상세 설명:** §"재현 방법" 참조. 범위 축소는 (a) 소형 모델(hidden 32, 2 layer), (b) 단일
  데이터셋(METR-LA) 우선, (c) 짧은 에폭·조기종료로 3060 8GB 내 학습.
- **10. 최초 프롬프트:** "DCRNN/Graph WaveNet 을 METR-LA 로 축소 재현. 스캐폴딩·기준선·평가
  골격(Phase 0)부터 로컬에서. 인접행렬 고정 vs 학습 절제와 다단계 오차 누적을 실험에 포함.
  데이터·가중치는 커밋 금지, GraphCast 패러다임 대응을 문서화."
- **11. 난이도:** 전반 **중**. ① 데이터 취득: **하~중**(연구용 공개, Google Drive 링크).
  ② 게이밍PC 구동: **중**(소형 STGNN 은 3060 8GB 에서 **실제 학습 가능** — 과제01 대비 쉬움).
- **12. 태그:** #traffic #forecasting #stgnn #gnn #dcrnn #graph-wavenet #graphcast #spatiotemporal

## 앵커 · 브리지 논문 대응 (RQ3: GraphCast ↔ 교통 STGNN)

| GraphCast (Lam+ 2023, 지구 기상) | 교통 STGNN (본 과제) |
|---|---|
| 지구 표면을 다중해상도 **격자 → 그래프**(정이십면체 메시)로 변환 | 도로망 **센서를 노드**로, 도로 인접/거리를 엣지로 하는 그래프 |
| 노드 특징 = 기상 상태(기온·기압·바람…) | 노드 특징 = 센서 속도(과거 관측 윈도우) |
| GNN 메시지 패싱으로 공간 상호작용 | 확산 그래프 합성곱(DCRNN) / adaptive 그래프(Graph WaveNet) |
| 한 스텝(6h) 예측 후 **자기회귀 롤아웃**으로 장기 예보 | 다단계(15/30/60분) 예측, 지평 따라 **오차 누적** 관찰(RQ2) |
| 고정 메시(지구 기하) | **고정 vs 학습** 인접행렬 절제(RQ1) — 교통은 그래프 정의 자체가 열린 문제 |

> 요지: 두 문제 모두 "**격자/센서를 그래프로 보고, GNN 으로 다음 상태를 예측·롤아웃**"하는 동일
> 패러다임. GraphCast 는 전지구 물리, 교통 STGNN 은 그 도시 규모 축소판으로 대응된다.

## 대표 논문·라이선스
> 방침: **서지정보(링크·DOI)만 기록**한다. PDF 는 저장소에 넣지 않는다. BibTeX 는 루트 [`CITATIONS.md`](../../CITATIONS.md).

- **앵커 논문:** Lam, R., Sanchez-Gonzalez, A., Willson, M., et al. (2023). *Learning skillful
  medium-range global weather forecasting*. **Science, 382(6677), 1416–1421.**
  DOI [10.1126/science.adi2336](https://doi.org/10.1126/science.adi2336). SCI(E) **O**. (GraphCast)
- **브리지 논문 1:** Li, Y., Yu, R., Shahabi, C., & Liu, Y. (2018). *Diffusion Convolutional
  Recurrent Neural Network: Data-Driven Traffic Forecasting*. **ICLR 2018.**
  arXiv [1707.01926](https://arxiv.org/abs/1707.01926). 학회 논문(SCI(E) 해당없음). (DCRNN)
- **브리지 논문 2:** Wu, Z., Pan, S., Long, G., Jiang, J., & Zhang, C. (2019). *Graph WaveNet for
  Deep Spatial-Temporal Graph Modeling*. **IJCAI 2019.**
  DOI [10.24963/ijcai.2019/264](https://doi.org/10.24963/ijcai.2019/264) · arXiv [1906.00121](https://arxiv.org/abs/1906.00121). (Graph WaveNet)
- **데이터 라이선스:** METR-LA / PEMS-BAY — **연구용 공개**(DCRNN 저장소 배포, 원 PeMS 데이터는
  Caltrans 제공). 재배포보다 원 출처 링크 사용 권장.
- **코드 라이선스:** 참조 구현(DCRNN, Graph WaveNet)은 각 저장소 라이선스 확인 후 명시 — Phase 1 확정.

### 논문에서 확정한 설정 (config/문서 반영, 성능수치는 *참조용*)
- **시간 프로토콜:** 과거 **12스텝(60분)** → 미래 **12스텝(60분)**, 5분 간격. 평가 지평 **3/6/12 = 15/30/60분**.
- **표준 지표:** **MAE · RMSE · MAPE**(결측=0 마스크). — `src/metrics.py` 에 구현.
- **기준선 정의:** `copy-last`(persistence, 마지막 관측 복사) / `Historical Average`(관측 평균; 계절형은 Phase 1).
- **데이터 분할:** 시간순 **7:1:2**(train/val/test) — DCRNN 표준.
- ⚠️ 논문 보고 성능수치는 **참조용**일 뿐, 우리 결과 표에 절대 옮겨 적지 않는다(재현성 규칙).

## 재현 방법 (실제 명령) — Phase 진행에 따라 채움
```bash
# 0) 환경 (과제 01 과 분리된 경량 스택)
python -m venv .venv-traffic && source .venv-traffic/bin/activate   # Windows: .venv-traffic\Scripts\activate
pip install -r tasks/task-02-traffic-stgnn/requirements.txt

# 1) 데이터(연구용 공개, METR-LA) — Phase 1 실제 취득 방법(gdown, 공식 DCRNN Drive 폴더)
pip install gdown
bash scripts/download_data.sh --subset metr-la --fetch   # opt-in 취득 + 배치검증
#   (--fetch 없이 실행하면 안내+배치검증만. 데이터는 data/ 로 gitignore 차단, 커밋 안 함)

# 2) 스모크런 — 합성 시계열로 기준선→지표 파이프라인 검증(실데이터·모델 불필요, numpy)
bash scripts/smoke.sh

# 3) 기준선 실측(Phase 1) — 실 METR-LA test split, MAE/RMSE/MAPE @3/6/12 + 오차 누적
python scripts/eval_baselines.py            # → results/<run_id>/{metrics.json,summary.md}

# 4) 모델 wiring 검증(선택, torch 필요) — 합성 배치 1스텝 순전파/역전파
python scripts/train.py --dry-run

# 5) STGNN 학습 → 평가 — Phase 2 (예정)
# python scripts/train.py --config config/base.yaml
# python scripts/eval.py  --config config/base.yaml --ckpt results/<run_id>/best.pt
```

## 실행 계획 (Phase)
- **Phase 0 (스캐폴딩):** 로컬 3060. ✅ 완료 — 폴더/지표/기준선/모델 골격/스모크/문서.
- **Phase 1 (데이터 취득 + 기준선 실측):** 로컬(CPU). ✅ 완료 — METR-LA 취득(gdown, 연구용 공개) →
  윈도우 12→12·시간순 70/10/20 분할·z-score 스케일러(train만) → copy-last·seasonal-HA 실측 →
  horizon 3/6/12 MAE/RMSE/MAPE + 오차 누적(RQ2). 재현: `python scripts/eval_baselines.py`.
- **Phase 2 (STGNN 학습·절제):** 로컬 3060. ⏳ 예정 — 소형 STGNN 실제 학습 → 기준선 대비 →
  인접행렬 fixed/learned 절제(RQ1) → 하드웨어(VRAM/시간) 기록.

## 결과 (실제 실행값만 기입) — Phase 1, 실 METR-LA **test** split (6,850 윈도우), 원 단위(mph)·masked
> 재현: `python scripts/eval_baselines.py` → `results/<run_id>/{metrics.json,summary.md}`
> (seed=42, 시간순 train 23,974 / val 3,425 / test 6,850, 결측=0 마스크).

| 모델 | MAE@15m | MAE@30m | MAE@60m | RMSE@60m | MAPE@60m(%) | MAE 기울기(스텝당) |
|---|---|---|---|---|---|---|
| copy-last (persistence) | 4.017 | 5.094 | 6.795 | 14.209 | 16.71 | **+0.332** (누적) |
| seasonal-HA (DCRNN 정의) | 4.187 | 4.187 | 4.187 | 7.852 | 13.03 | ~0.000 (평탄) |
| STGNN (fixed adj) | — | — | — | — | — | — (Phase 2, RQ1) |
| STGNN (learned adj) | — | — | — | — | — | — (Phase 2, RQ1) |
| *(참조용)* *논문 HA (Li+ 2018 Table 1)* | *4.16* | *4.16* | *4.16* | *7.80* | *13.0* | *~0 (평탄)* |

> ⚠️ 논문 HA 는 **참조용**(full METR-LA, HA 주기/결측처리 세부 차이). 우리 seasonal-HA(4.19)와 근접하나
> 직접 비교는 주의하며 **우리 결과로 옮겨 적지 않는다**. copy-last 는 논문 미보고(참조값 없음).

### RQ2 — 다단계 오차 누적 (기준선에서 먼저 관찰)
- **copy-last:** MAE 가 지평 따라 **단조 증가** 3.05→…→6.80(@5→@60분), 스텝당 +0.332, @60분/@5분 **2.23×**.
  persistence 는 시간이 갈수록 마지막 관측이 낡아 오차가 쌓인다 — 전형적 누적.
- **seasonal-HA:** MAE 가 모든 지평에서 **평탄**(≈4.19, 기울기 0). 타깃 시각의 주(week) 계절 슬롯만
  보므로 "얼마나 먼 미래냐"와 무관 → 누적 없음.
- 대조 요지: **오차 누적은 예측 방식의 성질**이다. Phase 2 STGNN 이 이 누적을 얼마나 완화하는지가 관전 포인트.

## 한계 / 미확인
- **STGNN 미학습**(Phase 2). 위 결과 표의 STGNN 행은 공란(값 지어내지 않음). 현재는 기준선만 실측.
- STGNN(`src/model.py`)은 **import 만 검증된 골격**이다(로컬에 torch 미설치). 순전파/형상·학습·성능은
  Phase 2(torch 설치 후 `python scripts/train.py --dry-run` 형상 assert → 실학습) 에서 확인한다.
- **seasonal-HA vs 논문 HA**: 4.19 vs 4.16 으로 근접하나 HA 주기(주간)·결측 마스킹 구현 차이가 있어
  **직접 비교 아님**(참조용). copy-last 는 논문 미보고라 참조값 자체가 없음.
- **PEMS-BAY 미실측**: 파일은 취득했으나(gitignore) 이번 실측은 METR-LA 만. PEMS-BAY 는 Phase 2 확장.
- **z-score 스케일러는 저장만**(`data/processed/scaler.json`, gitignore) — Phase 2 STGNN 학습용이며 기준선
  지표는 원 단위(mph)로 계산했다.
- `smoke.sh` 수치는 합성 더미이며 성능 보고가 아니다(`synthetic_dummy=true`, 위 실측과 별개).

## 완료 정의 (DoD) 체크
**Phase 0 (스캐폴딩) — 완료**
- [x] `smoke.sh` 통과(합성 기준선→지표 파이프라인, numpy-only)
- [x] 지표(MAE/RMSE/MAPE masked, 지평별)·기준선(copy-last·HA)·STGNN 골격·그래프 유틸 구현
- [x] 시드 고정(`common/seeding.py`) + 경량 환경 파일(`requirements.txt`, 과제01과 분리)
- [x] 데이터 git 미포함(.gitignore 차단) + `download_data.sh` + 라이선스 명시
- [x] 앵커/브리지 논문 인용(GraphCast DOI·SCI(E) O / DCRNN·Graph WaveNet)
- [x] GraphCast 격자→그래프→롤아웃 개념 대응 문서화(RQ3)

**Phase 1 (데이터 취득 + 기준선 실측) — 완료**
- [x] METR-LA 취득(gdown, 연구용 공개) — data/(gitignore), 커밋 없음. 검증 (T,N)=(34272,207)·결측 8.11%
- [x] 전처리: 윈도우 12→12 · 시간순 70/10/20 · z-score 스케일러(train만, 저장)
- [x] copy-last·seasonal-HA **실측** — MAE/RMSE/MAPE @15/30/60분 (`results/`, 실제 값)
- [x] 다단계 오차 누적(RQ2) 기준선에서 관찰·기록 — copy-last 누적 vs HA 평탄
- [x] 논문값은 참조용으로만 분리 표기(직접 비교 주의)

**미완 — Phase 2 이월**
- [ ] 소형 STGNN **실제 학습** + 기준선 대비 지표 표(results/, 실제 값)
- [ ] 절제 실험(인접행렬 fixed/learned/hybrid/identity) 실측 — RQ1
- [ ] 게이밍PC(3060 8GB) 실제 학습 시간·VRAM 기록
- [ ] (선택) PEMS-BAY 확장
