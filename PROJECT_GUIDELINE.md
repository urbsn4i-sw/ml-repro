# 프로젝트 지침 (PROJECT_GUIDELINE) — ML 재현 과제 실행용

> 이 문서는 앞서 정리한 **재설계 과제(1: 3D 점유 월드모델 / 2: 도시 교통 시공간 예측 / 3: 교차 임바디먼트 내비 / 4: Mamba 기반 LiDAR·점군)** 를 GitHub + HuggingFace에서 Claude Code 또는 Codex로 **축소 재현·공개**하기 위한 실행 규약이다.
> 사람과 AI 에이전트가 함께 읽는 "프로젝트 헌법 + 운영 매뉴얼"이다. 세부 코딩 규칙은 `.claude/rules/`(Claude Code)와 `AGENTS.md`(Codex)에 동기화한다.

---

## 0. 적용 범위와 기본 전제
- **목표:** 각 과제를 게이밍 PC / Colab 수준에서 돌아가는 **축소 재현 리포**로 구현하고, GitHub 공개 + (선택)HuggingFace Hub 배포까지 한다. SOTA 재현이 아니라 **원리 재현·학습**이 목적이다.
- **기본 구조:** 모노레포(하나의 저장소에 과제별 폴더). 과제별로 저장소를 나누고 싶으면 `tasks/<id>/`를 각각 독립 리포로 승격하면 된다.
- **언어:** 문서·주석·커밋 메시지는 한국어 기본(영문 병기 허용). 식별자·경로는 영문.

## 1. 최우선 원칙 (에이전트가 반드시 지킴)
1. **결과를 지어내지 않는다.** 실제 실행 로그·`metrics.json`에서 나온 수치만 보고한다. 돌리지 않은 결과, 추정 성능을 사실처럼 쓰지 않는다. 미검증·실패는 "미확인/한계"로 정직하게 남긴다. (원본 정리 지침의 사실성 원칙 계승)
2. **재현성 우선.** 시드 고정, 환경 동결, config 기반 실행. "그때는 됐다"가 아니라 "이 명령으로 재현된다".
3. **컴퓨트 예산 준수.** 각 과제의 난이도 항목(데이터 취득성·게이밍PC 구동성)을 지키고, 항상 소규모 서브셋·스모크런부터 시작한다.
4. **논문·라이선스 인용 유지.** 각 과제 README에 앵커/브리지 논문(저자·연도·학회·DOI·SCI(E) 여부)과 데이터·코드 라이선스를 명시한다.
5. **대용량·비밀은 git에 넣지 않는다.** 데이터셋·체크포인트·API 키·토큰은 커밋 금지(§4, §7).

## 2. 저장소 구조
```
repo-root/
├─ README.md                  # 프로젝트 개요, 과제 목록, 재현 방법 요약
├─ PROJECT_GUIDELINE.md       # (이 문서) 마스터 지침
├─ CLAUDE.md                  # Claude Code용 운영 규약(작고 안정적, 인덱스)
├─ AGENTS.md                  # Codex/기타 에이전트용(= CLAUDE.md와 동기화)
├─ .claude/
│  └─ rules/                  # Claude Code 모듈 규칙(경로 스코프 가능)
│     ├─ python.md
│     ├─ reproducibility.md
│     └─ data-and-hub.md
├─ environment.yml            # conda 환경(권장) 또는
├─ requirements.txt           # pip 핀 버전
├─ pyproject.toml             # (선택) 패키징/도구 설정
├─ .gitignore                 # data/, checkpoints/, *.ckpt, .env 등 제외
├─ .gitattributes             # (필요 시) Git LFS 설정
├─ common/                    # 공통 유틸(시드, 로깅, 지표, HF 헬퍼)
│  ├─ seeding.py
│  ├─ metrics.py
│  └─ hub.py
└─ tasks/
   ├─ task-01-occworld-spatial/
   │  ├─ README.md            # 12항목 과제 카드 + 재현 명령 + DoD 체크
   │  ├─ config/              # yaml 설정(하이퍼파라미터)
   │  ├─ src/                 # 모델/데이터/학습 로직
   │  ├─ scripts/             # download_data.sh, train.py, eval.py, smoke.sh
   │  ├─ notebooks/           # 축소 재현 데모(Colab 가능)
   │  ├─ results/             # metrics.json, 표(csv/md), 작은 figure만
   │  ├─ MODEL_CARD.md        # (배포 시) 모델 카드
   │  └─ DATASET_CARD.md      # (해당 시) 데이터 카드
   ├─ task-02-traffic-stgnn/
   ├─ task-03-crossembodiment-nav/
   └─ task-04-pointmamba-slam/
```
- 루트 문서는 **작고 안정적**으로 유지한다. 깊고 경로별인 규칙은 `.claude/rules/`로 분리한다.

## 3. 환경·재현성 규약
- **Python 버전 고정**(예: 3.10/3.11), 의존성은 **핀 버전**으로 기록(`requirements.txt` 또는 `environment.yml`).
- **시드 고정**: `common/seeding.py`로 `random / numpy / torch`(+`cudnn.deterministic`) 일괄 설정. 모든 스크립트 시작부에서 호출.
- **설정 관리**: 하이퍼파라미터는 `config/*.yaml`에. CLI로 오버라이드(예: `--config config/base.yaml lr=1e-3`). 하드코딩 금지.
- **런 메타데이터 기록**: 각 실행은 `results/<run_id>/`에 config 스냅샷·git commit hash·하드웨어(GPU명/VRAM)·소요시간·최종 지표를 남긴다.
- **스모크런 필수**: 본 학습 전에 `scripts/smoke.sh`(소량 데이터·1~2 스텝)로 파이프라인이 도는지 먼저 확인.

## 4. 데이터·모델 아티팩트 취급 (HuggingFace 규약)
- **git에 넣지 않는 것**: 원천 데이터셋, 대용량 전처리물, 체크포인트(`*.ckpt`, `*.pt`, `*.safetensors`), 로그 대용량, `.env`/토큰.
- **데이터 취득**: `scripts/download_data.sh`로 원 출처 또는 **HF `datasets`/Hub**에서 내려받는다. 항상 **소규모 서브셋 옵션**을 제공(`--subset mini`).
- **가중치/모델 공유**: 학습 결과 가중치는 **HF Hub 모델 리포**로 push(`huggingface_hub`), README에 `MODEL_CARD.md` 포함. git엔 다운로드/업로드 스크립트만.
- **결과물 커밋 대상**: `metrics.json`, 지표 표(csv/md), 소형 figure(수백 KB) 등 **작은 산출물만**.
- **라이선스·이용약관 명시**: 과제별로 데이터 라이선스를 README에 적는다. 특히 **nuScenes/Occ3D는 비상업 라이선스·등록 필요**, **METR-LA/PEMS-BAY·ModelNet40·ShapeNet 등은 연구용 공개**. Open X-Embodiment는 하위 데이터셋별 라이선스가 다르므로 사용한 서브셋의 라이선스를 각각 표기.
- **LFS는 최소화**: 정말 저장소에 둬야 할 중간 크기 산출물만 `.gitattributes`로 LFS 지정. 원칙은 "Hub에 두고 링크".

## 5. 과제 1건 실행 프로토콜 (표준 워크플로우)
각 과제는 아래 순서로 진행하고, 각 단계는 커밋 단위로 남긴다.
1. **과제 카드 확정** — `tasks/<id>/README.md`에 12항목(문제/왜 어려운가/RQ/실험 설계/최초 프롬프트/난이도/태그 등) + 대표 논문·라이선스 기입.
2. **이슈 생성** — GitHub Issue 1개 = 과제 1개. 본문에 DoD 체크리스트(§6) 복사.
3. **브랜치** — `feat/task-01-occworld` 형태.
4. **데이터** — `download_data.sh --subset mini` 로 소규모만.
5. **기준선 먼저** — 단순 기준선(예: copy-last, Historical Average, 무작위 스캔, 단일 도메인)부터 구현·측정. 이게 "비교 기준"이자 파이프라인 검증.
6. **본 모델** — 공개 구현(OccWorld / DCRNN·Graph WaveNet / OXE 정책 / PointMamba)을 통합하거나 축소 구현.
7. **평가** — `eval.py` → `results/metrics.json` + 지표 표(과제별 표준 지표: §8).
8. **데모 노트북** — Colab에서 돌아가는 축소 재현 노트북.
9. **문서화** — README 재현 명령, (배포 시) MODEL_CARD/DATASET_CARD 갱신.
10. **PR** — DoD 통과 확인 후 병합. 실패·한계는 "한계" 섹션에 정직하게.
11. **(선택) 배포** — HF Hub 모델/Spaces, GitHub Release.

## 6. 완료 정의 (Definition of Done) — PR 병합 전 체크
- [ ] 문서화된 단일 명령으로 재현됨: `bash scripts/smoke.sh` 통과 + `python scripts/train.py --config ...` 동작.
- [ ] **기준선 대비 지표 표**가 `results/`에 존재(수치는 실제 실행값).
- [ ] 시드 고정 + 환경 파일(`requirements.txt`/`environment.yml`) 존재.
- [ ] 데이터·가중치 git 미포함 + `download_data.sh` 존재 + 라이선스 명시.
- [ ] 대표 논문(앵커/브리지) 인용 + DOI/링크 + SCI(E) 여부.
- [ ] Colab/게이밍PC 구동 가능성 및 실제 사용 하드웨어 기록.
- [ ] 실패·미검증 항목은 "한계/미확인"으로 정직하게 기술.
- [ ] 비밀키·토큰·대용량 파일이 커밋에 없음(사전 훅/CI로 확인 권장).

## 7. 에이전트 협업 규칙 (Claude Code · Codex 공통)
- **정직성**: 결과 조작 금지. 실행되지 않은 코드의 성능을 보고하지 않는다.
- **작은 단위 커밋**: 각 커밋은 (가능하면) 통과 상태. 커밋 메시지에 "무엇/왜".
- **큰 변경은 계획 먼저**: 구조를 바꾸는 작업은 계획(plan)을 먼저 제시하고 승인 후 실행.
- **비밀·대용량 차단**: `.env`, 토큰, 데이터, 체크포인트 커밋 금지. 발견 시 즉시 중단·보고.
- **네트워크/다운로드**: 스크립트로만. 임의 URL 실행 금지. 라이선스 확인 후 사용.
- **CI에 강제 규칙 이관**: "반드시" 규칙(포맷·린트·비밀 스캔·대용량 차단)은 문서 문구가 아니라 pre-commit/CI로 강제.
- **동기화**: `CLAUDE.md`와 `AGENTS.md`는 항상 동일 내용 유지. 규칙을 바꾸면 둘 다 수정.

## 8. 과제별 표준 지표·데이터·대표 논문 (요약 표)

| 과제 | 표준 지표 | 축소 데이터 | 앵커 논문 | 브리지 논문 |
|---|---|---|---|---|
| 01 점유 월드모델 | 미래 mIoU, ego L2, 충돌률, 롤아웃 안정성 | nuScenes mini + Occ3D | Hafner+ 2025, *Nature*, 10.1038/s41586-025-08744-2 (SCI(E)) | Zheng+ 2024 *OccWorld*, ECCV, arXiv:2311.16038 |
| 02 교통 STGNN | MAE/RMSE/MAPE (h=3/6/12) | METR-LA, PEMS-BAY | Lam+ 2023, *Science*, 10.1126/science.adi2336 (SCI(E)) | Li+ 2018 *DCRNN* ICLR / Wu+ 2019 *Graph WaveNet* IJCAI |
| 03 교차 임바디먼트 내비 | 성공률, 미학습 도메인 성공률, ATE/RPE | OXE 서브셋 (+Habitat/KITTI) | O'Neill+ 2024 *Open X-Embodiment*, ICRA, 10.1109/ICRA57147.2024.10611477 | (앵커=브리지) |
| 04 PointMamba/SLAM | OA/mIoU/mAP, 처리량, 메모리, 길이 스케일링 | ModelNet40, ShapeNetPart (+SemanticKITTI 소셋) | Gu&Dao 2024 *Mamba*, COLM, arXiv:2312.00752 | Liang+ 2024 *PointMamba*, NeurIPS, arXiv:2402.10739 |

## 9. Claude Code / Codex 세팅 방법
- **Claude Code**: 루트에 `CLAUDE.md`(자동 로드, 세션 시작 시 주입) + `.claude/rules/*.md`(모듈 규칙, 경로 스코프 가능). 개인 로컬 설정은 `CLAUDE.local.md`(자동 gitignore). `claude` 실행 후 `/init`로 초안 생성 → 이 지침 기준으로 수정. 최신 규약은 공식 문서(code.claude.com/docs/en/memory) 확인.
- **Codex**: 루트에 `AGENTS.md`(= CLAUDE.md 동기화본)를 둔다. Codex는 이 파일을 프로젝트 규약으로 읽는다. 최신 규약은 OpenAI Codex 문서 확인.
- 두 도구를 함께 쓰면 **한쪽에서 규칙을 바꿀 때 반드시 다른 쪽도 갱신**한다(§7).

## 10. 시작하는 법 (초기 부트스트랩 프롬프트 예시)
아래를 Claude Code/Codex 세션에 그대로 던져 부트스트랩할 수 있다.

> "이 저장소를 PROJECT_GUIDELINE.md 규약대로 초기화해줘. (1) §2 구조로 폴더/파일 스캐폴딩, (2) `common/seeding.py`·`common/metrics.py` 작성, (3) `tasks/task-02-traffic-stgnn/`부터 시작: METR-LA mini 다운로드 스크립트, Historical Average 기준선, 소형 STGNN 학습/평가, `results/metrics.json`과 지표 표 생성. 큰 데이터·체크포인트는 커밋하지 말고, DoD 체크리스트를 이슈로 만들어줘. 먼저 계획을 보여주고 승인받은 뒤 실행해."

(진입장벽이 가장 낮은 **과제 02(교통 STGNN)**로 시작하는 것을 권장한다.)
