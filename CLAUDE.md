# CLAUDE.md — ML 재현 프로젝트 운영 규약 (Claude Code)

> 이 파일은 Claude Code가 매 세션 자동으로 읽는 프로젝트 메모리다. **작고 안정적**으로 유지한다.
> 상세 규약은 `PROJECT_GUIDELINE.md`, 세부 코딩 규칙은 `.claude/rules/`를 참조한다.
> Codex와 병행 시 `AGENTS.md`와 항상 동일하게 유지한다.

## 프로젝트 개요
- **3D 점유 월드모델(OccWorld / DreamerV3)** 을 게이밍PC/Colab 수준에서 **축소 재현·공개**하는 저장소. 원래 더 큰 재현 시리즈의 일부였고, **교통 예측**은 별도 저장소([urban-traffic-forecasting](https://github.com/urbsn4i-sw/urban-traffic-forecasting))로 분리됨(nav/LiDAR는 계획 단계).
- 목적은 SOTA 재현이 아니라 **원리 재현·학습**. 표준 지표·논문은 `PROJECT_GUIDELINE.md` §8.

## 절대 규칙 (충돌 시 이 규칙이 우선)
1. **결과를 지어내지 않는다.** 실제 실행 로그/`metrics.json`의 값만 보고. 미검증·실패는 "미확인/한계"로 명시.
2. **대용량·비밀 커밋 금지.** 데이터셋·체크포인트(`*.ckpt/*.pt/*.safetensors`)·`.env`·토큰은 절대 커밋하지 않는다.
3. **재현성 우선.** 모든 스크립트는 `common/seeding.py`로 시드 고정, config(yaml) 기반, 하드코딩 금지.
4. **소규모부터.** 본 학습 전 `scripts/smoke.sh`(소량·1~2스텝)로 파이프라인 검증. 데이터는 항상 `--subset mini` 우선.
5. **논문·라이선스 인용.** 과제 README에 앵커/브리지 논문(DOI·SCI(E) 여부)과 데이터/코드 라이선스 명시.

## 작업 방식
- 작업 1건 = GitHub Issue 1건. 브랜치 `feat/<slug>-...`, 작은 단위 커밋(각 커밋은 통과 상태 지향).
- 구조를 바꾸는 큰 작업은 **계획을 먼저 제시하고 승인 후 실행**.
- 표준 실행 순서: 과제카드 → 이슈 → 데이터(mini) → **기준선 먼저** → 본 모델 → 평가 → 데모 노트북 → 문서 → PR(DoD 통과). 상세는 `PROJECT_GUIDELINE.md` §5~§6.

## 저장소 지도 (한눈에)
- `PROJECT_GUIDELINE.md` — 마스터 지침(사람+에이전트)
- `.claude/rules/` — 모듈 규칙(python / reproducibility / data-and-hub)
- `common/` — 시드·지표·HF 헬퍼
- `tasks/task-01-occworld-spatial/` — OccWorld 과제 (README 12항목 카드 · config · src · scripts · notebooks · results)

## 자주 쓰는 명령
- 스모크런: `bash tasks/<id>/scripts/smoke.sh`
- 학습: `python tasks/<id>/scripts/train.py --config tasks/<id>/config/base.yaml`
- 평가: `python tasks/<id>/scripts/eval.py --config ... --ckpt <path>`
- 데이터: `bash tasks/<id>/scripts/download_data.sh --subset mini`

## 하지 말 것
- 돌리지 않은 결과를 성능으로 보고하기 / 임의 URL 실행 / 라이선스 미확인 데이터 사용 / 루트 문서를 장문으로 부풀리기.
