# 규칙: Python 코딩

- 식별자·경로·파일명은 영문. 주석·docstring은 한국어 기본(영문 병기 허용).
- 하드코딩 금지: 하이퍼파라미터·경로는 `config/*.yaml`로, CLI 오버라이드 허용.
- 모든 실행 스크립트는 시작부에서 `common.seeding.set_seed(cfg.seed)` 호출.
- 선택적 무거운 의존성(torch/mmcv 등)은 **지연 임포트**로 감싸 Phase 0 미설치 환경에서도 유틸이 import 되게 한다.
- 표준 라이브러리 우선, 의존성 추가는 최소화하고 `environment.yml`에 핀 버전으로 기록.
