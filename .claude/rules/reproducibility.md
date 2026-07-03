# 규칙: 재현성

- 시드 고정(random/numpy/torch + cudnn.deterministic), config 기반 실행.
- 각 실행은 `results/<run_id>/`에 config 스냅샷·git commit hash·하드웨어(GPU/VRAM)·소요시간·최종 지표를 남긴다.
- 본 학습/추론 전 `smoke.sh`(소량·1~2 스텝)로 파이프라인부터 검증.
- **결과를 지어내지 않는다.** `metrics.json` 등 실제 실행값만 보고. 미검증·실패는 "한계/미확인"으로 명시.
- 환경 이관(로컬→Colab 등)이 발생하면 **어디서 왜 이관했는지**를 README/results 로그에 기록.
