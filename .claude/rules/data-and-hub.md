# 규칙: 데이터 · HuggingFace Hub

- git에 넣지 않는 것: 원천 데이터셋, 대용량 전처리물(`*.pkl/*.npz/*.bin` 등), 체크포인트(`*.ckpt/*.pt/*.safetensors`), `.env`/토큰. `.gitignore`로 전 경로 차단.
- 데이터 취득은 `scripts/download_data.sh`로만. 임의 URL 실행 금지. 라이선스 확인 후 사용.
- **nuScenes/Occ3D는 비상업·등록 필요** — 자동 다운로드하지 말고 수동 취득 안내만 출력.
- HF 업로드는 **화이트리스트(허용 확장자: .py .md .json .yaml .yml .png .gif .csv)** 방식(`common/hub.py`).
  업로드 직전 `assert_clean()` 게이트가 허용 목록 밖 파일을 발견하면 **중단하고 사람에게 확인**.
- 커밋 대상 산출물은 작은 것만: `metrics.json`, 지표 표(csv/md), 소형 figure(수백 KB).
