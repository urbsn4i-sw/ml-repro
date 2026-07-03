# 점유 기준선 copy-last (실 Occ3D-nuScenes mini) — occ-copylast-mini-20260703T055134Z

- 프로토콜: 과거 4f→미래 6f @2Hz, mask_camera=True, free=17 제외
- 집계: 지평별 전역 confusion 누적. git `d5d6c4dad2` seed 42
- ⚠️ 논문 Copy&Paste 값은 reference_only(우리 결과 아님). annotations.json 교차확인은 미확인/한계.

## val (씬 2, 윈도우 63)

| 지표(%) | @1s | @2s | @3s | 6지평평균 |
|---|---|---|---|---|
| mIoU | 10.75 | 6.40 | 5.10 | 8.76 |
| IoU  | 27.21 | 21.11 | 18.35 | 24.13 |

## train (씬 8, 윈도우 251)

| 지표(%) | @1s | @2s | @3s | 6지평평균 |
|---|---|---|---|---|
| mIoU | 22.95 | 19.01 | 17.39 | 21.17 |
| IoU  | 40.40 | 34.84 | 32.37 | 37.41 |
