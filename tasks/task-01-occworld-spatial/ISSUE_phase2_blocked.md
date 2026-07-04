# Phase 2 blocked: OccWorld pretrained weights & temporal pkl unavailable (author Tsinghua cloud links inactive)

> GitHub Issue 초안. 원격이 붙으면 `gh issue create --title ... --body-file ...` 또는 웹으로 등록.
> Task: task-01-occworld-spatial / Branch: `feat/task-01-phase2-occworld`

## Labels
`task-01` · `phase-2` · `blocked` · `external-dependency`

## 요약 (Summary)
OccWorld 사전학습 추론(Phase 2)을 진행하려면 저자가 배포한 **사전학습 가중치**와 **temporal pkl**이 필요하다. 두 파일 모두 저자 **Tsinghua cloud(Seafile) 공유 링크**에만 있는데, 해당 공유가 **현재 비활성**이라 취득할 수 없다. 값을 지어낼 수 없으므로 **OccWorld 모델↔기준선 대비 지표는 공란으로 유지**한다(외부요인에 의한 한계).

## 필요한 파일 (Required, 미확보)
- `nuscenes_infos_val_temporal_v3_scene.pkl` — full val temporal pkl. 링크: `https://cloud.tsinghua.edu.cn/d/9e231ed16e4a4caca3bd/`
- 사전학습 ckpt (`epoch_125.pth` 등) — config `load_from='out/occworld/epoch_125.pth'`. 링크: `https://cloud.tsinghua.edu.cn/d/ff4612b2453841fba7a5/`
  (출처: OccWorld 원 리포 README "Preparing" 3–4단계, github.com/wzzheng/OccWorld)

## 시도한 것 / 관측 (What we tried)
- 호스트 `cloud.tsinghua.edu.cn` 자체는 도달 가능(TCP/TLS OK, HTTP 200).
- 그러나 두 공유 토큰(`ff4612…`, `9e231ed…`) 모두:
  - Seafile dirents API → `{"error_msg":"Share link ... not found."}`
  - `/d/<token>/` 페이지 → 200이지만 **7221바이트 generic SPA 셸**(파일 목록 없음)
  - 파일 직접 다운로드(`?p=/<name>&dl=1`) → pkl(수백 MB)이 아니라 **7221바이트 HTML 셸** 반환
- **대조 실험**: 무작위 가짜 토큰 `deadbeefdeadbeef0000` 도 **완전히 동일**하게 반응(200 셸 + API "not found"). → 실제 링크가 유효 링크처럼 동작하지 않음(만료/비활성/게이트 추정).
- 브라우저·wget/curl 어느 쪽으로도 실제 바이트 취득 실패. **억지 우회는 하지 않음**(정책·라이선스 준수).

## 우리가 이미 준비해둔 것 (Ready — 링크 복구 시 즉시 재개)
- **코드 조사(2-A) 완료**: `eval_metric_stp3.py` + `config/occworld.py` 요구사항 파악.
- **재생성 경로 = 전략 B(필터)**: 원 리포에 pkl **생성 스크립트 없음**(`pickle.dump`/`create_data`/`tools/` 부재, `dump(`은 `cfg.dump`뿐). 따라서 배포 full `val` pkl을 우리 mini 씬(**scene-0103, scene-0916** — 둘 다 nuScenes val ⊂)으로 **필터링**하면 유효한 mini val pkl 확보 가능.
  - pkl 구조: `data['infos']` = `{scene_name: [frame_info, ...]}`, 각 frame_info = `token`, `gt_ego_fut_trajs`, `pose_mode`, `ego2global_*`, `lidar2ego_*`, `cams{data_path, sensor2lidar_*, cam_intrinsic}`.
  - 필터 방법: `pickle.load` → `data['infos'] = {k: v for k,v in data['infos'].items() if k in {'scene-0103','scene-0916'}}` → `pickle.dump`. gts는 이미 10씬 보유.
- **점유 gts ✅ 보유**: `data/nuscenes/gts/{scene}/{token}/labels.npz` (mini 10씬, 200×200×16, free=17).
- **8GB OOM 예상 지점**: VQVAE 인코더 12프레임×200²×128ch 활성화 ≈2.5GB→배수로 8GB 초과 가능. 완화(eval 프레임 축소·`no_grad`·청크), 실패 시 **Colab(T4/L4 16GB) 이관**(사전 합의).
- **의존성 리스크**: eval도 `mmdet3d`/`mmcv`/`mmengine` 필수(dataset.py import). sm_86/py3.8 빌드가 관문.

## 재개 조건 (Unblock criteria)
아래 중 하나 충족 시 재개:
1. 저자 Tsinghua cloud 링크 **복구**, 또는
2. 가중치·temporal pkl의 **대체 미러/재배포** 확보(라이선스 확인 후).

## 재개 절차 (Next steps when unblocked)
1. **2-B2** — full val pkl 취득 → mini 씬 필터(전략 B) → mini val pkl 생성(gitignore 차단 위치).
2. **2-D** — py3.8 + mmcv/mmdet3d/spconv 환경 설치(WSL2). 빌드 실패 시 Colab 이관.
3. **2-E** — `eval_metric_stp3.py` smoke(소량) → OccWorld 미래 mIoU/IoU/ego-L2 실측 → 기준선 표에 모델 행 채움.

## 참고 (References)
- OccWorld: Zheng et al. (2024), ECCV, arXiv:2311.16038 / 코드 github.com/wzzheng/OccWorld
- 원 clone(리포 밖): `~/occworld-upstream` (커밋 안 됨; 재개 시 재사용 가능)
