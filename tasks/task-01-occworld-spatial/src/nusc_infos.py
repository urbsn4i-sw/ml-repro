"""nuScenes v1.0-mini temporal info 빌더 — **표준 라이브러리(json)만** 사용.

목적: nuscenes-devkit / mmdet3d 설치 없이, mini 메타데이터 json 테이블에서
씬별 ego 궤적(키프레임 순서)과 temporal info 를 만든다. (download_data.sh 전략 [A])

핵심 사실
  - nuScenes 키프레임(sample)은 2Hz. scene 은 first_sample_token 에서 next 링크로 이어짐.
  - 각 keyframe 의 ego 위치는 LIDAR_TOP sample_data 의 ego_pose(global translation) 사용.
  - ego translation 은 global 좌표계 → 예측 vs GT 의 L2 는 좌표계 강체변환에 불변.

공식 mini split (nuscenes-devkit `splits.py` 기준, 총 10 씬):
  - mini_train(8), mini_val(2).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

# nuscenes-devkit splits.py 의 공식 mini split (하드코딩이 아니라 '데이터셋 정의' 상수)
MINI_TRAIN = [
    "scene-0061", "scene-0553", "scene-0655", "scene-0757",
    "scene-0796", "scene-1077", "scene-1094", "scene-1100",
]
MINI_VAL = ["scene-0103", "scene-0916"]


def _load_table(meta_dir: Path, name: str) -> list[dict]:
    return json.loads((meta_dir / f"{name}.json").read_text(encoding="utf-8"))


def build_scene_trajectories(nusc_root: str | Path, version: str = "v1.0-mini") -> dict[str, Any]:
    """씬별 키프레임 ego 궤적을 만든다.

    반환 dict:
      {
        "version": ...,
        "scenes": {
            scene_name: {
                "sample_tokens": [...],           # 시간순 keyframe token
                "timestamps_us": [...],           # 각 keyframe timestamp(마이크로초)
                "ego_xy": [[x, y], ...],          # global ego translation (m)
                "ego_xyz": [[x, y, z], ...],
                "split": "train"|"val"|"unknown",
            }, ...
        },
        "counts": {...},
      }
    """
    root = Path(nusc_root)
    meta = root / version

    scenes = _load_table(meta, "scene")
    samples = _load_table(meta, "sample")
    sample_data = _load_table(meta, "sample_data")
    ego_poses = _load_table(meta, "ego_pose")

    sample_by_token = {s["token"]: s for s in samples}
    ego_by_token = {e["token"]: e for e in ego_poses}

    # sample_token → (keyframe) LIDAR_TOP 의 ego_pose_token
    #   keyframe LIDAR_TOP sample_data: is_key_frame=True 이고 filename 이 samples/LIDAR_TOP/...
    lidar_ego_of_sample: dict[str, str] = {}
    for sd in sample_data:
        if not sd.get("is_key_frame"):
            continue
        fn = sd.get("filename", "")
        if fn.startswith("samples/LIDAR_TOP") or "/LIDAR_TOP/" in fn:
            lidar_ego_of_sample[sd["sample_token"]] = sd["ego_pose_token"]

    def split_of(scene_name: str) -> str:
        if scene_name in MINI_TRAIN:
            return "train"
        if scene_name in MINI_VAL:
            return "val"
        return "unknown"

    out_scenes: dict[str, Any] = {}
    for sc in scenes:
        name = sc["name"]
        # first_sample_token → next 링크로 keyframe 순서 복원
        tokens: list[str] = []
        tok = sc["first_sample_token"]
        while tok:
            tokens.append(tok)
            tok = sample_by_token[tok]["next"]

        ts, xy, xyz = [], [], []
        for t in tokens:
            ego_tok = lidar_ego_of_sample.get(t)
            if ego_tok is None:
                # keyframe 에 LIDAR_TOP 이 없으면(비정상) 이 프레임은 건너뛴다 — 값을 지어내지 않음
                continue
            pose = ego_by_token[ego_tok]
            tr = pose["translation"]  # [x, y, z] global
            ts.append(sample_by_token[t]["timestamp"])
            xy.append([float(tr[0]), float(tr[1])])
            xyz.append([float(tr[0]), float(tr[1]), float(tr[2])])

        out_scenes[name] = {
            "sample_tokens": tokens,
            "timestamps_us": ts,
            "ego_xy": xy,
            "ego_xyz": xyz,
            "split": split_of(name),
        }

    counts = {
        "num_scenes": len(out_scenes),
        "num_keyframes": sum(len(s["ego_xy"]) for s in out_scenes.values()),
        "num_train_scenes": sum(1 for s in out_scenes.values() if s["split"] == "train"),
        "num_val_scenes": sum(1 for s in out_scenes.values() if s["split"] == "val"),
        "num_unknown_scenes": sum(1 for s in out_scenes.values() if s["split"] == "unknown"),
    }
    return {"version": version, "scenes": out_scenes, "counts": counts}
