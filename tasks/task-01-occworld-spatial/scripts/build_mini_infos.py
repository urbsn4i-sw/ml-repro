#!/usr/bin/env python
"""nuScenes v1.0-mini temporal info 생성(전략 [A], 무설치).

json 테이블만 읽어 씬별 ego 궤적/temporal info 를 만들고 pkl 로 저장한다.
"몇 개 씬이 잡히는지"를 출력해 검증한다(재현성 규칙: 생성 절차·검증 기록).

pkl 은 대용량 아님(궤적 좌표만)이나, 규칙상 *.pkl 은 .gitignore 로 커밋 차단된다.
"""
from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

_HERE = Path(__file__).resolve()
_TASK_DIR = _HERE.parents[1]
sys.path.insert(0, str(_TASK_DIR / "src"))

import nusc_infos as NI  # noqa: E402


def main(argv=None) -> int:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except (AttributeError, ValueError):
            pass

    p = argparse.ArgumentParser(description="nuScenes mini temporal info 생성")
    p.add_argument("--nusc-root", type=Path, default=_TASK_DIR / "data" / "nuscenes")
    p.add_argument("--version", default="v1.0-mini")
    p.add_argument("--out", type=Path, default=_TASK_DIR / "data" / "nuscenes_mini_infos.pkl")
    args = p.parse_args(argv)

    if not (args.nusc_root / args.version).is_dir():
        sys.exit(f"[infos] 오류: 메타 디렉토리 없음: {args.nusc_root / args.version}\n"
                 f"        download_data.sh 안내대로 v1.0-mini 를 배치했는지 확인.")

    infos = NI.build_scene_trajectories(args.nusc_root, args.version)
    c = infos["counts"]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "wb") as f:
        pickle.dump(infos, f)

    print(f"[infos] version={infos['version']}  → 저장: {args.out}")
    print(f"[infos] 씬 {c['num_scenes']}개 (train {c['num_train_scenes']} / "
          f"val {c['num_val_scenes']} / unknown {c['num_unknown_scenes']}), "
          f"키프레임 총 {c['num_keyframes']}개")
    for name, s in sorted(infos["scenes"].items()):
        print(f"    - {name} [{s['split']}]: {len(s['ego_xy'])} keyframes")

    # 검증: mini 는 씬 10, train8/val2 가 기대치
    ok = (c["num_scenes"] == 10 and c["num_train_scenes"] == 8 and c["num_val_scenes"] == 2)
    print(f"[infos] 기대치(씬10/train8/val2) 일치: {ok}")
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
