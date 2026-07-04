"""HuggingFace Hub 업로드 헬퍼 — 화이트리스트(허용 확장자) 방식.

설계 원칙 (PROJECT_GUIDELINE.md §4, 사용자 지시):
  - 블랙리스트(정규식 거부)가 아니라 "화이트리스트(허용 확장자만)"로 만든다.
  - 허용 확장자에 없는 파일 / 데이터 경로는 업로드 대상에서 "원천 제외"한다.
  - 업로드 직전, 허용 목록에 없는 파일이 스테이징에 포함됐는지 검사한다.
    포함됐으면 즉시 중단(RuntimeError)하고 사람에게 확인을 요청한다.

이 모듈은 Phase 0에서 "검사 로직"만 확정한다. 실제 push는 huggingface_hub가
설치되고(Phase 2+) 사용자가 명시적으로 실행할 때만 동작한다.

사용 예:
    python -m common.hub \
        --repo-id urbsn4i-sw/occworld-mini \
        --repo-type model \
        --src tasks/task-01-occworld-spatial \
        --dry-run            # 먼저 무엇이 올라갈지 확인 (기본값)

    # 실제 업로드는 --no-dry-run + HF_TOKEN 환경변수 필요
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# 허용 확장자 화이트리스트 (소문자, 점 포함). 이 목록에 없으면 업로드 불가.
ALLOWED_EXTENSIONS: frozenset[str] = frozenset(
    {".py", ".md", ".json", ".yaml", ".yml", ".png", ".gif", ".csv"}
)

# 확장자가 무엇이든 "경로에 이 세그먼트가 있으면" 무조건 제외 (데이터/비밀/캐시).
DENY_PATH_SEGMENTS: frozenset[str] = frozenset(
    {"data", "gts", "nuscenes", "occ3d", "samples", "sweeps", "__pycache__", "secrets"}
)

# 이미지로 취급할 확장자(화이트리스트를 통과하지만 '씬 덤프' 위험이 있는 것들).
IMAGE_EXTENSIONS: frozenset[str] = frozenset({".png", ".gif", ".jpg", ".jpeg"})

# 이미지 덤프 방어 기본 임계값 (assert_clean 인자로 조정 가능; 하드코딩 최소화).
#   - 화이트리스트는 .png/.gif 를 확장자로 허용하지만, "방법 설명용 소량 figure"와
#     "원본 씬을 복원한 대량 이미지 덤프"를 구분하지 못한다. 아래 값싼 가드로 메운다.
DEFAULT_MAX_IMAGE_COUNT = 20        # 이미지 개수 상한
DEFAULT_MAX_IMAGE_TOTAL_MB = 10.0   # 이미지 총 용량 상한(MB)
DEFAULT_MAX_SINGLE_IMAGE_MB = 2.0   # 개별 이미지 상한(MB) — 초과 시 '씬 덤프 의심'


def is_allowed(rel_path: Path) -> bool:
    """상대경로가 화이트리스트 규칙을 통과하면 True."""
    parts = {p.lower() for p in rel_path.parts}
    if parts & DENY_PATH_SEGMENTS:
        return False
    # .env 등 확장자 없는 비밀 파일 방어
    name = rel_path.name.lower()
    if name.startswith(".env"):
        return False
    return rel_path.suffix.lower() in ALLOWED_EXTENSIONS


def scan(src: Path) -> tuple[list[Path], list[Path]]:
    """src 트리를 훑어 (허용, 거부) 상대경로 목록을 돌려준다."""
    allowed: list[Path] = []
    rejected: list[Path] = []
    for path in sorted(src.rglob("*")):
        if not path.is_file():
            continue
        if ".git" in path.parts:
            continue
        rel = path.relative_to(src)
        (allowed if is_allowed(rel) else rejected).append(rel)
    return allowed, rejected


def _is_image(rel: Path) -> bool:
    return rel.suffix.lower() in IMAGE_EXTENSIONS


def check_image_dump(
    src: Path,
    allowed: list[Path],
    max_count: int = DEFAULT_MAX_IMAGE_COUNT,
    max_total_mb: float = DEFAULT_MAX_IMAGE_TOTAL_MB,
    max_single_mb: float = DEFAULT_MAX_SINGLE_IMAGE_MB,
) -> list[str]:
    """화이트리스트를 통과한 이미지들이 '대량 씬 덤프'로 의심되는지 검사.

    반환: 위반 메시지 목록(비어 있으면 통과). 실제 파일 크기(src/rel)를 stat 한다.
    """
    images = [r for r in allowed if _is_image(r)]
    violations: list[str] = []
    if not images:
        return violations

    sizes = {r: (src / r).stat().st_size for r in images}
    total_mb = sum(sizes.values()) / 1e6

    if len(images) > max_count:
        violations.append(
            f"이미지 개수 {len(images)}개 > 상한 {max_count}개 (대량 이미지 덤프 의심)"
        )
    if total_mb > max_total_mb:
        violations.append(
            f"이미지 총 용량 {total_mb:.1f}MB > 상한 {max_total_mb}MB (대량 덤프 의심)"
        )
    big = [(r, sz / 1e6) for r, sz in sizes.items() if sz / 1e6 > max_single_mb]
    for r, mb in sorted(big, key=lambda x: -x[1]):
        violations.append(f"개별 이미지 {mb:.1f}MB > 상한 {max_single_mb}MB: {r} (씬 덤프 의심)")
    return violations


def assert_clean(
    src: Path,
    max_image_count: int = DEFAULT_MAX_IMAGE_COUNT,
    max_image_total_mb: float = DEFAULT_MAX_IMAGE_TOTAL_MB,
    max_single_image_mb: float = DEFAULT_MAX_SINGLE_IMAGE_MB,
) -> list[Path]:
    """업로드 직전 게이트. (1) 화이트리스트 밖 파일 또는 (2) 이미지 덤프 의심이면 중단.

    Returns 허용 파일 목록. 위반이 있으면 RuntimeError(사람 확인 요청).
    임계값은 인자로 조정 가능(하드코딩 최소화).
    """
    allowed, rejected = scan(src)
    if rejected:
        lines = "\n".join(f"  - {r}" for r in rejected)
        raise RuntimeError(
            "업로드 중단: 허용 목록(화이트리스트)에 없는 파일이 포함됐습니다.\n"
            f"허용 확장자: {sorted(ALLOWED_EXTENSIONS)}\n"
            f"거부된 파일 {len(rejected)}개:\n{lines}\n"
            "→ 위 파일들을 제거하거나, 정말 업로드가 필요하면 사람에게 확인을 받으세요."
        )

    img_violations = check_image_dump(
        src, allowed, max_image_count, max_image_total_mb, max_single_image_mb
    )
    if img_violations:
        lines = "\n".join(f"  - {v}" for v in img_violations)
        raise RuntimeError(
            "업로드 중단: 이미지 덤프 방어 게이트에 걸렸습니다.\n"
            f"{lines}\n"
            "→ '방법 설명용 소량 figure'만 올리세요. 정말 필요하면 사람 확인 후 "
            "임계값(max_image_count/total_mb/single_mb)을 명시적으로 올려 재실행하세요."
        )
    return allowed


def upload(repo_id: str, repo_type: str, src: Path, dry_run: bool = True) -> None:
    allowed = assert_clean(src)  # 게이트 통과 못 하면 여기서 RuntimeError

    print(f"[hub] src={src}  repo={repo_type}:{repo_id}")
    print(f"[hub] 업로드 대상 {len(allowed)}개 (화이트리스트 통과):")
    for rel in allowed:
        print(f"  + {rel}")

    if dry_run:
        print("[hub] --dry-run: 실제 업로드는 하지 않았습니다. (--no-dry-run으로 실행)")
        return

    token = os.environ.get("HF_TOKEN")
    if not token:
        sys.exit("[hub] 오류: 환경변수 HF_TOKEN 이 없습니다. 업로드 중단.")

    try:
        from huggingface_hub import HfApi  # noqa: PLC0415  (지연 임포트: Phase 0 미설치 허용)
    except ImportError:
        sys.exit(
            "[hub] huggingface_hub 미설치. Phase 2+ 환경에서 `pip install huggingface_hub` 후 실행."
        )

    api = HfApi(token=token)
    api.create_repo(repo_id=repo_id, repo_type=repo_type, exist_ok=True)
    for rel in allowed:
        api.upload_file(
            path_or_fileobj=str(src / rel),
            path_in_repo=str(rel).replace(os.sep, "/"),
            repo_id=repo_id,
            repo_type=repo_type,
        )
    print(f"[hub] 완료: {len(allowed)}개 업로드 → {repo_type}:{repo_id}")


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="HF Hub 화이트리스트 업로드")
    p.add_argument("--repo-id", required=True, help="예: urbsn4i-sw/occworld-mini")
    p.add_argument("--repo-type", default="model", choices=["model", "dataset", "space"])
    p.add_argument("--src", required=True, type=Path, help="업로드할 로컬 디렉토리")
    p.add_argument("--dry-run", dest="dry_run", action="store_true", default=True)
    p.add_argument("--no-dry-run", dest="dry_run", action="store_false")
    return p.parse_args(argv)


if __name__ == "__main__":
    args = _parse_args()
    if not args.src.is_dir():
        sys.exit(f"[hub] 오류: src 디렉토리가 없습니다: {args.src}")
    upload(args.repo_id, args.repo_type, args.src, dry_run=args.dry_run)
