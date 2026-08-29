"""
Deploy the API to a Hugging Face Space.

Assembles a staging folder with exactly what the Space needs, then uploads
it. The GitHub repository keeps its own README; the Space gets the one in
deploy/space/ with the YAML configuration Spaces requires.

Usage:
    python scripts/deploy_space.py
    python scripts/deploy_space.py --dry-run
"""

import argparse
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from config.config import (  # noqa: E402
    HF_MODEL_REPO,
    HF_SPACE_REPO,
    HF_TOKEN,
    PROJECT_ROOT,
)

# Exactly what the container needs. Nothing else goes near the Space.
INCLUDE_DIRECTORIES = ["src", "config"]
INCLUDE_FILES = ["Dockerfile", "requirements-serve.txt", ".dockerignore"]


def build_staging(destination: Path) -> None:
    """Copy the Space contents into a temporary folder."""
    for name in INCLUDE_DIRECTORIES:
        shutil.copytree(
            PROJECT_ROOT / name,
            destination / name,
            # __pycache__ is compiled bytecode for your machine. It is
            # useless in the image and would only add size.
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
        )

    for name in INCLUDE_FILES:
        source = PROJECT_ROOT / name
        if source.exists():
            shutil.copy2(source, destination / name)

    # The Space README, with the YAML block Spaces reads for configuration.
    shutil.copy2(
        PROJECT_ROOT / "deploy" / "space" / "README.md", destination / "README.md"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Deploy the API to a Space.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Assemble and list the contents without uploading.",
    )
    args = parser.parse_args()

    if not HF_SPACE_REPO:
        print("ERROR: HF_SPACE_REPO is not set. Add it to .env, for example:")
        print("  HF_SPACE_REPO=your-username/ieee-cis-fraud-api")
        sys.exit(1)

    if not args.dry_run and not HF_TOKEN:
        print("ERROR: HF_TOKEN is not set. See step6.md section 1.")
        sys.exit(1)

    with tempfile.TemporaryDirectory() as temporary:
        staging = Path(temporary) / "space"
        staging.mkdir()
        build_staging(staging)

        files = sorted(
            path.relative_to(staging).as_posix()
            for path in staging.rglob("*")
            if path.is_file()
        )
        total_kb = (
            sum(path.stat().st_size for path in staging.rglob("*") if path.is_file())
            / 1024
        )

        print(f"Staged {len(files)} files, {total_kb:.0f} KB:")
        for name in files[:25]:
            print(f"  {name}")
        if len(files) > 25:
            print(f"  ... and {len(files) - 25} more")

        if args.dry_run:
            print("\nDry run. Nothing uploaded.")
            return

        from huggingface_hub import HfApi

        api = HfApi(token=HF_TOKEN)

        print(f"\nCreating or updating {HF_SPACE_REPO} ...")
        api.create_repo(
            repo_id=HF_SPACE_REPO,
            repo_type="space",
            space_sdk="docker",
            exist_ok=True,
        )

        # The Space needs to know which Model Hub repository to pull from.
        # This is a variable rather than a secret because the model repo is
        # public, so there is nothing sensitive about the name. D-67.
        api.add_space_variable(
            repo_id=HF_SPACE_REPO, key="HF_MODEL_REPO", value=HF_MODEL_REPO
        )

        print("Uploading ...")
        api.upload_folder(
            folder_path=str(staging),
            repo_id=HF_SPACE_REPO,
            repo_type="space",
            commit_message="Deploy fraud detection API",
        )

        print(f"\nDone: https://huggingface.co/spaces/{HF_SPACE_REPO}")
        print("The first build takes about five to ten minutes.")
        print("Watch progress on the Space's Logs tab.")


if __name__ == "__main__":
    main()
