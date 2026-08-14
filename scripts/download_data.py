"""
Download the IEEE-CIS Fraud Detection dataset from Kaggle into data/raw.

Prerequisites:
  1. A Kaggle account that has JOINED the competition and accepted its rules.
     Without this, the download fails with a 403 error.
  2. Kaggle credentials set up (run `kaggle auth login`, or place kaggle.json
     in your user folder under .kaggle).

Usage:
  python scripts/download_data.py
  python scripts/download_data.py --force     # re-download even if files exist
"""

import argparse
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

# Add the project root to the list of places Python looks for imports.
# Without this line, running the script directly cannot find the config package,
# because Python only looks in the script's own folder by default.
sys.path.append(str(Path(__file__).resolve().parents[1]))

from config.config import (  # noqa: E402
    KAGGLE_COMPETITION,
    RAW_DATA_DIR,
    EXPECTED_RAW_FILES,
    ensure_directories,
)


def check_kaggle_cli_available() -> None:
    """Confirm the kaggle command exists before trying to use it."""
    # shutil.which searches the system PATH for an executable and returns
    # its location, or None if it is not found.
    if shutil.which("kaggle") is None:
        print("ERROR: the 'kaggle' command was not found.")
        print("Fix: activate your virtual environment, then run:")
        print("     pip install kaggle")
        sys.exit(1)  # exit code 1 means "finished with an error"

    print("Kaggle CLI found.")


def files_already_present() -> bool:
    """Return True only if every expected CSV is already on disk."""
    # all() returns True when every item in the list is True.
    return all(file_path.exists() for file_path in EXPECTED_RAW_FILES)


def download_competition_files() -> Path:
    """
    Run the Kaggle CLI to download the competition archive.

    Returns the path to the downloaded zip file.
    """
    print(f"\nDownloading '{KAGGLE_COMPETITION}' into {RAW_DATA_DIR} ...")
    print("This is roughly 120 MB and may take a few minutes.\n")

    # The command as a list of pieces. Passing a list rather than one long
    # string avoids problems with spaces in folder names.
    #   -p  : where to put the download
    #   -o  : overwrite anything already there
    command = [
        "kaggle",
        "competitions",
        "download",
        KAGGLE_COMPETITION,
        "-p",
        str(RAW_DATA_DIR),
        "-o",
    ]

    # check=False means "do not raise an exception automatically", so we can
    # print a helpful message ourselves instead of a raw stack trace.
    result = subprocess.run(command, check=False)

    # Older versions of the CLI expect the competition name after -c.
    # If the first attempt failed, try that older form before giving up.
    if result.returncode != 0:
        print("\nFirst attempt failed. Retrying with the older -c flag syntax ...")
        legacy_command = [
            "kaggle",
            "competitions",
            "download",
            "-c",
            KAGGLE_COMPETITION,
            "-p",
            str(RAW_DATA_DIR),
            "-o",
        ]
        result = subprocess.run(legacy_command, check=False)

    if result.returncode != 0:
        print("\nERROR: the download failed.")
        print("Most common causes, in order of likelihood:")
        print("  1. You have not joined the competition and accepted its rules.")
        print(f"     Go to https://www.kaggle.com/competitions/{KAGGLE_COMPETITION}/rules")
        print("  2. Your credentials are missing or expired. Run: kaggle auth login")
        print("  3. No internet connection, or a proxy is blocking the request.")
        sys.exit(1)

    zip_path = RAW_DATA_DIR / f"{KAGGLE_COMPETITION}.zip"
    if not zip_path.exists():
        print(f"ERROR: expected {zip_path} after download, but it is not there.")
        sys.exit(1)

    print(f"\nDownloaded: {zip_path.name}")
    return zip_path


def extract_archive(zip_path: Path) -> None:
    """Unzip the archive into data/raw and then delete the zip."""
    print(f"\nExtracting {zip_path.name} ...")

    # "with" makes sure the zip file is closed properly even if an error occurs.
    with zipfile.ZipFile(zip_path, "r") as archive:
        members = archive.namelist()
        for index, member in enumerate(members, start=1):
            print(f"  [{index}/{len(members)}] {member}")
            archive.extract(member, RAW_DATA_DIR)

    # The zip is no longer needed and takes up 120 MB.
    zip_path.unlink()
    print("\nExtraction complete. Archive removed.")


def report_results() -> None:
    """Print each expected file with its size, and flag anything missing."""
    print("\n" + "=" * 60)
    print("FILES IN data/raw")
    print("=" * 60)

    missing = []
    for file_path in EXPECTED_RAW_FILES:
        if file_path.exists():
            # st_size is in bytes. Divide twice by 1024 to reach megabytes.
            size_mb = file_path.stat().st_size / 1024 / 1024
            print(f"  OK       {file_path.name:<28} {size_mb:>9.1f} MB")
        else:
            print(f"  MISSING  {file_path.name}")
            missing.append(file_path.name)

    if missing:
        print(f"\nWARNING: {len(missing)} expected file(s) missing: {missing}")
        sys.exit(1)

    print("\nAll expected files are present.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download the IEEE-CIS Fraud Detection dataset from Kaggle."
    )
    parser.add_argument(
        "--force",
        action="store_true",  # makes it a simple on/off flag with no value
        help="Download again even if the files already exist.",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("IEEE-CIS FRAUD DETECTION: DATA DOWNLOAD")
    print("=" * 60)

    ensure_directories()
    check_kaggle_cli_available()

    if files_already_present() and not args.force:
        print("\nAll files are already present. Nothing to do.")
        print("Use --force to download them again.")
        report_results()
        return

    zip_path = download_competition_files()
    extract_archive(zip_path)
    report_results()

    print("\nNext: run  python scripts/verify_data.py")


# This guard means the code only runs when the file is executed directly,
# not when it is imported by another module.
if __name__ == "__main__":
    main()
