"""
Refuse to commit anything that looks like an API token.

Run automatically by pre-commit on the files being committed. The point is
that a credential should be impossible to commit by accident, rather than
something you have to remember not to do.

The patterns cover the tokens this project touches. Add more as needed.
"""

import re
import sys
from pathlib import Path

# Each pattern is (what it is, the shape it takes).
PATTERNS = [
    ("Hugging Face token", re.compile(r"hf_[A-Za-z0-9]{30,}")),
    ("GitHub token", re.compile(r"gh[pousr]_[A-Za-z0-9]{30,}")),
    ("AWS access key", re.compile(r"AKIA[0-9A-Z]{16}")),
    ("OpenAI key", re.compile(r"sk-[A-Za-z0-9]{32,}")),
    ("private key block", re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----")),
]

# Files that are allowed to contain the patterns, because they describe
# them rather than containing a real one.
ALLOWED = {"scripts/check_no_secrets.py", ".env.example"}

# Reading a 100 MB Parquet file looking for text would be pointless and slow.
SKIP_SUFFIXES = {".parquet", ".joblib", ".png", ".db", ".zip", ".csv"}


def main() -> int:
    problems = []

    for name in sys.argv[1:]:
        path = Path(name)

        if path.as_posix() in ALLOWED:
            continue
        if path.suffix.lower() in SKIP_SUFFIXES:
            continue
        if not path.is_file():
            continue

        # errors="ignore" so a stray non-text file cannot crash the hook.
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue

        for label, pattern in PATTERNS:
            if pattern.search(text):
                problems.append(f"  {path}: looks like a {label}")

    if problems:
        print("\nCOMMIT BLOCKED: possible credentials found.\n")
        print("\n".join(problems))
        print(
            "\nMove the value into .env, which is git-ignored, and read it with\n"
            "os.getenv(). If this is a false alarm, add the path to ALLOWED in\n"
            "scripts/check_no_secrets.py.\n"
        )
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
