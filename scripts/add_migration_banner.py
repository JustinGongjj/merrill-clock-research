"""Insert a migration banner at the top of a repository README.

Usage: python3 scripts/add_migration_banner.py <repo_dir> <banner_text_file>
"""
from __future__ import annotations

import sys
from pathlib import Path


def main() -> int:
    repo = Path(sys.argv[1])
    banner_file = Path(sys.argv[2])
    readme = repo / "README.md"
    banner = banner_file.read_text(encoding="utf-8").rstrip() + "\n\n"

    text = readme.read_text(encoding="utf-8")
    if "merrill-clock-research" in text.split("\n\n")[0]:
        print(f"banner already present in {readme}")
        return 0
    readme.write_text(banner + text, encoding="utf-8")
    print(f"banner added: {readme}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
