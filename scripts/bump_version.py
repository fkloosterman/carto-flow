#!/usr/bin/env python3
"""
Version bump script - single source of truth for version management.
This script updates version numbers in all necessary files.
"""

import argparse
import re
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Bump version in pyproject.toml and mkdocs.yml")
    parser.add_argument("version", help="New version number (must follow semver format)")
    args = parser.parse_args()

    version = args.version.strip()

    # Validate semantic version format
    if not re.match(
        r"^[0-9]+\.[0-9]+\.[0-9]+(\-[a-zA-Z0-9]+(\.[a-zA-Z0-9]+)*)?(\+[a-zA-Z0-9]+(\.[a-zA-Z0-9]+)*)?$", version
    ):
        print(f"Error: Version '{version}' does not follow semantic versioning format")
        print("Example valid versions: 1.0.0, 1.2.3-beta.1, 2.0.0+build.123")
        sys.exit(1)

    # Update pyproject.toml
    pyproject_path = Path("pyproject.toml")
    if not pyproject_path.exists():
        print("Error: pyproject.toml not found in current directory")
        sys.exit(1)

    content = pyproject_path.read_text(encoding="utf-8")
    updated_content = re.sub(r'^version = ".*"$', f'version = "{version}"', content, flags=re.MULTILINE)
    pyproject_path.write_text(updated_content, encoding="utf-8")
    print(f"Updated version in pyproject.toml to {version}")

    # Update mkdocs.yml
    mkdocs_path = Path("mkdocs.yml")
    if not mkdocs_path.exists():
        print("Error: mkdocs.yml not found in current directory")
        sys.exit(1)

    content = mkdocs_path.read_text(encoding="utf-8")
    updated_content = re.sub(r"version: .*", f"version: {version}", content, flags=re.MULTILINE)
    mkdocs_path.write_text(updated_content, encoding="utf-8")
    print(f"Updated version in mkdocs.yml to {version}")

    # Update src/carto_flow/__init__.py
    init_path = Path("src/carto_flow/__init__.py")
    if not init_path.exists():
        print("Error: src/carto_flow/__init__.py not found in current directory")
        sys.exit(1)

    content = init_path.read_text(encoding="utf-8")
    updated_content = re.sub(r'__version__ = ".*"', f'__version__ = "{version}"', content, flags=re.MULTILINE)
    init_path.write_text(updated_content, encoding="utf-8")
    print(f"Updated version in src/carto_flow/__init__.py to {version}")

    # Note: uv.lock should be updated separately using `uv sync` if dependencies change
    # This ensures consistency between the package version and dependencies

    print("✅ Version bump completed successfully")


if __name__ == "__main__":
    main()
