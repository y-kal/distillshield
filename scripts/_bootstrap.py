from __future__ import annotations

import sys
from pathlib import Path


def bootstrap_repo_packages() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    package_roots = [
        repo_root / "apps/api",
        repo_root / "packages/core/src",
        repo_root / "packages/storage/src",
        repo_root / "packages/synthetic_data/src",
        repo_root / "packages/feature_pipeline/src",
        repo_root / "packages/models/src",
        repo_root / "packages/llm_adapter/src",
        repo_root / "packages/eval/src",
    ]
    for package_root in reversed(package_roots):
        package_root_str = str(package_root)
        if package_root_str not in sys.path:
            sys.path.insert(0, package_root_str)
