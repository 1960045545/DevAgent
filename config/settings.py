from __future__ import annotations

import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ENV_FILE = PROJECT_ROOT / ".env"


def load_project_env(*, override: bool = False) -> Path:
    """Load the single project-level .env file."""
    try:
        from dotenv import load_dotenv
    except ImportError:
        _load_env_fallback(PROJECT_ENV_FILE, override=override)
        return PROJECT_ENV_FILE

    load_dotenv(
        dotenv_path=PROJECT_ENV_FILE,
        override=override,
    )
    return PROJECT_ENV_FILE


def _load_env_fallback(
    path: Path,
    *,
    override: bool,
) -> None:
    if not path.exists():
        return

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line.removeprefix("export ").strip()
        if "=" not in line:
            continue

        name, value = line.split("=", 1)
        name = name.strip()
        value = value.strip()
        if not name:
            continue
        if (
            len(value) >= 2
            and value[0] == value[-1]
            and value[0] in {"'", '"'}
        ):
            value = value[1:-1]
        if override or name not in os.environ:
            os.environ[name] = value
