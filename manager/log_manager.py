from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path


_CONFIGURED = False


def configure_logging(
    *,
    level: int = logging.INFO,
    console_level: int = logging.WARNING,
    log_dir: str | Path | None = None,
    log_file_name: str = "agent.log",
) -> None:
    global _CONFIGURED

    if _CONFIGURED:
        return

    root = logging.getLogger()
    root.setLevel(min(level, console_level))

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(funcName)s:%(lineno)d | %(message)s"
    )

    console_handler = logging.StreamHandler()
    console_handler.setLevel(console_level)
    console_handler.setFormatter(formatter)
    root.addHandler(console_handler)

    target_dir = (
        Path(log_dir)
        if log_dir
        else Path(__file__).resolve().parent.parent / "logs"
    )
    target_dir.mkdir(parents=True, exist_ok=True)

    file_handler = RotatingFileHandler(
        target_dir / log_file_name,
        maxBytes=5_000_000,
        backupCount=3,
        encoding="utf-8",
    )
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)
    root.addHandler(file_handler)

    _CONFIGURED = True
