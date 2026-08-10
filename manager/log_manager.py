from __future__ import annotations

import logging
import os
from datetime import date
from pathlib import Path


_CONFIGURED = False


class DailyFileHandler(logging.FileHandler):
    def __init__(
        self,
        log_dir: str | Path,
        log_file_name: str = "agent.log",
        *,
        encoding: str = "utf-8",
    ) -> None:
        self._log_dir = Path(log_dir)
        self._log_file_name = log_file_name
        self._current_date = date.today()
        self._log_dir.mkdir(parents=True, exist_ok=True)
        super().__init__(self._build_path(self._current_date), encoding=encoding)

    def _build_path(self, current_date: date) -> Path:
        return self._log_dir / f"{current_date.isoformat()}-{self._log_file_name}"

    def _switch_if_needed(self) -> None:
        current_date = date.today()
        if current_date == self._current_date:
            return

        self.acquire()
        try:
            if current_date == self._current_date:
                return
            if self.stream:
                self.stream.close()
            self.baseFilename = os.fspath(self._build_path(current_date))
            self.stream = self._open()
            self._current_date = current_date
        finally:
            self.release()

    def emit(self, record: logging.LogRecord) -> None:
        self._switch_if_needed()
        super().emit(record)


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

    file_handler = DailyFileHandler(
        target_dir,
        log_file_name,
        encoding="utf-8",
    )
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)
    root.addHandler(file_handler)

    _CONFIGURED = True
