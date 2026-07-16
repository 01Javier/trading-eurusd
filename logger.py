"""
logger.py
Small CSV logger for demo trading and research runs.

The logger intentionally writes plain CSV files instead of modifying the Excel
tracker used for manual research. This keeps automation logs append-only and
easy to audit.
"""
from __future__ import annotations

import csv
import os
from datetime import datetime, timezone
from typing import Mapping, Any

from config import LOGS_DIR


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


class CsvLogger:
    def __init__(self, path: str | None = None):
        self.path = path or os.path.join(LOGS_DIR, "demo_bot_events.csv")

    def write(self, event: str, fields: Mapping[str, Any] | None = None) -> None:
        payload = {"ts_utc": utc_now_iso(), "event": event}
        if fields:
            payload.update({str(k): v for k, v in fields.items()})

        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        file_exists = os.path.exists(self.path)
        existing_fields: list[str] = []
        if file_exists:
            with open(self.path, "r", newline="", encoding="utf-8") as f:
                reader = csv.reader(f)
                existing_fields = next(reader, [])

        fieldnames = list(dict.fromkeys(existing_fields + list(payload.keys())))
        needs_rewrite = file_exists and existing_fields != fieldnames
        rows: list[dict[str, Any]] = []
        if needs_rewrite:
            with open(self.path, "r", newline="", encoding="utf-8") as f:
                rows = list(csv.DictReader(f))

        mode = "w" if needs_rewrite else "a"
        with open(self.path, mode, newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if not file_exists or needs_rewrite:
                writer.writeheader()
            for row in rows:
                writer.writerow(row)
            writer.writerow(payload)
