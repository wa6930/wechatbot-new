"""Durable synthetic/capture inbox and dry-run receipts, never a live sender."""

import json
import sqlite3
import time
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path

from . import Event, normalize_event


class InboxStore:
    def __init__(self, path):
        self.path = Path(path).resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as db:
            db.executescript("""
                CREATE TABLE IF NOT EXISTS inbox (
                    account_id TEXT NOT NULL,
                    chat_id TEXT NOT NULL,
                    message_id TEXT NOT NULL,
                    payload TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'pending',
                    created_at REAL NOT NULL,
                    PRIMARY KEY(account_id, chat_id, message_id)
                );
                CREATE TABLE IF NOT EXISTS replies (
                    request_id TEXT PRIMARY KEY,
                    account_id TEXT NOT NULL,
                    chat_id TEXT NOT NULL,
                    text TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'dry_run',
                    created_at REAL NOT NULL
                );
            """)

    @contextmanager
    def _connect(self):
        db = sqlite3.connect(self.path, timeout=5)
        db.row_factory = sqlite3.Row
        try:
            with db:
                yield db
        finally:
            db.close()

    def enqueue(self, payload) -> bool:
        event = normalize_event(payload)
        with self._connect() as db:
            cursor = db.execute(
                "INSERT OR IGNORE INTO inbox "
                "(account_id,chat_id,message_id,payload,created_at) VALUES(?,?,?,?,?)",
                (
                    event.account_id,
                    event.chat_id,
                    event.message_id,
                    json.dumps(asdict(event), ensure_ascii=False),
                    time.time(),
                ),
            )
            return cursor.rowcount == 1

    def claim(self, limit=20) -> list[Event]:
        if not isinstance(limit, int) or limit < 1 or limit > 1000:
            raise ValueError("Invalid batch size")
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            rows = db.execute(
                "SELECT * FROM inbox WHERE status='pending' "
                "ORDER BY created_at,rowid LIMIT ?",
                (limit,),
            ).fetchall()
            for row in rows:
                db.execute(
                    "UPDATE inbox SET status='dispatched' "
                    "WHERE account_id=? AND chat_id=? AND message_id=?",
                    (row["account_id"], row["chat_id"], row["message_id"]),
                )
            return [normalize_event(json.loads(row["payload"])) for row in rows]

    def finish(self, event: Event, state="completed") -> None:
        if state not in {"completed", "ignored", "expired", "failed", "cancelled"}:
            raise ValueError("Invalid receipt state")
        with self._connect() as db:
            cursor = db.execute(
                "UPDATE inbox SET status=? "
                "WHERE account_id=? AND chat_id=? AND message_id=?",
                (state, event.account_id, event.chat_id, event.message_id),
            )
            if cursor.rowcount != 1:
                raise ValueError("Unknown inbox event")

    def record_reply(self, request_id, account_id, chat_id, text) -> bool:
        if not all(
            isinstance(v, str) and v for v in (request_id, account_id, chat_id, text)
        ):
            raise ValueError("Reply identity and text are required")
        with self._connect() as db:
            return (
                db.execute(
                    "INSERT OR IGNORE INTO replies "
                    "(request_id,account_id,chat_id,text,created_at) VALUES(?,?,?,?,?)",
                    (request_id, account_id, chat_id, text, time.time()),
                ).rowcount
                == 1
            )

    def replies(self) -> list[dict]:
        with self._connect() as db:
            return [
                dict(row)
                for row in db.execute("SELECT * FROM replies ORDER BY created_at,rowid")
            ]
