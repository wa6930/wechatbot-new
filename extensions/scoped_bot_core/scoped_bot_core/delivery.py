"""Durable outbound intent ledger; this module never sends or imports WeChat.

Prepare an immutable logical action, claim it immediately before its external
effect, and persist the outcome. A claimed action is never made sendable again,
including after a crash or an ambiguous timeout. This provides at-most-one
dispatch admission, not proof of exactly-once delivery by a GUI client.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import secrets
import sqlite3
import time
from collections.abc import Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


class IntentConflict(ValueError):
    """The logical action slot was already bound to different arguments."""


class InvalidTransition(RuntimeError):
    """The outcome would replay or contradict a previously recorded action."""


class InvalidReceipt(ValueError):
    """The supplied readback does not identify this exact outbound action."""


@dataclass(frozen=True, slots=True)
class DeliveryIntent:
    action_id: str
    request_id: str
    action_index: int
    account_id: str
    chat_id: str
    text_sha256: str
    mode: str
    state: str
    owner_id: str | None
    created_at: float
    updated_at: float
    expires_at: float | None
    reason_code: str | None
    receipt: dict[str, Any] | None


@dataclass(frozen=True, slots=True)
class DeliveryClaim:
    action_id: str
    claim_token: str = field(repr=False)
    owner_id: str
    mode: str
    text_sha256: str


def _identity(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or not value.strip()
        or len(value) > 1024
        or any(ord(character) < 32 for character in value)
    ):
        raise ValueError("Invalid " + label)
    return value.strip()


def _action_index(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError("action_index must be a nonnegative integer")
    return value


def _timestamp(value: Any, label: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value < 0
    ):
        raise ValueError("Invalid " + label)
    return float(value)


def _reason(value: Any) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[a-z][a-z0-9_]{0,63}", value):
        raise ValueError("reason_code must be a short content-free identifier")
    return value


def text_sha256(text: str) -> str:
    if not isinstance(text, str):
        raise ValueError("Text must be a string")
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def make_action_id(request_id: str, action_index: int = 0) -> str:
    """Bind a slot to the globally stable request ID, independent of target."""
    slot = [_identity(request_id, "request_id"), _action_index(action_index)]
    return hashlib.sha256(
        json.dumps(slot, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


class DeliveryLedger:
    def __init__(self, path, *, clock=time.time):
        self.path = Path(path).expanduser().resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.clock = clock
        with self._connect() as db:
            db.execute("PRAGMA journal_mode=WAL")
            db.executescript("""
                CREATE TABLE IF NOT EXISTS delivery_intents (
                    action_id TEXT PRIMARY KEY,
                    request_id TEXT NOT NULL,
                    action_index INTEGER NOT NULL,
                    account_id TEXT NOT NULL,
                    chat_id TEXT NOT NULL,
                    text_sha256 TEXT NOT NULL,
                    mode TEXT NOT NULL,
                    state TEXT NOT NULL,
                    owner_id TEXT,
                    claim_token TEXT,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL,
                    expires_at REAL,
                    reason_code TEXT,
                    receipt_json TEXT,
                    UNIQUE(request_id, action_index)
                );
                CREATE TABLE IF NOT EXISTS delivery_history (
                    sequence INTEGER PRIMARY KEY AUTOINCREMENT,
                    action_id TEXT NOT NULL REFERENCES delivery_intents(action_id),
                    state TEXT NOT NULL,
                    observed_at REAL NOT NULL,
                    reason_code TEXT,
                    receipt_json TEXT
                );
                CREATE TABLE IF NOT EXISTS delivery_receipt_owners (
                    account_id TEXT NOT NULL,
                    chat_id TEXT NOT NULL,
                    message_id TEXT NOT NULL,
                    action_id TEXT NOT NULL UNIQUE
                        REFERENCES delivery_intents(action_id),
                    PRIMARY KEY(account_id, chat_id, message_id)
                );
            """)
        # Preserve receipt exclusivity when opening an earlier prototype ledger.
        with self._connect(write=True) as db:
            for row in db.execute(
                "SELECT * FROM delivery_intents WHERE state='confirmed'"
            ).fetchall():
                try:
                    receipt = json.loads(row["receipt_json"])
                except (TypeError, ValueError):
                    raise InvalidReceipt("Saved confirmed receipt is invalid") from None
                self._reserve_receipt(db, row, self._validated_receipt(row, receipt))

    @contextmanager
    def _connect(self, *, write=False):
        db = sqlite3.connect(self.path, timeout=5, isolation_level=None)
        db.row_factory = sqlite3.Row
        try:
            db.execute("PRAGMA foreign_keys=ON")
            db.execute("PRAGMA synchronous=FULL")
            if write:
                db.execute("BEGIN IMMEDIATE")
            try:
                yield db
            except BaseException:
                if db.in_transaction:
                    db.rollback()
                raise
            else:
                if db.in_transaction:
                    db.commit()
        finally:
            db.close()

    @staticmethod
    def _intent(row: sqlite3.Row) -> DeliveryIntent:
        return DeliveryIntent(
            **{
                key: row[key]
                for key in (
                    "action_id",
                    "request_id",
                    "action_index",
                    "account_id",
                    "chat_id",
                    "text_sha256",
                    "mode",
                    "state",
                    "owner_id",
                    "created_at",
                    "updated_at",
                    "expires_at",
                    "reason_code",
                )
            },
            receipt=json.loads(row["receipt_json"]) if row["receipt_json"] else None,
        )

    def _row(self, db, action_id):
        row = db.execute(
            "SELECT * FROM delivery_intents WHERE action_id=?", (action_id,)
        ).fetchone()
        if row is None:
            raise ValueError("Unknown delivery action")
        return row

    def _transition(self, db, row, state, *, reason_code=None, receipt=None):
        now = _timestamp(self.clock(), "clock")
        encoded = json.dumps(receipt, sort_keys=True) if receipt is not None else None
        if state == "confirmed":
            self._reserve_receipt(db, row, receipt)
        db.execute(
            "UPDATE delivery_intents SET state=?,updated_at=?,reason_code=?,receipt_json=? "
            "WHERE action_id=?",
            (state, now, reason_code, encoded, row["action_id"]),
        )
        db.execute(
            "INSERT INTO delivery_history(action_id,state,observed_at,reason_code,receipt_json) "
            "VALUES(?,?,?,?,?)",
            (row["action_id"], state, now, reason_code, encoded),
        )
        return self._intent(self._row(db, row["action_id"]))

    def prepare(
        self,
        *,
        request_id,
        account_id,
        chat_id,
        text,
        action_index=0,
        mode="dry_run",
        expires_at=None,
    ) -> DeliveryIntent:
        request_id = _identity(request_id, "request_id")
        account_id = _identity(account_id, "account_id")
        chat_id = _identity(chat_id, "chat_id")
        action_index = _action_index(action_index)
        if not isinstance(text, str) or not text.strip():
            raise ValueError("Nonempty text is required")
        if mode not in {"dry_run", "live"}:
            raise ValueError("mode must be dry_run or live")
        if expires_at is not None:
            expires_at = _timestamp(expires_at, "expires_at")
        action_id = make_action_id(request_id, action_index)
        digest = text_sha256(text)
        binding = (
            request_id,
            action_index,
            account_id,
            chat_id,
            digest,
            mode,
            expires_at,
        )
        with self._connect(write=True) as db:
            row = db.execute(
                "SELECT * FROM delivery_intents WHERE request_id=? AND action_index=?",
                (request_id, action_index),
            ).fetchone()
            if row is not None:
                existing = tuple(
                    row[key]
                    for key in (
                        "request_id",
                        "action_index",
                        "account_id",
                        "chat_id",
                        "text_sha256",
                        "mode",
                        "expires_at",
                    )
                )
                if existing != binding or row["action_id"] != action_id:
                    raise IntentConflict(
                        "Delivery action slot has different immutable arguments"
                    )
                return self._intent(row)
            now = _timestamp(self.clock(), "clock")
            db.execute(
                "INSERT INTO delivery_intents "
                "(action_id,request_id,action_index,account_id,chat_id,text_sha256,mode,state,"
                "created_at,updated_at,expires_at) VALUES(?,?,?,?,?,?,?,'prepared',?,?,?)",
                (
                    action_id,
                    request_id,
                    action_index,
                    account_id,
                    chat_id,
                    digest,
                    mode,
                    now,
                    now,
                    expires_at,
                ),
            )
            db.execute(
                "INSERT INTO delivery_history(action_id,state,observed_at) VALUES(?,'prepared',?)",
                (action_id, now),
            )
            return self._intent(self._row(db, action_id))

    def get(self, action_id: str) -> DeliveryIntent | None:
        action_id = _identity(action_id, "action_id")
        with self._connect() as db:
            row = db.execute(
                "SELECT * FROM delivery_intents WHERE action_id=?", (action_id,)
            ).fetchone()
            return self._intent(row) if row is not None else None

    def claim(self, action_id: str, owner_id: str) -> DeliveryClaim | None:
        """Persist admission before any GUI side effect; never grant it twice."""
        action_id = _identity(action_id, "action_id")
        owner_id = _identity(owner_id, "owner_id")
        with self._connect(write=True) as db:
            row = self._row(db, action_id)
            if row["state"] != "prepared":
                return None
            now = _timestamp(self.clock(), "clock")
            if row["expires_at"] is not None and now >= row["expires_at"]:
                self._transition(db, row, "expired", reason_code="deadline_elapsed")
                return None
            token = secrets.token_hex(32)
            db.execute(
                "UPDATE delivery_intents SET owner_id=?,claim_token=? WHERE action_id=?",
                (owner_id, token, action_id),
            )
            self._transition(db, row, "executing")
            return DeliveryClaim(
                action_id, token, owner_id, row["mode"], row["text_sha256"]
            )

    @staticmethod
    def _check_claim(row, claim_token):
        if (
            not isinstance(claim_token, str)
            or not re.fullmatch(r"[0-9a-f]{64}", claim_token)
            or not row["claim_token"]
            or not secrets.compare_digest(row["claim_token"], claim_token)
        ):
            raise PermissionError("Delivery claim does not own this action")

    def _record_outcome(self, action_id, claim_token, state, reason_code):
        reason_code = _reason(reason_code)
        with self._connect(write=True) as db:
            row = self._row(db, _identity(action_id, "action_id"))
            self._check_claim(row, claim_token)
            if state == "dry_run" and row["mode"] != "dry_run":
                raise InvalidTransition("A live action cannot have a dry-run outcome")
            if row["state"] == state:
                return self._intent(row)
            if row["state"] != "executing":
                raise InvalidTransition("Delivery action already has an outcome")
            return self._transition(db, row, state, reason_code=reason_code)

    def record_dry_run(self, action_id, claim_token) -> DeliveryIntent:
        return self._record_outcome(action_id, claim_token, "dry_run", "dry_run_only")

    def record_not_sent(
        self, action_id, claim_token, reason_code="not_sent"
    ) -> DeliveryIntent:
        """Record proven non-delivery without creating automatic retry authority."""
        return self._record_outcome(action_id, claim_token, "not_sent", reason_code)

    def record_unknown(
        self, action_id, claim_token, reason_code="outcome_unknown"
    ) -> DeliveryIntent:
        return self._record_outcome(action_id, claim_token, "unknown", reason_code)

    def _validated_receipt(self, row, receipt):
        required = {"source", "account_id", "chat_id", "message_id", "text_sha256"}
        if (
            not isinstance(receipt, Mapping)
            or not required.issubset(receipt)
            or set(receipt) - required - {"observed_at"}
            or receipt["source"] != "db_readback"
            or receipt["account_id"] != row["account_id"]
            or receipt["chat_id"] != row["chat_id"]
            or receipt["text_sha256"] != row["text_sha256"]
        ):
            raise InvalidReceipt("Readback does not match the delivery action")
        try:
            message_id = _identity(receipt["message_id"], "receipt message_id")
            observed_at = _timestamp(
                receipt.get("observed_at", self.clock()), "observed_at"
            )
        except ValueError:
            raise InvalidReceipt("Readback identity or timestamp is invalid") from None
        if observed_at < row["created_at"]:
            raise InvalidReceipt("Readback observation predates the delivery intent")
        return dict(receipt, message_id=message_id, observed_at=observed_at)

    @staticmethod
    def _reserve_receipt(db, row, receipt):
        scope = (receipt["account_id"], receipt["chat_id"], receipt["message_id"])
        owner = db.execute(
            "SELECT action_id FROM delivery_receipt_owners "
            "WHERE account_id=? AND chat_id=? AND message_id=?",
            scope,
        ).fetchone()
        if owner is not None:
            if owner["action_id"] != row["action_id"]:
                raise InvalidReceipt(
                    "Database receipt is already assigned to another action"
                )
            return
        existing = db.execute(
            "SELECT 1 FROM delivery_receipt_owners WHERE action_id=?",
            (row["action_id"],),
        ).fetchone()
        if existing is not None:
            raise InvalidReceipt(
                "Action is already assigned a different database receipt"
            )
        db.execute(
            "INSERT INTO delivery_receipt_owners(account_id,chat_id,message_id,action_id) "
            "VALUES(?,?,?,?)",
            (*scope, row["action_id"]),
        )

    def confirm_delivery(
        self, action_id, claim_token, receipt: Mapping[str, Any]
    ) -> DeliveryIntent:
        """Record caller-verified DB evidence; this method does not read WeChat.

        A UI success flag alone is insufficient. The caller must perform actual
        scope-specific DB readback and compare the exact outbound text first.
        """
        with self._connect(write=True) as db:
            row = self._row(db, _identity(action_id, "action_id"))
            self._check_claim(row, claim_token)
            if row["mode"] != "live":
                raise InvalidTransition("A dry-run action cannot be confirmed as sent")
            normalized = self._validated_receipt(row, receipt)
            required = {"source", "account_id", "chat_id", "message_id", "text_sha256"}
            if row["state"] == "confirmed":
                saved = json.loads(row["receipt_json"])
                if any(saved[key] != normalized[key] for key in required):
                    raise InvalidReceipt(
                        "A different delivery receipt is already recorded"
                    )
                return self._intent(row)
            if row["state"] not in {"executing", "unknown"}:
                raise InvalidTransition("This outcome cannot be confirmed as sent")
            return self._transition(db, row, "confirmed", receipt=normalized)

    def reconcile_unknown(
        self, action_id, receipt: Mapping[str, Any]
    ) -> DeliveryIntent:
        """Confirm an unknown action after trusted local DB reconciliation only.

        This is a local controller operation, never an LLM-facing tool. The
        controller must read the exact account/chat DB and verify the outbound
        message and text before calling it. It neither returns/recreates claim
        tokens nor admits another dispatch. Prepared and executing actions are
        deliberately ineligible: recovery must classify abandoned work first.
        """
        with self._connect(write=True) as db:
            row = self._row(db, _identity(action_id, "action_id"))
            if row["state"] != "unknown" or row["mode"] != "live":
                raise InvalidTransition("Only an unknown live action can be reconciled")
            normalized = self._validated_receipt(row, receipt)
            return self._transition(
                db, row, "confirmed", reason_code="local_reconciled", receipt=normalized
            )

    def recover_incomplete(self) -> list[DeliveryIntent]:
        """Mark abandoned claims unknown, without resending or stopping workers.

        Call after taking exclusive ownership of the adapter runtime. The old
        claim token remains valid only for a later matching DB confirmation.
        """
        with self._connect(write=True) as db:
            rows = db.execute(
                "SELECT * FROM delivery_intents WHERE state='executing' ORDER BY created_at,action_id"
            ).fetchall()
            return [
                self._transition(db, row, "unknown", reason_code="process_restarted")
                for row in rows
            ]

    def history(self, action_id: str) -> list[dict[str, Any]]:
        with self._connect() as db:
            return [
                {
                    "state": row["state"],
                    "observed_at": row["observed_at"],
                    "reason_code": row["reason_code"],
                    "receipt": json.loads(row["receipt_json"])
                    if row["receipt_json"]
                    else None,
                }
                for row in db.execute(
                    "SELECT * FROM delivery_history WHERE action_id=? ORDER BY sequence",
                    (_identity(action_id, "action_id"),),
                )
            ]
