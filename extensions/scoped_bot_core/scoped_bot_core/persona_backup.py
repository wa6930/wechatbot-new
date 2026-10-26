"""Versioned encrypted persona snapshots and pure backup scheduling decisions.

Callers supply selected, already sanitized data and a separately held 32-byte
key. This module never reads/writes files, scans a store, executes restored
content, or invokes Git/network/model/WeChat operations.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import math
import re
import secrets
import time
from dataclasses import dataclass
from typing import Any

FORMAT = "astrbot-wechat-persona"
SCHEMA_VERSION = 1
MAX_SNAPSHOT_BYTES = 8 * 1024 * 1024
MAX_ENVELOPE_BYTES = 12 * 1024 * 1024
MAX_STRING_BYTES = 128 * 1024
MAX_RECORDS = 10000
MAX_NODES = 150000
_SCOPE = {"account_id", "chat_id", "member_id", "persona_id"}
_SECRET_MARKER = re.compile(
    r"-----BEGIN (?:[A-Z ]*PRIVATE KEY|OPENSSH PRIVATE KEY)-----"
    r"|\b(?:api[_ -]?key|password|passwd|access[_ -]?token|client[_ -]?secret)\s*[:=]"
    r"|\bsk-[A-Za-z0-9_-]{20,}",
    re.IGNORECASE,
)


class SnapshotError(ValueError):
    """A snapshot or envelope violates the bounded, versioned data contract."""


class BackupIntegrityError(SnapshotError):
    """The key, authentication tag, manifest, or serialized data is invalid."""


class BackupDependencyError(RuntimeError):
    """The optional cryptography runtime is unavailable."""


def _aesgcm():
    try:
        from cryptography.hazmat.primitives.ciphers.aead import AESGCM
    except ImportError as exc:
        raise BackupDependencyError("Install the pinned cryptography runtime") from exc
    return AESGCM


def _canonical(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (ValueError, TypeError, UnicodeError, RecursionError) as exc:
        raise SnapshotError("Invalid JSON data") from exc


def _exact(value: Any, keys: set[str]) -> None:
    if type(value) is not dict or set(value) != keys:
        raise SnapshotError("Unexpected fields or missing required fields")


def _number(value: Any, *, nullable: bool = False) -> None:
    if value is None and nullable:
        return
    try:
        valid = type(value) in (int, float) and math.isfinite(value) and value >= 0
    except OverflowError:
        valid = False
    if not valid:
        raise SnapshotError("Expected a finite nonnegative number")


def _version(value: Any) -> None:
    if type(value) is not int or not 1 <= value <= 2**31 - 1:
        raise SnapshotError("Expected a positive version")


def _text(value: Any, *, nullable: bool = False, identity: bool = False) -> None:
    if value is None and nullable:
        return
    if type(value) is not str or not value.strip():
        raise SnapshotError("Expected nonempty text")
    limit = 1024 if identity else MAX_STRING_BYTES
    if identity and len(value) > 256:
        raise SnapshotError("Identity text exceeds snapshot limit")
    try:
        if len(value.encode("utf-8")) > limit:
            raise SnapshotError("Text exceeds snapshot limit")
    except UnicodeError as exc:
        raise SnapshotError("Invalid text encoding") from exc
    if identity and any(ord(char) < 32 for char in value):
        raise SnapshotError("Invalid identity text")
    if _SECRET_MARKER.search(value):
        raise SnapshotError("Credential-like content is not exportable")


def _rows(value: Any) -> None:
    if type(value) is not list or len(value) > MAX_RECORDS:
        raise SnapshotError("Expected a bounded record list")


def _bounded_tree(value: Any, depth: int = 0, counter: list[int] | None = None):
    if counter is None:
        counter = [0, 0]
    counter[0] += 1
    counter[1] += 8  # Conservative punctuation/numeric allowance per node.
    if depth > 12 or counter[0] > MAX_NODES:
        raise SnapshotError("Snapshot nesting or item limit exceeded")
    if type(value) is dict:
        for key, item in value.items():
            if type(key) is not str or len(key) > 64:
                raise SnapshotError("Invalid field name")
            counter[1] += len(key.encode("utf-8"))
            _bounded_tree(item, depth + 1, counter)
    elif type(value) is list:
        _rows(value)
        for item in value:
            _bounded_tree(item, depth + 1, counter)
    elif type(value) is str:
        _text(value)
        counter[1] += len(value.encode("utf-8"))
    elif value is not None and type(value) not in (bool, int, float):
        raise SnapshotError("Only JSON-compatible values are accepted")
    elif type(value) in (int, float):
        _number(value)
    if counter[1] > MAX_SNAPSHOT_BYTES:
        raise SnapshotError("Snapshot byte budget exceeded")


def _scoped(row: dict, account_id: str, persona_id: str) -> tuple:
    for name in _SCOPE:
        _text(row[name], identity=True)
    if row["account_id"] != account_id or row["persona_id"] != persona_id:
        raise SnapshotError("Mixed account or persona snapshot")
    return tuple(row[name] for name in sorted(_SCOPE))


def _evidence_ids(value: Any) -> None:
    _rows(value)
    if len(value) > 256:
        raise SnapshotError("Too many evidence references")
    for item in value:
        _text(item, identity=True)
    if len(set(value)) != len(value):
        raise SnapshotError("Duplicate evidence reference")


def validate_snapshot(snapshot: dict) -> dict:
    """Validate selected persona data and return a detached JSON-compatible copy.

    Evidence contains references/metadata only; no inbox, raw transcripts,
    credentials, tools, tasks, configuration, or delivery ledger fields exist.
    Free text must have been sanitized by its owner before calling this API.
    """
    _bounded_tree(snapshot)
    _exact(snapshot, {"persona", "state", "memories", "growth"})
    persona = snapshot["persona"]
    _exact(persona, {"account_id", "persona_id", "core_versions"})
    for name in ("account_id", "persona_id"):
        _text(persona[name], identity=True)
    account_id, persona_id = persona["account_id"], persona["persona_id"]
    _rows(persona["core_versions"])
    if not persona["core_versions"]:
        raise SnapshotError("At least one core version is required")
    versions = set()
    for row in persona["core_versions"]:
        _exact(row, {"version", "core_text", "candidate_id", "created_at"})
        _version(row["version"])
        if row["version"] in versions:
            raise SnapshotError("Duplicate core version")
        versions.add(row["version"])
        _text(row["core_text"])
        _text(row["candidate_id"], nullable=True, identity=True)
        _number(row["created_at"])

    memories = snapshot["memories"]
    _exact(memories, {"records", "evidence", "audit"})
    _rows(memories["evidence"])
    evidence = set()
    for row in memories["evidence"]:
        _exact(row, _SCOPE | {"evidence_id", "kind", "observed_at", "created_at"})
        scope = _scoped(row, account_id, persona_id)
        _text(row["evidence_id"], identity=True)
        _text(row["kind"], identity=True)
        if row["kind"] not in {
            "self_report",
            "observed",
            "owner_confirmed",
            "correction",
            "quote",
            "joke",
            "inference",
        }:
            raise SnapshotError("Unsupported evidence kind")
        _number(row["observed_at"])
        _number(row["created_at"])
        key = (*scope, row["evidence_id"])
        if key in evidence:
            raise SnapshotError("Duplicate evidence record")
        evidence.add(key)

    def references(row, scope):
        _evidence_ids(row["evidence_ids"])
        if any((*scope, item) not in evidence for item in row["evidence_ids"]):
            raise SnapshotError("Missing same-scope evidence")

    _rows(memories["records"])
    memory_keys = set()
    for row in memories["records"]:
        _exact(
            row,
            _SCOPE
            | {
                "memory_id",
                "kind",
                "content",
                "evidence_ids",
                "confidence",
                "importance",
                "observed_at",
                "created_at",
                "expires_at",
                "status",
                "supersedes_id",
            },
        )
        scope = _scoped(row, account_id, persona_id)
        for name in ("memory_id", "kind", "status"):
            _text(row[name], identity=True)
        if row["kind"] not in {
            "fact",
            "preference",
            "shared_experience",
            "relationship",
        }:
            raise SnapshotError("Unsupported memory kind")
        if row["status"] not in {"active", "superseded", "revoked"}:
            raise SnapshotError("Unsupported memory status")
        _text(row["content"])
        _text(row["supersedes_id"], nullable=True, identity=True)
        for name in ("observed_at", "created_at", "confidence", "importance"):
            _number(row[name])
        if row["confidence"] > 1 or row["importance"] > 1:
            raise SnapshotError("Confidence or importance outside unit interval")
        _number(row["expires_at"], nullable=True)
        references(row, scope)
        key = (*scope, row["memory_id"])
        if key in memory_keys:
            raise SnapshotError("Duplicate memory record")
        memory_keys.add(key)
    for row in memories["records"]:
        scope = _scoped(row, account_id, persona_id)
        predecessor = row["supersedes_id"]
        if predecessor is not None and (*scope, predecessor) not in memory_keys:
            raise SnapshotError("Missing same-scope superseded record")

    _rows(memories["audit"])
    for row in memories["audit"]:
        _exact(row, _SCOPE | {"memory_id", "action", "evidence_ids", "created_at"})
        scope = _scoped(row, account_id, persona_id)
        _text(row["memory_id"], identity=True)
        if row["action"] not in {"created", "corrected", "revoked"}:
            raise SnapshotError("Unsupported memory audit action")
        if (*scope, row["memory_id"]) not in memory_keys:
            raise SnapshotError("Missing audit memory record")
        _number(row["created_at"])
        references(row, scope)

    _rows(snapshot["state"])
    state_keys = set()
    for row in snapshot["state"]:
        _exact(row, _SCOPE | {"version", "state_text", "evidence_ids", "updated_at"})
        scope = _scoped(row, account_id, persona_id)
        _version(row["version"])
        _text(row["state_text"])
        _number(row["updated_at"])
        references(row, scope)
        if scope in state_keys:
            raise SnapshotError("Duplicate current state scope")
        state_keys.add(scope)

    _rows(snapshot["growth"])
    growth_keys = set()
    for row in snapshot["growth"]:
        _exact(
            row,
            _SCOPE
            | {
                "candidate_id",
                "base_version",
                "proposed_core",
                "rationale",
                "evidence_ids",
                "status",
                "created_at",
                "reviewed_at",
            },
        )
        scope = _scoped(row, account_id, persona_id)
        for name in ("candidate_id", "status"):
            _text(row[name], identity=True)
        if row["status"] not in {"pending", "accepted", "rejected"}:
            raise SnapshotError("Unsupported growth status")
        for name in ("proposed_core", "rationale"):
            _text(row[name])
        _version(row["base_version"])
        if row["base_version"] not in versions:
            raise SnapshotError("Missing growth base core version")
        _number(row["created_at"])
        _number(row["reviewed_at"], nullable=True)
        references(row, scope)
        if row["candidate_id"] in growth_keys:
            raise SnapshotError("Duplicate growth candidate")
        growth_keys.add(row["candidate_id"])
    if any(
        row["candidate_id"] is not None and row["candidate_id"] not in growth_keys
        for row in persona["core_versions"]
    ):
        raise SnapshotError("Missing core growth provenance")

    encoded = _canonical(snapshot)
    if len(encoded) > MAX_SNAPSHOT_BYTES:
        raise SnapshotError("Snapshot byte limit exceeded")
    return json.loads(encoded)


def _manifest(snapshot: dict) -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "sha256": hashlib.sha256(_canonical(snapshot)).hexdigest(),
        "counts": {
            "core_versions": len(snapshot["persona"]["core_versions"]),
            "state": len(snapshot["state"]),
            "memories": len(snapshot["memories"]["records"]),
            "evidence": len(snapshot["memories"]["evidence"]),
            "audit": len(snapshot["memories"]["audit"]),
            "growth": len(snapshot["growth"]),
        },
    }


def _key(key: bytes) -> None:
    if type(key) is not bytes or len(key) != 32:
        raise SnapshotError("An external 32-byte AES-256 key is required")


def seal_snapshot(
    snapshot: dict, key: bytes, *, created_at: float | None = None
) -> bytes:
    """Return an authenticated encrypted version; neither key nor identity is cleartext."""
    _key(key)
    selected = validate_snapshot(snapshot)
    timestamp = time.time() if created_at is None else created_at
    _number(timestamp)
    header = {
        "format": FORMAT,
        "schema_version": SCHEMA_VERSION,
        "cipher": "AES-256-GCM",
        "version_id": secrets.token_hex(16),
        "created_at": timestamp,
    }
    payload = _canonical({"snapshot": selected, "manifest": _manifest(selected)})
    nonce = secrets.token_bytes(12)
    ciphertext = _aesgcm()(key).encrypt(nonce, payload, _canonical(header))
    return _canonical(
        header
        | {
            "nonce": base64.b64encode(nonce).decode("ascii"),
            "ciphertext": base64.b64encode(ciphertext).decode("ascii"),
        }
    )


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise BackupIntegrityError("Duplicate serialized field")
        result[key] = value
    return result


def _decode(raw: bytes) -> Any:
    try:
        return json.loads(raw, object_pairs_hook=_unique_object)
    except (ValueError, UnicodeError, RecursionError) as exc:
        raise BackupIntegrityError("Invalid serialized backup") from exc


def open_snapshot(envelope: bytes, key: bytes) -> dict:
    """Authenticate, validate schema/hash/counts and return inert detached data.

    This never imports a database, applies a persona, changes an account,
    schedules tasks, or replays a pending send.
    """
    _key(key)
    if type(envelope) is not bytes or len(envelope) > MAX_ENVELOPE_BYTES:
        raise BackupIntegrityError("Backup byte limit exceeded or invalid input")
    try:
        encoded = _decode(envelope)
        _exact(
            encoded,
            {
                "format",
                "schema_version",
                "cipher",
                "version_id",
                "created_at",
                "nonce",
                "ciphertext",
            },
        )
        if (
            encoded["format"] != FORMAT
            or type(encoded["schema_version"]) is not int
            or encoded["schema_version"] != SCHEMA_VERSION
            or encoded["cipher"] != "AES-256-GCM"
            or type(encoded["version_id"]) is not str
            or re.fullmatch(r"[0-9a-f]{32}", encoded["version_id"]) is None
        ):
            raise BackupIntegrityError("Unsupported backup format or version")
        _number(encoded["created_at"])
        header = {
            name: value
            for name, value in encoded.items()
            if name not in {"nonce", "ciphertext"}
        }
        nonce = base64.b64decode(encoded["nonce"], validate=True)
        ciphertext = base64.b64decode(encoded["ciphertext"], validate=True)
        if len(nonce) != 12 or len(ciphertext) > MAX_SNAPSHOT_BYTES + 4096:
            raise BackupIntegrityError("Invalid nonce or encrypted payload size")
        try:
            plaintext = _aesgcm()(key).decrypt(nonce, ciphertext, _canonical(header))
        except BackupDependencyError:
            raise
        except Exception:
            raise BackupIntegrityError("Backup authentication failed") from None
        payload = _decode(plaintext)
        _exact(payload, {"snapshot", "manifest"})
        snapshot = validate_snapshot(payload["snapshot"])
        expected = _canonical(_manifest(snapshot))
        if not hmac.compare_digest(_canonical(payload["manifest"]), expected):
            raise BackupIntegrityError("Backup manifest mismatch")
        return snapshot
    except BackupDependencyError:
        raise
    except BackupIntegrityError:
        raise
    except (ValueError, TypeError, KeyError, OverflowError) as exc:
        raise BackupIntegrityError("Invalid backup structure") from exc


@dataclass(frozen=True, slots=True)
class BackupHistory:
    last_success_at: float | None = None
    last_failure_at: float | None = None
    consecutive_failures: int = 0


@dataclass(frozen=True, slots=True)
class BackupDecision:
    local_checkpoint: bool
    encrypted_git_candidate: bool
    local_due_at: float | None
    git_due_at: float | None


def _due(now: float, interval: int, important: bool, history: BackupHistory) -> float:
    if not isinstance(history, BackupHistory):
        raise SnapshotError("Expected backup history")
    for stamp in (history.last_success_at, history.last_failure_at):
        _number(stamp, nullable=True)
        if stamp is not None and stamp > now:
            raise SnapshotError("Backup history cannot be in the future")
    count = history.consecutive_failures
    if type(count) is not int or not 0 <= count <= 2**31 - 1:
        raise SnapshotError("Invalid failure count")
    if count and history.last_failure_at is None:
        raise SnapshotError("Failure timestamp required")
    due = (
        now
        if important or history.last_success_at is None
        else history.last_success_at + interval
    )
    if count and (
        history.last_success_at is None
        or history.last_failure_at >= history.last_success_at
    ):
        backoff = min(60 * 2 ** min(count - 1, 6), 3600)
        due = max(due, history.last_failure_at + backoff)
    return due


def plan_backup(
    *,
    now: float,
    changed: bool,
    important_change: bool = False,
    local: BackupHistory = BackupHistory(),
    git: BackupHistory = BackupHistory(),
) -> BackupDecision:
    """Plan changed-data hourly local/daily encrypted Git candidates, without I/O.

    The caller records successful revisions, persists per-channel history, and
    performs any actual jobs separately. A Git candidate is not a commit/push.
    Important changes bypass cadence, while retries still respect backoff.
    """
    _number(now)
    if type(changed) is not bool or type(important_change) is not bool:
        raise SnapshotError("Explicit boolean change flags required")
    local_due = _due(now, 3600, important_change, local)
    git_due = _due(now, 86400, important_change, git)
    if not changed and not important_change:
        return BackupDecision(False, False, None, None)
    return BackupDecision(now >= local_due, now >= git_due, local_due, git_due)
