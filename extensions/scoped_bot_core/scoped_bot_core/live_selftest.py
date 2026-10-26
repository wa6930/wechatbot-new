"""One-shot FileHelper admission and proof; no GUI, WeChat, or VM imports.

The injected GUI implementation is a trusted local controller, never an LLM
tool. It must prove a stable peer/builtin identity independently of display
titles, and staging must never send. Database implementations must return an
all-source-shard baseline and complete bounded fresh reads. This module cannot
turn a title or an unverified boolean into that proof.

An unknown outcome is durably recorded *before* the single Enter attempt.
Restart/reconciliation can inspect evidence but can never dispatch again.
"""

from __future__ import annotations

import json
import math
import re
import sqlite3
import time
from collections.abc import Callable, Mapping
from contextlib import AbstractContextManager, closing
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Protocol

from .delivery import DeliveryIntent, DeliveryLedger, make_action_id, text_sha256

FILEHELPER = "filehelper"
_MARKER = re.compile(r"ASTRBOT_SELFTEST_[0-9A-Fa-f]{16,64}\Z", re.ASCII)
_SHA256 = re.compile(r"[0-9a-f]{64}\Z", re.ASCII)
_EVIDENCE_KINDS = {"stable_peer_id", "builtin_filehelper"}


class SelftestRejected(ValueError):
    """A content-free reason code, safe to include in local status output."""


def _identity(value, reason):
    if (
        not isinstance(value, str)
        or not value
        or len(value) > 512
        or any(character.isspace() or ord(character) < 32 for character in value)
    ):
        raise SelftestRejected(reason)
    return value


def _integer(value, reason):
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise SelftestRejected(reason)
    return value


def _marker(value):
    if not isinstance(value, str) or not _MARKER.fullmatch(value):
        raise SelftestRejected("invalid_selftest_marker")
    return value


@dataclass(frozen=True)
class TargetBinding:
    account_id: str
    chat_id: str
    evidence_kind: str
    binding_id: str
    window_id: str
    input_id: str
    unlocked: bool
    foreground: bool
    focused: bool

    def key(self):
        return (
            self.account_id,
            self.chat_id,
            self.evidence_kind,
            self.binding_id,
            self.window_id,
            self.input_id,
        )

    def validate(self, account_id, *, focused=False):
        if self.account_id != account_id or self.chat_id != FILEHELPER:
            raise SelftestRejected("target_scope_mismatch")
        if self.evidence_kind not in _EVIDENCE_KINDS:
            raise SelftestRejected("unproven_target_binding")
        for value in (self.binding_id, self.window_id, self.input_id):
            _identity(value, "invalid_target_binding")
        if self.unlocked is not True or self.foreground is not True:
            raise SelftestRejected("desktop_not_ready")
        if focused and self.focused is not True:
            raise SelftestRejected("input_not_focused")


@dataclass(frozen=True)
class ShardBoundary:
    shard: str
    max_local_id: int
    max_sort_seq: int = 0


@dataclass(frozen=True)
class Watermark:
    account_id: str
    chat_id: str
    boundaries: tuple[ShardBoundary, ...]
    complete: bool

    def validate(self, account_id):
        if self.account_id != account_id or self.chat_id != FILEHELPER:
            raise SelftestRejected("watermark_scope_mismatch")
        if self.complete is not True or not self.boundaries:
            raise SelftestRejected("incomplete_watermark")
        seen = set()
        for boundary in self.boundaries:
            if not isinstance(boundary, ShardBoundary):
                raise SelftestRejected("invalid_watermark")
            _identity(boundary.shard, "invalid_shard")
            _integer(boundary.max_local_id, "invalid_boundary")
            _integer(boundary.max_sort_seq, "invalid_boundary")
            if boundary.shard in seen:
                raise SelftestRejected("duplicate_shard")
            seen.add(boundary.shard)

    @classmethod
    def from_probe(cls, probe: Mapping):
        """Accept the guest's private metadata, never the hashed host report.

        ``complete`` must certify that every observed source message shard was
        read, including zero baselines for shards without a FileHelper table,
        and that the shard inventory stayed unchanged during that read.
        """
        if (
            not isinstance(probe, Mapping)
            or type(probe.get("schema_version")) is not int
            or probe["schema_version"] != 1
            or probe.get("approved_chat_ids") != [FILEHELPER]
            or not isinstance(probe.get("watermarks"), list)
        ):
            raise SelftestRejected("invalid_probe_metadata")
        try:
            boundaries = tuple(
                ShardBoundary(row["shard"], row["max_local_id"], row["max_sort_seq"])
                for row in probe["watermarks"]
            )
        except (KeyError, TypeError):
            raise SelftestRejected("invalid_probe_metadata") from None
        result = cls(
            probe.get("account_id"), FILEHELPER, boundaries, probe.get("complete")
        )
        _identity(result.account_id, "invalid_account")
        result.validate(result.account_id)
        return result


@dataclass(frozen=True)
class ReadbackRow:
    account_id: str
    chat_id: str
    shard: str
    local_id: int
    sender_username: str
    local_type: int
    content: str


@dataclass(frozen=True)
class ReadbackBatch:
    account_id: str
    chat_id: str
    shards: tuple[str, ...]
    complete: bool
    rows: tuple[ReadbackRow, ...]


@dataclass(frozen=True)
class SelftestProof:
    action_id: str
    account_id: str
    text_sha256: str
    watermark: Watermark
    binding_key: tuple[str, ...]


class ExclusiveLease(Protocol):
    def assert_owned(self) -> None:
        """Raise unless this controller still exclusively owns the desktop."""


class SelftestGui(Protocol):
    def prepare_filehelper(self) -> TargetBinding:
        """Open only the proven builtin FileHelper; never send or clear drafts."""

    def inspect_target(self) -> TargetBinding:
        """Freshly inspect identity, window, input, desktop, and focus."""

    def read_draft(self) -> str | None: ...

    def stage_marker(self, marker: str) -> None:
        """Revalidate builtin identity/control, input without Enter or fallback.

        Never clear a nonempty draft; fail unless the same proven input control
        still belongs to the expected WeChat process/window and is focused.
        """

    def press_enter_once(self) -> None:
        """Revalidate the bound control/PID/focus, then attempt one Enter.

        No blind global Enter, built-in retries, or fallback is permitted.
        """


class SelftestDatabase(Protocol):
    def watermark(self) -> Watermark: ...

    def read_new_rows(self, watermark: Watermark) -> ReadbackBatch:
        """Bounded fresh complete query; preserve stable peer/self/shard identity."""


class ProofStore(Protocol):
    def save(self, proof: SelftestProof) -> None: ...

    def load(self, action_id: str) -> SelftestProof | None: ...


class SqliteProofStore:
    """Durable immutable metadata only; no marker, draft, or chat body is stored.

    The caller chooses a private local runtime directory, outside Git/shared
    folders. This class does not configure Windows directory ACLs.
    """

    def __init__(self, path):
        self.path = Path(path).expanduser().resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with closing(sqlite3.connect(self.path)) as db:
            db.execute("PRAGMA journal_mode=WAL")
            db.execute("PRAGMA synchronous=FULL")
            db.execute(
                "CREATE TABLE IF NOT EXISTS live_selftest_proofs "
                "(action_id TEXT PRIMARY KEY, proof_json TEXT NOT NULL)"
            )
            db.commit()

    def save(self, proof):
        encoded = json.dumps(asdict(proof), sort_keys=True, separators=(",", ":"))
        with closing(sqlite3.connect(self.path)) as db:
            db.execute("PRAGMA synchronous=FULL")
            with db:
                db.execute(
                    "INSERT OR IGNORE INTO live_selftest_proofs VALUES(?,?)",
                    (proof.action_id, encoded),
                )
                existing = db.execute(
                    "SELECT proof_json FROM live_selftest_proofs WHERE action_id=?",
                    (proof.action_id,),
                ).fetchone()
                if existing[0] != encoded:
                    raise SelftestRejected("proof_conflict")

    def load(self, action_id):
        with closing(sqlite3.connect(self.path)) as db:
            row = db.execute(
                "SELECT proof_json FROM live_selftest_proofs WHERE action_id=?",
                (action_id,),
            ).fetchone()
        if row is None:
            return None
        try:
            raw = json.loads(row[0])
            watermark = raw["watermark"]
            return SelftestProof(
                raw["action_id"],
                raw["account_id"],
                raw["text_sha256"],
                Watermark(
                    watermark["account_id"],
                    watermark["chat_id"],
                    tuple(ShardBoundary(**value) for value in watermark["boundaries"]),
                    watermark["complete"],
                ),
                tuple(raw["binding_key"]),
            )
        except (KeyError, TypeError, ValueError):
            raise SelftestRejected("invalid_saved_proof") from None


def matching_receipt(proof: SelftestProof, marker: str, batch: ReadbackBatch):
    """Validate a complete fixed-scope read and accept exactly one new own row."""
    _marker(marker)
    proof.watermark.validate(proof.account_id)
    if proof.text_sha256 != text_sha256(marker):
        raise SelftestRejected("marker_hash_mismatch")
    if (
        not isinstance(batch, ReadbackBatch)
        or batch.complete is not True
        or batch.account_id != proof.account_id
        or batch.chat_id != FILEHELPER
    ):
        raise SelftestRejected("incomplete_or_wrong_readback")
    boundaries = {
        value.shard: value.max_local_id for value in proof.watermark.boundaries
    }
    if (
        not isinstance(batch.shards, tuple)
        or len(batch.shards) != len(set(batch.shards))
        or set(batch.shards) != set(boundaries)
    ):
        raise SelftestRejected("readback_shards_changed")
    matches, seen = [], set()
    for row in batch.rows:
        if (
            not isinstance(row, ReadbackRow)
            or row.account_id != proof.account_id
            or row.chat_id != FILEHELPER
            or row.shard not in boundaries
        ):
            raise SelftestRejected("row_scope_mismatch")
        _integer(row.local_id, "invalid_row_identity")
        key = (row.shard, row.local_id)
        if key in seen:
            raise SelftestRejected("duplicate_readback_row")
        seen.add(key)
        if row.local_id <= boundaries[row.shard]:
            continue
        if (
            row.sender_username == proof.account_id
            and type(row.local_type) is int
            and row.local_type == 1
            and row.content == marker
        ):
            matches.append(row)
    if len(matches) > 1:
        raise SelftestRejected("multiple_matching_rows")
    if not matches:
        return None
    row = matches[0]
    return {
        "source": "db_readback",
        "account_id": proof.account_id,
        "chat_id": FILEHELPER,
        "message_id": json.dumps([row.shard, row.local_id], separators=(",", ":")),
        "text_sha256": proof.text_sha256,
    }


class FileHelperSelftest:
    """Explicit local selftest only; this class is not an AstrBot live sender."""

    def __init__(
        self,
        *,
        ledger: DeliveryLedger,
        proofs: ProofStore,
        gui: SelftestGui,
        database: SelftestDatabase,
        lease_factory: Callable[[], AbstractContextManager[ExclusiveLease]],
        account_id: str,
        owner_id: str,
        max_readbacks: int = 5,
        poll_interval: float = 1,
        sleep: Callable[[float], None] = time.sleep,
    ):
        self.account_id = _identity(account_id, "invalid_account")
        self.owner_id = _identity(owner_id, "invalid_owner")
        if type(max_readbacks) is not int or not 1 <= max_readbacks <= 30:
            raise ValueError("max_readbacks must be between 1 and 30")
        if (
            isinstance(poll_interval, bool)
            or not isinstance(poll_interval, (int, float))
            or not math.isfinite(poll_interval)
            or not 0 <= poll_interval <= 5
        ):
            raise ValueError("poll_interval must be between 0 and 5 seconds")
        self.ledger, self.proofs = ledger, proofs
        self.gui, self.database, self.lease_factory = gui, database, lease_factory
        self.max_readbacks, self.poll_interval, self.sleep = (
            max_readbacks,
            poll_interval,
            sleep,
        )

    def _poll(self, proof, marker, lease):
        for attempt in range(self.max_readbacks):
            lease.assert_owned()
            receipt = matching_receipt(
                proof, marker, self.database.read_new_rows(proof.watermark)
            )
            if receipt is not None:
                return self.ledger.reconcile_unknown(proof.action_id, receipt)
            if attempt + 1 < self.max_readbacks:
                self.sleep(self.poll_interval)
        return self.ledger.get(proof.action_id)

    def run(self, *, request_id: str, marker: str) -> DeliveryIntent:
        """At most one Enter for an immutable request; never retry unknown work."""
        marker = _marker(marker)
        intent = self.ledger.prepare(
            request_id=request_id,
            account_id=self.account_id,
            chat_id=FILEHELPER,
            text=marker,
            mode="live",
        )
        if intent.state != "prepared":
            return intent
        with self.lease_factory() as lease:
            lease.assert_owned()
            claim = self.ledger.claim(intent.action_id, self.owner_id)
            if claim is None:
                return self.ledger.get(intent.action_id)
            try:
                binding = self.gui.prepare_filehelper()
                binding.validate(self.account_id)
                watermark = self.database.watermark()
                watermark.validate(self.account_id)
                if self.gui.read_draft() != "":
                    raise SelftestRejected("draft_not_empty_or_unreadable")
                proof = SelftestProof(
                    intent.action_id,
                    self.account_id,
                    text_sha256(marker),
                    watermark,
                    binding.key(),
                )
                self.proofs.save(proof)
                lease.assert_owned()
                # The DB snapshot and durable proof can take time. Reject a
                # moved conversation before even leaving a marker draft in it.
                before_stage = self.gui.inspect_target()
                before_stage.validate(self.account_id, focused=True)
                if before_stage.key() != binding.key():
                    raise SelftestRejected("target_binding_changed")
                if self.gui.read_draft() != "":
                    raise SelftestRejected("draft_not_empty_or_unreadable")
                self.gui.stage_marker(marker)
                observed = self.gui.inspect_target()
                observed.validate(self.account_id, focused=True)
                if observed.key() != binding.key():
                    raise SelftestRejected("target_binding_changed")
                if self.gui.read_draft() != marker:
                    raise SelftestRejected("draft_readback_mismatch")
                # Recheck identity/focus after draft readback, immediately before
                # arming dispatch. Never replace this with a cached title.
                final = self.gui.inspect_target()
                final.validate(self.account_id, focused=True)
                if final.key() != binding.key():
                    raise SelftestRejected("target_binding_changed")
                lease.assert_owned()
            except BaseException as error:
                reason = (
                    str(error)
                    if isinstance(error, SelftestRejected)
                    else "preflight_failed"
                )
                result = self.ledger.record_not_sent(
                    intent.action_id, claim.claim_token, reason
                )
                if not isinstance(error, Exception):
                    raise
                return result
            # Persist conservative uncertainty before the only operation that
            # could send. A crash between these lines may mean no actual send;
            # that still MUST NOT authorize another Enter.
            self.ledger.record_unknown(
                intent.action_id, claim.claim_token, "dispatch_armed"
            )
            try:
                lease.assert_owned()
                armed_target = self.gui.inspect_target()
                armed_target.validate(self.account_id, focused=True)
                if armed_target.key() != binding.key():
                    raise SelftestRejected("target_binding_changed")
                if self.gui.read_draft() != marker:
                    raise SelftestRejected("draft_readback_mismatch")
                self.gui.press_enter_once()
            except Exception:
                # An exception may follow a successfully delivered key. Read
                # only; never attempt Enter or any fallback a second time.
                pass
            try:
                return self._poll(proof, marker, lease)
            except Exception:
                return self.ledger.get(intent.action_id)

    def reconcile(self, *, request_id: str, marker: str) -> DeliveryIntent:
        """Read-only recovery, including after a crash; this never calls GUI."""
        marker = _marker(marker)
        action_id = make_action_id(request_id)
        intent = self.ledger.get(action_id)
        if intent is None or intent.state != "unknown":
            raise SelftestRejected("not_an_unknown_action")
        if (
            intent.account_id != self.account_id
            or intent.chat_id != FILEHELPER
            or intent.mode != "live"
            or intent.text_sha256 != text_sha256(marker)
        ):
            raise SelftestRejected("action_scope_mismatch")
        with self.lease_factory() as lease:
            lease.assert_owned()
            proof = self.proofs.load(action_id)
            if (
                proof is None
                or proof.action_id != action_id
                or proof.account_id != self.account_id
                or proof.text_sha256 != intent.text_sha256
                or not _SHA256.fullmatch(proof.text_sha256)
                or len(proof.binding_key) != 6
                or proof.binding_key[0] != self.account_id
                or proof.binding_key[1] != FILEHELPER
                or proof.binding_key[2] not in _EVIDENCE_KINDS
            ):
                raise SelftestRejected("missing_or_invalid_saved_proof")
            try:
                return self._poll(proof, marker, lease)
            except Exception:
                return self.ledger.get(action_id)
