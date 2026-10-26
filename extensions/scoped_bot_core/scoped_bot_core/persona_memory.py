"""Local, evidence-bound member memory and explicitly reviewed persona versions.

This is a trusted-controller API, not an LLM tool or an authentication boundary.
It stores no source message bodies. Callers must register only evidence whose
identity/provenance they have verified, and authenticate owner actions upstream.
"""

import json
import math
import sqlite3
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

from .context_budget import MemoryCandidate, MemoryScope

_DIRECT = {"self_report", "observed", "owner_confirmed", "correction"}
_KINDS = {"fact", "preference", "shared_experience", "relationship"}


def _text(value, maximum=8192):
    if not isinstance(value, str) or not value.strip() or len(value) > maximum:
        raise ValueError("Non-empty bounded text is required")
    return value


def _number(value, minimum=0, maximum=None):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("A finite number is required")
    if not math.isfinite(value) or value < minimum:
        raise ValueError("Number outside allowed range")
    if maximum is not None and value > maximum:
        raise ValueError("Number outside allowed range")
    return value


def _scope(scope):
    if not isinstance(scope, MemoryScope):
        raise ValueError("A stable account/chat/member/persona scope is required")
    return (scope.account_id, scope.chat_id, scope.member_id, scope.persona_id)


def _limit(value):
    if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= 64:
        raise ValueError("Retrieval limit must be between 1 and 64")
    return value


@dataclass(frozen=True, slots=True)
class MemoryRecord:
    scope: MemoryScope
    memory_id: str
    kind: str
    content: str
    evidence_ids: tuple[str, ...]
    confidence: float
    importance: float
    observed_at: float
    created_at: float
    expires_at: float | None
    status: str
    supersedes_id: str | None

    def to_candidate(self):
        """Confidence is evidence quality, never an authorization decision."""
        if self.status != "active":
            raise ValueError("Inactive audit memory cannot become recall context")
        return MemoryCandidate(
            self.scope,
            self.memory_id,
            self.content,
            self.importance * self.confidence,
            self.expires_at,
        )


class PersonaMemoryStore:
    def __init__(self, path, *, clock=time.time):
        self.path = Path(path).resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.clock = clock
        with self._connect() as db:
            db.executescript("""
                CREATE TABLE IF NOT EXISTS pm_evidence (
                    account_id TEXT, chat_id TEXT, member_id TEXT, persona_id TEXT,
                    evidence_id TEXT, kind TEXT NOT NULL, observed_at REAL NOT NULL,
                    created_at REAL NOT NULL,
                    PRIMARY KEY(account_id,chat_id,member_id,persona_id,evidence_id)
                );
                CREATE TABLE IF NOT EXISTS pm_members (
                    account_id TEXT, chat_id TEXT, member_id TEXT,
                    nickname TEXT NOT NULL, updated_at REAL NOT NULL,
                    PRIMARY KEY(account_id,chat_id,member_id)
                );
                CREATE TABLE IF NOT EXISTS pm_memory (
                    account_id TEXT, chat_id TEXT, member_id TEXT, persona_id TEXT,
                    memory_id TEXT, kind TEXT NOT NULL, content TEXT NOT NULL,
                    evidence_ids TEXT NOT NULL, confidence REAL NOT NULL,
                    importance REAL NOT NULL, observed_at REAL NOT NULL,
                    created_at REAL NOT NULL, expires_at REAL,
                    status TEXT NOT NULL, supersedes_id TEXT,
                    PRIMARY KEY(account_id,chat_id,member_id,persona_id,memory_id)
                );
                CREATE INDEX IF NOT EXISTS pm_memory_retrieval ON pm_memory
                    (account_id,chat_id,member_id,persona_id,status,importance DESC);
                CREATE TABLE IF NOT EXISTS pm_audit (
                    account_id TEXT, chat_id TEXT, member_id TEXT, persona_id TEXT,
                    memory_id TEXT, action TEXT, evidence_ids TEXT, created_at REAL
                );
                CREATE TABLE IF NOT EXISTS pm_personas (
                    account_id TEXT, persona_id TEXT, owner_id TEXT NOT NULL,
                    current_version INTEGER NOT NULL,
                    PRIMARY KEY(account_id,persona_id)
                );
                CREATE TABLE IF NOT EXISTS pm_core_versions (
                    account_id TEXT, persona_id TEXT, version INTEGER,
                    core_text TEXT NOT NULL, candidate_id TEXT, created_at REAL,
                    PRIMARY KEY(account_id,persona_id,version)
                );
                CREATE TABLE IF NOT EXISTS pm_growth (
                    account_id TEXT, chat_id TEXT, member_id TEXT, persona_id TEXT,
                    candidate_id TEXT, base_version INTEGER NOT NULL,
                    proposed_core TEXT NOT NULL, rationale TEXT NOT NULL,
                    evidence_ids TEXT NOT NULL, status TEXT NOT NULL,
                    created_at REAL NOT NULL, reviewed_at REAL,
                    PRIMARY KEY(account_id,chat_id,member_id,persona_id,candidate_id)
                );
                CREATE TABLE IF NOT EXISTS pm_state (
                    account_id TEXT, chat_id TEXT, member_id TEXT, persona_id TEXT,
                    version INTEGER NOT NULL, state_text TEXT NOT NULL,
                    evidence_ids TEXT NOT NULL, updated_at REAL NOT NULL,
                    PRIMARY KEY(account_id,chat_id,member_id,persona_id)
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

    def _now(self):
        return _number(self.clock())

    def register_evidence(self, scope, evidence_id, *, kind, observed_at):
        """Provenance assertion by a trusted capture/controller, not by a model."""
        key = _scope(scope)
        _text(evidence_id, 256)
        _text(kind, 32)
        if kind not in _DIRECT | {"quote", "joke", "inference"}:
            raise ValueError("Unsupported evidence kind")
        now = self._now()
        _number(observed_at, maximum=now)
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            existing = db.execute(
                "SELECT kind,observed_at FROM pm_evidence WHERE account_id=? "
                "AND chat_id=? AND member_id=? AND persona_id=? AND evidence_id=?",
                (*key, evidence_id),
            ).fetchone()
            if existing:
                if (existing["kind"], existing["observed_at"]) != (kind, observed_at):
                    raise ValueError("Evidence identity is immutable")
                return False
            db.execute(
                "INSERT INTO pm_evidence VALUES(?,?,?,?,?,?,?,?)",
                (*key, evidence_id, kind, observed_at, now),
            )
        return True

    def _evidence(self, db, scope, evidence_ids):
        if (
            not isinstance(evidence_ids, (tuple, list))
            or not 1 <= len(evidence_ids) <= 16
        ):
            raise ValueError("One to sixteen explicit evidence IDs are required")
        ids = tuple(dict.fromkeys(_text(item, 256) for item in evidence_ids))
        observed = []
        for evidence_id in ids:
            row = db.execute(
                "SELECT kind,observed_at FROM pm_evidence WHERE account_id=? "
                "AND chat_id=? AND member_id=? AND persona_id=? AND evidence_id=?",
                (*_scope(scope), evidence_id),
            ).fetchone()
            if row is None or row["kind"] not in _DIRECT:
                raise ValueError("Missing, out-of-scope or indirect evidence")
            observed.append(row["observed_at"])
        return ids, max(observed)

    def update_nickname(self, scope, nickname):
        key = _scope(scope)
        _text(nickname, 128)
        with self._connect() as db:
            db.execute(
                "INSERT INTO pm_members VALUES(?,?,?,?,?) ON CONFLICT "
                "(account_id,chat_id,member_id) DO UPDATE SET "
                "nickname=excluded.nickname,updated_at=excluded.updated_at",
                (*key[:3], nickname, self._now()),
            )

    def _row(self, db, scope, memory_id):
        _text(memory_id, 256)
        row = db.execute(
            "SELECT * FROM pm_memory WHERE account_id=? AND chat_id=? "
            "AND member_id=? AND persona_id=? AND memory_id=?",
            (*_scope(scope), memory_id),
        ).fetchone()
        if row is None:
            raise ValueError("Unknown memory in selected scope")
        return row

    @staticmethod
    def _record(row):
        return MemoryRecord(
            MemoryScope(
                *(
                    row[key]
                    for key in ("account_id", "chat_id", "member_id", "persona_id")
                )
            ),
            row["memory_id"],
            row["kind"],
            row["content"],
            tuple(json.loads(row["evidence_ids"])),
            row["confidence"],
            row["importance"],
            row["observed_at"],
            row["created_at"],
            row["expires_at"],
            row["status"],
            row["supersedes_id"],
        )

    def remember(
        self,
        scope,
        *,
        kind,
        content,
        evidence_ids,
        confidence,
        importance,
        expires_at=None,
        supersedes_id=None,
    ):
        key = _scope(scope)
        _text(content)
        _text(kind, 32)
        if kind not in _KINDS:
            raise ValueError("Unsupported factual memory kind")
        _number(confidence, maximum=1)
        _number(importance, maximum=1)
        now = self._now()
        if expires_at is not None:
            _number(expires_at, minimum=now)
            if expires_at == now:
                raise ValueError("New memory must not already be expired")
        memory_id = uuid.uuid4().hex
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            ids, observed_at = self._evidence(db, scope, evidence_ids)
            if supersedes_id is not None:
                prior = self._row(db, scope, supersedes_id)
                if prior["status"] != "active":
                    raise ValueError("Only an active memory may be corrected")
            db.execute(
                "INSERT INTO pm_memory VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    *key,
                    memory_id,
                    kind,
                    content,
                    json.dumps(ids),
                    confidence,
                    importance,
                    observed_at,
                    now,
                    expires_at,
                    "active",
                    supersedes_id,
                ),
            )
            if supersedes_id is not None:
                db.execute(
                    "UPDATE pm_memory SET status='superseded' WHERE account_id=? "
                    "AND chat_id=? AND member_id=? AND persona_id=? AND memory_id=?",
                    (*key, supersedes_id),
                )
            db.execute(
                "INSERT INTO pm_audit VALUES(?,?,?,?,?,?,?,?)",
                (
                    *key,
                    memory_id,
                    "corrected" if supersedes_id else "created",
                    json.dumps(ids),
                    now,
                ),
            )
            return self._record(self._row(db, scope, memory_id))

    def revoke(self, scope, memory_id, *, evidence_ids):
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            ids, _ = self._evidence(db, scope, evidence_ids)
            row = self._row(db, scope, memory_id)
            if row["status"] != "active":
                raise ValueError("Only active memory may be revoked")
            db.execute(
                "UPDATE pm_memory SET status='revoked' WHERE account_id=? "
                "AND chat_id=? AND member_id=? AND persona_id=? AND memory_id=?",
                (*_scope(scope), memory_id),
            )
            db.execute(
                "INSERT INTO pm_audit VALUES(?,?,?,?,?,?,?,?)",
                (*_scope(scope), memory_id, "revoked", json.dumps(ids), self._now()),
            )

    def get(self, scope, memory_id):
        """Explicit audit lookup includes revoked/superseded content."""
        with self._connect() as db:
            return self._record(self._row(db, scope, memory_id))

    def retrieve(self, scope, *, limit=8, min_confidence=0.0):
        _limit(limit)
        _number(min_confidence, maximum=1)
        with self._connect() as db:
            rows = db.execute(
                "SELECT * FROM pm_memory WHERE account_id=? AND chat_id=? "
                "AND member_id=? AND persona_id=? AND status='active' "
                "AND confidence>=? AND (expires_at IS NULL OR expires_at>?) "
                "ORDER BY importance*confidence DESC,observed_at DESC,memory_id LIMIT ?",
                (*_scope(scope), min_confidence, self._now(), limit),
            ).fetchall()
            return tuple(self._record(row) for row in rows)

    def candidates(self, scope, **options):
        return tuple(
            record.to_candidate() for record in self.retrieve(scope, **options)
        )

    def find_active(self, scope, contents, *, limit=2):
        """Exact scoped content lookup for user-directed correction/revocation.

        No nickname, substring or semantic guess can select another fact. The
        caller must refuse ambiguity instead of taking an arbitrary first row.
        """
        _limit(limit)
        if not isinstance(contents, tuple) or not 1 <= len(contents) <= 4:
            raise ValueError("A bounded tuple of exact memory contents is required")
        for content in contents:
            _text(content)
        with self._connect() as db:
            rows = db.execute(
                "SELECT * FROM pm_memory WHERE account_id=? AND chat_id=? "
                "AND member_id=? AND persona_id=? AND status='active' "
                "AND (expires_at IS NULL OR expires_at>?) AND content IN ("
                + ",".join("?" for _ in contents)
                + ") ORDER BY created_at DESC,memory_id LIMIT ?",
                (*_scope(scope), self._now(), *contents, limit),
            ).fetchall()
            return tuple(self._record(row) for row in rows)

    def recent_observation_count(self, scope, *, since, prefix, limit=8):
        """A capped count, including withdrawn facts so quota cannot be reset."""
        _number(since)
        _text(prefix, 64)
        _limit(limit)
        with self._connect() as db:
            rows = db.execute(
                "SELECT 1 FROM pm_memory WHERE account_id=? AND chat_id=? "
                "AND member_id=? AND persona_id=? AND kind='preference' "
                "AND created_at>=? AND substr(content,1,?)=? LIMIT ?",
                (*_scope(scope), since, len(prefix), prefix, limit),
            ).fetchall()
            return len(rows)

    def create_persona(self, account_id, persona_id, *, owner_id, core_text):
        for value in (account_id, persona_id, owner_id):
            _text(value, 256)
        _text(core_text, 32768)
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            if db.execute(
                "SELECT 1 FROM pm_personas WHERE account_id=? AND persona_id=?",
                (account_id, persona_id),
            ).fetchone():
                raise ValueError("Persona core already exists")
            db.execute(
                "INSERT INTO pm_personas VALUES(?,?,?,1)",
                (account_id, persona_id, owner_id),
            )
            db.execute(
                "INSERT INTO pm_core_versions VALUES(?,?,1,?,NULL,?)",
                (account_id, persona_id, core_text, self._now()),
            )

    def core(self, account_id, persona_id, *, version=None):
        _text(account_id, 256)
        _text(persona_id, 256)
        if version is not None:
            if isinstance(version, bool) or not isinstance(version, int) or version < 1:
                raise ValueError("Version must be a positive integer")
        with self._connect() as db:
            if version is None:
                row = db.execute(
                    "SELECT current_version FROM pm_personas WHERE account_id=? AND persona_id=?",
                    (account_id, persona_id),
                ).fetchone()
                if row is None:
                    raise ValueError("Unknown persona")
                version = row[0]
            row = db.execute(
                "SELECT version,core_text,candidate_id,created_at FROM pm_core_versions "
                "WHERE account_id=? AND persona_id=? AND version=?",
                (account_id, persona_id, version),
            ).fetchone()
            if row is None:
                raise ValueError("Unknown persona version")
            return dict(row)

    def set_state(self, scope, *, state_text, evidence_ids, expected_version=0):
        """Scoped recent state may change; it cannot overwrite persona core."""
        _text(state_text)
        if (
            isinstance(expected_version, bool)
            or not isinstance(expected_version, int)
            or expected_version < 0
        ):
            raise ValueError("Expected version must be a nonnegative integer")
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            ids, _ = self._evidence(db, scope, evidence_ids)
            row = db.execute(
                "SELECT version FROM pm_state WHERE account_id=? AND chat_id=? "
                "AND member_id=? AND persona_id=?",
                _scope(scope),
            ).fetchone()
            version = row[0] if row else 0
            if version != expected_version:
                raise ValueError("State version changed")
            db.execute(
                "INSERT INTO pm_state VALUES(?,?,?,?,?,?,?,?) ON CONFLICT "
                "(account_id,chat_id,member_id,persona_id) DO UPDATE SET "
                "version=excluded.version,state_text=excluded.state_text,"
                "evidence_ids=excluded.evidence_ids,updated_at=excluded.updated_at",
                (*_scope(scope), version + 1, state_text, json.dumps(ids), self._now()),
            )
            return version + 1

    def state(self, scope):
        with self._connect() as db:
            row = db.execute(
                "SELECT version,state_text,evidence_ids,updated_at FROM pm_state "
                "WHERE account_id=? AND chat_id=? AND member_id=? AND persona_id=?",
                _scope(scope),
            ).fetchone()
            if row is None:
                return None
            result = dict(row)
            result["evidence_ids"] = tuple(json.loads(result["evidence_ids"]))
            return result

    def propose_growth(self, scope, *, proposed_core, rationale, evidence_ids):
        _text(proposed_core, 32768)
        _text(rationale)
        candidate_id = uuid.uuid4().hex
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            ids, _ = self._evidence(db, scope, evidence_ids)
            persona = db.execute(
                "SELECT current_version FROM pm_personas "
                "WHERE account_id=? AND persona_id=?",
                (scope.account_id, scope.persona_id),
            ).fetchone()
            if persona is None:
                raise ValueError("Unknown persona")
            db.execute(
                "INSERT INTO pm_growth VALUES(?,?,?,?,?,?,?,?,?,?,?,NULL)",
                (
                    *_scope(scope),
                    candidate_id,
                    persona[0],
                    proposed_core,
                    rationale,
                    json.dumps(ids),
                    "pending",
                    self._now(),
                ),
            )
        return candidate_id

    def review_growth(self, scope, candidate_id, *, authenticated_actor_id, accept):
        """Only an upstream-authenticated owner may explicitly version the core.

        Never expose authenticated_actor_id as a model-controlled tool argument.
        Caller must re-authenticate the actual message sender on every review.
        """
        _scope(scope)
        _text(candidate_id, 256)
        _text(authenticated_actor_id, 256)
        if not isinstance(accept, bool):
            raise ValueError("Explicit boolean decision is required")
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            persona = db.execute(
                "SELECT * FROM pm_personas WHERE account_id=? AND persona_id=?",
                (scope.account_id, scope.persona_id),
            ).fetchone()
            if persona is None or persona["owner_id"] != authenticated_actor_id:
                raise ValueError("Authenticated owner is required")
            candidate = db.execute(
                "SELECT * FROM pm_growth WHERE account_id=? AND chat_id=? "
                "AND member_id=? AND persona_id=? AND candidate_id=?",
                (*_scope(scope), candidate_id),
            ).fetchone()
            if candidate is None or candidate["status"] != "pending":
                raise ValueError("Unknown or reviewed growth candidate")
            if accept and persona["current_version"] != candidate["base_version"]:
                raise ValueError(
                    "Core version changed; propose a fresh reviewed change"
                )
            now = self._now()
            version = persona["current_version"]
            if accept:
                version += 1
                db.execute(
                    "INSERT INTO pm_core_versions VALUES(?,?,?,?,?,?)",
                    (
                        scope.account_id,
                        scope.persona_id,
                        version,
                        candidate["proposed_core"],
                        candidate_id,
                        now,
                    ),
                )
                db.execute(
                    "UPDATE pm_personas SET current_version=? WHERE account_id=? AND persona_id=?",
                    (version, scope.account_id, scope.persona_id),
                )
            db.execute(
                "UPDATE pm_growth SET status=?,reviewed_at=? WHERE account_id=? AND chat_id=? "
                "AND member_id=? AND persona_id=? AND candidate_id=?",
                (
                    "accepted" if accept else "rejected",
                    now,
                    *_scope(scope),
                    candidate_id,
                ),
            )
            return version

    def export_snapshot(self, account_id, persona_id, *, max_rows=10000):
        """Export all selected persona data coherently or fail; never truncate.

        Includes inactive memory and audit history because inheritance needs
        corrections and revocations. This plaintext object is private data;
        encrypt it before Git or other transport. Ownership/permissions, source
        message bodies, credentials and pending sends are deliberately absent.
        No restore/import or authorization effect is provided here.
        """
        _text(account_id, 256)
        _text(persona_id, 256)
        if (
            isinstance(max_rows, bool)
            or not isinstance(max_rows, int)
            or not 1 <= max_rows <= 100000
        ):
            raise ValueError("Snapshot max_rows must be between 1 and 100000")
        # All identifiers below are code-owned SQL constants, never caller text.
        specs = (
            ("pm_core_versions", "version,core_text,candidate_id,created_at"),
            ("pm_state", "*"),
            ("pm_memory", "*"),
            ("pm_evidence", "*"),
            ("pm_audit", "*"),
            ("pm_growth", "*"),
        )
        exported = {}
        with self._connect() as db:
            db.execute("BEGIN")
            if not db.execute(
                "SELECT 1 FROM pm_personas WHERE account_id=? AND persona_id=?",
                (account_id, persona_id),
            ).fetchone():
                raise ValueError("Unknown persona")
            total = 0
            for table, columns in specs:
                where = " WHERE account_id=? AND persona_id=?"
                total += db.execute(
                    f"SELECT COUNT(*) FROM {table}" + where,
                    (account_id, persona_id),
                ).fetchone()[0]
                if total > max_rows:
                    raise ValueError("Full snapshot exceeds max_rows; nothing exported")
                rows = db.execute(
                    f"SELECT {columns} FROM {table}" + where + " ORDER BY rowid",
                    (account_id, persona_id),
                ).fetchall()
                values = []
                for row in rows:
                    value = dict(row)
                    if "evidence_ids" in value:
                        value["evidence_ids"] = json.loads(value["evidence_ids"])
                    values.append(value)
                exported[table] = values
        return {
            "persona": {
                "account_id": account_id,
                "persona_id": persona_id,
                "core_versions": exported["pm_core_versions"],
            },
            "state": exported["pm_state"],
            "memories": {
                "records": exported["pm_memory"],
                "evidence": exported["pm_evidence"],
                "audit": exported["pm_audit"],
            },
            "growth": exported["pm_growth"],
        }
