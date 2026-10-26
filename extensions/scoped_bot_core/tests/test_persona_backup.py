"""Synthetic encrypted inheritance, validation and scheduling; no external I/O."""

import base64
import copy
import json
import math
import unittest
from unittest.mock import patch

from scoped_bot_core.persona_backup import (
    MAX_ENVELOPE_BYTES,
    MAX_STRING_BYTES,
    BackupDependencyError,
    BackupHistory,
    BackupIntegrityError,
    SnapshotError,
    open_snapshot,
    plan_backup,
    seal_snapshot,
    validate_snapshot,
)

KEY = bytes(range(32))  # Public synthetic test key; never use for real data.
SCOPE = {
    "account_id": "synthetic-account",
    "persona_id": "synthetic-persona",
    "chat_id": "synthetic-chat",
    "member_id": "synthetic-member",
}


def snapshot():
    return {
        "persona": {
            "account_id": SCOPE["account_id"],
            "persona_id": SCOPE["persona_id"],
            "core_versions": [
                {
                    "version": 1,
                    "core_text": "先倾听，有自己的分寸。",
                    "candidate_id": None,
                    "created_at": 100.0,
                }
            ],
        },
        "state": [
            SCOPE
            | {
                "version": 1,
                "state_text": "平静",
                "evidence_ids": ["e1"],
                "updated_at": 101.0,
            }
        ],
        "memories": {
            "evidence": [
                SCOPE
                | {
                    "evidence_id": "e1",
                    "kind": "self_report",
                    "observed_at": 100.0,
                    "created_at": 101.0,
                }
            ],
            "records": [
                SCOPE
                | {
                    "memory_id": "m1",
                    "kind": "preference",
                    "content": "喜欢不加糖的茶。",
                    "evidence_ids": ["e1"],
                    "confidence": 0.9,
                    "importance": 0.7,
                    "observed_at": 100.0,
                    "created_at": 101.0,
                    "expires_at": None,
                    "status": "active",
                    "supersedes_id": None,
                }
            ],
            "audit": [
                SCOPE
                | {
                    "memory_id": "m1",
                    "action": "created",
                    "evidence_ids": ["e1"],
                    "created_at": 101.0,
                }
            ],
        },
        "growth": [
            SCOPE
            | {
                "candidate_id": "g1",
                "base_version": 1,
                "proposed_core": "共同经历后，更愿意耐心倾听。",
                "rationale": "来自可追溯的共同经历。",
                "evidence_ids": ["e1"],
                "status": "pending",
                "created_at": 102.0,
                "reviewed_at": None,
            }
        ],
    }


def forge_payload(envelope, mutate):
    """Key-owner corruption exercises manifest/schema checks after valid AEAD."""
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM

    raw = json.loads(envelope)
    header = {
        name: value
        for name, value in raw.items()
        if name not in {"nonce", "ciphertext"}
    }
    canonical_header = json.dumps(
        header, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode()
    nonce = base64.b64decode(raw["nonce"])
    content = AESGCM(KEY).decrypt(
        nonce, base64.b64decode(raw["ciphertext"]), canonical_header
    )
    payload = json.loads(content)
    mutate(payload)
    modified = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode()
    raw["ciphertext"] = base64.b64encode(
        AESGCM(KEY).encrypt(nonce, modified, canonical_header)
    ).decode()
    return json.dumps(raw).encode()


class PersonaBackupTests(unittest.TestCase):
    def seal(self, data=None):
        return seal_snapshot(
            snapshot() if data is None else data, KEY, created_at=200.0
        )

    def test_round_trip_is_detached_and_contains_full_audit(self):
        source = snapshot()
        restored = open_snapshot(self.seal(source), KEY)
        self.assertEqual(restored, source)
        restored["memories"]["records"][0]["content"] = "changed locally"
        self.assertNotEqual(restored, source)
        self.assertEqual(source["memories"]["audit"][0]["action"], "created")

    def test_random_versions_and_nonces_do_not_expose_persona(self):
        first, second = self.seal(), self.seal()
        self.assertNotEqual(first, second)
        left, right = json.loads(first), json.loads(second)
        self.assertNotEqual(left["version_id"], right["version_id"])
        self.assertNotEqual(left["nonce"], right["nonce"])
        self.assertNotIn(SCOPE["persona_id"].encode(), first)
        self.assertNotIn("喜欢不加糖的茶".encode(), first)
        self.assertNotIn("manifest", left)
        self.assertNotIn("key", left)

    def test_wrong_key_and_ciphertext_tampering_fail(self):
        envelope = self.seal()
        with self.assertRaises(BackupIntegrityError):
            open_snapshot(envelope, b"x" * 32)
        raw = json.loads(envelope)
        cipher = bytearray(base64.b64decode(raw["ciphertext"]))
        cipher[0] ^= 1
        raw["ciphertext"] = base64.b64encode(cipher).decode()
        with self.assertRaises(BackupIntegrityError):
            open_snapshot(json.dumps(raw).encode(), KEY)

    def test_header_nonce_and_schema_tampering_fail(self):
        for changes in (
            {"created_at": 201.0},
            {"version_id": "0" * 32},
            {"nonce": base64.b64encode(b"0" * 12).decode()},
            {"schema_version": 2},
            {"cipher": "none"},
            {"format": "other"},
            {"extra": "unknown"},
            {"schema_version": True},
        ):
            with self.subTest(changes=changes):
                raw = json.loads(self.seal()) | changes
                with self.assertRaises(BackupIntegrityError):
                    open_snapshot(json.dumps(raw).encode(), KEY)

    def test_valid_aead_does_not_bypass_hash_counts_or_schema(self):
        def change_hash(payload):
            payload["manifest"]["sha256"] = "0" * 64

        def change_count(payload):
            payload["manifest"]["counts"]["audit"] = 999

        def change_data(payload):
            payload["snapshot"]["state"][0]["state_text"] = "changed"

        def add_send_state(payload):
            payload["snapshot"]["pending_tasks"] = []

        for mutate in (change_hash, change_count, change_data, add_send_state):
            with self.subTest(mutate=mutate.__name__):
                with self.assertRaises(BackupIntegrityError):
                    open_snapshot(forge_payload(self.seal(), mutate), KEY)

    def test_duplicate_json_fields_and_invalid_input_fail(self):
        for value in (
            b'{"format":"first","format":"second"}',
            b"not-json",
            b"[]",
            b"\xff",
            b"x" * (MAX_ENVELOPE_BYTES + 1),
            "text instead of bytes",
        ):
            with self.subTest(kind=type(value).__name__):
                with self.assertRaises(BackupIntegrityError):
                    open_snapshot(value, KEY)

    def test_invalid_keys_and_missing_dependency_are_explicit(self):
        for key in (b"short", b"x" * 31, b"x" * 33, "x" * 32, bytearray(32)):
            with self.subTest(kind=type(key).__name__):
                with self.assertRaises(SnapshotError):
                    seal_snapshot(snapshot(), key)
        with patch(
            "scoped_bot_core.persona_backup._aesgcm",
            side_effect=BackupDependencyError("missing"),
        ):
            with self.assertRaises(BackupDependencyError):
                self.seal()

    def test_unknown_nested_fields_credentials_and_raw_history_are_rejected(self):
        variants = []
        for target, key in (
            ("top", "history"),
            ("top", "delivery_ledger"),
            ("persona", "owner_id"),
            ("persona", "api_key"),
            ("memory", "raw_message"),
            ("growth", "tool_calls"),
        ):
            row = snapshot()
            where = {
                "top": row,
                "persona": row["persona"],
                "memory": row["memories"]["records"][0],
                "growth": row["growth"][0],
            }[target]
            where[key] = "synthetic"
            variants.append(row)
        for row in variants:
            with self.assertRaises(SnapshotError):
                self.seal(row)
        for marker in (
            "password = synthetic",
            "api_key: synthetic",
            "-----BEGIN PRIVATE KEY-----",
            "sk-" + "a" * 30,
        ):
            row = snapshot()
            row["memories"]["records"][0]["content"] = marker
            with self.assertRaises(SnapshotError):
                self.seal(row)

    def test_free_text_allowlist_is_not_a_claim_of_sanitization(self):
        row = snapshot()
        # This could be private information. Schemas cannot infer its meaning.
        row["memories"]["records"][0]["content"] = "synthetic private phrase"
        self.assertEqual(open_snapshot(self.seal(row), KEY), row)

    def test_record_bounds_and_nonfinite_values_fail_without_truncation(self):
        for content in ("x" * (MAX_STRING_BYTES + 1), "\ud800"):
            row = snapshot()
            row["memories"]["records"][0]["content"] = content
            with self.assertRaises(SnapshotError):
                self.seal(row)
        for number in (math.nan, math.inf, -1, True, 10**1000):
            row = snapshot()
            row["memories"]["records"][0]["confidence"] = number
            with self.assertRaises(SnapshotError):
                self.seal(row)
        row = snapshot()
        row["memories"]["records"] *= 10001
        with self.assertRaises(SnapshotError):
            self.seal(row)

    def test_long_cjk_core_and_unicode_identifiers_are_supported(self):
        row = snapshot()
        row["persona"]["core_versions"][0]["core_text"] = "性格" * 16384
        row["persona"]["persona_id"] = "人格" * 128
        for domain in (row["state"], row["growth"], *row["memories"].values()):
            for item in domain:
                item["persona_id"] = row["persona"]["persona_id"]
        self.assertEqual(open_snapshot(self.seal(row), KEY), row)

    def test_scope_reference_and_provenance_corruption_fail(self):
        def foreign_scope(row):
            row["memories"]["records"][0]["member_id"] = "other-member"

        def missing_evidence(row):
            row["memories"]["evidence"] = []

        def duplicate_evidence(row):
            row["memories"]["evidence"] *= 2

        def duplicate_memory(row):
            row["memories"]["records"] *= 2

        def missing_predecessor(row):
            row["memories"]["records"][0]["supersedes_id"] = "missing"

        def missing_candidate(row):
            row["persona"]["core_versions"][0]["candidate_id"] = "missing"

        def missing_version(row):
            row["growth"][0]["base_version"] = 99

        def wrong_status(row):
            row["growth"][0]["status"] = "execute_now"

        for mutate in (
            foreign_scope,
            missing_evidence,
            duplicate_evidence,
            duplicate_memory,
            missing_predecessor,
            missing_candidate,
            missing_version,
            wrong_status,
        ):
            row = snapshot()
            mutate(row)
            with self.subTest(mutate=mutate.__name__):
                with self.assertRaises(SnapshotError):
                    validate_snapshot(row)

    def test_validation_does_not_mutate_input(self):
        row = snapshot()
        original = copy.deepcopy(row)
        checked = validate_snapshot(row)
        self.assertEqual(row, original)
        checked["growth"].clear()
        self.assertEqual(row, original)


class BackupScheduleTests(unittest.TestCase):
    def test_initial_changed_data_is_due_and_unchanged_data_is_not(self):
        self.assertTrue(plan_backup(now=100, changed=True).local_checkpoint)
        self.assertTrue(plan_backup(now=100, changed=True).encrypted_git_candidate)
        unchanged = plan_backup(now=100, changed=False)
        self.assertFalse(unchanged.local_checkpoint)
        self.assertFalse(unchanged.encrypted_git_candidate)
        self.assertIsNone(unchanged.local_due_at)

    def test_hourly_local_and_daily_git_decisions_are_separate(self):
        history = BackupHistory(last_success_at=100)
        plan = plan_backup(now=3700, changed=True, local=history, git=history)
        self.assertTrue(plan.local_checkpoint)
        self.assertFalse(plan.encrypted_git_candidate)
        self.assertEqual(plan.git_due_at, 86500)
        self.assertTrue(
            plan_backup(now=86500, changed=True, git=history).encrypted_git_candidate
        )

    def test_important_change_bypasses_cadence(self):
        history = BackupHistory(last_success_at=100)
        plan = plan_backup(
            now=101, changed=False, important_change=True, local=history, git=history
        )
        self.assertTrue(plan.local_checkpoint)
        self.assertTrue(plan.encrypted_git_candidate)

    def test_git_retry_backoff_does_not_block_local_checkpoint(self):
        failed = BackupHistory(last_failure_at=100, consecutive_failures=1)
        plan = plan_backup(now=159, changed=True, git=failed)
        self.assertTrue(plan.local_checkpoint)
        self.assertFalse(plan.encrypted_git_candidate)
        self.assertEqual(plan.git_due_at, 160)
        self.assertTrue(
            plan_backup(now=160, changed=True, git=failed).encrypted_git_candidate
        )

    def test_repeated_failures_have_capped_backoff_even_for_important_change(self):
        failed = BackupHistory(last_failure_at=100, consecutive_failures=1000)
        plan = plan_backup(now=101, changed=True, important_change=True, local=failed)
        self.assertFalse(plan.local_checkpoint)
        self.assertEqual(plan.local_due_at, 3700)

    def test_success_clears_effect_of_older_failure(self):
        history = BackupHistory(
            last_success_at=200, last_failure_at=100, consecutive_failures=5
        )
        self.assertTrue(
            plan_backup(now=3800, changed=True, local=history).local_checkpoint
        )

    def test_same_timestamp_failure_still_respects_retry_backoff(self):
        history = BackupHistory(
            last_success_at=100, last_failure_at=100, consecutive_failures=1
        )
        plan = plan_backup(now=101, changed=True, important_change=True, local=history)
        self.assertFalse(plan.local_checkpoint)
        self.assertEqual(plan.local_due_at, 160)

    def test_invalid_history_and_flags_fail(self):
        for history in (
            BackupHistory(last_success_at=200),
            BackupHistory(last_failure_at=-1),
            BackupHistory(consecutive_failures=1),
            BackupHistory(last_failure_at=100, consecutive_failures=True),
        ):
            with self.subTest(history=history):
                with self.assertRaises(SnapshotError):
                    plan_backup(now=100, changed=True, local=history)
        with self.assertRaises(SnapshotError):
            plan_backup(now=100, changed=1)


if __name__ == "__main__":
    unittest.main()
