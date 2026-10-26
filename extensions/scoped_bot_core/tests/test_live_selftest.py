"""Adversarial one-shot FileHelper contracts; no GUI, client, VM or network."""

import importlib
import json
import sqlite3
import unittest
from contextlib import closing
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory

from scoped_bot_core.delivery import DeliveryLedger, make_action_id, text_sha256
from scoped_bot_core.live_selftest import (
    FILEHELPER,
    FileHelperSelftest,
    ReadbackBatch,
    ReadbackRow,
    SelftestProof,
    SelftestRejected,
    ShardBoundary,
    SqliteProofStore,
    TargetBinding,
    Watermark,
    matching_receipt,
)

ACCOUNT = "wxid_synthetic_account"
MARKER = "ASTRBOT_SELFTEST_0123456789abcdef"
BOUNDARIES = (
    ShardBoundary("message__message_0.db", 10, 500),
    ShardBoundary("message__message_1.db", 0, 0),
)


class FakeLease:
    def __init__(self):
        self.owned = False
        self.fail = False
        self.entries = 0

    def __enter__(self):
        if self.owned:
            raise RuntimeError("exclusive_lease_unavailable")
        self.owned = True
        self.entries += 1
        return self

    def __exit__(self, *_args):
        self.owned = False

    def assert_owned(self):
        if not self.owned or self.fail:
            raise RuntimeError("exclusive_lease_lost")


class FakeGui:
    def __init__(self, ledger):
        self.ledger = ledger
        self.binding = TargetBinding(
            ACCOUNT,
            FILEHELPER,
            "builtin_filehelper",
            "proven_builtin_identity",
            "process1_window1",
            "input1",
            True,
            True,
            True,
        )
        self.draft = ""
        self.staged = None
        self.before_enter = None
        self.after_stage = None
        self.enter_error = None
        self.enter_count = 0
        self.prepare_count = 0
        self.inspections = 0
        self.on_inspect = None

    def prepare_filehelper(self):
        self.prepare_count += 1
        return self.binding

    def inspect_target(self):
        self.inspections += 1
        if self.on_inspect:
            self.on_inspect(self.inspections)
        return self.binding

    def read_draft(self):
        return self.draft

    def stage_marker(self, marker):
        self.staged = marker
        self.draft = marker
        if self.after_stage:
            self.after_stage()

    def press_enter_once(self):
        self.enter_count += 1
        if self.before_enter:
            self.before_enter()
        if self.enter_error:
            raise self.enter_error


class FakeDb:
    def __init__(self):
        self.baseline = Watermark(ACCOUNT, FILEHELPER, BOUNDARIES, True)
        self.batches = []
        self.reads = 0
        self.row = ReadbackRow(
            ACCOUNT, FILEHELPER, BOUNDARIES[0].shard, 11, ACCOUNT, 1, MARKER
        )

    def watermark(self):
        return self.baseline

    def batch(self, rows=None, **changes):
        batch = ReadbackBatch(
            ACCOUNT,
            FILEHELPER,
            tuple(value.shard for value in BOUNDARIES),
            True,
            tuple(rows if rows is not None else [self.row]),
        )
        return replace(batch, **changes)

    def read_new_rows(self, watermark):
        assert watermark == self.baseline
        self.reads += 1
        result = self.batches.pop(0) if self.batches else self.batch()
        if isinstance(result, BaseException):
            raise result
        return result


class FileHelperSelftestTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory(prefix="synthetic-live-selftest-")
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name)
        self.ledger = DeliveryLedger(self.path / "delivery.sqlite3", clock=lambda: 100)
        self.proofs = SqliteProofStore(self.path / "proof.sqlite3")
        self.lease = FakeLease()
        self.db = FakeDb()
        self.gui = FakeGui(self.ledger)
        self.controller = self.new_controller()

    def new_controller(self, **changes):
        values = dict(
            ledger=self.ledger,
            proofs=self.proofs,
            gui=self.gui,
            database=self.db,
            lease_factory=lambda: self.lease,
            account_id=ACCOUNT,
            owner_id="synthetic_owner",
            max_readbacks=2,
            poll_interval=0,
            sleep=lambda _: None,
        )
        values.update(changes)
        return FileHelperSelftest(**values)

    def run_selftest(self, request_id="request-1", marker=MARKER):
        return self.controller.run(request_id=request_id, marker=marker)

    def proof(self):
        return SelftestProof(
            make_action_id("request-1"),
            ACCOUNT,
            text_sha256(MARKER),
            self.db.baseline,
            self.gui.binding.key(),
        )

    def test_valid_new_own_text_is_confirmed_after_durable_unknown(self):
        def assert_armed():
            self.assertTrue(self.lease.owned)
            intent = self.ledger.get(make_action_id("request-1"))
            self.assertEqual(intent.state, "unknown")
            self.assertEqual(intent.reason_code, "dispatch_armed")
            self.assertEqual(self.proofs.load(intent.action_id), self.proof())

        self.gui.before_enter = assert_armed
        result = self.run_selftest()
        self.assertEqual(result.state, "confirmed")
        self.assertEqual(
            json.loads(result.receipt["message_id"]), [BOUNDARIES[0].shard, 11]
        )
        self.assertEqual(self.gui.enter_count, 1)
        self.assertEqual(
            [item["state"] for item in self.ledger.history(result.action_id)],
            ["prepared", "executing", "unknown", "confirmed"],
        )

    def test_repeated_run_after_confirmed_does_not_even_open_gui(self):
        first = self.run_selftest()
        second = self.run_selftest()
        self.assertEqual(first, second)
        self.assertEqual(self.gui.enter_count, 1)
        self.assertEqual(self.gui.prepare_count, 1)

    def test_title_only_ambiguous_wrong_peer_and_changed_window_are_not_sent(self):
        cases = [
            dict(evidence_kind="display_title"),
            dict(chat_id="wxid_same_display_name"),
            dict(account_id="wxid_other_account"),
            dict(binding_id=""),
            dict(unlocked=False),
            dict(foreground=False),
        ]
        for index, changes in enumerate(cases):
            with self.subTest(changes=changes):
                self.gui.binding = replace(
                    TargetBinding(
                        ACCOUNT,
                        FILEHELPER,
                        "builtin_filehelper",
                        "builtin",
                        "window",
                        "input",
                        True,
                        True,
                        True,
                    ),
                    **changes,
                )
                self.assertEqual(
                    self.run_selftest("preflight-" + str(index)).state, "not_sent"
                )
        self.assertEqual(self.gui.enter_count, 0)
        self.assertIsNone(self.gui.staged)

    def test_existing_or_unreadable_draft_is_not_modified(self):
        for index, draft in enumerate(("existing user draft", " ", None)):
            with self.subTest(draft=draft):
                self.gui.draft = draft
                self.assertEqual(
                    self.run_selftest("draft-" + str(index)).state, "not_sent"
                )
                self.assertEqual(self.gui.draft, draft)
        self.assertIsNone(self.gui.staged)
        self.assertEqual(self.gui.enter_count, 0)

    def test_one_wrong_character_or_unreadable_staged_value_never_enters(self):
        for index, value in enumerate((MARKER[:-1] + "e", None)):
            with self.subTest(value=value):
                self.gui.draft = ""
                self.gui.after_stage = lambda value=value: setattr(
                    self.gui, "draft", value
                )
                result = self.run_selftest("stage-" + str(index))
                self.assertEqual(result.state, "not_sent")
        self.assertEqual(self.gui.enter_count, 0)

    def test_target_window_input_or_focus_changed_after_staging_never_enters(self):
        original = self.gui.binding
        for index, changes in enumerate(
            (
                dict(window_id="window2"),
                dict(input_id="input2"),
                dict(chat_id="wrong"),
                dict(focused=False),
            )
        ):
            with self.subTest(changes=changes):
                self.gui.binding, self.gui.draft = original, ""
                self.gui.after_stage = lambda changes=changes: setattr(
                    self.gui, "binding", replace(original, **changes)
                )
                self.assertEqual(
                    self.run_selftest("changed-" + str(index)).state, "not_sent"
                )
        self.assertEqual(self.gui.enter_count, 0)

    def test_final_identity_is_rechecked_after_draft_read(self):
        def move_after_read(count):
            if count == 3:
                self.gui.binding = replace(self.gui.binding, chat_id="wrong")

        self.gui.on_inspect = move_after_read
        self.assertEqual(self.run_selftest().state, "not_sent")
        self.assertEqual(self.gui.enter_count, 0)

    def test_identity_change_after_arming_is_unknown_without_enter(self):
        def move_after_arming(count):
            if count == 4:
                self.gui.binding = replace(self.gui.binding, chat_id="wrong")

        self.gui.on_inspect = move_after_arming
        self.db.batches = [self.db.batch(rows=[]), self.db.batch(rows=[])]
        self.assertEqual(self.run_selftest().state, "unknown")
        self.assertEqual(self.gui.enter_count, 0)
        self.assertEqual(self.run_selftest().state, "unknown")
        self.assertEqual(self.gui.enter_count, 0)

    def test_target_changed_during_watermark_does_not_even_stage_a_draft(self):
        original_watermark = self.db.watermark

        def slow_watermark():
            baseline = original_watermark()
            self.gui.binding = replace(self.gui.binding, chat_id="wrong")
            return baseline

        self.db.watermark = slow_watermark
        self.assertEqual(self.run_selftest().state, "not_sent")
        self.assertIsNone(self.gui.staged)
        self.assertEqual(self.gui.draft, "")
        self.assertEqual(self.gui.enter_count, 0)
        self.assertEqual(self.run_selftest().state, "not_sent")
        self.assertEqual(self.gui.enter_count, 0)

    def test_an_independent_controller_cannot_reenter_an_armed_request(self):
        second_gui = FakeGui(self.ledger)
        second = self.new_controller(gui=second_gui, owner_id="second_owner")

        def concurrent_attempt():
            result = second.run(request_id="request-1", marker=MARKER)
            self.assertEqual(result.state, "unknown")

        self.gui.before_enter = concurrent_attempt
        self.assertEqual(self.run_selftest().state, "confirmed")
        self.assertEqual(self.gui.enter_count, 1)
        self.assertEqual(second_gui.prepare_count, 0)
        self.assertEqual(second_gui.enter_count, 0)

    def test_exclusive_lease_unavailable_does_not_claim_or_touch_gui(self):
        self.lease.owned = True
        with self.assertRaises(RuntimeError):
            self.run_selftest()
        self.assertEqual(self.ledger.get(make_action_id("request-1")).state, "prepared")
        self.assertEqual(self.gui.prepare_count, 0)
        self.assertEqual(self.gui.enter_count, 0)

    def test_lease_lost_before_dispatch_never_enters(self):
        self.gui.after_stage = lambda: setattr(self.lease, "fail", True)
        self.assertEqual(self.run_selftest().state, "not_sent")
        self.assertEqual(self.gui.enter_count, 0)

    def test_incomplete_duplicate_or_wrong_scope_watermarks_never_stage(self):
        original = self.db.baseline
        for index, baseline in enumerate(
            (
                replace(original, complete=False),
                replace(original, account_id="other"),
                replace(original, boundaries=()),
                replace(original, boundaries=(BOUNDARIES[0], BOUNDARIES[0])),
                replace(
                    original, boundaries=(replace(BOUNDARIES[0], max_local_id=True),)
                ),
            )
        ):
            with self.subTest(index=index):
                self.db.baseline = baseline
                self.assertEqual(
                    self.run_selftest("bad-watermark-" + str(index)).state, "not_sent"
                )
        self.assertIsNone(self.gui.staged)
        self.assertEqual(self.gui.enter_count, 0)

    def test_enter_exception_can_only_reconcile_and_never_retry(self):
        self.gui.enter_error = RuntimeError("key might already have been delivered")
        result = self.run_selftest()
        self.assertEqual(result.state, "confirmed")
        self.assertEqual(self.gui.enter_count, 1)
        self.run_selftest()
        self.assertEqual(self.gui.enter_count, 1)

    def test_enter_interrupt_keeps_unknown_and_repeated_run_never_sends(self):
        self.gui.enter_error = KeyboardInterrupt()
        with self.assertRaises(KeyboardInterrupt):
            self.run_selftest()
        self.assertEqual(self.ledger.get(make_action_id("request-1")).state, "unknown")
        self.gui.enter_error = None
        self.assertEqual(self.run_selftest().state, "unknown")
        self.assertEqual(self.gui.enter_count, 1)

    def test_timeout_keeps_unknown_without_resend_and_reconcile_uses_no_gui(self):
        self.db.batches = [self.db.batch(rows=[]), self.db.batch(rows=[])]
        self.assertEqual(self.run_selftest().state, "unknown")
        self.assertEqual(self.run_selftest().state, "unknown")
        self.assertEqual(self.gui.enter_count, 1)
        self.assertEqual(self.db.reads, 2)
        self.controller.gui = None
        self.proofs = SqliteProofStore(self.path / "proof.sqlite3")
        self.controller.proofs = self.proofs
        result = self.controller.reconcile(request_id="request-1", marker=MARKER)
        self.assertEqual(result.state, "confirmed")
        self.assertEqual(self.gui.enter_count, 1)

    def test_readback_exception_partial_or_changed_shards_remain_unknown(self):
        for index, batch in enumerate(
            (
                RuntimeError("bounded read failed"),
                self.db.batch(complete=False),
                self.db.batch(shards=(BOUNDARIES[0].shard,)),
                self.db.batch(account_id="other"),
            )
        ):
            with self.subTest(index=index):
                self.gui.draft = ""
                self.db.batches = [batch]
                self.assertEqual(
                    self.run_selftest("read-failure-" + str(index)).state, "unknown"
                )
        self.assertEqual(self.gui.enter_count, 4)

    def test_delay_polls_read_only_after_one_enter(self):
        self.db.batches = [self.db.batch(rows=[]), self.db.batch()]
        self.assertEqual(self.run_selftest().state, "confirmed")
        self.assertEqual(self.db.reads, 2)
        self.assertEqual(self.gui.enter_count, 1)

    def test_wrong_sender_type_text_old_row_and_other_scope_never_confirm(self):
        changes = [
            dict(sender_username="wxid_other"),
            dict(sender_username=""),
            dict(sender_username="2"),
            dict(local_type=49),
            dict(local_type=True),
            dict(content=MARKER + "extra"),
            dict(local_id=10),
            dict(shard="unobserved_shard"),
            dict(chat_id="other"),
            dict(account_id="other"),
        ]
        for values in changes:
            with self.subTest(values=values):
                batch = self.db.batch(rows=[replace(self.db.row, **values)])
                try:
                    receipt = matching_receipt(self.proof(), MARKER, batch)
                except SelftestRejected:
                    receipt = None
                self.assertIsNone(receipt)

    def test_same_seq_and_repeated_local_id_in_another_shard_are_distinct(self):
        # A zero-baseline source shard may hold id=1 despite another shard's
        # max id=10. No global local-id or strict sort-seq comparison is used.
        row = replace(self.db.row, shard=BOUNDARIES[1].shard, local_id=1)
        receipt = matching_receipt(self.proof(), MARKER, self.db.batch(rows=[row]))
        self.assertEqual(json.loads(receipt["message_id"]), [BOUNDARIES[1].shard, 1])

    def test_multiple_matching_or_duplicate_rows_cannot_be_arbitrarily_selected(self):
        for rows in (
            [self.db.row, replace(self.db.row, local_id=12)],
            [self.db.row, self.db.row],
        ):
            with self.subTest(rows=len(rows)):
                with self.assertRaises(SelftestRejected):
                    matching_receipt(self.proof(), MARKER, self.db.batch(rows=rows))

    def test_metadata_store_is_immutable_and_contains_no_marker_or_draft(self):
        proof = self.proof()
        self.proofs.save(proof)
        self.proofs.save(proof)
        with self.assertRaises(SelftestRejected):
            self.proofs.save(
                replace(proof, watermark=replace(proof.watermark, complete=False))
            )
        self.assertEqual(
            SqliteProofStore(self.path / "proof.sqlite3").load(proof.action_id), proof
        )
        with closing(sqlite3.connect(self.path / "proof.sqlite3")) as db:
            stored = db.execute(
                "SELECT proof_json FROM live_selftest_proofs"
            ).fetchone()[0]
        self.assertNotIn(MARKER, stored)
        self.assertNotIn("content", stored)

    def test_proof_save_failure_does_not_stage_or_enter(self):
        class BrokenProofs:
            def save(self, proof):
                raise OSError("disk unavailable")

        self.controller.proofs = BrokenProofs()
        self.assertEqual(self.run_selftest().state, "not_sent")
        self.assertIsNone(self.gui.staged)
        self.assertEqual(self.gui.enter_count, 0)

    def test_invalid_marker_never_touches_gui_lease_or_ledger(self):
        for value in (
            "",
            "hello",
            "ASTRBOT_SELFTEST_0123456789abcde\n",
            "ASTRBOT_SELFTEST_中文0123456789abcdef",
            MARKER + " ",
            None,
        ):
            with self.subTest(value=value):
                with self.assertRaises(SelftestRejected):
                    self.run_selftest(marker=value)
        self.assertEqual(self.lease.entries, 0)
        self.assertEqual(self.gui.prepare_count, 0)
        self.assertIsNone(self.ledger.get(make_action_id("request-1")))

    def test_reconcile_wrong_marker_or_missing_evidence_never_calls_gui(self):
        self.db.batches = [self.db.batch(rows=[]), self.db.batch(rows=[])]
        self.run_selftest()
        self.controller.gui = None
        with self.assertRaises(SelftestRejected):
            self.controller.reconcile(
                request_id="request-1", marker="ASTRBOT_SELFTEST_ffffffffffffffff"
            )
        self.controller.proofs = SqliteProofStore(self.path / "different-proof.sqlite3")
        with self.assertRaises(SelftestRejected):
            self.controller.reconcile(request_id="request-1", marker=MARKER)
        self.assertEqual(self.gui.enter_count, 1)

    def test_probe_schema_covers_all_shards_and_requires_complete_true(self):
        probe = dict(
            schema_version=1,
            account_id=ACCOUNT,
            approved_chat_ids=[FILEHELPER],
            complete=True,
            watermarks=[
                dict(
                    shard=value.shard,
                    max_local_id=value.max_local_id,
                    max_sort_seq=value.max_sort_seq,
                )
                for value in BOUNDARIES
            ],
        )
        self.assertEqual(Watermark.from_probe(probe), self.db.baseline)
        for changes in (
            dict(complete=False),
            dict(complete=None),
            dict(approved_chat_ids=["same title"]),
            dict(schema_version=True),
            dict(watermarks=[]),
        ):
            with self.subTest(changes=changes):
                with self.assertRaises(SelftestRejected):
                    Watermark.from_probe(dict(probe, **changes))

    def test_import_is_pure_and_does_not_expose_live_implementation(self):
        module = importlib.import_module("scoped_bot_core.live_selftest")
        self.assertFalse(hasattr(module, "WeChat"))
        self.assertFalse(hasattr(module, "WeChatUIA"))


if __name__ == "__main__":
    unittest.main()
