"""Durable outbound admission tests; no WeChat, GUI, network, or VM access."""

import sqlite3
import unittest
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from pathlib import Path
from tempfile import TemporaryDirectory

from scoped_bot_core.delivery import (
    DeliveryLedger,
    IntentConflict,
    InvalidReceipt,
    InvalidTransition,
    make_action_id,
    text_sha256,
)

ACCOUNT_ID = "synthetic-account"
CHAT_ID = "synthetic-group@chatroom"
TEXT = "SYNTHETIC_BODY_MUST_NOT_BE_STORED_IN_THE_LEDGER"


class DeliveryLedgerTests(unittest.TestCase):
    def setUp(self):
        self.directory = TemporaryDirectory(prefix="delivery-ledger-")
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "ledger.sqlite3"
        self.now = [100.0]
        self.ledger = self.open_ledger()

    def open_ledger(self):
        return DeliveryLedger(self.path, clock=lambda: self.now[0])

    def prepare(self, request_id="request-1", **changes):
        arguments = {
            "request_id": request_id,
            "account_id": ACCOUNT_ID,
            "chat_id": CHAT_ID,
            "text": TEXT,
        }
        arguments.update(changes)
        return self.ledger.prepare(**arguments)

    def claimed(self, request_id="request-1", **changes):
        intent = self.prepare(request_id, **changes)
        claim = self.ledger.claim(intent.action_id, "synthetic-worker")
        self.assertIsNotNone(claim)
        return intent, claim

    def receipt(self, **changes):
        result = {
            "source": "db_readback",
            "account_id": ACCOUNT_ID,
            "chat_id": CHAT_ID,
            "message_id": "synthetic-account/group/message_0.db/100",
            "text_sha256": text_sha256(TEXT),
            "observed_at": 101.0,
        }
        result.update(changes)
        return result

    def test_defaults_are_dry_run_and_prepare_is_idempotent(self):
        first = self.prepare()
        self.assertEqual(first.mode, "dry_run")
        self.assertEqual(first.state, "prepared")
        self.assertEqual(self.prepare(), first)
        self.assertEqual(
            [entry["state"] for entry in self.ledger.history(first.action_id)],
            ["prepared"],
        )

    def test_logical_slot_cannot_change_account_destination_text_or_mode(self):
        original = self.prepare()
        for changes in (
            {"account_id": "synthetic-other-account"},
            {"chat_id": "synthetic-other-group@chatroom"},
            {"text": "A replacement answer"},
            {"mode": "live"},
            {"expires_at": 200.0},
        ):
            with self.subTest(changes=changes):
                with self.assertRaises(IntentConflict):
                    self.prepare(**changes)
                self.assertEqual(self.ledger.get(original.action_id), original)
        with closing(sqlite3.connect(self.path)) as db:
            self.assertEqual(
                db.execute("SELECT COUNT(*) FROM delivery_intents").fetchone()[0], 1
            )

    def test_action_id_depends_on_request_slot_and_not_destination(self):
        original = self.prepare()
        self.assertEqual(original.action_id, make_action_id("request-1", 0))
        next_slot = self.prepare(
            action_index=1, chat_id="synthetic-explicit-new-target"
        )
        another_request = self.prepare("request-2")
        self.assertEqual(
            len({original.action_id, next_slot.action_id, another_request.action_id}), 3
        )

    def test_only_one_connection_can_claim_the_same_prepared_action(self):
        intent = self.prepare()
        second_ledger = self.open_ledger()

        def claim(index):
            ledger = self.ledger if index % 2 else second_ledger
            return ledger.claim(intent.action_id, "worker-" + str(index))

        with ThreadPoolExecutor(max_workers=8) as pool:
            claims = list(pool.map(claim, range(24)))
        self.assertEqual(sum(item is not None for item in claims), 1)
        self.assertEqual(self.open_ledger().get(intent.action_id).state, "executing")
        self.assertEqual(
            [entry["state"] for entry in self.ledger.history(intent.action_id)],
            ["prepared", "executing"],
        )

    def test_claim_is_durable_before_external_dispatch_would_begin(self):
        intent, claim = self.claimed(mode="live")
        restarted = self.open_ledger()
        self.assertEqual(restarted.get(intent.action_id).state, "executing")
        self.assertIsNone(restarted.claim(intent.action_id, "new-worker"))
        recovered = restarted.recover_incomplete()
        self.assertEqual([item.action_id for item in recovered], [intent.action_id])
        self.assertEqual(restarted.get(intent.action_id).state, "unknown")
        self.assertIsNone(restarted.claim(intent.action_id, "new-worker"))
        self.assertEqual(restarted.recover_incomplete(), [])
        self.assertEqual(claim.action_id, intent.action_id)

    def test_recovery_preserves_prepared_and_terminal_actions(self):
        pending = self.prepare("pending")
        finished, claim = self.claimed("finished")
        self.ledger.record_dry_run(finished.action_id, claim.claim_token)
        abandoned, _ = self.claimed("abandoned", mode="live")
        self.open_ledger().recover_incomplete()
        self.assertEqual(self.ledger.get(pending.action_id).state, "prepared")
        self.assertEqual(self.ledger.get(finished.action_id).state, "dry_run")
        self.assertEqual(self.ledger.get(abandoned.action_id).state, "unknown")
        self.assertIsNotNone(self.ledger.claim(pending.action_id, "replacement-worker"))

    def test_outcomes_never_reopen_the_original_action(self):
        for outcome in ("dry_run", "not_sent", "unknown", "confirmed"):
            with self.subTest(outcome=outcome):
                mode = "dry_run" if outcome == "dry_run" else "live"
                intent, claim = self.claimed("request-" + outcome, mode=mode)
                if outcome == "confirmed":
                    self.ledger.confirm_delivery(
                        intent.action_id, claim.claim_token, self.receipt()
                    )
                else:
                    getattr(self.ledger, "record_" + outcome)(
                        intent.action_id, claim.claim_token
                    )
                reloaded = self.open_ledger()
                self.assertEqual(reloaded.get(intent.action_id).state, outcome)
                self.assertIsNone(
                    reloaded.claim(intent.action_id, "replacement-worker")
                )
                self.assertEqual(
                    self.prepare("request-" + outcome, mode=mode).state, outcome
                )

    def test_wrong_claim_token_cannot_write_an_outcome(self):
        intent, claim = self.claimed(mode="live")
        for token in ("not-the-claim-token", "λ", None):
            with self.subTest(token=token):
                with self.assertRaises(PermissionError):
                    self.ledger.record_unknown(intent.action_id, token)
        with self.assertRaises(PermissionError):
            self.ledger.confirm_delivery(
                intent.action_id, "not-the-claim-token", self.receipt()
            )
        self.assertEqual(self.ledger.get(intent.action_id).state, "executing")
        self.assertNotIn(claim.claim_token, repr(claim))

    def test_dry_run_cannot_be_confirmed_as_a_real_delivery(self):
        intent, claim = self.claimed()
        with self.assertRaises(InvalidTransition):
            self.ledger.confirm_delivery(
                intent.action_id, claim.claim_token, self.receipt()
            )
        self.assertEqual(self.ledger.get(intent.action_id).state, "executing")

    def test_live_action_cannot_receive_a_dry_run_outcome(self):
        intent, claim = self.claimed(mode="live")
        with self.assertRaises(InvalidTransition):
            self.ledger.record_dry_run(intent.action_id, claim.claim_token)
        self.assertEqual(self.ledger.get(intent.action_id).state, "executing")

    def test_gui_success_is_not_database_confirmation(self):
        intent, claim = self.claimed(mode="live")
        for receipt in ({"success": True}, self.receipt(source="gui_success")):
            with self.subTest(receipt=receipt):
                with self.assertRaises(InvalidReceipt):
                    self.ledger.confirm_delivery(
                        intent.action_id, claim.claim_token, receipt
                    )
        self.assertEqual(self.ledger.get(intent.action_id).state, "executing")

    def test_readback_must_match_scope_text_and_message_identity(self):
        intent, claim = self.claimed(mode="live")
        for changes in (
            {"account_id": "another-account"},
            {"chat_id": "another-group@chatroom"},
            {"text_sha256": text_sha256("another answer")},
            {"message_id": ""},
            {"message_id": True},
            {"observed_at": float("nan")},
            {"observed_at": -1},
            {"content": TEXT},
        ):
            with self.subTest(changes=changes):
                with self.assertRaises(InvalidReceipt):
                    self.ledger.confirm_delivery(
                        intent.action_id, claim.claim_token, self.receipt(**changes)
                    )
                self.assertEqual(self.ledger.get(intent.action_id).state, "executing")

    def test_unknown_result_can_only_be_resolved_by_original_matching_readback(self):
        intent, claim = self.claimed(mode="live")
        self.ledger.record_unknown(
            intent.action_id, claim.claim_token, "transport_timeout"
        )
        self.assertIsNone(
            self.open_ledger().claim(intent.action_id, "replacement-worker")
        )
        with self.assertRaises(InvalidTransition):
            self.ledger.record_not_sent(intent.action_id, claim.claim_token)
        with self.assertRaises(InvalidReceipt):
            self.ledger.confirm_delivery(
                intent.action_id, claim.claim_token, self.receipt(chat_id="wrong-group")
            )
        confirmed = self.open_ledger().confirm_delivery(
            intent.action_id, claim.claim_token, self.receipt()
        )
        self.assertEqual(confirmed.state, "confirmed")
        self.assertEqual(confirmed.receipt["message_id"], self.receipt()["message_id"])
        self.assertIsNone(self.ledger.claim(intent.action_id, "replacement-worker"))

    def test_crash_recovered_unknown_accepts_readback_without_rerunning_dispatch(self):
        intent, claim = self.claimed(mode="live")
        restarted = self.open_ledger()
        restarted.recover_incomplete()
        confirmed = restarted.confirm_delivery(
            intent.action_id, claim.claim_token, self.receipt()
        )
        self.assertEqual(confirmed.state, "confirmed")
        self.assertEqual(
            [entry["state"] for entry in restarted.history(intent.action_id)],
            ["prepared", "executing", "unknown", "confirmed"],
        )

    def test_local_reconciliation_does_not_need_a_lost_process_claim_token(self):
        intent, claim = self.claimed(mode="live")
        del claim
        restarted = self.open_ledger()
        restarted.recover_incomplete()
        confirmed = restarted.reconcile_unknown(intent.action_id, self.receipt())
        self.assertEqual(confirmed.state, "confirmed")
        self.assertEqual(confirmed.reason_code, "local_reconciled")
        self.assertFalse(hasattr(confirmed, "claim_token"))
        self.assertIsNone(restarted.claim(intent.action_id, "new-worker"))
        self.assertEqual(restarted.get(intent.action_id).receipt["chat_id"], CHAT_ID)

    def test_local_reconciliation_rejects_prepared_executing_and_terminal_states(self):
        for state in ("prepared", "executing", "not_sent", "confirmed", "dry_run"):
            with self.subTest(state=state):
                mode = "dry_run" if state == "dry_run" else "live"
                intent = self.prepare("reconcile-" + state, mode=mode)
                if state != "prepared":
                    claim = self.ledger.claim(intent.action_id, "worker")
                    if state == "confirmed":
                        self.ledger.confirm_delivery(
                            intent.action_id, claim.claim_token, self.receipt()
                        )
                    elif state in {"not_sent", "dry_run"}:
                        getattr(self.ledger, "record_" + state)(
                            intent.action_id, claim.claim_token
                        )
                with self.assertRaises(InvalidTransition):
                    self.ledger.reconcile_unknown(intent.action_id, self.receipt())
                self.assertEqual(self.ledger.get(intent.action_id).state, state)

    def test_local_reconciliation_requires_matching_database_evidence(self):
        intent, claim = self.claimed(mode="live")
        self.ledger.record_unknown(intent.action_id, claim.claim_token)
        for changes in (
            {"account_id": "other-account"},
            {"chat_id": "other-group@chatroom"},
            {"text_sha256": text_sha256("other text")},
            {"source": "gui_success"},
            {"message_id": ""},
            {"content": TEXT},
        ):
            with self.subTest(changes=changes):
                with self.assertRaises(InvalidReceipt):
                    self.ledger.reconcile_unknown(
                        intent.action_id, self.receipt(**changes)
                    )
                self.assertEqual(self.ledger.get(intent.action_id).state, "unknown")
                self.assertIsNone(self.ledger.claim(intent.action_id, "new-worker"))

    def test_unknown_dry_run_is_never_reconciled_as_real_delivery(self):
        intent, claim = self.claimed()
        self.ledger.record_unknown(intent.action_id, claim.claim_token)
        with self.assertRaises(InvalidTransition):
            self.ledger.reconcile_unknown(intent.action_id, self.receipt())
        self.assertEqual(self.ledger.get(intent.action_id).state, "unknown")

    def test_competing_local_reconciliation_records_only_one_confirmation(self):
        intent, claim = self.claimed(mode="live")
        self.ledger.record_unknown(intent.action_id, claim.claim_token)
        second_ledger = self.open_ledger()

        def reconcile(index):
            ledger = self.ledger if index % 2 else second_ledger
            try:
                ledger.reconcile_unknown(intent.action_id, self.receipt())
            except InvalidTransition:
                return False
            return True

        with ThreadPoolExecutor(max_workers=4) as pool:
            results = list(pool.map(reconcile, range(12)))
        self.assertEqual(sum(results), 1)
        self.assertEqual(
            [entry["state"] for entry in self.ledger.history(intent.action_id)],
            ["prepared", "executing", "unknown", "confirmed"],
        )

    def test_confirmed_receipt_is_idempotent_and_cannot_be_replaced(self):
        intent, claim = self.claimed(mode="live")
        first = self.ledger.confirm_delivery(
            intent.action_id, claim.claim_token, self.receipt()
        )
        repeated = self.ledger.confirm_delivery(
            intent.action_id, claim.claim_token, self.receipt(observed_at=999)
        )
        self.assertEqual(repeated, first)
        self.assertEqual(len(self.ledger.history(intent.action_id)), 3)
        with self.assertRaises(InvalidReceipt):
            self.ledger.confirm_delivery(
                intent.action_id,
                claim.claim_token,
                self.receipt(message_id="different-message"),
            )
        with self.assertRaises(InvalidTransition):
            self.ledger.record_unknown(intent.action_id, claim.claim_token)
        self.assertEqual(self.ledger.get(intent.action_id), first)

    def test_one_database_message_cannot_confirm_two_different_actions(self):
        first, first_claim = self.claimed("first-send", mode="live")
        second, second_claim = self.claimed("second-send", mode="live")
        self.ledger.confirm_delivery(
            first.action_id, first_claim.claim_token, self.receipt()
        )
        with self.assertRaises(InvalidReceipt):
            self.ledger.confirm_delivery(
                second.action_id, second_claim.claim_token, self.receipt()
            )
        self.assertEqual(self.ledger.get(first.action_id).state, "confirmed")
        self.assertEqual(self.ledger.get(second.action_id).state, "executing")
        self.assertEqual(len(self.ledger.history(second.action_id)), 2)

    def test_local_reconciliation_cannot_reuse_another_actions_receipt(self):
        first, first_claim = self.claimed("first-send", mode="live")
        second, second_claim = self.claimed("second-send", mode="live")
        self.ledger.confirm_delivery(
            first.action_id, first_claim.claim_token, self.receipt()
        )
        self.ledger.record_unknown(second.action_id, second_claim.claim_token)
        with self.assertRaises(InvalidReceipt):
            self.open_ledger().reconcile_unknown(second.action_id, self.receipt())
        self.assertEqual(self.ledger.get(second.action_id).state, "unknown")

    def test_receipt_message_identity_is_scoped_to_account_and_chat(self):
        for index, changes in enumerate(
            (
                {},
                {"account_id": "other-account"},
                {"chat_id": "other-group@chatroom"},
            )
        ):
            with self.subTest(changes=changes):
                intent, claim = self.claimed(
                    "scope-" + str(index), mode="live", **changes
                )
                result = self.ledger.confirm_delivery(
                    intent.action_id, claim.claim_token, self.receipt(**changes)
                )
                self.assertEqual(result.state, "confirmed")

    def test_receipt_observation_cannot_predate_its_intent(self):
        intent, claim = self.claimed(mode="live")
        with self.assertRaises(InvalidReceipt):
            self.ledger.confirm_delivery(
                intent.action_id, claim.claim_token, self.receipt(observed_at=1)
            )
        self.ledger.record_unknown(intent.action_id, claim.claim_token)
        with self.assertRaises(InvalidReceipt):
            self.ledger.reconcile_unknown(intent.action_id, self.receipt(observed_at=1))
        self.assertEqual(self.ledger.get(intent.action_id).state, "unknown")

    def test_reopening_backfills_receipt_ownership_for_earlier_prototype_schema(self):
        first, first_claim = self.claimed("first-send", mode="live")
        self.ledger.confirm_delivery(
            first.action_id, first_claim.claim_token, self.receipt()
        )
        with closing(sqlite3.connect(self.path)) as db, db:
            db.execute("DROP TABLE delivery_receipt_owners")
        restarted = self.open_ledger()
        second = restarted.prepare(
            request_id="second-send",
            account_id=ACCOUNT_ID,
            chat_id=CHAT_ID,
            text=TEXT,
            mode="live",
        )
        second_claim = restarted.claim(second.action_id, "replacement-worker")
        with self.assertRaises(InvalidReceipt):
            restarted.confirm_delivery(
                second.action_id, second_claim.claim_token, self.receipt()
            )

    def test_proven_non_delivery_is_terminal_and_does_not_authorize_retry(self):
        intent, claim = self.claimed(mode="live")
        first = self.ledger.record_not_sent(
            intent.action_id, claim.claim_token, "target_unavailable"
        )
        self.assertEqual(
            self.ledger.record_not_sent(intent.action_id, claim.claim_token), first
        )
        self.assertEqual(len(self.ledger.history(intent.action_id)), 3)
        with self.assertRaises(InvalidTransition):
            self.ledger.confirm_delivery(
                intent.action_id, claim.claim_token, self.receipt()
            )
        self.assertIsNone(self.ledger.claim(intent.action_id, "new-worker"))

    def test_expired_prepared_action_cannot_dispatch_even_after_clock_rollback(self):
        intent = self.prepare(expires_at=101.0)
        self.now[0] = 101.0
        self.assertIsNone(self.ledger.claim(intent.action_id, "worker"))
        self.assertEqual(self.ledger.get(intent.action_id).state, "expired")
        self.now[0] = 99.0
        self.assertIsNone(self.open_ledger().claim(intent.action_id, "another-worker"))
        self.assertEqual(self.prepare(expires_at=101.0).state, "expired")

    def test_invalid_deadlines_and_action_indexes_are_rejected(self):
        for expires_at in (True, -1, float("nan"), float("inf")):
            with self.subTest(expires_at=expires_at):
                with self.assertRaises(ValueError):
                    self.prepare(expires_at=expires_at)
        for action_index in (True, -1, 1.5, "0"):
            with self.subTest(action_index=action_index):
                with self.assertRaises(ValueError):
                    self.prepare(action_index=action_index)

    def test_freeform_exception_text_cannot_be_saved_as_a_reason_code(self):
        intent, claim = self.claimed(mode="live")
        secret_like_error = "arbitrary exception containing path or credential text"
        with self.assertRaises(ValueError) as rejected:
            self.ledger.record_unknown(
                intent.action_id, claim.claim_token, secret_like_error
            )
        self.assertNotIn(secret_like_error, str(rejected.exception))
        self.assertEqual(self.ledger.get(intent.action_id).state, "executing")

    def test_ledger_stores_hash_and_receipt_but_never_plaintext_or_claim_in_history(
        self,
    ):
        intent, claim = self.claimed(mode="live")
        self.ledger.confirm_delivery(
            intent.action_id, claim.claim_token, self.receipt()
        )
        with closing(sqlite3.connect(self.path)) as db:
            dump = "\n".join(db.iterdump())
        self.assertNotIn(TEXT, dump)
        self.assertEqual(intent.text_sha256, text_sha256(TEXT))
        history = repr(self.ledger.history(intent.action_id))
        self.assertNotIn(claim.claim_token, history)
        self.assertNotIn(TEXT, history)
        local_copy = self.ledger.get(intent.action_id)
        local_copy.receipt["chat_id"] = "changed-in-memory"
        self.assertEqual(self.ledger.get(intent.action_id).receipt["chat_id"], CHAT_ID)

    def test_ledger_can_share_a_database_without_modifying_existing_tables(self):
        with closing(sqlite3.connect(self.path)) as db, db:
            db.execute("CREATE TABLE preexisting(value TEXT)")
            db.execute("INSERT INTO preexisting VALUES('keep')")
        self.open_ledger()
        self.prepare()
        with closing(sqlite3.connect(self.path)) as db:
            self.assertEqual(
                db.execute("SELECT value FROM preexisting").fetchall(), [("keep",)]
            )

    def test_unknown_action_cannot_be_claimed_or_given_a_receipt(self):
        self.assertIsNone(self.ledger.get("missing-action"))
        with self.assertRaises(ValueError):
            self.ledger.claim("missing-action", "worker")
        with self.assertRaises(ValueError):
            self.ledger.confirm_delivery(
                "missing-action", "missing-token", self.receipt()
            )


if __name__ == "__main__":
    unittest.main()
