"""Synthetic persistence, scope isolation and reviewed persona growth cases."""

import unittest
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory

from scoped_bot_core.context_budget import MemoryScope
from scoped_bot_core.persona_memory import PersonaMemoryStore


class PersonaMemoryTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory(prefix="wechat-persona-synthetic-")
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "persona.sqlite3"
        self.now = 1000.0
        self.store = PersonaMemoryStore(self.path, clock=lambda: self.now)
        self.scope = MemoryScope(
            "synthetic-account", "group-1", "member-1", "persona-1"
        )
        self.store.create_persona(
            self.scope.account_id,
            self.scope.persona_id,
            owner_id="synthetic-owner",
            core_text="稳定价值与边界",
        )
        self.evidence("source-1")

    def evidence(self, source, scope=None, kind="self_report", observed_at=900):
        return self.store.register_evidence(
            scope or self.scope, source, kind=kind, observed_at=observed_at
        )

    def remember(self, **changes):
        values = dict(
            kind="preference",
            content="喜欢喝不加糖的茶",
            evidence_ids=("source-1",),
            confidence=0.9,
            importance=0.7,
        )
        values.update(changes)
        return self.store.remember(self.scope, **values)

    def growth(self, text="经历后更愿意先倾听"):
        return self.store.propose_growth(
            self.scope,
            proposed_core=text,
            rationale="有明确共同经历",
            evidence_ids=("source-1",),
        )

    def test_reopen_retains_evidence_memory_and_core(self):
        record = self.remember()
        reopened = PersonaMemoryStore(self.path, clock=lambda: self.now)
        self.assertEqual(reopened.retrieve(self.scope), (record,))
        self.assertEqual(
            reopened.core(self.scope.account_id, self.scope.persona_id)["core_text"],
            "稳定价值与边界",
        )
        self.assertEqual(record.observed_at, 900)
        self.assertEqual(record.created_at, 1000)

    def test_exact_active_lookup_is_scoped_bounded_and_rejects_substrings(self):
        target = "日常本人自述：我喜欢喝茶"
        first = self.remember(content=target)
        self.remember(content=target)
        self.remember(content=target)
        self.assertEqual(len(self.store.find_active(self.scope, (target,))), 2)
        self.assertEqual(self.store.find_active(self.scope, ("我喜欢喝茶",)), ())
        for field in ("account_id", "chat_id", "member_id", "persona_id"):
            other = replace(self.scope, **{field: "other-stable-id"})
            self.assertEqual(self.store.find_active(other, (target,)), ())
        self.store.revoke(self.scope, first.memory_id, evidence_ids=("source-1",))
        self.assertNotIn(first, self.store.find_active(self.scope, (target,), limit=64))
        for contents in ([], (), (target,) * 5):
            with self.assertRaises(ValueError):
                self.store.find_active(self.scope, contents)

    def test_exact_lookup_excludes_expired_and_observation_quota_includes_withdrawn(
        self,
    ):
        prefix = "日常本人自述："
        old = self.remember(content=prefix + "我喜欢喝茶", expires_at=1001)
        self.store.revoke(self.scope, old.memory_id, evidence_ids=("source-1",))
        self.remember(content=prefix + "我喜欢咖啡", expires_at=1001)
        self.remember(kind="fact", content=prefix + "我喜欢柠檬水")
        self.remember(content="本人自述：我喜欢白水")
        self.now = 1002
        self.assertEqual(
            self.store.find_active(self.scope, (prefix + "我喜欢咖啡",)), ()
        )
        self.assertEqual(
            self.store.recent_observation_count(self.scope, since=0, prefix=prefix), 2
        )
        self.assertEqual(
            self.store.recent_observation_count(
                self.scope, since=0, prefix=prefix, limit=1
            ),
            1,
        )
        self.assertEqual(
            self.store.recent_observation_count(self.scope, since=1001, prefix=prefix),
            0,
        )
        for field in ("account_id", "chat_id", "member_id", "persona_id"):
            other = replace(self.scope, **{field: "other-stable-id"})
            self.assertEqual(
                self.store.recent_observation_count(other, since=0, prefix=prefix), 0
            )

    def test_memory_retrieval_and_direct_lookup_enforce_all_four_ids(self):
        record = self.remember()
        for field in ("account_id", "chat_id", "member_id", "persona_id"):
            with self.subTest(field=field):
                other = replace(self.scope, **{field: "other-stable-id"})
                self.assertEqual(self.store.retrieve(other), ())
                with self.assertRaises(ValueError):
                    self.store.get(other, record.memory_id)
                with self.assertRaises(ValueError):
                    self.store.revoke(
                        other, record.memory_id, evidence_ids=("source-1",)
                    )
        self.assertEqual(self.store.retrieve(self.scope), (record,))

    def test_same_nickname_and_rename_do_not_merge_members(self):
        other = replace(self.scope, member_id="member-2")
        self.store.update_nickname(self.scope, "小明")
        self.store.update_nickname(other, "小明")
        record = self.remember()
        self.store.update_nickname(self.scope, "新昵称")
        self.assertEqual(self.store.retrieve(self.scope), (record,))
        self.assertEqual(self.store.retrieve(other), ())

    def test_evidence_id_is_idempotent_but_cannot_change_provenance(self):
        self.assertFalse(self.evidence("source-1"))
        with self.assertRaises(ValueError):
            self.evidence("source-1", kind="observed")

    def test_unknown_and_cross_scope_evidence_cannot_create_memory(self):
        other = replace(self.scope, chat_id="group-2")
        self.evidence("only-group-2", scope=other)
        for evidence in ((), ("missing",), ("only-group-2",)):
            with self.subTest(evidence=evidence):
                with self.assertRaises(ValueError):
                    self.remember(evidence_ids=evidence)
        self.assertEqual(self.store.retrieve(self.scope), ())

    def test_quote_joke_and_inference_never_become_factual_evidence(self):
        for kind in ("quote", "joke", "inference"):
            self.evidence(kind, kind=kind)
            with self.assertRaises(ValueError):
                self.remember(evidence_ids=(kind,))
        self.assertEqual(self.store.retrieve(self.scope), ())

    def test_evidence_future_time_and_nonfinite_values_are_rejected(self):
        for value in (1001, float("nan"), float("inf"), True, "900"):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    self.evidence("bad", observed_at=value)

    def test_correction_atomically_supersedes_old_content(self):
        old = self.remember()
        self.evidence("correction", kind="correction", observed_at=950)
        new = self.remember(
            content="最近改为少量加糖",
            evidence_ids=("correction",),
            supersedes_id=old.memory_id,
        )
        self.assertEqual(self.store.retrieve(self.scope), (new,))
        self.assertEqual(new.supersedes_id, old.memory_id)
        self.assertEqual(self.store.get(self.scope, old.memory_id).status, "superseded")
        with self.assertRaises(ValueError):
            self.remember(supersedes_id=old.memory_id)

    def test_failed_correction_does_not_retire_original(self):
        old = self.remember()
        with self.assertRaises(ValueError):
            self.remember(supersedes_id=old.memory_id, evidence_ids=("missing",))
        self.assertEqual(self.store.retrieve(self.scope), (old,))

    def test_revoke_removes_recall_but_preserves_explicit_audit_lookup(self):
        record = self.remember()
        self.store.revoke(self.scope, record.memory_id, evidence_ids=("source-1",))
        self.assertEqual(self.store.retrieve(self.scope), ())
        self.assertEqual(self.store.get(self.scope, record.memory_id).status, "revoked")
        with self.assertRaises(ValueError):
            self.store.revoke(self.scope, record.memory_id, evidence_ids=("source-1",))

    def test_expired_memory_is_excluded_without_erasing_history(self):
        record = self.remember(expires_at=1001)
        self.assertEqual(self.store.retrieve(self.scope), (record,))
        self.now = 1001
        self.assertEqual(self.store.retrieve(self.scope), ())
        self.assertEqual(self.store.get(self.scope, record.memory_id).status, "active")

    def test_retrieval_is_bounded_ranked_and_confidence_filtered(self):
        for i in range(12):
            self.remember(content=f"虚构偏好-{i}", importance=i / 12, confidence=0.9)
        low = self.remember(content="未充分确定", confidence=0.1, importance=1)
        records = self.store.retrieve(self.scope, limit=3, min_confidence=0.5)
        self.assertEqual(len(records), 3)
        self.assertEqual(records[0].content, "虚构偏好-11")
        self.assertNotIn(low, records)
        self.assertTrue(
            all(a.importance >= b.importance for a, b in zip(records, records[1:]))
        )

    def test_retrieval_and_content_parameters_reject_unbounded_requests(self):
        for value in (0, -1, 65, True, 2.5):
            with self.assertRaises(ValueError):
                self.store.retrieve(self.scope, limit=value)
        for changes in (
            dict(content="x" * 8193),
            dict(confidence=1.1),
            dict(importance=float("nan")),
            dict(kind="stereotype"),
            dict(evidence_ids=("source-1",) * 17),
            dict(expires_at=999),
        ):
            with self.subTest(changes=changes):
                with self.assertRaises(ValueError):
                    self.remember(**changes)

    def test_context_candidates_preserve_scope_evidence_score_and_expiry(self):
        record = self.remember(expires_at=1100)
        (candidate,) = self.store.candidates(self.scope)
        self.assertEqual(candidate.scope, self.scope)
        self.assertEqual(candidate.memory_id, record.memory_id)
        self.assertAlmostEqual(candidate.priority, 0.63)
        self.assertEqual(candidate.expires_at, 1100)

    def test_inactive_audit_records_cannot_be_converted_back_into_recall(self):
        old = self.remember()
        self.remember(supersedes_id=old.memory_id)
        revoked = self.remember(content="后来撤回的内容")
        self.store.revoke(self.scope, revoked.memory_id, evidence_ids=("source-1",))
        for memory_id in (old.memory_id, revoked.memory_id):
            with self.subTest(memory_id=memory_id):
                with self.assertRaises(ValueError):
                    self.store.get(self.scope, memory_id).to_candidate()

    def test_core_cannot_be_recreated_or_changed_by_growth_proposal(self):
        candidate = self.growth()
        self.assertTrue(candidate)
        self.assertEqual(
            self.store.core(self.scope.account_id, self.scope.persona_id)["version"], 1
        )
        with self.assertRaises(ValueError):
            self.store.create_persona(
                self.scope.account_id,
                self.scope.persona_id,
                owner_id="new-owner",
                core_text="覆盖核心",
            )

    def test_nonowner_and_wrong_scope_cannot_review_growth(self):
        candidate = self.growth()
        with self.assertRaises(ValueError):
            self.store.review_growth(
                self.scope,
                candidate,
                authenticated_actor_id="group-member",
                accept=True,
            )
        with self.assertRaises(ValueError):
            self.store.review_growth(
                replace(self.scope, chat_id="group-2"),
                candidate,
                authenticated_actor_id="synthetic-owner",
                accept=True,
            )
        self.assertEqual(
            self.store.core(self.scope.account_id, self.scope.persona_id)["version"], 1
        )

    def test_owner_review_creates_immutable_inheritable_core_version(self):
        candidate = self.growth()
        version = self.store.review_growth(
            self.scope, candidate, authenticated_actor_id="synthetic-owner", accept=True
        )
        self.assertEqual(version, 2)
        self.assertEqual(
            self.store.core(self.scope.account_id, self.scope.persona_id, version=1)[
                "core_text"
            ],
            "稳定价值与边界",
        )
        self.assertEqual(
            self.store.core(self.scope.account_id, self.scope.persona_id)[
                "candidate_id"
            ],
            candidate,
        )
        with self.assertRaises(ValueError):
            self.store.review_growth(
                self.scope,
                candidate,
                authenticated_actor_id="synthetic-owner",
                accept=True,
            )

    def test_rejected_growth_and_stale_proposal_do_not_silently_change_core(self):
        rejected = self.growth()
        self.assertEqual(
            self.store.review_growth(
                self.scope,
                rejected,
                authenticated_actor_id="synthetic-owner",
                accept=False,
            ),
            1,
        )
        first = self.growth("确认的一次变化")
        stale = self.growth("基于旧核心的另一变化")
        self.store.review_growth(
            self.scope, first, authenticated_actor_id="synthetic-owner", accept=True
        )
        with self.assertRaises(ValueError):
            self.store.review_growth(
                self.scope, stale, authenticated_actor_id="synthetic-owner", accept=True
            )
        self.assertEqual(
            self.store.core(self.scope.account_id, self.scope.persona_id)["core_text"],
            "确认的一次变化",
        )

    def test_recent_state_is_scoped_versioned_and_separate_from_core(self):
        self.assertIsNone(self.store.state(self.scope))
        self.assertEqual(
            self.store.set_state(
                self.scope, state_text="近期想了解茶具", evidence_ids=("source-1",)
            ),
            1,
        )
        with self.assertRaises(ValueError):
            self.store.set_state(
                self.scope, state_text="过期写入", evidence_ids=("source-1",)
            )
        self.assertEqual(
            self.store.set_state(
                self.scope,
                state_text="已经了解茶具",
                evidence_ids=("source-1",),
                expected_version=1,
            ),
            2,
        )
        self.assertEqual(self.store.state(self.scope)["state_text"], "已经了解茶具")
        self.assertIsNone(self.store.state(replace(self.scope, chat_id="group-2")))
        self.assertEqual(
            self.store.core(self.scope.account_id, self.scope.persona_id)["version"], 1
        )

    def test_two_controllers_cannot_apply_conflicting_corrections(self):
        original = self.remember()
        second = PersonaMemoryStore(self.path, clock=lambda: self.now)

        def correct(store):
            try:
                return store.remember(
                    self.scope,
                    kind="preference",
                    content="更正",
                    evidence_ids=("source-1",),
                    confidence=0.9,
                    importance=0.8,
                    supersedes_id=original.memory_id,
                )
            except ValueError:
                return None

        with ThreadPoolExecutor(max_workers=2) as pool:
            outcomes = list(pool.map(correct, (self.store, second)))
        self.assertEqual(sum(outcome is not None for outcome in outcomes), 1)
        self.assertEqual(len(self.store.retrieve(self.scope)), 1)

    def test_export_is_full_not_retrieval_page_and_preserves_inactive_history(self):
        old = self.remember()
        self.remember(supersedes_id=old.memory_id)
        revoked = self.remember(content="撤回的偏好")
        self.store.revoke(self.scope, revoked.memory_id, evidence_ids=("source-1",))
        for i in range(10):
            self.remember(content=f"虚构记录-{i}")
        candidate = self.growth()
        self.store.review_growth(
            self.scope, candidate, authenticated_actor_id="synthetic-owner", accept=True
        )
        self.store.set_state(
            self.scope, state_text="近期状态", evidence_ids=("source-1",)
        )
        exported = self.store.export_snapshot(
            self.scope.account_id, self.scope.persona_id
        )
        self.assertEqual(len(exported["memories"]["records"]), 13)
        self.assertEqual(
            {row["status"] for row in exported["memories"]["records"]},
            {"active", "revoked", "superseded"},
        )
        self.assertEqual(len(exported["persona"]["core_versions"]), 2)
        self.assertEqual(len(exported["state"]), 1)
        self.assertEqual(exported["growth"][0]["status"], "accepted")
        self.assertEqual(exported["memories"]["audit"][-1]["action"], "created")
        self.assertNotIn("owner_id", exported["persona"])

    def test_export_does_not_cross_account_or_persona_and_fails_instead_of_truncating(
        self,
    ):
        self.remember()
        other = replace(self.scope, account_id="other-account")
        self.store.create_persona(
            other.account_id,
            other.persona_id,
            owner_id="another-owner",
            core_text="另一个核心",
        )
        self.evidence("other-source", scope=other)
        exported = self.store.export_snapshot(
            self.scope.account_id, self.scope.persona_id
        )
        self.assertEqual(len(exported["memories"]["evidence"]), 1)
        with self.assertRaises(ValueError):
            self.store.export_snapshot(
                self.scope.account_id, self.scope.persona_id, max_rows=1
            )
        with self.assertRaises(ValueError):
            self.store.export_snapshot("unknown-account", self.scope.persona_id)


if __name__ == "__main__":
    unittest.main()
