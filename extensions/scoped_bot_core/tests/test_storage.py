"""收件箱持久化契约：临时 SQLite 与虚构消息，无微信或网络依赖。"""

import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from tempfile import TemporaryDirectory

from scoped_bot_core import normalize_event
from scoped_bot_core.storage import InboxStore


def synthetic_payload(message_id="message-1", **changes):
    payload = {
        "account_id": "synthetic-account",
        "chat_id": "synthetic-group@chatroom",
        "sender_id": "synthetic-member",
        "message_id": message_id,
        "content": "仅供离线验证的虚构消息",
        "sender_name": "虚构成员",
        "is_group": True,
        "is_self": False,
        "mentioned_ids": ["synthetic-bot"],
    }
    payload.update(changes)
    return payload


class InboxStoreTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory(prefix="wechat-bridge-contract-")
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "inbox.sqlite3"
        self.store = self.open_store()

    def open_store(self):
        store = InboxStore(self.path)
        # 实现若提供显式关闭方法，就在删除临时目录之前释放句柄。
        close = getattr(store, "close", None)
        if callable(close):
            self.addCleanup(close)
        return store

    def test_duplicate_enqueue_preserves_first_payload_and_is_claimed_once(self):
        self.assertTrue(self.store.enqueue(synthetic_payload(content="原始合成文本")))
        self.assertFalse(
            self.store.enqueue(synthetic_payload(content="重复事件不得覆盖原文"))
        )
        claimed = self.store.claim()
        self.assertEqual(len(claimed), 1)
        self.assertEqual(claimed[0].content, "原始合成文本")
        self.assertEqual(self.store.claim(), [])
        self.assertFalse(self.store.enqueue(synthetic_payload()))
        self.store.finish(claimed[0])
        self.assertFalse(self.store.enqueue(synthetic_payload()))
        self.assertEqual(self.store.claim(), [])

    def test_same_message_id_does_not_collide_between_accounts_or_groups(self):
        payloads = [
            synthetic_payload(),
            synthetic_payload(chat_id="synthetic-other-group@chatroom"),
            synthetic_payload(account_id="synthetic-other-account"),
        ]
        for payload in payloads:
            self.assertTrue(self.store.enqueue(payload))
        claimed = self.store.claim()
        self.assertEqual(
            {(e.account_id, e.chat_id, e.message_id) for e in claimed},
            {(p["account_id"], p["chat_id"], p["message_id"]) for p in payloads},
        )

    def test_reopening_keeps_pending_but_does_not_replay_dispatched(self):
        self.assertTrue(self.store.enqueue(synthetic_payload("claimed-before-restart")))
        self.assertTrue(self.store.enqueue(synthetic_payload("pending-before-restart")))
        dispatched = self.store.claim(limit=1)
        self.assertEqual(len(dispatched), 1)
        dispatched_id = dispatched[0].message_id

        reopened = self.open_store()
        remaining = reopened.claim()
        self.assertEqual(len(remaining), 1)
        self.assertNotEqual(remaining[0].message_id, dispatched_id)
        self.assertEqual(reopened.claim(), [])
        self.assertEqual(self.store.claim(), [])

    def test_claim_limit_does_not_drop_or_repeat_remaining_events(self):
        expected = {"synthetic-message-" + str(i) for i in range(5)}
        for message_id in expected:
            self.assertTrue(self.store.enqueue(synthetic_payload(message_id)))
        first = self.store.claim(limit=2)
        second = self.store.claim(limit=20)
        self.assertEqual(len(first), 2)
        self.assertEqual(len(second), 3)
        self.assertEqual({e.message_id for e in first + second}, expected)
        self.assertEqual(self.store.claim(), [])

    def test_two_store_instances_claim_atomically_without_double_dispatch(self):
        expected = {"concurrent-message-" + str(i) for i in range(8)}
        for message_id in expected:
            self.store.enqueue(synthetic_payload(message_id))
        second_store = self.open_store()
        with ThreadPoolExecutor(max_workers=2) as executor:
            batches = list(
                executor.map(
                    lambda store: store.claim(limit=4), [self.store, second_store]
                )
            )
        claimed = [event for batch in batches for event in batch]
        self.assertEqual(len(claimed), 8)
        self.assertEqual({e.message_id for e in claimed}, expected)
        self.assertEqual(self.store.claim(), [])

    def test_finishing_unknown_event_is_rejected(self):
        unknown = normalize_event(synthetic_payload("never-enqueued"))
        with self.assertRaises(ValueError):
            self.store.finish(unknown)

    def test_invalid_payload_is_not_persisted(self):
        for missing in ("account_id", "chat_id", "sender_id", "message_id"):
            with self.subTest(missing=missing):
                payload = synthetic_payload("invalid-" + missing)
                del payload[missing]
                with self.assertRaises(ValueError):
                    self.store.enqueue(payload)
        self.assertEqual(self.store.claim(), [])
        self.assertTrue(self.store.enqueue(synthetic_payload("valid-after-errors")))
        self.assertEqual(len(self.store.claim()), 1)

    def test_numeric_message_id_is_normalized_before_deduplication(self):
        self.assertTrue(self.store.enqueue(synthetic_payload(0)))
        self.assertFalse(self.store.enqueue(synthetic_payload("0")))
        claimed = self.store.claim()
        self.assertEqual(len(claimed), 1)
        self.assertEqual(claimed[0].message_id, "0")

    def test_reply_request_id_is_unique_and_cannot_overwrite_text_or_destination(self):
        self.assertTrue(
            self.store.record_reply(
                "request-1",
                "synthetic-account",
                "synthetic-group@chatroom",
                "第一份虚构回复",
            )
        )
        self.assertFalse(
            self.store.record_reply(
                "request-1",
                "synthetic-other-account",
                "synthetic-other-group@chatroom",
                "覆盖尝试",
            )
        )
        replies = self.store.replies()
        self.assertEqual(len(replies), 1)
        expected = {
            "request_id": "request-1",
            "account_id": "synthetic-account",
            "chat_id": "synthetic-group@chatroom",
            "text": "第一份虚构回复",
            "status": "dry_run",
        }
        for field, value in expected.items():
            self.assertEqual(replies[0][field], value)

    def test_reopening_preserves_dry_run_replies_and_reply_deduplication(self):
        self.store.record_reply(
            "request-1", "synthetic-account", "synthetic-chat", "离线回复"
        )
        reopened = self.open_store()
        self.assertFalse(
            reopened.record_reply(
                "request-1", "synthetic-account", "synthetic-chat", "迟到重试"
            )
        )
        self.assertTrue(
            reopened.record_reply(
                "request-2", "synthetic-account", "synthetic-chat", "第二份回复"
            )
        )
        replies = reopened.replies()
        self.assertEqual({r["request_id"] for r in replies}, {"request-1", "request-2"})
        self.assertTrue(all(r["status"] == "dry_run" for r in replies))


if __name__ == "__main__":
    unittest.main()
