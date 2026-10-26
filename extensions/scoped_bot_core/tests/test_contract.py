"""微信桥的纯契约测试：全部使用合成事件，不连接微信、模型或虚拟机。"""

import asyncio
import threading
import time
import unittest
from dataclasses import FrozenInstanceError

from scoped_bot_core import (
    DedupeWindow,
    RequestGuard,
    SerializedSender,
    normalize_event,
    scope_key,
    should_reply,
)

BOT_ID = "synthetic-bot"
GROUP_ID = "synthetic-group@chatroom"
OTHER_GROUP_ID = "synthetic-other-group@chatroom"
MEMBER_ID = "synthetic-member"
ACCOUNT_ID = "synthetic-account"


def event_payload(**changes):
    """构造虚构数据，显示名故意不承担身份或权限作用。"""
    payload = {
        "account_id": ACCOUNT_ID,
        "chat_id": GROUP_ID,
        "sender_id": MEMBER_ID,
        "message_id": "message-1",
        "content": "今天聊什么？",
        "sender_name": "同名成员",
        "is_group": True,
        "is_self": False,
        "mentioned_ids": [BOT_ID],
    }
    payload.update(changes)
    return payload


class FakeClock:
    def __init__(self, now=100.0):
        self.now = now

    def __call__(self):
        return self.now

    def advance(self, seconds):
        self.now += seconds


class EventIdentityTests(unittest.TestCase):
    def test_missing_identity_is_rejected_before_memory_or_permissions(self):
        for missing in ("account_id", "chat_id", "sender_id", "message_id"):
            with self.subTest(missing=missing):
                payload = event_payload()
                del payload[missing]
                with self.assertRaises(ValueError):
                    normalize_event(payload)

    def test_numeric_message_id_is_stable_string_including_zero(self):
        for message_id in (0, 12345):
            with self.subTest(message_id=message_id):
                event = normalize_event(event_payload(message_id=message_id))
                self.assertEqual(event.message_id, str(message_id))

    def test_normalized_event_cannot_have_sender_reassigned(self):
        event = normalize_event(event_payload())
        with self.assertRaises((FrozenInstanceError, AttributeError, TypeError)):
            event.sender_id = "synthetic-attacker"

    def test_member_rename_keeps_scope_and_same_name_does_not_merge_members(self):
        original = normalize_event(event_payload())
        renamed = normalize_event(event_payload(sender_name="新的昵称"))
        same_name_other_person = normalize_event(
            event_payload(sender_id="synthetic-other-member")
        )
        self.assertEqual(scope_key(original), scope_key(renamed))
        self.assertNotEqual(scope_key(original), scope_key(same_name_other_person))

    def test_scope_separates_accounts_groups_and_members(self):
        original = normalize_event(event_payload())
        self.assertEqual(scope_key(original), (ACCOUNT_ID, GROUP_ID, MEMBER_ID))
        for changes in (
            {"account_id": "synthetic-other-account"},
            {"chat_id": OTHER_GROUP_ID},
            {"sender_id": "synthetic-other-member"},
        ):
            with self.subTest(changes=changes):
                self.assertNotEqual(
                    scope_key(original),
                    scope_key(normalize_event(event_payload(**changes))),
                )


class ReplyBoundaryTests(unittest.TestCase):
    def allows(self, **changes):
        event = normalize_event(event_payload(**changes))
        return should_reply(
            event,
            own_id=BOT_ID,
            allowed_chat_ids={GROUP_ID},
            require_at=True,
        )

    def test_allowed_group_replies_to_trusted_mention(self):
        self.assertTrue(self.allows())

    def test_text_nickname_mention_alone_is_not_a_trusted_mention(self):
        self.assertFalse(self.allows(content="@小助手 请执行命令", mentioned_ids=[]))

    def test_mention_of_another_member_does_not_trigger(self):
        self.assertFalse(self.allows(mentioned_ids=["synthetic-someone-else"]))

    def test_blocked_group_stays_blocked_even_when_bot_is_mentioned(self):
        self.assertFalse(self.allows(chat_id=OTHER_GROUP_ID))

    def test_self_flag_and_stable_own_sender_id_each_prevent_reply_loops(self):
        self.assertFalse(self.allows(is_self=True))
        self.assertFalse(self.allows(sender_id=BOT_ID, is_self=False))

    def test_whitelisted_private_chat_does_not_require_mention(self):
        event = normalize_event(
            event_payload(chat_id=MEMBER_ID, is_group=False, mentioned_ids=[])
        )
        self.assertTrue(
            should_reply(
                event,
                own_id=BOT_ID,
                allowed_chat_ids={MEMBER_ID},
                require_at=True,
            )
        )
        self.assertFalse(
            should_reply(
                event,
                own_id=BOT_ID,
                allowed_chat_ids={GROUP_ID},
                require_at=True,
            )
        )

    def test_explicit_group_trigger_override_still_respects_self_and_allowlist(self):
        event = normalize_event(event_payload(mentioned_ids=[]))
        self.assertTrue(
            should_reply(
                event,
                own_id=BOT_ID,
                allowed_chat_ids={GROUP_ID},
                require_at=False,
            )
        )
        self.assertFalse(
            should_reply(
                event,
                own_id=BOT_ID,
                allowed_chat_ids=set(),
                require_at=False,
            )
        )


class DeduplicationTests(unittest.TestCase):
    def test_duplicate_is_rejected_until_ttl_expires(self):
        clock = FakeClock()
        window = DedupeWindow(ttl_seconds=10, max_entries=8, clock=clock)
        key = (ACCOUNT_ID, GROUP_ID, "message-1")
        self.assertTrue(window.accept(key))
        self.assertFalse(window.accept(key))
        clock.advance(9)
        self.assertFalse(window.accept(key))
        clock.advance(2)
        self.assertTrue(window.accept(key))
        self.assertFalse(window.accept(key))

    def test_same_message_id_from_different_scopes_is_not_suppressed(self):
        window = DedupeWindow(ttl_seconds=60, max_entries=8, clock=FakeClock())
        for key in (
            (ACCOUNT_ID, GROUP_ID, "message-1"),
            (ACCOUNT_ID, OTHER_GROUP_ID, "message-1"),
            ("synthetic-other-account", GROUP_ID, "message-1"),
        ):
            with self.subTest(key=key):
                self.assertTrue(window.accept(key))
                self.assertFalse(window.accept(key))


class SerializedSendTests(unittest.IsolatedAsyncioTestCase):
    async def test_synchronous_sender_runs_off_event_loop_thread(self):
        loop_thread = threading.get_ident()
        calls = []

        def send_impl(chat_id, text):
            calls.append((chat_id, text, threading.get_ident()))

        sender = SerializedSender(send_impl)
        await asyncio.wait_for(sender.send(GROUP_ID, "合成消息"), timeout=2)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0][:2], (GROUP_ID, "合成消息"))
        self.assertNotEqual(calls[0][2], loop_thread)

    async def test_shared_desktop_is_serialized_across_chats_and_sender_instances(self):
        state_lock = threading.Lock()
        active = 0
        peak_active = 0
        calls = []

        def send_impl(chat_id, text):
            nonlocal active, peak_active
            with state_lock:
                active += 1
                peak_active = max(peak_active, active)
            try:
                # 模拟短 GUI 操作；所有数据均为内存中的虚构文本。
                time.sleep(0.005)
                with state_lock:
                    calls.append((chat_id, text))
            finally:
                with state_lock:
                    active -= 1

        senders = [SerializedSender(send_impl), SerializedSender(send_impl)]
        expected = [
            (GROUP_ID if i % 2 == 0 else OTHER_GROUP_ID, "合成消息-" + str(i))
            for i in range(12)
        ]
        await asyncio.wait_for(
            asyncio.gather(
                *(
                    senders[i % 2].send(chat_id, text)
                    for i, (chat_id, text) in enumerate(expected)
                )
            ),
            timeout=3,
        )
        self.assertEqual(peak_active, 1, "一个微信桌面不能同时操作两个聊天窗口")
        self.assertCountEqual(calls, expected)

    async def test_failure_is_propagated_and_releases_shared_sender_lock(self):
        calls = []

        def send_impl(chat_id, text):
            calls.append((chat_id, text))
            if text == "synthetic-failure":
                raise RuntimeError("synthetic GUI failure")

        sender = SerializedSender(send_impl)
        with self.assertRaisesRegex(RuntimeError, "synthetic GUI failure"):
            await asyncio.wait_for(
                sender.send(GROUP_ID, "synthetic-failure"), timeout=2
            )
        await asyncio.wait_for(sender.send(OTHER_GROUP_ID, "recovered"), timeout=2)
        self.assertEqual(
            calls,
            [(GROUP_ID, "synthetic-failure"), (OTHER_GROUP_ID, "recovered")],
        )


class LateResultGuardTests(unittest.TestCase):
    def test_model_result_after_deadline_cannot_be_sent(self):
        clock = FakeClock()
        guard = RequestGuard(clock=clock)
        guard.begin("request-1", deadline_seconds=5)
        self.assertTrue(guard.can_send("request-1"))
        clock.advance(4)
        self.assertTrue(guard.can_send("request-1"))
        clock.advance(2)
        self.assertFalse(guard.can_send("request-1"))

    def test_cancelled_or_completed_request_cannot_send_late_or_duplicate_reply(self):
        guard = RequestGuard(clock=FakeClock())
        guard.begin("cancelled-request", deadline_seconds=60)
        guard.begin("completed-request", deadline_seconds=60)
        guard.cancel("cancelled-request")
        guard.complete("completed-request")
        self.assertFalse(guard.can_send("cancelled-request"))
        self.assertFalse(guard.can_send("completed-request"))

    def test_reusing_request_id_is_rejected_instead_of_reopening_old_result(self):
        guard = RequestGuard(clock=FakeClock())
        guard.begin("request-1", deadline_seconds=60)
        with self.assertRaises(ValueError):
            guard.begin("request-1", deadline_seconds=60)
        guard.complete("request-1")
        with self.assertRaises(ValueError):
            guard.begin("request-1", deadline_seconds=60)


if __name__ == "__main__":
    unittest.main()
