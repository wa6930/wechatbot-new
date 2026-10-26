"""Account-scoped messages and send guards, without importing a live client."""

import asyncio
import math
import threading
import time
from collections import OrderedDict
from collections.abc import Callable, Hashable, Mapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class Event:
    account_id: str
    chat_id: str
    sender_id: str
    message_id: str
    content: str
    sender_name: str = ""
    is_group: bool = False
    is_self: bool = False
    mentioned_ids: tuple[str, ...] = ()
    received_at: float = 0.0
    mention_source: str = "none"
    text_mention_matched: bool = False
    source: str = "synthetic"
    source_timestamp: float | None = None
    message_type: str = "text"


def normalize_event(payload: Mapping[str, Any]) -> Event:
    """Validate transport identities without inferring them from display names.

    Args:
        payload: A trusted transport envelope, not model-generated JSON.

    Returns:
        An immutable event.

    Raises:
        ValueError: Required identities or field types are invalid.
    """
    identities = {}
    for key in ("account_id", "chat_id", "sender_id", "message_id"):
        value = payload.get(key)
        if (
            key == "message_id"
            and isinstance(value, int)
            and not isinstance(value, bool)
        ):
            value = str(value)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"Invalid transport identity: {key}")
        identities[key] = value.strip()
    if not isinstance(payload.get("content"), str):
        raise ValueError("content must be a string")
    for key in ("is_group", "is_self", "text_mention_matched"):
        if key in payload and not isinstance(payload[key], bool):
            raise ValueError(f"{key} must be a boolean")
    mentions = payload.get("mentioned_ids", ())
    if not isinstance(mentions, (list, tuple)) or any(
        not isinstance(item, str) or not item.strip() for item in mentions
    ):
        raise ValueError("mentioned_ids must contain stable string identities")
    received_at = payload.get("received_at", time.time())
    if isinstance(received_at, bool) or not isinstance(received_at, (float, int)):
        raise ValueError("received_at must be a timestamp")
    if not math.isfinite(received_at) or received_at < 0:
        raise ValueError("received_at must be a finite timestamp")
    mention_source = payload.get("mention_source", "transport" if mentions else "none")
    if mention_source not in {"none", "transport", "text_fallback"}:
        raise ValueError("Unknown mention source")
    source_timestamp = payload.get("source_timestamp")
    if source_timestamp is not None and (
        isinstance(source_timestamp, bool)
        or not isinstance(source_timestamp, (float, int))
        or not math.isfinite(source_timestamp)
    ):
        raise ValueError("source_timestamp must be a finite timestamp")
    message_type = payload.get("message_type", payload.get("type", "text"))
    if message_type != "text":
        raise ValueError("Only text messages are supported in this prototype")
    return Event(
        **identities,
        content=payload["content"],
        sender_name=str(payload.get("sender_name", "")),
        is_group=payload.get("is_group", identities["chat_id"].endswith("@chatroom")),
        is_self=payload.get("is_self", False),
        mentioned_ids=tuple(dict.fromkeys(item.strip() for item in mentions)),
        received_at=float(received_at),
        mention_source=mention_source,
        text_mention_matched=payload.get("text_mention_matched", False),
        source=str(payload.get("source", "synthetic")),
        source_timestamp=source_timestamp,
        message_type=message_type,
    )


def scope_key(event: Event) -> tuple[str, str, str]:
    return event.account_id, event.chat_id, event.sender_id


def should_reply(
    event: Event, *, own_id: str, allowed_chat_ids: set[str], require_at: bool = True
) -> bool:
    return (
        not event.is_self
        and event.sender_id != own_id
        and event.chat_id in allowed_chat_ids
        and (not event.is_group or not require_at or own_id in event.mentioned_ids)
    )


class DedupeWindow:
    """Bounded in-process suppression; persistent receipt storage is separate."""

    def __init__(self, ttl_seconds=3600, max_entries=10000, clock=time.monotonic):
        if ttl_seconds <= 0 or max_entries <= 0:
            raise ValueError("Dedupe limits must be positive")
        self.ttl = ttl_seconds
        self.capacity = max_entries
        self.clock = clock
        self.entries: OrderedDict[Hashable, float] = OrderedDict()
        self.lock = threading.Lock()

    def accept(self, key: Hashable) -> bool:
        with self.lock:
            now = self.clock()
            while self.entries and next(iter(self.entries.values())) <= now:
                self.entries.popitem(last=False)
            if key in self.entries:
                return False
            self.entries[key] = now + self.ttl
            while len(self.entries) > self.capacity:
                self.entries.popitem(last=False)
            return True


# One executor for one process-wide desktop. Cancelling an awaiting coroutine
# never releases the desktop to a second worker while the first is still running.
_desktop_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="wechat-ui")


class SerializedSender:
    def __init__(self, send_impl: Callable[[str, str], Any]):
        self.send_impl = send_impl

    async def send(self, chat_id: str, text: str, *, can_send=None):
        def dispatch():
            if can_send is not None and not can_send():
                raise TimeoutError("Request expired or was already claimed")
            return self.send_impl(chat_id, text)

        return await asyncio.wrap_future(_desktop_executor.submit(dispatch))


class RequestGuard:
    """Block cancelled, late and repeated results without reopening old IDs."""

    def __init__(self, clock=time.monotonic, max_entries=10000):
        self.clock = clock
        self.max_entries = max_entries
        self.entries: dict[str, tuple[float, str]] = {}
        self.lock = threading.Lock()

    def begin(self, request_id: str, deadline_seconds: float) -> None:
        if (
            not request_id
            or not math.isfinite(deadline_seconds)
            or deadline_seconds <= 0
        ):
            raise ValueError("Request identity and positive deadline are required")
        with self.lock:
            now = self.clock()
            # SQLite owns durable deduplication. In-memory tombstones only cover
            # the response window, so normal traffic cannot exhaust this guard.
            for identity, (deadline, state) in list(self.entries.items()):
                if deadline <= now and state not in {"pending", "claimed"}:
                    del self.entries[identity]
            if request_id in self.entries:
                raise ValueError("A request ID cannot be reused")
            if len(self.entries) >= self.max_entries:
                raise RuntimeError("Request guard is full; restart after draining")
            self.entries[request_id] = now + deadline_seconds, "pending"

    def can_send(self, request_id: str) -> bool:
        with self.lock:
            deadline, state = self.entries.get(request_id, (0, "unknown"))
            return state == "pending" and self.clock() < deadline

    def claim(self, request_id: str) -> bool:
        """Atomically claim one send immediately before entering the UI worker."""
        with self.lock:
            deadline, state = self.entries.get(request_id, (0, "unknown"))
            if state != "pending" or self.clock() >= deadline:
                return False
            self.entries[request_id] = deadline, "claimed"
            return True

    def expired(self, request_id: str) -> bool:
        with self.lock:
            deadline, state = self.entries.get(request_id, (0, "unknown"))
            return state == "pending" and self.clock() >= deadline

    def cancel_all(self) -> None:
        with self.lock:
            for identity, (deadline, state) in list(self.entries.items()):
                if state in {"pending", "claimed"}:
                    self.entries[identity] = deadline, "cancelled"

    def cancel(self, request_id: str) -> None:
        with self.lock:
            if request_id in self.entries:
                self.entries[request_id] = self.entries[request_id][0], "cancelled"

    def complete(self, request_id: str) -> None:
        with self.lock:
            if request_id in self.entries:
                self.entries[request_id] = self.entries[request_id][0], "completed"
