"""Local waking policy: immutable scope decisions, never permissions or tools.

Only a selected group's new text reaches this module. Ordinary selection uses
bounded local rules and persistent quotas; no classification model is called.
Display aliases are private fallback evidence, never stable member identities.

Primary reference: pinned AstrBot v4.29.3 core/pipeline/waking_check/stage.py
WakingCheckStage.process separates At/Reply/prefix waking from stable sender-ID
administrator roles. The ambient gates and exact cooldowns below are local
bridge extensions, not behavior claimed to be supplied by that upstream stage.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import sqlite3
import threading
import time
from contextlib import closing
from dataclasses import dataclass, field, replace
from pathlib import Path

from . import Event


def _identity(value):
    return (
        isinstance(value, str)
        and bool(value)
        and len(value) <= 1024
        and not any(char.isspace() or ord(char) < 32 for char in value)
    )


@dataclass(frozen=True, slots=True)
class ReplyDecision:
    account_id: str
    chat_id: str
    sender_id: str
    message_id: str
    should_reply: bool
    reason: str
    observe: bool = True

    def validate_for(self, event):
        if not isinstance(event, Event) or any(
            not _identity(getattr(self, name))
            or getattr(self, name) != getattr(event, name)
            for name in ("account_id", "chat_id", "sender_id", "message_id")
        ):
            raise ValueError("Reply decision source binding differs")
        if (
            type(self.should_reply) is not bool
            or type(self.observe) is not bool
            or not isinstance(self.reason, str)
            or not re.fullmatch(r"[a-z][a-z0-9_]{0,79}", self.reason)
        ):
            raise ValueError("Invalid local reply decision")


@dataclass(frozen=True, slots=True)
class MentionIdentity:
    bot_id: str
    aliases: tuple[str, ...] = field(repr=False)
    metadata_source: str = "scoped_contact_nickname_and_remark"

    def matches(self, text):
        if not isinstance(text, str):
            return False
        # WeChat commonly uses U+2005 after a mention. Exact names and explicit
        # delimiters avoid @bot matching @bot2, email addresses, or substrings.
        return any(
            re.search(
                r"(?<![A-Za-z0-9_@.])@"
                + re.escape(alias)
                + r"(?=$|[\s，。！？、；：,.!?;:])",
                text,
            )
            for alias in self.aliases
        )

    def leading_action_text(self, text):
        """Return only a suffix after a proven leading bot token, else None.

        Horizontal spacing and the common U+2005 mention separator are allowed.
        A quoted, mid-sentence, unknown or adjacent longer name is never trimmed.
        The original Event content remains the source fingerprint and evidence.
        """
        if not isinstance(text, str):
            return None
        for alias in sorted(self.aliases, key=len, reverse=True):
            match = re.match(
                r"[^\S\r\n]*@" + re.escape(alias) + r"(?:[^\S\r\n]+|$)",
                text,
            )
            if match:
                return text[match.end() :]
        return None


def derive_mention_identity(runtime):
    account = runtime.get("account_id") if isinstance(runtime, dict) else None
    members = runtime.get("members") if isinstance(runtime, dict) else None
    if (
        not _identity(account)
        or not isinstance(members, list)
        or not 2 <= len(members) <= 500
    ):
        raise ValueError("Private mention roster is invalid")
    names, ids = {}, set()
    for member in members:
        identity = member.get("username") if isinstance(member, dict) else None
        if not _identity(identity) or identity in ids:
            raise ValueError("Private mention roster identity is ambiguous")
        ids.add(identity)
        for key in ("nick_name", "remark"):
            name = member.get(key)
            if name is None or name == "":
                continue
            if (
                not isinstance(name, str)
                or name != name.strip()
                or len(name) > 128
                or any(ord(char) < 32 for char in name)
                or "@" in name
            ):
                continue
            try:
                name.encode("utf-8")
            except UnicodeError:
                continue
            names.setdefault(name, set()).add(identity)
    if account not in ids:
        raise ValueError("Bot is not in private mention roster")
    aliases = tuple(
        sorted(name for name, owners in names.items() if owners == {account})
    )
    return MentionIdentity(account, aliases)


def mark_mentions(event, identity):
    """Fallback never fabricates protocol mentioned_ids or grants authority."""
    if event.mention_source == "transport":
        return event
    matched = bool(
        identity
        and identity.bot_id == event.account_id
        and identity.matches(event.content)
    )
    return replace(
        event,
        mentioned_ids=(),
        mention_source="text_fallback" if matched else "none",
        text_mention_matched=matched,
    )


@dataclass(frozen=True, slots=True)
class PolicyLimits:
    group_cooldown: float = 60
    member_cooldown: float = 180
    hourly_cap: int = 6

    def __post_init__(self):
        if (
            any(
                type(value) not in (int, float)
                or not math.isfinite(value)
                or not 0 <= value <= 3600
                for value in (self.group_cooldown, self.member_cooldown)
            )
            or type(self.hourly_cap) is not int
            or not 1 <= self.hourly_cap <= 24
        ):
            raise ValueError("Local ambient limits are invalid")


class AmbientQuota:
    """Reserve ordinary replies locally; a failed send still consumes its slot."""

    def __init__(self, path=None):
        self.path = Path(path).resolve() if path is not None else None
        self.lock, self.rows = threading.Lock(), []
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with closing(sqlite3.connect(self.path)) as db, db:
                db.execute(
                    "CREATE TABLE IF NOT EXISTS ambient_admissions (identity TEXT PRIMARY KEY, account TEXT, chat TEXT, sender TEXT, admitted_at REAL)"
                )

    @staticmethod
    def _admit(rows, event, now, limits):
        encoded = json.dumps(
            [event.account_id, event.chat_id, event.sender_id, event.message_id],
            separators=(",", ":"),
        )
        identity = hashlib.sha256(encoded.encode()).hexdigest()
        scoped = [
            row
            for row in rows
            if row[1:3] == (event.account_id, event.chat_id) and row[4] > now - 3600
        ]
        if any(row[0] == identity for row in scoped):
            return "ambient_selected", None
        if len(scoped) >= limits.hourly_cap:
            return "ambient_hour_cap", None
        if any(now - row[4] < limits.group_cooldown for row in scoped):
            return "ambient_group_cooldown", None
        if any(
            row[3] == event.sender_id and now - row[4] < limits.member_cooldown
            for row in scoped
        ):
            return "ambient_member_cooldown", None
        return "ambient_selected", (
            identity,
            event.account_id,
            event.chat_id,
            event.sender_id,
            now,
        )

    def reserve(self, event, now, limits):
        with self.lock:
            if self.path is None:
                self.rows = [row for row in self.rows if row[4] > now - 3600]
                reason, row = self._admit(self.rows, event, now, limits)
                if row:
                    self.rows.append(row)
                return reason
            with closing(sqlite3.connect(self.path, timeout=0.2)) as db, db:
                db.execute("BEGIN IMMEDIATE")
                db.execute(
                    "DELETE FROM ambient_admissions WHERE admitted_at<=?", (now - 3600,)
                )
                rows = db.execute(
                    "SELECT identity,account,chat,sender,admitted_at FROM ambient_admissions WHERE account=? AND chat=?",
                    (event.account_id, event.chat_id),
                ).fetchall()
                reason, row = self._admit(rows, event, now, limits)
                if row:
                    db.execute("INSERT INTO ambient_admissions VALUES(?,?,?,?,?)", row)
                return reason


class ReplyPolicy:
    def __init__(
        self,
        account_id,
        chat_id,
        *,
        bot_id=None,
        mention_identity=None,
        quota=None,
        limits=None,
        clock=time.time,
    ):
        if (
            not _identity(account_id)
            or not _identity(chat_id)
            or not chat_id.endswith("@chatroom")
        ):
            raise ValueError("Policy must have one fixed account and group")
        self.account_id, self.chat_id = account_id, chat_id
        self.bot_id = bot_id or account_id
        self.mention_identity, self.quota = mention_identity, quota or AmbientQuota()
        self.limits, self.clock = limits or PolicyLimits(), clock

    def decide(self, event, *, action_kind="none"):
        def result(allowed, reason, observe=True):
            decision = ReplyDecision(
                event.account_id,
                event.chat_id,
                event.sender_id,
                event.message_id,
                allowed,
                reason,
                observe,
            )
            decision.validate_for(event)
            return decision

        if (
            event.account_id != self.account_id
            or event.chat_id != self.chat_id
            or not event.is_group
        ):
            return result(False, "scope_rejected", False)
        if event.is_self or event.sender_id in {self.bot_id, self.account_id}:
            return result(False, "self_ignored", False)
        if event.message_type != "text" or not event.content.strip():
            return result(False, "nontext_or_empty", False)
        if event.mention_source == "transport" and self.bot_id in event.mentioned_ids:
            return result(True, "transport_mention")
        if self.mention_identity and self.mention_identity.matches(event.content):
            return result(True, "text_mention_fallback")
        if action_kind == "help":
            return result(True, "help_requested")
        if action_kind in {"remember", "forget", "correct"}:
            return result(True, "memory_command")
        if action_kind in {"search_web", "search_meme", "search_rejected"}:
            return result(True, "search_requested")
        if event.content.startswith("/ai "):
            return result(True, "legacy_prefix")
        if not re.search(
            r"[?？]|怎么|如何|为什么|能不能|可不可以|有没有|是否|多少|哪里|什么|你觉得|你认为|请问|吗[。！\s]*$",
            event.content,
        ):
            return result(False, "ambient_not_question")
        relevant = (
            bool(re.search(r"AI|人工智能|机器人|助手|\bbot\b", event.content, re.I))
            or bool(
                re.search(
                    r"^\s*(?:你|请问[，,：:\s]*你)|"
                    r"(?:^|[，,：:\s])(?:大家|有人知道|有没有推荐|问一下|求助|怎么办)",
                    event.content,
                )
            )
            or (
                "你" in event.content
                and any(
                    word in event.content
                    for word in (
                        "觉得",
                        "建议",
                        "帮我",
                        "推荐",
                        "知道",
                        "怎么看",
                        "认为",
                        "能不能",
                        "教我",
                    )
                )
            )
        )
        if not relevant:
            return result(False, "ambient_not_relevant")
        try:
            reason = self.quota.reserve(event, self.clock(), self.limits)
        except (OSError, sqlite3.Error):
            reason = "ambient_state_unavailable"
        return result(reason == "ambient_selected", reason)
