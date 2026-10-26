"""Explicit public search actions; no model, history, database or network IO.

The caller strips only a trusted leading bot mention, admits the immutable
request, then invokes the search client. Ordinary chat never becomes a query.
Result metadata remains untrusted text, never prompts or executable commands.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from .public_search import (
    PublicResult,
    PublicSearchPage,
    SearchBlocked,
    public_query,
    public_url,
)

_REQUEST_PREFIX = r"(?:(?:你)?(?:能不能|能否|可以)?(?:请|麻烦)?(?:帮我|给我|替我)?\s*)"


@dataclass(frozen=True, slots=True)
class SearchIntent:
    kind: str
    query: str = field(repr=False)

    def __post_init__(self):
        if self.kind not in {"web", "meme"}:
            raise SearchBlocked("search_kind_rejected")
        object.__setattr__(self, "query", public_query(self.query))


def parse_search_intent(text):
    """Return an explicit action or None; reject unsafe/missing query locally."""
    if not isinstance(text, str) or len(text) > 512:
        return None
    value = text.strip()
    if re.match(
        r"^(?:你)?(?:搜索|联网搜索|表情包)(?:功能|能力)(?:好用|怎么|如何|是|有|能)",
        value,
    ):
        return None
    if re.fullmatch(
        r"(?:你)?(?:能|可以|会|支持|能不能)?(?:联网)?(?:搜索|搜|查)(?:吗|么|不)?[?？]?",
        value,
    ):
        return None
    command = re.fullmatch(r"/(search|meme)(?:\s+(.*))?", value, re.I)
    if command:
        return SearchIntent(
            "web" if command[1].lower() == "search" else "meme", command[2] or ""
        )
    # Anchored request syntax avoids quoted examples, incidental keywords and
    # "我想知道怎么搜索" turning into network requests. Do not remove clauses or
    # extract secret-looking fragments: validate the complete requested suffix.
    meme = re.fullmatch(
        _REQUEST_PREFIX
        + r"(?:(?:搜索|搜|找|查|发|来|整)(?:一下|下|一张|一个|几张|几个|张|个)?\s*)"
        r"(?:(?:关于|有关)\s*)?(.*?)\s*(?:的)?表情包[吧呀呗呢。！!]*",
        value,
    )
    if meme:
        return SearchIntent("meme", meme[1].strip())
    meme_prefix = re.fullmatch(
        _REQUEST_PREFIX + r"(?:搜索|搜|找|发)(?:一下|下)?\s*表情包\s*[:：]?\s*(.+)",
        value,
    )
    if meme_prefix:
        return SearchIntent("meme", meme_prefix[1].strip().rstrip("。！!"))
    web = re.fullmatch(
        _REQUEST_PREFIX + r"(?:联网查|联网搜|网上查|网上搜|搜索|搜|查)(?:一下|下)?"
        r"\s*[:：]?\s*(.*)",
        value,
    )
    if web:
        return SearchIntent("web", web[1].strip().rstrip("。！!"))
    return None


def render_web_results(page, *, limit=3):
    """Format actual returned sources as plain WeChat text, without synthesis."""
    if (
        not isinstance(page, PublicSearchPage)
        or page.provider not in {"bing_cn_rss", "duckduckgo_html"}
        or type(limit) is not int
        or not 1 <= limit <= 3
    ):
        raise SearchBlocked("search_result_rejected")
    lines = ["捞到这些网页，摘要来自搜索结果，点来源可以核对："]
    for row in page.results[:limit]:
        if not isinstance(row, PublicResult) or not public_url(row.page_url):
            raise SearchBlocked("search_result_rejected")
        title = " ".join(row.title.split())[:90]
        snippet = " ".join(row.snippet.split())[:120]
        if not title:
            raise SearchBlocked("search_result_rejected")
        lines.extend((f"{len(lines) // 3 + 1}. {title}", snippet, row.page_url))
    if len(lines) == 1:
        raise SearchBlocked("no_results_or_provider_blocked")
    reply = "\n".join(line for line in lines if line)
    if len(reply.encode("utf-8")) > 3500:
        raise SearchBlocked("search_result_rejected")
    return reply


def meme_candidates(page, *, limit=3):
    """Return proven search metadata only; media download/send stays separate."""
    if (
        not isinstance(page, PublicSearchPage)
        or page.provider != "fabiaoqing_html"
        or type(limit) is not int
        or not 1 <= limit <= 3
    ):
        raise SearchBlocked("search_result_rejected")
    rows = page.results[:limit]
    if not rows or any(
        not isinstance(row, PublicResult)
        or not public_url(row.page_url)
        or not public_url(row.image_url)
        for row in rows
    ):
        raise SearchBlocked("search_result_rejected")
    return rows
