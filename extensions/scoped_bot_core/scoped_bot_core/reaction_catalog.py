"""Inert catalog of owner-reviewed reaction images; no search or file IO.

Only fixed public mood words resolve to reviewed candidates. The trusted caller
loads the catalog and separately prepares/decodes actual image bytes. Approval
here binds those bytes' resulting metadata to the reviewed SHA256 and source;
it does not infer semantic safety from a title or permit chat to add entries.
"""

from __future__ import annotations

import hashlib
import re
import time
import unicodedata
from dataclasses import dataclass
from types import MappingProxyType

from .media_assets import (
    MediaAsset,
    MediaBlocked,
    validate_asset_url,
    validate_page_url,
)
from .public_search import PublicResult, PublicSearchPage, SearchBlocked
from .reaction_policy import REACTION_QUERIES

MAX_REACTIONS = 6
MAX_TITLE_CHARS = 120
MAX_TITLE_BYTES = 512


@dataclass(frozen=True, slots=True)
class _ApprovedImage:
    title: str
    page_url: str
    image_url: str
    sha256: str


def _title(value):
    try:
        valid = (
            isinstance(value, str)
            and bool(value.strip())
            and len(value) <= MAX_TITLE_CHARS
            and len(value.encode("utf-8")) <= MAX_TITLE_BYTES
            and not any(
                unicodedata.category(char) in {"Cc", "Cf", "Cs"} for char in value
            )
        )
    except UnicodeError:
        valid = False
    if not valid:
        raise ValueError("curated_reaction_title_rejected")
    return value.strip()


class CuratedReactionSource:
    """Select approved metadata without transmitting chat or making a request.

    Construction must use trusted deployment data. Returned candidates still
    require normal bounded download/decode and ``accept_asset`` before queueing.
    The approval file is not a declaration that an image is licensed for reuse.
    """

    __slots__ = ("_by_query",)

    def __init__(self, value):
        if (
            type(value) is not dict
            or set(value) != {"schema_version", "items"}
            or type(value["schema_version"]) is not int
            or value["schema_version"] != 1
            or type(value["items"]) is not list
            or len(value["items"]) > MAX_REACTIONS
        ):
            raise ValueError("curated_reaction_catalog_rejected")
        by_query = {}
        for row in value["items"]:
            if (
                type(row) is not dict
                or set(row) != {"reaction", "title", "page_url", "image_url", "sha256"}
                or not isinstance(row["reaction"], str)
                or row["reaction"] not in REACTION_QUERIES
                or not isinstance(row["sha256"], str)
                or not re.fullmatch(r"[0-9a-f]{64}", row["sha256"])
            ):
                raise ValueError("curated_reaction_item_rejected")
            query = REACTION_QUERIES[row["reaction"]]
            if query in by_query:
                raise ValueError("curated_reaction_duplicate")
            by_query[query] = _ApprovedImage(
                _title(row["title"]),
                validate_page_url(row["page_url"]),
                validate_asset_url(row["image_url"]),
                row["sha256"],
            )
        self._by_query = MappingProxyType(by_query)

    @classmethod
    def from_dict(cls, value):
        return cls(value)

    def _approved(self, query, error_class):
        if not isinstance(query, str) or query not in self._by_query:
            raise error_class("curated_reaction_unavailable")
        return self._by_query[query]

    def search(self, query, *, kind="meme"):
        if kind != "meme":
            raise SearchBlocked("curated_reaction_kind_rejected")
        item = self._approved(query, SearchBlocked)
        return PublicSearchPage(
            "fabiaoqing_html",
            hashlib.sha256(query.encode("utf-8")).hexdigest(),
            (PublicResult(item.title, item.page_url, image_url=item.image_url),),
            time.time(),
        )

    def accept_asset(self, query, asset):
        item = self._approved(query, MediaBlocked)
        if type(asset) is not MediaAsset:
            raise MediaBlocked("curated_reaction_asset_rejected")
        asset.validate()
        if (
            asset.sha256 != item.sha256
            or asset.source_url != item.image_url
            or asset.page_url != item.page_url
            or asset.frames != 1
            or asset.image_format not in {"PNG", "JPEG"}
        ):
            raise MediaBlocked("curated_reaction_asset_rejected")
        return asset
