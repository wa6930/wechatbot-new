"""Reviewed-byte binding and fixed public queries, without network or disk IO."""

import copy
import hashlib
import unittest
from dataclasses import replace
from unittest.mock import patch

from scoped_bot_core.media_assets import MediaAsset, MediaBlocked
from scoped_bot_core.public_search import SearchBlocked
from scoped_bot_core.reaction_catalog import CuratedReactionSource
from scoped_bot_core.reaction_policy import REACTION_QUERIES


def row(reaction="happy", **changes):
    value = {
        "reaction": reaction,
        "title": "合成开心反应图",
        "page_url": "https://www.fabiaoqing.com/biaoqing/detail/id/100.html",
        "image_url": "https://img.soutula.com/bmiddle/synthetic-happy.jpg",
        "sha256": "a" * 64,
    }
    value.update(changes)
    return value


def catalog(*items):
    return {"schema_version": 1, "items": list(items or (row(),))}


def asset(**changes):
    value = MediaAsset(
        "a" * 64 + ".jpg",
        "a" * 64,
        "b" * 32,
        100,
        "image/jpeg",
        "JPEG",
        64,
        64,
        1,
        row()["image_url"],
        row()["page_url"],
    )
    return replace(value, **changes)


class CuratedReactionTests(unittest.TestCase):
    def test_fixed_word_returns_only_reviewed_candidate_and_correct_digest(self):
        source = CuratedReactionSource.from_dict(catalog())
        with patch("scoped_bot_core.reaction_catalog.time.time", return_value=12.5):
            page = source.search("开心")
        self.assertEqual(page.provider, "fabiaoqing_html")
        self.assertEqual(page.fetched_at, 12.5)
        self.assertEqual(page.query_sha256, hashlib.sha256("开心".encode()).hexdigest())
        self.assertEqual(len(page.results), 1)
        result = page.results[0]
        self.assertEqual(result.title, row()["title"])
        self.assertEqual(result.page_url, row()["page_url"])
        self.assertEqual(result.image_url, row()["image_url"])
        self.assertEqual(result.license_status, "unknown")

    def test_all_six_fixed_words_are_supported_when_individually_reviewed(self):
        source = CuratedReactionSource.from_dict(
            catalog(*(row(mood) for mood in REACTION_QUERIES))
        )
        for query in REACTION_QUERIES.values():
            with self.subTest(query=query):
                self.assertEqual(len(source.search(query, kind="meme").results), 1)

    def test_unreviewed_mood_is_unavailable_instead_of_searching(self):
        source = CuratedReactionSource.from_dict(catalog())
        with self.assertRaisesRegex(SearchBlocked, "^curated_reaction_unavailable$"):
            source.search("谢谢")

    def test_empty_catalog_is_safe_and_has_no_unreviewed_fallback(self):
        source = CuratedReactionSource.from_dict({"schema_version": 1, "items": []})
        with self.assertRaises(SearchBlocked):
            source.search("开心")

    def test_arbitrary_private_query_or_mood_key_never_resolves_or_leaks(self):
        source = CuratedReactionSource.from_dict(catalog())
        for query in (
            "happy",
            "开心 合成成员PRIVATE",
            "token=synthetic-private-value",
            "https://example.com/",
            " 开心",
            "开心\n",
            {"query": "开心"},
            None,
        ):
            with self.subTest(query=query):
                with self.assertRaises(SearchBlocked) as caught:
                    source.search(query)
                self.assertEqual(str(caught.exception), "curated_reaction_unavailable")

    def test_source_cannot_be_used_as_general_web_search(self):
        source = CuratedReactionSource.from_dict(catalog())
        with self.assertRaisesRegex(SearchBlocked, "^curated_reaction_kind_rejected$"):
            source.search("开心", kind="web")

    def test_original_config_cannot_change_reviewed_hash_or_metadata(self):
        value = catalog()
        source = CuratedReactionSource.from_dict(value)
        value["items"][0]["sha256"] = "c" * 64
        value["items"][0]["image_url"] = "https://img.soutula.com/bmiddle/other.jpg"
        value["items"].clear()
        self.assertEqual(source.accept_asset("开心", asset()), asset())
        self.assertEqual(source.search("开心").results[0].image_url, row()["image_url"])

    def test_catalog_schema_count_and_extra_fields_are_rejected(self):
        variants = (
            None,
            [],
            {},
            {"schema_version": True, "items": []},
            {"schema_version": 2, "items": []},
            {"schema_version": 1, "items": ()},
            {"schema_version": 1, "items": [], "trusted": True},
            {"schema_version": 1, "items": [row()] * 7},
        )
        for value in variants:
            with self.subTest(value=value), self.assertRaises(ValueError):
                CuratedReactionSource.from_dict(value)

    def test_malformed_item_unknown_mood_and_duplicate_mood_are_rejected(self):
        bad = (
            None,
            [],
            row("unknown"),
            row(True),
            row(
                [],
            ),
            row(extra="not permitted"),
        )
        missing = row()
        del missing["page_url"]
        for item in (*bad, missing):
            with self.subTest(item=item), self.assertRaises(ValueError):
                CuratedReactionSource.from_dict(catalog(item))
        with self.assertRaisesRegex(ValueError, "^curated_reaction_duplicate$"):
            CuratedReactionSource.from_dict(catalog(row(), row()))

    def test_hash_must_be_exact_lowercase_sha256(self):
        for value in (None, 1, "A" * 64, "a" * 63, "a" * 65, "z" * 64):
            with self.subTest(value=value), self.assertRaises(ValueError):
                CuratedReactionSource.from_dict(catalog(row(sha256=value)))

    def test_titles_cannot_be_control_payloads_or_unbounded(self):
        for value in (
            None,
            "",
            "   ",
            "x" * 121,
            "开心\n图片",
            "开心\t图片",
            "开心\x7f图片",
            "开心\u202e图片",
            "开心\ud800图片",
        ):
            with self.subTest(value=repr(value)), self.assertRaises(ValueError):
                CuratedReactionSource.from_dict(catalog(row(title=value)))

    def test_page_and_asset_urls_use_existing_fixed_origin_validators(self):
        for field, value in (
            ("page_url", "https://example.com/private"),
            ("page_url", "https://www.fabiaoqing.com/biaoqing/detail/id/1.html?x=1"),
            ("image_url", "https://example.com/pic.jpg"),
            ("image_url", "https://img.soutula.com/bmiddle/x.jpg?token=synthetic"),
            ("image_url", "https://user:synthetic@img.soutula.com/bmiddle/x.jpg"),
            ("image_url", "file:///private.png"),
        ):
            with (
                self.subTest(field=field, value=value),
                self.assertRaises(MediaBlocked),
            ):
                CuratedReactionSource.from_dict(catalog(row(**{field: value})))

    def test_asset_exactly_matches_reviewed_bytes_and_sources(self):
        source = CuratedReactionSource.from_dict(catalog())
        approved = asset()
        self.assertIs(source.accept_asset("开心", approved), approved)

    def test_changed_hash_same_origin_url_or_page_is_rejected(self):
        source = CuratedReactionSource.from_dict(catalog())
        variants = (
            asset(sha256="c" * 64, asset_name="c" * 64 + ".jpg"),
            asset(source_url="https://img.soutula.com/bmiddle/other.jpg"),
            asset(page_url="https://www.fabiaoqing.com/biaoqing/detail/id/101.html"),
        )
        for value in variants:
            with (
                self.subTest(value=value),
                self.assertRaisesRegex(
                    MediaBlocked, "^curated_reaction_asset_rejected$"
                ),
            ):
                source.accept_asset("开心", value)

    def test_valid_png_is_allowed_when_its_exact_bytes_were_reviewed(self):
        image_url = "https://img.soutula.com/bmiddle/synthetic-happy.png"
        source = CuratedReactionSource.from_dict(catalog(row(image_url=image_url)))
        value = asset(
            asset_name="a" * 64 + ".png",
            mime_type="image/png",
            image_format="PNG",
            source_url=image_url,
        )
        self.assertIs(source.accept_asset("开心", value), value)

    def test_gif_webp_or_multiframe_never_becomes_automatic_reaction(self):
        for mime, image_format, extension, frames in (
            ("image/gif", "GIF", ".gif", 1),
            ("image/webp", "WEBP", ".webp", 1),
            ("image/jpeg", "JPEG", ".jpg", 2),
        ):
            image_url = "https://img.soutula.com/bmiddle/synthetic-happy" + extension
            source = CuratedReactionSource.from_dict(catalog(row(image_url=image_url)))
            value = asset(
                asset_name="a" * 64 + extension,
                mime_type=mime,
                image_format=image_format,
                source_url=image_url,
                frames=frames,
            )
            with (
                self.subTest(image_format=image_format),
                self.assertRaises(MediaBlocked),
            ):
                source.accept_asset("开心", value)

    def test_asset_type_invalid_metadata_and_unreviewed_query_are_rejected(self):
        source = CuratedReactionSource.from_dict(catalog())
        for value in (None, {}, copy.copy(row()), asset(width=0), asset(md5="invalid")):
            with self.subTest(value=value), self.assertRaises(MediaBlocked):
                source.accept_asset("开心", value)
        with self.assertRaisesRegex(MediaBlocked, "^curated_reaction_unavailable$"):
            source.accept_asset("谢谢", asset())


if __name__ == "__main__":
    unittest.main()
