"""A local author asset is provenance, never a runtime download permission."""

import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from scoped_bot_core.media_assets import (
    MediaAssetCache,
    MediaBlocked,
    fetch_image_bytes,
    validate_asset_url,
    validate_page_url,
)
from scoped_bot_core.public_search import PublicResult

URL = "https://kiriyuki.com/media/drag/thumb/" + "a" * 64 + ".png"
PAGE = "https://kiriyuki.com/showcase?p=%E5%A4%A7%E8%82%A5%E9%B1%BC"


class WhaleMediaOriginsTests(unittest.TestCase):
    def test_exact_static_author_thumbnail_provenance_only(self):
        self.assertEqual(validate_asset_url(URL), URL)
        self.assertEqual(validate_page_url(PAGE), PAGE)
        for url in (
            URL.replace("https:", "http:"),
            URL.replace("kiriyuki.com", "kiriyuki.com.evil.example"),
            URL.replace("kiriyuki.com", "user:pass@kiriyuki.com"),
            URL.replace("kiriyuki.com", "kiriyuki.com:443"),
            URL + "?redirect=https://evil.example/",
            URL + "#fragment",
            URL.replace("/media/drag/thumb/", "/arbitrary/"),
            URL.replace("a" * 64, "../" + "a" * 64),
            URL.replace(".png", ".gif"),
            URL.replace("a" * 64, "a" * 63),
        ):
            with self.subTest(url=url):
                with self.assertRaises(MediaBlocked):
                    validate_asset_url(url)

    def test_only_single_whale_pack_page_is_accepted(self):
        for value in (
            PAGE + "&other=1",
            PAGE + "#anything",
            "https://kiriyuki.com/",
            "https://kiriyuki.com/showcase?p=GPT",
            PAGE.replace("kiriyuki.com", "evil.example"),
        ):
            with self.subTest(value=value):
                with self.assertRaises(MediaBlocked):
                    validate_page_url(value)

    def test_runtime_fetch_refuses_author_source_before_network(self):
        with patch("scoped_bot_core.media_assets.subprocess.run") as run:
            with self.assertRaisesRegex(MediaBlocked, "reviewed_local_import"):
                fetch_image_bytes(URL)
            run.assert_not_called()
        with tempfile.TemporaryDirectory(prefix="whale-origin-synthetic-") as folder:
            fetch = Mock(
                side_effect=AssertionError("No network for local author assets")
            )
            cache = MediaAssetCache(Path(folder) / "cache", fetch=fetch)
            with self.assertRaisesRegex(MediaBlocked, "reviewed_local_import"):
                cache.prepare(
                    PublicResult("synthetic", PAGE, image_url=URL),
                    deadline=time.monotonic() + 5,
                )
            fetch.assert_not_called()


if __name__ == "__main__":
    unittest.main()
