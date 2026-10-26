"""Synthetic image fixtures and fake fetch only; no desktop or public IO."""

import io
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from PIL import Image

from scoped_bot_core.media_assets import (
    MAX_BYTES,
    MAX_FRAMES,
    MediaAsset,
    MediaAssetCache,
    MediaBlocked,
    fetch_image_bytes,
    inspect_image,
)

URL = "https://img.soutula.com/bmiddle/synthetic.gif"
PAGE = "https://www.fabiaoqing.com/biaoqing/detail/id/123.html"


def fixture(kind="PNG", size=(16, 12), frames=1):
    output = io.BytesIO()
    images = [
        Image.new("RGB", size, (index * 17 % 255, 40, 80)) for index in range(frames)
    ]
    images[0].save(
        output, format=kind, save_all=frames > 1, append_images=images[1:], duration=30
    )
    return output.getvalue()


class MediaAssetTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.now, self.fetches = 10.0, []
        self.raw, self.mime = fixture(), "image/png"

        def fetch(url, **kwargs):
            self.fetches.append((url, kwargs))
            return self.mime, self.raw

        self.cache = MediaAssetCache(
            Path(self.temp.name) / "media-cache", fetch=fetch, clock=lambda: self.now
        )
        self.candidate = SimpleNamespace(image_url=URL, page_url=PAGE)

    def prepare(self):
        return self.cache.prepare(self.candidate, deadline=20)

    def test_png_roundtrip_binds_canonical_name_bytes_hash_and_image_properties(self):
        asset = self.prepare()
        self.assertEqual(
            (asset.image_format, asset.width, asset.height, asset.frames),
            ("PNG", 16, 12, 1),
        )
        self.assertEqual(asset.asset_name, asset.sha256 + ".png")
        decoded, path = self.cache.load(asset.canonical_json(), deadline=20)
        self.assertEqual(decoded, asset)
        self.assertEqual(path.read_bytes(), self.raw)
        self.assertEqual(len(self.fetches), 1)
        self.assertEqual(self.fetches[0][1]["timeout"], 6)

    def test_animated_gif_preserves_original_bytes_and_frame_count(self):
        self.mime, self.raw = "image/gif", fixture("GIF", frames=3)
        asset = self.prepare()
        self.assertEqual((asset.frames, asset.image_format), (3, "GIF"))
        self.assertEqual(
            self.cache.load(asset.canonical_json(), deadline=20)[1].read_bytes(),
            self.raw,
        )

    def test_rejects_untrusted_origin_queries_credentials_and_page_before_fetch(self):
        bad = [
            "http://img.soutula.com/bmiddle/a.png",
            "https://127.0.0.1/bmiddle/a.png",
            "https://img.soutula.com.evil.test/bmiddle/a.png",
            "https://img.soutula.com@evil.test/bmiddle/a.png",
            "https://u:p@img.soutula.com/bmiddle/a.png",
            "https://img.soutula.com/bmiddle/a.png?token=secret",
            "https://img.soutula.com/bmiddle/a.png#x",
            "https://img.soutula.com/../a.png",
            "https://img.soutula.com/bmiddle/%2e%2e.png",
            "https://img.soutula.com/bmiddle/a.svg",
        ]
        for value in bad:
            with self.subTest(value=value), self.assertRaises(MediaBlocked):
                self.candidate.image_url = value
                self.prepare()
        self.candidate.image_url, self.candidate.page_url = URL, "https://evil.test/"
        with self.assertRaises(MediaBlocked):
            self.prepare()
        self.assertEqual(self.fetches, [])

    def test_deadline_before_fetch_and_after_fetch_never_persists_asset(self):
        self.now = 21
        with self.assertRaises(MediaBlocked):
            self.prepare()
        self.assertEqual(self.fetches, [])
        self.now = 10
        self.cache.fetch = lambda *_args, **_kwargs: (
            setattr(self, "now", 21) or self.mime,
            self.raw,
        )
        with self.assertRaises(MediaBlocked):
            self.prepare()
        self.assertEqual(list(self.cache.root.iterdir()), [])

    def test_magic_mime_corruption_size_and_dimensions_fail_closed(self):
        for raw, mime in (
            (self.raw, "image/jpeg"),
            (b"<script>x</script>", "image/png"),
            (self.raw, "text/html"),
            (b"x" * (MAX_BYTES + 1), "image/png"),
            (self.raw[:28], "image/png"),
            (fixture(size=(2049, 1)), "image/png"),
        ):
            with (
                self.subTest(mime=mime, size=len(raw)),
                self.assertRaises(MediaBlocked),
            ):
                inspect_image(raw, mime, deadline=20, clock=lambda: 10)

    def test_frame_limit_blocks_animation_before_loading_all_frames(self):
        raw = fixture("GIF", size=(2, 2), frames=MAX_FRAMES + 1)
        with self.assertRaisesRegex(MediaBlocked, "dimensions_or_frames"):
            inspect_image(raw, "image/gif", deadline=20, clock=lambda: 10)

    def test_manifest_rejects_absolute_path_traversal_extra_keys_and_noncanonical_json(
        self,
    ):
        asset = self.prepare()
        for changed in (
            replace(asset, asset_name="../a.png"),
            replace(asset, asset_name=str(self.cache.root / "outside.png")),
            replace(asset, md5="0" * 31),
            replace(asset, frames=True),
        ):
            with self.assertRaises(MediaBlocked):
                self.cache.load(
                    json.dumps(changed.__dict__)
                    if hasattr(changed, "__dict__")
                    else json.dumps(
                        {
                            **json.loads(asset.canonical_json()),
                            "asset_name": changed.asset_name,
                            "md5": changed.md5,
                            "frames": changed.frames,
                        }
                    ),
                    deadline=20,
                )
        for value in (
            json.dumps(json.loads(asset.canonical_json()), indent=2),
            asset.canonical_json()[:-1] + ',"path":"C:/private"}',
            "[]",
            "null",
        ):
            with self.assertRaises(MediaBlocked):
                MediaAsset.from_json(value)

    def test_changed_cache_bytes_and_hash_fail_before_send(self):
        asset = self.prepare()
        path = self.cache.root / asset.asset_name
        path.write_bytes(bytes([self.raw[0] ^ 1]) + self.raw[1:])
        with self.assertRaisesRegex(MediaBlocked, "hash_changed"):
            self.cache.load(asset.canonical_json(), deadline=20)

    def test_existing_foreign_file_is_never_overwritten(self):
        asset = self.prepare()
        path = self.cache.root / asset.asset_name
        path.write_bytes(b"foreign")
        with self.assertRaises(MediaBlocked):
            self.prepare()
        self.assertEqual(path.read_bytes(), b"foreign")

    def test_image_decoder_exception_is_fixed_reason_not_content_or_path(self):
        with patch("PIL.Image.open", side_effect=OSError("sensitive private path")):
            with self.assertRaises(MediaBlocked) as error:
                self.prepare()
        self.assertEqual(str(error.exception), "media_decode_rejected")

    def test_fetch_unapproved_url_never_launches_child(self):
        with patch("subprocess.run") as runner:
            with self.assertRaises(MediaBlocked):
                fetch_image_bytes("https://evil.test/private")
        runner.assert_not_called()


if __name__ == "__main__":
    unittest.main()
