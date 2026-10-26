"""Bounded public meme assets; inert metadata, private content-addressed cache.

Only explicit Fabiaoqing search candidates can be fetched. Importing performs
no network, image decoder, GUI, clipboard, configuration, or cache IO.
"""

from __future__ import annotations

import base64
import hashlib
import io
import json
import os
import re
import stat
import subprocess
import time
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from urllib.error import URLError
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, ProxyHandler, Request, build_opener

MAX_BYTES = 2 * 1024 * 1024
MAX_SIDE = 2048
MAX_PIXELS = 4 * 1024 * 1024
MAX_FRAMES = 60
MAX_TOTAL_PIXELS = 12 * 1024 * 1024
FETCH_SECONDS = 6
MIME_FORMAT = {
    "image/png": ("PNG", ".png"),
    "image/jpeg": ("JPEG", ".jpg"),
    "image/gif": ("GIF", ".gif"),
    "image/webp": ("WEBP", ".webp"),
}
POWERSHELL = r"C:\Windows\System32\WindowsPowerShell\v1.0\powershell.exe"
_WINDOWS_FETCH = r"""
$ErrorActionPreference='Stop'
$client=$null; $handler=$null; $response=$null; $stream=$null; $memory=$null; $cts=$null
try {
  Add-Type -AssemblyName System.Net.Http
  $data=([Console]::In.ReadToEnd() | ConvertFrom-Json); $uri=[Uri]$data.url
  if ($uri.Scheme -ne 'https' -or $uri.Host -ne 'img.soutula.com' -or $uri.UserInfo -or $uri.Port -ne 443 -or $uri.Query -or $uri.Fragment -or $uri.AbsolutePath -notmatch '^/bmiddle/[A-Za-z0-9_-]+\.(jpg|jpeg|png|gif|webp)$') { throw 'origin_rejected' }
  $handler=New-Object System.Net.Http.HttpClientHandler
  $handler.AllowAutoRedirect=$false; $handler.UseCookies=$false; $handler.UseProxy=$false
  $handler.UseDefaultCredentials=$false; $handler.DefaultProxyCredentials=$null
  $client=New-Object System.Net.Http.HttpClient($handler)
  $client.Timeout=[TimeSpan]::FromSeconds(6)
  $client.DefaultRequestHeaders.UserAgent.ParseAdd('Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/131.0 Safari/537.36')
  $client.DefaultRequestHeaders.Accept.ParseAdd('image/png,image/jpeg,image/gif,image/webp')
  $client.DefaultRequestHeaders.Referrer=[Uri]'https://www.fabiaoqing.com/'
  $cts=New-Object System.Threading.CancellationTokenSource; $cts.CancelAfter(6000)
  $response=$client.GetAsync($uri,[System.Net.Http.HttpCompletionOption]::ResponseHeadersRead,$cts.Token).GetAwaiter().GetResult()
  if ([int]$response.StatusCode -ne 200 -or $response.RequestMessage.RequestUri.AbsoluteUri -ne $uri.AbsoluteUri) { throw 'response_rejected' }
  $mime=$response.Content.Headers.ContentType.MediaType
  if ($mime -notin @('image/png','image/jpeg','image/gif','image/webp') -or $response.Content.Headers.ContentLength -gt 2097152) { throw 'type_or_size_rejected' }
  $stream=$response.Content.ReadAsStreamAsync().GetAwaiter().GetResult()
  $memory=New-Object System.IO.MemoryStream; $buffer=New-Object byte[] 32768
  do {
    $wanted=[Math]::Min($buffer.Length,2097153-[int]$memory.Length)
    $count=$stream.ReadAsync($buffer,0,$wanted,$cts.Token).GetAwaiter().GetResult()
    if ($count -gt 0) { $memory.Write($buffer,0,$count) }
    if ($memory.Length -gt 2097152) { throw 'size_rejected' }
  } while ($count -gt 0)
  [Console]::Out.Write((@{ready=$true;mime=$mime;body_base64=[Convert]::ToBase64String($memory.ToArray())}|ConvertTo-Json -Compress))
} catch { [Console]::Out.Write('{"ready":false}'); exit 1 }
finally { foreach ($resource in @($stream,$memory,$response,$client,$handler,$cts)) { if ($resource) { $resource.Dispose() } } }
"""


class MediaBlocked(ValueError):
    """Fixed reason only; never include URLs, paths or upstream bytes."""


def _deadline(deadline, clock):
    if type(deadline) not in (int, float) or not deadline > clock():
        raise MediaBlocked("media_deadline_elapsed")


def validate_asset_url(value):
    try:
        parts = urlsplit(value)
        valid = (
            isinstance(value, str)
            and len(value) <= 512
            and parts.scheme == "https"
            and (
                parts.netloc == "img.soutula.com"
                and re.fullmatch(
                    r"/bmiddle/[A-Za-z0-9_-]+\.(?:jpg|jpeg|png|gif|webp)",
                    parts.path,
                    re.I,
                )
                or parts.netloc == "kiriyuki.com"
                and re.fullmatch(r"/media/drag/thumb/[0-9a-f]{64}\.png", parts.path)
            )
            and not parts.query
            and not parts.fragment
        )
    except (ValueError, TypeError, AttributeError):
        valid = False
    if not valid:
        raise MediaBlocked("media_origin_rejected")
    return value


def validate_page_url(value):
    if not isinstance(value, str) or not (
        re.fullmatch(
            r"https://www\.fabiaoqing\.com/biaoqing/detail/id/[0-9]+\.html", value
        )
        or value == "https://kiriyuki.com/showcase?p=%E5%A4%A7%E8%82%A5%E9%B1%BC"
    ):
        raise MediaBlocked("media_source_page_rejected")
    return value


@dataclass(frozen=True, slots=True)
class MediaAsset:
    asset_name: str
    sha256: str
    md5: str
    byte_length: int
    mime_type: str
    image_format: str
    width: int
    height: int
    frames: int
    source_url: str
    page_url: str

    def validate(self):
        validate_asset_url(self.source_url)
        validate_page_url(self.page_url)
        if (
            not isinstance(self.sha256, str)
            or not re.fullmatch(r"[0-9a-f]{64}", self.sha256)
            or not isinstance(self.md5, str)
            or not re.fullmatch(r"[0-9a-f]{32}", self.md5)
            or self.mime_type not in MIME_FORMAT
            or any(
                type(value) is not int
                for value in (self.byte_length, self.width, self.height, self.frames)
            )
            or not 1 <= self.byte_length <= MAX_BYTES
            or not 1 <= self.width <= MAX_SIDE
            or not 1 <= self.height <= MAX_SIDE
            or self.width * self.height > MAX_PIXELS
            or not 1 <= self.frames <= MAX_FRAMES
            or self.width * self.height * self.frames > MAX_TOTAL_PIXELS
        ):
            raise MediaBlocked("media_metadata_rejected")
        expected_format, extension = MIME_FORMAT[self.mime_type]
        if (
            self.image_format != expected_format
            or self.asset_name != self.sha256 + extension
        ):
            raise MediaBlocked("media_metadata_rejected")
        return self

    def canonical_json(self):
        self.validate()
        return json.dumps(
            asdict(self), ensure_ascii=True, sort_keys=True, separators=(",", ":")
        )

    @classmethod
    def from_json(cls, value):
        if not isinstance(value, str) or len(value) > 4096:
            raise MediaBlocked("media_metadata_rejected")
        try:
            asset = cls(**json.loads(value))
            if asset.canonical_json() != value:
                raise MediaBlocked("media_metadata_not_canonical")
        except (TypeError, ValueError, OverflowError):
            raise MediaBlocked("media_metadata_rejected") from None
        return asset


def inspect_image(raw, mime, *, deadline, clock=time.monotonic):
    """Decode bounded image frames; no conversion, animation or hash claims lost."""
    _deadline(deadline, clock)
    if (
        not isinstance(raw, bytes)
        or not 1 <= len(raw) <= MAX_BYTES
        or mime not in MIME_FORMAT
    ):
        raise MediaBlocked("media_bytes_or_type_rejected")
    expected = MIME_FORMAT[mime][0]
    magic = (
        raw.startswith(b"\x89PNG\r\n\x1a\n")
        if expected == "PNG"
        else raw.startswith(b"\xff\xd8\xff")
        if expected == "JPEG"
        else raw.startswith((b"GIF87a", b"GIF89a"))
        if expected == "GIF"
        else raw.startswith(b"RIFF") and raw[8:12] == b"WEBP"
    )
    if not magic:
        raise MediaBlocked("media_magic_mismatch")
    try:
        from PIL import Image

        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(raw)) as image:
                width, height = image.size
                frames = getattr(image, "n_frames", 1)
                if (
                    image.format != expected
                    or not 1 <= width <= MAX_SIDE
                    or not 1 <= height <= MAX_SIDE
                    or width * height > MAX_PIXELS
                    or not 1 <= frames <= MAX_FRAMES
                    or width * height * frames > MAX_TOTAL_PIXELS
                ):
                    raise MediaBlocked("media_dimensions_or_frames_rejected")
                image.verify()
            with Image.open(io.BytesIO(raw)) as image:
                for index in range(frames):
                    _deadline(deadline, clock)
                    image.seek(index)
                    if image.size != (width, height):
                        raise MediaBlocked("media_frame_dimensions_changed")
                    image.load()
    except MediaBlocked:
        raise
    except Exception:
        raise MediaBlocked("media_decode_rejected") from None
    _deadline(deadline, clock)
    return width, height, frames


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, *_args, **_kwargs):
        raise MediaBlocked("media_redirect_rejected")


def fetch_image_bytes(url, *, timeout=FETCH_SECONDS):
    """One direct credentialless GET, no host settings or shell interpolation."""
    validate_asset_url(url)
    if urlsplit(url).netloc == "kiriyuki.com":
        raise MediaBlocked("media_requires_reviewed_local_import")
    if type(timeout) not in (int, float) or not 0 < timeout <= FETCH_SECONDS:
        raise MediaBlocked("media_timeout_rejected")
    if os.name == "nt":
        try:
            completed = subprocess.run(
                [
                    POWERSHELL,
                    "-NoLogo",
                    "-NoProfile",
                    "-NonInteractive",
                    "-Command",
                    _WINDOWS_FETCH,
                ],
                input=json.dumps({"url": url}).encode(),
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                timeout=min(timeout + 1, FETCH_SECONDS + 1),
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                check=False,
            )
            if completed.returncode != 0 or len(completed.stdout) > MAX_BYTES * 2:
                raise MediaBlocked("media_network_failed")
            result = json.loads(completed.stdout)
            if result.get("ready") is not True:
                raise MediaBlocked("media_network_failed")
            raw = base64.b64decode(result["body_base64"], validate=True)
            mime = result["mime"]
        except MediaBlocked:
            raise
        except (OSError, ValueError, KeyError, TypeError, subprocess.TimeoutExpired):
            raise MediaBlocked("media_network_failed") from None
    else:
        try:
            request = Request(
                url,
                headers={
                    "User-Agent": "Mozilla/5.0",
                    "Accept": "image/png,image/jpeg,image/gif,image/webp",
                    "Accept-Encoding": "identity",
                    "Referer": "https://www.fabiaoqing.com/",
                },
            )
            with build_opener(ProxyHandler({}), _NoRedirect()).open(
                request, timeout=timeout
            ) as response:
                if response.status != 200 or response.geturl() != url:
                    raise MediaBlocked("media_response_rejected")
                mime, raw = (
                    response.headers.get_content_type(),
                    response.read(MAX_BYTES + 1),
                )
        except MediaBlocked:
            raise
        except (URLError, TimeoutError, OSError):
            raise MediaBlocked("media_network_failed") from None
    if mime not in MIME_FORMAT or not 1 <= len(raw) <= MAX_BYTES:
        raise MediaBlocked("media_bytes_or_type_rejected")
    return mime, raw


def _plain(path, *, directory=False):
    info = path.lstat()
    if stat.S_ISLNK(info.st_mode) or getattr(info, "st_file_attributes", 0) & 0x400:
        raise MediaBlocked("media_cache_redirected")
    if not (stat.S_ISDIR(info.st_mode) if directory else stat.S_ISREG(info.st_mode)):
        raise MediaBlocked("media_cache_type_rejected")
    return info


class MediaAssetCache:
    def __init__(self, root, *, fetch=fetch_image_bytes, clock=time.monotonic):
        configured = Path(root).absolute()
        if not configured.parent.is_dir():
            raise MediaBlocked("media_cache_parent_missing")
        for ancestor in (configured.parent, *configured.parent.parents):
            _plain(ancestor, directory=True)
        if not configured.exists():
            configured.mkdir()
        _plain(configured, directory=True)
        self.root, self.fetch, self.clock = configured, fetch, clock

    def _path(self, asset):
        asset.validate()
        _plain(self.root, directory=True)
        path = self.root / asset.asset_name
        if path.parent != self.root or path.resolve().parent != self.root.resolve():
            raise MediaBlocked("media_cache_redirected")
        return path

    def prepare(self, candidate, *, deadline):
        _deadline(deadline, self.clock)
        source, page = (
            validate_asset_url(candidate.image_url),
            validate_page_url(candidate.page_url),
        )
        if urlsplit(source).netloc == "kiriyuki.com":
            # Provenance permission is not permission for runtime HTTP fetch.
            # Authored assets enter only through the reviewed local import.
            raise MediaBlocked("media_requires_reviewed_local_import")
        mime, raw = self.fetch(
            source, timeout=min(FETCH_SECONDS, deadline - self.clock())
        )
        _deadline(deadline, self.clock)
        width, height, frames = inspect_image(
            raw, mime, deadline=deadline, clock=self.clock
        )
        sha = hashlib.sha256(raw).hexdigest()
        asset = MediaAsset(
            sha + MIME_FORMAT[mime][1],
            sha,
            hashlib.md5(raw, usedforsecurity=False).hexdigest(),
            len(raw),
            mime,
            MIME_FORMAT[mime][0],
            width,
            height,
            frames,
            source,
            page,
        )
        path = self._path(asset)
        if path.exists():
            self.load(asset.canonical_json(), deadline=deadline)
        else:
            # Exclusive creation never follows or overwrites a symlink/file.
            try:
                with path.open("xb") as handle:
                    handle.write(raw)
                    handle.flush()
                    os.fsync(handle.fileno())
            except FileExistsError:
                self.load(asset.canonical_json(), deadline=deadline)
            except OSError:
                raise MediaBlocked("media_cache_write_failed") from None
            self.load(asset.canonical_json(), deadline=deadline)
        return asset

    def load(self, media_json, *, deadline):
        _deadline(deadline, self.clock)
        asset = MediaAsset.from_json(media_json)
        path = self._path(asset)
        try:
            before = _plain(path)
            if before.st_size != asset.byte_length or before.st_size > MAX_BYTES:
                raise MediaBlocked("media_cache_size_changed")
            with path.open("rb") as handle:
                opened = os.fstat(handle.fileno())
                raw = handle.read(MAX_BYTES + 1)
            after = _plain(path)
            if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (
                opened.st_dev,
                opened.st_ino,
                opened.st_size,
                opened.st_mtime_ns,
            ) or (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns) != (
                opened.st_dev,
                opened.st_ino,
                opened.st_size,
                opened.st_mtime_ns,
            ):
                raise MediaBlocked("media_cache_file_changed")
        except MediaBlocked:
            raise
        except OSError:
            raise MediaBlocked("media_cache_read_failed") from None
        if (
            hashlib.sha256(raw).hexdigest() != asset.sha256
            or hashlib.md5(raw, usedforsecurity=False).hexdigest() != asset.md5
        ):
            raise MediaBlocked("media_cache_hash_changed")
        if inspect_image(raw, asset.mime_type, deadline=deadline, clock=self.clock) != (
            asset.width,
            asset.height,
            asset.frames,
        ):
            raise MediaBlocked("media_cache_image_changed")
        return asset, path
