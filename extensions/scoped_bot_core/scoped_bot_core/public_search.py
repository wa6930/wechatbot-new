"""Independent, read-only public search; no bot imports or import-time IO.

Only the explicit query is transmitted to a fixed public HTML endpoint. Results
are untrusted metadata, not instructions, downloaded assets, or proof of rights.
No result URL is fetched here. HTML providers have no API stability guarantee;
blocking/markup changes fail visibly instead of inventing results or retrying.
"""

from __future__ import annotations

import base64
import hashlib
import ipaddress
import json
import os
import re
import subprocess
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass, replace
from html.parser import HTMLParser
from urllib.error import HTTPError, URLError
from urllib.parse import parse_qs, quote, urlencode, urljoin, urlsplit
from urllib.request import HTTPRedirectHandler, ProxyHandler, Request, build_opener

MAX_HTML = 524_288
MAX_RESULTS = 5
TIMEOUT = 6
WEB_ROOT = "https://html.duckduckgo.com/html/"
BING_ROOT = "https://cn.bing.com/search"
MEME_ROOT = "https://www.fabiaoqing.com"
IS_WINDOWS = os.name == "nt"
WINDOWS_POWERSHELL = r"C:\Windows\System32\WindowsPowerShell\v1.0\powershell.exe"
# Fixed program only: the validated public URL is data on stdin, never shell
# text. Windows' native TLS route works with providers that reject OpenSSL.
# All settings below belong to this one HttpClient/child, not the host system.
_WINDOWS_HTTP = r"""
$ErrorActionPreference='Stop'
$client=$null; $handler=$null; $response=$null; $stream=$null; $memory=$null; $cts=$null
try {
  Add-Type -AssemblyName System.Net.Http
  $inputObject=([Console]::In.ReadToEnd() | ConvertFrom-Json)
  $uri=[Uri]$inputObject.url
  $handler=New-Object System.Net.Http.HttpClientHandler
  $handler.AllowAutoRedirect=$false; $handler.UseCookies=$false; $handler.UseProxy=$false
  $handler.UseDefaultCredentials=$false; $handler.DefaultProxyCredentials=$null
  $client=New-Object System.Net.Http.HttpClient($handler)
  $client.Timeout=[TimeSpan]::FromSeconds(6)
  $client.DefaultRequestHeaders.UserAgent.ParseAdd('Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/131.0 Safari/537.36')
  $client.DefaultRequestHeaders.Accept.ParseAdd('text/html, application/rss+xml, application/xml, text/xml')
  $cts=New-Object System.Threading.CancellationTokenSource
  $cts.CancelAfter(6000)
  $response=$client.GetAsync($uri,[System.Net.Http.HttpCompletionOption]::ResponseHeadersRead,$cts.Token).GetAwaiter().GetResult()
  if ([int]$response.StatusCode -ne 200) { throw 'status_rejected' }
  $types=if ($uri.Host -eq 'cn.bing.com') { @('application/rss+xml','application/xml','text/xml','text/rss+xml') } else { @('text/html','application/xhtml+xml') }
  if ($response.Content.Headers.ContentType.MediaType -notin $types) { throw 'type_rejected' }
  if ($response.RequestMessage.RequestUri.Host -ne $uri.Host) { throw 'origin_rejected' }
  if ($response.Content.Headers.ContentLength -gt 524288) { throw 'size_rejected' }
  $stream=$response.Content.ReadAsStreamAsync().GetAwaiter().GetResult()
  $memory=New-Object System.IO.MemoryStream
  $buffer=New-Object byte[] 32768
  do {
    $wanted=[Math]::Min($buffer.Length,524289-[int]$memory.Length)
    $count=$stream.ReadAsync($buffer,0,$wanted,$cts.Token).GetAwaiter().GetResult()
    if ($count -gt 0) { $memory.Write($buffer,0,$count) }
    if ($memory.Length -gt 524288) { throw 'size_rejected' }
  } while ($count -gt 0)
  [Console]::Out.Write((@{ready=$true;body_base64=[Convert]::ToBase64String($memory.ToArray())}|ConvertTo-Json -Compress))
} catch {
  [Console]::Out.Write('{"ready":false,"reason":"windows_http_failure"}')
  exit 1
} finally {
  foreach ($resource in @($stream,$memory,$response,$client,$handler,$cts)) { if ($resource) { $resource.Dispose() } }
}
"""


class SearchBlocked(RuntimeError):
    """A fixed reason code only; never upstream response bodies or query logs."""


@dataclass(frozen=True, slots=True)
class PublicResult:
    title: str
    page_url: str
    snippet: str = ""
    image_url: str | None = None
    license_status: str = "unknown"


@dataclass(frozen=True, slots=True)
class PublicSearchPage:
    provider: str
    query_sha256: str
    results: tuple[PublicResult, ...]
    fetched_at: float


def _sensitive_query(value):
    # Refuse concrete values, not the mere topic of keys, tokens or contacts.
    # This is a local guard on an explicit query, never permission to serialize
    # a message, profile, history or secret-bearing configuration into search.
    question = (
        r"(?:什么|怎么|如何|哪里|哪种|怎样|是否|啥|多少|能否|可以|需要|有没有|从哪)"
    )
    assignment = r"\s*(?:是|为|[=:：]|\bis\b)\s*(?!" + question + r")[^\s，。；;]{3,}"
    return bool(
        re.search(
            r"wxid_|@chatroom|(?:sk-|ghp_|github_pat_|xox[baprs]-)[A-Za-z0-9_-]{8,}|"
            r"(?:api[ _-]?key|(?:access[ _-]?|refresh[ _-]?)?token|password|passwd|secret|"
            r"authorization|密码|口令|密钥|秘钥|验证码|令牌)" + assignment,
            value,
            re.I,
        )
        or re.search(
            r"(?:微信(?:号|ID)|联系人(?:ID|编号)|用户ID|account[ _-]?id|user[ _-]?id|QQ号)"
            + assignment,
            value,
            re.I,
        )
        or re.search(
            r"[\w.!#$%&'*+/=?^_`{|}~-]+\s*@\s*(?:[\w-]+\.)+[\w-]{2,}|"
            r"(?<!\d)1[3-9]\d{9}(?!\d)|"
            r"(?<![\w+])\+\d(?:[ ()-]*\d){7,14}(?!\d)|"
            r"(?:手机(?:号|号码)?|电话|联系方式)\s*(?:是|为|[=:：])?\s*\+?\d[\d ()-]{4,}\d",
            value,
            re.I,
        )
    )


def public_query(value):
    if (
        not isinstance(value, str)
        or not 1 <= len(value.strip()) <= 80
        or any(ord(char) < 32 for char in value)
        or _sensitive_query(value)
    ):
        raise SearchBlocked("public_query_rejected")
    value = " ".join(value.split())
    try:
        if len(value.encode("utf-8")) > 240:
            raise SearchBlocked("public_query_rejected")
    except UnicodeError:
        raise SearchBlocked("public_query_rejected") from None
    return value


def _text(value, limit):
    return " ".join("".join(c for c in value if ord(c) >= 32).split())[:limit]


def public_url(value):
    if not isinstance(value, str) or len(value) > 2048 or "\\" in value:
        return None
    try:
        parts = urlsplit(value)
        host = parts.hostname.rstrip(".") if parts.hostname else None
        if (
            parts.scheme != "https"
            or not host
            or parts.username is not None
            or parts.password is not None
            or parts.port not in (None, 443)
            or any(ord(c) <= 32 for c in value)
            or "." not in host
            or host.endswith((".local", ".localhost", ".internal"))
        ):
            return None
        try:
            address = ipaddress.ip_address(host)
        except ValueError:
            if not re.fullmatch(r"[A-Za-z0-9.-]+", host) or re.fullmatch(
                r"[0-9.]+", host
            ):
                return None
        else:
            if not address.is_global:
                return None
    except ValueError:
        return None
    return value


def _web_url(value):
    if not isinstance(value, str) or not value:
        return None
    try:
        absolute = urljoin("https://duckduckgo.com", value)
        parts = urlsplit(absolute)
    except ValueError:
        return None
    if parts.hostname == "duckduckgo.com" and parts.path == "/l/":
        values = parse_qs(parts.query).get("uddg", [])
        return public_url(values[0]) if len(values) == 1 else None
    return public_url(absolute)


class _WebParser(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.rows, self.active, self.parts, self.url = [], None, [], None

    def handle_starttag(self, tag, attrs):
        if tag == "a":
            attrs = dict(attrs)
            classes = attrs.get("class", "").split()
            self.active = (
                "title"
                if "result__a" in classes
                else "snippet"
                if "result__snippet" in classes
                else None
            )
            self.parts, self.url = [], _web_url(attrs.get("href", ""))

    def handle_data(self, data):
        if self.active and sum(map(len, self.parts)) < 2048:
            self.parts.append(data)

    def handle_endtag(self, tag):
        if tag != "a" or self.active is None:
            return
        text = _text("".join(self.parts), 400)
        if self.active == "title" and self.url and text:
            if self.url not in {row.page_url for row in self.rows}:
                self.rows.append(PublicResult(text[:160], self.url))
        elif (
            self.active == "snippet"
            and self.rows
            and self.url == self.rows[-1].page_url
        ):
            self.rows[-1] = replace(self.rows[-1], snippet=text)
        self.active = None


class _MemeParser(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.rows, self.link = [], None

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == "a":
            href = attrs.get("href", "")
            self.link = (
                MEME_ROOT + href
                if re.fullmatch(r"/biaoqing/detail/id/[0-9]+\.html", href)
                else None
            )
        if tag != "img" or "bqppsearch" not in attrs.get("class", "").split():
            return
        asset = public_url(attrs.get("data-original"))
        if not asset or not self.link:
            return
        parts = urlsplit(asset)
        if (
            parts.hostname != "img.soutula.com"
            or not re.fullmatch(
                r"/bmiddle/[A-Za-z0-9_-]+\.(?:jpg|jpeg|png|gif|webp)", parts.path, re.I
            )
            or parts.query
            or parts.fragment
        ):
            return
        title = _text(attrs.get("title") or attrs.get("alt") or "", 160)
        if title and asset not in {row.image_url for row in self.rows}:
            self.rows.append(PublicResult(title, self.link, image_url=asset))

    def handle_endtag(self, tag):
        if tag == "a":
            self.link = None


def parse_public_html(provider, content):
    if not isinstance(content, str) or len(content.encode("utf-8")) > MAX_HTML:
        raise SearchBlocked("search_page_rejected")
    if provider not in {"duckduckgo_html", "fabiaoqing_html"}:
        raise SearchBlocked("search_provider_rejected")
    parser = _WebParser() if provider == "duckduckgo_html" else _MemeParser()
    parser.feed(content)
    parser.close()
    if not parser.rows:
        raise SearchBlocked("no_results_or_provider_blocked")
    return tuple(parser.rows[:MAX_RESULTS])


def parse_bing_rss(content):
    """Read bounded untrusted RSS metadata, without entities or result fetches."""
    if not isinstance(content, str):
        raise SearchBlocked("search_page_rejected")
    try:
        if len(content.encode("utf-8")) > MAX_HTML:
            raise SearchBlocked("search_page_rejected")
    except UnicodeError:
        raise SearchBlocked("search_page_rejected") from None
    if re.search(r"<!\s*(?:DOCTYPE|ENTITY)\b", content, re.I):
        raise SearchBlocked("search_xml_rejected")
    try:
        root = ET.fromstring(content)
    except ET.ParseError:
        raise SearchBlocked("search_xml_rejected") from None
    if root.tag != "rss" or len(tuple(root.iter())) > 2048:
        raise SearchBlocked("search_xml_rejected")
    rows, seen = [], set()
    for item in root.findall("./channel/item")[:100]:
        url = public_url(item.findtext("link"))
        title = _text(re.sub(r"<[^>]{0,1000}>", " ", item.findtext("title") or ""), 160)
        snippet = _text(
            re.sub(r"<[^>]{0,1000}>", " ", item.findtext("description") or ""), 300
        )
        if url and url not in seen and title:
            rows.append(PublicResult(title, url, snippet))
            seen.add(url)
            if len(rows) == MAX_RESULTS:
                break
    if not rows:
        raise SearchBlocked("no_results_or_provider_blocked")
    return tuple(rows)


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def _fetch_windows_html(url):
    try:
        result = subprocess.run(
            [
                WINDOWS_POWERSHELL,
                "-NoLogo",
                "-NoProfile",
                "-NonInteractive",
                "-Command",
                _WINDOWS_HTTP,
            ],
            input=json.dumps({"url": url}).encode("ascii"),
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            timeout=8,
            check=False,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        if result.returncode or len(result.stdout) > (MAX_HTML * 4 // 3 + 2048):
            raise SearchBlocked("windows_http_failure")
        value = json.loads(result.stdout.decode("utf-8-sig"))
        if (
            not isinstance(value, dict)
            or set(value) != {"ready", "body_base64"}
            or value["ready"] is not True
        ):
            raise SearchBlocked("windows_http_failure")
        raw = base64.b64decode(value["body_base64"], validate=True)
        if len(raw) > MAX_HTML:
            raise SearchBlocked("search_page_too_large")
        return raw.decode("utf-8", errors="strict")
    except (OSError, subprocess.TimeoutExpired, UnicodeError, ValueError, TypeError):
        raise SearchBlocked("windows_http_failure") from None


def fetch_public_html(url):
    """One fixed-origin GET, bounded response, verified TLS, no auth/cookies.

    Socket timeout is not an OS hard deadline; an integrating runner must keep
    its independent request watchdog and never extend a WeChat reply deadline.
    """
    parts = urlsplit(url)
    if not public_url(url) or not (
        (parts.hostname == "html.duckduckgo.com" and parts.path == "/html/")
        or (
            parts.hostname == "cn.bing.com"
            and parts.path == "/search"
            and set(parse_qs(parts.query)) == {"format", "q"}
            and parse_qs(parts.query).get("format") == ["rss"]
            and len(parse_qs(parts.query).get("q", [])) == 1
        )
        or (
            parts.hostname == "www.fabiaoqing.com"
            and re.fullmatch(r"/search/bqb/keyword/[^/]+/type/bq\.html", parts.path)
        )
    ):
        raise SearchBlocked("search_origin_rejected")
    if IS_WINDOWS:
        return _fetch_windows_html(url)
    request = Request(
        url,
        headers={
            "User-Agent": "Mozilla/5.0 (compatible; WeChatPublicSearch/1.0)",
            "Accept": "text/html, application/rss+xml, application/xml, text/xml",
            "Accept-Encoding": "identity",
        },
    )
    try:
        # A per-client empty proxy handler does not modify host environment or
        # settings and avoids forwarding inherited proxy credentials.
        opener = build_opener(ProxyHandler({}), _NoRedirect())
        with opener.open(request, timeout=TIMEOUT) as response:
            allowed_types = (
                {"application/rss+xml", "application/xml", "text/xml", "text/rss+xml"}
                if parts.hostname == "cn.bing.com"
                else {"text/html", "application/xhtml+xml"}
            )
            if (
                response.status != 200
                or response.headers.get_content_type() not in allowed_types
            ):
                raise SearchBlocked("search_response_rejected")
            raw = response.read(MAX_HTML + 1)
            if len(raw) > MAX_HTML:
                raise SearchBlocked("search_page_too_large")
            return raw.decode("utf-8", errors="strict")
    except HTTPError as exc:
        raise SearchBlocked(
            "search_rate_limited" if exc.code == 429 else "search_http_failure"
        ) from None
    except (URLError, TimeoutError, OSError, UnicodeError):
        raise SearchBlocked("search_network_or_encoding_failure") from None


class PublicSearchClient:
    def __init__(
        self, *, fetch=fetch_public_html, clock=time.time, web_provider="bing_cn_rss"
    ):
        if web_provider not in {"bing_cn_rss", "duckduckgo_html"}:
            raise SearchBlocked("search_provider_rejected")
        self.fetch, self.clock = fetch, clock
        self.web_provider = web_provider

    def search(self, query, *, kind="web"):
        query = public_query(query)
        if kind == "web":
            provider = self.web_provider
            url = (
                BING_ROOT + "?" + urlencode({"format": "rss", "q": query})
                if provider == "bing_cn_rss"
                else WEB_ROOT + "?" + urlencode({"q": query})
            )
        elif kind == "meme":
            provider = "fabiaoqing_html"
            url = (
                MEME_ROOT
                + "/search/bqb/keyword/"
                + quote(query, safe="")
                + "/type/bq.html"
            )
        else:
            raise SearchBlocked("search_kind_rejected")
        content = self.fetch(url)
        rows = (
            parse_bing_rss(content)
            if provider == "bing_cn_rss"
            else parse_public_html(provider, content)
        )
        return PublicSearchPage(
            provider, hashlib.sha256(query.encode()).hexdigest(), rows, self.clock()
        )
