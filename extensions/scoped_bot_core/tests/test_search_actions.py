"""Explicit intent and result formatting tests; no network, chat or model IO."""

import unittest
from dataclasses import FrozenInstanceError

from scoped_bot_core.public_search import (
    PublicResult,
    PublicSearchPage,
    SearchBlocked,
)
from scoped_bot_core.search_actions import (
    SearchIntent,
    meme_candidates,
    parse_search_intent,
    render_web_results,
)


class SearchActionTests(unittest.TestCase):
    def test_only_explicit_queries_become_search_intents(self):
        queries = {
            "帮我搜一下Python官方文档": ("web", "Python官方文档"),
            "麻烦帮我联网查今天香港天气": ("web", "今天香港天气"),
            "请查一下Python怎么安装": ("web", "Python怎么安装"),
            "你帮我搜一下Python官方文档": ("web", "Python官方文档"),
            "能不能帮我搜索一下Python官方文档": ("web", "Python官方文档"),
            "搜下 Python 文档": ("web", "Python 文档"),
            "/search Python official documentation": (
                "web",
                "Python official documentation",
            ),
            "/meme 开心": ("meme", "开心"),
            "帮我找开心表情包": ("meme", "开心"),
            "给我找个开心的表情包": ("meme", "开心"),
            "搜索开心表情包": ("meme", "开心"),
            "发个摸鱼表情包": ("meme", "摸鱼"),
            "来张开心表情包吧": ("meme", "开心"),
            "找表情包：开心": ("meme", "开心"),
        }
        for text, expected in queries.items():
            with self.subTest(text=text):
                intent = parse_search_intent(text)
                self.assertEqual((intent.kind, intent.query), expected)
                self.assertNotIn(expected[1], repr(intent))
        with self.assertRaises(FrozenInstanceError):
            parse_search_intent("搜Python").query = "changed"

    def test_ordinary_chat_help_quotes_and_untrusted_at_tokens_do_not_search(self):
        for text in (
            "我喜欢手机摄影",
            "Python怎么安装？",
            "你能搜索吗？",
            "你会发什么表情包",
            "我想知道怎么搜索",
            "他说帮我搜一下Python",
            "‘帮我搜一下Python’怎么用",
            "@未核验机器人 帮我搜一下Python",
            "如果我说搜Python怎么办",
            "搜索功能好用吗？",
            "你可以搜索吗？",
            "你能联网搜索吗？",
            "x" * 513,
            None,
            1,
        ):
            with self.subTest(text=text):
                self.assertIsNone(parse_search_intent(text))

    def test_missing_secret_or_private_query_fails_locally(self):
        for text in (
            "/search",
            "/meme",
            "搜一下",
            "帮我搜token=PRIVATE",
            "查一下密码是PRIVATE",
            "帮我找wxid_private表情包",
            "搜 someone@example.com",
        ):
            with (
                self.subTest(text=text),
                self.assertRaisesRegex(SearchBlocked, "^public_query_rejected$"),
            ):
                parse_search_intent(text)
        with self.assertRaisesRegex(SearchBlocked, "^search_kind_rejected$"):
            SearchIntent("shell", "whoami")

    def test_source_rendering_is_plain_bounded_metadata_not_model_synthesis(self):
        rows = tuple(
            PublicResult(
                f"Public title {i}", f"https://example.com/{i}", "Public snippet"
            )
            for i in range(5)
        )
        page = PublicSearchPage("bing_cn_rss", "public-hash", rows, 100)
        text = render_web_results(page)
        self.assertIn("1. Public title 0", text)
        self.assertIn("3. Public title 2", text)
        self.assertNotIn("title 3", text)
        self.assertIn("https://example.com/0", text)
        self.assertNotIn("public-hash", text)
        self.assertLess(len(text.encode()), 3500)
        with self.assertRaisesRegex(SearchBlocked, "^search_result_rejected$"):
            render_web_results(
                PublicSearchPage(
                    "bing_cn_rss",
                    "x",
                    (PublicResult("Private", "https://127.0.0.1/"),),
                    100,
                )
            )
        with self.assertRaisesRegex(SearchBlocked, "^search_result_rejected$"):
            render_web_results(page, limit=True)

    def test_meme_candidates_keep_provenance_and_licence_without_download_or_send(self):
        row = PublicResult(
            "开心",
            "https://www.fabiaoqing.com/biaoqing/detail/id/123.html",
            image_url="https://img.soutula.com/bmiddle/public.jpg",
        )
        page = PublicSearchPage("fabiaoqing_html", "public-hash", (row,), 100)
        self.assertEqual(meme_candidates(page), (row,))
        self.assertEqual(meme_candidates(page)[0].license_status, "unknown")
        with self.assertRaisesRegex(SearchBlocked, "^search_result_rejected$"):
            meme_candidates(PublicSearchPage("bing_cn_rss", "x", (row,), 100))


if __name__ == "__main__":
    unittest.main()
