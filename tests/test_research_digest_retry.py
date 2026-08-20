from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from app.research_digest import (
    DigestMetadata,
    ResearchDigestError,
    build_digest_prompt,
    build_chinese_research_digest,
)


def successful_response(text: str) -> SimpleNamespace:
    return SimpleNamespace(
        status_code=200,
        output={"choices": [{"message": {"content": text}}]},
    )


class ResearchDigestRetryTests(unittest.TestCase):
    def test_brief_fallback_prompt_forbids_external_fact_completion(self) -> None:
        prompt = build_digest_prompt(
            "SOURCE_KIND: EPISODE_BRIEF_FALLBACK\n\nOriginal episode description:\nA short description.",
            DigestMetadata(title="Episode"),
        )

        self.assertIn("只能使用 Original episode description 和元数据逐字提供的信息", prompt)
        self.assertIn("广告主的宣传语不能扩写成节目论据", prompt)
        self.assertIn("待获取完整音频或官方逐字稿", prompt)
        self.assertIn("不得编造媒体报道、数据库或未来报告", prompt)

    def test_timeout_retries_once_and_reports_visible_states(self) -> None:
        states: list[tuple[str, int, int]] = []
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_path = root / "episode.txt"
            output_path = root / "episode.zh.md"
            source_path.write_text("A complete source transcript.", encoding="utf-8")

            with (
                patch(
                    "app.research_digest.dashscope.Generation.call",
                    side_effect=[TimeoutError("request timed out"), successful_response("# 中文纪要")],
                ) as generation_call,
                patch("app.research_digest.time.sleep") as sleep_call,
            ):
                result = build_chinese_research_digest(
                    transcript_path=source_path,
                    output_path=output_path,
                    metadata=DigestMetadata(title="Episode"),
                    api_key="test-key",
                    request_timeout_seconds=37,
                    status_callback=lambda state, attempt, attempts: states.append(
                        (state, attempt, attempts)
                    ),
                )

            self.assertEqual(result, output_path)
            self.assertEqual(generation_call.call_count, 2)
            self.assertEqual(
                [call.kwargs["request_timeout"] for call in generation_call.call_args_list],
                [37, 37],
            )
            self.assertEqual(
                [call.kwargs["max_tokens"] for call in generation_call.call_args_list],
                [8192, 8192],
            )
            self.assertEqual(
                states,
                [
                    ("requesting", 1, 2),
                    ("retrying", 2, 2),
                    ("requesting", 2, 2),
                    ("succeeded", 2, 2),
                ],
            )
            sleep_call.assert_called_once_with(1.0)
            self.assertEqual(output_path.read_text(encoding="utf-8"), "# 中文纪要\n")

    def test_retryable_server_error_succeeds_on_second_attempt(self) -> None:
        server_error = SimpleNamespace(
            status_code=503,
            code="ServiceUnavailable",
            message="temporarily unavailable",
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_path = root / "episode.txt"
            output_path = root / "episode.zh.md"
            source_path.write_text("Source", encoding="utf-8")

            with (
                patch(
                    "app.research_digest.dashscope.Generation.call",
                    side_effect=[server_error, successful_response("整理成功")],
                ) as generation_call,
                patch("app.research_digest.time.sleep"),
            ):
                build_chinese_research_digest(
                    source_path,
                    output_path,
                    DigestMetadata(title="Episode"),
                    "test-key",
                )

            self.assertEqual(generation_call.call_count, 2)
            self.assertEqual(output_path.read_text(encoding="utf-8"), "整理成功\n")

    def test_authentication_error_does_not_retry_or_overwrite_cache(self) -> None:
        auth_error = SimpleNamespace(
            status_code=401,
            code="InvalidApiKey",
            message="invalid api key",
        )
        states: list[tuple[str, int, int]] = []
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_path = root / "episode.txt"
            output_path = root / "episode.zh.md"
            source_path.write_text("Source", encoding="utf-8")
            output_path.write_text("旧缓存\n", encoding="utf-8")

            with patch(
                "app.research_digest.dashscope.Generation.call",
                return_value=auth_error,
            ) as generation_call:
                with self.assertRaisesRegex(ResearchDigestError, "InvalidApiKey"):
                    build_chinese_research_digest(
                        source_path,
                        output_path,
                        DigestMetadata(title="Episode"),
                        "test-key",
                        status_callback=lambda state, attempt, attempts: states.append(
                            (state, attempt, attempts)
                        ),
                    )

            generation_call.assert_called_once()
            self.assertEqual(states, [("requesting", 1, 2), ("failed", 1, 2)])
            self.assertEqual(output_path.read_text(encoding="utf-8"), "旧缓存\n")

    def test_truncated_response_does_not_overwrite_existing_digest(self) -> None:
        truncated = successful_response("# 未完成的中文纪要")
        truncated.output["choices"][0]["finish_reason"] = "length"
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_path = root / "episode.txt"
            output_path = root / "episode.zh.md"
            source_path.write_text("A long complete transcript.", encoding="utf-8")
            output_path.write_text("旧稿\n", encoding="utf-8")

            with patch(
                "app.research_digest.dashscope.Generation.call",
                return_value=truncated,
            ):
                with self.assertRaisesRegex(ResearchDigestError, "输出上限"):
                    build_chinese_research_digest(
                        source_path,
                        output_path,
                        DigestMetadata(title="Episode"),
                        "test-key",
                    )

            self.assertEqual(output_path.read_text(encoding="utf-8"), "旧稿\n")


if __name__ == "__main__":
    unittest.main()
