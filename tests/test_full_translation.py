from __future__ import annotations

import re
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from app.research_digest import (
    FullTranslationMetadata,
    ResearchDigestError,
    build_chinese_full_translation,
    split_translation_chunks,
)


def successful_response(text: str) -> SimpleNamespace:
    return SimpleNamespace(
        status_code=200,
        output={
            "choices": [
                {
                    "message": {
                        "content": text,
                    }
                }
            ]
        },
    )


def normalized_text(text: str) -> str:
    return re.sub(r"\s+", "", text)


class FullTranslationTests(unittest.TestCase):
    def test_long_source_is_split_without_losing_text(self) -> None:
        paragraphs = [
            f"Paragraph {index}: " + (f"token-{index} alpha beta gamma. " * 70)
            for index in range(1, 8)
        ]
        source_text = "\r\n\r\n".join(paragraphs)

        chunks = split_translation_chunks(source_text, max_chars=1200)

        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(0 < len(chunk) <= 1200 for chunk in chunks))
        self.assertEqual(
            normalized_text(source_text),
            normalized_text("".join(chunks)),
        )

    def test_builder_calls_dashscope_for_every_chunk_reports_progress_and_replaces_atomically(
        self,
    ) -> None:
        source_text = "\n\n".join(
            f"Section {index}. " + (f"source-{index} evidence and detail. " * 170)
            for index in range(1, 7)
        )
        expected_chunks = split_translation_chunks(source_text)
        translated_parts = [
            f"译文第 {index} 段，保留数字 {index}。"
            for index in range(1, len(expected_chunks) + 1)
        ]
        progress: list[tuple[int, int]] = []

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_path = root / "episode.txt"
            output_path = root / "episode.zh.full.md"
            source_path.write_text(source_text, encoding="utf-8")
            original_replace = Path.replace

            with (
                patch(
                    "app.research_digest.dashscope.Generation.call",
                    side_effect=[successful_response(text) for text in translated_parts],
                ) as generation_call,
                patch.object(
                    Path,
                    "replace",
                    autospec=True,
                    side_effect=lambda path, target: original_replace(path, target),
                ) as replace_call,
            ):
                result = build_chinese_full_translation(
                    source_path=source_path,
                    output_path=output_path,
                    metadata=FullTranslationMetadata(
                        title="An English Episode",
                        source_name="Example Podcast",
                        published_text="2026-07-25",
                        webpage_url="https://example.com/episode",
                    ),
                    api_key="test-key",
                    progress_callback=lambda done, total: progress.append((done, total)),
                )

            self.assertEqual(result, output_path)
            self.assertEqual(generation_call.call_count, len(expected_chunks))
            self.assertEqual(
                progress,
                [
                    (index, len(expected_chunks))
                    for index in range(1, len(expected_chunks) + 1)
                ],
            )

            for index, (call, source_chunk) in enumerate(
                zip(generation_call.call_args_list, expected_chunks, strict=True),
                start=1,
            ):
                user_prompt = call.kwargs["messages"][1]["content"]
                self.assertIn(source_chunk, user_prompt)
                self.assertIn(f"第 {index}/{len(expected_chunks)} 段", user_prompt)

            replace_call.assert_called_once()
            temporary_path, replace_target = replace_call.call_args.args
            self.assertEqual(replace_target, output_path)
            self.assertEqual(temporary_path.parent, output_path.parent)
            self.assertTrue(temporary_path.name.startswith(output_path.name + "."))
            self.assertTrue(temporary_path.name.endswith(".tmp"))
            self.assertTrue(output_path.exists())
            self.assertFalse(temporary_path.exists())
            rendered = output_path.read_text(encoding="utf-8")
            self.assertIn("# An English Episode", rendered)
            self.assertIn("> 来源：Example Podcast", rendered)
            for translated in translated_parts:
                self.assertIn(translated, rendered)

    def test_failed_later_chunk_does_not_overwrite_existing_cache(self) -> None:
        source_text = "\n\n".join(
            [
                "First section. " + ("alpha beta gamma. " * 400),
                "Second section. " + ("delta epsilon zeta. " * 400),
            ]
        )
        self.assertGreater(len(split_translation_chunks(source_text)), 1)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_path = root / "episode.txt"
            output_path = root / "episode.zh.full.md"
            source_path.write_text(source_text, encoding="utf-8")
            output_path.write_text("原有完整缓存\n", encoding="utf-8")
            failed_response = SimpleNamespace(status_code=500, output={})

            with patch(
                "app.research_digest.dashscope.Generation.call",
                side_effect=[successful_response("第一段译文"), failed_response],
            ):
                with self.assertRaisesRegex(ResearchDigestError, r"第 2/\d+ 段"):
                    build_chinese_full_translation(
                        source_path=source_path,
                        output_path=output_path,
                        metadata=FullTranslationMetadata(title="Failure case"),
                        api_key="test-key",
                    )

            self.assertEqual(
                output_path.read_text(encoding="utf-8"),
                "原有完整缓存\n",
            )
            self.assertFalse(list(root.glob(output_path.name + ".*.tmp")))

    def test_brief_fallback_translation_is_clearly_labeled(self) -> None:
        source_text = (
            "SOURCE_KIND: EPISODE_BRIEF_FALLBACK\n\n"
            "This is an episode description, not a complete transcript."
        )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_path = root / "episode.brief.txt"
            output_path = root / "episode.zh.full.md"
            source_path.write_text(source_text, encoding="utf-8")

            with patch(
                "app.research_digest.dashscope.Generation.call",
                return_value=successful_response("这是一段节目简介，并非完整转录。"),
            ):
                build_chinese_full_translation(
                    source_path=source_path,
                    output_path=output_path,
                    metadata=FullTranslationMetadata(title="Brief fallback"),
                    api_key="test-key",
                )

            rendered = output_path.read_text(encoding="utf-8")
            self.assertIn(
                "> 当前原文是节目简介兜底稿，并非完整逐字转录。",
                rendered,
            )
            self.assertIn("这是一段节目简介，并非完整转录。", rendered)

    def test_token_limit_does_not_publish_partial_translation(self) -> None:
        truncated_response = successful_response("只有一部分译文")
        truncated_response.output["choices"][0]["finish_reason"] = "length"

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_path = root / "episode.txt"
            output_path = root / "episode.zh.full.md"
            source_path.write_text("A complete English paragraph.", encoding="utf-8")
            output_path.write_text("原有完整缓存\n", encoding="utf-8")

            with patch(
                "app.research_digest.dashscope.Generation.call",
                return_value=truncated_response,
            ):
                with self.assertRaisesRegex(ResearchDigestError, "输出上限"):
                    build_chinese_full_translation(
                        source_path=source_path,
                        output_path=output_path,
                        metadata=FullTranslationMetadata(title="Token limit"),
                        api_key="test-key",
                    )

            self.assertEqual(output_path.read_text(encoding="utf-8"), "原有完整缓存\n")


if __name__ == "__main__":
    unittest.main()
