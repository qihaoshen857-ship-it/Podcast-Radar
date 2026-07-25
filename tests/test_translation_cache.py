from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from main import (
    FULL_TRANSLATION_MODEL,
    FULL_TRANSLATION_PROMPT_VERSION,
    YTAudioDownloaderApp,
)


class TranslationCacheTests(unittest.TestCase):
    def test_tts_cleanup_preserves_real_years_versions_and_numbers(self) -> None:
        app = object.__new__(YTAudioDownloaderApp)
        text, truncated = app.prepare_tts_text(
            "# 标题\n2026年收入增长 12%。\n5.6版本保留数字。\n1. 第一条列表",
            0,
        )

        self.assertFalse(truncated)
        self.assertIn("2026年收入增长 12%", text)
        self.assertIn("5.6版本保留数字", text)
        self.assertIn("第一条列表", text)

    def test_translation_cache_requires_matching_source_hash_and_prompt(self) -> None:
        app = object.__new__(YTAudioDownloaderApp)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "episode.txt"
            translation = root / "episode.zh.full.md"
            source.write_text("English source", encoding="utf-8")
            translation.write_text("中文译文", encoding="utf-8")
            source_hash = app.source_sha256(source)
            record = {
                "full_translation_path": str(translation),
                "full_translation_source_path": str(source),
                "full_translation_source_sha256": source_hash,
                "full_translation_model": FULL_TRANSLATION_MODEL,
                "full_translation_prompt_version": FULL_TRANSLATION_PROMPT_VERSION,
            }

            self.assertEqual(
                app.translation_cache_matches(record, source, source_hash),
                translation,
            )
            source.write_text("Changed English source", encoding="utf-8")
            self.assertIsNone(
                app.translation_cache_matches(record, source, app.source_sha256(source))
            )

    def test_tts_record_is_bound_to_exact_source_and_hash(self) -> None:
        app = object.__new__(YTAudioDownloaderApp)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            transcript = root / "episode.txt"
            digest = root / "episode.zh.md"
            audio = root / "episode.wav"
            transcript.write_text("English transcript", encoding="utf-8")
            digest.write_text("中文纪要", encoding="utf-8")
            audio.write_bytes(b"RIFF" + b"\0" * 64)
            app.research_library = {
                "items": {
                    "episode": {
                        "transcript_path": str(transcript),
                        "digest_path": str(digest),
                    }
                }
            }
            app.save_research_library = lambda: None

            app.attach_reader_tts_record(audio, transcript)

            record = app.research_library["items"]["episode"]
            transcript_key = str(transcript.resolve())
            self.assertEqual(record["tts_source_path"], transcript_key)
            self.assertEqual(
                record["tts_source_sha256"],
                app.source_sha256(transcript),
            )
            self.assertIn(transcript_key, record["tts_audio_by_source"])
            self.assertNotIn(str(digest.resolve()), record["tts_audio_by_source"])

    def test_clearing_stale_translation_removes_generated_cache_files(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            translation_root = root / "translations"
            tts_root = root / "tts"
            translation_root.mkdir()
            tts_root.mkdir()
            translation = translation_root / "episode.zh.full.md"
            audio = tts_root / "episode.wav"
            translation.write_text("旧译文", encoding="utf-8")
            audio.write_bytes(b"old audio")
            record = {
                "full_translation_path": str(translation),
                "full_translation_source_sha256": "old",
                "full_translation_tts_audio_path": str(audio),
            }

            with (
                patch("main.RESEARCH_TRANSLATION_DIR", translation_root),
                patch("main.RESEARCH_TTS_DIR", tts_root),
            ):
                YTAudioDownloaderApp.clear_full_translation_cache(record, delete_files=True)

            self.assertFalse(translation.exists())
            self.assertFalse(audio.exists())
            self.assertNotIn("full_translation_path", record)
            self.assertNotIn("full_translation_tts_audio_path", record)

    def test_clearing_record_never_deletes_paths_outside_cache_roots(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            translation_root = root / "translations"
            tts_root = root / "tts"
            translation_root.mkdir()
            tts_root.mkdir()
            external = root / "user-note.md"
            external.write_text("用户文件", encoding="utf-8")
            record = {"full_translation_path": str(external)}

            with (
                patch("main.RESEARCH_TRANSLATION_DIR", translation_root),
                patch("main.RESEARCH_TTS_DIR", tts_root),
            ):
                YTAudioDownloaderApp.clear_full_translation_cache(record, delete_files=True)

            self.assertTrue(external.exists())
            self.assertNotIn("full_translation_path", record)

    def test_translation_selection_never_replaces_full_translation_audio(self) -> None:
        app = object.__new__(YTAudioDownloaderApp)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            translation = root / "episode.zh.full.md"
            selection_audio = root / "selection.wav"
            full_audio = root / "full.wav"
            translation.write_text("中文全文内容", encoding="utf-8")
            selection_audio.write_bytes(b"selection")
            full_audio.write_bytes(b"full")
            app.research_library = {
                "items": {
                    "episode": {
                        "full_translation_path": str(translation),
                    }
                }
            }
            app.save_research_library = lambda: None
            source_hash = app.source_sha256(translation)

            app.attach_reader_tts_record(
                selection_audio,
                translation,
                source_hash,
                "selection-hash",
                "selection",
            )
            record = app.research_library["items"]["episode"]
            self.assertNotIn("full_translation_tts_audio_path", record)
            self.assertEqual(record["tts_scope"], "selection")

            app.attach_reader_tts_record(
                full_audio,
                translation,
                source_hash,
                "full-hash",
                "full_translation",
            )
            self.assertEqual(record["full_translation_tts_audio_path"], str(full_audio))
            self.assertEqual(record["full_translation_tts_scope"], "full_translation")
            self.assertEqual(record["full_translation_tts_text_sha256"], "full-hash")

    def test_changed_source_removes_stale_bound_tts_and_companion_text(self) -> None:
        app = object.__new__(YTAudioDownloaderApp)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            tts_root = root / "tts"
            tts_root.mkdir()
            source = root / "episode.txt"
            audio = tts_root / "episode.wav"
            audio_text = tts_root / "episode.txt"
            source.write_text("new source", encoding="utf-8")
            audio.write_bytes(b"old audio")
            audio_text.write_text("old spoken text", encoding="utf-8")
            source_key = str(source.resolve())
            record = {
                "tts_audio_by_source": {
                    source_key: {
                        "audio_path": str(audio),
                        "source_sha256": "old-hash",
                        "text_sha256": "old-text",
                        "scope": "page",
                    }
                },
                "tts_audio_path": str(audio),
                "tts_source_path": source_key,
                "tts_source_sha256": "old-hash",
                "tts_text_sha256": "old-text",
                "tts_scope": "page",
            }

            with patch("main.RESEARCH_TTS_DIR", tts_root):
                changed = app.clear_stale_tts_cache_for_source(
                    record,
                    source,
                    app.source_sha256(source),
                )

            self.assertTrue(changed)
            self.assertFalse(audio.exists())
            self.assertFalse(audio_text.exists())
            self.assertNotIn("tts_audio_by_source", record)
            self.assertNotIn("tts_audio_path", record)


if __name__ == "__main__":
    unittest.main()
