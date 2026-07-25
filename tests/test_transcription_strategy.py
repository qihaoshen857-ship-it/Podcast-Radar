from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Event
from unittest.mock import patch

from app.transcription import (
    TranscriptionError,
    TranscriptionSettings,
    effective_asr_workers,
    merge_segment_texts,
    transcribe_segments,
    transcribe_segments_cloud,
    transcribe_segments_local_first,
    write_segment_checkpoint,
)


def make_settings(
    api_key: str = "",
    *,
    strategy: str | None = None,
    workers: int = 4,
) -> TranscriptionSettings:
    values = {
        "api_key": api_key,
        "target_segment_seconds": 120,
        "max_segment_seconds": 180,
        "workers": workers,
    }
    if strategy is not None:
        values["strategy"] = strategy
    return TranscriptionSettings(**values)


def test_local_transcription_does_not_require_api_key() -> None:
    settings = make_settings("")
    settings.validate()
    assert settings.strategy == "local_first"


def test_default_strategy_runs_local_whisper_before_cloud() -> None:
    reports: list[str] = []
    chunks = [Path("chunk.wav")]
    with (
        patch("app.transcription.transcribe_segments_local", return_value=["local text"]) as local,
        patch("app.transcription.transcribe_segments_cloud") as cloud,
    ):
        texts, backend = transcribe_segments(chunks, make_settings("cloud-key"), reports.append)

    assert texts == ["local text"]
    assert backend == "local_whisper"
    local.assert_called_once()
    cloud.assert_not_called()
    assert reports[0].startswith("准备使用本地 faster-whisper")


def test_cloud_fast_with_key_goes_directly_to_cloud() -> None:
    reports: list[str] = []
    chunks = [Path("chunk_000.wav"), Path("chunk_001.wav")]
    settings = make_settings("cloud-key", strategy="cloud_fast")
    with (
        patch("app.transcription.transcribe_segments_local") as local,
        patch(
            "app.transcription.transcribe_segments_cloud",
            return_value=["cloud first", "cloud second"],
        ) as cloud,
    ):
        texts, backend = transcribe_segments(chunks, settings, reports.append)

    assert texts == ["cloud first", "cloud second"]
    assert backend == "aliyun_cloud_fast"
    local.assert_not_called()
    cloud.assert_called_once()
    assert any("极速云端" in message for message in reports)


def test_cloud_fast_without_key_reports_and_falls_back_to_local() -> None:
    reports: list[str] = []
    chunks = [Path("chunk.wav")]
    settings = make_settings("", strategy="cloud_fast")
    with (
        patch("app.transcription.transcribe_segments_local", return_value=["local text"]) as local,
        patch("app.transcription.transcribe_segments_cloud") as cloud,
    ):
        texts, backend = transcribe_segments(chunks, settings, reports.append)

    assert texts == ["local text"]
    assert backend == "local_whisper"
    local.assert_called_once()
    cloud.assert_not_called()
    assert any(
        "未配置 DASHSCOPE_API_KEY" in message and "本地" in message
        for message in reports
    )


def test_cloud_fast_failure_uses_same_checkpoint_for_local_completion() -> None:
    reports: list[str] = []
    chunks = [Path("chunk_000.wav"), Path("chunk_001.wav")]
    settings = make_settings("cloud-key", strategy="cloud_fast")
    with TemporaryDirectory() as temp_dir:
        checkpoint_dir = Path(temp_dir) / "episode.segments"
        with (
            patch(
                "app.transcription.transcribe_segments_cloud",
                side_effect=TranscriptionError("one cloud chunk failed"),
            ) as cloud,
            patch(
                "app.transcription.transcribe_segments_local",
                return_value=["cached cloud text", "local completion"],
            ) as local,
        ):
            texts, backend = transcribe_segments(
                chunks,
                settings,
                reports.append,
                checkpoint_dir,
            )

    assert texts == ["cached cloud text", "local completion"]
    assert backend == "aliyun_cloud_fast_with_local_fallback"
    assert cloud.call_args.args[3] == checkpoint_dir
    assert local.call_args.args[2] == checkpoint_dir
    assert any("本地 Whisper 补齐" in message for message in reports)


def test_cloud_is_used_only_after_local_failure() -> None:
    reports: list[str] = []
    chunks = [Path("chunk.wav")]
    with (
        patch(
            "app.transcription.transcribe_segments_local",
            side_effect=TranscriptionError("local unavailable"),
        ) as local,
        patch("app.transcription.transcribe_segments_cloud", return_value=["cloud text"]) as cloud,
    ):
        texts, backend = transcribe_segments_local_first(chunks, make_settings("cloud-key"), reports.append)

    assert texts == ["cloud text"]
    assert backend == "local_whisper_with_aliyun_fallback"
    local.assert_called_once()
    cloud.assert_called_once()
    assert any("本地 Whisper 失败" in message for message in reports)
    assert any("开始阿里云 qwen3-asr-flash 备用" in message for message in reports)


def test_local_failure_without_key_does_not_call_cloud() -> None:
    chunks = [Path("chunk.wav")]
    with (
        patch(
            "app.transcription.transcribe_segments_local",
            side_effect=TranscriptionError("local unavailable"),
        ),
        patch("app.transcription.transcribe_segments_cloud") as cloud,
    ):
        try:
            transcribe_segments_local_first(chunks, make_settings(""))
        except TranscriptionError as exc:
            assert "未配置 DASHSCOPE_API_KEY" in str(exc)
        else:
            raise AssertionError("本地失败且没有 Key 时应返回明确错误")

    cloud.assert_not_called()


def test_cloud_worker_count_uses_requested_value_with_safe_upper_bound() -> None:
    assert effective_asr_workers(make_settings("key", workers=1)) == 1
    assert effective_asr_workers(make_settings("key", workers=4)) == 4
    assert effective_asr_workers(make_settings("key", workers=8)) == 8
    assert effective_asr_workers(make_settings("key", workers=16)) == 8


def test_segment_merge_removes_audio_overlap_without_reordering() -> None:
    texts = [
        "We need faster transcription for every podcast.",
        "For every podcast, the next point is accuracy.",
        "The next point is accuracy. Then we can publish.",
    ]

    assert merge_segment_texts(texts) == (
        "We need faster transcription for every podcast.\n"
        "the next point is accuracy.\n"
        "Then we can publish."
    )


def test_segment_merge_keeps_non_overlapping_text() -> None:
    assert merge_segment_texts(["first chunk", "second chunk"]) == "first chunk\nsecond chunk"


def test_cloud_results_keep_source_order_and_resume_from_segment_cache() -> None:
    chunks = [
        Path("chunk_000.wav"),
        Path("chunk_001.wav"),
        Path("chunk_002.wav"),
    ]
    settings = make_settings("cloud-key", strategy="cloud_fast", workers=4)
    release_first_chunk = Event()

    def transcribe_out_of_order(
        chunk_path: Path,
        api_key: str,
        segment_index: int,
        segment_total: int,
        _report,
    ) -> str:
        assert api_key == "cloud-key"
        assert segment_total == 3
        if segment_index == 1:
            assert release_first_chunk.wait(timeout=2)
        elif segment_index == 3:
            release_first_chunk.set()
        return f"cloud {chunk_path.stem[-3:]}"

    with TemporaryDirectory() as temp_dir:
        checkpoint_dir = Path(temp_dir) / "episode.segments"
        write_segment_checkpoint(checkpoint_dir, 1, "cached 001")

        with patch(
            "app.transcription.transcribe_segment",
            side_effect=transcribe_out_of_order,
        ) as transcribe:
            texts = transcribe_segments_cloud(
                chunks,
                settings,
                checkpoint_dir=checkpoint_dir,
            )

        assert texts == ["cloud 000", "cached 001", "cloud 002"]
        assert transcribe.call_count == 2
        called_segment_numbers = {
            call.args[2]
            for call in transcribe.call_args_list
        }
        assert called_segment_numbers == {1, 3}
        assert (checkpoint_dir / "segment_000.txt").read_text(encoding="utf-8").strip() == "cloud 000"
        assert (checkpoint_dir / "segment_002.txt").read_text(encoding="utf-8").strip() == "cloud 002"

        with patch("app.transcription.transcribe_segment") as transcribe_again:
            resumed = transcribe_segments_cloud(
                chunks,
                settings,
                checkpoint_dir=checkpoint_dir,
            )

        assert resumed == texts
        transcribe_again.assert_not_called()
