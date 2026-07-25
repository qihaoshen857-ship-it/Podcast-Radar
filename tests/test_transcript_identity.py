from __future__ import annotations

from main import (
    VideoItem,
    YTAudioDownloaderApp,
    cached_transcript_identity_conflict,
    episode_titles_match,
    extract_declared_transcript_title,
    transcript_source_text,
    validate_official_transcript_identity,
)


def make_item() -> VideoItem:
    return VideoItem(
        video_id="rss-current",
        title="How to Get the Most Out of Fable 5 and GPT-5.6 Sol",
        duration_text="26:37",
        webpage_url="https://podcasters.spotify.com/example",
        source_type="podcast_rss",
        source_name="The AI Daily Brief: Artificial Intelligence News and Analysis",
        published_text="2026-07-21",
    )


def test_extracts_declared_transcript_title() -> None:
    text = "# The Self-Driving Company — Transcript (2026-07-19)\n\nBody"
    assert extract_declared_transcript_title(text) == "The Self-Driving Company"


def test_rejects_adjacent_episode_transcript() -> None:
    item = make_item()
    wrong = "# The Self-Driving Company — Transcript (2026-07-19)\n\n" + ("Replit agents. " * 100)
    accepted, reason = validate_official_transcript_identity(item, wrong)
    assert accepted is False
    assert "The Self-Driving Company" in reason


def test_accepts_same_episode_transcript() -> None:
    item = make_item()
    correct = "# How to Get the Most Out of Fable 5 and GPT-5.6 Sol — Transcript (2026-07-21)\n\n" + (
        "Prompting frontier models. " * 100
    )
    accepted, reason = validate_official_transcript_identity(item, correct)
    assert accepted is True
    assert "标题匹配" in reason


def test_title_match_tolerates_small_punctuation_changes() -> None:
    matched, score = episode_titles_match(
        "How to Get the Most Out of Fable 5 and GPT-5.6 Sol",
        "How to Get the Most Out of Fable 5 & GPT 5.6 Sol",
    )
    assert matched is True
    assert score >= 0.72


def test_official_candidates_do_not_probe_two_days_away() -> None:
    app = object.__new__(YTAudioDownloaderApp)
    candidates = app.official_transcript_url_candidates(make_item())
    assert any("2026-07-21" in url for url in candidates)
    assert any("2026-07-20" in url for url in candidates)
    assert any("2026-07-22" in url for url in candidates)
    assert all("2026-07-19" not in url for url in candidates)
    assert all("2026-07-23" not in url for url in candidates)


def test_cached_wrong_episode_is_flagged(tmp_path) -> None:
    path = tmp_path / "current.txt"
    path.write_text(
        "# The Self-Driving Company — Transcript (2026-07-19)\n\n" + ("Replit agents. " * 100),
        encoding="utf-8",
    )

    conflict, reason = cached_transcript_identity_conflict(make_item(), path)

    assert conflict is True
    assert "The Self-Driving Company" in reason


def test_audio_asr_cache_without_official_heading_is_accepted(tmp_path) -> None:
    path = tmp_path / "current.txt"
    path.write_text("Welcome to this episode about Fable 5 and GPT-5.6 Sol.\n", encoding="utf-8")

    conflict, reason = cached_transcript_identity_conflict(make_item(), path)

    assert conflict is False
    assert reason == ""


def test_transcript_source_labels_distinguish_asr_and_brief() -> None:
    assert "ASR" in transcript_source_text({"transcript_origin": "audio_asr"})
    assert "本地 Whisper" in transcript_source_text(
        {"transcript_origin": "audio_asr", "transcript_engine": "local_whisper"}
    )
    assert "阿里云 ASR 补齐" in transcript_source_text(
        {
            "transcript_origin": "audio_asr",
            "transcript_engine": "local_whisper_with_aliyun_fallback",
        }
    )
    assert "非完整转录" in transcript_source_text({"transcript_origin": "episode_brief_fallback"})
