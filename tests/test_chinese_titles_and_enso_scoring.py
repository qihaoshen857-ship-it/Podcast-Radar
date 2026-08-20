from __future__ import annotations

from main import (
    DEFAULT_WEATHER_AGRI_PODCAST_FEEDS,
    ENSO_WEATHER_BONUS,
    VideoItem,
    YTAudioDownloaderApp,
    investment_value_score,
    preferred_record_title,
)


def weather_item(title: str, description: str = "") -> VideoItem:
    return VideoItem(
        video_id=title,
        title=title,
        duration_text="42:00",
        webpage_url="https://example.com/weather",
        source_type="podcast_rss",
        source_name=DEFAULT_WEATHER_AGRI_PODCAST_FEEDS[0].name,
        description_text=description,
    )


def test_library_and_reader_prefer_cached_chinese_title() -> None:
    record = {
        "title": "What El Nino Means for Global Crops",
        "title_zh": "厄尔尼诺对全球作物的影响",
    }

    assert preferred_record_title(record) == "厄尔尼诺对全球作物的影响"


def test_title_falls_back_to_original_when_translation_is_missing() -> None:
    assert preferred_record_title({"title": "Weather Outlook"}) == "Weather Outlook"
    assert preferred_record_title(
        {"title": "Weather Outlook", "title_zh": "Weather Outlook"}
    ) == "Weather Outlook"
    assert preferred_record_title({}, "Local transcript") == "Local transcript"


def test_result_index_keeps_library_snapshot_and_uses_chinese_title(tmp_path) -> None:
    transcript = tmp_path / "english-title.txt"
    transcript.write_text("transcript", encoding="utf-8")
    app = object.__new__(YTAudioDownloaderApp)

    records = YTAudioDownloaderApp.collect_result_records(
        app,
        library_records=[
            {
                "title": "English title",
                "title_zh": "已汉化的节目标题",
                "transcript_path": str(transcript),
            }
        ],
        transcription_paths=[],
        download_dir_text=str(tmp_path / "missing"),
    )

    indexed = next(record for record in records if record["path"] == str(transcript))
    assert indexed["title"] == "已汉化的节目标题"


def test_enso_episode_receives_explicit_weather_priority_bonus() -> None:
    enso_score, enso_reasons = investment_value_score(
        weather_item("El Niño is returning: the global crop outlook")
    )
    ordinary_score, ordinary_reasons = investment_value_score(
        weather_item("Drought and rainfall: the global crop outlook")
    )

    assert ENSO_WEATHER_BONUS >= 2.0
    assert enso_score > ordinary_score
    assert "厄尔尼诺/ENSO 重点" in enso_reasons
    assert "厄尔尼诺/ENSO 重点" not in ordinary_reasons


def test_chinese_enso_terms_receive_the_same_priority() -> None:
    score, reasons = investment_value_score(
        weather_item("厄尔尼诺与拉尼娜转换对农产品的影响")
    )

    assert score >= 8.0
    assert "厄尔尼诺/ENSO 重点" in reasons
