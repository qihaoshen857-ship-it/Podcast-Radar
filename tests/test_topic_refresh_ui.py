from __future__ import annotations

from main import (
    PIXEL_DINO_FRAMES,
    TOPIC_FILTER_LABELS,
    progress_marker_position,
    topic_display_label,
    topic_download_title,
    topic_filter_variant,
)


def test_selected_topic_button_uses_black_accent_variant() -> None:
    current_topic = "weather_agri"

    variants = {
        topic: topic_filter_variant(topic, current_topic)
        for topic in TOPIC_FILTER_LABELS
    }

    assert variants["weather_agri"] == "accent"
    assert variants["all"] == "secondary"
    assert variants["ai"] == "secondary"


def test_topic_header_makes_current_filter_explicit() -> None:
    assert topic_download_title("all") == "今日雷达"
    assert topic_download_title("ai") == "今日雷达 · AI"
    assert topic_download_title("weather_agri") == "今日雷达 · 天气"
    assert topic_download_title("science_wellness") == "今日雷达 · 养生"


def test_manual_and_favorite_views_do_not_leave_comprehensive_selected() -> None:
    assert topic_display_label("manual") == "手动链接"
    assert topic_download_title("favorites") == "我的精选"
    assert all(
        topic_filter_variant(topic, "manual") == "secondary"
        for topic in TOPIC_FILTER_LABELS
    )
    assert all(
        topic_filter_variant(topic, "favorites") == "secondary"
        for topic in TOPIC_FILTER_LABELS
    )


def test_dino_progress_marker_stays_inside_rounded_track() -> None:
    assert progress_marker_position(300, -10, 100, 20) == 4
    assert progress_marker_position(300, 0, 100, 20) == 4
    assert progress_marker_position(300, 50, 100, 20) == 140
    assert progress_marker_position(300, 100, 100, 20) == 276
    assert progress_marker_position(300, 120, 100, 20) == 276


def test_pixel_dinosaur_has_two_distinct_running_frames() -> None:
    assert len(PIXEL_DINO_FRAMES) == 2
    assert PIXEL_DINO_FRAMES[0] != PIXEL_DINO_FRAMES[1]
    assert all(any("#" in row for row in frame) for frame in PIXEL_DINO_FRAMES)
