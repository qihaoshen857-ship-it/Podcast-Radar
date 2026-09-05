from __future__ import annotations

from main import (
    VideoItem,
    YTAudioDownloaderApp,
    ellipsize_wrapped_text,
    strip_repeated_title_prefix,
    text_fits_wrapped_lines,
)


def episode(title: str, description: str = "") -> VideoItem:
    return VideoItem(
        video_id="layout-test",
        title=title,
        duration_text="42:00",
        webpage_url="https://example.com/episode",
        description_text=description,
    )


def app_with_library(records: dict | None = None) -> YTAudioDownloaderApp:
    app = object.__new__(YTAudioDownloaderApp)
    app.research_library = {"items": records or {}}
    return app


def test_rss_summary_drops_repeated_english_title_prefix() -> None:
    title = "How to Build a Profitable AI Startup Without Venture Capital"
    description = f"{title} — A practical conversation about customers and distribution."

    assert strip_repeated_title_prefix(description, title) == (
        "A practical conversation about customers and distribution."
    )


def test_card_summary_marks_exact_title_repeat_as_non_summary() -> None:
    item = episode("The Complete Guide to AI Agents", "The Complete Guide to AI Agents")

    assert YTAudioDownloaderApp.card_summary_text(app_with_library(), item) == "暂无独立简介"


def test_card_keeps_clean_summary_when_it_does_not_repeat_title() -> None:
    item = episode(
        "The Complete Guide to AI Agents",
        "Founders explain how they deploy reliable agents in production.",
    )

    assert YTAudioDownloaderApp.card_summary_text(app_with_library(), item) == (
        "Founders explain how they deploy reliable agents in production."
    )


def test_short_title_is_not_removed_from_the_start_of_a_longer_word() -> None:
    assert strip_repeated_title_prefix("AIrtable workflows for founders", "AI") == (
        "AIrtable workflows for founders"
    )


def test_translated_title_tooltip_preserves_complete_original_title() -> None:
    item = episode("A Very Long Original English Episode Title About AI Infrastructure")
    app = app_with_library(
        {
            item.video_id: {
                "title_zh": "关于人工智能基础设施的长篇访谈",
            }
        }
    )

    assert app.card_display_title(item) == "关于人工智能基础设施的长篇访谈"
    assert app.card_original_title_text(item) == item.title
    assert app.card_title_tooltip_text(item) == (
        "关于人工智能基础设施的长篇访谈\n英文原题："
        "A Very Long Original English Episode Title About AI Infrastructure"
    )


def test_long_english_title_is_ellipsized_within_two_lines() -> None:
    title = (
        "How Founders Can Build and Distribute Useful Artificial Intelligence "
        "Products Without Raising Venture Capital"
    )
    measure = lambda value: len(value) * 10

    rendered = ellipsize_wrapped_text(title, max_width=180, max_lines=2, measure=measure)

    assert rendered.endswith("…")
    assert len(rendered) < len(title)
    assert text_fits_wrapped_lines(rendered, 180, 2, measure)


def test_short_title_is_not_modified() -> None:
    title = "AI Startup Stories"

    assert ellipsize_wrapped_text(title, 300, 2, lambda value: len(value) * 10) == title
