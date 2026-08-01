from __future__ import annotations

from main import (
    AI_STARTUP_FEED_NAMES,
    DEFAULT_AI_STARTUP_PODCAST_FEEDS,
    DEFAULT_AI_STARTUP_XIAOYUZHOU_FEEDS,
    PODCAST_FEED_GROUPS,
    PODCAST_GROUP_LABELS,
    XIAOYUZHOU_FEED_GROUPS,
    VideoItem,
    infer_item_topic,
    is_ai_startup_story_item,
    is_relevant_podcast_item,
)


def test_ai_startup_topic_has_dedicated_sources_and_label() -> None:
    assert PODCAST_GROUP_LABELS["ai_startup"] == "AI创业"
    assert len(DEFAULT_AI_STARTUP_PODCAST_FEEDS) == 6
    assert len(DEFAULT_AI_STARTUP_XIAOYUZHOU_FEEDS) == 1
    assert PODCAST_FEED_GROUPS["ai_startup"] == DEFAULT_AI_STARTUP_PODCAST_FEEDS
    assert XIAOYUZHOU_FEED_GROUPS["ai_startup"] == DEFAULT_AI_STARTUP_XIAOYUZHOU_FEEDS
    assert all(feed in PODCAST_FEED_GROUPS["all"] for feed in DEFAULT_AI_STARTUP_PODCAST_FEEDS)
    assert all(feed in XIAOYUZHOU_FEED_GROUPS["all"] for feed in DEFAULT_AI_STARTUP_XIAOYUZHOU_FEEDS)


def test_ai_startup_source_is_classified_before_general_ai_keywords() -> None:
    item = VideoItem(
        video_id="startup-1",
        title="How we built an AI sales agent",
        duration_text="42:00",
        webpage_url="https://example.com/startup-1",
        source_name=DEFAULT_AI_STARTUP_PODCAST_FEEDS[0].name,
    )
    assert item.source_name in AI_STARTUP_FEED_NAMES
    assert infer_item_topic(item) == "ai_startup"


def test_filter_keeps_ai_tool_use_and_founder_execution_stories() -> None:
    assert is_ai_startup_story_item(
        "How I use Claude Code + MCPs to run my marketing",
    )
    assert is_ai_startup_story_item(
        "Why Domain Experts Are Winning In The Age Of AI",
        "YC founders explain how they find customers and build a startup from zero.",
    )
    assert is_ai_startup_story_item(
        "AI 工具如何帮一人公司获得第一批客户",
    )


def test_filter_rejects_pure_ai_industry_news_and_non_ai_business() -> None:
    assert not is_ai_startup_story_item(
        "World Models, Explained",
        "A research overview of the latest AI model architecture.",
    )
    assert not is_ai_startup_story_item(
        "Are OpenAI and Anthropic Overvalued? The Open-Source AI Reality",
        "A market update and valuation debate.",
    )
    assert not is_ai_startup_story_item(
        "Grok 4.5 is a bigger deal than Fable",
        "A discussion of the newest model release and benchmarks.",
        "The Startup Ideas Podcast",
    )
    assert not is_ai_startup_story_item(
        "Leading Anthropic's First Ever Round | Will Open Source Threaten Anthropic's Business",
        "An investor debates margins, valuations and the future of AI labs.",
    )
    assert not is_ai_startup_story_item(
        "How a restaurant founder opened ten new stores",
        "A practical business story without AI tools.",
    )


def test_rss_relevance_gate_uses_description_for_missing_axis() -> None:
    yc_feed = next(feed for feed in DEFAULT_AI_STARTUP_PODCAST_FEEDS if feed.name == "Y Combinator Startup Podcast")
    assert is_relevant_podcast_item(
        yc_feed,
        "Why Domain Experts Are Winning In The Age Of AI",
        "Founders share the product and customer lessons behind their startups.",
    )
    assert not is_relevant_podcast_item(
        yc_feed,
        "World Models, Explained",
        "A technical discussion of model research and benchmarks.",
    )
