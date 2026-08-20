from __future__ import annotations

import json
from unittest.mock import patch

from app.person_monitor_service import (
    archive_artwork_url,
    canonicalize_url,
    discover_elon_archive_interviews,
    discover_itunes_episodes,
    item_dedupe_key,
    load_registry,
    module_resources_dir,
    parse_feed_entries,
    refresh_person,
    source_identity_matches,
    title_identity_matches,
    youtube_thumbnail_url,
)


def test_title_identity_gate_rejects_third_party_title_mention() -> None:
    aliases = ["Elon Musk"]
    assert title_identity_matches("#400 – Elon Musk: SpaceX and AI", aliases) == ["elon musk"]
    assert title_identity_matches("Sam Altman: OpenAI, Elon Musk, AGI", aliases) == []


def test_curated_title_gate_recovers_guest_name_away_from_title_start() -> None:
    source = {
        "identity_gate": "title_contains_alias",
        "required_title_terms": ["interview", "with dylan patel"],
    }
    assert source_identity_matches(
        "AI Infrastructure with Dylan Patel of SemiAnalysis",
        ["Dylan Patel"],
        source,
    ) == ["dylan patel"]
    assert source_identity_matches(
        "An interview with Dylan Patel",
        ["Dylan Patel"],
        source,
    ) == ["dylan patel"]
    assert source_identity_matches(
        "Sam Altman on chips and Elon Musk",
        ["Elon Musk"],
        {"identity_gate": "title_contains_alias", "required_title_terms": ["interview"]},
    ) == []


def test_source_gate_respects_explicit_negative_terms() -> None:
    source = {
        "identity_gate": "title_contains_alias",
        "excluded_title_terms": ["elon musk vs."],
    }
    assert source_identity_matches(
        "Elon Musk vs. Sam Altman: Our Reaction",
        ["Elon Musk"],
        source,
    ) == []


def test_parse_rss_entries_keeps_episode_identity_fields() -> None:
    content = b"""<?xml version="1.0" encoding="UTF-8"?>
    <rss version="2.0"><channel><title>Research Feed</title>
      <image><url>https://example.com/feed-cover.jpg</url></image>
      <item><title>#1 - Dylan Patel: AI Infrastructure</title>
        <link>https://example.com/dylan</link><guid>episode-1</guid>
        <description>Direct interview</description>
        <enclosure url="https://cdn.example.com/dylan.mp3" type="audio/mpeg"/>
        <duration>01:23:45</duration>
        <pubDate>Sun, 20 Jul 2026 12:00:00 +0000</pubDate>
      </item>
    </channel></rss>"""
    channel, entries = parse_feed_entries(content)
    assert channel == "Research Feed"
    assert entries == [
        {
            "title": "#1 - Dylan Patel: AI Infrastructure",
            "link": "https://example.com/dylan",
            "guid": "episode-1",
            "description": "Direct interview",
            "published": "Sun, 20 Jul 2026 12:00:00 +0000",
            "author": "Research Feed",
            "artwork_url": "https://example.com/feed-cover.jpg",
            "audio_url": "https://cdn.example.com/dylan.mp3",
            "duration": "01:23:45",
        }
    ]


def test_parse_rss_entries_prefers_episode_thumbnail() -> None:
    content = b"""<?xml version="1.0" encoding="UTF-8"?>
    <rss xmlns:media="http://search.yahoo.com/mrss/"><channel>
      <title>Research Feed</title>
      <image><url>https://example.com/feed-cover.jpg</url></image>
      <item><title>Elon Musk Interview</title>
        <link>https://example.com/elon</link>
        <media:thumbnail url="https://example.com/episode-cover.jpg"/>
      </item>
    </channel></rss>"""
    _channel, entries = parse_feed_entries(content)
    assert entries[0]["artwork_url"] == "https://example.com/episode-cover.jpg"


def test_youtube_thumbnail_derivation_is_stable() -> None:
    assert youtube_thumbnail_url(
        "https://www.youtube.com/watch?v=eKg-nFDIPkQ"
    ) == "https://i.ytimg.com/vi/eKg-nFDIPkQ/hqdefault.jpg"
    assert (
        archive_artwork_url(
            {
                "source": "https://youtu.be/yG5Quo6xA74",
                "url": "https://elonmuskarchive.org/video/example",
            }
        )
        == "https://i.ytimg.com/vi/yG5Quo6xA74/hqdefault.jpg"
    )


def test_embedded_registry_and_snapshot_are_self_contained() -> None:
    registry = load_registry()
    assert [person["slug"] for person in registry["people"]] == [
        "dylan-patel",
        "elon-musk",
    ]
    module_root = module_resources_dir()
    manifest = json.loads((module_root / "module.json").read_text(encoding="utf-8"))
    assert manifest["runtime_policy"]["runtime_import_from_what_he_said"] is False
    people = {person["slug"]: person for person in registry["people"]}
    assert len(people["dylan-patel"]["feeds"]) >= 15
    assert len(people["elon-musk"]["feeds"]) >= 10
    assert people["dylan-patel"]["feeds"][0]["locator"] == (
        "https://api.substack.com/feed/podcast/69345.rss"
    )
    assert people["elon-musk"]["feeds"][2]["locator"] == (
        "https://api.substack.com/feed/podcast/69345.rss"
    )
    assert people["dylan-patel"]["searches"][0]["kind"] == "itunes_episode_search"
    elon_search_kinds = {
        source["kind"] for source in people["elon-musk"]["searches"]
    }
    assert elon_search_kinds == {
        "elon_archive_interviews",
        "itunes_episode_search",
    }
    copied = module_root / "vendor" / "private_person_monitor_snapshot"
    assert (copied / "UPSTREAM.lock").exists()
    assert len(list((copied / "person_packs").rglob("*.yaml"))) == 8


def test_canonical_url_and_title_date_dedupe_are_stable() -> None:
    assert canonicalize_url(
        "https://example.com/episode/?utm_source=test&id=4#player"
    ) == "https://example.com/episode?id=4"
    first = {
        "title": "Elon Musk — AI and Space",
        "published_at": "2026-02-05T12:00:00Z",
        "url": "https://example.com/one",
    }
    replay_same_day = {
        "title": "Elon Musk — AI and Space",
        "published_at": "2026-02-05T20:00:00Z",
        "url": "https://example.com/two",
    }
    later_episode = {
        "title": "Elon Musk — AI and Space",
        "published_at": "2026-03-05T12:00:00Z",
        "url": "https://example.com/three",
    }
    assert item_dedupe_key(first) == item_dedupe_key(replay_same_day)
    assert item_dedupe_key(first) != item_dedupe_key(later_episode)


def test_itunes_search_keeps_trusted_direct_guest_candidate_only() -> None:
    class FakeResponse:
        status_code = 200

        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict:
            return {
                "results": [
                    {
                        "trackId": 1,
                        "trackName": "#2404 - Elon Musk",
                        "collectionName": "The Joe Rogan Experience",
                        "trackViewUrl": "https://podcasts.apple.com/episode/1",
                        "episodeUrl": "https://cdn.example.com/episode-1.mp3",
                        "trackTimeMillis": 3_725_000,
                        "description": "Elon Musk discusses AI and space.",
                        "artworkUrl600": "https://example.com/joe-rogan-600.jpg",
                        "releaseDate": "2025-10-31T00:00:00Z",
                    },
                    {
                        "trackId": 2,
                        "trackName": "What Elon Musk Gets Wrong",
                        "collectionName": "Unknown Commentary Show",
                        "trackViewUrl": "https://podcasts.apple.com/episode/2",
                        "releaseDate": "2025-10-30T00:00:00Z",
                    },
                ]
            }

    source = {
        "key": "itunes-elon",
        "name": "Apple Podcasts",
        "kind": "itunes_episode_search",
        "locator": "https://itunes.apple.com/search",
        "queries": ["Elon Musk"],
        "trusted_collections": ["The Joe Rogan Experience"],
        "required_title_terms": ["- elon musk"],
        "identity_gate": "title_contains_alias",
        "expected_role": "guest_or_speaker",
    }
    spec = {"slug": "elon-musk", "identity_aliases": ["Elon Musk"]}
    with patch("app.person_monitor_service.requests.get", return_value=FakeResponse()):
        items, status = discover_itunes_episodes(spec, source)
    assert [item["title"] for item in items] == ["#2404 - Elon Musk"]
    assert items[0]["artwork_url"] == "https://example.com/joe-rogan-600.jpg"
    assert items[0]["audio_url"] == "https://cdn.example.com/episode-1.mp3"
    assert items[0]["duration_text"] == "01:02:05"
    assert items[0]["source_type"] == "podcast_rss"
    assert items[0]["monitoring_classification"]["status"] == "direct_expression_candidate"
    assert (
        items[0]["monitoring_classification"][
            "speaker_or_author_confirmation_required"
        ]
        is True
    )
    assert status["item_count"] == 1
    assert status["related_mentions_filtered"] == 1


def test_elon_archive_keeps_latest_transcript_backed_interviews() -> None:
    class FakeResponse:
        status_code = 200
        text = (
            '<meta property="og:image" '
            'content="https://example.com/economist-elon.jpg">'
        )

        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict:
            return {
                "total": 156,
                "results": [
                    {
                        "id": "economist-elon-musk-2026-07-23",
                        "type": "interview",
                        "date": "2026-07-23",
                        "title": "The Economist — Interview with Elon Musk",
                        "sourceLabel": "The Economist",
                        "url": "https://elonmuskarchive.org/video/economist-elon-musk-2026-07-23",
                        "source": "https://www.economist.com/interview",
                    },
                    {
                        "id": "tesla-q2-2026",
                        "type": "earnings",
                        "date": "2026-07-22",
                        "title": "Tesla Q2 2026 earnings call",
                        "url": "https://elonmuskarchive.org/video/tesla-q2-2026",
                    },
                ],
            }

    source = {
        "key": "elon-musk-archive-interviews",
        "name": "Elon Musk Archive 公开访谈档案",
        "kind": "elon_archive_interviews",
        "locator": "https://elonmuskarchive.org/agents/search",
        "result_limit": 30,
    }
    spec = {"slug": "elon-musk", "identity_aliases": ["Elon Musk"]}
    with patch("app.person_monitor_service.requests.get", return_value=FakeResponse()):
        items, status = discover_elon_archive_interviews(spec, source)

    assert [item["title"] for item in items] == [
        "The Economist — Interview with Elon Musk"
    ]
    classification = items[0]["monitoring_classification"]
    assert classification["discovery_tier"] == "verified_archive"
    assert classification["speaker_or_author_confirmation_required"] is False
    assert items[0]["published_at"] == "2026-07-23T00:00:00Z"
    assert items[0]["artwork_url"] == "https://example.com/economist-elon.jpg"
    assert items[0]["official_transcript_url"] == (
        "https://elonmuskarchive.org/video/economist-elon-musk-2026-07-23"
    )
    assert items[0]["source_type"] == "verified_archive"
    assert status["item_count"] == 1


def test_refresh_person_reports_progress_and_orders_by_latest_date(tmp_path) -> None:
    spec = {
        "slug": "elon-musk",
        "canonical_name": "Elon Musk",
        "display_name_zh": "埃隆·马斯克 / Elon Musk",
        "organization": "Tesla / SpaceX",
        "feeds": [
            {"key": "archive", "name": "已核验档案", "kind": "podcast_rss"},
            {"key": "feed", "name": "固定节目", "kind": "podcast_rss"},
        ],
        "searches": [],
    }
    old_archive = {
        "candidate_key": "archive-old",
        "title": "Old verified interview",
        "published_at": "2025-12-01T00:00:00Z",
        "url": "https://example.com/archive-old",
        "monitoring_classification": {"discovery_tier": "verified_archive"},
    }
    latest_feed = {
        "candidate_key": "feed-new",
        "title": "Latest curated interview",
        "published_at": "2026-08-18T00:00:00Z",
        "url": "https://example.com/feed-new",
        "monitoring_classification": {"discovery_tier": "curated_feed"},
    }

    def fake_discover(_spec, source, *, max_items):
        del max_items
        item = old_archive if source["key"] == "archive" else latest_feed
        return [item], {
            "source_key": source["key"],
            "source_name": source["name"],
            "kind": source["kind"],
            "status": "succeeded",
            "item_count": 1,
        }

    progress = []
    previous = {
        "schema_version": "1.0.0",
        "items": [old_archive],
        "item_count": 1,
    }
    with (
        patch("app.person_monitor_service.person_spec", return_value=spec),
        patch("app.person_monitor_service.load_person_export", return_value=previous),
        patch("app.person_monitor_service.discover_source", side_effect=fake_discover),
        patch("app.person_monitor_service.export_dir", return_value=tmp_path),
    ):
        payload = refresh_person(
            "elon-musk",
            progress_callback=lambda completed, total, status: progress.append(
                (completed, total, status["source_key"])
            ),
        )

    assert [item["title"] for item in payload["items"][:2]] == [
        "Latest curated interview",
        "Old verified interview",
    ]
    assert [entry[:2] for entry in progress] == [(1, 2), (2, 2)]
    assert {entry[2] for entry in progress} == {"archive", "feed"}
    assert payload["refresh_stats"]["new_candidates"] == 1
    assert (tmp_path / "elon-musk.json").exists()


def test_refresh_person_does_not_recount_candidates_below_display_limit(tmp_path) -> None:
    spec = {
        "slug": "dylan-patel",
        "canonical_name": "Dylan Patel",
        "display_name_zh": "Dylan Patel / SemiAnalysis",
        "organization": "SemiAnalysis",
        "feeds": [{"key": "feed", "name": "Feed", "kind": "podcast_rss"}],
        "searches": [],
    }
    candidates = [
        {
            "candidate_key": f"item-{index}",
            "title": f"Interview {index}",
            "published_at": f"2026-08-{31 - index:02d}T00:00:00Z",
            "url": f"https://example.com/{index}",
            "monitoring_classification": {"discovery_tier": "curated_feed"},
        }
        for index in range(5)
    ]
    previous = {
        "schema_version": "1.0.0",
        "items": candidates[:3],
        "item_count": 3,
    }
    status = {
        "source_key": "feed",
        "source_name": "Feed",
        "kind": "podcast_rss",
        "status": "succeeded",
        "item_count": len(candidates),
    }
    with (
        patch("app.person_monitor_service.person_spec", return_value=spec),
        patch("app.person_monitor_service.load_person_export", return_value=previous),
        patch(
            "app.person_monitor_service.discover_source",
            return_value=(candidates, status),
        ),
        patch("app.person_monitor_service.export_dir", return_value=tmp_path),
    ):
        payload = refresh_person("dylan-patel", max_items=3)

    assert [item["candidate_key"] for item in payload["items"]] == [
        "item-0",
        "item-1",
        "item-2",
    ]
    assert payload["refresh_stats"]["new_candidates"] == 0
