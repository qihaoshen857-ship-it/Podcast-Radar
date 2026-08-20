from __future__ import annotations

from main import (
    VideoItem,
    extract_elon_archive_transcript_html,
    person_monitor_video_item,
)


def test_person_monitor_rss_candidate_maps_to_direct_audio() -> None:
    item = person_monitor_video_item(
        {
            "candidate_key": "dylan-feed:episode-25",
            "title": "Dylan Patel on AI Infrastructure",
            "url": "https://example.com/episode-25",
            "audio_url": "https://cdn.example.com/episode-25.mp3",
            "duration_text": "01:05:02",
            "published_at": "2026-08-17T12:00:00Z",
            "author_or_channel": "Jordan Nanos",
            "monitoring_classification": {"discovery_tier": "curated_feed"},
        },
        "Dylan Patel / SemiAnalysis",
    )

    assert item.video_id.startswith("person-")
    assert item.source_type == "podcast_rss"
    assert item.audio_url == "https://cdn.example.com/episode-25.mp3"
    assert item.published_text == "2026-08-17"
    assert item.topic == "人物访谈：Dylan Patel / SemiAnalysis"


def test_verified_archive_maps_original_media_and_transcript_page() -> None:
    archive_url = "https://elonmuskarchive.org/video/example"
    item = person_monitor_video_item(
        {
            "candidate_key": "archive:example",
            "title": "Interview with Elon Musk",
            "url": archive_url,
            "original_source_url": "https://www.youtube.com/watch?v=abcdefghijk",
            "monitoring_classification": {"discovery_tier": "verified_archive"},
        },
        "Elon Musk",
    )

    assert item.source_type == "youtube"
    assert item.webpage_url == "https://www.youtube.com/watch?v=abcdefghijk"
    assert item.official_transcript_url == archive_url


def test_stale_directory_page_without_media_is_not_queued() -> None:
    item = person_monitor_video_item(
        {
            "candidate_key": "itunes:stale",
            "title": "Old Dylan Patel listing",
            "url": "https://podcasts.apple.com/us/podcast/example/id1?i=2",
            "monitoring_classification": {"discovery_tier": "directory_candidate"},
        },
        "Dylan Patel",
    )

    assert item.available is False
    assert item.webpage_url.startswith("https://podcasts.apple.com/")


def test_archive_html_extracts_speaker_labelled_full_transcript() -> None:
    item = VideoItem(
        video_id="person-example",
        title="Interview with Elon Musk",
        duration_text="-",
        webpage_url="https://example.com/video",
        source_name="Example",
        published_text="2026-08-20",
    )
    page_html = """
    <html><body>
      <h1>Interview with Elon Musk</h1>
      <h2>Transcript</h2>
      <div>
        <div class="mb-1 text-sm font-semibold text-accent">Host</div>
        <button aria-label="Play from here"><span>Hello </span><span>Elon.</span></button>
      </div>
      <div>
        <div class="mb-1 text-sm font-semibold text-accent">Elon Musk</div>
        <button aria-label="Play from here"><span>Thanks for having me.</span></button>
      </div>
    </body></html>
    """

    transcript = extract_elon_archive_transcript_html(item, page_html)
    assert transcript.startswith("# Interview with Elon Musk — Full Transcript")
    assert "### Host\n\nHello Elon." in transcript
    assert "### Elon Musk\n\nThanks for having me." in transcript
