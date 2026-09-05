from datetime import datetime, timezone
import pytest
import main
from app.embodied_intelligence import TOPIC, is_embodied_item, score_embodied_item


def item(title, **kwargs):
    return main.VideoItem(video_id="sample", title=title, webpage_url="https://example.com/episode",
                          duration_text="45:00", **kwargs)


@pytest.mark.parametrize("title,description,source", [
    ("Humanoid VLA with a robotics founder", "", "No Priors"),
    ("灵巧手与人形机器人量产：对话创始人", "", "小宇宙"),
    ("World models for robots", "sim-to-real policy training", ""),
    ("NEO’s Hands", "", "YouTube | 1X"),
    ("Apollo at work", "", "YouTube | Apptronik"),
    ("Omni-modal whole-body mobile manipulation", "", ""),
])
def test_embodied_kept(title, description, source):
    assert is_embodied_item(title, description, source)


@pytest.mark.parametrize("title,description,source", [
    ("Tesla earnings and FSD robotaxi", "humanoid robots mentioned in sponsor footer", ""),
    ("AI agents automate email marketing", "world models", "No Priors"),
    ("Figure raises money", "", "News"),
    ("World models for video generation", "A new image model", ""),
    ("Industrial robots weld steel", "Traditional fixed automation", ""),
    ("Innovation at Agility: Digit does TikTok", "", "YouTube | Agility Robotics"),
    ("F.03 Livestream: Day 8", "", "YouTube | Figure"),
    ("Optimus prime movie", "", ""),
])
def test_unrelated_or_pure_entertainment_rejected(title, description, source):
    assert not is_embodied_item(title, description, source)


def test_robot_topic_precedes_ai_and_old_cache():
    episode = item("Humanoid VLA founder interview", source_name=main.DEFAULT_AI_PODCAST_FEEDS[1].name)
    assert main.infer_item_topic(episode) == TOPIC
    record = {"title": episode.title, "source_name": episode.source_name, "topic": "ai"}
    app = object.__new__(main.YTAudioDownloaderApp)
    assert app.cached_record_matches_topic(record, TOPIC)
    assert not app.cached_record_matches_topic(record, "ai")


def test_robot_scores_do_not_double_count_popularity_or_boilerplate():
    episode = item("Humanoid VLA founder interview", play_count=100000000,
                   description_text="VLA robot policy training. Follow us https://tiktok.com/example")
    main.update_item_importance_score(episode)
    assert episode.importance_score == episode.investment_score
    assert "表演" not in episode.importance_reason


def test_delivery_evidence_beats_provisional_plan():
    now = datetime(2026, 9, 5, tzinfo=timezone.utc)
    delivered, reasons = score_embodied_item("Humanoid robots delivered: 200 units", now=now)
    planned, plan_reasons = score_embodied_item("Humanoid robots target 200 units of production", now=now)
    assert delivered > planned
    assert any("来源自述" in r for r in reasons)
    assert any("计划" in r for r in plan_reasons)


def test_future_date_has_no_freshness_bonus():
    now = datetime(2026, 9, 5, tzinfo=timezone.utc)
    _, reasons = score_embodied_item("Humanoid robot VLA", source_name="Official",
                                     published_text="2026-09-06", now=now)
    assert "近两周更新" not in reasons


def test_robot_tab_group_and_dedup():
    assert main.topic_download_title(TOPIC) == "今日雷达 · 具身智能"
    assert list(main.TOPIC_FILTER_LABELS) == ["all", "ai", "ai_startup", TOPIC, "weather_agri", "science_wellness"]
    assert len(main.DEFAULT_EMBODIED_YOUTUBE_FEEDS) >= 7
    assert len(main.DEFAULT_EMBODIED_PODCAST_FEEDS) >= 3
    episode = item("人形机器人灵巧手数据", source_name=main.DEFAULT_AI_PODCAST_FEEDS[1].name)
    other = item("AI models and compute", source_name=main.DEFAULT_AI_PODCAST_FEEDS[0].name)
    app = object.__new__(main.YTAudioDownloaderApp)
    app.current_topic = "all"
    app.video_items = [episode, other]
    sections = app.card_sections()
    assert [s[0] for s in sections] == ["AI 播客", "具身智能 / 人形机器人"]
    assert sections[1][2] == [episode]
    duplicate = item(episode.title, source_name=episode.source_name)
    rows, count = main.dedupe_episode_rows([(None, episode), (None, duplicate)])
    assert len(rows) == 1 and count == 1


def test_robot_filter_applied_before_per_feed_limit(monkeypatch):
    feed = main.DEFAULT_EMBODIED_PODCAST_FEEDS[0]
    unrelated = [(None, item(f"AI software update {i}")) for i in range(8)]
    robot = item("Humanoid robotics founder interview")
    monkeypatch.setattr(main, "discover_podcast_feeds", lambda topic: ([feed], 0))
    monkeypatch.setattr(main, "discover_youtube_feeds", lambda topic: ([], 0))
    monkeypatch.setattr(main, "discover_xiaoyuzhou_feeds", lambda topic: ([], 0))
    monkeypatch.setattr(main, "fetch_podcast_feed_items", lambda feed: (unrelated + [(None, robot)], 0))
    _, rows, _, _ = main.fetch_podcast_updates(TOPIC)
    assert rows == [robot]
