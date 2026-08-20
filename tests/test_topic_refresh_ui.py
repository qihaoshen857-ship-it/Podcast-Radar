from __future__ import annotations

from main import (
    PIXEL_DINO_FRAMES,
    TOPIC_FILTER_LABELS,
    YTAudioDownloaderApp,
    progress_marker_position,
    topic_display_label,
    topic_download_title,
    topic_filter_variant,
)


class DummyVar:
    def __init__(self) -> None:
        self.values: list[object] = []

    def set(self, value: object) -> None:
        self.values.append(value)


class DummyProgressBar:
    def __init__(self) -> None:
        self.calls: list[object] = []

    def winfo_exists(self) -> bool:
        return True

    def stop(self) -> None:
        self.calls.append("stop")

    def configure(self, **kwargs: object) -> None:
        self.calls.append(("configure", kwargs))

    def start(self, interval: int) -> None:
        self.calls.append(("start", interval))


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


def test_task_progress_starts_dinosaur_before_first_percent() -> None:
    app = type("DummyApp", (), {})()
    app.progress_value = DummyVar()
    app.progress_percent_var = DummyVar()
    app.progress_detail_var = DummyVar()
    app.progress_bar = DummyProgressBar()

    YTAudioDownloaderApp.begin_immediate_task_progress(
        app,
        "准备中",
        "正在启动转译队列",
    )

    assert app.progress_value.values == [0.0]
    assert app.progress_percent_var.values == ["准备中"]
    assert app.progress_detail_var.values == ["正在启动转译队列"]
    assert app.progress_bar.calls == [
        "stop",
        ("configure", {"mode": "indeterminate"}),
        ("start", 84),
    ]


def test_zero_percent_event_keeps_dinosaur_running() -> None:
    app = type("DummyApp", (), {})()
    app.progress_value = DummyVar()
    app.progress_percent_var = DummyVar()
    app.progress_detail_var = DummyVar()
    app.progress_bar = DummyProgressBar()
    app.update_row = lambda *_args, **_kwargs: None

    YTAudioDownloaderApp.on_progress(
        app,
        {
            "video_id": "episode",
            "percent": 0,
            "detail": "正在准备音频",
            "row_progress": 0,
            "row_status": "转录准备中",
        },
    )

    assert app.progress_percent_var.values == ["准备中"]
    assert app.progress_bar.calls == [
        "stop",
        ("configure", {"mode": "indeterminate"}),
        ("start", 84),
    ]
