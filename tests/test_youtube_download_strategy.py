from main import build_download_attempts, build_youtube_download_options


def test_youtube_download_uses_yt_dlp_client_auto_selection() -> None:
    progress_hook = lambda _payload: None

    opts = build_youtube_download_options(
        "/tmp/%(title)s.%(ext)s",
        "m4a",
        progress_hook,
        "/tmp/ffmpeg",
    )

    assert "extractor_args" not in opts
    assert opts["format"] == "bestaudio/best"
    assert opts["ffmpeg_location"] == "/tmp/ffmpeg"
    assert opts["progress_hooks"] == [progress_hook]
    assert opts["postprocessors"] == [
        {
            "key": "FFmpegExtractAudio",
            "preferredcodec": "m4a",
            "preferredquality": "192",
        }
    ]


def test_cookie_attempts_preserve_automatic_client_selection() -> None:
    base_opts = build_youtube_download_options(
        "/tmp/%(title)s.%(ext)s",
        "m4a",
        lambda _payload: None,
    )

    attempts = build_download_attempts(base_opts, "", "chrome")

    assert [label for label, _opts in attempts] == [
        "chrome 浏览器登录态",
        "无 cookies 直连",
    ]
    assert all("extractor_args" not in attempt_opts for _label, attempt_opts in attempts)
