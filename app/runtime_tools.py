from __future__ import annotations

import os
import shutil
from pathlib import Path


def resolve_ffmpeg() -> str | None:
    explicit_path = os.getenv("PODCAST_TRANSCRIBER_FFMPEG")
    if explicit_path and Path(explicit_path).expanduser().exists():
        return str(Path(explicit_path).expanduser())

    system_path = shutil.which("ffmpeg")
    if system_path:
        return system_path

    try:
        import imageio_ffmpeg

        bundled_path = imageio_ffmpeg.get_ffmpeg_exe()
        if bundled_path and Path(bundled_path).exists():
            return str(bundled_path)
    except Exception:
        return None

    return None


def ensure_ffmpeg_on_path() -> str | None:
    ffmpeg_path = resolve_ffmpeg()
    if not ffmpeg_path:
        return None
    parent = str(Path(ffmpeg_path).parent)
    current_path = os.environ.get("PATH", "")
    path_parts = current_path.split(os.pathsep) if current_path else []
    if parent not in path_parts:
        os.environ["PATH"] = parent + os.pathsep + current_path
    return ffmpeg_path
