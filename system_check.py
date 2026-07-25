from __future__ import annotations

import importlib
import json
import shutil
import sys
from pathlib import Path

from app.transcription import TranscriptionSettings, load_api_key


ROOT = Path(__file__).resolve().parent
DOTENV_PATH = ROOT / ".env"
SETTINGS_PATH = ROOT / "settings.json"
REQUIRED_MODULES = [
    "yt_dlp",
    "dashscope",
    "librosa",
    "soundfile",
    "silero_vad",
    "torch",
    "torchaudio",
    "numpy",
    "websocket",
]


def ok(label: str, detail: str = "") -> None:
    print(f"OK   {label}{': ' + detail if detail else ''}")


def miss(label: str, detail: str = "") -> None:
    print(f"MISS {label}{': ' + detail if detail else ''}")


def main() -> int:
    failures = 0
    print(f"Project: {ROOT}")
    print(f"Python: {sys.version.split()[0]} ({sys.executable})")

    for module in REQUIRED_MODULES:
        try:
            importlib.import_module(module)
            ok(f"module {module}")
        except Exception as exc:
            failures += 1
            miss(f"module {module}", str(exc))

    for tool in ("ffmpeg", "ffprobe", "deno"):
        path = shutil.which(tool)
        if path:
            ok(tool, path)
        else:
            failures += 1
            miss(tool)

    api_key = load_api_key(DOTENV_PATH)
    if api_key:
        ok("DASHSCOPE_API_KEY", f"loaded ({len(api_key)} chars)")
    else:
        failures += 1
        miss("DASHSCOPE_API_KEY", ".env missing or empty")

    if SETTINGS_PATH.exists():
        try:
            settings = json.loads(SETTINGS_PATH.read_text(encoding="utf-8"))
            ok("settings.json", f"browser={settings.get('browser_cookies')}, format={settings.get('audio_format')}")
            download_dir = Path(settings.get("download_dir") or ROOT / "downloads").expanduser()
            download_dir.mkdir(parents=True, exist_ok=True)
            ok("download_dir", str(download_dir))
            TranscriptionSettings(
                api_key=api_key,
                target_segment_seconds=int(settings.get("target_segment_seconds") or 120),
                max_segment_seconds=int(settings.get("max_segment_seconds") or 180),
                workers=int(settings.get("transcription_workers") or 4),
                strategy=str(settings.get("transcription_strategy") or "local_first"),
            ).validate()
            ok("transcription settings")
        except Exception as exc:
            failures += 1
            miss("settings.json", str(exc))
    else:
        failures += 1
        miss("settings.json")

    chrome_exists = Path("/Applications/Google Chrome.app").exists() or Path("~/Applications/Google Chrome.app").expanduser().exists()
    if chrome_exists:
        ok("Google Chrome")
    else:
        failures += 1
        miss("Google Chrome", "needed for browser cookie mode=chrome")

    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
