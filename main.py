from __future__ import annotations

import json
import html
import math
import os
import platform
import queue
import re
import shutil
import socket
import subprocess
import threading
import time
import hashlib
import webbrowser
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from difflib import SequenceMatcher
from email.utils import parsedate_to_datetime
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from urllib.parse import unquote, urlparse
import tkinter as tk
from xml.etree import ElementTree as ET

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
try:
    from PIL import Image, ImageDraw, ImageTk
except Exception:  # noqa: BLE001
    Image = None
    ImageDraw = None
    ImageTk = None
from yt_dlp import YoutubeDL
from yt_dlp.utils import DownloadError

from app.edge_cookie_export import export_edge_cookies
from app.person_monitor_module import PersonMonitorPage
from app.runtime_tools import ensure_ffmpeg_on_path
from app.research_digest import (
    DigestMetadata,
    EpisodeDetailMetadata,
    FullTranslationMetadata,
    ResearchDigestError,
    build_bilingual_episode_detail,
    build_chinese_full_translation,
    build_chinese_research_digest,
    translate_episode_card_summaries,
    translate_episode_titles,
)
from app.transcription import (
    SUPPORTED_EXTENSIONS,
    TranscriptionError,
    TranscriptionSettings,
    ensure_runtime_available,
    load_api_key,
    save_api_key,
    transcribe_file,
)


APP_NAME = "Podcast Radar"
APP_VERSION = "0.4.36"
TRANSCRIPTION_MODE_OPTIONS = {
    "本地优先": "local_first",
    "极速云端": "cloud_fast",
}
TRANSCRIPTION_MODE_LABELS = {value: key for key, value in TRANSCRIPTION_MODE_OPTIONS.items()}
REQUEST_USER_AGENT = (
    f"PodcastRadar/{APP_VERSION} "
    "(Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/126.0 Safari/537.36"
)
PROJECT_DIR = Path(__file__).resolve().parent


def resolve_app_dir() -> Path:
    if getattr(sys, "frozen", False):
        # Keep the legacy support-directory name so an app rename never splits
        # the user's downloads, settings, cache, or research library.
        return (Path.home() / "Library" / "Application Support" / "ResearchPodcastRadar").expanduser()
    return PROJECT_DIR


def ensure_runtime_path() -> None:
    runtime_candidates: list[Path] = []
    env_runtime = os.getenv("PODCAST_TRANSCRIBER_RUNTIME_DIR")
    if env_runtime:
        runtime_candidates.append(Path(env_runtime).expanduser())

    if getattr(sys, "frozen", False):
        executable_dir = Path(sys.executable).resolve()
        runtime_candidates.append(executable_dir.parent.parent / "Resources" / ".runtime")
        runtime_candidates.append(executable_dir.parent.parent.parent / "Resources" / ".runtime")

    runtime_candidates.append(PROJECT_DIR / ".runtime")

    for runtime_dir in runtime_candidates:
        runtime_bin = runtime_dir / "bin"
        if runtime_bin.exists():
            bin_path = str(runtime_bin)
            current_path = os.environ.get("PATH", "")
            path_parts = current_path.split(os.pathsep) if current_path else []
            if bin_path not in path_parts:
                os.environ["PATH"] = bin_path + os.pathsep + current_path
            return


APP_DIR = resolve_app_dir()
SETTINGS_PATH = APP_DIR / "settings.json"
DOTENV_PATH = APP_DIR / ".env"
DEFAULT_DOWNLOAD_DIR = APP_DIR / "downloads"
AUTO_EDGE_COOKIE_FILE = APP_DIR / "auto_edge_cookies.txt"
ARTWORK_CACHE_DIR = APP_DIR / ".cache" / "artwork"
RESEARCH_CACHE_DIR = APP_DIR / ".cache" / "research"
RESEARCH_AUDIO_DIR = RESEARCH_CACHE_DIR / "audio"
RESEARCH_DIGEST_DIR = RESEARCH_CACHE_DIR / "digests"
RESEARCH_TRANSLATION_DIR = RESEARCH_CACHE_DIR / "translations"
RESEARCH_TTS_DIR = RESEARCH_CACHE_DIR / "tts"
RESEARCH_LIBRARY_PATH = APP_DIR / "research_library.json"
FULL_TRANSLATION_MODEL = "qwen-plus"
FULL_TRANSLATION_PROMPT_VERSION = "1"
DEFAULT_BROWSER_ORDER = ("chrome", "edge", "firefox")
DEFAULT_TTS_PYTHON = Path.home() / "qwen3_tts_local" / ".venv" / "bin" / "python"
DEFAULT_TTS_MODEL = "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"
DEFAULT_TTS_SPEAKER = "Vivian"
DEFAULT_TTS_LANGUAGE = "Auto"
DEFAULT_TTS_MAX_CHARS = 1800
QWEN_TTS_BRIDGE_SOURCE = r'''from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run local Qwen3-TTS from Podcast Radar.")
    parser.add_argument("--text-file", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--speaker", default="Vivian")
    parser.add_argument("--language", default="Auto")
    parser.add_argument("--instruct", default="")
    parser.add_argument("--max-chars", type=int, default=1800)
    parser.add_argument("--chunk-chars", type=int, default=1200)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", default="auto")
    return parser.parse_args()


def torch_dtype(torch_module, dtype_name: str, device: str):
    if dtype_name and dtype_name != "auto":
        return getattr(torch_module, dtype_name)
    if device == "mps":
        return torch_module.float16
    return torch_module.float32


def choose_device(torch_module, requested: str) -> str:
    requested = (requested or "auto").strip().lower()
    if requested and requested != "auto":
        return requested
    if getattr(torch_module.backends, "mps", None) and torch_module.backends.mps.is_available():
        return "mps"
    return "cpu"


def emit(payload: dict, *, stream=None) -> None:
    print(json.dumps(payload, ensure_ascii=False), file=stream or sys.stdout, flush=True)


def split_tts_chunks(text: str, max_chars: int) -> list[str]:
    text = re.sub(r"\s+", " ", text or "").strip()
    if not text:
        return []
    max_chars = max(300, int(max_chars or 1200))
    chunks: list[str] = []
    while len(text) > max_chars:
        search_start = max(0, int(max_chars * 0.55))
        candidates = []
        for boundary in ("。", "！", "？", ". ", "? ", "! ", "；", "; ", "，", ", "):
            index = text.rfind(boundary, search_start, max_chars + 1)
            if index >= search_start:
                candidates.append(index + len(boundary))
        cut = max(candidates) if candidates else max_chars
        chunk = text[:cut].strip()
        if chunk:
            chunks.append(chunk)
        text = text[cut:].lstrip()
    if text:
        chunks.append(text)
    return chunks


def main() -> int:
    args = parse_args()
    partial_out = None
    try:
        text = Path(args.text_file).read_text(encoding="utf-8", errors="ignore").strip()
        if not text:
            raise ValueError("empty text")
        if args.max_chars > 0 and len(text) > args.max_chars:
            text = text[: args.max_chars].rstrip()
        chunks = split_tts_chunks(text, args.chunk_chars)
        if not chunks:
            raise ValueError("empty text")

        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
        os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
        os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

        import numpy as np
        import soundfile as sf
        import torch
        from qwen_tts import Qwen3TTSModel

        requested_device = choose_device(torch, args.device)
        load_error = None
        try:
            tts = Qwen3TTSModel.from_pretrained(
                args.model,
                device_map=requested_device,
                dtype=torch_dtype(torch, args.dtype, requested_device),
                attn_implementation=None,
            )
            used_device = requested_device
        except Exception as exc:  # noqa: BLE001
            load_error = f"{type(exc).__name__}: {exc}"
            if requested_device == "cpu":
                raise
            tts = Qwen3TTSModel.from_pretrained(
                args.model,
                device_map="cpu",
                dtype=torch.float32,
                attn_implementation=None,
            )
            used_device = "cpu"

        out = Path(args.output).expanduser()
        out.parent.mkdir(parents=True, exist_ok=True)
        partial_out = out.with_name(f"{out.stem}.part-{os.getpid()}{out.suffix}")
        partial_out.unlink(missing_ok=True)
        writer = None
        sample_rate = 0
        total_samples = 0
        try:
            for index, chunk in enumerate(chunks, start=1):
                wavs, sr = tts.generate_custom_voice(
                    text=chunk,
                    language=(args.language or "Auto").strip() or "Auto",
                    speaker=(args.speaker or "Vivian").strip() or "Vivian",
                    instruct=(args.instruct or "").strip() or None,
                    non_streaming_mode=True,
                )
                wav = np.asarray(wavs[0], dtype=np.float32).reshape(-1)
                if writer is None:
                    sample_rate = int(sr)
                    writer = sf.SoundFile(
                        str(partial_out),
                        mode="w",
                        samplerate=sample_rate,
                        channels=1,
                        subtype="PCM_16",
                    )
                elif int(sr) != sample_rate:
                    raise RuntimeError(f"sample rate changed: {sample_rate} -> {sr}")
                writer.write(wav)
                total_samples += len(wav)
                if index < len(chunks):
                    pause = np.zeros(int(sample_rate * 0.22), dtype=np.float32)
                    writer.write(pause)
                    total_samples += len(pause)
                emit({"progress": True, "done": index, "total": len(chunks)})
        finally:
            if writer is not None:
                writer.close()
        if not partial_out.exists() or partial_out.stat().st_size <= 44:
            raise RuntimeError("no audio generated")
        audio_info = sf.info(str(partial_out))
        if int(audio_info.frames) <= 0 or int(audio_info.samplerate) <= 0:
            raise RuntimeError("generated audio is incomplete")
        os.replace(partial_out, out)
        emit(
            {
                "ok": True,
                "output": str(out),
                "sample_rate": sample_rate,
                "seconds": round(total_samples / float(sample_rate), 2) if sample_rate else 0,
                "device": used_device,
                "load_warning": load_error or "",
                "chunks": len(chunks),
            }
        )
        return 0
    except Exception as exc:  # noqa: BLE001
        if partial_out is not None:
            partial_out.unlink(missing_ok=True)
        emit({"ok": False, "error": f"{type(exc).__name__}: {exc}"}, stream=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
'''

IS_MAC = platform.system() == "Darwin"
FONT_UI = "SF Pro Text" if IS_MAC else "Microsoft YaHei UI"
FONT_DISPLAY = "SF Pro Display" if IS_MAC else "Microsoft YaHei UI"
FONT_MONO = "SF Mono" if IS_MAC else "Consolas"

COLOR_BG = "#f5f5f7"
COLOR_SURFACE = "#ffffff"
COLOR_SURFACE_ALT = "#f2f2f7"
COLOR_TEXT = "#1d1d1f"
COLOR_MUTED = "#6e6e73"
COLOR_BORDER = "#d2d2d7"
COLOR_BLUE = "#0071e3"
COLOR_BLUE_DARK = "#005bb5"
COLOR_GREEN = "#34c759"
COLOR_PURPLE = "#af52de"
COLOR_RED = "#ff3b30"
COLOR_SIDEBAR = "#fbfbfd"
COLOR_SIDEBAR_ACTIVE = "#e8f1ff"
COLOR_CARD = "#ffffff"
COLOR_CARD_SELECTED = "#eef6ff"
COLOR_CARD_SHADOW = "#e4e7ec"
COLOR_INPUT = "#ffffff"

ITUNES_SEARCH_URL = "https://itunes.apple.com/search"
AI_DISCOVERY_LIMIT = 64
AI_EPISODES_PER_FEED = 4
YOUTUBE_EPISODES_PER_SOURCE = 3
XIAOYUZHOU_EPISODES_PER_SOURCE = 3
MAX_RSS_ENTRIES_PER_FEED = 160
AUTO_FETCH_DELAY_MS = 250
CARD_TRANSLATION_VISIBLE_LIMIT = 24
CARD_WIDTH = 196
CARD_HEIGHT = 376
CARD_INNER_WIDTH = 174
CARD_INNER_HEIGHT = 354
CARD_COVER_SIZE = 164
FEED_ROW_HEIGHT = 110
FEED_ROW_COVER_SIZE = 72
DETAIL_COVER_SIZE = 138
CARD_GAP_X = 26
CARD_GAP_Y = 30
MIN_VALID_AUDIO_BYTES = 4096
AUDIO_DOWNLOAD_CONNECT_TIMEOUT = 10
AUDIO_DOWNLOAD_READ_TIMEOUT = 75
AUDIO_DOWNLOAD_CHUNK_SIZE = 1024 * 512
TEXTISH_AUDIO_CONTENT_TYPES = {
    "application/json",
    "application/problem+json",
    "application/xml",
    "text/html",
    "text/plain",
    "text/xml",
}
KNOWN_AUDIO_EXTENSIONS = {"mp3", "m4a", "wav", "mp4", "aac", "ogg", "oga", "opus", "flac"}
WEATHER_AGRI_SOURCE_ALLOW_ALL = {
    "WeatherBrains",
    "Weather Geeks",
    "The Climate Pod",
    "Climate One",
    "NOAA Ocean Podcast",
    "AgriTalk",
    "Agri-Pulse Daybreak",
    "Grain Markets and Other Stuff",
    "Successful Farming Podcast",
}
WEATHER_AGRI_TITLE_KEYWORDS = (
    "agri",
    "agriculture",
    "farm",
    "farmer",
    "crop",
    "corn",
    "soy",
    "soybean",
    "wheat",
    "grain",
    "cotton",
    "sugar",
    "coffee",
    "cocoa",
    "rice",
    "cattle",
    "hog",
    "livestock",
    "dairy",
    "fertilizer",
    "usda",
    "wasde",
    "weather",
    "climate",
    "rain",
    "rainfall",
    "drought",
    "flood",
    "heat",
    "storm",
    "hurricane",
    "el nino",
    "el niño",
    "la nina",
    "la niña",
    "enso",
    "brazil",
    "argentina",
    "china",
    "market",
    "markets",
    "exports",
    "ethanol",
)
WEATHER_AGRI_REJECT_KEYWORDS = (
    "campfire",
    "hunting",
    "favorite bucks",
    "outdoor adventures",
    "deer",
)
AI_STARTUP_AI_KEYWORDS = (
    "ai",
    "artificial intelligence",
    "generative ai",
    "llm",
    "large language model",
    "agent",
    "agentic",
    "chatgpt",
    "claude",
    "codex",
    "cursor",
    "openai",
    "anthropic",
    "gemini",
    "grok",
    "copilot",
    "machine learning",
    "automation",
    "vibe coding",
    "mcp",
    "人工智能",
    "生成式 ai",
    "大模型",
    "智能体",
    "ai 工具",
    "ai工具",
    "数字员工",
)
AI_STARTUP_STORY_KEYWORDS = (
    "founder",
    "co-founder",
    "cofounder",
    "entrepreneur",
    "startup",
    "build",
    "built",
    "launch",
    "product-market fit",
    "pmf",
    "bootstrapped",
    "bootstrap",
    "solo business",
    "solo founder",
    "indie hacker",
    "from zero",
    "0 to 1",
    "go-to-market",
    "gtm",
    "revenue",
    "arr",
    "mrr",
    "customer",
    "pricing",
    "sales",
    "marketing",
    "agency",
    "business",
    "company",
    "ceo",
    "cpo",
    "cto",
    "how i use",
    "workflow",
    "monetize",
    "创始人",
    "创业者",
    "创业",
    "从 0 到 1",
    "从0到1",
    "一人公司",
    "产品",
    "商业化",
    "获客",
    "收入",
    "营收",
    "客户",
    "定价",
    "出海",
    "融资",
    "增长",
    "落地",
)
AI_STARTUP_STRONG_STORY_KEYWORDS = (
    "founder",
    "co-founder",
    "cofounder",
    "entrepreneur",
    "startup",
    "bootstrapped",
    "solo founder",
    "indie hacker",
    "product-market fit",
    "0 to 1",
    "创始人",
    "创业者",
    "创业",
    "从 0 到 1",
    "从0到1",
    "一人公司",
)
AI_STARTUP_TITLE_EXECUTION_KEYWORDS = (
    "founder",
    "co-founder",
    "cofounder",
    "entrepreneur",
    "startup",
    "build",
    "built",
    "launch",
    "product-market fit",
    "pmf",
    "bootstrapped",
    "solo business",
    "solo founder",
    "indie hacker",
    "0 to 1",
    "go-to-market",
    "gtm",
    "revenue",
    "arr",
    "mrr",
    "customer",
    "pricing",
    "sales",
    "marketing",
    "agency",
    "how i use",
    "workflow",
    "monetize",
    "创始人",
    "创业者",
    "创业",
    "从 0 到 1",
    "从0到1",
    "一人公司",
    "商业化",
    "获客",
    "收入",
    "营收",
    "客户",
    "定价",
    "出海",
)
AI_STARTUP_NEWS_KEYWORDS = (
    "daily brief",
    "news roundup",
    "weekly news",
    "market update",
    "latest ai news",
    "model release",
    "benchmark",
    "earnings",
    "overvalued",
    "ban chinese",
    "geopolitics",
    "chip war",
    "industry update",
    "explained",
    "bigger deal",
    "release matters",
    "first ever round",
    "margins matter",
    "will open source threaten",
    "series a is hard",
    "fool or genius",
    "sounds the alarm",
    "who wins the ai war",
    "china vs america",
    "offers trump",
    "who wins",
    "open models vs",
    "model war",
    "frontier models",
    "electricity",
    "energy sovereignty",
    "ai capex",
    "export controls",
    "ai infrastructure",
    "demand for compute",
    "token spend",
    "hiring ai researchers",
    "model is the product",
    "power is the bottleneck",
    "新闻速递",
    "一周新闻",
    "行业周报",
    "模型测评",
    "发布会",
    "产业新闻",
    "政策解读",
    "大厂财报",
    "估值争论",
    "芯片禁令",
)
AI_STARTUP_NATIVE_STORY_FEED_NAMES = {
    "The Startup Ideas Podcast",
    "The AI Ventures Podcast",
    "Pitch, Build, Scale",
    "小宇宙 | 十字路口 Crossing",
}
AI_PODCAST_SEARCHES: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    ("Latent Space", "Latent Space AI Engineer", ("latent", "space")),
    ("No Priors", "No Priors Artificial Intelligence", ("no", "priors")),
    ("The Cognitive Revolution", "The Cognitive Revolution AI", ("cognitive", "revolution")),
    ("Practical AI", "Practical AI Daniel Whitenack Chris Benson", ("practical", "ai")),
    ("Eye on A.I.", "Eye on A.I. Craig Smith", ("eye", "a.i.")),
    ("The AI Daily Brief", "The AI Daily Brief Artificial Intelligence News", ("ai", "daily", "brief")),
    ("Dwarkesh Podcast", "Dwarkesh Podcast AI", ("dwarkesh",)),
    ("Hard Fork", "Hard Fork New York Times", ("hard", "fork")),
)


@dataclass
class VideoItem:
    video_id: str
    title: str
    duration_text: str
    webpage_url: str
    source_type: str = "youtube"
    audio_url: str = ""
    source_name: str = ""
    published_text: str = ""
    artwork_url: str = ""
    description_text: str = ""
    play_count: int | None = None
    importance_score: float = 0.0
    investment_score: float = 0.0
    importance_reason: str = ""
    scored_at: str = ""
    available: bool = True
    status: str = "未开始"
    progress: str = "-"


TRANSCRIPT_HEADING_PATTERN = re.compile(
    r"^#\s*(.+?)\s+[\u2014\u2013-]\s*(?:full\s+)?transcript\b.*$",
    re.IGNORECASE,
)
TITLE_STOPWORDS = {
    "a",
    "an",
    "and",
    "for",
    "from",
    "in",
    "of",
    "on",
    "out",
    "the",
    "to",
    "with",
}


def normalize_episode_title(value: str) -> str:
    decoded = html.unescape(value or "").lower()
    return " ".join(re.findall(r"[a-z0-9]+|[\u4e00-\u9fff]+", decoded))


def extract_declared_transcript_title(text: str) -> str:
    for raw_line in (text or "").splitlines()[:12]:
        line = raw_line.strip()
        if not line:
            continue
        heading = TRANSCRIPT_HEADING_PATTERN.match(line)
        if heading:
            return heading.group(1).strip()
        title_field = re.match(r"^(?:episode\s+)?title\s*:\s*(.+)$", line, re.IGNORECASE)
        if title_field:
            return title_field.group(1).strip()
    return ""


def episode_titles_match(expected_title: str, observed_title: str) -> tuple[bool, float]:
    expected = normalize_episode_title(expected_title)
    observed = normalize_episode_title(observed_title)
    if not expected or not observed:
        return False, 0.0
    if expected == observed:
        return True, 1.0
    if min(len(expected), len(observed)) >= 18 and (expected in observed or observed in expected):
        return True, 0.95

    ratio = SequenceMatcher(None, expected, observed).ratio()
    expected_tokens = {token for token in expected.split() if token not in TITLE_STOPWORDS}
    observed_tokens = {token for token in observed.split() if token not in TITLE_STOPWORDS}
    common = expected_tokens & observed_tokens
    token_coverage = len(common) / max(1, min(len(expected_tokens), len(observed_tokens)))
    matched = ratio >= 0.72 or (len(common) >= 3 and token_coverage >= 0.70)
    return matched, max(ratio, token_coverage)


def validate_official_transcript_identity(item: VideoItem, text: str) -> tuple[bool, str]:
    declared_title = extract_declared_transcript_title(text)
    if not declared_title:
        return False, "官方稿未声明节目标题"
    matched, score = episode_titles_match(item.title, declared_title)
    if matched:
        return True, f"标题匹配 {score:.0%}"
    return False, f"返回《{declared_title}》，与当前《{item.title}》不符"


def cached_transcript_identity_conflict(item: VideoItem, path: Path) -> tuple[bool, str]:
    try:
        sample = path.read_text(encoding="utf-8", errors="replace")[:12000]
    except OSError as exc:
        return True, f"无法读取缓存：{exc}"
    declared_title = extract_declared_transcript_title(sample)
    if not declared_title:
        return False, ""
    matched, _score = episode_titles_match(item.title, declared_title)
    if matched:
        return False, ""
    return True, f"缓存声明为《{declared_title}》"


def quarantine_transcript_cache(path: Path) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    target = path.with_name(f"{path.stem}.identity-mismatch-{stamp}{path.suffix}")
    counter = 1
    while target.exists():
        target = path.with_name(f"{path.stem}.identity-mismatch-{stamp}-{counter}{path.suffix}")
        counter += 1
    path.replace(target)
    return target


def transcript_source_text(record: dict | None) -> str:
    if not isinstance(record, dict):
        return ""
    origin = str(record.get("transcript_origin") or "").strip()
    engine = str(record.get("transcript_engine") or "").strip()
    if origin == "audio_asr":
        if engine == "local_whisper":
            return "转录来源：当前节目音频 / 视频 · 本地 Whisper ASR"
        if engine == "local_whisper_with_aliyun_fallback":
            return "转录来源：本地 Whisper 优先 · 阿里云 ASR 补齐"
        if engine == "aliyun_cloud_fast":
            return "转录来源：当前节目音频 / 视频 · 阿里云并发 ASR"
        if engine == "aliyun_cloud_fast_with_local_fallback":
            return "转录来源：阿里云并发 ASR · 本地 Whisper 补齐"
        return "转录来源：当前节目音频 / 视频 ASR"
    labels = {
        "verified_official": "转录来源：已校验同期官方稿",
        "episode_brief_fallback": "资料来源：RSS 简介应急（非完整转录）",
        "existing_cache": "转录来源：本地缓存（来源沿用）",
        "identity_mismatch_quarantined": "转录状态：错配缓存已隔离，等待重新转录",
    }
    return labels.get(origin, "")


@dataclass(frozen=True)
class PodcastFeed:
    name: str
    feed_url: str
    source: str
    priority: int
    artwork_url: str = ""


@dataclass(frozen=True)
class YouTubeFeed:
    name: str
    channel_id: str
    source: str
    priority: int
    channel_url: str
    topic: str
    artwork_url: str = ""

    @property
    def feed_url(self) -> str:
        return f"https://www.youtube.com/feeds/videos.xml?channel_id={self.channel_id}"


@dataclass(frozen=True)
class XiaoyuzhouFeed:
    name: str
    podcast_url: str
    source: str
    priority: int
    topic: str
    artwork_url: str = ""

    @property
    def feed_url(self) -> str:
        return self.podcast_url


DEFAULT_AI_PODCAST_FEEDS: tuple[PodcastFeed, ...] = (
    PodcastFeed("Latent Space: The AI Engineer Podcast", "https://api.substack.com/feed/podcast/1084089.rss", "Latent.Space", 0),
    PodcastFeed("No Priors: Artificial Intelligence | Technology | Startups", "https://feeds.megaphone.fm/nopriors", "Conviction", 1),
    PodcastFeed('"The Cognitive Revolution" | AI Builders, Researchers, and Live Player Analysis', "https://feeds.megaphone.fm/RINTP3108857801", "Turpentine", 2),
    PodcastFeed("Practical AI", "https://feeds.transistor.fm/practical-ai-machine-learning-data-science-llm", "Practical AI LLC", 3),
    PodcastFeed("Eye On A.I.", "https://rss.libsyn.com/shows/123267/destinations/727317.xml", "Craig S. Smith", 4),
    PodcastFeed("The AI Daily Brief: Artificial Intelligence News and Analysis", "https://anchor.fm/s/f7cac464/podcast/rss", "Nathaniel Whittemore", 5),
    PodcastFeed("Dwarkesh Podcast", "https://apple.dwarkesh-podcast.workers.dev/feed.rss", "Dwarkesh Patel", 6),
    PodcastFeed("Hard Fork", "https://feeds.simplecast.com/6HKOhNgS", "The New York Times", 7),
)

DEFAULT_AI_STARTUP_PODCAST_FEEDS: tuple[PodcastFeed, ...] = (
    PodcastFeed(
        "The Startup Ideas Podcast",
        "https://rss2.flightcast.com/ordbkg8yojpehffas7vr7qpc.xml",
        "Greg Isenberg",
        0,
    ),
    PodcastFeed(
        "Y Combinator Startup Podcast",
        "https://anchor.fm/s/8c1524bc/podcast/rss",
        "Y Combinator",
        1,
    ),
    PodcastFeed(
        "A Product Market Fit Show | Startup Podcast for Founders",
        "https://rss.buzzsprout.com/1889238.rss",
        "Mistral.vc",
        2,
    ),
    PodcastFeed(
        "The SaaS Podcast - Real Lessons on Growing Profitable SaaS",
        "https://feeds.megaphone.fm/AHARO1075645324",
        "Omer Khan",
        3,
    ),
    PodcastFeed(
        "Pitch, Build, Scale",
        "https://anchor.fm/s/102d93688/podcast/rss",
        "Chris Fanchi",
        4,
    ),
    PodcastFeed(
        "The AI Ventures Podcast",
        "https://rss.buzzsprout.com/2613681.rss",
        "Adam Knees",
        5,
    ),
)

DEFAULT_WEATHER_AGRI_PODCAST_FEEDS: tuple[PodcastFeed, ...] = (
    PodcastFeed("WeatherBrains", "https://weatherbrains.libsyn.com/rss", "Big Brains Media", 0),
    PodcastFeed("Weather Geeks", "https://rss.art19.com/weather-geeks", "Weather Group Television", 1),
    PodcastFeed("The Climate Pod", "https://rss.libsyn.com/shows/191481/destinations/1344981.xml", "The Climate Pod", 2),
    PodcastFeed("Climate One", "https://feeds.megaphone.fm/CCC9544803627", "Commonwealth Club", 3),
    PodcastFeed("NOAA Ocean Podcast", "https://oceanservice.noaa.gov/rss/noaa-ocean-podcast.xml", "NOAA National Ocean Service", 4),
    PodcastFeed("AgriTalk", "https://www.omnycontent.com/d/playlist/1d004633-2e9e-47f1-9b89-a754011a2aa9/d3ba7031-c040-46d0-989d-a75800efd850/e50771fb-e99c-4188-8ef4-a75800f02002/podcast.rss", "AgriTalk", 5),
    PodcastFeed("Brownfield Ag News", "https://rss.art19.com/brownfield-ag-news", "Brownfield Ag News", 6),
    PodcastFeed("Agri-Pulse Daybreak", "https://www.agri-pulse.com/media/podcasts/109-agri-pulse-daybreak.rss", "Agri-Pulse", 7),
    PodcastFeed("Grain Markets and Other Stuff", "https://rss.buzzsprout.com/793022.rss", "Joe Vaclavik", 8),
    PodcastFeed("Successful Farming Podcast", "https://feeds.megaphone.fm/successful-farming-podcast", "Successful Farming", 9),
)

DEFAULT_SCIENCE_WELLNESS_PODCAST_FEEDS: tuple[PodcastFeed, ...] = (
    PodcastFeed("Huberman Lab", "https://feeds.megaphone.fm/hubermanlab", "Scicomm Media", 0),
    PodcastFeed("ZOE Science & Nutrition", "https://feeds.megaphone.fm/ZOELIMITED9301524082", "ZOE", 1),
    PodcastFeed("FoundMyFitness", "https://rss.libsyn.com/shows/51714/destinations/184296.xml", "Rhonda Patrick, Ph.D.", 2),
    PodcastFeed("Nutrition Facts with Dr. Greger", "https://nutritionfacts.org/audio/feed/podcast/", "Michael Greger, M.D. FACLM", 3),
    PodcastFeed("Sigma Nutrition Radio", "https://rss.libsyn.com/shows/53224/destinations/192888.xml", "Danny Lennon", 4),
    PodcastFeed("The Proof with Simon Hill", "https://feeds.megaphone.fm/IHINL7082362864", "Live better for longer", 5),
)

DEFAULT_AI_YOUTUBE_FEEDS: tuple[YouTubeFeed, ...] = (
    YouTubeFeed("YouTube | Lex Fridman", "UCSHZKyawb77ixDdsGog4iWA", "Lex Fridman", 0, "https://www.youtube.com/@lexfridman", "ai"),
    YouTubeFeed("YouTube | Dwarkesh Patel", "UCXl4i9dYBrFOabk0xGmbkRA", "Dwarkesh Patel", 1, "https://www.youtube.com/@DwarkeshPatel", "ai"),
    YouTubeFeed("YouTube | No Priors", "UCSI7h9hydQ40K5MJHnCrQvw", "No Priors", 2, "https://www.youtube.com/@NoPriorsPodcast", "ai"),
    YouTubeFeed("YouTube | Latent Space", "UCxBcwypKK-W3GHd_RZ9FZrQ", "Latent Space", 3, "https://www.youtube.com/@LatentSpacePod", "ai"),
    YouTubeFeed("YouTube | Cognitive Revolution", "UCjNRVMBVI30Sak_p6HRWhIA", "Cognitive Revolution", 4, "https://www.youtube.com/@CognitiveRevolutionPodcast", "ai"),
    YouTubeFeed("YouTube | AI Explained", "UCNJ1Ymd5yFuUPtn21xtRbbw", "AI Explained", 5, "https://www.youtube.com/@aiexplained-official", "ai"),
    YouTubeFeed("YouTube | Andrej Karpathy", "UCXUPKJO5MZQN11PqgIvyuvQ", "Andrej Karpathy", 6, "https://www.youtube.com/@AndrejKarpathy", "ai"),
    YouTubeFeed("YouTube | Matthew Berman", "UCawZsQWqfGSbCI5yjkdVkTA", "Matthew Berman", 7, "https://www.youtube.com/@matthew_berman", "ai"),
    YouTubeFeed("YouTube | a16z", "UC9cn0TuPq4dnbTY-CBsm8XA", "a16z", 8, "https://www.youtube.com/@a16z", "ai"),
)

DEFAULT_SCIENCE_WELLNESS_YOUTUBE_FEEDS: tuple[YouTubeFeed, ...] = (
    YouTubeFeed("YouTube | Huberman Lab", "UC2D2CMWXMOVWx7giW1n3LIg", "Andrew Huberman", 0, "https://www.youtube.com/@hubermanlab", "science_wellness"),
    YouTubeFeed("YouTube | Peter Attia MD", "UC8kGsMa0LygSX9nkBcBH1Sg", "Peter Attia MD", 1, "https://www.youtube.com/peterattiamd", "science_wellness"),
    YouTubeFeed("YouTube | FoundMyFitness", "UCWF8SqJVNlx-ctXbLswcTcA", "Rhonda Patrick", 2, "https://www.youtube.com/user/FoundMyFitness", "science_wellness"),
    YouTubeFeed("YouTube | ZOE", "UCa09am-cOsC-FSgr_nLkFFA", "ZOE", 3, "https://www.youtube.com/@joinZOE", "science_wellness"),
    YouTubeFeed("YouTube | NutritionFacts.org", "UCddn8dUxYdgJz3Qr5mjADtA", "NutritionFacts.org", 4, "https://www.youtube.com/@NutritionFactsOrg", "science_wellness"),
    YouTubeFeed("YouTube | The Proof with Simon Hill", "UChRSPNKMRhZxBmbivC8L_FA", "The Proof", 5, "https://www.youtube.com/@TheProofWithSimonHill", "science_wellness"),
)

DEFAULT_AI_XIAOYUZHOU_FEEDS: tuple[XiaoyuzhouFeed, ...] = (
    XiaoyuzhouFeed("小宇宙 | AI Odyssey", "https://www.xiaoyuzhoufm.com/podcast/6533d6ae5871431eb097dea7", "AI Odyssey", 0, "ai"),
    XiaoyuzhouFeed("小宇宙 | 人民公园说AI", "https://www.xiaoyuzhoufm.com/podcast/65257ff6e8ce9deaf70a65e9", "JustSayAI", 1, "ai"),
    XiaoyuzhouFeed("小宇宙 | AI炼金术", "https://www.xiaoyuzhoufm.com/podcast/63e9ef4de99bdef7d39944c8", "AI炼金术", 2, "ai"),
    XiaoyuzhouFeed("小宇宙 | AI每周谈", "https://www.xiaoyuzhoufm.com/podcast/688a34636f5a275f1cba40fd", "四维逻辑", 3, "ai"),
)

DEFAULT_AI_STARTUP_XIAOYUZHOU_FEEDS: tuple[XiaoyuzhouFeed, ...] = (
    XiaoyuzhouFeed(
        "小宇宙 | 十字路口 Crossing",
        "https://www.xiaoyuzhoufm.com/podcast/60502e253c92d4f62c2a9577",
        "十字路口 Crossing",
        0,
        "ai_startup",
    ),
)

DEFAULT_SCIENCE_WELLNESS_XIAOYUZHOU_FEEDS: tuple[XiaoyuzhouFeed, ...] = (
    XiaoyuzhouFeed("小宇宙 | 健康有方FM", "https://www.xiaoyuzhoufm.com/podcast/670e3d971985fe85db1fd06b", "澎湃新闻健康频道", 0, "science_wellness"),
    XiaoyuzhouFeed("小宇宙 | 漫谈健康", "https://www.xiaoyuzhoufm.com/podcast/68ec5982475ee5285f3dbd0c", "漫谈健康", 1, "science_wellness"),
    XiaoyuzhouFeed("小宇宙 | 食之有未", "https://www.xiaoyuzhoufm.com/podcast/65607489b0f6328de2cd7822", "MissGreen绿蔬蔬", 2, "science_wellness"),
)

WEATHER_AGRI_FEED_NAMES = {feed.name for feed in DEFAULT_WEATHER_AGRI_PODCAST_FEEDS}
AI_FEED_NAMES = {feed.name for feed in DEFAULT_AI_PODCAST_FEEDS + DEFAULT_AI_YOUTUBE_FEEDS + DEFAULT_AI_XIAOYUZHOU_FEEDS}
AI_STARTUP_FEED_NAMES = {
    feed.name
    for feed in DEFAULT_AI_STARTUP_PODCAST_FEEDS + DEFAULT_AI_STARTUP_XIAOYUZHOU_FEEDS
}
SCIENCE_WELLNESS_FEED_NAMES = {
    feed.name
    for feed in DEFAULT_SCIENCE_WELLNESS_PODCAST_FEEDS
    + DEFAULT_SCIENCE_WELLNESS_YOUTUBE_FEEDS
    + DEFAULT_SCIENCE_WELLNESS_XIAOYUZHOU_FEEDS
}

PODCAST_FEED_GROUPS: dict[str, tuple[PodcastFeed, ...]] = {
    "ai": DEFAULT_AI_PODCAST_FEEDS,
    "ai_startup": DEFAULT_AI_STARTUP_PODCAST_FEEDS,
    "weather_agri": DEFAULT_WEATHER_AGRI_PODCAST_FEEDS,
    "science_wellness": DEFAULT_SCIENCE_WELLNESS_PODCAST_FEEDS,
    "all": (
        DEFAULT_AI_PODCAST_FEEDS
        + DEFAULT_AI_STARTUP_PODCAST_FEEDS
        + DEFAULT_WEATHER_AGRI_PODCAST_FEEDS
        + DEFAULT_SCIENCE_WELLNESS_PODCAST_FEEDS
    ),
}
YOUTUBE_FEED_GROUPS: dict[str, tuple[YouTubeFeed, ...]] = {
    "ai": DEFAULT_AI_YOUTUBE_FEEDS,
    "ai_startup": (),
    "weather_agri": (),
    "science_wellness": DEFAULT_SCIENCE_WELLNESS_YOUTUBE_FEEDS,
    "all": DEFAULT_AI_YOUTUBE_FEEDS + DEFAULT_SCIENCE_WELLNESS_YOUTUBE_FEEDS,
}
XIAOYUZHOU_FEED_GROUPS: dict[str, tuple[XiaoyuzhouFeed, ...]] = {
    "ai": DEFAULT_AI_XIAOYUZHOU_FEEDS,
    "ai_startup": DEFAULT_AI_STARTUP_XIAOYUZHOU_FEEDS,
    "weather_agri": (),
    "science_wellness": DEFAULT_SCIENCE_WELLNESS_XIAOYUZHOU_FEEDS,
    "all": (
        DEFAULT_AI_XIAOYUZHOU_FEEDS
        + DEFAULT_AI_STARTUP_XIAOYUZHOU_FEEDS
        + DEFAULT_SCIENCE_WELLNESS_XIAOYUZHOU_FEEDS
    ),
}
PODCAST_GROUP_LABELS = {
    "ai": "AI 播客",
    "ai_startup": "AI创业",
    "weather_agri": "天气农业",
    "science_wellness": "科学养生",
    "all": "综合雷达",
}
TOPIC_FILTER_LABELS = {
    "all": "综合",
    "ai": "AI",
    "ai_startup": "AI创业",
    "weather_agri": "天气",
    "science_wellness": "养生",
}


def topic_filter_variant(button_topic: str, current_topic: str) -> str:
    """Return the visual variant for a radar topic filter button."""
    return "accent" if button_topic == current_topic else "secondary"


def topic_display_label(topic: str) -> str:
    if topic == "manual":
        return "手动链接"
    if topic == "favorites":
        return "我的精选"
    return TOPIC_FILTER_LABELS.get(topic, PODCAST_GROUP_LABELS.get(topic, "播客"))


def topic_download_title(topic: str) -> str:
    if topic == "all":
        return "今日雷达"
    if topic == "favorites":
        return "我的精选"
    return f"今日雷达 · {topic_display_label(topic)}"

FEED_PRIORITY_BY_NAME = {
    feed.name: feed.priority
    for feed in (
        DEFAULT_AI_PODCAST_FEEDS
        + DEFAULT_AI_STARTUP_PODCAST_FEEDS
        + DEFAULT_WEATHER_AGRI_PODCAST_FEEDS
        + DEFAULT_SCIENCE_WELLNESS_PODCAST_FEEDS
        + DEFAULT_AI_YOUTUBE_FEEDS
        + DEFAULT_SCIENCE_WELLNESS_YOUTUBE_FEEDS
        + DEFAULT_AI_XIAOYUZHOU_FEEDS
        + DEFAULT_AI_STARTUP_XIAOYUZHOU_FEEDS
        + DEFAULT_SCIENCE_WELLNESS_XIAOYUZHOU_FEEDS
    )
}

INVESTMENT_KEYWORDS: tuple[tuple[tuple[str, ...], float, str], ...] = (
    (("founder", "co-founder", "entrepreneur", "创始人", "创业者"), 1.0, "创始人一手经验"),
    (("revenue", "arr", "mrr", "pricing", "go-to-market", "商业化", "获客", "营收", "定价"), 1.1, "商业化验证"),
    (("product-market fit", "pmf", "bootstrapped", "solo founder", "0 to 1", "从0到1", "一人公司"), 0.9, "从 0 到 1"),
    (("nvidia", "gpu", "h100", "h200", "gb200", "blackwell", "accelerator"), 1.3, "AI 算力"),
    (("data center", "datacenter", "capex", "power", "electricity", "grid", "cooling"), 1.2, "数据中心/电力链"),
    (("openai", "anthropic", "claude", "gemini", "deepseek", "qwen", "llama"), 1.0, "核心模型"),
    (("agent", "agents", "workflow", "automation", "enterprise ai", "productivity"), 0.9, "AI 应用落地"),
    (("inference", "training", "token", "model cost", "compute cost"), 0.8, "推理/训练成本"),
    (("semiconductor", "chip", "memory", "hbm", "advanced packaging", "tsmc", "asml"), 1.1, "半导体链"),
    (("startup", "funding", "valuation", "ipo", "m&a", "acquisition"), 0.8, "一级/并购信号"),
    (("corn", "soy", "soybean", "wheat", "grain", "cotton", "sugar", "coffee", "cocoa", "rice"), 1.0, "农产品"),
    (("wasde", "usda", "export", "exports", "inventory", "stocks", "yield", "planting"), 1.1, "供需数据"),
    (("el nino", "el niño", "la nina", "la niña", "enso", "drought", "rainfall", "heat", "flood"), 1.2, "天气扰动"),
    (("brazil", "argentina", "china", "india", "black sea", "ukraine", "russia"), 0.8, "关键产区/贸易"),
    (("fertilizer", "potash", "phosphate", "urea", "ethanol", "diesel", "freight"), 0.8, "农化/成本"),
    (("longevity", "obesity", "glp-1", "sleep", "metabolic", "nutrition", "protein"), 0.7, "健康消费/医药"),
)


def format_duration(seconds) -> str:
    if seconds in (None, ""):
        return "-"
    seconds = int(seconds)
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    if h > 0:
        return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


def parse_count(value) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        if value < 0:
            return None
        return int(value)

    text = str(value).strip().lower()
    if not text or text in {"-", "none", "null", "nan"}:
        return None
    text = text.replace(",", "").replace("次播放", "").replace("播放", "").strip()

    multiplier = 1.0
    if "亿" in text:
        multiplier = 100000000.0
    elif "万" in text:
        multiplier = 10000.0
    elif "m" in text:
        multiplier = 1000000.0
    elif "k" in text:
        multiplier = 1000.0

    match = re.search(r"(\d+(?:\.\d+)?)", text)
    if not match:
        return None
    return int(float(match.group(1)) * multiplier)


def format_count(value: int | None) -> str:
    if value is None:
        return "未知"
    if value >= 100000000:
        return f"{value / 100000000:.1f}亿"
    if value >= 10000:
        return f"{value / 10000:.1f}万"
    return f"{value:,}"


def format_file_size(value: int) -> str:
    size = float(max(0, value))
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024 or unit == "GB":
            if unit == "B":
                return f"{int(size)} {unit}"
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} GB"


def extract_count_from_mapping(data: dict | None) -> int | None:
    if not isinstance(data, dict):
        return None
    direct_keys = (
        "view_count",
        "viewCount",
        "play_count",
        "playCount",
        "play_count_total",
        "playCountTotal",
        "plays",
        "listen_count",
        "listenCount",
        "listenedCount",
        "download_count",
        "downloadCount",
    )
    text_keys = (
        "view_count_text",
        "viewCountText",
        "play_count_text",
        "playCountText",
        "playsText",
        "listenCountText",
    )
    for key in direct_keys + text_keys:
        count = parse_count(data.get(key))
        if count is not None:
            return count
    for key in ("stats", "stat", "analytics", "episodeStats", "episodeStat", "playInfo"):
        count = extract_count_from_mapping(data.get(key))
        if count is not None:
            return count
    return None


def duration_minutes(duration_text: str) -> float | None:
    text = (duration_text or "").strip()
    if not text or text == "-":
        return None
    if ":" in text:
        try:
            parts = [int(part) for part in text.split(":")]
        except ValueError:
            return None
        seconds = 0
        for part in parts:
            seconds = seconds * 60 + part
        return seconds / 60
    match = re.search(r"(\d+(?:\.\d+)?)", text)
    if not match:
        return None
    return float(match.group(1))


def parse_item_date(value: str) -> datetime | None:
    text = (value or "").strip()
    if not text or text in {"-", "日期未知", "未知"}:
        return None
    for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%Y%m%d"):
        try:
            return datetime.strptime(text[:10], fmt).replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


def text_contains_keyword(text: str, keyword: str) -> bool:
    normalized_text = html.unescape(text or "").lower().replace("’", "'")
    normalized_keyword = keyword.lower()
    if normalized_keyword in {"ai", "llm", "arr", "mrr", "pmf", "mcp", "gtm"}:
        return bool(
            re.search(
                rf"(?<![a-z0-9]){re.escape(normalized_keyword)}(?![a-z0-9])",
                normalized_text,
            )
        )
    return normalized_keyword in normalized_text


def text_matches_any_keyword(text: str, keywords: tuple[str, ...]) -> bool:
    return any(text_contains_keyword(text, keyword) for keyword in keywords)


def strip_negated_ai_phrases(text: str) -> str:
    cleaned = re.sub(
        r"\b(?:without|no|not using|does not use|doesn't use)\s+(?:any\s+)?(?:ai|artificial intelligence)(?:\s+tools?)?\b",
        " ",
        text or "",
        flags=re.IGNORECASE,
    )
    return re.sub(r"(?:不使用|没有|无)\s*AI(?:\s*工具)?", " ", cleaned, flags=re.IGNORECASE)


def is_ai_startup_story_item(title: str, description: str = "", source_name: str = "") -> bool:
    title_text = strip_html_text(title)
    description_text = strip_html_text(description)
    description_focus = description_text[:700]
    combined_text = f"{title_text} {description_focus}".strip()
    title_ai_text = strip_negated_ai_phrases(title_text)
    combined_ai_text = strip_negated_ai_phrases(combined_text)

    title_has_ai = text_matches_any_keyword(title_ai_text, AI_STARTUP_AI_KEYWORDS)
    description_opening_has_ai = text_matches_any_keyword(
        strip_negated_ai_phrases(description_focus[:450]),
        AI_STARTUP_AI_KEYWORDS,
    )
    description_has_strong_story = text_matches_any_keyword(
        description_focus,
        AI_STARTUP_STRONG_STORY_KEYWORDS,
    )
    title_has_strong_story = text_matches_any_keyword(title_text, AI_STARTUP_STRONG_STORY_KEYWORDS)
    title_has_execution = text_matches_any_keyword(title_text, AI_STARTUP_TITLE_EXECUTION_KEYWORDS)
    title_looks_like_news = text_matches_any_keyword(title_text, AI_STARTUP_NEWS_KEYWORDS)

    if title_looks_like_news:
        return False
    if source_name in AI_STARTUP_NATIVE_STORY_FEED_NAMES:
        return text_matches_any_keyword(combined_ai_text, AI_STARTUP_AI_KEYWORDS) and text_matches_any_keyword(
            combined_text,
            AI_STARTUP_STORY_KEYWORDS,
        )
    return (
        (title_has_ai and title_has_execution)
        or (title_has_ai and description_has_strong_story)
        or (title_has_strong_story and description_opening_has_ai)
    )


def infer_item_topic(item: VideoItem) -> str:
    if item.source_name in AI_STARTUP_FEED_NAMES:
        return "ai_startup"
    if item.source_name in AI_FEED_NAMES:
        return "ai"
    if item.source_name in WEATHER_AGRI_FEED_NAMES:
        return "weather_agri"
    if item.source_name in SCIENCE_WELLNESS_FEED_NAMES:
        return "science_wellness"
    text = f"{item.source_name} {item.title} {item.description_text}".lower()
    if any(keyword in text for keyword in WEATHER_AGRI_TITLE_KEYWORDS):
        return "weather_agri"
    if is_ai_startup_story_item(item.title, item.description_text, item.source_name):
        return "ai_startup"
    if any(keyword in text for keyword in ("ai", "llm", "model", "gpu", "openai", "agent", "data center")):
        return "ai"
    if any(keyword in text for keyword in ("health", "nutrition", "sleep", "longevity", "protein", "metabolic")):
        return "science_wellness"
    return "manual"


def play_count_component(play_count: int | None) -> float:
    if play_count is None:
        return 5.0
    if play_count <= 0:
        return 0.5
    return max(0.5, min(10.0, 1.2 + math.log10(play_count + 1) * 1.8))


def investment_value_score(item: VideoItem) -> tuple[float, list[str]]:
    topic = infer_item_topic(item)
    score = {
        "ai": 5.4,
        "ai_startup": 5.2,
        "weather_agri": 5.8,
        "science_wellness": 3.8,
        "manual": 4.2,
    }.get(topic, 4.2)
    reasons = [PODCAST_GROUP_LABELS.get(topic, "手动来源")]

    text = f"{item.source_name} {item.title} {item.description_text}".lower()
    keyword_hits: list[str] = []
    keyword_bonus = 0.0
    for keywords, weight, label in INVESTMENT_KEYWORDS:
        if any(keyword in text for keyword in keywords):
            keyword_bonus += weight
            if label not in keyword_hits:
                keyword_hits.append(label)
    if keyword_hits:
        score += min(3.0, keyword_bonus)
        reasons.extend(keyword_hits[:3])

    priority = FEED_PRIORITY_BY_NAME.get(item.source_name)
    if priority is not None:
        source_bonus = max(0.0, 0.8 - min(priority, 10) * 0.07)
        score += source_bonus
        if source_bonus >= 0.35:
            reasons.append("高优先级来源")

    published_at = parse_item_date(item.published_text)
    if published_at:
        age_days = max(0, (datetime.now(timezone.utc) - published_at.astimezone(timezone.utc)).days)
        if age_days <= 3:
            score += 0.8
            reasons.append("近 3 日更新")
        elif age_days <= 14:
            score += 0.45
            reasons.append("近两周更新")
        elif age_days > 90:
            score -= 0.6
            reasons.append("时效偏旧")

    minutes = duration_minutes(item.duration_text)
    if minutes is not None:
        if 15 <= minutes <= 90:
            score += 0.35
        elif minutes < 8:
            score -= 0.25
        elif minutes > 150:
            score -= 0.25

    return max(0.0, min(10.0, score)), reasons


def update_item_importance_score(item: VideoItem) -> None:
    investment_score, reasons = investment_value_score(item)
    heat_score = play_count_component(item.play_count)
    if item.play_count is None:
        total = investment_score
    else:
        total = heat_score * 0.42 + investment_score * 0.58
    item.investment_score = round(investment_score, 1)
    item.importance_score = round(max(0.0, min(10.0, total)), 1)
    play_reason = f"播放量 {format_count(item.play_count)}" if item.play_count is not None else "播放量未知"
    reason_text = " / ".join([play_reason, f"投资价值 {item.investment_score:.1f}", *reasons[:4]])
    item.importance_reason = reason_text[:160]
    item.scored_at = datetime.now(timezone.utc).isoformat()


def refresh_youtube_metrics(item: VideoItem) -> None:
    if item.source_type != "youtube" or not item.webpage_url:
        return
    opts = {
        "skip_download": True,
        "quiet": True,
        "no_warnings": True,
        "socket_timeout": 15,
        "extractor_retries": 1,
        "extractor_args": {"youtube": {"player_client": ["web"]}},
    }
    with YoutubeDL(opts) as ydl:
        info = ydl.extract_info(item.webpage_url, download=False)
    play_count = extract_count_from_mapping(info)
    if play_count is not None:
        item.play_count = play_count
    if not item.source_name:
        item.source_name = str(info.get("channel") or info.get("uploader") or "").strip()
    if not item.artwork_url:
        item.artwork_url = str(info.get("thumbnail") or "").strip()
    if not item.description_text:
        item.description_text = strip_html_text(info.get("description") or info.get("summary") or "")
    if not item.published_text:
        upload_date = str(info.get("upload_date") or "").strip()
        if len(upload_date) == 8 and upload_date.isdigit():
            item.published_text = f"{upload_date[:4]}-{upload_date[4:6]}-{upload_date[6:]}"


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def safe_unlink_generated_cache(path: Path, allowed_root: Path) -> bool:
    """Delete only app-generated cache files inside the expected cache root."""
    try:
        resolved_path = path.expanduser().resolve()
        resolved_root = allowed_root.expanduser().resolve()
        resolved_path.relative_to(resolved_root)
    except (OSError, ValueError):
        return False
    if not resolved_path.is_file():
        return False
    try:
        resolved_path.unlink()
    except OSError:
        return False
    return True


def safe_unlink_tts_cache_artifact(path: Path) -> bool:
    deleted = safe_unlink_generated_cache(path, RESEARCH_TTS_DIR)
    companion_text = path.expanduser().with_suffix(".txt")
    safe_unlink_generated_cache(companion_text, RESEARCH_TTS_DIR)
    return deleted


def unique_preserve_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    unique: list[str] = []
    for value in values:
        cleaned = (value or "").strip()
        if not cleaned or cleaned in seen:
            continue
        seen.add(cleaned)
        unique.append(cleaned)
    return unique


def build_retry_session(total_retries: int = 2) -> requests.Session:
    retry = Retry(
        total=total_retries,
        connect=total_retries,
        read=total_retries,
        status=total_retries,
        backoff_factor=0.8,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset({"GET", "HEAD"}),
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry, pool_connections=4, pool_maxsize=8)
    session = requests.Session()
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    session.headers.update({"User-Agent": REQUEST_USER_AGENT})
    return session


def audio_request_headers(item: VideoItem | None = None) -> dict[str, str]:
    referer = (item.webpage_url if item else "") or "https://podcasters.spotify.com/"
    return {
        "User-Agent": REQUEST_USER_AGENT,
        "Accept": "audio/*,video/*,application/octet-stream,*/*",
        "Accept-Language": "en-US,en;q=0.9,zh-CN;q=0.8,zh;q=0.7",
        "Connection": "close",
        "Referer": referer,
    }


def response_content_type(response: requests.Response) -> str:
    return (response.headers.get("content-type") or "").split(";", 1)[0].strip().lower()


def is_textish_audio_response(response: requests.Response) -> bool:
    content_type = response_content_type(response)
    if not content_type:
        return False
    if content_type in TEXTISH_AUDIO_CONTENT_TYPES:
        return True
    return content_type.startswith("text/") and content_type not in {"text/vtt"}


def content_type_audio_extension(content_type: str) -> str:
    mapping = {
        "audio/aac": "aac",
        "audio/flac": "flac",
        "audio/m4a": "m4a",
        "audio/mp4": "m4a",
        "audio/mpeg": "mp3",
        "audio/ogg": "ogg",
        "audio/opus": "opus",
        "audio/wav": "wav",
        "audio/x-m4a": "m4a",
        "audio/x-wav": "wav",
        "video/mp4": "mp4",
    }
    return mapping.get(content_type.split(";", 1)[0].strip().lower(), "")


def resolved_audio_extension(url: str, content_type: str = "") -> str:
    ext = detect_audio_extension(url)
    if ext in KNOWN_AUDIO_EXTENSIONS:
        return ext
    content_ext = content_type_audio_extension(content_type)
    if content_ext:
        return content_ext
    return "m4a"


def is_ffprobe_available() -> bool:
    return resolve_sibling_tool("ffprobe") is not None


def is_readable_audio_cache(path: Path) -> bool:
    if not is_valid_audio_cache(path):
        return False
    if not is_ffprobe_available():
        return True
    return probe_audio_duration_seconds(path) is not None


def validate_audio_cache(path: Path, label: str = "音频") -> None:
    if not is_valid_audio_cache(path):
        raise RuntimeError(f"{label}为空或不完整")
    if is_ffprobe_available() and probe_audio_duration_seconds(path) is None:
        raise RuntimeError(f"{label}不是可识别的音频文件")


def compact_command_output(text: str, max_chars: int = 260) -> str:
    lines = [line.strip() for line in (text or "").splitlines() if line.strip()]
    compact = " | ".join(lines[-4:] or lines)
    return clean_ui_status(compact or "未知错误", max_chars)


def ensure_qwen_tts_bridge_script() -> Path:
    ensure_dir(APP_DIR / ".cache")
    script_path = APP_DIR / ".cache" / "qwen3_tts_bridge.py"
    if not script_path.exists() or script_path.read_text(encoding="utf-8", errors="ignore") != QWEN_TTS_BRIDGE_SOURCE:
        script_path.write_text(QWEN_TTS_BRIDGE_SOURCE, encoding="utf-8")
    return script_path


def parse_json_line(text: str) -> dict | None:
    for line in reversed((text or "").splitlines()):
        line = line.strip()
        if not line.startswith("{") or not line.endswith("}"):
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            return payload
    return None


def is_audio_result_path(path: Path) -> bool:
    return path.suffix.lower() in {".wav", ".mp3", ".m4a", ".aac", ".flac", ".ogg"}


def browser_is_available(browser: str) -> bool:
    browser = browser.strip().lower()
    system = platform.system()
    if system == "Darwin":
        candidates = {
            "chrome": [
                Path("/Applications/Google Chrome.app"),
                Path("~/Applications/Google Chrome.app").expanduser(),
            ],
            "edge": [
                Path("/Applications/Microsoft Edge.app"),
                Path("~/Applications/Microsoft Edge.app").expanduser(),
            ],
            "firefox": [
                Path("/Applications/Firefox.app"),
                Path("~/Applications/Firefox.app").expanduser(),
            ],
        }
        return any(path.exists() for path in candidates.get(browser, []))
    command_names = {
        "chrome": ["google-chrome", "google-chrome-stable", "chrome", "chromium"],
        "edge": ["microsoft-edge", "microsoft-edge-stable", "msedge"],
        "firefox": ["firefox"],
    }
    return any(shutil.which(command) for command in command_names.get(browser, []))


def available_browser_cookie_modes() -> list[str]:
    return [browser for browser in DEFAULT_BROWSER_ORDER if browser_is_available(browser)]


def default_browser_cookie_mode() -> str:
    modes = available_browser_cookie_modes()
    return modes[0] if modes else "none"


def create_rounded_rect(
    canvas: tk.Canvas,
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    radius: int,
    **kwargs,
) -> int:
    radius = min(radius, (x2 - x1) // 2, (y2 - y1) // 2)
    points = [
        x1 + radius,
        y1,
        x2 - radius,
        y1,
        x2,
        y1,
        x2,
        y1 + radius,
        x2,
        y2 - radius,
        x2,
        y2,
        x2 - radius,
        y2,
        x1 + radius,
        y2,
        x1,
        y2,
        x1,
        y2 - radius,
        x1,
        y1 + radius,
        x1,
        y1,
    ]
    return canvas.create_polygon(points, smooth=True, splinesteps=16, **kwargs)


PIXEL_DINO_FRAMES: tuple[tuple[str, ...], ...] = (
    (
        "        ####",
        "       ##  ###",
        "       #######",
        "       ###    ",
        "#     #####   ",
        "##   #######  ",
        "###########   ",
        " #########    ",
        "  #######     ",
        "   ##  ##     ",
        "   #    ##    ",
    ),
    (
        "        ####",
        "       ##  ###",
        "       #######",
        "       ###    ",
        "#     #####   ",
        "##   #######  ",
        "###########   ",
        " #########    ",
        "  #######     ",
        "   ##  ##     ",
        "  ##    #     ",
    ),
)


def progress_marker_position(
    width: float,
    value: float,
    maximum: float,
    marker_width: float,
    padding: float = 4.0,
) -> float:
    """Position a marker within a progress track and clamp out-of-range values."""
    usable_width = max(0.0, float(width) - marker_width - padding * 2)
    if maximum <= 0:
        ratio = 0.0
    else:
        ratio = max(0.0, min(1.0, float(value) / float(maximum)))
    return padding + usable_width * ratio


class RoundedDinoProgressBar(tk.Canvas):
    """Rounded progress track with a tiny skateboarding pixel dinosaur."""

    TRACK_COLOR = "#e2e5eb"
    TRAIL_COLOR = "#b9dcff"
    DINO_COLOR = "#1d1d1f"
    DINO_EYE_COLOR = "#a8ff35"
    SKATEBOARD_COLOR = "#a8ff35"
    SKATEBOARD_EDGE_COLOR = "#5edfff"
    SKATEBOARD_WHEEL_COLOR = "#af52de"
    CANVAS_HEIGHT = 28
    MARKER_WIDTH = 28.0
    MARKER_PADDING = 4.0

    def __init__(
        self,
        parent: tk.Misc,
        *,
        variable: tk.DoubleVar,
        maximum: float = 100,
        mode: str = "determinate",
        bg: str = COLOR_SURFACE,
    ) -> None:
        super().__init__(
            parent,
            height=self.CANVAS_HEIGHT,
            bg=bg,
            highlightthickness=0,
            bd=0,
            relief="flat",
        )
        self._variable = variable
        self._maximum = max(1.0, float(maximum))
        self._mode = mode
        self._explicit_running = False
        self._animation_job: str | None = None
        self._interval_ms = 84
        self._frame_index = 0
        self._indeterminate_x = self.MARKER_PADDING
        self._indeterminate_direction = 1
        self._trace_id = self._variable.trace_add("write", self._on_variable_changed)
        self.bind("<Configure>", self._on_resize)
        self.after_idle(self._render)

    def configure(self, cnf=None, **kwargs):
        options: dict = {}
        if isinstance(cnf, dict):
            options.update(cnf)
        options.update(kwargs)
        mode = options.pop("mode", None)
        maximum = options.pop("maximum", None)
        if mode is not None:
            if mode not in {"determinate", "indeterminate"}:
                raise tk.TclError(f"unsupported progress mode: {mode}")
            if mode != self._mode:
                self._mode = mode
                self._indeterminate_x = self.MARKER_PADDING
                self._indeterminate_direction = 1
        if maximum is not None:
            self._maximum = max(1.0, float(maximum))
        result = super().configure(**options) if options else None
        self._render()
        self._ensure_animation()
        return result

    config = configure

    def start(self, interval: int | float | None = None) -> None:
        if interval is not None:
            # ttk uses very short intervals by default; pixel legs remain clearer
            # and use less CPU at roughly 12 frames per second.
            self._interval_ms = max(70, int(interval))
        self._explicit_running = True
        self._ensure_animation()
        self._render()

    def stop(self) -> None:
        self._explicit_running = False
        self._cancel_animation()
        self._render()

    def destroy(self) -> None:
        self._cancel_animation()
        try:
            self._variable.trace_remove("write", self._trace_id)
        except (tk.TclError, ValueError):
            pass
        super().destroy()

    def _on_resize(self, _event=None) -> None:
        if self._mode == "indeterminate":
            max_x = self._indeterminate_max_x()
            self._indeterminate_x = max(
                self.MARKER_PADDING,
                min(max_x, self._indeterminate_x),
            )
        self._render()

    def _on_variable_changed(self, *_args) -> None:
        self._render()
        self._ensure_animation()

    def _current_value(self) -> float:
        try:
            return float(self._variable.get())
        except (tk.TclError, TypeError, ValueError):
            return 0.0

    def _determinate_running(self) -> bool:
        value = self._current_value()
        return self._mode == "determinate" and 0.0 < value < self._maximum

    def _should_animate(self) -> bool:
        return self._explicit_running or self._determinate_running()

    def _ensure_animation(self) -> None:
        if not self._should_animate():
            self._cancel_animation()
            return
        if self._animation_job is None:
            self._animation_job = self.after(self._interval_ms, self._animate)

    def _cancel_animation(self) -> None:
        if self._animation_job is None:
            return
        try:
            self.after_cancel(self._animation_job)
        except tk.TclError:
            pass
        self._animation_job = None

    def _animate(self) -> None:
        self._animation_job = None
        self._frame_index = (self._frame_index + 1) % len(PIXEL_DINO_FRAMES)
        if self._mode == "indeterminate" and self._explicit_running:
            self._move_indeterminate_dino()
        self._render()
        self._ensure_animation()

    def _indeterminate_max_x(self) -> float:
        width = max(float(self.winfo_width()), 80.0)
        return max(
            self.MARKER_PADDING,
            width - self.MARKER_WIDTH - self.MARKER_PADDING,
        )

    def _move_indeterminate_dino(self) -> None:
        min_x = self.MARKER_PADDING
        max_x = self._indeterminate_max_x()
        travel_step = max(3.0, min(8.0, float(self.winfo_width()) / 140.0))
        self._indeterminate_x += travel_step * self._indeterminate_direction
        if self._indeterminate_x >= max_x:
            self._indeterminate_x = max_x
            self._indeterminate_direction = -1
        elif self._indeterminate_x <= min_x:
            self._indeterminate_x = min_x
            self._indeterminate_direction = 1

    def _render(self) -> None:
        if not self.winfo_exists():
            return
        width = max(float(self.winfo_width()), 80.0)
        height = max(float(self.winfo_height()), float(self.CANVAS_HEIGHT))
        self.delete("all")

        track_x1 = 1.0
        track_x2 = width - 1.0
        track_height = 10.0
        track_y1 = (height - track_height) / 2
        track_y2 = track_y1 + track_height
        create_rounded_rect(
            self,
            track_x1,
            track_y1,
            track_x2,
            track_y2,
            5,
            fill=self.TRACK_COLOR,
            outline="",
        )

        value = self._current_value()
        if self._mode == "indeterminate":
            marker_x = self._indeterminate_x
            self._draw_indeterminate_trail(
                marker_x,
                track_y1,
                track_y2,
                width,
            )
            show_dino = self._explicit_running
            direction = self._indeterminate_direction
        else:
            marker_x = progress_marker_position(
                width,
                value,
                self._maximum,
                self.MARKER_WIDTH,
                self.MARKER_PADDING,
            )
            ratio = max(0.0, min(1.0, value / self._maximum))
            fill_x2 = track_x1 + (track_x2 - track_x1) * ratio
            if fill_x2 > track_x1:
                create_rounded_rect(
                    self,
                    track_x1,
                    track_y1,
                    fill_x2,
                    track_y2,
                    5,
                    fill=COLOR_BLUE,
                    outline="",
                )
            show_dino = value > 0.0 or self._explicit_running
            direction = 1

        if show_dino:
            marker_y = max(1.0, (height - 21.0) / 2)
            self._draw_pixel_dino(
                marker_x + 4.0,
                marker_y,
                direction,
            )
            self._draw_pixel_skateboard(
                marker_x,
                marker_y + 15.5,
                direction,
            )

    def _draw_indeterminate_trail(
        self,
        marker_x: float,
        track_y1: float,
        track_y2: float,
        width: float,
    ) -> None:
        trail_width = max(34.0, min(72.0, width * 0.09))
        if self._indeterminate_direction > 0:
            trail_x1 = max(1.0, marker_x - trail_width)
            trail_x2 = min(width - 1.0, marker_x + self.MARKER_WIDTH * 0.75)
        else:
            trail_x1 = max(1.0, marker_x + self.MARKER_WIDTH * 0.25)
            trail_x2 = min(width - 1.0, marker_x + self.MARKER_WIDTH + trail_width)
        if trail_x2 > trail_x1:
            create_rounded_rect(
                self,
                trail_x1,
                track_y1,
                trail_x2,
                track_y2,
                5,
                fill=self.TRAIL_COLOR,
                outline="",
            )

    def _draw_pixel_dino(self, x: float, y: float, direction: int) -> None:
        frame = PIXEL_DINO_FRAMES[self._frame_index]
        pixel_size = 1.35
        width_cells = max(len(row) for row in frame)
        for row_index, row in enumerate(frame):
            for column_index, cell in enumerate(row):
                if cell != "#":
                    continue
                visual_column = (
                    column_index
                    if direction >= 0
                    else width_cells - column_index - 1
                )
                x1 = x + visual_column * pixel_size
                y1 = y + row_index * pixel_size
                self.create_rectangle(
                    x1,
                    y1,
                    x1 + pixel_size + 0.2,
                    y1 + pixel_size + 0.2,
                    fill=self.DINO_COLOR,
                    outline="",
                )

        eye_column = 10 if direction >= 0 else width_cells - 11
        eye_x = x + eye_column * pixel_size
        eye_y = y + pixel_size
        self.create_rectangle(
            eye_x,
            eye_y,
            eye_x + pixel_size * 0.72,
            eye_y + pixel_size * 0.72,
            fill=self.DINO_EYE_COLOR,
            outline="",
        )

    def _draw_pixel_skateboard(
        self,
        x: float,
        y: float,
        direction: int,
    ) -> None:
        board_x1 = x + 1.0
        board_x2 = x + self.MARKER_WIDTH - 1.0
        create_rounded_rect(
            self,
            board_x1,
            y,
            board_x2,
            y + 3.2,
            2,
            fill=self.SKATEBOARD_COLOR,
            outline="",
        )

        nose_x1 = board_x2 - 4.0 if direction >= 0 else board_x1
        nose_x2 = board_x2 if direction >= 0 else board_x1 + 4.0
        self.create_rectangle(
            nose_x1,
            y,
            nose_x2,
            y + 1.2,
            fill=self.SKATEBOARD_EDGE_COLOR,
            outline="",
        )

        wheel_y = y + 4.2
        wheel_bob = 0.7 if self._frame_index else 0.0
        for wheel_x in (x + 6.0, x + self.MARKER_WIDTH - 6.0):
            self.create_oval(
                wheel_x - 1.7,
                wheel_y - 1.1 + wheel_bob,
                wheel_x + 1.7,
                wheel_y + 2.3 + wheel_bob,
                fill=self.SKATEBOARD_WHEEL_COLOR,
                outline="",
            )


class YTAudioDownloaderApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title(f"{APP_NAME} v{APP_VERSION}")
        self.root.geometry("1280x820")
        self.root.minsize(1080, 700)

        self.events: queue.Queue = queue.Queue()
        self.video_items: list[VideoItem] = []
        self.video_map: dict[str, VideoItem] = {}
        self.selected_item_ids: set[str] = set()
        self.card_widgets: dict[str, dict[str, tk.Widget]] = {}
        self.card_images: list[tk.PhotoImage] = []
        self.card_image_map: dict[str, tk.PhotoImage] = {}
        self.artwork_loading: set[str] = set()
        self.current_source_label = ""
        self.current_topic = "all"
        self.focused_item_id = ""
        self.card_columns = 0
        self.research_library: dict = {}
        self.favorite_item_ids: set[str] = set()
        self.detail_windows: dict[str, dict[str, object]] = {}
        self.cards_area: tk.Frame | None = None
        self.detail_canvas: tk.Canvas | None = None
        self.detail_content: tk.Frame | None = None
        self.detail_panel_visible = False
        self.fetching = False
        self.downloading = False
        self.researching = False
        self.stop_requested = False
        self.transcribing = False
        self.transcription_stop_requested = False
        self.auto_digests_running = False
        self.full_translation_inflight: dict[str, dict[str, object]] = {}
        self.translation_request_serial = 0
        self.latest_translation_request_id = 0
        self.reader_navigation_serial = 0
        self.transcription_paths: list[Path] = []
        self.transcription_rows: dict[Path, str] = {}
        self.transcription_statuses: dict[Path, str] = {}
        self.result_rows: dict[str, dict[str, str]] = {}
        self.current_page_key = ""
        self.results_dirty = True
        self.results_cache_ready = False
        self.results_records_cache: list[dict[str, object]] = []
        self.results_refresh_job: str | None = None
        self.results_render_job: str | None = None
        self.results_refresh_running = False
        self.results_refresh_pending = False
        self.results_render_records: list[dict[str, object]] = []
        self.results_render_index = 0
        self.results_refresh_generation = 0
        self.results_cache_version = 0
        self.results_render_target_version = 0
        self.results_rendered_cache_version = 0
        self.results_summary_cache = ""
        self.research_library_save_job: str | None = None
        self.research_library_dirty = False
        self.page_content_flush_job: str | None = None
        self.event_processing_blocked_until = 0.0
        self.download_render_dirty = False
        self.cards_render_job: str | None = None
        self.cards_render_plan: list[tuple[str, object]] = []
        self.cards_render_index = 0
        self.cards_render_row = 0
        self.dirty_card_ids: set[str] = set()
        self.pending_artwork_ids: set[str] = set()
        self.pending_transcription_visuals: dict[Path, tuple[str, str]] = {}
        self.pending_transcription_logs: list[str] = []
        self.restart_transcription_when_idle = False
        self.auto_transcribe_path_map: dict[Path, str] = {}
        self.pending_research_item_ids: set[str] = set()
        self.manual_link_expanded = False
        self.topic_filter_buttons: dict[str, tk.Canvas] = {}
        self.refresh_indicator_hide_job: str | None = None
        self.refresh_owns_progress = False

        self.url_var = tk.StringVar()
        self.download_dir_var = tk.StringVar()
        self.audio_format_var = tk.StringVar(value="m4a")
        self.cookie_file_var = tk.StringVar()
        self.browser_cookie_var = tk.StringVar(value=default_browser_cookie_mode())
        self.auto_transcribe_var = tk.BooleanVar(value=True)
        self.api_key_var = tk.StringVar()
        self.transcription_mode_var = tk.StringVar(value="本地优先")
        self.target_segment_seconds_var = tk.StringVar(value="120")
        self.max_segment_seconds_var = tk.StringVar(value="180")
        self.transcription_workers_var = tk.StringVar(value="4")
        self.status_var = tk.StringVar(value="准备就绪")
        self.summary_var = tk.StringVar(value="还没有读取任何节目")
        self.refresh_indicator_var = tk.StringVar(value="")
        self.current_task_var = tk.StringVar(value="当前任务：无")
        self.results_summary_var = tk.StringVar(value="还没有生成转录或纪要")
        self.reader_title_var = tk.StringVar(value="选择一条资料开始阅读")
        self.reader_meta_var = tk.StringVar(value="资料库里的 TXT 转录和 ZH 纪要会在这里直接打开")
        self.reader_path_var = tk.StringVar(value="")
        self.reader_current_path: Path | None = None
        self.reader_current_result_type = ""
        self.reader_translation_source_path: Path | None = None
        self.reader_translation_status_var = tk.StringVar(value="中文全文：打开英文转录后可按需生成")
        self.reader_tts_audio_path: Path | None = None
        self.reader_tts_source_path: Path | None = None
        self.reader_tts_scope = ""
        self.reader_tts_text_sha256 = ""
        self.tts_running = False
        self.tts_play_process: subprocess.Popen | None = None
        self.tts_generation_process: subprocess.Popen | None = None
        self.tts_generation_lock = threading.Lock()
        self.app_closing = False
        self.tts_python_var = tk.StringVar(value=str(DEFAULT_TTS_PYTHON))
        self.tts_model_var = tk.StringVar(value=DEFAULT_TTS_MODEL)
        self.tts_speaker_var = tk.StringVar(value=DEFAULT_TTS_SPEAKER)
        self.tts_language_var = tk.StringVar(value=DEFAULT_TTS_LANGUAGE)
        self.tts_instruct_var = tk.StringVar(value="")
        self.tts_max_chars_var = tk.StringVar(value=str(DEFAULT_TTS_MAX_CHARS))
        self.tts_status_var = tk.StringVar(value="本地朗读：空闲")

        self.download_title_var = tk.StringVar(value="今日雷达")
        self.selected_count_var = tk.StringVar(value="已选 0 / 0")
        self.progress_percent_var = tk.StringVar(value="0%")
        self.progress_detail_var = tk.StringVar(value="等待开始")

        self.progress_value = tk.DoubleVar(value=0)

        self.load_settings()
        self.load_research_library()
        self.prefill_initial_cached_radar()
        self.build_ui()
        self.root.protocol("WM_DELETE_WINDOW", self.on_app_close)
        self.root.after(120, self.process_events)
        self.root.after(AUTO_FETCH_DELAY_MS, self.fetch_all_updates)

    def load_settings(self) -> None:
        settings = {}
        if SETTINGS_PATH.exists():
            try:
                settings = json.loads(SETTINGS_PATH.read_text(encoding="utf-8"))
            except Exception:
                settings = {}

        default_dir = settings.get("download_dir") or str(DEFAULT_DOWNLOAD_DIR)
        self.download_dir_var.set(default_dir)
        self.audio_format_var.set(settings.get("audio_format") or "m4a")
        self.cookie_file_var.set(settings.get("cookie_file") or "")
        saved_browser_mode = settings.get("browser_cookies") or default_browser_cookie_mode()
        if saved_browser_mode in {"chrome", "edge", "firefox"} and not browser_is_available(saved_browser_mode):
            saved_browser_mode = default_browser_cookie_mode()
        self.browser_cookie_var.set(saved_browser_mode)
        self.auto_transcribe_var.set(bool(settings.get("auto_transcribe", True)))
        saved_transcription_strategy = str(settings.get("transcription_strategy") or "local_first")
        self.transcription_mode_var.set(
            TRANSCRIPTION_MODE_LABELS.get(saved_transcription_strategy, "本地优先")
        )
        self.target_segment_seconds_var.set(str(settings.get("target_segment_seconds") or "120"))
        self.max_segment_seconds_var.set(str(settings.get("max_segment_seconds") or "180"))
        self.transcription_workers_var.set(str(settings.get("transcription_workers") or "4"))
        self.tts_python_var.set(str(settings.get("tts_python") or DEFAULT_TTS_PYTHON))
        self.tts_model_var.set(str(settings.get("tts_model") or DEFAULT_TTS_MODEL))
        self.tts_speaker_var.set(str(settings.get("tts_speaker") or DEFAULT_TTS_SPEAKER))
        self.tts_language_var.set(str(settings.get("tts_language") or DEFAULT_TTS_LANGUAGE))
        self.tts_instruct_var.set(str(settings.get("tts_instruct") or ""))
        self.tts_max_chars_var.set(str(settings.get("tts_max_chars") or DEFAULT_TTS_MAX_CHARS))
        self.api_key_var.set(load_api_key(DOTENV_PATH))
        ensure_dir(Path(default_dir))

    def save_settings(self) -> None:
        payload = {
            "download_dir": self.download_dir_var.get().strip(),
            "audio_format": self.audio_format_var.get().strip(),
            "cookie_file": self.cookie_file_var.get().strip(),
            "browser_cookies": self.browser_cookie_var.get().strip(),
            "auto_transcribe": self.auto_transcribe_var.get(),
            "transcription_strategy": TRANSCRIPTION_MODE_OPTIONS.get(
                self.transcription_mode_var.get(),
                "local_first",
            ),
            "target_segment_seconds": self.target_segment_seconds_var.get().strip(),
            "max_segment_seconds": self.max_segment_seconds_var.get().strip(),
            "transcription_workers": self.transcription_workers_var.get().strip(),
            "tts_python": self.tts_python_var.get().strip(),
            "tts_model": self.tts_model_var.get().strip(),
            "tts_speaker": self.tts_speaker_var.get().strip(),
            "tts_language": self.tts_language_var.get().strip(),
            "tts_instruct": self.tts_instruct_var.get().strip(),
            "tts_max_chars": self.tts_max_chars_var.get().strip(),
        }
        SETTINGS_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def load_research_library(self) -> None:
        default_payload = {"version": 1, "items": {}}
        if not RESEARCH_LIBRARY_PATH.exists():
            self.research_library = default_payload
            self.favorite_item_ids = set()
            return
        try:
            payload = json.loads(RESEARCH_LIBRARY_PATH.read_text(encoding="utf-8"))
        except Exception:
            payload = default_payload
        if not isinstance(payload, dict):
            payload = default_payload
        items = payload.get("items")
        if not isinstance(items, dict):
            payload["items"] = {}
        self.research_library = payload
        self.favorite_item_ids = {
            item_id
            for item_id, record in self.research_library.get("items", {}).items()
            if isinstance(record, dict) and record.get("favorite")
        }

    def save_research_library(self) -> None:
        self.research_library_dirty = True
        if self.research_library_save_job is not None:
            try:
                self.root.after_cancel(self.research_library_save_job)
            except tk.TclError:
                pass
        self.research_library_save_job = self.root.after(500, self.flush_research_library)

    def flush_research_library(self) -> None:
        self.research_library_save_job = None
        if not self.research_library_dirty:
            return
        try:
            RESEARCH_LIBRARY_PATH.write_text(
                json.dumps(self.research_library, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except OSError as exc:
            self.research_library_dirty = True
            self.set_status(f"资料库保存失败：{clean_ui_status(str(exc), 160)}")
            return
        self.research_library_dirty = False

    def on_app_close(self) -> None:
        self.app_closing = True
        if self.refresh_indicator_hide_job is not None:
            try:
                self.root.after_cancel(self.refresh_indicator_hide_job)
            except tk.TclError:
                pass
            self.refresh_indicator_hide_job = None
        if self.research_library_save_job is not None:
            try:
                self.root.after_cancel(self.research_library_save_job)
            except tk.TclError:
                pass
            self.research_library_save_job = None
        self.flush_research_library()
        self.stop_reader_tts_audio(update_status=False)
        self.stop_tts_generation()
        self.root.destroy()

    def prefill_initial_cached_radar(self) -> None:
        cached_items = self.cached_podcast_items("all")
        if not cached_items:
            return
        self.current_topic = "all"
        self.current_source_label = "综合雷达本地快照"
        self.video_items = cached_items
        for item in self.video_items:
            update_item_importance_score(item)
        self.video_map = {item.video_id: item for item in self.video_items}
        self.summary_var.set(f"快照：综合雷达本地快照 | 可用节目 {len(self.video_items)} 个")
        self.status_var.set("已先加载本地快照，正在后台刷新最新节目...")
        self.current_task_var.set("当前任务：等待后台刷新")
        self.update_selected_count()

    def configure_ttk_style(self) -> None:
        style = ttk.Style(self.root)
        try:
            style.theme_use("clam")
        except Exception:
            pass
        style.configure("TNotebook", background=COLOR_BG, borderwidth=0)
        style.configure(
            "TNotebook.Tab",
            background=COLOR_BG,
            foreground=COLOR_MUTED,
            borderwidth=0,
            padding=(18, 9),
            font=(FONT_UI, 11),
        )
        style.map(
            "TNotebook.Tab",
            background=[("selected", COLOR_SURFACE)],
            foreground=[("selected", COLOR_TEXT)],
        )
        style.configure(
            "Treeview",
            background=COLOR_SURFACE,
            fieldbackground=COLOR_SURFACE,
            foreground=COLOR_TEXT,
            borderwidth=0,
            rowheight=30,
            font=(FONT_UI, 10),
        )
        style.configure(
            "Treeview.Heading",
            background=COLOR_SURFACE_ALT,
            foreground=COLOR_MUTED,
            relief="flat",
            font=(FONT_UI, 10, "bold"),
        )
        style.map("Treeview", background=[("selected", "#d8ebff")], foreground=[("selected", COLOR_TEXT)])
        style.configure(
            "Horizontal.TProgressbar",
            troughcolor=COLOR_BORDER,
            background=COLOR_BLUE,
            bordercolor=COLOR_BORDER,
            lightcolor=COLOR_BLUE,
            darkcolor=COLOR_BLUE,
        )
        style.configure("TCombobox", padding=(8, 5), font=(FONT_UI, 10))

    def ui_label(
        self,
        parent: tk.Misc,
        text: str | None = None,
        *,
        textvariable: tk.StringVar | None = None,
        size: int = 10,
        weight: str = "normal",
        fg: str = COLOR_TEXT,
        bg: str = COLOR_BG,
    ) -> tk.Label:
        return tk.Label(
            parent,
            text=text,
            textvariable=textvariable,
            fg=fg,
            bg=bg,
            font=(FONT_UI, size, weight),
        )

    def ui_entry(self, parent: tk.Misc, variable: tk.StringVar, *, width: int = 40, show: str = "") -> tk.Entry:
        return tk.Entry(
            parent,
            textvariable=variable,
            width=width,
            show=show,
            bg=COLOR_INPUT,
            fg=COLOR_TEXT,
            insertbackground=COLOR_TEXT,
            relief="solid",
            borderwidth=1,
            highlightthickness=1,
            highlightbackground=COLOR_BORDER,
            highlightcolor=COLOR_BLUE,
            font=(FONT_UI, 11),
        )

    def ui_button_palette(self, variant: str) -> tuple[str, str, str]:
        palettes = {
            "primary": (COLOR_BLUE, COLOR_BLUE_DARK, "white"),
            "secondary": (COLOR_SURFACE, "#edf3ff", COLOR_TEXT),
            "success": (COLOR_GREEN, "#28bd4f", "#08140d"),
            "accent": (COLOR_TEXT, "#3a3a3c", "white"),
            "danger": (COLOR_RED, "#d9362e", "white"),
        }
        return palettes.get(variant, palettes["secondary"])

    def ui_button(
        self,
        parent: tk.Misc,
        text: str,
        command,
        *,
        variant: str = "secondary",
        strong: bool = False,
        padx: int = 13,
        pady: int = 7,
    ) -> tk.Canvas:
        bg, active_bg, fg = self.ui_button_palette(variant)
        width = max(58, len(text) * 13 + padx * 2)
        height = 18 + pady * 2
        try:
            parent_bg = parent.cget("bg")
        except tk.TclError:
            parent_bg = COLOR_BG
        canvas = tk.Canvas(
            parent,
            bg=parent_bg,
            width=width,
            height=height,
            highlightthickness=0,
            bd=0,
            cursor="pointinghand",
        )
        button_bg = create_rounded_rect(
            canvas,
            1,
            1,
            width - 1,
            height - 1,
            min(14, height // 2),
            fill=bg,
            outline=COLOR_BORDER if variant == "secondary" else bg,
            width=1,
        )
        button_text = canvas.create_text(
            width // 2,
            height // 2,
            text=text,
            fill=fg,
            font=(FONT_UI, 10, "bold" if strong else "normal"),
        )
        canvas._ui_button_bg = button_bg
        canvas._ui_button_text = button_text
        canvas._ui_button_variant = variant

        def on_enter(_event) -> None:
            current_variant = getattr(canvas, "_ui_button_variant", variant)
            _bg, current_active_bg, _fg = self.ui_button_palette(current_variant)
            canvas.itemconfigure(
                button_bg,
                fill=current_active_bg,
                outline=COLOR_BORDER if current_variant == "secondary" else current_active_bg,
            )

        def on_leave(_event) -> None:
            current_variant = getattr(canvas, "_ui_button_variant", variant)
            current_bg, _active_bg, _fg = self.ui_button_palette(current_variant)
            canvas.itemconfigure(
                button_bg,
                fill=current_bg,
                outline=COLOR_BORDER if current_variant == "secondary" else current_bg,
            )

        def on_click(_event) -> str:
            command()
            return "break"

        for tag in (canvas,):
            tag.bind("<Button-1>", on_click)
            tag.bind("<Enter>", on_enter)
            tag.bind("<Leave>", on_leave)
        return canvas

    def configure_ui_button(
        self,
        button: tk.Canvas,
        text: str,
        variant: str,
        *,
        strong: bool | None = None,
    ) -> None:
        bg_item = getattr(button, "_ui_button_bg", None)
        text_item = getattr(button, "_ui_button_text", None)
        if bg_item is None or text_item is None:
            return
        bg, _active_bg, fg = self.ui_button_palette(variant)
        button._ui_button_variant = variant
        button.itemconfigure(
            bg_item,
            fill=bg,
            outline=COLOR_BORDER if variant == "secondary" else bg,
        )
        button.itemconfigure(text_item, text=text, fill=fg)
        if strong is not None:
            button.itemconfigure(
                text_item,
                font=(FONT_UI, 10, "bold" if strong else "normal"),
            )

    def ui_menu_button(
        self,
        parent: tk.Misc,
        text: str,
        items: list[tuple[str, callable]],
    ) -> tk.Menubutton:
        menu_button = tk.Menubutton(
            parent,
            text=text,
            fg=COLOR_TEXT,
            bg=COLOR_SURFACE_ALT,
            activeforeground=COLOR_TEXT,
            activebackground=COLOR_BORDER,
            relief="flat",
            bd=0,
            padx=12,
            pady=6,
            font=(FONT_UI, 10, "bold"),
            cursor="pointinghand",
        )
        menu = tk.Menu(
            menu_button,
            tearoff=0,
            bg=COLOR_SURFACE_ALT,
            fg=COLOR_TEXT,
            activebackground=COLOR_BLUE,
            activeforeground="#ffffff",
            bd=0,
            relief="flat",
        )
        for label, command in items:
            menu.add_command(label=label, command=command)
        menu_button.configure(menu=menu)
        return menu_button

    def build_ui(self) -> None:
        self.configure_ttk_style()
        self.root.configure(bg=COLOR_BG)

        shell = tk.Frame(self.root, bg=COLOR_BG)
        shell.pack(fill="both", expand=True)

        self.sidebar = tk.Frame(shell, bg=COLOR_SIDEBAR, width=210)
        self.sidebar.pack(side="left", fill="y")
        self.sidebar.pack_propagate(False)

        self.content_root = tk.Frame(shell, bg=COLOR_BG)
        self.content_root.pack(side="left", fill="both", expand=True)
        self.content_root.columnconfigure(0, weight=1)
        self.content_root.rowconfigure(0, weight=1)

        self.pages: dict[str, tk.Frame] = {}
        self.nav_buttons: dict[str, tk.Label] = {}
        self.build_sidebar()

        download_page = tk.Frame(self.content_root, bg=COLOR_BG)
        transcription_page = tk.Frame(self.content_root, bg=COLOR_BG)
        reader_page = tk.Frame(self.content_root, bg=COLOR_BG)
        results_page = tk.Frame(self.content_root, bg=COLOR_BG)
        person_monitor_page = tk.Frame(self.content_root, bg=COLOR_BG)
        settings_page = tk.Frame(self.content_root, bg=COLOR_BG)
        self.pages = {
            "download": download_page,
            "transcription": transcription_page,
            "reader": reader_page,
            "results": results_page,
            "person_monitor": person_monitor_page,
            "settings": settings_page,
        }

        self.build_download_page(download_page)
        self.build_transcription_tab(transcription_page)
        self.build_reader_tab(reader_page)
        self.build_results_tab(results_page)
        self.person_monitor_page = PersonMonitorPage(self, person_monitor_page)
        self.build_settings_tab(settings_page)
        for page in self.pages.values():
            page.grid(row=0, column=0, sticky="nsew")
        self.show_page("download")

    def build_sidebar(self) -> None:
        brand = tk.Frame(self.sidebar, bg=COLOR_SIDEBAR, padx=22, pady=24)
        brand.pack(fill="x")
        tk.Label(
            brand,
            text="Podcast",
            fg=COLOR_TEXT,
            bg=COLOR_SIDEBAR,
            font=(FONT_DISPLAY, 22, "bold"),
        ).pack(anchor="w")
        tk.Label(
            brand,
            text="Radar",
            fg=COLOR_BLUE,
            bg=COLOR_SIDEBAR,
            font=(FONT_DISPLAY, 22, "bold"),
        ).pack(anchor="w")
        tk.Label(
            brand,
            text="AI / Weather / Health",
            fg=COLOR_MUTED,
            bg=COLOR_SIDEBAR,
            font=(FONT_UI, 10),
        ).pack(anchor="w", pady=(8, 0))

        nav_specs = [
            ("download", "今日雷达"),
            ("person_monitor", "人物监控"),
            ("favorites", "我的精选"),
            ("reader", "阅读"),
            ("results", "资料库"),
            ("settings", "设置"),
        ]
        for page_key, title in nav_specs:
            button = tk.Label(
                self.sidebar,
                text=title,
                justify="left",
                anchor="w",
                padx=16,
                pady=12,
                fg=COLOR_TEXT,
                bg=COLOR_SIDEBAR,
                font=(FONT_UI, 11, "bold"),
                cursor="pointinghand",
            )
            button.pack(fill="x", padx=14, pady=4)
            if page_key == "favorites":
                button.bind("<Button-1>", lambda _event: self.show_favorite_items())
            else:
                button.bind("<Button-1>", lambda _event, key=page_key: self.show_page(key))
            self.nav_buttons[page_key] = button

        tk.Label(
            self.sidebar,
            text=f"本机 · Chrome · v{APP_VERSION}",
            fg=COLOR_MUTED,
            bg=COLOR_SIDEBAR,
            font=(FONT_UI, 9),
        ).pack(side="bottom", anchor="w", padx=22, pady=20)

    def set_active_nav(self, nav_key: str) -> None:
        for key, button in self.nav_buttons.items():
            button.configure(bg=COLOR_SIDEBAR_ACTIVE if key == nav_key else COLOR_SIDEBAR)

    def show_page(self, page_key: str, *, nav_key: str | None = None) -> None:
        if page_key not in self.pages:
            return
        self.set_active_nav(nav_key or page_key)
        if page_key == self.current_page_key:
            if page_key == "person_monitor":
                self.person_monitor_page.on_show()
            elif page_key == "results":
                self.ensure_results_current()
            else:
                self.schedule_visible_page_flush(delay_ms=0)
            return
        if page_key != "results":
            self.cancel_results_jobs(cancel_refresh=True)
        if page_key != "download":
            self.cancel_cards_render(mark_dirty=True)
        self.current_page_key = page_key
        self.pages[page_key].tkraise()
        self.event_processing_blocked_until = time.monotonic() + 0.04
        self.schedule_visible_page_flush(delay_ms=45)
        if page_key == "results":
            self.root.after(45, self.ensure_results_current)
        elif page_key == "person_monitor":
            self.root.after(45, self.person_monitor_page.on_show)

    def schedule_visible_page_flush(self, delay_ms: int = 35) -> None:
        if self.page_content_flush_job is not None:
            try:
                self.root.after_cancel(self.page_content_flush_job)
            except tk.TclError:
                pass
        self.page_content_flush_job = self.root.after(delay_ms, self.flush_visible_page_updates)

    def flush_visible_page_updates(self) -> None:
        self.page_content_flush_job = None
        if self.current_page_key == "download":
            if self.download_render_dirty:
                self.render_cards()
                return
            pending_ids = list(self.dirty_card_ids)[:8]
            for video_id in pending_ids:
                self.dirty_card_ids.discard(video_id)
                self.refresh_card_visual(video_id)
            artwork_ids = list(self.pending_artwork_ids)[:2]
            for video_id in artwork_ids:
                self.pending_artwork_ids.discard(video_id)
                self.apply_artwork_to_card(video_id)
            if self.dirty_card_ids or self.pending_artwork_ids:
                self.schedule_visible_page_flush(delay_ms=12)
        elif self.current_page_key == "transcription":
            pending_rows = list(self.pending_transcription_visuals.items())[:16]
            for path, (status, output) in pending_rows:
                self.pending_transcription_visuals.pop(path, None)
                self.apply_transcription_row_visual(path, status, output)
            if self.pending_transcription_logs:
                messages = self.pending_transcription_logs[:80]
                del self.pending_transcription_logs[:80]
                self.insert_transcription_logs(messages)
            if self.pending_transcription_visuals or self.pending_transcription_logs:
                self.schedule_visible_page_flush(delay_ms=12)

    def build_download_page(self, parent: tk.Frame) -> None:
        player = tk.Frame(parent, bg=COLOR_BG, height=52)
        player.pack(fill="x")
        player.pack_propagate(False)
        tk.Label(
            player,
            textvariable=self.download_title_var,
            fg=COLOR_TEXT,
            bg=COLOR_BG,
            font=(FONT_DISPLAY, 22, "bold"),
        ).pack(side="left", padx=32)
        self.ui_label(player, textvariable=self.selected_count_var, fg=COLOR_MUTED, bg=COLOR_BG).pack(side="left", padx=(0, 12))
        topic_buttons = [
            ("science_wellness", self.fetch_science_wellness_updates, 13, (5, 26)),
            ("weather_agri", self.fetch_weather_agri_updates, 13, 5),
            ("ai_startup", self.fetch_ai_startup_updates, 13, 5),
            ("ai", self.fetch_ai_updates, 13, 5),
            ("all", self.fetch_all_updates, 15, 5),
        ]
        for topic, command, button_padx, pack_padx in topic_buttons:
            button = self.ui_button(
                player,
                TOPIC_FILTER_LABELS[topic],
                command,
                variant=topic_filter_variant(topic, self.current_topic),
                strong=topic == self.current_topic,
                padx=button_padx,
                pady=5,
            )
            button.pack(side="right", padx=pack_padx, pady=9)
            self.topic_filter_buttons[topic] = button
        self.update_topic_filter_buttons()

        body = tk.Frame(parent, bg=COLOR_BG, padx=32, pady=8)
        body.pack(fill="both", expand=True)

        controls = tk.Frame(body, bg=COLOR_BG)
        controls.pack(fill="x", pady=(0, 7))
        self.download_controls = controls
        for text, cmd, variant in [
            ("待处理", self.select_available_rows, "secondary"),
            ("下载", self.download_selected, "success"),
            ("转译", self.research_selected, "primary"),
            ("精选", self.favorite_selected, "accent"),
        ]:
            self.ui_button(controls, text, cmd, variant=variant, padx=12, pady=5).pack(side="left", padx=(0, 7))
        self.ui_menu_button(
            controls,
            "更多",
            [
                ("全选", self.select_all_rows),
                ("清空选择", self.clear_selection),
                ("下载全部", self.download_all_available),
                ("下载精选", self.download_favorite_visible),
                ("取消精选", self.unfavorite_selected),
                ("停止任务", self.request_stop),
                ("转录任务详情", lambda: self.show_page("transcription")),
                ("下载设置", lambda: self.show_page("settings")),
            ],
        ).pack(side="left", padx=(0, 8))
        self.manual_link_toggle = self.ui_button(
            controls,
            "＋ 添加链接",
            self.toggle_manual_link_panel,
            variant="secondary",
            padx=11,
            pady=5,
        )
        self.manual_link_toggle.pack(side="right")

        tk.Checkbutton(
            controls,
            text="自动转录",
            variable=self.auto_transcribe_var,
            command=self.save_settings,
            fg=COLOR_TEXT,
            bg=COLOR_BG,
            selectcolor=COLOR_SURFACE,
            activebackground=COLOR_BG,
            activeforeground=COLOR_TEXT,
            font=(FONT_UI, 10),
        ).pack(side="left", padx=(2, 0))

        self.manual_link_panel = tk.Frame(body, bg=COLOR_SURFACE_ALT, padx=12, pady=8)
        self.ui_label(
            self.manual_link_panel,
            text="播客 / 视频链接",
            fg=COLOR_MUTED,
            bg=COLOR_SURFACE_ALT,
            weight="bold",
        ).pack(side="left", padx=(0, 10))
        self.manual_link_entry = self.ui_entry(self.manual_link_panel, self.url_var, width=72)
        self.manual_link_entry.pack(side="left", fill="x", expand=True, ipady=5)
        self.ui_button(
            self.manual_link_panel,
            "读取",
            self.fetch_videos,
            variant="accent",
            strong=True,
            padx=14,
            pady=5,
        ).pack(side="left", padx=(8, 0))
        self.ui_button(
            self.manual_link_panel,
            "清空",
            lambda: self.url_var.set(""),
            variant="secondary",
            padx=11,
            pady=5,
        ).pack(side="left", padx=(6, 0))

        summary_line = tk.Frame(body, bg=COLOR_BG)
        summary_line.pack(fill="x", pady=(0, 4))
        self.summary_label = self.ui_label(
            summary_line,
            textvariable=self.summary_var,
            fg=COLOR_BLUE,
            bg=COLOR_BG,
            weight="bold",
        )
        self.summary_label.pack(side="left", anchor="w")
        self.refresh_indicator = tk.Label(
            summary_line,
            textvariable=self.refresh_indicator_var,
            fg="white",
            bg=COLOR_BLUE,
            padx=9,
            pady=3,
            font=(FONT_UI, 9, "bold"),
        )

        self.cards_area = tk.Frame(body, bg=COLOR_BG)
        self.cards_area.pack(fill="both", expand=True)
        self.feed_panel = tk.Frame(self.cards_area, bg=COLOR_BG)
        self.feed_panel.pack(side="left", fill="both", expand=True)
        self.detail_panel = tk.Frame(self.cards_area, bg=COLOR_SURFACE, width=330, padx=18, pady=18)
        self.detail_panel.pack_propagate(False)

        self.cards_canvas = tk.Canvas(self.feed_panel, bg=COLOR_BG, highlightthickness=0)
        self.cards_scrollbar = ttk.Scrollbar(self.feed_panel, orient="vertical", command=self.cards_canvas.yview)
        self.cards_frame = tk.Frame(self.cards_canvas, bg=COLOR_BG)
        self.cards_window = self.cards_canvas.create_window((0, 0), window=self.cards_frame, anchor="nw")
        self.cards_canvas.configure(yscrollcommand=self.cards_scrollbar.set)
        self.cards_canvas.pack(side="left", fill="both", expand=True)
        self.cards_scrollbar.pack(side="right", fill="y")
        self.cards_frame.bind("<Configure>", lambda _event: self.cards_canvas.configure(scrollregion=self.cards_canvas.bbox("all")))
        self.cards_canvas.bind("<Configure>", self.on_cards_canvas_configure)
        self.root.bind_all("<MouseWheel>", self.on_cards_mousewheel, add="+")
        self.root.bind_all("<Button-4>", self.on_cards_mousewheel, add="+")
        self.root.bind_all("<Button-5>", self.on_cards_mousewheel, add="+")

        footer = tk.Frame(parent, bg=COLOR_SURFACE, padx=28, pady=6)
        footer.pack(fill="x")
        footer_top = tk.Frame(footer, bg=COLOR_SURFACE)
        footer_top.pack(fill="x")
        self.ui_label(footer_top, textvariable=self.current_task_var, fg=COLOR_TEXT, bg=COLOR_SURFACE, weight="bold").pack(side="left")
        progress_line = tk.Frame(footer_top, bg=COLOR_SURFACE)
        progress_line.pack(side="left", fill="x", expand=True, padx=(20, 0))
        self.progress_bar = RoundedDinoProgressBar(
            progress_line,
            variable=self.progress_value,
            maximum=100,
            bg=COLOR_SURFACE,
        )
        self.progress_bar.pack(side="left", fill="x", expand=True)
        self.ui_label(progress_line, textvariable=self.progress_percent_var, fg=COLOR_MUTED, bg=COLOR_SURFACE).pack(side="left", padx=(10, 0))
        self.ui_label(footer, textvariable=self.status_var, fg=COLOR_MUTED, bg=COLOR_SURFACE).pack(anchor="w", pady=(3, 0))
        self.render_cards()

    def update_topic_filter_buttons(self) -> None:
        for topic, button in self.topic_filter_buttons.items():
            selected = topic == self.current_topic
            self.configure_ui_button(
                button,
                TOPIC_FILTER_LABELS[topic],
                topic_filter_variant(topic, self.current_topic),
                strong=selected,
            )

    def begin_feed_refresh(self, topic: str, *, showing_cache: bool) -> None:
        label = topic_display_label(topic)
        if self.refresh_indicator_hide_job is not None:
            try:
                self.root.after_cancel(self.refresh_indicator_hide_job)
            except tk.TclError:
                pass
            self.refresh_indicator_hide_job = None

        indicator = getattr(self, "refresh_indicator", None)
        if isinstance(indicator, tk.Label):
            self.refresh_indicator_var.set(f"↻ 正在刷新 · {label}")
            indicator.configure(bg=COLOR_BLUE, fg="white")
            if not indicator.winfo_manager():
                indicator.pack(side="right", padx=(12, 0))

        self.current_task_var.set(f"当前任务：正在刷新{label}最新节目")
        self.set_status(
            f"正在后台刷新{label}最新节目，当前列表仍可浏览..."
            if showing_cache
            else f"正在搜索{label}最新节目，请稍候..."
        )
        self.refresh_owns_progress = not (
            self.downloading or self.researching or self.transcribing
        )
        if self.refresh_owns_progress and hasattr(self, "progress_bar"):
            self.progress_value.set(0)
            self.progress_bar.stop()
            self.progress_bar.configure(mode="indeterminate")
            self.progress_bar.start(12)
            self.progress_percent_var.set("刷新中")

    def finish_feed_refresh(self, *, success: bool, used_cache: bool = False) -> None:
        label = topic_display_label(self.current_topic)
        if self.refresh_owns_progress and hasattr(self, "progress_bar"):
            self.progress_bar.stop()
            self.progress_bar.configure(mode="determinate")
            self.progress_value.set(0)
            self.progress_percent_var.set("0%")
        self.refresh_owns_progress = False

        indicator = getattr(self, "refresh_indicator", None)
        if not isinstance(indicator, tk.Label):
            return
        if success:
            self.refresh_indicator_var.set(f"✓ 刷新完成 · {label}")
            indicator.configure(bg=COLOR_GREEN, fg=COLOR_TEXT)
            hide_delay = 1800
        elif used_cache:
            self.refresh_indicator_var.set(f"! 实时刷新失败 · 已用缓存")
            indicator.configure(bg=COLOR_RED, fg="white")
            hide_delay = 4200
        else:
            self.refresh_indicator_var.set(f"! 刷新失败 · {label}")
            indicator.configure(bg=COLOR_RED, fg="white")
            hide_delay = 4200
        if not indicator.winfo_manager():
            indicator.pack(side="right", padx=(12, 0))
        self.refresh_indicator_hide_job = self.root.after(
            hide_delay,
            self.hide_feed_refresh_indicator,
        )

    def hide_feed_refresh_indicator(self) -> None:
        self.refresh_indicator_hide_job = None
        indicator = getattr(self, "refresh_indicator", None)
        if isinstance(indicator, tk.Label):
            indicator.pack_forget()
        self.refresh_indicator_var.set("")

    def toggle_manual_link_panel(self) -> None:
        self.manual_link_expanded = not self.manual_link_expanded
        if self.manual_link_expanded:
            self.manual_link_panel.pack(
                fill="x",
                after=self.download_controls,
                pady=(0, 8),
            )
            self.configure_ui_button(self.manual_link_toggle, "收起链接", "secondary")
            self.root.after(30, self.manual_link_entry.focus_set)
            return
        self.manual_link_panel.pack_forget()
        self.configure_ui_button(self.manual_link_toggle, "＋ 添加链接", "secondary")

    def on_cards_mousewheel(self, event: tk.Event) -> str | None:
        units = self.card_scroll_units(event)
        if units == 0:
            return None
        if self.is_detail_scroll_event(event):
            canvas = self.detail_canvas
            if isinstance(canvas, tk.Canvas):
                canvas.yview_scroll(units, "units")
                return "break"
        if not hasattr(self, "cards_canvas"):
            return None
        if not self.is_cards_scroll_event(event):
            return None
        self.cards_canvas.yview_scroll(units, "units")
        return "break"

    def is_cards_scroll_event(self, event: tk.Event) -> bool:
        if not self.cards_area or not hasattr(self, "cards_canvas"):
            return False
        try:
            widget = self.root.winfo_containing(event.x_root, event.y_root)
        except (AttributeError, tk.TclError):
            widget = getattr(event, "widget", None)
        return isinstance(widget, tk.Widget) and self.is_descendant(widget, self.cards_area)

    def is_detail_scroll_event(self, event: tk.Event) -> bool:
        if not self.detail_panel_visible or not hasattr(self, "detail_panel"):
            return False
        try:
            widget = self.root.winfo_containing(event.x_root, event.y_root)
        except (AttributeError, tk.TclError):
            widget = getattr(event, "widget", None)
        return isinstance(widget, tk.Widget) and self.is_descendant(widget, self.detail_panel)

    def is_descendant(self, widget: tk.Widget, ancestor: tk.Widget) -> bool:
        current: tk.Widget | None = widget
        while current is not None:
            if current is ancestor:
                return True
            current = current.master
        return False

    def card_scroll_units(self, event: tk.Event) -> int:
        button_number = getattr(event, "num", None)
        if button_number == 4:
            return -3
        if button_number == 5:
            return 3

        delta = int(getattr(event, "delta", 0) or 0)
        if delta == 0:
            return 0
        if IS_MAC:
            magnitude = 1 if abs(delta) < 120 else max(1, abs(delta) // 120)
            return -magnitude if delta > 0 else magnitude
        units = int(-delta / 120)
        if units == 0:
            units = -1 if delta > 0 else 1
        return units

    def on_cards_canvas_configure(self, event: tk.Event) -> None:
        self.cards_canvas.itemconfigure(self.cards_window, width=event.width)

    def calculate_card_columns(self, available_width: int | None = None) -> int:
        if available_width is None and hasattr(self, "cards_canvas"):
            available_width = self.cards_canvas.winfo_width()
        width = max(360, available_width or 0)
        cell_width = CARD_WIDTH + CARD_GAP_X
        return max(1, min(5, width // cell_width))

    def cancel_cards_render(self, mark_dirty: bool = False) -> None:
        if self.cards_render_job is not None:
            try:
                self.root.after_cancel(self.cards_render_job)
            except tk.TclError:
                pass
            self.cards_render_job = None
        if mark_dirty and self.cards_render_index < len(self.cards_render_plan):
            self.download_render_dirty = True

    def render_cards(self) -> None:
        if not hasattr(self, "cards_frame"):
            return
        self.cancel_cards_render()
        for child in self.cards_frame.winfo_children():
            child.destroy()
        self.card_widgets = {}
        self.card_images = []
        self.card_image_map = {}
        self.card_columns = 1
        self.cards_frame.columnconfigure(0, weight=1)
        self.cards_render_plan = []
        self.cards_render_index = 0
        self.cards_render_row = 0
        self.download_render_dirty = False

        if not self.video_items:
            self.focused_item_id = ""
            empty = tk.Frame(self.cards_frame, bg=COLOR_BG, pady=80)
            empty.grid(row=0, column=0, sticky="ew")
            empty_text = (
                "还没有精选节目。遇到值得长期保留的内容，先选中卡片，再点「精选」。"
                if self.current_topic == "favorites"
                else "点击右上角「综合雷达」，分区查看 AI、AI创业、天气农业和科学养生播客更新。"
            )
            tk.Label(
                empty,
                text=empty_text,
                fg=COLOR_MUTED,
                bg=COLOR_BG,
                font=(FONT_UI, 14),
            ).pack()
            self.render_inline_detail(None)
            return

        sections = self.card_sections()
        visible_ids = [item.video_id for _, _, items in sections for item in items]
        if not self.focused_item_id or self.focused_item_id not in visible_ids:
            self.focused_item_id = ""
        for section_index, (title, subtitle, items) in enumerate(sections):
            if title:
                self.cards_render_plan.append(("header", (section_index, title, subtitle, len(items))))
            self.cards_render_plan.extend(("item", item) for item in items)
        self.render_next_card_batch()

    def render_next_card_batch(self) -> None:
        self.cards_render_job = None
        if self.current_page_key not in {"", "download"}:
            self.download_render_dirty = True
            return
        started_at = time.perf_counter()
        rendered = 0
        while self.cards_render_index < len(self.cards_render_plan):
            kind, payload = self.cards_render_plan[self.cards_render_index]
            if kind == "header":
                section_index, title, subtitle, count = payload
                header = self.create_section_header(self.cards_frame, title, subtitle, count)
                header.grid(
                    row=self.cards_render_row,
                    column=0,
                    sticky="ew",
                    pady=((0 if section_index == 0 else 12), 6),
                )
            else:
                item = payload
                episode_row = self.create_episode_row(self.cards_frame, item)
                episode_row.grid(
                    row=self.cards_render_row,
                    column=0,
                    sticky="ew",
                    padx=(0, 8),
                    pady=(0, 6),
                )
                self.dirty_card_ids.discard(item.video_id)
            self.cards_render_row += 1
            self.cards_render_index += 1
            rendered += 1
            if rendered >= 3 or time.perf_counter() - started_at >= 0.012:
                break
        if self.cards_render_index < len(self.cards_render_plan):
            self.cards_render_job = self.root.after(12, self.render_next_card_batch)
            return
        self.download_render_dirty = False
        self.render_inline_detail(self.video_map.get(self.focused_item_id) if self.focused_item_id else None)
        if self.pending_artwork_ids or self.dirty_card_ids:
            self.schedule_visible_page_flush(delay_ms=12)

    def card_sections(self) -> list[tuple[str, str, list[VideoItem]]]:
        if self.current_topic == "favorites":
            return [("我的精选", "跨日期收藏，适合稍后下载、转录或阅读", self.video_items)]
        if self.current_topic != "all":
            return [("", "", self.video_items)]

        ai_items: list[VideoItem] = []
        ai_startup_items: list[VideoItem] = []
        weather_agri_items: list[VideoItem] = []
        science_wellness_items: list[VideoItem] = []
        other_items: list[VideoItem] = []
        for item in self.video_items:
            if item.source_name in WEATHER_AGRI_FEED_NAMES:
                weather_agri_items.append(item)
            elif item.source_name in AI_STARTUP_FEED_NAMES:
                ai_startup_items.append(item)
            elif item.source_name in AI_FEED_NAMES:
                ai_items.append(item)
            elif item.source_name in SCIENCE_WELLNESS_FEED_NAMES:
                science_wellness_items.append(item)
            else:
                other_items.append(item)

        sections: list[tuple[str, str, list[VideoItem]]] = []
        if ai_items:
            sections.append(("AI 播客", "模型、算力、产品与产业动态", ai_items))
        if ai_startup_items:
            sections.append(("AI创业", "创始人故事、产品从 0 到 1、获客与商业化", ai_startup_items))
        if weather_agri_items:
            sections.append(("天气农业", "ENSO、气候、作物与农产品市场", weather_agri_items))
        if science_wellness_items:
            sections.append(("科学养生", "营养、运动、睡眠、神经科学与健康寿命", science_wellness_items))
        if other_items:
            sections.append(("其他来源", "手动或未归类播客", other_items))
        return sections or [("", "", self.video_items)]

    def create_section_header(self, parent: tk.Misc, title: str, subtitle: str, count: int) -> tk.Frame:
        header = tk.Frame(parent, bg=COLOR_BG)
        left = tk.Frame(header, bg=COLOR_BG)
        left.pack(side="left", fill="x", expand=True)
        tk.Label(
            left,
            text=title,
            fg=COLOR_TEXT,
            bg=COLOR_BG,
            font=(FONT_DISPLAY, 16, "bold"),
        ).pack(side="left")
        tk.Label(
            left,
            text=subtitle,
            fg=COLOR_MUTED,
            bg=COLOR_BG,
            font=(FONT_UI, 10),
        ).pack(side="left", padx=(10, 0), pady=(3, 0))
        tk.Label(
            header,
            text=f"{count} 条",
            fg=COLOR_MUTED,
            bg=COLOR_BG,
            font=(FONT_UI, 10, "bold"),
        ).pack(side="right", padx=(12, CARD_GAP_X))
        return header

    def create_episode_row(self, parent: tk.Misc, item: VideoItem) -> tk.Canvas:
        selected = item.video_id in self.selected_item_ids
        focused = item.video_id == self.focused_item_id
        bg = COLOR_CARD_SELECTED if selected else COLOR_CARD
        border = COLOR_BLUE if focused else (COLOR_PURPLE if selected else COLOR_BORDER)
        row_height = FEED_ROW_HEIGHT + 4
        row = tk.Canvas(
            parent,
            bg=COLOR_BG,
            height=row_height,
            highlightthickness=0,
            bd=0,
            cursor="pointinghand",
        )
        row._card_bg = bg
        row._card_border = border
        row._card_focused = focused
        create_rounded_rect(
            row,
            3,
            3,
            120,
            row_height - 1,
            16,
            fill=COLOR_CARD_SHADOW,
            outline=COLOR_CARD_SHADOW,
            tags=("row_shadow",),
        )
        create_rounded_rect(
            row,
            1,
            1,
            118,
            row_height - 3,
            16,
            fill=bg,
            outline=border,
            width=2 if focused else 1,
            tags=("row_shape",),
        )
        row_inner = tk.Frame(row, bg=bg, padx=9, pady=8)
        row_inner.grid_propagate(False)
        row_inner.grid_columnconfigure(0, minsize=FEED_ROW_COVER_SIZE + 12, weight=0)
        row_inner.grid_columnconfigure(1, weight=1)
        row_inner.grid_columnconfigure(2, minsize=126, weight=0)
        row_inner.grid_rowconfigure(0, weight=1)
        row_window = row.create_window(9, 7, window=row_inner, anchor="nw")

        def resize_row(event: tk.Event) -> None:
            width = max(140, event.width)
            height = max(row_height, event.height)
            current_bg = getattr(row, "_card_bg", bg)
            current_border = getattr(row, "_card_border", border)
            current_focused = bool(getattr(row, "_card_focused", focused))
            row.delete("row_shadow")
            row.delete("row_shape")
            create_rounded_rect(
                row,
                3,
                3,
                width - 1,
                height - 1,
                16,
                fill=COLOR_CARD_SHADOW,
                outline=COLOR_CARD_SHADOW,
                tags=("row_shadow",),
            )
            create_rounded_rect(
                row,
                1,
                1,
                width - 3,
                height - 3,
                16,
                fill=current_bg,
                outline=current_border,
                width=2 if current_focused else 1,
                tags=("row_shape",),
            )
            row.tag_lower("row_shape")
            row.tag_lower("row_shadow")
            row.coords(row_window, 9, 7)
            row.itemconfigure(row_window, width=max(1, width - 20), height=max(1, height - 16))

        row.bind("<Configure>", resize_row)

        cover = tk.Canvas(row_inner, width=FEED_ROW_COVER_SIZE, height=FEED_ROW_COVER_SIZE, bg=bg, highlightthickness=0, bd=0)
        cover.grid(row=0, column=0, sticky="nw", padx=(0, 12))
        photo = self.load_artwork_photo(item.artwork_url, FEED_ROW_COVER_SIZE)
        if photo:
            cover.create_image(0, 0, anchor="nw", image=photo)
            self.card_images.append(photo)
        else:
            self.draw_placeholder_cover(cover, item, FEED_ROW_COVER_SIZE)
            self.queue_artwork_load(item)

        content = tk.Frame(row_inner, bg=bg)
        content.grid(row=0, column=1, sticky="nsew")
        title = tk.Label(
            content,
            text=self.card_display_title(item),
            fg=COLOR_TEXT,
            bg=bg,
            justify="left",
            anchor="w",
            wraplength=430,
            font=(FONT_UI, 12, "bold"),
            height=1,
        )
        title.pack(fill="x")
        summary = tk.Label(
            content,
            text=self.card_summary_text(item),
            fg=COLOR_MUTED,
            bg=bg,
            justify="left",
            anchor="nw",
            wraplength=430,
            font=(FONT_UI, 10),
            height=1,
        )
        summary.pack(fill="x", pady=(3, 0))
        meta_line = tk.Frame(content, bg=bg)
        meta_line.pack(fill="x", pady=(3, 0))
        meta = tk.Label(
            meta_line,
            text=self.card_meta_text(item),
            fg=COLOR_MUTED,
            bg=bg,
            anchor="w",
            font=(FONT_UI, 9),
        )
        meta.pack(side="left")
        published = tk.Label(
            content,
            text=self.card_published_text(item),
            fg=COLOR_BLUE,
            bg=bg,
            anchor="w",
            font=(FONT_UI, 9, "bold"),
        )
        published.pack(fill="x", pady=(2, 0))

        right = tk.Frame(row_inner, bg=bg, width=126, height=FEED_ROW_HEIGHT - 18)
        right.grid(row=0, column=2, sticky="nsew", padx=(12, 0))
        right.pack_propagate(False)
        selected_badge = tk.Label(
            right,
            text="✓ 已选" if selected else "",
            fg="#08140d" if selected else COLOR_MUTED,
            bg=COLOR_GREEN if selected else bg,
            font=(FONT_UI, 8, "bold"),
            padx=6,
            pady=1,
        )
        selected_badge.pack(anchor="e", pady=(0, 1))
        importance = tk.Label(
            right,
            text=self.card_importance_text(item),
            fg=self.card_importance_color(item),
            bg=bg,
            justify="right",
            anchor="e",
            wraplength=132,
            font=(FONT_UI, 9, "bold"),
            height=1,
        )
        importance.pack(fill="x", pady=(0, 2))
        status = tk.Label(
            right,
            text=self.card_status_text(item),
            fg=COLOR_BLUE if selected else COLOR_MUTED,
            bg=bg,
            justify="right",
            anchor="e",
            wraplength=132,
            font=(FONT_UI, 9),
            height=1,
        )
        status.pack(fill="x")
        select_button = self.ui_button(
            right,
            "取消" if selected else "选择",
            lambda video_id=item.video_id: self.toggle_card_selection(video_id),
            variant="secondary" if selected else "primary",
            padx=8,
            pady=3,
        )
        select_button.pack(anchor="e", pady=(2, 0))

        self.card_widgets[item.video_id] = {
            "row": row,
            "row_inner": row_inner,
            "cover": cover,
            "cover_size": FEED_ROW_COVER_SIZE,
            "content": content,
            "meta_line": meta_line,
            "right": right,
            "title": title,
            "summary": summary,
            "meta": meta,
            "published": published,
            "selected_badge": selected_badge,
            "importance": importance,
            "status": status,
            "select_button": select_button,
        }
        for widget in (row, row_inner, cover, content, title, summary, meta_line, meta, published, right, importance, status):
            self.bind_row_focus(widget, item.video_id)
        row.bind("<Double-Button-1>", lambda _event, item_id=item.video_id: self.show_episode_detail(item_id))
        return row

    def bind_row_focus(self, widget: tk.Widget, video_id: str) -> None:
        widget.bind("<Button-1>", lambda _event, item_id=video_id: self.focus_episode(item_id))

    def focus_episode(self, video_id: str) -> None:
        if video_id not in self.video_map:
            return
        previous = self.focused_item_id
        self.focused_item_id = video_id
        if previous and previous != video_id:
            self.refresh_card_visual(previous)
        self.refresh_card_visual(video_id)
        self.render_inline_detail(self.video_map.get(video_id))

    def show_inline_detail_panel(self) -> None:
        if not hasattr(self, "detail_panel") or self.detail_panel_visible:
            return
        self.detail_panel.pack(side="right", fill="y", padx=(14, 0))
        self.detail_panel_visible = True

    def hide_inline_detail_panel(self) -> None:
        if not hasattr(self, "detail_panel") or not self.detail_panel_visible:
            return
        self.detail_panel.pack_forget()
        self.detail_canvas = None
        self.detail_content = None
        self.detail_panel_visible = False

    def close_inline_detail(self) -> None:
        previous = self.focused_item_id
        self.focused_item_id = ""
        self.hide_inline_detail_panel()
        if previous:
            self.refresh_card_visual(previous)

    def render_inline_detail(self, item: VideoItem | None) -> None:
        if not hasattr(self, "detail_panel"):
            return
        for child in self.detail_panel.winfo_children():
            child.destroy()
        self.detail_canvas = None
        self.detail_content = None

        if item is None:
            self.hide_inline_detail_panel()
            return

        self.show_inline_detail_panel()

        detail_canvas = tk.Canvas(self.detail_panel, bg=COLOR_SURFACE, highlightthickness=0, bd=0)
        detail_scrollbar = ttk.Scrollbar(self.detail_panel, orient="vertical", command=detail_canvas.yview)
        detail_content = tk.Frame(detail_canvas, bg=COLOR_SURFACE)
        detail_window = detail_canvas.create_window((0, 0), window=detail_content, anchor="nw")

        detail_canvas.configure(yscrollcommand=detail_scrollbar.set)
        detail_canvas.pack(side="left", fill="both", expand=True)
        detail_scrollbar.pack(side="right", fill="y")
        detail_content.bind(
            "<Configure>",
            lambda _event, canvas=detail_canvas: canvas.configure(scrollregion=canvas.bbox("all")),
        )
        detail_canvas.bind(
            "<Configure>",
            lambda event, canvas=detail_canvas, window_id=detail_window: canvas.itemconfigure(window_id, width=event.width),
        )
        self.detail_canvas = detail_canvas
        self.detail_content = detail_content
        detail_parent = detail_content
        wrap_width = 270

        detail_head = tk.Frame(detail_parent, bg=COLOR_SURFACE)
        detail_head.pack(fill="x", pady=(0, 12))
        tk.Label(
            detail_head,
            text="节目详情",
            fg=COLOR_TEXT,
            bg=COLOR_SURFACE,
            font=(FONT_DISPLAY, 16, "bold"),
        ).pack(side="left")
        self.ui_button(
            detail_head,
            "关闭",
            self.close_inline_detail,
            variant="secondary",
            padx=10,
            pady=4,
        ).pack(side="right")

        cover = tk.Canvas(detail_parent, width=DETAIL_COVER_SIZE, height=DETAIL_COVER_SIZE, bg=COLOR_SURFACE, highlightthickness=0, bd=0)
        cover.pack(anchor="w")
        photo = self.load_artwork_photo(item.artwork_url, DETAIL_COVER_SIZE)
        if photo:
            cover.create_image(0, 0, anchor="nw", image=photo)
            self.card_images.append(photo)
        else:
            self.draw_placeholder_cover(cover, item, DETAIL_COVER_SIZE)
            self.queue_artwork_load(item)

        tk.Label(
            detail_parent,
            text=self.card_display_title(item),
            fg=COLOR_TEXT,
            bg=COLOR_SURFACE,
            justify="left",
            anchor="w",
            wraplength=wrap_width,
            font=(FONT_DISPLAY, 17, "bold"),
        ).pack(fill="x", pady=(14, 5))

        original_title = self.card_original_title_text(item)
        if original_title:
            tk.Label(
                detail_parent,
                text=original_title,
                fg=COLOR_MUTED,
                bg=COLOR_SURFACE,
                justify="left",
                anchor="w",
                wraplength=wrap_width,
                font=(FONT_UI, 9),
            ).pack(fill="x", pady=(0, 8))

        meta_text = " · ".join(part for part in [item.source_name, item.published_text, item.duration_text] if part and part != "-")
        tk.Label(
            detail_parent,
            text=meta_text or "-",
            fg=COLOR_BLUE,
            bg=COLOR_SURFACE,
            justify="left",
            anchor="w",
            wraplength=wrap_width,
            font=(FONT_UI, 10, "bold"),
        ).pack(fill="x", pady=(0, 10))

        tk.Label(
            detail_parent,
            text=self.card_importance_detail_text(item),
            fg=self.card_importance_color(item),
            bg=COLOR_SURFACE,
            justify="left",
            anchor="w",
            wraplength=wrap_width,
            font=(FONT_UI, 10, "bold"),
        ).pack(fill="x", pady=(0, 5))
        tk.Label(
            detail_parent,
            text=item.importance_reason or "重要性评分等待生成",
            fg=COLOR_MUTED,
            bg=COLOR_SURFACE,
            justify="left",
            anchor="w",
            wraplength=wrap_width,
            font=(FONT_UI, 9),
        ).pack(fill="x", pady=(0, 10))

        record = self.research_library.get("items", {}).get(item.video_id, {})
        status_text = self.card_status_text(item)
        tk.Label(
            detail_parent,
            text=status_text,
            fg=COLOR_GREEN if item.video_id in self.selected_item_ids else COLOR_MUTED,
            bg=COLOR_SURFACE,
            justify="left",
            anchor="w",
            wraplength=wrap_width,
            font=(FONT_UI, 10),
        ).pack(fill="x", pady=(0, 12))

        source_text = transcript_source_text(record)
        if source_text:
            source_color = COLOR_MUTED
            if record.get("transcript_origin") == "audio_asr":
                source_color = COLOR_GREEN
            elif record.get("transcript_origin") == "verified_official":
                source_color = COLOR_BLUE
            elif record.get("transcript_origin") == "episode_brief_fallback":
                source_color = COLOR_PURPLE
            tk.Label(
                detail_parent,
                text=source_text,
                fg=source_color,
                bg=COLOR_SURFACE,
                justify="left",
                anchor="w",
                wraplength=wrap_width,
                font=(FONT_UI, 10, "bold"),
            ).pack(fill="x", pady=(0, 12))

        if item.webpage_url:
            link_box = tk.Frame(detail_parent, bg=COLOR_SURFACE)
            link_box.pack(fill="x", pady=(0, 14))
            tk.Label(
                link_box,
                text="原链接",
                fg=COLOR_TEXT,
                bg=COLOR_SURFACE,
                font=(FONT_UI, 11, "bold"),
            ).pack(anchor="w", pady=(0, 4))
            link_label = tk.Label(
                link_box,
                text=compact_url(item.webpage_url, 110),
                fg=COLOR_BLUE,
                bg=COLOR_SURFACE,
                justify="left",
                anchor="w",
                wraplength=wrap_width,
                cursor="pointinghand",
                font=(FONT_UI, 9),
            )
            link_label.pack(fill="x", pady=(0, 7))
            link_label.bind("<Button-1>", lambda _event, url=item.webpage_url: webbrowser.open(url))
            link_actions = tk.Frame(link_box, bg=COLOR_SURFACE)
            link_actions.pack(fill="x")
            self.ui_button(
                link_actions,
                "打开原链接",
                lambda url=item.webpage_url: webbrowser.open(url),
                variant="secondary",
                padx=10,
                pady=5,
            ).pack(side="left", padx=(0, 8))
            self.ui_button(
                link_actions,
                "复制链接",
                lambda url=item.webpage_url: self.copy_text_to_clipboard(url, "原链接"),
                variant="secondary",
                padx=10,
                pady=5,
            ).pack(side="left")

        action_top = tk.Frame(detail_parent, bg=COLOR_SURFACE)
        action_top.pack(fill="x", pady=(0, 8))
        self.ui_button(
            action_top,
            "取消选择" if item.video_id in self.selected_item_ids else "选择",
            lambda video_id=item.video_id: self.toggle_card_selection(video_id),
            variant="secondary" if item.video_id in self.selected_item_ids else "primary",
            padx=12,
            pady=6,
        ).pack(side="left", padx=(0, 8))
        self.ui_button(
            action_top,
            "确认转译",
            lambda item=item: self.start_single_research_from_detail(item),
            variant="success",
            padx=12,
            pady=6,
        ).pack(side="left")

        action_bottom = tk.Frame(detail_parent, bg=COLOR_SURFACE)
        action_bottom.pack(fill="x", pady=(0, 16))
        favorite_text = "取消精选" if item.video_id in self.favorite_item_ids else "标为精选"
        favorite_action = False if item.video_id in self.favorite_item_ids else True
        self.ui_button(
            action_bottom,
            favorite_text,
            lambda item=item, favorite=favorite_action: self.set_item_favorite(item, favorite),
            variant="accent" if favorite_action else "secondary",
            padx=12,
            pady=6,
        ).pack(side="left", padx=(0, 8))
        self.ui_button(
            action_bottom,
            "详情",
            lambda video_id=item.video_id: self.show_episode_detail(video_id),
            variant="secondary",
            padx=12,
            pady=6,
        ).pack(side="left")

        digest_path = Path(str(record.get("digest_path"))) if isinstance(record, dict) and record.get("digest_path") else None
        transcript_path = self.item_transcript_path(item)
        translation_path = (
            Path(str(record.get("full_translation_path")))
            if isinstance(record, dict) and record.get("full_translation_path")
            else None
        )
        existing_results = [
            ("阅读纪要", digest_path, "success", "ZH 纪要"),
            ("阅读转录", transcript_path, "primary", "TXT 转录"),
            ("阅读译文", translation_path, "accent", "中文译文"),
        ]
        existing_results = [
            (label, path, variant, result_type)
            for label, path, variant, result_type in existing_results
            if path and path.exists()
        ]
        if existing_results:
            tk.Label(
                detail_parent,
                text="已生成资料",
                fg=COLOR_TEXT,
                bg=COLOR_SURFACE,
                font=(FONT_UI, 11, "bold"),
            ).pack(anchor="w", pady=(0, 8))
            result_actions = tk.Frame(detail_parent, bg=COLOR_SURFACE)
            result_actions.pack(fill="x", pady=(0, 16))
            detail_title = self.card_display_title(item)
            for label, path, variant, result_type in existing_results:
                self.ui_button(
                    result_actions,
                    label,
                    lambda result_path=path, result_type=result_type, title=detail_title: self.load_reader_path(
                        result_path,
                        title=title,
                        result_type=result_type,
                    ),
                    variant=variant,
                    padx=10,
                    pady=5,
                ).pack(side="left", padx=(0, 8))
            first_path = existing_results[0][1]
            self.ui_button(
                result_actions,
                "所在文件夹",
                lambda result_path=first_path: self.reveal_path(result_path),
                variant="secondary",
                padx=10,
                pady=5,
            ).pack(side="left")
            if transcript_path and transcript_path.exists():
                translation_actions = tk.Frame(detail_parent, bg=COLOR_SURFACE)
                translation_actions.pack(fill="x", pady=(0, 16))
                brief_fallback = transcript_path.name.endswith(".brief.txt")
                self.ui_button(
                    translation_actions,
                    "看简介译文" if brief_fallback else "看译文",
                    lambda item=item: self.show_item_translation(item),
                    variant="primary",
                    padx=10,
                    pady=5,
                ).pack(side="left", padx=(0, 8))
                self.ui_button(
                    translation_actions,
                    "播放简介译文" if brief_fallback else "播放译文",
                    lambda item=item: self.play_item_translation(item),
                    variant="accent",
                    padx=10,
                    pady=5,
                ).pack(side="left")

        tk.Label(
            detail_parent,
            text="中文简介",
            fg=COLOR_TEXT,
            bg=COLOR_SURFACE,
            font=(FONT_UI, 11, "bold"),
        ).pack(anchor="w", pady=(2, 6))
        tk.Label(
            detail_parent,
            text=self.card_summary_text(item),
            fg=COLOR_TEXT,
            bg=COLOR_SURFACE,
            justify="left",
            anchor="nw",
            wraplength=wrap_width,
            font=(FONT_UI, 10),
        ).pack(fill="x")

        digest_snippet = self.card_digest_snippet(item)
        if digest_snippet:
            tk.Label(
                detail_parent,
                text="中文纪要",
                fg=COLOR_TEXT,
                bg=COLOR_SURFACE,
                font=(FONT_UI, 11, "bold"),
            ).pack(anchor="w", pady=(16, 6))
            tk.Label(
                detail_parent,
                text=digest_snippet,
                fg=COLOR_MUTED,
                bg=COLOR_SURFACE,
                justify="left",
                anchor="nw",
                wraplength=wrap_width,
                font=(FONT_UI, 10),
            ).pack(fill="x")

        description = clamp_text(item.description_text, 420) if item.description_text else ""
        if description:
            tk.Label(
                detail_parent,
                text="原始简介",
                fg=COLOR_TEXT,
                bg=COLOR_SURFACE,
                font=(FONT_UI, 11, "bold"),
            ).pack(anchor="w", pady=(16, 6))
            tk.Label(
                detail_parent,
                text=description,
                fg=COLOR_MUTED,
                bg=COLOR_SURFACE,
                justify="left",
                anchor="nw",
                wraplength=wrap_width,
                font=(FONT_UI, 9),
            ).pack(fill="x")

    def create_episode_card(self, parent: tk.Misc, item: VideoItem) -> tk.Canvas:
        selected = item.video_id in self.selected_item_ids
        bg = COLOR_CARD_SELECTED if selected else COLOR_CARD
        border = COLOR_PURPLE if selected else COLOR_BORDER
        card_width = CARD_WIDTH
        card_height = CARD_HEIGHT
        card = tk.Canvas(parent, bg=COLOR_BG, width=card_width, height=card_height, highlightthickness=0, bd=0)
        card_bg = create_rounded_rect(
            card,
            2,
            2,
            card_width - 2,
            card_height - 2,
            18,
            fill=bg,
            outline=border,
            width=2,
        )
        inner = tk.Frame(card, bg=bg, width=CARD_INNER_WIDTH, height=CARD_INNER_HEIGHT)
        inner.pack_propagate(False)
        card.create_window(11, 11, window=inner, anchor="nw")
        selected_badge = tk.Canvas(
            card,
            width=64,
            height=24,
            bg=bg,
            highlightthickness=0,
            bd=0,
        )
        selected_badge_bg = create_rounded_rect(
            selected_badge,
            1,
            1,
            63,
            23,
            11,
            fill=COLOR_GREEN,
            outline=COLOR_GREEN,
            width=1,
        )
        selected_badge_text = selected_badge.create_text(
            32,
            12,
            text="✓ 已选",
            fill="#08140d",
            font=(FONT_UI, 9, "bold"),
        )
        selected_badge_window = card.create_window(
            card_width - 13,
            15,
            window=selected_badge,
            anchor="ne",
            state="normal" if selected else "hidden",
        )

        cover = tk.Canvas(inner, width=CARD_COVER_SIZE, height=CARD_COVER_SIZE, bg=bg, highlightthickness=0, bd=0)
        cover.pack()
        photo = self.load_artwork_photo(item.artwork_url, CARD_COVER_SIZE)
        if photo:
            cover.create_image(0, 0, anchor="nw", image=photo)
            self.card_images.append(photo)
        else:
            self.draw_placeholder_cover(cover, item, CARD_COVER_SIZE)
            self.queue_artwork_load(item)

        title = tk.Label(
            inner,
            text=self.card_display_title(item),
            fg=COLOR_TEXT,
            bg=bg,
            justify="left",
            anchor="nw",
            wraplength=CARD_COVER_SIZE,
            font=(FONT_UI, 10, "bold"),
            height=2,
        )
        title.pack(fill="x", pady=(10, 2))

        summary = tk.Label(
            inner,
            text=self.card_summary_text(item),
            fg=COLOR_MUTED,
            bg=bg,
            justify="left",
            anchor="nw",
            wraplength=CARD_COVER_SIZE,
            font=(FONT_UI, 9),
            height=3,
        )
        summary.pack(fill="x", pady=(0, 2))

        meta = tk.Label(
            inner,
            text=self.card_meta_text(item),
            fg=COLOR_MUTED,
            bg=bg,
            justify="left",
            anchor="nw",
            wraplength=CARD_COVER_SIZE,
            font=(FONT_UI, 9),
            height=2,
        )
        meta.pack(fill="x")

        published = tk.Label(
            inner,
            text=self.card_published_text(item),
            fg=COLOR_BLUE,
            bg=bg,
            justify="left",
            anchor="w",
            wraplength=CARD_COVER_SIZE,
            font=(FONT_UI, 9, "bold"),
            height=1,
        )
        published.pack(fill="x", pady=(5, 0))

        importance = tk.Label(
            inner,
            text=self.card_importance_text(item),
            fg=self.card_importance_color(item),
            bg=bg,
            justify="left",
            anchor="w",
            wraplength=CARD_COVER_SIZE,
            font=(FONT_UI, 9, "bold"),
            height=1,
        )
        importance.pack(fill="x", pady=(4, 0))

        digest_preview = tk.Label(
            inner,
            text=self.card_digest_preview_text(item),
            fg=COLOR_MUTED,
            bg=bg,
            justify="left",
            anchor="nw",
            wraplength=CARD_COVER_SIZE,
            font=(FONT_UI, 8),
            height=1,
        )
        digest_preview.pack(fill="x", pady=(4, 0))

        status = tk.Label(
            inner,
            text=self.card_status_text(item),
            fg=COLOR_BLUE if selected else COLOR_MUTED,
            bg=bg,
            justify="left",
            anchor="w",
            wraplength=CARD_COVER_SIZE,
            font=(FONT_UI, 9),
            height=1,
        )
        status.pack(fill="x", pady=(8, 0))

        actions = tk.Frame(inner, bg=bg)
        actions.pack(side="bottom", fill="x", pady=(8, 0))
        self.ui_button(actions, "网页", lambda url=item.webpage_url: webbrowser.open(url), variant="secondary", padx=8, pady=4).pack(side="left")
        select_button = self.ui_button(
            actions,
            "取消" if selected else "选择",
            lambda video_id=item.video_id: self.toggle_card_selection(video_id),
            variant="secondary" if selected else "primary",
            padx=8,
            pady=4,
        )
        select_button.pack(side="right")

        self.card_widgets[item.video_id] = {
            "card": card,
            "card_bg": card_bg,
            "selected_badge": selected_badge,
            "selected_badge_bg": selected_badge_bg,
            "selected_badge_text": selected_badge_text,
            "selected_badge_window": selected_badge_window,
            "inner": inner,
            "title": title,
            "summary": summary,
            "meta": meta,
            "published": published,
            "importance": importance,
            "digest_preview": digest_preview,
            "status": status,
            "actions": actions,
            "cover": cover,
            "select_button": select_button,
        }
        self.bind_card_detail(card, item.video_id)
        self.bind_card_detail(inner, item.video_id)
        self.bind_card_detail(cover, item.video_id)
        self.bind_card_detail(title, item.video_id)
        self.bind_card_detail(summary, item.video_id)
        self.bind_card_detail(meta, item.video_id)
        self.bind_card_detail(status, item.video_id)
        return card

    def bind_card_detail(self, widget: tk.Widget, video_id: str) -> None:
        widget.bind("<Button-1>", lambda _event, item_id=video_id: self.show_episode_detail(item_id))

    def draw_placeholder_cover(self, canvas: tk.Canvas, item: VideoItem, size: int) -> None:
        palette = ["#5e5ce6", "#0a84ff", "#30d158", "#ff9f0a", "#ff375f", "#64d2ff"]
        color = palette[int(hashlib.sha1(item.video_id.encode("utf-8")).hexdigest(), 16) % len(palette)]
        create_rounded_rect(canvas, 0, 0, size, size, 14, fill=color, outline=color)
        words = (item.source_name or item.title or "AI").split()
        initials = "".join(word[:1] for word in words[:2]).upper() or "AI"
        canvas.create_text(
            size / 2,
            size / 2 - 10,
            text=initials[:4],
            fill="white",
            font=(FONT_DISPLAY, 30, "bold"),
        )
        canvas.create_text(
            size / 2,
            size - 26,
            text=(item.source_name or "Podcast")[:18],
            fill="white",
            font=(FONT_UI, 10, "bold"),
        )

    def artwork_cache_path(self, artwork_url: str) -> Path:
        cache_key = hashlib.sha1(artwork_url.encode("utf-8")).hexdigest()
        return ARTWORK_CACHE_DIR / f"{cache_key}.img"

    def queue_artwork_load(self, item: VideoItem) -> None:
        if not item.artwork_url or Image is None or ImageTk is None:
            return
        cache_path = self.artwork_cache_path(item.artwork_url)
        if cache_path.exists() or item.video_id in self.artwork_loading:
            return
        self.artwork_loading.add(item.video_id)
        threading.Thread(target=self.artwork_worker, args=(item.video_id, item.artwork_url), daemon=True).start()

    def artwork_worker(self, video_id: str, artwork_url: str) -> None:
        try:
            ensure_dir(ARTWORK_CACHE_DIR)
            cache_path = self.artwork_cache_path(artwork_url)
            if not cache_path.exists():
                resp = requests.get(artwork_url, timeout=10, headers={"User-Agent": "Mozilla/5.0"})
                resp.raise_for_status()
                cache_path.write_bytes(resp.content)
            self.events.put(("artwork_ready", video_id))
        except Exception:
            pass
        finally:
            self.artwork_loading.discard(video_id)

    def apply_artwork_to_card(self, video_id: str) -> None:
        if self.current_page_key != "download":
            self.pending_artwork_ids.add(video_id)
            return
        item = self.video_map.get(video_id)
        widgets = self.card_widgets.get(video_id)
        if not item or not widgets:
            if item:
                self.pending_artwork_ids.add(video_id)
            return
        cover_size = int(widgets.get("cover_size") or CARD_COVER_SIZE)
        photo = self.load_artwork_photo(item.artwork_url, cover_size)
        if not photo:
            return
        cover = widgets.get("cover")
        if not isinstance(cover, tk.Canvas):
            return
        cover.delete("all")
        cover.create_image(0, 0, anchor="nw", image=photo)
        self.card_image_map[video_id] = photo
        if video_id == self.focused_item_id:
            self.render_inline_detail(item)

    def load_artwork_photo(self, artwork_url: str, size: int) -> tk.PhotoImage | None:
        if not artwork_url or Image is None or ImageDraw is None or ImageTk is None:
            return None
        try:
            ensure_dir(ARTWORK_CACHE_DIR)
            cache_path = self.artwork_cache_path(artwork_url)
            if not cache_path.exists():
                return None

            image = Image.open(cache_path).convert("RGBA")
            width, height = image.size
            side = min(width, height)
            left = (width - side) // 2
            top = (height - side) // 2
            image = image.crop((left, top, left + side, top + side)).resize((size, size))
            mask = Image.new("L", (size, size), 0)
            draw = ImageDraw.Draw(mask)
            draw.rounded_rectangle((0, 0, size, size), radius=14, fill=255)
            image.putalpha(mask)
            return ImageTk.PhotoImage(image)
        except Exception:
            return None

    def toggle_card_selection(self, video_id: str) -> None:
        if video_id not in self.video_map:
            return
        if video_id in self.selected_item_ids:
            self.selected_item_ids.remove(video_id)
        else:
            self.selected_item_ids.add(video_id)
        self.refresh_card_visual(video_id)
        self.update_selected_count()

    def show_episode_detail(self, video_id: str) -> None:
        item = self.video_map.get(video_id)
        if not item:
            return
        existing = self.detail_windows.get(video_id, {}).get("window")
        if isinstance(existing, tk.Toplevel) and existing.winfo_exists():
            existing.lift()
            existing.focus_force()
            return

        window = tk.Toplevel(self.root)
        window.title(item.title[:80])
        window.configure(bg=COLOR_BG)
        window.geometry("920x760")
        window.minsize(760, 560)

        shell = tk.Frame(window, bg=COLOR_BG, padx=22, pady=18)
        shell.pack(fill="both", expand=True)

        header = tk.Frame(shell, bg=COLOR_BG)
        header.pack(fill="x", pady=(0, 14))
        cover = tk.Canvas(header, width=96, height=96, bg=COLOR_BG, highlightthickness=0, bd=0)
        cover.pack(side="left", padx=(0, 18))
        photo = self.load_artwork_photo(item.artwork_url, 96)
        detail_photo = photo
        if photo:
            cover.create_image(0, 0, anchor="nw", image=photo)
            self.card_images.append(photo)
        else:
            self.draw_placeholder_cover(cover, item, 96)

        title_box = tk.Frame(header, bg=COLOR_BG)
        title_box.pack(side="left", fill="x", expand=True)
        tk.Label(
            title_box,
            text=item.title,
            fg=COLOR_TEXT,
            bg=COLOR_BG,
            font=(FONT_DISPLAY, 18, "bold"),
            justify="left",
            anchor="w",
            wraplength=620,
        ).pack(fill="x")
        meta_text = " · ".join(part for part in [item.source_name, item.published_text, item.duration_text] if part and part != "-")
        tk.Label(
            title_box,
            text=meta_text,
            fg=COLOR_MUTED,
            bg=COLOR_BG,
            font=(FONT_UI, 10),
            anchor="w",
        ).pack(fill="x", pady=(7, 0))

        buttons = tk.Frame(shell, bg=COLOR_BG)
        buttons.pack(fill="x", pady=(0, 12))
        self.ui_button(buttons, "选择/取消", lambda video_id=video_id: self.toggle_card_selection(video_id), variant="primary", padx=12, pady=6).pack(side="left", padx=(0, 8))
        self.ui_button(buttons, "标为精选", lambda item=item: self.set_item_favorite(item, True), variant="accent", padx=12, pady=6).pack(side="left", padx=(0, 8))
        self.ui_button(buttons, "刷新翻译", lambda item=item: self.refresh_episode_detail_translation(item), variant="secondary", padx=12, pady=6).pack(side="left", padx=(0, 8))
        self.ui_button(buttons, "确认转译", lambda item=item: self.start_single_research_from_detail(item), variant="success", padx=12, pady=6).pack(side="left", padx=(0, 8))
        self.ui_button(buttons, "打开网页", lambda url=item.webpage_url: webbrowser.open(url), variant="secondary", padx=12, pady=6).pack(side="left", padx=(0, 8))

        transcript_path = self.item_transcript_path(item)
        if transcript_path and transcript_path.exists():
            translation_buttons = tk.Frame(shell, bg=COLOR_BG)
            translation_buttons.pack(fill="x", pady=(0, 12))
            tk.Label(
                translation_buttons,
                text="英文转录深读",
                fg=COLOR_MUTED,
                bg=COLOR_BG,
                font=(FONT_UI, 10, "bold"),
            ).pack(side="left", padx=(0, 10))
            self.ui_button(
                translation_buttons,
                "看简介译文" if transcript_path.name.endswith(".brief.txt") else "看译文",
                lambda item=item: self.show_item_translation(item),
                variant="primary",
                padx=12,
                pady=6,
            ).pack(side="left", padx=(0, 8))
            self.ui_button(
                translation_buttons,
                "播放简介译文" if transcript_path.name.endswith(".brief.txt") else "播放译文",
                lambda item=item: self.play_item_translation(item),
                variant="accent",
                padx=12,
                pady=6,
            ).pack(side="left")

        text_frame = tk.Frame(shell, bg=COLOR_BG)
        text_frame.pack(fill="both", expand=True)
        detail_text = tk.Text(
            text_frame,
            bg=COLOR_INPUT,
            fg=COLOR_TEXT,
            insertbackground=COLOR_TEXT,
            relief="flat",
            wrap="word",
            font=(FONT_UI, 12),
            padx=16,
            pady=14,
        )
        self.configure_detail_text_tags(detail_text)
        scrollbar = ttk.Scrollbar(text_frame, orient="vertical", command=detail_text.yview)
        detail_text.configure(yscrollcommand=scrollbar.set)
        detail_text.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        self.detail_windows[video_id] = {"window": window, "text": detail_text, "photo": detail_photo}
        window.protocol("WM_DELETE_WINDOW", lambda item_id=video_id: self.close_episode_detail(item_id))
        self.render_episode_detail_text(item, loading=True)
        self.ensure_episode_detail_translation(item)

    def close_episode_detail(self, video_id: str) -> None:
        widgets = self.detail_windows.pop(video_id, {})
        window = widgets.get("window")
        if isinstance(window, tk.Toplevel) and window.winfo_exists():
            window.destroy()

    def render_episode_detail_text(self, item: VideoItem, loading: bool = False) -> None:
        widgets = self.detail_windows.get(item.video_id)
        text_widget = widgets.get("text") if widgets else None
        if not isinstance(text_widget, tk.Text) or not text_widget.winfo_exists():
            return
        record = self.research_library.get("items", {}).get(item.video_id, {})
        translated = record.get("detail_translation") if isinstance(record, dict) else ""
        digest_path = record.get("digest_path") if isinstance(record, dict) else ""
        sections = [
            "# 节目详情",
            "",
            f"标题 / Title: {item.title}",
            f"来源 / Source: {item.source_name or '-'}",
            f"发布时间 / Published: {item.published_text or '-'}",
            f"重要性 / Importance: {item.importance_score:.1f}/10",
            f"播放量 / Plays: {format_count(item.play_count)}",
            f"评分依据 / Rationale: {item.importance_reason or '-'}",
            f"链接 / Link: {item.webpage_url or '-'}",
            "",
        ]
        if translated:
            sections.extend([translated.strip(), ""])
        elif loading:
            sections.extend(["正在生成中文沉浸式翻译，英文原文会保留在下方...", ""])
        else:
            sections.extend(["中文翻译暂不可用。", ""])
        if digest_path:
            sections.extend([f"中文研究纪要: {digest_path}", ""])
        sections.extend(
            [
                "## English Original",
                item.description_text.strip() or "No original episode description was provided by the feed.",
            ]
        )
        text_widget.configure(state="normal")
        text_widget.delete("1.0", "end")
        self.insert_rendered_markdown(text_widget, "\n".join(sections))
        text_widget.configure(state="disabled")

    def configure_detail_text_tags(self, text_widget: tk.Text) -> None:
        text_widget.tag_configure(
            "h1",
            foreground=COLOR_TEXT,
            font=(FONT_DISPLAY, 20, "bold"),
            spacing1=10,
            spacing3=12,
        )
        text_widget.tag_configure(
            "h2",
            foreground=COLOR_TEXT,
            font=(FONT_DISPLAY, 15, "bold"),
            spacing1=16,
            spacing3=7,
        )
        text_widget.tag_configure(
            "h3",
            foreground=COLOR_TEXT,
            font=(FONT_DISPLAY, 13, "bold"),
            spacing1=12,
            spacing3=5,
        )
        text_widget.tag_configure(
            "body",
            foreground=COLOR_TEXT,
            font=(FONT_UI, 12),
            spacing3=5,
        )
        text_widget.tag_configure(
            "bold",
            foreground=COLOR_TEXT,
            font=(FONT_UI, 12, "bold"),
        )
        text_widget.tag_configure(
            "meta",
            foreground=COLOR_MUTED,
            font=(FONT_UI, 11),
            spacing3=4,
        )
        text_widget.tag_configure(
            "quote",
            foreground=COLOR_MUTED,
            font=(FONT_UI, 11),
            lmargin1=18,
            lmargin2=18,
            spacing1=6,
            spacing3=8,
        )
        text_widget.tag_configure(
            "bullet",
            foreground=COLOR_TEXT,
            font=(FONT_UI, 12),
            lmargin1=18,
            lmargin2=34,
            spacing3=5,
        )
        text_widget.tag_configure(
            "link",
            foreground=COLOR_BLUE,
            font=(FONT_UI, 11),
            underline=True,
        )

    def insert_rendered_markdown(self, text_widget: tk.Text, markdown_text: str) -> None:
        for raw_line in markdown_text.splitlines():
            line = raw_line.rstrip()
            stripped = line.strip()
            if stripped.startswith("```"):
                continue
            if not stripped:
                text_widget.insert("end", "\n", ("body",))
                continue
            if stripped.startswith("# "):
                text_widget.insert("end", stripped[2:].strip() + "\n", ("h1",))
                continue
            if stripped.startswith("## "):
                text_widget.insert("end", stripped[3:].strip() + "\n", ("h2",))
                continue
            if stripped.startswith("### "):
                text_widget.insert("end", stripped[4:].strip() + "\n", ("h3",))
                continue
            if stripped.startswith(">"):
                self.insert_markdown_inline(text_widget, stripped.lstrip("> ").strip(), "quote")
                text_widget.insert("end", "\n", ("quote",))
                continue
            if stripped.startswith(("- ", "* ")):
                text_widget.insert("end", "• ", ("bullet",))
                self.insert_markdown_inline(text_widget, stripped[2:].strip(), "bullet")
                text_widget.insert("end", "\n", ("bullet",))
                continue
            numbered = re.match(r"^(\d+\.)\s+(.*)$", stripped)
            if numbered:
                text_widget.insert("end", numbered.group(1) + " ", ("bullet", "bold"))
                self.insert_markdown_inline(text_widget, numbered.group(2), "bullet")
                text_widget.insert("end", "\n", ("bullet",))
                continue
            if stripped.startswith(("标题 /", "来源 /", "发布时间 /", "链接 /", "Title:", "Source:", "Published:", "Link:")):
                self.insert_markdown_inline(text_widget, stripped, "meta")
                text_widget.insert("end", "\n", ("meta",))
                continue
            self.insert_markdown_inline(text_widget, stripped, "body")
            text_widget.insert("end", "\n", ("body",))

    def insert_markdown_inline(self, text_widget: tk.Text, text: str, base_tag: str) -> None:
        cursor = 0
        pattern = re.compile(r"(\*\*[^*]+\*\*|\*[^*]+\*)")
        for match in pattern.finditer(text):
            if match.start() > cursor:
                self.insert_markdown_linked_text(text_widget, text[cursor : match.start()], (base_tag,))
            marked = match.group(0)
            content = marked.strip("*")
            self.insert_markdown_linked_text(text_widget, content, (base_tag, "bold"))
            cursor = match.end()
        if cursor < len(text):
            self.insert_markdown_linked_text(text_widget, text[cursor:], (base_tag,))

    def insert_markdown_linked_text(self, text_widget: tk.Text, text: str, tags: tuple[str, ...]) -> None:
        cursor = 0
        url_pattern = re.compile(r"https?://\S+")
        for match in url_pattern.finditer(text):
            if match.start() > cursor:
                text_widget.insert("end", text[cursor : match.start()], tags)
            text_widget.insert("end", match.group(0), tags + ("link",))
            cursor = match.end()
        if cursor < len(text):
            text_widget.insert("end", text[cursor:], tags)

    def ensure_episode_detail_translation(self, item: VideoItem) -> None:
        record = self.research_library.get("items", {}).get(item.video_id, {})
        if isinstance(record, dict) and record.get("detail_translation"):
            return
        threading.Thread(target=self.detail_translation_worker, args=(item,), daemon=True).start()

    def refresh_episode_detail_translation(self, item: VideoItem) -> None:
        record = self.research_library.setdefault("items", {}).get(item.video_id, {})
        if isinstance(record, dict):
            record.pop("detail_translation", None)
            record.pop("detail_translated_at", None)
            self.research_library.setdefault("items", {})[item.video_id] = record
            self.save_research_library()
        self.render_episode_detail_text(item, loading=True)
        self.ensure_episode_detail_translation(item)

    def detail_translation_worker(self, item: VideoItem) -> None:
        try:
            translation = build_bilingual_episode_detail(
                EpisodeDetailMetadata(
                    title=item.title,
                    source_name=item.source_name,
                    published_text=item.published_text,
                    duration_text=item.duration_text,
                    webpage_url=item.webpage_url,
                    original_description=item.description_text,
                    topic_hint=self.current_topic,
                ),
                self.api_key_var.get().strip(),
            )
            self.events.put(("detail_translation_ready", {"video_id": item.video_id, "translation": translation}))
        except Exception as exc:  # noqa: BLE001
            self.events.put(("detail_translation_error", {"video_id": item.video_id, "error": str(exc)}))

    def apply_episode_detail_translation(self, payload: dict) -> None:
        video_id = payload.get("video_id")
        item = self.video_map.get(video_id)
        if not item:
            return
        record = self.video_item_record(item)
        previous = self.research_library.setdefault("items", {}).get(video_id, {})
        if isinstance(previous, dict):
            favorite = bool(previous.get("favorite"))
            previous.update(record)
            record = previous
            record["favorite"] = favorite
        record["detail_translation"] = payload.get("translation", "")
        record["detail_translated_at"] = datetime.now(timezone.utc).isoformat()
        self.research_library.setdefault("items", {})[video_id] = record
        self.save_research_library()
        self.render_episode_detail_text(item, loading=False)

    def show_episode_detail_error(self, payload: dict) -> None:
        video_id = payload.get("video_id")
        item = self.video_map.get(video_id)
        if not item:
            return
        widgets = self.detail_windows.get(video_id)
        text_widget = widgets.get("text") if widgets else None
        if isinstance(text_widget, tk.Text) and text_widget.winfo_exists():
            text_widget.configure(state="normal")
            text_widget.insert("2.0", f"\n详情翻译失败：{payload.get('error')}\n\n")
            text_widget.configure(state="disabled")

    def start_single_research_from_detail(self, item: VideoItem) -> None:
        self.selected_item_ids = {item.video_id}
        self.refresh_card_visual(item.video_id)
        self.update_selected_count()
        self.research_selected()

    def refresh_card_visual_or_defer(self, video_id: str) -> None:
        if self.current_page_key == "download" and video_id in self.card_widgets:
            self.refresh_card_visual(video_id)
            return
        self.dirty_card_ids.add(video_id)

    def refresh_card_visual(self, video_id: str) -> None:
        widgets = self.card_widgets.get(video_id)
        item = self.video_map.get(video_id)
        if not widgets or not item:
            return
        if "row" in widgets:
            self.refresh_episode_row_visual(video_id, widgets, item)
            return
        selected = video_id in self.selected_item_ids
        bg = COLOR_CARD_SELECTED if selected else COLOR_CARD
        border = COLOR_PURPLE if selected else COLOR_BORDER
        for widget in widgets.values():
            if isinstance(widget, tk.Widget):
                try:
                    if widget is widgets.get("card"):
                        widget.configure(bg=COLOR_BG)
                    else:
                        widget.configure(bg=bg)
                except tk.TclError:
                    pass
        try:
            widgets["card"].itemconfigure(widgets["card_bg"], fill=bg, outline=border)
            widgets["card"].itemconfigure(
                widgets["selected_badge_window"],
                state="normal" if selected else "hidden",
            )
            widgets["title"].configure(text=self.card_display_title(item))
            widgets["summary"].configure(text=self.card_summary_text(item))
            widgets["meta"].configure(text=self.card_meta_text(item))
            widgets["published"].configure(text=self.card_published_text(item))
            widgets["importance"].configure(
                text=self.card_importance_text(item),
                fg=self.card_importance_color(item),
            )
            widgets["digest_preview"].configure(text=self.card_digest_preview_text(item))
            widgets["status"].configure(
                fg=COLOR_BLUE if selected else COLOR_MUTED,
                text=self.card_status_text(item),
            )
            self.configure_ui_button(
                widgets["select_button"],
                text="取消" if selected else "选择",
                variant="secondary" if selected else "primary",
            )
        except tk.TclError:
            pass

    def refresh_episode_row_visual(self, video_id: str, widgets: dict[str, tk.Widget], item: VideoItem) -> None:
        selected = video_id in self.selected_item_ids
        focused = video_id == self.focused_item_id
        bg = COLOR_CARD_SELECTED if selected else COLOR_CARD
        border = COLOR_BLUE if focused else (COLOR_PURPLE if selected else COLOR_BORDER)
        for key in ("row_inner", "cover", "content", "meta_line", "right", "title", "summary", "meta", "published", "importance", "status"):
            widget = widgets.get(key)
            if isinstance(widget, tk.Widget):
                try:
                    widget.configure(bg=bg)
                except tk.TclError:
                    pass
        row = widgets.get("row")
        if isinstance(row, tk.Canvas):
            try:
                row._card_bg = bg
                row._card_border = border
                row._card_focused = focused
                row.itemconfigure(
                    "row_shape",
                    fill=bg,
                    outline=border,
                    width=2 if focused else 1,
                )
            except tk.TclError:
                pass
        try:
            widgets["title"].configure(text=self.card_display_title(item))
            widgets["summary"].configure(text=self.card_summary_text(item))
            widgets["meta"].configure(text=self.card_meta_text(item))
            widgets["published"].configure(text=self.card_published_text(item))
            widgets["importance"].configure(
                text=self.card_importance_text(item),
                fg=self.card_importance_color(item),
            )
            widgets["status"].configure(
                fg=COLOR_BLUE if selected else COLOR_MUTED,
                text=self.card_status_text(item),
            )
            widgets["selected_badge"].configure(
                text="✓ 已选" if selected else "",
                bg=COLOR_GREEN if selected else bg,
                fg="#08140d" if selected else COLOR_MUTED,
            )
            self.configure_ui_button(
                widgets["select_button"],
                text="取消" if selected else "选择",
                variant="secondary" if selected else "primary",
            )
        except tk.TclError:
            pass
        if focused:
            self.render_inline_detail(item)

    def card_status_text(self, item: VideoItem) -> str:
        selected_prefix = "✅ 已选 · " if item.video_id in self.selected_item_ids else ""
        prefix = "精选 · " if item.video_id in self.favorite_item_ids else ""
        prefix = selected_prefix + prefix
        status = clean_ui_status(item.status, 12)
        progress = clean_ui_status(item.progress, 18)
        if "失败" in status and progress not in {"-", "查看底部提示"}:
            progress = "查看底部提示"
        return f"{prefix}{status} · {progress}"

    def card_meta_text(self, item: VideoItem) -> str:
        return " · ".join(part for part in [item.source_name, item.duration_text] if part and part != "-")

    def card_importance_text(self, item: VideoItem) -> str:
        if not item.scored_at:
            update_item_importance_score(item)
        return f"重要性 {item.importance_score:.1f}/10"

    def card_importance_detail_text(self, item: VideoItem) -> str:
        if not item.scored_at:
            update_item_importance_score(item)
        return f"重要性 {item.importance_score:.1f}/10 · 播放量 {format_count(item.play_count)}"

    def card_importance_color(self, item: VideoItem) -> str:
        score = item.importance_score
        if score >= 8.0:
            return COLOR_GREEN
        if score >= 6.5:
            return COLOR_BLUE
        if score >= 5.0:
            return COLOR_PURPLE
        return COLOR_MUTED

    def card_published_text(self, item: VideoItem) -> str:
        return f"发布时间：{item.published_text or '未知'}"

    def card_summary_text(self, item: VideoItem) -> str:
        translated = self.card_translated_summary(item)
        if translated:
            return translated
        if item.description_text:
            return clamp_text(strip_html_text(item.description_text), 72)
        return "暂无简介"

    def card_digest_preview_text(self, item: VideoItem) -> str:
        snippet = self.card_digest_snippet(item)
        if not snippet:
            return ""
        return f"中文纪要：{snippet}"

    def card_digest_snippet(self, item: VideoItem) -> str:
        record = self.research_library.get("items", {}).get(item.video_id, {})
        if not isinstance(record, dict):
            return ""
        digest_path = str(record.get("digest_path", "")).strip()
        if not digest_path:
            return ""
        path = Path(digest_path)
        if not path.exists():
            return ""
        try:
            raw = path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            return ""
        for line in raw.splitlines():
            text = line.strip()
            if not text:
                continue
            if text.startswith("#"):
                continue
            if text.startswith((">", "- ", "•", "—", "* ")):
                continue
            return clamp_text(text, 48)
        return ""

    def card_display_title(self, item: VideoItem) -> str:
        translated = self.card_translated_title(item)
        return translated or item.title

    def card_original_title_text(self, item: VideoItem) -> str:
        translated = self.card_translated_title(item)
        if translated and translated.strip() != item.title.strip():
            return item.title
        return ""

    def card_translated_title(self, item: VideoItem) -> str:
        record = self.research_library.get("items", {}).get(item.video_id, {})
        if isinstance(record, dict):
            return str(record.get("title_zh") or "").strip()
        return ""

    def card_translated_summary(self, item: VideoItem) -> str:
        record = self.research_library.get("items", {}).get(item.video_id, {})
        if isinstance(record, dict):
            return str(record.get("summary_zh") or "").strip()
        return ""

    def _items_for_ids(self, item_ids: list[str] | set[str] | tuple[str, ...] | None) -> list[VideoItem]:
        if item_ids is None:
            return list(self.video_items)
        items: list[VideoItem] = []
        seen: set[str] = set()
        for video_id in item_ids:
            if not video_id or video_id in seen:
                continue
            item = self.video_map.get(video_id)
            if item is None:
                continue
            seen.add(video_id)
            items.append(item)
        return items

    def _start_research_item_translations(self, item_ids: list[str] | set[str] | tuple[str, ...] | None) -> None:
        effective_ids = item_ids
        if not effective_ids and self.pending_research_item_ids:
            effective_ids = self.pending_research_item_ids
        items = self._items_for_ids(effective_ids)
        if not items:
            return
        self.set_status(f"整理完成，正在同步更新 {len(items)} 张卡片的中文标题与简介...")
        self.ensure_card_title_translations(items)

    def ensure_card_title_translations(self, items: list[VideoItem] | None = None) -> None:
        api_key = self.api_key_var.get().strip()
        if not api_key:
            return
        target_items = list(items) if items is not None else list(self.video_items[:CARD_TRANSLATION_VISIBLE_LIMIT])
        missing_items = [
            item
            for item in target_items
            if not self.card_translated_title(item) and not contains_cjk(item.title)
        ]
        if not missing_items:
            self.ensure_card_summary_translations(target_items)
            return
        self.set_status(f"正在后台翻译 {len(missing_items)} 个卡片标题...")
        threading.Thread(target=self.card_title_translation_worker, args=(missing_items, api_key), daemon=True).start()
        self.ensure_card_summary_translations(target_items)

    def ensure_card_summary_translations(self, items: list[VideoItem] | None = None) -> None:
        api_key = self.api_key_var.get().strip()
        if not api_key:
            return
        target_items = list(items) if items is not None else list(self.video_items[:CARD_TRANSLATION_VISIBLE_LIMIT])
        missing_items = [
            item
            for item in target_items
            if not self.card_translated_summary(item)
        ]
        if not missing_items:
            return
        self.set_status(f"正在后台生成 {len(missing_items)} 个卡片中文简介...")
        threading.Thread(target=self.card_summary_translation_worker, args=(missing_items, api_key), daemon=True).start()

    def card_title_translation_worker(self, items: list[VideoItem], api_key: str) -> None:
        try:
            batch_size = 24
            for start in range(0, len(items), batch_size):
                batch = items[start : start + batch_size]
                translated_titles = translate_episode_titles([item.title for item in batch], api_key)
                self.events.put(
                    (
                        "card_titles_ready",
                        {
                            item.video_id: translated
                            for item, translated in zip(batch, translated_titles)
                        },
                    )
                )
        except Exception as exc:  # noqa: BLE001
            self.events.put(("card_titles_error", f"卡片标题翻译失败：{exc}"))

    def card_summary_translation_worker(self, items: list[VideoItem], api_key: str) -> None:
        try:
            batch_size = 16
            for start in range(0, len(items), batch_size):
                batch = items[start : start + batch_size]
                summaries = translate_episode_card_summaries(
                    [
                        {
                            "title": item.title,
                            "source_name": item.source_name,
                            "published_text": item.published_text,
                            "description_text": item.description_text,
                        }
                        for item in batch
                    ],
                    api_key,
                )
                self.events.put(
                    (
                        "card_summaries_ready",
                        {
                            item.video_id: summary
                            for item, summary in zip(batch, summaries)
                        },
                    )
                )
        except Exception as exc:  # noqa: BLE001
            self.events.put(("card_summaries_error", f"卡片中文简介生成失败：{exc}"))

    def apply_card_title_translations(self, translations: dict) -> None:
        records = self.research_library.setdefault("items", {})
        for video_id, title_zh in translations.items():
            item = self.video_map.get(video_id)
            if not item or not str(title_zh).strip():
                continue
            record = records.get(video_id, {})
            if not isinstance(record, dict):
                record = {}
            favorite = bool(record.get("favorite"))
            record.update(self.video_item_record(item))
            record["favorite"] = favorite
            record["title_zh"] = str(title_zh).strip()
            record["title_zh_updated_at"] = datetime.now(timezone.utc).isoformat()
            records[video_id] = record
            self.refresh_card_visual_or_defer(video_id)
        self.save_research_library()
        self.set_status("卡片标题中文翻译已更新")

    def apply_card_summary_translations(self, translations: dict) -> None:
        records = self.research_library.setdefault("items", {})
        updated = 0
        for video_id, summary_zh in translations.items():
            item = self.video_map.get(video_id)
            if not item or not str(summary_zh).strip():
                continue
            record = records.get(video_id, {})
            if not isinstance(record, dict):
                record = {}
            favorite = bool(record.get("favorite"))
            record.update(self.video_item_record(item))
            record["favorite"] = favorite
            record["summary_zh"] = clamp_text(str(summary_zh).strip(), 80)
            record["summary_zh_updated_at"] = datetime.now(timezone.utc).isoformat()
            records[video_id] = record
            self.refresh_card_visual_or_defer(video_id)
            updated += 1
        if updated:
            self.save_research_library()
            self.set_status("卡片中文简介已更新")

    def build_transcription_tab(self, parent: tk.Frame) -> None:
        toolbar = tk.Frame(parent, bg=COLOR_BG, padx=24, pady=18)
        toolbar.pack(fill="x")
        toolbar.columnconfigure(5, weight=1)

        tk.Label(
            toolbar,
            text="处理中",
            fg=COLOR_TEXT,
            bg=COLOR_BG,
            font=(FONT_DISPLAY, 22, "bold"),
        ).grid(row=0, column=0, columnspan=2, sticky="w", pady=(0, 12))
        self.ui_label(
            toolbar,
            textvariable=self.current_task_var,
            fg=COLOR_BLUE,
            bg=COLOR_BG,
            weight="bold",
        ).grid(row=0, column=2, columnspan=4, sticky="e", pady=(0, 12))

        buttons = [
            ("选择文件", self.choose_transcription_files, "primary"),
            ("开始转录", self.start_transcription_queue, "success"),
            ("停止", self.request_transcription_stop, "danger"),
            ("清空完成", self.clear_finished_transcriptions, "secondary"),
            ("资料库", self.show_results_page, "accent"),
        ]
        for index, (text, command, variant) in enumerate(buttons):
            self.ui_button(toolbar, text, command, variant=variant).grid(row=1, column=index, padx=(0, 8))

        self.ui_label(
            toolbar,
            text="这里集中看下载、ASR 转录和中文整理状态；完成后去「资料库」或「阅读」。",
            fg=COLOR_MUTED,
            bg=COLOR_BG,
        ).grid(row=2, column=0, columnspan=6, sticky="w", pady=(10, 0))

        table_frame = tk.Frame(parent, bg=COLOR_BG, padx=24, pady=4)
        table_frame.pack(fill="both", expand=True)
        columns = ("filename", "status", "output")
        self.transcription_tree = ttk.Treeview(table_frame, columns=columns, show="headings", height=15)
        self.transcription_tree.heading("filename", text="文件名")
        self.transcription_tree.heading("status", text="状态")
        self.transcription_tree.heading("output", text="输出文件")
        self.transcription_tree.column("filename", width=460, anchor="w")
        self.transcription_tree.column("status", width=180, anchor="center")
        self.transcription_tree.column("output", width=430, anchor="w")
        scroll = ttk.Scrollbar(table_frame, orient="vertical", command=self.transcription_tree.yview)
        self.transcription_tree.configure(yscrollcommand=scroll.set)
        self.transcription_tree.bind("<Double-1>", self.open_selected_transcription_output)
        self.transcription_tree.grid(row=0, column=0, sticky="nsew")
        scroll.grid(row=0, column=1, sticky="ns")
        table_frame.columnconfigure(0, weight=1)
        table_frame.rowconfigure(0, weight=1)

        log_frame = tk.LabelFrame(
            parent,
            text="转录日志",
            fg=COLOR_TEXT,
            bg=COLOR_BG,
            padx=10,
            pady=8,
            font=(FONT_UI, 10, "bold"),
        )
        log_frame.pack(fill="both", padx=24, pady=(6, 18))
        self.transcription_log_text = tk.Text(
            log_frame,
            height=10,
            wrap="word",
            state="disabled",
            bg=COLOR_SURFACE,
            fg=COLOR_TEXT,
            insertbackground=COLOR_TEXT,
            relief="flat",
            font=(FONT_MONO, 9),
        )
        self.transcription_log_text.pack(fill="both", expand=True)

    def build_reader_tab(self, parent: tk.Frame) -> None:
        shell = tk.Frame(parent, bg=COLOR_BG, padx=24, pady=18)
        shell.pack(fill="both", expand=True)

        header = tk.Frame(shell, bg=COLOR_BG)
        header.pack(fill="x")
        tk.Label(
            header,
            textvariable=self.reader_title_var,
            fg=COLOR_TEXT,
            bg=COLOR_BG,
            font=(FONT_DISPLAY, 22, "bold"),
            anchor="w",
            justify="left",
        ).pack(side="left", fill="x", expand=True)
        self.ui_button(header, "资料库", self.show_results_page, variant="secondary", padx=12, pady=6).pack(side="right", padx=(8, 0))
        self.ui_button(header, "外部打开", self.open_reader_current_path, variant="secondary", padx=12, pady=6).pack(side="right", padx=(8, 0))
        self.ui_button(header, "所在文件夹", self.reveal_reader_current_path, variant="secondary", padx=12, pady=6).pack(side="right", padx=(8, 0))

        self.ui_label(shell, textvariable=self.reader_meta_var, fg=COLOR_BLUE, bg=COLOR_BG).pack(anchor="w", pady=(8, 0))
        path_label = self.ui_label(shell, textvariable=self.reader_path_var, fg=COLOR_MUTED, bg=COLOR_BG)
        path_label.pack(anchor="w", pady=(2, 10))

        translation_bar = tk.Frame(shell, bg="#eef6ff", padx=12, pady=9)
        translation_bar.pack(fill="x", pady=(0, 8))
        self.ui_label(
            translation_bar,
            textvariable=self.reader_translation_status_var,
            fg=COLOR_BLUE_DARK,
            bg="#eef6ff",
            weight="bold",
        ).pack(side="left", fill="x", expand=True)
        self.ui_button(
            translation_bar,
            "停止播放",
            self.stop_reader_tts_audio,
            variant="secondary",
            padx=10,
            pady=5,
        ).pack(side="right", padx=(8, 0))
        self.ui_button(
            translation_bar,
            "播放译文",
            self.play_reader_translation,
            variant="accent",
            padx=10,
            pady=5,
        ).pack(side="right", padx=(8, 0))
        self.ui_button(
            translation_bar,
            "看译文",
            self.show_reader_translation,
            variant="primary",
            padx=10,
            pady=5,
        ).pack(side="right", padx=(8, 0))

        tts_bar = tk.Frame(shell, bg=COLOR_SURFACE_ALT, padx=12, pady=9)
        tts_bar.pack(fill="x", pady=(0, 10))
        self.ui_label(
            tts_bar,
            textvariable=self.tts_status_var,
            fg=COLOR_TEXT,
            bg=COLOR_SURFACE_ALT,
            weight="bold",
        ).pack(side="left", fill="x", expand=True)
        self.ui_button(tts_bar, "音频文件夹", self.open_tts_dir, variant="secondary", padx=10, pady=5).pack(side="right", padx=(8, 0))
        self.ui_button(tts_bar, "播放音频", self.play_reader_tts_audio, variant="secondary", padx=10, pady=5).pack(side="right", padx=(8, 0))
        self.ui_button(tts_bar, "朗读当前页", self.generate_reader_tts_current, variant="accent", padx=10, pady=5).pack(side="right", padx=(8, 0))
        self.ui_button(tts_bar, "朗读选中", self.generate_reader_tts_selected, variant="success", padx=10, pady=5).pack(side="right", padx=(8, 0))

        body = tk.Frame(shell, bg=COLOR_SURFACE, padx=0, pady=0)
        body.pack(fill="both", expand=True)
        self.reader_text = tk.Text(
            body,
            wrap="word",
            state="disabled",
            bg=COLOR_SURFACE,
            fg=COLOR_TEXT,
            insertbackground=COLOR_TEXT,
            relief="flat",
            padx=22,
            pady=18,
            font=(FONT_UI, 12),
        )
        self.reader_text.tag_configure("heading", foreground=COLOR_TEXT, font=(FONT_DISPLAY, 16, "bold"), spacing1=10, spacing3=6)
        self.reader_text.tag_configure("h1", foreground=COLOR_TEXT, font=(FONT_DISPLAY, 19, "bold"), spacing1=12, spacing3=8)
        self.reader_text.tag_configure("h2", foreground=COLOR_TEXT, font=(FONT_DISPLAY, 16, "bold"), spacing1=12, spacing3=6)
        self.reader_text.tag_configure("h3", foreground=COLOR_TEXT, font=(FONT_DISPLAY, 14, "bold"), spacing1=10, spacing3=5)
        self.reader_text.tag_configure("body", foreground=COLOR_TEXT, font=(FONT_UI, 12), spacing3=4)
        self.reader_text.tag_configure("bold", font=(FONT_UI, 12, "bold"))
        self.reader_text.tag_configure("quote", foreground=COLOR_MUTED, font=(FONT_UI, 11), lmargin1=18, lmargin2=18, spacing1=2, spacing3=4)
        self.reader_text.tag_configure("bullet", foreground=COLOR_TEXT, font=(FONT_UI, 12), lmargin1=18, lmargin2=36, spacing3=3)
        self.reader_text.tag_configure("meta", foreground=COLOR_MUTED, font=(FONT_UI, 11), spacing3=3)
        self.reader_text.tag_configure("link", foreground=COLOR_BLUE, underline=True)
        self.reader_text.tag_configure("muted", foreground=COLOR_MUTED)
        reader_scroll = ttk.Scrollbar(body, orient="vertical", command=self.reader_text.yview)
        self.reader_text.configure(yscrollcommand=reader_scroll.set)
        self.reader_text.pack(side="left", fill="both", expand=True)
        reader_scroll.pack(side="right", fill="y")
        self.set_reader_text("从左侧进入「资料库」，双击任意 TXT 转录或 ZH 纪要，就会在这里打开。")

    def build_results_tab(self, parent: tk.Frame) -> None:
        header = tk.Frame(parent, bg=COLOR_BG, padx=24, pady=18)
        header.pack(fill="x")
        tk.Label(
            header,
            text="资料库",
            fg=COLOR_TEXT,
            bg=COLOR_BG,
            font=(FONT_DISPLAY, 22, "bold"),
        ).pack(side="left")
        self.ui_label(header, textvariable=self.results_summary_var, fg=COLOR_MUTED, bg=COLOR_BG).pack(side="left", padx=(14, 0), pady=(5, 0))

        actions = tk.Frame(parent, bg=COLOR_BG, padx=24)
        actions.pack(fill="x", pady=(0, 8))
        for text, command, variant in [
            ("刷新", self.refresh_results, "secondary"),
            ("阅读", self.read_selected_result, "success"),
            ("外部打开", self.open_selected_result, "secondary"),
            ("所在文件夹", self.reveal_selected_result, "primary"),
            ("复制路径", self.copy_selected_result_path, "secondary"),
            ("下载目录", self.open_download_dir, "secondary"),
            ("纪要目录", self.open_digest_dir, "secondary"),
        ]:
            self.ui_button(actions, text, command, variant=variant, padx=12, pady=6).pack(side="left", padx=(0, 8))

        hint = tk.Frame(parent, bg=COLOR_SURFACE_ALT, padx=14, pady=10)
        hint.pack(fill="x", padx=24, pady=(0, 10))
        tk.Label(
            hint,
            text="TXT 是完整转录，ZH 是中文纪要，中文译文用于逐段深读，TTS 是本地朗读音频。双击文本进入阅读页；双击音频会用系统播放器打开。",
            fg=COLOR_TEXT,
            bg=COLOR_SURFACE_ALT,
            font=(FONT_UI, 10),
            anchor="w",
        ).pack(fill="x")

        table_frame = tk.Frame(parent, bg=COLOR_BG, padx=24, pady=4)
        table_frame.pack(fill="both", expand=True)
        columns = ("type", "title", "updated", "path")
        self.results_tree = ttk.Treeview(table_frame, columns=columns, show="headings", height=18)
        self.results_tree.heading("type", text="类型")
        self.results_tree.heading("title", text="节目 / 文件")
        self.results_tree.heading("updated", text="更新时间")
        self.results_tree.heading("path", text="位置")
        self.results_tree.column("type", width=96, anchor="center")
        self.results_tree.column("title", width=420, anchor="w")
        self.results_tree.column("updated", width=150, anchor="center")
        self.results_tree.column("path", width=560, anchor="w")
        self.results_tree.bind("<Double-1>", self.read_selected_result)
        scroll = ttk.Scrollbar(table_frame, orient="vertical", command=self.results_tree.yview)
        self.results_tree.configure(yscrollcommand=scroll.set)
        self.results_tree.grid(row=0, column=0, sticky="nsew")
        scroll.grid(row=0, column=1, sticky="ns")
        table_frame.columnconfigure(0, weight=1)
        table_frame.rowconfigure(0, weight=1)

    def build_settings_tab(self, parent: tk.Frame) -> None:
        download_form = tk.Frame(parent, bg=COLOR_BG, padx=28, pady=18)
        download_form.pack(fill="x")
        download_form.columnconfigure(1, weight=1)
        tk.Label(
            download_form,
            text="下载与来源",
            fg=COLOR_TEXT,
            bg=COLOR_BG,
            font=(FONT_DISPLAY, 16, "bold"),
        ).grid(row=0, column=0, columnspan=3, sticky="w", pady=(0, 12))
        self.ui_label(download_form, text="保存目录", fg=COLOR_TEXT, bg=COLOR_BG, weight="bold").grid(
            row=1, column=0, sticky="w"
        )
        self.ui_label(download_form, textvariable=self.download_dir_var, fg=COLOR_MUTED, bg=COLOR_BG).grid(
            row=1, column=1, sticky="ew", padx=(14, 10)
        )
        self.ui_button(download_form, "选择目录", self.choose_download_dir, variant="secondary", padx=10, pady=5).grid(
            row=1, column=2, sticky="e"
        )
        download_options = tk.Frame(download_form, bg=COLOR_BG)
        download_options.grid(row=2, column=1, columnspan=2, sticky="w", padx=(14, 0), pady=(12, 0))
        self.ui_label(download_options, text="音频格式", fg=COLOR_TEXT, bg=COLOR_BG, weight="bold").pack(side="left")
        ttk.Combobox(
            download_options,
            textvariable=self.audio_format_var,
            values=["m4a", "mp3"],
            state="readonly",
            width=6,
        ).pack(side="left", padx=(8, 18))
        self.ui_label(download_options, text="浏览器登录态", fg=COLOR_TEXT, bg=COLOR_BG, weight="bold").pack(side="left")
        ttk.Combobox(
            download_options,
            textvariable=self.browser_cookie_var,
            values=["auto", "chrome", "edge", "firefox", "none"],
            state="readonly",
            width=9,
        ).pack(side="left", padx=(8, 12))
        self.ui_button(
            download_options,
            "保存",
            self.save_download_settings,
            variant="primary",
            strong=True,
            padx=12,
            pady=5,
        ).pack(side="left")

        form = tk.Frame(parent, bg=COLOR_BG, padx=28, pady=12)
        form.pack(fill="x")
        tk.Label(
            form,
            text="语音转录",
            fg=COLOR_TEXT,
            bg=COLOR_BG,
            font=(FONT_DISPLAY, 16, "bold"),
        ).grid(row=0, column=0, columnspan=2, sticky="w", pady=(0, 16))

        self.ui_label(
            form,
            text="转录模式",
            fg=COLOR_TEXT,
            bg=COLOR_BG,
            weight="bold",
        ).grid(row=1, column=0, sticky="w", pady=7)
        ttk.Combobox(
            form,
            textvariable=self.transcription_mode_var,
            values=list(TRANSCRIPTION_MODE_OPTIONS),
            state="readonly",
            width=14,
        ).grid(row=1, column=1, sticky="w", padx=(14, 0), pady=7)

        fields = [
            ("DASHSCOPE_API_KEY（可选）", self.api_key_var, True),
            ("目标切分时长（秒）", self.target_segment_seconds_var, False),
            ("最大切分时长（秒）", self.max_segment_seconds_var, False),
            ("云端并发数", self.transcription_workers_var, False),
        ]
        for row, (label, variable, is_secret) in enumerate(fields, start=2):
            self.ui_label(
                form,
                text=label,
                fg=COLOR_TEXT,
                bg=COLOR_BG,
                weight="bold",
            ).grid(row=row, column=0, sticky="w", pady=7)
            entry = self.ui_entry(form, variable, width=72 if is_secret else 16, show="*" if is_secret else "")
            entry.grid(row=row, column=1, sticky="w", padx=(14, 0), pady=7)

        self.ui_button(form, "保存设置", self.save_transcription_settings, variant="primary", strong=True, padx=16, pady=8).grid(
            row=6, column=1, sticky="w", padx=(14, 0), pady=(16, 0)
        )

        self.ui_label(
            form,
            text="本地优先会先用 Whisper；极速云端会把音频切片后并发转录，缺少 Key 或云端片段失败时由本地 Whisper 补齐。",
            fg=COLOR_BLUE,
            bg=COLOR_BG,
        ).grid(row=7, column=0, columnspan=2, sticky="w", pady=(18, 0))

        tk.Label(
            form,
            text="本地朗读 TTS",
            fg=COLOR_TEXT,
            bg=COLOR_BG,
            font=(FONT_DISPLAY, 16, "bold"),
        ).grid(row=8, column=0, columnspan=2, sticky="w", pady=(28, 16))

        tts_fields = [
            ("Qwen3-TTS Python", self.tts_python_var, 72),
            ("模型", self.tts_model_var, 72),
            ("说话人", self.tts_speaker_var, 18),
            ("语种", self.tts_language_var, 18),
            ("控制指令", self.tts_instruct_var, 72),
            ("单次朗读上限（字）", self.tts_max_chars_var, 18),
        ]
        for row, (label, variable, width) in enumerate(tts_fields, start=9):
            self.ui_label(
                form,
                text=label,
                fg=COLOR_TEXT,
                bg=COLOR_BG,
                weight="bold",
            ).grid(row=row, column=0, sticky="w", pady=7)
            field_frame = tk.Frame(form, bg=COLOR_BG)
            field_frame.grid(row=row, column=1, sticky="w", padx=(14, 0), pady=7)
            self.ui_entry(field_frame, variable, width=width).pack(side="left")
            if label == "Qwen3-TTS Python":
                self.ui_button(field_frame, "选择", self.choose_tts_python, variant="secondary", padx=10, pady=5).pack(side="left", padx=(8, 0))

    def show_results_page(self) -> None:
        self.show_page("results")

    def show_favorite_items(self) -> None:
        self.current_topic = "favorites"
        self.download_title_var.set(topic_download_title(self.current_topic))
        self.update_topic_filter_buttons()
        self.show_page("download", nav_key="favorites")
        favorite_items = self.cached_podcast_items("favorites")
        self.apply_fetch_payload(
            {
                "source_title": "跨日期精选收藏",
                "items": favorite_items,
                "skipped": 0,
                "subtitle_hint": (
                    f"已加载 {len(favorite_items)} 条精选节目"
                    if favorite_items
                    else "还没有精选节目，可在今日雷达选中卡片后点击「精选」"
                ),
                "cache_preview": True,
            },
            start_translations=False,
            persist_records=False,
        )
        self.current_task_var.set("当前任务：已打开精选收藏")
        self.set_active_nav("favorites")

    def collect_result_records(
        self,
        *,
        library_records: list[dict[str, object]] | None = None,
        transcription_paths: list[Path] | None = None,
        download_dir_text: str | None = None,
    ) -> list[dict[str, object]]:
        records: list[dict[str, object]] = []
        seen: set[str] = set()

        def add_record(result_type: str, title: str, raw_path: object) -> None:
            if not raw_path:
                return
            try:
                path = Path(str(raw_path)).expanduser()
            except Exception:
                return
            if not path.exists() or not path.is_file():
                return
            try:
                resolved = str(path.resolve())
                stat = path.stat()
            except OSError:
                return
            key = f"{result_type}|{resolved}"
            if key in seen:
                return
            seen.add(key)
            updated_at = datetime.fromtimestamp(stat.st_mtime)
            records.append(
                {
                    "type": result_type,
                    "title": title.strip() or self.result_title_from_path(path),
                    "updated": updated_at.strftime("%Y-%m-%d %H:%M"),
                    "path": str(path),
                    "sort": stat.st_mtime,
                }
            )

        if library_records is None:
            items = self.research_library.get("items", {})
            if isinstance(items, dict):
                library_records = [record for record in items.values() if isinstance(record, dict)]
            else:
                library_records = []
        for record in library_records:
            title = str(record.get("title") or "").strip()
            add_record("ZH 纪要", title, record.get("digest_path"))
            add_record("TXT 转录", title, record.get("transcript_path"))
            add_record("中文译文", title, record.get("full_translation_path"))
            audio_map = record.get("tts_audio_by_source")
            if isinstance(audio_map, dict):
                for source_path, entry in audio_map.items():
                    if not isinstance(entry, dict) or not entry.get("text_sha256") or not entry.get("scope"):
                        continue
                    audio_path = self.validated_tts_audio_path(
                        source_raw=source_path,
                        audio_raw=entry.get("audio_path"),
                        source_hash=entry.get("source_sha256"),
                    )
                    add_record("TTS 音频", title, audio_path)
            if record.get("tts_text_sha256") and record.get("tts_scope"):
                legacy_audio = self.validated_tts_audio_path(
                    source_raw=record.get("tts_source_path"),
                    audio_raw=record.get("tts_audio_path"),
                    source_hash=record.get("tts_source_sha256"),
                )
                add_record("TTS 音频", title, legacy_audio)
            if (
                record.get("full_translation_tts_scope") == "full_translation"
                and record.get("full_translation_tts_text_sha256")
            ):
                translated_audio = self.validated_tts_audio_path(
                    source_raw=record.get("full_translation_tts_source_path"),
                    audio_raw=record.get("full_translation_tts_audio_path"),
                    source_hash=record.get("full_translation_tts_source_sha256"),
                )
                add_record("TTS 音频", title, translated_audio)

        paths_snapshot = transcription_paths if transcription_paths is not None else self.transcription_paths
        for source_path in paths_snapshot:
            add_record("TXT 转录", source_path.stem, source_path.with_suffix(".txt"))

        download_dir_value = download_dir_text
        if download_dir_value is None:
            download_dir_value = self.download_dir_var.get().strip()
        scan_dirs = [
            Path(download_dir_value or DEFAULT_DOWNLOAD_DIR),
            DEFAULT_DOWNLOAD_DIR,
            RESEARCH_AUDIO_DIR,
        ]
        for directory in scan_dirs:
            if not directory.exists():
                continue
            for path in directory.glob("*.txt"):
                add_record("TXT 转录", self.result_title_from_path(path), path)

        if RESEARCH_DIGEST_DIR.exists():
            for path in RESEARCH_DIGEST_DIR.glob("*.zh.md"):
                add_record("ZH 纪要", self.result_title_from_path(path), path)

        records.sort(key=lambda item: float(item.get("sort") or 0), reverse=True)
        return records

    def result_title_from_path(self, path: Path) -> str:
        title = path.name
        for suffix in (".zh.full.md", ".zh.md", ".txt", ".md", ".wav", ".mp3", ".m4a", ".flac"):
            if title.lower().endswith(suffix):
                title = title[: -len(suffix)]
                break
        title = re.sub(r"\s*\[[^\]]+\]\s*$", "", title)
        return title.strip() or path.stem

    def mark_results_dirty(self, refresh_visible: bool = False) -> None:
        self.results_dirty = True
        if refresh_visible and self.current_page_key == "results":
            self.schedule_results_refresh()

    def ensure_results_current(self) -> None:
        if not hasattr(self, "results_tree"):
            return
        if self.current_page_key != "results":
            return
        if self.results_cache_ready and not self.results_dirty:
            self.render_results_from_cache_if_needed()
            return
        if self.results_refresh_running:
            self.results_summary_var.set("资料库后台索引中...")
            return
        self.schedule_results_refresh(delay_ms=120)

    def schedule_results_refresh(self, delay_ms: int = 220) -> None:
        if not hasattr(self, "results_tree"):
            return
        if self.results_refresh_job is not None:
            try:
                self.root.after_cancel(self.results_refresh_job)
            except tk.TclError:
                pass
        self.results_summary_var.set("资料库刷新中...")
        self.results_refresh_job = self.root.after(delay_ms, self.run_scheduled_results_refresh)

    def run_scheduled_results_refresh(self) -> None:
        self.results_refresh_job = None
        if self.current_page_key != "results":
            return
        self.refresh_results()

    def cancel_results_jobs(self, cancel_refresh: bool = False) -> None:
        if cancel_refresh and self.results_refresh_job is not None:
            try:
                self.root.after_cancel(self.results_refresh_job)
            except tk.TclError:
                pass
            self.results_refresh_job = None
        if self.results_render_job is not None:
            try:
                self.root.after_cancel(self.results_render_job)
            except tk.TclError:
                pass
            self.results_render_job = None

    def refresh_results(self, force: bool = True) -> None:
        if not hasattr(self, "results_tree"):
            return
        if force:
            self.results_dirty = True
        if self.results_cache_ready and not self.results_dirty:
            self.render_results_from_cache_if_needed()
            return
        if self.results_refresh_running:
            if force:
                self.results_refresh_pending = True
            self.results_summary_var.set("资料库后台索引中...")
            return
        self.cancel_results_jobs(cancel_refresh=True)
        if self.results_cache_ready and self.results_records_cache:
            self.results_summary_var.set(f"{self.results_summary_cache} · 后台更新中...")
            self.render_results_from_cache_if_needed()
        else:
            self.results_summary_var.set("资料库后台索引中...")
        snapshot = self.results_refresh_snapshot()
        self.results_refresh_generation += 1
        generation = self.results_refresh_generation
        self.results_refresh_running = True
        self.results_refresh_pending = False
        threading.Thread(target=self.results_refresh_worker, args=(generation, snapshot), daemon=True).start()

    def results_refresh_snapshot(self) -> dict[str, object]:
        items = self.research_library.get("items", {})
        if isinstance(items, dict):
            library_records = [dict(record) for record in items.values() if isinstance(record, dict)]
        else:
            library_records = []
        return {
            "library_records": library_records,
            "transcription_paths": list(self.transcription_paths),
            "download_dir_text": self.download_dir_var.get().strip(),
        }

    def results_refresh_worker(self, generation: int, snapshot: dict[str, object]) -> None:
        started_at = time.monotonic()
        try:
            records = self.collect_result_records(
                library_records=list(snapshot.get("library_records") or []),
                transcription_paths=list(snapshot.get("transcription_paths") or []),
                download_dir_text=str(snapshot.get("download_dir_text") or ""),
            )
            self.events.put(
                (
                    "results_refresh_done",
                    {
                        "generation": generation,
                        "records": records,
                        "elapsed": time.monotonic() - started_at,
                    },
                )
            )
        except Exception as exc:  # noqa: BLE001
            self.events.put(
                (
                    "results_refresh_error",
                    {
                        "generation": generation,
                        "error": clean_ui_status(str(exc), 240),
                    },
                )
            )

    def update_results_cache(self, records: list[dict[str, object]], elapsed: float = 0.0) -> None:
        self.results_records_cache = records
        self.results_cache_ready = True
        self.results_dirty = False
        self.results_cache_version += 1
        digest_count = sum(1 for item in records if item.get("type") == "ZH 纪要")
        transcript_count = sum(1 for item in records if item.get("type") == "TXT 转录")
        translation_count = sum(1 for item in records if item.get("type") == "中文译文")
        tts_count = sum(1 for item in records if item.get("type") == "TTS 音频")
        elapsed_text = f" · {elapsed:.1f}s" if elapsed >= 0.1 else ""
        self.results_summary_cache = (
            f"已索引 {len(records)} 个结果：纪要 {digest_count}，转录 {transcript_count}，"
            f"译文 {translation_count}，朗读 {tts_count}{elapsed_text}"
        )

    def on_results_refresh_done(self, payload: dict) -> None:
        if int(payload.get("generation") or 0) != self.results_refresh_generation:
            return
        self.results_refresh_running = False
        records = list(payload.get("records") or [])
        elapsed = float(payload.get("elapsed") or 0)
        self.update_results_cache(records, elapsed)
        if self.current_page_key == "results":
            self.start_results_render(records)
        if self.results_refresh_pending:
            self.results_refresh_pending = False
            self.results_dirty = True
            if self.current_page_key == "results":
                self.schedule_results_refresh(delay_ms=300)

    def on_results_refresh_error(self, payload: dict) -> None:
        if int(payload.get("generation") or 0) != self.results_refresh_generation:
            return
        self.results_refresh_running = False
        error = str(payload.get("error") or "未知错误")
        self.results_summary_var.set(f"资料库刷新失败：{error}")
        self.set_status(f"资料库刷新失败：{error}")
        if self.results_refresh_pending:
            self.results_refresh_pending = False
            if self.current_page_key == "results":
                self.schedule_results_refresh(delay_ms=500)

    def render_results_from_cache_if_needed(self) -> None:
        if self.current_page_key != "results" or not self.results_cache_ready:
            return
        if self.results_rendered_cache_version == self.results_cache_version:
            return
        self.start_results_render(self.results_records_cache)

    def start_results_render(self, records: list[dict[str, object]]) -> None:
        if not hasattr(self, "results_tree") or self.current_page_key != "results":
            return
        self.cancel_results_jobs(cancel_refresh=False)
        for row in self.results_tree.get_children():
            self.results_tree.delete(row)
        self.result_rows = {}
        self.results_render_records = records[:160]
        self.results_render_index = 0
        self.results_render_target_version = self.results_cache_version
        if not self.results_render_records:
            self.results_rendered_cache_version = self.results_render_target_version
            self.results_summary_var.set(self.results_summary_cache or "还没有生成转录或纪要")
            return
        self.results_summary_var.set(f"资料库刷新中... 0/{len(self.results_render_records)}")
        self.render_next_result_batch()

    def render_next_result_batch(self) -> None:
        if not hasattr(self, "results_tree"):
            return
        if self.current_page_key != "results":
            self.results_render_job = None
            return
        batch_size = 12
        records = self.results_render_records
        start = self.results_render_index
        end = min(start + batch_size, len(records))
        for record in records[start:end]:
            path_text = str(record.get("path") or "")
            row_id = hashlib.sha1(f"{record.get('type')}|{path_text}".encode("utf-8")).hexdigest()
            self.result_rows[row_id] = {
                "type": str(record.get("type") or ""),
                "title": str(record.get("title") or ""),
                "updated": str(record.get("updated") or ""),
                "path": path_text,
            }
            self.results_tree.insert(
                "",
                "end",
                iid=row_id,
                values=(
                    self.result_rows[row_id]["type"],
                    self.result_rows[row_id]["title"],
                    self.result_rows[row_id]["updated"],
                    path_text,
                ),
            )
        self.results_render_index = end
        if end < len(records):
            self.results_summary_var.set(f"资料库刷新中... {end}/{len(records)}")
            self.results_render_job = self.root.after(8, self.render_next_result_batch)
            return
        self.results_render_job = None
        self.results_rendered_cache_version = self.results_render_target_version
        self.results_summary_var.set(self.results_summary_cache)

    def selected_result_record(self) -> dict[str, str] | None:
        if not hasattr(self, "results_tree"):
            return None
        selection = self.results_tree.selection()
        if not selection:
            children = self.results_tree.get_children()
            if not children:
                messagebox.showinfo("没有结果", "还没有找到转录 TXT 或中文纪要。")
                return None
            first = children[0]
            self.results_tree.selection_set(first)
            selection = (first,)
        return self.result_rows.get(selection[0])

    def read_selected_result(self, _event: tk.Event | None = None) -> None:
        record = self.selected_result_record()
        if not record:
            return
        if is_audio_result_path(Path(record["path"])):
            self.open_path(Path(record["path"]))
            return
        self.load_reader_path(
            Path(record["path"]),
            title=record.get("title", ""),
            result_type=record.get("type", ""),
        )

    def item_transcript_path(self, item: VideoItem) -> Path | None:
        records = self.research_library.setdefault("items", {})
        record = records.get(item.video_id)
        if not isinstance(record, dict):
            record = self.video_item_record(item)
            records[item.video_id] = record

        raw_path = str(record.get("transcript_path") or "").strip()
        if raw_path:
            candidate = Path(raw_path).expanduser()
            if candidate.exists() and candidate.is_file():
                conflict, _reason = cached_transcript_identity_conflict(item, candidate)
                if not conflict:
                    return self.bind_item_transcript_path(item, record, candidate)

        search_dirs: list[Path] = [RESEARCH_AUDIO_DIR]
        configured_download_dir = self.download_dir_var.get().strip()
        if configured_download_dir:
            search_dirs.append(Path(configured_download_dir).expanduser())
        search_dirs.append(DEFAULT_DOWNLOAD_DIR)
        unique_dirs: list[Path] = []
        seen_dirs: set[str] = set()
        for directory in search_dirs:
            key = str(directory)
            if key not in seen_dirs:
                seen_dirs.add(key)
                unique_dirs.append(directory)

        candidate = None
        for directory in unique_dirs:
            candidate = find_existing_transcript(directory, item.video_id)
            if candidate:
                conflict, _reason = cached_transcript_identity_conflict(item, candidate)
                if not conflict:
                    break
                candidate = None
        if candidate is None:
            for directory in unique_dirs:
                candidate = find_existing_brief_transcript(directory, item.video_id)
                if candidate:
                    conflict, _reason = cached_transcript_identity_conflict(item, candidate)
                    if not conflict:
                        break
                    candidate = None
        if candidate is None:
            return None
        return self.bind_item_transcript_path(item, record, candidate)

    def bind_item_transcript_path(self, item: VideoItem, record: dict, candidate: Path) -> Path:
        previous_transcript_raw = str(record.get("transcript_path") or "").strip()
        changed = previous_transcript_raw != str(candidate)
        try:
            transcript_hash = self.source_sha256(candidate)
        except OSError:
            transcript_hash = ""
        if previous_transcript_raw:
            previous_transcript = Path(previous_transcript_raw).expanduser()
            try:
                path_changed = previous_transcript.resolve() != candidate.resolve()
            except OSError:
                path_changed = previous_transcript != candidate
            if path_changed and self.clear_stale_tts_cache_for_source(
                record,
                previous_transcript,
                force=True,
            ):
                changed = True
        if self.clear_stale_tts_cache_for_source(record, candidate, transcript_hash):
            changed = True
        cached_translation_hash = str(record.get("full_translation_source_sha256") or "")
        if cached_translation_hash and transcript_hash and cached_translation_hash != transcript_hash:
            self.clear_full_translation_cache(record, delete_files=True)
            changed = True
        record.update(
            {
                **self.video_item_record(item),
                "transcript_path": str(candidate),
                "transcript_sha256": transcript_hash,
                "transcript_kind": (
                    "episode_brief_fallback" if candidate.name.endswith(".brief.txt") else "transcript"
                ),
                "transcript_origin": (
                    "episode_brief_fallback"
                    if candidate.name.endswith(".brief.txt")
                    else (
                        str(record.get("transcript_origin"))
                        if str(record.get("transcript_origin") or "") not in {"", "episode_brief_fallback"}
                        else "existing_cache"
                    )
                ),
            }
        )
        if changed:
            self.save_research_library()
            self.mark_results_dirty(refresh_visible=True)
        return candidate

    @staticmethod
    def source_sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()

    def library_record_for_path(self, path: Path) -> tuple[str, dict] | None:
        try:
            target = path.expanduser().resolve()
        except OSError:
            target = path.expanduser()
        records = self.research_library.setdefault("items", {})
        for record_key, record in records.items():
            if not isinstance(record, dict):
                continue
            for field in (
                "digest_path",
                "transcript_path",
                "full_translation_path",
                "tts_audio_path",
                "full_translation_tts_audio_path",
            ):
                raw_path = str(record.get(field) or "").strip()
                if not raw_path:
                    continue
                try:
                    candidate = Path(raw_path).expanduser().resolve()
                except OSError:
                    candidate = Path(raw_path).expanduser()
                if candidate == target:
                    return str(record_key), record
        return None

    def ensure_local_translation_record(self, source_path: Path, title: str) -> tuple[str, dict]:
        existing = self.library_record_for_path(source_path)
        if existing:
            return existing
        resolved = str(source_path.expanduser().resolve())
        record_key = "local-" + hashlib.sha1(resolved.encode("utf-8")).hexdigest()[:16]
        records = self.research_library.setdefault("items", {})
        record = records.get(record_key)
        if not isinstance(record, dict):
            record = {
                "video_id": record_key,
                "title": title.strip() or self.result_title_from_path(source_path),
                "source_name": "本地资料",
                "source_type": "local_text",
                "published_text": "",
                "webpage_url": "",
                "transcript_path": resolved,
                "transcript_kind": (
                    "episode_brief_fallback" if source_path.name.endswith(".brief.txt") else "transcript"
                ),
                "updated_at": datetime.now(timezone.utc).isoformat(),
            }
            records[record_key] = record
            self.save_research_library()
        return record_key, record

    def full_translation_output_path(self, record_key: str, title: str) -> Path:
        short_key = hashlib.sha1(record_key.encode("utf-8")).hexdigest()[:10]
        title_prefix = sanitize_filename(title)[:60] or "译文"
        return RESEARCH_TRANSLATION_DIR / f"{title_prefix} [{short_key}].zh.full.md"

    @staticmethod
    def clear_full_translation_cache(record: dict, *, delete_files: bool) -> None:
        if delete_files:
            cache_roots = {
                "full_translation_path": RESEARCH_TRANSLATION_DIR,
                "full_translation_tts_audio_path": RESEARCH_TTS_DIR,
            }
            for field, cache_root in cache_roots.items():
                raw_path = str(record.get(field) or "").strip()
                if raw_path:
                    if cache_root == RESEARCH_TTS_DIR:
                        safe_unlink_tts_cache_artifact(Path(raw_path))
                    else:
                        safe_unlink_generated_cache(Path(raw_path), cache_root)
        for field in (
            "full_translation_path",
            "full_translation_source_path",
            "full_translation_source_sha256",
            "full_translation_model",
            "full_translation_prompt_version",
            "full_translated_at",
            "full_translation_tts_audio_path",
            "full_translation_tts_source_path",
            "full_translation_tts_source_sha256",
            "full_translation_tts_text_sha256",
            "full_translation_tts_scope",
            "full_translation_tts_generated_at",
        ):
            record.pop(field, None)

    def clear_stale_tts_cache_for_source(
        self,
        record: dict,
        source_path: Path,
        current_hash: str = "",
        *,
        force: bool = False,
    ) -> bool:
        try:
            source_key = str(source_path.expanduser().resolve())
        except OSError:
            source_key = str(source_path.expanduser())
        changed = False
        audio_map = record.get("tts_audio_by_source")
        if isinstance(audio_map, dict):
            entry = audio_map.get(source_key)
            if isinstance(entry, dict):
                stored_hash = str(entry.get("source_sha256") or "")
                if force or not current_hash or stored_hash != current_hash:
                    raw_audio = str(entry.get("audio_path") or "").strip()
                    if raw_audio:
                        safe_unlink_tts_cache_artifact(Path(raw_audio))
                    audio_map.pop(source_key, None)
                    changed = True
            if not audio_map:
                record.pop("tts_audio_by_source", None)

        legacy_source = str(record.get("tts_source_path") or "").strip()
        try:
            legacy_matches = (
                bool(legacy_source)
                and Path(legacy_source).expanduser().resolve() == source_path.expanduser().resolve()
            )
        except OSError:
            legacy_matches = legacy_source == source_key
        legacy_hash = str(record.get("tts_source_sha256") or "")
        if legacy_matches and (force or not current_hash or legacy_hash != current_hash):
            raw_audio = str(record.get("tts_audio_path") or "").strip()
            if raw_audio:
                safe_unlink_tts_cache_artifact(Path(raw_audio))
            for field in (
                "tts_audio_path",
                "tts_source_path",
                "tts_source_sha256",
                "tts_text_sha256",
                "tts_scope",
                "tts_generated_at",
            ):
                record.pop(field, None)
            changed = True
        return changed

    def validated_tts_audio_path(
        self,
        *,
        source_raw: object,
        audio_raw: object,
        source_hash: object,
    ) -> Path | None:
        source_text = str(source_raw or "").strip()
        audio_text = str(audio_raw or "").strip()
        expected_hash = str(source_hash or "").strip()
        if not source_text or not audio_text or not expected_hash:
            return None
        source_path = Path(source_text).expanduser()
        audio_path = Path(audio_text).expanduser()
        if not source_path.exists() or not source_path.is_file():
            return None
        if not audio_path.exists() or not is_readable_audio_cache(audio_path):
            return None
        try:
            if self.source_sha256(source_path) != expected_hash:
                return None
        except OSError:
            return None
        return audio_path

    def translation_cache_matches(self, record: dict, source_path: Path, source_hash: str) -> Path | None:
        raw_translation_path = str(record.get("full_translation_path") or "").strip()
        if not raw_translation_path:
            return None
        translation_path = Path(raw_translation_path).expanduser()
        if not translation_path.exists() or not translation_path.is_file():
            return None
        if str(record.get("full_translation_source_sha256") or "") != source_hash:
            return None
        if str(record.get("full_translation_model") or "") != FULL_TRANSLATION_MODEL:
            return None
        if str(record.get("full_translation_prompt_version") or "") != FULL_TRANSLATION_PROMPT_VERSION:
            return None
        stored_source = str(record.get("full_translation_source_path") or "").strip()
        if stored_source:
            try:
                if Path(stored_source).expanduser().resolve() != source_path.expanduser().resolve():
                    return None
            except OSError:
                return None
        return translation_path

    def show_item_translation(self, item: VideoItem) -> None:
        source_path = self.item_transcript_path(item)
        if source_path is None:
            messagebox.showinfo(
                "还没有英文转录",
                "这条节目还没有可用的转录文本。请先点“确认转译”，程序会按设置中的转录模式生成。",
            )
            return
        record = self.research_library.setdefault("items", {}).setdefault(item.video_id, self.video_item_record(item))
        self.start_full_translation(
            item.video_id,
            record,
            source_path,
            self.card_display_title(item),
            autoplay=False,
        )

    def play_item_translation(self, item: VideoItem) -> None:
        source_path = self.item_transcript_path(item)
        if source_path is None:
            messagebox.showinfo(
                "还没有英文转录",
                "这条节目还没有可用的转录文本。请先点“确认转译”，程序会按设置中的转录模式生成。",
            )
            return
        record = self.research_library.setdefault("items", {}).setdefault(item.video_id, self.video_item_record(item))
        self.start_full_translation(
            item.video_id,
            record,
            source_path,
            self.card_display_title(item),
            autoplay=True,
        )

    def show_reader_translation(self) -> None:
        source_path = self.reader_translation_source_path
        if source_path is None or not source_path.exists():
            messagebox.showinfo(
                "当前资料不能生成译文",
                "请先打开英文 TXT 转录。中文纪要没有对应英文原文时，不会重复翻译。",
            )
            return
        match = self.library_record_for_path(source_path)
        if match:
            record_key, record = match
        else:
            record_key, record = self.ensure_local_translation_record(
                source_path,
                self.reader_title_var.get().strip(),
            )
        self.start_full_translation(
            record_key,
            record,
            source_path,
            self.reader_title_var.get().strip(),
            autoplay=False,
        )

    def play_reader_translation(self) -> None:
        source_path = self.reader_translation_source_path
        if source_path is None or not source_path.exists():
            messagebox.showinfo(
                "当前资料不能播放译文",
                "请先打开英文 TXT 转录，再点“播放译文”。",
            )
            return
        match = self.library_record_for_path(source_path)
        if match:
            record_key, record = match
        else:
            record_key, record = self.ensure_local_translation_record(
                source_path,
                self.reader_title_var.get().strip(),
            )
        self.start_full_translation(
            record_key,
            record,
            source_path,
            self.reader_title_var.get().strip(),
            autoplay=True,
        )

    def start_full_translation(
        self,
        record_key: str,
        record: dict,
        source_path: Path,
        title: str,
        *,
        autoplay: bool,
    ) -> None:
        source_path = source_path.expanduser()
        if not source_path.exists() or not source_path.is_file():
            messagebox.showwarning("找不到英文原文", f"路径不存在：\n{source_path}")
            return
        try:
            source_hash = self.source_sha256(source_path)
        except OSError as exc:
            messagebox.showerror("读取英文原文失败", str(exc))
            return

        display_title = title.strip() or str(record.get("title") or "").strip() or self.result_title_from_path(source_path)
        translation_label = "简介译文" if source_path.name.endswith(".brief.txt") else "中文全文"
        self.translation_request_serial += 1
        request_id = self.translation_request_serial
        self.latest_translation_request_id = request_id
        request_state: dict[str, object] = {
            "request_id": request_id,
            "autoplay": autoplay,
            "source": str(source_path.resolve()),
            "reader_navigation_serial": self.reader_navigation_serial,
            "start_page_key": self.current_page_key,
        }
        cached_path = self.translation_cache_matches(record, source_path, source_hash)
        if cached_path:
            self.reader_translation_status_var.set(f"{translation_label}：已命中本地缓存")
            self.load_reader_path(cached_path, title=display_title, result_type="中文译文")
            if autoplay:
                expected_path = str(cached_path.resolve())
                expected_navigation = self.reader_navigation_serial
                self.root.after(
                    120,
                    lambda: self.play_or_generate_loaded_translation(
                        expected_path,
                        expected_navigation,
                    ),
                )
            return

        output_path = self.full_translation_output_path(record_key, display_title)
        source_token = f"record:{record_key}"
        if source_token in self.full_translation_inflight:
            active_state = self.full_translation_inflight[source_token]
            if str(active_state.get("source") or "") == str(source_path.resolve()):
                request_state["autoplay"] = bool(active_state.get("autoplay")) or autoplay
                self.full_translation_inflight[source_token] = request_state
                message = (
                    f"{translation_label}：正在生成，完成后会自动播放"
                    if request_state["autoplay"]
                    else f"{translation_label}：正在生成，请稍候"
                )
                self.reader_translation_status_var.set(message)
                self.set_status("这篇资料的中文全文正在生成，已合并本次操作...")
            else:
                self.reader_translation_status_var.set("中文全文：旧转录的译文正在收尾，请稍后再点")
                self.set_status("同一节目的旧转录仍在翻译；完成后会自动丢弃过期结果。")
            return
        api_key = self.api_key_var.get().strip()
        if not api_key:
            messagebox.showerror(
                "缺少翻译接口",
                "请先在“设置”页填写 DASHSCOPE_API_KEY。全文翻译只在点击后调用。",
            )
            return

        metadata = FullTranslationMetadata(
            title=display_title,
            source_name=str(record.get("source_name") or ""),
            published_text=str(record.get("published_text") or ""),
            webpage_url=str(record.get("webpage_url") or ""),
        )
        self.full_translation_inflight[source_token] = request_state
        self.reader_translation_status_var.set(f"{translation_label}：准备按原文分段翻译...")
        self.set_status(f"正在生成{translation_label}：{display_title}")

        def worker() -> None:
            try:
                build_chinese_full_translation(
                    source_path=source_path,
                    output_path=output_path,
                    metadata=metadata,
                    api_key=api_key,
                    model=FULL_TRANSLATION_MODEL,
                    progress_callback=lambda done, total: self.events.put(
                        (
                            "full_translation_progress",
                            {
                                "source_token": source_token,
                                "title": display_title,
                                "translation_label": translation_label,
                                "done": done,
                                "total": total,
                            },
                        )
                    ),
                )
                self.events.put(
                    (
                        "full_translation_ready",
                        {
                            "record_key": record_key,
                            "source": str(source_path),
                            "source_token": source_token,
                            "source_hash": source_hash,
                            "output": str(output_path),
                            "title": display_title,
                            "translation_label": translation_label,
                            "autoplay": autoplay,
                        },
                    )
                )
            except Exception as exc:  # noqa: BLE001
                self.events.put(
                    (
                        "full_translation_error",
                        {
                            "source_token": source_token,
                            "title": display_title,
                            "error": clean_ui_status(str(exc), 300),
                        },
                    )
                )

        threading.Thread(target=worker, daemon=True).start()

    def on_full_translation_progress(self, payload: dict) -> None:
        source_token = str(payload.get("source_token") or "")
        request_state = self.full_translation_inflight.get(source_token, {})
        if not self.translation_request_is_active(request_state):
            return
        done = int(payload.get("done") or 0)
        total = max(1, int(payload.get("total") or 1))
        title = str(payload.get("title") or "当前资料")
        translation_label = str(payload.get("translation_label") or "中文全文")
        self.reader_translation_status_var.set(f"{translation_label}：正在翻译 {done}/{total} 段")
        self.set_status(f"《{title}》全文翻译进度 {done}/{total}")

    def on_full_translation_ready(self, payload: dict) -> None:
        source_token = str(payload.get("source_token") or "")
        request_state = self.full_translation_inflight.pop(source_token, {})
        interactive = self.translation_request_is_active(request_state)
        autoplay = bool(request_state.get("autoplay"))
        source_path = Path(str(payload.get("source") or "")).expanduser()
        output_path = Path(str(payload.get("output") or "")).expanduser()
        try:
            current_hash = self.source_sha256(source_path)
        except OSError as exc:
            safe_unlink_generated_cache(output_path, RESEARCH_TRANSLATION_DIR)
            self.on_full_translation_error(
                {
                    "source_token": source_token,
                    "error": f"译文生成后无法复核英文原文：{exc}",
                    "interactive": interactive,
                }
            )
            return
        if current_hash != str(payload.get("source_hash") or ""):
            safe_unlink_generated_cache(output_path, RESEARCH_TRANSLATION_DIR)
            self.on_full_translation_error(
                {
                    "source_token": source_token,
                    "error": "生成期间英文原文发生变化，已丢弃旧译文，请重新点击。",
                    "interactive": interactive,
                }
            )
            return

        record_key = str(payload.get("record_key") or "")
        records = self.research_library.setdefault("items", {})
        record = records.get(record_key)
        if not isinstance(record, dict):
            safe_unlink_generated_cache(output_path, RESEARCH_TRANSLATION_DIR)
            self.on_full_translation_error(
                {
                    "source_token": source_token,
                    "error": "这条资料已从资料库移除，刚生成的译文未写入。",
                    "interactive": interactive,
                }
            )
            return
        current_source_raw = str(record.get("transcript_path") or "").strip()
        try:
            current_source_matches = (
                bool(current_source_raw)
                and Path(current_source_raw).expanduser().resolve() == source_path.resolve()
            )
        except OSError:
            current_source_matches = False
        if not current_source_matches:
            safe_unlink_generated_cache(output_path, RESEARCH_TRANSLATION_DIR)
            self.on_full_translation_error(
                {
                    "source_token": source_token,
                    "error": "生成期间这条节目的英文转录已更新，旧译文已丢弃，请重新点击。",
                    "interactive": interactive,
                }
            )
            return
        previous_translation = str(record.get("full_translation_path") or "").strip()
        if previous_translation:
            try:
                translation_changed_path = (
                    Path(previous_translation).expanduser().resolve() != output_path.resolve()
                )
            except OSError:
                translation_changed_path = True
            if translation_changed_path:
                safe_unlink_generated_cache(Path(previous_translation), RESEARCH_TRANSLATION_DIR)
        previous_audio = str(record.get("full_translation_tts_audio_path") or "").strip()
        if previous_audio:
            try:
                safe_unlink_tts_cache_artifact(Path(previous_audio))
            except OSError:
                pass
        for field in (
            "full_translation_tts_audio_path",
            "full_translation_tts_source_path",
            "full_translation_tts_source_sha256",
            "full_translation_tts_text_sha256",
            "full_translation_tts_scope",
            "full_translation_tts_generated_at",
        ):
            record.pop(field, None)
        record.update(
            {
                "full_translation_path": str(output_path),
                "full_translation_source_path": str(source_path),
                "full_translation_source_sha256": current_hash,
                "full_translation_model": FULL_TRANSLATION_MODEL,
                "full_translation_prompt_version": FULL_TRANSLATION_PROMPT_VERSION,
                "full_translated_at": datetime.now(timezone.utc).isoformat(),
            }
        )
        self.save_research_library()
        self.mark_results_dirty(refresh_visible=True)
        if interactive:
            translation_label = str(payload.get("translation_label") or "中文全文")
            self.reader_translation_status_var.set(f"{translation_label}：生成完成，已缓存")
            self.set_status(f"中文全文译文已生成：{output_path.name}")
            self.load_reader_path(
                output_path,
                title=str(payload.get("title") or ""),
                result_type="中文译文",
            )
            if autoplay:
                expected_path = str(output_path.resolve())
                expected_navigation = self.reader_navigation_serial
                self.root.after(
                    120,
                    lambda: self.play_or_generate_loaded_translation(
                        expected_path,
                        expected_navigation,
                    ),
                )
        else:
            self.set_status(f"中文全文已在后台生成并缓存：{output_path.name}")

    def on_full_translation_error(self, payload: dict) -> None:
        source_token = str(payload.get("source_token") or "")
        request_state = self.full_translation_inflight.pop(source_token, {})
        interactive_raw = payload.get("interactive")
        interactive = (
            bool(interactive_raw)
            if interactive_raw is not None
            else self.translation_request_is_active(request_state)
        )
        message = str(payload.get("error") or "未知错误")
        self.set_status(f"中文全文翻译失败：{message}")
        if interactive:
            self.reader_translation_status_var.set(f"中文全文失败：{message}")
            messagebox.showerror("中文全文翻译失败", message)

    def translation_request_is_active(self, request_state: dict[str, object]) -> bool:
        if not request_state:
            return False
        return (
            int(request_state.get("request_id") or 0) == self.latest_translation_request_id
            and int(request_state.get("reader_navigation_serial") or 0) == self.reader_navigation_serial
            and str(request_state.get("start_page_key") or "") == self.current_page_key
        )

    def play_or_generate_loaded_translation(
        self,
        expected_path: str = "",
        expected_navigation: int | None = None,
    ) -> None:
        if not self.reader_current_path:
            return
        if expected_path:
            try:
                if str(self.reader_current_path.resolve()) != expected_path:
                    return
            except OSError:
                return
        if expected_navigation is not None and expected_navigation != self.reader_navigation_serial:
            return
        visible_text = self.reader_text.get("1.0", "end-1c") if hasattr(self, "reader_text") else ""
        prepared_text, _truncated = self.prepare_tts_text(visible_text, 0)
        expected_text_hash = hashlib.sha256(prepared_text.encode("utf-8")).hexdigest() if prepared_text else ""
        if (
            self.reader_tts_audio_path
            and self.reader_tts_audio_path.exists()
            and self.reader_tts_scope == "full_translation"
            and self.reader_tts_text_sha256 == expected_text_hash
        ):
            self.play_reader_tts_audio()
            return
        self.generate_reader_tts(selection_only=False, auto_play=True, full_text=True)

    def load_reader_path(self, path: Path, *, title: str = "", result_type: str = "") -> None:
        path = path.expanduser()
        if not path.exists() or not path.is_file():
            messagebox.showwarning("找不到文件", f"路径不存在：\n{path}")
            return
        try:
            stat = path.stat()
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError as exc:
            messagebox.showerror("读取失败", str(exc))
            return

        max_chars = 600000
        if len(text) > max_chars:
            text = text[:max_chars] + "\n\n[内容过长，阅读页已截断；原文件仍可用“外部打开”查看完整内容。]"

        self.reader_navigation_serial += 1
        self.stop_reader_tts_audio(update_status=False)
        display_title = title.strip() or self.result_title_from_path(path)
        if result_type.strip():
            display_type = result_type.strip()
        elif path.name.endswith(".zh.full.md"):
            display_type = "中文译文"
        elif path.name.endswith(".zh.md"):
            display_type = "ZH 纪要"
        else:
            display_type = "TXT 转录"
        updated = datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M")
        self.reader_current_path = path
        self.reader_current_result_type = display_type
        self.reader_translation_source_path = None
        self.reader_tts_audio_path = None
        self.reader_tts_source_path = None
        self.reader_tts_scope = ""
        self.reader_tts_text_sha256 = ""
        self.set_reader_text(text)
        visible_text = self.reader_text.get("1.0", "end-1c") if hasattr(self, "reader_text") else text
        full_tts_text, _ = self.prepare_tts_text(visible_text, 0)
        page_tts_text, _ = self.prepare_tts_text(visible_text, self.tts_max_chars())
        full_tts_hash = hashlib.sha256(full_tts_text.encode("utf-8")).hexdigest() if full_tts_text else ""
        page_tts_hash = hashlib.sha256(page_tts_text.encode("utf-8")).hexdigest() if page_tts_text else ""

        match = self.library_record_for_path(path)
        if not match and path.suffix.lower() == ".txt":
            match = self.ensure_local_translation_record(path, display_title)
        if match:
            _record_key, record = match
            transcript_raw = str(record.get("transcript_path") or "").strip()
            transcript_path = Path(transcript_raw).expanduser() if transcript_raw else None
            source_is_brief = bool(
                transcript_path
                and (
                    transcript_path.name.endswith(".brief.txt")
                    or str(record.get("transcript_kind") or "") == "episode_brief_fallback"
                )
            )
            translation_raw = str(record.get("full_translation_path") or "").strip()
            translation_path = Path(translation_raw).expanduser() if translation_raw else None
            if transcript_path and transcript_path.exists():
                self.reader_translation_source_path = transcript_path
            elif path.suffix.lower() == ".txt":
                self.reader_translation_source_path = path

            is_translation = False
            if translation_path:
                try:
                    is_translation = translation_path.resolve() == path.resolve()
                except OSError:
                    is_translation = translation_path == path
            audio_raw = ""
            expected_audio_source = ""
            expected_audio_hash = ""
            expected_text_hash = ""
            audio_scope = ""
            if is_translation:
                audio_raw = str(record.get("full_translation_tts_audio_path") or "").strip()
                expected_audio_source = str(record.get("full_translation_tts_source_path") or "").strip()
                expected_audio_hash = str(record.get("full_translation_tts_source_sha256") or "").strip()
                expected_text_hash = str(record.get("full_translation_tts_text_sha256") or "").strip()
                audio_scope = str(record.get("full_translation_tts_scope") or "").strip()
            else:
                source_key = str(path.resolve())
                audio_map = record.get("tts_audio_by_source")
                audio_entry = audio_map.get(source_key) if isinstance(audio_map, dict) else None
                if isinstance(audio_entry, dict):
                    audio_raw = str(audio_entry.get("audio_path") or "").strip()
                    expected_audio_source = source_key
                    expected_audio_hash = str(audio_entry.get("source_sha256") or "").strip()
                    expected_text_hash = str(audio_entry.get("text_sha256") or "").strip()
                    audio_scope = str(audio_entry.get("scope") or "").strip()
                else:
                    audio_raw = str(record.get("tts_audio_path") or "").strip()
                    expected_audio_source = str(record.get("tts_source_path") or "").strip()
                    expected_audio_hash = str(record.get("tts_source_sha256") or "").strip()
                    expected_text_hash = str(record.get("tts_text_sha256") or "").strip()
                    audio_scope = str(record.get("tts_scope") or "").strip()
            if audio_raw:
                audio_path = Path(audio_raw).expanduser()
                try:
                    source_matches = (
                        bool(expected_audio_source)
                        and Path(expected_audio_source).expanduser().resolve() == path.resolve()
                    )
                    hash_matches = (
                        bool(expected_audio_hash)
                        and self.source_sha256(path) == expected_audio_hash
                    )
                    expected_current_text_hash = (
                        full_tts_hash
                        if audio_scope == "full_translation"
                        else page_tts_hash if audio_scope == "page" else ""
                    )
                    text_matches = (
                        bool(expected_text_hash)
                        and bool(expected_current_text_hash)
                        and expected_text_hash == expected_current_text_hash
                    )
                except OSError:
                    source_matches = False
                    hash_matches = False
                    text_matches = False
                if (
                    audio_path.exists()
                    and audio_path.is_file()
                    and source_matches
                    and hash_matches
                    and text_matches
                ):
                    self.reader_tts_audio_path = audio_path
                    self.reader_tts_source_path = path
                    self.reader_tts_scope = audio_scope
                    self.reader_tts_text_sha256 = expected_text_hash

            if is_translation:
                self.reader_translation_status_var.set(
                    "简介译文：已载入（当前只有简介兜底稿，非完整逐字转录）"
                    if source_is_brief
                    else "中文全文：已载入本地缓存"
                )
            elif translation_path and translation_path.exists():
                self.reader_translation_status_var.set(
                    "简介译文：已有缓存（非完整逐字转录）"
                    if source_is_brief
                    else "中文全文：已有缓存，点击“看译文”打开"
                )
            elif self.reader_translation_source_path:
                self.reader_translation_status_var.set(
                    "简介译文：当前只有简介兜底稿；可先生成，但建议后续补做 Whisper 全转录"
                    if source_is_brief
                    else "中文全文：点击“看译文”按需生成"
                )
            else:
                self.reader_translation_status_var.set("中文全文：当前资料没有对应英文转录")
        elif path.suffix.lower() == ".txt":
            self.reader_translation_source_path = path
            self.reader_translation_status_var.set(
                "简介译文：当前只有简介兜底稿，非完整逐字转录"
                if path.name.endswith(".brief.txt")
                else "中文全文：点击“看译文”按需生成"
            )
        else:
            self.reader_translation_status_var.set("中文全文：当前资料没有对应英文转录")

        if self.reader_tts_audio_path:
            self.tts_status_var.set(f"本地朗读：已有音频 {self.reader_tts_audio_path.name}")
        else:
            self.tts_status_var.set("本地朗读：当前资料还没有音频")
        self.reader_title_var.set(display_title)
        self.reader_meta_var.set(f"{display_type} · {updated} · {format_file_size(stat.st_size)}")
        self.reader_path_var.set(str(path))
        self.show_page("reader")

    def set_reader_text(self, text: str) -> None:
        if not hasattr(self, "reader_text"):
            return
        self.reader_text.configure(state="normal")
        self.reader_text.delete("1.0", "end")
        if not text.strip():
            self.reader_text.insert("end", "这个文件暂时没有可显示的文本。", "muted")
        else:
            self.insert_rendered_markdown(self.reader_text, text)
        self.reader_text.see("1.0")
        self.reader_text.configure(state="disabled")

    def open_reader_current_path(self) -> None:
        if not self.reader_current_path:
            messagebox.showinfo("还没有资料", "先从「资料库」选择一条 TXT 转录或 ZH 纪要。")
            return
        self.open_path(self.reader_current_path)

    def reveal_reader_current_path(self) -> None:
        if not self.reader_current_path:
            messagebox.showinfo("还没有资料", "先从「资料库」选择一条 TXT 转录或 ZH 纪要。")
            return
        self.reveal_path(self.reader_current_path)

    def choose_tts_python(self) -> None:
        current = Path(self.tts_python_var.get().strip() or str(DEFAULT_TTS_PYTHON)).expanduser()
        chosen = filedialog.askopenfilename(
            title="选择 Qwen3-TTS 虚拟环境里的 python",
            initialdir=str(current.parent if current.parent.exists() else Path.home()),
            filetypes=[("Python", "python*"), ("所有文件", "*.*")],
        )
        if chosen:
            self.tts_python_var.set(chosen)
            self.save_settings()

    def generate_reader_tts_selected(self) -> None:
        self.generate_reader_tts(selection_only=True)

    def generate_reader_tts_current(self) -> None:
        self.generate_reader_tts(selection_only=False)

    def reader_text_for_tts(self, selection_only: bool) -> tuple[str, bool]:
        if not hasattr(self, "reader_text"):
            return "", False
        try:
            selected_text = self.reader_text.get("sel.first", "sel.last")
        except tk.TclError:
            selected_text = ""
        if selected_text.strip():
            return selected_text, True
        if selection_only:
            return "", False
        return self.reader_text.get("1.0", "end-1c"), False

    def tts_max_chars(self) -> int:
        try:
            value = int(self.tts_max_chars_var.get().strip())
        except ValueError:
            value = DEFAULT_TTS_MAX_CHARS
        return max(300, min(8000, value))

    def prepare_tts_text(self, text: str, max_chars: int) -> tuple[str, bool]:
        cleaned = text or ""
        cleaned = re.sub(r"(?s)```.*?```", " ", cleaned)
        cleaned = re.sub(r"!\[[^\]]*\]\([^)]+\)", " ", cleaned)
        cleaned = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", cleaned)
        cleaned = re.sub(r"`([^`]+)`", r"\1", cleaned)
        cleaned = re.sub(r"https?://\S+", " ", cleaned)
        cleaned_lines: list[str] = []
        for line in cleaned.splitlines():
            line = re.sub(r"^\s{0,3}#{1,6}\s*", "", line)
            line = re.sub(r"^\s{0,3}(?:>|[-*+])\s+", "", line)
            line = re.sub(r"^\s{0,3}\d{1,3}[.)]\s+", "", line)
            cleaned_lines.append(line)
        cleaned = "\n".join(cleaned_lines)
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        if max_chars <= 0:
            return cleaned, False
        if len(cleaned) <= max_chars:
            return cleaned, False
        return cleaned[:max_chars].rstrip(" ，。；、,.") + "。", True

    def generate_reader_tts(
        self,
        selection_only: bool = False,
        *,
        auto_play: bool = False,
        full_text: bool = False,
    ) -> None:
        if self.tts_running:
            messagebox.showinfo("正在生成", "本地朗读音频还在生成中。")
            return
        if not self.reader_current_path:
            messagebox.showinfo("还没有资料", "先从「资料库」选择一条 TXT 转录或 ZH 纪要。")
            return

        if full_text and hasattr(self, "reader_text"):
            raw_text = self.reader_text.get("1.0", "end-1c")
            from_selection = False
        else:
            raw_text, from_selection = self.reader_text_for_tts(selection_only)
        if not raw_text.strip():
            messagebox.showinfo("没有选中文本", "请先在阅读区选中一段文字，或使用“朗读当前页”。")
            return

        max_chars = 0 if full_text else self.tts_max_chars()
        text, truncated = self.prepare_tts_text(raw_text, max_chars)
        if not text:
            messagebox.showinfo("没有可朗读文本", "当前内容清理后为空。")
            return
        tts_scope = (
            "full_translation"
            if full_text and self.reader_current_result_type == "中文译文"
            else "selection" if from_selection else "page"
        )
        tts_text_sha256 = hashlib.sha256(text.encode("utf-8")).hexdigest()

        chunk_chars = min(1500, max(600, self.tts_max_chars()))
        tts_config = {
            "python": self.tts_python_var.get().strip() or str(DEFAULT_TTS_PYTHON),
            "model": self.tts_model_var.get().strip() or DEFAULT_TTS_MODEL,
            "speaker": self.tts_speaker_var.get().strip() or DEFAULT_TTS_SPEAKER,
            "language": self.tts_language_var.get().strip() or DEFAULT_TTS_LANGUAGE,
            "instruct": self.tts_instruct_var.get().strip(),
            "max_chars": str(max_chars),
            "chunk_chars": str(chunk_chars),
        }
        python_path = Path(tts_config["python"]).expanduser()
        if not python_path.exists():
            messagebox.showerror(
                "找不到 TTS 环境",
                f"没有找到 Qwen3-TTS Python：\n{python_path}\n\n可在「设置」里重新选择。",
            )
            return

        ensure_dir(RESEARCH_TTS_DIR)
        source_path = self.reader_current_path.expanduser().resolve()
        try:
            source_hash = self.source_sha256(source_path)
        except OSError as exc:
            messagebox.showerror("读取朗读原文失败", str(exc))
            return
        source_title = self.reader_title_var.get().strip() or self.result_title_from_path(source_path)
        digest = hashlib.sha1(
            "|".join(
                [
                    str(source_path),
                    text,
                    tts_config["model"],
                    tts_config["speaker"],
                    tts_config["language"],
                    tts_config["instruct"],
                    tts_config["chunk_chars"],
                    "full" if full_text else "limited",
                    "chunked-v2",
                ]
            ).encode("utf-8")
        ).hexdigest()[:10]
        title_prefix = sanitize_filename(source_title)[:60]
        output_path = RESEARCH_TTS_DIR / f"{title_prefix} - 朗读 [{digest}].wav"
        text_path = RESEARCH_TTS_DIR / f"{title_prefix} - 朗读 [{digest}].txt"
        text_path.write_text(text + "\n", encoding="utf-8")

        if output_path.exists() and is_readable_audio_cache(output_path):
            self.reader_tts_audio_path = output_path
            self.reader_tts_source_path = source_path
            self.tts_status_var.set(f"本地朗读：已命中缓存 {output_path.name}")
            self.reader_tts_scope = tts_scope
            self.reader_tts_text_sha256 = tts_text_sha256
            self.attach_reader_tts_record(
                output_path,
                source_path,
                source_hash,
                tts_text_sha256,
                tts_scope,
            )
            self.mark_results_dirty(refresh_visible=True)
            if auto_play:
                self.play_reader_tts_audio()
            return

        self.save_settings()
        self.tts_running = True
        scope_text = "整篇译文" if full_text else ("选中文本" if from_selection else "当前页")
        trunc_text = f"，已按 {max_chars} 字截断" if truncated else ""
        self.tts_status_var.set(f"本地朗读：正在分段生成 {scope_text}{trunc_text}，首次加载模型会比较慢")
        self.set_status("正在调用本机 Qwen3-TTS 生成朗读音频...")
        threading.Thread(
            target=self.tts_worker,
            args=(
                text_path,
                output_path,
                source_path,
                source_hash,
                tts_text_sha256,
                tts_scope,
                source_title,
                truncated,
                auto_play,
                self.reader_navigation_serial,
                self.current_page_key,
                tts_config,
            ),
            daemon=True,
        ).start()

    def tts_worker(
        self,
        text_path: Path,
        output_path: Path,
        source_path: Path,
        source_hash: str,
        tts_text_sha256: str,
        tts_scope: str,
        source_title: str,
        truncated: bool,
        auto_play: bool,
        reader_navigation_serial: int,
        start_page_key: str,
        tts_config: dict[str, str],
    ) -> None:
        python_path = Path(tts_config.get("python") or str(DEFAULT_TTS_PYTHON)).expanduser()
        bridge_path = ensure_qwen_tts_bridge_script()
        command = [
            str(python_path),
            str(bridge_path),
            "--text-file",
            str(text_path),
            "--output",
            str(output_path),
            "--model",
            tts_config.get("model") or DEFAULT_TTS_MODEL,
            "--speaker",
            tts_config.get("speaker") or DEFAULT_TTS_SPEAKER,
            "--language",
            tts_config.get("language") or DEFAULT_TTS_LANGUAGE,
            "--instruct",
            tts_config.get("instruct") or "",
            "--max-chars",
            tts_config.get("max_chars") or str(DEFAULT_TTS_MAX_CHARS),
            "--chunk-chars",
            tts_config.get("chunk_chars") or "1200",
        ]
        env = os.environ.copy()
        env.setdefault("TOKENIZERS_PARALLELISM", "false")
        env.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
        env.setdefault("HF_HUB_DISABLE_XET", "1")
        process: subprocess.Popen | None = None
        try:
            if self.app_closing:
                raise RuntimeError("应用正在退出，已取消朗读生成")
            process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                env=env,
                bufsize=1,
            )
            with self.tts_generation_lock:
                if self.app_closing:
                    terminate_process_tree(process)
                    raise RuntimeError("应用正在退出，已取消朗读生成")
                self.tts_generation_process = process
            output_lines: list[str] = []
            final_payload: dict | None = None
            if process.stdout is not None:
                for raw_line in process.stdout:
                    line = raw_line.rstrip()
                    if line:
                        output_lines.append(line)
                    event_payload = parse_json_line(line)
                    if not event_payload:
                        continue
                    if event_payload.get("progress"):
                        self.events.put(
                            (
                                "tts_progress",
                                {
                                    "source": str(source_path),
                                    "done": event_payload.get("done"),
                                    "total": event_payload.get("total"),
                                },
                            )
                        )
                    if event_payload.get("ok"):
                        final_payload = event_payload
            return_code = process.wait()
            if return_code != 0 or not final_payload or not final_payload.get("ok"):
                err_payload = parse_json_line("\n".join(output_lines)) or {}
                error_text = err_payload.get("error") or "\n".join(output_lines[-8:]) or "未知错误"
                raise RuntimeError(clean_ui_status(error_text, 260))
            self.events.put(
                (
                    "tts_done",
                    {
                        "output": str(output_path),
                        "source": str(source_path),
                        "source_hash": source_hash,
                        "text_sha256": tts_text_sha256,
                        "scope": tts_scope,
                        "title": source_title,
                        "seconds": final_payload.get("seconds"),
                        "sample_rate": final_payload.get("sample_rate"),
                        "device": final_payload.get("device"),
                        "load_warning": final_payload.get("load_warning") or "",
                        "chunks": final_payload.get("chunks"),
                        "truncated": truncated,
                        "auto_play": auto_play,
                        "reader_navigation_serial": reader_navigation_serial,
                        "start_page_key": start_page_key,
                    },
                )
            )
        except Exception as exc:  # noqa: BLE001
            safe_unlink_tts_cache_artifact(output_path)
            self.events.put(
                (
                    "tts_error",
                    {
                        "source": str(source_path),
                        "error": clean_ui_status(str(exc), 300),
                        "reader_navigation_serial": reader_navigation_serial,
                        "start_page_key": start_page_key,
                    },
                )
            )
        finally:
            with self.tts_generation_lock:
                if self.tts_generation_process is process:
                    self.tts_generation_process = None

    def on_tts_progress(self, payload: dict) -> None:
        source = str(payload.get("source") or "")
        if (
            self.current_page_key != "reader"
            or not self.reader_current_path
            or str(self.reader_current_path.expanduser().resolve()) != source
        ):
            return
        done = int(payload.get("done") or 0)
        total = max(1, int(payload.get("total") or 1))
        self.tts_status_var.set(f"本地朗读：正在生成第 {done}/{total} 段")
        self.set_status(f"本机 Qwen3-TTS 进度 {done}/{total}")

    def on_tts_done(self, payload: dict) -> None:
        self.tts_running = False
        output_path = Path(str(payload.get("output") or "")).expanduser()
        source_path = Path(str(payload.get("source") or "")).expanduser()
        expected_hash = str(payload.get("source_hash") or "")
        try:
            current_hash = self.source_sha256(source_path)
        except OSError as exc:
            safe_unlink_tts_cache_artifact(output_path)
            self.on_tts_error(
                {
                    "source": str(source_path),
                    "error": f"朗读生成后无法复核原文：{exc}",
                    "reader_navigation_serial": payload.get("reader_navigation_serial"),
                    "start_page_key": payload.get("start_page_key"),
                }
            )
            return
        if not expected_hash or current_hash != expected_hash:
            safe_unlink_tts_cache_artifact(output_path)
            self.on_tts_error(
                {
                    "source": str(source_path),
                    "error": "朗读生成期间原文已变化，旧音频已丢弃，请重新点击。",
                    "reader_navigation_serial": payload.get("reader_navigation_serial"),
                    "start_page_key": payload.get("start_page_key"),
                }
            )
            return
        if not is_readable_audio_cache(output_path):
            safe_unlink_tts_cache_artifact(output_path)
            self.on_tts_error(
                {
                    "source": str(source_path),
                    "error": "本地 TTS 返回的音频不完整，未写入缓存。",
                    "reader_navigation_serial": payload.get("reader_navigation_serial"),
                    "start_page_key": payload.get("start_page_key"),
                }
            )
            return
        tts_text_sha256 = str(payload.get("text_sha256") or "")
        tts_scope = str(payload.get("scope") or "")
        self.attach_reader_tts_record(
            output_path,
            source_path,
            expected_hash,
            tts_text_sha256,
            tts_scope,
        )
        self.mark_results_dirty(refresh_visible=True)
        is_current = False
        if self.reader_current_path:
            try:
                is_current = self.reader_current_path.expanduser().resolve() == source_path.resolve()
            except OSError:
                is_current = self.reader_current_path == source_path
        is_current = (
            is_current
            and int(payload.get("reader_navigation_serial") or 0) == self.reader_navigation_serial
            and str(payload.get("start_page_key") or "") == self.current_page_key
        )
        if not is_current:
            self.set_status(f"TTS 音频已在后台生成：{output_path.name}")
            return

        self.reader_tts_audio_path = output_path
        self.reader_tts_source_path = source_path
        self.reader_tts_scope = tts_scope
        self.reader_tts_text_sha256 = tts_text_sha256
        seconds_text = f"{float(payload.get('seconds') or 0):.1f}s" if payload.get("seconds") else "已生成"
        device_text = str(payload.get("device") or "").strip()
        suffix = f" · {device_text}" if device_text else ""
        if payload.get("chunks"):
            suffix += f" · {int(payload.get('chunks') or 1)} 段"
        if payload.get("truncated"):
            suffix += " · 已截断"
        self.tts_status_var.set(f"本地朗读：{output_path.name} · {seconds_text}{suffix}")
        self.set_status(f"TTS 音频已生成：{output_path.name}")
        if payload.get("auto_play"):
            self.play_reader_tts_audio()

    def on_tts_error(self, payload: object) -> None:
        self.tts_running = False
        if isinstance(payload, dict):
            message = str(payload.get("error") or "未知错误")
            source = str(payload.get("source") or "")
            if self.reader_current_path and source:
                try:
                    is_current = str(self.reader_current_path.expanduser().resolve()) == source
                    if payload.get("reader_navigation_serial") is not None:
                        is_current = (
                            is_current
                            and int(payload.get("reader_navigation_serial") or 0) == self.reader_navigation_serial
                            and str(payload.get("start_page_key") or "") == self.current_page_key
                        )
                    if not is_current:
                        self.set_status(f"后台朗读生成失败：{message}")
                        return
                except OSError:
                    pass
        else:
            message = str(payload)
        self.tts_status_var.set(f"本地朗读失败：{message}")
        self.set_status(f"本地朗读失败：{message}")
        messagebox.showerror("本地朗读失败", message)

    def attach_reader_tts_record(
        self,
        output_path: Path,
        source_path: Path,
        source_hash: str = "",
        text_sha256: str = "",
        scope: str = "page",
    ) -> None:
        match = self.library_record_for_path(source_path)
        if not match:
            return
        _record_key, record = match
        try:
            source_key = str(source_path.expanduser().resolve())
            if not source_hash:
                source_hash = self.source_sha256(source_path)
        except OSError:
            return
        translation_raw = str(record.get("full_translation_path") or "").strip()
        is_translation = False
        if translation_raw:
            try:
                is_translation = Path(translation_raw).expanduser().resolve() == source_path.expanduser().resolve()
            except OSError:
                is_translation = False
        if is_translation and scope == "full_translation":
            previous_audio = str(record.get("full_translation_tts_audio_path") or "").strip()
            if previous_audio and Path(previous_audio).expanduser() != output_path:
                safe_unlink_tts_cache_artifact(Path(previous_audio))
            record["full_translation_tts_audio_path"] = str(output_path)
            record["full_translation_tts_source_path"] = source_key
            record["full_translation_tts_source_sha256"] = source_hash
            record["full_translation_tts_text_sha256"] = text_sha256
            record["full_translation_tts_scope"] = scope
            record["full_translation_tts_generated_at"] = datetime.now(timezone.utc).isoformat()
        else:
            audio_map = record.get("tts_audio_by_source")
            if not isinstance(audio_map, dict):
                audio_map = {}
                record["tts_audio_by_source"] = audio_map
            previous_entry = audio_map.get(source_key)
            if isinstance(previous_entry, dict):
                previous_audio = str(previous_entry.get("audio_path") or "").strip()
                if previous_audio and Path(previous_audio).expanduser() != output_path:
                    safe_unlink_tts_cache_artifact(Path(previous_audio))
            generated_at = datetime.now(timezone.utc).isoformat()
            audio_map[source_key] = {
                "audio_path": str(output_path),
                "source_sha256": source_hash,
                "text_sha256": text_sha256,
                "scope": scope,
                "generated_at": generated_at,
            }
            record["tts_audio_path"] = str(output_path)
            record["tts_source_path"] = source_key
            record["tts_source_sha256"] = source_hash
            record["tts_text_sha256"] = text_sha256
            record["tts_scope"] = scope
            record["tts_generated_at"] = generated_at
        self.save_research_library()

    def play_reader_tts_audio(self) -> None:
        if not self.reader_tts_audio_path or not self.reader_tts_audio_path.exists():
            messagebox.showinfo("还没有朗读音频", "先在阅读页生成一次朗读音频。")
            return
        if IS_MAC:
            self.stop_reader_tts_audio(update_status=False)
            self.tts_play_process = subprocess.Popen(["afplay", str(self.reader_tts_audio_path)])
            self.tts_status_var.set(f"本地朗读：正在播放 {self.reader_tts_audio_path.name}")
        else:
            self.open_path(self.reader_tts_audio_path)

    def stop_reader_tts_audio(self, update_status: bool = True) -> None:
        process = self.tts_play_process
        self.tts_play_process = None
        if process is not None and process.poll() is None:
            try:
                process.terminate()
                process.wait(timeout=1.5)
            except Exception:  # noqa: BLE001
                try:
                    process.kill()
                except Exception:
                    pass
        if update_status:
            if self.reader_tts_audio_path and self.reader_tts_audio_path.exists():
                self.tts_status_var.set(f"本地朗读：已停止 · {self.reader_tts_audio_path.name}")
            else:
                self.tts_status_var.set("本地朗读：已停止")

    def stop_tts_generation(self) -> None:
        with self.tts_generation_lock:
            process = self.tts_generation_process
            self.tts_generation_process = None
        if process is not None:
            terminate_process_tree(process)

    def open_tts_dir(self) -> None:
        ensure_dir(RESEARCH_TTS_DIR)
        self.open_path(RESEARCH_TTS_DIR)

    def open_selected_result(self, _event: tk.Event | None = None) -> None:
        record = self.selected_result_record()
        if not record:
            return
        if is_audio_result_path(Path(record["path"])):
            self.open_path(Path(record["path"]))
            return
        self.open_path(Path(record["path"]))

    def reveal_selected_result(self) -> None:
        record = self.selected_result_record()
        if not record:
            return
        self.reveal_path(Path(record["path"]))

    def copy_selected_result_path(self) -> None:
        record = self.selected_result_record()
        if not record:
            return
        path_text = record["path"]
        self.copy_text_to_clipboard(path_text, "结果路径")
        self.set_status(f"已复制结果路径：{path_text}")

    def copy_text_to_clipboard(self, text: str, label: str = "文本") -> None:
        self.root.clipboard_clear()
        self.root.clipboard_append(text)
        self.set_status(f"已复制{label}")

    def open_selected_transcription_output(self, _event: tk.Event | None = None) -> None:
        selection = self.transcription_tree.selection()
        if not selection:
            return
        values = self.transcription_tree.item(selection[0], "values")
        output = str(values[2] if len(values) >= 3 else "").strip()
        if not output:
            messagebox.showinfo("还没有输出", "这条任务还没有生成转录 TXT。")
            return
        self.open_path(Path(output))

    def open_download_dir(self) -> None:
        download_dir = Path(self.download_dir_var.get().strip() or DEFAULT_DOWNLOAD_DIR)
        ensure_dir(download_dir)
        self.open_path(download_dir)

    def open_digest_dir(self) -> None:
        ensure_dir(RESEARCH_DIGEST_DIR)
        self.open_path(RESEARCH_DIGEST_DIR)

    def open_path(self, path: Path) -> None:
        path = path.expanduser()
        if not path.exists():
            messagebox.showwarning("找不到文件", f"路径不存在：\n{path}")
            return
        try:
            if IS_MAC:
                subprocess.run(["open", str(path)], check=False)
            elif platform.system() == "Windows":
                os.startfile(path)  # type: ignore[attr-defined]
            else:
                subprocess.run(["xdg-open", str(path)], check=False)
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("打开失败", str(exc))

    def reveal_path(self, path: Path) -> None:
        path = path.expanduser()
        if not path.exists():
            messagebox.showwarning("找不到文件", f"路径不存在：\n{path}")
            return
        try:
            if IS_MAC:
                subprocess.run(["open", "-R", str(path)], check=False)
            else:
                self.open_path(path.parent)
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("打开失败", str(exc))

    def build_transcription_settings(self) -> TranscriptionSettings:
        try:
            settings = TranscriptionSettings(
                api_key=self.api_key_var.get().strip(),
                target_segment_seconds=int(self.target_segment_seconds_var.get().strip()),
                max_segment_seconds=int(self.max_segment_seconds_var.get().strip()),
                workers=int(self.transcription_workers_var.get().strip()),
                strategy=TRANSCRIPTION_MODE_OPTIONS.get(
                    self.transcription_mode_var.get(),
                    "local_first",
                ),
            )
        except ValueError as exc:
            raise TranscriptionError("切分时长和并发数必须填写整数。") from exc
        settings.validate()
        return settings

    def save_transcription_settings(self) -> None:
        try:
            self.build_transcription_settings()
            save_api_key(DOTENV_PATH, self.api_key_var.get())
            self.save_settings()
        except (OSError, TranscriptionError) as exc:
            messagebox.showerror("保存失败", str(exc))
            return
        messagebox.showinfo("设置已保存", "转录设置已保存到本机。")

    def save_download_settings(self) -> None:
        self.save_settings()
        self.set_status("下载目录、音频格式和浏览器登录态已保存")

    def choose_download_dir(self) -> None:
        chosen = filedialog.askdirectory(initialdir=self.download_dir_var.get().strip() or str(APP_DIR))
        if chosen:
            self.download_dir_var.set(chosen)
            self.save_settings()

    def choose_cookie_file(self) -> None:
        chosen = filedialog.askopenfilename(
            title="选择 cookies.txt",
            filetypes=[("Cookie 文件", "*.txt"), ("所有文件", "*.*")],
            initialdir=str(APP_DIR),
        )
        if chosen:
            self.cookie_file_var.set(chosen)
            self.save_settings()

    def choose_transcription_files(self) -> None:
        chosen = filedialog.askopenfilenames(
            title="选择需要转录的音频或视频文件",
            filetypes=[
                ("支持的文件", "*.mp3 *.m4a *.wav *.mp4"),
                ("所有文件", "*.*"),
            ],
        )
        if chosen:
            self.add_transcription_files([Path(path) for path in chosen])

    def add_transcription_files(self, paths: list[Path], force_pending: bool = False) -> int:
        added = 0
        for raw_path in paths:
            path = raw_path.resolve()
            if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
                self.append_transcription_log(f"已跳过不支持的格式：{path.name}")
                continue
            if path in self.transcription_rows:
                if not force_pending:
                    self.append_transcription_log(f"文件已在转录列表中：{path.name}")
                    continue
                row_id = self.transcription_rows[path]
                output_path = path.with_suffix(".txt")
                row_status = "待处理"
                self.transcription_tree.item(
                    row_id,
                    values=(path.name, row_status, str(output_path) if output_path.exists() else ""),
                )
                self.transcription_statuses[path] = row_status
                added += 1
                continue

            output_path = path.with_suffix(".txt")
            status = "文本已存在，已跳过" if output_path.exists() else "待处理"
            row_id = self.transcription_tree.insert("", "end", values=(path.name, status, str(output_path) if output_path.exists() else ""))
            self.transcription_paths.append(path)
            self.transcription_rows[path] = row_id
            self.transcription_statuses[path] = status
            added += 1
        if added:
            self.append_transcription_log(f"已加入 {added} 个文件。")
        return added

    def start_transcription_queue(self, auto_mode: bool = False) -> None:
        if self.transcribing:
            messagebox.showinfo("提示", "当前已有转录任务在进行中。")
            return
        if not auto_mode:
            self.auto_transcribe_path_map = {}
        pending_paths = [
            path
            for path in self.transcription_paths
            if self.transcription_statuses.get(path) in {"待处理", "失败"}
        ]
        if not pending_paths:
            messagebox.showinfo("提示", "当前没有需要转录的文件。")
            return
        try:
            settings = self.build_transcription_settings()
            ensure_runtime_available()
        except TranscriptionError as exc:
            messagebox.showerror("无法开始转录", str(exc))
            self.append_transcription_log(f"无法开始转录：{exc}")
            return

        self.transcription_stop_requested = False
        self.transcribing = True
        self.append_transcription_log(f"转录队列已启动，共 {len(pending_paths)} 个文件。")
        threading.Thread(
            target=self.transcription_worker,
            args=(pending_paths, settings),
            daemon=True,
        ).start()

    def transcription_worker(self, paths: list[Path], settings: TranscriptionSettings) -> None:
        completed = 0
        skipped = 0
        failed = 0
        auto_digest_jobs: list[tuple[Path, str]] = []
        for index, path in enumerate(paths, start=1):
            if self.transcription_stop_requested:
                break
            self.events.put(("transcription_row", (path, "处理中", "")))
            self.events.put(("transcription_log", f"[{index}/{len(paths)}] 开始处理：{path.name}"))
            try:
                result = transcribe_file(
                    path,
                    settings,
                    lambda message, current=path.name: self.events.put(("transcription_log", f"[{current}] {message}")),
                )
                if result.skipped:
                    skipped += 1
                    status = "文本已存在，已跳过"
                else:
                    completed += 1
                    status = "已完成"
                video_id = self.auto_transcribe_path_map.get(path)
                if video_id:
                    item = self._item_for_auto_digest(video_id)
                    if item:
                        self.events.put(
                            (
                                "research_record",
                                {
                                    **self.video_item_record(item),
                                    "audio_cache_path": str(path),
                                    "transcript_path": str(result.output_path),
                                    "transcript_kind": "transcript",
                                    "transcript_origin": "audio_asr",
                                    "transcript_engine": result.backend,
                                    "transcript_identity_verified": True,
                                },
                            )
                        )
                    if settings.api_key.strip():
                        auto_digest_jobs.append((result.output_path, video_id))
                    else:
                        self.events.put(
                            (
                                "transcription_log",
                                f"[{path.name}] 转录已完成；未配置 API Key，跳过中文纪要。",
                            )
                        )
                self.events.put(("transcription_row", (path, status, str(result.output_path))))
            except Exception as exc:  # noqa: BLE001
                failed += 1
                self.events.put(("transcription_row", (path, "失败", "")))
                self.events.put(("transcription_log", f"[{path.name}] 转录失败：{exc}"))

        self.events.put(
            (
                "transcription_done",
                {
                    "completed": completed,
                    "skipped": skipped,
                    "failed": failed,
                    "stopped": self.transcription_stop_requested,
                    "auto_digest_jobs": [{"transcript_path": str(path), "video_id": video_id} for path, video_id in auto_digest_jobs],
                    "auto_api_key": settings.api_key,
                },
            )
        )

    def request_transcription_stop(self) -> None:
        if not self.transcribing:
            self.append_transcription_log("当前没有正在执行的转录队列。")
            return
        self.transcription_stop_requested = True
        self.append_transcription_log("已请求停止，当前文件结束后会停下。")

    def clear_finished_transcriptions(self) -> None:
        removable_statuses = {"已完成", "文本已存在，已跳过"}
        remaining_paths: list[Path] = []
        for path in self.transcription_paths:
            if self.transcription_statuses.get(path) in removable_statuses:
                self.transcription_tree.delete(self.transcription_rows.pop(path))
                self.transcription_statuses.pop(path, None)
            else:
                remaining_paths.append(path)
        self.transcription_paths = remaining_paths
        self.append_transcription_log("已清空完成和跳过的任务。")

    def update_transcription_row(self, path: Path, status: str, output: str) -> None:
        self.transcription_statuses[path] = status
        if output and Path(output).exists():
            self.mark_results_dirty(refresh_visible=True)
            if status in {"已完成", "文本已存在，已跳过"}:
                self.set_status(f"转录结果已生成：{Path(output).name}，可在左侧「资料库」查看。")
        if self.current_page_key != "transcription":
            self.pending_transcription_visuals[path] = (status, output)
            return
        self.apply_transcription_row_visual(path, status, output)

    def apply_transcription_row_visual(self, path: Path, status: str, output: str) -> None:
        row_id = self.transcription_rows.get(path)
        if not row_id:
            return
        current_values = list(self.transcription_tree.item(row_id, "values"))
        current_values[1] = status
        if output:
            current_values[2] = output
        self.transcription_tree.item(row_id, values=current_values)

    def append_transcription_log(self, message: str) -> None:
        timestamp = time.strftime("%H:%M:%S")
        line = f"[{timestamp}] {message}\n"
        if self.current_page_key != "transcription":
            self.pending_transcription_logs.append(line)
            if len(self.pending_transcription_logs) > 2000:
                del self.pending_transcription_logs[:-2000]
            return
        self.insert_transcription_logs([line])

    def insert_transcription_logs(self, lines: list[str]) -> None:
        if not lines:
            return
        self.transcription_log_text.configure(state="normal")
        self.transcription_log_text.insert("end", "".join(lines))
        self.transcription_log_text.see("end")
        self.transcription_log_text.configure(state="disabled")

    def on_transcription_finished(self, payload: dict) -> None:
        self.transcribing = False
        message = (
            f"转录队列结束：成功 {payload['completed']} 个，"
            f"跳过 {payload['skipped']} 个，失败 {payload['failed']} 个。"
        )
        if payload["stopped"]:
            message = f"转录队列已停止。{message}"
        elif payload["completed"] or payload["skipped"]:
            message += " 可在左侧「资料库」查看 TXT。"
        self.append_transcription_log(message)
        self.mark_results_dirty(refresh_visible=True)
        if self.restart_transcription_when_idle:
            self.restart_transcription_when_idle = False
            self.start_transcription_queue()

    def _normalize_auto_transcribe_payload(
        self,
        payload: list[Path] | list[tuple[Path, str]] | list[object],
    ) -> list[tuple[Path, str]]:
        normalized: list[tuple[Path, str]] = []
        for item in payload:
            raw_path = None
            raw_video_id = ""

            if isinstance(item, Path):
                raw_path = item
            elif isinstance(item, (list, tuple)):
                if len(item) == 2:
                    candidate_path, candidate_video_id = item
                    if isinstance(candidate_path, (list, tuple)) and len(candidate_path) == 2:
                        candidate_path, candidate_video_id = candidate_path
                    if isinstance(candidate_path, (Path, str, os.PathLike)):
                        raw_path = candidate_path
                        raw_video_id = candidate_video_id
                    else:
                        continue
                else:
                    continue
            elif isinstance(item, str):
                raw_path = item
            else:
                continue

            if raw_path is None:
                self.append_transcription_log(f"跳过无法识别的自动转录项：{item}")
                continue
            path = Path(raw_path)
            video_id = str(raw_video_id).strip()
            normalized.append((path, video_id))
        return normalized

    def handle_auto_transcribe(self, paths: list[Path] | list[tuple[Path, str]]) -> None:
        auto_items = self._normalize_auto_transcribe_payload(paths)
        if not auto_items:
            return
        self.append_transcription_log(f"下载队列已交付 {len(auto_items)} 个音频文件。")
        normalized_map: dict[Path, str] = {}
        for path, video_id in auto_items:
            if not video_id:
                continue
            try:
                resolved = path.resolve()
            except Exception as exc:
                self.append_transcription_log(f"跳过无效转录路径：{path}，原因：{exc}")
                continue
            normalized_map[resolved] = video_id
        self.auto_transcribe_path_map = normalized_map
        self.add_transcription_files([path for path, _video_id in auto_items], force_pending=True)
        pending_exists = any(
            self.transcription_statuses.get(path) in {"待处理", "失败"}
            for path in self.transcription_paths
            if path in self.auto_transcribe_path_map
        )
        if not pending_exists:
            return
        if self.transcribing:
            self.restart_transcription_when_idle = True
            return
        self.start_transcription_queue(auto_mode=True)

    def _parse_auto_digest_jobs(self, jobs_raw: object) -> list[tuple[Path, str]]:
        parsed: list[tuple[Path, str]] = []
        seen: set[tuple[str, str]] = set()

        if not isinstance(jobs_raw, list):
            return parsed

        for entry in jobs_raw:
            transcript_path: Path | None = None
            video_id: str = ""

            if isinstance(entry, dict):
                raw_path = entry.get("transcript_path")
                if isinstance(raw_path, Path):
                    transcript_path = raw_path
                elif isinstance(raw_path, str):
                    transcript_path = Path(raw_path)
                video_id = str(entry.get("video_id") or "").strip()
            elif isinstance(entry, (list, tuple)) and len(entry) == 2:
                raw_path, raw_video_id = entry
                if isinstance(raw_path, Path):
                    transcript_path = raw_path
                elif isinstance(raw_path, str):
                    transcript_path = Path(raw_path)
                video_id = str(raw_video_id).strip()

            if not transcript_path or not video_id:
                continue

            key = (str(transcript_path), video_id)
            if key in seen:
                continue
            seen.add(key)
            parsed.append((transcript_path, video_id))

        return parsed

    def _audio_path_from_transcript(self, transcript_path: Path) -> Path:
        transcript_text = str(transcript_path)
        if transcript_text.lower().endswith(".txt"):
            return Path(transcript_text[:-4])
        return transcript_path

    def _item_for_auto_digest(self, video_id: str) -> VideoItem | None:
        item = self.video_map.get(video_id)
        if item:
            return item

        record = self.research_library.get("items", {}).get(video_id)
        if not isinstance(record, dict):
            return None

        title = str(record.get("title") or "").strip()
        if not title:
            return None

        item = VideoItem(
            video_id=video_id,
            title=title,
            duration_text=str(record.get("duration_text") or "-"),
            webpage_url=str(record.get("webpage_url") or ""),
            source_name=str(record.get("source_name") or ""),
            published_text=str(record.get("published_text") or ""),
            play_count=parse_count(record.get("play_count")),
            importance_score=float(record.get("importance_score") or 0.0),
            investment_score=float(record.get("investment_score") or 0.0),
            importance_reason=str(record.get("importance_reason") or ""),
            scored_at=str(record.get("importance_scored_at") or ""),
        )
        if not item.scored_at:
            update_item_importance_score(item)
        return item

    def start_auto_digest_jobs(self, payload: object) -> None:
        if self.auto_digests_running:
            self.set_status("自动中文整理已在进行中，已忽略本次新增任务。")
            return

        if isinstance(payload, dict):
            raw_jobs = payload.get("jobs")
            api_key = str(payload.get("api_key") or self.api_key_var.get()).strip()
        else:
            raw_jobs = payload
            api_key = self.api_key_var.get().strip()

        jobs = self._parse_auto_digest_jobs(raw_jobs)
        if not jobs:
            return

        self.auto_digests_running = True
        self.current_task_var.set(f"当前任务：自动中文整理 {len(jobs)} 条")
        self.events.put(("task_status", f"当前任务：自动中文整理 {len(jobs)} 条"))
        first_video_id = jobs[0][1] if jobs else ""
        self.events.put(
            (
                "progress",
                {
                    "video_id": first_video_id,
                    "percent": 0.0,
                    "detail": f"准备自动中文整理 {len(jobs)} 条",
                    "row_progress": "准备",
                    "row_status": "待整理",
                },
            )
        )
        threading.Thread(target=self.auto_digest_worker, args=(jobs, api_key), daemon=True).start()

    def auto_digest_worker(self, jobs: list[tuple[Path, str]], api_key: str) -> None:
        completed = 0
        skipped = 0
        failed = 0
        total = len(jobs)
        failures: list[dict[str, str]] = []
        processed_item_ids: list[str] = []

        for index, (transcript_path, video_id) in enumerate(jobs, start=1):
            if self.stop_requested:
                break
            if video_id and video_id not in processed_item_ids:
                processed_item_ids.append(video_id)

            item = self._item_for_auto_digest(video_id)
            self.events.put(("row_status", (video_id, "中文整理中", "启动")))

            if not transcript_path.exists():
                failed += 1
                self.events.put(("log", f"[{video_id}] 转录文件不存在，无法自动整理：{transcript_path}"))
                self.events.put(("row_status", (video_id, "整理失败", "转录文件不存在")))
                failures.append({"title": str(item.title) if item else video_id, "error": "转录文件不存在"})
                continue

            transcript_text = transcript_path.read_text(encoding="utf-8", errors="ignore").strip()
            if not transcript_text:
                failed += 1
                self.events.put(("log", f"[{video_id}] 转录文本为空，无法自动整理：{transcript_path}"))
                self.events.put(("row_status", (video_id, "整理失败", "转录文本为空")))
                failures.append({"title": str(item.title) if item else video_id, "error": "转录文本为空"})
                continue

            if not item:
                failed += 1
                self.events.put(("log", f"[{video_id}] 未找到节目元数据，跳过自动整理。"))
                self.events.put(("row_status", (video_id, "整理失败", "未找到节目元数据")))
                failures.append({"title": video_id, "error": "未找到节目元数据"})
                continue

            audio_path = self._audio_path_from_transcript(transcript_path)
            digest_path = self.research_digest_path(item)
            try:
                self.events.put(("row_status", (video_id, "中文整理中", "提取要点")))

                if digest_path.exists() and digest_path.stat().st_mtime >= transcript_path.stat().st_mtime:
                    skipped += 1
                    self.events.put(("log", f"《{item.title}》中文整理已存在，已复用：{digest_path.name}"))
                    progress = (index / max(total, 1)) * 100
                    self.events.put(
                        (
                            "progress",
                            {
                                "video_id": video_id,
                                "percent": progress,
                                "detail": f"已复用 {index}/{total} 个",
                                "row_progress": "已复用",
                                "row_status": "已整理",
                            },
                        )
                    )
                else:
                    build_chinese_research_digest(
                        transcript_path=transcript_path,
                        output_path=digest_path,
                        metadata=DigestMetadata(
                            title=item.title,
                            source_name=item.source_name,
                            published_text=item.published_text,
                            duration_text=item.duration_text,
                            webpage_url=item.webpage_url,
                            topic_hint=self.current_topic,
                        ),
                        api_key=api_key,
                    )
                    completed += 1
                    self.events.put(("log", f"《{item.title}》中文整理完成：{digest_path.name}"))

                    progress = (index / max(total, 1)) * 100
                    self.events.put(
                        (
                            "progress",
                            {
                                "video_id": video_id,
                                "percent": progress,
                                "detail": f"已整理 {index}/{total} 个",
                                "row_progress": "中文纪要",
                                "row_status": "已整理",
                            },
                        )
                    )

                self.events.put(
                    (
                        "research_record",
                        {
                            **self.video_item_record(item),
                            "audio_cache_path": str(audio_path),
                            "transcript_path": str(transcript_path),
                            "digest_path": str(digest_path),
                            "digested_at": datetime.now(timezone.utc).isoformat(),
                        },
                    )
                )
                self.events.put(("row_status", (video_id, "已整理", "中文纪要")))
            except Exception as exc:  # noqa: BLE001
                failed += 1
                clean_error = clean_ui_status(str(exc), 240)
                failures.append({"title": item.title, "error": clean_error})
                self.events.put(("log", f"《{item.title}》中文整理失败：{clean_error}"))
                self.events.put(("row_status", (video_id, "整理失败", "查看底部提示")))

        self.events.put(
            (
                "auto_digest_done",
                {
                    "completed": completed,
                    "skipped": skipped,
                    "failed": failed,
                    "item_ids": processed_item_ids,
                    "stopped": self.stop_requested,
                    "total": total,
                    "failures": failures,
                },
            )
        )

    def on_auto_digest_done(self, payload: dict) -> None:
        self.auto_digests_running = False
        if hasattr(self, "progress_bar"):
            self.progress_bar.stop()
        self.current_task_var.set("当前任务：自动中文整理完成")
        message = (
            f"自动中文整理结束：新整理 {payload['completed']} 个，"
            f"复用 {payload['skipped']} 个，失败 {payload['failed']} 个。"
        )
        if payload.get("stopped"):
            message = f"自动中文整理已停止。{message}"

        if payload.get("completed") or payload.get("skipped"):
            self.set_status(message + " 可在左侧「资料库」查看中文纪要。")
        else:
            self.set_status(message)
        self.progress_detail_var.set(message)
        self.progress_value.set(100.0 if payload.get("total") else 0.0)
        self.progress_percent_var.set("100%" if payload.get("total") else "0%")
        self.mark_results_dirty(refresh_visible=True)
        failures = payload.get("failures") or []
        if failures:
            first_failure = failures[0]
            self.set_status(f"中文整理失败：{first_failure.get('title', '')[:16]} - {first_failure.get('error', '未知错误')}" )
        video_ids = payload.get("item_ids")
        if isinstance(video_ids, (list, tuple, set)) and video_ids:
            self._start_research_item_translations(video_ids)
        else:
            self._start_research_item_translations(self.pending_research_item_ids)

    def set_status(self, text: str) -> None:
        self.status_var.set(text)

    def process_events(self) -> None:
        blocked_for = self.event_processing_blocked_until - time.monotonic()
        if blocked_for > 0:
            self.root.after(max(1, int(blocked_for * 1000)), self.process_events)
            return
        processed = 0
        started_at = time.perf_counter()
        max_events_per_tick = 12
        time_budget_seconds = 0.008
        while processed < max_events_per_tick:
            try:
                event = self.events.get_nowait()
            except queue.Empty:
                break
            processed += 1

            event_type = event[0]
            payload = event[1] if len(event) > 1 else None

            if event_type == "fetch_success":
                self.on_fetch_success(payload)
            elif event_type == "fetch_error":
                self.fetching = False
                self.finish_feed_refresh(success=False)
                self.current_task_var.set("当前任务：刷新失败")
                self.set_status(payload)
                messagebox.showerror("读取失败", payload)
            elif event_type == "row_status":
                video_id, status, progress = payload
                self.update_row(video_id, status=status, progress=progress)
            elif event_type == "task_status":
                self.current_task_var.set(payload)
            elif event_type == "progress":
                self.on_progress(payload)
            elif event_type == "download_done":
                self.on_download_finished(payload)
            elif event_type == "queue_stopped":
                self.downloading = False
                if hasattr(self, "progress_bar"):
                    self.progress_bar.stop()
                self.current_task_var.set("当前任务：下载已停止")
                self.set_status("下载队列已停止")
            elif event_type == "log":
                self.set_status(payload)
            elif event_type == "transcription_row":
                path, status, output = payload
                self.update_transcription_row(path, status, output)
            elif event_type == "transcription_log":
                self.append_transcription_log(payload)
            elif event_type == "transcription_done":
                self.on_transcription_finished(payload)
                auto_jobs = payload.get("auto_digest_jobs") if isinstance(payload, dict) else None
                if auto_jobs:
                    self.events.put(
                        (
                            "auto_digest_jobs",
                            {
                                "jobs": auto_jobs,
                                "api_key": str(payload.get("auto_api_key") or self.api_key_var.get()).strip(),
                            },
                        )
                    )
            elif event_type == "results_refresh_done":
                self.on_results_refresh_done(payload)
            elif event_type == "results_refresh_error":
                self.on_results_refresh_error(payload)
            elif event_type == "artwork_ready":
                self.apply_artwork_to_card(payload)
            elif event_type == "auto_transcribe":
                self.handle_auto_transcribe(payload)
            elif event_type == "auto_digest_jobs":
                self.start_auto_digest_jobs(payload)
            elif event_type == "auto_digest_done":
                self.on_auto_digest_done(payload)
            elif event_type == "research_record":
                self.update_research_record(payload)
            elif event_type == "importance_record":
                self.update_research_record(payload)
            elif event_type == "research_done":
                self.on_research_finished(payload)
            elif event_type == "detail_translation_ready":
                self.apply_episode_detail_translation(payload)
            elif event_type == "detail_translation_error":
                self.show_episode_detail_error(payload)
            elif event_type == "card_titles_ready":
                self.apply_card_title_translations(payload)
            elif event_type == "card_titles_error":
                self.set_status(payload)
            elif event_type == "card_summaries_ready":
                self.apply_card_summary_translations(payload)
            elif event_type == "card_summaries_error":
                self.set_status(payload)
            elif event_type == "full_translation_progress":
                self.on_full_translation_progress(payload)
            elif event_type == "full_translation_ready":
                self.on_full_translation_ready(payload)
            elif event_type == "full_translation_error":
                self.on_full_translation_error(payload)
            elif event_type == "tts_progress":
                self.on_tts_progress(payload)
            elif event_type == "tts_done":
                self.on_tts_done(payload)
            elif event_type == "tts_error":
                self.on_tts_error(payload)

            if time.perf_counter() - started_at >= time_budget_seconds:
                break

        next_delay = 16 if not self.events.empty() else 120
        self.root.after(next_delay, self.process_events)

    def fetch_videos(self) -> None:
        if self.fetching:
            self.set_status(
                f"正在刷新{topic_display_label(self.current_topic)}，完成后再读取新链接。"
            )
            return

        url = self.url_var.get().strip()
        if not url:
            self.fetch_all_updates()
            return

        self.fetching = True
        self.current_topic = "manual"
        self.download_title_var.set(topic_download_title(self.current_topic))
        self.update_topic_filter_buttons()
        self.set_active_nav("download")
        self.summary_var.set("正在读取，请稍候...")
        self.begin_feed_refresh("manual", showing_cache=False)
        threading.Thread(target=self.fetch_worker, args=(url,), daemon=True).start()

    def fetch_current_source(self) -> None:
        if self.url_var.get().strip():
            self.fetch_videos()
            return
        self.fetch_all_updates()

    def fetch_ai_updates(self) -> None:
        self.fetch_topic_updates("ai")

    def fetch_ai_startup_updates(self) -> None:
        self.fetch_topic_updates("ai_startup")

    def fetch_weather_agri_updates(self) -> None:
        self.fetch_topic_updates("weather_agri")

    def fetch_science_wellness_updates(self) -> None:
        self.fetch_topic_updates("science_wellness")

    def fetch_all_updates(self) -> None:
        self.fetch_topic_updates("all")

    def fetch_topic_updates(self, topic: str) -> None:
        if self.fetching:
            self.set_status(
                f"正在刷新{topic_display_label(self.current_topic)}，完成后再切换栏目。"
            )
            return
        label = PODCAST_GROUP_LABELS.get(topic, "播客")
        self.url_var.set("")
        self.current_topic = topic
        self.download_title_var.set(topic_download_title(topic))
        self.update_topic_filter_buttons()
        self.set_active_nav("download")
        cached_items = self.cached_podcast_items(topic)
        if cached_items and not self.video_items:
            self.apply_fetch_payload(
                {
                    "source_title": f"{label}本地快照",
                    "items": cached_items,
                    "skipped": 0,
                    "subtitle_hint": f"已先显示本地快照 {len(cached_items)} 条，正在后台刷新最新节目...",
                    "cache_preview": True,
                },
                start_translations=False,
                persist_records=False,
            )
            self.current_task_var.set("当前任务：后台刷新最新节目")
        elif cached_items:
            self.set_status(f"已显示本地快照，正在后台刷新{label}最新节目...")
            self.current_task_var.set("当前任务：后台刷新最新节目")
        else:
            self.summary_var.set(f"正在搜索{label}更新...")
        self.fetching = True
        self.begin_feed_refresh(topic, showing_cache=bool(cached_items))
        threading.Thread(target=self.fetch_topic_worker, args=(topic,), daemon=True).start()

    def fetch_worker(self, url: str) -> None:
        try:
            source_title, items, skipped, subtitle_hint = fetch_video_items(url)
            self.events.put(
                (
                    "fetch_success",
                    {
                        "source_title": source_title,
                        "items": items,
                        "skipped": skipped,
                        "subtitle_hint": subtitle_hint,
                    },
                )
            )
        except Exception as exc:
            self.events.put(("fetch_error", f"读取节目列表失败：{exc}"))

    def fetch_topic_worker(self, topic: str) -> None:
        try:
            source_title, items, skipped, subtitle_hint = fetch_podcast_updates(topic)
            self.events.put(
                (
                    "fetch_success",
                    {
                        "source_title": source_title,
                        "items": items,
                        "skipped": skipped,
                        "subtitle_hint": subtitle_hint,
                    },
                )
            )
        except Exception as exc:
            cached_items = self.cached_podcast_items(topic)
            if cached_items:
                label = PODCAST_GROUP_LABELS.get(topic, "播客")
                self.events.put(
                    (
                        "fetch_success",
                        {
                            "source_title": f"{label}本地缓存",
                            "items": cached_items,
                            "skipped": 0,
                            "subtitle_hint": f"实时搜索暂时失败，已显示本地缓存 {len(cached_items)} 条；错误：{clean_ui_status(str(exc), 100)}",
                            "cache_preview": True,
                        },
                    )
                )
                return
            self.events.put(("fetch_error", f"搜索播客失败：{exc}"))

    def cached_podcast_items(self, topic: str) -> list[VideoItem]:
        records = self.research_library.get("items", {})
        if not isinstance(records, dict):
            return []

        rows: list[tuple[datetime, VideoItem]] = []
        for item_id, record in records.items():
            if not isinstance(record, dict):
                continue
            if not self.cached_record_matches_topic(record, topic):
                continue
            title = str(record.get("title") or "").strip()
            audio_url = str(record.get("audio_url") or "").strip()
            webpage_url = str(record.get("webpage_url") or "").strip()
            if not title or not (audio_url or webpage_url):
                continue
            item = VideoItem(
                video_id=str(record.get("video_id") or item_id),
                title=title,
                duration_text=str(record.get("duration_text") or "-"),
                webpage_url=webpage_url,
                source_type=str(record.get("source_type") or "podcast_rss"),
                audio_url=audio_url,
                source_name=str(record.get("source_name") or ""),
                published_text=str(record.get("published_text") or ""),
                artwork_url=str(record.get("artwork_url") or ""),
                description_text=str(record.get("description_text") or ""),
                play_count=parse_count(record.get("play_count")),
                importance_score=float(record.get("importance_score") or 0.0),
                investment_score=float(record.get("investment_score") or 0.0),
                importance_reason=str(record.get("importance_reason") or ""),
                scored_at=str(record.get("importance_scored_at") or record.get("scored_at") or ""),
                available=bool(audio_url or webpage_url),
                status="快照",
            )
            rows.append((self.cached_record_sort_datetime(record), item))

        rows.sort(key=lambda row: row[0], reverse=True)
        return [item for _timestamp, item in rows[:AI_DISCOVERY_LIMIT]]

    def cached_record_matches_topic(self, record: dict, topic: str) -> bool:
        if topic == "all":
            return True
        if topic == "favorites":
            return bool(record.get("favorite"))
        record_topic = str(record.get("topic") or "").strip()
        if record_topic == topic:
            return True
        source_name = str(record.get("source_name") or "")
        if topic == "ai":
            return source_name in AI_FEED_NAMES
        if topic == "ai_startup":
            return source_name in AI_STARTUP_FEED_NAMES
        if topic == "weather_agri":
            return source_name in WEATHER_AGRI_FEED_NAMES
        if topic == "science_wellness":
            return source_name in SCIENCE_WELLNESS_FEED_NAMES
        return False

    def cached_record_sort_datetime(self, record: dict) -> datetime:
        for key in ("published_text", "updated_at", "digested_at", "downloaded_at"):
            parsed = parse_podcast_datetime(str(record.get(key) or ""))
            if parsed:
                return parsed
        return datetime.fromtimestamp(0, tz=timezone.utc)

    def apply_fetch_payload(
        self,
        payload: dict,
        *,
        start_translations: bool = True,
        persist_records: bool = True,
    ) -> None:
        self.current_source_label = payload["source_title"]
        self.video_items = payload["items"]
        for item in self.video_items:
            update_item_importance_score(item)
        self.video_map = {item.video_id: item for item in self.video_items}
        if persist_records:
            self.persist_visible_item_records()
        self.selected_item_ids = set()
        self.focused_item_id = ""
        if self.current_page_key == "download":
            self.render_cards()
        else:
            self.cancel_cards_render()
            self.download_render_dirty = True

        skipped = payload["skipped"]
        source_label = "快照" if payload.get("cache_preview") else "来源"
        self.summary_var.set(f"{source_label}：{self.current_source_label} | 可用节目 {len(self.video_items)} 个 | 跳过不可用 {skipped} 个")
        self.current_task_var.set("当前任务：已读取节目列表" if not payload.get("cache_preview") else "当前任务：后台刷新中")
        self.progress_value.set(0)
        self.progress_percent_var.set("0%")
        self.progress_detail_var.set("等待开始下载")
        subtitle_hint = payload.get("subtitle_hint")
        self.set_status(subtitle_hint or "节目列表读取完成")
        self.update_selected_count()
        if start_translations:
            self.ensure_card_title_translations()

    def on_fetch_success(self, payload: dict) -> None:
        self.fetching = False
        is_cache_preview = bool(payload.get("cache_preview"))
        self.apply_fetch_payload(
            payload,
            start_translations=not is_cache_preview,
            persist_records=not is_cache_preview,
        )
        if is_cache_preview:
            self.current_task_var.set("当前任务：实时刷新失败，已显示本地缓存")
            self.finish_feed_refresh(success=False, used_cache=True)
        else:
            self.finish_feed_refresh(success=True)

    def update_row(self, video_id: str, status: str | None = None, progress: str | None = None) -> None:
        item = self.video_map.get(video_id)
        if not item:
            return
        if status is not None:
            item.status = clean_ui_status(status)
        if progress is not None:
            item.progress = clean_ui_status(progress)
        if "失败" in item.status and item.progress not in {"-", "查看底部提示"}:
            item.progress = "查看底部提示"
        self.refresh_card_visual_or_defer(video_id)

    def update_selected_count(self) -> None:
        selected = len(self.selected_item_ids)
        total = len(self.video_items)
        self.selected_count_var.set(f"已选 {selected} / {total}")

    def select_all_rows(self) -> None:
        self.selected_item_ids = {item.video_id for item in self.video_items}
        for item_id in self.video_map:
            self.refresh_card_visual(item_id)
        self.update_selected_count()

    def clear_selection(self) -> None:
        self.selected_item_ids = set()
        for item_id in self.video_map:
            self.refresh_card_visual(item_id)
        self.update_selected_count()

    def select_available_rows(self) -> None:
        self.selected_item_ids = {item.video_id for item in self.video_items if item.available}
        for item_id in self.video_map:
            self.refresh_card_visual(item_id)
        self.update_selected_count()

    def get_selected_items(self) -> list[VideoItem]:
        return [item for item in self.video_items if item.video_id in self.selected_item_ids]

    def favorite_selected(self) -> None:
        items = self.get_selected_items()
        if not items:
            messagebox.showwarning("提示", "请先选择至少一个节目。")
            return
        for item in items:
            self.set_item_favorite(item, True)
        self.set_status(f"已标记精选 {len(items)} 个节目")

    def unfavorite_selected(self) -> None:
        items = self.get_selected_items()
        if not items:
            messagebox.showwarning("提示", "请先选择至少一个节目。")
            return
        for item in items:
            self.set_item_favorite(item, False)
        self.set_status(f"已取消精选 {len(items)} 个节目")

    def download_favorite_visible(self) -> None:
        items = [item for item in self.video_items if item.video_id in self.favorite_item_ids]
        if not items:
            messagebox.showwarning("提示", "当前列表里还没有精选节目。")
            return
        self.start_download_queue(items)

    def set_item_favorite(self, item: VideoItem, favorite: bool) -> None:
        if favorite:
            self.favorite_item_ids.add(item.video_id)
        else:
            self.favorite_item_ids.discard(item.video_id)
        record = self.video_item_record(item)
        previous = self.research_library.setdefault("items", {}).get(item.video_id, {})
        if isinstance(previous, dict):
            previous.update(record)
            record = previous
        record["favorite"] = favorite
        record["favorite_updated_at"] = datetime.now(timezone.utc).isoformat()
        self.research_library.setdefault("items", {})[item.video_id] = record
        self.save_research_library()
        if self.current_topic == "favorites" and not favorite:
            self.video_items = [visible for visible in self.video_items if visible.video_id != item.video_id]
            self.video_map = {visible.video_id: visible for visible in self.video_items}
            self.selected_item_ids.discard(item.video_id)
            if self.focused_item_id == item.video_id:
                self.focused_item_id = ""
            self.update_selected_count()
            self.render_cards()
            self.summary_var.set(f"来源：跨日期精选收藏 | 可用节目 {len(self.video_items)} 个")
            return
        self.refresh_card_visual(item.video_id)

    def video_item_record(self, item: VideoItem) -> dict:
        return {
            "video_id": item.video_id,
            "title": item.title,
            "source_name": item.source_name,
            "source_type": item.source_type,
            "published_text": item.published_text,
            "duration_text": item.duration_text,
            "webpage_url": item.webpage_url,
            "audio_url": item.audio_url,
            "artwork_url": item.artwork_url,
            "description_text": item.description_text,
            "play_count": item.play_count,
            "play_count_text": format_count(item.play_count),
            "importance_score": item.importance_score,
            "investment_score": item.investment_score,
            "importance_reason": item.importance_reason,
            "importance_scored_at": item.scored_at,
            "topic": self.current_topic,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }

    def snapshot_item_importance_for_download(self, item: VideoItem) -> None:
        if item.source_type == "youtube":
            try:
                refresh_youtube_metrics(item)
            except Exception as exc:  # noqa: BLE001
                self.events.put(("log", f"《{item.title}》播放量刷新失败，沿用已有分数：{clean_error_message(str(exc))}"))
        update_item_importance_score(item)
        snapshot_at = datetime.now(timezone.utc).isoformat()
        self.events.put(
            (
                "importance_record",
                {
                    **self.video_item_record(item),
                    "importance_snapshot_at": snapshot_at,
                    "play_count_at_download": item.play_count,
                    "importance_score_at_download": item.importance_score,
                    "investment_score_at_download": item.investment_score,
                },
            )
        )
        self.events.put(("log", f"《{item.title}》重要性 {item.importance_score:.1f}/10：{item.importance_reason}"))

    def persist_visible_item_records(self) -> None:
        records = self.research_library.setdefault("items", {})
        changed = False
        for item in self.video_items:
            record = self.video_item_record(item)
            previous = records.get(item.video_id, {})
            if isinstance(previous, dict):
                favorite = bool(previous.get("favorite"))
                previous.update(record)
                previous["favorite"] = favorite
                record = previous
            records[item.video_id] = record
            changed = True
        if changed:
            self.save_research_library()

    def download_selected(self) -> None:
        self.start_download_queue(self.get_selected_items())

    def download_all_available(self) -> None:
        items = [item for item in self.video_items if item.available]
        self.start_download_queue(items)

    def research_selected(self) -> None:
        if self.researching:
            messagebox.showinfo("提示", "当前已有研究整理任务在进行中。")
            return
        if self.downloading:
            messagebox.showinfo("提示", "当前有下载任务在进行中，结束后再开始整理。")
            return
        items = self.get_selected_items()
        if not items:
            messagebox.showwarning("提示", "请先选择至少一个节目。")
            return
        selected_mode = self.transcription_mode_var.get()
        if selected_mode == "极速云端":
            transcription_flow = "极速云端切片并发转录"
            fallback_note = "云端片段失败时，本地 Whisper 会补齐缺失片段。"
        else:
            transcription_flow = "本地 Whisper 转录"
            fallback_note = "本地转录失败且已配置 Key 时，会尝试阿里云 ASR。"
        if not messagebox.askyesno(
            "确认转录整理",
            (
                f"将整理选中的 {len(items)} 个节目。\n\n"
                f"流程：下载到隐藏缓存 -> {transcription_flow} -> 生成中文研究笔记 Markdown。\n"
                f"{fallback_note}\n"
                "正式下载到本地仍由“下载已选/下载精选”完成。\n\n"
                "是否开始？"
            ),
        ):
            return

        try:
            settings = self.build_transcription_settings()
            ensure_runtime_available()
        except TranscriptionError as exc:
            messagebox.showerror("无法开始整理", str(exc))
            return

        ensure_dir(RESEARCH_AUDIO_DIR)
        ensure_dir(RESEARCH_DIGEST_DIR)
        self.stop_requested = False
        self.researching = True
        self.pending_research_item_ids = {item.video_id for item in items}
        self.current_task_var.set(f"当前任务：准备整理 {len(items)} 个节目")
        self.set_status("研究整理队列已启动")
        self.progress_value.set(0)
        self.progress_percent_var.set("0%")
        self.progress_detail_var.set("准备整理")
        browser_mode = self.browser_cookie_var.get().strip()
        if browser_mode in {"chrome", "edge", "firefox"} and not browser_is_available(browser_mode):
            browser_mode = default_browser_cookie_mode()
            self.browser_cookie_var.set(browser_mode)
            self.save_settings()
        threading.Thread(
            target=self.research_worker,
            args=(
                items,
                settings,
                self.audio_format_var.get().strip(),
                self.cookie_file_var.get().strip(),
                browser_mode,
            ),
            daemon=True,
        ).start()

    def research_worker(
        self,
        items: list[VideoItem],
        settings: TranscriptionSettings,
        audio_format: str,
        cookie_file: str,
        browser_cookies: str,
    ) -> None:
        completed = 0
        failed = 0
        skipped = 0
        lightweight = 0
        failures: list[dict[str, str]] = []
        processed_item_ids: list[str] = []
        total = len(items)
        download_dir = Path(self.download_dir_var.get().strip() or DEFAULT_DOWNLOAD_DIR)

        for index, item in enumerate(items, start=1):
            if self.stop_requested:
                break
            if item.video_id and item.video_id not in processed_item_ids:
                processed_item_ids.append(item.video_id)
            self.events.put(("task_status", f"当前任务：正在整理第 {index}/{total} 个 - {item.title}"))
            self.events.put(("row_status", (item.video_id, "整理准备", "0%")))
            try:
                previous_record = self.research_library.get("items", {}).get(item.video_id, {})
                if not isinstance(previous_record, dict):
                    previous_record = {}
                existing_transcript = (
                    find_existing_transcript(RESEARCH_AUDIO_DIR, item.video_id)
                    or find_existing_transcript(download_dir, item.video_id)
                )
                transcript_path: Path | None = None
                transcript_origin = ""
                transcript_engine = ""
                existing_brief = find_existing_brief_transcript(RESEARCH_AUDIO_DIR, item.video_id)
                audio_path: Path | None = (
                    find_existing_download(RESEARCH_AUDIO_DIR, item.video_id)
                    or find_existing_download(download_dir, item.video_id)
                )
                if existing_transcript is not None:
                    conflict, conflict_reason = cached_transcript_identity_conflict(item, existing_transcript)
                    if conflict:
                        quarantined = quarantine_transcript_cache(existing_transcript)
                        self.events.put(
                            (
                                "log",
                                f"《{item.title}》已隔离题文不符的旧转录：{conflict_reason}；"
                                f"备份为 {quarantined.name}，本次改走当前音频 ASR。",
                            )
                        )
                    else:
                        transcript_path = existing_transcript
                        prior_origin = str(previous_record.get("transcript_origin") or "").strip()
                        transcript_origin = prior_origin or "existing_cache"
                        transcript_engine = str(previous_record.get("transcript_engine") or "").strip()
                if transcript_path is not None:
                    self.events.put(("log", f"《{item.title}》已找到本地转录，直接复用：{transcript_path.name}"))
                    self.events.put(("row_status", (item.video_id, "复用转录", "TXT")))
                elif self.official_transcript_url_candidates(item):
                    self.events.put(("row_status", (item.video_id, "官方文稿", "读取中")))
                    transcript_path = self.fetch_official_transcript(item)
                    if transcript_path is not None:
                        transcript_origin = "verified_official"
                        self.events.put(("log", f"《{item.title}》已使用官方 Transcript：{transcript_path.name}"))
                        self.events.put(("row_status", (item.video_id, "官方转录", "已获取")))
                    else:
                        self.events.put(("log", f"《{item.title}》没有可验证的同期官方稿，将下载当前音频并进行 ASR。"))
                if transcript_path is None and existing_brief is not None:
                    self.events.put(
                        (
                            "log",
                            f"《{item.title}》存在旧的简介轻量稿，本次仍会重试当前音频；"
                            "只有媒体依旧不可用时才复用该稿。",
                        )
                    )
                if transcript_path is None and audio_path is None:
                    download_error = ""
                    try:
                        cookie_file, _was_skipped, audio_path = self.download_single_item(
                            item,
                            RESEARCH_AUDIO_DIR,
                            audio_format,
                            cookie_file,
                            browser_cookies,
                            False,
                            index,
                            total,
                        )
                    except Exception as exc:
                        download_error = clean_ui_status(str(exc), 220)
                        transcript_path = self.fetch_official_transcript(item)
                        if transcript_path is None:
                            transcript_path = existing_brief or self.build_episode_brief_transcript(item, download_error)
                            if transcript_path is not None:
                                transcript_origin = "episode_brief_fallback"
                        else:
                            transcript_origin = "verified_official"
                        if transcript_path is None:
                            raise RuntimeError(
                                "音频下载失败，且未找到官方 Transcript 或足够长的节目简介。"
                                f"原始错误：{download_error}"
                            ) from exc
                        if transcript_path.name.endswith(".brief.txt"):
                            lightweight += 1
                            self.events.put(("log", f"《{item.title}》音频下载失败，已改用节目简介生成轻量研究稿：{transcript_path.name}"))
                            self.events.put(("row_status", (item.video_id, "轻量整理", "基于简介")))
                        else:
                            self.events.put(("log", f"《{item.title}》音频下载失败，已改用官方 Transcript：{transcript_path.name}"))
                            self.events.put(("row_status", (item.video_id, "官方转录", "已获取")))
                def report_transcription(message: str, current=item, queue_index=index, queue_total=total) -> None:
                    self.events.put(("log", f"《{current.title}》{message}"))
                    if "准备使用本地 faster-whisper" in message or "正在加载本地 faster-whisper" in message:
                        self.events.put(("row_status", (current.video_id, "本地转录中", "Whisper")))
                        self.events.put(("task_status", f"当前任务：本地 Whisper 转录第 {queue_index}/{queue_total} 个 - {current.title}"))
                        return
                    if "本地 Whisper 失败" in message:
                        self.events.put(("row_status", (current.video_id, "切换云端备用", "阿里云 ASR")))
                        return
                    if "开始极速云端转录" in message:
                        self.events.put(("row_status", (current.video_id, "云端并发转录", "阿里云 ASR")))
                        self.events.put(("task_status", f"当前任务：极速云端转录第 {queue_index}/{queue_total} 个 - {current.title}"))
                        return
                    if "正在用本地 Whisper 补齐" in message:
                        self.events.put(("row_status", (current.video_id, "本地补转", "Whisper")))
                        self.events.put(("task_status", f"当前任务：本地补齐第 {queue_index}/{queue_total} 个 - {current.title}"))
                        return
                    if "开始阿里云 qwen3-asr-flash 备用" in message:
                        self.events.put(("row_status", (current.video_id, "云端备用转录", "阿里云 ASR")))
                        self.events.put(("task_status", f"当前任务：阿里云备用转录第 {queue_index}/{queue_total} 个 - {current.title}"))
                        return
                    progress_match = re.search(r"片段处理进度：(\d+)/(\d+)", message)
                    if progress_match:
                        done = int(progress_match.group(1))
                        segment_total = max(1, int(progress_match.group(2)))
                        overall = ((queue_index - 1) + done / segment_total) / max(1, queue_total) * 100
                        self.events.put(
                            (
                                "progress",
                                {
                                    "video_id": current.video_id,
                                    "percent": overall,
                                    "detail": f"第 {queue_index}/{queue_total} 个 | ASR 片段 {done}/{segment_total}",
                                    "row_progress": f"ASR {done}/{segment_total}",
                                    "row_status": "转录中",
                                },
                            )
                        )
                        return
                    cache_match = re.search(r"片段缓存命中：(\d+)/(\d+)", message)
                    if cache_match:
                        done = int(cache_match.group(1))
                        segment_total = max(1, int(cache_match.group(2)))
                        overall = ((queue_index - 1) + done / segment_total) / max(1, queue_total) * 100
                        self.events.put(
                            (
                                "progress",
                                {
                                    "video_id": current.video_id,
                                    "percent": overall,
                                    "detail": f"第 {queue_index}/{queue_total} 个 | 已复用 ASR 片段 {done}/{segment_total}",
                                    "row_progress": f"复用 {done}/{segment_total}",
                                    "row_status": "转录续跑",
                                },
                            )
                        )
                        return
                    failed_match = re.search(r"片段失败：(\d+)/(\d+)", message)
                    if failed_match:
                        segment_index = int(failed_match.group(1))
                        segment_total = max(1, int(failed_match.group(2)))
                        self.events.put(("row_status", (current.video_id, "片段失败", f"{segment_index}/{segment_total}")))
                        self.events.put(("task_status", f"当前任务：ASR 片段失败第 {queue_index}/{queue_total} 个，片段 {segment_index}/{segment_total}"))
                        return
                    if "本地转录失败" in message:
                        self.events.put(("row_status", (current.video_id, "本地转录失败", "请查看日志")))
                        return
                    active_match = re.search(r"片段(?:开始|重试|完成)：(\d+)/(\d+)", message)
                    if active_match:
                        segment_index = int(active_match.group(1))
                        segment_total = max(1, int(active_match.group(2)))
                        if settings.strategy == "cloud_fast":
                            self.events.put(
                                (
                                    "row_status",
                                    (current.video_id, "云端并发转录", f"并发处理中 · {segment_total} 片"),
                                )
                            )
                            return
                        overall = ((queue_index - 1) + max(0, segment_index - 1) / segment_total) / max(1, queue_total) * 100
                        self.events.put(
                            (
                                "progress",
                                {
                                    "video_id": current.video_id,
                                    "percent": overall,
                                    "detail": f"第 {queue_index}/{queue_total} 个 | ASR 片段 {segment_index}/{segment_total}",
                                    "row_progress": f"ASR {segment_index}/{segment_total}",
                                    "row_status": "转录中",
                                },
                            )
                        )
                        self.events.put(("task_status", f"当前任务：ASR 转录第 {queue_index}/{queue_total} 个，片段 {segment_index}/{segment_total}"))
                        return
                    if "VAD" in message or "切分" in message:
                        self.events.put(("row_status", (current.video_id, "音频切分", "处理中")))
                        return
                    if "准备调用" in message or "开始阿里云" in message:
                        segment_match = re.search(r"共\s*(\d+)\s*个片段", message)
                        segment_text = f"{segment_match.group(1)} 个片段" if segment_match else "等待ASR返回"
                        overall = (queue_index - 1) / max(1, queue_total) * 100
                        self.events.put(
                            (
                                "progress",
                                {
                                    "video_id": current.video_id,
                                    "percent": overall,
                                    "detail": f"第 {queue_index}/{queue_total} 个 | ASR 准备调用，{segment_text}",
                                    "row_progress": "等待ASR返回",
                                    "row_status": "转录中",
                                },
                            )
                        )
                        self.events.put(("task_status", f"当前任务：ASR 准备调用第 {queue_index}/{queue_total} 个 - {current.title}"))
                        return

                if transcript_path is None:
                    if audio_path is None:
                        raise RuntimeError("没有可用音频或官方转录文本。")
                    initial_engine_text = "阿里云并发 ASR" if settings.strategy == "cloud_fast" else "本地 Whisper"
                    self.events.put(("row_status", (item.video_id, "转录中", initial_engine_text)))
                    result = transcribe_file(
                        audio_path,
                        settings,
                        report_transcription,
                    )
                    transcript_path = result.output_path
                    transcript_origin = "audio_asr"
                    transcript_engine = result.backend
                digest_path = self.research_digest_path(item)
                digest_ready = False
                if digest_path.exists() and digest_path.stat().st_mtime >= transcript_path.stat().st_mtime:
                    skipped += 1
                    digest_ready = True
                    self.events.put(("log", f"《{item.title}》中文整理已存在，已复用：{digest_path.name}"))
                elif not settings.api_key.strip():
                    completed += 1
                    self.events.put(
                        (
                            "log",
                            f"《{item.title}》转录文本已就绪；未配置 DASHSCOPE_API_KEY，已跳过中文纪要。",
                        )
                    )
                else:
                    row_status = "轻量整理中" if transcript_path.name.endswith(".brief.txt") else "中文整理中"
                    row_progress = "基于简介" if transcript_path.name.endswith(".brief.txt") else "提取重点"
                    self.events.put(("row_status", (item.video_id, row_status, row_progress)))
                    build_chinese_research_digest(
                        transcript_path=transcript_path,
                        output_path=digest_path,
                        metadata=DigestMetadata(
                            title=item.title,
                            source_name=item.source_name,
                            published_text=item.published_text,
                            duration_text=item.duration_text,
                            webpage_url=item.webpage_url,
                            topic_hint=self.current_topic,
                        ),
                        api_key=settings.api_key,
                    )
                    completed += 1
                    digest_ready = True

                self.events.put(
                    (
                        "research_record",
                        {
                            **self.video_item_record(item),
                            "audio_cache_path": str(audio_path) if audio_path else "",
                            "transcript_path": str(transcript_path),
                            "transcript_kind": "episode_brief_fallback" if transcript_path.name.endswith(".brief.txt") else "transcript",
                            "transcript_origin": transcript_origin or "unknown",
                            "transcript_engine": transcript_engine,
                            "transcript_identity_verified": transcript_origin in {"verified_official", "audio_asr"},
                            "digest_path": str(digest_path) if digest_ready else "",
                            "digested_at": datetime.now(timezone.utc).isoformat() if digest_ready else "",
                        },
                    )
                )
                if transcript_path.name.endswith(".brief.txt"):
                    final_status = "已轻量整理"
                    final_progress = "简介纪要"
                elif digest_ready:
                    final_status = "已整理"
                    final_progress = "中文纪要"
                else:
                    final_status = "已转录"
                    final_progress = "TXT"
                self.events.put(("row_status", (item.video_id, final_status, final_progress)))
                overall = index / total * 100
                self.events.put(
                    (
                        "progress",
                        {
                            "video_id": item.video_id,
                            "percent": overall,
                            "detail": f"已整理 {index}/{total} 个",
                            "row_progress": final_progress,
                            "row_status": final_status,
                        },
                    )
                )
            except Exception as exc:  # noqa: BLE001
                failed += 1
                clean_error = clean_ui_status(str(exc), 180)
                failures.append({"title": item.title, "error": clean_error})
                self.events.put(("row_status", (item.video_id, "整理失败", "查看底部提示")))
                self.events.put(("log", f"《{item.title}》整理失败：{clean_error}"))

        self.events.put(
            (
                    "research_done",
                    {
                        "completed": completed,
                        "skipped": skipped,
                        "failed": failed,
                        "lightweight": lightweight,
                        "item_ids": processed_item_ids,
                        "stopped": self.stop_requested,
                        "total": total,
                        "failures": failures,
                    },
                )
            )

    def official_transcript_url_candidates(self, item: VideoItem) -> list[str]:
        candidates: list[str] = []
        if "aidailybrief.ai/e/" in item.webpage_url:
            base = item.webpage_url.split("?", 1)[0].rstrip("/")
            candidates.append(base + "/transcript.md")
            candidates.append(base + ".md")

        if "AI Daily Brief" in item.source_name:
            published_at = parse_podcast_datetime(item.published_text)
            if published_at:
                # RSS 时区最多只允许前后 1 天偏差；更远日期容易取到别的节目。
                for delta_days in (0, -1, 1):
                    day = (published_at + timedelta(days=delta_days)).strftime("%Y-%m-%d")
                    candidates.append(f"https://aidailybrief.ai/e/{day}/transcript.md")
                    candidates.append(f"https://aidailybrief.ai/e/{day}.md")

        unique: list[str] = []
        for url in candidates:
            if url not in unique:
                unique.append(url)
        return unique

    def fetch_official_transcript(self, item: VideoItem) -> Path | None:
        urls = self.official_transcript_url_candidates(item)
        if not urls:
            return None

        headers = {"User-Agent": REQUEST_USER_AGENT, "Accept": "text/markdown,text/plain,*/*"}
        session = build_retry_session(total_retries=2)
        last_error = ""
        for url in urls:
            try:
                resp = session.get(url, timeout=(8, 24), headers=headers)
                if resp.status_code != 200:
                    body_hint = clean_ui_status(resp.text, 120)
                    last_error = f"HTTP {resp.status_code}: {body_hint}" if body_hint else f"HTTP {resp.status_code}"
                    continue
                text = resp.text.strip()
                if len(text) < 1000:
                    last_error = "返回内容太短，不像可整理文本"
                    continue
                identity_ok, identity_reason = validate_official_transcript_identity(item, text)
                if not identity_ok:
                    last_error = f"官方稿身份校验失败：{identity_reason}"
                    self.events.put(("log", f"《{item.title}》已拒绝错期官方稿：{identity_reason}（{url}）"))
                    continue
                ensure_dir(RESEARCH_AUDIO_DIR)
                transcript_path = RESEARCH_AUDIO_DIR / f"{sanitize_filename(item.title)} [{item.video_id}].txt"
                transcript_path.write_text(text + "\n", encoding="utf-8")
                return transcript_path
            except Exception as exc:  # noqa: BLE001
                last_error = clean_ui_status(str(exc), 120)

        if last_error:
            self.events.put(("log", f"《{item.title}》官方 Transcript 获取失败：{last_error}"))
        return None

    def build_episode_brief_transcript(self, item: VideoItem, failure_reason: str = "") -> Path | None:
        description = strip_html_text(item.description_text or "")
        if len(description) < 80:
            return None

        ensure_dir(RESEARCH_AUDIO_DIR)
        transcript_path = RESEARCH_AUDIO_DIR / f"{sanitize_filename(item.title)} [{item.video_id}].brief.txt"
        lines = [
            "# Episode Brief Fallback",
            "",
            "SOURCE_KIND: EPISODE_BRIEF_FALLBACK",
            "",
            "注意：这不是完整逐字转录。音频下载或官方 Transcript 暂时不可用，本文件仅基于 RSS/节目页简介生成，供快速判断价值。后续如果拿到完整 Transcript，应重新整理。",
            "",
            f"Title: {item.title}",
            f"Source: {item.source_name or 'Unknown'}",
            f"Published: {item.published_text or 'Unknown'}",
            f"Duration: {item.duration_text or 'Unknown'}",
            f"Episode URL: {item.webpage_url or 'Unknown'}",
            f"Audio URL: {item.audio_url or 'Unknown'}",
        ]
        if failure_reason:
            lines.append(f"Fallback reason: {clean_ui_status(failure_reason, 220)}")
        lines.extend(["", "Original episode description:", "", description])
        transcript_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
        return transcript_path

    def research_digest_path(self, item: VideoItem) -> Path:
        return RESEARCH_DIGEST_DIR / f"{sanitize_filename(item.title)} [{item.video_id}].zh.md"

    def start_download_queue(self, items: list[VideoItem]) -> None:
        if self.downloading:
            messagebox.showinfo("提示", "当前已有下载任务在进行中。")
            return
        if self.researching:
            messagebox.showinfo("提示", "当前有研究整理任务在进行中，结束后再开始下载。")
            return
        if not items:
            messagebox.showwarning("提示", "请先选择至少一个节目。")
            return

        download_dir = Path(self.download_dir_var.get().strip() or DEFAULT_DOWNLOAD_DIR)
        ensure_dir(download_dir)
        self.save_settings()

        effective_cookie_file = self.cookie_file_var.get().strip()
        browser_mode = self.browser_cookie_var.get().strip()
        if browser_mode in {"chrome", "edge", "firefox"} and not browser_is_available(browser_mode):
            fallback_mode = default_browser_cookie_mode()
            self.browser_cookie_var.set(fallback_mode)
            browser_mode = fallback_mode
            self.save_settings()
            self.set_status(f"未找到所选浏览器登录态，已切换为 {fallback_mode}")
        allow_edge_refresh = False
        has_youtube_items = any(item.source_type == "youtube" for item in items)
        if has_youtube_items and browser_mode == "edge":
            allow_restart = messagebox.askyesno(
                "提醒主动关闭当前 Edge",
                "开始下载前，请先主动关闭当前所有 Edge 窗口，避免未保存的内容丢失。\n\n点击“是”后，程序会自动强制关闭仍在运行的 Edge 进程，再重新启动 Edge 并导出 YouTube Cookie。\n\n是否继续？",
            )
            if allow_restart:
                try:
                    self.set_status("正在自动刷新 Edge 的 YouTube 登录态...")
                    effective_cookie_file = refresh_edge_cookies_via_cdp()
                    self.cookie_file_var.set(effective_cookie_file)
                    self.save_settings()
                    self.set_status("已自动刷新 Edge Cookie，准备开始下载")
                    allow_edge_refresh = True
                except Exception as exc:
                    messagebox.showwarning(
                        "自动刷新失败",
                        f"脚本没能自动刷新 Edge Cookie。\n\n将继续尝试使用当前 cookie 设置下载。\n\n原因：{exc}",
                    )
            else:
                self.set_status("已跳过自动刷新 Edge Cookie，将使用当前设置继续下载")

        self.stop_requested = False
        self.downloading = True
        self.progress_value.set(0)
        self.progress_percent_var.set("0%")
        self.progress_detail_var.set("准备下载")
        self.current_task_var.set(f"当前任务：准备下载 {len(items)} 个节目")
        self.set_status("下载队列已启动")

        threading.Thread(
            target=self.download_worker,
            args=(
                items,
                download_dir,
                self.audio_format_var.get().strip(),
                effective_cookie_file,
                browser_mode,
                allow_edge_refresh,
                self.auto_transcribe_var.get(),
            ),
            daemon=True,
        ).start()

    def request_stop(self) -> None:
        if not self.downloading and not self.researching:
            self.set_status("当前没有正在执行的下载或整理队列")
            return
        self.stop_requested = True
        self.set_status("已请求停止，当前这一条结束后会停下")

    def on_progress(self, payload: dict) -> None:
        video_id = payload["video_id"]
        percent = payload["percent"]
        detail = payload["detail"]
        row_progress = payload["row_progress"]
        row_status = payload["row_status"]

        self.progress_value.set(percent)
        self.progress_percent_var.set(f"{percent:.1f}%")
        self.progress_detail_var.set(detail)
        self.update_row(video_id, status=row_status, progress=row_progress)

    def on_download_finished(self, payload: dict) -> None:
        self.downloading = False
        if hasattr(self, "progress_bar"):
            self.progress_bar.stop()
        completed = payload["completed"]
        failed = payload["failed"]
        skipped = payload["skipped"]
        stopped = payload["stopped"]
        total = payload["total"]

        if stopped:
            message = f"下载已停止。成功 {completed} 个，跳过 {skipped} 个，失败 {failed} 个，原计划 {total} 个。"
            self.current_task_var.set("当前任务：下载已停止")
        else:
            message = f"下载完成。成功 {completed} 个，跳过 {skipped} 个，失败 {failed} 个，共 {total} 个。"
            self.current_task_var.set("当前任务：下载队列已完成")

        self.progress_value.set(100 if total > 0 and not stopped else self.progress_value.get())
        self.progress_percent_var.set("100%" if total > 0 and not stopped else self.progress_percent_var.get())
        self.progress_detail_var.set(message)
        self.set_status(message)
        messagebox.showinfo("下载结果", message)

    def update_research_record(self, record: dict) -> None:
        item_id = record.get("video_id")
        if not item_id:
            return
        item = self.video_map.get(item_id)
        if item:
            if "play_count" in record:
                item.play_count = parse_count(record.get("play_count"))
            if "importance_score" in record:
                item.importance_score = float(record.get("importance_score") or item.importance_score or 0.0)
            if "investment_score" in record:
                item.investment_score = float(record.get("investment_score") or item.investment_score or 0.0)
            if "importance_reason" in record:
                item.importance_reason = str(record.get("importance_reason") or item.importance_reason or "")
            if "importance_scored_at" in record:
                item.scored_at = str(record.get("importance_scored_at") or item.scored_at or "")
        records = self.research_library.setdefault("items", {})
        previous = records.get(item_id, {})
        if isinstance(previous, dict):
            favorite = bool(previous.get("favorite"))
            incoming_transcript = str(record.get("transcript_path") or "").strip()
            if incoming_transcript:
                incoming_path = Path(incoming_transcript).expanduser()
                try:
                    incoming_hash = self.source_sha256(incoming_path)
                except OSError:
                    incoming_hash = ""
                if incoming_hash:
                    record["transcript_sha256"] = incoming_hash
                previous_transcript_raw = str(previous.get("transcript_path") or "").strip()
                if previous_transcript_raw:
                    previous_transcript_path = Path(previous_transcript_raw).expanduser()
                    try:
                        transcript_path_changed = previous_transcript_path.resolve() != incoming_path.resolve()
                    except OSError:
                        transcript_path_changed = previous_transcript_path != incoming_path
                    if transcript_path_changed:
                        self.clear_stale_tts_cache_for_source(
                            previous,
                            previous_transcript_path,
                            force=True,
                        )
                self.clear_stale_tts_cache_for_source(previous, incoming_path, incoming_hash)
                cached_source_path = str(previous.get("full_translation_source_path") or "").strip()
                cached_source_hash = str(previous.get("full_translation_source_sha256") or "").strip()
                path_changed = False
                if cached_source_path:
                    try:
                        path_changed = Path(cached_source_path).expanduser().resolve() != incoming_path.resolve()
                    except OSError:
                        path_changed = True
                content_changed = bool(cached_source_hash and incoming_hash and cached_source_hash != incoming_hash)
                if path_changed or content_changed:
                    self.clear_full_translation_cache(previous, delete_files=True)
            previous.update(record)
            previous["favorite"] = favorite
            record = previous
        else:
            record["favorite"] = item_id in self.favorite_item_ids
        records[item_id] = record
        self.save_research_library()
        self.refresh_card_visual_or_defer(item_id)
        self.mark_results_dirty(refresh_visible=True)

    def on_research_finished(self, payload: dict) -> None:
        self.researching = False
        if hasattr(self, "progress_bar"):
            self.progress_bar.stop()
        lightweight = int(payload.get("lightweight") or 0)
        message = (
            f"研究整理结束：新整理 {payload['completed']} 个，"
            f"复用 {payload['skipped']} 个，轻量整理 {lightweight} 个，失败 {payload['failed']} 个。"
        )
        if payload.get("stopped"):
            message = f"研究整理已停止。{message}"
        self.current_task_var.set("当前任务：研究整理队列已完成")
        self.progress_detail_var.set(message)
        failures = payload.get("failures") or []
        popup_message = message
        if failures:
            first = failures[0]
            detail = f"失败原因：{first.get('title', '')[:28]} - {first.get('error', '未知错误')}"
            self.set_status(clean_ui_status(detail, 180))
            popup_message += "\n\n首个失败原因：\n" + clean_ui_status(first.get("error", "未知错误"), 240)
        else:
            self.set_status("研究整理完成，可在左侧「资料库」或节目详情里阅读中文纪要。")
            popup_message += "\n\n结果入口：左侧「资料库」，或点击节目卡片查看详情。"
        self.mark_results_dirty(refresh_visible=True)
        messagebox.showinfo("研究整理结果", popup_message)
        item_ids = payload.get("item_ids")
        if isinstance(item_ids, (list, tuple, set)) and item_ids:
            self._start_research_item_translations(item_ids)
        else:
            self._start_research_item_translations(self.pending_research_item_ids)
        self.pending_research_item_ids = set()

    def download_worker(
        self,
        items: list[VideoItem],
        download_dir: Path,
        audio_format: str,
        cookie_file: str,
        browser_cookies: str,
        allow_edge_refresh: bool,
        auto_transcribe: bool,
    ) -> None:
        completed = 0
        failed = 0
        skipped = 0
        total = len(items)
        transcription_paths: list[tuple[Path, str]] = []

        for index, item in enumerate(items, start=1):
            if self.stop_requested:
                break

            self.events.put(("task_status", f"当前任务：正在下载第 {index}/{total} 个 - {item.title}"))
            self.events.put(("row_status", (item.video_id, "排队中", item.progress)))

            try:
                cookie_file, was_skipped, audio_path = self.download_single_item(
                    item,
                    download_dir,
                    audio_format,
                    cookie_file,
                    browser_cookies,
                    allow_edge_refresh,
                    index,
                    total,
                )
                if was_skipped:
                    skipped += 1
                else:
                    completed += 1
                if auto_transcribe:
                    transcription_paths.append((audio_path, item.video_id))
            except Exception as exc:
                failed += 1
                self.events.put(("row_status", (item.video_id, "下载失败", str(exc))))
                self.events.put(("log", f"《{item.title}》下载失败：{exc}"))

        stopped = self.stop_requested
        if stopped:
            self.events.put(("queue_stopped", None))
        self.events.put(
            (
                "download_done",
                {
                    "completed": completed,
                    "failed": failed,
                    "skipped": skipped,
                    "stopped": stopped,
                    "total": total,
                },
            )
        )
        if auto_transcribe and transcription_paths:
            self.events.put(("auto_transcribe", transcription_paths))

    def download_single_item(
        self,
        item: VideoItem,
        download_dir: Path,
        audio_format: str,
        cookie_file: str,
        browser_cookies: str,
        allow_edge_refresh: bool,
        queue_index: int,
        queue_total: int,
    ) -> tuple[str, bool, Path]:
        self.snapshot_item_importance_for_download(item)
        existing_file = find_existing_download(download_dir, item.video_id)
        if existing_file:
            self.events.put(("row_status", (item.video_id, "已存在，已跳过", existing_file.name)))
            self.events.put(("log", f"《{item.title}》已存在，跳过下载：{existing_file.name}"))
            return cookie_file, True, existing_file

        if item.source_type in {"xiaoyuzhou", "podcast_rss"}:
            progress_label = "小宇宙直链下载" if item.source_type == "xiaoyuzhou" else "RSS 音频下载"
            final_path = self.download_direct_audio_item(
                item,
                download_dir,
                audio_format,
                queue_index,
                queue_total,
                progress_label,
            )
            return cookie_file, False, final_path

        outtmpl = str(download_dir / "%(title).180B [%(id)s].%(ext)s")

        def progress_hook(d: dict) -> None:
            status = d.get("status")
            if status == "downloading":
                total_bytes = d.get("total_bytes") or d.get("total_bytes_estimate") or 0
                downloaded_bytes = d.get("downloaded_bytes") or 0
                percent = (downloaded_bytes / total_bytes * 100) if total_bytes else 0.0
                speed = human_speed(d.get("speed"))
                eta = d.get("eta")
                eta_text = f"{int(eta)} 秒" if eta is not None else "-"
                detail = f"第 {queue_index}/{queue_total} 个 | {speed} | ETA {eta_text}"
                row_progress = f"{percent:.1f}%"
                self.events.put(
                    (
                        "progress",
                        {
                            "video_id": item.video_id,
                            "percent": percent,
                            "detail": detail,
                            "row_progress": row_progress,
                            "row_status": "下载中",
                        },
                    )
                )
            elif status == "finished":
                self.events.put(
                    (
                        "progress",
                        {
                            "video_id": item.video_id,
                            "percent": 100.0,
                            "detail": f"第 {queue_index}/{queue_total} 个 | 下载完成，正在处理音频",
                            "row_progress": "100%",
                            "row_status": "处理中",
                        },
                    )
                )

        base_opts = build_youtube_download_options(
            outtmpl,
            audio_format,
            progress_hook,
            ensure_ffmpeg_on_path(),
        )

        self.events.put(("row_status", (item.video_id, "准备下载", "0%")))
        last_errors: list[str] = []

        for attempt_index in range(2):
            attempts = build_download_attempts(base_opts, cookie_file, browser_cookies)
            errors: list[str] = []
            raw_error = ""
            for label, ydl_opts in attempts:
                self.events.put(("log", f"《{item.title}》尝试方式：{label}"))
                try:
                    with YoutubeDL(ydl_opts) as ydl:
                        ydl.download([item.webpage_url])
                    self.events.put(("row_status", (item.video_id, "下载完成", "100%")))
                    self.events.put(("log", f"《{item.title}》下载完成"))
                    final_path = find_existing_download(download_dir, item.video_id)
                    if final_path is None:
                        raise RuntimeError("下载完成但未找到输出音频文件")
                    return cookie_file, False, final_path
                except DownloadError as exc:
                    raw_error = str(exc)
                    clean_error = clean_error_message(raw_error)
                    errors.append(f"{label}: {clean_error}")
                    self.events.put(("row_status", (item.video_id, f"尝试失败({label})", item.progress)))
                    self.events.put(("log", f"《{item.title}》{label}失败：{clean_error}"))

            last_errors = errors
            if (
                attempt_index == 0
                and allow_edge_refresh
                and browser_cookies == "edge"
                and is_cookie_retry_error(raw_error)
            ):
                self.events.put(("log", f"《{item.title}》疑似登录态失效，正在自动刷新 Edge Cookie 并重试一次"))
                self.events.put(("row_status", (item.video_id, "刷新 Cookie 后重试", item.progress)))
                cookie_file = refresh_edge_cookies_via_cdp()
                continue
            break

        raise RuntimeError("；".join(last_errors))

    def download_direct_audio_item(
        self,
        item: VideoItem,
        download_dir: Path,
        audio_format: str,
        queue_index: int,
        queue_total: int,
        progress_label: str,
    ) -> Path:
        if not item.audio_url:
            raise RuntimeError("没有找到可下载的音频地址")

        ensure_dir(download_dir)
        audio_urls = direct_audio_url_candidates(item.audio_url)
        base_name = f"{sanitize_filename(item.title)} [{item.video_id}]"
        final_path = download_dir / f"{base_name}.{audio_format}"
        headers = audio_request_headers(item)
        self.events.put(("row_status", (item.video_id, "准备下载", "0%")))

        temp_paths: list[Path] = []
        last_error = ""

        def temp_path_for(method_label: str, attempt_url: str, content_type: str = "") -> tuple[Path, str]:
            source_ext = resolved_audio_extension(attempt_url, content_type)
            safe_method = re.sub(r"[^a-z0-9]+", "-", method_label.lower()).strip("-") or "direct"
            digest = hashlib.sha1(attempt_url.encode("utf-8")).hexdigest()[:10]
            temp_path = download_dir / f".{item.video_id}.{safe_method}.{digest}.{source_ext}"
            temp_paths.append(temp_path)
            return temp_path, source_ext

        def report_download_progress(method_label: str, downloaded: int, total_bytes: int) -> None:
            percent = (downloaded / total_bytes * 100) if total_bytes else 0.0
            detail = f"第 {queue_index}/{queue_total} 个 | {progress_label} | {method_label}"
            row_progress = f"{percent:.1f}%" if total_bytes else f"{downloaded / 1024 / 1024:.1f} MB"
            self.events.put(
                (
                    "progress",
                    {
                        "video_id": item.video_id,
                        "percent": percent,
                        "detail": detail,
                        "row_progress": row_progress,
                        "row_status": "下载中",
                    },
                )
            )

        def download_with_requests(attempt_url: str) -> tuple[Path, str]:
            session = build_retry_session(total_retries=2)
            with session.get(
                attempt_url,
                stream=True,
                timeout=(AUDIO_DOWNLOAD_CONNECT_TIMEOUT, AUDIO_DOWNLOAD_READ_TIMEOUT),
                headers=headers,
                allow_redirects=True,
            ) as resp:
                resp.raise_for_status()
                if is_textish_audio_response(resp):
                    body_hint = clean_ui_status(resp.text, 160)
                    raise RuntimeError(f"返回 {response_content_type(resp)}，不像音频：{body_hint}")
                temp_path, source_ext = temp_path_for("requests", resp.url or attempt_url, response_content_type(resp))
                temp_path.unlink(missing_ok=True)
                total_bytes = int(resp.headers.get("content-length") or 0)
                downloaded = 0
                with open(temp_path, "wb") as f:
                    for chunk in resp.iter_content(chunk_size=AUDIO_DOWNLOAD_CHUNK_SIZE):
                        if self.stop_requested:
                            raise RuntimeError("已手动停止当前下载队列")
                        if not chunk:
                            continue
                        f.write(chunk)
                        downloaded += len(chunk)
                        report_download_progress("requests", downloaded, total_bytes)
                validate_audio_cache(temp_path, "requests 下载结果")
                return temp_path, source_ext

        def download_with_curl(attempt_url: str) -> tuple[Path, str]:
            curl_path = shutil.which("curl")
            if not curl_path:
                raise RuntimeError("系统没有找到 curl")
            temp_path, source_ext = temp_path_for("curl", attempt_url)
            temp_path.unlink(missing_ok=True)
            self.events.put(
                (
                    "progress",
                    {
                        "video_id": item.video_id,
                        "percent": 0.0,
                        "detail": f"第 {queue_index}/{queue_total} 个 | {progress_label} | curl 兜底",
                        "row_progress": "curl",
                        "row_status": "下载中",
                    },
                )
            )
            cmd = [
                curl_path,
                "--location",
                "--fail",
                "--silent",
                "--show-error",
                "--http1.1",
                "--retry",
                "2",
                "--retry-delay",
                "1",
                "--connect-timeout",
                str(AUDIO_DOWNLOAD_CONNECT_TIMEOUT),
                "--max-time",
                "1800",
                "--user-agent",
                REQUEST_USER_AGENT,
                "-H",
                "Accept: audio/*,video/*,application/octet-stream,*/*",
                "-H",
                f"Referer: {headers.get('Referer', '')}",
                "--output",
                str(temp_path),
                attempt_url,
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=1900)
            if result.returncode != 0:
                temp_path.unlink(missing_ok=True)
                raise RuntimeError(compact_command_output(result.stderr or result.stdout))
            validate_audio_cache(temp_path, "curl 下载结果")
            return temp_path, source_ext

        def download_with_ytdlp(attempt_url: str) -> tuple[Path, str]:
            source_ext = resolved_audio_extension(attempt_url)
            safe_method = "yt-dlp"
            digest = hashlib.sha1(attempt_url.encode("utf-8")).hexdigest()[:10]
            temp_stem = download_dir / f".{item.video_id}.{safe_method}.{digest}"
            for stale_path in temp_stem.parent.glob(temp_stem.name + ".*"):
                stale_path.unlink(missing_ok=True)
            self.events.put(
                (
                    "progress",
                    {
                        "video_id": item.video_id,
                        "percent": 0.0,
                        "detail": f"第 {queue_index}/{queue_total} 个 | {progress_label} | yt-dlp 兜底",
                        "row_progress": "yt-dlp",
                        "row_status": "下载中",
                    },
                )
            )

            def ytdlp_progress_hook(d: dict) -> None:
                if d.get("status") != "downloading":
                    return
                downloaded = int(d.get("downloaded_bytes") or 0)
                total_bytes = int(d.get("total_bytes") or d.get("total_bytes_estimate") or 0)
                report_download_progress("yt-dlp", downloaded, total_bytes)

            ydl_opts = {
                "format": "bestaudio/best",
                "outtmpl": str(temp_stem) + ".%(ext)s",
                "quiet": True,
                "no_warnings": True,
                "socket_timeout": 20,
                "retries": 2,
                "fragment_retries": 2,
                "extractor_retries": 1,
                "http_headers": headers,
                "progress_hooks": [ytdlp_progress_hook],
            }
            with YoutubeDL(ydl_opts) as ydl:
                ydl.download([attempt_url])
            candidates = [path for path in temp_stem.parent.glob(temp_stem.name + ".*") if path.is_file()]
            if not candidates:
                raise RuntimeError("yt-dlp 未生成音频文件")
            temp_path = max(candidates, key=lambda path: path.stat().st_size)
            temp_paths.append(temp_path)
            source_ext = temp_path.suffix.lower().lstrip(".") or source_ext
            validate_audio_cache(temp_path, "yt-dlp 下载结果")
            return temp_path, source_ext

        downloaded_path: Path | None = None
        source_ext = audio_format
        for method_label, downloader in (
            ("requests", download_with_requests),
            ("curl", download_with_curl),
            ("yt-dlp", download_with_ytdlp),
        ):
            for attempt_url in audio_urls:
                if self.stop_requested:
                    raise RuntimeError("已手动停止当前下载队列")
                self.events.put(("log", f"《{item.title}》音频下载尝试：{method_label} | {compact_url(attempt_url)}"))
                try:
                    downloaded_path, source_ext = downloader(attempt_url)
                    self.events.put(("log", f"《{item.title}》音频下载命中：{method_label}"))
                    break
                except Exception as exc:  # noqa: BLE001
                    last_error = clean_error_message(str(exc))
                    self.events.put(("log", f"《{item.title}》{method_label} 失败：{last_error}"))
            if downloaded_path is not None:
                break

        if downloaded_path is None:
            for temp_path in temp_paths:
                temp_path.unlink(missing_ok=True)
            raise RuntimeError(f"音频直链下载失败：{last_error or item.audio_url}")

        if audio_format != source_ext:
            self.events.put(
                (
                    "progress",
                    {
                        "video_id": item.video_id,
                        "percent": 0.0,
                        "detail": f"第 {queue_index}/{queue_total} 个 | 下载完成，开始转换为 {audio_format}",
                        "row_progress": "转码 0%",
                        "row_status": "转换中",
                    },
                )
            )
            def report_conversion_progress(percent: float, position_text: str) -> None:
                detail_suffix = f" | {position_text}" if position_text else ""
                self.events.put(
                    (
                        "progress",
                        {
                            "video_id": item.video_id,
                            "percent": percent,
                            "detail": f"第 {queue_index}/{queue_total} 个 | 转换为 {audio_format} {percent:.1f}%{detail_suffix}",
                            "row_progress": f"转码 {percent:.1f}%",
                            "row_status": "转换中",
                        },
                    )
                )

            convert_audio_file(downloaded_path, final_path, report_conversion_progress)
            downloaded_path.unlink(missing_ok=True)
        else:
            final_path.unlink(missing_ok=True)
            downloaded_path.replace(final_path)

        validate_audio_cache(final_path, "最终音频")
        for temp_path in temp_paths:
            if temp_path != final_path:
                temp_path.unlink(missing_ok=True)
        self.events.put(("row_status", (item.video_id, "下载完成", "100%")))
        self.events.put(("log", f"《{item.title}》下载完成：{final_path.name}"))
        return final_path


def human_speed(speed) -> str:
    if not speed:
        return "-"
    units = ["B/s", "KB/s", "MB/s", "GB/s"]
    value = float(speed)
    for unit in units:
        if value < 1024 or unit == units[-1]:
            return f"{value:.1f} {unit}"
        value /= 1024
    return "-"


def clean_error_message(message: str) -> str:
    text = message.replace("ERROR: ", "").strip()
    replacements = {
        "Sign in to confirm you’re not a bot": "需要有效登录态，YouTube 触发了机器人校验",
        "Sign in to confirm you鈥檙e not a bot": "需要有效登录态，YouTube 触发了机器人校验",
        "Sign in to confirm you're not a bot": "需要有效登录态，YouTube 触发了机器人校验",
        "Requested format is not available": "当前没有拿到可下载音频格式",
        "Failed to decrypt with DPAPI": "浏览器登录态读取失败（Windows DPAPI 解密失败）",
        "Could not copy Chrome cookie database": "浏览器 Cookie 数据库正在被占用，暂时无法读取",
        "could not find firefox cookies database": "没有找到 Firefox 的登录态数据库",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text


def is_cookie_retry_error(message: str) -> bool:
    lowered = (message or "").lower()
    keywords = [
        'sign in to confirm',
        'requested format is not available',
        'cookie',
        'login_info',
        'player response',
        'bot',
    ]
    return any(keyword in lowered for keyword in keywords)


def find_existing_download(download_dir: Path, item_id: str) -> Path | None:
    pattern = re.compile(rf"\[{re.escape(item_id)}\]\.[^.]+$", re.IGNORECASE)
    candidates: list[Path] = []
    for path in sorted(download_dir.glob("*")):
        if not (path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS and pattern.search(path.name)):
            continue
        if not is_readable_audio_cache(path):
            path.unlink(missing_ok=True)
            continue
        candidates.append(path)
    if not candidates:
        return None
    return max(candidates, key=lambda path: (path.stat().st_size, path.stat().st_mtime))


def find_existing_transcript(search_dir: Path, item_id: str) -> Path | None:
    if not search_dir.exists():
        return None
    pattern = re.compile(rf"\[{re.escape(item_id)}\]\.txt$", re.IGNORECASE)
    candidates = [
        path
        for path in sorted(search_dir.glob("*.txt"))
        if path.is_file() and pattern.search(path.name) and is_valid_text_cache(path)
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda path: (path.stat().st_size, path.stat().st_mtime))


def find_existing_brief_transcript(search_dir: Path, item_id: str) -> Path | None:
    if not search_dir.exists():
        return None
    pattern = re.compile(rf"\[{re.escape(item_id)}\]\.brief\.txt$", re.IGNORECASE)
    candidates = [
        path
        for path in sorted(search_dir.glob("*.brief.txt"))
        if path.is_file() and pattern.search(path.name) and is_valid_text_cache(path)
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda path: (path.stat().st_size, path.stat().st_mtime))


def is_valid_text_cache(path: Path) -> bool:
    try:
        return path.is_file() and path.stat().st_size >= 256
    except OSError:
        return False


def is_valid_audio_cache(path: Path) -> bool:
    try:
        return path.is_file() and path.stat().st_size >= MIN_VALID_AUDIO_BYTES
    except OSError:
        return False


def find_edge_executable() -> Path:
    system = platform.system()
    if system == "Darwin":
        candidates = [
            Path("/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge"),
            Path("~/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge").expanduser(),
        ]
    elif system == "Windows":
        candidates = [
            Path(r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"),
            Path(r"C:\Program Files\Microsoft\Edge\Application\msedge.exe"),
        ]
    else:
        candidates = [
            Path("/usr/bin/microsoft-edge"),
            Path("/usr/bin/microsoft-edge-stable"),
            Path("/snap/bin/microsoft-edge"),
        ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError("没有找到 Microsoft Edge 可执行文件")


def stop_edge_processes() -> None:
    system = platform.system()
    if system == "Windows":
        subprocess.run(
            ["powershell", "-NoProfile", "-Command", "Stop-Process -Name msedge -Force -ErrorAction SilentlyContinue"],
            capture_output=True,
            text=True,
            timeout=20,
        )
        return
    if system == "Darwin":
        subprocess.run(
            ["osascript", "-e", 'tell application "Microsoft Edge" to quit'],
            capture_output=True,
            text=True,
            timeout=20,
        )
        time.sleep(3)
        subprocess.run(["pkill", "-x", "Microsoft Edge"], capture_output=True, text=True, timeout=20)
        return
    subprocess.run(["pkill", "-f", "microsoft-edge"], capture_output=True, text=True, timeout=20)


def terminate_process_tree(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return
    if platform.system() == "Windows":
        subprocess.run(
            ["taskkill", "/PID", str(process.pid), "/T", "/F"],
            capture_output=True,
            text=True,
            timeout=20,
        )
        return
    process.terminate()
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        process.kill()


def reserve_local_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def wait_for_cdp_ready(port: int, timeout_seconds: int = 20) -> None:
    deadline = time.time() + timeout_seconds
    last_error = None
    while time.time() < deadline:
        try:
            resp = requests.get(f"http://127.0.0.1:{port}/json", timeout=2)
            if resp.ok:
                return
        except Exception as exc:
            last_error = exc
        time.sleep(0.8)
    raise RuntimeError(f"Edge 调试端口未就绪: {last_error}")


def refresh_edge_cookies_via_cdp(output_path: Path | None = None) -> str:
    output_path = output_path or AUTO_EDGE_COOKIE_FILE
    edge_path = find_edge_executable()
    port = reserve_local_port()

    stop_edge_processes()
    time.sleep(2)

    process = subprocess.Popen(
        [
            str(edge_path),
            f"--remote-debugging-port={port}",
            "--remote-allow-origins=*",
            "--profile-directory=Default",
            "--no-first-run",
            "https://accounts.google.com/",
            "https://www.youtube.com",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    try:
        wait_for_cdp_ready(port)
        time.sleep(8)
        export_edge_cookies(port, output_path)
        if not Path(output_path).exists():
            raise RuntimeError("自动导出的 cookie 文件不存在")
        return str(output_path)
    finally:
        try:
            terminate_process_tree(process)
        except Exception:
            pass


def build_download_attempts(base_opts: dict, cookie_file: str, browser_cookies: str) -> list[tuple[str, dict]]:
    attempts: list[tuple[str, dict]] = []

    cookie_path = Path(cookie_file).expanduser() if cookie_file else None
    if cookie_path and cookie_path.exists():
        opts = dict(base_opts)
        opts["cookiefile"] = str(cookie_path)
        attempts.append(("cookies.txt", opts))

    browser_order: list[str] = []
    mode = (browser_cookies or "auto").strip().lower()
    if mode == "auto":
        browser_order = available_browser_cookie_modes()
    elif mode in {"chrome", "edge", "firefox"}:
        browser_order = [mode] if browser_is_available(mode) else []

    for browser in browser_order:
        opts = dict(base_opts)
        opts["cookiesfrombrowser"] = (browser,)
        attempts.append((f"{browser} 浏览器登录态", opts))

    attempts.append(("无 cookies 直连", dict(base_opts)))
    return attempts


def build_youtube_download_options(
    outtmpl: str,
    audio_format: str,
    progress_hook,
    ffmpeg_path: str | None = None,
) -> dict:
    """Build stable yt-dlp options without pinning a fragile YouTube client."""
    opts = {
        "format": "bestaudio/best",
        "outtmpl": outtmpl,
        "quiet": True,
        "no_warnings": True,
        "socket_timeout": 15,
        "retries": 1,
        "fragment_retries": 1,
        "extractor_retries": 1,
        "remote_components": {"ejs:github"},
        "progress_hooks": [progress_hook],
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": audio_format,
                "preferredquality": "192",
            }
        ],
    }
    if ffmpeg_path:
        opts["ffmpeg_location"] = ffmpeg_path
    return opts


def fetch_ai_podcast_updates() -> tuple[str, list[VideoItem], int, str | None]:
    return fetch_podcast_updates("ai")


def fetch_ai_startup_podcast_updates() -> tuple[str, list[VideoItem], int, str | None]:
    return fetch_podcast_updates("ai_startup")


def fetch_weather_agri_podcast_updates() -> tuple[str, list[VideoItem], int, str | None]:
    return fetch_podcast_updates("weather_agri")


def fetch_science_wellness_podcast_updates() -> tuple[str, list[VideoItem], int, str | None]:
    return fetch_podcast_updates("science_wellness")


def fetch_podcast_updates(topic: str = "all") -> tuple[str, list[VideoItem], int, str | None]:
    feeds, skipped = discover_podcast_feeds(topic)
    youtube_feeds, youtube_skipped = discover_youtube_feeds(topic)
    xiaoyuzhou_feeds, xiaoyuzhou_skipped = discover_xiaoyuzhou_feeds(topic)
    skipped += youtube_skipped
    skipped += xiaoyuzhou_skipped
    episode_rows: list[tuple[datetime | None, VideoItem]] = []
    seen_episode_keys: set[str] = set()
    source_count = len(feeds) + len(youtube_feeds) + len(xiaoyuzhou_feeds)

    with ThreadPoolExecutor(max_workers=min(10, max(1, source_count))) as executor:
        futures = {executor.submit(fetch_podcast_feed_items, feed): feed for feed in feeds}
        futures.update({executor.submit(fetch_youtube_feed_items, feed): feed for feed in youtube_feeds})
        futures.update({executor.submit(fetch_xiaoyuzhou_feed_items, feed): feed for feed in xiaoyuzhou_feeds})
        for future in as_completed(futures):
            feed = futures[future]
            try:
                rows, feed_skipped = future.result()
            except Exception:
                skipped += 1
                continue
            skipped += feed_skipped
            added_from_feed = 0
            source_limit = source_episode_limit(feed)
            for published_at, item in rows:
                episode_key = item.audio_url or item.webpage_url or item.video_id
                if episode_key in seen_episode_keys:
                    skipped += 1
                    continue
                seen_episode_keys.add(episode_key)
                episode_rows.append((published_at, item))
                added_from_feed += 1
                if added_from_feed >= source_limit:
                    break

    if not episode_rows:
        raise RuntimeError("未能从公开播客 RSS、YouTube 频道源或小宇宙节目页读取到可处理节目。")

    epoch = datetime.fromtimestamp(0, tz=timezone.utc)
    episode_rows.sort(key=lambda row: row[0] or epoch, reverse=True)
    episode_rows, duplicate_count = dedupe_episode_rows(episode_rows)
    skipped += duplicate_count
    items = [item for _, item in episode_rows[:AI_DISCOVERY_LIMIT]]
    label = PODCAST_GROUP_LABELS.get(topic, "播客")
    hint = (
        f"已搜索 {len(feeds)} 个 RSS 源 + {len(youtube_feeds)} 个 YouTube 频道源 + {len(xiaoyuzhou_feeds)} 个小宇宙源，"
        f"按发布时间保留最新 {len(items)} 条。"
    )
    return f"{label}自动发现", items, skipped, hint


def discover_ai_podcast_feeds() -> tuple[list[PodcastFeed], int]:
    return discover_podcast_feeds("ai")


def source_episode_limit(feed: PodcastFeed | YouTubeFeed | XiaoyuzhouFeed) -> int:
    if isinstance(feed, YouTubeFeed):
        return YOUTUBE_EPISODES_PER_SOURCE
    if isinstance(feed, XiaoyuzhouFeed):
        return XIAOYUZHOU_EPISODES_PER_SOURCE
    return AI_EPISODES_PER_FEED


def dedupe_episode_rows(rows: list[tuple[datetime | None, VideoItem]]) -> tuple[list[tuple[datetime | None, VideoItem]], int]:
    deduped: list[tuple[datetime | None, VideoItem]] = []
    index_by_key: dict[str, int] = {}
    duplicates = 0

    for row in rows:
        _published_at, item = row
        key = episode_content_key(item)
        existing_index = index_by_key.get(key)
        if existing_index is None:
            index_by_key[key] = len(deduped)
            deduped.append(row)
            continue

        duplicates += 1
        existing_item = deduped[existing_index][1]
        if should_prefer_episode_candidate(item, existing_item):
            deduped[existing_index] = row

    return deduped, duplicates


def episode_content_key(item: VideoItem) -> str:
    title_key = normalize_search_text(item.title)
    if len(title_key) < 12:
        return item.audio_url or item.webpage_url or item.video_id
    date_key = item.published_text if re.match(r"\d{4}-\d{2}-\d{2}$", item.published_text or "") else ""
    return f"{date_key}|{title_key}"


def should_prefer_episode_candidate(new_item: VideoItem, existing_item: VideoItem) -> bool:
    if new_item.source_type == "youtube" and existing_item.source_type != "youtube":
        return True
    if new_item.source_type != "youtube" and existing_item.source_type == "youtube":
        return False
    return bool(new_item.description_text) and not bool(existing_item.description_text)


def discover_podcast_feeds(topic: str = "all") -> tuple[list[PodcastFeed], int]:
    feeds = list(PODCAST_FEED_GROUPS.get(topic, PODCAST_FEED_GROUPS["all"]))
    seen_urls: set[str] = set()
    unique_feeds: list[PodcastFeed] = []
    skipped = 0
    for feed in feeds:
        if feed.feed_url in seen_urls:
            skipped += 1
            continue
        seen_urls.add(feed.feed_url)
        unique_feeds.append(feed)
    return unique_feeds, skipped


def discover_youtube_feeds(topic: str = "all") -> tuple[list[YouTubeFeed], int]:
    feeds = list(YOUTUBE_FEED_GROUPS.get(topic, YOUTUBE_FEED_GROUPS["all"]))
    seen_channel_ids: set[str] = set()
    unique_feeds: list[YouTubeFeed] = []
    skipped = 0
    for feed in feeds:
        if feed.channel_id in seen_channel_ids:
            skipped += 1
            continue
        seen_channel_ids.add(feed.channel_id)
        unique_feeds.append(feed)
    return unique_feeds, skipped


def discover_xiaoyuzhou_feeds(topic: str = "all") -> tuple[list[XiaoyuzhouFeed], int]:
    feeds = list(XIAOYUZHOU_FEED_GROUPS.get(topic, XIAOYUZHOU_FEED_GROUPS["all"]))
    seen_urls: set[str] = set()
    unique_feeds: list[XiaoyuzhouFeed] = []
    skipped = 0
    for feed in feeds:
        if feed.feed_url in seen_urls:
            skipped += 1
            continue
        seen_urls.add(feed.feed_url)
        unique_feeds.append(feed)
    return unique_feeds, skipped


def resolve_ai_podcast_feed(priority: int, label: str, term: str, expected_words: tuple[str, ...]) -> PodcastFeed | None:
    results = search_itunes_podcasts(term)
    match = choose_podcast_result(results, expected_words)
    if not match:
        return None
    feed_url = (match.get("feedUrl") or "").strip()
    if not feed_url:
        return None
    return PodcastFeed(
        name=(match.get("collectionName") or label).strip(),
        feed_url=feed_url,
        source=(match.get("artistName") or "Apple Podcasts").strip(),
        priority=priority,
        artwork_url=(match.get("artworkUrl600") or match.get("artworkUrl100") or "").strip(),
    )


def search_itunes_podcasts(term: str) -> list[dict]:
    session = build_retry_session(total_retries=2)
    resp = session.get(
        ITUNES_SEARCH_URL,
        params={
            "term": term,
            "media": "podcast",
            "entity": "podcast",
            "limit": 8,
            "country": "US",
        },
        timeout=12,
        headers={"User-Agent": REQUEST_USER_AGENT},
    )
    resp.raise_for_status()
    return resp.json().get("results") or []


def choose_podcast_result(results: list[dict], expected_words: tuple[str, ...]) -> dict | None:
    candidates = [result for result in results if result.get("feedUrl")]
    if not candidates:
        return None

    for result in candidates:
        haystack = normalize_search_text(
            " ".join(
                [
                    str(result.get("collectionName") or ""),
                    str(result.get("artistName") or ""),
                ]
            )
        )
        if all(normalize_search_text(word) in haystack for word in expected_words):
            return result

    return candidates[0]


def normalize_search_text(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def contains_cjk(value: str) -> bool:
    return bool(re.search(r"[\u4e00-\u9fff]", value or ""))


def clamp_text(value: str, max_chars: int) -> str:
    compact = re.sub(r"\s+", " ", strip_html_text(value or "")).strip()
    if len(compact) <= max_chars:
        return compact
    return compact[: max(0, max_chars - 1)].rstrip(" ，。；、") + "…"


def clean_ui_status(value: str, max_chars: int = 48) -> str:
    cleaned = strip_html_text(value or "")
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    if not cleaned:
        return "-"
    if len(cleaned) <= max_chars:
        return cleaned
    return cleaned[: max(0, max_chars - 1)].rstrip(" ，。；、") + "…"


def compact_url(url: str, max_chars: int = 120) -> str:
    text = (url or "").strip()
    if len(text) <= max_chars:
        return text
    parsed = urlparse(text)
    if parsed.scheme and parsed.netloc:
        path = parsed.path or ""
        suffix = path[-48:] if len(path) > 48 else path
        compact = f"{parsed.scheme}://{parsed.netloc}/…{suffix}"
        if len(compact) <= max_chars:
            return compact
    return text[: max(0, max_chars - 1)].rstrip("/?&") + "…"


def compact_ffmpeg_error(stderr: str, max_chars: int = 260) -> str:
    lines = [line.strip() for line in (stderr or "").splitlines() if line.strip()]
    useful = [
        line
        for line in lines
        if not line.lower().startswith(
            (
                "ffmpeg version",
                "copyright",
                "built with",
                "configuration:",
                "libav",
                "libsw",
                "libpostproc",
            )
        )
    ]
    text = " | ".join(useful[-4:] or lines[-4:])
    return clean_ui_status(text or "未知错误", max_chars)


def is_relevant_podcast_item(feed: PodcastFeed, title: str, description: str = "") -> bool:
    if feed.name in AI_STARTUP_FEED_NAMES:
        return is_ai_startup_story_item(title, description, feed.name)
    if feed.name not in WEATHER_AGRI_FEED_NAMES:
        return True

    normalized_title = normalize_search_text(title)
    if any(keyword in normalized_title for keyword in WEATHER_AGRI_REJECT_KEYWORDS):
        return False
    if feed.name in WEATHER_AGRI_SOURCE_ALLOW_ALL:
        return True
    return any(keyword in normalized_title for keyword in WEATHER_AGRI_TITLE_KEYWORDS)


def fetch_podcast_feed_items(feed: PodcastFeed) -> tuple[list[tuple[datetime | None, VideoItem]], int]:
    session = build_retry_session(total_retries=2)
    resp = session.get(feed.feed_url, timeout=(10, 30), headers={"User-Agent": REQUEST_USER_AGENT})
    resp.raise_for_status()
    root = ET.fromstring(resp.content)

    entries = podcast_feed_entries(root)
    feed_artwork_url = feed_image_url(root) or feed.artwork_url
    rows: list[tuple[datetime | None, VideoItem]] = []
    skipped = 0

    for entry in entries[:MAX_RSS_ENTRIES_PER_FEED]:
        title = first_child_text(entry, "title")
        audio_url = enclosure_url(entry)
        if not title or not audio_url:
            skipped += 1
            continue
        description_text = episode_description(entry)
        if not is_relevant_podcast_item(feed, title, description_text):
            skipped += 1
            continue

        link = episode_link(entry) or feed.feed_url
        guid = first_child_text(entry, "guid") or first_child_text(entry, "id") or link or audio_url
        published_at = parse_podcast_datetime(
            first_child_text(entry, "pubDate")
            or first_child_text(entry, "published")
            or first_child_text(entry, "updated")
        )
        duration_text = parse_podcast_duration(first_child_text(entry, "duration"))
        date_prefix = published_at.astimezone().strftime("%Y-%m-%d") if published_at else "日期未知"
        artwork_url = podcast_image_url(entry) or feed_artwork_url
        item_id = "rss-" + hashlib.sha1(f"{feed.feed_url}|{guid}|{audio_url}".encode("utf-8")).hexdigest()[:14]
        rows.append(
            (
                published_at,
                VideoItem(
                    video_id=item_id,
                    title=title.strip(),
                    duration_text=duration_text,
                    webpage_url=link,
                    source_type="podcast_rss",
                    audio_url=audio_url,
                    source_name=feed.name,
                    published_text=date_prefix,
                    artwork_url=artwork_url,
                    description_text=description_text,
                    play_count=podcast_entry_play_count(entry),
                ),
            )
        )

    epoch = datetime.fromtimestamp(0, tz=timezone.utc)
    rows.sort(key=lambda row: row[0] or epoch, reverse=True)
    return rows, skipped


def fetch_youtube_feed_items(feed: YouTubeFeed) -> tuple[list[tuple[datetime | None, VideoItem]], int]:
    session = build_retry_session(total_retries=2)
    resp = session.get(feed.feed_url, timeout=(10, 30), headers={"User-Agent": REQUEST_USER_AGENT})
    resp.raise_for_status()
    root = ET.fromstring(resp.content)

    entries = [child for child in list(root) if local_tag(child.tag) == "entry"]
    rows: list[tuple[datetime | None, VideoItem]] = []
    skipped = 0

    for entry in entries[:MAX_RSS_ENTRIES_PER_FEED]:
        video_id = first_child_text(entry, "videoId")
        title = first_child_text(entry, "title") or youtube_media_text(entry, "title")
        if not video_id or not title:
            skipped += 1
            continue

        link = youtube_entry_link(entry) or f"https://www.youtube.com/watch?v={video_id}"
        published_at = parse_podcast_datetime(first_child_text(entry, "published") or first_child_text(entry, "updated"))
        date_prefix = published_at.astimezone().strftime("%Y-%m-%d") if published_at else "日期未知"
        description_text = youtube_media_text(entry, "description")
        if feed.topic == "ai_startup" and not is_ai_startup_story_item(
            title,
            description_text,
            feed.name,
        ):
            skipped += 1
            continue
        artwork_url = youtube_thumbnail_url(entry) or feed.artwork_url
        rows.append(
            (
                published_at,
                VideoItem(
                    video_id=video_id,
                    title=title.strip(),
                    duration_text="-",
                    webpage_url=link,
                    source_type="youtube",
                    source_name=feed.name,
                    published_text=date_prefix,
                    artwork_url=artwork_url,
                    description_text=strip_html_text(description_text),
                ),
            )
        )

    epoch = datetime.fromtimestamp(0, tz=timezone.utc)
    rows.sort(key=lambda row: row[0] or epoch, reverse=True)
    return rows, skipped


def fetch_xiaoyuzhou_feed_items(feed: XiaoyuzhouFeed) -> tuple[list[tuple[datetime | None, VideoItem]], int]:
    _source_title, items, skipped, _subtitle_hint = fetch_xiaoyuzhou_items(feed.podcast_url)
    rows: list[tuple[datetime | None, VideoItem]] = []
    for item in items:
        item.source_name = feed.name
        if feed.topic == "ai_startup" and not is_ai_startup_story_item(
            item.title,
            item.description_text,
            feed.name,
        ):
            skipped += 1
            continue
        if feed.artwork_url and not item.artwork_url:
            item.artwork_url = feed.artwork_url
        rows.append((parse_item_date(item.published_text), item))

    epoch = datetime.fromtimestamp(0, tz=timezone.utc)
    rows.sort(key=lambda row: row[0] or epoch, reverse=True)
    return rows, skipped


def youtube_entry_link(entry: ET.Element) -> str:
    direct_link = first_child_text(entry, "link")
    if direct_link.startswith("http"):
        return direct_link
    for child in list(entry):
        if local_tag(child.tag) != "link":
            continue
        href = (child.attrib.get("href") or child.attrib.get("url") or "").strip()
        if href.startswith("http"):
            return href
    return ""


def youtube_media_group(entry: ET.Element) -> ET.Element | None:
    for child in list(entry):
        if local_tag(child.tag).lower() == "group":
            return child
    return None


def youtube_media_text(entry: ET.Element, child_name: str) -> str:
    group = youtube_media_group(entry)
    if group is None:
        return ""
    return first_child_text(group, child_name)


def youtube_thumbnail_url(entry: ET.Element) -> str:
    group = youtube_media_group(entry)
    if group is None:
        return ""
    for child in list(group):
        tag = local_tag(child.tag).lower()
        if tag in {"thumbnail", "content"}:
            url = (child.attrib.get("url") or child.attrib.get("href") or "").strip()
            if url:
                return url
    return ""


def podcast_feed_entries(root: ET.Element) -> list[ET.Element]:
    if local_tag(root.tag) == "rss":
        channel = first_child(root, "channel")
        if channel is None:
            return []
        return [child for child in list(channel) if local_tag(child.tag) == "item"]
    if local_tag(root.tag) == "feed":
        return [child for child in list(root) if local_tag(child.tag) == "entry"]
    return [child for child in root.iter() if local_tag(child.tag) in {"item", "entry"}]


def feed_image_url(root: ET.Element) -> str:
    candidates: list[ET.Element] = []
    channel = first_child(root, "channel")
    if channel is not None:
        candidates.extend(list(channel))
    candidates.extend(list(root))

    for child in candidates:
        tag = local_tag(child.tag).lower()
        if tag == "image":
            href = (child.attrib.get("href") or child.attrib.get("url") or "").strip()
            if href:
                return href
            url = first_child_text(child, "url")
            if url:
                return url
        if tag == "thumbnail":
            url = (child.attrib.get("url") or child.attrib.get("href") or "").strip()
            if url:
                return url
    return ""


def local_tag(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def first_child(element: ET.Element, child_name: str) -> ET.Element | None:
    for child in list(element):
        if local_tag(child.tag) == child_name:
            return child
    return None


def first_child_text(element: ET.Element, child_name: str) -> str:
    child = first_child(element, child_name)
    if child is None or child.text is None:
        return ""
    return child.text.strip()


def episode_description(entry: ET.Element) -> str:
    candidates: list[str] = []
    for child in list(entry):
        tag = local_tag(child.tag).lower()
        if tag in {"description", "summary", "subtitle", "encoded", "content"} and child.text:
            candidates.append(child.text)
    for text in candidates:
        cleaned = strip_html_text(text)
        if cleaned:
            return cleaned
    return ""


def podcast_entry_play_count(entry: ET.Element) -> int | None:
    for child in list(entry):
        tag = local_tag(child.tag).lower()
        if tag in {"playcount", "play_count", "plays", "listencount", "downloadcount", "downloads"}:
            count = parse_count(child.text or child.attrib.get("value") or child.attrib.get("count"))
            if count is not None:
                return count
        for attr in ("playCount", "play_count", "plays", "listenCount", "downloadCount"):
            count = parse_count(child.attrib.get(attr))
            if count is not None:
                return count
    return None


def strip_html_text(value: str) -> str:
    text = html.unescape(value or "")
    text = re.sub(r"(?is)<(script|style).*?</\1>", " ", text)
    text = re.sub(r"(?s)<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def enclosure_url(entry: ET.Element) -> str:
    fallback = ""
    for child in list(entry):
        tag = local_tag(child.tag)
        if tag == "enclosure":
            url = (child.attrib.get("url") or "").strip()
            media_type = (child.attrib.get("type") or "").lower()
            if url and (not media_type or media_type.startswith("audio/")):
                return url
            fallback = fallback or url
        if tag == "link":
            rel = (child.attrib.get("rel") or "").lower()
            media_type = (child.attrib.get("type") or "").lower()
            url = (child.attrib.get("href") or child.attrib.get("url") or "").strip()
            if url and (rel == "enclosure" or media_type.startswith("audio/")):
                return url
            if url and not fallback:
                fallback = url
    return fallback


def episode_link(entry: ET.Element) -> str:
    direct_link = first_child_text(entry, "link")
    if direct_link.startswith("http"):
        return direct_link
    for child in list(entry):
        if local_tag(child.tag) != "link":
            continue
        rel = (child.attrib.get("rel") or "alternate").lower()
        href = (child.attrib.get("href") or child.attrib.get("url") or "").strip()
        if href and rel in {"alternate", ""}:
            return href
    return direct_link


def podcast_image_url(entry: ET.Element) -> str:
    for child in list(entry):
        tag = local_tag(child.tag).lower()
        if tag in {"image", "thumbnail", "content"}:
            url = (child.attrib.get("href") or child.attrib.get("url") or "").strip()
            media_type = (child.attrib.get("type") or "").lower()
            if url and (tag != "content" or media_type.startswith("image/")):
                return url
    return ""


def parse_podcast_datetime(value: str) -> datetime | None:
    if not value:
        return None
    try:
        parsed = parsedate_to_datetime(value)
    except Exception:
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except Exception:
            return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


def parse_podcast_duration(value: str) -> str:
    value = (value or "").strip()
    if not value:
        return "-"
    if value.isdigit():
        return format_duration(int(value))
    if ":" not in value:
        return value
    try:
        parts = [int(part) for part in value.split(":")]
    except ValueError:
        return value
    seconds = 0
    for part in parts:
        seconds = seconds * 60 + part
    return format_duration(seconds)


def fetch_video_items(url: str) -> tuple[str, list[VideoItem], int, str | None]:
    if is_xiaoyuzhou_url(url):
        return fetch_xiaoyuzhou_items(url)
    return fetch_youtube_items(url)


def fetch_youtube_items(url: str) -> tuple[str, list[VideoItem], int, str | None]:
    opts = {
        "extract_flat": "in_playlist",
        "skip_download": True,
        "quiet": True,
        "no_warnings": True,
        "playlistend": 500,
    }

    with YoutubeDL(opts) as ydl:
        info = ydl.extract_info(url, download=False)

    seen_playlist_urls = set()
    raw_entries: list[dict] = []

    def collect(entry_or_info: dict) -> None:
        entries = entry_or_info.get("entries") or []
        if not entries:
            raw_entries.append(entry_or_info)
            return

        for entry in entries:
            if not entry:
                continue
            nested_url = entry.get("url") or entry.get("webpage_url")
            if looks_like_playlist(entry, nested_url):
                if nested_url in seen_playlist_urls:
                    continue
                seen_playlist_urls.add(nested_url)
                with YoutubeDL(opts) as nested_ydl:
                    nested_info = nested_ydl.extract_info(nested_url, download=False)
                collect(nested_info)
            else:
                raw_entries.append(entry)

    collect(info)

    items: list[VideoItem] = []
    skipped = 0
    seen_video_ids = set()

    for entry in raw_entries:
        video_id = entry.get("id")
        title = (entry.get("title") or "").strip()
        if not video_id or not title or video_id in seen_video_ids:
            skipped += 1
            continue
        if title.startswith("[Private video]") or title.startswith("[Deleted video]"):
            skipped += 1
            continue

        webpage_url = entry.get("webpage_url") or entry.get("url")
        if webpage_url and not str(webpage_url).startswith("http"):
            webpage_url = f"https://www.youtube.com/watch?v={video_id}"
        if not webpage_url:
            webpage_url = f"https://www.youtube.com/watch?v={video_id}"

        seen_video_ids.add(video_id)
        items.append(
            VideoItem(
                video_id=video_id,
                title=title,
                duration_text=format_duration(entry.get("duration")),
                webpage_url=webpage_url,
                source_type="youtube",
                description_text=strip_html_text(entry.get("description") or entry.get("summary") or ""),
                play_count=extract_count_from_mapping(entry),
            )
        )

    source_title = info.get("title") or "未命名来源"
    for item in items:
        item.source_name = source_title
    return source_title, items, skipped, None


def is_xiaoyuzhou_url(url: str) -> bool:
    return "xiaoyuzhoufm.com" in url.lower()


def fetch_xiaoyuzhou_items(url: str) -> tuple[str, list[VideoItem], int, str | None]:
    session = build_retry_session(total_retries=2)
    html = session.get(url, timeout=(10, 30), headers={"User-Agent": REQUEST_USER_AGENT}).text
    match = re.search(r'<script id="__NEXT_DATA__" type="application/json">(.*?)</script>', html)
    if not match:
        raise RuntimeError("未能从小宇宙页面中提取节目数据")

    data = json.loads(match.group(1))
    podcast = data.get("props", {}).get("pageProps", {}).get("podcast")
    if not podcast:
        raise RuntimeError("未能从小宇宙页面中提取播客信息")

    items: list[VideoItem] = []
    skipped = 0
    for episode in podcast.get("episodes", []):
        eid = episode.get("eid")
        title = (episode.get("title") or "").strip()
        audio_url = (
            episode.get("enclosure", {}).get("url")
            or episode.get("media", {}).get("source", {}).get("url")
            or ""
        )
        if not eid or not title or not audio_url:
            skipped += 1
            continue

        published_at = parse_podcast_datetime(str(episode.get("pubDate") or ""))
        published_text = published_at.astimezone().strftime("%Y-%m-%d") if published_at else ""
        items.append(
            VideoItem(
                video_id=eid,
                title=title,
                duration_text=format_duration(episode.get("duration")),
                webpage_url=f"https://www.xiaoyuzhoufm.com/episode/{eid}",
                source_type="xiaoyuzhou",
                audio_url=audio_url,
                published_text=published_text,
                description_text=strip_html_text(episode.get("description") or episode.get("shownotes") or ""),
                play_count=extract_count_from_mapping(episode),
            )
        )

    source_title = podcast.get("title") or "未命名小宇宙播客"
    for item in items:
        item.source_name = source_title
    subtitle_hint = "小宇宙页面里未发现可直接下载的字幕文本；当前只发现 transcript mediaId 占位。"
    return source_title, items, skipped, subtitle_hint


def looks_like_playlist(entry: dict, nested_url: str | None) -> bool:
    if not nested_url:
        return False
    if "playlist?list=" in nested_url:
        return True
    ie_key = (entry.get("ie_key") or "").lower()
    if "tab" in ie_key or "playlist" in ie_key:
        return True
    return False


def sanitize_filename(name: str) -> str:
    cleaned = re.sub(r'[<>:"/\\|?*]+', " ", name)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned[:180] or "untitled"


def detect_audio_extension(audio_url: str) -> str:
    parsed = urlparse(audio_url)
    path = unquote(parsed.path or "").lower()
    suffix = Path(path).suffix.lower().lstrip(".")
    if suffix in KNOWN_AUDIO_EXTENSIONS:
        return suffix
    match = re.search(r"\.(mp3|m4a|wav|mp4|aac|ogg|oga|opus|flac)(?:/|$)", path)
    if match:
        return match.group(1)
    return "m4a"


def direct_audio_url_candidates(audio_url: str) -> list[str]:
    urls = [audio_url]
    parsed = urlparse(audio_url)
    netloc = parsed.netloc.lower()
    marker = "/redirect.mp3/"
    if netloc.endswith("podtrac.com") and marker in parsed.path:
        redirected_path = parsed.path.split(marker, 1)[1].lstrip("/")
        if "/" in redirected_path:
            decoded_path = unquote(redirected_path)
            urls.append("https://" + decoded_path)
            segments = [segment for segment in decoded_path.split("/") if segment]
            for index, segment in enumerate(segments):
                if "." not in segment or segment.endswith(".mp3"):
                    continue
                tail = "/".join(segments[index:])
                if "/" in tail:
                    urls.append("https://" + tail)
    if netloc.endswith("anchor.fm") and "/podcast/play/" in parsed.path:
        encoded_target = parsed.path.rsplit("/", 1)[-1]
        fallback = unquote(encoded_target)
        if fallback.startswith("https://") and fallback not in urls:
            urls.append(fallback)
    return unique_preserve_order(urls)


def resolve_sibling_tool(tool_name: str) -> str | None:
    ensure_runtime_path()
    tool_path = shutil.which(tool_name)
    if tool_path:
        return tool_path
    ffmpeg_path = shutil.which("ffmpeg")
    if ffmpeg_path:
        candidate = Path(ffmpeg_path).with_name(tool_name)
        if candidate.exists():
            return str(candidate)
    return None


def probe_audio_duration_seconds(source_path: Path) -> float | None:
    ffprobe_path = resolve_sibling_tool("ffprobe")
    if not ffprobe_path:
        return None
    try:
        result = subprocess.run(
            [
                ffprobe_path,
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                str(source_path),
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
    except Exception:
        return None
    if result.returncode != 0:
        return None
    try:
        duration = float(result.stdout.strip())
    except ValueError:
        return None
    return duration if duration > 0 else None


def parse_ffmpeg_progress_seconds(progress_value: str) -> float | None:
    value = progress_value.strip()
    if not value or value == "N/A":
        return None
    try:
        return max(float(value) / 1_000_000, 0.0)
    except ValueError:
        pass
    if ":" not in value:
        return None
    try:
        hours_text, minutes_text, seconds_text = value.split(":", 2)
        return int(hours_text) * 3600 + int(minutes_text) * 60 + float(seconds_text)
    except ValueError:
        return None


def convert_audio_file(source_path: Path, target_path: Path, progress_callback=None) -> None:
    target_path.unlink(missing_ok=True)
    ffmpeg_path = resolve_sibling_tool("ffmpeg") or "ffmpeg"
    duration_seconds = probe_audio_duration_seconds(source_path)
    cmd = [
        ffmpeg_path,
        "-nostdin",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-progress",
        "pipe:1",
        "-nostats",
        "-i",
        str(source_path),
        "-vn",
        str(target_path),
    ]
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
    )
    stderr_text = ""
    last_reported_percent = -1.0
    last_reported_at = 0.0
    try:
        if progress_callback:
            progress_callback(0.0, "")
        if process.stdout is not None:
            for raw_line in process.stdout:
                key, _, raw_value = raw_line.strip().partition("=")
                if key not in {"out_time_ms", "out_time_us", "out_time"}:
                    continue
                converted_seconds = parse_ffmpeg_progress_seconds(raw_value)
                if converted_seconds is None or not progress_callback:
                    continue
                if duration_seconds:
                    percent = min(max(converted_seconds / duration_seconds * 100, 0.0), 99.0)
                else:
                    percent = 0.0
                now = time.monotonic()
                if percent - last_reported_percent >= 0.5 or now - last_reported_at >= 1.0:
                    position_text = format_duration(converted_seconds)
                    if duration_seconds:
                        position_text = f"{position_text}/{format_duration(duration_seconds)}"
                    progress_callback(percent, position_text)
                    last_reported_percent = percent
                    last_reported_at = now
        if process.stderr is not None:
            stderr_text = process.stderr.read()
        returncode = process.wait()
    except Exception:
        terminate_process_tree(process)
        raise
    if returncode != 0:
        target_path.unlink(missing_ok=True)
        raise RuntimeError(f"音频转换失败：{compact_ffmpeg_error(stderr_text)}")
    if not is_valid_audio_cache(target_path):
        target_path.unlink(missing_ok=True)
        raise RuntimeError("音频转换失败：输出文件为空或不完整")
    if progress_callback:
        position_text = format_duration(duration_seconds) if duration_seconds else ""
        progress_callback(100.0, position_text)


def main() -> None:
    ensure_dir(APP_DIR)
    ensure_runtime_path()
    ensure_dir(DEFAULT_DOWNLOAD_DIR)
    root = tk.Tk()
    style = ttk.Style(root)
    try:
        style.theme_use("clam")
    except Exception:
        pass
    app = YTAudioDownloaderApp(root)
    if "--start-page" in sys.argv:
        page_index = sys.argv.index("--start-page") + 1
        if page_index < len(sys.argv):
            start_page = sys.argv[page_index].strip()
            if start_page in app.pages:
                root.after(80, lambda key=start_page: app.show_page(key))
    root.mainloop()


if __name__ == "__main__":
    main()
