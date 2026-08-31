from __future__ import annotations

import concurrent.futures
import io
import json
import os
import random
import re
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import dashscope
import librosa
import numpy as np
import soundfile as sf
from dotenv import load_dotenv
from silero_vad import get_speech_timestamps, load_silero_vad
try:
    from faster_whisper import WhisperModel
except Exception:  # noqa: BLE001
    WhisperModel = None

from app.runtime_tools import ensure_ffmpeg_on_path


WAV_SAMPLE_RATE = 16000
SPLIT_TRIGGER_SECONDS = 180
MAX_API_BYTES = 10 * 1024 * 1024
SUPPORTED_EXTENSIONS = {".mp3", ".m4a", ".wav", ".mp4"}
MAX_API_RETRIES = 4
API_RETRY_SLEEP_SECONDS = (2.0, 6.0)
MIN_AUDIO_BYTES = 4096
ASR_REQUEST_TIMEOUT_SECONDS = 150
ASR_MAX_WORKERS = 8
SEGMENT_CACHE_VERSION = 3
SEGMENT_OVERLAP_SECONDS = 1.5
LOCAL_ASR_MODEL = "medium"
LOCAL_ASR_DEVICE = "cpu"
LOCAL_ASR_COMPUTE_TYPE = "int8"
TRANSCRIPTION_STRATEGIES = {"local_first", "cloud_fast"}

ProgressCallback = Callable[[str], None]


class TranscriptionError(RuntimeError):
    """本地切分转录错误。"""


@dataclass
class TranscriptionSettings:
    api_key: str
    target_segment_seconds: int = 120
    max_segment_seconds: int = 180
    workers: int = 4
    strategy: str = "local_first"

    def validate(self) -> None:
        if not 10 <= self.target_segment_seconds <= SPLIT_TRIGGER_SECONDS:
            raise TranscriptionError("目标切分时长必须在 10 到 180 秒之间。")
        if not self.target_segment_seconds <= self.max_segment_seconds <= SPLIT_TRIGGER_SECONDS:
            raise TranscriptionError("最大切分时长必须大于等于目标时长，并且不超过 180 秒。")
        if not 1 <= self.workers <= 16:
            raise TranscriptionError("并发数必须在 1 到 16 之间。")
        if self.strategy not in TRANSCRIPTION_STRATEGIES:
            raise TranscriptionError("转录模式无效，请选择“本地优先”或“极速云端”。")


@dataclass
class TranscriptionResult:
    source_path: Path
    output_path: Path
    skipped: bool
    segment_count: int
    backend: str


_LOCAL_WHISPER_MODELS: dict[str, WhisperModel] = {}
_VAD_MODEL = None


def load_env_api_key(dotenv_path: Path, variable_name: str) -> str:
    load_dotenv(dotenv_path=dotenv_path, override=False)
    return os.getenv(variable_name, "").strip()


def load_api_key(dotenv_path: Path) -> str:
    return load_env_api_key(dotenv_path, "DASHSCOPE_API_KEY")


def load_deepseek_api_key(dotenv_path: Path) -> str:
    return load_env_api_key(dotenv_path, "DEEPSEEK_API_KEY")


def save_env_api_key(dotenv_path: Path, variable_name: str, api_key: str) -> None:
    existing_lines = dotenv_path.read_text(encoding="utf-8").splitlines() if dotenv_path.exists() else []
    replacement = f"{variable_name}={api_key.strip()}"
    output_lines: list[str] = []
    replaced = False
    for line in existing_lines:
        if line.strip().startswith(f"{variable_name}="):
            output_lines.append(replacement)
            replaced = True
        else:
            output_lines.append(line)
    if not replaced:
        output_lines.append(replacement)
    dotenv_path.write_text("\n".join(output_lines).rstrip() + "\n", encoding="utf-8")
    os.environ[variable_name] = api_key.strip()


def save_api_key(dotenv_path: Path, api_key: str) -> None:
    save_env_api_key(dotenv_path, "DASHSCOPE_API_KEY", api_key)


def save_deepseek_api_key(dotenv_path: Path, api_key: str) -> None:
    save_env_api_key(dotenv_path, "DEEPSEEK_API_KEY", api_key)


def ensure_runtime_available() -> None:
    if ensure_ffmpeg_on_path() is None:
        raise TranscriptionError("未找到 ffmpeg。请重新运行 setup_macos.command，或安装系统 ffmpeg。")


def transcribe_file(
    source_path: Path | str | os.PathLike,
    settings: TranscriptionSettings,
    progress_callback: ProgressCallback | None = None,
) -> TranscriptionResult:
    settings.validate()
    ensure_runtime_available()
    if not isinstance(source_path, (str, os.PathLike)):
        raise TranscriptionError(f"转录路径类型异常：{type(source_path).__name__}")
    source_path = Path(source_path)
    source_path = source_path.resolve()
    output_path = source_path.with_suffix(".txt")
    if output_path.exists():
        return TranscriptionResult(
            source_path=source_path,
            output_path=output_path,
            skipped=True,
            segment_count=0,
            backend="existing_cache",
        )
    if source_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise TranscriptionError(f"暂不支持的文件格式：{source_path.suffix}")
    if not source_path.exists():
        raise TranscriptionError(f"文件不存在：{source_path}")
    if source_path.stat().st_size < MIN_AUDIO_BYTES:
        raise TranscriptionError(f"音频缓存为空或不完整：{source_path.name}")

    report = progress_callback or (lambda _message: None)
    report("正在加载媒体并转换为 16 kHz 单声道音频")
    wav = load_audio(source_path)
    duration_seconds = len(wav) / WAV_SAMPLE_RATE
    report(f"媒体加载完成，时长 {format_duration(duration_seconds)}")

    temp_dir = Path(tempfile.mkdtemp(prefix="audio_transcriber_"))
    try:
        segments = build_segments(
            wav,
            target_segment_seconds=settings.target_segment_seconds,
            max_segment_seconds=settings.max_segment_seconds,
            report=report,
        )
        chunk_paths = save_segments(segments, temp_dir)
        checkpoint_dir = output_path.with_suffix(".segments")
        cache_metadata = build_segment_cache_metadata(
            source_path,
            settings,
            len(chunk_paths),
            segments,
        )
        prepare_segment_cache(checkpoint_dir, cache_metadata)
        texts, backend = transcribe_segments(
            chunk_paths,
            settings,
            report,
            checkpoint_dir,
        )
        merged_text = merge_segment_texts(texts)
        if not merged_text:
            raise TranscriptionError("转录完成，但未返回可用文本。")
        temp_output_path = output_path.with_suffix(".txt.tmp")
        temp_output_path.write_text(merged_text + "\n", encoding="utf-8")
        temp_output_path.replace(output_path)
        report(f"转录完成：{output_path.name}")
        return TranscriptionResult(
            source_path=source_path,
            output_path=output_path,
            skipped=False,
            segment_count=len(chunk_paths),
            backend=backend,
        )
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def load_audio(source_path: Path) -> np.ndarray:
    if source_path.stat().st_size < MIN_AUDIO_BYTES:
        raise TranscriptionError(f"音频缓存为空或不完整：{source_path.name}")
    try:
        return load_audio_with_ffmpeg(source_path)
    except Exception as ffmpeg_exc:  # noqa: BLE001
        try:
            wav_data, _ = librosa.load(str(source_path), sr=WAV_SAMPLE_RATE, mono=True)
            return np.asarray(wav_data, dtype=np.float32)
        except Exception as librosa_exc:  # noqa: BLE001
            raise TranscriptionError(
                f"读取媒体失败：ffmpeg={ffmpeg_exc}；librosa={librosa_exc}"
            ) from librosa_exc


def load_audio_with_ffmpeg(source_path: Path) -> np.ndarray:
    ffmpeg_path = ensure_ffmpeg_on_path() or "ffmpeg"
    command = [
        ffmpeg_path,
        "-nostdin",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(source_path),
        "-ar",
        str(WAV_SAMPLE_RATE),
        "-ac",
        "1",
        "-c:a",
        "pcm_s16le",
        "-f",
        "wav",
        "-",
    ]
    result = subprocess.run(command, capture_output=True, timeout=1800)
    if result.returncode != 0:
        error_text = compact_ffmpeg_error(result.stderr.decode("utf-8", errors="ignore"))
        raise TranscriptionError(f"ffmpeg 读取媒体失败：{error_text or '未知错误'}")
    with io.BytesIO(result.stdout) as buffer:
        wav_data, _ = sf.read(buffer, dtype="float32")
    return np.asarray(wav_data, dtype=np.float32)


def cached_vad_model():
    global _VAD_MODEL
    if _VAD_MODEL is None:
        _VAD_MODEL = load_silero_vad(onnx=True)
    return _VAD_MODEL


def load_audio_with_librosa(source_path: Path) -> np.ndarray:
    try:
        wav_data, _ = librosa.load(str(source_path), sr=WAV_SAMPLE_RATE, mono=True)
        return np.asarray(wav_data, dtype=np.float32)
    except Exception as exc:  # noqa: BLE001
        raise TranscriptionError(f"librosa 读取媒体失败：{exc}") from exc


def compact_ffmpeg_error(stderr: str) -> str:
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
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) > 260:
        return text[:259].rstrip(" ,.;，。；、") + "…"
    return text


def build_segments(
    wav: np.ndarray,
    target_segment_seconds: int = 120,
    max_segment_seconds: int = 180,
    report: ProgressCallback | None = None,
) -> list[tuple[int, int, np.ndarray]]:
    if len(wav) == 0:
        raise TranscriptionError("音频内容为空。")
    reporter = report or (lambda _message: None)
    if len(wav) / WAV_SAMPLE_RATE < SPLIT_TRIGGER_SECONDS:
        return [(0, len(wav), wav)]

    reporter("音频达到 180 秒，正在使用 Silero VAD 查找自然停顿")
    try:
        vad_model = cached_vad_model()
        speech_timestamps = get_speech_timestamps(
            wav,
            vad_model,
            sampling_rate=WAV_SAMPLE_RATE,
            return_seconds=False,
            min_speech_duration_ms=1500,
            min_silence_duration_ms=500,
        )
        if not speech_timestamps:
            raise ValueError("VAD 未检测到语音片段")
        segments = _split_near_speech_starts(wav, speech_timestamps, target_segment_seconds, max_segment_seconds)
        reporter(f"VAD 切分完成，共 {len(segments)} 个片段")
        return segments
    except Exception as exc:  # noqa: BLE001
        reporter(f"VAD 切分失败，改用固定长度切分：{exc}")
        return split_fixed_length(wav, max_segment_seconds)


def _split_near_speech_starts(
    wav: np.ndarray,
    speech_timestamps: list[dict],
    target_segment_seconds: int,
    max_segment_seconds: int,
) -> list[tuple[int, int, np.ndarray]]:
    potential_splits = {0, len(wav)}
    potential_splits.update(int(item["start"]) for item in speech_timestamps if "start" in item)
    sorted_splits = sorted(potential_splits)
    target_samples = target_segment_seconds * WAV_SAMPLE_RATE
    chosen_splits = {0, len(wav)}
    target = target_samples
    while target < len(wav):
        chosen_splits.add(min(sorted_splits, key=lambda point: abs(point - target)))
        target += target_samples

    max_samples = max_segment_seconds * WAV_SAMPLE_RATE
    final_splits = [0]
    for end in sorted(chosen_splits)[1:]:
        start = final_splits[-1]
        while end - start > max_samples:
            start += max_samples
            final_splits.append(start)
        if end > final_splits[-1]:
            final_splits.append(end)
    return _segments_from_split_points(wav, final_splits)


def split_fixed_length(wav: np.ndarray, max_segment_seconds: int) -> list[tuple[int, int, np.ndarray]]:
    max_samples = max_segment_seconds * WAV_SAMPLE_RATE
    split_points = list(range(0, len(wav), max_samples))
    if not split_points or split_points[-1] != len(wav):
        split_points.append(len(wav))
    return _segments_from_split_points(wav, split_points)


def _segments_from_split_points(wav: np.ndarray, split_points: list[int]) -> list[tuple[int, int, np.ndarray]]:
    segments = []
    overlap_samples = int(SEGMENT_OVERLAP_SECONDS * WAV_SAMPLE_RATE)
    for index, (start, end) in enumerate(zip(split_points, split_points[1:])):
        if index > 0:
            start = max(0, start - overlap_samples)
        if end > start:
            segments.append((start, end, wav[start:end]))
    return segments


def save_segments(segments: list[tuple[int, int, np.ndarray]], temp_dir: Path) -> list[Path]:
    temp_dir.mkdir(parents=True, exist_ok=True)
    chunk_paths = []
    for index, (_, _, wav_data) in enumerate(segments):
        chunk_path = temp_dir / f"chunk_{index:03d}.wav"
        sf.write(chunk_path, wav_data, WAV_SAMPLE_RATE)
        if chunk_path.stat().st_size > MAX_API_BYTES:
            raise TranscriptionError(f"切分片段仍超过 10 MB：{chunk_path.name}")
        chunk_paths.append(chunk_path)
    return chunk_paths


def transcribe_segments(
    chunk_paths: list[Path],
    settings: TranscriptionSettings,
    report: ProgressCallback | None = None,
    checkpoint_dir: Path | None = None,
) -> tuple[list[str], str]:
    """Run the selected ASR strategy while keeping local Whisper as a safe fallback."""
    settings.validate()
    if settings.strategy == "cloud_fast":
        return transcribe_segments_cloud_fast(chunk_paths, settings, report, checkpoint_dir)
    return transcribe_segments_local_first(chunk_paths, settings, report, checkpoint_dir)


def transcribe_segments_cloud_fast(
    chunk_paths: list[Path],
    settings: TranscriptionSettings,
    report: ProgressCallback | None = None,
    checkpoint_dir: Path | None = None,
) -> tuple[list[str], str]:
    reporter = report or (lambda _message: None)
    if not settings.api_key.strip():
        reporter("极速云端未配置 DASHSCOPE_API_KEY，已改用本地 Whisper。")
        texts = transcribe_segments_local(chunk_paths, reporter, checkpoint_dir)
        return texts, "local_whisper"

    effective_workers = effective_asr_workers(settings)
    worker_note = (
        f"设置并发 {settings.workers}，安全并发 {effective_workers}"
        if settings.workers != effective_workers
        else f"并发数 {effective_workers}"
    )
    reporter(
        f"开始极速云端转录：阿里云 qwen3-asr-flash，共 {len(chunk_paths)} 个片段，{worker_note}"
    )
    try:
        texts = transcribe_segments_cloud(chunk_paths, settings, reporter, checkpoint_dir)
        return texts, "aliyun_cloud_fast"
    except TranscriptionError as cloud_error:
        reporter(f"极速云端有片段失败：{cloud_error}")
        reporter("正在用本地 Whisper 补齐失败片段；云端已完成的片段会直接复用。")
        try:
            texts = transcribe_segments_local(chunk_paths, reporter, checkpoint_dir)
        except TranscriptionError as local_error:
            raise TranscriptionError(
                f"极速云端和本地补转均未完成。云端错误：{cloud_error}；本地错误：{local_error}"
            ) from local_error
        return texts, "aliyun_cloud_fast_with_local_fallback"


def transcribe_segments_local_first(
    chunk_paths: list[Path],
    settings: TranscriptionSettings,
    report: ProgressCallback | None = None,
    checkpoint_dir: Path | None = None,
) -> tuple[list[str], str]:
    reporter = report or (lambda _message: None)
    reporter(f"准备使用本地 faster-whisper（{LOCAL_ASR_MODEL}），共 {len(chunk_paths)} 个片段")
    try:
        texts = transcribe_segments_local(chunk_paths, reporter, checkpoint_dir)
        return texts, "local_whisper"
    except TranscriptionError as local_error:
        reporter(f"本地 Whisper 失败：{local_error}")
        if not settings.api_key.strip():
            raise TranscriptionError(
                "本地 Whisper 转录失败，且未配置 DASHSCOPE_API_KEY，无法启用阿里云 ASR 备用。"
                f"本地错误：{local_error}"
            ) from local_error

        effective_workers = effective_asr_workers(settings)
        worker_note = (
            f"，设置并发 {settings.workers}，稳态并发 {effective_workers}"
            if settings.workers != effective_workers
            else f"，并发数 {effective_workers}"
        )
        reporter(f"开始阿里云 qwen3-asr-flash 备用，共 {len(chunk_paths)} 个片段{worker_note}")
        texts = transcribe_segments_cloud(chunk_paths, settings, reporter, checkpoint_dir)
        return texts, "local_whisper_with_aliyun_fallback"


def transcribe_segments_cloud(
    chunk_paths: list[Path],
    settings: TranscriptionSettings,
    report: ProgressCallback | None = None,
    checkpoint_dir: Path | None = None,
) -> list[str]:
    reporter = report or (lambda _message: None)
    results = load_segment_checkpoint_texts(checkpoint_dir, len(chunk_paths)) if checkpoint_dir else {}
    cached_count = len(results)
    if cached_count:
        reporter(f"片段缓存命中：{cached_count}/{len(chunk_paths)}")
    missing = [index for index in range(len(chunk_paths)) if index not in results]
    if not missing:
        reporter(f"片段处理进度：{len(chunk_paths)}/{len(chunk_paths)}")
        return [results[index] for index in range(len(chunk_paths))]

    errors: list[str] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=effective_asr_workers(settings)) as executor:
        futures = {
            executor.submit(
                transcribe_segment,
                chunk_paths[index],
                settings.api_key,
                index + 1,
                len(chunk_paths),
                reporter,
            ): index
            for index in missing
        }
        completed = cached_count
        for future in concurrent.futures.as_completed(futures):
            index = futures[future]
            try:
                text = future.result()
                results[index] = text
                if checkpoint_dir is not None:
                    write_segment_checkpoint(checkpoint_dir, index, text)
                completed += 1
                reporter(f"片段处理进度：{completed}/{len(chunk_paths)}")
            except Exception as exc:  # noqa: BLE001
                errors.append(f"{index + 1}/{len(chunk_paths)}: {exc}")
                reporter(f"片段失败：{index + 1}/{len(chunk_paths)}（阿里云：{exc}）")

    if errors:
        raise TranscriptionError("阿里云转录未能完成全部片段，已缓存成功片段；下次会从缺失片段继续：" + "；".join(errors[:3]))
    return [results[index] for index in range(len(chunk_paths))]


def transcribe_segments_local(
    chunk_paths: list[Path],
    report: ProgressCallback | None = None,
    checkpoint_dir: Path | None = None,
) -> list[str]:
    reporter = report or (lambda _message: None)
    if WhisperModel is None:
        raise TranscriptionError("本地转录未安装：缺少 faster-whisper，请先执行 pip install faster-whisper。")

    results = load_segment_checkpoint_texts(checkpoint_dir, len(chunk_paths)) if checkpoint_dir else {}
    cached_count = len(results)
    if cached_count:
        reporter(f"片段缓存命中：{cached_count}/{len(chunk_paths)}")
    missing = [index for index in range(len(chunk_paths)) if index not in results]
    if not missing:
        reporter(f"片段处理进度：{len(chunk_paths)}/{len(chunk_paths)}")
        return [results[index] for index in range(len(chunk_paths))]

    model = build_local_whisper_model(reporter)
    completed = cached_count
    errors: list[str] = []
    for index in missing:
        try:
            text = transcribe_segment_local(model, chunk_paths[index], index + 1, len(chunk_paths), reporter)
            results[index] = text
            if checkpoint_dir is not None:
                write_segment_checkpoint(checkpoint_dir, index, text)
            completed += 1
            reporter(f"片段处理进度：{completed}/{len(chunk_paths)}")
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{index + 1}/{len(chunk_paths)}: {exc}")
            reporter(f"片段失败：{index + 1}/{len(chunk_paths)}")
    if errors:
        raise TranscriptionError("本地转录失败：未能完成全部片段：" + "；".join(errors[:3]))

    return [results[index] for index in range(len(chunk_paths))]


def build_local_whisper_model(report: ProgressCallback | None = None) -> WhisperModel:
    reporter = report or (lambda _message: None)
    model_key = f"{LOCAL_ASR_MODEL}:{LOCAL_ASR_DEVICE}:{LOCAL_ASR_COMPUTE_TYPE}"
    cached = _LOCAL_WHISPER_MODELS.get(model_key)
    if cached is not None:
        return cached

    reporter(f"正在加载本地 faster-whisper 模型：{LOCAL_ASR_MODEL}")
    if WhisperModel is None:
        raise TranscriptionError("本地转录未安装：缺少 faster-whisper，请先执行 pip install faster-whisper。")
    model = WhisperModel(
        LOCAL_ASR_MODEL,
        device=LOCAL_ASR_DEVICE,
        compute_type=LOCAL_ASR_COMPUTE_TYPE,
    )
    _LOCAL_WHISPER_MODELS[model_key] = model
    return model


def effective_asr_workers(settings: TranscriptionSettings) -> int:
    return max(1, min(settings.workers, ASR_MAX_WORKERS))


def build_segment_cache_metadata(
    source_path: Path,
    settings: TranscriptionSettings,
    segment_count: int,
    segments: list[tuple[int, int, np.ndarray]] | None = None,
) -> dict:
    stat = source_path.stat()
    effective_engine = (
        "qwen3-asr-flash"
        if settings.strategy == "cloud_fast" and settings.api_key.strip()
        else f"faster-whisper:{LOCAL_ASR_MODEL}"
    )
    return {
        "version": SEGMENT_CACHE_VERSION,
        "source_path": str(source_path),
        "source_size": stat.st_size,
        "source_mtime_ns": stat.st_mtime_ns,
        "target_segment_seconds": settings.target_segment_seconds,
        "max_segment_seconds": settings.max_segment_seconds,
        "segment_overlap_seconds": SEGMENT_OVERLAP_SECONDS,
        "segment_count": segment_count,
        "segment_layout": (
            [{"start_sample": start, "end_sample": end} for start, end, _wav_data in segments]
            if segments is not None
            else None
        ),
        "transcription_strategy": settings.strategy,
        "effective_engine": effective_engine,
        "local_model": LOCAL_ASR_MODEL,
        "cloud_model": "qwen3-asr-flash",
        "cloud_options": {"enable_lid": True, "enable_itn": False},
        "merge_version": 1,
    }


def prepare_segment_cache(checkpoint_dir: Path, metadata: dict) -> None:
    metadata_path = checkpoint_dir / "metadata.json"
    if metadata_path.exists():
        try:
            existing = json.loads(metadata_path.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            existing = None
        if existing != metadata:
            shutil.rmtree(checkpoint_dir, ignore_errors=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")


def load_segment_checkpoint_texts(checkpoint_dir: Path | None, segment_count: int) -> dict[int, str]:
    if checkpoint_dir is None or not checkpoint_dir.exists():
        return {}
    results: dict[int, str] = {}
    for index in range(segment_count):
        path = segment_checkpoint_path(checkpoint_dir, index)
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8", errors="ignore").strip()
        if text:
            results[index] = text
    return results


def write_segment_checkpoint(checkpoint_dir: Path, index: int, text: str) -> None:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    path = segment_checkpoint_path(checkpoint_dir, index)
    tmp_path = path.with_suffix(".tmp")
    tmp_path.write_text(text.strip() + "\n", encoding="utf-8")
    tmp_path.replace(path)


def segment_checkpoint_path(checkpoint_dir: Path, index: int) -> Path:
    return checkpoint_dir / f"segment_{index:03d}.txt"


def transcribe_segment(
    chunk_path: Path,
    api_key: str,
    segment_index: int = 1,
    segment_total: int = 1,
    report: ProgressCallback | None = None,
) -> str:
    audio_source = str(chunk_path.resolve())
    last_error: Exception | None = None
    reporter = report or (lambda _message: None)
    for attempt in range(1, MAX_API_RETRIES + 1):
        try:
            if attempt == 1:
                reporter(f"片段开始：{segment_index}/{segment_total}")
            else:
                reporter(f"片段重试：{segment_index}/{segment_total}，第 {attempt}/{MAX_API_RETRIES} 次")
            response = dashscope.MultiModalConversation.call(
                api_key=api_key,
                model="qwen3-asr-flash",
                messages=[
                    {"role": "system", "content": [{"text": ""}]},
                    {"role": "user", "content": [{"audio": audio_source}]},
                ],
                result_format="message",
                asr_options={"enable_lid": True, "enable_itn": False},
                request_timeout=ASR_REQUEST_TIMEOUT_SECONDS,
            )
            status_code = getattr(response, "status_code", None)
            if status_code != 200:
                raise TranscriptionError(_format_dashscope_error(response))
            text = _extract_response_text(response)
            if not text:
                raise TranscriptionError(f"接口未返回可用文本：{_format_dashscope_error(response)}")
            reporter(f"片段完成：{segment_index}/{segment_total}")
            return post_text_process(text)
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            if attempt < MAX_API_RETRIES:
                time.sleep(random.uniform(*API_RETRY_SLEEP_SECONDS))
    raise TranscriptionError(f"片段转录失败，已重试 {MAX_API_RETRIES} 次：{last_error}")


def transcribe_segment_local(
    model: WhisperModel,
    chunk_path: Path,
    segment_index: int,
    segment_total: int,
    report: ProgressCallback | None = None,
) -> str:
    reporter = report or (lambda _message: None)
    reporter(f"片段开始：{segment_index}/{segment_total}")
    try:
        segments, _ = model.transcribe(
            str(chunk_path.resolve()),
            vad_filter=False,
            condition_on_previous_text=False,
        )
        text = "".join(part.text for part in segments).strip()
        if not text:
            raise TranscriptionError("接口未返回可用文本。")
        reporter(f"片段完成：{segment_index}/{segment_total}")
        return post_text_process(text)
    except Exception as exc:  # noqa: BLE001
        raise TranscriptionError(f"本地片段转录失败：{exc}") from exc


def _format_dashscope_error(response: object) -> str:
    def read_field(name: str) -> object:
        if isinstance(response, dict):
            return response.get(name)
        return getattr(response, name, None)

    parts: list[str] = []
    status_code = read_field("status_code")
    if status_code:
        parts.append(f"HTTP {status_code}")
    for field in ("code", "message", "request_id"):
        value = read_field(field)
        if value:
            parts.append(f"{field}={value}")
    if parts:
        text = "；".join(parts)
    else:
        text = str(response)
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) > 360:
        return text[:359].rstrip(" ,.;，。；、") + "…"
    return text or "未知 DashScope 错误"


def _extract_response_text(response: object) -> str:
    output = getattr(response, "output", None)
    if output is None and isinstance(response, dict):
        output = response.get("output")
    choices = (output or {}).get("choices") or []
    if not choices:
        return ""
    content = (choices[0].get("message") or {}).get("content") or []
    return "\n".join(
        str(item.get("text", "")).strip()
        for item in content
        if isinstance(item, dict) and str(item.get("text", "")).strip()
    )


def merge_segment_texts(texts: list[str]) -> str:
    """Merge ordered ASR chunks and remove text repeated by the audio overlap."""
    cleaned = [text.strip() for text in texts if text and text.strip()]
    if not cleaned:
        return ""

    merged_parts = [cleaned[0]]
    merged_text = cleaned[0]
    for current in cleaned[1:]:
        cut_index = _segment_overlap_cut_index(merged_text, current)
        remainder = current[cut_index:].lstrip(" \t\r\n,.;:!?，。；：！？、-—")
        if not remainder:
            continue
        merged_parts.append(remainder)
        merged_text = "\n".join(merged_parts)
    return "\n".join(merged_parts).strip()


def _segment_overlap_cut_index(previous: str, current: str, max_tokens: int = 80) -> int:
    previous_tokens = _normalized_tokens_with_spans(previous)
    current_tokens = _normalized_tokens_with_spans(current)
    max_overlap = min(len(previous_tokens), len(current_tokens), max_tokens)
    for token_count in range(max_overlap, 2, -1):
        previous_suffix = [item[0] for item in previous_tokens[-token_count:]]
        current_prefix = [item[0] for item in current_tokens[:token_count]]
        if previous_suffix != current_prefix:
            continue
        chinese_only = all(len(token) == 1 and "\u3400" <= token <= "\u9fff" for token in current_prefix)
        if chinese_only and token_count < 4:
            continue
        return current_tokens[token_count - 1][2]
    return 0


def _normalized_tokens_with_spans(text: str) -> list[tuple[str, int, int]]:
    token_pattern = re.compile(r"[A-Za-z0-9]+(?:['’][A-Za-z0-9]+)?|[\u3400-\u4dbf\u4e00-\u9fff]")
    return [
        (match.group(0).casefold().replace("’", "'"), match.start(), match.end())
        for match in token_pattern.finditer(text)
    ]


def post_text_process(text: str, threshold: int = 20) -> str:
    text = re.sub(rf"(.)\1{{{threshold},}}", r"\1", text)
    return _remove_repeated_patterns(text, threshold)


def _remove_repeated_patterns(text: str, threshold: int, max_pattern_length: int = 20) -> str:
    index = 0
    result: list[str] = []
    while index < len(text):
        matched = False
        for pattern_length in range(1, min(max_pattern_length, len(text) - index) + 1):
            pattern = text[index : index + pattern_length]
            repeated = 1
            while text.startswith(pattern, index + repeated * pattern_length):
                repeated += 1
            if repeated >= threshold:
                result.append(pattern)
                index += repeated * pattern_length
                matched = True
                break
        if not matched:
            result.append(text[index])
            index += 1
    return "".join(result)


def format_duration(seconds: float) -> str:
    total_seconds = int(seconds)
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"
