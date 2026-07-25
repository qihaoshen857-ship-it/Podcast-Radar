#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import queue
import shutil
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

import main
from app.research_digest import DigestMetadata, build_chinese_research_digest
from app.transcription import load_api_key


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit cached podcast transcript titles.")
    parser.add_argument("--library", type=Path, required=True, help="research_library.json path")
    parser.add_argument("--repair", action="store_true", help="quarantine mismatches and rebuild records")
    parser.add_argument(
        "--brief-on-missing",
        action="store_true",
        help="create a clearly-labelled RSS brief when no verified transcript is available",
    )
    return parser.parse_args()


def item_from_record(video_id: str, record: dict) -> main.VideoItem:
    return main.VideoItem(
        video_id=video_id,
        title=str(record.get("title") or "Untitled"),
        duration_text=str(record.get("duration_text") or ""),
        webpage_url=str(record.get("webpage_url") or ""),
        source_type=str(record.get("source_type") or "podcast_rss"),
        audio_url=str(record.get("audio_url") or ""),
        source_name=str(record.get("source_name") or ""),
        published_text=str(record.get("published_text") or ""),
        artwork_url=str(record.get("artwork_url") or ""),
        description_text=str(record.get("description_text") or ""),
    )


def atomic_write_json(path: Path, payload: dict) -> None:
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
        temp_path = Path(handle.name)
    os.replace(temp_path, path)


def configure_live_cache(library_path: Path) -> None:
    app_dir = library_path.parent
    main.RESEARCH_AUDIO_DIR = app_dir / ".cache" / "research" / "audio"
    main.RESEARCH_DIGEST_DIR = app_dir / ".cache" / "research" / "digests"
    main.ensure_dir(main.RESEARCH_AUDIO_DIR)
    main.ensure_dir(main.RESEARCH_DIGEST_DIR)


def build_digest(item: main.VideoItem, transcript_path: Path, api_key: str) -> Path:
    digest_path = main.RESEARCH_DIGEST_DIR / (
        f"{main.sanitize_filename(item.title)} [{item.video_id}].zh.md"
    )
    build_chinese_research_digest(
        transcript_path=transcript_path,
        output_path=digest_path,
        metadata=DigestMetadata(
            title=item.title,
            source_name=item.source_name,
            published_text=item.published_text,
            duration_text=item.duration_text,
            webpage_url=item.webpage_url,
        ),
        api_key=api_key,
    )
    return digest_path


def main_cli() -> int:
    args = parse_args()
    library_path = args.library.expanduser().resolve()
    payload = json.loads(library_path.read_text(encoding="utf-8"))
    records = payload.get("items", {})
    configure_live_cache(library_path)

    mismatches: list[tuple[str, dict, main.VideoItem, Path, str]] = []
    for video_id, record in records.items():
        if not isinstance(record, dict):
            continue
        raw_path = str(record.get("transcript_path") or "").strip()
        if not raw_path:
            continue
        transcript_path = Path(raw_path).expanduser()
        if not transcript_path.exists() or transcript_path.name.endswith(".brief.txt"):
            continue
        item = item_from_record(video_id, record)
        conflict, reason = main.cached_transcript_identity_conflict(item, transcript_path)
        if conflict:
            mismatches.append((video_id, record, item, transcript_path, reason))

    print(json.dumps({"audited": len(records), "mismatches": len(mismatches)}, ensure_ascii=False))
    for video_id, _record, item, path, reason in mismatches:
        print(f"MISMATCH\t{video_id}\t{item.title}\t{reason}\t{path}")

    if not args.repair or not mismatches:
        return 1 if mismatches else 0

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = library_path.with_name(f"{library_path.stem}.pre-identity-repair-{stamp}.json")
    shutil.copy2(library_path, backup_path)
    print(f"BACKUP\t{backup_path}")

    app = object.__new__(main.YTAudioDownloaderApp)
    app.events = queue.Queue()
    api_key = load_api_key(library_path.parent / ".env")
    if not api_key:
        api_key = load_api_key(Path(__file__).resolve().parents[1] / ".env")

    repaired = 0
    for video_id, record, item, transcript_path, reason in mismatches:
        quarantined_transcript = main.quarantine_transcript_cache(transcript_path)
        raw_digest = str(record.get("digest_path") or "").strip()
        quarantined_digest = ""
        if raw_digest:
            digest_path = Path(raw_digest).expanduser()
            if digest_path.exists():
                quarantined_digest = str(main.quarantine_transcript_cache(digest_path))

        replacement = app.fetch_official_transcript(item)
        origin = "verified_official" if replacement is not None else ""
        verified = replacement is not None
        if replacement is None and args.brief_on_missing:
            replacement = app.build_episode_brief_transcript(
                item,
                "已隔离题文不符的旧缓存；当前网络无法取得可验证的完整媒体。",
            )
            origin = "episode_brief_fallback" if replacement is not None else ""

        new_digest: Path | None = None
        digest_error = ""
        if replacement is not None and api_key:
            try:
                new_digest = build_digest(item, replacement, api_key)
            except Exception as exc:  # noqa: BLE001
                digest_error = str(exc)

        record.update(
            {
                "transcript_path": str(replacement) if replacement else "",
                "transcript_kind": (
                    "episode_brief_fallback"
                    if replacement is not None and replacement.name.endswith(".brief.txt")
                    else ("transcript" if replacement is not None else "")
                ),
                "transcript_origin": origin or "identity_mismatch_quarantined",
                "transcript_identity_verified": verified,
                "digest_path": str(new_digest) if new_digest else "",
                "identity_repaired_at": datetime.now(timezone.utc).isoformat(),
                "identity_repair_reason": reason,
            }
        )
        repaired += 1
        print(
            "REPAIRED\t"
            f"{video_id}\t{origin or 'quarantined_only'}\t{replacement or ''}\t"
            f"backup_transcript={quarantined_transcript}\tbackup_digest={quarantined_digest}\t"
            f"digest_error={digest_error}"
        )

    atomic_write_json(library_path, payload)
    print(f"SAVED\t{library_path}\t{repaired}")
    while not app.events.empty():
        event = app.events.get_nowait()
        print(f"EVENT\t{event}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main_cli())
