"""Read-only live feed audit. Does not call translation, ASR or write user settings."""
import argparse
import concurrent.futures
import json
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import main


def probe(feed):
    try:
        parser = (main.fetch_podcast_feed_items if isinstance(feed, main.PodcastFeed)
                  else main.fetch_youtube_feed_items if isinstance(feed, main.YouTubeFeed)
                  else main.fetch_xiaoyuzhou_feed_items)
        rows, skipped = parser(feed)
        matches = [item for _, item in rows if main.is_embodied_item(item.title, item.description_text, item.source_name)]
        items = []
        for item in matches[:4]:
            main.update_item_importance_score(item)
            items.append(asdict(item))
        return {"source": feed.name, "url": feed.feed_url, "ok": True, "parsed": len(rows),
                "matched": len(matches), "skipped": skipped, "samples": items}
    except Exception as exc:
        return {"source": feed.name, "url": feed.feed_url, "ok": False,
                "error": type(exc).__name__, "samples": []}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    feeds = (list(main.DEFAULT_EMBODIED_YOUTUBE_FEEDS)
             + list(main.DEFAULT_EMBODIED_PODCAST_FEEDS)
             + list(main.DEFAULT_EMBODIED_XIAOYUZHOU_FEEDS))
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        sources = list(executor.map(probe, feeds))
    payload = {"version": main.APP_VERSION, "checked_at": datetime.now(timezone.utc).isoformat(),
               "sources_ok": sum(source["ok"] for source in sources), "sources_total": len(sources),
               "sources": sources}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in payload.items() if key != "sources"}))
