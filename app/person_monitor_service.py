"""Isolated service layer for the optional person-monitor page.

The host imports no what-he-said runtime code. Person packs and provenance are
copied under ``modules/person_monitor`` and RSS work only runs on user request.
"""

from __future__ import annotations

import hashlib
import html
import json
import os
import re
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit
from xml.etree import ElementTree as ET

import requests

EXPORT_SCHEMA_VERSION = "1.0.0"
MODULE_VERSION = "0.2.0"
PERSON_MONITOR_USER_AGENT = "PodcastRadar-PersonMonitor/0.2.0"
REQUEST_TIMEOUT = (10, 45)
TRACKING_QUERY_KEYS = {
    "fbclid",
    "gclid",
    "mc_cid",
    "mc_eid",
    "ref",
    "source",
}


def module_resources_dir() -> Path:
    frozen_root = getattr(sys, "_MEIPASS", None)
    if frozen_root:
        return Path(frozen_root) / "modules" / "person_monitor"
    return Path(__file__).resolve().parents[1] / "modules" / "person_monitor"


def person_monitor_data_dir() -> Path:
    # This legacy directory is intentionally stable across the app rename.
    return (
        Path.home()
        / "Library"
        / "Application Support"
        / "ResearchPodcastRadar"
        / "person-monitor"
    )


def export_dir() -> Path:
    return person_monitor_data_dir() / "exports"


def load_registry() -> dict[str, Any]:
    payload = json.loads((module_resources_dir() / "people.json").read_text(encoding="utf-8"))
    people = payload.get("people")
    if not isinstance(people, list) or not people:
        raise ValueError("人物监控配置为空")
    return payload


def person_specs() -> list[dict[str, Any]]:
    return [item for item in load_registry()["people"] if isinstance(item, dict)]


def person_spec(slug: str) -> dict[str, Any]:
    for record in person_specs():
        if record.get("slug") == slug:
            return record
    raise KeyError(f"未知人物：{slug}")


def load_person_export(slug: str) -> dict[str, Any]:
    live_path = export_dir() / f"{slug}.json"
    seed_path = module_resources_dir() / "seed_exports" / f"{slug}.json"
    for candidate in (live_path, seed_path):
        if not candidate.exists():
            continue
        try:
            payload = json.loads(candidate.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(payload, dict) and payload.get("schema_version") == EXPORT_SCHEMA_VERSION:
            return payload
    return empty_person_export(person_spec(slug))


def empty_person_export(spec: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": EXPORT_SCHEMA_VERSION,
        "generated_at": datetime.now(UTC).isoformat(),
        "producer": {"name": "person-monitor-embedded", "version": MODULE_VERSION},
        "person": {
            "slug": spec["slug"],
            "canonical_name": spec["canonical_name"],
            "display_name_zh": spec["display_name_zh"],
            "organization": spec.get("organization", ""),
        },
        "source_status": [],
        "items": [],
        "item_count": 0,
    }


def normalize_text(value: str) -> str:
    unescaped = html.unescape(re.sub(r"<[^>]+>", " ", value or ""))
    return " ".join(unescaped.split()).casefold()


def title_identity_matches(title: str, aliases: list[str]) -> list[str]:
    normalized_title = normalize_text(title)
    identity_slot = re.sub(
        r"^(?:\[[^\]]+\]\s*)?(?:#?\d+\s*[–—\-:|]\s*)?",
        "",
        normalized_title,
    )
    separators = (" ", ":", "：", "|", "｜", "–", "—", "-", "'", "’", ",", "，")
    normalized_aliases = [normalize_text(alias) for alias in aliases if normalize_text(alias)]
    return [
        alias
        for alias in normalized_aliases
        if identity_slot == alias
        or any(identity_slot.startswith(alias + mark) for mark in separators)
    ]


def source_identity_matches(
    title: str,
    aliases: list[str],
    source: dict[str, Any],
) -> list[str]:
    """Apply a source-specific discovery gate without confirming the speaker.

    The strict default keeps the original leading-name behavior. Curated feeds
    may opt into an anywhere-in-title match for formats such as
    ``Interview with Dylan Patel``. This only creates a candidate; downstream
    transcript or speaker evidence is still required before attribution.
    """

    normalized_title = normalize_text(title)
    excluded_terms = [
        normalize_text(str(value))
        for value in source.get("excluded_title_terms", [])
        if normalize_text(str(value))
    ]
    if any(term in normalized_title for term in excluded_terms):
        return []

    required_terms = [
        normalize_text(str(value))
        for value in source.get("required_title_terms", [])
        if normalize_text(str(value))
    ]
    if required_terms and not any(term in normalized_title for term in required_terms):
        return []

    strict_matches = title_identity_matches(title, aliases)
    if strict_matches or source.get("identity_gate") != "title_contains_alias":
        return strict_matches

    normalized_aliases = [normalize_text(alias) for alias in aliases if normalize_text(alias)]
    return [alias for alias in normalized_aliases if alias in normalized_title]


def canonicalize_url(value: str) -> str:
    raw = str(value or "").strip()
    if not raw:
        return ""
    try:
        parsed = urlsplit(raw)
    except ValueError:
        return raw
    if not parsed.scheme or not parsed.netloc:
        return raw
    filtered_query = [
        (key, item)
        for key, item in parse_qsl(parsed.query, keep_blank_values=True)
        if not key.casefold().startswith("utm_")
        and key.casefold() not in TRACKING_QUERY_KEYS
    ]
    path = parsed.path.rstrip("/") or "/"
    return urlunsplit(
        (
            parsed.scheme.casefold(),
            parsed.netloc.casefold(),
            path,
            urlencode(filtered_query, doseq=True),
            "",
        )
    )


def item_dedupe_key(item: dict[str, Any]) -> str:
    normalized_title = normalize_text(str(item.get("title") or ""))
    published_date = str(item.get("published_at") or "")[:10]
    if normalized_title and published_date:
        return f"title-date:{normalized_title}:{published_date}"
    canonical_url = canonicalize_url(str(item.get("url") or ""))
    if canonical_url:
        return f"url:{canonical_url}"
    if normalized_title:
        return f"title:{normalized_title}"
    return f"candidate:{item.get('candidate_key') or ''}"


def request_get(url: str, **kwargs: Any) -> requests.Response:
    """Retry one transient network failure without delaying the UI worker."""

    last_error: Exception | None = None
    for attempt in range(2):
        try:
            response = requests.get(url, **kwargs)
            if attempt == 0 and response.status_code in {429, 500, 502, 503, 504}:
                continue
            return response
        except requests.RequestException as exc:
            last_error = exc
            if attempt == 1:
                raise
    if last_error is not None:
        raise last_error
    raise RuntimeError("网络请求未返回结果")


def parse_publication_time(raw: str) -> str | None:
    value = (raw or "").strip()
    if not value:
        return None
    try:
        parsed = parsedate_to_datetime(value)
    except (TypeError, ValueError, OverflowError):
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC).isoformat().replace("+00:00", "Z")


def _local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1].casefold()


def _first_text(element: ET.Element, names: tuple[str, ...]) -> str:
    wanted = {name.casefold() for name in names}
    for child in element.iter():
        if _local_name(child.tag) in wanted and child.text and child.text.strip():
            return child.text.strip()
    return ""


def _entry_link(element: ET.Element) -> str:
    for child in element.iter():
        if _local_name(child.tag) != "link":
            continue
        href = str(child.attrib.get("href") or "").strip()
        if href:
            return href
        if child.text and child.text.strip():
            return child.text.strip()
    return ""


def parse_feed_entries(content: bytes) -> tuple[str, list[dict[str, str]]]:
    root = ET.fromstring(content)
    channel = next((node for node in root.iter() if _local_name(node.tag) == "channel"), root)
    channel_title = _first_text(channel, ("title",)) or "官方 Feed"
    entries = [node for node in root.iter() if _local_name(node.tag) in {"item", "entry"}]
    parsed: list[dict[str, str]] = []
    for entry in entries:
        title = _first_text(entry, ("title",))
        link = _entry_link(entry)
        guid = _first_text(entry, ("guid", "id")) or link
        description = _first_text(entry, ("description", "summary", "content"))
        published = _first_text(entry, ("pubdate", "published", "updated"))
        author = _first_text(entry, ("author", "creator")) or channel_title
        if title and (link or guid):
            parsed.append(
                {
                    "title": title,
                    "link": link or guid,
                    "guid": guid or link,
                    "description": description,
                    "published": published,
                    "author": author,
                }
            )
    return channel_title, parsed


def discover_feed(
    spec: dict[str, Any],
    source: dict[str, Any],
    *,
    max_items: int = 5,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    locator = str(source["locator"])
    response = request_get(
        locator,
        headers={"User-Agent": PERSON_MONITOR_USER_AGENT},
        timeout=REQUEST_TIMEOUT,
    )
    response.raise_for_status()
    channel_title, entries = parse_feed_entries(response.content)
    identity_aliases = [str(value) for value in spec.get("identity_aliases", [])]
    query_aliases = [normalize_text(str(value)) for value in source.get("query_aliases", [])]
    items: list[dict[str, Any]] = []
    related_mentions_filtered = 0
    query_matches = 0

    for entry in entries:
        searchable = normalize_text(
            " ".join(
                [entry.get("title", ""), entry.get("description", ""), entry.get("author", "")]
            )
        )
        if query_aliases and not any(alias in searchable for alias in query_aliases):
            continue
        query_matches += 1
        matched = source_identity_matches(entry["title"], identity_aliases, source)
        if source.get("expected_role") == "guest_or_speaker" and not matched:
            related_mentions_filtered += 1
            continue
        url = entry["link"]
        provider_id = entry["guid"] or url
        candidate_hash = hashlib.sha256(provider_id.encode("utf-8")).hexdigest()[:24]
        items.append(
            {
                "schema_version": "0.3.0",
                "candidate_key": f"{source['key']}:{candidate_hash}",
                "source_key": source["key"],
                "provider_item_id": provider_id,
                "url": url,
                "title": entry["title"],
                "author_or_channel": entry.get("author") or channel_title,
                "published_at": parse_publication_time(entry.get("published", "")),
                "monitoring_classification": {
                    "person_slug": spec["slug"],
                    "status": "direct_expression_candidate",
                    "identity_match_basis": "title_strong_alias",
                    "matched_aliases": matched,
                    "discovery_tier": "curated_feed",
                    "speaker_or_author_confirmation_required": True,
                },
            }
        )
        if len(items) >= max_items:
            break

    status = {
        "source_key": source["key"],
        "source_name": source.get("name", source["key"]),
        "kind": source.get("kind", "podcast_rss"),
        "status": "succeeded",
        "item_count": len(items),
        "related_mentions_filtered": related_mentions_filtered,
        "diagnostics": {
            "feed_entries": len(entries),
            "query_matches": query_matches,
            "selection_mode": "identity_gate",
        },
    }
    return items, status


def discover_itunes_episodes(
    spec: dict[str, Any],
    source: dict[str, Any],
    *,
    max_items: int = 15,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    endpoint = str(source.get("locator") or "https://itunes.apple.com/search")
    identity_aliases = [str(value) for value in spec.get("identity_aliases", [])]
    trusted_collections = [
        normalize_text(str(value))
        for value in source.get("trusted_collections", [])
        if normalize_text(str(value))
    ]
    blocked_collections = [
        normalize_text(str(value))
        for value in source.get("blocked_collections", [])
        if normalize_text(str(value))
    ]
    raw_results: list[dict[str, Any]] = []
    request_count = 0

    for query in source.get("queries", []):
        response = request_get(
            endpoint,
            params={
                "term": str(query),
                "media": "podcast",
                "entity": "podcastEpisode",
                "limit": int(source.get("result_limit", 100)),
                "country": str(source.get("country", "US")),
            },
            headers={"User-Agent": PERSON_MONITOR_USER_AGENT},
            timeout=REQUEST_TIMEOUT,
        )
        response.raise_for_status()
        payload = response.json()
        request_count += 1
        results = payload.get("results", []) if isinstance(payload, dict) else []
        raw_results.extend(item for item in results if isinstance(item, dict))

    items: list[dict[str, Any]] = []
    query_matches = 0
    related_mentions_filtered = 0
    seen_provider_ids: set[str] = set()

    for entry in raw_results:
        title = str(entry.get("trackName") or "").strip()
        collection = str(entry.get("collectionName") or "").strip()
        normalized_collection = normalize_text(collection)
        provider_id = str(
            entry.get("trackId")
            or entry.get("episodeGuid")
            or entry.get("trackViewUrl")
            or entry.get("episodeUrl")
            or ""
        )
        if not title or not provider_id or provider_id in seen_provider_ids:
            continue
        seen_provider_ids.add(provider_id)

        if blocked_collections and any(
            pattern in normalized_collection for pattern in blocked_collections
        ):
            related_mentions_filtered += 1
            continue
        if trusted_collections and not any(
            pattern in normalized_collection for pattern in trusted_collections
        ):
            related_mentions_filtered += 1
            continue

        matched = source_identity_matches(title, identity_aliases, source)
        if not matched:
            related_mentions_filtered += 1
            continue
        query_matches += 1

        url = str(
            entry.get("trackViewUrl")
            or entry.get("episodeUrl")
            or entry.get("collectionViewUrl")
            or ""
        ).strip()
        if not url:
            continue
        candidate_hash = hashlib.sha256(provider_id.encode("utf-8")).hexdigest()[:24]
        items.append(
            {
                "schema_version": "0.3.0",
                "candidate_key": f"{source['key']}:{candidate_hash}",
                "source_key": source["key"],
                "provider_item_id": provider_id,
                "url": url,
                "title": title,
                "author_or_channel": collection
                or str(entry.get("artistName") or "Apple Podcasts"),
                "published_at": parse_publication_time(
                    str(entry.get("releaseDate") or "")
                ),
                "monitoring_classification": {
                    "person_slug": spec["slug"],
                    "status": "direct_expression_candidate",
                    "identity_match_basis": "itunes_title_alias",
                    "matched_aliases": matched,
                    "discovery_tier": "directory_candidate",
                    "speaker_or_author_confirmation_required": True,
                },
            }
        )
        if len(items) >= max_items:
            break

    status = {
        "source_key": source["key"],
        "source_name": source.get("name", source["key"]),
        "kind": "itunes_episode_search",
        "status": "succeeded",
        "item_count": len(items),
        "related_mentions_filtered": related_mentions_filtered,
        "diagnostics": {
            "requests": request_count,
            "search_results": len(raw_results),
            "query_matches": query_matches,
            "selection_mode": "trusted_directory_identity_gate",
        },
    }
    return items, status


def discover_source(
    spec: dict[str, Any],
    source: dict[str, Any],
    *,
    max_items: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    kind = str(source.get("kind") or "podcast_rss")
    if kind == "itunes_episode_search":
        return discover_itunes_episodes(spec, source, max_items=max_items)
    return discover_feed(spec, source, max_items=max_items)


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temp_path = Path(temp_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, path)
    finally:
        temp_path.unlink(missing_ok=True)


def refresh_person(
    slug: str,
    *,
    max_items: int = 40,
    max_items_per_source: int = 12,
) -> dict[str, Any]:
    spec = person_spec(slug)
    previous = load_person_export(slug)
    all_items: list[dict[str, Any]] = []
    configured_sources = [
        source
        for source in [*spec.get("feeds", []), *spec.get("searches", [])]
        if isinstance(source, dict) and source.get("enabled", True)
    ]
    indexed_results: dict[int, tuple[list[dict[str, Any]], dict[str, Any]]] = {}

    def run_source(
        index: int,
        source: dict[str, Any],
    ) -> tuple[int, list[dict[str, Any]], dict[str, Any]]:
        try:
            items, status = discover_source(
                spec,
                source,
                max_items=max_items_per_source,
            )
        except Exception as exc:  # noqa: BLE001 - failures stay inside module boundary
            items = []
            status = {
                "source_key": source.get("key", "unknown"),
                "source_name": source.get("name", "未知来源"),
                "kind": source.get("kind", "podcast_rss"),
                "status": "failed",
                "error_type": type(exc).__name__,
                "error": str(exc) or type(exc).__name__,
            }
        return index, items, status

    if configured_sources:
        worker_count = min(8, len(configured_sources))
        with ThreadPoolExecutor(
            max_workers=worker_count,
            thread_name_prefix=f"person-monitor-{slug}",
        ) as executor:
            futures = [
                executor.submit(run_source, index, source)
                for index, source in enumerate(configured_sources)
            ]
            for future in as_completed(futures):
                index, items, status = future.result()
                indexed_results[index] = (items, status)

    statuses: list[dict[str, Any]] = []
    for index in range(len(configured_sources)):
        items, status = indexed_results[index]
        all_items.extend(items)
        statuses.append(status)

    successful = any(item.get("status") == "succeeded" for item in statuses)
    stale = not successful and bool(previous.get("items"))
    previous_items = [
        item for item in previous.get("items", []) if isinstance(item, dict)
    ]
    current_item_count = len(all_items)
    all_items.extend(previous_items)

    unique_items: dict[str, dict[str, Any]] = {}
    for item in all_items:
        unique_items.setdefault(item_dedupe_key(item), item)
    ordered_items = sorted(
        unique_items.values(),
        key=lambda item: (
            item.get("monitoring_classification", {}).get("discovery_tier")
            == "curated_feed",
            str(item.get("published_at") or ""),
        ),
        reverse=True,
    )[:max_items]

    payload = empty_person_export(spec)
    payload.update(
        {
            "generated_at": datetime.now(UTC).isoformat(),
            "source_status": statuses,
            "items": ordered_items,
            "item_count": len(ordered_items),
            "stale": stale,
            "refresh_stats": {
                "current_candidates": current_item_count,
                "retained_history": len(previous_items),
                "deduplicated_candidates": len(unique_items),
                "display_limit": max_items,
            },
        }
    )
    atomic_write_json(export_dir() / f"{slug}.json", payload)
    return payload
