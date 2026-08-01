"""Tkinter page adapter for the optional person-monitor module."""

from __future__ import annotations

import queue
import subprocess
import threading
import webbrowser
from datetime import UTC, datetime, timedelta
from typing import Any
import tkinter as tk
from tkinter import ttk

import requests

from app.person_monitor_service import (
    export_dir,
    load_person_export,
    person_specs,
    refresh_person,
)

COLOR_BG = "#f5f5f7"
COLOR_SURFACE = "#ffffff"
COLOR_SURFACE_ALT = "#f2f2f7"
COLOR_TEXT = "#1d1d1f"
COLOR_MUTED = "#6e6e73"
COLOR_BORDER = "#d2d2d7"
COLOR_CARD_SHADOW = "#e4e7ec"
COLOR_BLUE = "#0071e3"
COLOR_BLUE_DARK = "#005bb5"
COLOR_BLUE_SOFT = "#e8f1ff"
COLOR_GREEN = "#34c759"
COLOR_GREEN_DARK = "#188038"
COLOR_GREEN_SOFT = "#e9f8ee"
COLOR_ORANGE = "#ff9500"
COLOR_ORANGE_DARK = "#a85d00"
COLOR_ORANGE_SOFT = "#fff4df"
COLOR_RED = "#ff3b30"
FONT_UI = "SF Pro Text"
FONT_DISPLAY = "SF Pro Display"


def create_rounded_rect(
    canvas: tk.Canvas,
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    radius: int,
    **kwargs: Any,
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


class PersonMonitorPage:
    """Owns its UI, disk reads and refresh worker without touching host state."""

    def __init__(self, host: Any, parent: tk.Frame) -> None:
        self.host = host
        self.root: tk.Tk = host.root
        self.parent = parent
        self.people = {record["slug"]: record for record in person_specs()}
        self.selected_slug = "elon-musk" if "elon-musk" in self.people else next(iter(self.people))
        self.person_buttons: dict[str, tk.Canvas] = {}
        self.refresh_queue: queue.Queue[tuple[str, Any]] = queue.Queue()
        self.is_refreshing = False
        self.artwork_images: dict[str, tk.PhotoImage] = {}
        self.artwork_canvases: dict[str, tuple[tk.Canvas, str]] = {}
        self.artwork_loading: set[str] = set()
        self.artwork_waiters: dict[str, set[str]] = {}
        self.artwork_download_queue: queue.Queue[str] = queue.Queue()
        self.artwork_result_queue: queue.Queue[str] = queue.Queue()
        self.artwork_poll_scheduled = False
        for worker_index in range(4):
            threading.Thread(
                target=self._artwork_worker,
                name=f"person-artwork-{worker_index + 1}",
                daemon=True,
            ).start()

        self.person_name_var = tk.StringVar(master=self.root, value="人物监控")
        self.person_org_var = tk.StringVar(master=self.root, value="")
        self.person_topics_var = tk.StringVar(master=self.root, value="")
        self.item_count_var = tk.StringVar(master=self.root, value="0")
        self.source_count_var = tk.StringVar(master=self.root, value="0/0")
        self.filtered_count_var = tk.StringVar(master=self.root, value="0")
        self.updated_var = tk.StringVar(master=self.root, value="—")
        self.source_status_var = tk.StringVar(master=self.root, value="等待载入")
        self.notice_var = tk.StringVar(
            master=self.root,
            value="优质固定源 + 已核验公开访谈档案",
        )

        self._build()

    def _button(
        self,
        parent: tk.Widget,
        text: str,
        command: Any,
        *,
        primary: bool = False,
        width: int | None = None,
    ) -> tk.Canvas:
        del width
        return self.host.ui_button(
            parent,
            text,
            command,
            variant="primary" if primary else "secondary",
            strong=primary,
            padx=13,
            pady=5,
        )

    def _rounded_panel(
        self,
        parent: tk.Widget,
        *,
        height: int,
        fill: str = COLOR_SURFACE,
        border: str = COLOR_BORDER,
        radius: int = 16,
        padx: int = 16,
        pady: int = 12,
        shadow: bool = True,
    ) -> tuple[tk.Canvas, tk.Frame]:
        parent_bg = str(parent.cget("bg"))
        canvas = tk.Canvas(
            parent,
            bg=parent_bg,
            height=height,
            highlightthickness=0,
            bd=0,
        )
        inner = tk.Frame(canvas, bg=fill, padx=padx, pady=pady)
        inner_window = canvas.create_window(10, 7, window=inner, anchor="nw")

        def resize_panel(event: tk.Event) -> None:
            width = max(80, event.width)
            panel_height = max(height, event.height)
            canvas.delete("panel_shadow")
            canvas.delete("panel_shape")
            if shadow:
                create_rounded_rect(
                    canvas,
                    4,
                    4,
                    width - 2,
                    panel_height - 1,
                    radius,
                    fill=COLOR_CARD_SHADOW,
                    outline=COLOR_CARD_SHADOW,
                    tags=("panel_shadow",),
                )
            create_rounded_rect(
                canvas,
                1,
                1,
                width - 4,
                panel_height - 4,
                radius,
                fill=fill,
                outline=border,
                width=1,
                tags=("panel_shape",),
            )
            canvas.tag_lower("panel_shape")
            canvas.tag_lower("panel_shadow")
            canvas.coords(inner_window, 10, 7)
            canvas.itemconfigure(
                inner_window,
                width=max(1, width - 22),
                height=max(1, panel_height - 16),
            )

        canvas.bind("<Configure>", resize_panel)
        return canvas, inner

    def _build(self) -> None:
        self.parent.configure(bg=COLOR_BG)

        header = tk.Frame(self.parent, bg=COLOR_BG, height=52)
        header.pack(fill="x")
        header.pack_propagate(False)
        tk.Label(
            header,
            text="人物监控",
            fg=COLOR_TEXT,
            bg=COLOR_BG,
            font=(FONT_DISPLAY, 22, "bold"),
        ).pack(side="left", padx=(32, 12))
        tk.Label(
            header,
            textvariable=self.person_name_var,
            fg=COLOR_MUTED,
            bg=COLOR_BG,
            font=(FONT_UI, 10),
        ).pack(side="left")

        tabs = tk.Frame(header, bg=COLOR_BG)
        tabs.pack(side="right", padx=(0, 26), pady=9)
        for slug, record in self.people.items():
            button = self._button(
                tabs,
                str(record.get("canonical_name") or slug),
                lambda target=slug: self.select_person(target),
            )
            button.pack(side="left", padx=(5, 0))
            self.person_buttons[slug] = button

        body = tk.Frame(self.parent, bg=COLOR_BG, padx=32, pady=8)
        body.pack(fill="both", expand=True)

        controls = tk.Frame(body, bg=COLOR_BG)
        controls.pack(fill="x", pady=(0, 7))
        self.refresh_button = self._button(
            controls,
            "刷新当前人物",
            self.refresh_selected,
            primary=True,
        )
        self.refresh_button.pack(side="left", padx=(0, 7))
        self._button(controls, "打开数据目录", self.open_data_directory).pack(side="left")
        tk.Label(
            controls,
            textvariable=self.updated_var,
            fg=COLOR_MUTED,
            bg=COLOR_BG,
            font=(FONT_UI, 9),
        ).pack(side="right", padx=(8, 0))
        tk.Label(
            controls,
            text="最近刷新",
            fg=COLOR_MUTED,
            bg=COLOR_BG,
            font=(FONT_UI, 9),
        ).pack(side="right")

        self.summary_label = tk.Label(
            body,
            textvariable=self.source_status_var,
            fg=COLOR_BLUE,
            bg=COLOR_BG,
            anchor="w",
            font=(FONT_UI, 10, "bold"),
        )
        self.summary_label.pack(fill="x", pady=(0, 4))

        list_shell = tk.Frame(body, bg=COLOR_BG)
        list_shell.pack(fill="both", expand=True)
        self.items_canvas = tk.Canvas(
            list_shell,
            bg=COLOR_BG,
            highlightthickness=0,
            bd=0,
        )
        scrollbar = ttk.Scrollbar(list_shell, orient="vertical", command=self.items_canvas.yview)
        self.items_canvas.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side="right", fill="y")
        self.items_canvas.pack(side="left", fill="both", expand=True)

        self.items_frame = tk.Frame(self.items_canvas, bg=COLOR_BG)
        self.items_window = self.items_canvas.create_window(
            (0, 0),
            window=self.items_frame,
            anchor="nw",
        )
        self.items_frame.bind(
            "<Configure>",
            lambda _event: self.items_canvas.configure(scrollregion=self.items_canvas.bbox("all")),
        )
        self.items_canvas.bind(
            "<Configure>",
            lambda event: self.items_canvas.itemconfigure(self.items_window, width=event.width),
        )

    def on_show(self) -> None:
        payload = load_person_export(self.selected_slug)
        self._render(payload)
        if self._needs_refresh(payload):
            self.root.after(180, self.refresh_selected)

    def select_person(self, slug: str) -> None:
        if slug not in self.people or self.is_refreshing:
            return
        self.selected_slug = slug
        self.reload()

    def reload(self) -> None:
        payload = load_person_export(self.selected_slug)
        self._render(payload)

    def _render(self, payload: dict[str, Any]) -> None:
        person = payload.get("person", {})
        statuses = payload.get("source_status", [])
        items = payload.get("items", [])
        successful = sum(1 for item in statuses if item.get("status") == "succeeded")
        filtered = sum(int(item.get("related_mentions_filtered") or 0) for item in statuses)

        selected_spec = self.people.get(self.selected_slug, {})
        self.person_name_var.set(
            str(
                person.get("display_name_zh")
                or selected_spec.get("display_name_zh")
                or person.get("canonical_name")
                or self.selected_slug
            )
        )
        self.person_org_var.set(str(person.get("organization") or ""))
        self.person_topics_var.set(
            "关注方向 · "
            + "  /  ".join(str(value) for value in selected_spec.get("topics", [])[:5])
        )
        self.item_count_var.set(str(payload.get("item_count") or 0))
        self.source_count_var.set(f"{successful}/{len(statuses)}")
        self.filtered_count_var.set(str(filtered))
        self.updated_var.set(self._format_update_time(str(payload.get("generated_at") or "")))

        if payload.get("stale"):
            self.source_status_var.set(
                f"来源：刷新失败，保留上次成功结果｜收录 {len(items)} 条｜排除误报 {filtered} 条"
            )
        elif successful:
            archive_count = sum(
                1
                for item in items
                if item.get("monitoring_classification", {}).get("discovery_tier")
                == "verified_archive"
            )
            self.source_status_var.set(
                f"来源：{successful}/{len(statuses)} 个信源读取成功｜"
                f"收录 {len(items)} 条｜已核验档案 {archive_count} 条｜排除误报 {filtered} 条"
            )
        else:
            self.source_status_var.set(
                f"来源：当前显示内置快照｜收录 {len(items)} 条｜等待实时刷新"
            )

        for slug, button in self.person_buttons.items():
            selected = slug == self.selected_slug
            label = str(self.people.get(slug, {}).get("canonical_name") or slug)
            self.host.configure_ui_button(
                button,
                label,
                "accent" if selected else "secondary",
            )

        for child in self.items_frame.winfo_children():
            child.destroy()
        self.artwork_images = {}
        self.artwork_canvases = {}

        heading = tk.Frame(self.items_frame, bg=COLOR_BG)
        heading.pack(fill="x", pady=(0, 9))
        tk.Label(
            heading,
            text=f"{self.person_name_var.get()} · 最新访谈",
            fg=COLOR_TEXT,
            bg=COLOR_BG,
            font=(FONT_DISPLAY, 16, "bold"),
        ).pack(side="left")
        tk.Label(
            heading,
            textvariable=self.person_topics_var,
            fg=COLOR_MUTED,
            bg=COLOR_BG,
            font=(FONT_UI, 10),
        ).pack(side="left", padx=(10, 0), pady=(3, 0))
        tk.Label(
            heading,
            text=f"{len(items)} 条",
            fg=COLOR_MUTED,
            bg=COLOR_BG,
            font=(FONT_UI, 10, "bold"),
        ).pack(side="right", padx=(12, 20))

        if not items:
            empty_shell, empty = self._rounded_panel(
                self.items_frame,
                height=146,
                radius=18,
                padx=24,
                pady=30,
            )
            empty_shell.pack(fill="x", expand=True)
            tk.Label(
                empty,
                text="本轮没有高置信度候选",
                fg=COLOR_TEXT,
                bg=COLOR_SURFACE,
                font=(FONT_DISPLAY, 16, "bold"),
            ).pack()
            tk.Label(
                empty,
                text="没有通过姓名与来源门禁时保持为空，不把第三方提及归到本人名下。",
                fg=COLOR_MUTED,
                bg=COLOR_SURFACE,
                font=(FONT_UI, 10),
            ).pack(pady=(8, 0))
            return

        for item in items:
            self._render_item(item)

    def _render_item(self, item: dict[str, Any]) -> None:
        classification = item.get("monitoring_classification", {})
        tier = str(classification.get("discovery_tier") or "")
        if tier == "verified_archive":
            badge_text = "档案已核验"
            badge_fg = COLOR_GREEN_DARK
            badge_bg = COLOR_GREEN_SOFT
            summary_text = "公开访谈档案已核验，保留节目原始来源"
            status_text = "✓ 本人访谈"
        elif tier == "curated_feed":
            badge_text = "优质固定源"
            badge_fg = COLOR_BLUE
            badge_bg = COLOR_BLUE_SOFT
            summary_text = "来自人物监控优质固定信源"
            status_text = "固定源"
        else:
            badge_text = "跨节目发现"
            badge_fg = COLOR_ORANGE_DARK
            badge_bg = COLOR_ORANGE_SOFT
            summary_text = "跨节目目录发现，打开原文后继续核验"
            status_text = "待核验"

        card_shell, card = self._rounded_panel(
            self.items_frame,
            height=114,
            radius=16,
            padx=9,
            pady=8,
        )
        card_shell.pack(fill="x", pady=(0, 8))

        url = str(item.get("url") or "")
        actions = tk.Frame(card, bg=COLOR_SURFACE, width=126, height=92)
        actions.pack(side="right", fill="y", padx=(12, 0))
        actions.pack_propagate(False)
        tk.Label(
            actions,
            text=badge_text,
            fg=badge_fg,
            bg=COLOR_SURFACE,
            justify="right",
            anchor="e",
            wraplength=126,
            font=(FONT_UI, 9, "bold"),
        ).pack(fill="x", pady=(1, 2))
        tk.Label(
            actions,
            text=status_text,
            fg=COLOR_MUTED,
            bg=COLOR_SURFACE,
            justify="right",
            anchor="e",
            font=(FONT_UI, 9),
        ).pack(fill="x")
        self._button(
            actions,
            "查看档案" if tier == "verified_archive" else "打开原文",
            lambda target=url: webbrowser.open(target),
            primary=True,
        ).pack(anchor="e", pady=(4, 0))

        avatar = tk.Canvas(
            card,
            width=72,
            height=72,
            bg=COLOR_SURFACE,
            highlightthickness=0,
            bd=0,
        )
        avatar.pack(side="left", anchor="n", padx=(0, 12))
        candidate_key = str(item.get("candidate_key") or item.get("url") or id(item))
        artwork_url = str(item.get("artwork_url") or "").strip()
        self.artwork_canvases[candidate_key] = (avatar, artwork_url)
        if artwork_url:
            photo = self.host.load_artwork_photo(artwork_url, 72)
        else:
            photo = None
        if photo is not None:
            avatar.create_image(0, 0, anchor="nw", image=photo)
            self.artwork_images[candidate_key] = photo
        else:
            self._draw_person_placeholder(avatar, badge_bg, badge_fg)
            if artwork_url:
                self._schedule_artwork(candidate_key, artwork_url)

        content = tk.Frame(card, bg=COLOR_SURFACE)
        content.pack(side="left", fill="both", expand=True)
        tk.Label(
            content,
            text=str(item.get("title") or "未命名内容"),
            fg=COLOR_TEXT,
            bg=COLOR_SURFACE,
            anchor="w",
            justify="left",
            font=(FONT_UI, 12, "bold"),
            wraplength=560,
            height=1,
        ).pack(fill="x")
        tk.Label(
            content,
            text=summary_text,
            fg=COLOR_MUTED,
            bg=COLOR_SURFACE,
            anchor="w",
            justify="left",
            font=(FONT_UI, 10),
            wraplength=560,
            height=1,
        ).pack(fill="x", pady=(3, 0))
        tk.Label(
            content,
            text=str(item.get("author_or_channel") or item.get("source_key") or ""),
            fg=COLOR_MUTED,
            bg=COLOR_SURFACE,
            anchor="w",
            font=(FONT_UI, 9),
        ).pack(fill="x", pady=(3, 0))
        tk.Label(
            content,
            text=f"发布时间：{self._format_publication_date(str(item.get('published_at') or ''))}",
            fg=COLOR_BLUE,
            bg=COLOR_SURFACE,
            anchor="w",
            font=(FONT_UI, 9, "bold"),
        ).pack(fill="x", pady=(2, 0))

    def _draw_person_placeholder(
        self,
        avatar: tk.Canvas,
        badge_bg: str,
        badge_fg: str,
    ) -> None:
        create_rounded_rect(
            avatar,
            2,
            2,
            70,
            70,
            14,
            fill=badge_bg,
            outline=badge_fg,
            width=1,
        )
        avatar.create_text(
            36,
            31,
            text=self._person_initials(),
            fill=badge_fg,
            font=(FONT_DISPLAY, 15, "bold"),
        )
        avatar.create_text(
            36,
            53,
            text="PODCAST",
            fill=badge_fg,
            font=(FONT_UI, 7, "bold"),
        )

    def _schedule_artwork(self, candidate_key: str, artwork_url: str) -> None:
        self.artwork_waiters.setdefault(artwork_url, set()).add(candidate_key)
        if artwork_url not in self.artwork_loading:
            self.artwork_loading.add(artwork_url)
            self.artwork_download_queue.put(artwork_url)
        if not self.artwork_poll_scheduled:
            self.artwork_poll_scheduled = True
            self.root.after(120, self._poll_artwork_results)

    def _artwork_worker(self) -> None:
        while True:
            artwork_url = self.artwork_download_queue.get()
            try:
                cache_path = self.host.artwork_cache_path(artwork_url)
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                if cache_path.exists():
                    cache_path.unlink(missing_ok=True)
                response = requests.get(
                    artwork_url,
                    timeout=12,
                    headers={"User-Agent": "Mozilla/5.0 PodcastRadar/0.4.33"},
                )
                response.raise_for_status()
                content_type = str(response.headers.get("Content-Type") or "").casefold()
                if content_type and not content_type.startswith("image/"):
                    raise ValueError(f"封面返回了非图片类型：{content_type}")
                if len(response.content) < 256:
                    raise ValueError("封面图片内容过小")
                partial_path = cache_path.with_name(
                    f"{cache_path.name}.{threading.get_ident()}.partial"
                )
                partial_path.write_bytes(response.content)
                partial_path.replace(cache_path)
            except Exception:
                pass
            finally:
                self.artwork_result_queue.put(artwork_url)
                self.artwork_download_queue.task_done()

    def _poll_artwork_results(self) -> None:
        while True:
            try:
                artwork_url = self.artwork_result_queue.get_nowait()
            except queue.Empty:
                break
            self.artwork_loading.discard(artwork_url)
            for candidate_key in self.artwork_waiters.pop(artwork_url, set()):
                canvas_record = self.artwork_canvases.get(candidate_key)
                if not canvas_record or canvas_record[1] != artwork_url:
                    continue
                canvas = canvas_record[0]
                if not canvas.winfo_exists():
                    continue
                photo = self.host.load_artwork_photo(artwork_url, 72)
                if photo is None:
                    continue
                canvas.delete("all")
                canvas.create_image(0, 0, anchor="nw", image=photo)
                self.artwork_images[candidate_key] = photo
        if self.artwork_loading:
            self.root.after(120, self._poll_artwork_results)
        else:
            self.artwork_poll_scheduled = False

    def _person_initials(self) -> str:
        name = str(
            self.people.get(self.selected_slug, {}).get("canonical_name")
            or self.selected_slug
        )
        parts = [part for part in name.replace("-", " ").split() if part]
        if len(parts) >= 2:
            return "".join(part[0] for part in parts[:2]).upper()
        return name[:2].upper()

    def refresh_selected(self) -> None:
        if self.is_refreshing:
            return
        self.is_refreshing = True
        slug = self.selected_slug
        self.host.configure_ui_button(self.refresh_button, "刷新中…", "secondary")
        self.source_status_var.set(
            "来源：正在并行读取固定 RSS、公开访谈档案与 Apple Podcasts…"
        )
        worker = threading.Thread(
            target=self._refresh_worker,
            args=(slug,),
            name=f"person-monitor-{slug}",
            daemon=True,
        )
        worker.start()
        self.root.after(120, self._poll_refresh)

    def _refresh_worker(self, slug: str) -> None:
        try:
            payload = refresh_person(slug)
        except Exception as exc:  # noqa: BLE001 - contained inside module boundary
            self.refresh_queue.put(("error", f"{type(exc).__name__}: {exc}"))
            return
        self.refresh_queue.put(("success", payload))

    def _poll_refresh(self) -> None:
        try:
            status, result = self.refresh_queue.get_nowait()
        except queue.Empty:
            if self.is_refreshing:
                self.root.after(120, self._poll_refresh)
            return

        self.is_refreshing = False
        self.host.configure_ui_button(self.refresh_button, "刷新当前人物", "primary")
        if status == "success":
            self._render(result)
            self.notice_var.set("档案已核验优先；目录新发现继续保留待核验标记")
        else:
            self.source_status_var.set("来源：人物监控刷新失败，主程序其他功能不受影响")
            self.notice_var.set(str(result)[:140])

    def open_data_directory(self) -> None:
        target = export_dir()
        target.mkdir(parents=True, exist_ok=True)
        subprocess.Popen(["open", str(target)])

    @staticmethod
    def _needs_refresh(payload: dict[str, Any]) -> bool:
        if payload.get("seed_snapshot"):
            return True
        raw = str(payload.get("generated_at") or "")
        if not raw:
            return True
        try:
            parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
        except ValueError:
            return True
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=UTC)
        return datetime.now(UTC) - parsed.astimezone(UTC) > timedelta(hours=6)

    @staticmethod
    def _format_update_time(raw: str) -> str:
        if not raw:
            return "—"
        try:
            parsed = datetime.fromisoformat(raw.replace("Z", "+00:00")).astimezone()
        except ValueError:
            return "已载入"
        return parsed.strftime("%H:%M")

    @staticmethod
    def _format_publication_date(raw: str) -> str:
        if not raw:
            return "时间待核验"
        try:
            parsed = datetime.fromisoformat(raw.replace("Z", "+00:00")).astimezone()
        except ValueError:
            return raw[:10]
        return parsed.strftime("%Y-%m-%d")
