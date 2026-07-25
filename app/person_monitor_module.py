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
COLOR_BLUE = "#0071e3"
COLOR_BLUE_DARK = "#005bb5"
COLOR_GREEN = "#34c759"
COLOR_ORANGE = "#ff9500"
COLOR_RED = "#ff3b30"
FONT_UI = "SF Pro Text"
FONT_DISPLAY = "SF Pro Display"


class PersonMonitorPage:
    """Owns its UI, disk reads and refresh worker without touching host state."""

    def __init__(self, host: Any, parent: tk.Frame) -> None:
        self.host = host
        self.root: tk.Tk = host.root
        self.parent = parent
        self.people = {record["slug"]: record for record in person_specs()}
        self.selected_slug = "elon-musk" if "elon-musk" in self.people else next(iter(self.people))
        self.person_buttons: dict[str, tk.Button] = {}
        self.refresh_queue: queue.Queue[tuple[str, Any]] = queue.Queue()
        self.is_refreshing = False

        self.person_name_var = tk.StringVar(master=self.root, value="人物监控")
        self.person_org_var = tk.StringVar(master=self.root, value="")
        self.item_count_var = tk.StringVar(master=self.root, value="0")
        self.source_count_var = tk.StringVar(master=self.root, value="0/0")
        self.filtered_count_var = tk.StringVar(master=self.root, value="0")
        self.updated_var = tk.StringVar(master=self.root, value="—")
        self.source_status_var = tk.StringVar(master=self.root, value="等待载入")
        self.notice_var = tk.StringVar(master=self.root, value="按需读取，不影响节目下载和转录任务")

        self._build()

    def _button(
        self,
        parent: tk.Widget,
        text: str,
        command: Any,
        *,
        primary: bool = False,
        width: int | None = None,
    ) -> tk.Button:
        background = COLOR_BLUE if primary else COLOR_SURFACE_ALT
        foreground = "#ffffff" if primary else COLOR_TEXT
        active_background = COLOR_BLUE_DARK if primary else "#e5e5ea"
        button = tk.Button(
            parent,
            text=text,
            command=command,
            bg=background,
            fg=foreground,
            activebackground=active_background,
            activeforeground=foreground,
            relief="flat",
            bd=0,
            highlightthickness=0,
            padx=14,
            pady=8,
            font=(FONT_UI, 10, "bold"),
            cursor="pointinghand",
            width=width,
        )
        return button

    def _build(self) -> None:
        self.parent.configure(bg=COLOR_BG)

        header = tk.Frame(self.parent, bg=COLOR_BG, padx=26, pady=20)
        header.pack(fill="x")
        heading = tk.Frame(header, bg=COLOR_BG)
        heading.pack(side="left", fill="x", expand=True)
        tk.Label(
            heading,
            textvariable=self.person_name_var,
            fg=COLOR_TEXT,
            bg=COLOR_BG,
            font=(FONT_DISPLAY, 24, "bold"),
        ).pack(anchor="w")
        tk.Label(
            heading,
            textvariable=self.person_org_var,
            fg=COLOR_MUTED,
            bg=COLOR_BG,
            font=(FONT_UI, 10),
        ).pack(anchor="w", pady=(4, 0))

        self.refresh_button = self._button(
            header,
            "刷新当前人物",
            self.refresh_selected,
            primary=True,
        )
        self.refresh_button.pack(side="right", padx=(10, 0))
        self._button(header, "打开数据目录", self.open_data_directory).pack(side="right")

        body = tk.Frame(self.parent, bg=COLOR_BG, padx=26)
        body.pack(fill="both", expand=True)

        tabs = tk.Frame(body, bg=COLOR_BG)
        tabs.pack(fill="x", pady=(0, 14))
        for slug, record in self.people.items():
            button = self._button(
                tabs,
                str(record.get("canonical_name") or slug),
                lambda target=slug: self.select_person(target),
            )
            button.pack(side="left", padx=(0, 8))
            self.person_buttons[slug] = button

        metrics = tk.Frame(body, bg=COLOR_BG)
        metrics.pack(fill="x", pady=(0, 14))
        metric_specs = [
            ("候选条目", self.item_count_var, "姓名与来源门禁通过", COLOR_BLUE),
            ("来源状态", self.source_count_var, "本轮读取成功", COLOR_GREEN),
            ("过滤提及", self.filtered_count_var, "第三方标题与描述", COLOR_ORANGE),
            ("最近更新", self.updated_var, "本地生成时间", COLOR_TEXT),
        ]
        for index, (label, variable, detail, value_color) in enumerate(metric_specs):
            card = tk.Frame(
                metrics,
                bg=COLOR_SURFACE,
                padx=16,
                pady=13,
                highlightthickness=1,
                highlightbackground=COLOR_BORDER,
            )
            card.grid(row=0, column=index, sticky="nsew", padx=(0, 10 if index < 3 else 0))
            metrics.columnconfigure(index, weight=1, uniform="metric")
            tk.Label(
                card,
                text=label,
                fg=COLOR_MUTED,
                bg=COLOR_SURFACE,
                font=(FONT_UI, 9, "bold"),
            ).pack(anchor="w")
            tk.Label(
                card,
                textvariable=variable,
                fg=value_color,
                bg=COLOR_SURFACE,
                font=(FONT_DISPLAY, 22, "bold"),
            ).pack(anchor="w", pady=(6, 2))
            tk.Label(
                card,
                text=detail,
                fg=COLOR_MUTED,
                bg=COLOR_SURFACE,
                font=(FONT_UI, 9),
            ).pack(anchor="w")

        status_bar = tk.Frame(
            body,
            bg=COLOR_SURFACE,
            padx=14,
            pady=10,
            highlightthickness=1,
            highlightbackground=COLOR_BORDER,
        )
        status_bar.pack(fill="x", pady=(0, 14))
        tk.Label(
            status_bar,
            text="●",
            fg=COLOR_GREEN,
            bg=COLOR_SURFACE,
            font=(FONT_UI, 9),
        ).pack(side="left")
        tk.Label(
            status_bar,
            textvariable=self.source_status_var,
            fg=COLOR_TEXT,
            bg=COLOR_SURFACE,
            font=(FONT_UI, 10, "bold"),
        ).pack(side="left", padx=(7, 0))
        tk.Label(
            status_bar,
            textvariable=self.notice_var,
            fg=COLOR_MUTED,
            bg=COLOR_SURFACE,
            font=(FONT_UI, 9),
        ).pack(side="right")

        list_shell = tk.Frame(body, bg=COLOR_BG)
        list_shell.pack(fill="both", expand=True, pady=(0, 20))
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

        self.person_name_var.set(str(person.get("canonical_name") or self.selected_slug))
        self.person_org_var.set(str(person.get("organization") or ""))
        self.item_count_var.set(str(payload.get("item_count") or 0))
        self.source_count_var.set(f"{successful}/{len(statuses)}")
        self.filtered_count_var.set(str(filtered))
        self.updated_var.set(self._format_update_time(str(payload.get("generated_at") or "")))

        if payload.get("stale"):
            self.source_status_var.set("来源刷新失败，保留上次成功结果")
        elif successful:
            self.source_status_var.set(f"{successful} 个来源读取成功 · 结果已隔离保存")
        else:
            self.source_status_var.set("尚未执行实时刷新，当前显示内置快照")

        for slug, button in self.person_buttons.items():
            selected = slug == self.selected_slug
            button.configure(
                bg="#e8f1ff" if selected else COLOR_SURFACE_ALT,
                fg=COLOR_BLUE if selected else COLOR_TEXT,
                activebackground="#d8eaff" if selected else "#e5e5ea",
            )

        for child in self.items_frame.winfo_children():
            child.destroy()

        heading = tk.Frame(self.items_frame, bg=COLOR_BG)
        heading.pack(fill="x", pady=(0, 9))
        tk.Label(
            heading,
            text="本人做客与发言候选",
            fg=COLOR_TEXT,
            bg=COLOR_BG,
            font=(FONT_DISPLAY, 15, "bold"),
        ).pack(side="left")
        tk.Label(
            heading,
            text="EVIDENCE FIRST",
            fg=COLOR_BLUE,
            bg="#e8f1ff",
            padx=9,
            pady=4,
            font=(FONT_UI, 8, "bold"),
        ).pack(side="right")

        if not items:
            empty = tk.Frame(
                self.items_frame,
                bg=COLOR_SURFACE,
                padx=24,
                pady=46,
                highlightthickness=1,
                highlightbackground=COLOR_BORDER,
            )
            empty.pack(fill="both", expand=True)
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
        card = tk.Frame(
            self.items_frame,
            bg=COLOR_SURFACE,
            padx=16,
            pady=13,
            highlightthickness=1,
            highlightbackground=COLOR_BORDER,
        )
        card.pack(fill="x", pady=(0, 9))

        badge = tk.Label(
            card,
            text="?",
            fg="#ffffff",
            bg=COLOR_ORANGE,
            width=2,
            font=(FONT_UI, 9, "bold"),
        )
        badge.pack(side="left", anchor="n", padx=(0, 12))

        content = tk.Frame(card, bg=COLOR_SURFACE)
        content.pack(side="left", fill="x", expand=True)
        tk.Label(
            content,
            text=str(item.get("title") or "未命名内容"),
            fg=COLOR_TEXT,
            bg=COLOR_SURFACE,
            anchor="w",
            justify="left",
            font=(FONT_UI, 11, "bold"),
            wraplength=720,
        ).pack(fill="x")
        metadata = " · ".join(
            value
            for value in (
                self._format_publication_date(str(item.get("published_at") or "")),
                str(item.get("author_or_channel") or item.get("source_key") or ""),
                (
                    "跨节目新发现 · 待核验"
                    if item.get("monitoring_classification", {}).get("discovery_tier")
                    == "directory_candidate"
                    else "优质固定源 · 本人发言待核验"
                ),
            )
            if value
        )
        tk.Label(
            content,
            text=metadata,
            fg=COLOR_MUTED,
            bg=COLOR_SURFACE,
            anchor="w",
            font=(FONT_UI, 9),
        ).pack(fill="x", pady=(6, 0))

        url = str(item.get("url") or "")
        self._button(card, "打开原文", lambda target=url: webbrowser.open(target)).pack(
            side="right",
            anchor="n",
            padx=(12, 0),
        )

    def refresh_selected(self) -> None:
        if self.is_refreshing:
            return
        self.is_refreshing = True
        slug = self.selected_slug
        self.refresh_button.configure(text="刷新中…", state="disabled")
        self.source_status_var.set(
            "正在并行读取优质 RSS 与 Apple Podcasts，其他页面可继续使用"
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
        self.refresh_button.configure(text="刷新当前人物", state="normal")
        if status == "success":
            self._render(result)
            self.notice_var.set("优质固定源优先；跨节目新发现保留为待核验候选")
        else:
            self.source_status_var.set("人物监控刷新失败，主程序其他功能不受影响")
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
