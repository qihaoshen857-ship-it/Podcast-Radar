from __future__ import annotations

import json
import time
from pathlib import Path
from urllib.request import urlopen

from websocket import create_connection


AUTH_COOKIE_NAMES = {
    "LOGIN_INFO",
    "SID",
    "HSID",
    "SSID",
    "SAPISID",
    "__Secure-1PSID",
    "__Secure-3PSID",
}


def export_edge_cookies(port: int, output_path: Path) -> Path:
    tabs = _get_json_with_retry(f"http://127.0.0.1:{port}/json")
    page = next(
        (
            tab
            for tab in tabs
            if tab.get("type") == "page"
            and "youtube" in str(tab.get("url", "")).lower()
            and tab.get("webSocketDebuggerUrl")
        ),
        None,
    )
    if page is None:
        page = next((tab for tab in tabs if tab.get("webSocketDebuggerUrl")), None)
    if page is None:
        raise RuntimeError("没有找到可用的 Edge 调试页面")

    cookies = _request_cookies(str(page["webSocketDebuggerUrl"]))
    filtered = [
        cookie
        for cookie in cookies
        if any(domain in str(cookie.get("domain", "")) for domain in ("youtube.com", "google.com", "googlevideo.com"))
    ]

    lines = ["# Netscape HTTP Cookie File", "# Generated from Edge CDP", ""]
    for cookie in filtered:
        domain = str(cookie.get("domain", ""))
        include_subdomains = "TRUE" if domain.startswith(".") else "FALSE"
        path = str(cookie.get("path") or "/")
        secure = "TRUE" if cookie.get("secure") else "FALSE"
        expires = str(max(0, int(cookie.get("expires") or 0)))
        lines.append(
            "\t".join(
                [
                    domain,
                    include_subdomains,
                    path,
                    secure,
                    expires,
                    str(cookie.get("name", "")),
                    str(cookie.get("value", "")),
                ]
            )
        )

    output_path.write_text("\n".join(lines), encoding="utf-8")
    return output_path


def _get_json_with_retry(url: str, retries: int = 12, delay_seconds: float = 0.8) -> list[dict]:
    last_error: Exception | None = None
    for _ in range(retries):
        try:
            with urlopen(url, timeout=3) as response:
                payload = json.loads(response.read().decode("utf-8"))
            if isinstance(payload, list):
                return payload
            raise RuntimeError("Edge 调试接口返回的内容不是页面列表")
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            time.sleep(delay_seconds)
    raise RuntimeError(f"读取 Edge 调试页面失败：{last_error}")


def _request_cookies(websocket_url: str) -> list[dict]:
    websocket = create_connection(websocket_url, timeout=5, suppress_origin=True)
    try:
        for request_id in range(1, 9):
            websocket.send(json.dumps({"id": request_id, "method": "Storage.getCookies", "params": {}}))
            deadline = time.time() + 5
            while time.time() < deadline:
                payload = json.loads(websocket.recv())
                if payload.get("id") != request_id:
                    continue
                cookies = payload.get("result", {}).get("cookies") or []
                if any(cookie.get("name") in AUTH_COOKIE_NAMES for cookie in cookies) or request_id == 8:
                    return list(cookies)
                break
            time.sleep(1)
    finally:
        websocket.close()
    return []
