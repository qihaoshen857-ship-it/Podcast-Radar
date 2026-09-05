"""Shared discovery and evidence-weighted scoring for the robotics radar.

Channel IDs are resolved from the linked official channels, verified 2026-09-05.
Keep this module identical in the Mac and Windows repositories.
"""
from __future__ import annotations

import html
import re
import unicodedata
from datetime import datetime, timezone

TOPIC = "embodied_intelligence"
# name, channel ID, official channel URL. Only actual audio/video enters ASR.
YOUTUBE_SOURCES = (
    ("Figure", "UCYlq-KmwPjc1DtsGmthFqSQ", "https://www.youtube.com/@figureai"),
    ("1X", "UCoHslVexR2q57wUoCRfdUsg", "https://www.youtube.com/@1X-tech"),
    ("Apptronik", "UCo9jdosoSQtDgyTThukoCgg", "https://www.youtube.com/@apptronikofficial"),
    ("Agility Robotics", "UCN-StetwWuVYf-MU2_NVj4A", "https://www.youtube.com/@AgilityRobotics"),
    ("Boston Dynamics", "UC7vVhkEfw4nOGp8TyDk7RcQ", "https://www.youtube.com/@BostonDynamics"),
    ("Unitree Robotics", "UCsMbp4V8oxzHCMdOUP-3oWw", "https://www.youtube.com/@unitreerobotics"),
    ("NVIDIA", "UCHuiy8bXnmK5nisYHUd1J5g", "https://www.youtube.com/@NVIDIA"),
)
OFFICIAL_NAMES = {f"YouTube | {name}" for name, _, _ in YOUTUBE_SOURCES}
DEDICATED_NAMES = OFFICIAL_NAMES - {"YouTube | NVIDIA"}
STRONG = (
    "具身智能", "人形机器人", "仿人机器人", "通用机器人", "灵巧手",
    "embodied intelligence", "embodied ai", "humanoid", "physical ai",
    "vision language action", "vision-language-action", "vla",
    "robot foundation model", "generalist robot", "whole-body control",
    "whole-body mobile manipulation", "dexterous manipulation",
)
ROBOT = ("机器人", "robot", "robotics", "机器人本体")
TECH = (
    "world model", "世界模型", "sim-to-real", "sim to real", "仿真到现实",
    "teleoperation", "遥操作", "强化学习", "reinforcement learning",
    "运动控制", "locomotion", "manipulation", "robot data", "机器人数据",
    "autonomous", "自主", "policy", "端到端", "end-to-end",
)
PARTS = (
    "灵巧手", "关节模组", "执行器", "减速器", "滚柱丝杠", "触觉",
    "力矩传感", "actuator", "dexterous hand", "tactile", "torque sensor",
    "gearbox", "roller screw", "hands",
)
COMMERCIAL = (
    "量产", "产能", "良率", "交付", "订单", "成本", "部署", "生产线", "工厂",
    "production", "capacity", "yield", "delivered", "delivery", "deployed",
    "deployment", "order", "cost", "bom", "factory", "manufacturing", "ready stock",
)
EVIDENCE = ("data", "measured", "units", "hours", "数据", "实测", "已交付", "已部署", "已投产")
PLANNING = ("plan", "target", "aim to", "expect", "will ", "预计", "计划", "目标", "有望", "拟", "试产", "pilot")
ROLES = ("founder", "cofounder", "co-founder", "ceo", "cto", "chief scientist", "创始人", "首席科学家", "技术负责人")
INTERVIEW = ("interview", "podcast", "conversation", "with ", "访谈", "对话", "专访")
DEMO = ("dance", "dancing", "backflip", "tiktok", "combat competition", "跳舞", "后空翻", "格斗", "炫技")
ENTERTAINMENT = DEMO + ("fifa", "world cup", "redecorates", "viking row", "livestream", "superman", "athlete", "digit shuffle", "whirlybird", "new year greetings")
ROUNDUP = ("weekly news", "news roundup", "top stories", "新闻汇总", "一周快讯")
REPOST = ("reupload", "compilation", "转载", "混剪")
AUTO_ONLY = ("robotaxi", "self-driving", "fsd", "自动驾驶", "无人驾驶", "tesla earnings")
MODELS = ("atlas", "digit", "neo", "helix", "f.03", "figure 03", "unitree r1", "unitree g1", "unitree h1", "apollo")


def clean(value: str) -> str:
    value = re.sub(r"<[^>]+>", " ", html.unescape(value or ""))
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", value)).strip().lower()


def episode_focus(description: str) -> str:
    text = clean(description)
    text = re.split(r"(?:connect with|follow us|follow 1x|follow agility|subscribe to|our sponsors|sponsors:|赞助商|关注我们)", text)[0]
    return re.sub(r"https?://\S+", " ", text).strip()


def hits(text: str, words: tuple[str, ...]) -> bool:
    for word in words:
        word = word.strip()
        if re.search(r"[a-z]", word):
            if re.search(r"(?<![a-z0-9])" + re.escape(word) + r"(?:s|ics)?(?![a-z0-9])", text):
                return True
        elif word in text:
            return True
    return False


def is_embodied_item(title: str, description: str = "", source_name: str = "") -> bool:
    title_text = clean(title)
    focus = episode_focus(description)[:900]
    combined = title_text + " " + focus
    if hits(title_text, ENTERTAINMENT) and not hits(title_text, TECH + PARTS + COMMERCIAL):
        return False
    if hits(title_text, AUTO_ONLY) and not hits(title_text, STRONG + ("optimus",)):
        return False
    # VLA/world-model boilerplate in a channel description cannot rescue an unrelated title.
    if hits(title_text, STRONG):
        return True
    if hits(title_text, ROBOT) and hits(combined, TECH + PARTS + COMMERCIAL + INTERVIEW + ("ai",)):
        return True
    if source_name in DEDICATED_NAMES:
        return hits(title_text, MODELS + TECH + PARTS + COMMERCIAL + ROLES + ROBOT + ("compute", "training", "learning"))
    if hits(title_text, ("optimus",)) and hits(combined, ROBOT + STRONG):
        return True
    # Broad interviews must establish the subject in their opening, not a sponsor footer.
    opening = focus[:450]
    return hits(opening, STRONG) and hits(title_text, INTERVIEW + ROLES + ROBOT + TECH + PARTS)


def score_embodied_item(
    title: str, description: str = "", source_name: str = "",
    published_text: str = "", minutes: float | None = None,
    *, now: datetime | None = None,
) -> tuple[float, list[str]]:
    title_text = clean(title)
    focus = episode_focus(description)
    text = title_text + " " + focus[:1800]
    score = 5.5
    reasons = ["具身智能"]
    commercial = hits(text, COMMERCIAL)
    technical = hits(text, TECH + ("vla", "vision-language-action", "robot foundation model"))
    parts = hits(text, PARTS)
    quantitative = bool(re.search(r"(?:\$|¥|￥)\s*\d|\d[\d,.]*\s*(?:%|units|hours|robots|台|套|小时|万元|美元)", text))
    # Metadata is not independent verification: provisional plans receive a smaller bonus.
    if commercial:
        if hits(text, PLANNING) and not hits(text, ("已交付", "已部署", "已投产", "delivered", "deployed")):
            score += 0.5
            reasons.append("量产/部署计划，待验证")
        elif quantitative or hits(text, ("已交付", "已部署", "已投产", "delivered", "deployed", "ready stock")):
            score += 1.5
            reasons.append("量产/成本/部署数据（来源自述）")
        else:
            score += 0.5
            reasons.append("商业落地讨论，数据待核")
    if hits(text, ROLES) and hits(text, INTERVIEW) and (minutes is None or minutes >= 15):
        score += 1.2
        reasons.append("一手访谈线索")
    if technical:
        score += 1.0
        reasons.append("核心技术路线")
    if parts:
        score += 0.8
        reasons.append("关键零部件")
    reference_time = now or datetime.now(timezone.utc)
    try:
        published = datetime.strptime(published_text[:10], "%Y-%m-%d").replace(tzinfo=timezone.utc)
        age = (reference_time - published).days
    except (TypeError, ValueError):
        age = None
    if age is not None and 0 <= age <= 14 and source_name:
        score += 0.6
        reasons.append("近两周更新")
    if hits(title_text + " " + focus[:200], DEMO):
        score -= 1.5
        reasons.append("表演内容降权")
    if hits(title_text, ROUNDUP):
        score -= 1.0
        reasons.append("汇总内容降权")
    if hits(title_text, REPOST) or (minutes is not None and minutes < 8 and not (quantitative or technical or parts)):
        score -= 0.8
        reasons.append("短片/转载信息有限")
    return round(max(0.0, min(10.0, score)), 2), reasons
