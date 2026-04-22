from __future__ import annotations

import json
from typing import Any, Dict, List, Optional


# Shared routing helpers used by both the default LangChain path and the
# legacy compatibility orchestrator.
SMALL_TALK_PATTERNS = {
    "hi",
    "hello",
    "hey",
    "thanks",
    "thank you",
    "good morning",
    "good night",
    "bye",
    "goodbye",
    "see you",
    "你好",
    "您好",
    "嗨",
    "谢谢",
    "感谢",
    "早上好",
    "晚上好",
    "晚安",
    "再见",
    "拜拜",
    "哈囉",
    "謝謝",
    "感謝",
    "早安",
    "再見",
    "掰掰",
}

SMALL_TALK_SUBSTRING_TRIGGERS = (
    "你好",
    "您好",
    "嗨",
    "哈喽",
    "拜拜",
    "谢谢",
    "感谢",
    "哈囉",
    "掰掰",
    "謝謝",
    "感謝",
)


def normalize_sources(sources: Optional[List[str]]) -> List[str]:
    normalized: List[str] = []
    if not sources:
        return normalized
    for item in sources:
        if item is None:
            continue
        token = str(item).strip().lower()
        if token and token not in normalized:
            normalized.append(token)
    return normalized


def is_small_talk_query(query: str) -> bool:
    stripped = (query or "").strip()
    if not stripped:
        return True

    lowered = stripped.lower()
    if lowered in SMALL_TALK_PATTERNS or stripped in SMALL_TALK_PATTERNS:
        return True

    return any(token in stripped for token in SMALL_TALK_SUBSTRING_TRIGGERS)


def extract_json_object(text: str) -> Optional[Dict[str, Any]]:
    content = (text or "").strip()
    if not content:
        return None

    try:
        parsed = json.loads(content)
        return parsed if isinstance(parsed, dict) else None
    except json.JSONDecodeError:
        start = content.find("{")
        end = content.rfind("}")
        if start == -1 or end == -1 or end <= start:
            return None
        try:
            parsed = json.loads(content[start : end + 1])
            return parsed if isinstance(parsed, dict) else None
        except json.JSONDecodeError:
            return None


def coerce_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes", "y", "on", "需要"}:
            return True
        if normalized in {"false", "0", "no", "n", "off", "不需要"}:
            return False
    return default


def build_decision_prompt(query: str) -> str:
    return (
        "请判断下述用户问题是否需要实时搜索。"
        "如果无需搜索，请直接在answer字段内给出最终答案。"
        "如果需要搜索，将answer设为空字符串。\n\n"
        '输出严格的JSON，形如{"needs_search": bool, "reason": string, "answer": string}.\n'
        "用户问题:\n" + query
    )


def build_keyword_prompt(query: str) -> str:
    return (
        "请为以下问题生成不超过6个高质量的中英文双语搜索关键词或短语，"
        "每个关键词概念提供中英文版本，以数组形式返回JSON，"
        '例如{"keywords": ["关键词1", "keyword 1", "关键词2", "keyword 2"]}。\n\n'
        "规则：\n"
        "1. 关键词应覆盖查询核心信息\n"
        "2. 使用英文关键词提升搜索效果\n"
        "3. 对于体育比赛查询，添加'战报 highlights'、'得分统计 box score'等新闻/数据关键词\n"
        "4. 对于最新新闻查询，添加'最新 latest'、'新闻 news'等时效性关键词\n"
        "5. 避免只生成赛程/日程类关键词，应优先生成能获取详细内容的关键词\n"
        "6. 【重要】如果问题涉及时间范围（如前三年、近五年、历年），必须保留时间关键词\n"
        "7. 【重要】如果问题要求具体数值/数据（如具体值、股价数据、排名数字），必须添加'数据 data'、'具体 specific'等关键词\n"
        "8. 对于股价/金融查询，添加'stock price history'、'historical data'、'收盘价'等数据相关关键词\n\n"
        "用户问题:\n" + query
    )
