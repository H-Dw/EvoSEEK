from __future__ import annotations

import re

INSTRUCTION_PATTERNS = (
    re.compile(r"\bignore\s+(all\s+)?previous\s+instructions\b", re.IGNORECASE),
    re.compile(r"\b(system|developer)\s+prompt\b", re.IGNORECASE),
    re.compile(r"\b(call|invoke|execute|run)\s+(the\s+)?(tool|command|shell)\b", re.IGNORECASE),
    re.compile(r"忽略.{0,12}(指令|提示词)"),
    re.compile(r"执行.{0,8}(命令|工具)"),
)


def instruction_like_markers(text: str) -> tuple[str, ...]:
    return tuple(
        f"pattern_{index}"
        for index, pattern in enumerate(INSTRUCTION_PATTERNS, start=1)
        if pattern.search(text)
    )

