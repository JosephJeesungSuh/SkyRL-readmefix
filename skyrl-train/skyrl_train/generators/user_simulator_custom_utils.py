import json
import re
from typing import Any, Optional, Tuple

_CODE_FENCE_RE = re.compile(
    r"```(?:json|JSON)?\s*(.*?)\s*```",
    re.DOTALL,
)

def _strip_code_fences(s: str) -> Optional[str]:
    m = _CODE_FENCE_RE.search(s)
    return m.group(1) if m else None

def _find_balanced_json_span(s: str) -> Optional[Tuple[int, int]]:
    """
    Returns (start, end_exclusive) for the first balanced {...} or [...] block,
    ignoring braces/brackets inside string literals.
    """
    starts = []
    for i, ch in enumerate(s):
        if ch in "{[":
            starts.append((i, ch))
            break
    if not starts:
        return None

    start, open_ch = starts[0]
    close_ch = "}" if open_ch == "{" else "]"

    depth = 0
    in_str = False
    esc = False

    for j in range(start, len(s)):
        c = s[j]
        if in_str:
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == '"':
                in_str = False
            continue
        else:
            if c == '"':
                in_str = True
                continue
            if c == open_ch:
                depth += 1
            elif c == close_ch:
                depth -= 1
                if depth == 0:
                    return (start, j + 1)
            # If JSON is an object but we see array open, or vice versa,
            # we don't treat it as changing depth for the chosen type.
            # This keeps it simple and works well in practice.
    return None

def robust_json_loads(text: Any) -> Any:
    """
    Robustly parse JSON from LLM output.
    - Accepts str/bytes/other; converts to str.
    - Handles ```json ... ``` fences.
    - Handles extra text before/after JSON.
    """
    if text is None:
        raise ValueError("No text to parse")

    if isinstance(text, (bytes, bytearray)):
        s = text.decode("utf-8", errors="replace")
    else:
        s = str(text)

    s = s.strip()

    # 1) Strict parse
    try:
        return json.loads(s)
    except Exception:
        pass

    # 2) If fenced, parse inside fence
    inner = _strip_code_fences(s)
    if inner is not None:
        inner = inner.strip()
        try:
            return json.loads(inner)
        except Exception:
            # keep going
            s = inner  # continue cleanup on inner

    # 3) Extract first balanced JSON object/array from remaining text
    span = _find_balanced_json_span(s)
    if span is not None:
        candidate = s[span[0]:span[1]].strip()
        return json.loads(candidate)

    # 4) Nothing worked
    raise ValueError(f"Could not find valid JSON in text (len={len(s)})")
