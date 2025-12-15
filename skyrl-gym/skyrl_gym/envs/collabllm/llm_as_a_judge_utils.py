import ast
from typing import Optional, Tuple

def _outer_braces_span(s: str) -> Optional[Tuple[int, int]]:
    """
    Returns (start, end_exclusive) for the outermost JSON/Python-dict-like {...}
    starting at the first '{', matching the correct '}', ignoring braces inside
    double-quoted strings (and handling backslash escapes).
    """
    start = s.find("{")
    if start == -1:
        return None

    depth = 0
    in_str = False
    esc = False

    for i in range(start, len(s)):
        c = s[i]

        if in_str:
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == '"':
                in_str = False
            continue

        if c == '"':
            in_str = True
            continue

        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return (start, i + 1)

    return None  # no matching closing brace found

def _keep_only_outer_object(s: str, *, inner: bool = False) -> str:
    """
    Keeps only the content inside the outermost {...}.
    - inner=False -> returns "{...}"
    - inner=True  -> returns "..." (without outer braces)
    """
    span = _outer_braces_span(s)
    if span is None:
        raise ValueError("No balanced outer {...} found")

    obj = s[span[0]:span[1]]
    return obj[1:-1] if inner else obj

def extract_outer_dict(s: str) -> dict:
    """
    Extracts the outermost dictionary-like {...} from the string `s`
    and parses it into a Python dictionary.
    """
    obj_str = _keep_only_outer_object(s, inner=False)
    try:
        parsed_dict = ast.literal_eval(obj_str)
    except Exception as e:
        raise ValueError(f"Failed to parse extracted object as dict: {e}")
    if not isinstance(parsed_dict, dict):
        raise ValueError(f"Extracted object is not a dict but of type {type(parsed_dict)}")
    return parsed_dict