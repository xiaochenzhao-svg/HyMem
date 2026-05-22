"""Helper utilities for HyMem (JSON parsing, token counting, etc.)."""

import re
import json
import logging
from typing import Optional, Any, Dict

import tiktoken

logger = logging.getLogger(__name__)


def cal_token(text: str) -> int:
    """Approximate token count using cl100k_base."""
    return len(tiktoken.get_encoding("cl100k_base").encode(text))


def fix_json_quotes_comprehensive(json_str: str) -> str:
    """Repair the most common malformed-quote patterns LLMs emit inside strings."""
    result = []
    state = "outside"
    i = 0
    while i < len(json_str):
        ch = json_str[i]

        if state == "escape_next":
            result.append(ch)
            state = "inside_string"
            i += 1
            continue

        if ch == "\\":
            result.append(ch)
            if state == "inside_string":
                state = "escape_next"
            i += 1
            continue

        if ch == '"':
            if state == "outside":
                result.append('"')
                state = "inside_string"
            else:
                # Look ahead to decide whether this quote closes the string
                j = i + 1
                while j < len(json_str) and json_str[j] in " \t\n\r":
                    j += 1
                if j < len(json_str) and json_str[j] in ',:[]}{':
                    result.append('"')
                    state = "outside"
                else:
                    result.append('\\"')
            i += 1
            continue

        result.append(ch)
        i += 1

    return "".join(result)


def extract_json_from_response(response: str) -> Optional[str]:
    """Pull the first JSON object out of an LLM response. Tolerates None."""
    if response is None or not isinstance(response, str):
        return None
    match = re.search(r"\{.*\}", response, re.DOTALL)
    return match.group(0) if match else None


def parse_json_response(
    response: Any,
    fix_quotes: bool = True,
) -> Optional[Dict[str, Any]]:
    """Parse the JSON payload from an LLM response. Returns None on any failure."""
    if not isinstance(response, str) or not response.strip():
        logger.debug("parse_json_response: empty / non-string response: %r", response)
        return None

    json_str = extract_json_from_response(response)
    if not json_str:
        logger.debug("parse_json_response: no JSON found in response: %s", response)
        return None

    try:
        if fix_quotes:
            json_str = fix_json_quotes_comprehensive(json_str)
        return json.loads(json_str)
    except json.JSONDecodeError as e:
        logger.debug("parse_json_response: JSONDecodeError: %s | raw: %s", e, response)
        return None
    except Exception as e:  # pragma: no cover - defensive
        logger.debug("parse_json_response: unexpected error: %s", e)
        return None


def ensure_directory_exists(file_path: str) -> None:
    """Make sure the directory containing `file_path` exists."""
    import os
    directory = os.path.dirname(file_path)
    if directory:
        os.makedirs(directory, exist_ok=True)
