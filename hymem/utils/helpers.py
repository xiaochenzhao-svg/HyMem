"""
Helper utilities for HyMem.

This module provides utility functions for JSON parsing, text processing,
and other common operations used throughout the memory system.
"""

import re
import json
from typing import Optional, Any, Dict
import tiktoken

def cal_token(text: str):
    encoding = tiktoken.get_encoding("cl100k_base")
    tokens = encoding.encode(text)
    return len(tokens)


def fix_json_quotes_comprehensive(json_str: str) -> str:
    """
    Fix JSON string quotes issues comprehensively.
    
    This function handles various edge cases in JSON strings where quotes
    may be improperly escaped or formatted, ensuring valid JSON syntax.
    
    Args:
        json_str: Input JSON string that may have quote issues
        
    Returns:
        Fixed JSON string with properly formatted quotes
        
    Example:
        >>> bad_json = '{"key": "value with "quotes""}'
        >>> fixed = fix_json_quotes_comprehensive(bad_json)
        >>> json.loads(fixed)  # Will parse successfully
    """
    result = []
    state = 'outside'
    i = 0
    
    while i < len(json_str):
        char = json_str[i]
        
        # Handle escape sequences
        if state == 'escape_next':
            result.append(char)
            state = 'inside_string'
            i += 1
            continue
        
        if char == '\\':
            result.append(char)
            if state == 'inside_string':
                state = 'escape_next'
            i += 1
            continue
        
        # Handle quotes
        if char == '"':
            if state == 'outside':
                # Start of a string
                result.append('"')
                state = 'inside_string'
            elif state == 'inside_string':
                # Possible end of string
                # Look ahead to determine if this is the end
                j = i + 1
                while j < len(json_str) and json_str[j] in ' \t\n\r':
                    j += 1
                
                # If next non-whitespace char is JSON structure, this is end of string
                if j < len(json_str) and json_str[j] in ',:[]}{':
                    result.append('"')
                    state = 'outside'
                else:
                    # This quote is part of the string content, escape it
                    result.append('\\"')
            i += 1
            continue
        
        result.append(char)
        i += 1
    
    return ''.join(result)


def extract_json_from_response(response: str) -> Optional[str]:
    """
    Extract JSON content from a response string.
    
    Uses regex to find the first JSON object in the response.
    
    Args:
        response: Response string that may contain JSON
        
    Returns:
        Extracted JSON string, or None if not found
        
    Example:
        >>> response = "Here is the result: {\\"key\\": \\"value\\"}"
        >>> json_str = extract_json_from_response(response)
        >>> print(json_str)
        '{"key": "value"}'
    """
    pattern = r'\{.*\}'
    match = re.search(pattern, response, re.DOTALL)
    
    if match:
        return match.group(0)
    else:
        return None


def parse_json_response(
    response: str,
    fix_quotes: bool = True
) -> Optional[Dict[str, Any]]:
    """
    Parse JSON from a response string with error handling.
    
    This function extracts JSON from a response, optionally fixes quote issues,
    and parses it into a dictionary.
    
    Args:
        response: Response string containing JSON
        fix_quotes: Whether to fix quote issues before parsing
        
    Returns:
        Parsed JSON as dictionary, or None if parsing fails
        
    Example:
        >>> response = '{"result": {"finished": 1, "answer": "test"}}'
        >>> data = parse_json_response(response)
        >>> print(data['result']['answer'])
        'test'
    """
    # Extract JSON content
    json_str = extract_json_from_response(response)
    
    if not json_str:
        print(f"Error: No JSON found in response: {response}")
        return None
    
    try:
        # Optionally fix quotes
        if fix_quotes:
            json_str = fix_json_quotes_comprehensive(json_str)
        
        # Parse JSON
        data = json.loads(json_str)
        return data
        
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON response: {response}")
        print(f"JSONDecodeError: {e}")
        return None
    except Exception as e:
        print(f"Unexpected error parsing JSON: {e}")
        return None


def parse_timestamp(ts: str):
    """
    Parse timestamp string into datetime object.
    
    Args:
        ts: Timestamp string in format "I:%M %p on %d %B, %Y"
        
    Returns:
        datetime object
        
    Example:
        >>> from datetime import datetime
        >>> ts = "2:30 PM on 13 October, 2022"
        >>> dt = parse_timestamp(ts)
        >>> print(dt.year)
        2022
    """
    from datetime import datetime
    return datetime.strptime(ts, "%I:%M %p on %d %B, %Y")


def ensure_directory_exists(file_path: str) -> None:
    """
    Ensure the directory for a file path exists.
    
    Args:
        file_path: Path to a file (directory will be created if needed)
    """
    import os
    directory = os.path.dirname(file_path)
    if directory:
        os.makedirs(directory, exist_ok=True)


def format_session_text(speaker: str, text: str) -> str:
    """
    Format a conversation turn as text.
    
    Args:
        speaker: Speaker identifier
        text: Turn text content
        
    Returns:
        Formatted turn string
    """
    return f"Speaker {speaker} says : {text}"
