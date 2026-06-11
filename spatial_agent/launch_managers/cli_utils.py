"""Shared CLI utilities for launch managers."""

from typing import List


def parse_range_selection(input_str: str, max_val: int) -> List[int]:
    """Parse range notation like '1-5, 10, 15-16' into sorted unique indices.

    Returns 1-based indices within [1, max_val].
    """
    result = set()
    for part in input_str.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            try:
                start, end = part.split("-", 1)
                start, end = int(start.strip()), int(end.strip())
                for i in range(start, end + 1):
                    if 1 <= i <= max_val:
                        result.add(i)
            except ValueError:
                continue
        else:
            try:
                val = int(part)
                if 1 <= val <= max_val:
                    result.add(val)
            except ValueError:
                continue
    return sorted(result)
