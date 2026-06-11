"""VariableTracker: diff variables between steps and build summaries."""

from typing import Any, Dict, List, Tuple


class VariableTracker:
    """Stateless utility for tracking kernel variable changes."""

    @staticmethod
    def diff(
        prev: Dict[str, Dict[str, Any]],
        current: Dict[str, Dict[str, Any]],
    ) -> Tuple[Dict[str, Dict], Dict[str, Dict]]:
        """Compare previous and current variable registries.

        Returns:
            (new_vars, changed_vars) - dicts of variable info.
        """
        new_vars = {}
        changed_vars = {}
        for name, info in current.items():
            if name not in prev:
                new_vars[name] = info
            elif info != prev[name]:
                changed_vars[name] = info
        return new_vars, changed_vars

    @staticmethod
    def format_summary(name: str, info: Dict[str, Any]) -> str:
        """Format a variable's metadata as a concise summary string."""
        parts = [f"{name}: {info.get('type', '?')}"]

        if "shape" in info:
            parts.append(f"shape={info['shape']}")
        if "dtype" in info:
            parts.append(f"dtype={info['dtype']}")
        if "len" in info:
            parts.append(f"len={info['len']}")
        if "frame_indices" in info:
            fi = info["frame_indices"]
            if len(fi) > 6:
                fi_str = f"[{fi[0]}..{fi[-1]}] ({len(fi)} frames)"
            else:
                fi_str = str(fi)
            parts.append(f"frames={fi_str}")
        if "size_mb" in info and info["size_mb"] >= 1.0:
            parts.append(f"size={info['size_mb']:.1f}MB")
        if "keys" in info:
            keys = info["keys"]
            if len(keys) > 5:
                keys_str = str(keys[:5]) + "..."
            else:
                keys_str = str(keys)
            parts.append(f"keys={keys_str}")

        return ", ".join(parts)

    @staticmethod
    def check_large_variables(
        variables: Dict[str, Dict[str, Any]],
        max_size_mb: int = 500,
    ) -> List[str]:
        """Return warning strings for variables exceeding the size limit."""
        warnings = []
        for name, info in variables.items():
            size_mb = info.get("size_mb", 0)
            if size_mb > max_size_mb:
                warnings.append(
                    f"Variable '{name}' is {size_mb:.1f}MB (limit: {max_size_mb}MB). "
                    f"Consider deleting it with `del {name}` after extracting needed data."
                )
        return warnings
