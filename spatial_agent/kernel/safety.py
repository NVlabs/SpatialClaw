"""Security sandbox for LLM-generated code.

Uses AST-based static analysis to detect forbidden imports, builtins, and
file I/O operations BEFORE the code is executed in the Jupyter kernel.

Only LLM-generated code is checked.  Pre-installed tool implementations
(ReconstructTool, SAM3Tool, FeedbackModule, etc.) are trusted and can use any imports.
"""

import ast
import re
from typing import Optional, Set


FORBIDDEN_MODULES: Set[str] = {
    "os",
    "subprocess",
    "sys",
    "shutil",
    "pathlib",
    "io",
    "socket",
    "multiprocessing",
    "threading",
    "ctypes",
    "signal",
    "importlib",
    "pickle",
    "shelve",
    "glob",
    "tempfile",
    "webbrowser",
    "http",
    "urllib",
    "requests",
    "torch",
    "tensorflow",
    "jax",
}

FORBIDDEN_BUILTINS: Set[str] = {
    "open",
    "exec",
    "eval",
    "compile",
    "__import__",
    "breakpoint",
    "input",
}

# Regex patterns for additional runtime-evasion tricks
_FILE_IO_PATTERN = re.compile(
    r"""
    \bopen\s*\(            |  # open(...)
    \.read\s*\(            |  # .read(...)
    \.write\s*\(           |  # .write(...)
    \.readlines\s*\(       |  # .readlines(...)
    \.writelines\s*\(      |  # .writelines(...)
    \bPath\s*\(            |  # pathlib.Path(...)
    \.save\s*\(            |  # PIL .save(...)
    \.to_csv\s*\(          |  # pandas .to_csv(...)
    \.to_json\s*\(         |  # .to_json(...)
    \.to_parquet\s*\(         # .to_parquet(...)
    """,
    re.VERBOSE,
)


class SecuritySandbox:
    """Static analysis checker for LLM-generated code."""

    @staticmethod
    def check(code: str) -> Optional[str]:
        """Return ``None`` if the code is safe, or an error message string.

        This does NOT execute the code -- it only parses and inspects it.
        """
        # ----------------------------------------------------------
        # 1. Parse to AST
        # ----------------------------------------------------------
        try:
            tree = ast.parse(code)
        except SyntaxError as exc:
            return f"SyntaxError in generated code: {exc}"

        # ----------------------------------------------------------
        # 2. Walk AST nodes
        # ----------------------------------------------------------
        for node in ast.walk(tree):
            # --- Import checks ---
            if isinstance(node, ast.Import):
                for alias in node.names:
                    root = alias.name.split(".")[0]
                    if root in FORBIDDEN_MODULES:
                        return (
                            f"Forbidden import: '{alias.name}'. "
                            f"Module '{root}' is not allowed."
                        )

            if isinstance(node, ast.ImportFrom):
                if node.module:
                    root = node.module.split(".")[0]
                    if root in FORBIDDEN_MODULES:
                        return (
                            f"Forbidden import: 'from {node.module} import ...'. "
                            f"Module '{root}' is not allowed."
                        )

            # --- Forbidden builtin calls ---
            if isinstance(node, ast.Call):
                func = node.func
                name: Optional[str] = None
                if isinstance(func, ast.Name):
                    name = func.id
                elif isinstance(func, ast.Attribute):
                    name = func.attr

                if name in FORBIDDEN_BUILTINS:
                    return (
                        f"Forbidden builtin call: '{name}()'. "
                        f"This operation is not allowed."
                    )

        # ----------------------------------------------------------
        # 3. Regex fallback for patterns that AST might miss
        # ----------------------------------------------------------
        if _FILE_IO_PATTERN.search(code):
            # Whitelist: tools/feedback/vlm are allowed to .save() internally
            # but the LLM code should not use .save()
            for match in _FILE_IO_PATTERN.finditer(code):
                start = max(0, match.start() - 50)
                context = code[start : match.start()]
                if (
                    "tools." in context
                    or "feedback." in context
                    or "vlm." in context
                ):
                    continue
                return (
                    f"Potentially forbidden file I/O operation detected: "
                    f"'{match.group().strip()}'. "
                    f"File reading and writing are not allowed."
                )

        return None  # safe
