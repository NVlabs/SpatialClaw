"""Translate a ReAct tool-call into a Python code string.

The ReAct baseline constrains the LLM to emit one structured tool call per
step. This module parses the ``{"tool": ..., "args": {...}}`` payload and
converts it to a single-line assignment like::

    result_<step> = tools.SAM3.segment_image_by_text(image=InputImages[0], prompt="red car")

which is then handed to ``execute_node`` unchanged. Arguments are
**Python expression strings** validated by a restricted AST walker so the
structured interface cannot smuggle in arbitrary code (no binops, no
comprehensions, no nested calls).

Arg expressions may contain **one method call** per leaf, and only when the
call's receiver chain is rooted at a kernel-bound name (``result_<N>`` or
one of the built-in constants). This unlocks patterns like
``result_3.get_centroid_3d(result_1, frame=result_0.frame_indices[0], object=0)``
without permitting arbitrary code to be smuggled in.
"""

import ast
from typing import Any, Dict


ALLOWED_TOOL_PREFIXES = ("tools.",)
ALLOWED_FULL_NAMES = {"vlm.locate", "vlm.ask_with_thinking", "ReturnAnswer", "show"}


_LITERAL_AST_NODES = (
    ast.Expression,
    ast.Constant,
    ast.Name,
    ast.Load,
    ast.Attribute,
    ast.Subscript,
    ast.Slice,
    ast.Tuple,
    ast.List,
    ast.Dict,
    ast.UnaryOp,
    ast.USub,
    ast.UAdd,
    ast.Index,  # py<3.9 compat, harmless on newer
)


_KNOWN_KERNEL_NAMES = {
    "InputImages", "Metadata", "tools", "feedback", "vlm", "ReturnAnswer",
    "VisualFeedback", "PerFrameMask", "Reconstruction", "FrameImage",
    "RefImages",
    "None", "True", "False",
}


def _is_known_kernel_name(name: str) -> bool:
    """True if ``name`` is a kernel-bound base usable in ReAct expressions."""
    if name in _KNOWN_KERNEL_NAMES:
        return True
    # Per-video InputImages_<N> in multi-video mode.
    if name.startswith("InputImages_") and name[len("InputImages_"):].isdigit():
        return True
    if name.startswith("result_") and name[7:].isdigit():
        return True
    return False


def _is_known_base(node: ast.AST) -> bool:
    """True if ``node`` is an attribute/subscript chain rooted at a known name.

    Known roots are the always-bound kernel names and ``result_<int>``
    back-references produced by prior ReAct steps.
    """
    while isinstance(node, (ast.Attribute, ast.Subscript)):
        node = node.value
    if not isinstance(node, ast.Name):
        return False
    return _is_known_kernel_name(node.id)


def _validate_expr(node: ast.AST, *, allow_call: bool) -> bool:
    """Recursively check that an expression AST is ReAct-safe.

    Rules:
    - Literals, names, attribute access, subscripts/slices, tuples/lists/dicts,
      and unary +/- on constants are always allowed.
    - A single ``Call`` node is allowed only at positions where ``allow_call``
      is True, and only when its receiver chain is rooted at a known kernel
      base. The call's own arguments are then validated with ``allow_call=False``
      so nested calls cannot be smuggled in.
    - No ``*args`` (``ast.Starred``) or ``**kwargs`` (keyword with ``arg=None``).
    """
    if isinstance(node, ast.Call):
        if not allow_call:
            return False
        if not isinstance(node.func, ast.Attribute):
            return False
        if not _is_known_base(node.func.value):
            return False
        if node.func.attr.startswith("_"):
            return False
        for a in node.args:
            if isinstance(a, ast.Starred):
                return False
            if not _validate_expr(a, allow_call=False):
                return False
        for kw in node.keywords:
            if kw.arg is None:
                return False
            if not _validate_expr(kw.value, allow_call=False):
                return False
        return True
    if isinstance(node, ast.Expression):
        return _validate_expr(node.body, allow_call=allow_call)
    if isinstance(node, (ast.Tuple, ast.List)):
        return all(_validate_expr(e, allow_call=allow_call) for e in node.elts)
    if isinstance(node, ast.Dict):
        for k in node.keys:
            if k is not None and not _validate_expr(k, allow_call=allow_call):
                return False
        return all(_validate_expr(v, allow_call=allow_call) for v in node.values)
    if isinstance(node, ast.Subscript):
        return (
            _validate_expr(node.value, allow_call=allow_call)
            and _validate_expr(node.slice, allow_call=allow_call)
        )
    if isinstance(node, ast.Attribute):
        return _validate_expr(node.value, allow_call=allow_call)
    if isinstance(node, ast.Slice):
        for part in (node.lower, node.upper, node.step):
            if part is not None and not _validate_expr(part, allow_call=allow_call):
                return False
        return True
    if isinstance(node, ast.UnaryOp):
        return isinstance(node.op, (ast.USub, ast.UAdd)) and _validate_expr(
            node.operand, allow_call=allow_call
        )
    if isinstance(node, ast.Index):  # py<3.9 compat wrapper
        return _validate_expr(node.value, allow_call=allow_call)
    if isinstance(node, (ast.Name, ast.Constant, ast.Load)):
        return True
    return False


def _looks_like_expr(expr: str) -> bool:
    """Return True if ``expr`` should be injected verbatim as a Python expr.

    Parses ``expr`` and validates it against the ReAct-safe grammar (literals,
    attribute/subscript chains, and optionally a single method call rooted at
    a kernel-bound base). A bare single-identifier string (e.g. ``"B"``,
    ``"car"``) is treated as a literal UNLESS it names a known kernel
    variable or a ``result_<int>`` back-reference — otherwise it would
    silently become a NameError at runtime for single-letter multiple-choice
    answers and prompt text.
    """
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError:
        return False
    if not _validate_expr(tree, allow_call=True):
        return False
    body = tree.body
    if isinstance(body, ast.Name):
        return _is_known_kernel_name(body.id)
    return True


def _render_value(value: Any, arg_name: str) -> str:
    """Render a JSON arg value to a Python source fragment.

    - String values that parse as restricted Python expressions are emitted
      as-is (so ``"InputImages[0]"`` becomes ``InputImages[0]``).
    - Other strings are emitted as Python string literals (via ``repr``),
      so prose like ``"What is happening?"`` becomes ``'What is happening?'``
      without requiring the LLM to double-escape quotes.
    - Lists and dicts are rendered recursively with the same rule applied
      to each element/value, so ``["InputImages[0]", "InputImages[15]"]``
      becomes ``[InputImages[0], InputImages[15]]``.
    - ``None``/``True``/``False``/numbers pass straight through.
    """
    if isinstance(value, str):
        if _looks_like_expr(value):
            return value
        # Treat as literal string (prose, quoted text, bare words).
        return repr(value)
    if isinstance(value, bool) or value is None:
        return repr(value)
    if isinstance(value, (int, float)):
        return repr(value)
    if isinstance(value, list):
        return "[" + ", ".join(_render_value(v, arg_name) for v in value) + "]"
    if isinstance(value, dict):
        parts = []
        for k, v in value.items():
            if not isinstance(k, str):
                raise ValueError(
                    f"arg {arg_name!r}: nested dict keys must be strings, "
                    f"got {type(k).__name__}."
                )
            parts.append(f"{k!r}: {_render_value(v, arg_name)}")
        return "{" + ", ".join(parts) + "}"
    raise ValueError(
        f"arg {arg_name!r}: unsupported JSON value type "
        f"{type(value).__name__}."
    )


def _validate_tool_path(tool: str) -> None:
    if tool in ALLOWED_FULL_NAMES:
        return
    if not any(tool.startswith(p) for p in ALLOWED_TOOL_PREFIXES):
        raise ValueError(
            f"unknown tool {tool!r}. Allowed: {sorted(ALLOWED_FULL_NAMES)} "
            f"or a tools.<Module>.<method> path."
        )
    # Require at least tools.<Module>.<method> (three dotted parts)
    parts = tool.split(".")
    if len(parts) < 3 or not all(parts):
        raise ValueError(
            f"tool path {tool!r} is incomplete. Expected "
            f"tools.<Module>.<method> (e.g. tools.SAM3.segment_image_by_text)."
        )
    # Block import-time name lookups like tools.__class__ etc.
    for p in parts[1:]:
        if p.startswith("_"):
            raise ValueError(
                f"tool path {tool!r} contains a private attribute {p!r}; "
                f"only public tool methods are callable."
            )


def translate(tool_call: Dict[str, Any], step: int) -> str:
    """Render ``tool_call`` as a single-line Python assignment.

    Args:
        tool_call: ``{"tool": <dotted name>, "args": {<name>: <expr_str>, ...}}``.
            ``args`` may be omitted or empty.
        step: The current step index; used to name the result variable.

    Returns:
        A Python source string of the form
        ``result_<step> = <tool>(<kwargs>)`` (or without the assignment for
        ``ReturnAnswer``).

    Raises:
        ValueError with a descriptive message on any schema, tool, or arg
        validation failure. The caller is expected to surface the message
        to the LLM so it can retry.
    """
    if not isinstance(tool_call, dict):
        raise ValueError(
            f"tool_call must be a JSON object, got {type(tool_call).__name__}."
        )

    tool = tool_call.get("tool")
    if not isinstance(tool, str) or not tool.strip():
        raise ValueError('tool_call is missing the "tool" string field.')
    tool = tool.strip()
    _validate_tool_path(tool)

    args = tool_call.get("args", {})
    if args is None:
        args = {}
    if not isinstance(args, dict):
        raise ValueError(
            f'"args" must be a JSON object mapping kwarg name → expression '
            f"string, got {type(args).__name__}."
        )

    extra = set(tool_call.keys()) - {"tool", "args"}
    if extra:
        raise ValueError(
            f"tool_call has unexpected keys {sorted(extra)}. "
            f'Only "tool" and "args" are allowed.'
        )

    # `show` is variadic-positional (def show(self, *args, label="")); it
    # cannot accept the kwargs that ReAct-style tool calls naturally emit.
    # Pick the first image-like arg and pass it positionally, preserving
    # `label` as a kwarg if present.
    if tool == "show":
        positional_expr = None
        label_expr = None
        for name, value in args.items():
            if not isinstance(name, str) or not name.isidentifier():
                raise ValueError(
                    f"arg name {name!r} is not a valid Python identifier."
                )
            rendered = _render_value(value, name)
            if name == "label":
                label_expr = rendered
            elif positional_expr is None:
                positional_expr = rendered
            else:
                raise ValueError(
                    "show() accepts a single image/list arg (any kwarg "
                    "name like `image`, `images`, or `visual_input`); "
                    "pass multiple images as a list."
                )
        parts = []
        if positional_expr is not None:
            parts.append(positional_expr)
        if label_expr is not None:
            parts.append(f"label={label_expr}")
        call = f"show({', '.join(parts)})"
        return f"result_{step} = {call}"

    rendered_kwargs = []
    for name, value in args.items():
        if not isinstance(name, str) or not name.isidentifier():
            raise ValueError(
                f"arg name {name!r} is not a valid Python identifier."
            )
        rendered_kwargs.append(f"{name}={_render_value(value, name)}")

    call = f"{tool}({', '.join(rendered_kwargs)})"

    # ReturnAnswer doesn't need its result captured — the sentinel is set
    # inside its constructor.
    if tool == "ReturnAnswer":
        return call

    return f"result_{step} = {call}"
