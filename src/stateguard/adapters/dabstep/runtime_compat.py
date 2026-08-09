"""Reproducible compatibility fixes for the pinned smolagents 1.3 runtime.

The official DABStep environment pins smolagents 1.3.0.  Its local AST
interpreter has a few Python-semantics bugs that materially affected the first
managed run.  Keep the fixes in this repository and apply them at process
startup instead of editing the ephemeral virtualenv under /tmp.
"""
from __future__ import annotations

import ast
from typing import Any, Callable, Dict, List

import smolagents
import smolagents.agents as agents_module
import smolagents.local_python_executor as executor_module


DEFAULT_MAX_OPERATIONS = 100_000_000
RUNTIME_PATCH_VERSION = "dabstep-smolagents-1.3-compat-v1"
OFFICIAL_RUNTIME_PROFILE = "official"
COMPAT_RUNTIME_PROFILE = "compat-v1"
SUPPORTED_SMOLAGENTS_VERSION = "1.3.0"
_RUNTIME_MARKER = "_stateguard_dabstep_runtime_patch"

_APPLIED = False
_ORIGINAL_EVALUATE_AST: Callable[..., Any] | None = None
_ORIGINAL_PARSE_CODE_BLOBS: Callable[[str], str] | None = None


def _evaluate_for_cpython_semantics(
    for_loop: ast.For,
    state: Dict[str, Any],
    static_tools: Dict[str, Callable],
    custom_tools: Dict[str, Callable],
    authorized_imports: List[str],
) -> Any:
    """Evaluate ``for`` with real loop-level break/continue semantics.

    smolagents 1.3 catches ContinueException inside the loop over AST body
    nodes.  Its ``continue`` therefore advances to the next statement in the
    same iteration instead of the next item in the data iterator.
    """

    result = None
    iterator = executor_module.evaluate_ast(
        for_loop.iter, state, static_tools, custom_tools, authorized_imports
    )
    broke = False
    for counter in iterator:
        executor_module.set_value(
            for_loop.target,
            counter,
            state,
            static_tools,
            custom_tools,
            authorized_imports,
        )
        try:
            for node in for_loop.body:
                line_result = executor_module.evaluate_ast(
                    node,
                    state,
                    static_tools,
                    custom_tools,
                    authorized_imports,
                )
                if line_result is not None:
                    result = line_result
        except executor_module.ContinueException:
            continue
        except executor_module.BreakException:
            broke = True
            break

    if not broke:
        for node in for_loop.orelse:
            line_result = executor_module.evaluate_ast(
                node, state, static_tools, custom_tools, authorized_imports
            )
            if line_result is not None:
                result = line_result
    return result


def _evaluate_ast_compat(
    expression: ast.AST,
    state: Dict[str, Any],
    static_tools: Dict[str, Callable],
    custom_tools: Dict[str, Callable],
    authorized_imports: List[str] = executor_module.BASE_BUILTIN_MODULES,
) -> Any:
    """Add the missing comprehension result types without changing recursion."""

    if isinstance(expression, ast.GeneratorExp):
        # evaluate_listcomp already implements nested comprehensions and local
        # comprehension scopes correctly for this pinned release.  Return an
        # iterator so next(...) and normal iteration have Python-compatible
        # behavior.  The values are materialized eagerly, which is acceptable
        # for this bounded, read-only analytical runtime.
        values = executor_module.evaluate_listcomp(
            expression, state, static_tools, custom_tools, authorized_imports
        )
        return iter(values)
    if isinstance(expression, ast.SetComp):
        values = executor_module.evaluate_listcomp(
            expression, state, static_tools, custom_tools, authorized_imports
        )
        return set(values)
    assert _ORIGINAL_EVALUATE_AST is not None
    return _ORIGINAL_EVALUATE_AST(
        expression, state, static_tools, custom_tools, authorized_imports
    )


def _parse_code_blobs_compat(code_blob: str) -> str:
    """Recover a syntactically complete code block missing only its end fence.

    A genuinely truncated/incomplete Python program remains an error.  Running
    an arbitrary parseable prefix would silently change the model's action.
    """

    assert _ORIGINAL_PARSE_CODE_BLOBS is not None

    # The upstream parser returns any earlier complete block even when a later
    # block was cut off.  Detect that case first so partial model actions are
    # never silently executed.
    if code_blob.count("```") % 2 == 1:
        last_fence = code_blob.rfind("```")
        line_end = code_blob.find("\n", last_fence)
        header = code_blob[last_fence:line_end].strip().lower()
        if line_end >= 0 and header in {"```", "```py", "```python"}:
            candidate = code_blob[line_end + 1 :]
            candidate = candidate.removesuffix("<end_code>").strip()
            try:
                ast.parse(candidate)
            except SyntaxError as exc:
                raise ValueError(
                    "The final code block is truncated and syntactically "
                    "incomplete; refusing to execute an earlier partial action."
                ) from exc

            prefix = code_blob[:last_fence]
            prefix_code = ""
            if "```" in prefix:
                prefix_code = _ORIGINAL_PARSE_CODE_BLOBS(prefix)
            return "\n\n".join(
                part for part in (prefix_code.strip(), candidate) if part
            )

    return _ORIGINAL_PARSE_CODE_BLOBS(code_blob)


def configure_smolagents_runtime(
    profile: str = COMPAT_RUNTIME_PROFILE,
    *,
    max_operations: int = DEFAULT_MAX_OPERATIONS,
) -> Dict[str, Any]:
    """Configure one process-wide DABstep runtime profile."""

    if profile not in {OFFICIAL_RUNTIME_PROFILE, COMPAT_RUNTIME_PROFILE}:
        raise ValueError(f"unknown DABstep runtime profile: {profile}")
    version = getattr(smolagents, "__version__", "unknown")
    if version != SUPPORTED_SMOLAGENTS_VERSION:
        raise RuntimeError(
            f"DABstep runtime profiles require smolagents {SUPPORTED_SMOLAGENTS_VERSION}, "
            f"got {version}"
        )
    existing = getattr(executor_module, _RUNTIME_MARKER, None)
    if profile == OFFICIAL_RUNTIME_PROFILE:
        if existing is not None:
            raise RuntimeError(
                "cannot select the official runtime after compat-v1 was applied "
                "in this process; start a fresh process"
            )
        return {
            "runtime_profile": OFFICIAL_RUNTIME_PROFILE,
            "patch_version": None,
            "smolagents_version": version,
            "max_python_operations": executor_module.MAX_OPERATIONS,
            "fixes": [],
        }
    report = apply_smolagents_runtime_fixes(max_operations=max_operations)
    return {"runtime_profile": COMPAT_RUNTIME_PROFILE, **report}


def apply_smolagents_runtime_fixes(
    *, max_operations: int = DEFAULT_MAX_OPERATIONS
) -> Dict[str, Any]:
    """Apply idempotent process-local fixes and return auditable metadata."""

    global _APPLIED, _ORIGINAL_EVALUATE_AST, _ORIGINAL_PARSE_CODE_BLOBS
    version = getattr(smolagents, "__version__", "unknown")
    if version != SUPPORTED_SMOLAGENTS_VERSION:
        raise RuntimeError(
            f"compat-v1 requires smolagents {SUPPORTED_SMOLAGENTS_VERSION}, got {version}"
        )
    if max_operations < 10_000_000:
        raise ValueError("max_operations must be at least the official 10,000,000")
    existing = getattr(executor_module, _RUNTIME_MARKER, None)
    if existing not in {None, RUNTIME_PATCH_VERSION}:
        raise RuntimeError(f"another DABstep runtime patch is already active: {existing}")

    if not _APPLIED:
        _ORIGINAL_EVALUATE_AST = executor_module.evaluate_ast
        _ORIGINAL_PARSE_CODE_BLOBS = agents_module.parse_code_blobs
        executor_module.evaluate_for = _evaluate_for_cpython_semantics
        executor_module.evaluate_ast = _evaluate_ast_compat
        agents_module.parse_code_blobs = _parse_code_blobs_compat
        executor_module.BASE_PYTHON_TOOLS.setdefault("repr", repr)
        setattr(executor_module, _RUNTIME_MARKER, RUNTIME_PATCH_VERSION)
        _APPLIED = True

    executor_module.MAX_OPERATIONS = int(max_operations)
    return {
        "patch_version": RUNTIME_PATCH_VERSION,
        "smolagents_version": getattr(smolagents, "__version__", "unknown"),
        "max_python_operations": executor_module.MAX_OPERATIONS,
        "fixes": [
            "for-loop break/continue semantics",
            "GeneratorExp returns an iterator usable by next()",
            "SetComp support",
            "repr builtin",
            "missing closing code-fence recovery when Python is complete",
        ],
    }


__all__ = [
    "COMPAT_RUNTIME_PROFILE",
    "OFFICIAL_RUNTIME_PROFILE",
    "SUPPORTED_SMOLAGENTS_VERSION",
    "configure_smolagents_runtime",
    "DEFAULT_MAX_OPERATIONS",
    "RUNTIME_PATCH_VERSION",
    "apply_smolagents_runtime_fixes",
]
