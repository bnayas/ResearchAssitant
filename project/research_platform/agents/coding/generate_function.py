"""
coding.generate_function
────────────────────────
Tool: Generate a single simulation function from its contract.

Phase 4 refactor: supports recursive sub-function planning.  When the
LLM decides it needs helper functions, it can declare them.  Each
sub-function is generated, validated, and cached for reuse.
"""
from __future__ import annotations

import threading
from typing import Any

from ...contracts import AssistantId
from ...registry import ArtifactRegistry
from ..base import ToolContext, ToolDescriptor, ToolRequirement, ToolResult
from ..validators import dict_has_keys, is_non_empty_string


def _resolve_backend(llm_config: Any) -> Any:
    """Resolve an LLM backend from config or a pre-built backend.

    If llm_config already has a `complete` method, it is used directly.
    Otherwise, ``make_service_backend`` is called to build one.
    """
    if llm_config is None:
        return None
    if hasattr(llm_config, "complete"):
        return llm_config
    from ...assistants.backends import make_service_backend
    return make_service_backend("designer", llm_config)

# Module-level function cache — shared across invocations, thread-safe.
_function_cache: dict[str, str] = {}
_cache_lock = threading.Lock()

TOOL = ToolDescriptor(
    name="generate_function",
    display_name="Generate Function",
    description=(
        "Generate a single simulation function from its contract (name, "
        "signature, docstring).  One LLM call per function.  The function "
        "is validated (syntax-checked) independently.  If the LLM declares "
        "sub-function dependencies, those are recursively generated, "
        "validated, and cached for reuse by other tools."
    ),
    agent_id="coding",
    requirements=[
        ToolRequirement(
            name="function_spec",
            description=(
                "A dict with keys: name, signature, docstring, context. "
                "Describes the function contract to implement."
            ),
            type="dict",
            validator=dict_has_keys("name", "signature"),
            validator_description="Must contain at least 'name' and 'signature' keys",
        ),
        ToolRequirement(
            name="context",
            description="Additional context (other function signatures, model description)",
            type="str",
            required=False,
            default="",
        ),
    ],
    produces=["function_code"],
    tags=["coding", "generation"],
    idempotent=True,
    estimated_seconds=15.0,
)


def execute(
    context: ToolContext,
    *,
    registry: ArtifactRegistry,
    llm_config: Any = None,
) -> ToolResult:
    """Generate a function, recursively generating sub-functions as needed."""
    from ...assistants.backends import make_service_backend

    function_spec = context.inputs["function_spec"]
    func_name = function_spec["name"]
    additional_context = context.inputs.get("context", "")

    # Check cache first
    with _cache_lock:
        if func_name in _function_cache:
            code = _function_cache[func_name]
            ref = _save_function(registry, func_name, code, function_spec)
            return ToolResult(
                status="completed",
                artifacts=[ref],
                message=f"Returned cached function: {func_name}",
                metadata={"code": code, "function_name": func_name, "cached": True},
            )

    backend = _resolve_backend(llm_config)
    if backend is None:
        return ToolResult(status="failed", message="No LLM backend for code generation")

    # Build cache context for the LLM
    with _cache_lock:
        available_helpers = "\n---\n".join(
            f"# {name}\n{code}" for name, code in _function_cache.items()
        )
    cached_block = f"\n\nAvailable helper functions (already generated):\n{available_helpers}" if available_helpers else ""

    try:
        import json
        raw = backend.complete(
            system=(
                "You are a scientific Python code generator.\n"
                "Generate ONLY the requested function.  Include type hints and a docstring.\n"
                "Use numpy/scipy where appropriate.\n\n"
                "If you need helper/sub-functions that don't exist yet, declare them at the "
                "END of your response after a line containing only '# --- SUB-FUNCTIONS ---'.\n"
                "List each needed sub-function as a JSON comment:\n"
                '# SUB: {"name": "helper_name", "signature": "def helper_name(...)", "docstring": "..."}\n\n'
                "Return ONLY Python code, no markdown fences or explanation."
            ),
            messages=[{
                "role": "user",
                "content": (
                    f"Function contract:\n"
                    f"  Name: {func_name}\n"
                    f"  Signature: {function_spec['signature']}\n"
                    f"  Docstring: {function_spec.get('docstring', '')}\n"
                    f"  Context: {function_spec.get('context', '')}\n"
                    f"\nAdditional context:\n{additional_context}"
                    f"{cached_block}"
                ),
            }],
            temperature=0.0,
        )
        code = str(raw).strip()
    except Exception as exc:
        return ToolResult(status="failed", message=f"Generation failed: {exc}")

    # Extract sub-function declarations
    sub_specs = _extract_sub_function_specs(code)
    main_code = _strip_sub_declarations(code)

    # Validate main function syntax
    try:
        compile(main_code, f"<{func_name}>", "exec")
    except SyntaxError as exc:
        return ToolResult(
            status="failed",
            message=f"Generated code has syntax error: {exc}",
            metadata={"code": main_code, "error": str(exc)},
        )

    # Cache the main function
    with _cache_lock:
        _function_cache[func_name] = main_code

    # Recursively generate sub-functions (max depth 2)
    sub_artifacts = []
    depth = context.metadata.get("_recursion_depth", 0)
    if sub_specs and depth < 2:
        for sub_spec in sub_specs:
            sub_name = sub_spec.get("name", "")
            with _cache_lock:
                if sub_name in _function_cache:
                    continue  # Already cached

            sub_context = ToolContext(
                tool_name="generate_function",
                directive_id=context.directive_id,
                instruction=context.instruction,
                inputs={
                    "function_spec": sub_spec,
                    "context": f"This is a helper for {func_name}.\n{additional_context}",
                },
                output_dir=context.output_dir,
                metadata={"_recursion_depth": depth + 1},
            )
            sub_result = execute(sub_context, registry=registry, llm_config=llm_config)
            if sub_result.ok:
                sub_artifacts.extend(sub_result.artifacts)

    # Save main function
    ref = _save_function(registry, func_name, main_code, function_spec)
    all_artifacts = [ref] + sub_artifacts

    sub_names = [s.get("name", "?") for s in sub_specs]
    msg = f"Generated function: {func_name}"
    if sub_names:
        msg += f" (+ {len(sub_names)} sub-function(s): {', '.join(sub_names)})"

    return ToolResult(
        status="completed",
        artifacts=all_artifacts,
        message=msg,
        metadata={
            "code": main_code,
            "function_name": func_name,
            "sub_functions": sub_names,
            "cached": False,
        },
    )


def get_function_cache() -> dict[str, str]:
    """Read-only snapshot of the current function cache."""
    with _cache_lock:
        return dict(_function_cache)


def clear_function_cache() -> None:
    """Clear the function cache (useful between test runs)."""
    with _cache_lock:
        _function_cache.clear()


def _save_function(registry: ArtifactRegistry, name: str, code: str, spec: dict) -> Any:
    return registry.save_text(
        assistant=AssistantId.CODING_AGENT.value,
        kind="function_code",
        title=f"Function: {name}",
        filename=f"simulation/functions/{name}.py",
        text=code,
        summary=spec.get("docstring", name),
        metadata={"function_spec": spec},
        artifact_id=f"function-{name}",
    )


def _extract_sub_function_specs(code: str) -> list[dict[str, str]]:
    """Extract # SUB: {...} declarations from generated code."""
    import json as _json
    specs = []
    for line in code.splitlines():
        stripped = line.strip()
        if stripped.startswith("# SUB:"):
            try:
                payload = _json.loads(stripped[6:].strip())
                if isinstance(payload, dict) and "name" in payload:
                    payload.setdefault("signature", f"def {payload['name']}()")
                    payload.setdefault("docstring", "")
                    specs.append(payload)
            except (_json.JSONDecodeError, KeyError):
                pass
    return specs


def _strip_sub_declarations(code: str) -> str:
    """Remove the # --- SUB-FUNCTIONS --- section and SUB: lines."""
    lines = code.splitlines()
    result = []
    in_sub_section = False
    for line in lines:
        if "# --- SUB-FUNCTIONS ---" in line:
            in_sub_section = True
            continue
        if in_sub_section and line.strip().startswith("# SUB:"):
            continue
        if not in_sub_section:
            result.append(line)
    return "\n".join(result)
