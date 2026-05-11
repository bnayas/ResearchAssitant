"""
coding.design_spec
──────────────────
Tool: Convert a research goal into a validated SimulationSpec.

Phase 4 refactor: drives the SimulationDesigner directly instead of
going through LocalCodingAssistant.  Auto-answers clarification
questions from the article brief and literature context.
"""
from __future__ import annotations

import json
from typing import Any

from ...contracts import AssistantId
from ...registry import ArtifactRegistry
from ..base import ToolContext, ToolDescriptor, ToolRequirement, ToolResult
from ..validators import artifact_kind_is, is_non_empty_string

TOOL = ToolDescriptor(
    name="design_simulation_spec",
    display_name="Design Simulation Spec",
    description=(
        "Convert a research goal into a validated SimulationSpec.  Uses the "
        "SimulationDesigner LLM to produce variables, stopping conditions, "
        "state fields, and all code blocks.  Auto-answers designer clarification "
        "questions from article brief and literature context.  Returns a spec "
        "artifact ready for runtime materialisation."
    ),
    agent_id="coding",
    requirements=[
        ToolRequirement(
            name="instruction", description="The PI's research directive text",
            type="str", validator=is_non_empty_string,
            validator_description="Must be a non-empty instruction string",
        ),
        ToolRequirement(
            name="brief", description="The article brief ArtifactRef",
            type="ArtifactRef", validator=artifact_kind_is("article_brief"),
            validator_description="Must be an 'article_brief' artifact",
        ),
        ToolRequirement(
            name="literature", description="The literature review ArtifactRef",
            type="ArtifactRef", required=False,
        ),
        ToolRequirement(
            name="functions", description="List of generated function code ArtifactRefs",
            type="list[ArtifactRef]", required=False, default=[],
        ),
        ToolRequirement(
            name="sample_steps", description="Number of steps for sample runs",
            type="int", required=False, default=200,
        ),
    ],
    produces=["simulation_spec"],
    tags=["coding", "design", "spec"],
    idempotent=True,
    estimated_seconds=120.0,
)


def execute(
    context: ToolContext,
    *,
    registry: ArtifactRegistry,
    llm_config: Any = None,
) -> ToolResult:
    """Drive SimulationDesigner directly to produce a spec."""
    from sim_tool.contract import ClarificationRequest, QuestionTopic
    from sim_tool.designer import SimulationDesigner
    from sim_tool.tool import SimulationTool
    from ...assistants.backends import make_service_backend
    from ...runtime_assemble import _jsonify_spec

    brief = context.artifacts["brief"]
    literature = context.artifacts.get("literature")
    functions = context.artifacts.get("functions") or []
    instruction = context.inputs.get("instruction", context.instruction)
    sample_steps = context.inputs.get("sample_steps", 200)

    # Build research goal from artifacts
    inquiry_answers = context.metadata.get("inquiry_answers") or {}
    research_goal = _compose_research_goal(
        instruction, brief, literature, functions, inquiry_answers=inquiry_answers)

    # Create designer directly
    backend = make_service_backend("designer", llm_config)
    if backend is None:
        return ToolResult(status="failed", message="No LLM backend for spec design")

    tool = SimulationTool(
        output_root=context.output_dir,
        designer_backend=backend,
        analyst_backend=make_service_backend("analyst", llm_config),
    )

    try:
        result = tool.design(research_goal)
    except Exception as exc:
        return ToolResult(status="failed", message=f"Designer failed: {exc}")

    # Auto-answer clarification rounds using brief/literature context
    max_rounds = 4
    rounds = 0
    while isinstance(result, ClarificationRequest) and rounds < max_rounds:
        answers = _auto_answer(result, brief, literature)
        try:
            result = tool.answer(result.session_id, answers)
        except Exception as exc:
            return ToolResult(status="failed", message=f"Clarification round failed: {exc}")
        rounds += 1

    # If still clarifying, escalate to orchestrator
    if isinstance(result, ClarificationRequest):
        from ...service_contracts import OrchestrationRequestEnvelope
        questions = result.questions
        question_text = "\n".join(f"{q.index}. {q.text}" for q in questions)
        return ToolResult(
            status="needs_input",
            message=f"Designer needs answers after {rounds} auto-rounds:\n{question_text}",
            request=OrchestrationRequestEnvelope.new(
                resume_token=result.session_id,
                request_kind="clarification",
                question=question_text,
                expected_schema={"answers": {str(q.index): "string" for q in questions}},
                capability_hint="user",
            ),
        )

    # Spec is ready
    try:
        spec = tool._designer.get_spec(result.session_id)
        spec_payload = _jsonify_spec(spec)
    except Exception as exc:
        return ToolResult(status="failed", message=f"Failed to extract spec: {exc}")

    spec_ref = registry.save_json(
        assistant=AssistantId.CODING_AGENT.value,
        kind="simulation_spec",
        title="Simulation Spec",
        filename="simulation/spec.json",
        payload=spec_payload,
        summary=spec.description[:160] if spec.description else spec.name,
        metadata={"session_id": result.session_id},
        artifact_id="simulation-spec",
    )
    return ToolResult(
        status="completed",
        artifacts=[spec_ref],
        message=f"Spec designed: {spec.name} ({len(spec.variables)} variables)",
        metadata={"spec_payload": spec_payload, "session_id": result.session_id},
    )


def _compose_research_goal(
    instruction: str,
    brief: Any,
    literature: Any,
    functions: list[Any] | None = None,
    inquiry_answers: dict[str, Any] | None = None,
) -> str:
    """Build a rich research goal string from directive and artifacts."""
    params = brief.metadata.get("key_parameters") or {}
    figures = brief.metadata.get("expected_figures") or []
    model_desc = str(brief.metadata.get("model_description") or "")
    procedure = str(brief.metadata.get("procedure") or "")
    lit_summary = ""
    if literature:
        lit_summary = str(literature.metadata.get("synthesis") or literature.summary or "")

    parts = [instruction]
    if model_desc:
        parts.append(f"Model description:\n{model_desc}")
    if procedure:
        parts.append(f"Procedure:\n{procedure}")
    if params:
        parts.append(f"Key parameters:\n{json.dumps(params, indent=2)}")
    if figures:
        parts.append("Expected figures:\n" + "\n".join(f"- {f}" for f in figures))
    if lit_summary:
        parts.append(f"Related literature context:\n{lit_summary}")
    inquiries = brief.metadata.get("inquiries") or []
    if inquiries:
        parts.append(f"Open inquiries from upstream analysis:\n{json.dumps(inquiries, indent=2)}")
    if inquiry_answers:
        parts.append(f"Resolved inquiry answers:\n{json.dumps(inquiry_answers, indent=2)}")
    if functions:
        function_blocks = []
        for ref in functions:
            spec = (getattr(ref, "metadata", {}) or {}).get("function_spec") or {}
            name = spec.get("name") or getattr(ref, "title", "function")
            signature = spec.get("signature", "")
            docstring = spec.get("docstring", getattr(ref, "summary", ""))
            function_blocks.append(f"- {name}: {signature}\n  {docstring}")
        if function_blocks:
            parts.append(
                "Generated function contracts available for implementation:\n"
                + "\n".join(function_blocks)
            )
    return "\n\n".join(parts)


def _auto_answer(clarification: ClarificationRequest, brief: Any, literature: Any) -> dict[str, str]:
    """Auto-answer designer questions from article brief and literature."""
    from sim_tool.contract import QuestionTopic

    params = brief.metadata.get("key_parameters") or {}
    model_desc = str(brief.metadata.get("model_description") or "")
    lit_summary = ""
    if literature:
        lit_summary = str(literature.metadata.get("synthesis") or literature.summary or "")

    answers: dict[str, str] = {}
    for q in clarification.questions:
        if q.topic == QuestionTopic.VARIABLE_RANGE:
            text = (
                "Use the parameter values reported in the article. "
                f"Parsed parameters: {json.dumps(params)}"
            )
        elif q.topic == QuestionTopic.STOPPING_THRESHOLD:
            text = (
                "Use conservative stopping criteria: stable summary statistics "
                "over repeated intervals as success, numerical divergence as failure."
            )
        elif q.topic == QuestionTopic.STEP_BUDGET:
            text = (
                "Use a moderate sample budget and larger full-sweep budget. "
                "Enough steps to reproduce the paper's headline figures."
            )
        elif q.topic == QuestionTopic.PHYSICAL_UNITS:
            text = "Use the units and conventions stated in the article."
        else:
            text = (
                "Sweep the most important reported stochastic controls first. "
                f"Model context: {model_desc or lit_summary}"
            )
        answers[str(q.index)] = text
    return answers
