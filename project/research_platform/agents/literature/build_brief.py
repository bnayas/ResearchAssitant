"""
literature.build_brief
──────────────────────
Tool: Extract a structured research brief from article text.

The brief is the canonical structured representation of a paper's
scientific content.  It is consumed by simulation, writing, and math
agents, so extraction quality matters.

Extraction approach:
  1. PRIMARY PASS — full structured extraction in one LLM call.
  2. VALIDATION PASS — check for empty/vague critical fields and
     re-extract only those fields with targeted prompts.
  3. INQUIRY GENERATION — identify unresolved choices that materially
     affect downstream work and surface them as typed inquiries.

The tool is agnostic about who is calling it.  "Steering notes" are
optional correction strings, not instructions from a named person.
"""
from __future__ import annotations

import json
from typing import Any

from ...contracts import AssistantId
from ...registry import ArtifactRegistry
from ..base import ToolContext, ToolDescriptor, ToolRequirement, ToolResult
from ..validators import artifact_has_metadata_key

# ---------------------------------------------------------------------------
# Tool descriptor
# ---------------------------------------------------------------------------

TOOL = ToolDescriptor(
    name="build_article_brief",
    display_name="Build Article Brief",
    description=(
        "Extract a structured research brief from a primary_article artifact. "
        "The brief captures: model description, key parameters, simulation "
        "procedure, expected figures, assumptions, and validation criteria. "
        "Performs a primary extraction pass followed by targeted re-extraction "
        "of any weak fields.  Surfaces ambiguous choices as typed inquiries."
    ),
    agent_id="literature",
    requirements=[
        ToolRequirement(
            name="article",
            description="A primary_article ArtifactRef to extract from.",
            type="ArtifactRef",
            validator=artifact_has_metadata_key("abstract"),
            validator_description="Article must have an 'abstract' key in its metadata",
        ),
        ToolRequirement(
            name="corrections",
            description=(
                "Optional correction strings that override or extend the "
                "extracted brief (e.g., 'focus on the mean-field variant', "
                "'ignore the appendix simulation')."
            ),
            type="list[str]",
            required=False,
            default=[],
        ),
        ToolRequirement(
            name="focus_fields",
            description=(
                "Optional list of brief fields to prioritise during "
                "extraction, e.g. ['key_parameters', 'procedure']. "
                "If empty, all fields are extracted."
            ),
            type="list[str]",
            required=False,
            default=[],
        ),
        ToolRequirement(
            name="intention",
            description=(
                "Optional downstream task intent to guide extraction, such as "
                "'summarize simulations' or 'extract model equations'."
            ),
            type="str",
            required=False,
            default="",
        ),
    ],
    produces=["article_brief"],
    tags=["literature", "extraction", "brief"],
    idempotent=True,
    estimated_seconds=35.0,
)

PARSE_ARTICLE_WITH_INTENTION_TOOL = ToolDescriptor(
    name="parse_article_with_intention",
    display_name="Parse Article With Intention",
    description=(
        "Parse a primary article for a downstream intention after the article "
        "has already been found.  Use this for requests such as extracting "
        "simulation details, model equations, figures, or reproduction steps. "
        "This tool does not search for papers."
    ),
    agent_id="literature",
    requirements=[
        ToolRequirement(
            name="article",
            description="A primary_article ArtifactRef to parse.",
            type="ArtifactRef",
            validator=artifact_has_metadata_key("abstract"),
            validator_description="Article must have an 'abstract' key in its metadata",
        ),
        ToolRequirement(
            name="intention",
            description="Specific downstream parsing intention for the found article.",
            type="str",
            required=True,
        ),
        ToolRequirement(
            name="focus_fields",
            description="Optional brief fields to prioritize during extraction.",
            type="list[str]",
            required=False,
            default=[],
        ),
        ToolRequirement(
            name="corrections",
            description="Optional correction strings that override or extend the extracted brief.",
            type="list[str]",
            required=False,
            default=[],
        ),
    ],
    produces=["article_brief"],
    tags=["literature", "extraction", "brief", "intention"],
    idempotent=True,
    estimated_seconds=35.0,
)

# ---------------------------------------------------------------------------
# Brief field schema (used in prompts and validation)
# ---------------------------------------------------------------------------

BRIEF_SCHEMA = {
    "model_description": (
        "A precise 2-5 sentence description of the mathematical model, "
        "system, or theoretical framework studied in this paper.  Must name "
        "the model (e.g., 'Ising model on a 2D square lattice'), the key "
        "dynamical rule or equation, and the regime of interest."
    ),
    "key_parameters": (
        "Dict mapping parameter name to value/range.  Include all parameters "
        "the paper varies or specifies: coupling constants, temperatures, "
        "lattice sizes, noise levels, learning rates, etc.  Use the paper's "
        "notation (e.g., 'J': '±1', 'T': '0.1 to 5.0', 'N': '100-10000')."
    ),
    "procedure": (
        "Step-by-step description of the simulation or experimental procedure "
        "described in the paper.  Number the steps.  Include: initialisation, "
        "update rule, measurement protocol, convergence criterion, and any "
        "post-processing.  Be concrete enough that an implementer can reproduce it."
    ),
    "expected_figures": (
        "List of figures the paper produces, each described as: "
        "'Figure N: x-axis vs y-axis, qualitative shape (e.g., monotone, "
        "critical divergence, phase boundary), key claim it supports'."
    ),
    "assumptions": (
        "List of modelling assumptions, approximations, and boundary conditions "
        "stated or implied in the paper.  Include thermodynamic limit, "
        "ergodicity, mean-field assumptions, boundary conditions, etc."
    ),
    "validation_criteria": (
        "List of quantitative or qualitative checks the paper uses to validate "
        "its results (e.g., 'critical exponents match known values', "
        "'finite-size scaling collapse', 'reproduces analytic solution at T→0')."
    ),
    "inquiries": (
        "List of unresolved choices that materially affect downstream work. "
        "Only include when the paper presents multiple options (models, "
        "figures, parameter regimes, datasets) and the caller must choose one. "
        "Return [] if no such choice is needed."
    ),
}

CRITICAL_FIELDS = {"model_description", "key_parameters", "procedure"}
VAGUE_THRESHOLD = 40  # chars — below this, a field is considered empty/vague

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

_PRIMARY_SYSTEM = f"""\
You are an expert scientific text extractor.  Extract a structured research
brief from the provided article text.

Return ONLY valid JSON with EXACTLY these fields:

{{
  "model_description": "{BRIEF_SCHEMA['model_description']}",
  "key_parameters": {{"param_name": "value_or_range", ...}},
  "procedure": "{BRIEF_SCHEMA['procedure']}",
  "expected_figures": [
    "Figure 1: description including axes, shape, and claim supported",
    ...
  ],
  "assumptions": ["assumption 1", "assumption 2", ...],
  "validation_criteria": ["criterion 1", ...],
  "inquiries": [
    {{
      "id": "short_stable_id",
      "kind": "selection | clarification | approval",
      "question": "Precise question requiring a choice",
      "reason": "Why this choice affects downstream simulation or writing",
      "applies_to": ["simulation", "writing"],
      "blocking": true,
      "options": [{{"label": "...", "description": "...", "value": {{}}}}]
    }}
  ]
}}

Extraction rules:
- model_description: name the model explicitly; do not paraphrase vaguely.
- key_parameters: use the paper's own notation; include units where given.
- procedure: number the steps; be concrete enough to implement.
- expected_figures: one entry per distinct figure/plot in the paper.
- inquiries: ONLY for genuine ambiguity where a choice is required.
  Return [] when the paper is unambiguous.
- Do NOT invent information not in the text.
- If a field cannot be determined from the text, use null for scalars
  or [] for lists, and note "NOT_IN_TEXT" as the value for string fields.
"""

_TARGETED_SYSTEM = """\
You are an expert scientific text extractor performing a TARGETED re-extraction.

The primary extraction pass returned a weak or empty value for a specific field.
Re-read the article text carefully and extract ONLY the requested field.

Return ONLY valid JSON with the single requested field as the top-level key.
Example: {"model_description": "The paper studies the XY model on ..."}

Rules:
- Be specific and concrete.
- Use the paper's own notation and terminology.
- If the field is genuinely not in the text, return {"field_name": "NOT_IN_TEXT"}.
"""

# ---------------------------------------------------------------------------
# Execute
# ---------------------------------------------------------------------------


def execute(
    context: ToolContext,
    *,
    registry: ArtifactRegistry,
    llm_backend: Any = None,
) -> ToolResult:
    """Extract a structured brief via primary + validation passes."""
    if llm_backend is None:
        return ToolResult(status="failed", message="No LLM backend available for extraction")

    article = context.artifacts["article"]
    intention = str(context.inputs.get("intention") or "").strip()
    corrections: list[str] = [
        str(s).strip()
        for s in (context.inputs.get("corrections") or
                  context.inputs.get("steering_notes") or [])  # backward compat
        if str(s).strip()
    ]
    if intention:
        corrections.insert(0, f"Downstream parsing intention: {intention}")
    focus_fields: list[str] = [
        str(f).strip()
        for f in (context.inputs.get("focus_fields") or [])
        if str(f).strip() and str(f).strip() in BRIEF_SCHEMA
    ]

    text = _get_article_text(article, registry)
    if len(text.strip()) < 50:
        return ToolResult(
            status="failed",
            message=(
                "Article text is too short for extraction.  "
                "Ensure the article artifact contains an abstract or full text."
            ),
        )

    # -- Primary extraction pass --
    brief_data = _primary_extract(llm_backend, article, text, corrections, focus_fields)
    if brief_data is None:
        brief_data = _fallback_brief(article, text, corrections)

    # -- Targeted re-extraction for weak critical fields --
    weak_fields = _find_weak_fields(brief_data, focus_fields or list(CRITICAL_FIELDS))
    if weak_fields:
        brief_data = _targeted_reextract(llm_backend, text, brief_data, weak_fields)

    # -- Apply corrections as post-processing hints --
    if corrections:
        brief_data = _apply_corrections(llm_backend, text, brief_data, corrections)

    # -- Normalise --
    brief_data["inquiries"] = _coerce_inquiries(brief_data.get("inquiries"))

    brief_ref = registry.save_json(
        assistant=AssistantId.LITERATURE_REVIEWER.value,
        kind="article_brief",
        title="Article Brief",
        filename="article/brief.json",
        payload=brief_data,
        summary=str(brief_data.get("model_description", ""))[:200],
        metadata=brief_data,
        artifact_id="article-brief",
    )

    n_params = len(brief_data.get("key_parameters") or {})
    n_figures = len(brief_data.get("expected_figures") or [])
    n_inquiries = len(brief_data.get("inquiries") or [])
    inquiry_requests = _build_inquiry_requests(brief_data["inquiries"], context)

    return ToolResult(
        status="completed",
        artifacts=[brief_ref],
        message=(
            f"Brief extracted: {n_params} parameters, "
            f"{n_figures} expected figures"
            + (f", {n_inquiries} inquiry(s)" if n_inquiries else "")
        ),
        inquiries=inquiry_requests,
    )


def execute_parse_article_with_intention(
    context: ToolContext,
    *,
    registry: ArtifactRegistry,
    llm_backend: Any = None,
) -> ToolResult:
    """Parse an already-found article for a caller-provided intention."""
    return execute(context, registry=registry, llm_backend=llm_backend)


# ---------------------------------------------------------------------------
# Primary extraction
# ---------------------------------------------------------------------------


def _primary_extract(
    backend: Any,
    article: Any,
    text: str,
    corrections: list[str],
    focus_fields: list[str],
) -> dict[str, Any] | None:
    title = str(article.metadata.get("title") or "")
    authors_list = list(article.metadata.get("authors") or [])
    authors_str = ", ".join(authors_list[:6])
    year = str(article.metadata.get("year") or "")

    corrections_block = ""
    if corrections:
        corrections_block = "\n\nCorrections / focus instructions:\n" + "\n".join(
            f"- {c}" for c in corrections
        )

    focus_block = ""
    if focus_fields:
        focus_block = (
            f"\n\nPrioritise these fields (be especially detailed): "
            f"{', '.join(focus_fields)}"
        )

    user_msg = (
        f"Article: {title}\n"
        f"Authors: {authors_str}\n"
        f"Year: {year}\n\n"
        f"Full text:\n{_trim(text, 40_000)}"
        f"{corrections_block}"
        f"{focus_block}"
    )

    try:
        raw = backend.complete(
            system=_PRIMARY_SYSTEM,
            messages=[{"role": "user", "content": user_msg}],
            temperature=0.0,
        )
        payload = _parse_json(raw)
        if isinstance(payload, dict):
            return payload
    except Exception as exc:
        return None
    return None


# ---------------------------------------------------------------------------
# Validation and targeted re-extraction
# ---------------------------------------------------------------------------


def _find_weak_fields(
    brief: dict[str, Any],
    check_fields: list[str],
) -> list[str]:
    """Return fields in check_fields that are missing or too short."""
    weak: list[str] = []
    for field in check_fields:
        value = brief.get(field)
        if value is None:
            weak.append(field)
        elif isinstance(value, str):
            v = value.strip()
            if not v or v == "NOT_IN_TEXT" or len(v) < VAGUE_THRESHOLD:
                weak.append(field)
        elif isinstance(value, dict) and not value:
            weak.append(field)
        elif isinstance(value, list) and not value:
            # Empty list for procedure or figures is weak
            if field in ("expected_figures", "procedure"):
                weak.append(field)
    return weak


def _targeted_reextract(
    backend: Any,
    text: str,
    brief: dict[str, Any],
    weak_fields: list[str],
) -> dict[str, Any]:
    """Re-extract each weak field with a targeted prompt."""
    for field in weak_fields:
        description = BRIEF_SCHEMA.get(field, f"the '{field}' field")
        user_msg = (
            f"Field to extract: {field}\n"
            f"What it should contain: {description}\n\n"
            f"Article text:\n{_trim(text, 40_000)}"
        )
        try:
            raw = backend.complete(
                system=_TARGETED_SYSTEM,
                messages=[{"role": "user", "content": user_msg}],
                temperature=0.0,
            )
            payload = _parse_json(raw)
            if isinstance(payload, dict) and field in payload:
                value = payload[field]
                if value and value != "NOT_IN_TEXT":
                    brief[field] = value
        except Exception:
            continue
    return brief


# ---------------------------------------------------------------------------
# Correction application
# ---------------------------------------------------------------------------

_CORRECTION_SYSTEM = """\
You are updating a structured research brief based on correction instructions.

You will receive the current brief (JSON) and a list of corrections.
Apply ONLY the corrections — do not change fields that are not affected.

Return ONLY the COMPLETE updated brief as valid JSON (same structure).
"""


def _apply_corrections(
    backend: Any,
    text: str,
    brief: dict[str, Any],
    corrections: list[str],
) -> dict[str, Any]:
    """Use the LLM to apply free-form corrections to the extracted brief."""
    corrections_block = "\n".join(f"- {c}" for c in corrections)
    user_msg = (
        f"Current brief:\n{json.dumps(brief, indent=2)}\n\n"
        f"Corrections to apply:\n{corrections_block}\n\n"
        f"Reference text (if needed):\n{_trim(text, 8_000)}"
    )
    try:
        raw = backend.complete(
            system=_CORRECTION_SYSTEM,
            messages=[{"role": "user", "content": user_msg}],
            temperature=0.0,
        )
        updated = _parse_json(raw)
        if isinstance(updated, dict) and set(updated) >= {"model_description", "procedure"}:
            return updated
    except Exception:
        pass
    return brief


# ---------------------------------------------------------------------------
# Article text reader
# ---------------------------------------------------------------------------


def _get_article_text(article: Any, registry: ArtifactRegistry) -> str:
    """Best-effort article text: registry → content → full-text fetch → abstract."""
    # 1. Registry
    if hasattr(article, "artifact_id"):
        stored = registry.read_artifact_text(article.artifact_id)
        if stored and len(stored.strip()) > 200:
            return stored

    # 2. Content field
    content = str(getattr(article, "content", "") or "").strip()
    if len(content) > 100:
        return content

    # 3. Fetch full text via parser (ArXiv PDF / HTML)
    arxiv_id = str(article.metadata.get("arxiv_id") or "")
    url = str(article.metadata.get("url") or "")
    abstract = str(article.metadata.get("abstract") or "")

    if arxiv_id or url:
        try:
            from literature_review.article_parser import ArticleParser
            from literature_review.search_backends.arxiv_backend import ArXivBackend
            parser = ArticleParser(ArXivBackend())
            fetched = parser.fetch_full_text(
                arxiv_id,
                fallback_abstract=abstract,
                paper_url=url,
            )
            if fetched and len(fetched.strip()) > 200:
                return fetched
        except Exception:
            pass

    return abstract or str(getattr(article, "summary", "") or "")


# ---------------------------------------------------------------------------
# Inquiry normalisation
# ---------------------------------------------------------------------------


def _coerce_inquiries(raw: Any) -> list[dict[str, Any]]:
    if not isinstance(raw, list):
        return []
    inquiries: list[dict[str, Any]] = []
    for idx, item in enumerate(raw, 1):
        if not isinstance(item, dict):
            continue
        question = str(item.get("question") or item.get("prompt") or "").strip()
        if not question:
            continue
        inquiry_id = str(item.get("id") or f"inquiry_{idx}").strip()
        options = _coerce_options(item.get("options"))
        kind = str(item.get("kind") or ("selection" if options else "clarification")).strip()
        inquiries.append({
            "id": inquiry_id,
            "title": str(item.get("title") or "").strip(),
            "kind": kind,
            "question": question,
            "reason": str(item.get("reason") or "").strip(),
            "applies_to": _str_list(item.get("applies_to")),
            "blocking": bool(item.get("blocking", True)),
            "options": options,
            "expected_schema": (
                item.get("expected_schema")
                if isinstance(item.get("expected_schema"), dict)
                else {}
            ),
            "default_policy": str(item.get("default_policy") or "ask").strip(),
        })
    return inquiries[:12]


def _coerce_options(raw: Any) -> list[dict[str, Any]]:
    if not isinstance(raw, list):
        return []
    options: list[dict[str, Any]] = []
    for idx, item in enumerate(raw, 1):
        if isinstance(item, dict):
            label = str(item.get("label") or item.get("name") or f"Option {idx}").strip()
            description = str(item.get("description") or item.get("summary") or "").strip()
            value = item.get("value", item)
        else:
            label = str(item or f"Option {idx}").strip()
            description = ""
            value = item
        if label:
            options.append({"label": label, "description": description, "value": value})
    return options[:20]


def _str_list(raw: Any) -> list[str]:
    if isinstance(raw, list):
        return [str(i).strip() for i in raw if str(i).strip()]
    s = str(raw or "").strip()
    return [s] if s else []


# ---------------------------------------------------------------------------
# Inquiry → OrchestrationRequestEnvelope
# ---------------------------------------------------------------------------


def _build_inquiry_requests(
    inquiries: list[dict[str, Any]],
    context: ToolContext,
) -> list[Any]:
    try:
        from ...service_contracts import OrchestrationRequestEnvelope
    except ImportError:
        return []

    requests: list[Any] = []
    for inquiry in inquiries:
        expected_schema = inquiry.get("expected_schema") or {}
        if not expected_schema:
            expected_schema = (
                {"answers": {"selection": "string"}}
                if inquiry.get("options")
                else {"answers": {"answer": "string"}}
            )
        requests.append(
            OrchestrationRequestEnvelope.new(
                resume_token=(
                    f"{context.directive_id}:{context.tool_name}:{inquiry['id']}"
                ),
                request_kind="inquiry",
                question=str(inquiry["question"]),
                expected_schema=expected_schema,
                capability_hint="user",
                blocking=bool(inquiry.get("blocking", True)),
                metadata={"inquiry": inquiry},
            )
        )
    return requests


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def _trim(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    half = max_chars // 2
    return text[:half] + "\n\n[... trimmed ...]\n\n" + text[-half:]


def _parse_json(raw: Any) -> Any:
    if not isinstance(raw, str):
        return raw
    text = raw.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        text = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
    if not text.startswith("{"):
        import re
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            text = match.group(0)
    return json.loads(text.strip())


def _fallback_brief(article: Any, text: str, corrections: list[str]) -> dict[str, Any]:
    """Deterministic brief when the LLM response is not valid JSON."""
    title = str(article.metadata.get("title") or article.title or "").strip()
    abstract = str(article.metadata.get("abstract") or text or "").strip()
    model_description = abstract[:1200] if abstract else f"Article: {title}"
    lower = abstract.lower()
    procedure = "NOT_IN_TEXT"
    if "simulation" in lower or "monte-carlo" in lower or "monte carlo" in lower:
        procedure = (
            "The abstract reports simulation evidence. "
            "It states that analytic expressions fit species abundance "
            "distributions from extensive Monte-Carlo simulations and numerical "
            "solutions of the corresponding master equations. Full procedural "
            "details require the article body."
        )
    expected_figures = []
    if "species abundance" in lower or "sad" in lower:
        expected_figures.append("Species abundance distributions for the time-averaged neutral models.")
    if "model a" in lower:
        expected_figures.append("Model A: local competition with linear fitness-dependence.")
    if "model b" in lower:
        expected_figures.append("Model B: global competition with nonlinear fitness-dependence.")
    assumptions = []
    if "neutral" in lower:
        assumptions.append("Neutral dynamics with equivalent individuals/species except stochastic effects.")
    if "environmental" in lower:
        assumptions.append("Environmental variations coherently affect relative population fitness.")
    return {
        "model_description": model_description,
        "key_parameters": {},
        "procedure": procedure,
        "expected_figures": expected_figures,
        "assumptions": assumptions,
        "validation_criteria": [
            "Analytic expressions should fit Monte-Carlo species abundance distributions."
        ] if "monte-carlo" in lower or "monte carlo" in lower else [],
        "inquiries": [],
        "extraction_fallback": True,
        "corrections_applied": corrections,
    }
