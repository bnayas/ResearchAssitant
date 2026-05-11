"""
literature.prepare_lookup
─────────────────────────
Tool: Analyze a research question and produce a structured, multi-strategy
ArXiv search plan.

This tool is intentionally agnostic about caller identity.  It accepts any
research text — a question, an abstract, a paper title, a free-form
description — and returns a self-contained search plan that other tools can
execute without any further context.

ArXiv field syntax reference (used in generated query_string values):
  ti:     title
  au:     author surname (e.g. au:Kingma)
  abs:    abstract keywords
  cat:    subject category (e.g. cat:cs.LG)
  id:     specific paper ID
  all:    all fields (fallback)
  Booleans: AND  OR  ANDNOT  (uppercase, space-separated)
  Phrases:  "quoted phrase"
  Date:     submittedDate:[YYYYMMDD0000 TO YYYYMMDD2359]
"""
from __future__ import annotations

import json
import re
from typing import Any

from ...article_lookup import build_article_lookup_query
from ...contracts import AssistantId
from ...registry import ArtifactRegistry
from ..base import ToolContext, ToolDescriptor, ToolRequirement, ToolResult
from ..validators import is_non_empty_string

# ---------------------------------------------------------------------------
# ArXiv category map — used to ground LLM category suggestions
# ---------------------------------------------------------------------------

ARXIV_CS_CATS = {
    "cs.LG", "cs.AI", "cs.CV", "cs.CL", "cs.NE", "cs.RO", "cs.SY",
    "cs.IT", "cs.DC", "cs.DS", "cs.NA",
}
ARXIV_MATH_CATS = {
    "math.OC", "math.ST", "math.PR", "math.NA", "math.DS",
}
ARXIV_PHYS_CATS = {
    "cond-mat.str-el", "cond-mat.dis-nn", "physics.comp-ph",
    "quant-ph", "hep-th", "hep-lat", "gr-qc",
}
ARXIV_BIO_CATS = {
    "q-bio.QM", "q-bio.GN", "q-bio.NC", "q-bio.PE",
}
ARXIV_ECON_CATS = {"econ.GN", "econ.TH", "stat.ML", "stat.AP"}
KNOWN_CATS: set[str] = (
    ARXIV_CS_CATS | ARXIV_MATH_CATS | ARXIV_PHYS_CATS
    | ARXIV_BIO_CATS | ARXIV_ECON_CATS
)

# ---------------------------------------------------------------------------
# Tool descriptor
# ---------------------------------------------------------------------------

TOOL = ToolDescriptor(
    name="prepare_article_lookup",
    display_name="Prepare Article Lookup",
    description=(
        "Analyze a research question or text and produce a structured, "
        "multi-strategy ArXiv search plan.  The plan contains several "
        "query specs (each with title terms, abstract keywords, author "
        "filters, category filters, and date ranges) that the "
        "find_primary_article tool can execute.  Also identifies what "
        "kind of article is sought and what to look for in results."
    ),
    agent_id="literature",
    requirements=[
        ToolRequirement(
            name="instruction",
            description=(
                "Research text to analyze.  Can be a question, a free-form "
                "description, an abstract, a paper title, or any combination."
            ),
            type="str",
            validator=is_non_empty_string,
            validator_description="Must be a non-empty string",
        ),
        ToolRequirement(
            name="query",
            description="Optional alias for instruction; used by direct callers.",
            type="str",
            required=False,
            default="",
        ),
        ToolRequirement(
            name="topic_hint",
            description="Optional caller-supplied topic hint used as steering, not as a required search term.",
            type="str",
            required=False,
            default="",
        ),
        ToolRequirement(
            name="known_paper_ids",
            description=(
                "ArXiv IDs or DOIs already in the literature cache.  "
                "The planner will embed these in the plan so downstream "
                "tools skip re-fetching them."
            ),
            type="list[str]",
            required=False,
            default=[],
        ),
        ToolRequirement(
            name="target_count",
            description="Approximate number of relevant articles desired.",
            type="int",
            required=False,
            default=10,
        ),
        ToolRequirement(
            name="steering",
            description=(
                "Optional list of free-form correction notes that further "
                "constrain or redirect the search plan."
            ),
            type="list[str]",
            required=False,
            default=[],
        ),
    ],
    produces=["article_lookup_spec"],
    tags=["literature", "search", "planning"],
    idempotent=True,
    estimated_seconds=8.0,
)

# ---------------------------------------------------------------------------
# LLM prompt
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """\
You are an expert academic search strategist.  You will receive a research
text — this may be a question, a paper abstract, a title, or a description —
and you must produce a comprehensive, ArXiv-native search plan.

First split the request into article lookup fields and downstream work.
Search queries identify papers; they must not include action phrases for what
the caller wants to do after the paper is found.

Return ONLY valid JSON matching this schema exactly:

{
  "research_intent": "One sentence: what article(s) this search is trying to find",
  "article_target": "Only the article-identifying target: title/model/topic/authors/year",
  "task_intent": "Downstream work requested after the article is found",
  "article_type": "primary_source | survey | methodology | dataset | benchmark | any",
  "required_authors": ["Surname1", "Surname2"],
  "year_constraints": {"preferred_year": null, "year_from": null, "year_to": null},
  "title_phrases": ["near-exact title phrase"],
  "topic_terms": ["paper-identifying model or topic term"],
  "excluded_search_terms": ["instruction/action phrase removed from search"],
  "scope_criteria": [
    "Criterion a result must satisfy to be considered relevant",
    "..."
  ],
  "what_to_look_for": "What the caller should check once articles are found",
  "queries": [
    {
      "label": "Short human-readable label for this query strategy",
      "rationale": "Why this query is useful",
      "title_terms": ["exact or near-exact title words or phrases"],
      "abstract_keywords": ["topic keywords likely in abstract"],
      "authors": ["Surname1", "Surname2"],
      "categories": ["cat:cs.LG", "cat:cs.AI"],
      "year_from": null,
      "year_to": null,
      "sort_by": "relevance | submittedDate",
      "max_results": 15,
      "query_string": "ArXiv API query string built from the fields above"
    }
  ],
  "known_ids_to_exclude": [],
  "fallback_query": "Simple all-field query to use if structured queries fail"
}

Rules for query_string construction:
- Use field prefixes: ti: au: abs: cat:
- Booleans MUST be uppercase: AND OR ANDNOT
- Phrase search: ti:\"exact phrase\"
- Multiple authors: (au:Smith OR au:Jones)
- Multiple categories: (cat:cs.LG OR cat:cs.AI)
- Date range: submittedDate:[20170101000 TO 20231231235]
- Combine: ti:\"attention\" AND abs:transformer AND (cat:cs.LG OR cat:cs.CL)
- Do NOT invent author names; only include authors mentioned in the input.
- Do NOT put downstream tasks such as read, summarize, reproduce, describe,
  implement, simulate, or details in topic terms or queries unless they are
  literally part of a paper title or model name.
- Generate 2 to 5 distinct queries with different strategies (e.g., one
  title-focused, one abstract-keyword-focused, one author+topic).
- Make max_results proportional to specificity: title+author queries = 5,
  broad keyword queries = 20.
"""

_REVISION_PROMPT = """\
Revise an academic article lookup plan using search telemetry.

Return ONLY valid JSON with the same schema as the original lookup plan.
Keep any grounded required_authors and explicit year constraints unless the
telemetry proves they were not grounded.  Do not include downstream task
phrases as search terms.  Prefer simpler, broader structured fields when raw
results were zero; prefer stricter authors/title fields when noisy results had
wrong authors.
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
    """Analyze research text and produce a structured ArXiv search plan."""
    query = _resolve_query(context)
    known_ids: list[str] = list(context.inputs.get("known_paper_ids") or [])
    target_count: int = int(context.inputs.get("target_count") or 10)
    steering: list[str] = [
        str(s).strip()
        for s in (context.inputs.get("steering") or [])
        if str(s).strip()
    ]
    topic_hint = str(context.inputs.get("topic_hint") or "").strip()
    if topic_hint:
        steering.append(f"Topic hint: {topic_hint}")

    if llm_backend is None:
        plan = _fallback_plan(query, known_ids)
    else:
        plan = _llm_plan(llm_backend, query, known_ids, target_count, steering)

    # Validate and normalise the plan
    plan = _normalise_plan(plan, query, known_ids)

    # Check if the LLM flagged genuine ambiguity
    if plan.get("needs_clarification"):
        return _clarification_result(plan, context)

    spec_ref = registry.save_json(
        assistant=AssistantId.LITERATURE_REVIEWER.value,
        kind="article_lookup_spec",
        title="Article Lookup Plan",
        filename="article/lookup_spec.json",
        payload=plan,
        summary=plan.get("research_intent", query[:120]),
        metadata=plan,
        artifact_id="article-lookup-spec",
    )
    n_queries = len(plan.get("queries") or [])
    return ToolResult(
        status="completed",
        artifacts=[spec_ref],
        message=f"Search plan ready: {n_queries} query strategies — {plan.get('research_intent', '')}",
        metadata={"lookup_spec": plan},
    )


# ---------------------------------------------------------------------------
# LLM planning
# ---------------------------------------------------------------------------


def _llm_plan(
    backend: Any,
    query: str,
    known_ids: list[str],
    target_count: int,
    steering: list[str],
) -> dict[str, Any]:
    steering_block = ""
    if steering:
        steering_block = "\n\nAdditional constraints:\n" + "\n".join(
            f"- {s}" for s in steering
        )
    if known_ids:
        steering_block += (
            f"\n\nAlready known paper IDs (exclude from results): "
            f"{', '.join(known_ids[:20])}"
        )

    user_msg = (
        f"Research text:\n{query}\n\n"
        f"Target: approximately {target_count} relevant articles."
        f"{steering_block}"
    )
    try:
        raw = backend.complete(
            system=_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_msg}],
            temperature=0.0,
        )
        payload = json.loads(raw) if isinstance(raw, str) else raw
        if isinstance(payload, dict):
            payload["known_ids_to_exclude"] = list(
                set(list(payload.get("known_ids_to_exclude") or []) + known_ids)
            )
            return payload
    except Exception:
        pass
    return _fallback_plan(query, known_ids)


# ---------------------------------------------------------------------------
# Fallback (no LLM)
# ---------------------------------------------------------------------------


def _fallback_plan(query: str, known_ids: list[str]) -> dict[str, Any]:
    """Build a minimal search plan without an LLM."""
    base_query = build_article_lookup_query("", query)
    authors = _extract_surnames(query)
    year = _extract_year(query)
    excluded_terms = _instruction_terms(query)
    topic_phrases = _expand_topic_phrases(
        _extract_topic_phrases(query, authors, excluded_terms)
    )
    keywords = _filter_search_terms(
        [*topic_phrases, *[p.strip() for p in base_query.split(",") if p.strip()]],
        excluded_terms,
    )
    author_keys = {author.lower() for author in authors}
    keywords = [
        term for term in keywords
        if term.lower() not in author_keys
        and not any(re.search(rf"\b{re.escape(author.lower())}\b", term.lower()) for author in authors)
    ]

    queries: list[dict[str, Any]] = []

    # Query 1: broad article-identifying phrase with definite authors.
    title_terms = topic_phrases[:2]
    kw_query = _build_query_string({
        "all_terms": title_terms[:1],
        "abstract_keywords": [],
        "authors": authors[:2],
        "year_from": year,
        "year_to": year,
    }) or f'all:"{_strip_instruction_terms(query)[:80]}"'
    queries.append({
        "label": "Title/topic search",
        "rationale": "Match the article-identifying title or model terms.",
        "all_terms": title_terms[:1],
        "title_terms": title_terms[:1],
        "abstract_keywords": keywords[:6],
        "authors": authors[:2],
        "categories": [],
        "year_from": year,
        "year_to": year,
        "sort_by": "relevance",
        "max_results": 20,
        "query_string": kw_query,
    })

    # Query 2: author + keyword (only if authors found)
    if authors:
        au_part = " AND ".join(f'au:"{a}"' for a in authors[:2])
        au_query = f"({au_part})"
        if keywords:
            au_query += f' AND abs:"{keywords[0]}"'
        queries.append({
            "label": "Author + topic search",
            "rationale": "Narrow to known authors working on this topic",
            "title_terms": title_terms[:1],
            "abstract_keywords": keywords[:3],
            "authors": authors[:2],
            "categories": [],
            "year_from": year,
            "year_to": year,
            "sort_by": "relevance",
            "max_results": 10,
            "query_string": au_query,
        })

    return {
        "research_intent": f"Find articles about: {query[:100]}",
        "article_target": _strip_instruction_terms(query),
        "task_intent": "; ".join(excluded_terms),
        "article_type": "any",
        "required_authors": authors[:4],
        "year_constraints": {"preferred_year": year, "year_from": year, "year_to": year},
        "title_phrases": topic_phrases[:4],
        "topic_terms": keywords[:8],
        "excluded_search_terms": excluded_terms,
        "scope_criteria": [f"Related to: {query[:80]}"],
        "what_to_look_for": "Relevance to the stated research question",
        "queries": queries,
        "known_ids_to_exclude": known_ids,
        "fallback_query": base_query,
        "needs_clarification": False,
    }


# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------


def _normalise_plan(plan: dict[str, Any], query: str, known_ids: list[str]) -> dict[str, Any]:
    """Validate and normalise a plan dict produced by the LLM."""
    fallback = _fallback_plan(query, known_ids)
    excluded_search_terms = _dedupe_strings(
        _str_list(plan.get("excluded_search_terms"))
        or _instruction_terms(query)
    )
    required_authors = _dedupe_strings(
        _str_list(plan.get("required_authors"))
        or _authors_from_queries(plan.get("queries"))
        or list(fallback.get("required_authors") or [])
    )[:4]
    year_constraints = plan.get("year_constraints") if isinstance(plan.get("year_constraints"), dict) else {}
    preferred_year = _coerce_int(
        year_constraints.get("preferred_year")
        or plan.get("preferred_year")
        or fallback.get("year_constraints", {}).get("preferred_year")
    )
    year_from = _coerce_int(year_constraints.get("year_from") or plan.get("year_from") or preferred_year)
    year_to = _coerce_int(year_constraints.get("year_to") or plan.get("year_to") or preferred_year)
    title_phrases = _filter_search_terms(
        _str_list(plan.get("title_phrases")) or _titles_from_queries(plan.get("queries")),
        excluded_search_terms,
    )[:6]
    topic_terms = _filter_search_terms(
        _str_list(plan.get("topic_terms")) or list(fallback.get("topic_terms") or []),
        excluded_search_terms,
    )[:12]

    queries_raw = plan.get("queries")
    if not isinstance(queries_raw, list) or not queries_raw:
        queries_raw = fallback["queries"]

    queries: list[dict[str, Any]] = []
    for q in queries_raw:
        if not isinstance(q, dict):
            continue
        query_spec = dict(q)
        query_spec["authors"] = _dedupe_strings(
            _str_list(query_spec.get("authors")) or required_authors
        )[:4]
        query_spec["title_terms"] = _filter_search_terms(
            _str_list(query_spec.get("title_terms")),
            excluded_search_terms,
        )
        query_spec["all_terms"] = _filter_search_terms(
            _str_list(query_spec.get("all_terms")),
            excluded_search_terms,
        )
        query_spec["abstract_keywords"] = _filter_search_terms(
            _str_list(query_spec.get("abstract_keywords")),
            excluded_search_terms,
        )
        if year_from and query_spec.get("year_from") is None:
            query_spec["year_from"] = year_from
        if year_to and query_spec.get("year_to") is None:
            query_spec["year_to"] = year_to
        # Structured fields are authoritative; use LLM-authored query_string
        # only when no structured fields can produce a query.
        qs = _build_query_string(query_spec) or str(q.get("query_string") or "").strip()
        if not qs:
            continue
        # Validate categories
        raw_cats = q.get("categories") or []
        cats = [c for c in raw_cats if _valid_arxiv_cat(c)]
        queries.append({
            "label": str(q.get("label") or "Search"),
            "rationale": str(q.get("rationale") or ""),
            "title_terms": _str_list(query_spec.get("title_terms")),
            "all_terms": _str_list(query_spec.get("all_terms")),
            "abstract_keywords": _str_list(query_spec.get("abstract_keywords")),
            "authors": _str_list(query_spec.get("authors")),
            "categories": cats,
            "year_from": _coerce_int(query_spec.get("year_from")),
            "year_to": _coerce_int(query_spec.get("year_to")),
            "sort_by": str(q.get("sort_by") or "relevance"),
            "max_results": max(1, min(50, int(q.get("max_results") or 15))),
            "query_string": qs,
        })

    if not queries:
        queries = fallback["queries"]

    if required_authors:
        author_query = " AND ".join(f'au:"{author}"' for author in required_authors[:3])
        if author_query and author_query not in {q["query_string"] for q in queries}:
            queries.append({
                "label": "Author-only fallback",
                "rationale": "Relax keywords while preserving definite author constraints.",
                "title_terms": [],
                "abstract_keywords": [],
                "authors": required_authors[:3],
                "categories": [],
                "year_from": year_from,
                "year_to": year_to,
                "sort_by": "relevance",
                "max_results": 20,
                "query_string": author_query,
            })

    excl = list(set(list(plan.get("known_ids_to_exclude") or []) + known_ids))

    return {
        "research_intent": str(plan.get("research_intent") or query[:120]),
        "article_target": str(plan.get("article_target") or fallback.get("article_target") or query[:120]),
        "task_intent": str(plan.get("task_intent") or fallback.get("task_intent") or ""),
        "article_type": str(plan.get("article_type") or "any"),
        "required_authors": required_authors,
        "year_constraints": {
            "preferred_year": preferred_year,
            "year_from": year_from,
            "year_to": year_to,
        },
        "title_phrases": title_phrases,
        "topic_terms": topic_terms,
        "excluded_search_terms": excluded_search_terms,
        "scope_criteria": _str_list(plan.get("scope_criteria")),
        "what_to_look_for": str(plan.get("what_to_look_for") or ""),
        "queries": queries,
        "known_ids_to_exclude": excl,
        "fallback_query": str(plan.get("fallback_query") or query[:120]),
        "needs_clarification": bool(plan.get("needs_clarification")),
        "clarification_question": str(plan.get("clarification_question") or "").strip(),
    }


def _build_query_string(spec: dict[str, Any]) -> str:
    """Construct an ArXiv query string from spec fields when the LLM omitted one."""
    parts: list[str] = []
    all_terms = _str_list(spec.get("all_terms"))
    for term in all_terms[:2]:
        parts.append(f'all:"{term}"')
    if not all_terms:
        title_terms = _str_list(spec.get("title_terms"))
        if title_terms:
            phrase = title_terms[0]
            parts.append(f'ti:"{phrase}"')
        abs_kws = _str_list(spec.get("abstract_keywords"))
        if abs_kws:
            abs_part = " AND ".join(f'abs:"{kw}"' for kw in abs_kws[:3])
            parts.append(abs_part)
    authors = _str_list(spec.get("authors"))
    if authors:
        au_part = " AND ".join(f'au:"{a}"' for a in authors[:3])
        parts.append(au_part)
    cats = [c for c in (spec.get("categories") or []) if _valid_arxiv_cat(c)]
    if cats:
        cat_part = " OR ".join(cats[:3])
        parts.append(f"({cat_part})")
    year_from = _coerce_int(spec.get("year_from"))
    year_to = _coerce_int(spec.get("year_to"))
    if year_from and year_to:
        parts.append(
            f"submittedDate:[{year_from}01010000 TO {year_to}12312359]"
        )
    return " AND ".join(parts)


def revise_lookup_plan(
    backend: Any,
    *,
    original_query: str,
    previous_plan: dict[str, Any],
    telemetry: dict[str, Any],
    known_ids: list[str] | None = None,
) -> dict[str, Any]:
    """Ask the LLM for a better structured plan after search telemetry."""
    if backend is None:
        return _normalise_plan(previous_plan, original_query, known_ids or [])
    try:
        raw = backend.complete(
            system=_REVISION_PROMPT,
            messages=[{
                "role": "user",
                "content": (
                    f"Original request:\n{original_query}\n\n"
                    f"Previous lookup JSON:\n{json.dumps(previous_plan, ensure_ascii=True)}\n\n"
                    f"Search telemetry:\n{json.dumps(telemetry, ensure_ascii=True)}"
                ),
            }],
            temperature=0.0,
        )
        payload = json.loads(raw) if isinstance(raw, str) else raw
        if isinstance(payload, dict):
            payload["required_authors"] = (
                _str_list(previous_plan.get("required_authors"))
                or _str_list(payload.get("required_authors"))
            )
            payload["year_constraints"] = _merge_year_constraints(
                previous_plan.get("year_constraints"), payload.get("year_constraints")
            )
            return _normalise_plan(payload, original_query, known_ids or [])
    except Exception:
        pass
    return _normalise_plan(previous_plan, original_query, known_ids or [])


# ---------------------------------------------------------------------------
# Clarification path
# ---------------------------------------------------------------------------


def _clarification_result(plan: dict[str, Any], context: ToolContext) -> ToolResult:
    try:
        from ...service_contracts import OrchestrationRequestEnvelope
        req = OrchestrationRequestEnvelope.new(
            resume_token=context.tool_name,
            request_kind="clarification",
            question=plan.get("clarification_question", "Please clarify the research question."),
            expected_schema={"answers": {"clarification": "string"}},
            capability_hint="user",
            metadata={"lookup_spec": plan},
        )
    except Exception:
        req = None
    return ToolResult(
        status="needs_input",
        message=plan.get("clarification_question", "Please clarify the research question."),
        request=req,
        metadata={"lookup_spec": plan},
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _resolve_query(context: ToolContext) -> str:
    """Accept 'query', 'instruction', or context.instruction for backward compat."""
    return (
        str(context.inputs.get("query") or "").strip()
        or str(context.inputs.get("instruction") or "").strip()
        or str(getattr(context, "instruction", "") or "").strip()
    )


def _str_list(raw: Any) -> list[str]:
    if not isinstance(raw, list):
        return []
    return [str(item).strip() for item in raw if str(item).strip()]


def _dedupe_strings(values: list[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = str(value or "").strip()
        key = text.lower()
        if text and key not in seen:
            seen.add(key)
            result.append(text)
    return result


def _authors_from_queries(raw: Any) -> list[str]:
    authors: list[str] = []
    if isinstance(raw, list):
        for item in raw:
            if isinstance(item, dict):
                authors.extend(_str_list(item.get("authors")))
    return _dedupe_strings(authors)


def _titles_from_queries(raw: Any) -> list[str]:
    titles: list[str] = []
    if isinstance(raw, list):
        for item in raw:
            if isinstance(item, dict):
                titles.extend(_str_list(item.get("title_terms")))
                titles.extend(_str_list(item.get("all_terms")))
    return _dedupe_strings(titles)


def _filter_search_terms(terms: list[str], excluded_terms: list[str]) -> list[str]:
    excluded = {term.lower().strip() for term in excluded_terms}
    filtered: list[str] = []
    for term in terms:
        text = str(term or "").strip()
        if not text:
            continue
        key = text.lower()
        if key in excluded:
            continue
        if any(key == ex or key in ex for ex in excluded if len(ex.split()) >= 2):
            continue
        filtered.append(text)
    return _dedupe_strings(filtered)


def _instruction_terms(text: str) -> list[str]:
    action_words = {
        "read", "summarize", "summary", "describe", "reproduce", "reproduction",
        "simulate", "simulation", "simulations", "implement", "details", "answer",
        "compare", "explain", "extract", "parse",
    }
    found: list[str] = []
    lowered = text.lower()
    for word in action_words:
        if re.search(rf"\b{re.escape(word)}\b", lowered):
            found.append(word)
    for fragment in re.split(r"[-.;\n]+", text):
        words = [
            word
            for word in re.findall(r"[A-Za-z][A-Za-z-]{2,}", fragment.lower())
            if word in action_words
        ]
        if words:
            found.append(" ".join(words))
    return _dedupe_strings(found)


def _strip_instruction_terms(text: str) -> str:
    result = str(text or "")
    for term in _instruction_terms(text):
        if len(term.split()) == 1:
            result = re.sub(rf"\b{re.escape(term)}\b", "", result, flags=re.IGNORECASE)
    return re.sub(r"\s+", " ", result).strip(" -:;")


def _merge_year_constraints(previous: Any, revised: Any) -> dict[str, Any]:
    prev = previous if isinstance(previous, dict) else {}
    rev = revised if isinstance(revised, dict) else {}
    return {
        "preferred_year": prev.get("preferred_year") or rev.get("preferred_year"),
        "year_from": prev.get("year_from") or rev.get("year_from"),
        "year_to": prev.get("year_to") or rev.get("year_to"),
    }


def _coerce_int(value: Any) -> int | None:
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _valid_arxiv_cat(cat: str) -> bool:
    cat = str(cat or "").strip()
    if not cat:
        return False
    # Strip "cat:" prefix if present
    bare = cat.removeprefix("cat:")
    return bare in KNOWN_CATS or bool(re.match(r"^[a-z\-]+\.[A-Z]{2,}$", bare))


def _extract_surnames(text: str) -> list[str]:
    surnames: list[str] = []
    for m in re.finditer(
        r"\b([A-Z][A-Za-z'`\-]{2,})\s+(?:and|&|,)\s+([A-Z][A-Za-z'`\-]{2,})\b",
        text,
    ):
        surnames.extend([m.group(1), m.group(2)])
    return list(dict.fromkeys(surnames))[:4]


def _extract_year(text: str) -> int | None:
    years = [int(y) for y in re.findall(r"\b(19\d{2}|20\d{2})\b", text)]
    return years[0] if years else None


def _extract_topic_phrases(text: str, authors: list[str], excluded_terms: list[str]) -> list[str]:
    phrases: list[str] = []
    for match in re.finditer(r"\babout\s+([^\n.;]+?)(?:\s+-|$)", text, re.IGNORECASE):
        _append_phrase(phrases, match.group(1))
    for phrase in re.findall(r"\(([^()]{2,40})\)", text):
        _append_phrase(phrases, phrase)
    for phrase in re.findall(r'"([^"]{3,120})"', text):
        _append_phrase(phrases, phrase)
    author_keys = {author.lower() for author in authors}
    excluded = {term.lower() for term in excluded_terms}
    cleaned: list[str] = []
    for phrase in phrases:
        words = [
            word
            for word in re.findall(r"[A-Za-z][A-Za-z-]{1,}", phrase)
            if word.lower() not in author_keys and word.lower() not in excluded
        ]
        if words:
            value = " ".join(words)
            if value.lower() not in {item.lower() for item in cleaned}:
                cleaned.append(value)
    return cleaned[:6]


def _expand_topic_phrases(phrases: list[str]) -> list[str]:
    expanded: list[str] = []
    generic_tail = {"model", "models", "method", "methods", "approach", "approaches", "paper", "article"}
    for phrase in phrases:
        words = re.findall(r"[A-Za-z][A-Za-z-]{1,}", phrase)
        while words and (words[-1].isupper() or words[-1].lower() in generic_tail):
            words = words[:-1]
        if len(words) >= 2:
            _append_phrase(expanded, " ".join(words))
        _append_phrase(expanded, phrase)
    return expanded[:8]


def _append_phrase(values: list[str], phrase: str) -> None:
    text = re.sub(r"\s+", " ", str(phrase or "").strip(" -:;,."))
    if text and text.lower() not in {item.lower() for item in values}:
        values.append(text)
