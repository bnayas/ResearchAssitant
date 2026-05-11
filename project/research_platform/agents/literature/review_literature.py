"""
literature.review_literature
────────────────────────────
Tool: Multi-round academic literature search with a literature cache,
plan-driven ArXiv queries, per-paper scope validation, and synthesis.

Design principles:
  - Driven by the structured plan from prepare_article_lookup; does not
    require a "directive" or "PI instruction" to operate.
  - Maintains a literature cache (list of already-known paper dicts) so
    repeated calls avoid re-fetching known papers.
  - Each search round generates query variations from accepted papers,
    expanding coverage progressively.
  - Scope validation is a deterministic LLM call against explicit criteria
    from the lookup plan, not a vague relevance guess.
  - Synthesis is a structured pass: gap analysis, agreements,
    contradictions, open questions.
"""
from __future__ import annotations

import json
import logging
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from typing import Any

from ...contracts import AssistantId
from ...registry import ArtifactRegistry
from ..base import ToolContext, ToolDescriptor, ToolRequirement, ToolResult
from ..validators import artifact_kind_is

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Tool descriptor
# ---------------------------------------------------------------------------

TOOL = ToolDescriptor(
    name="review_related_literature",
    display_name="Review Related Literature",
    description=(
        "Execute the search plan from prepare_article_lookup against arXiv "
        "to find related papers.  Maintains a literature cache to avoid "
        "redundant fetches across calls.  Runs multiple search rounds with "
        "LLM-refined queries.  Validates each paper's scope against the "
        "plan's explicit criteria.  Produces a literature_review artifact "
        "with accepted papers and a structured synthesis."
    ),
    agent_id="literature",
    requirements=[
        ToolRequirement(
            name="article",
            description="Primary article ArtifactRef (the paper being studied).",
            type="ArtifactRef",
            validator=artifact_kind_is("primary_article"),
            validator_description="Must be a 'primary_article' artifact",
        ),
        ToolRequirement(
            name="brief",
            description="Article brief ArtifactRef produced by build_article_brief.",
            type="ArtifactRef",
            validator=artifact_kind_is("article_brief"),
            validator_description="Must be an 'article_brief' artifact",
        ),
        ToolRequirement(
            name="lookup_spec",
            description=(
                "Search plan from prepare_article_lookup.  If omitted, "
                "the tool builds a minimal plan from the article metadata."
            ),
            type="dict",
            required=False,
            default={},
        ),
        ToolRequirement(
            name="known_papers",
            description=(
                "Literature cache: list of paper dicts already reviewed "
                "in a previous call.  Papers whose arxiv_id or doi appears "
                "here will not be re-fetched or re-validated."
            ),
            type="list",
            required=False,
            default=[],
        ),
        ToolRequirement(
            name="max_papers",
            description="Maximum related papers to accept into the review.",
            type="int",
            required=False,
            default=12,
        ),
        ToolRequirement(
            name="max_rounds",
            description="Maximum search rounds (each round refines queries).",
            type="int",
            required=False,
            default=3,
        ),
        ToolRequirement(
            name="scope_strictness",
            description=(
                "How strictly scope validation filters papers.  "
                "'strict' = must clearly satisfy all criteria; "
                "'normal' = majority of criteria; "
                "'loose' = any thematic overlap."
            ),
            type="str",
            required=False,
            default="normal",
        ),
    ],
    produces=["literature_review"],
    tags=["literature", "search", "synthesis", "cache"],
    idempotent=False,
    estimated_seconds=90.0,
)

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

_QUERY_GEN_SYSTEM = """\
You are an academic search strategist.  Given a primary paper and a list of
papers already found, generate NEW arXiv search queries that will find
additional related work not yet captured.

Rules:
- Generate 2 to 4 distinct queries.
- Each query must use arXiv field syntax: ti: au: abs: cat:
- Booleans uppercase: AND OR ANDNOT
- Diversify: try different angles (method names, application domains,
  theoretical foundations, benchmark datasets, competing approaches).
- Do NOT repeat queries already tried.

Return ONLY a JSON array of strings (the query strings):
["ti:\"...\\" AND abs:...", "abs:\"...\" AND cat:cs.LG", ...]
"""

_SCOPE_SYSTEM = """\
You are an academic relevance judge.  Decide whether a candidate paper
satisfies the scope criteria for a literature review.

You will receive:
- The primary paper's title and model description.
- Explicit scope criteria (from the search plan).
- The candidate paper's title and abstract.
- A strictness level: strict | normal | loose.

Return ONLY a JSON object:
{
  "accept": true | false,
  "confidence": 0.0-1.0,
  "criteria_met": ["criterion text", ...],
  "reason": "one concise sentence"
}

Keep the response minimal.

For strict: all criteria must be met.
For normal: majority of criteria must be met.
For loose:  any meaningful thematic overlap.
"""

_SYNTHESIS_SYSTEM = """\
You are an expert academic writer producing a structured literature review
synthesis section.

You will receive the primary paper's context and a list of accepted related
papers.  Produce a structured synthesis covering:

1. Thematic clusters (group papers by sub-topic or method)
2. Agreements with the primary paper's approach
3. Contradictions or open debates
4. Gaps the primary paper addresses
5. Open questions that remain

Return ONLY valid JSON:
{
  "clusters": [
    {"label": "...", "paper_titles": [...], "summary": "..."}
  ],
  "agreements": ["..."],
  "contradictions": ["..."],
  "gaps_addressed": ["..."],
  "open_questions": ["..."],
  "prose_summary": "A 2-4 paragraph narrative synthesis suitable for inclusion in a paper."
}
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
    """Multi-round, cache-aware literature search and synthesis."""
    article = context.artifacts["article"]
    brief = context.artifacts["brief"]

    max_papers: int = int(context.inputs.get("max_papers") or 12)
    max_rounds: int = int(context.inputs.get("max_rounds") or 3)
    strictness: str = str(context.inputs.get("scope_strictness") or "normal")

    # Resolve lookup spec
    lookup_spec = _resolve_lookup_spec(context, article)
    scope_criteria = list(lookup_spec.get("scope_criteria") or [])

    # Build exclusion set from known_ids_to_exclude + known_papers cache
    known_papers_input: list[dict] = list(context.inputs.get("known_papers") or [])
    primary_id = str(article.metadata.get("arxiv_id") or article.artifact_id)
    excluded_ids: set[str] = {primary_id}
    excluded_ids.update(lookup_spec.get("known_ids_to_exclude") or [])
    cache_by_id: dict[str, dict] = {}
    for p in known_papers_input:
        pid = _paper_id(p)
        if pid:
            cache_by_id[pid] = p
            excluded_ids.add(pid)

    # Context for queries
    primary_title = str(article.metadata.get("title") or article.title or "")
    model_desc = str(brief.metadata.get("model_description") or "")
    abstract = str(article.metadata.get("abstract") or "")

    # Seed queries from plan
    active_queries: list[str] = [
        str(q.get("query_string") or "")
        for q in (lookup_spec.get("queries") or [])
        if str(q.get("query_string") or "").strip()
    ]
    if not active_queries:
        active_queries = [primary_title]

    tried_queries: set[str] = set()
    accepted: list[dict[str, Any]] = []
    all_seen_ids: set[str] = set(excluded_ids)

    for round_idx in range(max_rounds):
        if len(accepted) >= max_papers:
            break

        new_this_round: list[dict[str, Any]] = []

        for qs in active_queries:
            if qs in tried_queries or len(accepted) >= max_papers:
                continue
            tried_queries.add(qs)

            try:
                results = _arxiv_search(qs, max_results=15)
            except _RateLimited:
                break
            except Exception:
                continue

            for paper in results:
                if len(accepted) >= max_papers:
                    break
                pid = _paper_id(paper)
                if not pid or pid in all_seen_ids:
                    continue
                all_seen_ids.add(pid)

                # Scope validation
                if not _passes_scope(
                    llm_backend, paper,
                    primary_title, model_desc, scope_criteria, strictness,
                ):
                    continue

                accepted.append(paper)
                new_this_round.append(paper)

        # Refine queries for next round using accepted papers
        if round_idx < max_rounds - 1 and llm_backend and new_this_round:
            new_queries = _generate_refinement_queries(
                llm_backend,
                primary_title,
                model_desc,
                abstract,
                accepted,
                list(tried_queries),
            )
            active_queries = [q for q in new_queries if q not in tried_queries]
        else:
            break

    # Merge with known_papers cache (cache entries are already validated)
    all_papers = list(cache_by_id.values()) + accepted

    # Synthesis
    synthesis = _synthesise(llm_backend, all_papers, primary_title, model_desc)

    payload = {
        "accepted_papers": all_papers,
        "new_papers_this_run": accepted,
        "synthesis": synthesis,
        "search_queries_tried": list(tried_queries),
        "rounds_completed": round_idx + 1,
        "scope_criteria": scope_criteria,
        "strictness": strictness,
    }

    review_ref = registry.save_json(
        assistant=AssistantId.LITERATURE_REVIEWER.value,
        kind="literature_review",
        title="Literature Review",
        filename="article/literature_review.json",
        payload=payload,
        summary=(
            synthesis.get("prose_summary", "")[:200]
            if isinstance(synthesis, dict)
            else str(synthesis)[:200]
        ) or f"{len(all_papers)} related papers",
        metadata={
            "accepted_count": len(all_papers),
            "new_count": len(accepted),
            "accepted_papers": all_papers,
            "new_papers_this_run": accepted,
            "synthesis": synthesis,
            "synthesis_keys": list(synthesis.keys()) if isinstance(synthesis, dict) else [],
        },
        artifact_id="literature-review",
    )
    return ToolResult(
        status="completed",
        artifacts=[review_ref],
        message=(
            f"{len(accepted)} new paper(s) found this run; "
            f"{len(all_papers)} total in review."
        ),
    )


# ---------------------------------------------------------------------------
# Scope validation
# ---------------------------------------------------------------------------


def _passes_scope(
    backend: Any,
    paper: dict[str, Any],
    primary_title: str,
    model_desc: str,
    scope_criteria: list[str],
    strictness: str,
) -> bool:
    """Return True if the paper satisfies scope criteria."""
    if not scope_criteria or backend is None:
        # Without criteria or LLM, accept everything
        return True

    criteria_block = "\n".join(f"- {c}" for c in scope_criteria)
    user_msg = (
        f"Primary paper: {primary_title}\n"
        f"Model/topic: {model_desc[:300]}\n\n"
        f"Scope criteria:\n{criteria_block}\n\n"
        f"Strictness: {strictness}\n\n"
        f"Candidate title: {paper.get('title','')}\n"
        f"Candidate abstract: {str(paper.get('abstract',''))[:400]}"
    )
    try:
        raw = backend.complete(
            system=_SCOPE_SYSTEM,
            messages=[{"role": "user", "content": user_msg}],
            temperature=0.0,
            max_tokens=200,
        )
        result = json.loads(raw) if isinstance(raw, str) else raw
        return bool(result.get("accept", True))
    except Exception:
        return True  # Accept on LLM failure to avoid silent data loss


# ---------------------------------------------------------------------------
# Query refinement
# ---------------------------------------------------------------------------


def _generate_refinement_queries(
    backend: Any,
    primary_title: str,
    model_desc: str,
    abstract: str,
    accepted: list[dict],
    tried: list[str],
) -> list[str]:
    """Generate new ArXiv queries by analysing the accepted set."""
    titles_block = "\n".join(
        f"- {p.get('title', '')} ({p.get('year', '?')})"
        for p in accepted[-6:]
    )
    tried_block = "\n".join(f"- {q}" for q in tried[-8:])
    user_msg = (
        f"Primary paper: {primary_title}\n"
        f"Topic: {model_desc[:200]}\n\n"
        f"Already accepted papers:\n{titles_block}\n\n"
        f"Already tried queries:\n{tried_block}\n\n"
        "Generate new queries to find ADDITIONAL related work not yet found."
    )
    try:
        raw = backend.complete(
            system=_QUERY_GEN_SYSTEM,
            messages=[{"role": "user", "content": user_msg}],
            temperature=0.3,
        )
        queries = json.loads(raw) if isinstance(raw, str) else raw
        if isinstance(queries, list):
            return [str(q).strip() for q in queries if str(q).strip()]
    except Exception:
        pass
    return []


# ---------------------------------------------------------------------------
# Synthesis
# ---------------------------------------------------------------------------


def _synthesise(
    backend: Any,
    papers: list[dict],
    primary_title: str,
    model_desc: str,
) -> dict[str, Any] | str:
    if not papers:
        return {
            "clusters": [],
            "agreements": [],
            "contradictions": [],
            "gaps_addressed": [],
            "open_questions": [],
            "prose_summary": "No related papers found.",
        }
    if backend is None:
        return f"Found {len(papers)} related papers (no LLM for synthesis)."

    paper_block = "\n".join(
        f"{i+1}. \"{p.get('title','')}\" "
        f"({', '.join(p.get('authors', [])[:2])}, {p.get('year','?')}) — "
        f"{str(p.get('abstract',''))[:180]}..."
        for i, p in enumerate(papers)
    )
    user_msg = (
        f"Primary paper: {primary_title}\n"
        f"Model/approach: {model_desc[:300]}\n\n"
        f"Related papers ({len(papers)}):\n{paper_block}"
    )
    try:
        raw = backend.complete(
            system=_SYNTHESIS_SYSTEM,
            messages=[{"role": "user", "content": user_msg}],
            temperature=0.2,
        )
        result = json.loads(raw) if isinstance(raw, str) else raw
        if isinstance(result, dict):
            return result
    except Exception:
        pass
    return f"Synthesis of {len(papers)} related papers failed; manual review required."


# ---------------------------------------------------------------------------
# ArXiv search (self-contained copy to avoid circular imports)
# ---------------------------------------------------------------------------

ARXIV_API = "http://export.arxiv.org/api/query"
ARXIV_NS = "http://www.w3.org/2005/Atom"


def _arxiv_search(query: str, *, max_results: int = 15) -> list[dict[str, Any]]:
    params = urllib.parse.urlencode({
        "search_query": query,
        "start": 0,
        "max_results": max_results,
        "sortBy": "relevance",
        "sortOrder": "descending",
    })
    url = f"{ARXIV_API}?{params}"
    message = f"ArXiv query: {query} | max_results={max_results} | sortBy=relevance"
    print(message, flush=True)
    log.warning(message)
    req = urllib.request.Request(url, headers={"User-Agent": "research-platform/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            body = resp.read()
    except Exception as exc:
        if "429" in str(exc).lower():
            raise _RateLimited(str(exc)) from exc
        raise
    return _parse_arxiv_atom(body)


def _parse_arxiv_atom(body: bytes) -> list[dict[str, Any]]:
    import re as _re
    root = ET.fromstring(body)
    papers: list[dict[str, Any]] = []
    for entry in root.findall(f"{{{ARXIV_NS}}}entry"):
        title_el = entry.find(f"{{{ARXIV_NS}}}title")
        title = (title_el.text or "").strip().replace("\n", " ") if title_el is not None else ""
        authors = [
            (a.find(f"{{{ARXIV_NS}}}name").text or "").strip()
            for a in entry.findall(f"{{{ARXIV_NS}}}author")
            if a.find(f"{{{ARXIV_NS}}}name") is not None
        ]
        summary_el = entry.find(f"{{{ARXIV_NS}}}summary")
        abstract = (summary_el.text or "").strip().replace("\n", " ") if summary_el is not None else ""
        id_el = entry.find(f"{{{ARXIV_NS}}}id")
        raw_id = (id_el.text or "").strip() if id_el is not None else ""
        m = _re.search(r"(\d{4}\.\d{4,5})", raw_id)
        arxiv_id = m.group(1) if m else None
        url = raw_id
        for link in entry.findall(f"{{{ARXIV_NS}}}link"):
            if link.get("type") == "text/html":
                url = link.get("href", url)
                break
        cats = [t.get("term", "") for t in entry.findall(f"{{{ARXIV_NS}}}category")]
        pub_el = entry.find(f"{{{ARXIV_NS}}}published")
        year: int | None = None
        if pub_el is not None and pub_el.text:
            try:
                year = int(pub_el.text[:4])
            except ValueError:
                pass
        if title:
            papers.append({
                "title": title, "authors": authors, "abstract": abstract,
                "year": year, "arxiv_id": arxiv_id, "url": url,
                "categories": cats, "source": "arxiv",
            })
    return papers


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _paper_id(paper: dict[str, Any]) -> str:
    return (
        str(paper.get("arxiv_id") or "")
        or str(paper.get("doi") or "")
        or str(paper.get("title") or "")
    )


def _resolve_lookup_spec(
    context: ToolContext,
    article: Any,
) -> dict[str, Any]:
    """Resolve the lookup spec from inputs, falling back to article metadata."""
    spec = dict(context.inputs.get("lookup_spec") or {})
    if spec:
        return spec

    # Try to get the spec from an artifact input
    spec_artifact = context.artifacts.get("lookup_spec")
    if spec_artifact is not None:
        spec = dict(getattr(spec_artifact, "metadata", {}) or {})
        if spec:
            return spec

    # Build minimal plan from article metadata
    title = str(article.metadata.get("title") or article.title or "")
    abstract = str(article.metadata.get("abstract") or "")
    cats = list(article.metadata.get("categories") or [])
    cat_part = (
        "(" + " OR ".join(f"cat:{c}" for c in cats[:3]) + ")"
        if cats else ""
    )
    base_qs = f'abs:"{title[:60]}"'
    if cat_part:
        base_qs += f" AND {cat_part}"
    return {
        "research_intent": f"Find related work for: {title[:80]}",
        "scope_criteria": [f"Related to the topic of: {title[:60]}"],
        "queries": [{"label": "title-based", "query_string": base_qs, "max_results": 15}],
        "known_ids_to_exclude": [],
        "fallback_query": title,
    }


class _RateLimited(Exception):
    pass
