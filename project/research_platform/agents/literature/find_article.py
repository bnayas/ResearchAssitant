"""
literature.find_article
───────────────────────
Tool: Execute a structured search plan against arXiv (and optionally
Semantic Scholar as fallback) and return the single best matching paper.

Search is driven by the structured query specs produced by
prepare_article_lookup.  Each spec carries native ArXiv field syntax
(ti:, au:, abs:, cat:, submittedDate:) so queries are precise.

Returned artifact schema ("primary_article"):
    title         str
    authors       list[str]
    abstract      str
    year          int | None
    arxiv_id      str | None
    doi           str | None
    url           str
    categories    list[str]
    source        "arxiv" | "semantic_scholar" | "unknown"
    lookup_query  str   — the query that produced this result
    score         float — relevance score (0.0–1.0), LLM-assigned if available
"""
from __future__ import annotations

import json
import logging
import re
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from typing import Any

from ...contracts import AssistantId
from ...registry import ArtifactRegistry
from ..base import ToolContext, ToolDescriptor, ToolRequirement, ToolResult
from ..validators import is_non_empty_string
from .prepare_lookup import revise_lookup_plan

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Tool descriptor
# ---------------------------------------------------------------------------

TOOL = ToolDescriptor(
    name="find_primary_article",
    display_name="Find Primary Article",
    description=(
        "Execute a structured ArXiv search plan and return an artifact for "
        "the single best matching paper.  Accepts either an article_lookup_spec "
        "artifact (from prepare_article_lookup) or a raw query string.  Tries "
        "each query strategy in the plan in order, stops at the first confident "
        "match, and falls back to Semantic Scholar if arXiv returns nothing."
    ),
    agent_id="literature",
    requirements=[
        ToolRequirement(
            name="lookup_spec",
            description=(
                "Structured lookup plan produced by prepare_article_lookup, "
                "containing one or more ArXiv-native query specs."
            ),
            type="dict",
            required=False,
            default={},
        ),
        ToolRequirement(
            name="query",
            description=(
                "Raw search string used when no lookup_spec is provided.  "
                "May be an ArXiv-field-qualified query or a plain text phrase."
            ),
            type="str",
            required=False,
            default="",
        ),
        ToolRequirement(
            name="rerank",
            description=(
                "If true and an LLM backend is available, use the LLM to "
                "pick the best match from the candidate set before returning."
            ),
            type="bool",
            required=False,
            default=True,
        ),
        ToolRequirement(
            name="max_candidates",
            description="Maximum ArXiv results to fetch per query spec.",
            type="int",
            required=False,
            default=15,
        ),
    ],
    produces=["primary_article"],
    tags=["literature", "search", "arxiv"],
    idempotent=True,
    estimated_seconds=20.0,
)

# ---------------------------------------------------------------------------
# ArXiv API constants
# ---------------------------------------------------------------------------

ARXIV_API = "http://export.arxiv.org/api/query"
ARXIV_NS = "http://www.w3.org/2005/Atom"
ARXIV_NS_MAP = {
    "atom": ARXIV_NS,
    "arxiv": "http://arxiv.org/schemas/atom",
    "opensearch": "http://a9.com/-/spec/opensearch/1.1/",
}

# ---------------------------------------------------------------------------
# LLM reranker prompt
# ---------------------------------------------------------------------------

_RERANK_SYSTEM = """\
You are an academic paper relevance judge.

You will receive a research question and a numbered list of candidate papers.
Select the single paper that best answers the research question.

Rules:
- Choose based on topical relevance, not just surface keywords.
- Prefer papers that are likely to be THE target paper (primary source),
  not merely related work.
- If no candidate is a strong match, output 0.

Return ONLY a JSON object:
{"index": <1-based integer or 0 if none match>, "confidence": <0.0-1.0>,
 "reason": "one sentence"}

Keep the response minimal.
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
    """Search ArXiv using the plan and return the best matching paper."""
    lookup_spec = _resolve_spec(context)
    rerank: bool = bool(context.inputs.get("rerank", True))
    max_candidates: int = int(context.inputs.get("max_candidates") or 15)
    max_iterations: int = int(context.inputs.get("max_query_iterations") or 2)
    original_query = _resolve_query_text(context, lookup_spec)

    known_ids: set[str] = set(lookup_spec.get("known_ids_to_exclude") or [])

    # Build a combined candidate pool across all query specs
    candidates: list[dict[str, Any]] = []
    successful_query = ""
    all_errors: list[str] = []
    all_rejected: list[dict[str, Any]] = []
    tried_queries: list[str] = []

    for iteration in range(max(1, max_iterations)):
        query_specs = list(lookup_spec.get("queries") or [])
        fallback_query = str(lookup_spec.get("fallback_query") or "").strip()
        required_authors = _required_authors(lookup_spec)

        for spec in query_specs:
            qs = str(spec.get("query_string") or "").strip()
            if not qs or qs in tried_queries:
                continue
            tried_queries.append(qs)
            n = min(max_candidates, int(spec.get("max_results") or 15))
            sort_by = str(spec.get("sort_by") or "relevance")
            try:
                results = _arxiv_search(qs, max_results=n, sort_by=sort_by)
            except _RateLimited as e:
                all_errors.append(f"arXiv rate-limited: {e}")
                break
            except Exception as e:
                all_errors.append(f"Query '{spec.get('label','?')}' failed: {e}")
                continue

            new_results, rejected = _filter_candidates(results, known_ids, required_authors, lookup_spec)
            all_rejected.extend(rejected)
            candidates.extend(new_results)
            if not successful_query and new_results:
                successful_query = qs
            # Deduplicate as we go
            candidates = _dedup(candidates)
            if len(candidates) >= max_candidates * 2:
                break

        # Fallback: if no results, try a plain-text fallback
        if not candidates and fallback_query and fallback_query not in tried_queries:
            tried_queries.append(fallback_query)
            try:
                results = _arxiv_search(fallback_query, max_results=max_candidates)
                new_results, rejected = _filter_candidates(results, known_ids, required_authors, lookup_spec)
                all_rejected.extend(rejected)
                candidates = new_results
                successful_query = fallback_query if candidates else successful_query
            except Exception as e:
                all_errors.append(f"Fallback query failed: {e}")

        # Fallback: try Semantic Scholar if still nothing
        if not candidates:
            try:
                raw_query = fallback_query or (query_specs[0].get("query_string") if query_specs else "")
                if raw_query:
                    results = _semantic_scholar_search(raw_query, max_results=10)
                    candidates, rejected = _filter_candidates(results, known_ids, required_authors, lookup_spec)
                    all_rejected.extend(rejected)
            except Exception as e:
                all_errors.append(f"Semantic Scholar failed: {e}")

        if candidates or iteration >= max_iterations - 1 or llm_backend is None:
            break

        telemetry = {
            "iteration": iteration + 1,
            "tried_queries": tried_queries,
            "errors": all_errors[-8:],
            "raw_candidates_after_author_filter": len(candidates),
            "rejected_candidates": all_rejected[-12:],
            "reason": "No acceptable candidates were found.",
        }
        lookup_spec = revise_lookup_plan(
            llm_backend,
            original_query=original_query,
            previous_plan=lookup_spec,
            telemetry=telemetry,
            known_ids=list(known_ids),
        )

    if not candidates:
        error_summary = "; ".join(all_errors) or "No results found across all backends"
        return ToolResult(status="failed", message=error_summary)

    # Pick best match
    paper = _select_best(candidates, lookup_spec, llm_backend, rerank)
    article_ref = _build_artifact(paper, successful_query, registry)

    authors_str = ", ".join(paper.get("authors") or [])
    year = paper.get("year") or "?"
    return ToolResult(
        status="completed",
        artifacts=[article_ref],
        message=f'Found: "{paper.get("title","?")}" ({authors_str}, {year})',
    )


# ---------------------------------------------------------------------------
# ArXiv API
# ---------------------------------------------------------------------------


def _arxiv_search(
    query: str,
    *,
    max_results: int = 15,
    sort_by: str = "relevance",
    start: int = 0,
) -> list[dict[str, Any]]:
    """Call the ArXiv Atom API and return parsed paper dicts."""
    sort_map = {
        "relevance": "relevance",
        "date": "submittedDate",
        "submittedDate": "submittedDate",
        "lastUpdatedDate": "lastUpdatedDate",
    }
    params = urllib.parse.urlencode({
        "search_query": query,
        "start": start,
        "max_results": max_results,
        "sortBy": sort_map.get(sort_by, "relevance"),
        "sortOrder": "descending",
    })
    url = f"{ARXIV_API}?{params}"
    message = f"ArXiv query: {query} | max_results={max_results} | sortBy={sort_map.get(sort_by, 'relevance')}"
    print(message, flush=True)
    log.warning(message)
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "research-platform/1.0"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            body = resp.read()
    except Exception as exc:
        msg = str(exc).lower()
        if "429" in msg or "too many" in msg:
            raise _RateLimited(str(exc)) from exc
        raise

    return _parse_arxiv_atom(body)


def _parse_arxiv_atom(body: bytes) -> list[dict[str, Any]]:
    """Parse ArXiv Atom XML into a list of paper dicts."""
    root = ET.fromstring(body)
    papers: list[dict[str, Any]] = []

    for entry in root.findall(f"{{{ARXIV_NS}}}entry"):
        # Title
        title_el = entry.find(f"{{{ARXIV_NS}}}title")
        title = (title_el.text or "").strip().replace("\n", " ") if title_el is not None else ""

        # Authors
        authors = [
            (a.find(f"{{{ARXIV_NS}}}name").text or "").strip()
            for a in entry.findall(f"{{{ARXIV_NS}}}author")
            if a.find(f"{{{ARXIV_NS}}}name") is not None
        ]

        # Abstract
        summary_el = entry.find(f"{{{ARXIV_NS}}}summary")
        abstract = (summary_el.text or "").strip().replace("\n", " ") if summary_el is not None else ""

        # arXiv ID
        id_el = entry.find(f"{{{ARXIV_NS}}}id")
        raw_id = (id_el.text or "").strip() if id_el is not None else ""
        arxiv_id = _extract_arxiv_id(raw_id)

        # URL
        url = raw_id  # The entry id IS the canonical URL
        for link in entry.findall(f"{{{ARXIV_NS}}}link"):
            if link.get("type") == "text/html":
                url = link.get("href", url)
                break

        # DOI
        doi_el = entry.find("{http://arxiv.org/schemas/atom}doi")
        doi = (doi_el.text or "").strip() if doi_el is not None else None

        # Categories
        cats = [
            tag.get("term", "")
            for tag in entry.findall(f"{{{ARXIV_NS}}}category")
        ]

        # Published date → year
        pub_el = entry.find(f"{{{ARXIV_NS}}}published")
        year: int | None = None
        if pub_el is not None and pub_el.text:
            try:
                year = int(pub_el.text[:4])
            except ValueError:
                pass

        if not title:
            continue
        papers.append({
            "title": title,
            "authors": authors,
            "abstract": abstract,
            "year": year,
            "arxiv_id": arxiv_id,
            "doi": doi,
            "url": url,
            "categories": cats,
            "source": "arxiv",
        })

    return papers


def _extract_arxiv_id(raw: str) -> str | None:
    """Extract the bare arXiv ID (e.g. '2301.00001') from a URL or raw string."""
    if not raw:
        return None
    # http://arxiv.org/abs/2301.00001v2 → 2301.00001
    import re
    m = re.search(r"(\d{4}\.\d{4,5})(v\d+)?", raw)
    return m.group(1) if m else raw.rsplit("/", 1)[-1]


# ---------------------------------------------------------------------------
# Semantic Scholar fallback
# ---------------------------------------------------------------------------


def _semantic_scholar_search(query: str, max_results: int = 10) -> list[dict[str, Any]]:
    """Query Semantic Scholar public API as a fallback."""
    params = urllib.parse.urlencode({
        "query": query,
        "limit": max_results,
        "fields": "title,authors,abstract,year,externalIds,url",
    })
    url = f"https://api.semanticscholar.org/graph/v1/paper/search?{params}"
    req = urllib.request.Request(url, headers={"User-Agent": "research-platform/1.0"})
    with urllib.request.urlopen(req, timeout=20) as resp:
        data = json.loads(resp.read())

    papers: list[dict[str, Any]] = []
    for item in data.get("data") or []:
        ext = item.get("externalIds") or {}
        authors = [a.get("name", "") for a in (item.get("authors") or [])]
        papers.append({
            "title": str(item.get("title") or ""),
            "authors": authors,
            "abstract": str(item.get("abstract") or ""),
            "year": item.get("year"),
            "arxiv_id": ext.get("ArXiv"),
            "doi": ext.get("DOI"),
            "url": str(item.get("url") or ""),
            "categories": [],
            "source": "semantic_scholar",
        })
    return papers


# ---------------------------------------------------------------------------
# Candidate selection
# ---------------------------------------------------------------------------


def _select_best(
    candidates: list[dict[str, Any]],
    lookup_spec: dict[str, Any],
    llm_backend: Any,
    rerank: bool,
) -> dict[str, Any]:
    """Pick the best candidate, using LLM reranking if available and requested."""
    if len(candidates) == 1 or not rerank or llm_backend is None:
        return candidates[0]

    research_intent = str(lookup_spec.get("research_intent") or "")
    scope_criteria = lookup_spec.get("scope_criteria") or []
    scope_block = "\n".join(f"- {c}" for c in scope_criteria) if scope_criteria else ""

    summaries = "\n".join(
        f"{i+1}. \"{p.get('title','')}\" "
        f"({', '.join(p.get('authors', [])[:3])}, {p.get('year','?')}) — "
        f"{str(p.get('abstract',''))[:150]}..."
        for i, p in enumerate(candidates[:20])
    )

    user_msg = (
        f"Research intent: {research_intent}\n"
        + (f"Scope criteria:\n{scope_block}\n\n" if scope_block else "\n")
        + f"Candidates:\n{summaries}"
    )

    try:
        raw = llm_backend.complete(
            system=_RERANK_SYSTEM,
            messages=[{"role": "user", "content": user_msg}],
            temperature=0.0,
            max_tokens=128,
        )
        result = json.loads(raw) if isinstance(raw, str) else raw
        idx = int(result.get("index", 1)) - 1
        if 0 <= idx < len(candidates):
            return candidates[idx]
    except Exception:
        pass

    return candidates[0]


# ---------------------------------------------------------------------------
# Artifact builder
# ---------------------------------------------------------------------------


def _build_artifact(
    paper: dict[str, Any],
    query: str,
    registry: ArtifactRegistry,
) -> Any:
    metadata = {
        "title": paper.get("title", ""),
        "authors": paper.get("authors", []),
        "abstract": paper.get("abstract", ""),
        "year": paper.get("year"),
        "url": paper.get("url", ""),
        "arxiv_id": paper.get("arxiv_id"),
        "doi": paper.get("doi"),
        "categories": paper.get("categories", []),
        "source": paper.get("source", "unknown"),
        "lookup_query": query,
    }
    return registry.create(
        artifact_id="article-match",
        assistant=AssistantId.LITERATURE_REVIEWER.value,
        kind="primary_article",
        title=str(metadata["title"]),
        summary=str(metadata["abstract"])[:200],
        content=str(metadata["abstract"]),
        metadata=metadata,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _resolve_spec(context: ToolContext) -> dict[str, Any]:
    """Build a lookup spec from context inputs, accepting several input shapes."""
    # Direct spec from artifact_inputs or inline
    spec = dict(context.inputs.get("lookup_spec") or {})
    if not spec:
        # Legacy: instruction / topic_hint / lookup_spec from old API
        spec = dict(context.inputs.get("lookup_spec") or {})

    # Plain query string fallback
    raw_query = (
        str(context.inputs.get("query") or "").strip()
        or str(context.inputs.get("instruction") or "").strip()
        or str(getattr(context, "instruction", "") or "").strip()
    )
    if not spec and raw_query:
        spec = {
            "research_intent": raw_query[:120],
            "queries": [{
                "label": "Direct query",
                "query_string": raw_query,
                "sort_by": "relevance",
                "max_results": 15,
            }],
            "fallback_query": raw_query,
            "known_ids_to_exclude": [],
        }

    # If spec came from the artifact metadata dict inside ToolContext.artifacts
    if not spec:
        spec_artifact = context.artifacts.get("lookup_spec")
        if spec_artifact is not None:
            spec = dict(getattr(spec_artifact, "metadata", {}) or {})

    return spec


def _resolve_query_text(context: ToolContext, lookup_spec: dict[str, Any]) -> str:
    return (
        str(context.inputs.get("query") or "").strip()
        or str(context.inputs.get("instruction") or "").strip()
        or str(getattr(context, "instruction", "") or "").strip()
        or str(lookup_spec.get("article_target") or lookup_spec.get("research_intent") or "").strip()
    )


def _required_authors(lookup_spec: dict[str, Any]) -> list[str]:
    authors = [
        str(author).strip()
        for author in (lookup_spec.get("required_authors") or [])
        if str(author).strip()
    ]
    if authors:
        return authors
    for spec in lookup_spec.get("queries") or []:
        if isinstance(spec, dict):
            for author in spec.get("authors") or []:
                text = str(author).strip()
                if text and text.lower() not in {item.lower() for item in authors}:
                    authors.append(text)
    return authors


def _filter_candidates(
    papers: list[dict[str, Any]],
    known_ids: set[str],
    required_authors: list[str],
    lookup_spec: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    topic_terms = _lookup_topic_terms(lookup_spec)
    for paper in papers:
        pid = _paper_id(paper)
        if pid in known_ids:
            continue
        missing = _missing_required_authors(paper.get("authors") or [], required_authors)
        if missing:
            rejected.append({
                "title": paper.get("title", ""),
                "year": paper.get("year"),
                "authors": paper.get("authors") or [],
                "reason": "missing required author(s): " + ", ".join(missing),
            })
            continue
        if topic_terms and _candidate_topic_score(paper, topic_terms) <= 0:
            rejected.append({
                "title": paper.get("title", ""),
                "year": paper.get("year"),
                "authors": paper.get("authors") or [],
                "reason": "author match but no title/topic signal",
            })
            continue
        accepted.append(paper)
    return accepted, rejected


def _missing_required_authors(raw_authors: list[str], required_authors: list[str]) -> list[str]:
    if not required_authors:
        return []
    raw_text = " ".join(str(author) for author in raw_authors).lower()
    missing: list[str] = []
    for author in required_authors:
        surname = str(author or "").strip().split()[-1].lower()
        if surname and not re.search(rf"\b{re.escape(surname)}\b", raw_text):
            missing.append(author)
    return missing


def _lookup_topic_terms(lookup_spec: dict[str, Any]) -> list[str]:
    terms: list[str] = []
    for key in ("title_phrases", "topic_terms"):
        for term in lookup_spec.get(key) or []:
            text = str(term or "").strip()
            if text and text.lower() not in {item.lower() for item in terms}:
                terms.append(text)
    for spec in lookup_spec.get("queries") or []:
        if isinstance(spec, dict):
            for key in ("all_terms", "title_terms", "abstract_keywords"):
                for term in spec.get(key) or []:
                    text = str(term or "").strip()
                    if text and text.lower() not in {item.lower() for item in terms}:
                        terms.append(text)
    excluded = {str(term).strip().lower() for term in lookup_spec.get("excluded_search_terms") or []}
    return [
        term for term in terms
        if term.lower() not in excluded and len(_term_tokens(term)) > 0
    ][:12]


def _candidate_topic_score(paper: dict[str, Any], topic_terms: list[str]) -> int:
    text = f"{paper.get('title', '')}\n{paper.get('abstract', '')}".lower()
    score = 0
    strong_terms = [term for term in topic_terms if len(_term_tokens(term)) >= 2]
    terms_to_score = strong_terms or topic_terms
    for term in terms_to_score:
        normalized = str(term or "").strip().lower()
        if normalized and normalized in text:
            score += 3
            continue
        tokens = _term_tokens(normalized)
        hits = sum(1 for token in tokens if re.search(rf"\b{re.escape(token.rstrip('s'))}s?\b", text))
        if len(tokens) >= 2 and hits >= min(2, len(tokens)):
            score += hits
        elif len(tokens) == 1 and hits:
            score += 1
    return score


def _term_tokens(term: str) -> list[str]:
    stop = {
        "about", "article", "detail", "details", "model", "models", "paper",
        "read", "reproduce", "simulation", "simulations", "summarize", "the",
        "with",
    }
    return [
        token
        for token in re.findall(r"[a-z][a-z-]{2,}", str(term or "").lower())
        if token not in stop
    ]


def _paper_id(paper: dict[str, Any]) -> str:
    return paper.get("arxiv_id") or paper.get("doi") or paper.get("title", "")


def _dedup(papers: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    result: list[dict[str, Any]] = []
    for p in papers:
        pid = _paper_id(p)
        if pid not in seen:
            seen.add(pid)
            result.append(p)
    return result


class _RateLimited(Exception):
    pass
