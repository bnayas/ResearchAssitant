"""
reviewer/diff.py
=================
Annotated article builder and unified diff generator.

Responsibilities
----------------
1. Parse the raw article text into an ArticleView (section splitting).
2. Insert ReviewAnnotation markers into article text at the correct lines.
3. Produce a unified diff (original → annotated) that is:
   - Human-readable in a pull-request / GitHub style interface
   - Machine-parseable for re-ingestion by the AcademicWriter.respond_to_review()

Annotation insertion contract
------------------------------
Annotations are inserted AFTER the target line (not replacing it).
The original line is preserved verbatim.  The annotation appears as:

    Original line text...
    ~~> [REVIEW-ANN-001AB | CRITICAL] There is a sign error here.
        → Suggested: F = -∇V, not F = +∇V

Multiple annotations on the same line are inserted in severity order:
  critical → major → minor → question

The annotation prefix ~~> is chosen to be visually distinct and
unlikely to appear in article text.  It is stripped by the writer
when consuming the diff programmatically.

Unified diff format
-------------------
Standard unified diff with context lines = 3.
Header lines use pseudo-filenames:
    --- a/<article_id>.original.md
    +++ b/<article_id>.annotated.md
This allows the diff to be applied with `patch` or displayed by
standard diff viewers without any custom tooling.
"""

from __future__ import annotations

import difflib
import re
import uuid
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

from .contract import (
    AnnotatedArticle,
    AnnotationSeverity,
    ArticleSection,
    ArticleView,
    ReviewAnnotation,
)

# ── Severity ordering for multi-annotation lines ──────────────────────────────

_SEVERITY_ORDER: Dict[str, int] = {
    "critical": 0,
    "major": 1,
    "minor": 2,
    "question": 3,
}

ANNOTATION_PREFIX = "~~>"

# ── Article parser ────────────────────────────────────────────────────────────

# Heuristic patterns for markdown / plain-text section headers.
_SECTION_RE = re.compile(
    r"^(?P<hashes>#{1,4})\s+(?P<number>(?:\d+\.)*\d+)?\s*(?P<title>.+)$",
    re.MULTILINE,
)

# Patterns that hint a section contains math or empirical claims.
_MATH_HINT_RE = re.compile(
    r"[\$\\]|\\begin\{|=\s*[\d\.\-]|\bp[\s<>=]+0\.\d|O\(|Θ\(|≤|≥|∈|∀|∃|∫|∂",
)
_CLAIM_HINT_RE = re.compile(
    r"\b(shows?|demonstrates?|proves?|establishes?|confirms?|implies?|"
    r"find|found|observe[sd]?|results? in|yields?|achieves?|outperforms?)\b",
    re.IGNORECASE,
)


def parse_article(raw_text: str, article_id: Optional[str] = None) -> ArticleView:
    """
    Split a raw article text into an ArticleView with per-section metadata.

    Section detection uses markdown heading syntax (# / ## / ### / ####).
    Falls back to a single section containing the full text if no headers found.
    """
    if article_id is None:
        article_id = f"art-{uuid.uuid4().hex[:8]}"

    lines = raw_text.splitlines()
    total_lines = len(lines)

    # Extract title and authors from leading lines (heuristic)
    title = ""
    authors: List[str] = []
    abstract = ""
    for i, line in enumerate(lines[:20]):
        stripped = line.strip()
        if stripped.startswith("# ") and not title:
            title = stripped[2:].strip()
        elif re.match(r"^\*\*?Authors?:?\*\*?", stripped, re.I) or re.match(r"^By ", stripped, re.I):
            authors_text = re.sub(r"^\*\*?Authors?:?\*\*?\s*", "", stripped, flags=re.I).strip()
            authors = [a.strip() for a in re.split(r"[,;]|and", authors_text) if a.strip()]
        elif "abstract" in stripped.lower() and len(stripped) < 20:
            # Next paragraph is abstract
            para_lines = []
            for j in range(i + 1, min(i + 30, len(lines))):
                if lines[j].strip() and not lines[j].strip().startswith("#"):
                    para_lines.append(lines[j].strip())
                elif para_lines:
                    break
            abstract = " ".join(para_lines)

    # Find section boundaries
    boundaries: List[Tuple[int, str, str, int]] = []  # (line_idx, number, title, level)
    for match in _SECTION_RE.finditer(raw_text):
        level = len(match.group("hashes"))
        number = match.group("number") or ""
        sec_title = match.group("title").strip()
        line_idx = raw_text[: match.start()].count("\n")
        boundaries.append((line_idx, number, sec_title, level))

    if not boundaries:
        # No sections found: treat entire article as one section
        section = ArticleSection(
            section_id="s0",
            number="",
            title=title or "Full Article",
            body=raw_text,
            start_line=0,
            end_line=total_lines - 1,
            has_math=bool(_MATH_HINT_RE.search(raw_text)),
            has_claims=bool(_CLAIM_HINT_RE.search(raw_text)),
        )
        return ArticleView(
            article_id=article_id,
            title=title,
            authors=authors,
            abstract=abstract,
            sections=[section],
            raw_text=raw_text,
            total_lines=total_lines,
        )

    sections: List[ArticleSection] = []
    for idx, (start, number, sec_title, level) in enumerate(boundaries):
        end = boundaries[idx + 1][0] - 1 if idx + 1 < len(boundaries) else total_lines - 1
        body = "\n".join(lines[start: end + 1])
        sections.append(ArticleSection(
            section_id=f"s{idx + 1}",
            number=number,
            title=sec_title,
            body=body,
            start_line=start,
            end_line=end,
            has_math=bool(_MATH_HINT_RE.search(body)),
            has_claims=bool(_CLAIM_HINT_RE.search(body)),
        ))

    return ArticleView(
        article_id=article_id,
        title=title,
        authors=authors,
        abstract=abstract,
        sections=sections,
        raw_text=raw_text,
        total_lines=total_lines,
    )


# ── Annotated article builder ─────────────────────────────────────────────────


class ArticleAnnotator:
    """
    Inserts ReviewAnnotation markers into the original article text
    and produces a unified diff.

    Usage
    -----
        annotator = ArticleAnnotator(article_view)
        annotated = annotator.build(annotations)
    """

    def __init__(self, view: ArticleView) -> None:
        self._view = view

    def build(self, annotations: List[ReviewAnnotation]) -> AnnotatedArticle:
        """
        Produce the AnnotatedArticle from a list of annotations.

        Steps
        -----
        1. Group annotations by line number.
        2. Sort each group by severity (critical first).
        3. Re-build the article line-by-line, inserting annotation
           markers after the target line.
        4. Generate unified diff (original → annotated).
        5. Build severity count histogram.
        """
        # Group and sort
        by_line: Dict[int, List[ReviewAnnotation]] = defaultdict(list)
        for ann in annotations:
            by_line[ann.line_number].append(ann)

        for line_idx in by_line:
            by_line[line_idx].sort(
                key=lambda a: (_SEVERITY_ORDER.get(a.severity, 99), a.created_at)
            )

        # Re-build
        original_lines = self._view.raw_text.splitlines()
        annotated_lines: List[str] = []

        for i, line in enumerate(original_lines):
            annotated_lines.append(line)
            if i in by_line:
                for ann in by_line[i]:
                    annotated_lines.extend(self._format_annotation(ann))

        annotated_text = "\n".join(annotated_lines)

        # Unified diff
        original_file = f"a/{self._view.article_id}.original.md"
        annotated_file = f"b/{self._view.article_id}.annotated.md"
        diff_lines = list(difflib.unified_diff(
            original_lines,
            annotated_lines,
            fromfile=original_file,
            tofile=annotated_file,
            lineterm="",
            n=3,
        ))
        unified_diff = "\n".join(diff_lines)

        # Severity histogram
        counts: Dict[str, int] = {}
        for ann in annotations:
            counts[ann.severity] = counts.get(ann.severity, 0) + 1

        return AnnotatedArticle(
            article_id=self._view.article_id,
            original_text=self._view.raw_text,
            annotated_text=annotated_text,
            unified_diff=unified_diff,
            annotations=annotations,
            annotation_count_by_severity=counts,
        )

    def _format_annotation(self, ann: ReviewAnnotation) -> List[str]:
        """
        Format a single ReviewAnnotation as one or more lines to be
        inserted into the annotated article.
        """
        badge = f"[{ann.annotation_id[:7].upper()} | {ann.severity.upper()} | {ann.kind}]"
        comment_lines = ann.comment.splitlines()
        first_line = f"{ANNOTATION_PREFIX} {badge} {comment_lines[0]}"
        continuation = [f"    {l}" for l in comment_lines[1:]]
        result = [first_line] + continuation
        if ann.suggested_replacement:
            result.append(f"    → Suggested replacement: {ann.suggested_replacement}")
        return result


# ── Diff utilities ────────────────────────────────────────────────────────────


def annotation_count_summary(annotated: AnnotatedArticle) -> str:
    """Human-readable summary line for the annotation set."""
    counts = annotated.annotation_count_by_severity
    parts = [
        f"{counts.get('critical', 0)} critical",
        f"{counts.get('major', 0)} major",
        f"{counts.get('minor', 0)} minor",
        f"{counts.get('question', 0)} question",
    ]
    return "Annotations: " + ", ".join(parts)


def locate_line(view: ArticleView, target_text: str) -> int:
    """
    Find the 0-based line number of the first occurrence of target_text.
    Returns -1 if not found.  Used by the reviewer to pin annotations.
    """
    for i, line in enumerate(view.raw_text.splitlines()):
        if target_text.strip() in line:
            return i
    return -1


def extract_annotations_from_diff(unified_diff: str) -> List[Dict[str, str]]:
    """
    Parse the ~~> annotation lines back out of a unified diff.
    Useful for programmatic consumption by the AcademicWriter.

    Returns a list of dicts with keys: annotation_id, severity, kind, comment.
    """
    results = []
    badge_re = re.compile(
        r"\+~~>\s+\[([A-Z0-9\-]+)\s*\|\s*([A-Z]+)\s*\|\s*([a-z_]+)\]\s+(.*)"
    )
    for line in unified_diff.splitlines():
        m = badge_re.match(line)
        if m:
            results.append({
                "annotation_id": m.group(1).lower(),
                "severity": m.group(2).lower(),
                "kind": m.group(3),
                "comment": m.group(4),
            })
    return results
