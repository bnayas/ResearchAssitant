from __future__ import annotations

import argparse
import json
import uuid

from literature_review.contract import (
    LiteratureReviewTask,
    ScopeConstraint,
    SearchDepthConfig,
)
from literature_review.director_bridge import build_bridge_from_env


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a live literature review")
    parser.add_argument("query", help="Natural-language research question")
    parser.add_argument(
        "--topic",
        action="append",
        default=[],
        help="Explicit in-scope topic. Repeat for multiple topics.",
    )
    parser.add_argument(
        "--exclude-topic",
        action="append",
        default=[],
        help="Explicit out-of-scope topic. Repeat for multiple topics.",
    )
    parser.add_argument("--year-min", type=int, default=None)
    parser.add_argument("--year-max", type=int, default=None)
    parser.add_argument("--max-papers", type=int, default=10)
    parser.add_argument("--papers-per-query", type=int, default=8)
    parser.add_argument("--min-papers-threshold", type=int, default=4)
    parser.add_argument("--max-rounds", type=int, default=3)
    parser.add_argument("--max-term-variations", type=int, default=2)
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit a compact JSON result instead of a text summary.",
    )
    return parser


def _build_task(args: argparse.Namespace) -> LiteratureReviewTask:
    include_topics = args.topic or [args.query]
    return LiteratureReviewTask(
        task_id=f"lit-{uuid.uuid4().hex[:8]}",
        branch_id="cli",
        query=args.query,
        scope=ScopeConstraint(
            include_topics=include_topics,
            exclude_topics=args.exclude_topic,
            year_min=args.year_min,
            year_max=args.year_max,
            max_papers=args.max_papers,
        ),
        depth=SearchDepthConfig(
            max_rounds=args.max_rounds,
            max_term_variations=args.max_term_variations,
            min_papers_threshold=args.min_papers_threshold,
            papers_per_query=args.papers_per_query,
        ),
        requestor_agent="example_literature_review",
    )


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()

    bridge = build_bridge_from_env()
    result = bridge.run_sync(_build_task(args))
    artifact = result.artifact

    if args.json:
        payload = {
            "summary": artifact.summary_line(),
            "status": artifact.status,
            "is_sufficient": artifact.is_sufficient,
            "audit_passed": result.audit.passed,
            "papers": [
                {
                    "title": paper.title,
                    "year": paper.year,
                    "source": paper.source,
                    "url": paper.url,
                }
                for paper in artifact.papers
            ],
            "synthesis": artifact.synthesis,
        }
        print(json.dumps(payload, indent=2))
        return 0

    print(artifact.summary_line())
    print(f"audit_passed={result.audit.passed}")
    if artifact.insufficiency_reason:
        print(f"insufficiency={artifact.insufficiency_reason}")
    print("")
    print("Papers:")
    for paper in artifact.papers:
        year = paper.year if paper.year is not None else "?"
        print(f"- [{paper.source}] {paper.title} ({year})")
        print(f"  {paper.url}")
    print("")
    print("Synthesis:")
    print(artifact.synthesis)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
