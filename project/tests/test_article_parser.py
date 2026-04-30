from __future__ import annotations

import urllib.request

from literature_review.article_parser import ArticleParser
from literature_review.search_backends.base import RawPaper, SearchBackend


class _EmptyBackend(SearchBackend):
    name = "arxiv"

    async def search(self, keywords, max_results, year_min=None, year_max=None, categories=None):
        return []


class _SemanticHitBackend(SearchBackend):
    name = "semantic_scholar"

    async def search(self, keywords, max_results, year_min=None, year_max=None, categories=None):
        return [
            RawPaper(
                title="Stochastic Gradient Langevin Dynamics",
                authors=["Max Welling", "John Smith"],
                abstract="A paper about stochastic gradient Langevin dynamics.",
                url="https://example.org/sgld",
                year=2014,
                source=self.name,
                arxiv_id=None,
                doi="10.0000/example",
            )
        ]


def test_article_parser_falls_back_to_semantic_scholar_when_arxiv_returns_nothing():
    parser = ArticleParser(_EmptyBackend(), fallback_backends=[_SemanticHitBackend()])

    paper = parser.find_article("Smith, Welling, 2014, stochastic gradient Langevin dynamics")

    assert paper is not None
    assert paper.source == "semantic_scholar"
    assert paper.title == "Stochastic Gradient Langevin Dynamics"
    assert paper.doi == "10.0000/example"


class _MixedBackend(SearchBackend):
    name = "mixed"

    async def search(self, keywords, max_results, year_min=None, year_max=None, categories=None):
        return [
            RawPaper(
                title="Neutral dynamics with environmental noise",
                authors=["Different Author", "Another Person"],
                abstract="A paper about environmental noise.",
                url="https://example.org/wrong",
                year=2015,
                source=self.name,
                arxiv_id="1505.02888",
                doi=None,
            ),
            RawPaper(
                title="Theory of time-averaged neutral dynamics with environmental stochasticity",
                authors=["Matan Danino", "Nadav M. Shnerb"],
                abstract="Target paper.",
                url="https://example.org/right",
                year=2017,
                source=self.name,
                arxiv_id="1711.11332",
                doi=None,
            ),
        ]


def test_article_parser_enforces_required_authors_from_lookup_spec():
    parser = ArticleParser(_MixedBackend())

    paper = parser.find_article(
        "Danino, Shnerb, 2017, neutral dynamics",
        lookup_spec={
            "query_string": "Danino, Shnerb, 2017, neutral dynamics",
            "query_terms": ["Danino", "Shnerb", "2017", "neutral dynamics"],
            "required_authors": ["Danino", "Shnerb"],
            "preferred_year": 2017,
            "title_phrases": ["time-averaged neutral dynamics"],
        },
    )

    assert paper is not None
    assert paper.title == "Theory of time-averaged neutral dynamics with environmental stochasticity"
    assert paper.year == 2017
    assert paper.authors == ["Matan Danino", "Nadav M. Shnerb"]


def test_fetch_full_text_can_fallback_to_paper_url_html(monkeypatch):
    parser = ArticleParser(_EmptyBackend())
    body = "<html><body>" + ("Stochastic gradient Langevin dynamics full text. " * 80) + "</body></html>"

    class _Response:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self):
            return body.encode("utf-8")

    def _fake_urlopen(req, timeout=0):
        if isinstance(req, urllib.request.Request):
            assert req.full_url == "https://example.org/paper"
        return _Response()

    monkeypatch.setattr(urllib.request, "urlopen", _fake_urlopen)

    text = parser.fetch_full_text("", fallback_abstract="short abstract", paper_url="https://example.org/paper")

    assert "stochastic gradient langevin dynamics full text" in text.lower()
    assert len(text) > 1000
