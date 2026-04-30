"""
sim_tool.api_server
───────────────────
FastAPI app exposing the simulation orchestrator and literature reviewer.

Install extras first:
    uv sync --extra api --extra full

Run:
    uv run python -m sim_tool.api_server --host 127.0.0.1 --port 8001
"""
from __future__ import annotations

import argparse
import asyncio
import json
import queue
import threading
import time
import uuid
from dataclasses import dataclass, fields, is_dataclass
from enum import Enum
from pathlib import Path
from typing import Any, AsyncIterator, Optional

try:
    from fastapi import FastAPI, HTTPException
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import StreamingResponse
    from pydantic import BaseModel, ConfigDict, Field
except ImportError as exc:  # pragma: no cover - exercised only when deps missing
    raise SystemExit(
        "sim_tool.api_server requires FastAPI. Install with `uv sync --extra api`."
    ) from exc

from literature_review.contract import (
    LiteratureReviewTask,
    ScopeConstraint,
    SearchDepthConfig,
)
from literature_review.director_bridge import LiteratureReviewBridge
from literature_review.llm_interface import ThreadedAsyncAdapter
from literature_review.search_backends.arxiv_backend import ArXivBackend
from literature_review.search_backends.base import SearchBackend
from literature_review.search_backends.perplexity_backend import PerplexityBackend
from literature_review.search_backends.semantic_scholar_backend import (
    SemanticScholarBackend,
)
from research_platform.assistants import make_service_backend
from research_platform.flow_management import FlowManagementService
from research_platform.registry import ArtifactRegistry
from research_platform.service_contracts import AgentRuntimeProfile, MixedServiceRuntimeConfig, ServiceRuntimeConfig
from research_platform.writer_runtime import run_writer_pipeline

from .launcher import LaunchRuntimeConfig, Launcher
from .llm import LLMBackend, make_backend
from .orchestrator import OrchestratorUpdate, ResearchOrchestrator
from .tool import SimulationTool

try:
    from writer.contract import StreamChunk
    _WRITER_AVAILABLE = True
except ImportError:
    _WRITER_AVAILABLE = False


def _jsonify(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, dict):
        return {str(k): _jsonify(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonify(v) for v in value]
    if is_dataclass(value):
        payload = {f.name: _jsonify(getattr(value, f.name)) for f in fields(value)}
        payload["rendered"] = str(value)
        if hasattr(value, "data_files"):
            try:
                payload["data_files"] = _jsonify(getattr(value, "data_files"))
            except Exception:
                pass
        for attr in ("n_success", "n_failed", "n_max_steps", "n_crash"):
            if hasattr(value, attr):
                payload[attr] = getattr(value, attr)
        return payload
    return str(value)


def _http_error(exc: Exception) -> HTTPException:
    if isinstance(exc, KeyError):
        return HTTPException(status_code=404, detail=str(exc))
    if isinstance(exc, ValueError):
        return HTTPException(status_code=400, detail=str(exc))
    if isinstance(exc, TimeoutError) or "timed out" in str(exc).lower():
        return HTTPException(status_code=504, detail=str(exc))
    return HTTPException(status_code=500, detail=str(exc))


class AgentLLMConfig(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="ignore")

    provider: Optional[str] = None
    model: Optional[str] = None
    base_url: Optional[str] = Field(default=None, alias="url")
    api_key: Optional[str] = Field(default=None, alias="apiKey")
    timeout_seconds: Optional[float] = Field(default=None, alias="timeoutSeconds")
    enabled: bool = True


class LauncherConfigRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="ignore")

    primary_mode: Optional[str] = Field(default=None, alias="primaryMode")
    fallback_mode: Optional[str] = Field(default=None, alias="fallbackMode")
    docker_binary: Optional[str] = Field(default=None, alias="dockerBinary")
    docker_image: Optional[str] = Field(default=None, alias="dockerImage")
    docker_workdir: Optional[str] = Field(default=None, alias="dockerWorkdir")
    docker_auto_pull: Optional[bool] = Field(default=None, alias="dockerAutoPull")


class SimulationStartRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="ignore")

    research_goal: str = Field(alias="researchGoal")
    context_notes: str = Field(default="", alias="contextNotes")
    output_root: str = Field(default="./simulations", alias="outputRoot")
    sample_steps: int = Field(default=200, alias="sampleSteps")
    designer: Optional[AgentLLMConfig] = None
    analyst: Optional[AgentLLMConfig] = None
    launcher: Optional[LauncherConfigRequest] = None


class AnswersRequest(BaseModel):
    answers: dict[str, str]


class FeedbackRequest(BaseModel):
    feedback: str = ""


class WaitRequest(BaseModel):
    timeout_seconds: Optional[float] = Field(default=None, alias="timeoutSeconds")


class QueryRequest(BaseModel):
    question: str


class LiteratureArxivRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="ignore")

    enabled: bool = True
    sort_by: str = Field(default="relevance", alias="sortBy")


class LiteratureSemanticScholarRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="ignore")

    enabled: bool = True
    api_key: Optional[str] = Field(default=None, alias="apiKey")


class LiteraturePerplexityRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="ignore")

    enabled: bool = False
    api_key: Optional[str] = Field(default=None, alias="apiKey")
    model: str = "llama-3.1-sonar-large-128k-online"


class LiteratureReviewRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="ignore")

    query: str
    include_topics: list[str] = Field(default_factory=list, alias="includeTopics")
    exclude_topics: list[str] = Field(default_factory=list, alias="excludeTopics")
    year_min: Optional[int] = Field(default=None, alias="yearMin")
    year_max: Optional[int] = Field(default=None, alias="yearMax")
    max_papers: int = Field(default=10, alias="maxPapers")
    papers_per_query: int = Field(default=8, alias="papersPerQuery")
    min_papers_threshold: int = Field(default=4, alias="minPapersThreshold")
    max_rounds: int = Field(default=3, alias="maxRounds")
    max_term_variations: int = Field(default=2, alias="maxTermVariations")
    requestor_agent: str = Field(default="frontend", alias="requestorAgent")
    branch_id: str = Field(default="frontend", alias="branchId")
    llm: Optional[AgentLLMConfig] = None
    arxiv: LiteratureArxivRequest = Field(default_factory=LiteratureArxivRequest)
    semantic_scholar: LiteratureSemanticScholarRequest = Field(
        default_factory=LiteratureSemanticScholarRequest,
        alias="semanticScholar",
    )
    perplexity: LiteraturePerplexityRequest = Field(
        default_factory=LiteraturePerplexityRequest
    )


class PIDirectiveRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="ignore")

    directive_id: str = Field(default_factory=lambda: f"dir-{int(time.time())}", alias="directiveId")
    pi_name: str = Field(alias="piName")
    instruction: str
    topic_hint: str = Field(default="", alias="topicHint")
    phases: list[str] = Field(default_factory=list)
    output_dir: str = Field(default="./programs/directive-run", alias="outputDir")
    llm: Optional[AgentLLMConfig] = None
    agent_profiles: dict[str, AgentLLMConfig] = Field(default_factory=dict, alias="agentProfiles")


class DirectiveSteeringRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="ignore")

    checkpoint_id: str = Field(default="", alias="checkpointId")
    session_id: Optional[str] = Field(default=None, alias="sessionId")
    action: str = "continue"
    feedback: str = ""
    mode: Optional[str] = None
    reason: str = ""
    payload: dict[str, Any] = Field(default_factory=dict)


class DirectiveStopRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="ignore")

    reason: str = ""

def _make_llm_backend(service: str, config: Optional[AgentLLMConfig]) -> LLMBackend:
    if config is not None and not config.enabled:
        raise ValueError(f"{service} backend is disabled")
    provider = (config.provider or "").strip().lower() if config else ""
    if provider == "deterministic":
        raise ValueError(f"{service} requires an LLM backend, not deterministic")
    return make_service_backend(service, config)


def _make_launcher(config: Optional[LauncherConfigRequest]) -> Launcher:
    if config is None:
        return Launcher()
    runtime = LaunchRuntimeConfig(
        primary_mode=(config.primary_mode or "docker").strip().lower(),
        fallback_mode=(config.fallback_mode or "process").strip().lower(),
        docker_binary=(config.docker_binary or "docker").strip() or "docker",
        docker_image=(config.docker_image or "research-platform-launcher:latest").strip()
        or "research-platform-launcher:latest",
        docker_workdir=(config.docker_workdir or "/workspace").strip() or "/workspace",
        docker_auto_pull=bool(config.docker_auto_pull),
    )
    return Launcher(runtime=runtime)


def _compose_goal(research_goal: str, context_notes: str) -> str:
    research_goal = research_goal.strip()
    context_notes = context_notes.strip()
    if not context_notes:
        return research_goal
    return (
        f"{research_goal}\n\n"
        "Upstream context (treat as constraints and useful guidance, not optional prose):\n"
        f"{context_notes}"
    )


@dataclass
class _SimulationContext:
    tool: SimulationTool
    orchestrator: ResearchOrchestrator
    created_at: float


class SimulationSessionManager:
    def __init__(self) -> None:
        self._sessions: dict[str, _SimulationContext] = {}
        self._lock = threading.Lock()

    def start(self, request: SimulationStartRequest) -> OrchestratorUpdate:
        tool = SimulationTool(
            output_root=request.output_root,
            designer_backend=_make_llm_backend("designer", request.designer),
            analyst_backend=_make_llm_backend("analyst", request.analyst),
            launcher=_make_launcher(request.launcher),
        )
        orchestrator = ResearchOrchestrator(tool, sample_steps=request.sample_steps)
        update = orchestrator.start(
            _compose_goal(request.research_goal, request.context_notes)
        )
        with self._lock:
            self._sessions[update.session_id] = _SimulationContext(
                tool=tool,
                orchestrator=orchestrator,
                created_at=time.time(),
            )
        return update

    def _get(self, session_id: str) -> _SimulationContext:
        with self._lock:
            context = self._sessions.get(session_id)
        if context is None:
            raise KeyError(f"Unknown simulation session: {session_id}")
        return context

    def answer(self, session_id: str, answers: dict[str, str]) -> OrchestratorUpdate:
        return self._get(session_id).orchestrator.answer(session_id, answers)

    def approve(self, session_id: str) -> OrchestratorUpdate:
        return self._get(session_id).orchestrator.approve(session_id)

    def reject(self, session_id: str, feedback: str) -> OrchestratorUpdate:
        return self._get(session_id).orchestrator.reject(session_id, feedback)

    def run_sample(self, session_id: str) -> OrchestratorUpdate:
        return self._get(session_id).orchestrator.run_sample(session_id)

    def approve_sample(self, session_id: str) -> OrchestratorUpdate:
        return self._get(session_id).orchestrator.approve_sample(session_id)

    def get_status(self, session_id: str) -> OrchestratorUpdate:
        return self._get(session_id).orchestrator.get_status(session_id)

    def wait(self, session_id: str, timeout_seconds: Optional[float]) -> OrchestratorUpdate:
        return self._get(session_id).orchestrator.wait(session_id, timeout=timeout_seconds)

    def query(self, session_id: str, question: str) -> OrchestratorUpdate:
        return self._get(session_id).orchestrator.query(session_id, question)

    def approve_results(self, session_id: str) -> OrchestratorUpdate:
        return self._get(session_id).orchestrator.approve_results(session_id)

    def reject_results(self, session_id: str, feedback: str) -> OrchestratorUpdate:
        return self._get(session_id).orchestrator.reject_results(session_id, feedback)

    def apply_patch_and_rerun(self, session_id: str) -> OrchestratorUpdate:
        return self._get(session_id).orchestrator.apply_patch_and_rerun(session_id)


# ── Writer job service ────────────────────────────────────────────────────────

class WriterArtifactInput(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="ignore")

    artifact_id: Optional[str] = Field(default=None, alias="artifactId")
    assistant: Optional[str] = None
    agent: Optional[str] = None
    source_agent: Optional[str] = Field(default=None, alias="sourceAgent")
    kind: str = "source_artifact"
    title: str = ""
    summary: str = ""
    path: Optional[str] = None
    url: Optional[str] = None
    content: Optional[str] = None
    mime_type: str = Field(default="text/plain", alias="mimeType")
    metadata: dict[str, Any] = Field(default_factory=dict)


class WriterStartRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="ignore")

    description: str
    venue: str = "arXiv preprint"
    artifact_context: str = Field(default="", alias="artifactContext")
    artifact_catalog: list[WriterArtifactInput] = Field(
        default_factory=list,
        alias="artifactCatalog",
    )
    llm: Optional[AgentLLMConfig] = None


@dataclass
class _WriterJob:
    job_id: str
    event_queue: "queue.Queue[Optional[dict]]"
    result: Optional[dict] = None
    error: Optional[str] = None
    done: bool = False


@dataclass
class _DirectiveJob:
    directive_id: str
    event_queue: "queue.Queue[Optional[dict]]"
    flow_management: FlowManagementService
    mailbox: Any = None
    done: bool = False
    error: Optional[str] = None


class WriterService:
    def __init__(self) -> None:
        self._jobs: dict[str, _WriterJob] = {}
        self._lock = threading.Lock()

    def start(self, request: WriterStartRequest) -> str:
        if not _WRITER_AVAILABLE:
            raise ValueError("writer package not available — install research-platform with writer extras")
        job_id = f"wj-{uuid.uuid4().hex[:8]}"
        q: queue.Queue[Optional[dict]] = queue.Queue()
        job = _WriterJob(job_id=job_id, event_queue=q)
        with self._lock:
            self._jobs[job_id] = job

        llm_backend = _make_llm_backend("writer", request.llm)
        registry = ArtifactRegistry(root_dir=Path("./writer_jobs") / job_id)
        artifact_catalog = [
            item.model_dump(by_alias=False)
            for item in request.artifact_catalog
        ]

        def _run() -> None:
            """Run AcademicWriter in a background thread, emitting NDJSON events."""
            try:
                result = run_writer_pipeline(
                    llm_backend=llm_backend,
                    registry=registry,
                    description=request.description,
                    venue=request.venue,
                    artifact_catalog=artifact_catalog,
                    artifact_context=request.artifact_context,
                    stream_callback=lambda chunk: q.put(_chunk_to_dict(chunk)),
                )
                job.result = {
                    "job_id": job_id,
                    **result,
                    "artifacts": _jsonify(registry.all()),
                }
            except Exception as exc:
                job.error = str(exc)
                q.put({"kind": "error", "text": str(exc)})
            finally:
                job.done = True
                q.put(None)  # sentinel

        t = threading.Thread(target=_run, daemon=True)
        t.start()
        return job_id

    def _get(self, job_id: str) -> _WriterJob:
        with self._lock:
            job = self._jobs.get(job_id)
        if job is None:
            raise KeyError(f"Unknown writer job: {job_id}")
        return job

    def stream(self, job_id: str) -> AsyncIterator[str]:
        job = self._get(job_id)

        async def _gen() -> AsyncIterator[str]:
            loop = asyncio.get_event_loop()
            while True:
                item = await loop.run_in_executor(None, job.event_queue.get)
                if item is None:
                    break
                yield json.dumps(item) + "\n"

        return _gen()

    def result(self, job_id: str) -> dict[str, Any]:
        job = self._get(job_id)
        if not job.done:
            raise ValueError("Job not yet complete")
        if job.error:
            raise RuntimeError(job.error)
        return job.result or {}


def _chunk_to_dict(chunk: "StreamChunk") -> dict:
    return {
        "kind": chunk.kind,
        "text": chunk.text,
        "section_id": chunk.section_id,
        "payload": _jsonify(chunk.payload),
        "ts": chunk.ts,
    }


class _StreamingMailbox:
    def __init__(self, q: queue.Queue, output_dir: Path):
        self.q = q
        from comms.pi_email import PIMailbox
        self.underlying = PIMailbox(output_dir=output_dir)
        
    def append(self, email):
        self.underlying.append(email)
        self.q.put({"type": "email", "email": email.to_dict()})
        
    def all_emails(self):
        return self.underlying.all_emails()

class DirectiveService:
    def __init__(self) -> None:
        self._jobs: dict[str, _DirectiveJob] = {}
        self._lock = threading.Lock()

    def _register(self, job: _DirectiveJob) -> None:
        with self._lock:
            self._jobs[job.directive_id] = job

    def _get(self, directive_id: str) -> _DirectiveJob:
        with self._lock:
            job = self._jobs.get(directive_id)
        if job is None:
            raise KeyError(f"Unknown directive run: {directive_id}")
        return job

    def _close(self, directive_id: str) -> None:
        with self._lock:
            self._jobs.pop(directive_id, None)

    def run_stream(self, request: PIDirectiveRequest) -> AsyncIterator[str]:
        q: queue.Queue[Optional[dict]] = queue.Queue()
        flow = FlowManagementService(event_sink=lambda item: q.put({"type": "flow_event", "event": item}))
        job = _DirectiveJob(
            directive_id=request.directive_id,
            event_queue=q,
            flow_management=flow,
        )
        self._register(job)
        
        def _run():
            try:
                from sim_tool.research_directive import PIDirective, DirectiveRunner
                directive = PIDirective(
                    directive_id=request.directive_id,
                    pi_name=request.pi_name,
                    instruction=request.instruction,
                    topic_hint=request.topic_hint,
                    phases=request.phases,
                    output_dir=request.output_dir,
                    llm=request.llm,
                    agent_profiles=request.agent_profiles,
                )
                mailbox = _StreamingMailbox(q, Path(request.output_dir))
                job.mailbox = mailbox
                runner = DirectiveRunner(directive, mailbox, flow_management=flow)
                runner.run_all_phases()
            except Exception as e:
                job.error = str(e)
                q.put({"type": "status", "phase": "error", "message": str(e)})
            finally:
                job.done = True
                q.put(None)
                
        threading.Thread(target=_run, daemon=True).start()
        
        async def _gen() -> AsyncIterator[str]:
            loop = asyncio.get_event_loop()
            try:
                while True:
                    item = await loop.run_in_executor(None, q.get)
                    if item is None:
                        break
                    yield json.dumps(item) + "\n"
            finally:
                if job.done:
                    self._close(request.directive_id)
        return _gen()

    def submit_steering(self, directive_id: str, request: DirectiveSteeringRequest) -> dict[str, Any]:
        job = self._get(directive_id)
        target_session_id = request.session_id or job.flow_management.latest_session_id(directive_id)
        if not target_session_id:
            raise KeyError(f"No running session available for directive {directive_id}")
        mode = (request.mode or ("hard" if request.action.strip().lower() == "revise" else "soft")).strip().lower()
        payload = dict(request.payload)
        if not payload:
            payload = {
                "action": request.action,
                "feedback": request.feedback,
                "checkpoint_id": request.checkpoint_id,
            }
        injection = job.flow_management.inject_context(
            target_session_id,
            payload=payload,
            reason=request.reason or request.feedback or request.action,
            mode=mode,
            injected_by="api",
        )
        job.event_queue.put(
            {
                "type": "steering",
                "directiveId": directive_id,
                "checkpointId": request.checkpoint_id,
                "sessionId": target_session_id,
                "action": request.action,
                "feedback": request.feedback,
                "mode": mode,
            }
        )
        return {
            "directive_id": directive_id,
            "checkpoint_id": request.checkpoint_id,
            "session_id": target_session_id,
            "action": request.action,
            "feedback": request.feedback,
            "mode": injection.mode,
        }

    def stop(self, directive_id: str, reason: str = "") -> dict[str, Any]:
        job = self._get(directive_id)
        job.flow_management.request_stop(directive_id, reason)
        job.event_queue.put(
            {
                "type": "directive_stop_requested",
                "directiveId": directive_id,
                "reason": reason,
            }
        )
        return {
            "directive_id": directive_id,
            "status": "stop_requested",
            "reason": reason,
        }

    def snapshot(self, directive_id: str) -> dict[str, Any]:
        job = self._get(directive_id)
        mailbox = job.mailbox
        emails = []
        if mailbox is not None:
            emails = [email.to_dict() for email in mailbox.all_emails()]
        flow_snapshot = job.flow_management.get_workflow_snapshot(directive_id)
        return {
            "directive_id": directive_id,
            "done": job.done,
            "error": job.error,
            "emails": emails,
            "pending_steering": None,
            "stop_requested": flow_snapshot["stop_requested"],
            "stop_reason": flow_snapshot["stop_reason"],
            "flow": flow_snapshot,
        }


class LiteratureReviewService:
    def run(self, request: LiteratureReviewRequest) -> dict[str, Any]:
        backends: list[SearchBackend] = []
        if request.arxiv.enabled:
            backends.append(ArXivBackend(sort_by=request.arxiv.sort_by))
        if request.semantic_scholar.enabled:
            backends.append(
                SemanticScholarBackend(api_key=request.semantic_scholar.api_key)
            )
        if request.perplexity.enabled:
            backends.append(
                PerplexityBackend(
                    api_key=request.perplexity.api_key,
                    model=request.perplexity.model,
                )
            )
        if not backends:
            raise ValueError("At least one literature search backend must be enabled")

        llm_backend = _make_llm_backend("literature_review", request.llm)
        bridge = LiteratureReviewBridge(
            backends=backends,
            llm=ThreadedAsyncAdapter(llm_backend),
            gui_event_sink=None,
        )
        task = LiteratureReviewTask(
            task_id=f"lit-{int(time.time() * 1000)}",
            branch_id=request.branch_id,
            query=request.query,
            scope=ScopeConstraint(
                include_topics=request.include_topics or [request.query],
                exclude_topics=request.exclude_topics,
                year_min=request.year_min,
                year_max=request.year_max,
                max_papers=request.max_papers,
            ),
            depth=SearchDepthConfig(
                max_rounds=request.max_rounds,
                max_term_variations=request.max_term_variations,
                min_papers_threshold=request.min_papers_threshold,
                papers_per_query=request.papers_per_query,
            ),
            requestor_agent=request.requestor_agent,
        )
        result = bridge.run_sync(task)
        return {
            "artifact": _jsonify(result.artifact),
            "audit": _jsonify(result.audit),
        }


def create_app(
    simulation_manager: Optional[SimulationSessionManager] = None,
    literature_service: Optional[LiteratureReviewService] = None,
    writer_service: Optional[WriterService] = None,
    directive_service: Optional[DirectiveService] = None,
) -> FastAPI:
    simulation_manager = simulation_manager or SimulationSessionManager()
    literature_service = literature_service or LiteratureReviewService()
    writer_service = writer_service or WriterService()
    directive_service = directive_service or DirectiveService()

    app = FastAPI(title="Research Platform API", version="0.1.0")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/api/health/llm")
    def health_llm(request: AgentLLMConfig) -> dict[str, Any]:
        """
        Probe an LLM backend config with a minimal completion.
        Returns {ok, model, latency_ms, error}.
        """
        import time as _time
        try:
            backend = _make_llm_backend("probe", request)
            t0 = _time.monotonic()
            reply = backend.complete(
                system="You are a ping responder. Reply with exactly: pong",
                messages=[{"role": "user", "content": "ping"}],
                max_tokens=8,
                temperature=0.0,
            )
            latency_ms = round((_time.monotonic() - t0) * 1000)
            return {
                "ok": True,
                "model": getattr(backend, "model_name", request.model or "unknown"),
                "reply": reply.strip()[:40],
                "latency_ms": latency_ms,
                "error": None,
            }
        except Exception as exc:
            return {"ok": False, "model": None, "reply": None, "latency_ms": None, "error": str(exc)}

    @app.post("/api/simulations/start")
    def simulation_start(request: SimulationStartRequest) -> dict[str, Any]:
        try:
            return _jsonify(simulation_manager.start(request))
        except Exception as exc:
            raise _http_error(exc) from exc

    @app.post("/api/simulations/{session_id}/answer")
    def simulation_answer(session_id: str, request: AnswersRequest) -> dict[str, Any]:
        try:
            return _jsonify(simulation_manager.answer(session_id, request.answers))
        except Exception as exc:
            raise _http_error(exc) from exc

    @app.post("/api/simulations/{session_id}/approve")
    def simulation_approve(session_id: str) -> dict[str, Any]:
        try:
            return _jsonify(simulation_manager.approve(session_id))
        except Exception as exc:
            raise _http_error(exc) from exc

    @app.post("/api/simulations/{session_id}/reject")
    def simulation_reject(session_id: str, request: FeedbackRequest) -> dict[str, Any]:
        try:
            return _jsonify(simulation_manager.reject(session_id, request.feedback))
        except Exception as exc:
            raise _http_error(exc) from exc

    @app.post("/api/simulations/{session_id}/run-sample")
    def simulation_run_sample(session_id: str) -> dict[str, Any]:
        try:
            return _jsonify(simulation_manager.run_sample(session_id))
        except Exception as exc:
            raise _http_error(exc) from exc

    @app.post("/api/simulations/{session_id}/approve-sample")
    def simulation_approve_sample(session_id: str) -> dict[str, Any]:
        try:
            return _jsonify(simulation_manager.approve_sample(session_id))
        except Exception as exc:
            raise _http_error(exc) from exc

    @app.get("/api/simulations/{session_id}/status")
    def simulation_status(session_id: str) -> dict[str, Any]:
        try:
            return _jsonify(simulation_manager.get_status(session_id))
        except Exception as exc:
            raise _http_error(exc) from exc

    @app.post("/api/simulations/{session_id}/wait")
    def simulation_wait(session_id: str, request: WaitRequest) -> dict[str, Any]:
        try:
            return _jsonify(
                simulation_manager.wait(session_id, request.timeout_seconds)
            )
        except Exception as exc:
            raise _http_error(exc) from exc

    @app.post("/api/simulations/{session_id}/query")
    def simulation_query(session_id: str, request: QueryRequest) -> dict[str, Any]:
        try:
            return _jsonify(simulation_manager.query(session_id, request.question))
        except Exception as exc:
            raise _http_error(exc) from exc

    @app.post("/api/simulations/{session_id}/approve-results")
    def simulation_approve_results(session_id: str) -> dict[str, Any]:
        try:
            return _jsonify(simulation_manager.approve_results(session_id))
        except Exception as exc:
            raise _http_error(exc) from exc

    @app.post("/api/simulations/{session_id}/reject-results")
    def simulation_reject_results(
        session_id: str, request: FeedbackRequest
    ) -> dict[str, Any]:
        try:
            return _jsonify(
                simulation_manager.reject_results(session_id, request.feedback)
            )
        except Exception as exc:
            raise _http_error(exc) from exc

    @app.post("/api/simulations/{session_id}/apply-patch")
    def simulation_apply_patch(session_id: str) -> dict[str, Any]:
        try:
            return _jsonify(simulation_manager.apply_patch_and_rerun(session_id))
        except Exception as exc:
            raise _http_error(exc) from exc

    @app.post("/api/literature/reviews/run")
    def literature_run(request: LiteratureReviewRequest) -> dict[str, Any]:
        try:
            return literature_service.run(request)
        except Exception as exc:
            raise _http_error(exc) from exc

    @app.post("/api/literature/reviews/stream")
    def literature_stream(request: LiteratureReviewRequest) -> StreamingResponse:
        """Same as /run but returns result as a single NDJSON line when done."""
        async def _gen() -> AsyncIterator[str]:
            loop = asyncio.get_event_loop()
            try:
                result = await loop.run_in_executor(None, literature_service.run, request)
                yield json.dumps({"kind": "complete", "data": result}) + "\n"
            except Exception as exc:
                yield json.dumps({"kind": "error", "text": str(exc)}) + "\n"
        return StreamingResponse(_gen(), media_type="application/x-ndjson")

    @app.post("/api/writer/start")
    def writer_start(request: WriterStartRequest) -> dict[str, Any]:
        try:
            job_id = writer_service.start(request)
            return {"job_id": job_id}
        except Exception as exc:
            raise _http_error(exc) from exc

    @app.get("/api/writer/{job_id}/stream")
    def writer_stream(job_id: str) -> StreamingResponse:
        try:
            gen = writer_service.stream(job_id)
            return StreamingResponse(gen, media_type="application/x-ndjson")
        except Exception as exc:
            raise _http_error(exc) from exc

    @app.get("/api/writer/{job_id}/result")
    def writer_result(job_id: str) -> dict[str, Any]:
        try:
            return writer_service.result(job_id)
        except Exception as exc:
            raise _http_error(exc) from exc

    @app.post("/api/directives")
    def directives_stream(request: PIDirectiveRequest) -> StreamingResponse:
        try:
            gen = directive_service.run_stream(request)
            return StreamingResponse(gen, media_type="application/x-ndjson")
        except Exception as exc:
            raise _http_error(exc) from exc

    @app.post("/api/directives/{directive_id}/steering")
    def directives_steering(directive_id: str, request: DirectiveSteeringRequest) -> dict[str, Any]:
        try:
            return directive_service.submit_steering(directive_id, request)
        except Exception as exc:
            raise _http_error(exc) from exc

    @app.post("/api/directives/{directive_id}/stop")
    def directives_stop(directive_id: str, request: DirectiveStopRequest) -> dict[str, Any]:
        try:
            return directive_service.stop(directive_id, request.reason)
        except Exception as exc:
            raise _http_error(exc) from exc

    @app.get("/api/directives/{directive_id}/snapshot")
    def directives_snapshot(directive_id: str) -> dict[str, Any]:
        try:
            return directive_service.snapshot(directive_id)
        except Exception as exc:
            raise _http_error(exc) from exc

    return app


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Research Platform API server")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8001)
    args = parser.parse_args()

    try:
        import uvicorn
    except ImportError as exc:  # pragma: no cover - exercised only when deps missing
        raise SystemExit(
            "sim_tool.api_server requires uvicorn. Install with `uv sync --extra api`."
        ) from exc

    uvicorn.run(create_app(), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
