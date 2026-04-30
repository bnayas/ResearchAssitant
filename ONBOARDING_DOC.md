# Onboarding Guide

This document is the guided reading path for a developer who is new to the repository.

The goal is not to read every file. The goal is to build the right mental model quickly:

1. what the product is,
2. where orchestration lives,
3. which package owns which responsibility,
4. which functions and classes are the real entry points,
5. how to make a change without weakening subsystem boundaries.

## How To Use This Guide

Read in layers.

- First read the product-level docs so you know what the platform is supposed to do.
- Then read the shared platform layer, because that is the contract boundary between units.
- Only after that dive into a subsystem such as simulation, literature, or writing.
- End with tests, because the tests are easier to interpret once you know the shapes they are asserting.

For each file below, this guide tells you:

- why the file exists,
- which symbols to read first,
- what question that file answers.

## The Fastest Useful Pass

If you only have 30 to 60 minutes, do this:

1. Read [README.md](README.md) and [project/AGENTS.md](project/AGENTS.md).
2. Read [project/research_platform/contracts.py](project/research_platform/contracts.py).
3. Read [project/research_platform/article_lookup.py](project/research_platform/article_lookup.py).
4. Read [project/research_platform/workflow.py](project/research_platform/workflow.py).
5. Read [project/research_platform/assistants/literature.py](project/research_platform/assistants/literature.py).
6. Skim [project/research_platform/assistants/coding.py](project/research_platform/assistants/coding.py) and [project/research_platform/assistants/writer.py](project/research_platform/assistants/writer.py).
7. Read [project/tests/test_professor_workflow.py](project/tests/test_professor_workflow.py).

If you do only that, you should already understand:

- how a PI directive becomes a workflow,
- how the orchestrator talks to assistants,
- how artifacts and mailbox messages move through the system,
- where to patch an orchestration bug.

## Mental Model To Keep In Your Head

Use this model while reading:

- `research_platform` is the only layer that should know about multiple assistants at once.
- `sim_tool` owns simulation design, execution, and analysis.
- `literature_review` owns article discovery, article interrogation, and related-literature synthesis.
- `writer` turns upstream artifacts into a grounded draft.
- `reviewer` is a separate manuscript-review subsystem.
- `frontend` is a thin UI shell over API contracts.
- `comms` stores the PI-facing mailbox history.

When in doubt:

- shared contracts belong in `research_platform`,
- subsystem logic belongs in the subsystem that owns it,
- workflow sequencing belongs in the orchestrator,
- UI normalization belongs in frontend helpers, not backend code.

## Guided Reading Order

### 1. Start With The Product Shape

[README.md](README.md)

- Read for: the product summary, the repo layout, the quick-start commands, and the docs map.
- Question it answers: what is this repository for, and where does the real code live?

[project/README.md](project/README.md)

- Read for: the `project/` working model and the commands developers actually run from inside the package directory.
- Question it answers: what is inside `project/`, and what is source code versus workspace wrapper?

[project/AGENTS.md](project/AGENTS.md)

- Read for: the intended agent boundaries and the architectural vocabulary used across the repo.
- Start with: the architecture overview, then the sections for `sim_tool`, `literature_review`, and `writer`.
- Question it answers: what are the conceptual agents and what must each one never do?

[project/examples/danino_repro.json](project/examples/danino_repro.json)

- Read for: the shape of a real directive payload.
- Important note: this is an example fixture, not a live product default.
- Question it answers: what does the professor workflow accept as input?

### 2. Understand The Shared Platform Layer First

This is the most important reading block in the repo. If you understand these files, the rest of the codebase becomes much easier to place.

[project/research_platform/contracts.py](project/research_platform/contracts.py)

- Read first: `AssistantId`, `AttachmentRef`, `ArtifactRef`, `ArtifactRef.as_attachment()`, `MailboxMessage`, `TaskEnvelope`.
- Then read: `LiteratureAssistantPort`, `CodingAssistantPort`, `WriterAssistantPort`.
- Question it answers: what shapes are allowed to cross subsystem boundaries?
- Expected takeaway: if a payload crosses from one assistant to another, it should probably be represented here.

[project/research_platform/article_lookup.py](project/research_platform/article_lookup.py)

- Read first: `build_article_lookup_query()` and `article_query_candidates()`.
- Then read: `_derive_terms()` and `_extract_topic_phrases()`.
- Question it answers: what deterministic fallback exists if the live lookup-query preparation step needs a local normalizer?
- Why it matters: the live workflow now asks the literature assistant LLM to prepare the lookup query first, and this file exists as the generic fallback and cleanup path.
- Important boundary: this file should stay generic. It may extract author names, years, quoted phrases, and contiguous topic phrases from the directive text, but it should not contain example-specific domain phrase tables tied to one paper or one research area.

[project/research_platform/registry.py](project/research_platform/registry.py)

- Read first: `ArtifactRegistry.create()`, `save_text()`, `save_json()`.
- Then read: `get()`, `list_by_assistant()`, `next_filename()`.
- Question it answers: how are artifacts persisted and rediscovered between phases?
- Expected takeaway: this registry is the handoff mechanism that keeps peer assistants decoupled.

[project/research_platform/workflow.py](project/research_platform/workflow.py)

- Read first: `ProfessorWorkflowRunner.run_all_phases()`.
- Then read in order: `_phase_find_article()`, `_phase_parse_article()`, `_phase_literature()`, `_phase_simulation()`, `_phase_write()`.
- Finally read: `_task()`, `_send_message()`, `_send_error()`, `_on_coding_update()`.
- Question it answers: how does the professor workflow sequence the assistants and emit PI-visible updates?
- Expected takeaway: this file is the end-to-end orchestration state machine for the professor workflow.

[project/research_platform/assistants/literature.py](project/research_platform/assistants/literature.py)

- Read first: `LocalLiteratureAssistant.prepare_article_lookup_spec()` and `find_primary_article()`.
- Then read: `build_article_brief()`, `review_related_literature()`, `_normalize_answer()`, `_coerce_parameters()`, `_coerce_figures()`.
- Question it answers: how does the local literature subsystem get translated into the shared assistant contracts?
- Expected takeaway: this adapter owns article lookup, article brief extraction, related-literature synthesis, and the normalization of those results into registry artifacts.
- Important runtime detail: the LLM now prepares a structured JSON lookup spec with query terms, required authors, preferred year, year bounds, title phrases, and optional clarification fields. Primary-article lookup searches ArXiv and Semantic Scholar and then enforces those structured constraints when ranking results.
- Clarification hook: `prepare_article_lookup_spec()` may return `needs_clarification=true` plus a `clarification_question`. If it does, `workflow.py` pauses before any live search and asks the PI for one answer or permission to continue broadly.
- Rate-limit behavior: ArXiv and Semantic Scholar are both paced conservatively, and if all configured article-search backends return HTTP `429`, the workflow stops retrying and reports a rate-limit error instead of cycling through more query variants.

[project/research_platform/assistants/coding.py](project/research_platform/assistants/coding.py)

- Read first: `LocalCodingAssistant.execute_task()`.
- Then read: `_resolve_clarifications()`, `_compose_research_goal()`, `_analysis_summary()`, `_bug_ticket_artifact()`.
- Question it answers: how is the simulation subsystem exposed as one external coding assistant?
- Expected takeaway: the orchestrator does not call `sim_tool` internals directly; this adapter accepts a generic task plus source artifacts and turns the internal design/launch/analyze lifecycle into shared coding artifacts.

[project/research_platform/assistants/writer.py](project/research_platform/assistants/writer.py)

- Read first: `LocalWriterAssistant.draft_review()`.
- Then read: `_source_catalog()`.
- Question it answers: how does the writer receive upstream artifacts without knowing about peer assistants directly?
- Expected takeaway: the writer adapter consumes a registry-backed source catalog assembled by the orchestrator, not hard-coded peer-agent knowledge.

[project/research_platform/assistants/backends.py](project/research_platform/assistants/backends.py)

- Read first: `make_service_backend()`.
- Then read: `build_literature_review_bridge_from_env()`.
- Question it answers: where do environment-driven assistant backends and literature-review bridges get constructed?
- Expected takeaway: runtime backend selection belongs here, not inside the orchestrator or subsystem code.

[project/research_platform/assistants/errors.py](project/research_platform/assistants/errors.py) and [project/research_platform/assistants/utils.py](project/research_platform/assistants/utils.py)

- Skim: `AssistantExecutionError`, `truncate()`, and `jsonify()`.
- Question they answer: what small shared helper pieces do the adapters use?
- Expected takeaway: these files keep the main adapter modules smaller and keep error/serialization behavior consistent.

[project/research_platform/assistants/__init__.py](project/research_platform/assistants/__init__.py)

- Skim: the re-exports.
- Question it answers: what public assistant-layer API should other packages import?
- Expected takeaway: callers should import from `research_platform.assistants`, while the implementation stays split by assistant responsibility.

- Why the classes are called `Local*Assistant`: these are in-process adapters used by the current CLI and API server. The `*AssistantPort` protocols in `contracts.py` are the real boundary; a future remote or separately deployed assistant would implement the same port and replace the local adapter without changing `workflow.py`.

[project/research_platform/wiring.py](project/research_platform/wiring.py)

- Read first: `build_professor_workflow_runner()`.
- Then read: `build_literature_bridge_from_env()`.
- Question it answers: where are the real runtime dependencies constructed?
- Expected takeaway: CLI and API entrypoints should assemble dependencies here, not inside the assistants themselves.

[project/research_platform/writer_runtime.py](project/research_platform/writer_runtime.py)

- Read first: `build_registry_toolbox()`, `seed_registry_from_catalog()`, `run_writer_pipeline()`.
- Skim: `WriterLLMAdapter`.
- Question it answers: how does the real writer runtime consume artifact-registry data instead of test-only mock plumbing?
- Expected takeaway: if the writer is mis-grounded or missing artifacts, this file is usually part of the story.

[project/research_platform/steering.py](project/research_platform/steering.py)

- Read first: `SteeringCheckpoint`, `SteeringDecision`, and `SteeringController.submit()`.
- Then read: `SteeringController.open()`, `pending()`, and `wait()`.
- Question it answers: how does the professor interrupt the automatic workflow and correct the assistants in place?
- Expected takeaway: the streamed directive flow is not fire-and-forget anymore; it can pause at explicit review checkpoints and wait for PI steering.

### 3. Read The Simulation Subsystem Next

Once the platform layer makes sense, move to the subsystem with the largest typed lifecycle.

[project/sim_tool/contract.py](project/sim_tool/contract.py)

- Read first: `ClarificationRequest`, `SpecApproval`, `GeneratedArtifacts`, `RunResult`, `RunSummary`, `AnalysisResult`, `SpecPatch`, `BugTicket`.
- Skim: `QuestionTopic`, `Verdict`, `BugKind`.
- Question it answers: what are the typed stages of a simulation session?
- Expected takeaway: this file tells you what kinds of outputs the simulation pipeline can legally produce.

[project/sim_tool/tool.py](project/sim_tool/tool.py)

- Read first: `SimulationTool.start()`, `answer()`, `approve()`.
- Then read: `run_single()`, `run_sweep()`, `analyse_sweep()`, `apply_patch()`.
- Skim: `_generate()` and `_get()`.
- Question it answers: what is the subsystem façade that the orchestrator ultimately drives?
- Expected takeaway: this file exposes the simulation lifecycle without the higher-level professor workflow.

[project/sim_tool/orchestrator.py](project/sim_tool/orchestrator.py)

- Read first: `ResearchOrchestrator.start()`.
- Then read in order: `answer()`, `approve()`, `run_sample()`, `approve_sample()`, `wait()`, `approve_results()`, `apply_patch_and_rerun()`.
- Skim: `OrchestratorState`, `OrchestratorUpdate`, `_route_designer_result()`, `_run_full_sweep()`.
- Question it answers: how is the simulation lifecycle turned into a gated, stateful session?
- Expected takeaway: this file is about simulation-session control flow, not cross-assistant orchestration.

[project/sim_tool/api_server.py](project/sim_tool/api_server.py)

- Read first: `create_app()`.
- Then read the internal service classes in this order: `SimulationSessionManager`, `WriterService`, `DirectiveService`, `LiteratureReviewService`.
- Read the request models only after you understand the services: `SimulationStartRequest`, `LiteratureReviewRequest`, `WriterStartRequest`, `PIDirectiveRequest`.
- Question it answers: how do HTTP endpoints map onto the internal services and workflow runner?
- Expected takeaway: if the UI and the backend disagree, this file is one of the first places to inspect.

[project/sim_tool/run_directive.py](project/sim_tool/run_directive.py)

- Read first: `main()`.
- Question it answers: how does a JSON directive file become a mailbox-producing workflow run from the CLI?

[project/sim_tool/research_directive.py](project/sim_tool/research_directive.py)

- Read first: `PIDirective` and `DirectiveRunner`.
- Question it answers: what compatibility seam is preserved for older call sites?
- Expected takeaway: this file is a shim over `research_platform`, not the true home of the workflow logic.

[project/sim_tool/designer.py](project/sim_tool/designer.py)

- Skim after the files above.
- Read for: how the spec-design LLM is prompted and repaired.
- Question it answers: where do clarification questions and initial specs come from?

[project/sim_tool/launcher.py](project/sim_tool/launcher.py)

- Skim after `tool.py`.
- Read for: the deterministic execution path and bug classification.
- Question it answers: how does generated code actually get run?

[project/sim_tool/analyst.py](project/sim_tool/analyst.py)

- Skim after `launcher.py`.
- Read for: how run outputs turn into verdicts and patches.
- Question it answers: how are results classified after execution?

### 4. Read The Literature Subsystem

Now read the assistant that owns article lookup and related-paper synthesis.

[project/literature_review/contract.py](project/literature_review/contract.py)

- Read first: `ScopeConstraint`, `SearchDepthConfig`, `LiteratureReviewTask`, `LiteraturePaper`, `LiteratureReviewArtifact`, `StreamEvent`, `LiteratureAuditResult`.
- Question it answers: what are the stable data shapes inside the literature subsystem?

[project/literature_review/article_parser.py](project/literature_review/article_parser.py)

- Read first: `ArticleParser.find_article()`, `fetch_full_text()`, `extract_answers()`.
- Question it answers: how is one target article found and then interrogated?
- Expected takeaway: this file owns target-paper lookup and article-question answering, not related-literature synthesis.
- Runtime detail: `find_article()` now searches the configured primary backend and then fallback backends, while `fetch_full_text()` tries ar5iv first and then the paper URL as an HTML page before falling back to the abstract.
- Important note on `extract_answers()`: it returns a JSON-decoded dict keyed by the original question strings. That is deliberate. The caller asks several article-reading questions in one LLM call and needs a stable answer map instead of brittle positional parsing.

### 4A. Trace One Real Article-Lookup Flow

If you want to understand one function end to end, do not bounce randomly between files. Follow this exact chain.

#### Professor workflow: find the target article

Start here:

1. [project/sim_tool/run_directive.py](project/sim_tool/run_directive.py)
   Read `main()`.
   Question it answers: how does a JSON file become a directive run?

2. [project/sim_tool/research_directive.py](project/sim_tool/research_directive.py)
   Read `DirectiveRunner.__init__()` and `run_all_phases()`.
   Question it answers: how does the legacy entrypoint hand control to the shared workflow runner?

3. [project/research_platform/workflow.py](project/research_platform/workflow.py)
   Read `ProfessorWorkflowRunner.run_all_phases()`.
   Then read `_phase_find_article()`.
   Question it answers: how does the orchestrator prepare the literature task?

4. Still in [project/research_platform/workflow.py](project/research_platform/workflow.py)
   Read `_task()`.
   Question it answers: what metadata and instructions are actually handed to the assistant?

5. [project/research_platform/assistants/literature.py](project/research_platform/assistants/literature.py)
   Read `LocalLiteratureAssistant.prepare_article_lookup_spec()`.
   Then read `_normalize_lookup_spec()`.
   Question it answers: how does the literature assistant use the LLM to prepare the live structured article-search request, and when does it ask for clarification instead of searching immediately?

6. Still in [project/research_platform/assistants/literature.py](project/research_platform/assistants/literature.py)
   Read `LocalLiteratureAssistant.find_primary_article()`.
   Question it answers: how is the orchestrator’s task converted into a concrete article lookup, and when does the run stop because all providers are in cooldown?

7. [project/research_platform/article_lookup.py](project/research_platform/article_lookup.py)
   Read `build_article_lookup_query()`.
   Question it answers: what deterministic fallback does the assistant use if the LLM lookup-spec step fails or returns junk?

8. Back in [project/research_platform/workflow.py](project/research_platform/workflow.py)
   Re-read `_phase_find_article()`, especially the `needs_clarification` branch and the article-confirmation checkpoint.
   Question it answers: when does the PI get to clarify before lookup and when do they confirm after a candidate article is found?

9. [project/literature_review/article_parser.py](project/literature_review/article_parser.py)
   Read `ArticleParser.find_article()`.
   Question it answers: how is the prepared lookup query executed across backends and how are provider cooldown diagnostics recorded?

10. [project/literature_review/search_backends/arxiv_backend.py](project/literature_review/search_backends/arxiv_backend.py)
   Read `ArXivBackend.search()`, then `_build_query()`, `_build_or_query()`, and `_mark_rate_limited()`.
   Question it answers: what exact ArXiv queries get emitted and how does the backend back off after HTTP `429`?

11. [project/literature_review/search_backends/semantic_scholar_backend.py](project/literature_review/search_backends/semantic_scholar_backend.py)
   Read `SemanticScholarBackend.search()` and `_mark_rate_limited()`.
   Question it answers: how does the fallback provider get paced and cooled down in the shared unauthenticated pool?

12. [project/literature_review/scope_validator.py](project/literature_review/scope_validator.py)
   Read `raw_to_paper()`.
   Question it answers: how is the raw search result normalized into a typed literature paper?

Short version of the call chain:

`run_directive.main()`
→ `DirectiveRunner.run_all_phases()`
→ `ProfessorWorkflowRunner.run_all_phases()`
→ `ProfessorWorkflowRunner._phase_find_article()`
→ `ProfessorWorkflowRunner._task()`
→ `LocalLiteratureAssistant.prepare_article_lookup_spec()`
→ optional `ProfessorWorkflowRunner._checkpoint()` clarification pause
→ `build_article_lookup_query()` fallback only on failure
→ `LocalLiteratureAssistant.find_primary_article()`
→ `ArticleParser.find_article()`
→ `ArXivBackend.search()` + `SemanticScholarBackend.search()`
→ `raw_to_paper()`

Important steering hook:

- If `prepare_article_lookup_spec()` says the request is underspecified, the workflow opens a clarification checkpoint before article lookup starts.
- `ProfessorWorkflowRunner._checkpoint()` opens a `SteeringCheckpoint` before the workflow proceeds.
- If the PI says the wrong article was found, the runner records that note and reruns `_phase_find_article()` with a corrected query.

#### Professor workflow: turn the article into a reproduction brief

Once the article exists, follow this second chain:

1. In [project/research_platform/workflow.py](project/research_platform/workflow.py)
   Read `_phase_parse_article()`.

2. In [project/research_platform/assistants/literature.py](project/research_platform/assistants/literature.py)
   Read `LocalLiteratureAssistant.build_article_brief()`.

3. In [project/literature_review/article_parser.py](project/literature_review/article_parser.py)
   Read `fetch_full_text()` and `extract_answers()`.

4. Back in [project/research_platform/assistants/literature.py](project/research_platform/assistants/literature.py)
   Read `_normalize_answer()`, `_coerce_parameters()`, and `_coerce_figures()`.

5. In [project/research_platform/registry.py](project/research_platform/registry.py)
   Read `save_text()` and `save_json()`.

Short version of the call chain:

`ProfessorWorkflowRunner._phase_parse_article()`
→ `LocalLiteratureAssistant.build_article_brief()`
→ `ArticleParser.fetch_full_text()`
→ `ArticleParser.extract_answers()`
→ `_normalize_answer()` / `_coerce_parameters()` / `_coerce_figures()`
→ `ArtifactRegistry.save_text()` / `save_json()`

[project/literature_review/reviewer.py](project/literature_review/reviewer.py)

- Read first: `LiteratureReviewer.run()`.
- Then read: `_synthesize()` and `_emit()`.
- Question it answers: how does the system search multiple backends, filter papers, and synthesize accepted results?
- Expected takeaway: this is the main engine for related-literature review.

[project/literature_review/scope_validator.py](project/literature_review/scope_validator.py)

- Read first: `check_paper_scope()`.
- Then read: `_heuristic_scope_check()` and `raw_to_paper()`.
- Question it answers: how are accepted and rejected papers decided, and how are raw backend papers normalized?
- Expected takeaway: many literature regressions are actually scope-validator regressions.

[project/literature_review/auditor.py](project/literature_review/auditor.py)

- Read first: `LiteratureAuditor.audit()`.
- Then skim the check helpers `_check_year_min()` through `_check_search_log()`.
- Question it answers: what deterministic structural guarantees are enforced after a literature review completes?
- Expected takeaway: this is defense-in-depth, not the same job as semantic scope review.

[project/literature_review/director_bridge.py](project/literature_review/director_bridge.py)

- Read first: `LiteratureReviewBridge.run_sync()`.
- Then read: `_make_stream_callback()` and `build_bridge_from_env()`.
- Question it answers: how is the async literature engine adapted to synchronous callers and streamed UI updates?

[project/literature_review/search_backends/arxiv_backend.py](project/literature_review/search_backends/arxiv_backend.py)

- Read first: `ArXivBackend.search()`.
- Then skim: `_build_query()`, `_build_or_query()`, `_parse_atom()`.
- Question it answers: how do prepared keyword sets become ArXiv queries?
- Why it matters: if article lookup looks obviously wrong, inspect the query shaping here together with `research_platform.article_lookup`.

[project/literature_review/search_backends/semantic_scholar_backend.py](project/literature_review/search_backends/semantic_scholar_backend.py)

- Skim for rate limiting, API-key handling, and result normalization.
- Question it answers: what changes when a paid or rate-limited backend is enabled?

[project/literature_review/search_backends/perplexity_backend.py](project/literature_review/search_backends/perplexity_backend.py)

- Skim for query handling and response normalization.
- Question it answers: how does the optional online-LLM search path plug into the same review flow?

### 5. Read The Writer Subsystem

The writer is easier to understand after the registry and artifact contracts make sense.

[project/writer/contract.py](project/writer/contract.py)

- Read first: `GroundingClaim`, `SectionSpec`, `SectionDraft`, `TOCPlan`, `WriterArtifact`, `WritingTask`, `ReviewCycle`, `StreamChunk`.
- Skim: `Promise`, `FigureSpec`, `ReviewerDemand`, `RevisionResponse`.
- Question it answers: what structures are required for grounded multi-phase drafting?

[project/writer/tools.py](project/writer/tools.py)

- Read first: `WriterToolbox`, `AgentQueryTool`, `TaskRequestTool`.
- Skim: `ArtifactInfo`, `AgentQueryResult`, `TaskRequestResult`.
- Question it answers: what is the writer allowed to ask the rest of the system for?
- Important note: `make_mock_toolbox()` is useful for tests, but should not define live runtime behavior.

[project/writer/validator.py](project/writer/validator.py)

- Read first: `GroundingValidator.validate_section()` and `validate_paper()`.
- Then skim the check families `_t1_*`, `_t2_*`, `_t3_*`, `_t4_*`.
- Question it answers: what does the writer have to prove about its claims before a draft is accepted?
- Expected takeaway: this file encodes the grounding bar for the writer.

[project/writer/academic_writer.py](project/writer/academic_writer.py)

- Read first: `AcademicWriter.plan()`, `draft()`, `respond_to_review()`.
- Then read: `materialize_section()` and `materialize_paper()`.
- Skim next: `_parse_toc()`, `_build_section_draft()`, `_register_section_promises()`, `_topo_sort()`.
- Question it answers: how does the writer move from plan to section drafts to a final assembled paper?
- Expected takeaway: this file is large, so start with the public async methods and only then descend into the helpers.

[project/writer/promise_registry.py](project/writer/promise_registry.py)

- Read for: how forward references are stored and later resolved.
- Question it answers: how does the writer keep “we will show later” statements coherent across section-by-section drafting?

[project/research_platform/writer_runtime.py](project/research_platform/writer_runtime.py)

- Re-read this file now if the writer still feels abstract.
- Question it answers: how do artifact-registry entries become the writer’s concrete toolbox and store?

### 6. Read The Review And Mailbox Support Layers

These are adjacent to the main professor workflow, not inside it.

[project/reviewer/contract.py](project/reviewer/contract.py)

- Read first: `MathCheckRequest`, `MathCheckResult`, `ReviewFindings`, `AnnotatedArticle`, `ReviewDecision`, `ReviewTask`, `ReviewChunk`.
- Question it answers: what shapes does the manuscript-review pipeline operate on?

[project/reviewer/journal_reviewer.py](project/reviewer/journal_reviewer.py)

- Read first: `JournalReviewer.review()`.
- Then read in order: `_phase_parse()`, `_phase_analyse()`, `_phase_verify()`, `_phase_decide()`.
- Question it answers: how does the standalone manuscript-review pipeline work from text to decision?
- Expected takeaway: this file is intentionally separate from the professor workflow and does not consume upstream artifacts.

[project/reviewer/math_tools.py](project/reviewer/math_tools.py)

- Read first: `MathAgentRegistry.register()` and `dispatch()`.
- Skim: `CLAIM_KIND_TO_AGENT`.
- Question it answers: how are mathematical claims routed to specialized verifiers?

[project/comms/pi_email.py](project/comms/pi_email.py)

- Read first: `PIMailbox.append()`, `_save_email()`, `all_emails()`.
- Then skim: `_coerce_message()` and `_coerce_attachment()`.
- Question it answers: how do mailbox messages become persisted PI-facing records on disk?

### 7. Read The Frontend Last

Do not start with the UI. It is much easier once the backend contracts are already familiar.

[project/frontend/src/api.js](project/frontend/src/api.js)

- Read first: `healthCheck()`, `streamDirective()`, `literatureRun()`, `simStart()`, `writerStart()`.
- Then skim the helpers: `post()`, `get()`, `readNDJSON()`.
- Question it answers: what exact HTTP contract does the frontend think the backend provides?

[project/frontend/src/ResearchHub.jsx](project/frontend/src/ResearchHub.jsx)

- Read first: `ResearchHub()`, then `handleArtifact()`, `handleProfessorNameChange()`, `handleFocusChange()`.
- Question it answers: how does the top-level UI stitch the panels together and maintain shared desk state?

[project/frontend/src/DirectivesPanel.jsx](project/frontend/src/DirectivesPanel.jsx)

- Read first: `DirectivesPanel()`, then `handleSend()`.
- Skim: `EmailCard()` and `handleTogglePhase()`.
- Question it answers: how does the professor workflow launch from the UI, and how are mailbox updates rendered?

[project/frontend/src/LitReviewPanel.jsx](project/frontend/src/LitReviewPanel.jsx)

- Read first: `LitReviewPanel()`, then `handleRun()` and `handleExport()`.
- Skim: `PaperCard()`.
- Question it answers: how does a direct literature-review run map onto the backend contract?

[project/frontend/src/SimPanel.jsx](project/frontend/src/SimPanel.jsx)

- Read first: `SimPanel()`, then `handleStart()`, `processUpdate()`, and `doAction()`.
- Skim: `QAForm()` and `TextGate()`.
- Question it answers: how does the simulation state machine get projected into a UI with gates and approvals?

[project/frontend/src/WriterPanel.jsx](project/frontend/src/WriterPanel.jsx)

- Read first: `WriterPanel()`, then `handleStart()`, `handleStop()`, `handleExport()`.
- Question it answers: how do literature and simulation artifacts become a writer job from the UI?

[project/frontend/src/professorWorkflowView.js](project/frontend/src/professorWorkflowView.js)

- Read first: `attachmentHref()`, `buildWriterArtifactCatalog()`, `buildWriterArtifactContext()`.
- Question it answers: where is the cross-panel data normalization logic kept so it does not spread through React components?

[project/frontend/src/researchDeskState.js](project/frontend/src/researchDeskState.js)

- Read first: `buildDeskHeading()`, `buildFocusLabel()`, `loadDeskSettings()`, `saveDeskSettings()`.
- Question it answers: how is the desk-level UI state kept generic and user-driven rather than hard-coded to an example case?

[project/frontend/src/style.css](project/frontend/src/style.css)

- Skim for tokens, variables, and broad layout decisions.
- Question it answers: where are the shared visual primitives defined?

### 8. Read The Tests To Confirm Your Mental Model

Now that the code paths make sense, read the tests that lock behavior in place.

[project/tests/test_professor_workflow.py](project/tests/test_professor_workflow.py)

- Read first.
- Focus on: how fake assistants are injected, what mailbox subjects are asserted, and what the article-lookup regression covers.
- Question it answers: what behavior is considered stable for the shared workflow?

[project/tests/test_writer_api.py](project/tests/test_writer_api.py)

- Read next.
- Focus on: the real writer-service path over a concrete artifact registry.
- Question it answers: what integration guarantees does the writer API make?

[project/tests/test_api_server.py](project/tests/test_api_server.py)

- Read when API/UI behavior looks suspicious.
- Focus on: endpoint inputs, outputs, and streamed responses.
- Question it answers: what contract does the HTTP layer guarantee?

[project/tests/test_literature_review.py](project/tests/test_literature_review.py)

- Read when literature behavior changes.
- Focus on: scope rejections, accepted counts, and event emission.
- Question it answers: which literature behaviors are regression-sensitive?

[project/tests/test_orchestrator.py](project/tests/test_orchestrator.py)

- Read when simulation gate logic changes.
- Focus on: allowed state transitions and wrong-state errors.
- Question it answers: what is the legal lifecycle of a simulation session?

[project/tests/test_writer.py](project/tests/test_writer.py)

- Read when grounding or promise behavior changes.
- Question it answers: what are the expectations inside the writer engine itself?

[project/tests/ProfessorWorkflowView.test.js](project/tests/ProfessorWorkflowView.test.js)

- Read when artifact or attachment rendering changes.
- Focus on: `attachmentHref()`, `buildWriterArtifactCatalog()`, and `buildWriterArtifactContext()`.
- Question it answers: what cross-panel frontend data shapes are assumed to be stable?

[project/tests/ResearchDeskState.test.js](project/tests/ResearchDeskState.test.js)

- Read when desk heading, focus banner, or settings persistence changes.
- Question it answers: what generic desk-state behavior is expected from the new UI shell?

[project/tests/PIControl.test.js](project/tests/PIControl.test.js)

- Treat this as legacy or adjacent coverage.
- Read only if you are touching the older PI-control concepts that still influence tests or design vocabulary.

## Change-Oriented Reading Paths

### If you are changing the professor workflow

Read in this order:

1. [project/research_platform/workflow.py](project/research_platform/workflow.py)
2. [project/research_platform/article_lookup.py](project/research_platform/article_lookup.py)
3. [project/research_platform/assistants/literature.py](project/research_platform/assistants/literature.py)
4. [project/research_platform/assistants/coding.py](project/research_platform/assistants/coding.py)
5. [project/research_platform/assistants/writer.py](project/research_platform/assistants/writer.py)
6. [project/research_platform/wiring.py](project/research_platform/wiring.py)
7. [project/sim_tool/api_server.py](project/sim_tool/api_server.py)
8. [project/frontend/src/DirectivesPanel.jsx](project/frontend/src/DirectivesPanel.jsx)
9. [project/tests/test_professor_workflow.py](project/tests/test_professor_workflow.py)

### If you are changing simulation behavior

Read in this order:

1. [project/sim_tool/contract.py](project/sim_tool/contract.py)
2. [project/sim_tool/tool.py](project/sim_tool/tool.py)
3. [project/sim_tool/orchestrator.py](project/sim_tool/orchestrator.py)
4. [project/tests/test_orchestrator.py](project/tests/test_orchestrator.py)
5. [project/tests/test_api_server.py](project/tests/test_api_server.py)

### If you are changing literature behavior

Read in this order:

1. [project/literature_review/article_parser.py](project/literature_review/article_parser.py)
2. [project/literature_review/reviewer.py](project/literature_review/reviewer.py)
3. [project/literature_review/scope_validator.py](project/literature_review/scope_validator.py)
4. [project/research_platform/article_lookup.py](project/research_platform/article_lookup.py)
5. [project/tests/test_literature_review.py](project/tests/test_literature_review.py)

### If you are changing writer behavior

Read in this order:

1. [project/writer/academic_writer.py](project/writer/academic_writer.py)
2. [project/writer/validator.py](project/writer/validator.py)
3. [project/research_platform/writer_runtime.py](project/research_platform/writer_runtime.py)
4. [project/tests/test_writer.py](project/tests/test_writer.py)
5. [project/tests/test_writer_api.py](project/tests/test_writer_api.py)

### If you are changing the frontend

Read in this order:

1. [project/frontend/src/api.js](project/frontend/src/api.js)
2. [project/frontend/src/ResearchHub.jsx](project/frontend/src/ResearchHub.jsx)
3. [project/frontend/src/professorWorkflowView.js](project/frontend/src/professorWorkflowView.js)
4. [project/frontend/src/DirectivesPanel.jsx](project/frontend/src/DirectivesPanel.jsx)
5. [project/tests/ProfessorWorkflowView.test.js](project/tests/ProfessorWorkflowView.test.js)
6. [project/tests/ResearchDeskState.test.js](project/tests/ResearchDeskState.test.js)

## Commands You Will Actually Use

From the repo root:

```bash
uv sync --project project --extra full
uv sync --project project --extra api --extra full
npm --prefix project/frontend install
uv run --project project python -m pytest -q project/tests
node project/tests/ProfessorWorkflowView.test.js
node project/tests/ResearchDeskState.test.js
npm --prefix project/frontend run build
uv run --project project python -m sim_tool.api_server --host 127.0.0.1 --port 8001
uv run --project project python -m sim_tool.run_directive project/examples/danino_repro.json
```

## Troubleshooting

### Refresh the API server after code changes

The default command:

```bash
uv run --project project python -m sim_tool.api_server --host 127.0.0.1 --port 8001
```

does not enable auto-reload. If you changed backend code, "refreshing" the server means restarting the process.

If the server is running in your current terminal, stop it with `Ctrl-C` and start it again.

If it is running elsewhere, stop it first:

```bash
pkill -f "python -m sim_tool.api_server"
```

and then start it again:

```bash
uv run --project project python -m sim_tool.api_server --host 127.0.0.1 --port 8001
```

If you want automatic reload during backend development, run Uvicorn directly:

```bash
uv run --project project python -m uvicorn sim_tool.api_server:create_app --factory --host 127.0.0.1 --port 8001 --reload
```

That keeps the frontend proxy configuration unchanged because it still serves on port `8001`.

### API server fails with `address already in use`

If this command fails:

```bash
uv run --project project python -m sim_tool.api_server --host 127.0.0.1 --port 8001
```

and you see:

```text
[Errno 48] address already in use
```

then something is already listening on `127.0.0.1:8001`.

Check what owns the port:

```bash
lsof -nP -iTCP:8001 -sTCP:LISTEN
```

If it is an older API server you no longer need, stop it cleanly:

```bash
kill <PID>
```

If you want to be more aggressive and you know it is this repo's server:

```bash
pkill -f "python -m sim_tool.api_server"
```

If you are not sure whether the server is already healthy, probe it before killing anything:

```bash
curl http://127.0.0.1:8001/health
```

If that returns a healthy response, you can usually reuse the running server instead of starting a second one.

If you really need another copy, run it on a different port:

```bash
uv run --project project python -m sim_tool.api_server --host 127.0.0.1 --port 8002
```

Important:

- CLI calls can use any port you choose.
- The frontend Vite proxy is currently pinned to `127.0.0.1:8001` in `project/frontend/vite.config.js`.
- If you move the API server to `8002`, update the `/api` and `/health` proxy targets in `project/frontend/vite.config.js` before using the UI.

### Root URL on `8001` shows `{"detail":"Not Found"}`

The backend on port `8001` is an API server, not the frontend app.

- `http://127.0.0.1:8001/` is not expected to render the UI.
- `http://127.0.0.1:8001/health` is the basic backend health check.
- `http://127.0.0.1:8001/docs` is the FastAPI docs UI.

If you want the actual frontend, run:

```bash
npm --prefix project/frontend run dev
```

and open:

```text
http://127.0.0.1:3001
```

## Common Traps

- The active package is in `project/`, not at the repo root.
- `research_platform` is the contract boundary; avoid reintroducing direct peer-to-peer imports across subsystems.
- The writer depends on real artifact data. Mock-style stores are fine in tests, not in live service paths.
- Generated files under `project/programs/` are outputs, not source code.
- Example fixtures under `project/examples/` should guide tests and manual runs, but should not leak into live UI defaults or hard-coded workflow prompts.

## Good First Tasks

1. Run the deterministic tests once.
2. Read [project/tests/test_professor_workflow.py](project/tests/test_professor_workflow.py) and [project/research_platform/workflow.py](project/research_platform/workflow.py) together.
3. Start the API server locally.
4. Run one example directive from `project/examples/`.
5. Make one small docs-only or test-only change before attempting a cross-unit behavior change.
