import { useState, useRef, useCallback, useEffect } from "react";
import { literatureStream } from "./api.js";
import LLMConfigBlock from "./LLMConfigBlock.jsx";
import { buildFocusLabel } from "./researchDeskState.js";

const DEFAULT_QUERY = "";
const SAVED_LATER_KEY = "literatureSavedForLater";
const CONTEXT_KEY = "literatureContextPapers";

function paperKey(paper) {
  return paper?.url || paper?.arxiv_id || `${paper?.title || ""}-${paper?.year || ""}`;
}

function dedupePapers(papers) {
  const seen = new Set();
  const out = [];
  for (const paper of papers || []) {
    const key = paperKey(paper);
    if (!key || seen.has(key)) continue;
    seen.add(key);
    out.push(paper);
  }
  return out;
}

function loadPaperList(key) {
  try {
    const parsed = JSON.parse(localStorage.getItem(key) || "[]");
    return Array.isArray(parsed) ? parsed : [];
  } catch {
    return [];
  }
}

function downloadPaper(paper) {
  const blob = new Blob([JSON.stringify(paper, null, 2)], { type: "application/json" });
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = `${(paper.title || "paper").toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/^-|-$/g, "").slice(0, 80) || "paper"}.json`;
  a.click();
  URL.revokeObjectURL(a.href);
}


function Toggle({ on, onChange, label }) {
  return (
    <label className="toggle-wrap" style={{ userSelect: "none" }}>
      <div className={`toggle ${on ? "on" : ""}`} onClick={() => onChange(!on)} />
      <span style={{ fontSize: 10, color: on ? "var(--text-secondary)" : "var(--text-faint)", fontFamily: "var(--font-mono)" }}>
        {label}
      </span>
    </label>
  );
}

function AgentLabel({ color, text }) {
  return (
    <span className="badge" style={{ color, background: `${color}18` }}>{text}</span>
  );
}

function StreamLine({ text, type = "normal" }) {
  const isGate = type === "gate";
  const isErr  = type === "error";
  const isPhase = type === "phase";
  return (
    <div className={`stream-line ${isGate ? "gate" : ""} ${isErr ? "error" : ""}`}
      style={{ paddingTop: isPhase ? 8 : undefined, paddingBottom: isPhase ? 8 : undefined }}>
      <span style={{
        fontSize: 9, fontWeight: 700, letterSpacing: 1,
        color: isErr ? "var(--accent-red)" : isPhase ? "var(--accent-violet)" : "var(--text-faint)",
        fontFamily: "var(--font-mono)", minWidth: 60, flexShrink: 0,
      }}>
        {isErr ? "ERROR" : isPhase ? "PHASE" : "LIT"}
      </span>
      <span style={{ color: isErr ? "var(--accent-red)" : isPhase ? "var(--text-primary)" : "var(--text-secondary)", flex: 1 }}>
        {text}
      </span>
    </div>
  );
}

function PaperCard({
  paper,
  idx,
  variant = "accepted",
  onInclude,
  onSaveDesk,
  onAddContext,
  onSaveLater,
  onDownload,
}) {
  const [expanded, setExpanded] = useState(false);
  const rejected = variant === "rejected";
  const controlStyle = {
    fontSize: 8,
    padding: "4px 6px",
    letterSpacing: 0.7,
    whiteSpace: "nowrap",
  };
  return (
    <div className="paper-card slide-up" style={{
      animationDelay: `${idx * 0.05}s`,
      borderColor: rejected ? "rgba(244,63,94,0.22)" : undefined,
    }}>
      <div className="paper-title">{paper.title || "(no title)"}</div>
      <div className="paper-meta">
        {paper.year && <span className="badge" style={{ color: "var(--accent-blue)", background: "rgba(96,165,250,0.1)" }}>{paper.year}</span>}
        {paper.source && <span className="badge" style={{ color: "var(--text-faint)", background: "var(--bg-base)" }}>{paper.source}</span>}
        {rejected && <span className="badge" style={{ color: "var(--accent-red)", background: "rgba(244,63,94,0.08)" }}>REJECTED</span>}
        {paper.url && (
          <a href={paper.url} target="_blank" rel="noreferrer"
            style={{ fontSize: 9, color: "var(--accent-gold)", fontFamily: "var(--font-mono)", textDecoration: "none" }}>
            ↗ OPEN
          </a>
        )}
      </div>
      {paper.authors?.length > 0 && (
        <div style={{ fontSize: 9, color: "var(--text-faint)", fontFamily: "var(--font-mono)", lineHeight: 1.5, marginTop: 4 }}>
          {paper.authors.slice(0, 5).join(", ")}
        </div>
      )}
      {rejected && paper.scope_violation_reason && (
        <div style={{ fontSize: 9, color: "var(--accent-red)", fontFamily: "var(--font-mono)", lineHeight: 1.5, marginTop: 4 }}>
          {paper.scope_violation_reason}
        </div>
      )}
      {paper.abstract && (
        <>
          <div className="paper-abstract" style={{ WebkitLineClamp: expanded ? undefined : 3, overflow: expanded ? "visible" : "hidden", display: expanded ? "block" : "-webkit-box" }}>
            {paper.abstract}
          </div>
          <button onClick={() => setExpanded(!expanded)}
            style={{ marginTop: 4, background: "none", border: "none", color: "var(--text-faint)", fontSize: 9, fontFamily: "var(--font-mono)", cursor: "pointer", letterSpacing: 1 }}>
            {expanded ? "▲ LESS" : "▼ MORE"}
          </button>
        </>
      )}
      <div style={{ display: "flex", flexWrap: "wrap", gap: 6, marginTop: 8 }}>
        {rejected && <button className="btn" style={controlStyle} onClick={() => onInclude?.(paper)}>OVERRULE AUDIT</button>}
        <button className="btn" style={controlStyle} onClick={() => onSaveDesk?.(paper)}>SAVE TO DESK</button>
        <button className="btn" style={controlStyle} onClick={() => onAddContext?.(paper)}>ADD TO CONTEXT</button>
        <button className="btn" style={controlStyle} onClick={() => onSaveLater?.(paper)}>SAVE TO LATER</button>
        <button className="btn" style={controlStyle} onClick={() => onDownload?.(paper)}>DOWNLOAD</button>
      </div>
    </div>
  );
}

export default function LitReviewPanel({ globalLLM, onArtifact, onFocusChange, onAttachToDesk, isActive = true }) {
  const [query, setQuery] = useState(DEFAULT_QUERY);
  const [includeTopics, setIncludeTopics] = useState("");
  const [excludeTopics, setExcludeTopics] = useState("");
  const [yearMin, setYearMin]   = useState("2000");
  const [yearMax, setYearMax]   = useState("");
  const [maxPapers, setMaxPapers] = useState("12");
  const [maxRounds, setMaxRounds] = useState("3");

  // Load from localStorage if available
  const [useArxiv, setUseArxiv]   = useState(true);
  const [useSSch, setUseSSch]     = useState(false);
  const [usePerplexity, setUsePerplexity] = useState(false);
  const [ssApiKey, setSsApiKey]   = useState("");
  const [pplxKey, setPplxKey]     = useState("");

  useEffect(() => {
    try {
      const saved = localStorage.getItem("literatureAgentConfig");
      if (saved) {
        const parsed = JSON.parse(saved);
        if (parsed.arxiv !== undefined) setUseArxiv(parsed.arxiv.enabled);
        if (parsed.semantic_scholar !== undefined) {
          setUseSSch(parsed.semantic_scholar.enabled);
          if (parsed.semantic_scholar.api_key) setSsApiKey(parsed.semantic_scholar.api_key);
        }
        if (parsed.perplexity !== undefined) {
          setUsePerplexity(parsed.perplexity.enabled);
          if (parsed.perplexity.api_key) setPplxKey(parsed.perplexity.api_key);
        }
      }
    } catch (e) {
      console.error("Failed to load literatureAgentConfig from localStorage", e);
    }
  }, []);

  useEffect(() => {
    const config = {
      arxiv: { enabled: useArxiv },
      semantic_scholar: { enabled: useSSch, api_key: ssApiKey || null },
      perplexity: { enabled: usePerplexity, api_key: pplxKey || null },
    };
    localStorage.setItem("literatureAgentConfig", JSON.stringify(config));
  }, [useArxiv, useSSch, usePerplexity, ssApiKey, pplxKey]);

  const [llm, setLlm] = useState({
    provider: globalLLM?.provider || "lmstudio",
    url:      globalLLM?.url     || "http://localhost:1234/v1",
    model:    globalLLM?.model   || "auto",
    apiKey:   globalLLM?.apiKey  || "",
  });

  const setLlmField = (field, value) => setLlm(p => ({ ...p, [field]: value }));

  const [running, setRunning]   = useState(false);
  const [logs, setLogs]         = useState([]);
  const [result, setResult]     = useState(null); // { artifact, audit }
  const [error, setError]       = useState(null);
  const [showConfig, setShowConfig] = useState(true);
  const [manualIncluded, setManualIncluded] = useState([]);
  const [contextPapers, setContextPapers] = useState(() => loadPaperList(CONTEXT_KEY));
  const [savedLater, setSavedLater] = useState(() => loadPaperList(SAVED_LATER_KEY));

  const logRef = useRef(null);
  const addLog = useCallback((text, type = "normal") => {
    setLogs(p => [...p, { id: Date.now() + Math.random(), text, type }]);
  }, []);

  useEffect(() => {
    if (logRef.current) logRef.current.scrollTop = logRef.current.scrollHeight;
  }, [logs]);

  useEffect(() => {
    localStorage.setItem(CONTEXT_KEY, JSON.stringify(contextPapers));
  }, [contextPapers]);

  useEffect(() => {
    localStorage.setItem(SAVED_LATER_KEY, JSON.stringify(savedLater));
  }, [savedLater]);

  // Sync with global LLM changes
  useEffect(() => {
    if (globalLLM) setLlm(p => ({ ...p, ...globalLLM }));
  }, [globalLLM]);

  useEffect(() => {
    if (!isActive) return;
    onFocusChange?.(buildFocusLabel(query, includeTopics));
  }, [query, includeTopics, onFocusChange, isActive]);

  const handleRun = async () => {
    if (running) return;
    const normalizedQuery = query.trim();
    const topics = includeTopics.split(",").map(s => s.trim()).filter(Boolean);
    if (!normalizedQuery && topics.length === 0) {
      setError("Provide a research query or at least one include topic.");
      return;
    }

    setRunning(true);
    setError(null);
    setResult(null);
    setManualIncluded([]);
    setLogs([]);
    setShowConfig(false);

    const excl   = excludeTopics.split(",").map(s => s.trim()).filter(Boolean);

    const req = {
      query: normalizedQuery,
      includeTopics: topics.length ? topics : [normalizedQuery],
      excludeTopics: excl,
      yearMin: yearMin ? parseInt(yearMin) : null,
      yearMax: yearMax ? parseInt(yearMax) : null,
      maxPapers: parseInt(maxPapers) || 12,
      maxRounds: parseInt(maxRounds) || 3,
      minPapersThreshold: 4,
      papersPerQuery: 8,
      llm: { provider: llm.provider, url: llm.url, model: llm.model, apiKey: llm.apiKey || null, enabled: true },
      arxiv: { enabled: useArxiv, sortBy: "relevance" },
      semanticScholar: { enabled: useSSch, apiKey: ssApiKey || null },
      perplexity: { enabled: usePerplexity, apiKey: pplxKey || null },
    };

    addLog("Starting literature review…", "phase");
    addLog(`Query: "${normalizedQuery}"`, "normal");
    addLog(`Backends: ${[useArxiv && "ArXiv", useSSch && "Semantic Scholar", usePerplexity && "Perplexity"].filter(Boolean).join(", ")}`, "normal");

    try {
      addLog("Searching and synthesising with live results…", "normal");
      setResult({ artifact: { papers: [], removed_papers: [], synthesis: "", status: "running" }, audit: null });
      let streamError = null;
      await literatureStream(req, item => {
        if (item.kind === "error") {
          streamError = new Error(item.text || "Literature stream failed");
          return;
        }
        if (item.kind === "complete") {
          const data = item.data;
          const art = data?.artifact || {};
          const accepted = (art.papers || []).filter(p => p.in_scope !== false);
          const rejected = (art.removed_papers || []).length + (art.papers || []).filter(p => p.in_scope === false).length;
          setResult(data);
          addLog("Literature review complete.", "phase");
          addLog(`Accepted ${accepted.length} paper(s), rejected ${rejected}.`, "normal");
          if (art.synthesis) addLog("Synthesis generated.", "normal");
          const audit = data?.audit || {};
          if (audit.passed === false) {
            addLog(`Audit: FAILED — ${(audit.issues || audit.scope_violations_found || []).length} issue(s).`, "error");
          } else {
            addLog("Audit: PASSED.", "normal");
          }
          onArtifact?.("literature", data);
          return;
        }
        if (item.kind !== "event") return;
        const event = item.event || {};
        const payload = event.payload || {};
        if (event.event_type === "lookup_plan") {
          const authors = payload.required_authors?.length ? ` | authors: ${payload.required_authors.join(", ")}` : "";
          const years = payload.year_constraints?.preferred_year ? ` | year: ${payload.year_constraints.preferred_year}` : "";
          addLog(`Lookup target: ${payload.article_target || "(not specified)"}${authors}${years}`, "normal");
          if (payload.task_intent) addLog(`Parse intention: ${payload.task_intent}`, "normal");
          if (payload.excluded_search_terms?.length) addLog(`Excluded from search: ${payload.excluded_search_terms.join(", ")}`, "normal");
        } else if (event.event_type === "query_generated") {
          addLog(`Query ${payload.index}: ${payload.query || "(empty)"}${payload.rationale ? ` — ${payload.rationale}` : ""}`, "normal");
        } else if (event.event_type === "search_round_start") {
          addLog(`Round ${payload.round}: ${payload.keywords?.map(s => s.join(" + ")).join(" | ") || "search"}${payload.authors?.length ? ` | authors: ${payload.authors.join(", ")}` : ""}`, "normal");
        } else if (event.event_type === "keyword_refined") {
          addLog(`Refined keywords: ${payload.keywords?.map(s => s.join(" + ")).join(" | ") || "none"}`, "normal");
        } else if (event.event_type === "title_found") {
          const paper = { ...payload, in_scope: true };
          addLog(`Accepted: ${paper.title || "(untitled)"}${paper.year ? ` (${paper.year})` : ""}${paper.authors?.length ? ` — ${paper.authors.join(", ")}` : ""}`, "normal");
          setResult(prev => {
            const base = prev || { artifact: { papers: [], removed_papers: [], synthesis: "" }, audit: null };
            const art = base.artifact || {};
            return {
              ...base,
              artifact: {
                ...art,
                papers: dedupePapers([...(art.papers || []), paper]),
              },
            };
          });
        } else if (event.event_type === "scope_removed") {
          const paper = {
            ...payload,
            in_scope: false,
            scope_violation_reason: payload.reason || payload.scope_violation_reason,
          };
          setResult(prev => {
            const base = prev || { artifact: { papers: [], removed_papers: [], synthesis: "" }, audit: null };
            const art = base.artifact || {};
            return {
              ...base,
              artifact: {
                ...art,
                removed_papers: dedupePapers([...(art.removed_papers || []), paper]),
              },
            };
          });
        } else if (event.event_type === "round_summary") {
          addLog(`Round ${payload.round}: raw=${payload.raw}, accepted=${payload.in_scope}, validated=${payload.validated}, skipped=${payload.skipped_prefilter}.`, "normal");
        }
      });
      if (streamError) throw streamError;
    } catch (err) {
      setError(err.message);
      addLog(err.message, "error");
    } finally {
      setRunning(false);
    }
  };

  const handleIncludeRejected = paper => {
    const included = {
      ...paper,
      in_scope: true,
      audit_overridden: true,
      scope_violation_reason: null,
    };
    setManualIncluded(prev => dedupePapers([...prev, included]));
    addLog(`Audit overruled: ${paper.title || "(untitled)"}`, "normal");
  };

  const handleSaveDesk = paper => {
    onAttachToDesk?.("", [paper]);
    addLog(`Saved to desk: ${paper.title || "(untitled)"}`, "normal");
  };

  const handleAddContext = paper => {
    const next = dedupePapers([...contextPapers, paper]);
    setContextPapers(next);
    onArtifact?.("literature_context", { papers: next });
    addLog(`Added to context: ${paper.title || "(untitled)"}`, "normal");
  };

  const handleSaveLater = paper => {
    const next = dedupePapers([...savedLater, paper]);
    setSavedLater(next);
    addLog(`Saved for later: ${paper.title || "(untitled)"}`, "normal");
  };

  const handleExport = () => {
    if (!result?.artifact) return;
    const art = result.artifact;
    const papers = dedupePapers([...(art.papers || []).filter(p => p.in_scope !== false), ...manualIncluded]);
    const lines = [
      `# Literature Review: ${query}`,
      `**Date:** ${new Date().toISOString().slice(0, 10)}`,
      `**Papers:** ${papers.length}`,
      "",
      "## Synthesis",
      art.synthesis || "(no synthesis)",
      "",
      "## Papers",
      ...papers.map((p, i) =>
        `### ${i + 1}. ${p.title}\n- **Year:** ${p.year || "?"}\n- **URL:** ${p.url || "?"}\n\n${p.abstract || ""}`
      ),
    ];
    const blob = new Blob([lines.join("\n\n")], { type: "text/markdown" });
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = "literature_review.md";
    a.click();
  };

  // All papers from result; split by in_scope flag
  const allPapers = result?.artifact?.papers || [];
  const papers = dedupePapers([
    ...allPapers.filter(p => p.in_scope !== false),
    ...manualIncluded,
  ]);
  const acceptedPaperKeys = new Set(papers.map(paperKey));
  const rejectedPapers = dedupePapers([
    ...(result?.artifact?.removed_papers || []),
    ...allPapers.filter(p => p.in_scope === false),
  ]).filter(p => !acceptedPaperKeys.has(paperKey(p)));
  const synthesis   = result?.artifact?.synthesis || "";

  return (
    <div style={{ display: "flex", flexDirection: "column", height: "100%", overflow: "hidden" }}>

      {/* Toolbar */}
      <div style={{ display: "flex", alignItems: "center", gap: 10, padding: "10px 16px", borderBottom: "1px solid var(--border-faint)", background: "var(--bg-surface)", flexShrink: 0 }}>
        <AgentLabel color="var(--accent-violet)" text="LITERATURE REVIEWER" />
        {running && (
          <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
            <div className="dot pulse-anim" style={{ background: "var(--accent-violet)", boxShadow: "0 0 8px var(--accent-violet)" }} />
            <span style={{ fontSize: 9, color: "var(--accent-violet)", letterSpacing: 2 }}>SEARCHING</span>
          </div>
        )}
        {result && !running && (
          <span className="badge" style={{ color: "var(--accent-teal)", background: "rgba(52,211,153,0.1)" }}>
            ✓ {papers.length} PAPERS
          </span>
        )}
        <div style={{ flex: 1 }} />
        <button className="btn" onClick={() => setShowConfig(s => !s)} style={{ fontSize: 9 }}>
          {showConfig ? "▼ HIDE CONFIG" : "▲ SHOW CONFIG"}
        </button>
        {result && (
          <div style={{ display: "flex", gap: 6 }}>
            <button className="btn btn-violet" onClick={handleExport}>⬇ EXPORT .MD</button>
            <button className="btn btn-violet" onClick={() => onAttachToDesk?.(synthesis, papers)}>➕ ATTACH TO DESK</button>
          </div>
        )}
        <button
          className={`btn ${running ? "" : "btn-violet"}`}
          onClick={handleRun}
          disabled={running}
          style={{ minWidth: 120 }}
        >
          {running ? "⏳ RUNNING…" : "▶ RUN REVIEW"}
        </button>
      </div>

      <div style={{ flex: 1, display: "flex", overflow: "hidden" }}>

        {/* Config pane */}
        {showConfig && (
          <div style={{ width: 320, flexShrink: 0, overflowY: "auto", borderRight: "1px solid var(--border-faint)", padding: "14px 16px", display: "flex", flexDirection: "column", gap: 14, background: "var(--bg-surface)" }}>

            <div>
              <div className="sec-label">QUERY</div>
              <textarea className="ta" rows={3} value={query} onChange={e => setQuery(e.target.value)} placeholder="Research query…" />
            </div>

            <div>
              <div className="fld-label">INCLUDE TOPICS (comma-separated)</div>
              <textarea className="ta" rows={2} value={includeTopics} onChange={e => setIncludeTopics(e.target.value)} placeholder="topic1, topic2, …" />
            </div>

            <div>
              <div className="fld-label">EXCLUDE TOPICS</div>
              <input className="inp" value={excludeTopics} onChange={e => setExcludeTopics(e.target.value)} placeholder="Lotka-Volterra, …" />
            </div>

            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 8 }}>
              <div><div className="fld-label">YEAR MIN</div><input className="inp" value={yearMin} onChange={e => setYearMin(e.target.value)} placeholder="2000" /></div>
              <div><div className="fld-label">YEAR MAX</div><input className="inp" value={yearMax} onChange={e => setYearMax(e.target.value)} placeholder="now" /></div>
              <div><div className="fld-label">MAX PAPERS</div><input className="inp" type="number" value={maxPapers} onChange={e => setMaxPapers(e.target.value)} /></div>
              <div><div className="fld-label">MAX ROUNDS</div><input className="inp" type="number" value={maxRounds} onChange={e => setMaxRounds(e.target.value)} /></div>
            </div>

            <div className="divider" />

            <div>
              <div className="sec-label">BACKENDS</div>
              <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
                <Toggle on={useArxiv} onChange={setUseArxiv} label="ArXiv (free)" />
                <Toggle on={useSSch}  onChange={setUseSSch}  label="Semantic Scholar" />
                {useSSch && <input className="inp" value={ssApiKey} onChange={e => setSsApiKey(e.target.value)} placeholder="SS API key (optional)" type="password" />}
                <Toggle on={usePerplexity} onChange={setUsePerplexity} label="Perplexity" />
                {usePerplexity && <input className="inp" value={pplxKey} onChange={e => setPplxKey(e.target.value)} placeholder="Perplexity API key" type="password" />}
              </div>
            </div>

            <div className="divider" />

            <LLMConfigBlock
              label="LLM (LITERATURE)"
              provider={llm.provider} url={llm.url} model={llm.model} apiKey={llm.apiKey}
              onChange={setLlmField}
            />
          </div>
        )}

        {/* Main area: logs + results */}
        <div style={{ flex: 1, display: "flex", flexDirection: "column", overflow: "hidden" }}>

          {/* Log */}
          <div ref={logRef} className="stream-log" style={{ maxHeight: 140, borderBottom: "1px solid var(--border-faint)" }}>
            {logs.length === 0 && (
              <div style={{ padding: "20px 16px", color: "var(--text-ghost)", fontStyle: "italic", fontSize: 10 }}>
                Configure and click RUN REVIEW →
              </div>
            )}
            {logs.map(l => <StreamLine key={l.id} text={l.text} type={l.type} />)}
            {running && (
              <div style={{ padding: "3px 12px", display: "flex", gap: 6, alignItems: "center" }}>
                <span className="stream-cursor" />
                <span style={{ color: "var(--text-faint)", fontSize: 9 }}>waiting for backend…</span>
              </div>
            )}
          </div>

          {/* Results */}
          {!result ? (
            <div style={{ flex: 1, display: "flex", alignItems: "center", justifyContent: "center", padding: 40 }}>
              <div style={{ textAlign: "center", maxWidth: 400 }}>
                <div style={{ fontFamily: "var(--font-serif)", fontSize: 22, color: "var(--text-primary)", fontStyle: "italic", marginBottom: 8 }}>
                  Literature Reviewer
                </div>
                <div style={{ fontSize: 9, color: "var(--text-ghost)", letterSpacing: 2, marginBottom: 16 }}>
                  MULTI-ROUND ARXIV + SEMANTIC SCHOLAR
                </div>
                <div style={{ fontSize: 10, color: "var(--text-faint)", lineHeight: 1.7 }}>
                  Start from a paper title, author list, or topic query.<br />
                  Add include topics to tighten scope before running.
                </div>
              </div>
            </div>
          ) : (
            <div style={{ flex: 1, overflowY: "auto", padding: "14px 16px", display: "flex", flexDirection: "column", gap: 16 }}>

              {/* Synthesis */}
              {synthesis && (
                <div>
                  <div className="sec-label" style={{ color: "var(--accent-violet)", marginBottom: 10 }}>SYNTHESIS</div>
                  <div className="card" style={{ borderColor: "rgba(167,139,250,0.2)" }}>
                    <div style={{ fontSize: 11, color: "var(--text-secondary)", fontFamily: "var(--font-sans)", lineHeight: 1.8, whiteSpace: "pre-wrap" }}>
                      {synthesis}
                    </div>
                  </div>
                </div>
              )}

              {/* Audit */}
              {result?.audit && (
                <div style={{ display: "flex", gap: 8, alignItems: "center" }}>
                  <span className="badge" style={{
                    color: result.audit.passed ? "var(--accent-teal)" : "var(--accent-red)",
                    background: result.audit.passed ? "rgba(52,211,153,0.1)" : "rgba(244,63,94,0.1)",
                  }}>
                    {result.audit.passed ? "✓ AUDIT PASSED" : "✗ AUDIT FAILED"}
                  </span>
                  {result.audit.issues?.map((iss, i) => (
                    <span key={i} className="badge" style={{ color: "var(--accent-red)", background: "rgba(244,63,94,0.08)" }}>
                      {iss.check_id || `issue-${i}`}
                    </span>
                  ))}
                </div>
              )}

              {/* Papers */}
              <div>
                <div className="sec-label" style={{ color: "var(--accent-blue)" }}>
                  ACCEPTED PAPERS ({papers.length})
                </div>
                <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
                  {papers.map((p, i) => (
                    <PaperCard
                      key={paperKey(p) || i}
                      paper={p}
                      idx={i}
                      onSaveDesk={handleSaveDesk}
                      onAddContext={handleAddContext}
                      onSaveLater={handleSaveLater}
                      onDownload={downloadPaper}
                    />
                  ))}
                </div>
              </div>

              {/* Rejected papers */}
              {rejectedPapers.length > 0 && (
                <div>
                  <div className="sec-label" style={{ color: "var(--accent-red)" }}>
                    REJECTED PAPERS ({rejectedPapers.length})
                  </div>
                  <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
                    {rejectedPapers.map((p, i) => (
                      <PaperCard
                        key={paperKey(p) || i}
                        paper={p}
                        idx={i}
                        variant="rejected"
                        onInclude={handleIncludeRejected}
                        onSaveDesk={handleSaveDesk}
                        onAddContext={handleAddContext}
                        onSaveLater={handleSaveLater}
                        onDownload={downloadPaper}
                      />
                    ))}
                  </div>
                </div>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
