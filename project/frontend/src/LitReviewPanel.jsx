import { useState, useRef, useCallback, useEffect } from "react";
import { literatureRun } from "./api.js";
import LLMConfigBlock from "./LLMConfigBlock.jsx";
import { buildFocusLabel } from "./researchDeskState.js";

const DEFAULT_QUERY = "";


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

function PaperCard({ paper, idx }) {
  const [expanded, setExpanded] = useState(false);
  return (
    <div className="paper-card slide-up" style={{ animationDelay: `${idx * 0.05}s` }}>
      <div className="paper-title">{paper.title || "(no title)"}</div>
      <div className="paper-meta">
        {paper.year && <span className="badge" style={{ color: "var(--accent-blue)", background: "rgba(96,165,250,0.1)" }}>{paper.year}</span>}
        {paper.source && <span className="badge" style={{ color: "var(--text-faint)", background: "var(--bg-base)" }}>{paper.source}</span>}
        {paper.url && (
          <a href={paper.url} target="_blank" rel="noreferrer"
            style={{ fontSize: 9, color: "var(--accent-gold)", fontFamily: "var(--font-mono)", textDecoration: "none" }}>
            ↗ OPEN
          </a>
        )}
      </div>
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
    </div>
  );
}

export default function LitReviewPanel({ globalLLM, onArtifact, onFocusChange }) {
  const [query, setQuery] = useState(DEFAULT_QUERY);
  const [includeTopics, setIncludeTopics] = useState("");
  const [excludeTopics, setExcludeTopics] = useState("");
  const [yearMin, setYearMin]   = useState("2000");
  const [yearMax, setYearMax]   = useState("");
  const [maxPapers, setMaxPapers] = useState("12");
  const [maxRounds, setMaxRounds] = useState("3");

  const [useArxiv, setUseArxiv]   = useState(true);
  const [useSSch, setUseSSch]     = useState(false);  // disabled by default — needs API key to avoid rate-limits
  const [usePerplexity, setUsePerplexity] = useState(false);
  const [ssApiKey, setSsApiKey]   = useState("");
  const [pplxKey, setPplxKey]     = useState("");

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

  const logRef = useRef(null);
  const addLog = useCallback((text, type = "normal") => {
    setLogs(p => [...p, { id: Date.now() + Math.random(), text, type }]);
  }, []);

  useEffect(() => {
    if (logRef.current) logRef.current.scrollTop = logRef.current.scrollHeight;
  }, [logs]);

  // Sync with global LLM changes
  useEffect(() => {
    if (globalLLM) setLlm(p => ({ ...p, ...globalLLM }));
  }, [globalLLM]);

  useEffect(() => {
    onFocusChange?.(buildFocusLabel(query, includeTopics));
  }, [query, includeTopics, onFocusChange]);

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
      // The sync endpoint blocks until done — we show a spinner
      addLog("Searching and synthesising (this may take 30–120s)…", "normal");
      const data = await literatureRun(req);
      addLog("Literature review complete.", "phase");

      const art = data?.artifact || {};
      // Backend returns art.papers[] with in_scope boolean on each
      const allPapers = art.papers || [];
      const accepted  = allPapers.filter(p => p.in_scope !== false);
      const rejected  = allPapers.filter(p => p.in_scope === false).length;
      addLog(`Accepted ${accepted.length} paper(s), rejected ${rejected}.`, "normal");
      if (art.synthesis) addLog("Synthesis generated.", "normal");

      const audit = data?.audit || {};
      if (audit.passed === false) {
        addLog(`Audit: FAILED — ${audit.issues?.length || 0} issue(s).`, "error");
      } else {
        addLog("Audit: PASSED.", "normal");
      }

      setResult(data);
      if (onArtifact) onArtifact("literature", data);
    } catch (err) {
      setError(err.message);
      addLog(err.message, "error");
    } finally {
      setRunning(false);
    }
  };

  const handleExport = () => {
    if (!result?.artifact) return;
    const art = result.artifact;
    const papers = (art.papers || []).filter(p => p.in_scope !== false);
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
  const allPapers   = result?.artifact?.papers || [];
  const papers      = allPapers.filter(p => p.in_scope !== false);
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
          <button className="btn btn-violet" onClick={handleExport}>⬇ EXPORT .MD</button>
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
                  {papers.map((p, i) => <PaperCard key={p.url || i} paper={p} idx={i} />)}
                </div>
              </div>

              {/* Rejected count */}
              {allPapers.filter(p => p.in_scope === false).length > 0 && (
                <div style={{ fontSize: 9, color: "var(--text-faint)", letterSpacing: 1 }}>
                  {allPapers.filter(p => p.in_scope === false).length} paper(s) rejected by scope filter.
                </div>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
