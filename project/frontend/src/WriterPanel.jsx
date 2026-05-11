import { useState, useRef, useCallback, useEffect } from "react";
import { writerStart, writerStream, writerResult } from "./api.js";
import LLMConfigBlock from "./LLMConfigBlock.jsx";
import {
  buildWriterArtifactCatalog,
  buildWriterArtifactContext,
} from "./professorWorkflowView.js";
import { buildFocusLabel } from "./researchDeskState.js";


const DEFAULT_DESC = `Write a grounded review draft for the PI using the available article, literature, and simulation artifacts. Summarize the main model, the surrounding literature, the implemented reproduction, the outcome, and the main limitations.`;

const CHUNK_KIND_COLOR = {
  phase_change:     "var(--accent-violet)",
  tool_call:        "var(--accent-blue)",
  tool_result:      "var(--accent-teal)",
  grounding_claim:  "var(--accent-orange)",
  promise_created:  "var(--accent-gold)",
  promise_resolved: "var(--accent-teal)",
  validation_issue: "var(--accent-red)",
  error:            "var(--accent-red)",
  text_delta:       "var(--text-faint)",
  reasoning:        "var(--text-faint)",
};

function StreamChunkLine({ chunk }) {
  const color = CHUNK_KIND_COLOR[chunk.kind] || "var(--text-faint)";
  const isPhase = chunk.kind === "phase_change";
  const isErr   = chunk.kind === "error" || chunk.kind === "validation_issue";
  return (
    <div className={`stream-line ${isErr ? "error" : ""}`}
      style={{ paddingTop: isPhase ? 7 : undefined, paddingBottom: isPhase ? 7 : undefined }}>
      <span style={{ fontSize: 8, fontWeight: 700, letterSpacing: 1, color, background: `${color}12`, padding: "1px 4px", borderRadius: 2, minWidth: 80, flexShrink: 0, display: "inline-block", textAlign: "center", textTransform: "uppercase" }}>
        {chunk.kind.replace(/_/g, " ")}
      </span>
      <span style={{ flex: 1, color: isPhase ? "var(--text-primary)" : isErr ? "var(--accent-red)" : "var(--text-secondary)", whiteSpace: "pre-wrap", wordBreak: "break-word", fontSize: isPhase ? 11 : 10 }}>
        {chunk.text}
        {chunk.streaming && <span className="stream-cursor" />}
      </span>
    </div>
  );
}

function SectionTab({ sections, activeId, onSelect }) {
  if (!sections.length) return null;
  return (
    <div style={{ display: "flex", gap: 4, overflowX: "auto", padding: "8px 12px", borderBottom: "1px solid var(--border-faint)", flexShrink: 0, background: "var(--bg-surface)" }}>
      {sections.map(s => (
        <button key={s.id}
          onClick={() => onSelect(s.id)}
          className="btn"
          style={{ borderColor: activeId === s.id ? "var(--accent-violet)" : undefined, color: activeId === s.id ? "var(--accent-violet)" : undefined, background: activeId === s.id ? "rgba(167,139,250,0.1)" : undefined, fontSize: 9, whiteSpace: "nowrap" }}>
          {s.title || s.id}
        </button>
      ))}
    </div>
  );
}

export default function WriterPanel({ litArtifact, simArtifact, onArtifact, onFocusChange, isActive = true }) {
  const [description, setDescription] = useState(DEFAULT_DESC);
  const [venue, setVenue]     = useState("Research note");
  const [artCtx, setArtCtx]   = useState("");

  const [llm, setLlm]   = useState({ provider: "lmstudio", url: "http://localhost:1234/v1", model: "auto", apiKey: "" });
  const setLlmField = (f, v) => setLlm(p => ({ ...p, [f]: v }));

  const [running, setRunning]   = useState(false);
  const [jobId, setJobId]       = useState(null);
  const [chunks, setChunks]     = useState([]);
  const [sections, setSections] = useState([]); // [{id, title, content}]
  const [activeSection, setActiveSection] = useState(null);
  const [paper, setPaper]       = useState(null); // final assembled paper
  const [error, setError]       = useState(null);
  const [showConfig, setShowConfig] = useState(true);

  const logRef = useRef(null);
  const abortRef = useRef(null);

  useEffect(() => {
    if (logRef.current) logRef.current.scrollTop = logRef.current.scrollHeight;
  }, [chunks]);

  useEffect(() => {
    if (!isActive) return;
    onFocusChange?.(buildFocusLabel(description, venue));
  }, [description, venue, onFocusChange, isActive]);

  // Pre-fill artifact context from upstream panels
  useEffect(() => {
    const next = buildWriterArtifactContext(litArtifact, simArtifact);
    if (next) {
      setArtCtx(prev => prev ? prev : next);
    }
  }, [litArtifact, simArtifact]);

  const handleStart = async () => {
    setRunning(true);
    setError(null);
    setChunks([]);
    setSections([]);
    setActiveSection(null);
    setPaper(null);
    setShowConfig(false);

    const req = {
      description,
      venue,
      artifactContext: artCtx,
      artifactCatalog: buildWriterArtifactCatalog(litArtifact, simArtifact),
      llm: { provider: llm.provider, url: llm.url, model: llm.model, apiKey: llm.apiKey || null, enabled: true },
    };

    try {
      const { job_id } = await writerStart(req);
      setJobId(job_id);

      abortRef.current = new AbortController();
      const sectionBuf = {};

      await writerStream(job_id, (chunk) => {
        setChunks(p => [...p, { ...chunk, id: Date.now() + Math.random() }]);

        // Accumulate section text
        if (chunk.kind === "text_delta" && chunk.section_id) {
          sectionBuf[chunk.section_id] = (sectionBuf[chunk.section_id] || "") + chunk.text;
        }
        if (chunk.kind === "phase_change" && chunk.payload?.sections) {
          const newSecs = chunk.payload.sections.map(s => ({ id: s.id, title: s.title, content: "" }));
          setSections(newSecs);
          if (newSecs.length && !activeSection) setActiveSection(newSecs[0].id);
        }
        // Update section content live
        if (chunk.section_id && sectionBuf[chunk.section_id] !== undefined) {
          setSections(p => p.map(s => s.id === chunk.section_id ? { ...s, content: sectionBuf[chunk.section_id] } : s));
        }
      }, abortRef.current.signal);

      // Fetch final result
      try {
        const result = await writerResult(job_id);
        setPaper(result?.paper || result?.assembled_paper || null);
        if (onArtifact) onArtifact("writer", result);
      } catch { /* stream end is enough */ }

    } catch (e) {
      if (e.name !== "AbortError") {
        setError(e.message);
        setChunks(p => [...p, { id: Date.now(), kind: "error", text: e.message }]);
      }
    } finally {
      setRunning(false);
    }
  };

  const handleStop = () => {
    abortRef.current?.abort();
    setRunning(false);
  };

  const handleExport = () => {
    const content = paper || sections.map(s => `# ${s.title}\n\n${s.content}`).join("\n\n---\n\n");
    if (!content) return;
    const blob = new Blob([content], { type: "text/markdown" });
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = "mini_review.md";
    a.click();
  };

  const activeSectionObj = sections.find(s => s.id === activeSection);
  const phaseCount = chunks.filter(c => c.kind === "phase_change").length;
  const claimCount = chunks.filter(c => c.kind === "grounding_claim").length;

  return (
    <div style={{ display: "flex", flexDirection: "column", height: "100%", overflow: "hidden" }}>

      {/* Toolbar */}
      <div style={{ display: "flex", alignItems: "center", gap: 10, padding: "10px 16px", borderBottom: "1px solid var(--border-faint)", background: "var(--bg-surface)", flexShrink: 0 }}>
        <span className="badge" style={{ color: "var(--accent-pink)", background: "rgba(232,121,249,0.12)" }}>ACADEMIC WRITER</span>
        {running && <div className="dot pulse-anim" style={{ background: "var(--accent-pink)", boxShadow: "0 0 8px var(--accent-pink)" }} />}
        {sections.length > 0 && <span className="badge" style={{ color: "var(--accent-teal)", background: "rgba(52,211,153,0.1)" }}>{sections.length} SECTIONS</span>}
        {claimCount > 0 && <span className="badge" style={{ color: "var(--accent-orange)", background: "rgba(251,146,60,0.1)" }}>{claimCount} CLAIMS</span>}
        <div style={{ flex: 1 }} />
        <button className="btn" onClick={() => setShowConfig(s => !s)} style={{ fontSize: 9 }}>{showConfig ? "▼ HIDE" : "▲ CONFIG"}</button>
        {(paper || sections.length > 0) && <button className="btn btn-teal" onClick={handleExport}>⬇ EXPORT .MD</button>}
        {running
          ? <button className="btn btn-danger" onClick={handleStop}>■ STOP</button>
          : <button className="btn" style={{ background: "rgba(232,121,249,0.1)", borderColor: "var(--accent-pink)", color: "var(--accent-pink)", minWidth: 110 }} onClick={handleStart} disabled={running}>
              ▶ START WRITING
            </button>
        }
      </div>

      <div style={{ flex: 1, display: "flex", overflow: "hidden" }}>

        {/* Config */}
        {showConfig && (
          <div style={{ width: 300, flexShrink: 0, overflowY: "auto", borderRight: "1px solid var(--border-faint)", padding: "14px 16px", display: "flex", flexDirection: "column", gap: 12, background: "var(--bg-surface)" }}>
            <div>
              <div className="sec-label">WRITING TASK</div>
              <textarea className="ta" rows={4} value={description} onChange={e => setDescription(e.target.value)} placeholder="Describe the paper to write…" />
            </div>
            <div>
              <div className="fld-label">VENUE</div>
              <input className="inp" value={venue} onChange={e => setVenue(e.target.value)} placeholder="arXiv, Nature, etc." />
            </div>
            <div>
              <div className="fld-label">ARTIFACT CONTEXT (from upstream agents)</div>
              <textarea className="ta" rows={6} value={artCtx} onChange={e => setArtCtx(e.target.value)} placeholder="Paste literature synthesis and simulation results here…" />
            </div>
            <div className="divider" />
            <LLMConfigBlock label="WRITER LLM"
              provider={llm.provider} url={llm.url} model={llm.model} apiKey={llm.apiKey}
              onChange={setLlmField} />
            {error && <div style={{ fontSize: 9, color: "var(--accent-red)", fontFamily: "var(--font-mono)", padding: "6px 8px", background: "rgba(244,63,94,0.06)", borderRadius: 4 }}>{error}</div>}
          </div>
        )}

        {/* Main: log + section viewer */}
        <div style={{ flex: 1, display: "flex", flexDirection: "column", overflow: "hidden" }}>
          {!jobId ? (
            <div style={{ flex: 1, display: "flex", alignItems: "center", justifyContent: "center", padding: 40 }}>
              <div style={{ textAlign: "center", maxWidth: 420 }}>
                <div style={{ fontFamily: "var(--font-serif)", fontSize: 22, color: "var(--text-primary)", fontStyle: "italic", marginBottom: 8 }}>Academic Writer</div>
                <div style={{ fontSize: 9, color: "var(--text-ghost)", letterSpacing: 2, marginBottom: 12 }}>4-PHASE AUTHORING · GROUNDED IN ARTIFACTS</div>
                <div style={{ fontSize: 10, color: "var(--text-faint)", lineHeight: 1.7 }}>
                  Draft a grounded review from the artifacts collected so far.<br />
                  Run literature or simulation first, or paste artifact context directly.
                </div>
                {litArtifact && <div style={{ marginTop: 10 }}><span className="badge" style={{ color: "var(--accent-teal)", background: "rgba(52,211,153,0.1)" }}>✓ LITERATURE ARTIFACT AVAILABLE</span></div>}
              </div>
            </div>
          ) : (
            <div style={{ flex: 1, display: "flex", flexDirection: "column", overflow: "hidden" }}>
              {/* Event log */}
              <div ref={logRef} className="stream-log" style={{ maxHeight: 160, borderBottom: "1px solid var(--border-faint)" }}>
                {chunks.map(c => <StreamChunkLine key={c.id} chunk={c} />)}
                {running && (
                  <div style={{ padding: "3px 12px", display: "flex", gap: 6, alignItems: "center" }}>
                    <span className="stream-cursor" />
                    <span style={{ color: "var(--text-faint)", fontSize: 9 }}>drafting…</span>
                  </div>
                )}
              </div>

              {/* Section tabs + viewer */}
              <SectionTab sections={sections} activeId={activeSection} onSelect={setActiveSection} />

              <div style={{ flex: 1, overflowY: "auto", padding: "16px 20px" }}>
                {activeSectionObj ? (
                  <div>
                    <div style={{ fontSize: 13, color: "var(--text-primary)", fontFamily: "var(--font-serif)", fontStyle: "italic", marginBottom: 12 }}>
                      {activeSectionObj.title}
                    </div>
                    <div style={{ fontSize: 11, color: "var(--text-secondary)", fontFamily: "var(--font-sans)", lineHeight: 1.9, whiteSpace: "pre-wrap" }}>
                      {activeSectionObj.content || <span style={{ color: "var(--text-faint)", fontStyle: "italic" }}>(drafting…)</span>}
                      {running && activeSectionObj.id === sections[sections.length - 1]?.id && <span className="stream-cursor" />}
                    </div>
                  </div>
                ) : paper ? (
                  <div style={{ fontSize: 11, color: "var(--text-secondary)", fontFamily: "var(--font-sans)", lineHeight: 1.9, whiteSpace: "pre-wrap" }}>
                    {paper}
                  </div>
                ) : (
                  <div style={{ color: "var(--text-faint)", fontSize: 10, fontStyle: "italic" }}>
                    {running ? "Waiting for first section…" : "Paper complete — select a section above or export."}
                  </div>
                )}
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
