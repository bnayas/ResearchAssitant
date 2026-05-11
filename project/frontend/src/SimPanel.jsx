import { useState, useEffect, useRef, useCallback } from "react";
import {
  simStart, simAnswer, simApprove, simReject,
  simRunSample, simApproveSample, simApproveResults, simRejectResults,
  simApplyPatch, simWait,
} from "./api.js";
import LLMConfigBlock from "./LLMConfigBlock.jsx";
import { buildFocusLabel } from "./researchDeskState.js";


const STAGES = ["clarifying","awaiting_spec_approval","ready_to_sample","sample_running",
  "awaiting_sample_review","full_running","awaiting_results_review","complete","failed","aborted"];

const STAGE_COLOR = {
  clarifying: "#a78bfa", awaiting_spec_approval: "#34d399", ready_to_sample: "#60a5fa",
  sample_running: "#fb923c", awaiting_sample_review: "#f59e0b", full_running: "#fb923c",
  awaiting_results_review: "#f59e0b", complete: "#34d399", failed: "#f43f5e", aborted: "#f43f5e",
};

function AgentBadge({ color, text }) {
  return <span className="badge" style={{ color, background: `${color}18` }}>{text}</span>;
}

function StreamLine({ msg }) {
  const isGate = msg.type === "gate";
  const isErr  = msg.type === "error";
  const isPh   = msg.type === "phase";
  return (
    <div className={`stream-line ${isGate ? "gate" : ""} ${isErr ? "error" : ""}`}
      style={{ paddingTop: isPh ? 7 : undefined, paddingBottom: isPh ? 7 : undefined }}>
      <span style={{
        fontSize: 8, fontWeight: 700, letterSpacing: 1, minWidth: 66, flexShrink: 0,
        color: isErr ? "var(--accent-red)" : isGate ? "var(--accent-gold)" : isPh ? "var(--accent-violet)" : "var(--text-faint)",
        background: isErr ? "rgba(244,63,94,0.1)" : isGate ? "rgba(245,158,11,0.1)" : isPh ? "rgba(167,139,250,0.08)" : "transparent",
        padding: "1px 4px", borderRadius: 2, display: "inline-block", textAlign: "center",
      }}>
        {isErr ? "ERROR" : isGate ? "⬡ GATE" : isPh ? "STATE" : msg.agent || "SIM"}
      </span>
      <span style={{ flex: 1, color: isErr ? "var(--accent-red)" : isGate ? "var(--text-primary)" : "var(--text-secondary)", whiteSpace: "pre-wrap", wordBreak: "break-word" }}>
        {msg.text}
        {msg.streaming && <span className="stream-cursor" />}
      </span>
    </div>
  );
}

function QAForm({ questions, onSubmit }) {
  const [ans, setAns] = useState({});
  if (!questions?.length) return null;
  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
      <div className="sec-label" style={{ color: "var(--accent-violet)" }}>DESIGNER QUESTIONS</div>
      {questions.map(q => (
        <div key={q.id} className="card" style={{ borderColor: "rgba(167,139,250,0.2)" }}>
          <div style={{ fontSize: 10, color: "var(--accent-violet)", marginBottom: 4, fontWeight: 700 }}>{q.topic?.toUpperCase() || `Q${q.id}`}</div>
          <div style={{ fontSize: 10, color: "var(--text-secondary)", marginBottom: 6, lineHeight: 1.5 }}>{q.text}</div>
          <textarea className="ta" rows={2} value={ans[q.id] || ""} onChange={e => setAns(p => ({ ...p, [q.id]: e.target.value }))} placeholder="Your answer…" />
        </div>
      ))}
      <button className="btn btn-violet" style={{ width: "100%" }} onClick={() => onSubmit(ans)}>
        SUBMIT ANSWERS →
      </button>
    </div>
  );
}

function TextGate({ title, content, primaryLabel, primaryColor, onPrimary, secondaryLabel, onSecondary }) {
  const [rejMode, setRejMode] = useState(false);
  const [fb, setFb] = useState("");
  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
      <div className="sec-label" style={{ color: primaryColor || "var(--accent-teal)" }}>{title}</div>
      <div style={{ background: "var(--bg-base)", border: "1px solid var(--border)", borderRadius: 6, padding: "10px 12px", maxHeight: 240, overflowY: "auto", fontSize: 10, color: "var(--text-secondary)", whiteSpace: "pre-wrap", lineHeight: 1.6 }}>
        {content || "(no content)"}
      </div>
      {rejMode ? (
        <>
          <textarea className="ta" rows={3} value={fb} onChange={e => setFb(e.target.value)} placeholder="What needs to change?" />
          <div style={{ display: "flex", gap: 8 }}>
            <button className="btn" onClick={() => setRejMode(false)}>BACK</button>
            <button className="btn btn-danger" style={{ flex: 1 }} onClick={() => onSecondary(fb)}>{secondaryLabel} →</button>
          </div>
        </>
      ) : (
        <div style={{ display: "flex", gap: 8 }}>
          <button className="btn btn-danger" onClick={() => setRejMode(true)}>{secondaryLabel}</button>
          <button className="btn" style={{ flex: 1, borderColor: primaryColor, color: primaryColor, background: `${primaryColor}15` }} onClick={onPrimary}>✓ {primaryLabel} →</button>
        </div>
      )}
    </div>
  );
}

export default function SimPanel({ globalLLM, onArtifact, onFocusChange, isActive = true }) {
  const defaultGoal = "";

  const [goal, setGoal]         = useState(defaultGoal);
  const [contextNotes, setContextNotes] = useState("");
  const [sampleSteps, setSampleSteps] = useState("200");
  const [outputRoot, setOutputRoot] = useState("./simulations");

  const [designer, setDesigner] = useState({ provider: globalLLM?.provider || "lmstudio", url: globalLLM?.url || "http://localhost:1234/v1", model: globalLLM?.model || "auto", apiKey: globalLLM?.apiKey || "" });
  const [analyst,  setAnalyst]  = useState({ provider: globalLLM?.provider || "lmstudio", url: globalLLM?.url || "http://localhost:1234/v1", model: globalLLM?.model || "auto", apiKey: globalLLM?.apiKey || "" });
  const setDesignerField = (f, v) => setDesigner(p => ({ ...p, [f]: v }));
  const setAnalystField  = (f, v) => setAnalyst(p  => ({ ...p, [f]: v }));

  const [session, setSession]       = useState(null); // {sessionId, state, ...}
  const [messages, setMessages]     = useState([]);
  const [gateType, setGateType]     = useState(null); // clarify|spec_approval|sample_review|results_review
  const [gateData, setGateData]     = useState(null);
  const [polling, setPolling]       = useState(false);
  const [working, setWorking]       = useState(false);
  const [showConfig, setShowConfig] = useState(true);
  const [error, setError]           = useState(null);

  const streamRef = useRef(null);
  const pollTimer = useRef(null);

  useEffect(() => {
    if (streamRef.current) streamRef.current.scrollTop = streamRef.current.scrollHeight;
  }, [messages]);

  useEffect(() => {
    if (!isActive) return;
    onFocusChange?.(buildFocusLabel(goal));
  }, [goal, onFocusChange, isActive]);

  const addMsg = useCallback((agent, text, type = "normal") => {
    setMessages(p => [...p, { id: Date.now() + Math.random(), agent, text, type, streaming: false }]);
  }, []);

  // Process an OrchestratorUpdate from the backend and update gate/UI state
  const processUpdate = useCallback((upd) => {
    if (!upd) return;
    const state = upd.state || upd.status;
    setSession(s => ({ ...s, ...upd, state }));

    // Map state → gate
    if (state === "clarifying" && upd.questions?.length) {
      addMsg("DIRECTOR", `Designer asks ${upd.questions.length} question(s).`, "gate");
      setGateType("clarify");
      setGateData({ questions: upd.questions });
    } else if (state === "awaiting_spec_approval") {
      const content = upd.spec_summary || upd.rendered || "(spec ready)";
      addMsg("DIRECTOR", "Spec ready for PI approval.", "gate");
      setGateType("spec_approval");
      setGateData({ content });
    } else if (state === "ready_to_sample") {
      addMsg("LAUNCHER", "Script written. Ready to run sample.", "gate");
      setGateType("run_sample");
      setGateData({ content: upd.script_preview || "(script ready)" });
    } else if (state === "awaiting_sample_review") {
      const content = upd.analysis_summary || upd.rendered || "(sample complete)";
      addMsg("ANALYST", "Sample run complete. Review before full sweep.", "gate");
      setGateType("sample_review");
      setGateData({ content });
    } else if (state === "awaiting_results_review") {
      const content = upd.analysis_summary || upd.rendered || "(sweep complete)";
      addMsg("ANALYST", "Full sweep done. Final review.", "gate");
      setGateType("results_review");
      setGateData({ content });
      if (onArtifact) onArtifact("simulation", upd);
    } else if (state === "complete") {
      addMsg("DIRECTOR", "Session complete. Results available.", "phase");
      setGateType(null);
      setPolling(false);
    } else if (state === "failed" || state === "aborted") {
      addMsg("SYSTEM", `Session ${state}: ${upd.error || ""}`, "error");
      setGateType(null);
      setPolling(false);
    } else if (state === "sample_running" || state === "full_running") {
      addMsg("LAUNCHER", `Running (${state === "sample_running" ? "sample" : "full sweep"})…`, "phase");
      setGateType(null);
    }
  }, [addMsg, onArtifact]);

  // Poll for updates when a run is in progress
  useEffect(() => {
    if (!polling || !session?.sessionId) return;
    const tick = async () => {
      try {
        const upd = await simWait(session.sessionId, 15);
        processUpdate(upd);
        const st = upd.state || upd.status || "";
        if (["complete","failed","aborted","clarifying","awaiting_spec_approval",
             "ready_to_sample","awaiting_sample_review","awaiting_results_review"].includes(st)) {
          setPolling(false);
        }
      } catch (e) {
        addMsg("SYSTEM", `Poll error: ${e.message}`, "error");
        setPolling(false);
      }
    };
    pollTimer.current = setTimeout(tick, 500);
    return () => clearTimeout(pollTimer.current);
  }, [polling, session, processUpdate, addMsg]);

  const doAction = async (fn, label) => {
    setWorking(true);
    setError(null);
    try {
      const upd = await fn();
      addMsg("DIRECTOR", `Action: ${label}`, "phase");
      processUpdate(upd);
      // If now running, start polling
      const st = upd.state || upd.status || "";
      if (["sample_running","full_running"].includes(st)) setPolling(true);
    } catch (e) {
      setError(e.message);
      addMsg("SYSTEM", e.message, "error");
    } finally {
      setWorking(false);
    }
  };

  const handleStart = async () => {
    const researchGoal = goal.trim();
    if (!researchGoal) {
      setError("A research goal is required before launching the simulation workflow.");
      addMsg("SYSTEM", "Add a research goal before launching.", "error");
      return;
    }

    setWorking(true);
    setError(null);
    setMessages([]);
    setSession(null);
    setGateType(null);
    setGateData(null);
    setShowConfig(false);

    const req = {
      researchGoal,
      contextNotes,
      sampleSteps: parseInt(sampleSteps) || 200,
      outputRoot,
      designer: { provider: designer.provider, url: designer.url, model: designer.model, apiKey: designer.apiKey || null, timeoutSeconds: 120, enabled: true },
      analyst:  { provider: analyst.provider,  url: analyst.url,  model: analyst.model,  apiKey: analyst.apiKey  || null, timeoutSeconds: 120, enabled: true },
    };

    try {
      addMsg("DIRECTOR", "Starting research session…", "phase");
      const upd = await simStart(req);
      addMsg("DIRECTOR", `Session started: ${upd.session_id}`, "normal");
      setSession({ ...upd, sessionId: upd.session_id });
      processUpdate(upd);
    } catch (e) {
      setError(e.message);
      addMsg("SYSTEM", e.message, "error");
    } finally {
      setWorking(false);
    }
  };

  const sid = session?.sessionId;

  const stateColor = STAGE_COLOR[session?.state] || "var(--text-muted)";
  const isRunning = ["sample_running","full_running"].includes(session?.state);
  const isWaiting = gateType !== null;
  const isDone    = ["complete","failed","aborted"].includes(session?.state);

  return (
    <div style={{ display: "flex", flexDirection: "column", height: "100%", overflow: "hidden" }}>

      {/* Toolbar */}
      <div style={{ display: "flex", alignItems: "center", gap: 10, padding: "10px 16px", borderBottom: "1px solid var(--border-faint)", background: "var(--bg-surface)", flexShrink: 0 }}>
        <AgentBadge color="var(--accent-orange)" text="SIM TOOL" />
        {session?.state && (
          <span className="badge" style={{ color: stateColor, background: `${stateColor}15` }}>
            {(isRunning ? "● " : isWaiting ? "⬡ " : "✓ ") + session.state.toUpperCase().replace(/_/g, " ")}
          </span>
        )}
        {(isRunning || polling) && <div className="dot pulse-anim" style={{ background: "var(--accent-orange)", boxShadow: "0 0 8px var(--accent-orange)" }} />}
        <div style={{ flex: 1 }} />
        <button className="btn" onClick={() => setShowConfig(s => !s)} style={{ fontSize: 9 }}>{showConfig ? "▼ HIDE" : "▲ CONFIG"}</button>
        {isDone && <button className="btn" onClick={() => { setSession(null); setMessages([]); setGateType(null); setShowConfig(true); }}>NEW SESSION</button>}
        <button className="btn btn-primary" onClick={handleStart} disabled={working || (session && !isDone)} style={{ minWidth: 110 }}>
          {working ? "⏳ STARTING…" : "▶ LAUNCH"}
        </button>
      </div>

      <div style={{ flex: 1, display: "flex", overflow: "hidden" }}>

        {/* Config */}
        {showConfig && (
          <div style={{ width: 300, flexShrink: 0, overflowY: "auto", borderRight: "1px solid var(--border-faint)", padding: "14px 16px", display: "flex", flexDirection: "column", gap: 12, background: "var(--bg-surface)" }}>
            <div>
              <div className="sec-label">RESEARCH GOAL</div>
              <textarea
                className="ta"
                rows={5}
                value={goal}
                onChange={e => setGoal(e.target.value)}
                placeholder="Describe the simulation or reproduction goal."
              />
            </div>
            <div>
              <div className="fld-label">CONTEXT / LITERATURE NOTES</div>
              <textarea className="ta" rows={3} value={contextNotes} onChange={e => setContextNotes(e.target.value)} placeholder="Paste synthesis from literature review…" />
            </div>
            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 8 }}>
              <div><div className="fld-label">SAMPLE STEPS</div><input className="inp" type="number" value={sampleSteps} onChange={e => setSampleSteps(e.target.value)} /></div>
              <div><div className="fld-label">OUTPUT ROOT</div><input className="inp" value={outputRoot} onChange={e => setOutputRoot(e.target.value)} /></div>
            </div>
            <div className="divider" />
            <LLMConfigBlock label="DESIGNER LLM"
              provider={designer.provider} url={designer.url} model={designer.model} apiKey={designer.apiKey}
              onChange={setDesignerField} />
            <div className="divider" />
            <LLMConfigBlock label="ANALYST LLM"
              provider={analyst.provider} url={analyst.url} model={analyst.model} apiKey={analyst.apiKey}
              onChange={setAnalystField} />
          </div>
        )}

        {/* Stream + Gate panel */}
        <div style={{ flex: 1, display: "flex", overflow: "hidden" }}>

          {/* Stream */}
          <div style={{ flex: 1, display: "flex", flexDirection: "column", overflow: "hidden", borderRight: gateType ? "1px solid var(--border-faint)" : undefined }}>
            {!session ? (
              <div style={{ flex: 1, display: "flex", alignItems: "center", justifyContent: "center", padding: 40 }}>
                <div style={{ textAlign: "center", maxWidth: 420 }}>
                  <div style={{ fontFamily: "var(--font-serif)", fontSize: 22, color: "var(--text-primary)", fontStyle: "italic", marginBottom: 8 }}>Simulation Orchestrator</div>
                  <div style={{ fontSize: 9, color: "var(--text-ghost)", letterSpacing: 2, marginBottom: 12 }}>9-STATE PIPELINE · REAL EXECUTION</div>
                  <div style={{ fontSize: 10, color: "var(--text-faint)", lineHeight: 1.7 }}>
                    Describe the simulation goal and supporting context.<br />
                    Paste literature synthesis into Context Notes when available.
                  </div>
                </div>
              </div>
            ) : (
              <>
                <div style={{ padding: "5px 12px", borderBottom: "1px solid var(--border-faint)", background: "var(--bg-surface)", display: "flex", gap: 8, alignItems: "center", flexShrink: 0 }}>
                  <span style={{ fontSize: 8, color: "var(--text-ghost)", letterSpacing: 2 }}>AGENT STREAM</span>
                  <span style={{ fontSize: 8, color: "var(--text-faint)", marginLeft: "auto" }}>{messages.length} events · {sid}</span>
                </div>
                <div ref={streamRef} className="stream-log" style={{ flex: 1, paddingTop: 4, paddingBottom: 8 }}>
                  {messages.map(msg => <StreamLine key={msg.id} msg={msg} />)}
                  {(isRunning || polling) && (
                    <div style={{ padding: "3px 12px", display: "flex", gap: 6, alignItems: "center" }}>
                      <span className="stream-cursor" />
                      <span style={{ color: "var(--text-faint)", fontSize: 9 }}>running…</span>
                    </div>
                  )}
                  {isDone && !isWaiting && (
                    <div style={{ margin: "12px", padding: 12, background: "rgba(52,211,153,0.04)", border: "1px solid rgba(52,211,153,0.2)", borderRadius: 8, textAlign: "center" }}>
                      <div style={{ fontSize: 11, color: "var(--accent-teal)", fontWeight: 700, marginBottom: 6 }}>SESSION {session.state.toUpperCase()}</div>
                    </div>
                  )}
                </div>
              </>
            )}
          </div>

          {/* Gate panel */}
          {gateType && sid && (
            <div style={{ width: 320, flexShrink: 0, overflowY: "auto", padding: 14, background: "var(--bg-surface)", display: "flex", flexDirection: "column", gap: 12 }}>
              <div style={{ fontSize: 8, color: "var(--accent-red)", letterSpacing: 2, fontWeight: 700 }}>● PI DECISION REQUIRED</div>

              {gateType === "clarify" && (
                <QAForm questions={gateData?.questions || []} onSubmit={ans => doAction(() => simAnswer(sid, ans), "answer")} />
              )}

              {gateType === "spec_approval" && (
                <TextGate title="SPEC APPROVAL" content={gateData?.content}
                  primaryLabel="APPROVE SPEC" primaryColor="var(--accent-teal)"
                  secondaryLabel="REJECT"
                  onPrimary={() => doAction(() => simApprove(sid), "approve spec")}
                  onSecondary={fb => doAction(() => simReject(sid, fb), "reject spec")} />
              )}

              {gateType === "run_sample" && (
                <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
                  <div className="sec-label" style={{ color: "var(--accent-blue)" }}>SCRIPT READY</div>
                  <div style={{ background: "var(--bg-base)", border: "1px solid var(--border)", borderRadius: 6, padding: "8px 10px", maxHeight: 180, overflowY: "auto", fontSize: 9, color: "var(--text-secondary)", fontFamily: "var(--font-mono)", whiteSpace: "pre-wrap" }}>
                    {gateData?.content}
                  </div>
                  <button className="btn btn-primary" style={{ width: "100%" }} onClick={() => doAction(() => simRunSample(sid), "run sample")}>
                    ▶ RUN SAMPLE →
                  </button>
                </div>
              )}

              {gateType === "sample_review" && (
                <TextGate title="SAMPLE REVIEW" content={gateData?.content}
                  primaryLabel="PROCEED TO FULL SWEEP" primaryColor="var(--accent-teal)"
                  secondaryLabel="REVISE SPEC"
                  onPrimary={() => doAction(() => simApproveSample(sid), "approve sample")}
                  onSecondary={() => doAction(() => simReject(sid, "revise spec"), "reject sample")} />
              )}

              {gateType === "results_review" && (
                <TextGate title="RESULTS REVIEW" content={gateData?.content}
                  primaryLabel="ACCEPT & CLOSE" primaryColor="var(--accent-teal)"
                  secondaryLabel="ITERATE"
                  onPrimary={() => doAction(() => simApproveResults(sid), "approve results")}
                  onSecondary={fb => doAction(() => simRejectResults(sid, fb), "reject results")} />
              )}

              {error && <div style={{ fontSize: 9, color: "var(--accent-red)", fontFamily: "var(--font-mono)", padding: "6px 8px", background: "rgba(244,63,94,0.06)", borderRadius: 4, border: "1px solid rgba(244,63,94,0.2)" }}>{error}</div>}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
