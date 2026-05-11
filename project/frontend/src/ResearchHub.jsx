import { useState, useEffect, useCallback } from "react";
import DirectivesPanel  from "./DirectivesPanel.jsx";
import LitReviewPanel from "./LitReviewPanel.jsx";
import SimPanel       from "./SimPanel.jsx";
import WriterPanel    from "./WriterPanel.jsx";
import { healthCheck } from "./api.js";
import {
  DEFAULT_FOCUS_LABEL,
  buildDeskHeading,
  buildFocusLabel,
  loadDeskSettings,
  saveDeskSettings,
} from "./researchDeskState.js";

const TABS = [
  { id: "directives", label: "Professor Desk", icon: "📨", color: "var(--accent-teal)" },
  { id: "literature", label: "Literature",  icon: "📚", color: "var(--accent-violet)" },
  { id: "simulation", label: "Coding Lab",  icon: "⚗️",  color: "var(--accent-orange)" },
  { id: "writer",     label: "Review Draft", icon: "✍️",  color: "var(--accent-pink)" },
  { id: "review",     label: "Referee Room", icon: "🔬",  color: "var(--accent-blue)" },
];

const ASSISTANT_ROSTER = [
  { label: "Literature Reviewer", color: "var(--accent-violet)" },
  { label: "Coding Agent", color: "var(--accent-orange)" },
  { label: "Writer", color: "var(--accent-pink)" },
];

function TabPanel({ active, children }) {
  return (
    <div
      style={{
        flex: 1,
        minHeight: 0,
        overflow: "hidden",
        display: active ? "flex" : "none",
        flexDirection: "column",
      }}
    >
      {children}
    </div>
  );
}

export default function ResearchHub() {
  const [activeTab, setActiveTab]     = useState("directives");
  const [backendOk, setBackendOk]     = useState(null); // null|true|false
  const [tabStatus, setTabStatus]     = useState({ directives: "idle", literature: "idle", simulation: "idle", writer: "idle", review: "idle" });
  const [artifacts, setArtifacts]     = useState({}); // {literature, simulation, writer}
  const [deskSettings, setDeskSettings] = useState(() => loadDeskSettings());
  const [focusLabel, setFocusLabel] = useState(DEFAULT_FOCUS_LABEL);
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [deskContext, setDeskContext] = useState("");


  // Health check on mount
  useEffect(() => {
    healthCheck()
      .then(() => setBackendOk(true))
      .catch(() => setBackendOk(false));
  }, []);

  useEffect(() => {
    saveDeskSettings(deskSettings);
  }, [deskSettings]);

  const setStatus = useCallback((tab, status) => {
    setTabStatus(p => ({ ...p, [tab]: status }));
  }, []);

  const handleAttachToDesk = useCallback((synthesis, papers) => {
    const paperContext = papers.map(p => `- ${p.title} (${p.year || "?"}): ${p.abstract || ""}`).join("\n\n");
    const context = `LITERATURE SYNTHESIS:\n${synthesis || "(none)"}\n\nRELEVANT PAPERS:\n${paperContext}`;
    setDeskContext(context);
    setActiveTab("directives");
  }, []);

  const handleArtifact = useCallback((agent, data) => {
    setArtifacts(p => ({ ...p, [agent]: data }));
    setStatus(agent, "complete");
    if (agent === "literature") {
      const accepted = (data?.artifact?.papers || []).filter((paper) => paper.in_scope !== false);
      setFocusLabel(buildFocusLabel(accepted[0]?.title, data?.artifact?.synthesis));
    } else if (agent === "simulation") {
      setFocusLabel(buildFocusLabel(data?.message, data?.full_analysis?.rendered, data?.full_summary?.rendered));
    } else if (agent === "writer") {
      setFocusLabel(buildFocusLabel(data?.paper, data?.assembled_paper, "Review draft ready"));
    }
  }, [setStatus]);

  const handleProfessorNameChange = useCallback((value) => {
    setDeskSettings({ professorName: value });
  }, []);

  const handleFocusChange = useCallback((value) => {
    setFocusLabel(buildFocusLabel(value));
  }, []);

  const STATUS_COLOR = { idle: "var(--text-ghost)", running: "var(--accent-orange)", complete: "var(--accent-teal)", error: "var(--accent-red)" };

  return (
    <div style={{ display: "flex", flexDirection: "column", height: "100vh", background: "var(--bg-base)", overflow: "hidden", position: "relative" }}>

      {/* ── Global header ── */}
      <header style={{ display: "flex", alignItems: "center", gap: 18, padding: "10px 20px", borderBottom: "1px solid var(--border-faint)", background: "var(--bg-surface)", flexShrink: 0 }}>
        {/* Wordmark */}
        <div style={{ display: "flex", alignItems: "baseline", gap: 10 }}>
          <span style={{ fontFamily: "var(--font-serif)", fontSize: 24, color: "var(--accent-ink)", fontStyle: "italic", letterSpacing: -0.5 }}>
            {buildDeskHeading(deskSettings.professorName)}
          </span>
          <span style={{ fontSize: 8, color: "var(--text-muted)", letterSpacing: 3, fontFamily: "var(--font-mono)" }}>
            INSTRUCTION · DOSSIERS · ATTACHMENTS
          </span>
        </div>

        {/* Topic banner */}
        <div style={{ flex: 1, display: "flex", alignItems: "center", justifyContent: "center" }}>
          <div style={{ fontSize: 9, color: "var(--text-secondary)", fontFamily: "var(--font-mono)", letterSpacing: 1.5, background: "var(--bg-card)", border: "1px solid var(--border-soft)", borderRadius: 999, padding: "4px 12px" }}>
            {focusLabel}
          </div>
        </div>

        {/* Backend status */}
        <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
          <button
            className="btn"
            onClick={() => setSettingsOpen((open) => !open)}
            style={{ fontSize: 9 }}
          >
            {settingsOpen ? "CLOSE SETTINGS" : "SETTINGS"}
          </button>
          <div className="dot" style={{
            background: backendOk === null ? "var(--text-ghost)" : backendOk ? "var(--accent-teal)" : "var(--accent-red)",
            boxShadow: backendOk ? "0 0 6px var(--accent-teal)" : undefined,
          }} />
          <span style={{ fontSize: 8, letterSpacing: 2, fontFamily: "var(--font-mono)", color: backendOk === null ? "var(--text-faint)" : backendOk ? "var(--accent-teal)" : "var(--accent-red)" }}>
            {backendOk === null ? "CONNECTING" : backendOk ? "BACKEND OK" : "BACKEND DOWN"}
          </span>
          {backendOk === false && (
            <span style={{ fontSize: 8, color: "var(--text-ghost)", fontFamily: "var(--font-mono)" }}>
              Start: `uv run python -m sim_tool.api_server`
            </span>
          )}
        </div>
      </header>

      {settingsOpen && (
        <div style={{
          position: "absolute",
          top: 64,
          right: 20,
          width: 280,
          zIndex: 5,
          background: "var(--bg-surface)",
          border: "1px solid var(--border)",
          borderRadius: 10,
          boxShadow: "0 20px 50px rgba(15, 23, 42, 0.2)",
          padding: 16,
        }}>
          <div className="sec-label" style={{ marginBottom: 10 }}>Desk Settings</div>
          <div className="fld-label">Professor Name</div>
          <input
            className="inp"
            value={deskSettings.professorName}
            onChange={(event) => handleProfessorNameChange(event.target.value)}
            placeholder="Prof. Ada Lovelace"
          />
          <div style={{ marginTop: 8, fontSize: 10, color: "var(--text-faint)", lineHeight: 1.6 }}>
            This name is used in the desk heading and as the default PI name for directive runs.
          </div>
        </div>
      )}

      <div style={{ display: "flex", gap: 10, padding: "10px 20px", borderBottom: "1px solid var(--border-faint)", background: "var(--bg-paper)", flexShrink: 0, overflowX: "auto" }}>
        {ASSISTANT_ROSTER.map(member => (
          <div key={member.label} style={{ display: "flex", alignItems: "center", gap: 8, padding: "6px 10px", borderRadius: 999, border: "1px solid var(--border-soft)", background: "rgba(255,255,255,0.65)", flexShrink: 0 }}>
            <span className="dot" style={{ background: member.color, boxShadow: `0 0 0 4px ${member.color}18` }} />
            <span style={{ fontSize: 10, color: "var(--text-secondary)", letterSpacing: 0.7 }}>{member.label}</span>
          </div>
        ))}
      </div>

      {/* ── Tab bar ── */}
      <nav style={{ display: "flex", borderBottom: "1px solid var(--border-faint)", background: "var(--bg-surface)", flexShrink: 0, overflowX: "auto" }}>
        {TABS.map(tab => {
          const isActive = activeTab === tab.id;
          const status   = tabStatus[tab.id];
          return (
            <button key={tab.id}
              id={`tab-${tab.id}`}
              onClick={() => setActiveTab(tab.id)}
              style={{
                display: "flex", alignItems: "center", gap: 7,
                padding: "10px 20px", background: "transparent", border: "none",
                borderBottom: isActive ? `2px solid ${tab.color}` : "2px solid transparent",
                color: isActive ? tab.color : "var(--text-faint)",
                fontFamily: "var(--font-mono)", fontSize: 10, fontWeight: isActive ? 700 : 400,
                letterSpacing: 1, cursor: "pointer", transition: "all 0.15s", flexShrink: 0,
              }}>
              <span style={{ fontSize: 13 }}>{tab.icon}</span>
              {tab.label.toUpperCase()}
              {status !== "idle" && (
                <span className="badge" style={{ color: STATUS_COLOR[status], background: `${STATUS_COLOR[status]}15`, marginLeft: 2 }}>
                  {status === "running" ? "●" : status === "complete" ? "✓" : "✗"}
                </span>
              )}
            </button>
          );
        })}
      </nav>

      {/* ── Panel content ── */}
      <main style={{ flex: 1, overflow: "hidden", display: "flex", flexDirection: "column" }}>
        <TabPanel active={activeTab === "directives"}>
          <DirectivesPanel
            onArtifact={handleArtifact}
            professorName={deskSettings.professorName}
            onFocusChange={handleFocusChange}
            isActive={activeTab === "directives"}
            incomingContext={deskContext}
            onClearIncomingContext={() => setDeskContext("")}
          />
        </TabPanel>
        <TabPanel active={activeTab === "literature"}>
          <LitReviewPanel
            onArtifact={handleArtifact}
            onFocusChange={handleFocusChange}
            isActive={activeTab === "literature"}
            onAttachToDesk={handleAttachToDesk}
          />
        </TabPanel>
        <TabPanel active={activeTab === "simulation"}>
          <SimPanel
            onArtifact={handleArtifact}
            onFocusChange={handleFocusChange}
            isActive={activeTab === "simulation"}
          />
        </TabPanel>
        <TabPanel active={activeTab === "writer"}>
          <WriterPanel
            litArtifact={artifacts.literature}
            simArtifact={artifacts.simulation}
            onArtifact={handleArtifact}
            onFocusChange={handleFocusChange}
            isActive={activeTab === "writer"}
          />
        </TabPanel>
        <TabPanel active={activeTab === "review"}>
          <div style={{ flex: 1, display: "flex", alignItems: "center", justifyContent: "center", padding: 40 }}>
            <div style={{ textAlign: "center" }}>
              <div style={{ fontFamily: "var(--font-serif)", fontSize: 22, color: "var(--text-primary)", fontStyle: "italic", marginBottom: 8 }}>
                Journal Reviewer
              </div>
              <div style={{ fontSize: 9, color: "var(--text-ghost)", letterSpacing: 2, marginBottom: 16 }}>
                COMING SOON
              </div>
              <div style={{ fontSize: 10, color: "var(--text-faint)", lineHeight: 1.7 }}>
                4-phase article review with math claim verification<br />
                via CAS agents (SymPy / SciPy / Z3).
              </div>
            </div>
          </div>
        </TabPanel>
      </main>
    </div>
  );
}
