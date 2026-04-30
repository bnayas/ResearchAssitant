/**
 * LLMConfigBlock.jsx
 * Reusable LLM configuration panel with a live "Test LLM" probe button.
 * Props:
 *   label       — string, e.g. "DESIGNER LLM"
 *   provider, url, model, apiKey — controlled values
 *   onChange(field, value)       — setter
 */
import { useState } from "react";
import { probeLLM } from "./api.js";

const PROVIDERS = ["lmstudio", "anthropic", "openai", "ollama", "custom"];
const PROVIDER_DEFAULTS = {
  lmstudio:  { url: "http://localhost:1234/v1", model: "auto" },
  anthropic: { url: "https://api.anthropic.com/v1", model: "claude-sonnet-4-20250514" },
  openai:    { url: "https://api.openai.com/v1", model: "gpt-4o" },
  ollama:    { url: "http://localhost:11434/v1", model: "llama3.1" },
  custom:    { url: "", model: "" },
};

export { PROVIDERS, PROVIDER_DEFAULTS };

export default function LLMConfigBlock({ label = "LLM", provider, url, model, apiKey, onChange }) {
  const [probeState, setProbeState] = useState(null); // null | {ok, model, latency_ms, error, reply}
  const [probing, setProbing]       = useState(false);

  const handleProviderChange = (p) => {
    onChange("provider", p);
    const d = PROVIDER_DEFAULTS[p] || {};
    onChange("url", d.url || "");
    onChange("model", d.model || "");
  };

  const handleTest = async () => {
    setProbing(true);
    setProbeState(null);
    try {
      const result = await probeLLM({ provider, url, model, apiKey });
      setProbeState(result);
    } catch (e) {
      setProbeState({ ok: false, error: e.message });
    } finally {
      setProbing(false);
    }
  };

  const statusColor = probeState === null ? "var(--text-faint)"
    : probeState.ok ? "var(--accent-teal)"
    : "var(--accent-red)";

  return (
    <div>
      <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 8 }}>
        <div className="sec-label" style={{ margin: 0 }}>{label}</div>
        {probeState !== null && (
          <span className="badge" style={{ color: statusColor, background: `${statusColor}15` }}>
            {probeState.ok
              ? `✓ ${probeState.latency_ms}ms · ${probeState.model || "?"}`
              : `✗ ${(probeState.error || "failed").slice(0, 40)}`}
          </span>
        )}
      </div>

      <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
        <div>
          <div className="fld-label">PROVIDER</div>
          <select className="sel" style={{ width: "100%" }} value={provider}
            onChange={e => handleProviderChange(e.target.value)}>
            {PROVIDERS.map(p => <option key={p} value={p}>{p.toUpperCase()}</option>)}
          </select>
        </div>

        {provider !== "anthropic" && (
          <div>
            <div className="fld-label">ENDPOINT URL</div>
            <input className="inp" value={url} onChange={e => onChange("url", e.target.value)} placeholder="http://localhost:1234/v1" />
          </div>
        )}

        <div>
          <div className="fld-label">MODEL</div>
          <input className="inp" value={model} onChange={e => onChange("model", e.target.value)}
            placeholder={provider === "lmstudio" ? "auto (discovered)" : "model name"} />
        </div>

        <div>
          <div className="fld-label">API KEY</div>
          <input className="inp" type="password" value={apiKey} onChange={e => onChange("apiKey", e.target.value)}
            placeholder={provider === "lmstudio" || provider === "ollama" ? "(not required)" : "sk-…"} />
        </div>

        <button className="btn" onClick={handleTest} disabled={probing}
          style={{
            width: "100%", justifyContent: "center", marginTop: 2,
            borderColor: probeState?.ok ? "var(--accent-teal)" : probing ? "var(--border)" : "var(--border-mid)",
            color: probeState?.ok ? "var(--accent-teal)" : probeState?.ok === false ? "var(--accent-red)" : "var(--text-muted)",
          }}>
          {probing ? "⏳ TESTING…" : probeState?.ok ? "✓ LLM OK — RETEST" : probeState?.ok === false ? "✗ RETRY TEST" : "▶ TEST LLM CONNECTION"}
        </button>

        {probeState?.ok && (
          <div style={{ fontSize: 9, color: "var(--text-faint)", fontFamily: "var(--font-mono)", padding: "4px 8px", background: "rgba(52,211,153,0.04)", border: "1px solid rgba(52,211,153,0.15)", borderRadius: 4, lineHeight: 1.6 }}>
            reply: <span style={{ color: "var(--accent-teal)" }}>{probeState.reply}</span>
            {probeState.latency_ms && <span style={{ float: "right" }}>{probeState.latency_ms}ms</span>}
          </div>
        )}
        {probeState?.ok === false && (
          <div style={{ fontSize: 9, color: "var(--accent-red)", fontFamily: "var(--font-mono)", padding: "4px 8px", background: "rgba(244,63,94,0.04)", border: "1px solid rgba(244,63,94,0.15)", borderRadius: 4, lineHeight: 1.6, wordBreak: "break-all" }}>
            {probeState.error}
          </div>
        )}
      </div>
    </div>
  );
}
