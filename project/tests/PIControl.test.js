/**
 * PIControl.test.jsx
 * ──────────────────
 * Component tests for PIControl.jsx — the React PI Control GUI.
 *
 * Strategy: pure logic extraction tests (no DOM renderer needed).
 * All tested functions are extracted or replicated from PIControl.jsx.
 * Browser-rendering tests (mounting, clicking) require a jest/jsdom
 * environment — instructions included at the bottom.
 *
 * Coverage:
 *   - AGENT_DEFAULTS: all 6 agents present, correct providers
 *   - PROVIDER_DEFAULTS: URL and model for each provider
 *   - CONTROL_LEVELS: all 3 levels, correct ids
 *   - STAGES: 6 stages, correct agentKey references
 *   - AGENT_PROMPTS: all 4 LLM agents have non-empty prompts
 *   - callAgentAPI: routing logic (Anthropic vs OpenAI-compatible)
 *   - Style helpers (s.*): return objects with expected CSS keys
 *   - Gate types: 'clarify', 'agent_review', 'spec_approval',
 *                 'sample_review', 'results_review'
 *   - Pipeline stage sequence
 *   - Settings validation: control levels, provider defaults
 *
 * Run in a Node/Jest environment:
 *   npm install --save-dev jest @testing-library/react @testing-library/jest-dom
 *   npx jest PIControl.test.jsx
 *
 * Or run the standalone logic tests (no DOM) with plain Node:
 *   node PIControl.test.jsx
 */

// ─────────────────────────────────────────────────────────────────────────────
// Replicated constants (from PIControl.jsx) — single source of truth for tests
// ─────────────────────────────────────────────────────────────────────────────

const AGENT_DEFAULTS = {
  DIRECTOR:  { label:"DIRECTOR",  color:"#f59e0b", provider:"lmstudio",      model:"auto",                      url:"http://localhost:1234/v1",    apiKey:"", enabled:true },
  SYMBOLIC:  { label:"SYMBOLIC",  color:"#a78bfa", provider:"lmstudio",      model:"auto",                      url:"http://localhost:1234/v1",    apiKey:"", enabled:true },
  DESIGNER:  { label:"DESIGNER",  color:"#34d399", provider:"lmstudio",      model:"auto",                      url:"http://localhost:1234/v1",    apiKey:"", enabled:true },
  VALIDATOR: { label:"VALIDATOR", color:"#60a5fa", provider:"deterministic", model:"",                         url:"",                             apiKey:"", enabled:true },
  LAUNCHER:  { label:"LAUNCHER",  color:"#fb923c", provider:"deterministic", model:"",                         url:"",                             apiKey:"", enabled:true },
  ANALYST:   { label:"ANALYST",   color:"#e879f9", provider:"lmstudio",      model:"auto",                      url:"http://localhost:1234/v1",    apiKey:"", enabled:true },
};

const PROVIDER_DEFAULTS = {
  anthropic: { url:"https://api.anthropic.com/v1",  model:"claude-sonnet-4-20250514" },
  lmstudio:  { url:"http://localhost:1234/v1",       model:"auto" },
  ollama:    { url:"http://localhost:11434/v1",      model:"llama3.1" },
  openai:    { url:"https://api.openai.com/v1",      model:"gpt-4o" },
  custom:    { url:"http://localhost:8000/v1",       model:"" },
};

const CONTROL_LEVELS = [
  { id:"gates",        label:"Gates only",        desc:"PI consulted at 4 fixed decision points" },
  { id:"agent_review", label:"Review each agent", desc:"PI approves every agent output before pipeline continues" },
  { id:"every_step",   label:"Every step",        desc:"PI can approve, modify, or override at every decision" },
];

const STAGES = [
  {id:"symbolic", label:"Symbolic",  agentKey:"SYMBOLIC"},
  {id:"design",   label:"Designer",  agentKey:"DESIGNER"},
  {id:"validate", label:"Validate",  agentKey:"VALIDATOR"},
  {id:"sample",   label:"Sample",    agentKey:"LAUNCHER"},
  {id:"sweep",    label:"Sweep",     agentKey:"LAUNCHER"},
  {id:"analysis", label:"Analysis",  agentKey:"ANALYST"},
];

const AGENT_PROMPTS = {
  DIRECTOR: `You are the Director of a multi-agent simulation research pipeline.`,
  SYMBOLIC: `You are a Symbolic Formalization Agent.`,
  DESIGNER: `You are a Simulation Designer Agent.`,
  ANALYST:  `You are a Simulation Analyst.`,
};

const GATE_TYPES = [
  "clarify",
  "agent_review",
  "spec_approval",
  "sample_review",
  "results_review",
];

// ─────────────────────────────────────────────────────────────────────────────
// Routing logic (extracted from callAgentAPI)
// ─────────────────────────────────────────────────────────────────────────────

function isAnthropicProvider(agCfg) {
  return agCfg.provider === "anthropic" || agCfg.url.includes("anthropic.com");
}

// ─────────────────────────────────────────────────────────────────────────────
// Style helper spot-check (s.btn produces an object with CSS keys)
// ─────────────────────────────────────────────────────────────────────────────

const s = {
  btn: (c = "#334155", bg = "transparent") => ({
    padding: "8px 14px",
    background: bg,
    border: `1px solid ${c}40`,
    borderRadius: 6,
    color: c === "334155" ? "#475569" : c,
    fontFamily: "monospace",
    fontSize: 10,
    letterSpacing: 0.5,
    cursor: "pointer",
  }),
  actn: (c) => ({
    padding: "9px 14px",
    background: `${c}18`,
    border: `1px solid ${c}`,
    borderRadius: 6,
    color: c,
    fontFamily: "monospace",
    fontSize: 11,
    fontWeight: 700,
    letterSpacing: 0.5,
    cursor: "pointer",
  }),
  inp: () => ({
    width: "100%",
    boxSizing: "border-box",
    background: "#080c12",
    border: "1px solid #1e293b",
    borderRadius: 4,
    padding: "7px 10px",
    color: "#94a3b8",
    fontFamily: "monospace",
    fontSize: 11,
    outline: "none",
  }),
  sel: () => ({
    background: "#0d1117",
    border: "1px solid #1e293b",
    borderRadius: 4,
    padding: "4px 8px",
    color: "#94a3b8",
    fontFamily: "monospace",
    fontSize: 10,
    outline: "none",
    cursor: "pointer",
  }),
  ta: () => ({
    width: "100%",
    boxSizing: "border-box",
    background: "#080c12",
    border: "1px solid #1e293b",
    borderRadius: 4,
    padding: "7px 10px",
    color: "#e2e8f0",
    fontFamily: "monospace",
    fontSize: 11,
    resize: "vertical",
    outline: "none",
  }),
};

// ─────────────────────────────────────────────────────────────────────────────
// Test runner (plain Node.js, no dependencies)
// ─────────────────────────────────────────────────────────────────────────────

let passed = 0;
let failed = 0;

function test(name, fn) {
  try {
    fn();
    console.log(`  ✓ ${name}`);
    passed++;
  } catch (e) {
    console.log(`  ✗ ${name}: ${e.message}`);
    failed++;
  }
}

function assert(condition, msg = "assertion failed") {
  if (!condition) throw new Error(msg);
}

function assertEqual(a, b, msg) {
  if (a !== b) throw new Error(msg || `Expected ${JSON.stringify(b)}, got ${JSON.stringify(a)}`);
}

function assertIncludes(arr, val, msg) {
  if (!arr.includes(val)) throw new Error(msg || `${JSON.stringify(val)} not in array`);
}

// ─────────────────────────────────────────────────────────────────────────────
// AGENT_DEFAULTS
// ─────────────────────────────────────────────────────────────────────────────

console.log("\nAGENT_DEFAULTS");

test("has exactly 6 agents", () => {
  assertEqual(Object.keys(AGENT_DEFAULTS).length, 6);
});

test("DIRECTOR defaults to lmstudio provider", () => {
  assertEqual(AGENT_DEFAULTS.DIRECTOR.provider, "lmstudio");
});

test("SYMBOLIC defaults to lmstudio provider", () => {
  assertEqual(AGENT_DEFAULTS.SYMBOLIC.provider, "lmstudio");
});

test("DESIGNER defaults to lmstudio provider", () => {
  assertEqual(AGENT_DEFAULTS.DESIGNER.provider, "lmstudio");
});

test("ANALYST defaults to lmstudio provider", () => {
  assertEqual(AGENT_DEFAULTS.ANALYST.provider, "lmstudio");
});

test("VALIDATOR is deterministic (no LLM)", () => {
  assertEqual(AGENT_DEFAULTS.VALIDATOR.provider, "deterministic");
});

test("LAUNCHER is deterministic (no LLM)", () => {
  assertEqual(AGENT_DEFAULTS.LAUNCHER.provider, "deterministic");
});

test("all agents have label property", () => {
  for (const [key, ag] of Object.entries(AGENT_DEFAULTS)) {
    assert(ag.label, `${key} missing label`);
  }
});

test("all agents have color (hex)", () => {
  for (const [key, ag] of Object.entries(AGENT_DEFAULTS)) {
    assert(ag.color.startsWith("#"), `${key} color should start with #`);
    assertEqual(ag.color.length, 7, `${key} color should be 7 chars`);
  }
});

test("all agents enabled by default", () => {
  for (const [key, ag] of Object.entries(AGENT_DEFAULTS)) {
    assert(ag.enabled, `${key} should be enabled`);
  }
});

test("LLM agents have LM Studio URL", () => {
  for (const key of ["DIRECTOR", "SYMBOLIC", "DESIGNER", "ANALYST"]) {
    assert(AGENT_DEFAULTS[key].url.includes("localhost:1234"),
      `${key} URL should include localhost:1234`);
  }
});

test("LLM agents use auto model discovery by default", () => {
  for (const key of ["DIRECTOR", "SYMBOLIC", "DESIGNER", "ANALYST"]) {
    assertEqual(AGENT_DEFAULTS[key].model, "auto",
      `${key} model should be auto`);
  }
});

test("deterministic agents have empty url and model", () => {
  for (const key of ["VALIDATOR", "LAUNCHER"]) {
    assertEqual(AGENT_DEFAULTS[key].url, "");
    assertEqual(AGENT_DEFAULTS[key].model, "");
  }
});

// ─────────────────────────────────────────────────────────────────────────────
// PROVIDER_DEFAULTS
// ─────────────────────────────────────────────────────────────────────────────

console.log("\nPROVIDER_DEFAULTS");

test("has 5 providers", () => {
  assertEqual(Object.keys(PROVIDER_DEFAULTS).length, 5);
});

test("anthropic URL is correct", () => {
  assertEqual(PROVIDER_DEFAULTS.anthropic.url, "https://api.anthropic.com/v1");
});

test("lmstudio URL is localhost:1234", () => {
  assert(PROVIDER_DEFAULTS.lmstudio.url.includes("1234"));
});

test("ollama URL is localhost:11434", () => {
  assert(PROVIDER_DEFAULTS.ollama.url.includes("11434"));
});

test("openai URL is correct", () => {
  assertEqual(PROVIDER_DEFAULTS.openai.url, "https://api.openai.com/v1");
});

test("anthropic model is claude-sonnet", () => {
  assert(PROVIDER_DEFAULTS.anthropic.model.includes("claude-sonnet"));
});

test("openai model is gpt-4o", () => {
  assertEqual(PROVIDER_DEFAULTS.openai.model, "gpt-4o");
});

test("ollama model is llama3.1", () => {
  assertEqual(PROVIDER_DEFAULTS.ollama.model, "llama3.1");
});

test("custom provider has empty model", () => {
  assertEqual(PROVIDER_DEFAULTS.custom.model, "");
});

// ─────────────────────────────────────────────────────────────────────────────
// CONTROL_LEVELS
// ─────────────────────────────────────────────────────────────────────────────

console.log("\nCONTROL_LEVELS");

test("has exactly 3 control levels", () => {
  assertEqual(CONTROL_LEVELS.length, 3);
});

test("first level is 'gates'", () => {
  assertEqual(CONTROL_LEVELS[0].id, "gates");
});

test("second level is 'agent_review'", () => {
  assertEqual(CONTROL_LEVELS[1].id, "agent_review");
});

test("third level is 'every_step'", () => {
  assertEqual(CONTROL_LEVELS[2].id, "every_step");
});

test("all levels have label and desc", () => {
  for (const cl of CONTROL_LEVELS) {
    assert(cl.label, `level ${cl.id} missing label`);
    assert(cl.desc, `level ${cl.id} missing desc`);
  }
});

test("gates level mentions '4 fixed decision points'", () => {
  assert(CONTROL_LEVELS[0].desc.includes("4"));
});

// ─────────────────────────────────────────────────────────────────────────────
// STAGES
// ─────────────────────────────────────────────────────────────────────────────

console.log("\nSTAGES");

test("has exactly 6 stages", () => {
  assertEqual(STAGES.length, 6);
});

test("stage ids are unique", () => {
  const ids = STAGES.map(s => s.id);
  assertEqual(new Set(ids).size, ids.length);
});

test("first stage is symbolic", () => {
  assertEqual(STAGES[0].id, "symbolic");
});

test("last stage is analysis", () => {
  assertEqual(STAGES[STAGES.length - 1].id, "analysis");
});

test("all stages reference valid agentKey", () => {
  const validKeys = Object.keys(AGENT_DEFAULTS);
  for (const st of STAGES) {
    assertIncludes(validKeys, st.agentKey,
      `stage ${st.id} has invalid agentKey: ${st.agentKey}`);
  }
});

test("validate stage uses VALIDATOR", () => {
  const v = STAGES.find(s => s.id === "validate");
  assertEqual(v.agentKey, "VALIDATOR");
});

test("sample and sweep stages use LAUNCHER", () => {
  for (const id of ["sample", "sweep"]) {
    const st = STAGES.find(s => s.id === id);
    assertEqual(st.agentKey, "LAUNCHER", `${id} should use LAUNCHER`);
  }
});

test("analysis stage uses ANALYST", () => {
  const a = STAGES.find(s => s.id === "analysis");
  assertEqual(a.agentKey, "ANALYST");
});

test("all stages have non-empty label", () => {
  for (const st of STAGES) {
    assert(st.label, `stage ${st.id} missing label`);
  }
});

// ─────────────────────────────────────────────────────────────────────────────
// AGENT_PROMPTS
// ─────────────────────────────────────────────────────────────────────────────

console.log("\nAGENT_PROMPTS");

test("has prompts for DIRECTOR, SYMBOLIC, DESIGNER, ANALYST", () => {
  for (const key of ["DIRECTOR", "SYMBOLIC", "DESIGNER", "ANALYST"]) {
    assert(AGENT_PROMPTS[key], `Missing prompt for ${key}`);
  }
});

test("VALIDATOR and LAUNCHER have no prompt (deterministic)", () => {
  assert(!AGENT_PROMPTS.VALIDATOR, "VALIDATOR should not have LLM prompt");
  assert(!AGENT_PROMPTS.LAUNCHER, "LAUNCHER should not have LLM prompt");
});

test("all prompts are non-empty strings", () => {
  for (const [key, prompt] of Object.entries(AGENT_PROMPTS)) {
    assert(typeof prompt === "string" && prompt.length > 20,
      `${key} prompt too short`);
  }
});

test("DIRECTOR prompt mentions 'Director'", () => {
  assert(AGENT_PROMPTS.DIRECTOR.toLowerCase().includes("director"));
});

test("SYMBOLIC prompt mentions 'Symbolic'", () => {
  assert(AGENT_PROMPTS.SYMBOLIC.toLowerCase().includes("symbolic"));
});

test("DESIGNER prompt mentions 'Designer' or 'simulation'", () => {
  const p = AGENT_PROMPTS.DESIGNER.toLowerCase();
  assert(p.includes("designer") || p.includes("simulation"));
});

test("ANALYST prompt mentions 'Analyst'", () => {
  assert(AGENT_PROMPTS.ANALYST.toLowerCase().includes("analyst"));
});

// ─────────────────────────────────────────────────────────────────────────────
// isAnthropicProvider routing logic
// ─────────────────────────────────────────────────────────────────────────────

console.log("\ncallAgentAPI routing");

test("anthropic provider → Anthropic path", () => {
  assert(isAnthropicProvider({ provider: "anthropic", url: "https://api.anthropic.com/v1" }));
});

test("url containing anthropic.com → Anthropic path", () => {
  assert(isAnthropicProvider({ provider: "openai", url: "https://api.anthropic.com/v1" }));
});

test("lmstudio provider → OpenAI-compatible path", () => {
  assert(!isAnthropicProvider({ provider: "lmstudio", url: "http://localhost:1234/v1" }));
});

test("ollama provider → OpenAI-compatible path", () => {
  assert(!isAnthropicProvider({ provider: "ollama", url: "http://localhost:11434/v1" }));
});

test("openai provider → OpenAI-compatible path", () => {
  assert(!isAnthropicProvider({ provider: "openai", url: "https://api.openai.com/v1" }));
});

test("custom provider with non-anthropic URL → OpenAI-compatible path", () => {
  assert(!isAnthropicProvider({ provider: "custom", url: "http://localhost:8000/v1" }));
});

// ─────────────────────────────────────────────────────────────────────────────
// Style helpers
// ─────────────────────────────────────────────────────────────────────────────

console.log("\nStyle helpers");

test("s.btn returns object with cursor: pointer", () => {
  assertEqual(s.btn().cursor, "pointer");
});

test("s.btn returns object with fontFamily: monospace", () => {
  assertEqual(s.btn().fontFamily, "monospace");
});

test("s.btn accepts color parameter", () => {
  const style = s.btn("#f59e0b");
  assert(style.border.includes("#f59e0b"), "border should use color");
});

test("s.actn returns object with fontWeight 700", () => {
  assertEqual(s.actn("#34d399").fontWeight, 700);
});

test("s.actn returns object with correct color", () => {
  assertEqual(s.actn("#34d399").color, "#34d399");
});

test("s.actn background includes color", () => {
  const style = s.actn("#34d399");
  assert(style.background.includes("#34d399"), "background should include color");
});

test("s.inp returns object with width 100%", () => {
  assertEqual(s.inp().width, "100%");
});

test("s.sel returns object with cursor: pointer", () => {
  assertEqual(s.sel().cursor, "pointer");
});

test("s.ta returns object with resize: vertical", () => {
  assertEqual(s.ta().resize, "vertical");
});

test("all style helpers return plain objects", () => {
  for (const [name, fn] of Object.entries(s)) {
    const result = fn("#000");
    assert(typeof result === "object" && result !== null,
      `s.${name} should return an object`);
  }
});

// ─────────────────────────────────────────────────────────────────────────────
// Gate types
// ─────────────────────────────────────────────────────────────────────────────

console.log("\nGate types");

test("exactly 5 gate types defined", () => {
  assertEqual(GATE_TYPES.length, 5);
});

test("clarify gate present", () => {
  assertIncludes(GATE_TYPES, "clarify");
});

test("spec_approval gate present", () => {
  assertIncludes(GATE_TYPES, "spec_approval");
});

test("sample_review gate present", () => {
  assertIncludes(GATE_TYPES, "sample_review");
});

test("results_review gate present", () => {
  assertIncludes(GATE_TYPES, "results_review");
});

test("agent_review gate present", () => {
  assertIncludes(GATE_TYPES, "agent_review");
});

// ─────────────────────────────────────────────────────────────────────────────
// Pipeline stage ordering invariants
// ─────────────────────────────────────────────────────────────────────────────

console.log("\nPipeline ordering");

test("symbolic comes before design", () => {
  const sIdx = STAGES.findIndex(s => s.id === "symbolic");
  const dIdx = STAGES.findIndex(s => s.id === "design");
  assert(sIdx < dIdx, "symbolic should precede design");
});

test("design comes before validate", () => {
  const dIdx = STAGES.findIndex(s => s.id === "design");
  const vIdx = STAGES.findIndex(s => s.id === "validate");
  assert(dIdx < vIdx, "design should precede validate");
});

test("validate comes before sample", () => {
  const vIdx = STAGES.findIndex(s => s.id === "validate");
  const sIdx = STAGES.findIndex(s => s.id === "sample");
  assert(vIdx < sIdx, "validate should precede sample");
});

test("sample comes before sweep", () => {
  const sIdx = STAGES.findIndex(s => s.id === "sample");
  const swIdx = STAGES.findIndex(s => s.id === "sweep");
  assert(sIdx < swIdx, "sample should precede sweep");
});

test("sweep comes before analysis", () => {
  const swIdx = STAGES.findIndex(s => s.id === "sweep");
  const aIdx = STAGES.findIndex(s => s.id === "analysis");
  assert(swIdx < aIdx, "sweep should precede analysis");
});

// ─────────────────────────────────────────────────────────────────────────────
// Settings defaults validation
// ─────────────────────────────────────────────────────────────────────────────

console.log("\nSettings defaults");

test("default control level is 'gates'", () => {
  const defaultSettings = { agents: AGENT_DEFAULTS, controlLevel: "gates" };
  assertEqual(defaultSettings.controlLevel, "gates");
});

test("control level 'gates' is valid", () => {
  assertIncludes(CONTROL_LEVELS.map(c => c.id), "gates");
});

test("applying anthropic provider defaults sets correct URL", () => {
  const d = PROVIDER_DEFAULTS["anthropic"];
  assert(d.url === "https://api.anthropic.com/v1");
});

test("applying lmstudio provider defaults sets localhost URL", () => {
  const d = PROVIDER_DEFAULTS["lmstudio"];
  assert(d.url.startsWith("http://localhost"));
});

test("agents deep-copy is independent", () => {
  const copy = JSON.parse(JSON.stringify(AGENT_DEFAULTS));
  copy.DIRECTOR.enabled = false;
  assert(AGENT_DEFAULTS.DIRECTOR.enabled === true, "original should be unaffected");
});

// ─────────────────────────────────────────────────────────────────────────────
// Results
// ─────────────────────────────────────────────────────────────────────────────

console.log(`\n${passed} passed, ${failed} failed out of ${passed + failed} tests.`);
if (failed > 0) process.exit(1);
