/**
 * api.js — Typed fetch wrappers for all Research Platform backend endpoints.
 * Backend: FastAPI at http://127.0.0.1:8001 (proxied via Vite to /api)
 */

const BASE = "";  // Vite proxy handles /api → :8001

// ── Helpers ────────────────────────────────────────────────────────────────

async function post(path, body) {
  const res = await fetch(BASE + path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) {
    const text = await res.text().catch(() => "");
    throw new Error(`${res.status} ${res.statusText}: ${text.slice(0, 300)}`);
  }
  return res.json();
}

async function get(path) {
  const res = await fetch(BASE + path);
  if (!res.ok) {
    const text = await res.text().catch(() => "");
    throw new Error(`${res.status} ${res.statusText}: ${text.slice(0, 300)}`);
  }
  return res.json();
}

/**
 * Read a fetch Response body as NDJSON, calling onLine for each parsed object.
 * Stops on done or when abortSignal fires.
 */
async function readNDJSON(response, onLine) {
  const reader = response.body.getReader();
  const dec = new TextDecoder();
  let buf = "";
  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buf += dec.decode(value, { stream: true });
      const lines = buf.split("\n");
      buf = lines.pop() ?? "";
      for (const line of lines) {
        const trimmed = line.trim();
        if (!trimmed) continue;
        try { onLine(JSON.parse(trimmed)); } catch { /* skip malformed */ }
      }
    }
    // Flush remainder
    if (buf.trim()) {
      try { onLine(JSON.parse(buf.trim())); } catch { /* skip */ }
    }
  } finally {
    reader.releaseLock();
  }
}

// ── LLM probe ──────────────────────────────────────────────────────────────

/**
 * Test an LLM config with a minimal ping.
 * Returns {ok, model, latency_ms, reply, error}
 */
export async function probeLLM({ provider, url, model, apiKey }) {
  return post("/api/health/llm", {
    provider,
    base_url: url,
    model,
    api_key: apiKey || null,
    enabled: true,
  });
}

// ── Health ─────────────────────────────────────────────────────────────────

export async function healthCheck() {
  return get("/health");
}

// ── Literature Review ──────────────────────────────────────────────────────

/**
 * Run a literature review (synchronous — blocks until complete).
 * @param {object} req  Matches LiteratureReviewRequest schema
 * @returns {Promise<{artifact, audit}>}
 */
export async function literatureRun(req) {
  return post("/api/literature/reviews/run", req);
}

/**
 * Stream a literature review as NDJSON events.
 * Calls onEvent(event) for each StreamEvent object.
 */
export async function literatureStream(req, onEvent, signal) {
  const res = await fetch(BASE + "/api/literature/reviews/stream", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(req),
    signal,
  });
  if (!res.ok) {
    const text = await res.text().catch(() => "");
    throw new Error(`Literature stream ${res.status}: ${text.slice(0, 300)}`);
  }
  await readNDJSON(res, onEvent);
}

// ── Simulation ─────────────────────────────────────────────────────────────

export async function simStart(req) {
  return post("/api/simulations/start", req);
}

export async function simAction(sessionId, action, body = {}) {
  return post(`/api/simulations/${sessionId}/${action}`, body);
}

export async function simStatus(sessionId) {
  return get(`/api/simulations/${sessionId}/status`);
}

export async function simWait(sessionId, timeoutSeconds = 300) {
  return post(`/api/simulations/${sessionId}/wait`, { timeoutSeconds });
}

export async function simAnswer(sessionId, answers) {
  return simAction(sessionId, "answer", { answers });
}

export async function simApprove(sessionId) {
  return simAction(sessionId, "approve");
}

export async function simReject(sessionId, feedback = "") {
  return simAction(sessionId, "reject", { feedback });
}

export async function simRunSample(sessionId) {
  return simAction(sessionId, "run-sample");
}

export async function simApproveSample(sessionId) {
  return simAction(sessionId, "approve-sample");
}

export async function simApproveResults(sessionId) {
  return simAction(sessionId, "approve-results");
}

export async function simRejectResults(sessionId, feedback = "") {
  return simAction(sessionId, "reject-results", { feedback });
}

export async function simApplyPatch(sessionId) {
  return simAction(sessionId, "apply-patch");
}

// ── Writer ─────────────────────────────────────────────────────────────────

/**
 * Start an academic writing job.
 * @param {object} req  WriterStartRequest
 * @returns {Promise<{job_id}>}
 */
export async function writerStart(req) {
  return post("/api/writer/start", req);
}

/**
 * Stream writer events for a job.
 * @param {string} jobId
 * @param {function} onChunk  called with each StreamChunk JSON object
 * @param {AbortSignal} [signal]
 */
export async function writerStream(jobId, onChunk, signal) {
  const res = await fetch(BASE + `/api/writer/${jobId}/stream`, { signal });
  if (!res.ok) {
    const text = await res.text().catch(() => "");
    throw new Error(`Writer stream ${res.status}: ${text.slice(0, 300)}`);
  }
  await readNDJSON(res, onChunk);
}

/**
 * Get the final writer artifact (assembled paper + metadata).
 */
export async function writerResult(jobId) {
  return get(`/api/writer/${jobId}/result`);
}

// ── Directives ─────────────────────────────────────────────────────────────

export async function streamDirective(payload, onEvent) {
  const res = await fetch(BASE + "/api/directives", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  if (!res.ok) {
    throw new Error(`Failed to stream directive: ${res.status}`);
  }
  await readNDJSON(res, onEvent);
}

export async function submitDirectiveSteering(directiveId, payload) {
  return post(`/api/directives/${directiveId}/steering`, payload);
}

export async function stopDirective(directiveId, payload = {}) {
  return post(`/api/directives/${directiveId}/stop`, payload);
}

export async function getDirectiveSnapshot(directiveId) {
  return get(`/api/directives/${directiveId}/snapshot`);
}
