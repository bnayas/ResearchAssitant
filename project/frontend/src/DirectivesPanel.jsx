import { useState, useRef, useEffect } from "react";
import LLMConfigBlock from "./LLMConfigBlock.jsx";
import { streamDirective, submitDirectiveSteering, stopDirective, getDirectiveSnapshot } from "./api.js";
import {
  attachmentHref,
  buildAnswerFields,
  buildQuestionResponsePayload,
  getAwaitingSteering,
  hasAnswerValues,
  initialAnswersForSteering,
  parseJsonAttachment,
} from "./professorWorkflowView.js";
import { buildFocusLabel } from "./researchDeskState.js";

function emailKey(email) {
  return [
    email?.timestamp ?? "",
    email?.subject ?? "",
    email?.from_agent ?? "",
    String(email?.body || "").slice(0, 80),
  ].join("|");
}

function mergeEmails(existing, incoming) {
  const merged = [];
  const seen = new Set();
  [...existing, ...incoming].forEach((email) => {
    const key = emailKey(email);
    if (seen.has(key)) return;
    seen.add(key);
    merged.push(email);
  });
  const sorted = merged.sort((a, b) => (a.timestamp || 0) - (b.timestamp || 0));
  if (
    sorted.length === existing.length
    && sorted.every((email, index) => emailKey(email) === emailKey(existing[index]))
  ) {
    return existing;
  }
  return sorted;
}

function MessageBody({ text, defaultExpanded = false }) {
  const [expanded, setExpanded] = useState(defaultExpanded);
  const CHAR_LIMIT = defaultExpanded ? 5000 : 2000;
  const isLong = text && text.length > CHAR_LIMIT;
  const displayText = (!expanded && isLong)
    ? `${text.substring(0, Math.max(260, CHAR_LIMIT - 220)).trimEnd()}\n\n...\n\n${text.slice(-200).trimStart()}`
    : text;

  return (
    <div style={{ fontSize: 11, color: "var(--text-secondary)", whiteSpace: "pre-wrap", lineHeight: 1.6 }}>
      {displayText}
      {isLong && (
        <button
          className="btn"
          style={{ marginTop: 10, padding: "6px 10px", fontSize: 10, display: "block" }}
          onClick={() => setExpanded((value) => !value)}
        >
          {expanded ? "COLLAPSE MESSAGE" : "EXPAND MESSAGE"}
        </button>
      )}
    </div>
  );
}

function AttachmentRow({ attachments }) {
  const [expandedIndex, setExpandedIndex] = useState(null);

  if (!attachments || attachments.length === 0) return null;

  const expandedAtt = expandedIndex !== null ? attachments[expandedIndex] : null;
  const expandedPayload = expandedAtt ? parseJsonAttachment(expandedAtt) : null;

  return (
    <div style={{ borderTop: "1px solid var(--border-faint)", background: "var(--bg-surface)", padding: "12px 16px" }}>
      <div className="fld-label" style={{ marginBottom: 8 }}>Attachments ({attachments.length})</div>
      <div style={{ display: "flex", gap: 10, flexWrap: "wrap" }}>
        {attachments.map((att, aIdx) => (
          <div key={aIdx} style={{ display: "flex", alignItems: "center", gap: 6, background: "var(--bg-base)", border: "1px solid var(--border)", padding: "6px 10px", borderRadius: 4, fontSize: 10 }}>
            <span>{att.url ? "🔗" : "📎"}</span>
            <span style={{ color: "var(--text-primary)" }}>{att.name}</span>
            {parseJsonAttachment(att) ? (
              <button
                className="btn"
                onClick={() => setExpandedIndex(expandedIndex === aIdx ? null : aIdx)}
                style={{ fontSize: 9, padding: "2px 6px", marginLeft: 8 }}
              >
                {expandedIndex === aIdx ? "HIDE DETAILS" : "VIEW DETAILS"}
              </button>
            ) : attachmentHref(att) ? (
              <a href={attachmentHref(att)} target="_blank" rel="noreferrer" style={{ color: "var(--accent-blue)", textDecoration: "none", marginLeft: 8 }}>[OPEN]</a>
            ) : (
              <span style={{ color: "var(--text-muted)", marginLeft: 8, fontSize: 9 }}>[INLINE]</span>
            )}
          </div>
        ))}
      </div>
      {expandedPayload && (
        <div className="card slide-down" style={{ marginTop: 12, background: "var(--bg-paper)", padding: "12px" }}>
          {expandedPayload.title && <div style={{ fontWeight: "bold", fontSize: 12, marginBottom: 4, color: "var(--text-primary)" }}>{expandedPayload.title}</div>}
          {expandedPayload.authors && <div style={{ fontSize: 10, color: "var(--text-secondary)", marginBottom: 4 }}>{Array.isArray(expandedPayload.authors) ? expandedPayload.authors.join(", ") : expandedPayload.authors}</div>}
          {expandedPayload.year && <div style={{ fontSize: 10, color: "var(--text-muted)", marginBottom: 4 }}>Year: {expandedPayload.year}</div>}
          {expandedPayload.url && <div style={{ fontSize: 10, color: "var(--accent-blue)", marginBottom: 4 }}><a href={expandedPayload.url} target="_blank" rel="noreferrer" style={{ color: "inherit", textDecoration: "none" }}>URL: {expandedPayload.url}</a></div>}
          {expandedPayload.abstract && <div style={{ fontSize: 10, marginTop: 8, whiteSpace: "pre-wrap", color: "var(--text-secondary)", lineHeight: 1.6 }}>{expandedPayload.abstract}</div>}
          {!expandedPayload.title && !expandedPayload.abstract && (
            <pre style={{ fontSize: 9, overflowX: "auto", whiteSpace: "pre-wrap", color: "var(--text-secondary)", marginTop: 4 }}>
              {JSON.stringify(expandedPayload, null, 2)}
            </pre>
          )}
        </div>
      )}
    </div>
  );
}

function SteeringSummary({ email }) {
  const attachments = email?.attachments || [];
  const articleMatch = attachments
    .map(parseJsonAttachment)
    .find((payload) => payload?.title && (payload?.authors || payload?.short_id));
  const brief = attachments
    .map(parseJsonAttachment)
    .find((payload) => payload?.model_description && payload?.procedure);

  if (articleMatch) {
    return (
      <div className="card" style={{ background: "var(--bg-paper)", marginBottom: 12 }}>
        <div className="fld-label" style={{ marginBottom: 8 }}>Current Article Match</div>
        <div style={{ fontFamily: "var(--font-serif)", fontSize: 18, color: "var(--text-primary)", lineHeight: 1.35 }}>
          {articleMatch.title}
        </div>
        <div style={{ marginTop: 6, fontSize: 10, color: "var(--text-secondary)", lineHeight: 1.6 }}>
          {Array.isArray(articleMatch.authors) ? articleMatch.authors.join(", ") : ""}
          {(articleMatch.year || articleMatch.short_id) ? ` (${articleMatch.year || "?"})` : ""}
          {articleMatch.short_id ? ` — ${articleMatch.short_id}` : ""}
        </div>
        {articleMatch.lookup_query && (
          <div style={{ marginTop: 10, fontSize: 10, color: "var(--text-faint)", lineHeight: 1.6 }}>
            Lookup query: {articleMatch.lookup_query}
          </div>
        )}
        {articleMatch.abstract && (
          <div style={{ marginTop: 12, fontSize: 10, color: "var(--text-secondary)", lineHeight: 1.7 }}>
            {articleMatch.abstract}
          </div>
        )}
      </div>
    );
  }

  if (brief) {
    return (
      <div className="card" style={{ background: "var(--bg-paper)", marginBottom: 12 }}>
        <div className="fld-label" style={{ marginBottom: 8 }}>Current Article Brief</div>
        <div style={{ fontSize: 10, color: "var(--text-secondary)", lineHeight: 1.7 }}>
          <strong>Model:</strong> {brief.model_description || "(missing)"}
        </div>
        <div style={{ marginTop: 8, fontSize: 10, color: "var(--text-secondary)", lineHeight: 1.7 }}>
          <strong>Procedure:</strong> {brief.procedure || "(missing)"}
        </div>
      </div>
    );
  }

  return null;
}

function EmailCard({ email }) {
  const steering = getAwaitingSteering(email);

  return (
    <div className="card slide-up" style={{ padding: 0, overflow: "hidden", flexShrink: 0, borderColor: steering ? "var(--accent-orange)" : undefined }}>
      <div style={{ background: "var(--bg-surface)", padding: "12px 16px", borderBottom: "1px solid var(--border-faint)" }}>
        <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 4 }}>
          <div style={{ fontSize: 11, fontWeight: 600, color: "var(--text-primary)" }}>{email.from_agent} <span style={{ color: "var(--text-muted)", fontWeight: 400 }}>&lt;{email.from_addr}&gt;</span></div>
          <div style={{ fontSize: 10, color: "var(--text-muted)" }}>{new Date(email.timestamp * 1000).toLocaleTimeString()}</div>
        </div>
        <div style={{ fontSize: 10, color: "var(--text-muted)", marginBottom: 12 }}>To: {email.to_name}</div>
        <div style={{ fontSize: 12, fontWeight: 700, color: "var(--text-primary)" }}>{email.subject}</div>
        {steering && (
          <div style={{ marginTop: 8, fontSize: 9, color: "var(--accent-orange)", letterSpacing: 1.1, fontFamily: "var(--font-mono)" }}>
            AWAITING PI STEERING
          </div>
        )}
      </div>

      <div style={{ padding: "16px" }}>
        <SteeringSummary email={email} />
        <MessageBody text={email.body || ""} defaultExpanded={Boolean(steering)} />
      </div>

      <AttachmentRow attachments={email.attachments} />
    </div>
  );
}

export default function DirectivesPanel({ onArtifact, professorName, onFocusChange, incomingContext, onClearIncomingContext, isActive = true }) {
  const [directive, setDirective] = useState({
    instruction: "",
    topicHint: "",
    phases: { find_article: true, parse_article: true, literature: true, simulation: true, write: true },
    llm: { provider: "lmstudio", url: "http://localhost:1234/v1", model: "auto" }
  });

  const [emails, setEmails] = useState([]);
  const [isRunning, setIsRunning] = useState(false);
  const [error, setError] = useState(null);
  const [directiveId, setDirectiveId] = useState(null);
  const [pendingSteering, setPendingSteering] = useState(null);
  const [steeringFeedback, setSteeringFeedback] = useState("");
  const [steeringAnswers, setSteeringAnswers] = useState({});
  const [steeringBusy, setSteeringBusy] = useState(false);
  const [steeringError, setSteeringError] = useState(null);
  const [stopRequested, setStopRequested] = useState(false);
  const emailsEndRef = useRef(null);
  const mailboxScrollRef = useRef(null);
  const prevEmailCountRef = useRef(0);
  const shouldAutoScrollRef = useRef(true);
  const submittedCheckpointsRef = useRef(new Set());

  useEffect(() => {
    if (emails.length > prevEmailCountRef.current && shouldAutoScrollRef.current) {
      emailsEndRef.current?.scrollIntoView({ behavior: "smooth" });
    }
    prevEmailCountRef.current = emails.length;
  }, [emails.length]);

  useEffect(() => {
    if (!isActive) return;
    onFocusChange?.(buildFocusLabel(directive.topicHint, directive.instruction));
  }, [directive.topicHint, directive.instruction, onFocusChange, isActive]);

  useEffect(() => {
    if (incomingContext) {
      setDirective(p => ({
        ...p,
        instruction: p.instruction
          ? p.instruction + "\n\n" + incomingContext
          : incomingContext
      }));
      onClearIncomingContext?.();
    }
  }, [incomingContext, onClearIncomingContext]);

  useEffect(() => {
    if (!directiveId || !isRunning) return undefined;
    let cancelled = false;
    const poll = async () => {
      try {
        const snapshot = await getDirectiveSnapshot(directiveId);
        if (cancelled) return;
        if (Array.isArray(snapshot.emails) && snapshot.emails.length > 0) {
          setEmails((prev) => mergeEmails(prev, snapshot.emails));
        }
        if (snapshot.pending_steering && !submittedCheckpointsRef.current.has(snapshot.pending_steering.checkpoint_id)) {
          if (!pendingSteering || pendingSteering.checkpoint_id !== snapshot.pending_steering.checkpoint_id) {
            const matchingEmail = [...(snapshot.emails || [])].reverse().find(e => getAwaitingSteering(e)?.checkpoint_id === snapshot.pending_steering.checkpoint_id);
            setPendingSteering({
              ...snapshot.pending_steering,
              subject: matchingEmail?.subject || "Agent Request",
              fromAgent: matchingEmail?.from_agent || "Agent",
            });
            setSteeringAnswers(initialAnswersForSteering(snapshot.pending_steering));
          }
        }
        if (snapshot.stop_requested) {
          setStopRequested(true);
        }
        if (snapshot.done) {
          setIsRunning(false);
        }
      } catch {
        // Ignore polling errors; the live stream may still be healthy.
      }
    };
    poll();
    const intervalId = setInterval(poll, 1000);
    return () => {
      cancelled = true;
      clearInterval(intervalId);
    };
  }, [directiveId, isRunning, pendingSteering]);

  const handleSend = async () => {
    const instruction = directive.instruction.trim();
    const topicHint = directive.topicHint.trim();
    if (!instruction) {
      setError("Instruction is required before opening a workflow.");
      return;
    }

    setIsRunning(true);
    setError(null);
    setEmails([]);
    setPendingSteering(null);
    setSteeringFeedback("");
    setSteeringAnswers({});
    setSteeringBusy(false);
    setSteeringError(null);
    setStopRequested(false);

    const nextDirectiveId = `dir-${Date.now()}`;
    setDirectiveId(nextDirectiveId);

    const payload = {
      directiveId: nextDirectiveId,
      piName: professorName?.trim() || "Professor",
      instruction,
      topicHint,
      phases: Object.keys(directive.phases).filter(k => directive.phases[k]),
      llm: {
        provider: directive.llm.provider,
        url: directive.llm.url,
        model: directive.llm.model,
        timeoutSeconds: 3000.0
      }
    };

    try {
      const saved = localStorage.getItem("literatureAgentConfig");
      if (saved) {
        payload.agentProfiles = {
          literature_agent: {
            pluginConfig: JSON.parse(saved),
            enabled: true
          }
        };
      }
    } catch (e) {
      console.error(e);
    }

    try {
      await streamDirective(payload, (event) => {
        if (event.type === "email") {
          setEmails(prev => [...prev, event.email]);
          const steering = getAwaitingSteering(event.email);
          if (steering && !submittedCheckpointsRef.current.has(steering.checkpoint_id)) {
            setPendingSteering({
              ...steering,
              subject: event.email.subject,
              fromAgent: event.email.from_agent,
            });
            setSteeringAnswers(initialAnswersForSteering(steering));
            setSteeringFeedback("");
            setSteeringBusy(false);
            setSteeringError(null);
          }
        } else if (event.type === "status" && event.phase === "error") {
          setError(event.message);
        } else if (event.type === "steering") {
          setSteeringBusy(false);
        } else if (event.type === "directive_stop_requested") {
          setStopRequested(true);
        }
      });
    } catch (e) {
      setError(e.message);
    } finally {
      setIsRunning(false);
      setSteeringBusy(false);
    }
  };

  const handleTogglePhase = (key) => {
    setDirective(prev => ({ ...prev, phases: { ...prev.phases, [key]: !prev.phases[key] } }));
  };

  const setLlmField = (field, value) => {
    setDirective(prev => ({
      ...prev,
      llm: { ...prev.llm, [field]: value },
    }));
  };

  const handleSteering = async (action) => {
    if (!directiveId || !pendingSteering) return;
    const isAgentRequest = pendingSteering.kind === "agent_request";
    if (isAgentRequest) {
      if (!hasAnswerValues(steeringAnswers)) {
        setSteeringError("Answer the agent question before resuming the workflow.");
        return;
      }
      setSteeringBusy(true);
      setSteeringError(null);

      const answerText = Object.entries(steeringAnswers)
        .filter(([k, v]) => v && v.trim())
        .map(([k, v]) => `${k}: ${v}`)
        .join("\n\n");
      const mockEmail = {
        timestamp: Date.now() / 1000,
        subject: `Response to: ${pendingSteering.title || pendingSteering.subject}`,
        from_agent: professorName?.trim() || "Professor",
        from_addr: "professor@research.local",
        to_name: "Research Orchestrator",
        body: answerText,
        attachments: []
      };

      try {
        await submitDirectiveSteering(
          directiveId,
          buildQuestionResponsePayload(pendingSteering, steeringAnswers),
        );
        setEmails(prev => mergeEmails(prev, [mockEmail]));
        submittedCheckpointsRef.current.add(pendingSteering.checkpoint_id);
        setPendingSteering(null);
        setSteeringAnswers({});
      } catch (e) {
        setSteeringError(e.message);
      } finally {
        setSteeringBusy(false);
      }
      return;
    }
    if (action === "revise" && !steeringFeedback.trim()) {
      setSteeringError("Explain what should change before requesting a revision.");
      return;
    }
    setSteeringBusy(true);
    setSteeringError(null);

    const isRevise = action === "revise";
    const mockEmail = {
      timestamp: Date.now() / 1000,
      subject: isRevise ? `Revision requested: ${pendingSteering.title || pendingSteering.subject}` : `Confirmed: ${pendingSteering.title || pendingSteering.subject}`,
      from_agent: professorName?.trim() || "Professor",
      from_addr: "professor@research.local",
      to_name: "Research Orchestrator",
      body: isRevise ? steeringFeedback.trim() : "Approved. Proceed with the workflow.",
      attachments: []
    };

    try {
      await submitDirectiveSteering(directiveId, {
        checkpointId: pendingSteering.checkpoint_id,
        action,
        feedback: steeringFeedback.trim(),
      });
      setEmails(prev => mergeEmails(prev, [mockEmail]));
      submittedCheckpointsRef.current.add(pendingSteering.checkpoint_id);
      setPendingSteering(null);
      setSteeringFeedback("");
      setSteeringAnswers({});
    } catch (e) {
      setSteeringError(e.message);
    } finally {
      setSteeringBusy(false);
    }
  };

  const steeringEmail = pendingSteering
    ? [...emails].reverse().find((email) => getAwaitingSteering(email)?.checkpoint_id === pendingSteering.checkpoint_id) || null
    : null;

  const answerFields = pendingSteering?.kind === "agent_request"
    ? buildAnswerFields(pendingSteering)
    : [];

  const handleMailboxScroll = () => {
    const node = mailboxScrollRef.current;
    if (!node) return;
    const distanceFromBottom = node.scrollHeight - node.scrollTop - node.clientHeight;
    shouldAutoScrollRef.current = distanceFromBottom < 120;
  };

  const handleStop = async () => {
    if (!directiveId) return;
    setSteeringBusy(true);
    setSteeringError(null);
    try {
      await stopDirective(directiveId, {
        reason: steeringFeedback.trim(),
      });
      setPendingSteering(null);
      setStopRequested(true);
    } catch (e) {
      setSteeringError(e.message);
    } finally {
      setSteeringBusy(false);
    }
  };

  return (
    <div style={{ display: "flex", width: "100%", height: "100%", background: "var(--bg-base)" }}>
      {/* LEFT: Instruction Form */}
      <div style={{ width: 340, borderRight: "1px solid var(--border)", display: "flex", flexDirection: "column", background: "var(--bg-surface)", overflowY: "auto" }}>
        <div style={{ padding: 20 }}>
          <div className="sec-label">Professor Brief</div>

          <div style={{ marginBottom: 15 }}>
            <div className="fld-label">Professor</div>
            <div className="card" style={{ padding: "10px 12px", fontSize: 10, color: "var(--text-secondary)", lineHeight: 1.6 }}>
              {professorName?.trim() || "Professor"}
              <div style={{ color: "var(--text-faint)", marginTop: 4 }}>
                Set in Settings.
              </div>
            </div>
          </div>

          <div style={{ marginBottom: 15 }}>
            <div className="fld-label">Instruction to the Assistants</div>
            <textarea
              className="ta"
              rows={4}
              value={directive.instruction}
              onChange={e => setDirective({ ...directive, instruction: e.target.value })}
              placeholder="Describe the paper, task, or reproduction goal for the assistant team."
            />
          </div>

          <div style={{ marginBottom: 15 }}>
            <div className="fld-label">Topic Dossier (Keywords)</div>
            <input
              className="inp"
              value={directive.topicHint}
              onChange={e => setDirective({ ...directive, topicHint: e.target.value })}
              placeholder="keywords, methods, authors, datasets, phenomena"
            />
          </div>

          <div style={{ marginBottom: 20 }}>
            <div className="fld-label" style={{ marginBottom: 8 }}>Assistant Sequence</div>
            {Object.entries(directive.phases).map(([key, val]) => (
              <div key={key} className="toggle-wrap" style={{ marginBottom: 6 }} onClick={() => handleTogglePhase(key)}>
                <div className={`toggle ${val ? "on" : ""}`} />
                <span style={{ fontSize: 10, color: val ? "var(--text-primary)" : "var(--text-muted)" }}>{key.replace("_", " ")}</span>
              </div>
            ))}
          </div>

          <div className="divider" style={{ margin: "20px 0" }} />

          <div className="sec-label">Assistant LLM</div>
          <LLMConfigBlock
            label="ASSISTANT LLM"
            provider={directive.llm.provider}
            url={directive.llm.url}
            model={directive.llm.model}
            apiKey={directive.llm.apiKey}
            onChange={setLlmField}
          />

          <div style={{ marginTop: 24 }}>
            <button className="btn btn-teal" style={{ width: "100%", justifyContent: "center", padding: "12px" }} onClick={handleSend} disabled={isRunning}>
              {isRunning ? "SENDING INSTRUCTIONS..." : "▶ SEND INSTRUCTIONS"}
            </button>
            {(isRunning || pendingSteering) && (
              <button
                className="btn"
                style={{ width: "100%", justifyContent: "center", padding: "12px", marginTop: 10, borderColor: "var(--accent-red)", color: "var(--accent-red)" }}
                onClick={handleStop}
                disabled={steeringBusy}
              >
                {steeringBusy ? "STOPPING..." : "STOP WORKFLOW"}
              </button>
            )}
            <div style={{ fontSize: 10, color: "var(--text-faint)", marginTop: 8, lineHeight: 1.5 }}>
              Launches the selected assistant sequence and streams progress into the professor mailbox.
            </div>
            {stopRequested && (
              <div style={{ color: "var(--accent-orange)", fontSize: 10, marginTop: 8 }}>
                Stop requested. The workflow will halt at the first safe interruption point.
              </div>
            )}
            {error && <div style={{ color: "var(--accent-red)", fontSize: 10, marginTop: 8 }}>{error}</div>}
          </div>
        </div>
      </div>

      {/* RIGHT: Mailbox */}
      <div style={{ flex: 1, display: "flex", flexDirection: "column", background: "var(--bg-base)", position: "relative" }}>
        <div style={{ padding: "15px 20px", borderBottom: "1px solid var(--border)", background: "var(--bg-surface)", display: "flex", alignItems: "center", justifyContent: "space-between" }}>
          <div className="sec-label" style={{ margin: 0 }}>Professor Mailbox</div>
          <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
            {pendingSteering && (
              <div className="badge" style={{ background: "var(--accent-orange)", color: "#000" }}>
                AWAITING YOUR DECISION
              </div>
            )}
            {stopRequested && (
              <div className="badge" style={{ background: "var(--accent-red)", color: "#fff" }}>
                STOP REQUESTED
              </div>
            )}
            {isRunning && <div className="badge pulse-anim" style={{ background: "var(--accent-teal)", color: "#000" }}>WORKING</div>}
          </div>
        </div>

        <div
          ref={mailboxScrollRef}
          onScroll={handleMailboxScroll}
          style={{ flex: 1, padding: 20, overflowY: "auto", display: "flex", flexDirection: "column", gap: 15 }}
        >
          {emails.map((email) => (
            <EmailCard key={emailKey(email)} email={email} />
          ))}
          {pendingSteering && (
            <div className="card slide-up" style={{ flexShrink: 0, borderColor: "var(--accent-orange)", background: "var(--bg-surface)" }}>
              <div className="sec-label" style={{ marginBottom: 10 }}>Awaiting Your Steering</div>
              <div style={{ fontSize: 11, color: "var(--text-primary)", marginBottom: 6 }}>{pendingSteering.title || pendingSteering.subject}</div>
              <div style={{ fontSize: 10, color: "var(--text-secondary)", lineHeight: 1.6, whiteSpace: "pre-wrap", marginBottom: 12 }}>
                {pendingSteering.prompt}
              </div>
              {steeringEmail && <SteeringSummary email={steeringEmail} />}
              {steeringEmail && (
                <div className="card" style={{ background: "var(--bg-base)", marginBottom: 12 }}>
                  <div className="fld-label" style={{ marginBottom: 8 }}>Assistant Message</div>
                  <MessageBody text={steeringEmail.body || ""} defaultExpanded={true} />
                </div>
              )}
              {steeringEmail?.attachments?.length > 0 && (
                <div style={{ marginBottom: 12 }}>
                  <AttachmentRow attachments={steeringEmail.attachments} />
                </div>
              )}
              {pendingSteering.kind === "agent_request" ? (
                <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
                  {answerFields.map((field) => (
                    <label key={field.key} style={{ display: "flex", flexDirection: "column", gap: 6 }}>
                      <span className="fld-label">{field.label}</span>
                      <textarea
                        className="ta"
                        rows={field.key === "selection" ? 2 : 4}
                        value={steeringAnswers[field.key] || ""}
                        onChange={(event) => setSteeringAnswers((prev) => ({ ...prev, [field.key]: event.target.value }))}
                        placeholder={field.key === "selection" ? "Example: all, 1 and 2, or the target name" : "Answer the assistant question"}
                      />
                    </label>
                  ))}
                </div>
              ) : (
                <textarea
                  className="ta"
                  rows={4}
                  value={steeringFeedback}
                  onChange={(event) => setSteeringFeedback(event.target.value)}
                  placeholder="If you want a revision, describe what should change: correct article, correct model, missing parameter, wrong sweep, etc."
                />
              )}
              <div style={{ display: "flex", gap: 10, marginTop: 12, flexWrap: "wrap" }}>
                {pendingSteering.kind === "agent_request" ? (
                  <button
                    className="btn btn-teal"
                    onClick={() => handleSteering("answer")}
                    disabled={steeringBusy || !hasAnswerValues(steeringAnswers)}
                  >
                    {steeringBusy ? "SENDING..." : "SUBMIT RESPONSE"}
                  </button>
                ) : (
                  <>
                    <button className="btn btn-teal" onClick={() => handleSteering("continue")} disabled={steeringBusy}>
                      {steeringBusy ? "SENDING..." : "CONTINUE"}
                    </button>
                    <button
                      className="btn"
                      onClick={() => handleSteering("revise")}
                      disabled={steeringBusy || !steeringFeedback.trim()}
                      style={{ borderColor: "var(--accent-orange)", color: "var(--accent-orange)" }}
                    >
                      REQUEST REVISION
                    </button>
                  </>
                )}
                <button
                  className="btn"
                  onClick={handleStop}
                  disabled={steeringBusy}
                  style={{ borderColor: "var(--accent-red)", color: "var(--accent-red)" }}
                >
                  STOP WORKFLOW
                </button>
              </div>
              <div style={{ fontSize: 10, color: "var(--text-faint)", marginTop: 8, lineHeight: 1.5 }}>
                {pendingSteering.kind === "agent_request"
                  ? "Your response is sent back to the waiting agent session with the earlier conversation context."
                  : "Revision requests should explain what to change. If you want to stop instead of correcting, use `STOP WORKFLOW`."}
              </div>
              {steeringError && (
                <div style={{ color: "var(--accent-red)", fontSize: 10, marginTop: 10 }}>
                  {steeringError}
                </div>
              )}
            </div>
          )}
          {emails.length === 0 && isRunning && (
            <div className="card slide-up" style={{ flexShrink: 0, background: "var(--bg-surface)" }}>
              <div className="sec-label" style={{ marginBottom: 8 }}>Workflow Active</div>
              <div style={{ fontSize: 10, color: "var(--text-secondary)", lineHeight: 1.7 }}>
                The workflow was launched and the mailbox stream is open. Waiting for the first assistant update.
              </div>
            </div>
          )}
          {emails.length === 0 && !isRunning && (
            <div style={{ margin: "auto", color: "var(--text-ghost)", fontSize: 11, textAlign: "center" }}>
              No mailbox updates yet.<br />Open a workflow to brief the AI research assistants.
            </div>
          )}

          <div ref={emailsEndRef} />
        </div>
      </div>
    </div>
  );
}
