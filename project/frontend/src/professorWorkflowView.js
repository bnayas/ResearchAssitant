const LITERATURE_ASSISTANT = "literature_reviewer";
const CODING_ASSISTANT = "coding_agent";

function truncate(text, limit = 160) {
  const compact = String(text || "").replace(/\s+/g, " ").trim();
  if (!compact) return "";
  return compact.length <= limit ? compact : `${compact.slice(0, limit - 1).trimEnd()}…`;
}

export function attachmentHref(attachment) {
  if (!attachment) return null;
  if (attachment.url) return attachment.url;
  if (attachment.path) {
    return attachment.path.startsWith("file://")
      ? attachment.path
      : `file://${attachment.path}`;
  }
  return null;
}

export function getAwaitingSteering(email) {
  const steering = email?.metadata?.steering;
  if (!steering || steering.state !== "awaiting_pi") return null;
  return steering;
}

export function parseJsonAttachment(attachment) {
  if (!attachment?.content || attachment?.mime_type !== "application/json") return null;
  try {
    return JSON.parse(attachment.content);
  } catch {
    return null;
  }
}

export function buildWriterArtifactCatalog(litArtifact, simArtifact) {
  const catalog = [];
  const literature = litArtifact?.artifact;
  if (literature) {
    const papers = (literature.papers || []).filter(p => p.in_scope !== false);
    catalog.push({
      artifactId: "lit-synthesis",
      assistant: LITERATURE_ASSISTANT,
      kind: "literature_synthesis",
      title: "Literature Synthesis",
      summary: truncate(literature.synthesis || `${papers.length} related paper(s)`),
      content: literature.synthesis || "",
      metadata: {
        acceptedCount: papers.length,
        audit: litArtifact?.audit || {},
      },
    });
    papers.slice(0, 8).forEach((paper, index) => {
      catalog.push({
        artifactId: `lit-paper-${index + 1}`,
        assistant: LITERATURE_ASSISTANT,
        kind: "literature_paper",
        title: paper.title || `Paper ${index + 1}`,
        summary: truncate(`${paper.year || "?"} · ${paper.abstract || ""}`),
        url: paper.url || null,
        content: `${paper.title || ""}\n${paper.abstract || ""}`.trim(),
        metadata: { ...paper },
      });
    });
  }

  if (simArtifact?.full_summary) {
    catalog.push({
      artifactId: "sim-full-summary",
      assistant: CODING_ASSISTANT,
      kind: "simulation_summary",
      title: "Simulation Summary",
      summary: truncate(simArtifact?.message || simArtifact?.full_summary?.rendered || "Simulation summary"),
      content: JSON.stringify(simArtifact.full_summary, null, 2),
      metadata: simArtifact.full_summary,
    });
  }

  if (simArtifact?.full_analysis) {
    catalog.push({
      artifactId: "sim-full-analysis",
      assistant: CODING_ASSISTANT,
      kind: "simulation_analysis",
      title: "Simulation Analysis",
      summary: truncate(simArtifact?.full_analysis?.rendered || simArtifact?.message || "Simulation analysis"),
      content: JSON.stringify(simArtifact.full_analysis, null, 2),
      metadata: simArtifact.full_analysis,
    });
  }

  if (simArtifact?.artifacts?.script_path || simArtifact?.artifacts?.scriptPath) {
    catalog.push({
      artifactId: "sim-script",
      assistant: CODING_ASSISTANT,
      kind: "simulation_script",
      title: "Simulation Script",
      summary: truncate(simArtifact?.artifacts?.script_path || simArtifact?.artifacts?.scriptPath),
      path: simArtifact?.artifacts?.script_path || simArtifact?.artifacts?.scriptPath,
      metadata: simArtifact?.artifacts || {},
    });
  }

  return catalog;
}

export function buildWriterArtifactContext(litArtifact, simArtifact) {
  const sections = [];
  const literature = litArtifact?.artifact;
  if (literature) {
    const papers = (literature.papers || []).filter(p => p.in_scope !== false);
    sections.push(
      `## Literature Review\n${literature.synthesis || "(no synthesis)"}\n\n### Accepted Papers\n${
        papers.length
          ? papers.map(p => `- ${p.title} (${p.year || "?"}) — ${truncate(p.abstract || "", 120)}`).join("\n")
          : "- none"
      }`
    );
  }

  if (simArtifact?.full_summary || simArtifact?.full_analysis) {
    const summary = simArtifact?.full_summary?.rendered || simArtifact?.message || "(no simulation summary)";
    const analysis = simArtifact?.full_analysis?.rendered || "";
    sections.push(`## Simulation\n${summary}${analysis ? `\n\n### Analysis\n${analysis}` : ""}`);
  }

  return sections.join("\n\n").trim();
}
