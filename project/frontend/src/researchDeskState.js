const STORAGE_KEY = "researchDeskSettings";
export const DEFAULT_FOCUS_LABEL = "No active research focus";

function compactText(value) {
  return String(value || "").replace(/\s+/g, " ").trim();
}

export function normalizeProfessorName(value) {
  return compactText(value);
}

export function buildDeskHeading(professorName) {
  const name = normalizeProfessorName(professorName);
  return name ? `${name}'s Research Desk` : "Research Desk";
}

export function buildFocusLabel(...values) {
  for (const value of values) {
    const text = compactText(value);
    if (!text) continue;

    const parts = text
      .split(",")
      .map((part) => compactText(part))
      .filter(Boolean);
    if (parts.length >= 2) {
      return parts.slice(0, 3).join(" · ");
    }

    const sentence = compactText(text.split(/[\n.!?]/)[0] || text);
    if (!sentence) continue;
    return sentence.length <= 72 ? sentence : `${sentence.slice(0, 69).trimEnd()}…`;
  }
  return DEFAULT_FOCUS_LABEL;
}

export function loadDeskSettings(storage = globalThis?.localStorage ?? null) {
  if (!storage) return { professorName: "" };
  try {
    const raw = storage.getItem(STORAGE_KEY);
    if (!raw) return { professorName: "" };
    const parsed = JSON.parse(raw);
    return {
      professorName: normalizeProfessorName(parsed?.professorName),
    };
  } catch {
    return { professorName: "" };
  }
}

export function saveDeskSettings(settings, storage = globalThis?.localStorage ?? null) {
  const normalized = {
    professorName: normalizeProfessorName(settings?.professorName),
  };
  if (storage) {
    try {
      storage.setItem(STORAGE_KEY, JSON.stringify(normalized));
    } catch {
      // Ignore persistence failures and keep the in-memory settings.
    }
  }
  return normalized;
}
