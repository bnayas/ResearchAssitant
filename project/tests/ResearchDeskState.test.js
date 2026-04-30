import {
  DEFAULT_FOCUS_LABEL,
  buildDeskHeading,
  buildFocusLabel,
  loadDeskSettings,
  saveDeskSettings,
} from "../frontend/src/researchDeskState.js";

let passed = 0;
let failed = 0;

function test(name, fn) {
  try {
    fn();
    console.log(`  ✓ ${name}`);
    passed++;
  } catch (error) {
    console.log(`  ✗ ${name}: ${error.message}`);
    failed++;
  }
}

function assertEqual(actual, expected, message) {
  if (actual !== expected) {
    throw new Error(message || `Expected ${JSON.stringify(expected)}, got ${JSON.stringify(actual)}`);
  }
}

function makeStorage() {
  const store = new Map();
  return {
    getItem(key) {
      return store.has(key) ? store.get(key) : null;
    },
    setItem(key, value) {
      store.set(key, value);
    },
  };
}

console.log("\nResearchDeskState");

test("buildDeskHeading falls back to generic heading", () => {
  assertEqual(buildDeskHeading(""), "Research Desk");
});

test("buildDeskHeading uses professor name when available", () => {
  assertEqual(buildDeskHeading("Prof. Ada Lovelace"), "Prof. Ada Lovelace's Research Desk");
});

test("buildFocusLabel compacts comma separated topics", () => {
  assertEqual(
    buildFocusLabel("stochasticity, neutral dynamics, abundance distribution, extra"),
    "stochasticity · neutral dynamics · abundance distribution",
  );
});

test("buildFocusLabel falls back to default label", () => {
  assertEqual(buildFocusLabel("   "), DEFAULT_FOCUS_LABEL);
});

test("desk settings round trip through storage", () => {
  const storage = makeStorage();
  saveDeskSettings({ professorName: " Prof. Emmy Noether " }, storage);
  const loaded = loadDeskSettings(storage);
  assertEqual(loaded.professorName, "Prof. Emmy Noether");
});

console.log(`\n${passed} passed, ${failed} failed.`);
if (failed > 0) process.exit(1);
