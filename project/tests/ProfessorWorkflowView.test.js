import {
  attachmentHref,
  buildWriterArtifactCatalog,
  buildWriterArtifactContext,
  getAwaitingSteering,
} from "../frontend/src/professorWorkflowView.js";

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

function assert(condition, message = "assertion failed") {
  if (!condition) throw new Error(message);
}

function assertEqual(actual, expected, message) {
  if (actual !== expected) {
    throw new Error(message || `Expected ${JSON.stringify(expected)}, got ${JSON.stringify(actual)}`);
  }
}

const literatureResult = {
  artifact: {
    synthesis: "Related neutral theory papers.",
    papers: [
      {
        title: "Theory of time-averaged neutral dynamics with environmental stochasticity",
        abstract: "Neutral model with environmental noise.",
        year: 2017,
        url: "https://arxiv.org/abs/1711.11332v3",
        in_scope: true,
      },
    ],
  },
  audit: { passed: true },
};

const simulationResult = {
  message: "Full sweep finished.",
  full_summary: { rendered: "1 successful run." },
  full_analysis: { rendered: "Verdict: ok." },
  artifacts: { script_path: "/tmp/sim.py" },
};

console.log("\nProfessorWorkflowView");

test("attachmentHref prefers web URLs", () => {
  assertEqual(attachmentHref({ url: "https://example.com/paper" }), "https://example.com/paper");
});

test("attachmentHref converts local paths to file URLs", () => {
  assertEqual(attachmentHref({ path: "/tmp/report.md" }), "file:///tmp/report.md");
});

test("getAwaitingSteering extracts pending steering metadata", () => {
  const steering = getAwaitingSteering({
    metadata: {
      steering: {
        checkpoint_id: "dir-1:find_article",
        state: "awaiting_pi",
        title: "Review article",
      },
    },
  });
  assertEqual(steering.checkpoint_id, "dir-1:find_article");
});

test("buildWriterArtifactCatalog includes literature and simulation sources", () => {
  const catalog = buildWriterArtifactCatalog(literatureResult, simulationResult);
  const ids = catalog.map(item => item.artifactId);
  assert(ids.includes("lit-synthesis"), "missing literature synthesis artifact");
  assert(ids.includes("sim-full-summary"), "missing simulation summary artifact");
  assert(ids.includes("sim-full-analysis"), "missing simulation analysis artifact");
});

test("buildWriterArtifactContext stitches literature and simulation sections", () => {
  const context = buildWriterArtifactContext(literatureResult, simulationResult);
  assert(context.includes("## Literature Review"), "missing literature section");
  assert(context.includes("## Simulation"), "missing simulation section");
  assert(context.includes("Theory of time-averaged neutral dynamics"), "missing paper detail");
});

console.log(`\n${passed} passed, ${failed} failed.`);
if (failed > 0) process.exit(1);
