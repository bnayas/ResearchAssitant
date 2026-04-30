# Unit Guide: `frontend`

`frontend` is the React/Vite UI layer. It renders the professor workflow, the standalone subsystem panels, and the mailbox-style artifact view.

## What This Unit Owns

- the top-level application shell,
- the professor-workflow panel,
- standalone literature, simulation, and writer panels,
- attachment and artifact normalization for the UI,
- the shared visual language of the app.

## Main Files

- `project/frontend/src/ResearchHub.jsx` is the top-level application shell.
- `project/frontend/src/DirectivesPanel.jsx` renders the professor workflow and mailbox view.
- `project/frontend/src/LitReviewPanel.jsx` renders direct literature-review interactions.
- `project/frontend/src/SimPanel.jsx` renders direct simulation interactions.
- `project/frontend/src/WriterPanel.jsx` renders writer tasks over collected artifacts.
- `project/frontend/src/professorWorkflowView.js` normalizes mailbox attachments and writer bootstrap data for the UI.
- `project/frontend/src/api.js` wraps the backend HTTP calls.
- `project/frontend/src/style.css` defines the shared UI styling.

## Boundary Rules

- The frontend should stay thin over the API.
- Cross-panel data-shape normalization should stay in helpers, not spread across components.
- If a panel becomes complicated, check whether the real problem is a backend contract mismatch.

## Change Here When

Edit this package when you are:

- changing the visual design,
- improving mailbox or attachment rendering,
- fixing panel-to-panel artifact handoffs,
- adapting the UI to a backend contract change.

## Useful Tests

- `project/tests/PIControl.test.js`
- `project/tests/ProfessorWorkflowView.test.js`
- `npm --prefix project/frontend run build`

## Common Risks

- Assuming a local file path is already a browser-openable link.
- Duplicating backend contract logic across multiple React components.
- Preserving stale UI assumptions such as old artifact field names.
