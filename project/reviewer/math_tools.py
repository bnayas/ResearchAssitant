"""
reviewer/math_tools.py
=======================
Mathematical verification agents — API contracts and stub implementations.

This module specifies the FULL INTERFACE CONTRACT that each math verification
agent must satisfy.  None of the agents are implemented here.  The stubs
raise NotImplementedError or return mock results so the reviewer can be
developed and tested independently.

──────────────────────────────────────────────────────────────────────────────
EXPECTED AGENT API CONTRACTS
──────────────────────────────────────────────────────────────────────────────

Every math agent exposes a single async coroutine:

    async def check(request: MathCheckRequest) -> MathCheckResult

Agents are registered in MathAgentRegistry and selected by claim_kind.

──────────────────────────────────────────────────────────────────────────────
AGENT 1 – AlgebraVerifier
──────────────────────────────────────────────────────────────────────────────
Handles: algebraic_identity, bound_or_complexity, optimization_claim

Contract
--------
Input  : MathCheckRequest where claim_text contains a symbolic algebraic
         assertion expressed in LaTeX or Python-style notation.
         context provides the surrounding paragraph for disambiguation.

Output : MathCheckResult
         status="verified"    — CAS confirms the identity holds symbolically.
         status="refuted"     — CAS found a counterexample or disproof.
                                counterexample set to witness string.
                                correction set to corrected LaTeX.
         status="uncertain"   — claim is contingent on assumptions not stated
                                in the article text.
         status="unsupported" — claim uses notation or domain beyond agent scope.

Implementation guidance (not enforced here)
-------------------------------------------
  - Preferred CAS: SymPy for pure algebra; SageMath for number theory.
  - Parse LaTeX using latex2sympy2 or antlr4 grammar.
  - For complexity claims: verify via asymptotic analysis, not simulation.
  - For optimization: verify KKT conditions or convexity argument.
  - timeout_seconds from the request MUST be respected (asyncio.wait_for).
  - confidence should reflect symbolic certainty: 1.0 for CAS-verified,
    0.8 for heuristic, 0.5 for parse-only checks.

──────────────────────────────────────────────────────────────────────────────
AGENT 2 – NumericalVerifier
──────────────────────────────────────────────────────────────────────────────
Handles: numerical_result, differential_equation

Contract
--------
Input  : MathCheckRequest where claim_text states a numerical result or ODE
         solution.  context must include the equation/system being solved.

Output : MathCheckResult
         status="verified"    — numerical solver reproduces the claimed result
                                within tolerance (relative error < 1e-4).
         status="refuted"     — solver yields a materially different result.
                                correction contains the solver's result.
         status="uncertain"   — insufficient parameter info in article text.
         status="timeout"     — solver did not converge within timeout_seconds.

Implementation guidance
-----------------------
  - ODE: scipy.integrate.solve_ivp or Julia DifferentialEquations.jl.
  - NumericalResult: extract constants from claim_text; evaluate expression;
    compare to the stated value.
  - For PDEs: finite-difference or finite-element stub (mark unsupported if
    domain is not simple box geometry).
  - Report solver name and tolerance used in verdict.
  - Do NOT extrapolate beyond the article's stated domain.

──────────────────────────────────────────────────────────────────────────────
AGENT 3 – StatisticsVerifier
──────────────────────────────────────────────────────────────────────────────
Handles: statistical_claim

Contract
--------
Input  : MathCheckRequest where claim_text contains a statistical assertion
         such as a p-value, effect size, confidence interval, or power claim.
         context should include sample sizes and test description.

Output : MathCheckResult
         status="verified"    — re-computation from stated parameters matches
                                claimed value within rounding tolerance.
         status="refuted"     — re-computation yields materially different
                                value.  counterexample = recomputed value.
         status="uncertain"   — parameters (n, distribution, correction) not
                                fully specified in the article text.
         status="unsupported" — exotic test not in agent's library.

Implementation guidance
-----------------------
  - Use scipy.stats, pingouin, or statsmodels.
  - Extract: test name, n (per group), test statistic, df, claimed p-value.
  - Re-derive p from (test_statistic, df) and compare to stated p-value.
  - Check effect size formulas: Cohen's d, η², Cohen's f, Hedges' g.
  - For multiple comparisons: verify that Bonferroni / FDR correction was
    applied if the article claims it was.
  - confidence: 1.0 if fully parameterised; 0.6 if n inferred from context.

──────────────────────────────────────────────────────────────────────────────
AGENT 4 – ProofChecker
──────────────────────────────────────────────────────────────────────────────
Handles: proof_step

Contract
--------
Input  : MathCheckRequest where claim_text is one inference step in a proof.
         context contains the preceding and following proof steps.

Output : MathCheckResult
         status="verified"    — step is deductively valid given stated premises.
         status="refuted"     — a gap or error was found; counterexample
                                describes the invalid step.
         status="uncertain"   — step relies on a lemma not stated in the text.
         status="unsupported" — requires interactive theorem prover interaction.

Implementation guidance
-----------------------
  - Light path: parse step as natural-deduction rule; check via Z3 or Lean.
  - For induction: verify base case and inductive step separately.
  - For reductions: verify each direction of the bidirectional claim.
  - If Lean / Coq is available, attempt to typecheck the proof term.
  - confidence: 0.9 for Z3-verified; 0.6 for pattern-match only.
  - Do not hallucinate lemma proofs.  status="uncertain" is preferable.

──────────────────────────────────────────────────────────────────────────────
SHARED CONTRACT REQUIREMENTS (all agents)
──────────────────────────────────────────────────────────────────────────────
1. request_id is preserved verbatim in the MathCheckResult.
2. timeout_seconds from the request must be respected.
3. confidence ∈ [0, 1]; do not return confidence=1.0 for "uncertain".
4. verdict is a human-readable string suitable for inclusion in a review letter.
5. correction must be valid LaTeX if the claim_text used LaTeX notation.
6. Agents must be stateless between requests (no session state across calls).
7. Agents must not access external databases or the internet.
8. Agents must not consult the upstream simulation/analysis artifacts;
   they check only the mathematical claim as written in the article text.
──────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import asyncio
from typing import Callable, Dict, Optional

from .contract import MathCheckRequest, MathCheckResult, MathClaimKind

# ── Agent callable type ───────────────────────────────────────────────────────

MathAgentFn = Callable[[MathCheckRequest], "asyncio.Coroutine[None, None, MathCheckResult]"]

# ── Routing table: claim kind → preferred agent ───────────────────────────────

CLAIM_KIND_TO_AGENT: Dict[MathClaimKind, str] = {
    "algebraic_identity":    "AlgebraVerifier",
    "bound_or_complexity":   "AlgebraVerifier",
    "optimization_claim":    "AlgebraVerifier",
    "numerical_result":      "NumericalVerifier",
    "differential_equation": "NumericalVerifier",
    "statistical_claim":     "StatisticsVerifier",
    "proof_step":            "ProofChecker",
}

# ── Registry ──────────────────────────────────────────────────────────────────


class MathAgentRegistry:
    """
    Resolves a MathCheckRequest to the correct agent and dispatches it.

    At runtime the orchestrator calls registry.register(name, fn) for
    each available math agent.  The reviewer uses registry.dispatch(request).
    If no agent is registered for a claim kind, returns status="unsupported".
    """

    def __init__(self) -> None:
        self._agents: Dict[str, MathAgentFn] = {}

    def register(self, agent_name: str, fn: MathAgentFn) -> None:
        """Bind an agent name to its async check function."""
        self._agents[agent_name] = fn

    def is_registered(self, agent_name: str) -> bool:
        return agent_name in self._agents

    def registered_names(self) -> list[str]:
        return list(self._agents.keys())

    async def dispatch(self, request: MathCheckRequest) -> MathCheckResult:
        """
        Route a MathCheckRequest to the appropriate agent.
        Falls back to "unsupported" if no agent handles this claim kind.
        """
        agent_name = CLAIM_KIND_TO_AGENT.get(request.claim_kind)
        if agent_name is None or agent_name not in self._agents:
            return MathCheckResult(
                request_id=request.request_id,
                status="unsupported",
                verdict=(
                    f"No agent registered for claim kind '{request.claim_kind}'. "
                    f"Available: {self.registered_names()}"
                ),
                confidence=0.0,
            )

        fn = self._agents[agent_name]
        try:
            result = await asyncio.wait_for(fn(request), timeout=request.timeout_seconds)
            return result
        except asyncio.TimeoutError:
            return MathCheckResult(
                request_id=request.request_id,
                status="timeout",
                verdict=f"Agent '{agent_name}' timed out after {request.timeout_seconds}s",
                confidence=0.0,
            )
        except Exception as exc:
            return MathCheckResult(
                request_id=request.request_id,
                status="uncertain",
                verdict=f"Agent '{agent_name}' raised an exception: {exc}",
                confidence=0.0,
            )

    async def dispatch_all(
        self, requests: list[MathCheckRequest]
    ) -> list[MathCheckResult]:
        """Dispatch all requests concurrently and return results in order."""
        if not requests:
            return []
        tasks = [self.dispatch(req) for req in requests]
        return list(await asyncio.gather(*tasks))


# ── Stub implementations (NOT FOR PRODUCTION — develop and replace) ───────────

async def _stub_algebra_verifier(request: MathCheckRequest) -> MathCheckResult:
    """
    Stub: returns uncertain for every claim.
    Replace with SymPy / SageMath integration.

    Expected production implementation
    -----------------------------------
    from sympy import sympify, simplify, latex
    lhs, rhs = parse_latex_equation(request.claim_text)
    diff = simplify(lhs - rhs)
    if diff == 0:
        status, verdict = "verified", f"Identity holds: simplify(lhs-rhs)=0"
    else:
        status, verdict = "refuted", f"simplify(lhs-rhs)={diff}"
    ...
    """
    await asyncio.sleep(0)   # yield to event loop
    return MathCheckResult(
        request_id=request.request_id,
        status="uncertain",
        verdict="[STUB] AlgebraVerifier not implemented. Replace with SymPy integration.",
        confidence=0.0,
    )


async def _stub_numerical_verifier(request: MathCheckRequest) -> MathCheckResult:
    """
    Stub: returns uncertain.
    Replace with scipy / Julia DifferentialEquations integration.

    Expected production implementation
    -----------------------------------
    from scipy.integrate import solve_ivp
    ode_system = parse_ode(request.context)
    sol = solve_ivp(ode_system, t_span, y0, dense_output=True)
    claimed_val = extract_numerical_claim(request.claim_text)
    computed_val = sol.y[state_idx][-1]
    err = abs(computed_val - claimed_val) / (abs(claimed_val) + 1e-12)
    ...
    """
    await asyncio.sleep(0)
    return MathCheckResult(
        request_id=request.request_id,
        status="uncertain",
        verdict="[STUB] NumericalVerifier not implemented. Replace with scipy/Julia integration.",
        confidence=0.0,
    )


async def _stub_statistics_verifier(request: MathCheckRequest) -> MathCheckResult:
    """
    Stub: returns uncertain.
    Replace with scipy.stats / pingouin integration.

    Expected production implementation
    -----------------------------------
    from scipy import stats
    test_name, n, stat_val, claimed_p = extract_stat_claim(request.claim_text)
    _, computed_p = getattr(stats, test_name)(statistic=stat_val, df=n-1)
    if abs(computed_p - claimed_p) < 0.001:
        status = "verified"
    else:
        status = "refuted"
        counterexample = f"Recomputed p={computed_p:.4f}, claimed p={claimed_p}"
    ...
    """
    await asyncio.sleep(0)
    return MathCheckResult(
        request_id=request.request_id,
        status="uncertain",
        verdict="[STUB] StatisticsVerifier not implemented. Replace with scipy.stats integration.",
        confidence=0.0,
    )


async def _stub_proof_checker(request: MathCheckRequest) -> MathCheckResult:
    """
    Stub: returns uncertain.
    Replace with Z3 / Lean integration.

    Expected production implementation
    -----------------------------------
    from z3 import Solver, Not, parse_smt2_string
    solver = Solver()
    premises = parse_premises(request.context)
    conclusion = parse_step(request.claim_text)
    solver.add(premises)
    solver.add(Not(conclusion))
    result = solver.check()
    if result == z3.unsat:
        status = "verified"
    elif result == z3.sat:
        status = "refuted"
        counterexample = str(solver.model())
    ...
    """
    await asyncio.sleep(0)
    return MathCheckResult(
        request_id=request.request_id,
        status="uncertain",
        verdict="[STUB] ProofChecker not implemented. Replace with Z3/Lean integration.",
        confidence=0.0,
    )


def make_stub_registry() -> MathAgentRegistry:
    """
    Returns a MathAgentRegistry pre-populated with all four stub agents.
    Use for development / testing when real math agents are not available.
    """
    registry = MathAgentRegistry()
    registry.register("AlgebraVerifier",    _stub_algebra_verifier)
    registry.register("NumericalVerifier",  _stub_numerical_verifier)
    registry.register("StatisticsVerifier", _stub_statistics_verifier)
    registry.register("ProofChecker",       _stub_proof_checker)
    return registry
