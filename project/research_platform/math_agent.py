"""
research_platform.math_agent
────────────────────────────
Mixed math service with deterministic SciPy-backed tools and explicit NL adapters.
"""
from __future__ import annotations

import ast
import asyncio
import math
import operator
import re
from typing import Any, Callable, Optional

from reviewer.contract import MathCheckRequest, MathCheckResult
from reviewer.math_tools import MathAgentRegistry

from .assistants.backends import make_service_backend
from .service_contracts import AgentRuntimeProfile, MixedServiceRuntimeConfig

try:
    import numpy as np
except ImportError:  # pragma: no cover
    np = None

try:
    from scipy import integrate, optimize, stats
except ImportError:  # pragma: no cover
    integrate = None
    optimize = None
    stats = None


class MathAgentService:
    """Mixed math service: deterministic tool surface plus NL adapter endpoints."""

    def __init__(self, config: Optional[MixedServiceRuntimeConfig] = None) -> None:
        self._config = config or MixedServiceRuntimeConfig()
        self._nl_backend = None
        if self._config.nl_profile and self._config.nl_profile.enabled:
            self._nl_backend = make_service_backend("math_agent", self._config.nl_profile)

    def evaluate_expression(self, expression: str, *, variables: Optional[dict[str, Any]] = None) -> dict[str, Any]:
        return {"status": "ok", "value": _safe_eval(expression, variables or {})}

    def solve_equation(
        self,
        equation: str,
        *,
        variable: str = "x",
        bracket: Optional[list[float]] = None,
        guess: Optional[float] = None,
    ) -> dict[str, Any]:
        if optimize is None:
            return {"status": "unsupported", "reason": "scipy is not available"}
        lhs, rhs = _split_equation(equation)
        func = _compile_expression(f"({lhs}) - ({rhs})", variable_names={variable})
        bracket = bracket or [-1_000.0, 1_000.0]
        try:
            if guess is not None:
                result = optimize.root_scalar(lambda x: func({variable: x}), x0=float(guess), x1=float(guess) + 1.0)
            else:
                result = optimize.root_scalar(lambda x: func({variable: x}), bracket=(float(bracket[0]), float(bracket[1])))
        except Exception as exc:
            return {"status": "uncertain", "reason": str(exc)}
        if not result.converged:
            return {"status": "uncertain", "reason": "root solver did not converge"}
        return {"status": "ok", "variable": variable, "root": float(result.root), "iterations": int(result.iterations)}

    def solve_ode(
        self,
        derivative: str,
        *,
        t_span: list[float],
        y0: list[float],
        method: str = "RK45",
        eval_points: Optional[list[float]] = None,
    ) -> dict[str, Any]:
        if integrate is None:
            return {"status": "unsupported", "reason": "scipy is not available"}
        func = _compile_ode_expression(derivative)
        try:
            solution = integrate.solve_ivp(
                func,
                (float(t_span[0]), float(t_span[1])),
                [float(v) for v in y0],
                method=method,
                t_eval=[float(v) for v in eval_points] if eval_points else None,
            )
        except Exception as exc:
            return {"status": "uncertain", "reason": str(exc)}
        return {
            "status": "ok" if solution.success else "uncertain",
            "solver": "scipy.integrate.solve_ivp",
            "success": bool(solution.success),
            "message": str(solution.message),
            "t": [float(v) for v in solution.t],
            "y": [[float(item) for item in row] for row in solution.y],
        }

    def integrate_numeric(
        self,
        expression: str,
        *,
        variable: str = "x",
        bounds: list[float],
    ) -> dict[str, Any]:
        if integrate is None:
            return {"status": "unsupported", "reason": "scipy is not available"}
        func = _compile_expression(expression, variable_names={variable})
        result, error = integrate.quad(lambda x: func({variable: x}), float(bounds[0]), float(bounds[1]))
        return {"status": "ok", "value": float(result), "estimated_error": float(error)}

    def differentiate_numeric(
        self,
        expression: str,
        *,
        point: float,
        variable: str = "x",
        dx: float = 1e-6,
    ) -> dict[str, Any]:
        func = _compile_expression(expression, variable_names={variable})
        x0 = float(point)
        step = float(dx)
        derivative = (func({variable: x0 + step}) - func({variable: x0 - step})) / (2.0 * step)
        return {"status": "ok", "value": float(derivative), "point": x0}

    def compute_summary_statistic(self, values: list[float]) -> dict[str, Any]:
        if not values:
            return {"status": "uncertain", "reason": "no values provided"}
        n = len(values)
        mean = sum(values) / n
        variance = sum((value - mean) ** 2 for value in values) / max(n - 1, 1)
        return {
            "status": "ok",
            "count": n,
            "mean": mean,
            "variance": variance,
            "std_dev": math.sqrt(variance),
            "min": min(values),
            "max": max(values),
        }

    def recompute_stat_test(
        self,
        *,
        test: str,
        sample_a: list[float],
        sample_b: Optional[list[float]] = None,
    ) -> dict[str, Any]:
        if stats is None:
            return {"status": "unsupported", "reason": "scipy is not available"}
        normalized = test.strip().lower()
        if normalized in {"ttest_ind", "t_test_ind", "student_t"}:
            if sample_b is None:
                return {"status": "uncertain", "reason": "sample_b is required"}
            stat, p_value = stats.ttest_ind(sample_a, sample_b, equal_var=False)
            return {"status": "ok", "test": "ttest_ind", "statistic": float(stat), "p_value": float(p_value)}
        if normalized in {"ttest_1samp", "one_sample_t"}:
            stat, p_value = stats.ttest_1samp(sample_a, popmean=0.0)
            return {"status": "ok", "test": "ttest_1samp", "statistic": float(stat), "p_value": float(p_value)}
        return {"status": "unsupported", "reason": f"unsupported test: {test}"}

    def check_confidence_interval(
        self,
        *,
        mean: float,
        std_dev: float,
        n: int,
        confidence: float = 0.95,
    ) -> dict[str, Any]:
        if stats is None:
            return {"status": "unsupported", "reason": "scipy is not available"}
        if n <= 1:
            return {"status": "uncertain", "reason": "n must be greater than 1"}
        alpha = 1.0 - float(confidence)
        critical = stats.t.ppf(1.0 - alpha / 2.0, df=n - 1)
        margin = critical * (float(std_dev) / math.sqrt(n))
        return {
            "status": "ok",
            "confidence": float(confidence),
            "lower": float(mean - margin),
            "upper": float(mean + margin),
            "margin": float(margin),
        }

    def verify_numeric_claim(
        self,
        *,
        expression: str,
        claimed_value: float,
        tolerance: float = 1e-4,
        variables: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        computed = _safe_eval(expression, variables or {})
        error = abs(computed - float(claimed_value))
        return {
            "status": "verified" if error <= tolerance else "refuted",
            "computed_value": computed,
            "claimed_value": float(claimed_value),
            "absolute_error": error,
            "tolerance": float(tolerance),
        }

    def verify_statistical_claim(
        self,
        *,
        test: str,
        sample_a: list[float],
        sample_b: Optional[list[float]] = None,
        claimed_p_value: Optional[float] = None,
        tolerance: float = 1e-3,
    ) -> dict[str, Any]:
        outcome = self.recompute_stat_test(test=test, sample_a=sample_a, sample_b=sample_b)
        if outcome.get("status") != "ok" or claimed_p_value is None:
            return outcome
        error = abs(float(outcome["p_value"]) - float(claimed_p_value))
        outcome["claimed_p_value"] = float(claimed_p_value)
        outcome["absolute_error"] = error
        outcome["status"] = "verified" if error <= tolerance else "refuted"
        return outcome

    def verify_differential_equation_claim(
        self,
        *,
        derivative: str,
        t_span: list[float],
        y0: list[float],
        claimed_final_value: float,
        tolerance: float = 1e-4,
    ) -> dict[str, Any]:
        outcome = self.solve_ode(derivative, t_span=t_span, y0=y0)
        if outcome.get("status") not in {"ok", "uncertain"} or not outcome.get("y"):
            return outcome
        final_value = float(outcome["y"][0][-1])
        error = abs(final_value - float(claimed_final_value))
        outcome["claimed_final_value"] = float(claimed_final_value)
        outcome["final_value"] = final_value
        outcome["absolute_error"] = error
        outcome["status"] = "verified" if error <= tolerance else "refuted"
        return outcome

    def solve_from_text(self, question: str) -> dict[str, Any]:
        parsed = self._parse_text(question, mode="solve")
        if parsed["status"] != "ok":
            return parsed
        return self._dispatch_parsed(parsed["operation"], parsed["params"])

    def verify_from_text(self, question: str) -> dict[str, Any]:
        parsed = self._parse_text(question, mode="verify")
        if parsed["status"] != "ok":
            return parsed
        return self._dispatch_parsed(parsed["operation"], parsed["params"])

    def make_registry(self) -> MathAgentRegistry:
        registry = MathAgentRegistry()
        registry.register("AlgebraVerifier", self._unsupported_coroutine("unsupported", "Symbolic algebra is not supported by the SciPy terminal"))
        registry.register("NumericalVerifier", self._numerical_verifier)
        registry.register("StatisticsVerifier", self._statistics_verifier)
        registry.register("ProofChecker", self._unsupported_coroutine("unsupported", "Proof checking is not supported by the SciPy terminal"))
        return registry

    async def _numerical_verifier(self, request: MathCheckRequest) -> MathCheckResult:
        if request.claim_kind == "numerical_result":
            parsed = self.verify_from_text(request.claim_text)
            return _to_math_check_result(request, parsed)
        if request.claim_kind == "differential_equation":
            parsed = self.verify_differential_equation_claim(
                derivative=request.context or request.claim_text,
                t_span=[0.0, 1.0],
                y0=[0.0],
                claimed_final_value=_extract_first_float(request.claim_text, default=0.0),
            )
            return _to_math_check_result(request, parsed)
        return MathCheckResult(
            request_id=request.request_id,
            status="unsupported",
            verdict=f"Claim kind {request.claim_kind!r} is not supported by NumericalVerifier.",
            confidence=0.0,
        )

    async def _statistics_verifier(self, request: MathCheckRequest) -> MathCheckResult:
        parsed = self.verify_from_text(request.claim_text)
        return _to_math_check_result(request, parsed)

    def _unsupported_coroutine(self, status: str, verdict: str) -> Callable[[MathCheckRequest], Any]:
        async def _fn(request: MathCheckRequest) -> MathCheckResult:
            return MathCheckResult(
                request_id=request.request_id,
                status=status,
                verdict=verdict,
                confidence=0.0,
            )
        return _fn

    def _parse_text(self, question: str, *, mode: str) -> dict[str, Any]:
        text = str(question or "").strip()
        if not text:
            return {"status": "uncertain", "reason": "empty question"}
        if self._nl_backend is not None:
            structured = self._parse_with_llm(text, mode=mode)
            if structured is not None:
                return structured
        lower = text.lower()
        if "integral" in lower:
            match = re.search(r"integral of (.+?) from ([^ ]+) to ([^ ]+)", lower)
            if match:
                return {
                    "status": "ok",
                    "operation": "integrate_numeric",
                    "params": {
                        "expression": match.group(1),
                        "bounds": [float(match.group(2)), float(match.group(3))],
                    },
                }
        if "derivative" in lower:
            match = re.search(r"derivative of (.+?) at ([^ ]+)", lower)
            if match:
                return {
                    "status": "ok",
                    "operation": "differentiate_numeric",
                    "params": {
                        "expression": match.group(1),
                        "point": float(match.group(2)),
                    },
                }
        if "solve" in lower and "=" in text:
            return {
                "status": "ok",
                "operation": "solve_equation",
                "params": {"equation": text.split("solve", 1)[-1].strip()},
            }
        if "p-value" in lower or "p value" in lower:
            return {"status": "unsupported", "reason": "natural-language statistical parsing is limited; pass structured samples"}
        if mode == "verify":
            numeric = re.search(r"(.+?)\s*(?:=|equals|is)\s*(-?\d+(?:\.\d+)?)", text)
            if numeric:
                return {
                    "status": "ok",
                    "operation": "verify_numeric_claim",
                    "params": {
                        "expression": numeric.group(1).replace("equals", "").replace("is", "").strip(),
                        "claimed_value": float(numeric.group(2)),
                    },
                }
        return {
            "status": "ok",
            "operation": "evaluate_expression",
            "params": {"expression": text},
        }

    def _parse_with_llm(self, text: str, *, mode: str) -> Optional[dict[str, Any]]:
        try:
            raw = self._nl_backend.complete(
                system=(
                    "Parse natural-language math requests into JSON.\n"
                    "Return ONLY JSON with keys: status, operation, params, reason.\n"
                    "operation must be one of evaluate_expression, solve_equation, solve_ode, "
                    "integrate_numeric, differentiate_numeric, verify_numeric_claim, "
                    "verify_statistical_claim, verify_differential_equation_claim."
                ),
                messages=[{"role": "user", "content": f"mode={mode}\nquestion={text}"}],
                temperature=0.0,
            )
        except Exception:
            return None
        try:
            import json
            payload = json.loads(raw)
        except Exception:
            return None
        if payload.get("status") not in {"ok", "unsupported", "uncertain"}:
            return None
        if payload.get("status") == "ok" and not payload.get("operation"):
            return None
        return payload

    def _dispatch_parsed(self, operation: str, params: dict[str, Any]) -> dict[str, Any]:
        dispatch = {
            "evaluate_expression": self.evaluate_expression,
            "solve_equation": self.solve_equation,
            "solve_ode": self.solve_ode,
            "integrate_numeric": self.integrate_numeric,
            "differentiate_numeric": self.differentiate_numeric,
            "verify_numeric_claim": self.verify_numeric_claim,
            "verify_statistical_claim": self.verify_statistical_claim,
            "verify_differential_equation_claim": self.verify_differential_equation_claim,
        }
        fn = dispatch.get(operation)
        if fn is None:
            return {"status": "unsupported", "reason": f"unsupported operation: {operation}"}
        return fn(**params)


def _extract_first_float(text: str, *, default: float = 0.0) -> float:
    match = re.search(r"-?\d+(?:\.\d+)?", text)
    if not match:
        return default
    return float(match.group(0))


def _to_math_check_result(request: MathCheckRequest, payload: dict[str, Any]) -> MathCheckResult:
    status = str(payload.get("status") or "uncertain")
    verdict = payload.get("reason") or payload.get("message") or str(payload)
    if status == "verified":
        verdict = verdict or "Claim verified."
    elif status == "refuted":
        verdict = verdict or "Claim refuted."
    elif status == "ok":
        status = "verified"
        verdict = verdict or "Computed result is available."
    if status not in {"verified", "refuted", "uncertain", "timeout", "unsupported"}:
        status = "uncertain"
    correction = None
    if "computed_value" in payload:
        correction = str(payload["computed_value"])
    elif "final_value" in payload:
        correction = str(payload["final_value"])
    confidence = 1.0 if status == "verified" else 0.0 if status == "unsupported" else 0.6
    return MathCheckResult(
        request_id=request.request_id,
        status=status,
        verdict=str(verdict),
        correction=correction,
        confidence=confidence,
    )


def _split_equation(equation: str) -> tuple[str, str]:
    if "=" not in equation:
        return equation, "0"
    lhs, rhs = equation.split("=", 1)
    return lhs.strip(), rhs.strip()


_ALLOWED_BINOPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.Mod: operator.mod,
}
_ALLOWED_UNARY = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}
_ALLOWED_FUNCS = {
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "exp": math.exp,
    "sqrt": math.sqrt,
    "log": math.log,
    "abs": abs,
}


def _safe_eval(expression: str, variables: dict[str, Any]) -> float:
    return float(_eval_node(ast.parse(expression, mode="eval").body, variables))


def _compile_expression(expression: str, *, variable_names: set[str]) -> Callable[[dict[str, Any]], float]:
    tree = ast.parse(expression, mode="eval")

    def _fn(values: dict[str, Any]) -> float:
        env = {name: values.get(name, 0.0) for name in variable_names}
        return float(_eval_node(tree.body, env))

    return _fn


def _compile_ode_expression(expression: str) -> Callable[[float, list[float]], list[float]]:
    text = expression.strip()
    if text.startswith("[") and text.endswith("]"):
        pieces = [piece.strip() for piece in text[1:-1].split(",") if piece.strip()]
    else:
        pieces = [text]
    funcs = [_compile_expression(piece, variable_names={"t", "y", "y0", "y1", "y2"}) for piece in pieces]

    def _fn(t: float, y: list[float]) -> list[float]:
        env = {"t": t, "y": y[0] if y else 0.0}
        for index, value in enumerate(y):
            env[f"y{index}"] = value
        return [float(fn(env)) for fn in funcs]

    return _fn


def _eval_node(node: ast.AST, variables: dict[str, Any]) -> Any:
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Name):
        if node.id in variables:
            return variables[node.id]
        if node.id == "pi":
            return math.pi
        if node.id == "e":
            return math.e
        raise ValueError(f"unknown variable: {node.id}")
    if isinstance(node, ast.BinOp) and type(node.op) in _ALLOWED_BINOPS:
        return _ALLOWED_BINOPS[type(node.op)](_eval_node(node.left, variables), _eval_node(node.right, variables))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _ALLOWED_UNARY:
        return _ALLOWED_UNARY[type(node.op)](_eval_node(node.operand, variables))
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in _ALLOWED_FUNCS:
        args = [_eval_node(arg, variables) for arg in node.args]
        return _ALLOWED_FUNCS[node.func.id](*args)
    if isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name) and node.value.id in variables:
        value = variables[node.value.id]
        index = _eval_node(node.slice, variables) if not isinstance(node.slice, ast.Slice) else None
        return value[index]
    if isinstance(node, ast.List):
        return [_eval_node(item, variables) for item in node.elts]
    if isinstance(node, ast.Tuple):
        return tuple(_eval_node(item, variables) for item in node.elts)
    raise ValueError(f"unsupported expression: {ast.dump(node)}")
