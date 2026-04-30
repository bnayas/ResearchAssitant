"""
research_platform.runtime_assemble
──────────────────────────────────
Deterministic runtime assembly and execution service for mediated simulations.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
import json
from pathlib import Path
from typing import Any, Optional

from sim_tool.codegen import generate_notebook, generate_script
from sim_tool.contract import BugTicket, LaunchResult, RunOutcome, RunResult, RunSummary
from sim_tool.launcher import LaunchRuntimeConfig, Launcher
from sim_tool.models import SimulationSpec, StoppingCondition, Variable, VariableKind
from sim_tool.runner import build_sweep_configs

from .contracts import ArtifactRef, AssistantId
from .registry import ArtifactRegistry
from .service_contracts import RuntimeDiagnostics, ServiceRuntimeConfig


@dataclass
class RuntimeBundle:
    session_id: str
    script_path: str
    notebook_path: str
    spec_artifact_id: str
    script_artifact_id: str
    notebook_artifact_id: str


class RuntimeAssembleService:
    """Deterministic terminal service used by the mediated simulation stack."""

    def __init__(
        self,
        registry: ArtifactRegistry,
        *,
        config: Optional[ServiceRuntimeConfig] = None,
        launcher: Optional[Launcher] = None,
        output_root: Optional[str | Path] = None,
    ) -> None:
        self._registry = registry
        self._config = config or ServiceRuntimeConfig()
        self._output_root = Path(output_root or self._config.settings.get("output_root") or "./simulations")
        runtime = LaunchRuntimeConfig(
            primary_mode=str(self._config.settings.get("primary_mode", "docker")),
            fallback_mode=str(self._config.settings.get("fallback_mode", "process")),
            docker_binary=str(self._config.settings.get("docker_binary", "docker")),
            docker_image=str(
                self._config.settings.get("docker_image", "research-platform-launcher:latest")
            ),
            docker_workdir=str(self._config.settings.get("docker_workdir", "/workspace")),
            docker_auto_pull=bool(self._config.settings.get("docker_auto_pull", False)),
        )
        self._launcher = launcher or Launcher(runtime=runtime)

    def materialize_runtime(
        self,
        *,
        session_id: str,
        spec_payload: dict[str, Any],
        output_dir: Optional[str] = None,
        assistant: str = AssistantId.CODING_AGENT.value,
    ) -> dict[str, Any]:
        spec = _spec_from_payload(spec_payload)
        run_dir = Path(output_dir or (self._output_root / session_id))
        run_dir.mkdir(parents=True, exist_ok=True)

        safe = _slugify(spec.name)
        script_path = run_dir / f"{safe}.py"
        notebook_path = run_dir / f"{safe}.ipynb"
        spec_path = run_dir / "simulation_spec.json"

        generate_script(spec, script_path)
        generate_notebook(spec, notebook_path)
        spec_path.write_text(json.dumps(_jsonify_spec(spec), indent=2), encoding="utf-8")

        spec_ref = self._registry.create(
            artifact_id=f"{session_id}-runtime-spec",
            assistant=assistant,
            kind="simulation_spec",
            title="Simulation Spec",
            summary=spec.description[:160] if spec.description else spec.name,
            path=str(spec_path.resolve()),
            content=spec_path.read_text(encoding="utf-8"),
            mime_type="application/json",
            metadata={"session_id": session_id, "spec_name": spec.name},
        )
        script_ref = self._registry.create(
            artifact_id=f"{session_id}-runtime-script",
            assistant=assistant,
            kind="simulation_script",
            title="Simulation Script",
            summary=str(script_path.resolve()),
            path=str(script_path.resolve()),
            metadata={"session_id": session_id, "spec_name": spec.name},
        )
        notebook_ref = self._registry.create(
            artifact_id=f"{session_id}-runtime-notebook",
            assistant=assistant,
            kind="simulation_notebook",
            title="Simulation Notebook",
            summary=str(notebook_path.resolve()),
            path=str(notebook_path.resolve()),
            metadata={"session_id": session_id, "spec_name": spec.name},
        )
        return {
            "session_id": session_id,
            "script_path": str(script_path.resolve()),
            "notebook_path": str(notebook_path.resolve()),
            "script_artifact_id": script_ref.artifact_id,
            "notebook_artifact_id": notebook_ref.artifact_id,
            "spec_artifact_id": spec_ref.artifact_id,
        }

    def run_sample(
        self,
        *,
        session_id: str,
        spec_payload: dict[str, Any],
        bundle: dict[str, Any],
        sample_steps: int = 200,
        timeout_seconds: Optional[float] = None,
        assistant: str = AssistantId.CODING_AGENT.value,
    ) -> RuntimeDiagnostics:
        spec = _spec_from_payload(spec_payload)
        config = _default_config(spec)
        config["max_steps"] = max(1, min(int(sample_steps), int(spec.max_steps)))
        return self._launch(
            session_id=session_id,
            spec=spec,
            bundle=bundle,
            config=config,
            output_dir=self._output_root / session_id / "run_sample",
            timeout_seconds=timeout_seconds,
            assistant=assistant,
        )

    def run_sweep(
        self,
        *,
        session_id: str,
        spec_payload: dict[str, Any],
        bundle: dict[str, Any],
        timeout_seconds: Optional[float] = None,
        assistant: str = AssistantId.CODING_AGENT.value,
    ) -> RuntimeDiagnostics:
        spec = _spec_from_payload(spec_payload)
        configs = build_sweep_configs(spec)
        sweep_dir = self._output_root / session_id / "sweep"
        sweep_dir.mkdir(parents=True, exist_ok=True)

        results: list[RunResult] = []
        warnings: list[str] = []
        errors: list[str] = []
        diagnostics_payload: list[dict[str, Any]] = []
        raw_log_ids: list[str] = []

        for index, config in enumerate(configs):
            run_dir = sweep_dir / f"cfg-{index:03d}"
            diag = self._launch(
                session_id=session_id,
                spec=spec,
                bundle=bundle,
                config=config,
                output_dir=run_dir,
                timeout_seconds=timeout_seconds,
                assistant=assistant,
            )
            diagnostics_payload.append(diag.to_dict())
            warnings.extend(diag.warnings)
            errors.extend(diag.errors)
            if diag.raw_log_ref:
                raw_log_ids.append(diag.raw_log_ref)
            run_ref = self._registry.get(diag.result_ref) if diag.result_ref else None
            if run_ref and run_ref.path and Path(run_ref.path).exists():
                payload = json.loads(Path(run_ref.path).read_text(encoding="utf-8"))
                results.extend(_run_results_from_payload(payload))

        summary = RunSummary(
            session_id=session_id,
            results=results,
            sweep_dir=sweep_dir.resolve(),
        )
        summary_ref = self._registry.save_json(
            assistant=assistant,
            kind="runtime_run_summary",
            title="Runtime Sweep Summary",
            filename=f"{session_id}/runtime/sweep_summary.json",
            payload=_jsonify_run_summary(summary),
            summary=f"{len(summary.results)} configuration(s) executed",
            metadata={"session_id": session_id},
            artifact_id=f"{session_id}-runtime-sweep-summary",
        )
        diagnostics_ref = self._registry.save_json(
            assistant=assistant,
            kind="runtime_diagnostics",
            title="Runtime Sweep Diagnostics",
            filename=f"{session_id}/runtime/sweep_diagnostics.json",
            payload={
                "session_id": session_id,
                "diagnostics": diagnostics_payload,
            },
            summary=f"{len(diagnostics_payload)} diagnostics item(s)",
            metadata={"session_id": session_id},
            artifact_id=f"{session_id}-runtime-sweep-diagnostics",
        )
        return RuntimeDiagnostics(
            status="failed" if errors else "success",
            warnings=_dedupe(warnings),
            errors=_dedupe(errors),
            failure_class="runtime_error" if errors else "",
            metrics_summary=_metrics_from_summary(summary),
            result_ref=summary_ref.artifact_id,
            diagnostics_ref=diagnostics_ref.artifact_id,
            raw_log_ref=",".join(raw_log_ids) if raw_log_ids else None,
            metadata={"sweep_dir": str(sweep_dir.resolve())},
        )

    def read_results(self, artifact_id: str) -> Optional[dict[str, Any]]:
        text = self._registry.read_artifact_text(artifact_id)
        if not text:
            return None
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return {"text": text}

    def read_diagnostics(self, artifact_id: str) -> Optional[dict[str, Any]]:
        return self.read_results(artifact_id)

    def read_raw_logs(self, artifact_id: str) -> Optional[str]:
        return self._registry.read_artifact_text(artifact_id)

    def _launch(
        self,
        *,
        session_id: str,
        spec: SimulationSpec,
        bundle: dict[str, Any],
        config: dict[str, Any],
        output_dir: Path,
        timeout_seconds: Optional[float],
        assistant: str,
    ) -> RuntimeDiagnostics:
        script_path = Path(str(bundle["script_path"]))
        launched = self._launcher.launch(
            session_id=session_id,
            script_path=script_path,
            config=config,
            output_dir=output_dir,
            setup_code=spec.setup_code,
            timeout_seconds=timeout_seconds,
            stream_logs=False,
        )
        if isinstance(launched, BugTicket):
            return self._diagnostics_from_bug_ticket(
                session_id=session_id,
                ticket=launched,
                output_dir=output_dir,
                assistant=assistant,
            )
        return self._diagnostics_from_launch_result(
            session_id=session_id,
            launch_result=launched,
            output_dir=output_dir,
            assistant=assistant,
        )

    def _diagnostics_from_bug_ticket(
        self,
        *,
        session_id: str,
        ticket: BugTicket,
        output_dir: Path,
        assistant: str,
    ) -> RuntimeDiagnostics:
        raw_log_ref = self._registry.save_text(
            assistant=assistant,
            kind="runtime_raw_logs",
            title="Runtime Failure Logs",
            filename=f"{session_id}/runtime/{output_dir.name}_logs.txt",
            text=(
                f"stdout_tail:\n{ticket.stdout_tail}\n\n"
                f"stderr_tail:\n{ticket.stderr_tail}\n"
            ),
            summary=f"{ticket.kind.value}: {ticket.error_line or ticket.suggested_fix}",
            metadata={"session_id": session_id, "output_dir": str(output_dir.resolve())},
            artifact_id=f"{session_id}-{output_dir.name}-raw-logs",
        )
        diagnostics_ref = self._registry.save_json(
            assistant=assistant,
            kind="runtime_diagnostics",
            title="Runtime Failure Diagnostics",
            filename=f"{session_id}/runtime/{output_dir.name}_diagnostics.json",
            payload=_jsonify_bug_ticket(ticket),
            summary=ticket.suggested_fix or ticket.error_line or ticket.kind.value,
            metadata={"session_id": session_id, "output_dir": str(output_dir.resolve())},
            artifact_id=f"{session_id}-{output_dir.name}-diagnostics",
        )
        return RuntimeDiagnostics(
            status="failed",
            warnings=[],
            errors=[
                part
                for part in [ticket.error_line, ticket.suggested_fix, ticket.kind.value]
                if str(part).strip()
            ],
            failure_class=ticket.kind.value,
            metrics_summary={"exit_code": ticket.exit_code, "wall_time_seconds": ticket.wall_time_seconds},
            diagnostics_ref=diagnostics_ref.artifact_id,
            raw_log_ref=raw_log_ref.artifact_id,
            metadata={"output_dir": str(output_dir.resolve())},
        )

    def _diagnostics_from_launch_result(
        self,
        *,
        session_id: str,
        launch_result: LaunchResult,
        output_dir: Path,
        assistant: str,
    ) -> RuntimeDiagnostics:
        summary_ref = self._registry.save_json(
            assistant=assistant,
            kind="runtime_run_summary",
            title="Runtime Run Summary",
            filename=f"{session_id}/runtime/{output_dir.name}_summary.json",
            payload=_jsonify_run_summary(launch_result.run_summary),
            summary=f"{len(launch_result.run_summary.results)} result(s)",
            metadata={"session_id": session_id, "output_dir": str(output_dir.resolve())},
            artifact_id=f"{session_id}-{output_dir.name}-summary",
        )
        diagnostics_ref = self._registry.save_json(
            assistant=assistant,
            kind="runtime_diagnostics",
            title="Runtime Diagnostics",
            filename=f"{session_id}/runtime/{output_dir.name}_diagnostics.json",
            payload={
                "env_info": _jsonify_environment(launch_result.env_info),
                "launcher_log_entry": launch_result.launcher_log_entry,
                "pipeline_script_path": launch_result.pipeline_script_path,
            },
            summary=f"Launch succeeded for {output_dir.name}",
            metadata={"session_id": session_id, "output_dir": str(output_dir.resolve())},
            artifact_id=f"{session_id}-{output_dir.name}-diagnostics",
        )
        warnings = []
        for result in launch_result.run_summary.results:
            if result.outcome == RunOutcome.MAX_STEPS:
                warnings.append(result.reason)
            elif result.outcome == RunOutcome.FAILED:
                warnings.append(result.reason)
        return RuntimeDiagnostics(
            status="success",
            warnings=_dedupe(warnings),
            errors=[],
            failure_class="",
            metrics_summary=_metrics_from_summary(launch_result.run_summary),
            result_ref=summary_ref.artifact_id,
            diagnostics_ref=diagnostics_ref.artifact_id,
            metadata={
                "output_dir": str(output_dir.resolve()),
                "pipeline_script_path": launch_result.pipeline_script_path or "",
            },
        )


def _slugify(value: str) -> str:
    safe = value.lower().replace(" ", "_").replace("-", "_").replace("—", "")
    return "".join(c if c.isalnum() or c == "_" else "_" for c in safe).strip("_") or "simulation"


def _default_config(spec: SimulationSpec) -> dict[str, Any]:
    return {
        variable.name: variable.default
        for variable in spec.variables
    }


def _dedupe(items: list[str]) -> list[str]:
    return list(dict.fromkeys(str(item).strip() for item in items if str(item).strip()))


def _metrics_from_summary(summary: RunSummary) -> dict[str, Any]:
    return {
        "n_runs": len(summary.results),
        "n_success": summary.n_success,
        "n_failed": summary.n_failed,
        "n_max_steps": summary.n_max_steps,
        "n_crash": summary.n_crash,
        "sweep_dir": str(summary.sweep_dir),
        "data_files": [str(path) for path in summary.data_files],
    }


def _jsonify_environment(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if hasattr(value, "__dataclass_fields__"):
        return {
            name: _jsonify_environment(getattr(value, name))
            for name in value.__dataclass_fields__.keys()
        }
    if isinstance(value, dict):
        return {str(k): _jsonify_environment(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonify_environment(v) for v in value]
    return value


def _jsonify_bug_ticket(ticket: BugTicket) -> dict[str, Any]:
    payload = asdict(ticket)
    for key, value in list(payload.items()):
        if isinstance(value, Enum):
            payload[key] = value.value
    return payload


def _jsonify_run_summary(summary: RunSummary) -> dict[str, Any]:
    return {
        "session_id": summary.session_id,
        "sweep_dir": str(summary.sweep_dir),
        "results": [
            {
                "config": dict(result.config),
                "outcome": result.outcome.value,
                "reason": result.reason,
                "stop_condition_name": result.stop_condition_name,
                "steps_run": result.steps_run,
                "wall_time_seconds": result.wall_time_seconds,
                "output_dir": str(result.output_dir),
                "data_file": str(result.data_file) if result.data_file else None,
            }
            for result in summary.results
        ],
    }


def _run_results_from_payload(payload: dict[str, Any]) -> list[RunResult]:
    results: list[RunResult] = []
    for item in payload.get("results", []):
        try:
            outcome = RunOutcome(str(item.get("outcome", "crash")))
        except ValueError:
            outcome = RunOutcome.CRASH
        results.append(
            RunResult(
                config=dict(item.get("config") or {}),
                outcome=outcome,
                reason=str(item.get("reason") or ""),
                stop_condition_name=str(item.get("stop_condition_name") or ""),
                steps_run=int(item.get("steps_run") or 0),
                wall_time_seconds=float(item.get("wall_time_seconds") or 0.0),
                output_dir=Path(str(item.get("output_dir") or ".")),
                data_file=Path(str(item["data_file"])) if item.get("data_file") else None,
            )
        )
    return results


def _jsonify_spec(spec: SimulationSpec) -> dict[str, Any]:
    return {
        "name": spec.name,
        "description": spec.description,
        "variables": [
            {
                "name": variable.name,
                "description": variable.description,
                "kind": variable.kind.value,
                "default": variable.default,
                "min_val": variable.min_val,
                "max_val": variable.max_val,
                "step": variable.step,
                "choices": variable.choices,
                "unit": variable.unit,
                "sweep": variable.sweep,
                "sweep_values": variable.sweep_values,
            }
            for variable in spec.variables
        ],
        "stopping_conditions": [asdict(condition) for condition in spec.stopping_conditions],
        "state_fields": list(spec.state_fields),
        "setup_code": spec.setup_code,
        "precompute_code": spec.precompute_code,
        "initial_state_code": spec.initial_state_code,
        "step_code": spec.step_code,
        "progress_code": spec.progress_code,
        "config_assert_code": spec.config_assert_code,
        "state_assert_code": spec.state_assert_code,
        "output_variables": list(spec.output_variables),
        "data_log_variables": list(spec.data_log_variables),
        "data_log_interval": spec.data_log_interval,
        "checkpoint_interval": spec.checkpoint_interval,
        "max_steps": spec.max_steps,
        "progress_interval": spec.progress_interval,
        "time_estimate_seconds": spec.time_estimate_seconds,
        "time_estimate_explanation": spec.time_estimate_explanation,
    }


def _spec_from_payload(payload: dict[str, Any]) -> SimulationSpec:
    variables = [
        Variable(
            name=str(item["name"]),
            description=str(item.get("description") or ""),
            kind=VariableKind(str(item.get("kind") or "string")),
            default=item.get("default"),
            min_val=item.get("min_val"),
            max_val=item.get("max_val"),
            step=item.get("step"),
            choices=item.get("choices"),
            unit=item.get("unit"),
            sweep=bool(item.get("sweep", False)),
            sweep_values=item.get("sweep_values"),
        )
        for item in (payload.get("variables") or [])
    ]
    conditions = [
        StoppingCondition(
            kind=str(item.get("kind") or ""),
            name=str(item.get("name") or ""),
            description=str(item.get("description") or ""),
            check_expr=str(item.get("check_expr") or ""),
            reason_expr=str(item.get("reason_expr") or ""),
            save_on_trigger=bool(item.get("save_on_trigger", True)),
            priority=int(item.get("priority", 0)),
        )
        for item in (payload.get("stopping_conditions") or [])
    ]
    return SimulationSpec(
        name=str(payload.get("name") or "Simulation"),
        description=str(payload.get("description") or ""),
        variables=variables,
        stopping_conditions=conditions,
        state_fields=[tuple(item) for item in (payload.get("state_fields") or [])],
        setup_code=str(payload.get("setup_code") or ""),
        precompute_code=str(payload.get("precompute_code") or ""),
        initial_state_code=str(payload.get("initial_state_code") or ""),
        step_code=str(payload.get("step_code") or ""),
        progress_code=str(payload.get("progress_code") or ""),
        config_assert_code=str(payload.get("config_assert_code") or ""),
        state_assert_code=str(payload.get("state_assert_code") or ""),
        output_variables=list(payload.get("output_variables") or []),
        data_log_variables=list(payload.get("data_log_variables") or []),
        data_log_interval=int(payload.get("data_log_interval") or 1),
        checkpoint_interval=int(payload.get("checkpoint_interval") or 100),
        max_steps=int(payload.get("max_steps") or 1000),
        progress_interval=int(payload.get("progress_interval") or 100),
        time_estimate_seconds=float(payload.get("time_estimate_seconds") or 0.0),
        time_estimate_explanation=str(payload.get("time_estimate_explanation") or ""),
    )
