"""
sim_tool.llm_mock
──────────────────
Offline mock backend for demonstrations and tests without API access.
Simulates a scripted Ising model designer session.
"""
from __future__ import annotations
import json
from .llm import LLMBackend


_ISING_SPEC = {
    "spec_complete": True, "questions": [],
    "spec": {
        "name": "2D Ising Model (Mock)",
        "description": "Mock Metropolis MC for offline demo.",
        "variables": [{"name":"temperature","description":"T","kind":"float",
                       "default":2.269,"min_val":1.5,"max_val":4.0,"step":0.5,
                       "choices":None,"unit":"J/kB","sweep":True,
                       "sweep_values":[1.5,2.0,2.269,3.0,4.0]}],
        "state_fields":[["magnetisation","float",0.0],["energy","float",0.0]],
        "stopping_conditions":[
            {"kind":"success","name":"converged","description":"m stable",
             "check_expr":"state.step>50","reason_expr":"f'done at {state.step}'",
             "save_on_trigger":True,"priority":10},
            {"kind":"failure","name":"frozen","description":"lattice frozen",
             "check_expr":"state.step>10 and state.magnetisation==0.0",
             "reason_expr":"f'frozen at {state.step}'","save_on_trigger":True,"priority":20},
        ],
        "setup_code":"import math\nimport copy\n",
        "precompute_code":"    return {}\n",
        "initial_state_code":"    state=SimState()\n    state.magnetisation=0.5\n    state.energy=-1.0\n    return state\n",
        "step_code":"    new=copy.copy(state)\n    new.magnetisation=state.magnetisation*0.99\n    new.energy=state.energy-0.01\n    return new\n",
        "progress_code":"    return f'm={state.magnetisation:.3f}'\n",
        "config_assert_code":"    assert config.temperature>0\n",
        "state_assert_code":"    import math\n    assert math.isfinite(state.magnetisation)\n",
        "output_variables":["magnetisation","energy"],
        "data_log_variables":["magnetisation","energy"],
        "data_log_interval":5,"checkpoint_interval":100,
        "max_steps":200,"progress_interval":50,
        "time_estimate_seconds":5.0,"time_estimate_explanation":"mock run",
    }
}

_ANALYST_RESPONSE = json.dumps({
    "reasoning": "Mock analysis: all runs healthy.",
    "verdict": "ok", "reason": "All configurations completed normally.", "changes": [],
})


class MockBackend(LLMBackend):
    """Scripted responses for offline demonstration."""

    def __init__(self, scenario: str = "ising"):
        self.model_name = f"mock-{scenario}"
        self._call = 0

    def complete(self, system: str, messages: list, **kwargs) -> str:
        self._call += 1
        if "analyst" in system.lower() or "run summary" in str(messages):
            return _ANALYST_RESPONSE
        if "consolidat" in system.lower():
            return '{"what_failed":[],"rules":[]}'
        return json.dumps(_ISING_SPEC)
