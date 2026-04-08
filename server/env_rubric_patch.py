"""
env_rubric_patch.py — Instructions to integrate the BaseScenario rubric grader
================================================================================

This file contains the TWO additions needed in server/env.py to wire in
the new BaseScenario rubric system (scenarios/base_scenario.py + scenario_classes.py).

Step 1: Add this import block AFTER the existing alert_triage import line:

    from scenarios.alert_triage import get_alert_triage_scenario, ALERT_TRIAGE_SCENARIOS

Paste this immediately after:

    # ── v4.0: BaseScenario rubric grader (superior Strategy pattern) ──────────
    try:
        from scenarios.scenario_classes import ScenarioRubricAdapter as _RubricAdapter
        _RUBRIC_AVAILABLE = True
    except ImportError:
        _RUBRIC_AVAILABLE = False
        class _RubricAdapter:  # type: ignore
            @staticmethod
            def has_rubric(task_name): return False
            @staticmethod
            def grade(*a, **kw): return None
            @staticmethod
            def grade_details(*a, **kw): return None

Step 2: Add this new method to IncidentResponseEnv, AFTER get_score_breakdown():

    def get_rubric_score_breakdown(self) -> dict:
        \"\"\"
        v4.0: Returns the rubric-based 6D score breakdown using the new
        BaseScenario Strategy pattern (superior to incident-commander-env).

        Falls back to get_score_breakdown() if no rubric class is registered
        for the current task — fully backward compatible.

        Advantages over both the old graders.py approach and incident-commander-env:
          ✅  Per-dimension partial credit per scenario (not just global counters)
          ✅  Investigation-gate enforced declaratively at rubric level
          ✅  Exponential time-decay efficiency (vs step-ratio buckets)
          ✅  Each scenario self-documents its own grading criteria
          ✅  Adding new scenarios requires ZERO env.py edits
          ✅  grade_details() exposes per-criterion pass/fail for debugging
        \"\"\"
        if self._task_name == "alert_triage":
            return self._get_alert_triage_score()

        if not _RUBRIC_AVAILABLE or not _RubricAdapter.has_rubric(self._task_name or ""):
            # Graceful fallback: existing 6D grader
            return self.get_score_breakdown()

        obs_loop = _detect_observation_loop(self._actions_taken)

        rubric_details = _RubricAdapter.grade_details(
            task_name=self._task_name or "",
            actions=self._actions_taken,
            service_states=self._service_states,
            step_number=self._step_number,
            max_steps=self._scenario.max_steps if self._scenario else 30,
            incident_resolved=self._incident_resolved,
            obs_loop=obs_loop,
        )

        if rubric_details is None:
            return self.get_score_breakdown()

        dims = rubric_details["dimensions"]
        final = rubric_details["final_score"]
        failure_type = rubric_details["failure_type"]

        # Build feedback string showing per-check pass/fail
        check_summary = []
        for chk in rubric_details.get("checks", []):
            icon = "✅" if chk["passed"] else ("🔒" if chk.get("gate_failed") else "❌")
            check_summary.append(f"{icon} [{chk['dimension']}] {chk['name']}")
        checks_text = "\\n".join(check_summary[:12])  # cap at 12 for readability
        feedback = (
            f"[{failure_type}] Rubric grader ({len(rubric_details['checks'])} checks):\\n"
            f"{checks_text}\\n"
            f"→ final={final:.4f}"
        )

        return {
            "root_cause":    round(dims["root_cause"]["score"], 4),
            "remediation":   round(dims["remediation"]["score"], 4),
            "investigation": round(dims["investigation"]["score"], 4),
            "efficiency":    round(dims["efficiency"]["score"], 4),
            "safety":        round(dims["safety"]["score"], 4),
            "sequence":      round(dims["sequence"]["score"], 4),
            "final":         final,
            "feedback":      feedback,
            "failure_type":  failure_type,
            "observation_loop": rubric_details["observation_loop"],
            "_rubric_details":  rubric_details,      # full per-check breakdown
            "_rubric_grader":   True,                # flag to identify grader type
        }

    def get_rubric_grade_details(self) -> dict:
        \"\"\"
        v4.0: Return the full per-criterion rubric grading breakdown.
        Useful for debugging, UI drill-down, and training signal analysis.

        Returns dict with:
          task_id, final_score, failure_type, observation_loop,
          dimensions: {dim: {score, weight, weighted}},
          checks: [{name, dimension, weight, passed, gate_failed, partial_credit}]
        \"\"\"
        if not _RUBRIC_AVAILABLE or not _RubricAdapter.has_rubric(self._task_name or ""):
            return {"error": "No rubric class registered for this task", "task": self._task_name}

        obs_loop = _detect_observation_loop(self._actions_taken)
        return _RubricAdapter.grade_details(
            task_name=self._task_name or "",
            actions=self._actions_taken,
            service_states=self._service_states,
            step_number=self._step_number,
            max_steps=self._scenario.max_steps if self._scenario else 30,
            incident_resolved=self._incident_resolved,
            obs_loop=obs_loop,
        ) or {"error": "grade_details returned None"}
"""
