"""
Incident Response Environment — Core Logic

Implements the OpenEnv interface: reset(), step(), state()
with rich reward shaping, multi-task support, and deterministic grading.

v3.0 changes (ported + extended from ai-incident-openenv-trial):
  - Observation Loop Detection: ≥3 consecutive diagnosis steps penalised live
    and hard-capped in final score
  - Diagnosis Gate: gated scenarios (cascading_service_timeout, ssl_certificate_expiry,
    database_deadlock) penalise blind fixes with -0.05 per-step; full reward only
    if root-cause service was investigated first
  - Failure Type exposed in step info and score breakdown
  - Sequence tracking fields added to env state for grader
  - Honeypot-style per-step reward penalty for non-root-cause destructive actions
    on scenarios where root cause is known but agent ignores evidence

v2.1 changes:
  - Score clamped to open interval (0.01, 0.99) for OpenEnv validator compliance
  - Per-step feedback string in every observation (explains reward earned/lost)
  - Wires to standalone graders.py for external trajectory grading
  - Service alias normalisation in root cause declaration
  - Gated information: check_service_health hints about uninvestigated services
"""

from __future__ import annotations

import copy
import json
from typing import Any, Dict, List, Optional, Tuple

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import (
    ActionType,
    Alert,
    AlertSeverity,
    IncidentResponseAction,
    IncidentResponseObservation,
    IncidentResponseReward,
    IncidentResponseState,
    ServiceStatus,
)
from scenarios.definitions import ScenarioDef, get_scenario, list_tasks, ScenarioFactory
from scenarios.alert_triage import get_alert_triage_scenario, ALERT_TRIAGE_SCENARIOS

# ── Standalone grader import (project root) ───────────────────────────────
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
try:
    from graders import (
        grade as _external_grade,
        grade_alert_triage as _grade_alert_triage,
        _clamp as _clamp_score,
        _detect_observation_loop,
        _classify_failure_type,
        _OBSERVATION_LOOP_CAP,
        _GATED_SCENARIOS,
        _INVESTIGATION_ACTIONS,
        _DESTRUCTIVE_ACTIONS,
        _investigated_before_fix,
    )
except ImportError:
    def _external_grade(state):  # type: ignore
        return {}
    def _grade_alert_triage(state, scenario):  # type: ignore
        return {"total": 0.01, "breakdown": {}, "feedback": "grader unavailable"}
    def _clamp_score(x):  # type: ignore
        return round(max(0.01, min(0.99, x)), 4)
    def _detect_observation_loop(actions):  # type: ignore
        return False
    def _classify_failure_type(*args, **kwargs):  # type: ignore
        return "Unknown"
    _OBSERVATION_LOOP_CAP = 0.45
    _GATED_SCENARIOS = set()
    _INVESTIGATION_ACTIONS = set()
    _DESTRUCTIVE_ACTIONS = set()
    def _investigated_before_fix(*args, **kwargs):  # type: ignore
        return True

# ── Service alias map (mirrors graders.py for root-cause normalisation) ───
_SVC_ALIASES: Dict[str, str] = {
    "postgres": "postgres-primary", "postgresql": "postgres-primary",
    "postgres_primary": "postgres-primary", "primary_db": "primary-db",
    "primarydb": "primary-db", "db": "postgres-primary",
    "api": "api-gateway", "api_gateway": "api-gateway",
    "apigw": "api-gateway", "gateway": "api-gateway",
    "auth": "auth-service", "authentication": "auth-service",
    "order": "order-service", "orders": "order-service",
    "order_service": "order-service",
    "inventory": "inventory-service", "inventory_service": "inventory-service",
    "payment": "payment-service", "payments": "payment-service",
    "payment_service": "payment-service",
    "user": "user-api", "users": "user-api",
    "user_api": "user-api", "userapi": "user-api",
    "analytics": "analytics-service", "analytics_service": "analytics-service",
    "redis": "redis-cache", "cache": "redis-cache", "redis_cache": "redis-cache",
    "search": "search-service", "search_service": "search-service",
    "cert": "cert-manager", "tls": "cert-manager",
    "certmanager": "cert-manager", "cert_manager": "cert-manager",
    "nginx": "nginx-lb", "lb": "nginx-lb", "loadbalancer": "nginx-lb",
}


def _normalise_svc(name: str) -> str:
    n = name.lower().strip().replace("_", "-").replace(" ", "-")
    return _SVC_ALIASES.get(n, n)


class IncidentResponseEnv:
    """
    Production Incident Response & Root Cause Analysis Environment.

    An AI agent acts as an on-call SRE. It receives alerts about a production
    incident and must investigate logs, metrics, and configurations to diagnose
    root causes, determine blast radius, and execute remediations.
    """

    def __init__(self):
        self._scenario: Optional[ScenarioDef] = None
        self._task_name: Optional[str] = None
        self._step_number: int = 0
        self._done: bool = False
        self._cumulative_reward: float = 0.0
        self._services_investigated: Dict[str, set] = {}
        self._root_causes_declared: List[str] = []
        self._remediations_applied: List[Dict[str, Any]] = []
        self._correct_root_causes_found: int = 0
        self._correct_remediations_found: int = 0
        self._actions_taken: List[str] = []
        self._service_states: Dict[str, Dict[str, Any]] = {}
        self._active_alerts: List[Dict[str, str]] = []
        self._incident_resolved: bool = False
        self._last_action_error: Optional[str] = None
        # v2: dynamic degradation + safety tracking
        self._collateral_degraded: set = set()
        self._degradation_warnings: List[str] = []
        self._scenario_seed: Optional[int] = None
        # v2.1: per-step feedback
        self._last_feedback: str = ""
        # v3.0: observation loop + sequence tracking
        self._consecutive_diagnosis_count: int = 0
        self._observation_loop_warned: bool = False
        self._diagnosis_penalty_applied_at: int = -1  # step when loop penalty was first applied
        # alert_triage task state
        self._at_scenario: Optional[dict] = None
        self._at_submitted_severity: Optional[str] = None

    # ── OpenEnv Interface ─────────────────────────────────────────────────

    def reset(self, task_name: str = "db_connection_failure", seed: Optional[int] = None) -> IncidentResponseObservation:
        """Reset the environment with a specific task. Provide seed for procedural variation."""
        # ── Alert Triage fast-path ────────────────────────────────────────
        if task_name == "alert_triage":
            return self._reset_alert_triage(seed)
        # ── Standard incident-response tasks ─────────────────────────────
        if seed is not None:
            self._scenario = ScenarioFactory.generate(task_name, seed)
        else:
            self._scenario = get_scenario(task_name)
        self._task_name = task_name
        self._scenario_seed = seed
        self._step_number = 0
        self._done = False
        self._cumulative_reward = 0.0
        self._services_investigated = {}
        self._root_causes_declared = []
        self._remediations_applied = []
        self._correct_root_causes_found = 0
        self._correct_remediations_found = 0
        self._actions_taken = []
        self._incident_resolved = False
        self._last_action_error = None
        self._collateral_degraded = set()
        self._degradation_warnings = []
        self._last_feedback = "Incident started. Begin investigation."
        # v3.0 reset
        self._consecutive_diagnosis_count = 0
        self._observation_loop_warned = False
        self._diagnosis_penalty_applied_at = -1

        self._service_states = {}
        for name, svc in self._scenario.services.items():
            self._service_states[name] = {
                "healthy": svc.healthy,
                "response_time_ms": svc.response_time_ms,
                "error_rate": svc.error_rate,
                "cpu_percent": svc.cpu_percent,
                "memory_percent": svc.memory_percent,
                "connections_active": svc.connections_active,
                "version": svc.version,
                "config": copy.deepcopy(svc.config),
                "deployment_status": svc.deployment_status,
                "restarted": False,
                "scaled_replicas": 1,
                "rolled_back": False,
            }

        self._active_alerts = copy.deepcopy(self._scenario.alerts)

        return self._build_observation(
            action_result=(
                f"INCIDENT ALERT: {self._scenario.incident_summary}\n\n"
                f"You are the on-call SRE. Investigate the incident, identify root cause(s), and apply remediations.\n\n"
                f"Available services: {', '.join(self._scenario.services.keys())}\n\n"
                f"Available actions: {', '.join(at.value for at in ActionType)}\n\n"
                f"Action format: {{\"action_type\": \"<type>\", \"service_name\": \"<n>\", \"parameters\": {{...}}}}\n"
                f"  - investigate_logs: params={{\"keyword\": \"<optional filter>\"}}\n"
                f"  - check_metrics: params={{\"metric_type\": \"all|cpu|memory|...\"}}\n"
                f"  - read_config: no extra params\n"
                f"  - check_service_health: no extra params\n"
                f"  - run_diagnostic: no extra params\n"
                f"  - restart_service: no extra params\n"
                f"  - update_config: params={{\"key\": \"<config_key>\", \"value\": \"<new_value>\"}}\n"
                f"  - rollback_deployment: no extra params\n"
                f"  - scale_service: params={{\"replicas\": \"<count>\"}}\n"
                f"  - declare_root_cause: params={{\"cause\": \"<description>\"}}"
            ),
            action_success=True,
        )

    def step(self, action: IncidentResponseAction) -> Tuple[IncidentResponseObservation, float, bool, Dict[str, Any]]:
        """Execute an action and return (observation, reward, done, info)."""
        if self._done:
            obs = self._build_observation("Episode is already done. Call reset() to start a new episode.", False)
            return obs, 0.0, True, {"error": "episode_done"}
        # ── Alert Triage fast-path ────────────────────────────────────────
        if self._task_name == "alert_triage":
            return self._step_alert_triage(action)

        self._step_number += 1
        self._last_action_error = None

        action_result, action_success, reward, reward_breakdown = self._process_action(action)

        # ── v3.0: Observation Loop Tracking ──────────────────────────────
        action_type_str = action.action_type.value
        is_diagnosis_action = action_type_str in _INVESTIGATION_ACTIONS
        is_fix_action = action_type_str in _DESTRUCTIVE_ACTIONS or action_type_str == "update_config"

        if is_diagnosis_action:
            self._consecutive_diagnosis_count += 1
        else:
            self._consecutive_diagnosis_count = 0

        # Live penalty: ≥3 consecutive diagnosis steps with no fix
        if self._consecutive_diagnosis_count >= 3 and not self._observation_loop_warned:
            loop_penalty = -0.08
            reward += loop_penalty
            reward_breakdown["observation_loop_penalty"] = loop_penalty
            self._observation_loop_warned = True
            self._diagnosis_penalty_applied_at = self._step_number
            action_result += (
                "\n\n⚠ OBSERVATION LOOP WARNING: You have performed 3+ consecutive diagnosis "
                "actions without attempting any fix. Reward penalised (-0.08). "
                "Attempt a remediation — diagnose, then fix."
            )
        elif self._consecutive_diagnosis_count >= 3:
            # Ongoing loop: keep applying a smaller rolling penalty
            rolling_penalty = -0.03
            reward += rolling_penalty
            reward_breakdown["observation_loop_rolling"] = rolling_penalty

        # Reset loop warning once agent breaks the pattern with a fix
        if is_fix_action and self._consecutive_diagnosis_count == 0:
            self._observation_loop_warned = False

        # ── v3.0: Diagnosis Gate per-step enforcement ─────────────────────
        if (
            is_fix_action
            and self._task_name in _GATED_SCENARIOS
            and action.service_name
        ):
            investigated_first = _investigated_before_fix(
                self._actions_taken, action.service_name, action_type_str
            )
            if not investigated_first:
                gate_penalty = -0.05
                reward += gate_penalty
                reward_breakdown["diagnosis_gate_penalty"] = gate_penalty
                action_result += (
                    f"\n\n🔒 DIAGNOSIS GATE: You applied a fix to {action.service_name} without "
                    f"investigating it first. Investigate with investigate_logs or check_metrics "
                    f"before fixing for full credit. Penalty: {gate_penalty}"
                )

        # v2.1: Build per-step feedback string from the reward breakdown
        self._last_feedback = self._build_feedback(reward, reward_breakdown, action)

        action_desc = f"{action.action_type.value}"
        if action.service_name:
            action_desc += f"({action.service_name})"
        self._actions_taken.append(action_desc)

        # v2: Dynamic degradation — unresolved failures spread to dependent services
        self._degrade_system()

        self._check_resolution()

        if self._step_number >= self._scenario.max_steps:
            self._done = True
            if not self._incident_resolved:
                reward -= 0.05
                reward_breakdown["timeout_penalty"] = -0.05
                self._last_feedback += " | Timeout penalty: -0.05"

        if self._incident_resolved:
            steps_ratio = self._step_number / self._scenario.max_steps
            time_bonus = 0.10 if steps_ratio < 0.5 else (0.05 if steps_ratio < 0.75 else 0.0)
            reward += time_bonus
            reward_breakdown["time_bonus"] = time_bonus
            self._done = True
            if time_bonus > 0:
                self._last_feedback += f" | Time bonus: +{time_bonus:.2f} (fast resolution)"

        reward = max(-1.0, min(1.0, reward))
        self._cumulative_reward += reward

        # v3.0: Compute live failure type for info
        obs_loop = _detect_observation_loop(self._actions_taken)
        failure_type = _classify_failure_type(
            self._actions_taken,
            self._scenario.correct_remediations if self._scenario else [],
            self._incident_resolved,
            self._step_number,
            self._scenario.max_steps if self._scenario else 30,
            obs_loop,
        )

        obs = self._build_observation(action_result, action_success)
        info = {
            "reward_breakdown": reward_breakdown,
            "step": self._step_number,
            "incident_resolved": self._incident_resolved,
            "last_action_error": self._last_action_error,
            "feedback": self._last_feedback,
            # v3.0 additions
            "failure_type": failure_type,
            "observation_loop": obs_loop,
            "consecutive_diagnosis_count": self._consecutive_diagnosis_count,
        }

        return obs, reward, self._done, info

    def state(self) -> IncidentResponseState:
        """Return the full internal state."""
        return IncidentResponseState(
            task_name=self._task_name or "",
            task_difficulty=self._scenario.difficulty if self._scenario else "",
            step_number=self._step_number,
            elapsed_time_minutes=self._step_number * 2,
            services={k: v for k, v in self._service_states.items()},
            alerts=self._active_alerts,
            root_causes=self._root_causes_declared,
            agent_findings=[],
            agent_actions_taken=self._actions_taken,
            remediation_applied=[json.dumps(r) for r in self._remediations_applied],
            correct_remediations=[json.dumps(r) for r in self._scenario.correct_remediations] if self._scenario else [],
            incident_resolved=self._incident_resolved,
            cumulative_reward=self._cumulative_reward,
            done=self._done,
            max_steps=self._scenario.max_steps if self._scenario else 30,
            collateral_degraded_services=list(self._collateral_degraded),
            scenario_seed=self._scenario_seed,
        )

    def get_score(self) -> float:
        """Compute final grader score clamped to open interval (0.01, 0.99)."""
        return self.get_score_breakdown()["final"]

    def get_score_breakdown(self) -> Dict[str, float]:
        """
        Return full score breakdown. Routes to alert_triage grader when task is alert_triage.
        """
        if self._task_name == "alert_triage":
            return self._get_alert_triage_score()
        if not self._scenario:
            return {
                "root_cause": 0.0, "remediation": 0.0, "investigation": 0.0,
                "efficiency": 0.0, "safety": 0.0, "sequence": 0.0,
                "final": 0.01, "feedback": "No scenario loaded.",
                "failure_type": "Unknown", "observation_loop": False,
            }

        total_causes = max(len(self._scenario.root_causes), 1)
        root_cause_score = min(self._correct_root_causes_found / total_causes, 1.0)

        total_remediations = max(len(self._scenario.correct_remediations), 1)
        remediation_score = min(self._correct_remediations_found / total_remediations, 1.0)

        total_services = max(len(self._scenario.services), 1)
        investigation_score = min(len(self._services_investigated) / total_services, 1.0)

        if self._step_number == 0:
            efficiency_score = 0.0
        else:
            ratio = self._step_number / self._scenario.max_steps
            efficiency_score = 1.0 if ratio <= 0.4 else (0.7 if ratio <= 0.7 else (0.4 if ratio <= 1.0 else 0.1))

        safety_score = self._compute_safety_score()

        # v3.0: Pull sequence + failure_type from grader
        obs_loop = _detect_observation_loop(self._actions_taken)

        # Build _grade_components for grader.grade()
        _grade_components = {
            "root_cause": round(root_cause_score, 4),
            "remediation": round(remediation_score, 4),
            "investigation": round(investigation_score, 4),
            "efficiency": round(efficiency_score, 4),
            "safety": round(safety_score, 4),
        }

        state_dict = self.state().model_dump()
        state_dict["_grade_components"] = _grade_components
        state_dict["incident_resolved"] = self._incident_resolved

        grader_result = _external_grade(state_dict)
        sequence_score = grader_result.get("sequence", 0.0)
        failure_type = grader_result.get("failure_type", "Unknown")

        # v3.0 weights: root_cause 0.30, remediation 0.25, sequence 0.10 (new)
        raw_final = (
            root_cause_score    * 0.30
            + remediation_score   * 0.25
            + investigation_score * 0.15
            + efficiency_score    * 0.10
            + safety_score        * 0.10
            + sequence_score      * 0.10
        )

        final = _clamp_score(raw_final)

        # v3.0: Observation Loop hard cap
        if obs_loop and not self._incident_resolved:
            final = min(final, _OBSERVATION_LOOP_CAP)

        feedback = (
            f"[{failure_type}] "
            f"root_cause={root_cause_score:.2f}×0.30  remediation={remediation_score:.2f}×0.25  "
            f"investigation={investigation_score:.2f}×0.15  efficiency={efficiency_score:.2f}×0.10  "
            f"safety={safety_score:.2f}×0.10  sequence={sequence_score:.2f}×0.10  →  final={final:.4f}"
        )
        if obs_loop and not self._incident_resolved:
            feedback += f"  ⚠ Observation Loop cap ({_OBSERVATION_LOOP_CAP}) applied"

        return {
            "root_cause":    round(root_cause_score, 4),
            "remediation":   round(remediation_score, 4),
            "investigation": round(investigation_score, 4),
            "efficiency":    round(efficiency_score, 4),
            "safety":        round(safety_score, 4),
            "sequence":      round(sequence_score, 4),          # NEW v3.0
            "final":         final,
            "feedback":      feedback,
            "failure_type":  failure_type,                      # NEW v3.0
            "observation_loop": obs_loop,                       # NEW v3.0
            "_grade_components": _grade_components,
        }

    def _compute_safety_score(self) -> float:
        """
        Safety dimension: penalizes destructive actions on services that are NOT
        part of the correct remediation chain.
        v2.0: also penalizes collateral degradation caused by slow response.
        """
        correct_targets = {r.get("service_name", "") for r in self._scenario.correct_remediations}

        penalties = 0
        for action in self._remediations_applied:
            if action.get("action_type") in _DESTRUCTIVE_ACTIONS:
                target = action.get("service_name", "")
                if target and target not in correct_targets:
                    penalties += 1

        collateral_penalty = len(self._collateral_degraded) * 0.15
        raw = max(0.0, 1.0 - penalties * 0.25 - collateral_penalty)
        return round(raw, 4)

    def close(self):
        pass

    # ── Per-step feedback builder (v2.1 / v3.0) ───────────────────────────

    def _build_feedback(self, reward: float, breakdown: Dict[str, float], action: IncidentResponseAction) -> str:
        """Generate a human-readable explanation of the reward signal for this step."""
        parts = []
        sign = "+" if reward >= 0 else ""
        parts.append(f"Step {self._step_number}: reward={sign}{reward:.3f}")

        for key, val in breakdown.items():
            if key == "investigation" and val > 0:
                svc = action.service_name or ""
                parts.append(f"New service investigated ({svc}): +{val:.3f}")
            elif key == "redundant":
                parts.append(f"Redundant action (already investigated): {val:.3f}")
            elif key == "correct_root_cause":
                parts.append(f"Correct root cause declared: +{val:.3f}")
            elif key == "wrong_cause":
                parts.append(f"Wrong root cause — keep investigating: {val:.3f}")
            elif key == "correct_remediation":
                parts.append(f"Correct remediation applied: +{val:.3f}")
            elif key == "unnecessary":
                parts.append(f"Unnecessary remediation (wrong service): {val:.3f}")
            elif key == "invalid_service":
                parts.append(f"Invalid service name: {val:.3f}")
            elif key == "wrong_config":
                parts.append(f"Config changed but not a fix: {val:.3f}")
            elif key == "no_deployment":
                parts.append(f"No recent deployment to rollback: {val:.3f}")
            # v3.0 entries
            elif key == "observation_loop_penalty":
                parts.append(f"⚠ Observation Loop penalty (3+ diagnosis steps, no fix): {val:.3f}")
            elif key == "observation_loop_rolling":
                parts.append(f"⚠ Observation Loop ongoing penalty: {val:.3f}")
            elif key == "diagnosis_gate_penalty":
                parts.append(f"🔒 Diagnosis Gate penalty (fix without prior investigation): {val:.3f}")

        if self._scenario:
            remaining = self._scenario.max_steps - self._step_number
            if remaining <= 5:
                parts.append(f"⚠ {remaining} steps remaining!")

        if self._consecutive_diagnosis_count >= 2:
            parts.append(f"📋 Diagnosis streak: {self._consecutive_diagnosis_count} (fix something soon!)")

        return " | ".join(parts)

    # ── Action Processing ─────────────────────────────────────────────────

    def _process_action(self, action: IncidentResponseAction) -> Tuple[str, bool, float, Dict[str, float]]:
        at = action.action_type
        svc_name = action.service_name
        params = action.parameters

        needs_service = at not in (ActionType.DECLARE_ROOT_CAUSE,)
        if needs_service:
            if not svc_name or svc_name not in self._scenario.services:
                self._last_action_error = f"Unknown service: {svc_name}. Available: {list(self._scenario.services.keys())}"
                return self._last_action_error, False, -0.02, {"invalid_service": -0.02}

        if at == ActionType.INVESTIGATE_LOGS:
            return self._do_investigate_logs(svc_name, params)
        elif at == ActionType.CHECK_METRICS:
            return self._do_check_metrics(svc_name, params)
        elif at == ActionType.READ_CONFIG:
            return self._do_read_config(svc_name)
        elif at == ActionType.CHECK_SERVICE_HEALTH:
            return self._do_check_health(svc_name)
        elif at == ActionType.RUN_DIAGNOSTIC:
            return self._do_run_diagnostic(svc_name)
        elif at == ActionType.RESTART_SERVICE:
            return self._do_restart(svc_name, params)
        elif at == ActionType.UPDATE_CONFIG:
            return self._do_update_config(svc_name, params)
        elif at == ActionType.ROLLBACK_DEPLOYMENT:
            return self._do_rollback(svc_name)
        elif at == ActionType.SCALE_SERVICE:
            return self._do_scale(svc_name, params)
        elif at == ActionType.DECLARE_ROOT_CAUSE:
            return self._do_declare_root_cause(params)
        else:
            self._last_action_error = f"Unknown action: {at}"
            return self._last_action_error, False, -0.02, {"unknown_action": -0.02}

    def _track_investigation(self, svc_name: str, action_key: str) -> Tuple[bool, float]:
        if svc_name not in self._services_investigated:
            self._services_investigated[svc_name] = set()
        if action_key not in self._services_investigated[svc_name]:
            self._services_investigated[svc_name].add(action_key)
            return True, self._scenario.investigation_hints.get(svc_name, 0.01)
        return False, -0.01

    def _do_investigate_logs(self, svc_name, params):
        svc = self._scenario.services[svc_name]
        keyword = params.get("keyword", "").lower()
        filtered = [l for l in svc.logs if keyword in l["message"].lower() or keyword in l["level"].lower()] if keyword else svc.logs
        log_text = "\n".join(f"[{l['timestamp']}] {l['level']}: {l['message']}" for l in filtered)
        if not log_text:
            log_text = f"No logs matching '{keyword}' for {svc_name}"
        result = f"=== Logs: {svc_name} {'(filter: ' + keyword + ')' if keyword else ''} ===\n{log_text}"
        is_new, r = self._track_investigation(svc_name, "logs")
        bd = {"investigation": r} if is_new else {"redundant": r}
        return result, True, r, bd

    def _do_check_metrics(self, svc_name, params):
        svc = self._scenario.services[svc_name]
        metric_type = params.get("metric_type", "all")
        lines = [f"=== Metrics: {svc_name} ==="]
        for m in svc.metrics_history:
            ts = m.get("timestamp_min", 0)
            label = f"T{ts:+d}min" if ts else "NOW"
            if metric_type == "all":
                vals = " | ".join(f"{k}={v}" for k, v in m.items() if k != "timestamp_min")
            else:
                val = m.get(metric_type)
                vals = f"{metric_type}={val}" if val is not None else f"{metric_type}=N/A"
            lines.append(f"[{label}] {vals}")
        is_new, r = self._track_investigation(svc_name, "metrics")
        bd = {"investigation": r} if is_new else {"redundant": r}
        return "\n".join(lines), True, r * 0.8, bd

    def _do_read_config(self, svc_name):
        config = self._service_states[svc_name]["config"]
        result = f"=== Config: {svc_name} ===\n{json.dumps(config, indent=2)}"
        is_new, r = self._track_investigation(svc_name, "config")
        bd = {"investigation": r} if is_new else {"redundant": r}
        return result, True, r * 0.7, bd

    def _do_check_health(self, svc_name):
        state = self._service_states[svc_name]
        svc = self._scenario.services[svc_name]

        gate_hint = ""
        if svc_name not in self._services_investigated:
            gate_hint = (
                f"\n[HINT] No logs or metrics checked for {svc_name} yet. "
                f"Run investigate_logs or check_metrics first for richer diagnostics."
            )

        lines = [
            f"=== Health: {svc_name} ===",
            f"Status: {'HEALTHY' if state['healthy'] else 'UNHEALTHY'}",
            f"Response Time: {state['response_time_ms']}ms",
            f"Error Rate: {state['error_rate']*100:.1f}%",
            f"CPU: {state['cpu_percent']}% | Memory: {state['memory_percent']}%",
            f"Connections: {state['connections_active']}/{svc.connections_max}",
            f"Version: {state['version']} (prev: {svc.previous_version})",
            f"Deploy Status: {state['deployment_status']}",
            f"Dependencies: {', '.join(svc.dependencies) or 'none'}",
        ]
        if gate_hint:
            lines.append(gate_hint)

        is_new, r = self._track_investigation(svc_name, "health")
        bd = {"investigation": r} if is_new else {"redundant": r}
        return "\n".join(lines), True, r * 0.5, bd

    def _do_run_diagnostic(self, svc_name):
        svc = self._scenario.services[svc_name]
        result = f"=== Diagnostic: {svc_name} ===\n{svc.diagnostic_output}"
        is_new, r = self._track_investigation(svc_name, "diagnostic")
        bd = {"investigation": r} if is_new else {"redundant": r}
        return result, True, r * 1.2, bd

    def _do_restart(self, svc_name, params):
        state = self._service_states[svc_name]
        is_correct = self._check_remediation_match("restart_service", svc_name, params)
        self._remediations_applied.append({"action_type": "restart_service", "service_name": svc_name})
        if is_correct and not state["restarted"]:
            state["restarted"] = True
            state["healthy"] = True
            state["error_rate"] = 0.02
            state["response_time_ms"] = 50.0
            self._correct_remediations_found += 1
            return f"Service {svc_name} restarted successfully. Now healthy.", True, 0.15, {"correct_remediation": 0.15}
        elif state["restarted"]:
            return f"Service {svc_name} already restarted.", True, -0.03, {"redundant": -0.03}
        else:
            state["restarted"] = True
            return f"Service {svc_name} restarted (may not address root cause).", True, -0.02, {"unnecessary": -0.02}

    def _do_update_config(self, svc_name, params):
        state = self._service_states[svc_name]
        key = params.get("key", "")
        value = params.get("value", "")
        if not key:
            self._last_action_error = "update_config needs 'key' param"
            return self._last_action_error, False, -0.02, {"invalid_params": -0.02}
        old_val = state["config"].get(key, "<unset>")
        state["config"][key] = self._parse_config_value(value)
        self._remediations_applied.append({"action_type": "update_config", "service_name": svc_name, "key": key, "value": value})
        is_correct = self._check_remediation_match("update_config", svc_name, params)
        if is_correct:
            self._correct_remediations_found += 1
            state["healthy"] = True
            state["error_rate"] = max(0.0, state["error_rate"] - 0.5)
            return f"Config {svc_name}.{key} = {value} (was {old_val}). Fix applied.", True, 0.15, {"correct_remediation": 0.15}
        return f"Config {svc_name}.{key} = {value} (was {old_val}).", True, -0.02, {"wrong_config": -0.02}

    def _do_rollback(self, svc_name):
        state = self._service_states[svc_name]
        svc = self._scenario.services[svc_name]
        self._remediations_applied.append({"action_type": "rollback_deployment", "service_name": svc_name})
        is_correct = self._check_remediation_match("rollback_deployment", svc_name, {})
        if is_correct and not state["rolled_back"]:
            state["rolled_back"] = True
            state["version"] = svc.previous_version
            state["deployment_status"] = "rolled_back"
            state["healthy"] = True
            state["error_rate"] = max(0.0, state["error_rate"] - 0.3)
            self._correct_remediations_found += 1
            return f"Rolled back {svc_name} to {svc.previous_version}.", True, 0.15, {"correct_remediation": 0.15}
        elif state["rolled_back"]:
            return f"{svc_name} already rolled back.", True, -0.03, {"redundant": -0.03}
        elif svc.deployment_status != "recently_deployed":
            return f"No recent deployment on {svc_name} to rollback.", True, -0.03, {"no_deployment": -0.03}
        else:
            state["rolled_back"] = True
            state["version"] = svc.previous_version
            return f"Rolled back {svc_name} (may not help).", True, -0.01, {"wrong_rollback": -0.01}

    def _do_scale(self, svc_name, params):
        state = self._service_states[svc_name]
        replicas = int(params.get("replicas", 2))
        self._remediations_applied.append({"action_type": "scale_service", "service_name": svc_name, "replicas": replicas})
        is_correct = self._check_remediation_match("scale_service", svc_name, params)
        state["scaled_replicas"] = replicas
        if is_correct:
            self._correct_remediations_found += 1
            return f"Scaled {svc_name} to {replicas} replicas.", True, 0.12, {"correct_remediation": 0.12}
        return f"Scaled {svc_name} to {replicas} replicas.", True, -0.01, {"unnecessary_scale": -0.01}

    def _do_declare_root_cause(self, params):
        """
        v2.1: Service alias normalisation — "postgres" matches "postgres-primary",
        "order_service" matches "order-service", etc. Reduces false negatives from
        minor naming variations in the agent's declared cause.
        """
        cause = params.get("cause", "")
        if not cause:
            self._last_action_error = "declare_root_cause needs 'cause' param"
            return self._last_action_error, False, -0.02, {"invalid_params": -0.02}

        cause_normalised = cause
        for alias, canonical in _SVC_ALIASES.items():
            if alias in cause.lower():
                cause_normalised = cause_normalised.lower().replace(alias, canonical)

        self._root_causes_declared.append(cause)
        for i, keywords in enumerate(self._scenario.root_cause_keywords):
            for check_cause in [cause, cause_normalised]:
                cause_lower = check_cause.lower()
                match_count = sum(1 for kw in keywords if kw.lower() in cause_lower)
                if match_count >= 2:
                    already = any(
                        sum(1 for kw in keywords if kw.lower() in p.lower()) >= 2
                        for p in self._root_causes_declared[:-1]
                    )
                    if not already:
                        self._correct_root_causes_found += 1
                        return (
                            f"ROOT CAUSE: '{cause}' ✓ Matches known cause!",
                            True, 0.20, {"correct_root_cause": 0.20}
                        )
                    return (
                        f"ROOT CAUSE: '{cause}' — already identified.",
                        True, -0.02, {"duplicate": -0.02}
                    )

        return (
            f"ROOT CAUSE: '{cause}' ✗ Does not match. Keep investigating.",
            True, -0.05, {"wrong_cause": -0.05}
        )

    # ── Helpers ────────────────────────────────────────────────────────────

    def _check_remediation_match(self, action_type: str, svc_name: str, params: dict) -> bool:
        for rem in self._scenario.correct_remediations:
            if rem["action_type"] == action_type and rem.get("service_name") == svc_name:
                if action_type == "update_config":
                    ek = rem.get("parameters", {}).get("key", "")
                    ev = str(rem.get("parameters", {}).get("value", ""))
                    ak = params.get("key", "")
                    av = str(params.get("value", ""))
                    if ek == ak and (ev == av or ev in av):
                        return True
                else:
                    return True
        return False

    def _check_resolution(self):
        if not self._scenario:
            return
        if (self._correct_root_causes_found >= len(self._scenario.root_causes) and
                self._correct_remediations_found >= len(self._scenario.correct_remediations)):
            self._incident_resolved = True
            for alert in self._active_alerts:
                alert["severity"] = "resolved"

    def _degrade_system(self):
        """
        v2.0: Time-coupled dynamic degradation.
        Every 4 steps, services that depend on unhealthy uncorrected services
        progressively degrade. Creates genuine urgency missing from competitors.
        """
        if self._incident_resolved:
            return
        if self._step_number % 4 != 0:
            return

        correct_targets = {r.get("service_name", "") for r in self._scenario.correct_remediations}

        for svc_name, state in self._service_states.items():
            if svc_name in correct_targets:
                continue
            if svc_name in self._collateral_degraded:
                continue

            svc_def = self._scenario.services[svc_name]
            has_unhealthy_dep = any(
                not self._service_states[dep]["healthy"]
                for dep in svc_def.dependencies
                if dep in self._service_states
            )

            if has_unhealthy_dep and state["healthy"]:
                state["error_rate"] = min(1.0, round(state["error_rate"] + 0.10, 3))
                state["response_time_ms"] = min(60000.0, round(state["response_time_ms"] * 1.4, 1))

                if state["error_rate"] >= 0.50:
                    state["healthy"] = False
                    self._collateral_degraded.add(svc_name)
                    warning = (
                        f"⚠ DEGRADATION: {svc_name} tipped unhealthy (error_rate="
                        f"{state['error_rate']:.0%}) due to unresolved upstream failure. "
                        f"Elapsed: {self._step_number * 2} min."
                    )
                    self._degradation_warnings.append(warning)
                    self._active_alerts.append({
                        "alert_id": f"ALT-DEGRADE-{svc_name[:6].upper()}",
                        "severity": "high",
                        "service": svc_name,
                        "message": f"{svc_name} degraded due to upstream failure (collateral damage)",
                        "timestamp": f"T+{self._step_number * 2}min",
                    })

    def _build_observation(self, action_result: str, action_success: bool) -> IncidentResponseObservation:
        alerts = []
        for a in self._active_alerts:
            sev = a["severity"]
            try:
                sev_enum = AlertSeverity(sev)
            except ValueError:
                sev_enum = AlertSeverity.MEDIUM
            alerts.append(Alert(alert_id=a["alert_id"], severity=sev_enum, service=a["service"],
                                message=a["message"], timestamp=a["timestamp"]))
        statuses = []
        for name, st in self._service_states.items():
            svc_def = self._scenario.services[name]
            statuses.append(ServiceStatus(name=name, healthy=st["healthy"],
                                          response_time_ms=st["response_time_ms"],
                                          error_rate=st["error_rate"],
                                          uptime_seconds=svc_def.uptime_seconds))
        return IncidentResponseObservation(
            action_result=action_result, action_success=action_success,
            active_alerts=alerts, service_statuses=statuses,
            available_services=list(self._scenario.services.keys()),
            available_actions=[at.value for at in ActionType],
            step_number=self._step_number,
            max_steps=self._scenario.max_steps,
            elapsed_time_minutes=self._step_number * 2,
            incident_summary=self._scenario.incident_summary,
            task_name=self._task_name or "",
            task_difficulty=self._scenario.difficulty,
            degradation_warnings=list(self._degradation_warnings),
            feedback=self._last_feedback,
        )

    @staticmethod
    def _parse_config_value(value: str):
        if isinstance(value, str):
            if value.isdigit():
                return int(value)
            try:
                return float(value)
            except ValueError:
                pass
            if value.lower() in ("true", "false"):
                return value.lower() == "true"
        return value

    # =========================================================================
    # Alert Triage Task Implementation
    # =========================================================================

    def _reset_alert_triage(self, seed: Optional[int] = None) -> IncidentResponseObservation:
        """
        Reset into the alert_triage task.
        seed 0/None=AT-001, 1=AT-002, 2=AT-003 (cycles on seed % 3).
        """
        idx = (seed % len(ALERT_TRIAGE_SCENARIOS)) if seed is not None else 0
        sc = ALERT_TRIAGE_SCENARIOS[idx]

        self._task_name = "alert_triage"
        self._scenario_seed = seed
        self._at_scenario = sc
        self._at_submitted_severity = None
        self._step_number = 0
        self._done = False
        self._cumulative_reward = 0.0
        self._actions_taken = []
        self._last_action_error = None
        self._last_feedback = "Alert triage started. Investigate services, then submit_severity."
        self._consecutive_diagnosis_count = 0
        self._observation_loop_warned = False
        self._diagnosis_penalty_applied_at = -1
        # Clear SRE-specific state that doesn't apply here
        self._scenario = None  # type: ignore
        self._service_states = {}
        self._active_alerts = []
        self._services_investigated = {}
        self._root_causes_declared = []
        self._remediations_applied = []
        self._correct_root_causes_found = 0
        self._correct_remediations_found = 0
        self._incident_resolved = False
        self._collateral_degraded = set()
        self._degradation_warnings = []

        alert = sc.get("alert", {})
        symptoms_text = "\n".join(f"  - {s}" for s in alert.get("symptoms", []))
        known = ", ".join(sc.get("known_services", []))
        action_result = (
            f"ALERT TRIAGE: {sc['incident_summary']}\n\n"
            f"Alert ID:  {alert.get('id', 'N/A')}\n"
            f"Title:     {alert.get('title', 'N/A')}\n"
            f"Error rate: {alert.get('error_rate', 0)*100:.0f}%\n"
            f"Duration:  {alert.get('duration_minutes', 0)} min\n"
            f"Revenue impact: ${alert.get('revenue_impact_per_min', 0):,.0f}/min\n\n"
            f"Symptoms:\n{symptoms_text}\n\n"
            f"Available services: {known}\n\n"
            f"Available actions (max 3 steps):\n"
            f"  - investigate_logs: params={{\"keyword\": \"<optional>\"}}"
            f"  (service_name required)\n"
            f"  - check_metrics: no extra params\n"
            f"  - check_service_health: no extra params\n"
            f"  - run_diagnostic: no extra params\n"
            f"  - submit_severity: params={{\"severity\": \"P1|P2|P3|P4\"}}\n\n"
            f"Read the blast radius carefully.\n"
            f"High error rate ≠ P1 if there is graceful fallback and zero revenue impact.\n"
            f"Submit your classification with submit_severity."
        )
        return self._build_at_observation(action_result, True)

    def _step_alert_triage(self, action: IncidentResponseAction) -> Tuple[IncidentResponseObservation, float, bool, Dict[str, Any]]:
        """Handle one step inside the alert_triage task."""
        self._step_number += 1
        self._last_action_error = None
        sc = self._at_scenario
        at = action.action_type.value
        svc = (action.service_name or "").lower().strip()
        params = action.parameters
        reward = 0.0
        reward_breakdown: Dict[str, float] = {}

        # ── Investigation actions ─────────────────────────────────────
        if at in ("investigate_logs", "check_metrics", "check_service_health", "run_diagnostic"):
            known = [s.lower() for s in sc.get("known_services", [])]
            if not svc:
                action_result = f"Error: {at} requires a service_name."
                reward = -0.04
                reward_breakdown["no_service"] = reward
                self._last_action_error = action_result
            elif svc not in known:
                action_result = f"Unknown service '{svc}'. Known: {', '.join(known)}"
                reward = -0.05
                reward_breakdown["unknown_service"] = reward
                self._last_action_error = action_result
            else:
                tool_responses = sc.get("tool_responses", {})
                data = tool_responses.get(at, {}).get(svc)
                if data is None:
                    data = f"No {at} data available for '{svc}'."
                # Track investigation
                key = (at, svc)
                if svc not in self._services_investigated:
                    self._services_investigated[svc] = set()
                if key not in self._services_investigated.get(svc, set()):
                    self._services_investigated.setdefault(svc, set()).add(key)
                    reward = 0.04 if svc not in {s for s, _ in [k for k in [(s, a) for s, acts in self._services_investigated.items() for a in acts]]} else 0.02
                    reward = 0.04  # new service
                    reward_breakdown["new_investigation"] = reward
                else:
                    reward = -0.03
                    reward_breakdown["repeat"] = reward
                action_result = f"=== {at.replace('_', ' ').title()}: {svc} ===\n{data}"

        # ── Submit severity ─────────────────────────────────────────────
        elif at == "submit_severity":
            severity = str(params.get("severity", "")).upper().strip()
            if not severity:
                action_result = "Error: submit_severity requires params={\"severity\": \"P1|P2|P3|P4\"}"
                reward = -0.05
                reward_breakdown["no_severity"] = reward
                self._last_action_error = action_result
            else:
                self._at_submitted_severity = severity
                correct = sc.get("correct_severity", "P1")
                adjacent = set(sc.get("adjacent_severities", []))
                if severity == correct:
                    reward = 0.20
                    reward_breakdown["exact_severity"] = reward
                    action_result = f"Submitted severity: {severity} ✓ Correct!"
                elif severity in adjacent:
                    reward = 0.08
                    reward_breakdown["adjacent_severity"] = reward
                    action_result = f"Submitted severity: {severity} (adjacent — correct was {correct})"
                else:
                    reward = -0.10
                    reward_breakdown["wrong_severity"] = reward
                    action_result = f"Submitted severity: {severity} ✗ Incorrect (correct: {correct})"
                # Terminal — grader finalises
                self._done = True

        # ── Invalid / unsupported actions for this task ───────────────
        else:
            action_result = (
                f"Action '{at}' is not available in alert_triage. "
                f"Use: investigate_logs, check_metrics, check_service_health, "
                f"run_diagnostic, or submit_severity."
            )
            reward = -0.05
            reward_breakdown["invalid_action"] = reward
            self._last_action_error = action_result

        # ── Timeout ───────────────────────────────────────────────
        if self._step_number >= 3 and not self._done:
            reward -= 0.08
            reward_breakdown["timeout"] = -0.08
            self._done = True
            action_result += "\n\n⏰ Step budget exhausted (3 steps max). Episode ending."

        reward = max(-1.0, min(1.0, reward))
        self._cumulative_reward += reward

        # Track action
        action_str = at
        if svc:
            action_str += f"({svc})"
        self._actions_taken.append(action_str)

        self._last_feedback = self._build_at_feedback(reward, reward_breakdown)
        obs = self._build_at_observation(action_result, self._last_action_error is None)

        info = {
            "reward_breakdown": reward_breakdown,
            "step": self._step_number,
            "task": "alert_triage",
            "feedback": self._last_feedback,
            "submitted_severity": self._at_submitted_severity,
            "failure_type": "N/A",
            "observation_loop": False,
            "consecutive_diagnosis_count": 0,
        }
        return obs, reward, self._done, info

    def _get_alert_triage_score(self) -> dict:
        """Return the alert triage grader result for /score and /grader."""
        if not self._at_scenario:
            return {"final": 0.01, "feedback": "No alert_triage scenario loaded.",
                    "total": 0.01, "breakdown": {}}
        state_dict = {
            "agent_actions_taken": self._actions_taken,
            "step_number": self._step_number,
            "max_steps": 3,
        }
        result = _grade_alert_triage(state_dict, self._at_scenario)
        result["final"] = result["total"]
        result["failure_type"] = "Alert Triage"
        result["observation_loop"] = False
        return result

    def _build_at_observation(self, action_result: str, action_success: bool) -> IncidentResponseObservation:
        """Build an Observation for the alert_triage task (no service statuses, no scenario)."""
        sc = self._at_scenario or {}
        alert_data = sc.get("alert", {})
        severity_str = "high"
        try:
            alert_obj = Alert(
                alert_id=alert_data.get("id", "AT-UNKNOWN"),
                severity=AlertSeverity(severity_str),
                service=sc.get("known_services", [""])[0] if sc.get("known_services") else "unknown",
                message=alert_data.get("title", ""),
                timestamp="T+0min",
            )
            alerts = [alert_obj]
        except Exception:
            alerts = []

        known = sc.get("known_services", [])
        statuses = [
            ServiceStatus(name=svc, healthy=True, response_time_ms=None, error_rate=None)
            for svc in known
        ]
        at_actions = [
            "investigate_logs", "check_metrics", "check_service_health",
            "run_diagnostic", "submit_severity",
        ]
        return IncidentResponseObservation(
            action_result=action_result,
            action_success=action_success,
            active_alerts=alerts,
            service_statuses=statuses,
            available_services=known,
            available_actions=at_actions,
            step_number=self._step_number,
            max_steps=3,
            elapsed_time_minutes=self._step_number,
            incident_summary=sc.get("incident_summary", ""),
            task_name="alert_triage",
            task_difficulty="easy",
            degradation_warnings=[],
            feedback=self._last_feedback,
        )

    @staticmethod
    def _build_at_feedback(reward: float, breakdown: Dict[str, float]) -> str:
        sign = "+" if reward >= 0 else ""
        parts = [f"Alert triage: reward={sign}{reward:.3f}"]
        for key, val in breakdown.items():
            if key == "exact_severity":
                parts.append(f"Correct severity: +{val:.3f}")
            elif key == "adjacent_severity":
                parts.append(f"Adjacent severity (close): +{val:.3f}")
            elif key == "wrong_severity":
                parts.append(f"Wrong severity: {val:.3f}")
            elif key == "new_investigation":
                parts.append(f"New service investigated: +{val:.3f}")
            elif key == "repeat":
                parts.append(f"Repeated investigation: {val:.3f}")
            elif key == "timeout":
                parts.append("Step budget exhausted: -0.08")
        return " | ".join(parts)
