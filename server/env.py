"""
Incident Response Environment — Core Logic

Implements the OpenEnv interface: reset(), step(), state()
with rich reward shaping, multi-task support, and deterministic grading.
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
from scenarios.definitions import ScenarioDef, get_scenario, list_tasks


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

    # ── OpenEnv Interface ─────────────────────────────────────────────────

    def reset(self, task_name: str = "db_connection_failure") -> IncidentResponseObservation:
        """Reset the environment with a specific task."""
        self._scenario = get_scenario(task_name)
        self._task_name = task_name
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
                f"Action format: {{\"action_type\": \"<type>\", \"service_name\": \"<name>\", \"parameters\": {{...}}}}\n"
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

        self._step_number += 1
        self._last_action_error = None

        action_result, action_success, reward, reward_breakdown = self._process_action(action)

        action_desc = f"{action.action_type.value}"
        if action.service_name:
            action_desc += f"({action.service_name})"
        self._actions_taken.append(action_desc)

        self._check_resolution()

        if self._step_number >= self._scenario.max_steps:
            self._done = True
            if not self._incident_resolved:
                reward -= 0.05
                reward_breakdown["timeout_penalty"] = -0.05

        if self._incident_resolved:
            steps_ratio = self._step_number / self._scenario.max_steps
            time_bonus = 0.10 if steps_ratio < 0.5 else (0.05 if steps_ratio < 0.75 else 0.0)
            reward += time_bonus
            reward_breakdown["time_bonus"] = time_bonus
            self._done = True

        reward = max(-1.0, min(1.0, reward))
        self._cumulative_reward += reward

        obs = self._build_observation(action_result, action_success)
        info = {
            "reward_breakdown": reward_breakdown,
            "step": self._step_number,
            "incident_resolved": self._incident_resolved,
            "last_action_error": self._last_action_error,
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
        )

    def get_score(self) -> float:
        """
        Compute final grader score in [0.0, 1.0].

        Scoring weights:
          - Root cause identification: 40%
          - Remediation quality: 35%
          - Investigation thoroughness: 15%
          - Efficiency (fewer steps): 10%
        """
        if not self._scenario:
            return 0.0

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

        final = (root_cause_score * 0.40 + remediation_score * 0.35 +
                 investigation_score * 0.15 + efficiency_score * 0.10)

        return round(max(0.0, min(1.0, final)), 4)

    def close(self):
        pass

    # ── Action Processing ─────────────────────────────────────────────────

    def _process_action(self, action: IncidentResponseAction) -> Tuple[str, bool, float, Dict[str, float]]:
        reward = 0.0
        breakdown: Dict[str, float] = {}
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
        """Track investigation, return (is_new, reward)."""
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
        cause = params.get("cause", "")
        if not cause:
            self._last_action_error = "declare_root_cause needs 'cause' param"
            return self._last_action_error, False, -0.02, {"invalid_params": -0.02}
        self._root_causes_declared.append(cause)
        for i, keywords in enumerate(self._scenario.root_cause_keywords):
            cause_lower = cause.lower()
            match_count = sum(1 for kw in keywords if kw.lower() in cause_lower)
            if match_count >= 2:
                already = any(
                    sum(1 for kw in keywords if kw.lower() in p.lower()) >= 2
                    for p in self._root_causes_declared[:-1]
                )
                if not already:
                    self._correct_root_causes_found += 1
                    return f"ROOT CAUSE: '{cause}' ✓ Matches known cause!", True, 0.20, {"correct_root_cause": 0.20}
                return f"ROOT CAUSE: '{cause}' — already identified.", True, -0.02, {"duplicate": -0.02}
        return f"ROOT CAUSE: '{cause}' ✗ Does not match. Keep investigating.", True, -0.05, {"wrong_cause": -0.05}

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
