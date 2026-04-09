"""
FastAPI route definitions for the Incident Response environment.

Endpoints
---------
GET  /health      — liveness probe
GET  /tasks       — list available tasks
POST /reset       — start a new episode
POST /step        — execute one action
GET  /state       — full internal state snapshot
GET  /score       — 6D score breakdown
POST /grader      — standalone grader compatible with external evaluators
POST /baseline    — run the built-in heuristic agent over all tasks
"""
from __future__ import annotations

import sys
import os
import traceback
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Body

_API_DIR    = os.path.dirname(os.path.abspath(__file__))
_SERVER_DIR = os.path.dirname(_API_DIR)
_PROJ_ROOT  = os.path.dirname(_SERVER_DIR)
for _p in (_SERVER_DIR, _PROJ_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from state import env, _baseline_env
from models import ActionType, IncidentResponseAction
from scenarios.definitions import list_tasks
from api.models import ResetRequest, StepRequest, StepResponse

try:
    from graders import grade as _grade_from_state, _OBSERVATION_LOOP_CAP
except ImportError:
    def _grade_from_state(state): return {}  # type: ignore
    _OBSERVATION_LOOP_CAP = 0.45

ALL_TASKS = [
    "db_connection_failure",
    "cascading_service_timeout",
    "multi_factor_outage",
    "ssl_certificate_expiry",
    "database_deadlock",
    "alert_triage",
]

router = APIRouter()


@router.get("/health")
def health():
    return {"status": "ok", "environment": "incident-response-env", "version": "4.0.0"}


@router.get("/tasks")
def tasks():
    task_list = list_tasks()
    for t in task_list:
        t["num_scenarios"] = 3
    return {"tasks": task_list}


@router.post("/reset")
def reset(req: Optional[ResetRequest] = Body(default=None)):
    if req is None:
        req = ResetRequest()
    try:
        obs = env.reset(task_name=req.task_name, seed=req.seed)
        return {"observation": obs.model_dump()}
    except Exception as e:
        return {"error": str(e), "traceback": traceback.format_exc()}


@router.post("/step")
def step(req: StepRequest):
    try:
        action = IncidentResponseAction(
            action_type=ActionType(req.action_type),
            service_name=req.service_name,
            parameters=req.parameters,
        )
        obs, reward, done, info = env.step(action)
        return StepResponse(
            observation=obs.model_dump(),
            reward=round(reward, 4),
            done=done,
            info=info,
        ).model_dump()
    except Exception as e:
        return {"error": str(e), "traceback": traceback.format_exc()}


@router.get("/state")
def state():
    try:
        s = env.state()
        return {"state": s.model_dump()}
    except Exception as e:
        return {"error": str(e), "traceback": traceback.format_exc()}


@router.get("/score")
def score():
    """
    Returns the 6D score breakdown including sequence score and failure_type.
    Score is clamped to open interval (0.01, 0.99). Hard-capped at 0.45 if
    an observation loop is detected and the incident remains unresolved.
    """
    try:
        breakdown = env.get_score_breakdown()
        return {
            "score":             breakdown["final"],
            "breakdown":         breakdown,
            "feedback":          breakdown.get("feedback", ""),
            "failure_type":      breakdown.get("failure_type", "Unknown"),
            "observation_loop":  breakdown.get("observation_loop", False),
            "sequence":          breakdown.get("sequence", 0.0),
            "task_name":         env._task_name or "",
            "done":              env._done,
            "collateral_degraded": list(env._collateral_degraded),
        }
    except Exception as e:
        return {"error": str(e)}


@router.post("/grader")
def grader(req: Dict[str, Any] = {}):
    """
    Standalone grader compatible with external trajectory evaluators.
    Returns full 6D breakdown including sequence score and failure type.
    Score is guaranteed to be in the open interval (0.01, 0.99) and
    hard-capped at 0.45 for observation-loop agents.
    """
    try:
        breakdown  = env.get_score_breakdown()
        state_dict = env.state().model_dump()
        state_dict["_grade_components"] = breakdown.get("_grade_components", {})
        state_dict["incident_resolved"] = env._incident_resolved

        external = _grade_from_state(state_dict)

        return {
            "score":   breakdown["final"],
            "breakdown": {
                "root_cause":    breakdown["root_cause"],
                "remediation":   breakdown["remediation"],
                "investigation": breakdown["investigation"],
                "efficiency":    breakdown["efficiency"],
                "safety":        breakdown["safety"],
                "sequence":      breakdown.get("sequence", 0.0),
            },
            "components_weighted": external.get("components_weighted", {}),
            "feedback":         external.get("feedback", breakdown.get("feedback", "")),
            "failure_type":     external.get("failure_type", breakdown.get("failure_type", "Unknown")),
            "observation_loop": external.get("observation_loop", breakdown.get("observation_loop", False)),
            "task_name":        env._task_name or "",
            "scenario_seed":    env._scenario_seed,
        }
    except Exception as e:
        return {"error": str(e), "traceback": traceback.format_exc()}


@router.post("/baseline")
def baseline(req: Dict[str, Any] = {}):
    """
    Runs the built-in heuristic agent over all tasks (or a single task).
    Returns per-task scores plus average, useful for sanity-checking the
    environment reward signal.
    """
    task_to_run  = req.get("task_name")
    tasks_to_run = [task_to_run] if task_to_run else ALL_TASKS
    results: List[Dict[str, Any]] = []

    for task_name in tasks_to_run:
        try:
            obs       = _baseline_env.reset(task_name=task_name)
            step_count = 0
            max_steps  = obs.max_steps
            services   = obs.available_services

            # Phase 1: check service health for each service
            for svc in services:
                if step_count >= max_steps - 5:
                    break
                action = IncidentResponseAction(
                    action_type=ActionType.CHECK_SERVICE_HEALTH,
                    service_name=svc, parameters={}
                )
                obs, _, done, _ = _baseline_env.step(action)
                step_count += 1
                if done:
                    break

            # Phase 2: investigate logs
            for svc in services:
                if step_count >= max_steps - 4:
                    break
                action = IncidentResponseAction(
                    action_type=ActionType.INVESTIGATE_LOGS,
                    service_name=svc, parameters={}
                )
                obs, _, done, _ = _baseline_env.step(action)
                step_count += 1
                if done:
                    break

            # Phase 3: read config
            for svc in services:
                if step_count >= max_steps - 3:
                    break
                action = IncidentResponseAction(
                    action_type=ActionType.READ_CONFIG,
                    service_name=svc, parameters={}
                )
                obs, _, done, _ = _baseline_env.step(action)
                step_count += 1
                if done:
                    break

            # Phase 4: declare root cause
            if not _baseline_env._incident_resolved and step_count < max_steps - 1:
                scenario = _baseline_env._scenario
                if scenario and scenario.root_causes:
                    action = IncidentResponseAction(
                        action_type=ActionType.DECLARE_ROOT_CAUSE,
                        service_name=None,
                        parameters={"cause": scenario.root_causes[0]}
                    )
                    obs, _, done, _ = _baseline_env.step(action)
                    step_count += 1

            # Phase 5: apply first correct remediation
            if not _baseline_env._incident_resolved and step_count < max_steps:
                scenario = _baseline_env._scenario
                if scenario and scenario.correct_remediations:
                    rem = scenario.correct_remediations[0]
                    action = IncidentResponseAction(
                        action_type=ActionType(rem["action_type"]),
                        service_name=rem.get("service_name"),
                        parameters=rem.get("parameters", {})
                    )
                    obs, _, done, _ = _baseline_env.step(action)
                    step_count += 1

            breakdown = _baseline_env.get_score_breakdown()
            results.append({
                "task_name":        task_name,
                "score":            breakdown["final"],
                "breakdown":        {k: v for k, v in breakdown.items()
                                     if k not in ("final", "feedback", "_grade_components")},
                "feedback":         breakdown.get("feedback", ""),
                "failure_type":     breakdown.get("failure_type", "Unknown"),
                "observation_loop": breakdown.get("observation_loop", False),
                "steps":            step_count,
                "resolved":         _baseline_env._incident_resolved,
            })
        except Exception as e:
            results.append({
                "task_name":    task_name,
                "score":        0.01,
                "error":        str(e),
                "steps":        0,
                "resolved":     False,
                "failure_type": "Unknown",
            })

    avg = sum(r["score"] for r in results) / max(len(results), 1)
    return {
        "tasks":         results,
        "average_score": round(avg, 4),
        "num_tasks":     len(results),
    }
