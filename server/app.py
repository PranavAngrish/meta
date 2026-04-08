"""
FastAPI server exposing the Incident Response environment via HTTP.

v4.0 UI additions (ported & enhanced from cloud-incident-response):
  - Rich Markdown observation panel: service health table, alert badges, degradation warnings
  - Dedicated per-action parameter inputs (no raw JSON required)
  - Live action history table with step, icon, action, service, per-step reward
  - Episode state panel: progress bar, diagnosis streak, collateral degraded count
  - 6D score breakdown table with mini progress bars per dimension
  - Failure-type badge with colour-coded icons (Efficient Reasoner / Obs Loop / etc.)
  - Styled CSS sections, pulsing alert dot, Inter font, light-mode enforcement

v3.0 additions:
  - /score and /grader now return failure_type, sequence score, observation_loop flag
  - /grader feedback includes observation loop cap notice

v2.1 additions:
  - POST /baseline  — runs all 5 tasks with built-in heuristic, returns scores
  - POST /grader    — returns full 6D breakdown via standalone graders.py
  - GET  /score     — now includes per-step feedback and open-interval clamping
"""

from __future__ import annotations

import os
import sys
import json
import traceback
from typing import Any, Dict, List, Optional

import gradio as gr
from fastapi import Body, FastAPI
from fastapi.middleware.cors import CORSMiddleware
from uvicorn.middleware.proxy_headers import ProxyHeadersMiddleware
from pydantic import BaseModel

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from env import IncidentResponseEnv
from models import ActionType, IncidentResponseAction
from scenarios.definitions import list_tasks

# Import standalone grader
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
try:
    from graders import grade as _grade_from_state, _OBSERVATION_LOOP_CAP
except ImportError:
    def _grade_from_state(state): return {}  # type: ignore
    _OBSERVATION_LOOP_CAP = 0.45

app = FastAPI(title="Incident Response Environment", version="4.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
app.add_middleware(ProxyHeadersMiddleware, trusted_hosts="*")

env = IncidentResponseEnv()
_baseline_env = IncidentResponseEnv()

ALL_TASKS = [
    "db_connection_failure",
    "cascading_service_timeout",
    "multi_factor_outage",
    "ssl_certificate_expiry",
    "database_deadlock",
    "alert_triage",
]


class ResetRequest(BaseModel):
    task_name: str = "db_connection_failure"
    seed: Optional[int] = None


class StepRequest(BaseModel):
    action_type: str
    service_name: Optional[str] = None
    parameters: Dict[str, Any] = {}


class StepResponse(BaseModel):
    observation: dict
    reward: float
    done: bool
    info: dict


@app.get("/health")
def health():
    return {"status": "ok", "environment": "incident-response-env", "version": "4.0.0"}


@app.get("/tasks")
def tasks():
    task_list = list_tasks()
    for t in task_list:
        t["num_scenarios"] = 3
    return {"tasks": task_list}


@app.post("/reset")
def reset(req: Optional[ResetRequest] = Body(default=None)):
    if req is None:
        req = ResetRequest()
    try:
        obs = env.reset(task_name=req.task_name, seed=req.seed)
        return {"observation": obs.model_dump()}
    except Exception as e:
        return {"error": str(e), "traceback": traceback.format_exc()}


@app.post("/step")
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


@app.get("/state")
def state():
    try:
        s = env.state()
        return {"state": s.model_dump()}
    except Exception as e:
        return {"error": str(e), "traceback": traceback.format_exc()}


@app.get("/score")
def score():
    """
    v3.0: Returns 6D score breakdown including sequence score and failure_type.
    Score is clamped to open interval (0.01, 0.99). Hard-capped at 0.45 if
    observation loop detected and incident unresolved.
    """
    try:
        breakdown = env.get_score_breakdown()
        return {
            "score":            breakdown["final"],
            "breakdown":        breakdown,
            "feedback":         breakdown.get("feedback", ""),
            "failure_type":     breakdown.get("failure_type", "Unknown"),
            "observation_loop": breakdown.get("observation_loop", False),
            "sequence":         breakdown.get("sequence", 0.0),
            "task_name":        env._task_name or "",
            "done":             env._done,
            "collateral_degraded": list(env._collateral_degraded),
        }
    except Exception as e:
        return {"error": str(e)}


@app.post("/grader")
def grader(req: Dict[str, Any] = {}):
    """
    Standalone grader endpoint — compatible with external trajectory evaluators.
    v3.0: Returns full 6D breakdown (adds sequence + failure_type).
    Score guaranteed to be in open interval (0.01, 0.99).
    Hard-capped at 0.45 for observation-loop agents.
    """
    try:
        breakdown = env.get_score_breakdown()
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


@app.post("/baseline")
def baseline(req: Dict[str, Any] = {}):
    """
    Baseline endpoint — runs a built-in heuristic agent over all 5 tasks.
    v3.0: Baseline result now includes failure_type and sequence score per task.
    """
    task_to_run = req.get("task_name")
    tasks_to_run = [task_to_run] if task_to_run else ALL_TASKS
    results: List[Dict[str, Any]] = []

    for task_name in tasks_to_run:
        try:
            obs = _baseline_env.reset(task_name=task_name)
            step_count = 0
            max_steps = obs.max_steps
            services = obs.available_services

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

            if not _baseline_env._incident_resolved and step_count < max_steps - 1:
                scenario = _baseline_env._scenario
                if scenario and scenario.root_causes:
                    first_cause = scenario.root_causes[0]
                    action = IncidentResponseAction(
                        action_type=ActionType.DECLARE_ROOT_CAUSE,
                        service_name=None,
                        parameters={"cause": first_cause}
                    )
                    obs, _, done, _ = _baseline_env.step(action)
                    step_count += 1

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
                "task_name": task_name,
                "score": 0.01,
                "error": str(e),
                "steps": 0,
                "resolved": False,
                "failure_type": "Unknown",
            })

    avg = sum(r["score"] for r in results) / max(len(results), 1)
    return {
        "tasks":         results,
        "average_score": round(avg, 4),
        "num_tasks":     len(results),
    }


# =============================================================================
# Gradio Web UI  v4.0
# Enhanced UI inspired by cloud-incident-response, built for meta-hackathon's
# richer 6D scoring, dynamic degradation, and failure-type classification.
# =============================================================================

# ── Action category helpers ───────────────────────────────────────────────────
_DIAG_ACTIONS = {
    "investigate_logs", "check_metrics", "read_config",
    "check_service_health", "run_diagnostic",
}
_FIX_ACTIONS = {
    "restart_service", "update_config", "rollback_deployment", "scale_service",
}

_FAILURE_TYPE_ICON = {
    "Efficient Reasoner":        "🟢",
    "Symptom Chaser":            "🟡",
    "Lucky Guesser":             "🟠",
    "Stuck in Observation Loop": "🔴",
    "Late Corrector":            "🔵",
    "Unknown":                   "⚪",
}

_DIFFICULTY_BADGE = {
    "db_connection_failure":     "🟢 Easy",
    "cascading_service_timeout": "🟡 Medium",
    "ssl_certificate_expiry":    "🟡 Medium",
    "multi_factor_outage":       "🔴 Hard",
    "database_deadlock":         "🔴 Hard",
    "alert_triage":              "🔵 Easy (Triage)",
}

# Per-episode in-memory action history (cleared on reset)
_action_history: List[Dict[str, Any]] = []


# ── Markdown formatters ───────────────────────────────────────────────────────

def _fmt_obs(obs_dump: dict, action_result: str) -> str:
    """Build the rich observation panel shown after every step."""
    lines = []
    task = obs_dump.get("task_name", "—")
    diff = _DIFFICULTY_BADGE.get(task, "")
    step = obs_dump.get("step_number", 0)
    max_s = obs_dump.get("max_steps", 30)
    pct = int(step / max(max_s, 1) * 100)
    bar_filled = pct // 5
    bar = "█" * bar_filled + "░" * (20 - bar_filled)

    lines.append(f"### 📋 Task: `{task}` {diff}")
    lines.append(f"**Progress:** {step}/{max_s}  `{bar}` {pct}%")

    # Active alerts
    alerts = obs_dump.get("active_alerts", [])
    if alerts:
        lines.append("\n#### 🔔 Active Alerts")
        for a in alerts:
            sev = a.get("severity", "medium").upper()
            sev_icon = "🔴" if sev in ("CRITICAL", "HIGH") else ("🟡" if sev == "MEDIUM" else "🟢")
            lines.append(
                f"{sev_icon} `[{a.get('alert_id', '')}]` "
                f"**{a.get('service', '')}** — {a.get('message', '')} "
                f"*(T={a.get('timestamp', '')})*"
            )

    # Service health table
    statuses = obs_dump.get("service_statuses", [])
    if statuses:
        lines.append("\n#### 🖥️ Service Health")
        lines.append("| Service | Status | RT (ms) | Err% |")
        lines.append("|---|---|---|---|")
        for s in statuses:
            icon = "✅" if s.get("healthy") else "❌"
            rt_raw = s.get("response_time_ms")
            rt = f"{rt_raw:.0f}" if rt_raw is not None else "—"
            er_raw = s.get("error_rate")
            er = f"{er_raw * 100:.1f}%" if er_raw is not None else "—"
            lines.append(f"| `{s['name']}` | {icon} | {rt} | {er} |")

    # Degradation warnings (last 3)
    warn = obs_dump.get("degradation_warnings", [])
    if warn:
        lines.append("\n#### ⚠️ Degradation Warnings")
        for w in warn[-3:]:
            lines.append(f"- {w}")

    # Last action output
    if action_result:
        lines.append("\n---\n#### 📄 Last Action Output")
        preview = action_result[:1400]
        if len(action_result) > 1400:
            preview += "\n…(truncated)"
        lines.append(f"```\n{preview}\n```")

    return "\n".join(lines)


def _fmt_history(history: List[Dict[str, Any]]) -> str:
    """Format action history as a Markdown table with step, type icon, action, service, reward."""
    if not history:
        return "*No actions yet.*"
    lines = ["| Step | Type | Action | Service | Reward |"]
    lines.append("|:---:|:---:|---|---|---:|")
    for h in history:
        act = h.get("action", "")
        icon = "🔍" if act in _DIAG_ACTIONS else ("🔧" if act in _FIX_ACTIONS else "📝")
        r = h.get("reward", 0.0)
        r_str = f"+{r:.3f}" if r >= 0 else f"{r:.3f}"
        svc = h.get("service") or "—"
        lines.append(f"| {h['step']} | {icon} | `{act}` | `{svc}` | {r_str} |")
    return "\n".join(lines)


def _fmt_score_alert_triage(result: dict) -> str:
    """Render the alert triage grader result (severity score + investigation bonus)."""
    total = result.get("total", result.get("final", 0.0))
    bd = result.get("breakdown", {})
    fb = result.get("feedback", "")
    score_icon = "🟢" if total >= 0.8 else ("🟡" if total >= 0.5 else "🔴")

    submitted = bd.get("submitted_severity", "—")
    correct = bd.get("correct_severity", "—")
    sev_match = bd.get("severity_match", 0.0)
    inv_bonus = bd.get("investigation_bonus", 0.0)
    svcs_inv = bd.get("services_investigated", 0)
    act_types = bd.get("action_types_used", 0)

    lines = [
        f"### {score_icon} Alert Triage Score: **{total:.4f}** / 1.0",
        "",
        "| Component | Value |",
        "|---|---|",
        f"| **Submitted Severity** | `{submitted}` |",
        f"| **Correct Severity** | `{correct}` |",
        f"| **Severity Score** | `{sev_match:.2f}` (1.0 exact · 0.5 adjacent · 0.25 two-off) |",
        f"| **Investigation Bonus** | `+{inv_bonus:.2f}` ({svcs_inv} svcs · {act_types} action types) |",
        f"| **Total** | `{total:.4f}` |",
    ]
    if fb:
        lines.append(f"\n---\n> {fb}")
    return "\n".join(lines)


def _fmt_score(breakdown: dict) -> str:
    """Render score breakdown. Routes to alert-triage formatter when appropriate."""
    if not breakdown:
        return "*No score yet. Execute at least one step, then click Grade (6D).*"

    # Alert triage returns a different breakdown structure
    if breakdown.get("failure_type") == "Alert Triage" or "severity_match" in breakdown.get("breakdown", {}):
        return _fmt_score_alert_triage(breakdown)

    final = breakdown.get("final", 0.0)
    ft = breakdown.get("failure_type", "Unknown")
    ft_icon = _FAILURE_TYPE_ICON.get(ft, "⚪")
    obs_loop = breakdown.get("observation_loop", False)
    score_icon = "🟢" if final >= 0.7 else ("🟡" if final >= 0.4 else "🔴")

    lines = [f"### {score_icon} Score: **{final:.4f}** / 1.0  *(open interval 0.01–0.99)*"]
    lines.append(f"\n{ft_icon} **Failure Type:** {ft}")

    if obs_loop:
        lines.append(f"\n> ⚠️ *Observation loop detected — score hard-capped at {_OBSERVATION_LOOP_CAP}*")

    lines.append("\n#### 6D Breakdown")
    lines.append("| Dimension | Score | Weight | Weighted | Bar |")
    lines.append("|---|---:|---:|---:|---|")
    dims = [
        ("root_cause",    "Root Cause",    0.30),
        ("remediation",   "Remediation",   0.25),
        ("investigation", "Investigation", 0.15),
        ("efficiency",    "Efficiency",    0.10),
        ("safety",        "Safety",        0.10),
        ("sequence",      "Sequence",      0.10),
    ]
    for key, label, weight in dims:
        val = breakdown.get(key, 0.0)
        weighted = val * weight
        filled = int(val * 10)
        mini_bar = "▓" * filled + "░" * (10 - filled)
        lines.append(f"| {label} | {val:.2f} | ×{weight:.2f} | {weighted:.3f} | `{mini_bar}` |")

    fb = breakdown.get("feedback", "")
    if fb:
        lines.append(f"\n---\n> {fb}")

    return "\n".join(lines)


def _fmt_state_panel() -> str:
    """Compact episode state panel: task, progress, reward, resolution, streaks."""
    # Alert triage has no _scenario object — use dedicated path
    if env._task_name == "alert_triage":
        task = "alert_triage"
        diff = _DIFFICULTY_BADGE.get(task, "")
        step = env._step_number
        done = env._done
        cum_r = round(env._cumulative_reward, 4)
        status = "🏁 Done" if done else "⚡ Active"
        pct = int(step / 3 * 100)
        bar_filled = pct // 5
        bar = "█" * bar_filled + "░" * (20 - bar_filled)
        submitted = env._at_submitted_severity or "—"
        lines = [
            f"### {status}",
            "",
            "| Field | Value |",
            "|---|---|",
            f"| **Task** | `{task}` {diff} |",
            f"| **Progress** | {step}/3  `{bar}` {pct}% |",
            f"| **Cum. Reward** | `{cum_r:+.4f}` |",
            f"| **Submitted Severity** | `{submitted}` |",
        ]
        return "\n".join(lines)

    if not env._scenario:
        return "### ⏳ Ready\n\nSelect task → Reset → Begin"

    task = env._task_name or "—"
    diff = _DIFFICULTY_BADGE.get(task, "")
    step = env._step_number
    max_s = env._scenario.max_steps
    resolved = env._incident_resolved
    done = env._done
    cum_r = round(env._cumulative_reward, 4)
    status = "🏁 Resolved" if resolved else ("🏁 Done" if done else "⚡ Active")

    pct = int(step / max(max_s, 1) * 100)
    bar_filled = pct // 5
    bar = "█" * bar_filled + "░" * (20 - bar_filled)
    diag_streak = env._consecutive_diagnosis_count
    streak_warn = " ⚠️" if diag_streak >= 2 else ""
    collateral_count = len(env._collateral_degraded)

    lines = [
        f"### {status}",
        "",
        "| Field | Value |",
        "|---|---|",
        f"| **Task** | `{task}` {diff} |",
        f"| **Progress** | {step}/{max_s}  `{bar}` {pct}% |",
        f"| **Cum. Reward** | `{cum_r:+.4f}` |",
        f"| **Resolved** | {'✅' if resolved else '❌'} |",
        f"| **Diag Streak** | {diag_streak}{streak_warn} |",
        f"| **Collateral Degraded** | {collateral_count} service(s) |",
    ]
    if env._collateral_degraded:
        degraded_list = ", ".join(sorted(env._collateral_degraded))
        lines.append(f"| **Degraded Services** | `{degraded_list}` |")

    return "\n".join(lines)


# ── Gradio callbacks ──────────────────────────────────────────────────────────

def gr_reset(task_name: str, seed_str: str):
    global _action_history
    _action_history = []
    try:
        seed = int(seed_str) if seed_str.strip() else None
        obs = env.reset(task_name=task_name, seed=seed)
        obs_dump = obs.model_dump()
        services = obs.available_services
        return (
            _fmt_obs(obs_dump, obs.action_result),
            _fmt_history([]),
            _fmt_state_panel(),
            "*Start an episode first.*",
            "*Click Grade (6D) after executing actions.*",
            gr.Dropdown(choices=services, value=services[0] if services else None),
        )
    except Exception as e:
        err = f"❌ **Error:** {e}"
        return err, "", err, "", "", gr.Dropdown(choices=[])


def gr_step(
    action_type: str,
    service_name: str,
    keyword: str,
    config_key: str,
    config_val: str,
    replicas_str: str,
    cause_text: str,
    severity_val: str,
):
    global _action_history
    try:
        # Build params from dedicated fields — no raw JSON needed
        params: Dict[str, Any] = {}
        at = action_type

        if at == "investigate_logs" and keyword.strip():
            params["keyword"] = keyword.strip()
        elif at == "check_metrics":
            params["metric_type"] = "all"
        elif at == "update_config":
            if config_key.strip():
                params["key"] = config_key.strip()
            if config_val.strip():
                params["value"] = config_val.strip()
        elif at == "scale_service" and replicas_str.strip():
            try:
                params["replicas"] = int(replicas_str.strip())
            except ValueError:
                pass
        elif at == "declare_root_cause" and cause_text.strip():
            params["cause"] = cause_text.strip()
        elif at == "submit_severity" and severity_val.strip():
            params["severity"] = severity_val.strip()

        svc = service_name if service_name and service_name.strip() else None
        action = IncidentResponseAction(
            action_type=ActionType(at),
            service_name=svc,
            parameters=params,
        )
        obs, reward, done, info = env.step(action)
        obs_dump = obs.model_dump()

        # Append to in-memory history
        _action_history.append({
            "step":    obs.step_number,
            "action":  at,
            "service": svc,
            "reward":  reward,
        })

        # Reward panel
        ft = info.get("failure_type", "N/A")
        ft_icon = _FAILURE_TYPE_ICON.get(ft, "⚪")
        r_sign = "+" if reward >= 0 else ""
        reward_parts = [
            f"### Step reward: `{r_sign}{reward:.4f}`",
            f"**Cumulative:** `{env._cumulative_reward:+.4f}`",
            f"**Feedback:** {obs.feedback}",
            (
                f"{ft_icon} **Failure Type:** {ft}  |  "
                f"Obs Loop: `{info.get('observation_loop', False)}`  |  "
                f"Diag Streak: `{info.get('consecutive_diagnosis_count', 0)}`"
            ),
        ]
        if done:
            reward_parts.append(
                "\n---\n🏁 **Episode complete** — click **📊 Grade (6D)** for final score"
            )
        reward_md = "\n\n".join(reward_parts)

        return (
            _fmt_obs(obs_dump, obs.action_result),
            _fmt_history(_action_history),
            _fmt_state_panel(),
            reward_md,
        )
    except Exception as e:
        err = f"❌ **Error:** {e}"
        return err, "", "", err


def gr_grade():
    try:
        breakdown = env.get_score_breakdown()
        return _fmt_score(breakdown)
    except Exception as e:
        return f"❌ {e}"


def gr_state():
    return _fmt_state_panel()


# ── CSS & JS ──────────────────────────────────────────────────────────────────

CUSTOM_CSS = """
:root, html, body, .gradio-container { color-scheme: light !important; }
body.dark, html.dark, .dark {
    color-scheme: light !important;
    --body-background-fill: #ffffff !important;
    --background-fill-primary: #ffffff !important;
    --background-fill-secondary: #f8fafc !important;
}
.gradio-container {
    background: #ffffff !important;
    max-width: 1500px !important;
    margin: 0 auto !important;
}
.env-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 18px 16px;
    border-bottom: 2px solid #fecaca;
    margin-bottom: 18px;
    background: linear-gradient(135deg, #fff5f5, #ffffff);
    border-radius: 12px 12px 0 0;
}
.env-header-left {
    display: flex;
    align-items: center;
    gap: 12px;
    font-size: 1.4rem;
    font-weight: 800;
    color: #0f172a;
}
.env-header-dot {
    width: 13px;
    height: 13px;
    border-radius: 50%;
    background: #ef4444;
    box-shadow: 0 0 0 0 rgba(239, 68, 68, 0.5);
    animation: pulse-dot 2s ease-in-out infinite;
}
@keyframes pulse-dot {
    0%, 100% { box-shadow: 0 0 0 0 rgba(239, 68, 68, 0.4); }
    50%       { box-shadow: 0 0 0 8px rgba(239, 68, 68, 0); }
}
.env-header-right {
    font-size: 0.82rem;
    font-weight: 600;
    color: #94a3b8;
    text-transform: uppercase;
    letter-spacing: 0.08em;
}
.section-title {
    font-weight: 700;
    font-size: 0.82rem;
    color: #1e293b;
    margin: 14px 0 6px;
    padding: 5px 10px;
    background: #f1f5f9;
    border-radius: 5px;
    border-left: 3px solid #ef4444;
    text-transform: uppercase;
    letter-spacing: 0.06em;
}
.param-hint {
    font-size: 0.75rem;
    color: #64748b;
    margin: 3px 0 4px;
    padding: 4px 8px;
    background: #f8fafc;
    border-radius: 4px;
    border-left: 2px solid #cbd5e1;
}
"""

FORCE_LIGHT_JS = """
function() {
    document.body.classList.remove('dark');
    document.documentElement.classList.remove('dark');
    document.documentElement.style.setProperty('color-scheme', 'light');
}
"""

# ── Build Gradio Blocks ───────────────────────────────────────────────────────

with gr.Blocks(
    title="Incident Response Environment v4.0",
    css=CUSTOM_CSS,
    js=FORCE_LIGHT_JS,
    theme=gr.themes.Soft(
        primary_hue="red",
        neutral_hue="slate",
        font=gr.themes.GoogleFont("Inter"),
    ),
) as web_ui:

    gr.HTML("""
    <div class="env-header">
        <div class="env-header-left">
            <span class="env-header-dot"></span>
            🚨 Incident Response Environment
        </div>
        <span class="env-header-right">OpenEnv · v4.0 · 6D Scoring · Anti-Reward-Hacking</span>
    </div>
    """)


    # ════════════════════════════════════════════════════════════════════════════
    # ONBOARDING & EDUCATION TABS
    # ════════════════════════════════════════════════════════════════════════════

    with gr.Tabs():
        # ─────────────────────────────────────────────────────────────────────────
        # TAB 1: WELCOME & ONBOARDING
        # ─────────────────────────────────────────────────────────────────────────
        with gr.TabItem("🎯 Welcome & What Is This?", id="welcome"):
            with gr.Column():
                gr.Markdown("""
# 🚨 Welcome to Incident Response Environment

## What Is This System?

This is an **AI training environment** that teaches artificial intelligence systems how to respond to real-world **production incidents** — those 3 AM emergency calls when something breaks in a live service that millions of people rely on.

Think of it like a **flight simulator for SREs** (Site Reliability Engineers) — a safe place to practice diagnosing and fixing problems without consequences.

### Real-World Scenario

Imagine you're an on-call engineer and:
- 🔴 **ALERT**: Your payment system is returning errors
- 😰 Users can't check out
- 💸 You're losing $1,000s per minute
- ⏰ You have 30 minutes to fix it

**What do you do?**

1. **Investigate**: Check service logs, metrics, configurations
2. **Diagnose**: Figure out what's actually broken (root cause)
3. **Fix**: Apply the right remedy
4. **Verify**: Confirm the issue is resolved

This system simulates exactly that scenario.

---

## Key Concepts

### 🔍 Investigation
You investigate services to gather evidence:
- **Check logs**: See what the service is reporting
- **Check metrics**: Response time, error rates, CPU usage
- **Read config**: Check if configurations are correct
- **Check health**: Is this service up or down?

### 🎯 Diagnosis
Once you've gathered evidence, you declare what you think the root cause is:
- The database port is misconfigured
- A certificate expired
- A recent deployment introduced a bug
- A query is running in a loop

### 🔧 Remediation
You apply a fix:
- Update a configuration
- Restart a service
- Rollback a deployment
- Scale up replicas

### 📊 Scoring
The system grades you on 6 dimensions:
- **Root Cause Accuracy** (30%): Did you identify the right problem?
- **Remediation** (25%): Did you apply the right fix?
- **Investigation** (15%): Did you check all necessary services?
- **Efficiency** (10%): Did you solve it quickly?
- **Safety** (10%): Did you avoid making things worse?
- **Sequence** (10%): Did you diagnose before fixing?

---

## 🚀 Quick Start in 3 Steps

### Step 1️⃣: Pick a Scenario
Start with an **Easy** scenario to learn the basics:
- 🟢 **DB Connection Failure** — Single clear problem, straightforward fix

### Step 2️⃣: Start an Episode
Click **🔄 Reset** to launch the scenario. You'll see:
- Active alerts (what went wrong)
- Service health (which services are broken)
- Available services (what you can investigate)

### Step 3️⃣: Investigate → Diagnose → Fix
Click **▶️ Execute Action** repeatedly:
- Use 🔍 investigation actions to gather clues
- Use 📝 `declare_root_cause` when you know what it is
- Use 🔧 fix actions to apply the remedy

---

## Ready to Learn?

👉 Head to the **"🎓 Learn By Example"** tab to see a complete walkthrough!

Or jump straight to the sandbox in the **"⚙️ Interactive Sandbox"** tab to start playing.
                """)

        # ─────────────────────────────────────────────────────────────────────────
        # TAB 2: LEARN BY EXAMPLE
        # ─────────────────────────────────────────────────────────────────────────
        with gr.TabItem("🎓 Learn By Example", id="tutorial"):
            with gr.Column():
                gr.Markdown("""
# 🎓 Step-by-Step Example: Database Connection Failure

## Scenario Overview
**Task**: `db_connection_failure`  
**Difficulty**: 🟢 Easy  
**Time Budget**: 30 steps  
**Root Cause**: Database connection port is misconfigured

---

## Phase 1: Initial Alert 🚨

When you reset, you see:
```
Active Alerts:
  🔴 [ALT-001] CRITICAL — user-api at 2025-04-09 08:15:23
     Message: user-api returning 503 errors

Service Health:
  ❌ user-api       — 85% error rate, 450ms latency
  ✅ postgres-primary — Healthy
  ✅ nginx-lb       — Healthy
```

**What does this tell us?**
- user-api is broken (85% errors)
- Two other services are fine
- **Next**: Investigate user-api!

---

## Phase 2: Investigation 🔍

### Action 1: Check Service Health
```
Input:
  - Action Type: check_service_health
  - Service: user-api

Output:
  ✅ Investigated user-api for first time
  Reward: +0.04
```

**Why?** We confirmed the problem is real and specific to user-api.

---

### Action 2: Investigate Logs
```
Input:
  - Action Type: investigate_logs
  - Service: user-api
  - Keyword: connection

Output:
  Found in logs:
    [08:15:12] ERROR: Connection refused for postgres-primary:5433
    [08:15:13] ERROR: Failed to connect to database
    [08:15:14] ERROR: Connection refused for postgres-primary:5433

  Reward: +0.08
```

**What did we learn?**
- user-api is trying to connect to postgres-primary:5433
- But usually databases run on port 5432
- **5433 looks wrong!** 🚩

---

### Action 3: Check Metrics
```
Input:
  - Action Type: check_metrics
  - Service: user-api

Output:
  Memory: 256 MB / 512 MB
  CPU: 12%
  Response Time: 450ms (huge!)
  Error Rate: 85%

  Reward: +0.02
```

---

### Action 4: Read Config
```
Input:
  - Action Type: read_config
  - Service: user-api

Output:
  Config for user-api:
  {
    "db_host": "postgres-primary",
    "db_port": 5433,           ← ⚠️ WRONG! Should be 5432
    "db_user": "app",
    "db_password": "***",
    "max_connections": 100
  }

  Reward: +0.06
```

**Eureka!** 💡 We found it — **port 5433 should be 5432**!

---

## Phase 3: Diagnosis 🎯

### Action 5: Declare Root Cause
```
Input:
  - Action Type: declare_root_cause
  - Cause: "user-api db_port misconfigured as 5433 instead of 5432"

Output:
  ✅ ROOT CAUSE: Matches known cause!
  Reward: +0.20 (big reward!)

  Feedback: "Correct root cause identified! Now apply a fix."
```

**This tells the AI**: "You understood the problem. Now fix it."

---

## Phase 4: Remediation 🔧

### Action 6: Update Config
```
Input:
  - Action Type: update_config
  - Service: user-api
  - Config Key: db_port
  - Config Value: 5432

Output:
  ✅ Configuration updated successfully
  user-api is now healthy: 0% errors, 45ms latency
  Reward: +0.15

  🏁 EPISODE COMPLETE — Incident Resolved!
```

---

## Final Score

```
Final Score: 0.89 / 1.00

6D Breakdown:
  Root Cause:    1.00  (✅ Identified correctly)
  Remediation:   1.00  (✅ Applied correct fix)
  Investigation: 0.67  (⚠️ Didn't check postgres-primary)
  Efficiency:    1.00  (✅ Used steps efficiently)
  Safety:        1.00  (✅ No collateral damage)
  Sequence:      1.00  (✅ Diagnosed before fixing)
```

---

## Key Takeaways

✅ **Do**:
1. Investigate all services to build a complete picture
2. Look for patterns in logs (errors, timeouts, rejections)
3. Compare configurations to known good values
4. Declare root cause BEFORE trying fixes
5. Apply targeted fixes to specific services

❌ **Don't**:
1. Restart services randomly hoping luck fixes it
2. Fix something before investigating why it broke
3. Investigate the same service 5 times
4. Ignore what the logs are telling you

---

## Ready for More?

**Easy scenarios:**
- 🟢 db_connection_failure (just did this!)
- 🔵 alert_triage (classify alert severity)

**Medium scenarios:**
- 🟡 cascading_service_timeout (multiple services, cascading failures)
- 🟡 ssl_certificate_expiry (non-obvious cert issue)

**Hard scenarios:**
- 🔴 multi_factor_outage (3 simultaneous root causes)
- 🔴 database_deadlock (requires understanding of database locks)

👉 Now go to **"⚙️ Interactive Sandbox"** and try it yourself!
                """)

        # ─────────────────────────────────────────────────────────────────────────
        # TAB 3: FAQ & CONCEPTS
        # ─────────────────────────────────────────────────────────────────────────
        with gr.TabItem("❓ FAQ & Concepts", id="faq"):
            with gr.Column():
                gr.Markdown("""
# ❓ FAQ & Concepts

## Scoring & Evaluation

### Q: How is my score calculated?

**A:** Your score has **6 dimensions**, each weighted:

| Dimension       | Weight | Explanation |
|-----------------|--------|-------------|
| Root Cause      | 30%    | Did you identify the RIGHT problem? |
| Remediation     | 25%    | Did you apply the RIGHT fix? |
| Investigation   | 15%    | Did you check enough services? |
| Efficiency      | 10%    | Did you use steps wisely? |
| Safety          | 10%    | Did you avoid making it worse? |
| Sequence        | 10%    | Did you diagnose BEFORE fixing? |

**Example**: If you get 100% on Root Cause (0.30) + 100% on Remediation (0.25) but only 50% on Investigation (0.075), your total would be 0.625 / 1.0.

---

### Q: What does "Observation Loop" mean?

**A:** If you do **3+ diagnosis actions in a row WITHOUT any fix**, the system penalizes you:
- You get a **-0.08 penalty**
- Your score is **hard-capped at 0.45** even if perfect on everything else

**Why?** In real incidents, endless investigation wastes time. You must eventually commit to a hypothesis and try a fix.

**Fix**: Do an investigation, then declare a root cause, then apply a fix. Mix it up!

---

### Q: Why did my score get worse after I applied a fix?

**A:** Possibilities:

1. **Wrong diagnosis**: You fixed the wrong thing → Remediation score drops
2. **Collateral damage**: Your fix broke another service → Safety score drops
3. **Too late**: You took too many steps → Efficiency score drops

**Solution**: Review the feedback message — it will tell you what happened!

---

### Q: What's "Diagnosis Gate"?

**A:** Some scenarios have a "diagnosis gate" — you must investigate a service BEFORE you can fix it successfully.

**Example**: In `ssl_certificate_expiry`, you must investigate the `api-gateway` logs FIRST. If you try to fix it blind, you only get 50% of the remediation points.

**Why?** In real SRE work, you don't blindly restart systems — you investigate, diagnose, then fix.

---

### Q: What's "Collateral Degradation"?

**A:** Every 4 steps, if the incident isn't resolved:
- Unresolved upstream failures cascade
- Dependent services get worse (↑10% error rate, ↑40% latency)
- Eventually services fail completely

**Example**:
- Step 0-3: user-api down, others OK
- Step 4: payment-service starts getting errors (depends on user-api)
- Step 8: If still not fixed, order-service gets errors too
- Step 12: Entire stack might fail if unresolved

**Why?** Real incidents don't stay contained — they spread. Time pressure is real.

---

## Scenarios

### Q: Which scenario should I start with?

**A:** Start with 🟢 **db_connection_failure** — it has:
- Only 1 root cause
- Very clear symptoms
- No confusing red herrings
- Straightforward fix

After that, try 🔵 **alert_triage** — different task type (classify severity, not fix).

---

### Q: What's the difference between "easy" and "hard" scenarios?

**A:**

| Aspect | Easy | Hard |
|--------|------|------|
| # Services | 3 | 6 |
| # Root Causes | 1 | 2-3 simultaneously |
| Red Herrings | None | Many misleading clues |
| Cascades | No | Yes, spreading failures |
| Diagnosis Gates | No | Yes, catch blind fixes |
| Investigation Hints | Clear | Ambiguous |

---

### Q: What's "Alert Triage"?

**A:** **Different task type** — instead of fixing an incident, you **classify its severity**:

- **P1 (Critical)**: Complete outage or >$1k/min revenue loss
- **P2 (High)**: Major degradation, most users affected
- **P3 (Medium)**: Partial/minor, graceful fallback active
- **P4 (Low)**: Informational, zero user impact

**Goal**: Investigate the incident, then submit severity with `submit_severity` action.

**Budget**: Only 3 steps — must decide quickly!

---

## Actions & Mechanics

### Q: What's the difference between `investigate_logs` and `check_service_health`?

**A:**

| Action | Returns | Use When |
|--------|---------|----------|
| `check_service_health` | Health status, error rate, latency | Quick overview of service state |
| `investigate_logs` | Raw log excerpts | Need detailed error messages |
| `check_metrics` | CPU, memory, throughput | Understanding scale/capacity issues |
| `read_config` | Service configuration | Checking settings |
| `run_diagnostic` | Deep diagnostic for a service | Need advanced info |

**Pattern**: Use `check_service_health` first to get overview, then `investigate_logs` on suspicious services.

---

### Q: What actions fix things?

**A:**

| Action | When to Use |
|--------|------------|
| `update_config` | Configuration wrong (ports, limits, etc.) |
| `restart_service` | Service crashed or in bad state |
| `rollback_deployment` | Recent deployment broke things |
| `scale_service` | Service overloaded, needs more replicas |

---

### Q: Can I see what actions are available in a specific scenario?

**A:** After you click 🔄 **Reset**, the "📋 Action Controls" section shows all available actions for this scenario.

---

## Seeds & Reproducibility

### Q: What's a "Seed"?

**A:** A seed makes scenarios reproducible:
- **Same seed** = same incident every time
- **Different seed** = variation of the scenario (metric values shift ±12%)

**Use**: 
- Testing: Use seed=42 to get consistent results
- Training: Use different seeds to generate variety

**Leave blank** to get a random variation.

---

### Q: Can I see the same incident multiple times?

**A:** Yes! Use the same seed. This helps if you want to try a different investigation strategy on the exact same scenario.

---

## Troubleshooting

### Q: My episode ended but I don't think I fixed it?

**A:** Click 📊 **Grade (6D)** to see the full breakdown. The feedback will explain:
- What you diagnosed correctly/incorrectly
- What you fixed correctly/incorrectly
- Why your score is what it is

---

### Q: Why is my score showing 0.45 even though I diagnosed correctly?

**A:** Likely "Observation Loop" — you did 3+ diagnosis steps in a row without any fix action. Score is hard-capped at 0.45 in this case.

---

### Q: The system says "Diagnosis Gate" — what do I do?

**A:** You tried to fix a service without investigating it first. Some scenarios require you to:
1. Investigate the service
2. Declare the root cause
3. THEN apply the fix

Try replacing your fix action with `investigate_logs` or `read_config` first.

---

### Q: Can I try the same task again?

**A:** Yes! Click 🔄 **Reset** with the same task name (and optionally same seed) to run again.

---

## Advanced

### Q: Why do I need to declare root cause separately from fixing?

**A:** 
- **Root cause declaration** tests understanding (can you diagnose?)
- **Fix application** tests judgment (do you know HOW to fix it?)

Separating them:
- Forces diagnosis before action (safer)
- Tests reasoning (not just luck)
- Matches real SRE workflow

---

### Q: What if there are multiple root causes?

**A:** In hard scenarios like `multi_factor_outage`:
- 3 separate issues exist simultaneously
- You must identify ALL 3 to get full score
- You'll call `declare_root_cause` multiple times with different causes

---

### Q: How do I improve my score?

**A:** 
1. Investigate thoroughly (don't skip services)
2. Match your diagnosis to the root cause exactly
3. Apply the correct fix corresponding to your diagnosis
4. Don't do 3+ diagnosis actions in a row
5. Declare root cause BEFORE applying fixes
6. In alert triage, classify severity accurately

---

## Still Confused?

👉 Go to **"⚙️ Interactive Sandbox"** and just start playing! You'll learn fastest by doing.

The system will give you feedback after every action. Read the feedback — it will guide you.
                """)

        # ─────────────────────────────────────────────────────────────────────────
        # TAB 4: INTERACTIVE SANDBOX (THE ACTUAL INTERFACE)
        # ─────────────────────────────────────────────────────────────────────────
        with gr.TabItem("⚙️ Interactive Sandbox", id="sandbox"):

            with gr.Row(equal_height=False):

                # ── Left column: controls ─────────────────────────────────────────────
                with gr.Column(scale=2, min_width=380):

                    gr.HTML('<div class="section-title">🎯 Episode Setup</div>')
                    with gr.Row():
                        task_dd = gr.Dropdown(
                            choices=[
                                ("🟢 Easy — DB Connection Failure",      "db_connection_failure"),
                                ("🟡 Medium — Cascading Timeout",        "cascading_service_timeout"),
                                ("🟡 Medium — SSL Certificate Expiry",   "ssl_certificate_expiry"),
                                ("🔴 Hard — Multi-Factor Outage",        "multi_factor_outage"),
                                ("🔴 Hard — Database Deadlock",          "database_deadlock"),
                                ("🔵 Easy — Alert Triage (P1/P2/P3/P4)", "alert_triage"),
                            ],
                            value="db_connection_failure",
                            label="Task",
                            scale=2,
                        )
                        seed_tb = gr.Textbox(
                            label="Seed (optional)",
                            placeholder="e.g. 42",
                            value="",
                            scale=1,
                        )
                    reset_btn = gr.Button("🔄 Reset Environment", variant="secondary", size="lg")

                    gr.HTML('<div class="section-title">🎮 Action Controls</div>')
                    action_dd = gr.Dropdown(
                        choices=[
                            ("🔍 investigate_logs",     "investigate_logs"),
                            ("🔍 check_metrics",        "check_metrics"),
                            ("🔍 read_config",          "read_config"),
                            ("🔍 check_service_health", "check_service_health"),
                            ("🔍 run_diagnostic",       "run_diagnostic"),
                            ("🔧 restart_service",      "restart_service"),
                            ("🔧 update_config",        "update_config"),
                            ("🔧 rollback_deployment",  "rollback_deployment"),
                            ("🔧 scale_service",        "scale_service"),
                            ("📝 declare_root_cause",   "declare_root_cause"),
                            ("🔵 submit_severity",      "submit_severity"),
                        ],
                        value="investigate_logs",
                        label="Action Type",
                    )
                    service_dd = gr.Dropdown(
                        choices=[],
                        label="Target Service",
                        allow_custom_value=True,
                        info="Populated after Reset",
                    )

                    with gr.Accordion("📋 Action Parameters", open=True):
                        gr.HTML('<div class="param-hint">investigate_logs — optional keyword filter</div>')
                        keyword_tb = gr.Textbox(
                            label="Keyword (investigate_logs)",
                            placeholder="e.g. error, timeout, OOM, connection",
                            value="",
                        )

                        gr.HTML('<div class="param-hint">update_config — config key + new value</div>')
                        with gr.Row():
                            config_key_tb = gr.Textbox(
                                label="Config Key",
                                placeholder="e.g. max_connections",
                                scale=1,
                            )
                            config_val_tb = gr.Textbox(
                                label="Config Value",
                                placeholder="e.g. 100",
                                scale=1,
                            )

                        gr.HTML('<div class="param-hint">scale_service — desired replica count</div>')
                        replicas_tb = gr.Textbox(
                            label="Replicas (scale_service)",
                            placeholder="e.g. 3",
                            value="",
                        )

                        gr.HTML('<div class="param-hint">declare_root_cause — describe root cause in detail (service + failure mode)</div>')
                        cause_tb = gr.Textbox(
                            label="Root Cause Description",
                            placeholder="e.g. postgres-primary connection pool exhausted due to slow analytics query",
                            lines=3,
                            value="",
                        )

                        gr.HTML('<div class="param-hint">submit_severity (alert_triage only) — classify incident P1–P4</div>')
                        severity_dd = gr.Dropdown(
                            choices=[
                                ("— not submitting", ""),
                                ("🔴 P1 — Critical: complete outage or >$1k/min revenue loss", "P1"),
                                ("🟠 P2 — High: major degradation, most users affected", "P2"),
                                ("🟡 P3 — Medium: partial/minor, graceful fallback active", "P3"),
                                ("🟢 P4 — Low: informational, zero user impact", "P4"),
                            ],
                            value="",
                            label="Severity (submit_severity)",
                        )

                    step_btn = gr.Button("▶️ Execute Action", variant="primary", size="lg")

                    gr.HTML('<div class="section-title">📊 Scoring</div>')
                    with gr.Row():
                        grade_btn = gr.Button("📊 Grade (6D)", variant="secondary", size="sm")
                        state_btn = gr.Button("📋 State",      variant="secondary", size="sm")

                    gr.HTML('<div class="section-title">📌 Episode State</div>')
                    state_display = gr.Markdown("### ⏳ Ready\n\nSelect task → Reset → Begin")

                    # ── Right column: output panels ───────────────────────────────────────
                with gr.Column(scale=3, min_width=480):

                    gr.HTML('<div class="section-title">👁️ Observation</div>')
                    obs_display = gr.Markdown(
                        "### 👋 Welcome\n\nSelect a task and click **🔄 Reset** to begin."
                    )

                    gr.HTML('<div class="section-title">📜 Action History</div>')
                    history_display = gr.Markdown("*No actions yet.*")

                    gr.HTML('<div class="section-title">💰 Step Reward</div>')
                    reward_display = gr.Markdown("*Start an episode first.*")

                    gr.HTML('<div class="section-title">🏆 6D Score Breakdown</div>')
                    score_display = gr.Markdown(
                        "*Click **📊 Grade (6D)** after executing actions.*"
                    )

                # ── Wire up events ────────────────────────────────────────────────────────
                reset_btn.click(
                    fn=gr_reset,
                    inputs=[task_dd, seed_tb],
                    outputs=[obs_display, history_display, state_display, reward_display, score_display, service_dd],
                )
                step_btn.click(
                    fn=gr_step,
                    inputs=[action_dd, service_dd, keyword_tb, config_key_tb, config_val_tb, replicas_tb, cause_tb, severity_dd],
                    outputs=[obs_display, history_display, state_display, reward_display],
                )
                grade_btn.click(fn=gr_grade, outputs=[score_display])
                state_btn.click(fn=gr_state, outputs=[state_display])


app = gr.mount_gradio_app(app, web_ui, path="/web")


def main():
    import uvicorn
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run(app, host="0.0.0.0", port=port, proxy_headers=True, forwarded_allow_ips="*")


if __name__ == "__main__":
    main()
