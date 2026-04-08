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

    with gr.Accordion("📖 How to Use", open=False):
        gr.Markdown("""
### Quick Start
1. Select **Task** + optional **Seed** → click **🔄 Reset**
2. Choose **Action Type** + **Service** + fill relevant parameter fields → click **▶️ Execute**
3. Workflow: `investigate_logs` / `check_metrics` → `run_diagnostic` → `declare_root_cause` → fix action
4. Click **📊 Grade (6D)** at any time for the full score breakdown

### Tasks
| Task | Difficulty | Key Challenge |
|---|---|---|
| `db_connection_failure` | 🟢 Easy | Single root cause, direct fix |
| `cascading_service_timeout` | 🟡 Medium | Multi-hop cascade, gated diagnosis |
| `ssl_certificate_expiry` | 🟡 Medium | Cert config + renewal, gated |
| `multi_factor_outage` | 🔴 Hard | Multiple simultaneous root causes |
| `database_deadlock` | 🔴 Hard | Deadlock pattern, gated diagnosis |
| `alert_triage` | 🔵 Easy (Triage) | Classify severity P1–P4 in ≤3 steps |

### 6D Scoring
| Dimension | Weight | What it measures |
|---|---|---|
| Root Cause | ×0.30 | Correct `declare_root_cause` vs total required |
| Remediation | ×0.25 | Correct fix actions applied |
| Investigation | ×0.15 | Services investigated / total services |
| Efficiency | ×0.10 | Steps used relative to budget |
| Safety | ×0.10 | No destructive actions on wrong services |
| Sequence | ×0.10 | Diagnose before fix, early resolution |

### Anti-Reward-Hacking
- **Observation Loop:** ≥3 consecutive diagnosis steps without any fix → −0.08 penalty, score capped at 0.45
- **Diagnosis Gate:** Fixing a gated-scenario service without investigating first → 50% remediation credit
- **Collateral Degradation:** Unresolved failures propagate to dependent services every 4 steps
        """)

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
    uvicorn.run(app, host="0.0.0.0", port=port)


if __name__ == "__main__":
    main()
