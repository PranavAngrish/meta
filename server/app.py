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



CUSTOM_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

/* ── FORCE DARK EVERYWHERE ── */
*, *::before, *::after { box-sizing: border-box; }

.gradio-container {
    background: #060810 !important;
    max-width: 100% !important;
    margin: 0 !important;
    padding: 0 !important;
    font-family: 'Syne', sans-serif !important;
    min-height: 100vh;
}

body, html {
    background: #060810 !important;
    color: #f0f2f8 !important;
}

/* ── HIDE GRADIO CHROME ── */
footer { display: none !important; }
.svelte-1ipelgc { display: none !important; }
#component-0 { background: #060810 !important; }

/* ── TABS ── */
.tab-nav {
    background: #0d1117 !important;
    border-bottom: 1px solid rgba(255,255,255,0.07) !important;
    padding: 0 32px !important;
    gap: 0 !important;
}

.tab-nav button {
    font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important;
    font-size: 13px !important;
    color: #4a5568 !important;
    background: transparent !important;
    border: none !important;
    border-bottom: 2px solid transparent !important;
    border-radius: 0 !important;
    padding: 16px 22px !important;
    transition: all 0.2s !important;
    letter-spacing: 0.02em !important;
}

.tab-nav button:hover {
    color: #8b95aa !important;
    background: rgba(255,255,255,0.03) !important;
}

.tab-nav button.selected {
    color: #f0f2f8 !important;
    border-bottom: 2px solid #ff3a5c !important;
    background: rgba(255,58,92,0.06) !important;
}

/* ── PANELS / BLOCKS ── */
.gr-group, .gr-box, .gr-form, div.gr-block {
    background: #111620 !important;
    border: 1px solid rgba(255,255,255,0.07) !important;
    border-radius: 14px !important;
}

/* ── INPUTS ── */
input[type="text"], input[type="number"], textarea, select {
    background: rgba(255,255,255,0.04) !important;
    border: 1px solid rgba(255,255,255,0.08) !important;
    border-radius: 10px !important;
    color: #f0f2f8 !important;
    font-family: 'Syne', sans-serif !important;
    font-size: 13px !important;
    padding: 10px 14px !important;
    transition: border-color 0.2s, box-shadow 0.2s !important;
}

input[type="text"]:focus, input[type="number"]:focus, textarea:focus, select:focus {
    border-color: rgba(255,58,92,0.5) !important;
    box-shadow: 0 0 0 3px rgba(255,58,92,0.1) !important;
    outline: none !important;
}

input::placeholder, textarea::placeholder { color: #4a5568 !important; }

/* ── LABELS ── */
label span, .gr-input-label, .block-title {
    font-family: 'Syne', sans-serif !important;
    font-size: 11px !important;
    font-weight: 700 !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
    color: #4a5568 !important;
}

/* ── BUTTONS ── */
.gr-button {
    font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important;
    font-size: 13px !important;
    border-radius: 10px !important;
    letter-spacing: 0.02em !important;
    transition: all 0.2s !important;
    border: none !important;
    cursor: pointer !important;
}

.gr-button-primary {
    background: #ff3a5c !important;
    color: #fff !important;
    box-shadow: 0 0 20px rgba(255,58,92,0.2) !important;
}

.gr-button-primary:hover {
    background: #ff5575 !important;
    box-shadow: 0 0 32px rgba(255,58,92,0.35) !important;
    transform: translateY(-1px) !important;
}

.gr-button-secondary {
    background: rgba(255,255,255,0.05) !important;
    color: #8b95aa !important;
    border: 1px solid rgba(255,255,255,0.08) !important;
}

.gr-button-secondary:hover {
    background: rgba(255,255,255,0.09) !important;
    color: #f0f2f8 !important;
    border-color: rgba(255,255,255,0.15) !important;
}

/* ── MARKDOWN ── */
.gr-markdown, .md, .prose {
    color: #8b95aa !important;
    font-family: 'Syne', sans-serif !important;
    background: transparent !important;
}

.gr-markdown h1, .gr-markdown h2, .gr-markdown h3,
.md h1, .md h2, .md h3 {
    color: #f0f2f8 !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 800 !important;
}

.gr-markdown h3, .md h3 { color: #00d4ff !important; }

.gr-markdown code, .md code {
    background: rgba(255,58,92,0.12) !important;
    color: #ff6b7a !important;
    border-radius: 5px !important;
    padding: 2px 6px !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 12px !important;
    border: 1px solid rgba(255,58,92,0.2) !important;
}

.gr-markdown pre, .md pre {
    background: #080b12 !important;
    border: 1px solid rgba(255,255,255,0.07) !important;
    border-radius: 10px !important;
    padding: 16px !important;
}

.gr-markdown pre code, .md pre code {
    background: none !important;
    border: none !important;
    color: #00d4ff !important;
    padding: 0 !important;
}

.gr-markdown table, .md table {
    border-collapse: collapse !important;
    width: 100% !important;
}

.gr-markdown th, .md th {
    background: rgba(255,58,92,0.1) !important;
    color: #f0f2f8 !important;
    font-weight: 700 !important;
    padding: 10px 12px !important;
    border: 1px solid rgba(255,255,255,0.07) !important;
    font-size: 12px !important;
}

.gr-markdown td, .md td {
    padding: 9px 12px !important;
    border: 1px solid rgba(255,255,255,0.05) !important;
    color: #8b95aa !important;
    font-size: 13px !important;
}

.gr-markdown tr:hover td, .md tr:hover td {
    background: rgba(255,58,92,0.04) !important;
}

.gr-markdown blockquote, .md blockquote {
    border-left: 3px solid #00d4ff !important;
    background: rgba(0,212,255,0.06) !important;
    padding: 12px 16px !important;
    border-radius: 0 8px 8px 0 !important;
    color: #8b95aa !important;
    margin: 12px 0 !important;
}

.gr-markdown a, .md a { color: #00d4ff !important; }
.gr-markdown strong, .md strong { color: #f0f2f8 !important; }
.gr-markdown li, .md li { color: #8b95aa !important; line-height: 1.7 !important; }

/* ── DROPDOWN ── */
.gr-dropdown ul {
    background: #111620 !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    border-radius: 10px !important;
}

.gr-dropdown li {
    color: #8b95aa !important;
    font-family: 'Syne', sans-serif !important;
}

.gr-dropdown li:hover {
    background: rgba(255,58,92,0.1) !important;
    color: #f0f2f8 !important;
}

/* ── ACCORDION ── */
.gr-accordion {
    background: #111620 !important;
    border: 1px solid rgba(255,255,255,0.07) !important;
    border-radius: 10px !important;
}

.gr-accordion > button {
    font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important;
    color: #8b95aa !important;
    background: transparent !important;
    font-size: 13px !important;
    padding: 14px 16px !important;
}

.gr-accordion > button:hover { color: #f0f2f8 !important; }

/* ── SCROLLBAR ── */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: rgba(255,255,255,0.08); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: rgba(255,255,255,0.15); }

/* ── ROW / COLUMN GAPS ── */
.gr-row { gap: 16px !important; }
.gr-column { gap: 12px !important; }

/* ── SECTION HEADER HTML ── */
.sec-head {
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: #4a5568;
    padding: 10px 0 10px;
    display: flex;
    align-items: center;
    gap: 10px;
    font-family: 'Syne', sans-serif;
}
.sec-head::after { content: ''; flex: 1; height: 1px; background: rgba(255,255,255,0.06); }

.panel-header-html {
    display: flex;
    align-items: center;
    gap: 8px;
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: #4a5568;
    font-family: 'Syne', sans-serif;
    margin-bottom: 4px;
}

.panel-header-dot {
    width: 6px; height: 6px;
    border-radius: 50%;
    background: #ff3a5c;
    display: inline-block;
}

.diff-easy  { display:inline;background:rgba(0,229,160,0.1);color:#00e5a0;border:1px solid rgba(0,229,160,0.2);font-family:'JetBrains Mono',monospace;font-size:10px;font-weight:600;padding:2px 8px;border-radius:5px;letter-spacing:0.06em; }
.diff-med   { display:inline;background:rgba(255,176,32,0.1);color:#ffb020;border:1px solid rgba(255,176,32,0.2);font-family:'JetBrains Mono',monospace;font-size:10px;font-weight:600;padding:2px 8px;border-radius:5px;letter-spacing:0.06em; }
.diff-hard  { display:inline;background:rgba(255,58,92,0.1);color:#ff3a5c;border:1px solid rgba(255,58,92,0.2);font-family:'JetBrains Mono',monospace;font-size:10px;font-weight:600;padding:2px 8px;border-radius:5px;letter-spacing:0.06em; }
.diff-triage{ display:inline;background:rgba(0,212,255,0.1);color:#00d4ff;border:1px solid rgba(0,212,255,0.2);font-family:'JetBrains Mono',monospace;font-size:10px;font-weight:600;padding:2px 8px;border-radius:5px;letter-spacing:0.06em; }

.stat-mini {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 10px;
    padding: 12px 14px;
    font-family: 'Syne', sans-serif;
}
.stat-mini .sl { font-size:10px;font-weight:700;letter-spacing:0.1em;text-transform:uppercase;color:#4a5568;margin-bottom:4px; }
.stat-mini .sv { font-size:22px;font-weight:800;letter-spacing:-0.5px; }

.reward-pos { color: #00e5a0 !important; }
.reward-neg { color: #ff3a5c !important; }

.chip { display:inline-flex;align-items:center;gap:4px;font-family:'JetBrains Mono',monospace;font-size:11px;font-weight:500;padding:3px 10px;border-radius:100px; }
.chip-diag { background:rgba(0,212,255,0.1);color:#00d4ff; }
.chip-fix  { background:rgba(0,229,160,0.1);color:#00e5a0; }
.chip-decl { background:rgba(155,124,248,0.1);color:#9b7cf8; }

.log-out {
    font-family:'JetBrains Mono',monospace;
    font-size:12px;
    background:#080b12;
    border:1px solid rgba(255,255,255,0.06);
    border-radius:10px;
    padding:14px 16px;
    color:#8b95aa;
    line-height:1.9;
    max-height:220px;
    overflow-y:auto;
    white-space:pre-wrap;
    word-break:break-word;
}

.alert-item {
    display:flex;align-items:flex-start;gap:10px;
    padding:12px 14px;border-radius:10px;border:1px solid;
    margin-bottom:8px;font-family:'Syne',sans-serif;
}
.alert-crit { background:rgba(255,58,92,0.06);border-color:rgba(255,58,92,0.2); }
.alert-med  { background:rgba(255,176,32,0.06);border-color:rgba(255,176,32,0.2); }
.alert-low  { background:rgba(0,229,160,0.06);border-color:rgba(0,229,160,0.15); }

.abadge { font-family:'JetBrains Mono',monospace;font-size:10px;font-weight:600;padding:2px 8px;border-radius:5px;flex-shrink:0;margin-top:1px; }
.abadge-c { background:rgba(255,58,92,0.15);color:#ff3a5c;border:1px solid rgba(255,58,92,0.3); }
.abadge-m { background:rgba(255,176,32,0.15);color:#ffb020;border:1px solid rgba(255,176,32,0.3); }
.abadge-l { background:rgba(0,229,160,0.15);color:#00e5a0;border:1px solid rgba(0,229,160,0.2); }

.svc-tbl { width:100%;border-collapse:collapse;font-family:'JetBrains Mono',monospace;font-size:12px; }
.svc-tbl th { color:#4a5568;font-size:10px;letter-spacing:0.1em;text-transform:uppercase;padding:0 10px 8px;text-align:left;border-bottom:1px solid rgba(255,255,255,0.06); }
.svc-tbl td { padding:9px 10px;border-bottom:1px solid rgba(255,255,255,0.03);color:#8b95aa; }
.svc-tbl tr:last-child td { border-bottom:none; }

.hpill { display:inline-flex;align-items:center;gap:5px;font-size:11px;font-weight:600;padding:3px 10px;border-radius:100px; }
.hpill-ok  { background:rgba(0,229,160,0.1);color:#00e5a0; }
.hpill-err { background:rgba(255,58,92,0.1);color:#ff3a5c; }

.dim-card {
    background:rgba(255,255,255,0.03);border:1px solid rgba(255,255,255,0.06);
    border-radius:10px;padding:12px 14px;font-family:'Syne',sans-serif;
}
.dim-name { font-size:10px;font-weight:700;letter-spacing:0.1em;text-transform:uppercase;color:#4a5568;margin-bottom:6px; }
.dim-val  { font-size:22px;font-weight:800;letter-spacing:-0.5px;margin-bottom:6px; }
.dim-bar-bg { height:3px;background:rgba(255,255,255,0.05);border-radius:100px;overflow:hidden; }
.dim-bar { height:100%;border-radius:100px; }
.dim-wt  { font-family:'JetBrains Mono',monospace;font-size:10px;color:#4a5568;margin-top:4px; }

.prog-outer { height:4px;background:rgba(255,255,255,0.05);border-radius:100px;overflow:hidden;margin:10px 0 4px; }
.prog-inner { height:100%;border-radius:100px;background:linear-gradient(90deg,#ff3a5c,#00d4ff);transition:width 0.5s; }

.fail-banner { display:flex;align-items:center;gap:12px;padding:12px 16px;border-radius:10px;border:1px solid;margin-bottom:14px;font-family:'Syne',sans-serif; }
.fb-green { background:rgba(0,229,160,0.06);border-color:rgba(0,229,160,0.2); }
.fb-red   { background:rgba(255,58,92,0.06);border-color:rgba(255,58,92,0.2); }
.fb-amber { background:rgba(255,176,32,0.06);border-color:rgba(255,176,32,0.2); }
"""

# ── Helpers (keep your existing ones, add these) ──────────────────────────────

_DIAG_ACTIONS = {"investigate_logs","check_metrics","read_config","check_service_health","run_diagnostic"}
_FIX_ACTIONS  = {"restart_service","update_config","rollback_deployment","scale_service"}
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
    "alert_triage":              "🔵 Triage",
}
_action_history: List[Dict[str, Any]] = []


def _render_alerts(alerts):
    if not alerts:
        return "<div style='text-align:center;padding:32px 16px;color:#4a5568;font-family:Syne,sans-serif;font-size:13px;'>🔕 No alerts — reset to begin</div>"
    out = ""
    for a in alerts:
        sev = (a.get("severity","medium")).upper()
        cls = "alert-crit" if sev in ("CRITICAL","HIGH") else ("alert-med" if sev=="MEDIUM" else "alert-low")
        bcls = "abadge-c" if sev in ("CRITICAL","HIGH") else ("abadge-m" if sev=="MEDIUM" else "abadge-l")
        out += f"""<div class="alert-item {cls}">
  <span class="abadge {bcls}">{sev}</span>
  <div>
    <div style="font-size:13px;font-weight:700;color:#f0f2f8;margin-bottom:2px;">{a.get('service','')} <span style="font-family:'JetBrains Mono',monospace;font-size:10px;color:#4a5568;font-weight:400;">{a.get('alert_id','')}</span></div>
    <div style="font-size:12px;color:#8b95aa;line-height:1.5;">{a.get('message','')}</div>
  </div>
</div>"""
    return out


def _render_svc_table(statuses):
    if not statuses:
        return ""
    rows = ""
    for s in statuses:
        ok = s.get("healthy", True)
        hcls = "hpill-ok" if ok else "hpill-err"
        htxt = "OK" if ok else "ERROR"
        rt = f"{s['response_time_ms']:.0f}" if s.get("response_time_ms") is not None else "—"
        er_raw = s.get("error_rate")
        er = f"{er_raw*100:.1f}%" if er_raw is not None else "—"
        er_color = "#ff3a5c" if not ok else "#4a5568"
        rows += f"""<tr>
  <td style="color:#f0f2f8;font-weight:600;">{s['name']}</td>
  <td><span class="hpill {hcls}">● {htxt}</span></td>
  <td>{rt}</td>
  <td style="color:{er_color};">{er}</td>
</tr>"""
    return f"""<div style="font-family:'JetBrains Mono',monospace;font-size:10px;letter-spacing:0.12em;color:#4a5568;text-transform:uppercase;margin:16px 0 8px;">Service Health</div>
<table class="svc-tbl">
  <thead><tr><th>Service</th><th>Status</th><th>RT ms</th><th>Err%</th></tr></thead>
  <tbody>{rows}</tbody>
</table>"""


def _render_log(text):
    if not text:
        return ""
    import html as _html
    escaped = _html.escape(str(text))
    # basic coloring via inline spans
    import re
    escaped = re.sub(r'(ERROR|CRITICAL|FAILED|refused|Connection refused)', r'<span style="color:#ff3a5c;">\1</span>', escaped, flags=re.IGNORECASE)
    escaped = re.sub(r'(SUCCESS|healthy|resolved|updated successfully)', r'<span style="color:#00e5a0;">\1</span>', escaped, flags=re.IGNORECASE)
    escaped = re.sub(r'(\[\d{2}:\d{2}:\d{2}\]|\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2})', r'<span style="color:#4a5568;">\1</span>', escaped)
    return f'<div style="font-family:\'JetBrains Mono\',monospace;font-size:10px;letter-spacing:0.12em;color:#4a5568;text-transform:uppercase;margin:16px 0 8px;">Last Action Output</div><div class="log-out">{escaped}</div>'


def _render_obs(obs_dump, action_result):
    task = obs_dump.get("task_name", "—")
    diff = _DIFFICULTY_BADGE.get(task, "")
    step = obs_dump.get("step_number", 0)
    max_s = obs_dump.get("max_steps", 30)
    alerts_html = _render_alerts(obs_dump.get("active_alerts", []))
    svc_html = _render_svc_table(obs_dump.get("service_statuses", []))
    log_html = _render_log(action_result) if action_result else ""
    pct = int(step / max(max_s, 1) * 100)
    return f"""
<div style="font-family:'Syne',sans-serif;">
  <div style="display:flex;align-items:center;gap:10px;padding:14px 0 16px;border-bottom:1px solid rgba(255,255,255,0.06);margin-bottom:16px;">
    <span style="width:8px;height:8px;border-radius:50%;background:#ff3a5c;display:inline-block;animation:pulse 2s infinite;"></span>
    <span style="font-family:'JetBrains Mono',monospace;font-size:12px;color:#8b95aa;">{task}</span>
    <span style="margin-left:4px;" class="diff-{('easy' if 'Easy' in diff else 'med' if 'Medium' in diff else 'hard' if 'Hard' in diff else 'triage')}">{diff}</span>
    <span style="margin-left:auto;font-family:'JetBrains Mono',monospace;font-size:11px;color:#4a5568;">{step}/{max_s}</span>
  </div>
  <div class="prog-outer"><div class="prog-inner" style="width:{pct}%;"></div></div>
  <div style="display:flex;justify-content:space-between;font-family:'JetBrains Mono',monospace;font-size:10px;color:#4a5568;margin-bottom:16px;">
    <span>Step {step}/{max_s}</span><span>{pct}%</span>
  </div>
  <div style="font-family:'JetBrains Mono',monospace;font-size:10px;letter-spacing:0.12em;color:#4a5568;text-transform:uppercase;margin-bottom:10px;">Active Alerts</div>
  {alerts_html}
  {svc_html}
  {log_html}
</div>
<style>@keyframes pulse{{0%,100%{{opacity:1}}50%{{opacity:0.4}}}}</style>
"""


def _render_history(history):
    if not history:
        return "<div style='text-align:center;padding:32px;color:#4a5568;font-family:Syne,sans-serif;font-size:13px;'>📋 No actions yet</div>"
    rows = ""
    for h in history:
        act = h.get("action","")
        is_diag = act in _DIAG_ACTIONS
        is_fix  = act in _FIX_ACTIONS
        chip_cls = "chip-diag" if is_diag else ("chip-fix" if is_fix else "chip-decl")
        icon = "🔍" if is_diag else ("🔧" if is_fix else "📝")
        r = h.get("reward", 0.0)
        r_str = f"+{r:.3f}" if r >= 0 else f"{r:.3f}"
        r_col = "#00e5a0" if r >= 0 else "#ff3a5c"
        svc = h.get("service") or "—"
        rows += f"""<tr>
  <td style="color:#4a5568;">{h['step']}</td>
  <td><span class="chip {chip_cls}">{icon}</span></td>
  <td style="color:#f0f2f8;font-weight:600;">{act}</td>
  <td style="color:#8b95aa;">{svc}</td>
  <td style="color:{r_col};font-weight:700;">{r_str}</td>
</tr>"""
    return f"""<table style="width:100%;border-collapse:collapse;font-family:'JetBrains Mono',monospace;font-size:12px;">
  <thead><tr>
    <th style="color:#4a5568;font-size:10px;letter-spacing:0.1em;text-transform:uppercase;padding:0 10px 8px;text-align:left;border-bottom:1px solid rgba(255,255,255,0.06);">Step</th>
    <th style="color:#4a5568;font-size:10px;letter-spacing:0.1em;text-transform:uppercase;padding:0 10px 8px;text-align:left;border-bottom:1px solid rgba(255,255,255,0.06);">Type</th>
    <th style="color:#4a5568;font-size:10px;letter-spacing:0.1em;text-transform:uppercase;padding:0 10px 8px;text-align:left;border-bottom:1px solid rgba(255,255,255,0.06);">Action</th>
    <th style="color:#4a5568;font-size:10px;letter-spacing:0.1em;text-transform:uppercase;padding:0 10px 8px;text-align:left;border-bottom:1px solid rgba(255,255,255,0.06);">Service</th>
    <th style="color:#4a5568;font-size:10px;letter-spacing:0.1em;text-transform:uppercase;padding:0 10px 8px;text-align:left;border-bottom:1px solid rgba(255,255,255,0.06);">Reward</th>
  </tr></thead>
  <tbody>{rows}</tbody>
</table>"""


def _render_state_panel():
    if env._task_name == "alert_triage":
        step = env._step_number
        cum = round(env._cumulative_reward, 4)
        submitted = env._at_submitted_severity or "—"
        pct = int(step / 3 * 100)
        return f"""<div style="font-family:'Syne',sans-serif;">
  <div style="display:grid;grid-template-columns:1fr 1fr;gap:10px;margin-bottom:12px;">
    <div class="stat-mini"><div class="sl">Step</div><div class="sv" style="color:#00d4ff;">{step}/3</div></div>
    <div class="stat-mini"><div class="sl">Cum. Reward</div><div class="sv" style="color:#00e5a0;">{'+' if cum>=0 else ''}{cum:.4f}</div></div>
  </div>
  <div class="stat-mini"><div class="sl">Submitted Severity</div><div class="sv" style="color:#9b7cf8;font-size:18px;">{submitted}</div></div>
  <div class="prog-outer"><div class="prog-inner" style="width:{pct}%;"></div></div>
</div>"""

    if not env._scenario:
        return "<div style='color:#4a5568;font-family:Syne,sans-serif;font-size:13px;padding:16px 0;'>⏳ Select task → Reset → Begin</div>"

    task = env._task_name or "—"
    diff = _DIFFICULTY_BADGE.get(task,"")
    step = env._step_number
    max_s = env._scenario.max_steps
    resolved = env._incident_resolved
    done = env._done
    cum = round(env._cumulative_reward, 4)
    pct = int(step / max(max_s,1) * 100)
    status = "🏁 Resolved" if resolved else ("🏁 Done" if done else "⚡ Active")
    streak = env._consecutive_diagnosis_count
    collateral = len(env._collateral_degraded)
    cum_col = "#00e5a0" if cum >= 0 else "#ff3a5c"
    streak_col = "#ff3a5c" if streak >= 2 else "#ffb020"
    return f"""<div style="font-family:'Syne',sans-serif;">
  <div style="font-size:14px;font-weight:800;margin-bottom:12px;color:#f0f2f8;">{status}</div>
  <div style="display:grid;grid-template-columns:1fr 1fr;gap:10px;margin-bottom:12px;">
    <div class="stat-mini"><div class="sl">Step</div><div class="sv" style="color:#00d4ff;">{step}/{max_s}</div></div>
    <div class="stat-mini"><div class="sl">Cum. Reward</div><div class="sv" style="color:{cum_col};">{'+' if cum>=0 else ''}{cum:.4f}</div></div>
    <div class="stat-mini"><div class="sl">Diag Streak</div><div class="sv" style="color:{streak_col};">{streak}{'⚠' if streak>=2 else ''}</div></div>
    <div class="stat-mini"><div class="sl">Collateral</div><div class="sv" style="color:#9b7cf8;">{collateral}</div></div>
  </div>
  <div style="display:flex;align-items:center;gap:8px;font-size:13px;color:#8b95aa;">
    Resolved: {'<span style="color:#00e5a0;font-weight:700;">Yes ✓</span>' if resolved else '<span style="color:#ff3a5c;">No ✗</span>'}
  </div>
  <div class="prog-outer"><div class="prog-inner" style="width:{pct}%;"></div></div>
  <div style="display:flex;justify-content:space-between;font-family:'JetBrains Mono',monospace;font-size:10px;color:#4a5568;">
    <span>{step}/{max_s} steps</span><span>{pct}%</span>
  </div>
</div>"""


def _render_score(breakdown):
    if not breakdown:
        return "*Click **📊 Grade** after executing actions.*"

    final = breakdown.get("final", 0.0)
    ft = breakdown.get("failure_type", "Unknown")
    ft_icon = _FAILURE_TYPE_ICON.get(ft, "⚪")
    obs_loop = breakdown.get("observation_loop", False)
    score_col = "#00e5a0" if final >= 0.7 else ("#ffb020" if final >= 0.4 else "#ff3a5c")
    fb_cls = "fb-green" if final >= 0.7 else ("fb-red" if ft == "Stuck in Observation Loop" else "fb-amber")

    is_triage = ft == "Alert Triage" or "severity_match" in breakdown.get("breakdown", {})
    if is_triage:
        bd = breakdown.get("breakdown", {})
        sub = bd.get("submitted_severity","—")
        cor = bd.get("correct_severity","—")
        sev_m = bd.get("severity_match", 0.0)
        inv_b = bd.get("investigation_bonus", 0.0)
        return f"""### {ft_icon} Alert Triage Score: **{final:.4f}** / 1.0

| Component | Value |
|---|---|
| Submitted | `{sub}` |
| Correct | `{cor}` |
| Severity Score | `{sev_m:.2f}` |
| Investigation Bonus | `+{inv_b:.2f}` |
| **Total** | **`{final:.4f}`** |

{('> ⚠ ' + breakdown.get('feedback','')) if breakdown.get('feedback') else ''}
"""

    dims = [
        ("root_cause","Root Cause",0.30,"#ff3a5c"),
        ("remediation","Remediation",0.25,"#00d4ff"),
        ("investigation","Investigation",0.15,"#00e5a0"),
        ("efficiency","Efficiency",0.10,"#ffb020"),
        ("safety","Safety",0.10,"#9b7cf8"),
        ("sequence","Sequence",0.10,"#00e5a0"),
    ]
    rows = "\n".join(
        f"| {label} | {breakdown.get(key,0.0):.2f} | ×{wt:.2f} | {'▓'*int(breakdown.get(key,0)*10)}{'░'*(10-int(breakdown.get(key,0)*10))} |"
        for key, label, wt, _ in dims
    )
    fb = breakdown.get("feedback","")
    obs_note = f"\n> ⚠ Observation loop — score capped at {_OBSERVATION_LOOP_CAP}" if obs_loop else ""
    return f"""### {ft_icon} Score: **{final:.4f}** / 1.0

**Failure Type:** {ft}{obs_note}

#### 6D Breakdown
| Dimension | Score | Weight | Bar |
|---|---:|---:|---|
{rows}

{('> ' + fb) if fb else ''}
"""


# ── Gradio callbacks ──────────────────────────────────────────────────────────

def gr_reset(task_name, seed_str):
    global _action_history
    _action_history = []
    try:
        seed = int(seed_str) if seed_str.strip() else None
        obs = env.reset(task_name=task_name, seed=seed)
        obs_dump = obs.model_dump()
        services = obs.available_services
        return (
            _render_obs(obs_dump, obs.action_result),
            _render_history([]),
            _render_state_panel(),
            "*Execute an action then click Grade.*",
            "*Start an episode first.*",
            gr.Dropdown(choices=services, value=services[0] if services else None),
        )
    except Exception as e:
        err = f"❌ **Error:** {e}"
        return err, "", err, "", "", gr.Dropdown(choices=[])


def gr_step(action_type, service_name, keyword, config_key, config_val, replicas_str, cause_text, severity_val):
    global _action_history
    try:
        params: Dict[str, Any] = {}
        at = action_type
        if at == "investigate_logs" and keyword.strip():
            params["keyword"] = keyword.strip()
        elif at == "check_metrics":
            params["metric_type"] = "all"
        elif at == "update_config":
            if config_key.strip(): params["key"] = config_key.strip()
            if config_val.strip(): params["value"] = config_val.strip()
        elif at == "scale_service" and replicas_str.strip():
            try: params["replicas"] = int(replicas_str.strip())
            except ValueError: pass
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

        _action_history.append({"step": obs.step_number, "action": at, "service": svc, "reward": reward})

        ft = info.get("failure_type","N/A")
        ft_icon = _FAILURE_TYPE_ICON.get(ft,"⚪")
        r_sign = "+" if reward >= 0 else ""
        reward_md = f"""### Step reward: `{r_sign}{reward:.4f}`
**Cumulative:** `{env._cumulative_reward:+.4f}`

**Feedback:** {obs.feedback}

{ft_icon} **{ft}** | Obs Loop: `{info.get('observation_loop', False)}` | Diag Streak: `{info.get('consecutive_diagnosis_count', 0)}`
{chr(10) + '---' + chr(10) + '🏁 **Episode complete** — click **📊 Grade** for final score' if done else ''}
"""
        return (
            _render_obs(obs_dump, obs.action_result),
            _render_history(_action_history),
            _render_state_panel(),
            _render_score({}),
            reward_md,
        )
    except Exception as e:
        err = f"❌ **Error:** {e}"
        return err, "", "", "", err


def gr_grade():
    try:
        breakdown = env.get_score_breakdown()
        return _render_score(breakdown)
    except Exception as e:
        return f"❌ {e}"


def gr_state():
    return _render_state_panel()


# ── Build Gradio Blocks ───────────────────────────────────────────────────────

HEADER_HTML = """
<div style="background:linear-gradient(135deg,#ff3a5c 0%,#9b7cf8 50%,#00d4ff 100%);padding:28px 40px;display:flex;align-items:center;justify-content:space-between;position:relative;overflow:hidden;">
  <div style="display:flex;align-items:center;gap:14px;position:relative;z-index:1;">
    <div style="width:38px;height:38px;border-radius:10px;background:rgba(255,255,255,0.2);display:flex;align-items:center;justify-content:center;font-size:20px;backdrop-filter:blur(10px);border:1px solid rgba(255,255,255,0.3);">🚨</div>
    <div>
      <div style="font-family:'Syne',sans-serif;font-size:20px;font-weight:900;color:#fff;letter-spacing:-0.5px;line-height:1.1;">Incident Response Environment</div>
      <div style="font-family:'JetBrains Mono',monospace;font-size:11px;color:rgba(255,255,255,0.7);letter-spacing:0.1em;margin-top:2px;">SRE AI TRAINING PLATFORM · V4.0</div>
    </div>
  </div>
  <div style="display:flex;gap:8px;position:relative;z-index:1;">
    <span style="font-family:'JetBrains Mono',monospace;font-size:11px;font-weight:600;padding:5px 12px;border-radius:100px;border:1px solid rgba(255,255,255,0.4);color:#fff;background:rgba(0,0,0,0.2);">6D SCORING</span>
    <span style="font-family:'JetBrains Mono',monospace;font-size:11px;font-weight:600;padding:5px 12px;border-radius:100px;border:1px solid rgba(255,255,255,0.4);color:#fff;background:rgba(0,0,0,0.2);">● LIVE</span>
  </div>
</div>
<div style="height:3px;background:linear-gradient(90deg,#ff3a5c,#9b7cf8,#00d4ff);"></div>
"""

with gr.Blocks(
    title="Incident Response Environment v4.0",
    css=CUSTOM_CSS,
    theme=gr.themes.Base(),
) as web_ui:

    gr.HTML(HEADER_HTML)

    with gr.Tabs():

        # ── TAB 1: WELCOME ───────────────────────────────────────────────────
        with gr.TabItem("🎯 Overview"):
            gr.Markdown("""
# The Flight Simulator for SREs

Diagnose real production incidents in a safe, scored environment.
Teach AI agents to **investigate first, fix with precision**.

---

## Core Concepts

| Phase | Actions | Goal |
|-------|---------|------|
| 🔍 **Investigate** | check_service_health, investigate_logs, check_metrics, read_config | Gather evidence |
| 🎯 **Diagnose** | declare_root_cause | Name the problem precisely |
| 🔧 **Remediate** | update_config, restart_service, rollback_deployment, scale_service | Apply the targeted fix |

---

## 6D Scoring Model

| Dimension | Weight | What It Measures |
|-----------|--------|-----------------|
| Root Cause | **30%** | Did you identify the right problem? |
| Remediation | **25%** | Did you apply the right fix? |
| Investigation | **15%** | Did you check enough services? |
| Efficiency | **10%** | Did you use steps wisely? |
| Safety | **10%** | Did you avoid collateral damage? |
| Sequence | **10%** | Did you diagnose before fixing? |

> Scores are clamped to (0.01, 0.99). If an **observation loop** is detected (3+ diagnosis actions in a row with no fix), the score is **hard-capped at 0.45**.

---

## Scenarios

| Scenario | Difficulty | Root Causes | Notes |
|----------|-----------|------------|-------|
| db_connection_failure | 🟢 Easy | 1 | Best starting point |
| cascading_service_timeout | 🟡 Medium | 2 | Multi-hop cascade |
| ssl_certificate_expiry | 🟡 Medium | 1 | Diagnosis gate |
| multi_factor_outage | 🔴 Hard | 3 | Red herrings + cascades |
| database_deadlock | 🔴 Hard | 1 | Lock mechanics |
| alert_triage | 🔵 Triage | — | Classify P1–P4 in 3 steps |
            """)

        # ── TAB 2: WALKTHROUGH ───────────────────────────────────────────────
        with gr.TabItem("🎓 Walkthrough"):
            gr.Markdown("""
# Step-by-Step: DB Connection Failure

Follow along as we solve a real incident from alert to resolution.

**Task:** `db_connection_failure` · 🟢 Easy · 30 steps · 3 services

---

## Phase 1 — The Alert (on Reset)

```
● CRITICAL  [ALT-001]  user-api  T=08:15:23
  → user-api returning 503 errors

✗ user-api          error_rate=85%   rt=450ms
✓ postgres-primary  healthy
✓ nginx-lb          healthy
```

Two services are fine. One is broken. Start there.

---

## Phase 2 — Investigation (Steps 1–3)

**Step 1** — `check_service_health` → `user-api`
```
Reward: +0.04   (first-time investigation bonus)
```

**Step 2** — `investigate_logs` → `user-api` · keyword: `connection`
```
[08:15:12] ERROR: Connection refused for postgres-primary:5433
[08:15:13] ERROR: Failed to connect to database

→ Port 5433 is suspicious — default is 5432
Reward: +0.08
```

**Step 3** — `read_config` → `user-api`
```json
{ "db_host": "postgres-primary", "db_port": 5433 }
                                              ^^^^
                                         Should be 5432!
Reward: +0.06
```

---

## Phase 3 — Diagnosis (Step 4)

**`declare_root_cause`**
```
Cause: "user-api db_port misconfigured as 5433 instead of 5432"

✓ ROOT CAUSE MATCHED!
Reward: +0.20  ← biggest reward, tests understanding not luck
```

> Always declare root cause **before** fixing. Sequence score depends on it.

---

## Phase 4 — Remediation (Step 5)

**`update_config`** → `user-api` · key: `db_port` · value: `5432`
```
✓ Configuration updated
✓ user-api healthy: error_rate=0%  rt=45ms

🏁 INCIDENT RESOLVED
Reward: +0.15
```

---

## Final Score

| Dimension | Score | Why |
|-----------|-------|-----|
| Root Cause | 1.00 | ✅ Exact match |
| Remediation | 1.00 | ✅ Correct fix |
| Investigation | 0.67 | ⚠ Didn't check postgres-primary |
| Efficiency | 1.00 | ✅ Only 5 steps |
| Safety | 1.00 | ✅ No collateral damage |
| Sequence | 1.00 | ✅ Diagnosed before fixing |

**Final: 0.89 / 1.00 — Efficient Reasoner 🟢**

---

## Key Rules

- ✅ Investigate → Diagnose → Fix (in that order)
- ✅ Declare root cause with specific service + failure mode
- ❌ Don't restart services hoping luck fixes it
- ❌ Don't do 3+ diagnosis actions in a row (observation loop penalty)
- ⚠ Cascades spread every 4 steps if unresolved
            """)

        # ── TAB 3: FAQ ───────────────────────────────────────────────────────
        with gr.TabItem("❓ FAQ"):
            gr.Markdown("""
# FAQ & Concepts

### What is an observation loop?
3+ diagnosis actions in a row with no fix action triggers the observation loop detector.
Your score is **hard-capped at 0.45** and you receive a -0.08 penalty.
Fix it by alternating investigation with fix actions.

### What is collateral degradation?
Every 4 steps, if the incident isn't resolved, dependent services degrade:
error rates rise +10%, latency climbs +40%. Eventually the whole stack fails.
This represents real incident dynamics — **time pressure is genuine**.

### What is a diagnosis gate?
Some scenarios (like `ssl_certificate_expiry`) require you to investigate a specific service
**before** a fix will work. If you fix blind, you only receive 50% of remediation points.

### How does Alert Triage work?
Instead of fixing an incident, you classify severity in 3 steps:
- **P1** — Critical: complete outage or >$1k/min revenue loss
- **P2** — High: major degradation, most users affected
- **P3** — Medium: partial/minor, graceful fallback active
- **P4** — Low: informational, zero user impact

Score = severity accuracy (1.0 exact, 0.5 adjacent, 0.25 two-off) + investigation bonus.

### What does a seed do?
Same seed = identical incident every run. Different seed = same structure, metric values shift ±12%.
Leave blank for a random variation. Use `seed=42` for reproducible testing.

### Why declare root cause separately from fixing?
- Root cause tests **understanding** (can you diagnose?)
- Fix tests **judgment** (do you know how to fix it?)

This prevents lucky guessing from scoring well, and matches real SRE workflow.

### How do I improve my score?
1. Investigate multiple services (not just the broken one)
2. Match root cause declaration to the exact expected phrasing
3. Apply the fix that directly corresponds to your diagnosis
4. Never do 3+ diagnosis actions without a fix in between
5. Always declare root cause before applying any fix
            """)

        # ── TAB 4: SANDBOX ───────────────────────────────────────────────────
        with gr.TabItem("⚙️ Sandbox"):
            with gr.Row(equal_height=False):

                # LEFT COLUMN
                with gr.Column(scale=2, min_width=360):

                    gr.HTML('<div class="sec-head">Episode Setup</div>')
                    with gr.Group():
                        with gr.Column():
                            task_dd = gr.Dropdown(
                                choices=[
                                    ("🟢 Easy — DB Connection Failure",       "db_connection_failure"),
                                    ("🟡 Medium — Cascading Timeout",         "cascading_service_timeout"),
                                    ("🟡 Medium — SSL Certificate Expiry",    "ssl_certificate_expiry"),
                                    ("🔴 Hard — Multi-Factor Outage",         "multi_factor_outage"),
                                    ("🔴 Hard — Database Deadlock",           "database_deadlock"),
                                    ("🔵 Triage — Alert Severity (P1–P4)",    "alert_triage"),
                                ],
                                value="db_connection_failure",
                                label="Task",
                            )
                            seed_tb = gr.Textbox(label="Seed (optional)", placeholder="e.g. 42", value="")
                            reset_btn = gr.Button("↺ Reset Environment", variant="secondary", size="lg")

                    gr.HTML('<div class="sec-head">Episode State</div>')
                    state_display = gr.HTML(_render_state_panel())

                    gr.HTML('<div class="sec-head">Action Controls</div>')
                    with gr.Group():
                        with gr.Column():
                            action_dd = gr.Dropdown(
                                choices=[
                                    ("🔍 investigate_logs",      "investigate_logs"),
                                    ("🔍 check_metrics",         "check_metrics"),
                                    ("🔍 read_config",           "read_config"),
                                    ("🔍 check_service_health",  "check_service_health"),
                                    ("🔍 run_diagnostic",        "run_diagnostic"),
                                    ("🔧 restart_service",       "restart_service"),
                                    ("🔧 update_config",         "update_config"),
                                    ("🔧 rollback_deployment",   "rollback_deployment"),
                                    ("🔧 scale_service",         "scale_service"),
                                    ("📝 declare_root_cause",    "declare_root_cause"),
                                    ("🔵 submit_severity",       "submit_severity"),
                                ],
                                value="investigate_logs",
                                label="Action Type",
                            )
                            service_dd = gr.Dropdown(choices=[], label="Target Service", allow_custom_value=True)

                    with gr.Accordion("📋 Action Parameters", open=True):
                        keyword_tb  = gr.Textbox(label="Keyword (investigate_logs)", placeholder="e.g. error, timeout, connection", value="")
                        config_key_tb = gr.Textbox(label="Config Key (update_config)", placeholder="e.g. db_port", value="")
                        config_val_tb = gr.Textbox(label="Config Value (update_config)", placeholder="e.g. 5432", value="")
                        replicas_tb = gr.Textbox(label="Replicas (scale_service)", placeholder="e.g. 3", value="")
                        cause_tb    = gr.Textbox(label="Root Cause (declare_root_cause)", placeholder="e.g. user-api db_port misconfigured as 5433 instead of 5432", lines=3, value="")
                        severity_dd = gr.Dropdown(
                            choices=[
                                ("— not submitting", ""),
                                ("🔴 P1 — Critical outage / >$1k/min loss", "P1"),
                                ("🟠 P2 — High: major degradation",         "P2"),
                                ("🟡 P3 — Medium: partial, fallback active", "P3"),
                                ("🟢 P4 — Low: informational",              "P4"),
                            ],
                            value="", label="Severity (submit_severity)",
                        )

                    step_btn = gr.Button("▶ Execute Action", variant="primary", size="lg")

                    gr.HTML('<div class="sec-head">Scoring</div>')
                    with gr.Row():
                        grade_btn = gr.Button("📊 Grade (6D)", variant="secondary", size="sm")
                        state_btn = gr.Button("📋 Refresh State", variant="secondary", size="sm")

                # RIGHT COLUMN
                with gr.Column(scale=3, min_width=480):

                    gr.HTML('<div class="sec-head">Observation</div>')
                    obs_display = gr.HTML(
                        "<div style='text-align:center;padding:48px 16px;color:#4a5568;font-family:Syne,sans-serif;'>"
                        "<div style='font-size:32px;margin-bottom:12px;opacity:0.3;'>🔕</div>"
                        "<p style='font-size:13px;'>Select a task and click <strong style='color:#8b95aa;'>Reset Environment</strong> to begin.</p>"
                        "</div>"
                    )

                    gr.HTML('<div class="sec-head">Action History</div>')
                    history_display = gr.HTML(_render_history([]))

                    gr.HTML('<div class="sec-head">Step Reward</div>')
                    reward_display = gr.Markdown("*Start an episode first.*")

                    gr.HTML('<div class="sec-head">6D Score</div>')
                    score_display = gr.Markdown("*Click **📊 Grade** after executing actions.*")

            # Wire events
            reset_btn.click(
                fn=gr_reset,
                inputs=[task_dd, seed_tb],
                outputs=[obs_display, history_display, state_display, score_display, reward_display, service_dd],
            )
            step_btn.click(
                fn=gr_step,
                inputs=[action_dd, service_dd, keyword_tb, config_key_tb, config_val_tb, replicas_tb, cause_tb, severity_dd],
                outputs=[obs_display, history_display, state_display, score_display, reward_display],
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
