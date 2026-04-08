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
@import url('https://api.fontshare.com/v2/css?f[]=cabinet-grotesk@800,700,500,400&display=swap');
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;500&display=swap');

*, *::before, *::after { box-sizing: border-box; }

:root {
    --bg0: #080a0f;
    --bg1: #0d1018;
    --bg2: #111520;
    --bg3: #181c2a;
    --bg4: #1e2235;
    --line: rgba(255,255,255,0.06);
    --line2: rgba(255,255,255,0.10);
    --text0: #eef0f6;
    --text1: #9ba3bc;
    --text2: #5a6380;
    --text3: #373d55;
    --amber: #f5a623;
    --amber2: #ffc55a;
    --amber-bg: rgba(245,166,35,0.08);
    --cyan: #22d3ee;
    --cyan-bg: rgba(34,211,238,0.08);
    --green: #34d399;
    --green-bg: rgba(52,211,153,0.08);
    --red: #f87171;
    --red-bg: rgba(248,113,113,0.08);
    --violet: #a78bfa;
    --violet-bg: rgba(167,139,250,0.08);
    --r: 10px;
    --r2: 14px;
    --r3: 18px;
    --font-head: 'Cabinet Grotesk', system-ui, sans-serif;
    --font-mono: 'JetBrains Mono', 'Courier New', monospace;
}

body, html {
    background: var(--bg0) !important;
    color: var(--text0) !important;
    font-family: var(--font-head) !important;
    -webkit-font-smoothing: antialiased;
}

.gradio-container {
    background: var(--bg0) !important;
    max-width: 100% !important;
    margin: 0 !important;
    padding: 0 !important;
    min-height: 100vh;
}

footer, .svelte-1ipelgc { display: none !important; }
#component-0 { background: var(--bg0) !important; }

.tab-nav {
    background: var(--bg1) !important;
    border-bottom: 1px solid var(--line) !important;
    padding: 0 40px !important;
    gap: 0 !important;
}
.tab-nav button {
    font-family: var(--font-mono) !important;
    font-size: 11px !important;
    font-weight: 400 !important;
    letter-spacing: 0.08em !important;
    color: var(--text3) !important;
    background: transparent !important;
    border: none !important;
    border-bottom: 1.5px solid transparent !important;
    border-radius: 0 !important;
    padding: 16px 20px !important;
    transition: color 0.2s !important;
}
.tab-nav button:hover { color: var(--text1) !important; }
.tab-nav button.selected {
    color: var(--text0) !important;
    border-bottom-color: var(--amber) !important;
}

.gr-group, .gr-box, .gr-form, div.gr-block {
    background: var(--bg2) !important;
    border: 1px solid var(--line) !important;
    border-radius: var(--r2) !important;
}

input[type="text"], input[type="number"], textarea, select {
    background: var(--bg1) !important;
    border: 1px solid var(--line) !important;
    border-radius: var(--r) !important;
    color: var(--text0) !important;
    font-family: var(--font-mono) !important;
    font-size: 12px !important;
    padding: 10px 13px !important;
    transition: border-color 0.15s, box-shadow 0.15s !important;
    outline: none !important;
}
input:focus, textarea:focus, select:focus {
    border-color: rgba(245,166,35,0.35) !important;
    box-shadow: 0 0 0 3px rgba(245,166,35,0.06) !important;
}
input::placeholder, textarea::placeholder { color: var(--text3) !important; }

label span, .gr-input-label, .block-title {
    font-family: var(--font-mono) !important;
    font-size: 10px !important;
    font-weight: 400 !important;
    letter-spacing: 0.12em !important;
    text-transform: uppercase !important;
    color: var(--text2) !important;
}

.gr-button {
    font-family: var(--font-mono) !important;
    font-size: 11px !important;
    font-weight: 500 !important;
    letter-spacing: 0.08em !important;
    border-radius: var(--r) !important;
    transition: all 0.18s !important;
    cursor: pointer !important;
}
.gr-button-primary {
    background: var(--amber) !important;
    color: #0a0700 !important;
    border: none !important;
    font-weight: 700 !important;
    box-shadow: 0 1px 0 0 rgba(0,0,0,0.4), inset 0 1px 0 rgba(255,255,255,0.15) !important;
}
.gr-button-primary:hover {
    background: var(--amber2) !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 20px rgba(245,166,35,0.3) !important;
}
.gr-button-secondary {
    background: transparent !important;
    color: var(--text1) !important;
    border: 1px solid var(--line2) !important;
}
.gr-button-secondary:hover {
    background: var(--bg3) !important;
    color: var(--text0) !important;
}

.gr-markdown, .md, .prose {
    color: var(--text1) !important;
    font-family: var(--font-head) !important;
    background: transparent !important;
    font-size: 14px !important;
    line-height: 1.7 !important;
}
.gr-markdown h1, .md h1 {
    font-family: var(--font-head) !important;
    font-size: 28px !important; font-weight: 800 !important;
    color: var(--text0) !important; letter-spacing: -0.4px !important;
    margin-bottom: 6px !important;
}
.gr-markdown h2, .md h2 {
    font-size: 18px !important; font-weight: 700 !important;
    color: var(--text0) !important;
    margin-top: 28px !important; margin-bottom: 10px !important;
    padding-bottom: 8px !important;
    border-bottom: 1px solid var(--line) !important;
    font-family: var(--font-head) !important;
}
.gr-markdown h3, .md h3 {
    font-family: var(--font-mono) !important;
    font-size: 11px !important; font-weight: 400 !important;
    color: var(--amber) !important; letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
    margin-top: 20px !important; margin-bottom: 8px !important;
}
.gr-markdown code, .md code {
    font-family: var(--font-mono) !important;
    font-size: 11.5px !important;
    background: rgba(245,166,35,0.07) !important;
    color: var(--amber) !important;
    padding: 2px 6px !important;
    border-radius: 5px !important;
    border: 1px solid rgba(245,166,35,0.14) !important;
}
.gr-markdown pre, .md pre {
    background: var(--bg1) !important;
    border: 1px solid var(--line) !important;
    border-left: 2px solid var(--amber) !important;
    border-radius: var(--r) !important;
    padding: 16px 18px !important;
}
.gr-markdown pre code, .md pre code {
    background: none !important; border: none !important;
    color: var(--text1) !important; font-size: 12px !important;
    line-height: 1.85 !important; padding: 0 !important;
}
.gr-markdown table, .md table { border-collapse: collapse !important; width: 100% !important; margin: 14px 0 !important; }
.gr-markdown th, .md th {
    font-family: var(--font-mono) !important;
    font-size: 10px !important; letter-spacing: 0.12em !important;
    text-transform: uppercase !important; font-weight: 400 !important;
    color: var(--amber) !important;
    background: rgba(245,166,35,0.05) !important;
    padding: 9px 13px !important; border: 1px solid var(--line) !important;
}
.gr-markdown td, .md td {
    padding: 8px 13px !important;
    border: 1px solid rgba(255,255,255,0.04) !important;
    color: var(--text1) !important; font-size: 13px !important;
}
.gr-markdown tr:hover td { background: rgba(255,255,255,0.02) !important; }
.gr-markdown strong, .md strong { color: var(--text0) !important; font-weight: 700 !important; }
.gr-markdown blockquote, .md blockquote {
    border-left: 2px solid var(--cyan) !important;
    background: var(--cyan-bg) !important;
    padding: 10px 16px !important; margin: 14px 0 !important;
    border-radius: 0 var(--r) var(--r) 0 !important;
    color: var(--text1) !important;
}
.gr-markdown a, .md a { color: var(--cyan) !important; }
.gr-markdown li, .md li { margin-bottom: 5px !important; }

.gr-accordion {
    background: var(--bg2) !important;
    border: 1px solid var(--line) !important;
    border-radius: var(--r2) !important;
}
.gr-accordion > button {
    font-family: var(--font-mono) !important;
    font-size: 10px !important; letter-spacing: 0.12em !important;
    text-transform: uppercase !important;
    color: var(--text2) !important; background: transparent !important;
    padding: 14px 16px !important;
}
.gr-accordion > button:hover { color: var(--text0) !important; }

.gr-dropdown ul {
    background: var(--bg3) !important;
    border: 1px solid var(--line2) !important;
    border-radius: var(--r) !important;
    box-shadow: 0 8px 32px rgba(0,0,0,0.5) !important;
}
.gr-dropdown li { color: var(--text1) !important; font-family: var(--font-mono) !important; font-size: 12px !important; }
.gr-dropdown li:hover { background: var(--amber-bg) !important; color: var(--amber) !important; }

::-webkit-scrollbar { width: 4px; height: 4px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: var(--line2); border-radius: 2px; }
::-webkit-scrollbar-thumb:hover { background: rgba(245,166,35,0.3); }

.gr-row { gap: 18px !important; }
.gr-column { gap: 12px !important; }

/* ═══════════════ CUSTOM COMPONENTS ═══════════════ */

.ir-section {
    font-family: var(--font-mono, monospace);
    font-size: 10px; font-weight: 400;
    letter-spacing: 0.16em; text-transform: uppercase;
    color: var(--text3, #373d55);
    padding: 16px 0 8px;
    display: flex; align-items: center; gap: 10px;
    user-select: none;
}
.ir-section::after { content: ''; flex: 1; height: 1px; background: var(--line, rgba(255,255,255,0.06)); }

.mc {
    background: var(--bg2);
    border: 1px solid var(--line);
    border-radius: var(--r);
    padding: 14px 16px;
    position: relative; overflow: hidden;
    transition: border-color 0.2s;
}
.mc:hover { border-color: var(--line2); }
.mc-accent { position: absolute; top: 0; left: 0; right: 0; height: 2px; }
.mc-label { font-family: var(--font-mono); font-size: 9px; letter-spacing: 0.16em; text-transform: uppercase; color: var(--text2); margin-bottom: 8px; }
.mc-value { font-family: var(--font-head); font-size: 26px; font-weight: 800; letter-spacing: -0.5px; line-height: 1; color: var(--text0); }

.alert-card { display: flex; gap: 12px; padding: 13px 15px; border-radius: var(--r); border: 1px solid; margin-bottom: 8px; transition: transform 0.15s; }
.alert-card:hover { transform: translateX(3px); }
.alert-badge { font-family: var(--font-mono); font-size: 9px; font-weight: 500; letter-spacing: 0.12em; padding: 3px 8px; border-radius: 4px; flex-shrink: 0; margin-top: 1px; height: fit-content; }
.alert-name { font-family: var(--font-head); font-size: 13px; font-weight: 700; color: var(--text0); margin-bottom: 3px; }
.alert-id { font-family: var(--font-mono); font-size: 10px; color: var(--text2); font-weight: 300; margin-left: 6px; }
.alert-msg { font-family: var(--font-head); font-size: 12px; color: var(--text1); line-height: 1.5; }

.svc-row { display: grid; grid-template-columns: 1fr auto auto auto; align-items: center; padding: 10px 14px; border-bottom: 1px solid rgba(255,255,255,0.03); transition: background 0.15s; }
.svc-row:hover { background: rgba(255,255,255,0.02); }
.svc-row:last-child { border-bottom: none; }
.svc-name { font-family: var(--font-head); font-size: 13px; font-weight: 600; color: var(--text0); }
.svc-pill { display: inline-flex; align-items: center; gap: 5px; font-family: var(--font-mono); font-size: 10px; padding: 3px 9px; border-radius: 100px; }
.svc-stat { font-family: var(--font-mono); font-size: 11px; color: var(--text1); text-align: right; min-width: 70px; padding: 0 10px; }

.action-chip { display: inline-flex; align-items: center; gap: 4px; font-family: var(--font-mono); font-size: 10px; padding: 2px 8px; border-radius: 100px; white-space: nowrap; }

.log-block { font-family: var(--font-mono); font-size: 11px; line-height: 1.9; background: var(--bg1); border: 1px solid var(--line); border-left: 2px solid var(--cyan); border-radius: var(--r); padding: 13px 16px; color: var(--text1); max-height: 200px; overflow-y: auto; white-space: pre-wrap; word-break: break-all; }

.prog-wrap { position: relative; height: 3px; background: rgba(255,255,255,0.05); border-radius: 2px; overflow: hidden; margin: 10px 0 5px; }
.prog-fill { position: absolute; inset: 0; border-radius: 2px; background: linear-gradient(90deg, var(--amber), var(--cyan)); transition: width 0.5s cubic-bezier(.4,0,.2,1); }

.empty-state { display: flex; flex-direction: column; align-items: center; justify-content: center; padding: 44px 24px; text-align: center; }
.empty-icon { width: 40px; height: 40px; border-radius: 10px; background: var(--bg3); border: 1px solid var(--line); display: flex; align-items: center; justify-content: center; font-size: 18px; margin-bottom: 12px; opacity: 0.5; }
.empty-label { font-family: var(--font-head); font-size: 13px; color: var(--text2); line-height: 1.6; }
.empty-label strong { color: var(--text1); font-weight: 600; }

.hist-table { width: 100%; border-collapse: collapse; }
.hist-th { font-family: var(--font-mono); font-size: 9px; letter-spacing: 0.14em; text-transform: uppercase; color: var(--text3); padding: 0 12px 8px; text-align: left; border-bottom: 1px solid var(--line); }
.hist-td { font-family: var(--font-mono); font-size: 11px; padding: 9px 12px; border-bottom: 1px solid rgba(255,255,255,0.03); color: var(--text1); vertical-align: middle; }
.hist-row:hover .hist-td { background: rgba(255,255,255,0.015); }
.hist-row:last-child .hist-td { border-bottom: none; }

.panel-label { font-family: var(--font-mono); font-size: 9px; letter-spacing: 0.18em; text-transform: uppercase; color: var(--text3); margin-bottom: 6px; padding-left: 8px; border-left: 2px solid var(--amber); }

@keyframes pulse-dot { 0%,100%{opacity:1;} 50%{opacity:0.3;} }
.live-dot { width: 6px; height: 6px; border-radius: 50%; display: inline-block; flex-shrink: 0; animation: pulse-dot 2s ease-in-out infinite; }
"""


# ─── HELPERS ─────────────────────────────────────────────────────────────────

_DIAG_ACTIONS = {"investigate_logs","check_metrics","read_config","check_service_health","run_diagnostic"}
_FIX_ACTIONS  = {"restart_service","update_config","rollback_deployment","scale_service"}
_FAILURE_TYPE_ICON = {
    "Efficient Reasoner":        "◆",
    "Symptom Chaser":            "◇",
    "Lucky Guesser":             "○",
    "Stuck in Observation Loop": "⊗",
    "Late Corrector":            "◎",
    "Unknown":                   "·",
}
_DIFF_MAP = {
    "db_connection_failure":     ("Easy",   "var(--green)",  "var(--green-bg)"),
    "cascading_service_timeout": ("Medium", "var(--amber)",  "var(--amber-bg)"),
    "ssl_certificate_expiry":    ("Medium", "var(--amber)",  "var(--amber-bg)"),
    "multi_factor_outage":       ("Hard",   "var(--red)",    "var(--red-bg)"),
    "database_deadlock":         ("Hard",   "var(--red)",    "var(--red-bg)"),
    "alert_triage":              ("Triage", "var(--cyan)",   "var(--cyan-bg)"),
}
_action_history = []


def _render_alerts(alerts):
    if not alerts:
        return """<div class="empty-state">
  <div class="empty-icon">🔕</div>
  <div class="empty-label">No active alerts<br><strong>Reset an episode to populate</strong></div>
</div>"""
    out = ""
    for a in alerts:
        sev = (a.get("severity","medium")).upper()
        if sev in ("CRITICAL","HIGH"):
            bc, bb, bl = "var(--red)", "var(--red-bg)", "rgba(248,113,113,0.2)"
        elif sev == "MEDIUM":
            bc, bb, bl = "var(--amber)", "var(--amber-bg)", "rgba(245,166,35,0.2)"
        else:
            bc, bb, bl = "var(--green)", "var(--green-bg)", "rgba(52,211,153,0.2)"
        out += f"""
<div class="alert-card" style="background:{bb};border-color:{bl};">
  <span class="alert-badge" style="color:{bc};background:{bb};border:1px solid {bl};">{sev}</span>
  <div style="min-width:0;">
    <div class="alert-name">{a.get('service','')}<span class="alert-id">{a.get('alert_id','')}</span></div>
    <div class="alert-msg">{a.get('message','')}</div>
  </div>
</div>"""
    return out


def _render_services(statuses):
    if not statuses:
        return ""
    rows = ""
    for s in statuses:
        ok   = s.get("healthy", True)
        pc   = "var(--green)" if ok else "var(--red)"
        pb   = "var(--green-bg)" if ok else "var(--red-bg)"
        pt   = "● healthy" if ok else "● error"
        rt   = f"{s['response_time_ms']:.0f} ms" if s.get("response_time_ms") is not None else "—"
        er_r = s.get("error_rate")
        er   = f"{er_r*100:.1f}%" if er_r is not None else "—"
        ec   = "var(--red)" if not ok else "var(--text2)"
        rows += f"""
<div class="svc-row">
  <span class="svc-name">{s['name']}</span>
  <span class="svc-pill" style="color:{pc};background:{pb};border:1px solid {pc}25;">{pt}</span>
  <span class="svc-stat">{rt}</span>
  <span class="svc-stat" style="color:{ec};">{er}</span>
</div>"""
    return f"""
<div style="margin-top:20px;">
  <div class="panel-label" style="margin-bottom:0;">Service health</div>
  <div style="background:var(--bg2);border:1px solid var(--line);border-radius:var(--r);margin-top:8px;overflow:hidden;">
    <div style="display:grid;grid-template-columns:1fr auto auto auto;padding:7px 14px;border-bottom:1px solid var(--line);">
      <span style="font-family:var(--font-mono);font-size:9px;letter-spacing:0.14em;text-transform:uppercase;color:var(--text3);">Service</span>
      <span style="font-family:var(--font-mono);font-size:9px;letter-spacing:0.14em;text-transform:uppercase;color:var(--text3);min-width:80px;text-align:right;padding-right:10px;">Status</span>
      <span style="font-family:var(--font-mono);font-size:9px;letter-spacing:0.14em;text-transform:uppercase;color:var(--text3);min-width:70px;text-align:right;padding:0 10px;">Latency</span>
      <span style="font-family:var(--font-mono);font-size:9px;letter-spacing:0.14em;text-transform:uppercase;color:var(--text3);min-width:70px;text-align:right;">Err%</span>
    </div>
    {rows}
  </div>
</div>"""


def _render_log(text):
    if not text:
        return ""
    import html as _html, re
    esc = _html.escape(str(text))
    esc = re.sub(r'(ERROR|CRITICAL|FAILED|refused|Connection refused)',
                 r'<span style="color:var(--red);">\1</span>', esc, flags=re.IGNORECASE)
    esc = re.sub(r'(SUCCESS|healthy|resolved|updated successfully)',
                 r'<span style="color:var(--green);">\1</span>', esc, flags=re.IGNORECASE)
    esc = re.sub(r'(\[\d{2}:\d{2}:\d{2}\]|\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2})',
                 r'<span style="color:var(--text3);">\1</span>', esc)
    return f"""
<div style="margin-top:20px;">
  <div class="panel-label" style="border-left-color:var(--cyan);margin-bottom:8px;">Last output</div>
  <div class="log-block">{esc}</div>
</div>"""


def _render_obs(obs_dump, action_result):
    task  = obs_dump.get("task_name", "—")
    step  = obs_dump.get("step_number", 0)
    max_s = obs_dump.get("max_steps", 30)
    pct   = int(step / max(max_s, 1) * 100)
    dlabel, dcolor, dbg = _DIFF_MAP.get(task, ("?", "var(--text1)", "var(--bg3)"))
    alerts_html   = _render_alerts(obs_dump.get("active_alerts", []))
    services_html = _render_services(obs_dump.get("service_statuses", []))
    log_html      = _render_log(action_result) if action_result else ""
    return f"""
<div style="font-family:var(--font-head);">
  <div style="display:flex;align-items:center;gap:10px;padding:12px 0 14px;border-bottom:1px solid var(--line);margin-bottom:16px;">
    <span class="live-dot" style="background:var(--green);box-shadow:0 0 0 3px var(--green-bg);"></span>
    <span style="font-family:var(--font-mono);font-size:11px;color:var(--text1);letter-spacing:0.04em;">{task}</span>
    <span style="font-family:var(--font-mono);font-size:9px;padding:2px 9px;border-radius:4px;color:{dcolor};background:{dbg};border:1px solid {dcolor}25;">{dlabel}</span>
    <div style="margin-left:auto;display:flex;align-items:baseline;gap:6px;">
      <span style="font-family:var(--font-head);font-size:22px;font-weight:800;color:var(--text0);letter-spacing:-0.5px;">{step}</span>
      <span style="font-family:var(--font-mono);font-size:10px;color:var(--text2);">/ {max_s} steps</span>
    </div>
  </div>
  <div class="prog-wrap"><div class="prog-fill" style="width:{pct}%;"></div></div>
  <div style="display:flex;justify-content:space-between;font-family:var(--font-mono);font-size:9px;color:var(--text3);margin-bottom:18px;">
    <span>Progress</span><span>{pct}%</span>
  </div>
  <div class="panel-label" style="margin-bottom:10px;">Active alerts</div>
  {alerts_html}
  {services_html}
  {log_html}
</div>"""


def _render_history(history):
    if not history:
        return """<div class="empty-state">
  <div class="empty-icon">◻</div>
  <div class="empty-label">No actions yet<br><strong>Execute an action to begin</strong></div>
</div>"""
    rows = ""
    for h in history:
        act     = h.get("action","")
        is_diag = act in _DIAG_ACTIONS
        is_fix  = act in _FIX_ACTIONS
        if is_diag:   cc, cb, icon = "var(--cyan)", "var(--cyan-bg)", "◈ diag"
        elif is_fix:  cc, cb, icon = "var(--green)", "var(--green-bg)", "⬡ fix"
        else:         cc, cb, icon = "var(--amber)", "var(--amber-bg)", "◆ decl"
        r     = h.get("reward", 0.0)
        r_str = f"+{r:.3f}" if r >= 0 else f"{r:.3f}"
        r_col = "var(--green)" if r >= 0 else "var(--red)"
        rows += f"""
<tr class="hist-row">
  <td class="hist-td" style="color:var(--text3);width:36px;">{h['step']}</td>
  <td class="hist-td"><span class="action-chip" style="color:{cc};background:{cb};border:1px solid {cc}20;">{icon}</span></td>
  <td class="hist-td" style="color:var(--text0);font-weight:500;">{act}</td>
  <td class="hist-td" style="color:var(--text2);">{h.get('service') or '—'}</td>
  <td class="hist-td" style="color:{r_col};font-weight:500;text-align:right;">{r_str}</td>
</tr>"""
    return f"""
<div style="overflow-x:auto;">
<table class="hist-table">
  <thead><tr>{''.join(f'<th class="hist-th">{x}</th>' for x in ['#','type','action','service','reward'])}</tr></thead>
  <tbody>{rows}</tbody>
</table>
</div>"""


def _render_state_panel():
    if env._task_name == "alert_triage":
        step = env._step_number
        cum  = round(env._cumulative_reward, 4)
        sub  = env._at_submitted_severity or "—"
        pct  = int(step / 3 * 100)
        cc   = "var(--green)" if cum >= 0 else "var(--red)"
        return f"""
<div>
  <div style="display:grid;grid-template-columns:1fr 1fr;gap:8px;margin-bottom:8px;">
    <div class="mc"><div class="mc-accent" style="background:var(--cyan);"></div>
      <div class="mc-label">Step</div>
      <div class="mc-value" style="color:var(--cyan);">{step}<span style="font-size:14px;color:var(--text2);font-weight:400;">/3</span></div>
    </div>
    <div class="mc"><div class="mc-accent" style="background:{cc};"></div>
      <div class="mc-label">Cum. Reward</div>
      <div class="mc-value" style="color:{cc};font-size:20px;">{'+' if cum>=0 else ''}{cum:.4f}</div>
    </div>
  </div>
  <div class="mc" style="margin-bottom:8px;"><div class="mc-accent" style="background:var(--violet);"></div>
    <div class="mc-label">Submitted severity</div>
    <div class="mc-value" style="color:var(--violet);font-size:22px;">{sub}</div>
  </div>
  <div class="prog-wrap"><div class="prog-fill" style="width:{pct}%;"></div></div>
  <div style="font-family:var(--font-mono);font-size:9px;color:var(--text3);text-align:right;">{pct}%</div>
</div>"""

    if not env._scenario:
        return """<div class="empty-state" style="padding:28px 16px;">
  <div class="empty-icon">◌</div>
  <div class="empty-label">Select a task → Reset</div>
</div>"""

    task     = env._task_name or "—"
    step     = env._step_number
    max_s    = env._scenario.max_steps
    resolved = env._incident_resolved
    done     = env._done
    cum      = round(env._cumulative_reward, 4)
    pct      = int(step / max(max_s,1) * 100)
    streak   = env._consecutive_diagnosis_count
    coll     = len(env._collateral_degraded)
    cc       = "var(--green)" if cum >= 0 else "var(--red)"
    sc       = "var(--red)" if streak >= 2 else "var(--amber)" if streak == 1 else "var(--text2)"
    if resolved: status_c, status_t = "var(--green)", "resolved"
    elif done:   status_c, status_t = "var(--red)",   "done — not resolved"
    else:        status_c, status_t = "var(--amber)",  "active"
    dlabel, dcolor, dbg = _DIFF_MAP.get(task, ("?", "var(--text1)", "var(--bg3)"))

    return f"""
<div>
  <div style="display:flex;align-items:center;gap:8px;margin-bottom:12px;">
    <span class="live-dot" style="background:{status_c};box-shadow:0 0 0 3px {status_c}20;"></span>
    <span style="font-family:var(--font-mono);font-size:10px;color:{status_c};letter-spacing:0.08em;">{status_t}</span>
    <span style="margin-left:auto;font-family:var(--font-mono);font-size:9px;padding:2px 9px;border-radius:4px;color:{dcolor};background:{dbg};border:1px solid {dcolor}25;">{dlabel}</span>
  </div>
  <div style="display:grid;grid-template-columns:1fr 1fr;gap:8px;margin-bottom:8px;">
    <div class="mc"><div class="mc-accent" style="background:var(--cyan);"></div>
      <div class="mc-label">Step</div>
      <div class="mc-value" style="color:var(--cyan);font-size:22px;">{step}<span style="font-size:13px;color:var(--text2);font-weight:400;"> /{max_s}</span></div>
    </div>
    <div class="mc"><div class="mc-accent" style="background:{cc};"></div>
      <div class="mc-label">Cum. Reward</div>
      <div class="mc-value" style="color:{cc};font-size:18px;">{'+' if cum>=0 else ''}{cum:.4f}</div>
    </div>
    <div class="mc"><div class="mc-accent" style="background:{sc};"></div>
      <div class="mc-label">Diag streak</div>
      <div class="mc-value" style="color:{sc};font-size:22px;">{streak}{'⚠' if streak>=2 else ''}</div>
    </div>
    <div class="mc"><div class="mc-accent" style="background:var(--violet);"></div>
      <div class="mc-label">Collateral</div>
      <div class="mc-value" style="color:var(--violet);font-size:22px;">{coll}</div>
    </div>
  </div>
  <div style="display:flex;align-items:center;justify-content:space-between;
              background:var(--bg2);border:1px solid var(--line);border-radius:var(--r);
              padding:9px 14px;margin-bottom:8px;">
    <span style="font-family:var(--font-mono);font-size:10px;letter-spacing:0.1em;text-transform:uppercase;color:var(--text2);">Incident resolved</span>
    {'<span style="font-family:var(--font-mono);font-size:11px;color:var(--green);font-weight:500;">YES ✓</span>' if resolved else '<span style="font-family:var(--font-mono);font-size:11px;color:var(--red);">NO ✗</span>'}
  </div>
  <div class="prog-wrap"><div class="prog-fill" style="width:{pct}%;"></div></div>
  <div style="display:flex;justify-content:space-between;font-family:var(--font-mono);font-size:9px;color:var(--text3);">
    <span>{step} / {max_s} steps</span><span>{pct}%</span>
  </div>
</div>"""


def _render_score(breakdown):
    if not breakdown:
        return "*Execute actions, then click **Grade** to evaluate.*"
    final    = breakdown.get("final", 0.0)
    ft       = breakdown.get("failure_type", "Unknown")
    icon     = _FAILURE_TYPE_ICON.get(ft, "·")
    obs_loop = breakdown.get("observation_loop", False)
    is_triage = "severity_match" in breakdown.get("breakdown", {})
    if is_triage:
        bd  = breakdown.get("breakdown", {})
        return f"""### {icon} Triage Score: **{final:.4f}** / 1.0

| Component | Value |
|---|---|
| Submitted | `{bd.get('submitted_severity','—')}` |
| Correct | `{bd.get('correct_severity','—')}` |
| Severity score | `{bd.get('severity_match', 0.0):.2f}` |
| Investigation bonus | `+{bd.get('investigation_bonus', 0.0):.2f}` |
| **Total** | **`{final:.4f}`** |

{('> ⚠ ' + breakdown.get('feedback','')) if breakdown.get('feedback') else ''}"""

    dims = [
        ("root_cause","Root Cause",0.30),("remediation","Remediation",0.25),
        ("investigation","Investigation",0.15),("efficiency","Efficiency",0.10),
        ("safety","Safety",0.10),("sequence","Sequence",0.10),
    ]
    rows = "\n".join(
        f"| {l} | `{breakdown.get(k,0.0):.2f}` | ×{w:.2f} | {'█'*int(breakdown.get(k,0)*10)}{'░'*(10-int(breakdown.get(k,0)*10))} |"
        for k,l,w in dims
    )
    fb       = breakdown.get("feedback","")
    obs_note = "\n> ⚠ Observation loop — score capped at 0.45" if obs_loop else ""
    return f"""### {icon} Final Score: **{final:.4f}** / 1.0

**Pattern:** {ft}{obs_note}

#### Dimension breakdown
| Dimension | Score | Weight | Bar |
|---|---:|---:|---|
{rows}

{('> ' + fb) if fb else ''}"""


# ── Gradio callbacks ──────────────────────────────────────────────────────────

def gr_reset(task_name, seed_str):
    global _action_history
    _action_history = []
    try:
        seed = int(seed_str) if seed_str.strip() else None
        obs  = env.reset(task_name=task_name, seed=seed)
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
        params = {}
        at = action_type
        if at == "investigate_logs" and keyword.strip():     params["keyword"] = keyword.strip()
        elif at == "check_metrics":                          params["metric_type"] = "all"
        elif at == "update_config":
            if config_key.strip(): params["key"]   = config_key.strip()
            if config_val.strip(): params["value"] = config_val.strip()
        elif at == "scale_service" and replicas_str.strip():
            try: params["replicas"] = int(replicas_str.strip())
            except ValueError: pass
        elif at == "declare_root_cause" and cause_text.strip(): params["cause"] = cause_text.strip()
        elif at == "submit_severity" and severity_val.strip():  params["severity"] = severity_val.strip()

        svc    = service_name if service_name and service_name.strip() else None
        action = IncidentResponseAction(action_type=ActionType(at), service_name=svc, parameters=params)
        obs, reward, done, info = env.step(action)
        obs_dump = obs.model_dump()
        _action_history.append({"step": obs.step_number, "action": at, "service": svc, "reward": reward})

        ft      = info.get("failure_type","N/A")
        ft_icon = _FAILURE_TYPE_ICON.get(ft,"·")
        r_sign  = "+" if reward >= 0 else ""
        reward_md = f"""### Step reward: `{r_sign}{reward:.4f}`
**Cumulative:** `{env._cumulative_reward:+.4f}`

**Feedback:** {obs.feedback}

{ft_icon} **{ft}** · Loop: `{info.get('observation_loop', False)}` · Streak: `{info.get('consecutive_diagnosis_count', 0)}`
{chr(10) + '---' + chr(10) + '**Episode complete.** Click **Grade** for your final 6D score.' if done else ''}"""
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
    try:    return _render_score(env.get_score_breakdown())
    except Exception as e: return f"❌ {e}"

def gr_state():
    return _render_state_panel()


# ── Header ────────────────────────────────────────────────────────────────────

HEADER_HTML = """
<div style="background:#0d1018;border-bottom:1px solid rgba(255,255,255,0.06);
            padding:0 40px;height:62px;display:flex;align-items:center;
            justify-content:space-between;position:relative;">
  <div style="position:absolute;top:0;left:0;right:0;height:1px;
              background:linear-gradient(90deg,#f5a623 0%,#22d3ee 40%,transparent 100%);opacity:0.4;"></div>
  <div style="display:flex;align-items:center;gap:14px;">
    <div style="width:34px;height:34px;border-radius:9px;
                background:rgba(245,166,35,0.1);border:1px solid rgba(245,166,35,0.18);
                display:flex;align-items:center;justify-content:center;font-size:16px;">🚨</div>
    <div>
      <div style="font-family:'Cabinet Grotesk',system-ui,sans-serif;font-size:16px;
                  font-weight:800;color:#eef0f6;letter-spacing:-0.3px;line-height:1.1;">
        Incident Response Environment
      </div>
      <div style="font-family:'JetBrains Mono',monospace;font-size:9px;
                  color:#373d55;letter-spacing:0.18em;text-transform:uppercase;margin-top:2px;">
        SRE AI Training Platform · v4.0
      </div>
    </div>
  </div>
  <div style="display:flex;align-items:center;gap:8px;">
    <span style="font-family:'JetBrains Mono',monospace;font-size:9px;letter-spacing:0.14em;
                 text-transform:uppercase;color:#5a6380;padding:5px 11px;
                 border:1px solid rgba(255,255,255,0.07);border-radius:6px;">6D Scoring</span>
    <span style="font-family:'JetBrains Mono',monospace;font-size:9px;letter-spacing:0.14em;
                 text-transform:uppercase;color:#34d399;padding:5px 11px;
                 border:1px solid rgba(52,211,153,0.2);border-radius:6px;
                 background:rgba(52,211,153,0.07);">● Live</span>
  </div>
</div>
"""


# ── Blocks ────────────────────────────────────────────────────────────────────

with gr.Blocks(
    title="Incident Response Environment v4.0",
    css=CUSTOM_CSS,
    theme=gr.themes.Base(
        primary_hue=gr.themes.colors.orange,
        secondary_hue=gr.themes.colors.slate,
        neutral_hue=gr.themes.colors.slate,
        font=[gr.themes.GoogleFont("DM Sans"), "sans-serif"],
        font_mono=[gr.themes.GoogleFont("JetBrains Mono"), "monospace"],
    ),
) as web_ui:

    gr.HTML(HEADER_HTML)

    with gr.Tabs():

        with gr.TabItem("overview"):
            gr.Markdown("""
# The flight simulator for SREs

Train AI agents — and humans — to diagnose real production incidents with precision.

---

## Workflow

| Phase | Actions | Purpose |
|-------|---------|---------|
| **Investigate** | check_service_health, investigate_logs, check_metrics, read_config | Gather evidence |
| **Diagnose** | declare_root_cause | Name the problem |
| **Remediate** | update_config, restart_service, rollback_deployment, scale_service | Apply the fix |

---

## 6-Dimension Scoring

| Dimension | Weight | What it measures |
|-----------|--------|-----------------|
| Root Cause | **30%** | Did you identify the right problem? |
| Remediation | **25%** | Did you apply the right fix? |
| Investigation | **15%** | Did you check enough services? |
| Efficiency | **10%** | Did you use steps wisely? |
| Safety | **10%** | Did you avoid collateral damage? |
| Sequence | **10%** | Did you diagnose before fixing? |

> Observation loops (3+ diagnosis actions without a fix) cap the score at **0.45**.

---

## Scenarios

| Scenario | Difficulty | Root causes |
|----------|-----------|------------|
| db_connection_failure | Easy | 1 |
| cascading_service_timeout | Medium | 2 |
| ssl_certificate_expiry | Medium | 1 |
| multi_factor_outage | Hard | 3 |
| database_deadlock | Hard | 1 |
| alert_triage | Triage | classify P1–P4 |
            """)

        with gr.TabItem("walkthrough"):
            gr.Markdown("""
# Step-by-step: DB Connection Failure

A complete annotated solve from alert to resolution, scoring **0.89**.

---

## The alert (on reset)

```
● CRITICAL  [ALT-001]  user-api  T=08:15:23
  user-api returning 503 errors

✗ user-api          error_rate=85%   rt=450ms
✓ postgres-primary  healthy
✓ nginx-lb          healthy
```

---

## Investigation (steps 1–3)

**Step 1** — `check_service_health` → `user-api` → `Reward: +0.04`

**Step 2** — `investigate_logs` → `user-api` · keyword: `connection`
```
[08:15:12] ERROR: Connection refused for postgres-primary:5433
Port 5433 is suspicious — default is 5432.  Reward: +0.08
```

**Step 3** — `read_config` → `user-api`
```json
{ "db_port": 5433 }   ← should be 5432    Reward: +0.06
```

---

## Diagnosis (step 4)

`declare_root_cause` → `"user-api db_port misconfigured as 5433 instead of 5432"`
```
✓ ROOT CAUSE MATCHED    Reward: +0.20
```

---

## Remediation (step 5)

`update_config` → `user-api` · key: `db_port` · value: `5432`
```
✓ Configuration updated · INCIDENT RESOLVED    Reward: +0.15
```

---

## Final score: 0.89 / 1.0

| Dimension | Score | Note |
|-----------|-------|------|
| Root Cause | 1.00 | Exact match |
| Remediation | 1.00 | Correct fix |
| Investigation | 0.67 | postgres-primary not checked |
| Efficiency | 1.00 | 5 steps only |
| Safety | 1.00 | No collateral |
| Sequence | 1.00 | Diagnosed before fixing |
            """)

        with gr.TabItem("faq"):
            gr.Markdown("""
# FAQ

### What is an observation loop?
3+ consecutive diagnosis actions with no fix in between. Score hard-capped at 0.45, penalty –0.08 per occurrence.

### What is collateral degradation?
Every 4 steps without resolution: error rates +10%, latency +40%. Eventually the full stack fails.

### What is a diagnosis gate?
Some scenarios require investigating a specific service *before* a fix takes full effect. Blind fixes → 50% remediation points.

### How does Alert Triage work?
Classify P1/P2/P3/P4 in 3 steps. Score = severity accuracy (1.0 exact / 0.5 adjacent / 0.25 two-off) + investigation bonus.

### What does a seed do?
Same seed → identical incident. Leave blank for random. `seed=42` for reproducible benchmarking.

### How do I maximise my score?
Investigate → Diagnose → Fix in that order. Never chain 3+ diagnosis actions without a fix. Declare root cause *before* any remediation.
            """)

        with gr.TabItem("sandbox"):
            with gr.Row(equal_height=False):

                with gr.Column(scale=2, min_width=340):
                    gr.HTML('<div class="ir-section">Episode Setup</div>')
                    with gr.Group():
                        with gr.Column():
                            task_dd = gr.Dropdown(
                                choices=[
                                    ("Easy — DB Connection Failure",      "db_connection_failure"),
                                    ("Medium — Cascading Timeout",        "cascading_service_timeout"),
                                    ("Medium — SSL Certificate Expiry",   "ssl_certificate_expiry"),
                                    ("Hard — Multi-Factor Outage",        "multi_factor_outage"),
                                    ("Hard — Database Deadlock",          "database_deadlock"),
                                    ("Triage — Alert Severity P1–P4",     "alert_triage"),
                                ],
                                value="db_connection_failure", label="Task",
                            )
                            seed_tb   = gr.Textbox(label="Seed (optional)", placeholder="e.g. 42", value="")
                            reset_btn = gr.Button("Reset Environment", variant="secondary", size="lg")

                    gr.HTML('<div class="ir-section">Episode State</div>')
                    state_display = gr.HTML(_render_state_panel())

                    gr.HTML('<div class="ir-section">Action Controls</div>')
                    with gr.Group():
                        with gr.Column():
                            action_dd = gr.Dropdown(
                                choices=[
                                    ("investigate_logs",     "investigate_logs"),
                                    ("check_metrics",        "check_metrics"),
                                    ("read_config",          "read_config"),
                                    ("check_service_health", "check_service_health"),
                                    ("run_diagnostic",       "run_diagnostic"),
                                    ("restart_service",      "restart_service"),
                                    ("update_config",        "update_config"),
                                    ("rollback_deployment",  "rollback_deployment"),
                                    ("scale_service",        "scale_service"),
                                    ("declare_root_cause",   "declare_root_cause"),
                                    ("submit_severity",      "submit_severity"),
                                ],
                                value="investigate_logs", label="Action Type",
                            )
                            service_dd = gr.Dropdown(choices=[], label="Target Service", allow_custom_value=True)

                    with gr.Accordion("Action Parameters", open=True):
                        keyword_tb    = gr.Textbox(label="Keyword (investigate_logs)", placeholder="error, timeout, connection", value="")
                        config_key_tb = gr.Textbox(label="Config Key (update_config)", placeholder="e.g. db_port", value="")
                        config_val_tb = gr.Textbox(label="Config Value (update_config)", placeholder="e.g. 5432", value="")
                        replicas_tb   = gr.Textbox(label="Replicas (scale_service)", placeholder="e.g. 3", value="")
                        cause_tb      = gr.Textbox(label="Root Cause (declare_root_cause)",
                                                   placeholder="e.g. user-api db_port misconfigured as 5433 instead of 5432",
                                                   lines=3, value="")
                        severity_dd   = gr.Dropdown(
                            choices=[
                                ("— not submitting", ""),
                                ("P1 — Critical outage", "P1"),
                                ("P2 — High: major degradation", "P2"),
                                ("P3 — Medium: partial, fallback active", "P3"),
                                ("P4 — Low: informational", "P4"),
                            ],
                            value="", label="Severity (submit_severity)",
                        )

                    step_btn = gr.Button("Execute Action", variant="primary", size="lg")

                    gr.HTML('<div class="ir-section">Scoring</div>')
                    with gr.Row():
                        grade_btn = gr.Button("Grade (6D)", variant="secondary", size="sm")
                        state_btn = gr.Button("Refresh State", variant="secondary", size="sm")

                with gr.Column(scale=3, min_width=460):
                    gr.HTML('<div class="ir-section">Observation</div>')
                    obs_display = gr.HTML("""
<div class="empty-state">
  <div class="empty-icon">◌</div>
  <div class="empty-label">Select a task and click <strong>Reset Environment</strong> to begin</div>
</div>""")

                    gr.HTML('<div class="ir-section">Action History</div>')
                    history_display = gr.HTML(_render_history([]))

                    gr.HTML('<div class="ir-section">Step Reward</div>')
                    reward_display = gr.Markdown("*Start an episode first.*")

                    gr.HTML('<div class="ir-section">6D Score</div>')
                    score_display = gr.Markdown("*Execute actions then click **Grade**.*")

            reset_btn.click(
                fn=gr_reset, inputs=[task_dd, seed_tb],
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
