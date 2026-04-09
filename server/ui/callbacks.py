"""
Gradio event callback functions for the Incident Response sandbox UI.

Each callback is wired to a Gradio component event in layout.py.
The global _action_history list tracks step details for the history
table and is reset on every gr_reset() call.
"""
from __future__ import annotations

import sys
import os
from typing import Any, Dict, List, Optional

_UI_DIR     = os.path.dirname(os.path.abspath(__file__))
_SERVER_DIR = os.path.dirname(_UI_DIR)
_PROJ_ROOT  = os.path.dirname(_SERVER_DIR)
for _p in (_SERVER_DIR, _PROJ_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import gradio as gr

from state import env
from models import ActionType, IncidentResponseAction
from ui.renderers import (
    render_obs,
    render_obs_done,
    render_history,
    render_state_panel,
    render_score,
    render_step_detail,
    render_step_detail_reset,
    render_step_detail_done,
)

# Per-episode action history (reset on gr_reset)
_action_history: List[Dict[str, Any]] = []


# ── Reset ─────────────────────────────────────────────────────────────────────

def gr_reset(task_name: str, seed_str: str):
    """Reset the environment and return initial display state."""
    global _action_history
    _action_history = []
    try:
        seed = int(seed_str) if seed_str.strip() else None
        obs  = env.reset(task_name=task_name, seed=seed)
        obs_dump = obs.model_dump()
        services = obs.available_services
        return (
            render_obs(obs_dump, obs.action_result),
            render_history([]),
            render_state_panel(env),
            "*Execute an action then click Grade.*",
            render_step_detail_reset(obs_dump),
            gr.Dropdown(choices=services, value=services[0] if services else None),
            gr.Button("Execute Action", variant="primary", interactive=True),
        )
    except Exception as e:
        err = f"❌ **Error:** {e}"
        return (err, "", err, "", "", gr.Dropdown(choices=[]),
                gr.Button("Execute Action", variant="primary", interactive=True))


# ── Step ──────────────────────────────────────────────────────────────────────

def gr_step(
    action_type:  str,
    service_name: str,
    keyword:      str,
    config_key:   str,
    config_val:   str,
    replicas_str: str,
    cause_text:   str,
    severity_val: str,
):
    """Execute one action and return updated display state."""
    global _action_history

    # Guard: episode already finished — reject further actions
    if env._done:
        return (
            render_obs_done(),
            render_history(_action_history),
            render_state_panel(env),
            render_score(env.get_score_breakdown()),
            render_step_detail_done(),
            gr.Button("Episode Complete — Reset to continue", variant="secondary", interactive=False),
        )

    try:
        # Build parameter dict from the dedicated input fields
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

        svc    = service_name if service_name and service_name.strip() else None
        action = IncidentResponseAction(
            action_type=ActionType(at),
            service_name=svc,
            parameters=params,
        )
        obs, reward, done, info = env.step(action)
        obs_dump = obs.model_dump()

        _action_history.append({
            "step":    obs.step_number,
            "action":  at,
            "service": svc,
            "reward":  reward,
        })

        # Add cumulative action history to info for step detail stats bar
        info["action_history"] = _action_history

        step_detail_html = render_step_detail(
            obs_dump, reward, done, info,
            obs.feedback, obs.action_result,
            at, svc, obs.step_number,
        )

        # If episode just ended, auto-show final score
        score_md = render_score(env.get_score_breakdown()) if done else "*Click Grade (6D) for score.*"

        # Disable execute button once episode ends
        step_btn_update = (
            gr.Button("Episode Complete — Reset to continue", variant="secondary", interactive=False)
            if done else
            gr.Button("Execute Action", variant="primary", interactive=True)
        )

        return (
            render_obs(obs_dump, obs.action_result),
            render_history(_action_history),
            render_state_panel(env),
            score_md,
            step_detail_html,
            step_btn_update,
        )

    except Exception as e:
        err = f"❌ **Error:** {e}"
        return (err, "", "", "", err,
                gr.Button("Execute Action", variant="primary", interactive=True))


# ── Grade & refresh ──────────────────────────────────────────────────────────

def gr_grade():
    """Compute and display the 6D score breakdown."""
    try:
        return render_score(env.get_score_breakdown())
    except Exception as e:
        return f"❌ {e}"


def gr_state():
    """Refresh the episode state panel."""
    return render_state_panel(env)
