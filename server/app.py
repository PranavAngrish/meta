"""
FastAPI server exposing the Incident Response environment via HTTP.
"""

from __future__ import annotations

import os
import sys
import json
import traceback
from typing import Any, Dict, Optional

import gradio as gr
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from env import IncidentResponseEnv
from models import ActionType, IncidentResponseAction
from scenarios.definitions import list_tasks

app = FastAPI(title="Incident Response Environment", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

env = IncidentResponseEnv()


class ResetRequest(BaseModel):
    task_name: str = "db_connection_failure"


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
    return {"status": "ok", "environment": "incident-response-env"}


@app.get("/tasks")
def tasks():
    return {"tasks": list_tasks()}


@app.post("/reset")
def reset(req: ResetRequest):
    try:
        obs = env.reset(task_name=req.task_name)
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
        return StepResponse(observation=obs.model_dump(), reward=round(reward, 4), done=done, info=info).model_dump()
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
    try:
        return {"score": env.get_score(), "task_name": env._task_name or "", "done": env._done}
    except Exception as e:
        return {"error": str(e)}


# ── Gradio Web UI ────────────────────────────────────────────────────────

def gr_reset(task_name):
    try:
        obs = env.reset(task_name=task_name)
        return obs.action_result, json.dumps([s.model_dump() for s in obs.service_statuses], indent=2)
    except Exception as e:
        return str(e), ""


def gr_step(action_type, service_name, params_json):
    try:
        params = json.loads(params_json) if params_json.strip() else {}
        action = IncidentResponseAction(
            action_type=ActionType(action_type),
            service_name=service_name if service_name.strip() else None,
            parameters=params,
        )
        obs, reward, done, info = env.step(action)
        result = f"Reward: {reward:.4f} | Done: {done} | Step: {obs.step_number}/{obs.max_steps}\n\n{obs.action_result}"
        status = json.dumps([s.model_dump() for s in obs.service_statuses], indent=2)
        return result, status
    except Exception as e:
        return f"Error: {e}", ""


def gr_score():
    return f"Score: {env.get_score():.4f}"


with gr.Blocks(title="Incident Response Env") as web_ui:
    gr.Markdown("# 🚨 Incident Response Environment")
    gr.Markdown("Production incident response & root cause analysis RL environment")
    with gr.Row():
        task_dd = gr.Dropdown(choices=["db_connection_failure", "cascading_service_timeout", "multi_factor_outage"],
                              value="db_connection_failure", label="Task")
        reset_btn = gr.Button("Reset", variant="primary")
    with gr.Row():
        with gr.Column(scale=2):
            action_dd = gr.Dropdown(choices=[at.value for at in ActionType], value="investigate_logs", label="Action")
            service_tb = gr.Textbox(label="Service Name", placeholder="e.g. user-api")
            params_tb = gr.Textbox(label="Parameters (JSON)", placeholder='{"keyword": "error"}', value="{}")
            step_btn = gr.Button("Step", variant="primary")
            score_btn = gr.Button("Get Score")
        with gr.Column(scale=3):
            result_tb = gr.Textbox(label="Result", lines=15, interactive=False)
            status_tb = gr.Textbox(label="Service Statuses", lines=10, interactive=False)
            score_tb = gr.Textbox(label="Score", interactive=False)
    reset_btn.click(gr_reset, inputs=[task_dd], outputs=[result_tb, status_tb])
    step_btn.click(gr_step, inputs=[action_dd, service_tb, params_tb], outputs=[result_tb, status_tb])
    score_btn.click(gr_score, outputs=[score_tb])

app = gr.mount_gradio_app(app, web_ui, path="/web")

def main():
    import uvicorn
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run(app, host="0.0.0.0", port=port)


if __name__ == "__main__":
    main()
