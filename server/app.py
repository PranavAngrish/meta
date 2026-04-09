"""
Incident Response Environment — FastAPI + Gradio server entry point.

Architecture
------------
  server/
    app.py          ← this file: wires FastAPI + Gradio and starts Uvicorn
    state.py        ← shared IncidentResponseEnv singletons
    env.py          ← core RL environment logic
    api/
      routes.py     ← all HTTP endpoint handlers
      models.py     ← Pydantic request/response models
    ui/
      layout.py     ← Gradio Blocks definition
      callbacks.py  ← gr_reset / gr_step / gr_grade / gr_state
      renderers.py  ← pure HTML/Markdown rendering functions
      constants.py  ← shared display constants
      styles.py     ← CUSTOM_CSS + HEADER_HTML

Usage
-----
  python server/app.py              # development
  uvicorn server.app:app --reload   # hot-reload
  docker build -t ir-env . && docker run -p 7860:7860 ir-env
"""
from __future__ import annotations

import os
import sys

# Ensure both the project root and server/ directory are importable
_SERVER_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJ_ROOT  = os.path.dirname(_SERVER_DIR)
for _p in (_SERVER_DIR, _PROJ_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import gradio as gr
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from uvicorn.middleware.proxy_headers import ProxyHeadersMiddleware

from api.routes import router
from ui.layout import build_ui

# ── FastAPI application ───────────────────────────────────────────────────────
app = FastAPI(
    title="Incident Response Environment",
    version="4.0.0",
    description=(
        "OpenEnv-compatible RL training environment for SRE incident response. "
        "Supports 6 tasks with 6D scoring, procedural generation, and "
        "anti-reward-hacking mechanisms."
    ),
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(ProxyHeadersMiddleware, trusted_hosts="*")

# Root → redirect to the Gradio UI
@app.get("/")
def root():
    return RedirectResponse(url="/web")

# Mount all API routes
app.include_router(router)

# Mount Gradio UI at /web
web_ui = build_ui()
app    = gr.mount_gradio_app(app, web_ui, path="/web")


# ── Entry point ───────────────────────────────────────────────────────────────
def main() -> None:
    import uvicorn
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port,
        proxy_headers=True,
        forwarded_allow_ips="*",
    )


if __name__ == "__main__":
    main()
