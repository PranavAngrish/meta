"""
Gradio Blocks layout builder for the Incident Response sandbox UI.

Design goals
------------
- Two-column sandbox: narrow controls left, live output right
- Numbered steps guide the user through Setup → Act → Grade
- Episode state auto-updates on every action (no Refresh button)
- Grade button lives next to the score panel (right column) so it's always visible
- score_display is gr.HTML (not gr.Markdown) to avoid Gradio 6 re-render bugs

Call build_ui() to get the fully wired gr.Blocks instance.
"""
from __future__ import annotations

import sys
import os

_UI_DIR     = os.path.dirname(os.path.abspath(__file__))
_SERVER_DIR = os.path.dirname(_UI_DIR)
_PROJ_ROOT  = os.path.dirname(_SERVER_DIR)
for _p in (_SERVER_DIR, _PROJ_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import gradio as gr

from ui.styles import CUSTOM_CSS, HEADER_HTML
from ui.renderers import render_history, render_state_panel, render_score
from ui.callbacks import gr_reset, gr_step, gr_grade, gr_state

from state import env


# ── Small HTML helpers ────────────────────────────────────────────────────────

def _step_badge(n: str, label: str) -> str:
    return f"""
<div style="display:flex;align-items:center;gap:10px;padding:18px 0 10px;">
  <div style="width:22px;height:22px;border-radius:6px;
              background:var(--amber);display:flex;align-items:center;
              justify-content:center;font-family:var(--font-mono);
              font-size:11px;font-weight:700;color:#0a0700;flex-shrink:0;">{n}</div>
  <span style="font-family:var(--font-mono);font-size:10px;font-weight:400;
               letter-spacing:0.14em;text-transform:uppercase;
               color:var(--text2);">{label}</span>
  <div style="flex:1;height:1px;background:var(--line);"></div>
</div>"""


def _section_divider(label: str) -> str:
    return f"""
<div style="display:flex;align-items:center;gap:10px;padding:14px 0 8px;">
  <span style="font-family:var(--font-mono);font-size:9px;font-weight:400;
               letter-spacing:0.16em;text-transform:uppercase;
               color:var(--text3);">{label}</span>
  <div style="flex:1;height:1px;background:var(--line);"></div>
</div>"""


_EMPTY_OBS = """
<div class="empty-state">
  <div class="empty-icon" style="font-size:22px;">◌</div>
  <div class="empty-label">
    Choose a task and click <strong>New Episode</strong> to begin
  </div>
</div>"""

_EMPTY_STEP = """
<div class="empty-state">
  <div class="empty-icon" style="font-size:22px;">◈</div>
  <div class="empty-label">
    Step details appear here after each action
  </div>
</div>"""


# ── Builder ───────────────────────────────────────────────────────────────────

def build_ui() -> gr.Blocks:
    """Construct and return the fully wired Gradio Blocks application."""

    with gr.Blocks(title="Incident Response Environment v4.0") as web_ui:

        # Inject global CSS (Gradio 6 compatible — css param moved to launch())
        gr.HTML(f"<style>{CUSTOM_CSS}</style>")
        gr.HTML(HEADER_HTML)

        with gr.Tabs():

            # ── Overview ─────────────────────────────────────────────────────
            with gr.TabItem("Overview"):
                gr.Markdown("""
# The flight simulator for SREs

Train AI agents — and humans — to diagnose real production incidents with precision.

---

## How it works

| Phase | Actions | Goal |
|-------|---------|------|
| **Investigate** | check_service_health · investigate_logs · check_metrics · read_config · run_diagnostic | Gather evidence |
| **Diagnose** | declare_root_cause | Name the problem |
| **Remediate** | restart_service · update_config · rollback_deployment · scale_service | Apply the fix |

---

## 6-Dimension Scoring

| Dimension | Weight | Measures |
|-----------|--------|---------|
| Root Cause | **30%** | Correct failure mode identified? |
| Remediation | **25%** | Right fix on the right service? |
| Investigation | **15%** | Enough services checked? |
| Efficiency | **10%** | Steps used wisely? |
| Safety | **10%** | No collateral damage? |
| Sequence | **10%** | Diagnosed before fixing? |

> **Observation loop warning:** 3+ consecutive investigate actions without a fix → score capped at **0.45**

---

## Scenarios

| Task | Difficulty | Root causes | Steps |
|------|-----------|------------|-------|
| db_connection_failure | Easy | 1 | 20 |
| alert_triage | Easy (Triage) | P1–P4 classify | 3 |
| cascading_service_timeout | Medium | 2 | 25 |
| ssl_certificate_expiry | Medium | 1 | 25 |
| multi_factor_outage | Hard | 3 | 30 |
| database_deadlock | Hard | 1 | 30 |

→ Head to the **Sandbox** tab to start an episode.
                """)

            # ── Walkthrough ──────────────────────────────────────────────────
            with gr.TabItem("Walkthrough"):
                gr.Markdown("""
# Annotated solve: DB Connection Failure  ·  score 0.89

---

**On reset** — you see this alert:
```
● CRITICAL  user-api  "returning 503 errors"
✗ user-api   error_rate=85%   rt=450ms
✓ postgres-primary  healthy
✓ nginx-lb          healthy
```

---

**Step 1** — Action: `check_service_health` → `user-api`
```
Reward: +0.04   (first service investigated)
```

**Step 2** — Action: `investigate_logs` → `user-api` · keyword: `connection`
```
[08:15:12] ERROR: Connection refused for postgres-primary:5433
Port 5433 is suspicious — default is 5432.
Reward: +0.08
```

**Step 3** — Action: `read_config` → `user-api`
```json
{ "db_port": 5433 }   ← should be 5432
Reward: +0.06
```

**Step 4** — Action: `declare_root_cause`
> user-api db_port misconfigured as 5433 instead of 5432
```
✓ ROOT CAUSE MATCHED    Reward: +0.20
```

**Step 5** — Action: `update_config` → `user-api` · key: `db_port` · value: `5432`
```
✓ INCIDENT RESOLVED    Reward: +0.15
```

---

| Dimension | Score | Note |
|-----------|-------|------|
| Root Cause | 1.00 | Exact match |
| Remediation | 1.00 | Correct fix |
| Investigation | 0.67 | postgres-primary not checked |
| Efficiency | 1.00 | 5/20 steps used |
| Safety | 1.00 | No collateral damage |
| Sequence | 1.00 | Diagnosed before fixing |

**Final: 0.89 / 1.0 · Pattern: ◆ Efficient Reasoner**
                """)

            # ── FAQ ──────────────────────────────────────────────────────────
            with gr.TabItem("FAQ"):
                gr.Markdown("""
# Frequently Asked Questions

### What is an observation loop?
Three or more consecutive investigate actions with no fix in between.
Penalty: –0.08 on first occurrence, –0.03 rolling. Score hard-capped at **0.45**.

### What is collateral degradation?
Every 4 steps without resolution: +10% error rate and +40% latency cascade to dependent services. The stack degrades in real time — work quickly.

### What is a diagnosis gate?
Gated scenarios (`cascading_service_timeout`, `ssl_certificate_expiry`, `database_deadlock`) require investigating a specific service before a fix takes full effect. Blind fixes get **50% remediation credit only**.

### How does Alert Triage scoring work?
Classify the alert severity (P1/P2/P3/P4) in 3 steps.
Scoring: **1.0** exact · **0.5** adjacent · **0.25** two-off + investigation bonus.
A high error rate does not always mean P1 — check if a graceful fallback is active.

### What does a seed do?
Same seed → identical procedurally generated incident every time.
Leave blank for random variation. Use `42` for reproducible benchmarks.

### Optimal strategy
1. `check_service_health` on the unhealthy service first
2. `investigate_logs` with a relevant keyword
3. `declare_root_cause` before applying any fix
4. Apply the minimal correct remediation
5. Never chain 3+ investigate actions without a fix
                """)

            # ── Sandbox ──────────────────────────────────────────────────────
            with gr.TabItem("Sandbox"):
                with gr.Row(equal_height=False):

                    # ── Left: controls (narrower) ─────────────────────────
                    with gr.Column(scale=3, min_width=280):

                        # Step 1: Setup
                        gr.HTML(_step_badge("1", "New Episode"))
                        with gr.Group():
                            task_dd = gr.Dropdown(
                                choices=[
                                    ("Easy — DB Connection Failure",    "db_connection_failure"),
                                    ("Medium — Cascading Timeout",      "cascading_service_timeout"),
                                    ("Medium — SSL Expiry",             "ssl_certificate_expiry"),
                                    ("Hard — Multi-Factor Outage",      "multi_factor_outage"),
                                    ("Hard — Database Deadlock",        "database_deadlock"),
                                    ("Triage — Alert Severity P1–P4",   "alert_triage"),
                                ],
                                value="db_connection_failure",
                                label="Scenario",
                            )
                            seed_tb = gr.Textbox(
                                label="Seed  (leave blank for random)",
                                placeholder="e.g. 42",
                                value="",
                            )
                            reset_btn = gr.Button(
                                "🔄  New Episode",
                                variant="secondary",
                                size="lg",
                            )

                        # Step 2: Execute Action
                        gr.HTML(_step_badge("2", "Execute Action"))
                        with gr.Group():
                            action_dd = gr.Dropdown(
                                choices=[
                                    ("◈  check_service_health",  "check_service_health"),
                                    ("◈  investigate_logs",      "investigate_logs"),
                                    ("◈  check_metrics",         "check_metrics"),
                                    ("◈  read_config",           "read_config"),
                                    ("◈  run_diagnostic",        "run_diagnostic"),
                                    ("⬡  update_config",         "update_config"),
                                    ("⬡  restart_service",       "restart_service"),
                                    ("⬡  rollback_deployment",   "rollback_deployment"),
                                    ("⬡  scale_service",         "scale_service"),
                                    ("◆  declare_root_cause",    "declare_root_cause"),
                                    ("◆  submit_severity",       "submit_severity"),
                                ],
                                value="check_service_health",
                                label="Action",
                            )
                            service_dd = gr.Dropdown(
                                choices=[],
                                label="Target Service",
                                allow_custom_value=True,
                            )

                        with gr.Accordion("Parameters  (fill only what's needed)", open=False):
                            keyword_tb = gr.Textbox(
                                label="Keyword  —  investigate_logs",
                                placeholder="error, timeout, connection…",
                                value="",
                            )
                            cause_tb = gr.Textbox(
                                label="Root Cause  —  declare_root_cause",
                                placeholder="e.g. user-api db_port misconfigured as 5433 instead of 5432",
                                lines=3,
                                value="",
                            )
                            severity_dd = gr.Dropdown(
                                choices=[
                                    ("— not submitting",               ""),
                                    ("P1 — Critical: full outage",     "P1"),
                                    ("P2 — High: major degradation",   "P2"),
                                    ("P3 — Medium: partial, fallback", "P3"),
                                    ("P4 — Low: informational",        "P4"),
                                ],
                                value="",
                                label="Severity  —  submit_severity",
                            )
                            config_key_tb = gr.Textbox(
                                label="Config Key  —  update_config",
                                placeholder="e.g. db_port",
                                value="",
                            )
                            config_val_tb = gr.Textbox(
                                label="Config Value  —  update_config",
                                placeholder="e.g. 5432",
                                value="",
                            )
                            replicas_tb = gr.Textbox(
                                label="Replicas  —  scale_service",
                                placeholder="e.g. 3",
                                value="",
                            )

                        step_btn = gr.Button(
                            "▶  Execute Action",
                            variant="primary",
                            size="lg",
                        )

                        # Episode State (auto-refreshes on every action)
                        gr.HTML(_section_divider("Episode State"))
                        state_display = gr.HTML(render_state_panel(env))

                    # ── Right: live output (wider) ────────────────────────
                    with gr.Column(scale=5, min_width=400):

                        gr.HTML(_section_divider("Live Observation"))
                        obs_display = gr.HTML(_EMPTY_OBS)

                        gr.HTML(_section_divider("Last Action"))
                        step_detail_display = gr.HTML(_EMPTY_STEP)

                        with gr.Accordion("Action History", open=True):
                            history_display = gr.HTML(render_history([]))

                        # Score panel with Grade button inline
                        gr.HTML(f"""
<div style="display:flex;align-items:center;gap:10px;padding:14px 0 8px;">
  <span style="font-family:var(--font-mono);font-size:9px;font-weight:400;
               letter-spacing:0.16em;text-transform:uppercase;color:var(--text3);">6D Score</span>
  <div style="flex:1;height:1px;background:var(--line);"></div>
</div>""")
                        grade_btn = gr.Button("Grade Episode", variant="secondary", size="sm")
                        score_display = gr.HTML(render_score({}))

                # ── Event wiring ──────────────────────────────────────────
                _reset_outputs = [
                    obs_display,
                    history_display,
                    state_display,
                    score_display,
                    step_detail_display,
                    service_dd,
                    step_btn,
                ]
                _step_outputs = [
                    obs_display,
                    history_display,
                    state_display,
                    score_display,
                    step_detail_display,
                    step_btn,
                ]

                reset_btn.click(
                    fn=gr_reset,
                    inputs=[task_dd, seed_tb],
                    outputs=_reset_outputs,
                )
                step_btn.click(
                    fn=gr_step,
                    inputs=[
                        action_dd, service_dd,
                        keyword_tb, config_key_tb, config_val_tb,
                        replicas_tb, cause_tb, severity_dd,
                    ],
                    outputs=_step_outputs,
                )
                grade_btn.click(fn=gr_grade, outputs=[score_display])

    return web_ui
