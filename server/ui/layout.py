"""
Gradio Blocks layout builder for the Incident Response sandbox UI.

Call build_ui() to get the configured gr.Blocks instance.
The layout is intentionally separate from callbacks and renderers so that
the visual structure can be changed without touching business logic.
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
from ui.renderers import render_history, render_state_panel
from ui.callbacks import gr_reset, gr_step, gr_grade, gr_state

# Lazily import env for the initial state panel render
from state import env


def build_ui() -> gr.Blocks:
    """Construct and return the fully wired Gradio Blocks application."""

    with gr.Blocks(title="Incident Response Environment v4.0") as web_ui:

        # Inject CSS via HTML component (Gradio 6 compatible — css/theme moved to launch())
        gr.HTML(f"<style>{CUSTOM_CSS}</style>")
        gr.HTML(HEADER_HTML)

        with gr.Tabs():

            # ── Overview tab ─────────────────────────────────────────────────
            with gr.TabItem("overview"):
                gr.Markdown("""
# The flight simulator for SREs

Train AI agents — and humans — to diagnose real production incidents with precision and speed.

---

## Workflow

| Phase | Actions | Purpose |
|-------|---------|---------|
| **Investigate** | check_service_health, investigate_logs, check_metrics, read_config, run_diagnostic | Gather evidence |
| **Diagnose** | declare_root_cause | Name the problem |
| **Remediate** | update_config, restart_service, rollback_deployment, scale_service | Apply the fix |

---

## 6-Dimension Scoring

| Dimension | Weight | What it measures |
|-----------|--------|-----------------|
| Root Cause | **30%** | Did you identify the correct failure mode? |
| Remediation | **25%** | Did you apply the right fix to the right service? |
| Investigation | **15%** | Did you check enough of the affected services? |
| Efficiency | **10%** | Did you use your step budget wisely? |
| Safety | **10%** | Did you avoid destructive actions on healthy services? |
| Sequence | **10%** | Did you diagnose before fixing? |

> ⚠ Observation loops (3+ diagnosis actions without a fix) hard-cap the score at **0.45**.

---

## Scenarios

| Scenario | Difficulty | Root causes | Max steps |
|----------|-----------|------------|-----------|
| db_connection_failure | Easy | 1 | 20 |
| alert_triage | Easy (Triage) | classify P1–P4 | 3 |
| cascading_service_timeout | Medium | 2 | 25 |
| ssl_certificate_expiry | Medium | 1 | 25 |
| multi_factor_outage | Hard | 3 | 30 |
| database_deadlock | Hard | 1 | 30 |

---

## Anti-reward-hacking Mechanisms

| Mechanism | Trigger | Effect |
|-----------|---------|--------|
| Observation Loop Detection | ≥3 consecutive diagnosis steps | –0.08 penalty + 0.45 hard cap |
| Diagnosis Gate | Blind fix before investigating gated services | 50% remediation credit |
| Safety Penalty | Destructive action on healthy service | –0.25 per action |
| Dynamic Degradation | Unresolved failure every 4 steps | +10% error rate, +40% latency cascade |
                """)

            # ── Walkthrough tab ──────────────────────────────────────────────
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

**Step 1** — `check_service_health` → `user-api`
```
Reward: +0.04   (investigation credit, 1st new service explored)
```

**Step 2** — `investigate_logs` → `user-api` · keyword: `connection`
```
[08:15:12] ERROR: Connection refused for postgres-primary:5433
Port 5433 is suspicious — default is 5432.

Reward: +0.08
```

**Step 3** — `read_config` → `user-api`
```json
{ "db_port": 5433 }   ← should be 5432

Reward: +0.06
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
| Efficiency | 1.00 | 5 steps used out of 20 |
| Safety | 1.00 | No collateral damage |
| Sequence | 1.00 | Diagnosed before fixing |

**Pattern:** ◆ Efficient Reasoner
                """)

            # ── FAQ tab ──────────────────────────────────────────────────────
            with gr.TabItem("faq"):
                gr.Markdown("""
# FAQ

### What is an observation loop?
Three or more consecutive diagnosis actions with no fix in between.
Score is hard-capped at **0.45** and you receive a –0.08 penalty on
the first occurrence, –0.03 rolling.

### What is collateral degradation?
Every 4 steps without resolution, all services upstream of the
failure receive +10% error rate and +40% latency. Eventually the
entire stack degrades, raising the urgency of timely remediation.

### What is a diagnosis gate?
Some scenarios (cascading_service_timeout, ssl_certificate_expiry,
database_deadlock) require investigating a specific service *before*
a fix takes full effect. Blind fixes receive only **50%** remediation
credit, to prevent reward-hacking via lucky guessing.

### How does Alert Triage work?
Classify the severity of an ongoing alert (P1/P2/P3/P4) using up to 3
investigation actions. Scoring: **1.0** exact · **0.5** adjacent ·
**0.25** two-off, plus an investigation bonus for thorough triage.

### What does a seed do?
Same seed → identical procedurally generated incident. Leave blank for
random variation. Use `seed=42` for reproducible benchmarking runs.

### How do I maximise my score?
1. Investigate first — always check the unhealthy service before touching healthy ones.
2. Declare root cause *before* applying any remediation.
3. Never chain 3+ diagnosis actions without a fix.
4. Pick the minimal correct fix — avoid destructive actions on healthy services.
5. Resolve the incident in as few steps as possible for the efficiency bonus.
                """)

            # ── Sandbox tab ──────────────────────────────────────────────────
            with gr.TabItem("sandbox"):
                with gr.Row(equal_height=False):

                    # ── Left column: controls ─────────────────────────────
                    with gr.Column(scale=2, min_width=340):

                        gr.HTML('<div class="ir-section">Episode Setup</div>')
                        with gr.Group():
                            with gr.Column():
                                task_dd = gr.Dropdown(
                                    choices=[
                                        ("Easy — DB Connection Failure",    "db_connection_failure"),
                                        ("Medium — Cascading Timeout",      "cascading_service_timeout"),
                                        ("Medium — SSL Certificate Expiry", "ssl_certificate_expiry"),
                                        ("Hard — Multi-Factor Outage",      "multi_factor_outage"),
                                        ("Hard — Database Deadlock",        "database_deadlock"),
                                        ("Triage — Alert Severity P1–P4",   "alert_triage"),
                                    ],
                                    value="db_connection_failure",
                                    label="Task",
                                )
                                seed_tb   = gr.Textbox(
                                    label="Seed (optional)",
                                    placeholder="e.g. 42 — leave blank for random",
                                    value="",
                                )
                                reset_btn = gr.Button(
                                    "Reset Environment",
                                    variant="secondary",
                                    size="lg",
                                )

                        gr.HTML('<div class="ir-section">Episode State</div>')
                        state_display = gr.HTML(render_state_panel(env))

                        gr.HTML('<div class="ir-section">Action Controls</div>')
                        with gr.Group():
                            with gr.Column():
                                action_dd = gr.Dropdown(
                                    choices=[
                                        ("◈  investigate_logs",      "investigate_logs"),
                                        ("◈  check_metrics",         "check_metrics"),
                                        ("◈  read_config",           "read_config"),
                                        ("◈  check_service_health",  "check_service_health"),
                                        ("◈  run_diagnostic",        "run_diagnostic"),
                                        ("⬡  restart_service",       "restart_service"),
                                        ("⬡  update_config",         "update_config"),
                                        ("⬡  rollback_deployment",   "rollback_deployment"),
                                        ("⬡  scale_service",         "scale_service"),
                                        ("◆  declare_root_cause",    "declare_root_cause"),
                                        ("◆  submit_severity",       "submit_severity"),
                                    ],
                                    value="investigate_logs",
                                    label="Action Type",
                                )
                                service_dd = gr.Dropdown(
                                    choices=[],
                                    label="Target Service",
                                    allow_custom_value=True,
                                )

                        with gr.Accordion("Action Parameters", open=True):
                            keyword_tb    = gr.Textbox(
                                label="Keyword  ·  investigate_logs",
                                placeholder="error, timeout, connection, deadlock…",
                                value="",
                            )
                            config_key_tb = gr.Textbox(
                                label="Config Key  ·  update_config",
                                placeholder="e.g. db_port",
                                value="",
                            )
                            config_val_tb = gr.Textbox(
                                label="Config Value  ·  update_config",
                                placeholder="e.g. 5432",
                                value="",
                            )
                            replicas_tb   = gr.Textbox(
                                label="Replicas  ·  scale_service",
                                placeholder="e.g. 3",
                                value="",
                            )
                            cause_tb      = gr.Textbox(
                                label="Root Cause  ·  declare_root_cause",
                                placeholder="e.g. user-api db_port misconfigured as 5433 instead of 5432",
                                lines=3,
                                value="",
                            )
                            severity_dd   = gr.Dropdown(
                                choices=[
                                    ("— not submitting",               ""),
                                    ("P1 — Critical: full outage",     "P1"),
                                    ("P2 — High: major degradation",   "P2"),
                                    ("P3 — Medium: partial, fallback", "P3"),
                                    ("P4 — Low: informational",        "P4"),
                                ],
                                value="",
                                label="Severity  ·  submit_severity",
                            )

                        step_btn = gr.Button("Execute Action", variant="primary", size="lg")

                        gr.HTML('<div class="ir-section">Scoring</div>')
                        with gr.Row():
                            grade_btn = gr.Button("Grade (6D)", variant="secondary", size="sm")
                            state_btn = gr.Button("Refresh State", variant="secondary", size="sm")

                    # ── Right column: displays ────────────────────────────
                    with gr.Column(scale=3, min_width=460):

                        gr.HTML('<div class="ir-section">Environment Observation</div>')
                        obs_display = gr.HTML("""
<div class="empty-state">
  <div class="empty-icon">◌</div>
  <div class="empty-label">
    Select a task and click <strong>Reset Environment</strong> to begin
  </div>
</div>""")

                        gr.HTML('<div class="ir-section">Step Detail</div>')
                        step_detail_display = gr.HTML("""
<div class="empty-state">
  <div class="empty-icon">◈</div>
  <div class="empty-label">
    Step details appear here after each action<br>
    <strong>Reset an episode first</strong>
  </div>
</div>""")

                        gr.HTML('<div class="ir-section">Action History</div>')
                        history_display = gr.HTML(render_history([]))

                        gr.HTML('<div class="ir-section">6D Score</div>')
                        score_display = gr.Markdown(
                            "*Execute actions then click **Grade (6D)**.*"
                        )

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
                state_btn.click(fn=gr_state, outputs=[state_display])

    return web_ui
