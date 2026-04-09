"""
Pure HTML/Markdown rendering functions for the Incident Response Gradio UI.

Every function here is side-effect-free: it takes data, returns a string.
No imports from state.py — all environment data is passed in as arguments.
"""
from __future__ import annotations

import html as _html
import re
from typing import Any, Dict, List, Optional

from ui.constants import (
    DIFF_MAP, ACTION_META, DIAG_ACTIONS, FIX_ACTIONS, FAILURE_TYPE_ICON
)

try:
    from graders import _OBSERVATION_LOOP_CAP
except ImportError:
    _OBSERVATION_LOOP_CAP = 0.45


# ── Alerts ────────────────────────────────────────────────────────────────────

def render_alerts(alerts: List[Dict]) -> str:
    if not alerts:
        return """
<div class="empty-state">
  <div class="empty-icon">🔕</div>
  <div class="empty-label">No active alerts<br>
    <strong>Reset an episode to populate</strong>
  </div>
</div>"""
    out = ""
    for a in alerts:
        sev = (a.get("severity", "medium")).upper()
        if sev in ("CRITICAL", "HIGH"):
            bc, bb, bl = "var(--red)",   "var(--red-bg)",   "rgba(248,113,113,0.2)"
        elif sev == "MEDIUM":
            bc, bb, bl = "var(--amber)", "var(--amber-bg)", "rgba(245,166,35,0.2)"
        else:
            bc, bb, bl = "var(--green)", "var(--green-bg)", "rgba(52,211,153,0.2)"
        svc_name = _html.escape(str(a.get("service", "")))
        alert_id = _html.escape(str(a.get("alert_id", "")))
        msg      = _html.escape(str(a.get("message", "")))
        ts       = _html.escape(str(a.get("timestamp", "")))
        out += f"""
<div class="alert-card" style="background:{bb};border-color:{bl};">
  <span class="alert-badge" style="color:{bc};background:{bb};border:1px solid {bl};">{sev}</span>
  <div style="min-width:0;flex:1;">
    <div class="alert-name">{svc_name}<span class="alert-id">{alert_id}</span></div>
    <div class="alert-msg">{msg}</div>
    <div style="font-family:var(--font-mono);font-size:9px;color:var(--text3);margin-top:4px;">T={ts}</div>
  </div>
</div>"""
    return out


# ── Service health ────────────────────────────────────────────────────────────

def render_services(statuses: List[Dict]) -> str:
    if not statuses:
        return ""
    rows = ""
    for s in statuses:
        ok  = s.get("healthy", True)
        pc  = "var(--green)" if ok else "var(--red)"
        pb  = "var(--green-bg)" if ok else "var(--red-bg)"
        pt  = "● healthy" if ok else "● error"
        rt  = f"{s['response_time_ms']:.0f} ms" if s.get("response_time_ms") is not None else "—"
        er_r = s.get("error_rate")
        er  = f"{er_r * 100:.1f}%" if er_r is not None else "—"
        ec  = "var(--red)" if not ok else "var(--text2)"
        name = _html.escape(str(s.get("name", "")))
        rows += f"""
<div class="svc-row">
  <span class="svc-name">{name}</span>
  <span class="svc-pill" style="color:{pc};background:{pb};border:1px solid {pc}25;">{pt}</span>
  <span class="svc-stat">{rt}</span>
  <span class="svc-stat" style="color:{ec};">{er}</span>
</div>"""
    return f"""
<div style="margin-top:18px;">
  <div class="panel-label" style="margin-bottom:0;">Service health</div>
  <div style="background:var(--bg2);border:1px solid var(--line);
              border-radius:var(--r);margin-top:8px;overflow:hidden;">
    <div style="display:grid;grid-template-columns:1fr auto auto auto;
                padding:7px 14px;border-bottom:1px solid var(--line);">
      <span style="font-family:var(--font-mono);font-size:9px;letter-spacing:0.14em;
                   text-transform:uppercase;color:var(--text3);">Service</span>
      <span style="font-family:var(--font-mono);font-size:9px;letter-spacing:0.14em;
                   text-transform:uppercase;color:var(--text3);min-width:80px;
                   text-align:right;padding-right:10px;">Status</span>
      <span style="font-family:var(--font-mono);font-size:9px;letter-spacing:0.14em;
                   text-transform:uppercase;color:var(--text3);min-width:70px;
                   text-align:right;padding:0 10px;">Latency</span>
      <span style="font-family:var(--font-mono);font-size:9px;letter-spacing:0.14em;
                   text-transform:uppercase;color:var(--text3);min-width:70px;
                   text-align:right;">Err%</span>
    </div>
    {rows}
  </div>
</div>"""


# ── Action output log ─────────────────────────────────────────────────────────

def render_log(text: str, accent: str = "var(--cyan)") -> str:
    if not text:
        return ""
    esc = _html.escape(str(text))
    # Colour-code keywords
    esc = re.sub(
        r"(ERROR|CRITICAL|FAILED|refused|Connection refused|timeout|deadlock)",
        r'<span style="color:var(--red);">\1</span>', esc, flags=re.IGNORECASE
    )
    esc = re.sub(
        r"(SUCCESS|healthy|resolved|updated successfully|MATCHED|correct)",
        r'<span style="color:var(--green);">\1</span>', esc, flags=re.IGNORECASE
    )
    esc = re.sub(
        r"(WARNING|WARN|degraded|slow|high)",
        r'<span style="color:var(--amber);">\1</span>', esc, flags=re.IGNORECASE
    )
    esc = re.sub(
        r"(\[\d{2}:\d{2}:\d{2}\]|\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2})",
        r'<span style="color:var(--text3);">\1</span>', esc
    )
    return f"""
<div style="margin-top:12px;">
  <div class="panel-label" style="border-left-color:{accent};margin-bottom:8px;">Action output</div>
  <div class="log-block" style="border-left-color:{accent};">{esc}</div>
</div>"""


# ── Main observation panel ────────────────────────────────────────────────────

def render_obs(obs_dump: Dict, action_result: str) -> str:
    task  = obs_dump.get("task_name", "—")
    step  = obs_dump.get("step_number", 0)
    max_s = obs_dump.get("max_steps", 30)
    pct   = int(step / max(max_s, 1) * 100)
    dlabel, dcolor, dbg = DIFF_MAP.get(task, ("?", "var(--text1)", "var(--bg3)"))

    incident_summary = _html.escape(str(obs_dump.get("incident_summary", "")))

    alerts_html   = render_alerts(obs_dump.get("active_alerts", []))
    services_html = render_services(obs_dump.get("service_statuses", []))

    # Degradation warnings
    warn = obs_dump.get("degradation_warnings", [])
    degrade_html = ""
    if warn:
        items = "".join(
            f'<div style="margin-bottom:4px;color:var(--violet);">⚡ {_html.escape(w)}</div>'
            for w in warn[-3:]
        )
        degrade_html = f"""
<div style="margin-top:14px;">
  <div class="panel-label" style="border-left-color:var(--violet);margin-bottom:8px;">Degradation Cascade</div>
  <div style="background:var(--violet-bg);border:1px solid rgba(167,139,250,0.15);
              border-radius:var(--r);padding:10px 14px;
              font-family:var(--font-mono);font-size:11px;line-height:1.8;">
    {items}
  </div>
</div>"""

    # Incident context summary
    summary_html = ""
    if incident_summary:
        summary_html = f"""
<div style="margin-bottom:14px;background:var(--amber-bg);border:1px solid rgba(245,166,35,0.12);
            border-radius:var(--r);padding:10px 14px;">
  <div style="font-family:var(--font-mono);font-size:9px;text-transform:uppercase;
              letter-spacing:0.12em;color:var(--amber);margin-bottom:4px;">Incident</div>
  <div style="font-family:var(--font-head);font-size:12px;color:var(--text0);
              line-height:1.5;">{incident_summary}</div>
</div>"""

    return f"""
<div style="font-family:var(--font-head);">
  <div style="display:flex;align-items:center;gap:10px;padding:12px 0 14px;
              border-bottom:1px solid var(--line);margin-bottom:16px;">
    <span class="live-dot" style="background:var(--green);box-shadow:0 0 0 3px var(--green-bg);"></span>
    <span style="font-family:var(--font-mono);font-size:11px;color:var(--text1);letter-spacing:0.04em;">{task}</span>
    <span style="font-family:var(--font-mono);font-size:9px;padding:2px 9px;border-radius:4px;
                 color:{dcolor};background:{dbg};border:1px solid {dcolor}25;">{dlabel}</span>
    <div style="margin-left:auto;display:flex;align-items:baseline;gap:6px;">
      <span style="font-family:var(--font-head);font-size:22px;font-weight:800;
                   color:var(--text0);letter-spacing:-0.5px;">{step}</span>
      <span style="font-family:var(--font-mono);font-size:10px;color:var(--text2);">/ {max_s} steps</span>
    </div>
  </div>

  <div class="prog-wrap"><div class="prog-fill" style="width:{pct}%;"></div></div>
  <div style="display:flex;justify-content:space-between;font-family:var(--font-mono);
              font-size:9px;color:var(--text3);margin-bottom:18px;">
    <span>Progress</span><span>{pct}%</span>
  </div>

  {summary_html}

  <div class="panel-label" style="margin-bottom:10px;">Active alerts</div>
  {alerts_html}
  {services_html}
  {degrade_html}
</div>"""


# ── Episode done state (shown in obs panel when clicking after completion) ────

def render_obs_done() -> str:
    return """
<div style="background:var(--bg2);border:1px solid var(--line);border-radius:var(--r2);
            padding:36px;text-align:center;">
  <div style="font-size:32px;margin-bottom:14px;">🏁</div>
  <div style="font-family:var(--font-head);font-size:17px;font-weight:800;
              color:var(--text0);margin-bottom:8px;">Episode Already Complete</div>
  <div style="font-family:var(--font-mono);font-size:11px;color:var(--text2);line-height:1.9;">
    Actions are no longer accepted for this episode.<br>
    Click <strong style="color:var(--amber);">Grade (6D)</strong> to view your final score<br>
    or <strong style="color:var(--amber);">Reset Environment</strong> to start a new episode.
  </div>
</div>"""


# ── Step detail card ──────────────────────────────────────────────────────────

def render_step_detail(
    obs_dump:     Dict,
    reward:       float,
    done:         bool,
    info:         Dict,
    feedback:     str,
    action_result: str,
    action_type:  str,
    service_name: Optional[str],
    step_number:  int,
) -> str:
    """Rich per-step card shown after every action execution."""

    # Action metadata
    meta = ACTION_META.get(action_type, ("·", action_type, "var(--text2)", "var(--bg3)"))
    a_icon, a_label, a_color, a_bg = meta

    # Reward styling
    r_sign = "+" if reward >= 0 else ""
    r_col  = "var(--green)" if reward >= 0 else "var(--red)"
    svc    = _html.escape(str(service_name or "—"))
    at_esc = _html.escape(action_type)

    # Feedback callout
    feedback_html = ""
    if feedback:
        fb_esc = _html.escape(str(feedback))
        feedback_html = f"""
<div style="background:var(--cyan-bg);border-left:2px solid var(--cyan);
            padding:10px 14px;margin:12px 14px 0;
            border-radius:0 var(--r) var(--r) 0;">
  <div style="font-family:var(--font-mono);font-size:9px;text-transform:uppercase;
              letter-spacing:0.12em;color:var(--cyan);margin-bottom:5px;">Per-step feedback</div>
  <div style="font-family:var(--font-head);font-size:13px;color:var(--text0);line-height:1.5;">
    {fb_esc}
  </div>
</div>"""

    # Warnings
    streak   = info.get("consecutive_diagnosis_count", 0)
    obs_loop = info.get("observation_loop", False)
    warn_html = ""
    if obs_loop:
        warn_html = f"""
<div class="warn-loop" style="margin:10px 14px 0;">
  ⊗ Observation loop detected — score hard-capped at {_OBSERVATION_LOOP_CAP}.<br>
  Apply a remediation action to break the loop.
</div>"""
    elif streak >= 2:
        warn_html = f"""
<div class="warn-streak" style="margin:10px 14px 0;">
  ⚠ Diagnosis streak: {streak} consecutive investigate actions.<br>
  Apply a fix soon — at 3+ you will hit the observation loop cap.
</div>"""

    # Degradation warnings from obs
    degrade_warns = obs_dump.get("degradation_warnings", [])
    if degrade_warns:
        dw_items = " | ".join(_html.escape(w) for w in degrade_warns[-2:])
        warn_html += f"""
<div class="warn-degrade" style="margin:10px 14px 0;">
  ⚡ Cascade: {dw_items}
</div>"""

    # Service snapshot (mini table — 4 cols)
    statuses = obs_dump.get("service_statuses", [])
    svc_rows = ""
    for s in statuses:
        ok   = s.get("healthy", True)
        pc   = "var(--green)" if ok else "var(--red)"
        icon = "✓" if ok else "✗"
        rt   = f"{s['response_time_ms']:.0f}" if s.get("response_time_ms") is not None else "—"
        er_r = s.get("error_rate")
        er   = f"{er_r * 100:.0f}%" if er_r is not None else "—"
        ec   = "var(--red)" if not ok else "var(--text3)"
        nm   = _html.escape(str(s.get("name", "")))
        svc_rows += f"""
<tr>
  <td style="font-family:var(--font-mono);font-size:11px;color:var(--text1);
             padding:6px 10px;border-bottom:1px solid rgba(255,255,255,0.03);">{nm}</td>
  <td style="font-family:var(--font-mono);font-size:11px;color:{pc};
             padding:6px 10px;border-bottom:1px solid rgba(255,255,255,0.03);text-align:center;">{icon}</td>
  <td style="font-family:var(--font-mono);font-size:11px;color:var(--text2);
             padding:6px 10px;border-bottom:1px solid rgba(255,255,255,0.03);text-align:right;">{rt}ms</td>
  <td style="font-family:var(--font-mono);font-size:11px;color:{ec};
             padding:6px 10px;border-bottom:1px solid rgba(255,255,255,0.03);text-align:right;">{er}</td>
</tr>"""
    svc_snapshot_html = ""
    if svc_rows:
        svc_snapshot_html = f"""
<div style="padding:12px 14px 0;">
  <div class="panel-label" style="margin-bottom:8px;">Service snapshot</div>
  <div style="background:var(--bg1);border:1px solid var(--line);border-radius:var(--r);overflow:hidden;">
    <table style="width:100%;border-collapse:collapse;">
      <thead>
        <tr>
          <th style="font-family:var(--font-mono);font-size:9px;text-transform:uppercase;
                     letter-spacing:0.12em;color:var(--text3);padding:5px 10px;
                     border-bottom:1px solid var(--line);text-align:left;">Service</th>
          <th style="font-family:var(--font-mono);font-size:9px;text-transform:uppercase;
                     letter-spacing:0.12em;color:var(--text3);padding:5px 10px;
                     border-bottom:1px solid var(--line);text-align:center;">OK</th>
          <th style="font-family:var(--font-mono);font-size:9px;text-transform:uppercase;
                     letter-spacing:0.12em;color:var(--text3);padding:5px 10px;
                     border-bottom:1px solid var(--line);text-align:right;">RT</th>
          <th style="font-family:var(--font-mono);font-size:9px;text-transform:uppercase;
                     letter-spacing:0.12em;color:var(--text3);padding:5px 10px;
                     border-bottom:1px solid var(--line);text-align:right;">Err%</th>
        </tr>
      </thead>
      <tbody>{svc_rows}</tbody>
    </table>
  </div>
</div>"""

    # Log output
    log_html = render_log(action_result) if action_result else ""
    if log_html:
        log_html = f'<div style="padding:0 14px;">{log_html}</div>'

    # Episode done banner
    done_html = ""
    if done:
        resolved_msg = "Incident resolved! " if info.get("incident_resolved") else ""
        done_html = f"""
<div class="episode-done-banner">
  <div style="font-family:var(--font-head);font-size:15px;font-weight:800;
              color:var(--green);margin-bottom:6px;">🏁 Episode Complete</div>
  <div style="font-family:var(--font-mono);font-size:11px;color:var(--text2);line-height:1.7;">
    {resolved_msg}Click <strong style="color:var(--amber);">Grade (6D)</strong> to see your final score breakdown.
  </div>
</div>"""

    # Stats bar (right of header): investigation count, actions breakdown
    diag_count = sum(1 for h in info.get("action_history", []) if h.get("action") in DIAG_ACTIONS)
    fix_count  = sum(1 for h in info.get("action_history", []) if h.get("action") in FIX_ACTIONS)

    failure_type = info.get("failure_type", "N/A")
    ft_icon = FAILURE_TYPE_ICON.get(failure_type, "·")

    return f"""
<div class="step-card">
  <div class="step-card-header">
    <span style="font-family:var(--font-mono);font-size:10px;color:var(--text3);min-width:28px;">#{step_number}</span>
    <span class="action-chip" style="color:{a_color};background:{a_bg};border:1px solid {a_color}20;">
      {a_icon} {a_label}
    </span>
    <span style="font-family:var(--font-mono);font-size:11px;color:var(--text1);">{at_esc}</span>
    <span style="font-family:var(--font-head);font-size:12px;color:var(--text3);">→ {svc}</span>
    <div style="margin-left:auto;display:flex;flex-direction:column;align-items:flex-end;gap:1px;">
      <span style="font-family:var(--font-head);font-size:22px;font-weight:800;
                   color:{r_col};line-height:1;">{r_sign}{reward:.4f}</span>
      <span style="font-family:var(--font-mono);font-size:9px;color:var(--text3);">step reward</span>
    </div>
  </div>

  <div style="display:flex;gap:0;border-bottom:1px solid var(--line);
              padding:10px 14px;background:var(--bg3);">
    <div style="flex:1;text-align:center;">
      <div style="font-family:var(--font-mono);font-size:9px;color:var(--text3);
                  text-transform:uppercase;letter-spacing:0.1em;margin-bottom:3px;">Pattern</div>
      <div style="font-family:var(--font-mono);font-size:11px;color:var(--text1);">
        {ft_icon} {_html.escape(failure_type)}
      </div>
    </div>
    <div style="width:1px;background:var(--line);margin:0 10px;"></div>
    <div style="flex:1;text-align:center;">
      <div style="font-family:var(--font-mono);font-size:9px;color:var(--text3);
                  text-transform:uppercase;letter-spacing:0.1em;margin-bottom:3px;">Diag / Fix</div>
      <div style="font-family:var(--font-mono);font-size:11px;">
        <span style="color:var(--cyan);">{diag_count}</span>
        <span style="color:var(--text3);"> / </span>
        <span style="color:var(--green);">{fix_count}</span>
      </div>
    </div>
    <div style="width:1px;background:var(--line);margin:0 10px;"></div>
    <div style="flex:1;text-align:center;">
      <div style="font-family:var(--font-mono);font-size:9px;color:var(--text3);
                  text-transform:uppercase;letter-spacing:0.1em;margin-bottom:3px;">Obs Loop</div>
      <div style="font-family:var(--font-mono);font-size:11px;
                  color:{'var(--red)' if obs_loop else 'var(--green)'};">
        {'⊗ yes' if obs_loop else '✓ no'}
      </div>
    </div>
    <div style="width:1px;background:var(--line);margin:0 10px;"></div>
    <div style="flex:1;text-align:center;">
      <div style="font-family:var(--font-mono);font-size:9px;color:var(--text3);
                  text-transform:uppercase;letter-spacing:0.1em;margin-bottom:3px;">Streak</div>
      <div style="font-family:var(--font-mono);font-size:11px;
                  color:{'var(--red)' if streak >= 3 else 'var(--amber)' if streak >= 2 else 'var(--text2)'};">
        {streak}{'⚠' if streak >= 2 else ''}
      </div>
    </div>
  </div>

  {done_html}
  {warn_html}
  {feedback_html}
  {svc_snapshot_html}
  {log_html}
  <div style="height:14px;"></div>
</div>"""


# ── Step detail — reset welcome card ─────────────────────────────────────────

def render_step_detail_reset(obs_dump: Dict) -> str:
    """Welcome card shown after resetting the environment."""
    task     = obs_dump.get("task_name", "—")
    dlabel, dcolor, dbg = DIFF_MAP.get(task, ("?", "var(--text1)", "var(--bg3)"))
    summary  = _html.escape(str(obs_dump.get("incident_summary", "")))
    alerts   = obs_dump.get("active_alerts", [])
    statuses = obs_dump.get("service_statuses", [])
    unhealthy = [s for s in statuses if not s.get("healthy", True)]
    services = obs_dump.get("available_services", [])
    max_s    = obs_dump.get("max_steps", 30)

    alert_items = ""
    for a in alerts[:3]:
        sev  = a.get("severity", "medium").upper()
        sc   = "var(--red)" if sev in ("CRITICAL", "HIGH") else "var(--amber)"
        svcn = _html.escape(str(a.get("service", "")))
        msg  = _html.escape(str(a.get("message", "")))
        alert_items += f"""
<div style="display:flex;gap:10px;align-items:flex-start;margin-bottom:8px;">
  <span style="font-family:var(--font-mono);font-size:9px;color:{sc};
               background:{sc}15;border:1px solid {sc}30;padding:2px 7px;border-radius:4px;
               flex-shrink:0;margin-top:1px;">{sev}</span>
  <div>
    <div style="font-family:var(--font-head);font-size:12px;font-weight:700;
                color:var(--text0);">{svcn}</div>
    <div style="font-family:var(--font-head);font-size:11px;color:var(--text1);">{msg}</div>
  </div>
</div>"""

    unhealthy_html = ""
    if unhealthy:
        names = ", ".join(_html.escape(s.get("name", "")) for s in unhealthy)
        unhealthy_html = f"""
<div style="margin-top:10px;background:var(--red-bg);border:1px solid rgba(248,113,113,0.15);
            border-radius:var(--r);padding:9px 14px;font-family:var(--font-mono);
            font-size:11px;color:var(--red);">
  ✗ Degraded services: {names}
</div>"""

    return f"""
<div class="step-card">
  <div style="background:var(--amber-bg);border-bottom:1px solid rgba(245,166,35,0.15);
              padding:14px 16px;display:flex;align-items:center;gap:12px;">
    <span style="font-family:var(--font-mono);font-size:10px;color:var(--amber);
                 letter-spacing:0.12em;text-transform:uppercase;">Episode Reset</span>
    <span style="font-family:var(--font-mono);font-size:9px;padding:2px 9px;border-radius:4px;
                 color:{dcolor};background:{dbg};border:1px solid {dcolor}25;">{dlabel}</span>
    <span style="margin-left:auto;font-family:var(--font-mono);font-size:10px;color:var(--text2);">
      Budget: {max_s} steps
    </span>
  </div>

  <div style="padding:14px 16px;">
    <div class="panel-label" style="margin-bottom:8px;">Incident summary</div>
    <div style="font-family:var(--font-head);font-size:13px;color:var(--text0);
                line-height:1.6;margin-bottom:14px;">{summary}</div>
    <div class="panel-label" style="margin-bottom:8px;">Active alerts ({len(alerts)})</div>
    {alert_items or '<div style="color:var(--text3);font-family:var(--font-mono);font-size:11px;">No alerts</div>'}
    {unhealthy_html}
  </div>

  <div style="padding:0 16px 16px;">
    <div class="panel-label" style="margin-bottom:8px;">Suggested first action</div>
    <div style="background:var(--bg3);border:1px solid var(--line);border-radius:var(--r);
                padding:10px 14px;font-family:var(--font-mono);font-size:12px;color:var(--text1);">
      <span style="color:var(--cyan);">check_service_health</span> →
      <span style="color:var(--amber);">{_html.escape(unhealthy[0].get("name","") if unhealthy else (services[0] if services else ""))}</span>
      <div style="color:var(--text3);font-size:10px;margin-top:4px;">
        Start by triaging the unhealthy service to understand the blast radius.
      </div>
    </div>
  </div>
</div>"""


# ── Step detail — episode already done ───────────────────────────────────────

def render_step_detail_done() -> str:
    return """
<div class="step-card">
  <div style="padding:32px;text-align:center;">
    <div style="font-size:28px;margin-bottom:12px;">🏁</div>
    <div style="font-family:var(--font-head);font-size:16px;font-weight:800;
                color:var(--text0);margin-bottom:8px;">Episode Already Complete</div>
    <div style="font-family:var(--font-mono);font-size:11px;color:var(--text2);line-height:1.9;">
      No further actions are accepted for this episode.<br>
      Click <strong style="color:var(--amber);">Grade (6D)</strong> to see your score breakdown<br>
      or <strong style="color:var(--amber);">Reset Environment</strong> to start fresh.
    </div>
  </div>
</div>"""


# ── Action history table ──────────────────────────────────────────────────────

def render_history(history: List[Dict]) -> str:
    if not history:
        return """
<div class="empty-state">
  <div class="empty-icon">◻</div>
  <div class="empty-label">No actions yet<br>
    <strong>Execute an action to begin</strong>
  </div>
</div>"""

    cumulative = 0.0
    rows = ""
    for h in history:
        at      = h.get("action", "")
        meta    = ACTION_META.get(at, ("·", at, "var(--text2)", "var(--bg3)"))
        ic, lb, cc, cb = meta
        r       = h.get("reward", 0.0)
        cumulative += r
        r_str   = f"+{r:.3f}" if r >= 0 else f"{r:.3f}"
        r_col   = "var(--green)" if r >= 0 else "var(--red)"
        cum_col = "var(--green)" if cumulative >= 0 else "var(--red)"
        cum_str = f"+{cumulative:.3f}" if cumulative >= 0 else f"{cumulative:.3f}"
        svc     = _html.escape(str(h.get("service") or "—"))
        rows += f"""
<tr class="hist-row">
  <td class="hist-td" style="color:var(--text3);width:36px;">{h['step']}</td>
  <td class="hist-td">
    <span class="action-chip" style="color:{cc};background:{cb};border:1px solid {cc}20;">
      {ic} {lb}
    </span>
  </td>
  <td class="hist-td" style="color:var(--text0);font-weight:500;">{_html.escape(at)}</td>
  <td class="hist-td" style="color:var(--text2);">{svc}</td>
  <td class="hist-td" style="color:{r_col};font-weight:500;text-align:right;">{r_str}</td>
  <td class="hist-td" style="color:{cum_col};text-align:right;opacity:0.7;">{cum_str}</td>
</tr>"""

    return f"""
<div style="overflow-x:auto;">
<table class="hist-table">
  <thead>
    <tr>
      {''.join(f'<th class="hist-th">{x}</th>' for x in ['#','type','action','service','reward','cumulative'])}
    </tr>
  </thead>
  <tbody>{rows}</tbody>
</table>
</div>"""


# ── Episode state panel ───────────────────────────────────────────────────────

def render_state_panel(env_obj: Any) -> str:
    """Render the live episode state using env private attributes."""

    # Alert triage special path
    if env_obj._task_name == "alert_triage":
        step = env_obj._step_number
        cum  = round(env_obj._cumulative_reward, 4)
        sub  = getattr(env_obj, "_at_submitted_severity", None) or "—"
        pct  = int(step / 3 * 100)
        cc   = "var(--green)" if cum >= 0 else "var(--red)"
        return f"""
<div>
  <div style="display:grid;grid-template-columns:1fr 1fr;gap:8px;margin-bottom:8px;">
    <div class="mc"><div class="mc-accent" style="background:var(--cyan);"></div>
      <div class="mc-label">Step</div>
      <div class="mc-value" style="color:var(--cyan);">{step}
        <span style="font-size:14px;color:var(--text2);font-weight:400;">/3</span>
      </div>
    </div>
    <div class="mc"><div class="mc-accent" style="background:{cc};"></div>
      <div class="mc-label">Cum. Reward</div>
      <div class="mc-value" style="color:{cc};font-size:20px;">
        {'+' if cum >= 0 else ''}{cum:.4f}
      </div>
    </div>
  </div>
  <div class="mc" style="margin-bottom:8px;">
    <div class="mc-accent" style="background:var(--violet);"></div>
    <div class="mc-label">Submitted severity</div>
    <div class="mc-value" style="color:var(--violet);font-size:22px;">{sub}</div>
  </div>
  <div class="prog-wrap"><div class="prog-fill" style="width:{pct}%;"></div></div>
  <div style="font-family:var(--font-mono);font-size:9px;color:var(--text3);text-align:right;">{pct}%</div>
</div>"""

    # Not started yet
    if not env_obj._scenario:
        return """
<div class="empty-state" style="padding:28px 16px;">
  <div class="empty-icon">◌</div>
  <div class="empty-label">Select a task → Reset</div>
</div>"""

    task     = env_obj._task_name or "—"
    step     = env_obj._step_number
    max_s    = env_obj._scenario.max_steps
    resolved = env_obj._incident_resolved
    done     = env_obj._done
    cum      = round(env_obj._cumulative_reward, 4)
    pct      = int(step / max(max_s, 1) * 100)
    streak   = env_obj._consecutive_diagnosis_count
    coll     = len(env_obj._collateral_degraded)
    cc       = "var(--green)" if cum >= 0 else "var(--red)"
    sc       = ("var(--red)" if streak >= 3 else
                "var(--amber)" if streak >= 2 else "var(--text2)")
    dlabel, dcolor, dbg = DIFF_MAP.get(task, ("?", "var(--text1)", "var(--bg3)"))

    if resolved:
        status_c, status_t = "var(--green)", "resolved"
    elif done:
        status_c, status_t = "var(--red)",   "done — not resolved"
    else:
        status_c, status_t = "var(--amber)",  "active"

    return f"""
<div>
  <div style="display:flex;align-items:center;gap:8px;margin-bottom:12px;">
    <span class="live-dot" style="background:{status_c};box-shadow:0 0 0 3px {status_c}20;"></span>
    <span style="font-family:var(--font-mono);font-size:10px;color:{status_c};
                 letter-spacing:0.08em;">{status_t}</span>
    <span style="margin-left:auto;font-family:var(--font-mono);font-size:9px;padding:2px 9px;
                 border-radius:4px;color:{dcolor};background:{dbg};border:1px solid {dcolor}25;">{dlabel}</span>
  </div>
  <div style="display:grid;grid-template-columns:1fr 1fr;gap:8px;margin-bottom:8px;">
    <div class="mc"><div class="mc-accent" style="background:var(--cyan);"></div>
      <div class="mc-label">Step</div>
      <div class="mc-value" style="color:var(--cyan);font-size:22px;">{step}
        <span style="font-size:13px;color:var(--text2);font-weight:400;"> /{max_s}</span>
      </div>
    </div>
    <div class="mc"><div class="mc-accent" style="background:{cc};"></div>
      <div class="mc-label">Cum. Reward</div>
      <div class="mc-value" style="color:{cc};font-size:18px;">
        {'+' if cum >= 0 else ''}{cum:.4f}
      </div>
    </div>
    <div class="mc"><div class="mc-accent" style="background:{sc};"></div>
      <div class="mc-label">Diag streak</div>
      <div class="mc-value" style="color:{sc};font-size:22px;">
        {streak}{'⚠' if streak >= 2 else ''}
      </div>
    </div>
    <div class="mc"><div class="mc-accent" style="background:var(--violet);"></div>
      <div class="mc-label">Collateral</div>
      <div class="mc-value" style="color:var(--violet);font-size:22px;">{coll}</div>
    </div>
  </div>
  <div style="display:flex;align-items:center;justify-content:space-between;
              background:var(--bg2);border:1px solid var(--line);border-radius:var(--r);
              padding:9px 14px;margin-bottom:8px;">
    <span style="font-family:var(--font-mono);font-size:10px;letter-spacing:0.1em;
                 text-transform:uppercase;color:var(--text2);">Incident resolved</span>
    {'<span style="font-family:var(--font-mono);font-size:11px;color:var(--green);font-weight:500;">YES ✓</span>'
      if resolved else
     '<span style="font-family:var(--font-mono);font-size:11px;color:var(--red);">NO ✗</span>'}
  </div>
  <div class="prog-wrap"><div class="prog-fill" style="width:{pct}%;"></div></div>
  <div style="display:flex;justify-content:space-between;font-family:var(--font-mono);
              font-size:9px;color:var(--text3);">
    <span>{step} / {max_s} steps</span><span>{pct}%</span>
  </div>
</div>"""


# ── 6D Score panel ────────────────────────────────────────────────────────────

def render_score(breakdown: Dict) -> str:
    if not breakdown:
        return "*Execute actions, then click **Grade (6D)** to evaluate.*"

    # Alert triage has a different breakdown structure
    is_triage = "severity_match" in breakdown.get("breakdown", {})
    if is_triage:
        bd    = breakdown.get("breakdown", {})
        total = breakdown.get("final", 0.0)
        sc    = "🟢" if total >= 0.8 else ("🟡" if total >= 0.5 else "🔴")
        fb    = breakdown.get("feedback", "")
        return f"""### {sc} Alert Triage Score: **{total:.4f}** / 1.0

| Component | Value |
|---|---|
| Submitted | `{bd.get('submitted_severity', '—')}` |
| Correct | `{bd.get('correct_severity', '—')}` |
| Severity score | `{bd.get('severity_match', 0.0):.2f}` (1.0 exact · 0.5 adjacent · 0.25 two-off) |
| Investigation bonus | `+{bd.get('investigation_bonus', 0.0):.2f}` |
| **Total** | **`{total:.4f}`** |

{('> ' + fb) if fb else ''}"""

    final    = breakdown.get("final", 0.0)
    ft       = breakdown.get("failure_type", "Unknown")
    icon     = FAILURE_TYPE_ICON.get(ft, "·")
    obs_loop = breakdown.get("observation_loop", False)
    sc       = "🟢" if final >= 0.7 else ("🟡" if final >= 0.4 else "🔴")
    obs_note = f"\n> ⚠ Observation loop — score capped at {_OBSERVATION_LOOP_CAP}" if obs_loop else ""

    dims = [
        ("root_cause",    "Root Cause",    0.30),
        ("remediation",   "Remediation",   0.25),
        ("investigation", "Investigation", 0.15),
        ("efficiency",    "Efficiency",    0.10),
        ("safety",        "Safety",        0.10),
        ("sequence",      "Sequence",      0.10),
    ]
    rows = "\n".join(
        f"| {l} | `{breakdown.get(k, 0.0):.2f}` | ×{w:.2f} | "
        f"{'█' * int(breakdown.get(k, 0) * 10)}{'░' * (10 - int(breakdown.get(k, 0) * 10))} |"
        for k, l, w in dims
    )
    fb = breakdown.get("feedback", "")

    return f"""### {sc} Final Score: **{final:.4f}** / 1.0

**Pattern:** {icon} {ft}{obs_note}

#### 6D Breakdown
| Dimension | Score | Weight | Bar |
|---|---:|---:|---|
{rows}

{('> ' + fb) if fb else ''}"""
