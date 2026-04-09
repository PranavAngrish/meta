"""
Global CSS and static HTML fragments for the Incident Response Gradio UI.

Design language: dark-mode terminal aesthetic with amber/cyan accents,
Cabinet Grotesk for headings and JetBrains Mono for all monospace text.
"""

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
    --line:  rgba(255,255,255,0.06);
    --line2: rgba(255,255,255,0.10);
    --text0: #eef0f6;
    --text1: #9ba3bc;
    --text2: #5a6380;
    --text3: #373d55;
    --amber:      #f5a623;
    --amber2:     #ffc55a;
    --amber-bg:   rgba(245,166,35,0.08);
    --cyan:       #22d3ee;
    --cyan-bg:    rgba(34,211,238,0.08);
    --green:      #34d399;
    --green-bg:   rgba(52,211,153,0.08);
    --red:        #f87171;
    --red-bg:     rgba(248,113,113,0.08);
    --violet:     #a78bfa;
    --violet-bg:  rgba(167,139,250,0.08);
    --r:  10px;
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

/* ── Tabs ── */
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

/* ── Panels ── */
.gr-group, .gr-box, .gr-form, div.gr-block {
    background: var(--bg2) !important;
    border: 1px solid var(--line) !important;
    border-radius: var(--r2) !important;
}

/* ── Inputs ── */
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

/* ── Buttons ── */
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
/* Disabled state */
.gr-button[disabled], button:disabled {
    opacity: 0.38 !important;
    cursor: not-allowed !important;
    transform: none !important;
    box-shadow: none !important;
}

/* ── Markdown ── */
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

/* ── Accordion ── */
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

/* ── Dropdown ── */
.gr-dropdown ul {
    background: var(--bg3) !important;
    border: 1px solid var(--line2) !important;
    border-radius: var(--r) !important;
    box-shadow: 0 8px 32px rgba(0,0,0,0.5) !important;
}
.gr-dropdown li {
    color: var(--text1) !important;
    font-family: var(--font-mono) !important;
    font-size: 12px !important;
}
.gr-dropdown li:hover { background: var(--amber-bg) !important; color: var(--amber) !important; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 4px; height: 4px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: var(--line2); border-radius: 2px; }
::-webkit-scrollbar-thumb:hover { background: rgba(245,166,35,0.3); }

.gr-row    { gap: 18px !important; }
.gr-column { gap: 12px !important; }

/* ══════════════ CUSTOM COMPONENT CLASSES ══════════════ */

/* Section divider */
.ir-section {
    font-family: var(--font-mono, monospace);
    font-size: 10px; font-weight: 400;
    letter-spacing: 0.16em; text-transform: uppercase;
    color: var(--text3, #373d55);
    padding: 16px 0 8px;
    display: flex; align-items: center; gap: 10px;
    user-select: none;
}
.ir-section::after {
    content: ''; flex: 1; height: 1px;
    background: var(--line, rgba(255,255,255,0.06));
}

/* Metric card */
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
.mc-label {
    font-family: var(--font-mono);
    font-size: 9px; letter-spacing: 0.16em; text-transform: uppercase;
    color: var(--text2); margin-bottom: 8px;
}
.mc-value {
    font-family: var(--font-head);
    font-size: 26px; font-weight: 800;
    letter-spacing: -0.5px; line-height: 1; color: var(--text0);
}

/* Alert card */
.alert-card {
    display: flex; gap: 12px;
    padding: 13px 15px; border-radius: var(--r);
    border: 1px solid; margin-bottom: 8px;
    transition: transform 0.15s;
}
.alert-card:hover { transform: translateX(3px); }
.alert-badge {
    font-family: var(--font-mono);
    font-size: 9px; font-weight: 500; letter-spacing: 0.12em;
    padding: 3px 8px; border-radius: 4px;
    flex-shrink: 0; margin-top: 1px; height: fit-content;
}
.alert-name {
    font-family: var(--font-head);
    font-size: 13px; font-weight: 700; color: var(--text0); margin-bottom: 3px;
}
.alert-id {
    font-family: var(--font-mono);
    font-size: 10px; color: var(--text2); font-weight: 300; margin-left: 6px;
}
.alert-msg {
    font-family: var(--font-head);
    font-size: 12px; color: var(--text1); line-height: 1.5;
}

/* Service row */
.svc-row {
    display: grid; grid-template-columns: 1fr auto auto auto;
    align-items: center; padding: 10px 14px;
    border-bottom: 1px solid rgba(255,255,255,0.03);
    transition: background 0.15s;
}
.svc-row:hover { background: rgba(255,255,255,0.02); }
.svc-row:last-child { border-bottom: none; }
.svc-name { font-family: var(--font-head); font-size: 13px; font-weight: 600; color: var(--text0); }
.svc-pill {
    display: inline-flex; align-items: center; gap: 5px;
    font-family: var(--font-mono); font-size: 10px;
    padding: 3px 9px; border-radius: 100px;
}
.svc-stat {
    font-family: var(--font-mono); font-size: 11px; color: var(--text1);
    text-align: right; min-width: 70px; padding: 0 10px;
}

/* Action chip */
.action-chip {
    display: inline-flex; align-items: center; gap: 4px;
    font-family: var(--font-mono); font-size: 10px;
    padding: 2px 8px; border-radius: 100px; white-space: nowrap;
}

/* Log block */
.log-block {
    font-family: var(--font-mono); font-size: 11px; line-height: 1.9;
    background: var(--bg1); border: 1px solid var(--line);
    border-left: 2px solid var(--cyan);
    border-radius: var(--r); padding: 13px 16px;
    color: var(--text1); max-height: 220px; overflow-y: auto;
    white-space: pre-wrap; word-break: break-all;
}

/* Progress bar */
.prog-wrap {
    position: relative; height: 3px;
    background: rgba(255,255,255,0.05);
    border-radius: 2px; overflow: hidden; margin: 10px 0 5px;
}
.prog-fill {
    position: absolute; inset: 0; border-radius: 2px;
    background: linear-gradient(90deg, var(--amber), var(--cyan));
    transition: width 0.5s cubic-bezier(.4,0,.2,1);
}

/* Empty state */
.empty-state {
    display: flex; flex-direction: column;
    align-items: center; justify-content: center;
    padding: 44px 24px; text-align: center;
}
.empty-icon {
    width: 40px; height: 40px; border-radius: 10px;
    background: var(--bg3); border: 1px solid var(--line);
    display: flex; align-items: center; justify-content: center;
    font-size: 18px; margin-bottom: 12px; opacity: 0.5;
}
.empty-label {
    font-family: var(--font-head); font-size: 13px;
    color: var(--text2); line-height: 1.6;
}
.empty-label strong { color: var(--text1); font-weight: 600; }

/* History table */
.hist-table { width: 100%; border-collapse: collapse; }
.hist-th {
    font-family: var(--font-mono); font-size: 9px;
    letter-spacing: 0.14em; text-transform: uppercase;
    color: var(--text3); padding: 0 12px 8px;
    text-align: left; border-bottom: 1px solid var(--line);
}
.hist-td {
    font-family: var(--font-mono); font-size: 11px;
    padding: 9px 12px;
    border-bottom: 1px solid rgba(255,255,255,0.03);
    color: var(--text1); vertical-align: middle;
}
.hist-row:hover .hist-td { background: rgba(255,255,255,0.015); }
.hist-row:last-child .hist-td { border-bottom: none; }

/* Panel label */
.panel-label {
    font-family: var(--font-mono); font-size: 9px;
    letter-spacing: 0.18em; text-transform: uppercase;
    color: var(--text3); margin-bottom: 6px;
    padding-left: 8px; border-left: 2px solid var(--amber);
}

/* Step detail card */
.step-card {
    background: var(--bg2); border: 1px solid var(--line);
    border-radius: var(--r2); overflow: hidden;
}
.step-card-header {
    display: flex; align-items: center; gap: 12px;
    padding: 14px 16px; border-bottom: 1px solid var(--line);
    background: var(--bg3);
}
.step-reward-pos { color: var(--green); }
.step-reward-neg { color: var(--red); }

/* Score dimension bar */
.dim-bar {
    display: flex; gap: 10px; align-items: center;
    padding: 8px 14px; border-bottom: 1px solid rgba(255,255,255,0.03);
}
.dim-bar:last-child { border-bottom: none; }
.dim-label {
    font-family: var(--font-mono); font-size: 10px;
    color: var(--text2); width: 90px; flex-shrink: 0;
}
.dim-track {
    flex: 1; height: 4px; background: rgba(255,255,255,0.05);
    border-radius: 2px; overflow: hidden;
}
.dim-fill { height: 100%; border-radius: 2px; transition: width 0.4s ease; }
.dim-value {
    font-family: var(--font-mono); font-size: 10px;
    color: var(--text1); width: 36px; text-align: right;
}

/* Pulsing live dot */
@keyframes pulse-dot { 0%,100%{opacity:1;} 50%{opacity:0.3;} }
.live-dot {
    width: 6px; height: 6px; border-radius: 50%;
    display: inline-block; flex-shrink: 0;
    animation: pulse-dot 2s ease-in-out infinite;
}

/* Warning banners */
.warn-loop {
    background: var(--red-bg);
    border: 1px solid rgba(248,113,113,0.2);
    border-radius: var(--r); padding: 9px 14px; margin: 8px 14px;
    font-family: var(--font-mono); font-size: 11px; color: var(--red);
}
.warn-streak {
    background: var(--amber-bg);
    border: 1px solid rgba(245,166,35,0.2);
    border-radius: var(--r); padding: 9px 14px; margin: 8px 14px;
    font-family: var(--font-mono); font-size: 11px; color: var(--amber);
}
.warn-degrade {
    background: var(--violet-bg);
    border: 1px solid rgba(167,139,250,0.2);
    border-radius: var(--r); padding: 9px 14px; margin: 8px 14px;
    font-family: var(--font-mono); font-size: 11px; color: var(--violet);
}

/* Episode done banner */
.episode-done-banner {
    background: var(--green-bg);
    border: 1px solid rgba(52,211,153,0.2);
    border-radius: var(--r); padding: 14px 18px; margin: 12px 14px;
    text-align: center;
}
"""


HEADER_HTML = """
<div style="background:#0d1018;border-bottom:1px solid rgba(255,255,255,0.06);
            padding:0 40px;height:62px;display:flex;align-items:center;
            justify-content:space-between;position:relative;">
  <div style="position:absolute;top:0;left:0;right:0;height:1px;
              background:linear-gradient(90deg,#f5a623 0%,#22d3ee 40%,transparent 100%);
              opacity:0.4;"></div>
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
