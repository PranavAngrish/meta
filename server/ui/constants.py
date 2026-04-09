"""
Shared display constants for the Incident Response Gradio UI.

Centralising these prevents magic-string drift between renderers,
callbacks, and the layout builder.
"""
from __future__ import annotations

# ── Action categories ─────────────────────────────────────────────────────────
DIAG_ACTIONS: set[str] = {
    "investigate_logs",
    "check_metrics",
    "read_config",
    "check_service_health",
    "run_diagnostic",
}

FIX_ACTIONS: set[str] = {
    "restart_service",
    "update_config",
    "rollback_deployment",
    "scale_service",
}

DECLARE_ACTIONS: set[str] = {
    "declare_root_cause",
    "submit_severity",
}

# ── Failure-type display ──────────────────────────────────────────────────────
FAILURE_TYPE_ICON: dict[str, str] = {
    "Efficient Reasoner":        "◆",
    "Symptom Chaser":            "◇",
    "Lucky Guesser":             "○",
    "Stuck in Observation Loop": "⊗",
    "Late Corrector":            "◎",
    "Unknown":                   "·",
}

# ── Task difficulty metadata ──────────────────────────────────────────────────
# value: (label, CSS colour var, CSS background var)
DIFF_MAP: dict[str, tuple[str, str, str]] = {
    "db_connection_failure":     ("Easy",   "var(--green)",  "var(--green-bg)"),
    "cascading_service_timeout": ("Medium", "var(--amber)",  "var(--amber-bg)"),
    "ssl_certificate_expiry":    ("Medium", "var(--amber)",  "var(--amber-bg)"),
    "multi_factor_outage":       ("Hard",   "var(--red)",    "var(--red-bg)"),
    "database_deadlock":         ("Hard",   "var(--red)",    "var(--red-bg)"),
    "alert_triage":              ("Triage", "var(--cyan)",   "var(--cyan-bg)"),
}

# ── Action type display metadata ──────────────────────────────────────────────
# value: (icon, label, colour var, background var)
ACTION_META: dict[str, tuple[str, str, str, str]] = {
    "investigate_logs":     ("◈", "Investigate", "var(--cyan)",   "var(--cyan-bg)"),
    "check_metrics":        ("◈", "Metrics",     "var(--cyan)",   "var(--cyan-bg)"),
    "read_config":          ("◈", "Config",      "var(--cyan)",   "var(--cyan-bg)"),
    "check_service_health": ("◈", "Health",      "var(--cyan)",   "var(--cyan-bg)"),
    "run_diagnostic":       ("◈", "Diagnostic",  "var(--cyan)",   "var(--cyan-bg)"),
    "restart_service":      ("⬡", "Restart",     "var(--green)",  "var(--green-bg)"),
    "update_config":        ("⬡", "Update",      "var(--green)",  "var(--green-bg)"),
    "rollback_deployment":  ("⬡", "Rollback",    "var(--green)",  "var(--green-bg)"),
    "scale_service":        ("⬡", "Scale",       "var(--green)",  "var(--green-bg)"),
    "declare_root_cause":   ("◆", "Declare",     "var(--amber)",  "var(--amber-bg)"),
    "submit_severity":      ("◆", "Submit",      "var(--violet)", "var(--violet-bg)"),
}

ALL_TASKS = [
    "db_connection_failure",
    "cascading_service_timeout",
    "multi_factor_outage",
    "ssl_certificate_expiry",
    "database_deadlock",
    "alert_triage",
]
