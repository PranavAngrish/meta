"""
graders.py — Standalone deterministic graders for the Incident Response environment.

Public API:
    grade(env_state: dict) -> dict
        Returns: {"root_cause", "remediation", "investigation", "efficiency", "safety",
                  "sequence", "final", "feedback", "failure_type"}

All scores are in the open interval (0.01, 0.99) for OpenEnv validator compatibility.
Called by POST /grader and POST /baseline endpoints independently of live env state.

6-Dimensional Scoring (v3.0):
    root_cause   (30%) — correct root cause declarations vs required
    remediation  (25%) — correct remediation actions applied
    investigation(15%) — services investigated / total services
    efficiency   (10%) — steps used relative to max budget
    safety       (10%) — no destructive actions on healthy/unrelated services
    sequence     (10%) — reasoning quality: diagnosis before fix, correct ordering

v3.0 additions (ported from ai-incident-openenv-trial):
  1. Sequence Score (new 6th dimension) — rewards correct investigation→diagnosis→fix order
  2. Failure Type Classification — categorises episode into:
       Efficient Reasoner | Symptom Chaser | Lucky Guesser |
       Stuck in Observation Loop | Late Corrector
  3. Diagnosis Gate enforcement — in grader scoring, blind fixes on gated scenarios
     receive only 50% remediation credit
  4. Observation Loop Detection — hard-caps final score at 0.45 when agent spends
     ≥3 consecutive steps diagnosing without any fix attempt

Design principles:
    - Transparent: every component is named and weighted explicitly
    - Partial credit: agents get credit for partial progress
    - Safety dimension: discourages shotgun remediation
    - Sequence dimension: rewards investigation-first reasoning (anti lucky-guesser)
    - Anti-reward-hacking: observation loop hard cap, diagnosis gate partial credit
    - Open-interval: never returns exactly 0.0 or 1.0 (validator compliance)
    - Service alias normalisation: "order_service" matches "order-service"
"""

from __future__ import annotations

# ── Open-interval constants ────────────────────────────────────────────────
_MIN_SCORE = 0.01
_MAX_SCORE = 0.99

# Hard cap for observation-loop agents (≥3 consecutive diagnosis steps, no fix)
_OBSERVATION_LOOP_CAP = 0.45

# Diagnosis gate scenarios — blind fix receives only partial remediation credit
_GATED_SCENARIOS = {
    "ssl_certificate_expiry",
    "database_deadlock",
    "cascading_service_timeout",
}

# Investigation actions (used for sequence + loop detection)
_INVESTIGATION_ACTIONS = {
    "investigate_logs", "check_metrics", "read_config",
    "check_service_health", "run_diagnostic",
}

# Destructive/fix actions
_DESTRUCTIVE_ACTIONS = {"restart_service", "rollback_deployment", "scale_service"}


def _clamp(score: float) -> float:
    """Clamp score to open interval (0.01, 0.99) for OpenEnv validator compliance."""
    return round(max(_MIN_SCORE, min(_MAX_SCORE, score)), 4)


# ── Service alias normalisation ────────────────────────────────────────────

_ALIASES: dict[str, str] = {
    # Database
    "postgres": "postgres-primary",
    "postgresql": "postgres-primary",
    "postgres_primary": "postgres-primary",
    "primary_db": "primary-db",
    "primarydb": "primary-db",
    "db": "postgres-primary",
    # APIs / gateways
    "api": "api-gateway",
    "api_gateway": "api-gateway",
    "apigw": "api-gateway",
    "gateway": "api-gateway",
    # Auth
    "auth": "auth-service",
    "authentication": "auth-service",
    # Order
    "order": "order-service",
    "orders": "order-service",
    "order_service": "order-service",
    # Inventory
    "inventory": "inventory-service",
    "inventory_service": "inventory-service",
    # Payment
    "payment": "payment-service",
    "payments": "payment-service",
    "payment_service": "payment-service",
    # User
    "user": "user-api",
    "users": "user-api",
    "user_api": "user-api",
    "userapi": "user-api",
    # Analytics
    "analytics": "analytics-service",
    "analytics_service": "analytics-service",
    # Redis / cache
    "redis": "redis-cache",
    "cache": "redis-cache",
    "redis_cache": "redis-cache",
    # Search
    "search": "search-service",
    "search_service": "search-service",
    # Cert / TLS
    "cert": "cert-manager",
    "tls": "cert-manager",
    "certmanager": "cert-manager",
    "cert_manager": "cert-manager",
    # Misc
    "nginx": "nginx-lb",
    "lb": "nginx-lb",
    "loadbalancer": "nginx-lb",
}


def _normalise_svc(name: str) -> str:
    """Lowercase, strip, normalise separators, apply alias map."""
    n = name.lower().strip().replace("_", "-").replace(" ", "-")
    return _ALIASES.get(n, n)


def _svc_match(submitted: str, correct: str) -> bool:
    """Return True if submitted service name matches the correct one (with aliases)."""
    s = _normalise_svc(submitted)
    c = _normalise_svc(correct)
    if s == c:
        return True
    return s in c or c in s


# ── Observation Loop Detection ─────────────────────────────────────────────

def _detect_observation_loop(agent_actions: list[str]) -> bool:
    """
    v3.0: Detect if an agent spent ≥3 consecutive steps diagnosing without any fix.
    This catches agents that 'observation-loop' — spamming check_* with no fixes.
    """
    max_streak = 0
    current_streak = 0
    for action_str in agent_actions:
        # Extract action type from "investigate_logs(user-api)" format
        action_type = action_str.split("(")[0].strip()
        if action_type in _INVESTIGATION_ACTIONS:
            current_streak += 1
            max_streak = max(max_streak, current_streak)
        else:
            current_streak = 0
    return max_streak >= 3


def _root_cause_fixed_before_step(agent_actions: list[str], correct_remediations: list[dict]) -> int | None:
    """Return the step index (0-based) at which the root cause was first fixed, or None."""
    correct_targets = {r.get("service_name", "") for r in correct_remediations}
    correct_action_types = {r.get("action_type", "") for r in correct_remediations}
    for i, action_str in enumerate(agent_actions):
        action_type = action_str.split("(")[0].strip()
        # Extract service name from "action_type(service_name)" format
        svc = ""
        if "(" in action_str and action_str.endswith(")"):
            svc = action_str.split("(", 1)[1].rstrip(")")
        if action_type in correct_action_types:
            if any(_svc_match(svc, t) for t in correct_targets):
                return i
    return None


def _diagnosed_root_cause_before_fix(
    agent_actions: list[str],
    correct_remediations: list[dict],
    fix_step: int | None,
) -> bool:
    """
    Return True if the agent investigated the root-cause service at least once
    BEFORE applying the correct fix.
    """
    if fix_step is None:
        return False
    correct_targets = {r.get("service_name", "") for r in correct_remediations}
    for i, action_str in enumerate(agent_actions[:fix_step]):
        action_type = action_str.split("(")[0].strip()
        svc = ""
        if "(" in action_str and action_str.endswith(")"):
            svc = action_str.split("(", 1)[1].rstrip(")")
        if action_type in _INVESTIGATION_ACTIONS:
            if any(_svc_match(svc, t) for t in correct_targets):
                return True
    return False


# ── Failure Type Classification ────────────────────────────────────────────

def _classify_failure_type(
    agent_actions: list[str],
    correct_remediations: list[dict],
    incident_resolved: bool,
    step_number: int,
    max_steps: int,
    observation_loop: bool,
) -> str:
    """
    v3.0: Classify the episode into one of 5 failure types.

    Efficient Reasoner  — root cause fixed in first 40% of budget, diagnosis first, no symptom fixes
    Symptom Chaser      — applied fixes to non-root-cause services before the real fix
    Lucky Guesser       — fixed root cause without prior diagnosis
    Stuck in Obs Loop   — ≥3 consecutive diagnosis steps, no fix ever attempted
    Late Corrector      — root cause fixed but after 60% of step budget
    """
    fix_step = _root_cause_fixed_before_step(agent_actions, correct_remediations)
    diagnosed_first = _diagnosed_root_cause_before_fix(agent_actions, correct_remediations, fix_step)

    # Symptom chaser: applied destructive actions to non-correct-target services before fix
    correct_targets = {r.get("service_name", "") for r in correct_remediations}
    symptom_fixes = 0
    for action_str in agent_actions:
        action_type = action_str.split("(")[0].strip()
        svc = ""
        if "(" in action_str and action_str.endswith(")"):
            svc = action_str.split("(", 1)[1].rstrip(")")
        if action_type in _DESTRUCTIVE_ACTIONS:
            if svc and not any(_svc_match(svc, t) for t in correct_targets):
                symptom_fixes += 1

    if observation_loop and fix_step is None:
        return "Stuck in Observation Loop"

    if fix_step is not None:
        ratio = fix_step / max(max_steps, 1)
        if diagnosed_first and symptom_fixes == 0 and ratio <= 0.4:
            return "Efficient Reasoner"
        if not diagnosed_first:
            return "Lucky Guesser"
        if ratio > 0.6:
            return "Late Corrector"

    if symptom_fixes > 0:
        return "Symptom Chaser"

    if not incident_resolved:
        return "Stuck in Observation Loop"

    return "Late Corrector"


# ── Sequence Score ─────────────────────────────────────────────────────────

def _compute_sequence_score(
    agent_actions: list[str],
    correct_remediations: list[dict],
    incident_resolved: bool,
    step_number: int,
    max_steps: int,
    task_name: str,
    observation_loop: bool,
) -> float:
    """
    v3.0: Sequence score rewards:
      - Investigating root-cause service BEFORE fixing it
      - Fixing root cause early (within 40% of step budget)
      - Penalises symptom fixes before root-cause fix
      - Penalises observation loops
    """
    fix_step = _root_cause_fixed_before_step(agent_actions, correct_remediations)
    diagnosed_first = _diagnosed_root_cause_before_fix(agent_actions, correct_remediations, fix_step)

    score = 0.0

    # Reward: diagnosed before fix
    if fix_step is not None and diagnosed_first:
        score += 0.40

    # Reward: fix early in episode
    if fix_step is not None:
        ratio = fix_step / max(max_steps, 1)
        score += max(0.0, 1.0 - ratio) * 0.40

    # Reward: full resolution
    if incident_resolved:
        score += 0.20

    # Penalty: symptom fixes before root-cause fix
    correct_targets = {r.get("service_name", "") for r in correct_remediations}
    symptom_fix_count = 0
    for action_str in agent_actions:
        action_type = action_str.split("(")[0].strip()
        svc = ""
        if "(" in action_str and action_str.endswith(")"):
            svc = action_str.split("(", 1)[1].rstrip(")")
        if action_type in _DESTRUCTIVE_ACTIONS:
            if svc and not any(_svc_match(svc, t) for t in correct_targets):
                symptom_fix_count += 1

    score -= symptom_fix_count * 0.15

    # Penalty: observation loop
    if observation_loop:
        score -= 0.20

    return round(max(0.0, min(1.0, score)), 4)


# ── Main grader entry point ────────────────────────────────────────────────

def grade(env_state: dict) -> dict:
    """
    Grade an episode from its raw state dict (as returned by GET /state → state key).

    Args:
        env_state: dict with keys matching IncidentResponseState fields

    Returns:
        {
          "root_cause": float,        # 0-1 component score (weight 0.30)
          "remediation": float,       # 0-1 component score (weight 0.25)
          "investigation": float,     # 0-1 component score (weight 0.15)
          "efficiency": float,        # 0-1 component score (weight 0.10)
          "safety": float,            # 0-1 component score (weight 0.10)
          "sequence": float,          # 0-1 component score (weight 0.10) [NEW v3.0]
          "final": float,             # clamped to (0.01, 0.99)
          "feedback": str,            # human-readable explanation
          "failure_type": str,        # classification [NEW v3.0]
          "observation_loop": bool,   # True if agent looped on diagnosis [NEW v3.0]
          "components_weighted": dict,
        }
    """
    import json as _json

    # ── Extract state fields ───────────────────────────────────────────────
    step_number = int(env_state.get("step_number", 0))
    max_steps = int(env_state.get("max_steps", 30))
    incident_resolved = bool(env_state.get("incident_resolved", False))
    cumulative_reward = float(env_state.get("cumulative_reward", 0.0))
    collateral_degraded = list(env_state.get("collateral_degraded_services", []))
    agent_actions = list(env_state.get("agent_actions_taken", []))
    remediation_applied = list(env_state.get("remediation_applied", []))
    root_causes_declared = list(env_state.get("root_causes", []))
    correct_remediations_raw = list(env_state.get("correct_remediations", []))
    services = dict(env_state.get("services", {}))
    task_name = str(env_state.get("task_name", ""))

    correct_remediations: list[dict] = []
    for r in correct_remediations_raw:
        if isinstance(r, str):
            try:
                correct_remediations.append(_json.loads(r))
            except Exception:
                pass
        elif isinstance(r, dict):
            correct_remediations.append(r)

    remediation_dicts: list[dict] = []
    for r in remediation_applied:
        if isinstance(r, str):
            try:
                remediation_dicts.append(_json.loads(r))
            except Exception:
                pass
        elif isinstance(r, dict):
            remediation_dicts.append(r)

    # ── v3.0: Observation Loop Detection ──────────────────────────────────
    observation_loop = _detect_observation_loop(agent_actions)

    # ── Component scores ───────────────────────────────────────────────────
    _pre = env_state.get("_grade_components")
    if _pre:
        root_cause_score = float(_pre.get("root_cause", 0.0))
        remediation_score = float(_pre.get("remediation", 0.0))
        investigation_score = float(_pre.get("investigation", 0.0))
        efficiency_score = float(_pre.get("efficiency", 0.0))
        safety_score = float(_pre.get("safety", 0.0))
    else:
        root_cause_score = 1.0 if incident_resolved else _estimate_rc_score(root_causes_declared, cumulative_reward)
        remediation_score = _compute_remediation_score(
            remediation_dicts, correct_remediations, task_name, agent_actions
        )
        investigation_score = _compute_investigation_score(agent_actions, services)
        efficiency_score = _compute_efficiency_score(step_number, max_steps)
        safety_score = _compute_safety_score(remediation_dicts, correct_remediations, collateral_degraded)

    # ── v3.0: Sequence Score ───────────────────────────────────────────────
    sequence_score = _compute_sequence_score(
        agent_actions, correct_remediations, incident_resolved,
        step_number, max_steps, task_name, observation_loop
    )

    # ── v3.0: Failure Type Classification ─────────────────────────────────
    failure_type = _classify_failure_type(
        agent_actions, correct_remediations, incident_resolved,
        step_number, max_steps, observation_loop
    )

    # ── Weighted total (v3.0 weights) ──────────────────────────────────────
    # Weights rebalanced to make room for sequence dimension:
    #   root_cause: 0.35 → 0.30
    #   remediation: 0.30 → 0.25
    #   sequence: NEW 0.10
    raw_final = (
        root_cause_score   * 0.30
        + remediation_score  * 0.25
        + investigation_score* 0.15
        + efficiency_score   * 0.10
        + safety_score       * 0.10
        + sequence_score     * 0.10
    )

    final = _clamp(raw_final)

    # ── v3.0: Observation Loop Hard Cap ───────────────────────────────────
    if observation_loop and not incident_resolved:
        final = min(final, _OBSERVATION_LOOP_CAP)

    components_weighted = {
        "root_cause×0.30":    round(root_cause_score   * 0.30, 4),
        "remediation×0.25":   round(remediation_score  * 0.25, 4),
        "investigation×0.15": round(investigation_score * 0.15, 4),
        "efficiency×0.10":    round(efficiency_score   * 0.10, 4),
        "safety×0.10":        round(safety_score       * 0.10, 4),
        "sequence×0.10":      round(sequence_score     * 0.10, 4),
    }

    feedback = _build_feedback(
        root_cause_score, remediation_score, investigation_score,
        efficiency_score, safety_score, sequence_score,
        final, incident_resolved, collateral_degraded,
        step_number, max_steps, failure_type, observation_loop,
    )

    return {
        "root_cause":          round(root_cause_score, 4),
        "remediation":         round(remediation_score, 4),
        "investigation":       round(investigation_score, 4),
        "efficiency":          round(efficiency_score, 4),
        "safety":              round(safety_score, 4),
        "sequence":            round(sequence_score, 4),          # NEW v3.0
        "final":               final,
        "feedback":            feedback,
        "failure_type":        failure_type,                      # NEW v3.0
        "observation_loop":    observation_loop,                  # NEW v3.0
        "components_weighted": components_weighted,
    }


# ── Component scorers ──────────────────────────────────────────────────────

def _estimate_rc_score(declared: list[str], cumulative_reward: float) -> float:
    if not declared:
        return 0.0
    positive_reward = max(0.0, cumulative_reward)
    return min(0.6, 0.2 + positive_reward * 0.1) if declared else 0.0


def _compute_remediation_score(
    applied: list[dict],
    correct: list[dict],
    task_name: str,
    agent_actions: list[str],
) -> float:
    """
    v3.0: Diagnosis Gate — for gated scenarios, blind fixes (no prior investigation
    of root-cause service) receive only 50% remediation credit instead of full credit.
    This discourages lucky-guessing and rewards deliberate investigation.
    """
    if not correct:
        return 1.0
    if not applied:
        return 0.0

    matched = 0
    for rem in correct:
        c_type = rem.get("action_type", "")
        c_svc = rem.get("service_name", "")
        for app in applied:
            if app.get("action_type") == c_type and _svc_match(app.get("service_name", ""), c_svc):
                # v3.0: Diagnosis gate — check if agent investigated this service first
                if task_name in _GATED_SCENARIOS:
                    investigated_first = _investigated_before_fix(agent_actions, c_svc, c_type)
                    matched += 1.0 if investigated_first else 0.5  # 50% credit for blind fix
                else:
                    matched += 1
                break

    return round(min(1.0, matched / len(correct)), 4)


def _investigated_before_fix(agent_actions: list[str], target_svc: str, fix_action: str) -> bool:
    """Return True if agent investigated target_svc before applying the fix action."""
    fix_index = None
    for i, action_str in enumerate(agent_actions):
        action_type = action_str.split("(")[0].strip()
        svc = ""
        if "(" in action_str and action_str.endswith(")"):
            svc = action_str.split("(", 1)[1].rstrip(")")
        if action_type == fix_action and _svc_match(svc, target_svc):
            fix_index = i
            break
    if fix_index is None:
        return True  # Fix never applied — not a gate violation
    for action_str in agent_actions[:fix_index]:
        action_type = action_str.split("(")[0].strip()
        svc = ""
        if "(" in action_str and action_str.endswith(")"):
            svc = action_str.split("(", 1)[1].rstrip(")")
        if action_type in _INVESTIGATION_ACTIONS and _svc_match(svc, target_svc):
            return True
    return False


def _compute_investigation_score(actions: list[str], services: dict) -> float:
    if not services:
        return 0.0
    investigated_svcs: set[str] = set()
    for action_str in actions:
        action_type = action_str.split("(")[0].strip()
        if action_type in _INVESTIGATION_ACTIONS:
            if "(" in action_str and action_str.endswith(")"):
                svc = action_str.split("(", 1)[1].rstrip(")")
                investigated_svcs.add(svc)
    return round(min(1.0, len(investigated_svcs) / max(len(services), 1)), 4)


def _compute_efficiency_score(step_number: int, max_steps: int) -> float:
    if step_number == 0 or max_steps == 0:
        return 0.0
    ratio = step_number / max_steps
    if ratio <= 0.4:
        return 1.0
    elif ratio <= 0.7:
        return 0.7
    elif ratio <= 1.0:
        return 0.4
    return 0.1


def _compute_safety_score(applied: list[dict], correct: list[dict], collateral: list[str]) -> float:
    correct_targets = {r.get("service_name", "") for r in correct}
    penalties = 0
    for action in applied:
        if action.get("action_type") in _DESTRUCTIVE_ACTIONS:
            target = action.get("service_name", "")
            if target and not any(_svc_match(target, ct) for ct in correct_targets):
                penalties += 1
    collateral_penalty = len(collateral) * 0.15
    raw = max(0.0, 1.0 - penalties * 0.25 - collateral_penalty)
    return round(raw, 4)


# ── Feedback builder ───────────────────────────────────────────────────────

def _build_feedback(
    rc: float, rem: float, inv: float, eff: float, saf: float, seq: float,
    final: float, resolved: bool, collateral: list,
    steps: int, max_steps: int,
    failure_type: str, observation_loop: bool,
) -> str:
    lines = []
    status = "RESOLVED" if resolved else "UNRESOLVED"
    lines.append(f"Score: {final:.4f} ({status}) | Failure Type: {failure_type}")
    lines.append(
        f"  root_cause={rc:.2f}×0.30  remediation={rem:.2f}×0.25  "
        f"investigation={inv:.2f}×0.15  efficiency={eff:.2f}×0.10  "
        f"safety={saf:.2f}×0.10  sequence={seq:.2f}×0.10"
    )

    # Failure type advice
    if failure_type == "Stuck in Observation Loop":
        lines.append("  → Observation Loop detected — score hard-capped at 0.45. Attempt a fix after diagnosing.")
    elif failure_type == "Lucky Guesser":
        lines.append("  → Lucky Guesser — fix applied without prior diagnosis. Investigate first for full sequence credit.")
    elif failure_type == "Symptom Chaser":
        lines.append("  → Symptom Chaser — non-root-cause services fixed before root cause. Focus on root cause first.")
    elif failure_type == "Late Corrector":
        lines.append("  → Late Corrector — root cause fixed but after >60% of step budget. Act faster.")
    elif failure_type == "Efficient Reasoner":
        lines.append("  → Efficient Reasoner — excellent! Diagnosed and fixed root cause early with no wasted steps.")

    if rc < 0.5:
        lines.append("  → Root cause not identified — use declare_root_cause with specific service + failure mode.")
    if rem < 0.5:
        lines.append("  → Correct remediation not applied — check correct_remediations for required actions.")
    if inv < 0.5:
        lines.append("  → Investigate more services — check logs, metrics, configs before remediating.")
    if eff < 0.7:
        lines.append(f"  → Efficiency low — used {steps}/{max_steps} steps. Aim for under 40% of budget.")
    if saf < 0.7:
        lines.append("  → Safety penalty — destructive action on non-root-cause service, or collateral damage occurred.")
    if seq < 0.4:
        lines.append("  → Sequence score low — investigate root cause service before applying fix for sequence bonus.")
    if observation_loop:
        lines.append(f"  → ⚠ Observation Loop: ≥3 consecutive diagnosis steps with no fix. Score capped at {_OBSERVATION_LOOP_CAP}.")
    if collateral:
        lines.append(f"  → Collateral degraded services: {', '.join(collateral)}")

    return " | ".join(lines)


# ── Alert Triage Grader ────────────────────────────────────────────────────

_SEVERITY_ORDER = ["P1", "P2", "P3", "P4"]


def grade_alert_triage(env_state: dict, scenario: dict) -> dict:
    """
    Grade an alert_triage episode.

    Scoring:
      1.00 — exact severity match
      0.50 — adjacent severity (±1 step)
      0.25 — two steps off
      0.00 — wrong by 3+ or no submission

    Bonus for investigation quality:
      +0.10 if agent investigated ≥2 distinct services before submitting
      +0.05 if agent queried both logs/metrics AND health/diagnostic

    All final scores clamped to open interval (0.01, 0.99).
    """
    agent_actions = list(env_state.get("agent_actions_taken", []))
    step_number = int(env_state.get("step_number", 0))
    max_steps = int(env_state.get("max_steps", 3))

    correct_severity = scenario.get("correct_severity", "P1")
    adjacent = set(scenario.get("adjacent_severities", []))

    # Find submitted severity
    submitted = None
    for action_str in agent_actions:
        at = action_str.split("(")[0].strip()
        if at == "submit_severity":
            # Extract from format: submit_severity(severity=P1) or submit_severity(P1)
            if "(" in action_str and action_str.endswith(")"):
                inner = action_str.split("(", 1)[1].rstrip(")")
                # handle 'severity=P1' or just 'P1'
                if "=" in inner:
                    submitted = inner.split("=", 1)[1].strip().upper()
                else:
                    submitted = inner.strip().upper()
            break

    if not submitted:
        return {
            "total": 0.01,
            "breakdown": {"submitted": False, "severity_match": 0.0, "investigation_bonus": 0.0},
            "feedback": "No severity submitted — score 0.01 (minimum)",
        }

    # Base severity score
    if submitted == correct_severity:
        base, msg = 1.0, f"Exact match: {submitted}"
    elif submitted in adjacent:
        base, msg = 0.5, f"Adjacent: submitted {submitted}, correct {correct_severity}"
    else:
        try:
            dist = abs(_SEVERITY_ORDER.index(submitted) - _SEVERITY_ORDER.index(correct_severity))
        except ValueError:
            dist = 4
        base = 0.25 if dist == 2 else 0.0
        msg = f"Wrong: submitted {submitted}, correct {correct_severity} (distance={dist})"

    # Investigation bonus — rewards reading evidence before classifying
    investigated_svcs: set[str] = set()
    action_types_used: set[str] = set()
    for action_str in agent_actions:
        at = action_str.split("(")[0].strip()
        if at in _INVESTIGATION_ACTIONS:
            action_types_used.add(at)
            if "(" in action_str and action_str.endswith(")"):
                svc = action_str.split("(", 1)[1].rstrip(")")
                investigated_svcs.add(svc)

    breadth_bonus = 0.10 if len(investigated_svcs) >= 2 else (0.05 if len(investigated_svcs) == 1 else 0.0)
    depth_bonus = 0.05 if len(action_types_used) >= 2 else 0.0
    investigation_bonus = breadth_bonus + depth_bonus

    total = _clamp(min(1.0, base + investigation_bonus))

    return {
        "total": total,
        "breakdown": {
            "submitted_severity": submitted,
            "correct_severity": correct_severity,
            "severity_match": base,
            "investigation_bonus": investigation_bonus,
            "services_investigated": len(investigated_svcs),
            "action_types_used": len(action_types_used),
        },
        "feedback": (
            f"{msg} | investigation_bonus={investigation_bonus:.2f} "
            f"(svcs={len(investigated_svcs)}, action_types={len(action_types_used)}) "
            f"| total={total:.4f}"
        ),
    }
