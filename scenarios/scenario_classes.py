"""
Concrete BaseScenario implementations for meta-hackathon.
=========================================================

Each scenario class wraps one of the existing ScenarioDef functions and
expresses its grading logic as a declarative list of RubricCheck objects.

This is the feature borrowed from incident-commander-env (self-contained
rubric per scenario) but built SUPERIOR by:
  - Per-dimension weights instead of a single flat sum
  - Investigation-gate checks at rubric level
  - Plugging into the existing ScenarioDef / ScenarioFactory without changes
  - Compatible with seeded variation (rubric checks use action + state data,
    not hardcoded metric values, so they work on any variation)

Scenarios implemented
---------------------
  EasyDBConnectionFailureScenario     — db_connection_failure (easy)
  MediumCascadingTimeoutScenario      — cascading_service_timeout (medium)
  HardMultiFactorOutageScenario       — multi_factor_outage (hard)
  MediumSSLCertExpiryScenario         — ssl_certificate_expiry (medium)
  HardDatabaseDeadlockScenario        — database_deadlock (hard)

Usage
-----
  from scenarios.scenario_classes import SCENARIO_CLASS_REGISTRY, ScenarioRubricAdapter

  # Grade an existing episode using the new rubric:
  scenario_cls = SCENARIO_CLASS_REGISTRY["db_connection_failure"]
  details = scenario_cls().grade_details(
      actions=env._actions_taken,
      service_states=env._service_states,
      step_number=env._step_number,
      max_steps=env._scenario.max_steps,
      incident_resolved=env._incident_resolved,
  )

  # Or use ScenarioRubricAdapter to auto-select based on task name:
  from scenarios.scenario_classes import ScenarioRubricAdapter
  result = ScenarioRubricAdapter.grade_details_for(task_name, actions, states, ...)
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from scenarios.base_scenario import BaseScenario, RubricCheck

# ── shared helper ──────────────────────────────────────────────────────────


def _action_matches(actions: List[str], action_type: str, service: str = "") -> bool:
    """Return True if any action matches action_type (optionally also service)."""
    key = action_type.lower()
    svc = service.lower()
    for a in actions:
        a_low = a.lower()
        action_name = a_low.split("(")[0]
        if action_name == key:
            if not svc or svc in a_low:
                return True
    return False


def _action_type_count(actions: List[str], action_type: str, service: str = "") -> int:
    key = action_type.lower()
    svc = service.lower()
    return sum(
        1 for a in actions
        if a.lower().split("(")[0] == key and (not svc or svc in a.lower())
    )


def _investigated(actions: List[str], service: str) -> bool:
    """Return True if the agent investigated service with any diagnostic action."""
    diag = {"investigate_logs", "check_metrics", "read_config", "check_service_health", "run_diagnostic"}
    svc = service.lower()
    return any(a.lower().split("(")[0] in diag and svc in a.lower() for a in actions)


def _service_healthy(states: Dict[str, Any], service: str) -> bool:
    st = states.get(service, {})
    return bool(st.get("healthy", True))


def _config_value(states: Dict[str, Any], service: str, key: str) -> Any:
    st = states.get(service, {})
    cfg = st.get("config", {})
    return cfg.get(key)


# ─────────────────────────────────────────────────────────────────────────────
# 1. EASY: DB Connection Failure
# ─────────────────────────────────────────────────────────────────────────────

class EasyDBConnectionFailureScenario(BaseScenario):
    """
    Rubric for db_connection_failure.

    root_cause   (30%): declare user-api db_port misconfiguration
    remediation  (25%): fix db_port to 5432 on user-api
    investigation(15%): checked user-api logs + postgres-primary
    efficiency   (10%): computed dynamically by grader
    safety       (10%): didn't restart healthy postgres-primary
    sequence     (10%): investigated before fixing
    """
    task_id = "db_connection_failure"
    difficulty = "easy"
    description = "Single service database connection failure due to port misconfiguration"

    def get_rubric(self) -> List[RubricCheck]:
        return [
            # ── root_cause ────────────────────────────────────────────────
            RubricCheck(
                name="Declared db_port misconfiguration as root cause",
                check=lambda a, s: any(
                    "declare_root_cause" in act.lower()
                    and ("5433" in act or "port" in act.lower() or "db_port" in act.lower())
                    for act in a
                ),
                dimension="root_cause",
                weight=0.70,
            ),
            RubricCheck(
                name="Identified user-api as the failing service",
                check=lambda a, s: _investigated(a, "user-api"),
                dimension="root_cause",
                weight=0.30,
            ),

            # ── remediation ──────────────────────────────────────────────
            RubricCheck(
                name="Applied update_config to fix db_port on user-api",
                check=lambda a, s: (
                    _action_matches(a, "update_config", "user-api")
                    and str(_config_value(s, "user-api", "db_port")) == "5432"
                ),
                dimension="remediation",
                weight=1.0,
                gated=True,
                gate_service="user-api",
            ),

            # ── investigation ────────────────────────────────────────────
            RubricCheck(
                name="Investigated user-api logs",
                check=lambda a, s: _action_matches(a, "investigate_logs", "user-api"),
                dimension="investigation",
                weight=0.45,
            ),
            RubricCheck(
                name="Checked user-api config or diagnostics",
                check=lambda a, s: (
                    _action_matches(a, "read_config", "user-api")
                    or _action_matches(a, "run_diagnostic", "user-api")
                ),
                dimension="investigation",
                weight=0.35,
            ),
            RubricCheck(
                name="Verified postgres-primary is healthy",
                check=lambda a, s: _investigated(a, "postgres-primary"),
                dimension="investigation",
                weight=0.20,
            ),

            # ── safety ────────────────────────────────────────────────────
            RubricCheck(
                name="Did not restart healthy postgres-primary",
                check=lambda a, s: not _action_matches(a, "restart_service", "postgres-primary"),
                dimension="safety",
                weight=0.60,
            ),
            RubricCheck(
                name="Did not restart healthy nginx-lb",
                check=lambda a, s: not _action_matches(a, "restart_service", "nginx-lb"),
                dimension="safety",
                weight=0.40,
            ),

            # ── sequence ──────────────────────────────────────────────────
            RubricCheck(
                name="Investigated user-api before applying config fix",
                check=lambda a, s: (
                    # index of first investigation < index of first fix
                    next((i for i, x in enumerate(a) if "investigate_logs(user-api)" in x.lower()
                          or "read_config(user-api)" in x.lower()), 9999)
                    <
                    next((i for i, x in enumerate(a) if "update_config(user-api)" in x.lower()), 9999)
                ),
                dimension="sequence",
                weight=0.70,
            ),
            RubricCheck(
                name="Declared root cause before applying fix",
                check=lambda a, s: (
                    next((i for i, x in enumerate(a) if "declare_root_cause" in x.lower()), 9999)
                    <
                    next((i for i, x in enumerate(a) if "update_config(user-api)" in x.lower()), 9999)
                ),
                dimension="sequence",
                weight=0.30,
            ),
        ]


# ─────────────────────────────────────────────────────────────────────────────
# 2. MEDIUM: Cascading Service Timeout
# ─────────────────────────────────────────────────────────────────────────────

class MediumCascadingTimeoutScenario(BaseScenario):
    """
    Rubric for cascading_service_timeout.

    root_cause   (30%): identify inventory-service memory/GC as root cause
    remediation  (25%): restart inventory-service + fix jvm_heap_max
    investigation(15%): traced cascade: payment→order→inventory
    efficiency   (10%): dynamic
    safety       (10%): didn't restart healthy orders-db or payment-service
    sequence     (10%): investigated cascade order correctly
    """
    task_id = "cascading_service_timeout"
    difficulty = "medium"
    description = "Multi-service cascading timeout caused by memory leak in downstream service"

    def get_rubric(self) -> List[RubricCheck]:
        return [
            # ── root_cause ────────────────────────────────────────────────
            RubricCheck(
                name="Identified inventory-service memory/GC leak as root cause",
                check=lambda a, s: any(
                    "declare_root_cause" in act.lower()
                    and any(kw in act.lower() for kw in ["inventory", "memory", "gc", "heap", "garbage"])
                    for act in a
                ),
                dimension="root_cause",
                weight=0.80,
            ),
            RubricCheck(
                name="Did NOT blame the unrelated payment-service deployment",
                check=lambda a, s: not any(
                    "declare_root_cause" in act.lower() and "3.1.0" in act
                    for act in a
                ),
                dimension="root_cause",
                weight=0.20,
            ),

            # ── remediation ──────────────────────────────────────────────
            RubricCheck(
                name="Restarted inventory-service",
                check=lambda a, s: _action_matches(a, "restart_service", "inventory-service"),
                dimension="remediation",
                weight=0.45,
                gated=True,
                gate_service="inventory-service",
            ),
            RubricCheck(
                name="Updated jvm_heap_max to 4g on inventory-service",
                check=lambda a, s: (
                    _action_matches(a, "update_config", "inventory-service")
                    and str(_config_value(s, "inventory-service", "jvm_heap_max")) in ("4g", "4G", "4096m")
                ),
                dimension="remediation",
                weight=0.55,
                gated=True,
                gate_service="inventory-service",
            ),

            # ── investigation ────────────────────────────────────────────
            RubricCheck(
                name="Investigated inventory-service (the root cause service)",
                check=lambda a, s: _investigated(a, "inventory-service"),
                dimension="investigation",
                weight=0.45,
            ),
            RubricCheck(
                name="Traced cascade through order-service",
                check=lambda a, s: _investigated(a, "order-service"),
                dimension="investigation",
                weight=0.30,
            ),
            RubricCheck(
                name="Observed upstream payment-service symptoms",
                check=lambda a, s: _investigated(a, "payment-service"),
                dimension="investigation",
                weight=0.25,
            ),

            # ── safety ────────────────────────────────────────────────────
            RubricCheck(
                name="Did not restart healthy orders-db",
                check=lambda a, s: not _action_matches(a, "restart_service", "orders-db"),
                dimension="safety",
                weight=0.50,
            ),
            RubricCheck(
                name="Did not blame or restart payment-service (red herring deployment)",
                check=lambda a, s: not _action_matches(a, "rollback_deployment", "payment-service"),
                dimension="safety",
                weight=0.50,
            ),

            # ── sequence ──────────────────────────────────────────────────
            RubricCheck(
                name="Investigated inventory-service before restarting it",
                check=lambda a, s: (
                    next((i for i, x in enumerate(a) if "inventory-service" in x.lower()
                          and x.lower().split("(")[0] in {
                              "investigate_logs", "check_metrics", "run_diagnostic",
                              "read_config", "check_service_health"
                          }), 9999)
                    <
                    next((i for i, x in enumerate(a) if "restart_service(inventory-service)" in x.lower()), 9999)
                ),
                dimension="sequence",
                weight=0.60,
            ),
            RubricCheck(
                name="Followed cascade order: checked deeper services before shallower",
                check=lambda a, s: (
                    next((i for i, x in enumerate(a) if "inventory-service" in x.lower()), 9999)
                    <
                    next((i for i, x in enumerate(a) if "payment-service" in x.lower() and
                          x.lower().split("(")[0] in {"restart_service", "rollback_deployment"}), 9999)
                ),
                dimension="sequence",
                weight=0.40,
            ),
        ]


# ─────────────────────────────────────────────────────────────────────────────
# 3. HARD: Multi-Factor Outage
# ─────────────────────────────────────────────────────────────────────────────

class HardMultiFactorOutageScenario(BaseScenario):
    """
    Rubric for multi_factor_outage.

    Three root causes must all be found:
      1. api-gateway canary routing bug
      2. primary-db max_connections too low
      3. traffic spike awareness
    """
    task_id = "multi_factor_outage"
    difficulty = "hard"
    description = "Complex multi-factor outage: routing bug + connection pool exhaustion + traffic spike"

    def get_rubric(self) -> List[RubricCheck]:
        return [
            # ── root_cause ────────────────────────────────────────────────
            RubricCheck(
                name="Identified api-gateway canary routing bug",
                check=lambda a, s: any(
                    "declare_root_cause" in act.lower()
                    and any(kw in act.lower() for kw in ["canary", "routing", "gateway", "4.2.0", "misroute"])
                    for act in a
                ),
                dimension="root_cause",
                weight=0.38,
            ),
            RubricCheck(
                name="Identified primary-db connection exhaustion",
                check=lambda a, s: any(
                    "declare_root_cause" in act.lower()
                    and any(kw in act.lower() for kw in ["max_connections", "connection", "exhaust", "primary-db"])
                    for act in a
                ),
                dimension="root_cause",
                weight=0.38,
            ),
            RubricCheck(
                name="Identified traffic spike as contributing factor",
                check=lambda a, s: any(
                    "declare_root_cause" in act.lower()
                    and any(kw in act.lower() for kw in ["traffic", "spike", "campaign", "2x", "double"])
                    for act in a
                ),
                dimension="root_cause",
                weight=0.24,
            ),

            # ── remediation ──────────────────────────────────────────────
            RubricCheck(
                name="Rolled back api-gateway to v4.1.9",
                check=lambda a, s: _action_matches(a, "rollback_deployment", "api-gateway"),
                dimension="remediation",
                weight=0.40,
                gated=True,
                gate_service="api-gateway",
            ),
            RubricCheck(
                name="Increased primary-db max_connections",
                check=lambda a, s: (
                    _action_matches(a, "update_config", "primary-db")
                    and int(_config_value(s, "primary-db", "max_connections") or 0) >= 200
                ),
                dimension="remediation",
                weight=0.38,
                gated=True,
                gate_service="primary-db",
            ),
            RubricCheck(
                name="Scaled api-gateway replicas for traffic spike",
                check=lambda a, s: (
                    _action_matches(a, "scale_service", "api-gateway")
                    and s.get("api-gateway", {}).get("scaled_replicas", 1) >= 3
                ),
                dimension="remediation",
                weight=0.22,
                gated=True,
                gate_service="api-gateway",
            ),

            # ── investigation ────────────────────────────────────────────
            RubricCheck(
                name="Investigated api-gateway (routing bug source)",
                check=lambda a, s: _investigated(a, "api-gateway"),
                dimension="investigation",
                weight=0.30,
            ),
            RubricCheck(
                name="Investigated primary-db (connection pool source)",
                check=lambda a, s: _investigated(a, "primary-db"),
                dimension="investigation",
                weight=0.30,
            ),
            RubricCheck(
                name="Mapped blast radius (checked 3+ services)",
                check=lambda a, s: sum(
                    1 for svc in ["api-gateway", "auth-service", "product-service",
                                  "primary-db", "search-service", "redis-cache"]
                    if _investigated(a, svc)
                ) >= 3,
                dimension="investigation",
                weight=0.25,
            ),
            RubricCheck(
                name="Checked search-service for misrouted traffic evidence",
                check=lambda a, s: _investigated(a, "search-service"),
                dimension="investigation",
                weight=0.15,
            ),

            # ── safety ────────────────────────────────────────────────────
            RubricCheck(
                name="Did not blame or fix redis-cache (red herring)",
                check=lambda a, s: not (
                    _action_matches(a, "restart_service", "redis-cache")
                    or _action_matches(a, "update_config", "redis-cache")
                ),
                dimension="safety",
                weight=0.50,
            ),
            RubricCheck(
                name="Did not perform destructive actions on healthy services",
                check=lambda a, s: all(
                    not _action_matches(a, "restart_service", svc)
                    for svc in ["redis-cache"]
                ),
                dimension="safety",
                weight=0.50,
            ),

            # ── sequence ──────────────────────────────────────────────────
            RubricCheck(
                name="Rolled back api-gateway before scaling it",
                check=lambda a, s: (
                    next((i for i, x in enumerate(a) if "rollback_deployment(api-gateway)" in x.lower()), 9999)
                    <
                    next((i for i, x in enumerate(a) if "scale_service(api-gateway)" in x.lower()), 9999)
                ),
                dimension="sequence",
                weight=0.50,
            ),
            RubricCheck(
                name="Investigated api-gateway before rolling it back",
                check=lambda a, s: (
                    next((i for i, x in enumerate(a) if "api-gateway" in x.lower()
                          and x.lower().split("(")[0] in {
                              "investigate_logs", "read_config", "check_service_health", "run_diagnostic"
                          }), 9999)
                    <
                    next((i for i, x in enumerate(a) if "rollback_deployment(api-gateway)" in x.lower()), 9999)
                ),
                dimension="sequence",
                weight=0.50,
            ),
        ]


# ─────────────────────────────────────────────────────────────────────────────
# 4. MEDIUM: SSL Certificate Expiry
# ─────────────────────────────────────────────────────────────────────────────

class MediumSSLCertExpiryScenario(BaseScenario):
    """
    Rubric for ssl_certificate_expiry.

    Must identify expired cert, find renewed cert path, and apply config fix.
    Red herring: auth-service recent deployment.
    """
    task_id = "ssl_certificate_expiry"
    difficulty = "medium"
    description = "Platform-wide HTTPS failure from expired TLS certificate on api-gateway"

    def get_rubric(self) -> List[RubricCheck]:
        return [
            # ── root_cause ────────────────────────────────────────────────
            RubricCheck(
                name="Declared expired TLS certificate as root cause",
                check=lambda a, s: any(
                    "declare_root_cause" in act.lower()
                    and any(kw in act.lower() for kw in ["ssl", "cert", "tls", "expire", "2023"])
                    for act in a
                ),
                dimension="root_cause",
                weight=0.70,
            ),
            RubricCheck(
                name="Did NOT blame auth-service deployment (red herring)",
                check=lambda a, s: not any(
                    "declare_root_cause" in act.lower()
                    and "auth" in act.lower() and "4.1.0" in act
                    for act in a
                ),
                dimension="root_cause",
                weight=0.30,
            ),

            # ── remediation ──────────────────────────────────────────────
            RubricCheck(
                name="Updated ssl_cert_path to renewed cert on api-gateway",
                check=lambda a, s: (
                    _action_matches(a, "update_config", "api-gateway")
                    and "2025-renewed" in str(_config_value(s, "api-gateway", "ssl_cert_path") or "")
                ),
                dimension="remediation",
                weight=1.0,
                gated=True,
                gate_service="api-gateway",
            ),

            # ── investigation ────────────────────────────────────────────
            RubricCheck(
                name="Checked api-gateway config or diagnostics (found expired cert)",
                check=lambda a, s: (
                    _action_matches(a, "read_config", "api-gateway")
                    or _action_matches(a, "run_diagnostic", "api-gateway")
                    or _action_matches(a, "investigate_logs", "api-gateway")
                ),
                dimension="investigation",
                weight=0.50,
            ),
            RubricCheck(
                name="Checked cert-manager for renewal status",
                check=lambda a, s: _investigated(a, "cert-manager"),
                dimension="investigation",
                weight=0.35,
            ),
            RubricCheck(
                name="Observed downstream effects (auth-service, user-api)",
                check=lambda a, s: _investigated(a, "auth-service") or _investigated(a, "user-api"),
                dimension="investigation",
                weight=0.15,
            ),

            # ── safety ────────────────────────────────────────────────────
            RubricCheck(
                name="Did not restart auth-service (cascaded victim, not root cause)",
                check=lambda a, s: not _action_matches(a, "restart_service", "auth-service"),
                dimension="safety",
                weight=0.60,
            ),
            RubricCheck(
                name="Did not rollback auth-service deployment (false suspect)",
                check=lambda a, s: not _action_matches(a, "rollback_deployment", "auth-service"),
                dimension="safety",
                weight=0.40,
            ),

            # ── sequence ──────────────────────────────────────────────────
            RubricCheck(
                name="Inspected cert-manager or api-gateway before applying fix",
                check=lambda a, s: (
                    max(
                        next((i for i, x in enumerate(a) if "cert-manager" in x.lower()), -1),
                        next((i for i, x in enumerate(a) if "api-gateway" in x.lower()
                              and x.lower().split("(")[0] in {
                                  "investigate_logs", "read_config", "run_diagnostic",
                                  "check_service_health", "check_metrics"
                              }), -1),
                    ) < next((i for i, x in enumerate(a) if "update_config(api-gateway)" in x.lower()), 9999)
                    if any("update_config(api-gateway)" in x.lower() for x in a)
                    else True
                ),
                dimension="sequence",
                weight=1.0,
            ),
        ]


# ─────────────────────────────────────────────────────────────────────────────
# 5. HARD: Database Deadlock
# ─────────────────────────────────────────────────────────────────────────────

class HardDatabaseDeadlockScenario(BaseScenario):
    """
    Rubric for database_deadlock.

    Root cause: order-service v2.3.0 lock-order inversion → rollback.
    Red herrings: payment-service latency, user-service pool warning,
                  analytics-db contention.
    """
    task_id = "database_deadlock"
    difficulty = "hard"
    description = "Cascading deadlocks from lock-order inversion in order-service v2.3.0"

    def get_rubric(self) -> List[RubricCheck]:
        return [
            # ── root_cause ────────────────────────────────────────────────
            RubricCheck(
                name="Identified order-service v2.3.0 lock-order inversion as root cause",
                check=lambda a, s: any(
                    "declare_root_cause" in act.lower()
                    and any(kw in act.lower() for kw in
                            ["deadlock", "lock", "order-service", "2.3.0", "inversion", "v2.3.0"])
                    for act in a
                ),
                dimension="root_cause",
                weight=0.85,
            ),
            RubricCheck(
                name="Did NOT identify payment-service as root cause (red herring)",
                check=lambda a, s: not any(
                    "declare_root_cause" in act.lower() and "payment" in act.lower()
                    for act in a
                ),
                dimension="root_cause",
                weight=0.15,
            ),

            # ── remediation ──────────────────────────────────────────────
            RubricCheck(
                name="Rolled back order-service to v2.2.9",
                check=lambda a, s: (
                    _action_matches(a, "rollback_deployment", "order-service")
                    and s.get("order-service", {}).get("rolled_back", False)
                ),
                dimension="remediation",
                weight=1.0,
                gated=True,
                gate_service="order-service",
            ),

            # ── investigation ────────────────────────────────────────────
            RubricCheck(
                name="Investigated order-service (deployment and logs)",
                check=lambda a, s: (
                    _action_matches(a, "investigate_logs", "order-service")
                    or _action_matches(a, "run_diagnostic", "order-service")
                ),
                dimension="investigation",
                weight=0.35,
            ),
            RubricCheck(
                name="Checked primary-db deadlock evidence",
                check=lambda a, s: _investigated(a, "primary-db"),
                dimension="investigation",
                weight=0.35,
            ),
            RubricCheck(
                name="Checked user-service to understand deadlock partner",
                check=lambda a, s: _investigated(a, "user-service"),
                dimension="investigation",
                weight=0.20,
            ),
            RubricCheck(
                name="Assessed analytics-db to rule out or confirm amplifier",
                check=lambda a, s: _investigated(a, "analytics-db"),
                dimension="investigation",
                weight=0.10,
            ),

            # ── safety ────────────────────────────────────────────────────
            RubricCheck(
                name="Did not restart user-service (victim, not cause)",
                check=lambda a, s: not _action_matches(a, "restart_service", "user-service"),
                dimension="safety",
                weight=0.40,
            ),
            RubricCheck(
                name="Did not restart payment-service (downstream symptom)",
                check=lambda a, s: not _action_matches(a, "restart_service", "payment-service"),
                dimension="safety",
                weight=0.35,
            ),
            RubricCheck(
                name="Did not restart job-queue directly (symptom, not cause)",
                check=lambda a, s: not _action_matches(a, "restart_service", "job-queue"),
                dimension="safety",
                weight=0.25,
            ),

            # ── sequence ──────────────────────────────────────────────────
            RubricCheck(
                name="Investigated primary-db before rolling back order-service",
                check=lambda a, s: (
                    next((i for i, x in enumerate(a) if "primary-db" in x.lower()
                          and x.lower().split("(")[0] in {
                              "investigate_logs", "check_metrics", "run_diagnostic",
                              "check_service_health"
                          }), 9999)
                    <
                    next((i for i, x in enumerate(a) if "rollback_deployment(order-service)" in x.lower()), 9999)
                ),
                dimension="sequence",
                weight=0.55,
            ),
            RubricCheck(
                name="Declared root cause before rolling back",
                check=lambda a, s: (
                    next((i for i, x in enumerate(a) if "declare_root_cause" in x.lower()), 9999)
                    <
                    next((i for i, x in enumerate(a) if "rollback_deployment(order-service)" in x.lower()), 9999)
                ),
                dimension="sequence",
                weight=0.45,
            ),
        ]


# ─────────────────────────────────────────────────────────────────────────────
# Registry + Adapter
# ─────────────────────────────────────────────────────────────────────────────

SCENARIO_CLASS_REGISTRY: Dict[str, type] = {
    "db_connection_failure":     EasyDBConnectionFailureScenario,
    "cascading_service_timeout": MediumCascadingTimeoutScenario,
    "multi_factor_outage":       HardMultiFactorOutageScenario,
    "ssl_certificate_expiry":    MediumSSLCertExpiryScenario,
    "database_deadlock":         HardDatabaseDeadlockScenario,
}


class ScenarioRubricAdapter:
    """
    Convenience adapter that plugs the new rubric grading into the
    existing IncidentResponseEnv without any changes to env.py.

    Called from get_score_breakdown() when a scenario has a rubric class.
    Falls back silently if no rubric class is registered.
    """

    @staticmethod
    def has_rubric(task_name: str) -> bool:
        return task_name in SCENARIO_CLASS_REGISTRY

    @staticmethod
    def grade(
        task_name: str,
        actions: List[str],
        service_states: Dict[str, Any],
        step_number: int,
        max_steps: int,
        incident_resolved: bool,
        obs_loop: bool = False,
    ) -> Optional[float]:
        """
        Return rubric-based score, or None if no rubric class registered.
        """
        cls = SCENARIO_CLASS_REGISTRY.get(task_name)
        if cls is None:
            return None
        scenario_obj = cls()
        return scenario_obj.grade(
            actions=actions,
            service_states=service_states,
            step_number=step_number,
            max_steps=max_steps,
            incident_resolved=incident_resolved,
            obs_loop=obs_loop,
        )

    @staticmethod
    def grade_details(
        task_name: str,
        actions: List[str],
        service_states: Dict[str, Any],
        step_number: int,
        max_steps: int,
        incident_resolved: bool,
        obs_loop: bool = False,
    ) -> Optional[Dict[str, Any]]:
        """
        Return full rubric grade_details dict, or None if no rubric class.
        """
        cls = SCENARIO_CLASS_REGISTRY.get(task_name)
        if cls is None:
            return None
        scenario_obj = cls()
        return scenario_obj.grade_details(
            actions=actions,
            service_states=service_states,
            step_number=step_number,
            max_steps=max_steps,
            incident_resolved=incident_resolved,
            obs_loop=obs_loop,
        )
