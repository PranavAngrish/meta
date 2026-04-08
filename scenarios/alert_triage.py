"""
alert_triage.py — Alert Severity Classification scenarios for meta-hackathon.

Ported and EXTENDED from cloud-incident-response (Einstein & Sidra).

Three scenarios testing a fundamentally different capability:
  AT-001 — Clear P1: 84% errors, $14k/min revenue loss (trivial correct answer)
  AT-002 — Ambiguous P2: degraded but functional, $600/min (tests precision)
  AT-003 — Trap P3: 52% ML service errors, zero revenue impact (tests not over-classifying)

The agent gets max 3 steps to classify severity P1/P2/P3/P4.
Scoring: 1.0 exact, 0.5 adjacent, 0.25 two-off, 0.0 wrong.

This adds a FUNDAMENTALLY DIFFERENT task type missing from meta-hackathon:
  - Pure signal reading (no remediation needed)
  - Forces the agent to distinguish blast radius from error rate
  - Tests calibration: high error rate ≠ P1 if there's graceful fallback
"""

from __future__ import annotations

# ── Scenario definitions ──────────────────────────────────────────────────

ALERT_TRIAGE_SCENARIOS: list[dict] = [

    # ── AT-001: Clear P1 ─────────────────────────────────────────────────
    # DB connection pool exhaustion → auth down → checkout unavailable
    # 84% errors, $14k/min revenue loss → unambiguous P1
    {
        "scenario_id": "AT-001",
        "description": "Cascading failure across checkout pipeline. Assess severity from metrics.",
        "incident_summary": (
            "Alert fired: api-gateway error rate elevated. Multiple downstream services "
            "affected. Checkout appears unavailable. Assess the incident severity."
        ),
        "alert": {
            "id": "ALT-20250408-001",
            "title": "api-gateway error rate critical",
            "severity_fired": "UNCLASSIFIED",
            "affected_services": ["api-gateway", "auth-service", "postgres-primary"],
            "symptoms": [
                "api-gateway: HTTP 503 rate 84% (baseline: 0.1%)",
                "auth-service: connection timeout 97% of requests",
                "postgres-primary: connection pool 500/500 utilized",
                "checkout flow: unavailable",
                "user logins: failing",
            ],
            "error_rate": 0.84,
            "duration_minutes": 6,
            "revenue_impact_per_min": 14000,
        },
        "known_services": ["api-gateway", "auth-service", "postgres-primary"],
        "tool_responses": {
            "investigate_logs": {
                "api-gateway": (
                    "2025-04-08T10:04:12Z ERROR upstream timeout auth-service:8080\n"
                    "2025-04-08T10:04:13Z ERROR 503 Service Unavailable\n"
                    "2025-04-08T10:04:14Z ERROR circuit breaker OPEN — all downstream routes failing"
                ),
                "auth-service": (
                    "2025-04-08T10:04:10Z ERROR too many clients already (pool 500/500)\n"
                    "2025-04-08T10:04:11Z ERROR connection pool exhausted — rejecting new auth requests"
                ),
                "postgres-primary": (
                    "2025-04-08T10:04:00Z FATAL remaining slots reserved for superuser connections\n"
                    "2025-04-08T10:04:01Z LOG max_connections=500 active=500\n"
                    "2025-04-08T10:04:05Z ERROR new connections rejected — pool full"
                ),
            },
            "check_metrics": {
                "api-gateway": "5xx rate: 84% | p99: 30s | circuit_breaker: OPEN | RPS: 1200",
                "auth-service": "Error rate: 97% | DB wait: 28s | Queue depth: 1100",
                "postgres-primary": "Connections: 500/500 (100%) | CPU: 98% | Memory: 91%",
            },
            "check_service_health": {
                "api-gateway": "Status: UNHEALTHY | Error: 84% | Latency p99: 30000ms",
                "auth-service": "Status: UNHEALTHY | DB pool: exhausted",
                "postgres-primary": "Status: UNHEALTHY | Connections: maxed",
            },
            "run_diagnostic": {
                "api-gateway": "DIAGNOSIS: Circuit breaker OPEN. 84% of requests failing with 503. Upstream auth-service unreachable.",
                "auth-service": "DIAGNOSIS: Database connection pool exhausted (500/500). All auth requests failing.",
                "postgres-primary": "DIAGNOSIS: max_connections reached. Revenue-critical auth path blocked.",
            },
        },
        "correct_severity": "P1",
        "adjacent_severities": ["P2"],
        "severity_reasoning": (
            "P1: checkout unavailable + $14k/min revenue loss + 84% error rate across authentication. "
            "This is a complete outage of a revenue-critical user-facing flow."
        ),
    },

    # ── AT-002: Ambiguous P2 ─────────────────────────────────────────────
    # CDN cache invalidation storm → origin overloaded → pages slow
    # Core checkout still works, $600/min impact → P2 not P1
    {
        "scenario_id": "AT-002",
        "description": "Service degradation affecting page load times. Core transactions still operational.",
        "incident_summary": (
            "Alert fired: CDN cache performance severely degraded. "
            "Origin servers experiencing heavy load. Assess severity carefully."
        ),
        "alert": {
            "id": "ALT-20250408-002",
            "title": "CDN cache hit rate collapsed",
            "severity_fired": "UNCLASSIFIED",
            "affected_services": ["cdn-edge", "product-service", "image-service"],
            "symptoms": [
                "CDN cache hit rate: 2% (normal: 94%)",
                "product-service: CPU 95%, p99 latency 22s",
                "image-service: worker pool exhausted",
                "Product pages: loading extremely slowly",
                "Checkout: still functional (cart/payment unaffected)",
            ],
            "error_rate": 0.18,
            "duration_minutes": 12,
            "revenue_impact_per_min": 600,
        },
        "known_services": ["cdn-edge", "product-service", "image-service"],
        "tool_responses": {
            "investigate_logs": {
                "cdn-edge": (
                    "2025-04-08T10:22:00Z INFO cache MISS ratio: 98%\n"
                    "2025-04-08T10:20:11Z WARN mass cache purge — 2.4M keys invalidated\n"
                    "2025-04-08T10:20:10Z INFO purge pattern: /* (ALL keys) triggered by cronjob"
                ),
                "product-service": (
                    "2025-04-08T10:22:05Z WARN request queue depth: 15000\n"
                    "2025-04-08T10:22:06Z ERROR timeout retrieving images from image-service\n"
                    "2025-04-08T10:22:07Z WARN worker pool 95% utilized"
                ),
                "image-service": (
                    "2025-04-08T10:22:00Z WARN CPU throttled to 95%\n"
                    "2025-04-08T10:22:01Z ERROR worker pool exhausted — dropping image resize jobs\n"
                    "2025-04-08T10:22:02Z WARN memory at 93%"
                ),
            },
            "check_metrics": {
                "cdn-edge": "Cache hit: 2% | Origin RPS: 52000 (norm 1400) | Bandwidth: 940 Gbps",
                "product-service": "Origin RPS: 52k | Queue: 15,000 | CPU: 95% | p99: 22s",
                "image-service": "CPU: 95% | Memory: 93% | p99: 22s | Workers: 100%",
            },
            "check_service_health": {
                "cdn-edge": "Status: DEGRADED | Cache: 2% hit | Origin: overwhelmed",
                "product-service": "Status: DEGRADED | Not down, but slow. Checkout works.",
                "image-service": "Status: DEGRADED | Images slow, not failing entirely",
            },
            "run_diagnostic": {
                "cdn-edge": "DIAGNOSIS: All cache keys purged 12min ago. Origin traffic 37x normal. Pages loading slowly.",
                "product-service": "DIAGNOSIS: Overloaded by cache misses. Checkout pipeline unaffected.",
                "image-service": "DIAGNOSIS: Overloaded by image requests. No auth/checkout dependency.",
            },
        },
        "correct_severity": "P2",
        "adjacent_severities": ["P1", "P3"],
        "severity_reasoning": (
            "P2: Major degradation (22s page loads, most users affected) but checkout still works. "
            "$600/min revenue impact is significant but below P1 threshold ($1k/min). "
            "Not P1 because core transaction flow is functional."
        ),
    },

    # ── AT-003: Trap P3 ─────────────────────────────────────────────────
    # ML recommendation service errors → graceful fallback → zero user impact
    # 52% error rate but $0 revenue impact → P3 (trap: don't classify as P1/P2)
    {
        "scenario_id": "AT-003",
        "description": "Internal ML service reporting elevated errors. Determine actual user impact.",
        "incident_summary": (
            "Alert fired: recommendation-service error rate elevated to 52%. "
            "product-service is downstream. Assess severity based on actual impact."
        ),
        "alert": {
            "id": "ALT-20250408-003",
            "title": "recommendation-service error rate 52%",
            "severity_fired": "UNCLASSIFIED",
            "affected_services": ["recommendation-service", "product-service"],
            "symptoms": [
                "recommendation-service: error rate 52% (baseline: 1.5%)",
                "product-service: using fallback (default) recommendation logic",
                "User experience: showing default product recommendations",
                "Checkout: fully functional",
                "Revenue: no measurable change",
            ],
            "error_rate": 0.52,
            "duration_minutes": 28,
            "revenue_impact_per_min": 0,
        },
        "known_services": ["recommendation-service", "product-service", "redis-reco-cache"],
        "tool_responses": {
            "investigate_logs": {
                "recommendation-service": (
                    "2025-04-08T09:48:00Z ERROR ML model inference timeout (>5s)\n"
                    "2025-04-08T09:48:01Z WARN ML model server v2.4 overloaded\n"
                    "2025-04-08T09:48:02Z INFO fallback activated: returning popular items list"
                ),
                "product-service": (
                    "2025-04-08T09:48:05Z INFO recommendation-service returned fallback defaults\n"
                    "2025-04-08T09:48:06Z INFO serving page with default recs — checkout unaffected\n"
                    "2025-04-08T09:48:10Z INFO checkout pipeline: 100% success rate"
                ),
                "redis-reco-cache": (
                    "2025-04-08T09:48:00Z INFO cache hit rate: 91% — operating normally\n"
                    "2025-04-08T09:48:01Z INFO no errors — all GET/SET successful"
                ),
            },
            "check_metrics": {
                "recommendation-service": (
                    "Error rate: 52% | Fallback rate: 52% | User impact: NONE (graceful degradation) | "
                    "Inference timeout: 5s (model server overloaded)"
                ),
                "product-service": (
                    "Error rate: 0.08% (normal) | Checkout: 100% | Revenue: unchanged | "
                    "Reco mode: FALLBACK (default items)"
                ),
                "redis-reco-cache": "Hit rate: 91% | Memory: 28% | HEALTHY | No errors",
            },
            "check_service_health": {
                "recommendation-service": "Status: DEGRADED | ML model slow | Fallback: ACTIVE",
                "product-service": "Status: HEALTHY | Checkout: 100% | Using reco fallback",
                "redis-reco-cache": "Status: HEALTHY | 91% hit rate",
            },
            "run_diagnostic": {
                "recommendation-service": (
                    "DIAGNOSIS: ML model v2.4 inference exceeding 5s timeout. "
                    "Graceful fallback active — returning popular items instead. "
                    "No checkout or revenue impact. Root cause: model update 4h ago."
                ),
                "product-service": (
                    "DIAGNOSIS: Operating normally with fallback recommendations. "
                    "Checkout flow: 100% success. Revenue: unchanged. Not a P1 or P2."
                ),
                "redis-reco-cache": "DIAGNOSIS: HEALTHY. No issues.",
            },
        },
        "correct_severity": "P3",
        "adjacent_severities": ["P2", "P4"],
        "severity_reasoning": (
            "P3: Minor/partial issue. 52% error rate sounds alarming but graceful fallback "
            "means zero user-visible impact beyond slightly less personalized recommendations. "
            "Zero revenue impact. Checkout/login unaffected. Not P1 or P2."
        ),
    },
]


def get_alert_triage_scenario(index: int) -> dict:
    if index < 0 or index >= len(ALERT_TRIAGE_SCENARIOS):
        raise ValueError(f"AT scenario index {index} out of range (0-{len(ALERT_TRIAGE_SCENARIOS)-1})")
    return ALERT_TRIAGE_SCENARIOS[index]


def list_alert_triage_scenarios() -> list[dict]:
    return [
        {"scenario_id": s["scenario_id"], "correct_severity": s["correct_severity"],
         "description": s["description"]}
        for s in ALERT_TRIAGE_SCENARIOS
    ]
