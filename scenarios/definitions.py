"""
Scenario definitions for the Incident Response environment.

Each scenario defines:
- A set of services with their states, logs, metrics, configs
- Root causes the agent must identify
- Correct remediation actions
- Grading rubric with partial credit

Three difficulty levels:
  easy   — Single service, single root cause, obvious symptoms
  medium — Multi-service cascade, requires tracing through dependencies
  hard   — Multiple simultaneous root causes, red herrings, time pressure
"""

from __future__ import annotations
import copy
import random
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field


@dataclass
class ServiceDef:
    """Definition of a simulated service."""
    name: str
    healthy: bool
    response_time_ms: float
    error_rate: float
    cpu_percent: float
    memory_percent: float
    connections_active: int
    connections_max: int
    uptime_seconds: int
    version: str
    previous_version: str
    config: Dict[str, Any]
    correct_config: Dict[str, Any]  # what config should be
    logs: List[Dict[str, str]]  # list of log entries
    metrics_history: List[Dict[str, float]]  # recent metric snapshots
    dependencies: List[str]
    deployment_status: str  # "stable", "recently_deployed", "rolling_back"
    diagnostic_output: str


@dataclass
class ScenarioDef:
    """Complete scenario definition."""
    task_name: str
    difficulty: str
    incident_summary: str
    services: Dict[str, ServiceDef]
    alerts: List[Dict[str, str]]
    root_causes: List[str]
    root_cause_keywords: List[List[str]]  # keywords for each root cause (for grading)
    correct_remediations: List[Dict[str, Any]]  # action_type + params to fix
    remediation_keywords: List[List[str]]
    max_steps: int
    investigation_hints: Dict[str, float]  # service_name -> reward for investigating
    red_herrings: List[str] = field(default_factory=list)


def build_easy_scenario() -> ScenarioDef:
    """
    EASY: Database Connection Failure
    
    The 'user-api' service cannot connect to 'postgres-primary' because
    the database port in the config was changed from 5432 to 5433 during 
    a recent maintenance window. Single root cause, single service affected.
    """
    services = {
        "user-api": ServiceDef(
            name="user-api",
            healthy=False,
            response_time_ms=12000.0,
            error_rate=0.85,
            cpu_percent=12.0,
            memory_percent=35.0,
            connections_active=0,
            connections_max=100,
            uptime_seconds=3600,
            version="2.4.1",
            previous_version="2.4.0",
            config={
                "db_host": "postgres-primary",
                "db_port": 5433,  # WRONG - should be 5432
                "db_name": "users",
                "db_pool_size": 20,
                "db_timeout_ms": 5000,
                "log_level": "info",
                "cache_ttl_seconds": 300,
            },
            correct_config={
                "db_host": "postgres-primary",
                "db_port": 5432,  # CORRECT
                "db_name": "users",
                "db_pool_size": 20,
                "db_timeout_ms": 5000,
                "log_level": "info",
                "cache_ttl_seconds": 300,
            },
            logs=[
                {"timestamp": "2025-04-08T02:15:01Z", "level": "INFO", "message": "Service started, version 2.4.1"},
                {"timestamp": "2025-04-08T02:15:02Z", "level": "INFO", "message": "Attempting database connection to postgres-primary:5433"},
                {"timestamp": "2025-04-08T02:15:07Z", "level": "ERROR", "message": "Connection refused: postgres-primary:5433 - ECONNREFUSED"},
                {"timestamp": "2025-04-08T02:15:08Z", "level": "ERROR", "message": "Database connection pool failed to initialize"},
                {"timestamp": "2025-04-08T02:15:08Z", "level": "WARN", "message": "Retrying database connection (attempt 2/5)"},
                {"timestamp": "2025-04-08T02:15:13Z", "level": "ERROR", "message": "Connection refused: postgres-primary:5433 - ECONNREFUSED"},
                {"timestamp": "2025-04-08T02:15:18Z", "level": "ERROR", "message": "Connection refused: postgres-primary:5433 - ECONNREFUSED"},
                {"timestamp": "2025-04-08T02:15:23Z", "level": "ERROR", "message": "Connection refused: postgres-primary:5433 - ECONNREFUSED"},
                {"timestamp": "2025-04-08T02:15:28Z", "level": "ERROR", "message": "Connection refused: postgres-primary:5433 - ECONNREFUSED"},
                {"timestamp": "2025-04-08T02:15:28Z", "level": "CRITICAL", "message": "All 5 connection attempts failed. Service degraded - returning 503 for all DB-dependent routes"},
                {"timestamp": "2025-04-08T02:16:00Z", "level": "ERROR", "message": "GET /api/users/123 -> 503 Service Unavailable (no database connection)"},
                {"timestamp": "2025-04-08T02:16:05Z", "level": "ERROR", "message": "POST /api/users/login -> 503 Service Unavailable (no database connection)"},
            ],
            metrics_history=[
                {"timestamp_min": 0, "cpu": 12.0, "memory": 35.0, "req_per_sec": 150.0, "error_rate": 0.85, "p99_latency_ms": 12000.0},
                {"timestamp_min": -5, "cpu": 10.0, "memory": 34.0, "req_per_sec": 200.0, "error_rate": 0.80, "p99_latency_ms": 11000.0},
                {"timestamp_min": -10, "cpu": 5.0, "memory": 30.0, "req_per_sec": 250.0, "error_rate": 0.0, "p99_latency_ms": 45.0},
            ],
            dependencies=["postgres-primary"],
            deployment_status="stable",
            diagnostic_output="TCP connection test to postgres-primary:5433 FAILED (Connection refused)\nTCP connection test to postgres-primary:5432 SUCCESS (Connected in 2ms)\nDNS resolution for postgres-primary: 10.0.1.50 OK",
        ),
        "postgres-primary": ServiceDef(
            name="postgres-primary",
            healthy=True,
            response_time_ms=5.0,
            error_rate=0.0,
            cpu_percent=25.0,
            memory_percent=60.0,
            connections_active=45,
            connections_max=200,
            uptime_seconds=864000,
            version="15.4",
            previous_version="15.4",
            config={
                "port": 5432,
                "max_connections": 200,
                "shared_buffers": "2GB",
                "work_mem": "256MB",
                "listen_addresses": "*",
            },
            correct_config={
                "port": 5432,
                "max_connections": 200,
                "shared_buffers": "2GB",
                "work_mem": "256MB",
                "listen_addresses": "*",
            },
            logs=[
                {"timestamp": "2025-04-08T02:10:00Z", "level": "INFO", "message": "PostgreSQL 15.4 running on port 5432"},
                {"timestamp": "2025-04-08T02:15:00Z", "level": "INFO", "message": "Checkpoint complete: wrote 128 buffers, 0.5%"},
                {"timestamp": "2025-04-08T02:16:00Z", "level": "INFO", "message": "Active connections: 45/200"},
            ],
            metrics_history=[
                {"timestamp_min": 0, "cpu": 25.0, "memory": 60.0, "connections": 45, "queries_per_sec": 850.0, "replication_lag_ms": 0},
            ],
            dependencies=[],
            deployment_status="stable",
            diagnostic_output="PostgreSQL accepting connections on port 5432\nReplication status: streaming, lag 0ms\nDisk usage: 45% of 500GB\nAll tablespaces healthy",
        ),
        "nginx-lb": ServiceDef(
            name="nginx-lb",
            healthy=True,
            response_time_ms=2.0,
            error_rate=0.42,
            cpu_percent=8.0,
            memory_percent=15.0,
            connections_active=500,
            connections_max=10000,
            uptime_seconds=2592000,
            version="1.25.3",
            previous_version="1.25.3",
            config={
                "worker_processes": 4,
                "worker_connections": 10000,
                "upstream_user_api": "http://user-api:8080",
                "keepalive_timeout": 65,
            },
            correct_config={
                "worker_processes": 4,
                "worker_connections": 10000,
                "upstream_user_api": "http://user-api:8080",
                "keepalive_timeout": 65,
            },
            logs=[
                {"timestamp": "2025-04-08T02:16:00Z", "level": "WARN", "message": "upstream user-api returning 503 for 85% of requests"},
                {"timestamp": "2025-04-08T02:16:30Z", "level": "INFO", "message": "Health check: user-api UNHEALTHY"},
            ],
            metrics_history=[
                {"timestamp_min": 0, "cpu": 8.0, "memory": 15.0, "req_per_sec": 500.0, "upstream_5xx_rate": 0.42},
            ],
            dependencies=["user-api"],
            deployment_status="stable",
            diagnostic_output="Nginx running normally\nUpstream user-api: 1/1 backends UNHEALTHY\nAll other upstreams: healthy",
        ),
    }

    return ScenarioDef(
        task_name="db_connection_failure",
        difficulty="easy",
        incident_summary="INCIDENT: user-api returning 503 errors for all authenticated endpoints. Users unable to log in or access profiles. Started ~10 minutes ago. On-call alert triggered by error rate spike.",
        services=services,
        alerts=[
            {"alert_id": "ALT-001", "severity": "critical", "service": "user-api", "message": "Error rate >80% for user-api", "timestamp": "2025-04-08T02:16:00Z"},
            {"alert_id": "ALT-002", "severity": "high", "service": "nginx-lb", "message": "Upstream 5xx rate elevated", "timestamp": "2025-04-08T02:16:30Z"},
        ],
        root_causes=["user-api database port misconfiguration: db_port set to 5433 instead of 5432"],
        root_cause_keywords=[["port", "5433", "5432", "db_port", "misconfigur"]],
        correct_remediations=[
            {"action_type": "update_config", "service_name": "user-api", "parameters": {"key": "db_port", "value": "5432"}},
        ],
        remediation_keywords=[["update_config", "db_port", "5432"]],
        max_steps=20,
        investigation_hints={
            "user-api": 0.08,
            "postgres-primary": 0.06,
            "nginx-lb": 0.03,
        },
    )


def build_medium_scenario() -> ScenarioDef:
    """
    MEDIUM: Cascading Service Timeout
    
    payment-service -> order-service -> inventory-service
    inventory-service has a memory leak causing GC pauses and slow responses.
    order-service times out waiting for inventory-service.
    payment-service times out waiting for order-service.
    The agent must trace the cascade back to inventory-service's memory leak.
    Red herring: recent deployment to payment-service (unrelated).
    """
    services = {
        "payment-service": ServiceDef(
            name="payment-service",
            healthy=False,
            response_time_ms=30000.0,
            error_rate=0.65,
            cpu_percent=18.0,
            memory_percent=40.0,
            connections_active=85,
            connections_max=200,
            uptime_seconds=7200,
            version="3.1.0",
            previous_version="3.0.9",
            config={
                "order_service_url": "http://order-service:8081",
                "request_timeout_ms": 30000,
                "retry_count": 3,
                "circuit_breaker_threshold": 5,
                "stripe_api_version": "2024-12-18",
            },
            correct_config={
                "order_service_url": "http://order-service:8081",
                "request_timeout_ms": 30000,
                "retry_count": 3,
                "circuit_breaker_threshold": 5,
                "stripe_api_version": "2024-12-18",
            },
            logs=[
                {"timestamp": "2025-04-08T03:00:00Z", "level": "INFO", "message": "Deployed version 3.1.0 (added Apple Pay support)"},
                {"timestamp": "2025-04-08T03:30:00Z", "level": "WARN", "message": "Timeout calling order-service: POST /api/orders/create (30000ms exceeded)"},
                {"timestamp": "2025-04-08T03:30:05Z", "level": "ERROR", "message": "Payment processing failed: upstream timeout from order-service"},
                {"timestamp": "2025-04-08T03:30:10Z", "level": "WARN", "message": "Circuit breaker OPEN for order-service (5 consecutive failures)"},
                {"timestamp": "2025-04-08T03:30:15Z", "level": "ERROR", "message": "POST /api/payments/charge -> 504 Gateway Timeout"},
                {"timestamp": "2025-04-08T03:31:00Z", "level": "WARN", "message": "Timeout calling order-service: POST /api/orders/validate (30000ms exceeded)"},
                {"timestamp": "2025-04-08T03:31:30Z", "level": "INFO", "message": "Circuit breaker HALF-OPEN, testing order-service"},
                {"timestamp": "2025-04-08T03:31:35Z", "level": "ERROR", "message": "Circuit breaker test FAILED, returning to OPEN state"},
            ],
            metrics_history=[
                {"timestamp_min": 0, "cpu": 18.0, "memory": 40.0, "req_per_sec": 30.0, "error_rate": 0.65, "p99_latency_ms": 30000.0},
                {"timestamp_min": -15, "cpu": 15.0, "memory": 38.0, "req_per_sec": 80.0, "error_rate": 0.40, "p99_latency_ms": 25000.0},
                {"timestamp_min": -30, "cpu": 10.0, "memory": 35.0, "req_per_sec": 120.0, "error_rate": 0.0, "p99_latency_ms": 200.0},
            ],
            dependencies=["order-service"],
            deployment_status="recently_deployed",
            diagnostic_output="Health check: DEGRADED (upstream dependency failure)\nRecent deployment: v3.1.0 deployed 90 min ago (Apple Pay feature)\nDiff from v3.0.9: +ApplePayProvider class, no changes to order-service integration\nStripe API connection: OK\nLocal processing: OK (payment logic healthy)",
        ),
        "order-service": ServiceDef(
            name="order-service",
            healthy=False,
            response_time_ms=30000.0,
            error_rate=0.70,
            cpu_percent=22.0,
            memory_percent=45.0,
            connections_active=150,
            connections_max=200,
            uptime_seconds=259200,
            version="2.8.3",
            previous_version="2.8.3",
            config={
                "inventory_service_url": "http://inventory-service:8082",
                "request_timeout_ms": 25000,
                "db_host": "orders-db",
                "db_port": 5432,
                "retry_count": 2,
            },
            correct_config={
                "inventory_service_url": "http://inventory-service:8082",
                "request_timeout_ms": 25000,
                "db_host": "orders-db",
                "db_port": 5432,
                "retry_count": 2,
            },
            logs=[
                {"timestamp": "2025-04-08T03:25:00Z", "level": "WARN", "message": "Slow response from inventory-service: GET /api/inventory/check took 24500ms"},
                {"timestamp": "2025-04-08T03:26:00Z", "level": "ERROR", "message": "Timeout calling inventory-service: GET /api/inventory/reserve (25000ms exceeded)"},
                {"timestamp": "2025-04-08T03:27:00Z", "level": "ERROR", "message": "Failed to process order ORD-8821: inventory check timeout"},
                {"timestamp": "2025-04-08T03:28:00Z", "level": "WARN", "message": "Thread pool exhaustion approaching: 148/200 threads busy waiting on inventory-service"},
                {"timestamp": "2025-04-08T03:29:00Z", "level": "ERROR", "message": "Timeout calling inventory-service: POST /api/inventory/reserve (25000ms exceeded)"},
                {"timestamp": "2025-04-08T03:30:00Z", "level": "CRITICAL", "message": "Connection pool saturated: all threads blocked on inventory-service calls"},
            ],
            metrics_history=[
                {"timestamp_min": 0, "cpu": 22.0, "memory": 45.0, "req_per_sec": 15.0, "error_rate": 0.70, "p99_latency_ms": 30000.0},
                {"timestamp_min": -15, "cpu": 18.0, "memory": 42.0, "req_per_sec": 40.0, "error_rate": 0.45, "p99_latency_ms": 25000.0},
                {"timestamp_min": -45, "cpu": 10.0, "memory": 38.0, "req_per_sec": 100.0, "error_rate": 0.0, "p99_latency_ms": 150.0},
            ],
            dependencies=["inventory-service", "orders-db"],
            deployment_status="stable",
            diagnostic_output="Order DB connection: OK (45ms avg query time)\nInventory-service calls: FAILING (timeout after 25000ms)\nThread pool: 150/200 active (threads blocked on downstream)\nNo recent deployments",
        ),
        "inventory-service": ServiceDef(
            name="inventory-service",
            healthy=False,
            response_time_ms=25000.0,
            error_rate=0.30,
            cpu_percent=95.0,
            memory_percent=97.0,
            connections_active=180,
            connections_max=200,
            uptime_seconds=604800,
            version="1.5.2",
            previous_version="1.5.2",
            config={
                "db_host": "inventory-db",
                "db_port": 5432,
                "jvm_heap_max": "2g",
                "gc_type": "G1GC",
                "cache_provider": "redis",
                "cache_host": "redis-cluster",
            },
            correct_config={
                "db_host": "inventory-db",
                "db_port": 5432,
                "jvm_heap_max": "4g",  # needs increase
                "gc_type": "G1GC",
                "cache_provider": "redis",
                "cache_host": "redis-cluster",
            },
            logs=[
                {"timestamp": "2025-04-08T02:00:00Z", "level": "WARN", "message": "GC pause: 2500ms (G1 Young Generation)"},
                {"timestamp": "2025-04-08T02:30:00Z", "level": "WARN", "message": "GC pause: 4200ms (G1 Mixed Collection)"},
                {"timestamp": "2025-04-08T02:45:00Z", "level": "WARN", "message": "Heap usage at 85% (1.7GB/2.0GB)"},
                {"timestamp": "2025-04-08T03:00:00Z", "level": "ERROR", "message": "GC pause: 8500ms (Full GC) - heap pressure critical"},
                {"timestamp": "2025-04-08T03:10:00Z", "level": "ERROR", "message": "GC pause: 12000ms (Full GC) - application threads stopped"},
                {"timestamp": "2025-04-08T03:15:00Z", "level": "CRITICAL", "message": "Heap usage at 96% (1.92GB/2.0GB) - OutOfMemoryError imminent"},
                {"timestamp": "2025-04-08T03:20:00Z", "level": "ERROR", "message": "GC pause: 18000ms (Full GC) - response times severely degraded"},
                {"timestamp": "2025-04-08T03:25:00Z", "level": "ERROR", "message": "Request processing delayed: inventory check took 22000ms due to GC"},
                {"timestamp": "2025-04-08T03:28:00Z", "level": "WARN", "message": "Redis cache miss rate elevated: 78% (possible memory pressure evictions)"},
            ],
            metrics_history=[
                {"timestamp_min": 0, "cpu": 95.0, "memory": 97.0, "gc_pause_ms": 18000, "heap_used_gb": 1.94, "heap_max_gb": 2.0, "req_per_sec": 5.0},
                {"timestamp_min": -30, "cpu": 80.0, "memory": 90.0, "gc_pause_ms": 8500, "heap_used_gb": 1.8, "heap_max_gb": 2.0, "req_per_sec": 30.0},
                {"timestamp_min": -60, "cpu": 40.0, "memory": 65.0, "gc_pause_ms": 200, "heap_used_gb": 1.3, "heap_max_gb": 2.0, "req_per_sec": 100.0},
                {"timestamp_min": -120, "cpu": 25.0, "memory": 50.0, "gc_pause_ms": 50, "heap_used_gb": 1.0, "heap_max_gb": 2.0, "req_per_sec": 100.0},
            ],
            dependencies=["inventory-db", "redis-cluster"],
            deployment_status="stable",
            diagnostic_output="JVM heap: 1.94GB / 2.0GB (97% used) - CRITICAL\nGC activity: Full GC every 5-10 minutes, pauses up to 18 seconds\nThread dump: 160/200 threads in TIMED_WAITING (blocked during GC)\nMemory leak suspected: object count growing linearly since last restart (7 days ago)\nRecommendation: increase jvm_heap_max or restart service to reclaim memory",
        ),
        "orders-db": ServiceDef(
            name="orders-db",
            healthy=True,
            response_time_ms=8.0,
            error_rate=0.0,
            cpu_percent=15.0,
            memory_percent=55.0,
            connections_active=30,
            connections_max=200,
            uptime_seconds=2592000,
            version="15.4",
            previous_version="15.4",
            config={"port": 5432, "max_connections": 200},
            correct_config={"port": 5432, "max_connections": 200},
            logs=[
                {"timestamp": "2025-04-08T03:00:00Z", "level": "INFO", "message": "Active connections: 30/200"},
            ],
            metrics_history=[
                {"timestamp_min": 0, "cpu": 15.0, "memory": 55.0, "connections": 30, "queries_per_sec": 200.0},
            ],
            dependencies=[],
            deployment_status="stable",
            diagnostic_output="PostgreSQL healthy, accepting connections on port 5432",
        ),
    }

    return ScenarioDef(
        task_name="cascading_service_timeout",
        difficulty="medium",
        incident_summary="INCIDENT: Payment processing failing with 504 timeouts. 65% error rate on payment-service. Customers unable to complete purchases. payment-service was recently deployed (v3.1.0) ~90 min ago. Investigating if deployment is the cause.",
        services=services,
        alerts=[
            {"alert_id": "ALT-101", "severity": "critical", "service": "payment-service", "message": "Payment failure rate >60%", "timestamp": "2025-04-08T03:30:00Z"},
            {"alert_id": "ALT-102", "severity": "high", "service": "order-service", "message": "Thread pool near exhaustion", "timestamp": "2025-04-08T03:28:00Z"},
            {"alert_id": "ALT-103", "severity": "high", "service": "inventory-service", "message": "Memory usage >95%", "timestamp": "2025-04-08T03:15:00Z"},
        ],
        root_causes=["inventory-service memory leak causing GC pauses up to 18 seconds, blocking all request processing and causing upstream timeout cascade"],
        root_cause_keywords=[["inventory", "memory", "leak", "gc", "heap", "garbage"]],
        correct_remediations=[
            {"action_type": "restart_service", "service_name": "inventory-service", "parameters": {}},
            {"action_type": "update_config", "service_name": "inventory-service", "parameters": {"key": "jvm_heap_max", "value": "4g"}},
        ],
        remediation_keywords=[
            ["restart", "inventory"],
            ["update_config", "jvm_heap_max", "4g", "heap"],
        ],
        max_steps=25,
        investigation_hints={
            "inventory-service": 0.08,
            "order-service": 0.05,
            "payment-service": 0.03,
            "orders-db": 0.02,
        },
        red_herrings=["payment-service recent deployment v3.1.0 (Apple Pay feature — unrelated to timeout issue)"],
    )


def build_hard_scenario() -> ScenarioDef:
    """
    HARD: Multi-Factor Production Outage

    Three simultaneous issues:
    1. A bad deployment to api-gateway introduced a routing bug (sends 20% of traffic to wrong backend)
    2. Database connection pool exhaustion on primary-db (max_connections too low for current load)
    3. A marketing campaign spike doubled traffic to the frontend
    
    Red herrings:
    - CDN cache miss rate is high (normal for new campaign landing page)
    - Redis shows elevated latency (consequence, not cause)
    
    Agent must identify ALL THREE contributing factors and apply correct remediations.
    """
    services = {
        "api-gateway": ServiceDef(
            name="api-gateway",
            healthy=False,
            response_time_ms=5500.0,
            error_rate=0.45,
            cpu_percent=70.0,
            memory_percent=55.0,
            connections_active=2000,
            connections_max=5000,
            uptime_seconds=1800,
            version="4.2.0",
            previous_version="4.1.9",
            config={
                "routing_rules_version": "4.2.0",
                "rate_limit_per_second": 1000,
                "backend_auth_service": "http://auth-service:8080",
                "backend_product_service": "http://product-service:8081",
                "backend_search_service": "http://search-service:8082",
                "enable_canary": True,
                "canary_percentage": 20,
            },
            correct_config={
                "routing_rules_version": "4.1.9",
                "rate_limit_per_second": 1000,
                "backend_auth_service": "http://auth-service:8080",
                "backend_product_service": "http://product-service:8081",
                "backend_search_service": "http://search-service:8082",
                "enable_canary": False,
                "canary_percentage": 0,
            },
            logs=[
                {"timestamp": "2025-04-08T04:00:00Z", "level": "INFO", "message": "Deployed v4.2.0 with new canary routing rules"},
                {"timestamp": "2025-04-08T04:05:00Z", "level": "WARN", "message": "Canary route mismatch: /api/products routed to search-service for 20% of requests"},
                {"timestamp": "2025-04-08T04:10:00Z", "level": "ERROR", "message": "search-service returned 404 for /api/products/list (wrong backend)"},
                {"timestamp": "2025-04-08T04:15:00Z", "level": "ERROR", "message": "Routing error: 204 requests/min hitting wrong backend due to canary rule"},
                {"timestamp": "2025-04-08T04:20:00Z", "level": "WARN", "message": "Incoming request rate: 1800 req/s (normal: 900 req/s) - traffic spike detected"},
                {"timestamp": "2025-04-08T04:25:00Z", "level": "ERROR", "message": "Backend connection errors increasing: auth-service returning 503 intermittently"},
                {"timestamp": "2025-04-08T04:28:00Z", "level": "WARN", "message": "Rate limit approaching: 1800/1000 req/s (some requests being throttled)"},
            ],
            metrics_history=[
                {"timestamp_min": 0, "cpu": 70.0, "memory": 55.0, "req_per_sec": 1800.0, "error_rate": 0.45, "p99_latency_ms": 5500.0, "routing_errors_per_min": 204},
                {"timestamp_min": -15, "cpu": 55.0, "memory": 50.0, "req_per_sec": 1500.0, "error_rate": 0.30, "p99_latency_ms": 3000.0, "routing_errors_per_min": 180},
                {"timestamp_min": -30, "cpu": 30.0, "memory": 40.0, "req_per_sec": 900.0, "error_rate": 0.0, "p99_latency_ms": 100.0, "routing_errors_per_min": 0},
            ],
            dependencies=["auth-service", "product-service", "search-service"],
            deployment_status="recently_deployed",
            diagnostic_output="Canary routing ACTIVE: 20% of traffic using new routing rules\nRouting rule diff v4.1.9 -> v4.2.0:\n  - CHANGED: /api/products canary -> routes to search-service (BUG: should route to product-service)\n  - ADDED: /api/recommendations route\nTraffic: 1800 req/s (2x normal baseline of 900 req/s)\nRate limiting: actively throttling ~800 req/s",
        ),
        "auth-service": ServiceDef(
            name="auth-service",
            healthy=False,
            response_time_ms=8000.0,
            error_rate=0.35,
            cpu_percent=45.0,
            memory_percent=50.0,
            connections_active=95,
            connections_max=100,
            uptime_seconds=604800,
            version="2.1.0",
            previous_version="2.1.0",
            config={
                "db_host": "primary-db",
                "db_port": 5432,
                "db_pool_size": 50,
                "jwt_secret_rotation": "enabled",
                "session_ttl_minutes": 30,
            },
            correct_config={
                "db_host": "primary-db",
                "db_port": 5432,
                "db_pool_size": 50,
                "jwt_secret_rotation": "enabled",
                "session_ttl_minutes": 30,
            },
            logs=[
                {"timestamp": "2025-04-08T04:15:00Z", "level": "WARN", "message": "Database connection pool near limit: 48/50 active connections"},
                {"timestamp": "2025-04-08T04:20:00Z", "level": "ERROR", "message": "Cannot acquire DB connection: pool exhausted (50/50), waiting..."},
                {"timestamp": "2025-04-08T04:22:00Z", "level": "ERROR", "message": "Authentication request timed out waiting for DB connection"},
                {"timestamp": "2025-04-08T04:25:00Z", "level": "CRITICAL", "message": "primary-db rejecting new connections: too many clients already (max 100)"},
                {"timestamp": "2025-04-08T04:26:00Z", "level": "ERROR", "message": "POST /api/auth/login -> 503 (database unavailable)"},
            ],
            metrics_history=[
                {"timestamp_min": 0, "cpu": 45.0, "memory": 50.0, "req_per_sec": 400.0, "error_rate": 0.35, "db_pool_active": 50, "db_pool_waiting": 25},
                {"timestamp_min": -30, "cpu": 25.0, "memory": 45.0, "req_per_sec": 200.0, "error_rate": 0.0, "db_pool_active": 25, "db_pool_waiting": 0},
            ],
            dependencies=["primary-db"],
            deployment_status="stable",
            diagnostic_output="DB connection pool: 50/50 EXHAUSTED (25 requests waiting)\nprimary-db is rejecting new connections (at max_connections limit)\nNo recent deployments or config changes\nTraffic doubled due to external spike",
        ),
        "product-service": ServiceDef(
            name="product-service",
            healthy=False,
            response_time_ms=6000.0,
            error_rate=0.30,
            cpu_percent=55.0,
            memory_percent=60.0,
            connections_active=90,
            connections_max=100,
            uptime_seconds=604800,
            version="3.5.1",
            previous_version="3.5.1",
            config={
                "db_host": "primary-db",
                "db_port": 5432,
                "db_pool_size": 40,
                "cache_host": "redis-cache",
                "cache_ttl_seconds": 600,
            },
            correct_config={
                "db_host": "primary-db",
                "db_port": 5432,
                "db_pool_size": 40,
                "cache_host": "redis-cache",
                "cache_ttl_seconds": 600,
            },
            logs=[
                {"timestamp": "2025-04-08T04:20:00Z", "level": "WARN", "message": "DB connection pool saturated: 40/40"},
                {"timestamp": "2025-04-08T04:22:00Z", "level": "ERROR", "message": "primary-db: FATAL: too many connections for role 'product_svc'"},
                {"timestamp": "2025-04-08T04:24:00Z", "level": "WARN", "message": "Falling back to cache-only mode for product listings"},
                {"timestamp": "2025-04-08T04:25:00Z", "level": "ERROR", "message": "Cache miss for product detail - cannot serve without DB connection"},
            ],
            metrics_history=[
                {"timestamp_min": 0, "cpu": 55.0, "memory": 60.0, "req_per_sec": 500.0, "error_rate": 0.30, "cache_hit_rate": 0.40},
                {"timestamp_min": -30, "cpu": 30.0, "memory": 45.0, "req_per_sec": 300.0, "error_rate": 0.0, "cache_hit_rate": 0.85},
            ],
            dependencies=["primary-db", "redis-cache"],
            deployment_status="stable",
            diagnostic_output="DB pool: EXHAUSTED (40/40)\nprimary-db rejecting connections (global max_connections=100 reached)\nCache-only mode active but insufficient - cache hit rate dropped to 40%",
        ),
        "primary-db": ServiceDef(
            name="primary-db",
            healthy=False,
            response_time_ms=2500.0,
            error_rate=0.20,
            cpu_percent=90.0,
            memory_percent=80.0,
            connections_active=100,
            connections_max=100,
            uptime_seconds=2592000,
            version="15.4",
            previous_version="15.4",
            config={
                "max_connections": 100,
                "shared_buffers": "4GB",
                "work_mem": "512MB",
                "port": 5432,
            },
            correct_config={
                "max_connections": 300,  # needs increase
                "shared_buffers": "4GB",
                "work_mem": "512MB",
                "port": 5432,
            },
            logs=[
                {"timestamp": "2025-04-08T04:18:00Z", "level": "WARN", "message": "Connection count approaching max: 90/100"},
                {"timestamp": "2025-04-08T04:22:00Z", "level": "ERROR", "message": "FATAL: too many connections. max_connections=100, current=100"},
                {"timestamp": "2025-04-08T04:23:00Z", "level": "ERROR", "message": "FATAL: sorry, too many clients already (rejecting connection from 10.0.2.15)"},
                {"timestamp": "2025-04-08T04:25:00Z", "level": "ERROR", "message": "FATAL: sorry, too many clients already (rejecting connection from 10.0.2.20)"},
                {"timestamp": "2025-04-08T04:27:00Z", "level": "WARN", "message": "Query latency increased: avg 2500ms (normal: 15ms) - connection contention"},
            ],
            metrics_history=[
                {"timestamp_min": 0, "cpu": 90.0, "memory": 80.0, "connections": 100, "queries_per_sec": 300.0, "connection_rejections_per_min": 45},
                {"timestamp_min": -30, "cpu": 40.0, "memory": 65.0, "connections": 55, "queries_per_sec": 500.0, "connection_rejections_per_min": 0},
            ],
            dependencies=[],
            deployment_status="stable",
            diagnostic_output="max_connections: 100 (ALL IN USE)\nConnection breakdown: auth-service=48, product-service=40, admin=5, monitoring=7\nWaiting queue: 35 pending connections\nRecommendation: increase max_connections to accommodate traffic spike\nDisk I/O: normal, Replication: healthy",
        ),
        "redis-cache": ServiceDef(
            name="redis-cache",
            healthy=True,
            response_time_ms=45.0,
            error_rate=0.02,
            cpu_percent=30.0,
            memory_percent=70.0,
            connections_active=200,
            connections_max=1000,
            uptime_seconds=2592000,
            version="7.2",
            previous_version="7.2",
            config={"maxmemory": "4gb", "maxmemory_policy": "allkeys-lru"},
            correct_config={"maxmemory": "4gb", "maxmemory_policy": "allkeys-lru"},
            logs=[
                {"timestamp": "2025-04-08T04:20:00Z", "level": "WARN", "message": "Elevated command latency: avg 45ms (normal: 2ms) - high load"},
                {"timestamp": "2025-04-08T04:22:00Z", "level": "INFO", "message": "Key evictions: 1200/min (LRU policy active due to memory pressure)"},
            ],
            metrics_history=[
                {"timestamp_min": 0, "cpu": 30.0, "memory": 70.0, "ops_per_sec": 15000, "cache_hit_rate": 0.55, "evictions_per_min": 1200},
                {"timestamp_min": -30, "cpu": 10.0, "memory": 55.0, "ops_per_sec": 8000, "cache_hit_rate": 0.88, "evictions_per_min": 50},
            ],
            dependencies=[],
            deployment_status="stable",
            diagnostic_output="Redis functional but under pressure from traffic spike\nElevated latency is consequence of high request volume, not a root cause\nEviction rate high due to new campaign pages filling cache",
        ),
        "search-service": ServiceDef(
            name="search-service",
            healthy=True,
            response_time_ms=120.0,
            error_rate=0.15,
            cpu_percent=35.0,
            memory_percent=40.0,
            connections_active=60,
            connections_max=200,
            uptime_seconds=604800,
            version="1.8.0",
            previous_version="1.8.0",
            config={"elasticsearch_url": "http://es-cluster:9200", "index_prefix": "search"},
            correct_config={"elasticsearch_url": "http://es-cluster:9200", "index_prefix": "search"},
            logs=[
                {"timestamp": "2025-04-08T04:10:00Z", "level": "WARN", "message": "Receiving unexpected /api/products/* requests (not a search endpoint)"},
                {"timestamp": "2025-04-08T04:12:00Z", "level": "ERROR", "message": "404 Not Found: /api/products/list - no matching route in search-service"},
                {"timestamp": "2025-04-08T04:15:00Z", "level": "WARN", "message": "~200 misrouted requests/min for /api/products/* paths"},
            ],
            metrics_history=[
                {"timestamp_min": 0, "cpu": 35.0, "memory": 40.0, "req_per_sec": 250.0, "error_rate": 0.15, "misrouted_req_per_min": 200},
            ],
            dependencies=[],
            deployment_status="stable",
            diagnostic_output="Service healthy but receiving misrouted traffic\n~200 req/min for /api/products/* paths that don't exist in search-service\nThese appear to be routed here by api-gateway canary rules",
        ),
    }

    return ScenarioDef(
        task_name="multi_factor_outage",
        difficulty="hard",
        incident_summary="INCIDENT: Major production outage. 45% error rate across the platform. Users experiencing login failures, product page errors, and search returning 404s. Traffic is ~2x normal (marketing campaign launched today). api-gateway was deployed 30 min ago (v4.2.0). Multiple services reporting database connection issues.",
        services=services,
        alerts=[
            {"alert_id": "ALT-201", "severity": "critical", "service": "api-gateway", "message": "Error rate >40% platform-wide", "timestamp": "2025-04-08T04:25:00Z"},
            {"alert_id": "ALT-202", "severity": "critical", "service": "primary-db", "message": "Connection limit reached (100/100)", "timestamp": "2025-04-08T04:22:00Z"},
            {"alert_id": "ALT-203", "severity": "high", "service": "auth-service", "message": "Authentication failures >30%", "timestamp": "2025-04-08T04:25:00Z"},
            {"alert_id": "ALT-204", "severity": "high", "service": "product-service", "message": "Product API error rate >25%", "timestamp": "2025-04-08T04:24:00Z"},
            {"alert_id": "ALT-205", "severity": "medium", "service": "search-service", "message": "Elevated 404 rate from misrouted requests", "timestamp": "2025-04-08T04:15:00Z"},
        ],
        root_causes=[
            "api-gateway v4.2.0 canary routing bug: 20% of /api/products traffic misrouted to search-service",
            "primary-db max_connections=100 insufficient for doubled traffic load",
            "traffic spike from marketing campaign (2x normal) overwhelming current capacity",
        ],
        root_cause_keywords=[
            ["gateway", "canary", "routing", "misroute", "search-service", "4.2.0"],
            ["primary-db", "max_connections", "100", "connection", "exhaust", "limit"],
            ["traffic", "spike", "marketing", "campaign", "2x", "double", "capacity"],
        ],
        correct_remediations=[
            {"action_type": "rollback_deployment", "service_name": "api-gateway", "parameters": {}},
            {"action_type": "update_config", "service_name": "primary-db", "parameters": {"key": "max_connections", "value": "300"}},
            {"action_type": "scale_service", "service_name": "api-gateway", "parameters": {"replicas": "4"}},
        ],
        remediation_keywords=[
            ["rollback", "api-gateway"],
            ["update_config", "max_connections", "300", "primary-db"],
            ["scale", "api-gateway", "replicas"],
        ],
        max_steps=30,
        investigation_hints={
            "api-gateway": 0.05,
            "primary-db": 0.05,
            "auth-service": 0.04,
            "product-service": 0.04,
            "search-service": 0.04,
            "redis-cache": 0.02,
        },
        red_herrings=[
            "redis-cache elevated latency (consequence of traffic spike, not a root cause)",
            "CDN cache miss rate (normal for new campaign landing pages)",
        ],
    )


def build_ssl_expiry_scenario() -> ScenarioDef:
    """
    MEDIUM: SSL Certificate Expiry — Platform HTTPS Failure

    api-gateway's TLS certificate expired 2 days ago.
    cert-manager shows a pending renewal that requires manual approval.
    All HTTPS-dependent services cascade into errors.
    Red herring: auth-service was recently deployed (unrelated).

    Inspired by: Microsoft Teams global outage (Jan 29, 2020) caused by
    an expired TLS certificate on an identity service.
    # Postmortem reference: https://azure.microsoft.com/en-us/blog/details-of-the-january-29-2020-windows-azure-active-directory-outage/
    """
    services = {
        "api-gateway": ServiceDef(
            name="api-gateway",
            healthy=False,
            response_time_ms=35.0,
            error_rate=0.82,
            cpu_percent=18.0,
            memory_percent=22.0,
            connections_active=400,
            connections_max=5000,
            uptime_seconds=172800,
            version="3.9.4",
            previous_version="3.9.4",
            config={
                "ssl_cert_path": "/certs/api-gw-2023.crt",   # EXPIRED — should be 2025 cert
                "ssl_key_path": "/certs/api-gw-2023.key",
                "ssl_verify_client": False,
                "tls_min_version": "1.2",
                "cert_renewal_mode": "manual",
                "backend_timeout_ms": 30000,
            },
            correct_config={
                "ssl_cert_path": "/certs/api-gw-2025-renewed.crt",
                "ssl_key_path": "/certs/api-gw-2025-renewed.key",
                "ssl_verify_client": False,
                "tls_min_version": "1.2",
                "cert_renewal_mode": "manual",
                "backend_timeout_ms": 30000,
            },
            logs=[
                {"timestamp": "2025-04-06T00:00:01Z", "level": "WARN", "message": "TLS certificate /certs/api-gw-2023.crt expires in 0 days (expired: 2025-04-06T00:00:00Z)"},
                {"timestamp": "2025-04-06T00:01:00Z", "level": "ERROR", "message": "TLS handshake failed: certificate has expired (peer: 10.0.5.12)"},
                {"timestamp": "2025-04-06T00:01:05Z", "level": "ERROR", "message": "HTTPS connection rejected: SSL certificate verify failed — 'certificate has expired'"},
                {"timestamp": "2025-04-06T00:05:00Z", "level": "ERROR", "message": "TLS handshake failure rate: 82% (823/1005 connections rejected in last 5 min)"},
                {"timestamp": "2025-04-08T02:10:00Z", "level": "CRITICAL", "message": "SSL cert /certs/api-gw-2023.crt EXPIRED 2 days ago. All HTTPS traffic failing."},
                {"timestamp": "2025-04-08T02:10:30Z", "level": "ERROR", "message": "GET /api/products -> 000 (TLS handshake error — no response from client)"},
                {"timestamp": "2025-04-08T02:11:00Z", "level": "ERROR", "message": "GET /api/auth/login -> 000 (TLS handshake error)"},
            ],
            metrics_history=[
                {"timestamp_min": 0, "cpu": 18.0, "memory": 22.0, "req_per_sec": 200.0, "error_rate": 0.82, "tls_failures_per_min": 820},
                {"timestamp_min": -1440, "cpu": 20.0, "memory": 22.0, "req_per_sec": 950.0, "error_rate": 0.0, "tls_failures_per_min": 0},
            ],
            dependencies=["cert-manager"],
            deployment_status="stable",
            diagnostic_output=(
                "SSL Certificate Status:\n"
                "  Path: /certs/api-gw-2023.crt\n"
                "  Subject: CN=api.company.com\n"
                "  Issued: 2023-04-06T00:00:00Z\n"
                "  Expires: 2025-04-06T00:00:00Z  ← EXPIRED (2 days ago)\n"
                "  Status: EXPIRED\n\n"
                "Renewed certificate available at: /certs/api-gw-2025-renewed.crt\n"
                "  Expires: 2027-04-06T00:00:00Z\n"
                "  Status: VALID (awaiting config reload)\n\n"
                "Fix: Update ssl_cert_path to /certs/api-gw-2025-renewed.crt"
            ),
        ),
        "cert-manager": ServiceDef(
            name="cert-manager",
            healthy=True,
            response_time_ms=8.0,
            error_rate=0.0,
            cpu_percent=5.0,
            memory_percent=12.0,
            connections_active=2,
            connections_max=50,
            uptime_seconds=2592000,
            version="1.13.0",
            previous_version="1.13.0",
            config={
                "renewal_mode": "manual",
                "notify_days_before_expiry": 30,
                "auto_reload": False,
            },
            correct_config={
                "renewal_mode": "manual",
                "notify_days_before_expiry": 30,
                "auto_reload": False,
            },
            logs=[
                {"timestamp": "2025-03-07T00:00:00Z", "level": "WARN", "message": "Certificate api-gw-2023.crt expiring in 30 days — manual renewal required"},
                {"timestamp": "2025-03-20T00:00:00Z", "level": "WARN", "message": "Certificate api-gw-2023.crt expiring in 17 days — ACTION REQUIRED"},
                {"timestamp": "2025-04-04T00:00:00Z", "level": "ERROR", "message": "Certificate api-gw-2023.crt expiring in 2 days — CRITICAL"},
                {"timestamp": "2025-04-05T12:00:00Z", "level": "INFO", "message": "New certificate generated: api-gw-2025-renewed.crt (valid until 2027-04-06)"},
                {"timestamp": "2025-04-05T12:01:00Z", "level": "WARN", "message": "Certificate renewal PENDING: api-gateway config not updated to use new cert (manual mode)"},
                {"timestamp": "2025-04-08T02:00:00Z", "level": "CRITICAL", "message": "Certificate api-gw-2023.crt EXPIRED. api-gateway still using expired cert. Update required."},
            ],
            metrics_history=[
                {"timestamp_min": 0, "cpu": 5.0, "memory": 12.0, "certs_managed": 8, "certs_expired": 1, "pending_renewals": 1},
            ],
            dependencies=[],
            deployment_status="stable",
            diagnostic_output=(
                "cert-manager healthy. Managing 8 certificates.\n"
                "Certificate: api-gw-2023.crt — EXPIRED (2025-04-06)\n"
                "Renewed cert: api-gw-2025-renewed.crt — READY at /certs/api-gw-2025-renewed.crt\n"
                "Action required: Update api-gateway config to use renewed certificate."
            ),
        ),
        "auth-service": ServiceDef(
            name="auth-service",
            healthy=False,
            response_time_ms=8500.0,
            error_rate=0.78,
            cpu_percent=22.0,
            memory_percent=38.0,
            connections_active=40,
            connections_max=200,
            uptime_seconds=3600,
            version="4.1.0",
            previous_version="4.0.9",
            config={
                "token_expiry_seconds": 3600,
                "session_store": "redis",
                "upstream_gateway": "https://api-gateway:443",
            },
            correct_config={
                "token_expiry_seconds": 3600,
                "session_store": "redis",
                "upstream_gateway": "https://api-gateway:443",
            },
            logs=[
                {"timestamp": "2025-04-08T02:00:00Z", "level": "INFO", "message": "Deployed v4.1.0 (added MFA improvements)"},
                {"timestamp": "2025-04-08T02:10:00Z", "level": "ERROR", "message": "OAuth callback failed: upstream api-gateway returned SSL error"},
                {"timestamp": "2025-04-08T02:11:00Z", "level": "ERROR", "message": "POST /auth/login -> TLS handshake failure calling api-gateway"},
            ],
            metrics_history=[
                {"timestamp_min": 0, "cpu": 22.0, "memory": 38.0, "req_per_sec": 50.0, "error_rate": 0.78},
                {"timestamp_min": -120, "cpu": 18.0, "memory": 35.0, "req_per_sec": 180.0, "error_rate": 0.0},
            ],
            dependencies=["api-gateway"],
            deployment_status="recently_deployed",
            diagnostic_output=(
                "auth-service v4.1.0 deployed 2h ago (MFA improvements — no gateway changes)\n"
                "Errors are 100% TLS-related failures calling api-gateway upstream\n"
                "Service itself is healthy; failures are caused by api-gateway cert expiry\n"
                "Recent deployment to v4.1.0 is NOT the root cause"
            ),
        ),
        "user-api": ServiceDef(
            name="user-api",
            healthy=False,
            response_time_ms=9000.0,
            error_rate=0.74,
            cpu_percent=15.0,
            memory_percent=28.0,
            connections_active=30,
            connections_max=200,
            uptime_seconds=604800,
            version="2.4.1",
            previous_version="2.4.1",
            config={
                "gateway_url": "https://api-gateway:443",
                "retry_on_ssl_error": False,
            },
            correct_config={
                "gateway_url": "https://api-gateway:443",
                "retry_on_ssl_error": False,
            },
            logs=[
                {"timestamp": "2025-04-08T02:10:00Z", "level": "ERROR", "message": "GET /users/profile -> TLS certificate expired on api-gateway"},
                {"timestamp": "2025-04-08T02:11:00Z", "level": "ERROR", "message": "74% of requests failing: SSL handshake error from api-gateway"},
            ],
            metrics_history=[
                {"timestamp_min": 0, "cpu": 15.0, "memory": 28.0, "req_per_sec": 40.0, "error_rate": 0.74},
            ],
            dependencies=["api-gateway"],
            deployment_status="stable",
            diagnostic_output="Service healthy. All errors are SSL-related from api-gateway upstream.",
        ),
    }

    return ScenarioDef(
        task_name="ssl_certificate_expiry",
        difficulty="medium",
        incident_summary=(
            "INCIDENT: Platform-wide HTTPS failure. 82% of API requests failing with TLS errors. "
            "Users unable to log in or access any HTTPS endpoints. auth-service was recently deployed "
            "(v4.1.0, 2h ago). cert-manager shows 1 expired certificate. Started ~2 hours ago."
        ),
        services=services,
        alerts=[
            {"alert_id": "ALT-301", "severity": "critical", "service": "api-gateway", "message": "TLS failure rate >80% — HTTPS unavailable", "timestamp": "2025-04-08T02:10:00Z"},
            {"alert_id": "ALT-302", "severity": "high", "service": "cert-manager", "message": "1 expired certificate detected (api-gw-2023.crt)", "timestamp": "2025-04-08T02:00:00Z"},
            {"alert_id": "ALT-303", "severity": "high", "service": "auth-service", "message": "Authentication failure rate >75%", "timestamp": "2025-04-08T02:11:00Z"},
        ],
        root_causes=["api-gateway TLS certificate /certs/api-gw-2023.crt expired 2 days ago; renewed cert not yet applied to config"],
        root_cause_keywords=[["ssl", "certificate", "expired", "tls", "cert", "api-gw-2023", "2025-renewed"]],
        correct_remediations=[
            {"action_type": "update_config", "service_name": "api-gateway", "parameters": {"key": "ssl_cert_path", "value": "/certs/api-gw-2025-renewed.crt"}},
        ],
        remediation_keywords=[["update_config", "ssl_cert_path", "2025-renewed"]],
        max_steps=22,
        investigation_hints={
            "api-gateway": 0.09,
            "cert-manager": 0.08,
            "auth-service": 0.03,
            "user-api": 0.02,
        },
        red_herrings=["auth-service recent deployment v4.1.0 (MFA improvements, completely unrelated to TLS errors)"],
    )


def build_database_deadlock_scenario() -> ScenarioDef:
    """
    HARD: Database Deadlock Storm — Lock Order Inversion

    order-service v2.3.0 introduced a code change that acquires DB locks in
    order [orders_table → users_table], while user-service acquires them as
    [users_table → orders_table]. This creates a classic AB-BA deadlock under
    concurrent load. analytics-db sharing the primary DB amplifies contention.

    Three simultaneous issues:
    1. Lock order inversion in order-service v2.3.0 (root cause)
    2. analytics job running heavy aggregation queries on same DB (amplifier)
    3. job-queue backing up with failed order transactions (symptom)

    Red herrings:
    - payment-service elevated latency (waiting on order-service)
    - connection pool warning in user-service (consequence of deadlocks)

    Inspired by: GitLab database deadlock incident (2017) + Shopify order
    processing deadlocks under flash sale load.
    """
    services = {
        "order-service": ServiceDef(
            name="order-service",
            healthy=False,
            response_time_ms=45000.0,
            error_rate=0.68,
            cpu_percent=55.0,
            memory_percent=60.0,
            connections_active=180,
            connections_max=200,
            uptime_seconds=3600,
            version="2.3.0",
            previous_version="2.2.9",
            config={
                "db_host": "primary-db",
                "db_port": 5432,
                "db_pool_size": 50,
                "transaction_timeout_ms": 30000,
                "retry_on_deadlock": True,
                "max_deadlock_retries": 3,
            },
            correct_config={
                "db_host": "primary-db",
                "db_port": 5432,
                "db_pool_size": 50,
                "transaction_timeout_ms": 30000,
                "retry_on_deadlock": True,
                "max_deadlock_retries": 3,
            },
            logs=[
                {"timestamp": "2025-04-08T05:00:00Z", "level": "INFO", "message": "Deployed v2.3.0 (refactored order creation transaction for performance)"},
                {"timestamp": "2025-04-08T05:05:00Z", "level": "ERROR", "message": "Deadlock detected in transaction: waiting for lock on users row (held by user-service)"},
                {"timestamp": "2025-04-08T05:05:01Z", "level": "WARN", "message": "Deadlock retry #1 for order ORD-7821 (locks: orders → users)"},
                {"timestamp": "2025-04-08T05:05:15Z", "level": "ERROR", "message": "Deadlock retry #2 failed for ORD-7821: same deadlock pattern"},
                {"timestamp": "2025-04-08T05:06:00Z", "level": "ERROR", "message": "Deadlock retry #3 FAILED — order ORD-7821 aborted. Deadlock on: orders_table(row=82210), users_table(row=1502)"},
                {"timestamp": "2025-04-08T05:07:00Z", "level": "CRITICAL", "message": "Deadlock rate: 68% of transactions failing. Lock contention: order-service (orders→users) vs user-service (users→orders)"},
                {"timestamp": "2025-04-08T05:08:00Z", "level": "ERROR", "message": "DB transaction log: PID 44821 holding orders_table, waiting for users_table (PID 44830 holds it)"},
                {"timestamp": "2025-04-08T05:08:01Z", "level": "ERROR", "message": "DB transaction log: PID 44830 holding users_table, waiting for orders_table (PID 44821 holds it)"},
            ],
            metrics_history=[
                {"timestamp_min": 0, "cpu": 55.0, "memory": 60.0, "req_per_sec": 15.0, "error_rate": 0.68, "deadlocks_per_min": 42, "db_wait_time_ms": 45000},
                {"timestamp_min": -15, "cpu": 40.0, "memory": 50.0, "req_per_sec": 60.0, "error_rate": 0.30, "deadlocks_per_min": 18, "db_wait_time_ms": 20000},
                {"timestamp_min": -30, "cpu": 15.0, "memory": 40.0, "req_per_sec": 100.0, "error_rate": 0.0, "deadlocks_per_min": 0, "db_wait_time_ms": 12},
            ],
            dependencies=["primary-db", "user-service"],
            deployment_status="recently_deployed",
            diagnostic_output=(
                "DEADLOCK ANALYSIS:\n"
                "order-service v2.3.0 lock acquisition order: orders_table FIRST, then users_table\n"
                "user-service lock acquisition order: users_table FIRST, then orders_table\n"
                "This AB-BA pattern causes deadlock under concurrent order+user transactions.\n"
                "Change introduced in v2.3.0: 'Refactored createOrder() to batch-update users after orders'\n"
                "v2.2.9 lock order was: users_table FIRST, then orders_table (same as user-service)\n"
                "RECOMMENDATION: Rollback to v2.2.9"
            ),
        ),
        "user-service": ServiceDef(
            name="user-service",
            healthy=False,
            response_time_ms=12000.0,
            error_rate=0.35,
            cpu_percent=40.0,
            memory_percent=55.0,
            connections_active=145,
            connections_max=200,
            uptime_seconds=604800,
            version="5.1.2",
            previous_version="5.1.2",
            config={
                "db_host": "primary-db",
                "db_port": 5432,
                "db_pool_size": 60,
            },
            correct_config={
                "db_host": "primary-db",
                "db_port": 5432,
                "db_pool_size": 60,
            },
            logs=[
                {"timestamp": "2025-04-08T05:05:00Z", "level": "WARN", "message": "Deadlock detected: waiting for orders_table lock (held by order-service PID 44821)"},
                {"timestamp": "2025-04-08T05:06:00Z", "level": "ERROR", "message": "Transaction aborted due to deadlock — update user USR-1502 profile failed"},
                {"timestamp": "2025-04-08T05:07:00Z", "level": "WARN", "message": "DB connection pool: 145/200 (connections backing up due to deadlock wait)"},
                {"timestamp": "2025-04-08T05:08:00Z", "level": "ERROR", "message": "35% of user update requests failing: deadlock with order-service v2.3.0"},
            ],
            metrics_history=[
                {"timestamp_min": 0, "cpu": 40.0, "memory": 55.0, "req_per_sec": 80.0, "error_rate": 0.35, "db_pool_waiting": 45},
                {"timestamp_min": -30, "cpu": 20.0, "memory": 48.0, "req_per_sec": 200.0, "error_rate": 0.0, "db_pool_waiting": 0},
            ],
            dependencies=["primary-db"],
            deployment_status="stable",
            diagnostic_output=(
                "user-service v5.1.2 (no recent deployment, stable)\n"
                "Lock acquisition order: users_table FIRST, then orders_table (correct pattern)\n"
                "Deadlocks initiated by order-service v2.3.0 which changed lock order\n"
                "Connection pool: 145/200 — backing up due to deadlock waits (not a separate issue)"
            ),
        ),
        "primary-db": ServiceDef(
            name="primary-db",
            healthy=False,
            response_time_ms=1800.0,
            error_rate=0.25,
            cpu_percent=85.0,
            memory_percent=72.0,
            connections_active=180,
            connections_max=200,
            uptime_seconds=2592000,
            version="15.4",
            previous_version="15.4",
            config={"max_connections": 200, "deadlock_timeout": "1s", "lock_timeout": "30s"},
            correct_config={"max_connections": 200, "deadlock_timeout": "1s", "lock_timeout": "30s"},
            logs=[
                {"timestamp": "2025-04-08T05:05:00Z", "level": "ERROR", "message": "DEADLOCK DETECTED: PID 44821 (order-service) vs PID 44830 (user-service) on tables orders, users"},
                {"timestamp": "2025-04-08T05:06:00Z", "level": "ERROR", "message": "Deadlock count last 60s: 42 — CRITICAL (normal: 0)"},
                {"timestamp": "2025-04-08T05:07:00Z", "level": "WARN", "message": "Lock wait queue depth: 89 transactions waiting (normal: 0-2)"},
                {"timestamp": "2025-04-08T05:08:00Z", "level": "ERROR", "message": "query stats: deadlock_query='UPDATE orders SET ... WHERE user_id=$1' vs 'UPDATE users SET ... WHERE id IN (SELECT user_id FROM orders ...)'"},
            ],
            metrics_history=[
                {"timestamp_min": 0, "cpu": 85.0, "memory": 72.0, "connections": 180, "deadlocks_per_min": 42, "lock_waits_per_min": 290, "queries_per_sec": 85.0},
                {"timestamp_min": -30, "cpu": 25.0, "memory": 60.0, "connections": 50, "deadlocks_per_min": 0, "lock_waits_per_min": 2, "queries_per_sec": 450.0},
            ],
            dependencies=[],
            deployment_status="stable",
            diagnostic_output=(
                "DEADLOCK FREQUENCY: 42/min (CRITICAL)\n"
                "Competing transactions:\n"
                "  order-service: BEGIN → LOCK orders(row X) → LOCK users(row Y)\n"
                "  user-service:  BEGIN → LOCK users(row Y) → LOCK orders(row X)\n"
                "Pattern: AB-BA deadlock caused by lock order inversion in order-service v2.3.0\n"
                "PostgreSQL resolves by aborting one victim per deadlock cycle\n"
                "Also: analytics-job running heavy aggregation adds lock pressure\n"
                "Primary fix: rollback order-service to v2.2.9"
            ),
        ),
        "analytics-db": ServiceDef(
            name="analytics-db",
            healthy=True,
            response_time_ms=3200.0,
            error_rate=0.05,
            cpu_percent=70.0,
            memory_percent=65.0,
            connections_active=45,
            connections_max=200,
            uptime_seconds=604800,
            version="15.4",
            previous_version="15.4",
            config={"max_connections": 200, "shared_buffers": "2GB"},
            correct_config={"max_connections": 200, "shared_buffers": "2GB"},
            logs=[
                {"timestamp": "2025-04-08T05:00:00Z", "level": "INFO", "message": "Daily analytics job started: ORDER_SUMMARY_AGG (estimated runtime: 45 min)"},
                {"timestamp": "2025-04-08T05:05:00Z", "level": "WARN", "message": "Analytics job running slower than usual: 2800ms avg query (normal: 800ms)"},
                {"timestamp": "2025-04-08T05:08:00Z", "level": "WARN", "message": "Heavy sequential scan on orders table acquiring shared locks — contributing to contention"},
            ],
            metrics_history=[
                {"timestamp_min": 0, "cpu": 70.0, "memory": 65.0, "shared_lock_holds": 1200, "query_time_ms": 3200},
            ],
            dependencies=[],
            deployment_status="stable",
            diagnostic_output=(
                "analytics-db running daily aggregation job ORDER_SUMMARY_AGG\n"
                "Job acquires SHARED locks on orders table — amplifies deadlock contention\n"
                "NOT the root cause, but should be deferred when order-service deadlocks are occurring\n"
                "Stopping analytics job temporarily will reduce (but not eliminate) deadlock rate"
            ),
        ),
        "payment-service": ServiceDef(
            name="payment-service",
            healthy=False,
            response_time_ms=40000.0,
            error_rate=0.60,
            cpu_percent=30.0,
            memory_percent=42.0,
            connections_active=60,
            connections_max=200,
            uptime_seconds=604800,
            version="3.1.0",
            previous_version="3.1.0",
            config={"order_service_url": "http://order-service:8081", "timeout_ms": 30000},
            correct_config={"order_service_url": "http://order-service:8081", "timeout_ms": 30000},
            logs=[
                {"timestamp": "2025-04-08T05:07:00Z", "level": "ERROR", "message": "Order creation timeout (30000ms) — order-service deadlocked"},
                {"timestamp": "2025-04-08T05:08:00Z", "level": "WARN", "message": "60% payment failure rate — all failures are 504 timeouts from order-service"},
            ],
            metrics_history=[
                {"timestamp_min": 0, "cpu": 30.0, "memory": 42.0, "req_per_sec": 20.0, "error_rate": 0.60},
            ],
            dependencies=["order-service"],
            deployment_status="stable",
            diagnostic_output="payment-service healthy. Failures are 100% timeout-related from order-service. Not a root cause.",
        ),
        "job-queue": ServiceDef(
            name="job-queue",
            healthy=False,
            response_time_ms=200.0,
            error_rate=0.45,
            cpu_percent=25.0,
            memory_percent=35.0,
            connections_active=20,
            connections_max=100,
            uptime_seconds=604800,
            version="2.0.1",
            previous_version="2.0.1",
            config={"queue_backend": "redis", "dead_letter_threshold": 5},
            correct_config={"queue_backend": "redis", "dead_letter_threshold": 5},
            logs=[
                {"timestamp": "2025-04-08T05:07:00Z", "level": "WARN", "message": "Dead letter queue depth: 1842 (normal <50) — order transactions failing"},
                {"timestamp": "2025-04-08T05:08:00Z", "level": "ERROR", "message": "Queue backlog: 8500 pending order jobs (normal: <200)"},
            ],
            metrics_history=[
                {"timestamp_min": 0, "cpu": 25.0, "memory": 35.0, "queue_depth": 8500, "dlq_depth": 1842, "processing_rate": 5.0},
            ],
            dependencies=["order-service"],
            deployment_status="stable",
            diagnostic_output="job-queue backing up due to order-service failures. This is a symptom, not the cause.",
        ),
    }

    return ScenarioDef(
        task_name="database_deadlock",
        difficulty="hard",
        incident_summary=(
            "INCIDENT: Order processing failures at 68% error rate. Payment failures cascading. "
            "order-service was deployed 8 minutes ago (v2.3.0). Database showing 42 deadlocks/min "
            "(normal: 0). Analytics daily job is also running. Job queue depth: 8500 (normal <200)."
        ),
        services=services,
        alerts=[
            {"alert_id": "ALT-401", "severity": "critical", "service": "order-service", "message": "Order failure rate 68% — deadlock storm detected", "timestamp": "2025-04-08T05:07:00Z"},
            {"alert_id": "ALT-402", "severity": "critical", "service": "primary-db", "message": "Deadlock rate: 42/min (threshold: 5/min)", "timestamp": "2025-04-08T05:06:00Z"},
            {"alert_id": "ALT-403", "severity": "high", "service": "payment-service", "message": "Payment failure rate 60%", "timestamp": "2025-04-08T05:08:00Z"},
            {"alert_id": "ALT-404", "severity": "high", "service": "job-queue", "message": "Queue depth 8500 — critical backlog", "timestamp": "2025-04-08T05:08:00Z"},
        ],
        root_causes=[
            "order-service v2.3.0 lock order inversion: acquires orders_table then users_table (v2.2.9 was reversed), causing AB-BA deadlock with user-service",
        ],
        root_cause_keywords=[
            ["order-service", "deadlock", "lock", "v2.3.0", "inversion", "rollback", "orders", "users"],
        ],
        correct_remediations=[
            {"action_type": "rollback_deployment", "service_name": "order-service", "parameters": {}},
        ],
        remediation_keywords=[["rollback", "order-service"]],
        max_steps=28,
        investigation_hints={
            "order-service": 0.09,
            "primary-db": 0.08,
            "user-service": 0.06,
            "analytics-db": 0.04,
            "payment-service": 0.03,
            "job-queue": 0.02,
        },
        red_herrings=[
            "payment-service elevated latency (symptom of order-service deadlock, not a root cause)",
            "user-service connection pool warning (consequence of deadlock waits, not a separate issue)",
            "analytics-db running heavy aggregation (amplifier, not root cause — fix order-service first)",
        ],
    )


class ScenarioFactory:
    """
    Procedurally generates seeded variations of base scenarios.

    v2.0 Design (superior to competitors' simple randomization):
    - Seed-reproducible: same seed always produces same scenario
    - Varies metric values within physically realistic bounds
    - Randomizes red herring assignment from a curated pool
    - Supports scenario selection by difficulty for curriculum learning
    - From 5 archetypes × seeded variation = thousands of unique episodes
      suitable for large-scale RL training runs

    Usage:
        scenario = ScenarioFactory.generate("cascading_service_timeout", seed=42)
        scenario = ScenarioFactory.generate_by_difficulty("hard", seed=99)
    """

    # Pool of extra red herrings to inject across scenarios
    _EXTRA_RED_HERRINGS = [
        "monitoring-service showing elevated CPU (metric collection overhead, unrelated)",
        "CDN cache miss rate elevated (expected for new content rollout, not an issue)",
        "log aggregator reporting 2% packet loss (known infrastructure issue, unrelated)",
        "internal metrics dashboard showing stale data (exporter restart lag, benign)",
        "staging environment showing similar errors (pre-existing test issue, unrelated)",
        "backup job running slower than usual (disk I/O contention from backup, normal)",
        "load balancer health check latency +5ms (expected during high load, normal)",
    ]

    _TASKS_BY_DIFFICULTY = {
        "easy": ["db_connection_failure"],
        "medium": ["cascading_service_timeout", "ssl_certificate_expiry"],
        "hard": ["multi_factor_outage", "database_deadlock"],
    }

    @classmethod
    def generate(cls, task_name: str, seed: int) -> ScenarioDef:
        """
        Generate a seeded variation of a named scenario.

        The variation applies realistic noise to metric values, slightly adjusts
        investigation hint weights, and may inject an additional red herring.
        The root cause and correct remediation are always preserved exactly.
        """
        rng = random.Random(seed)
        base = get_scenario(task_name)
        return cls._apply_variation(base, rng, seed)

    @classmethod
    def generate_by_difficulty(cls, difficulty: str, seed: int) -> ScenarioDef:
        """Select a random task at the given difficulty level, then vary it by seed."""
        rng = random.Random(seed)
        task_pool = cls._TASKS_BY_DIFFICULTY.get(difficulty, cls._TASKS_BY_DIFFICULTY["easy"])
        task_name = rng.choice(task_pool)
        return cls.generate(task_name, seed)

    @classmethod
    def _apply_variation(cls, base: ScenarioDef, rng: random.Random, seed: int) -> ScenarioDef:
        """Apply parameterized variation while preserving correctness."""
        varied = copy.deepcopy(base)

        # Vary metric values ±12% to prevent overfitting to exact values
        for svc in varied.services.values():
            svc.response_time_ms = round(svc.response_time_ms * rng.uniform(0.90, 1.12), 1)
            svc.error_rate = min(1.0, round(svc.error_rate * rng.uniform(0.92, 1.08), 3))
            svc.cpu_percent = min(100.0, round(svc.cpu_percent * rng.uniform(0.88, 1.12), 1))
            svc.memory_percent = min(100.0, round(svc.memory_percent * rng.uniform(0.90, 1.10), 1))

        # Vary investigation hints slightly (reward shaping texture)
        for svc_name in varied.investigation_hints:
            varied.investigation_hints[svc_name] = round(
                varied.investigation_hints[svc_name] * rng.uniform(0.80, 1.20), 3
            )

        # 60% chance to inject one extra red herring from the pool
        if rng.random() < 0.60:
            extra = rng.choice(cls._EXTRA_RED_HERRINGS)
            if extra not in varied.red_herrings:
                varied.red_herrings.append(extra)

        # Tag incident summary with seed for reproducibility auditing
        varied.incident_summary = varied.incident_summary.rstrip() + f" [variant seed={seed}]"
        return varied


SCENARIOS = {
    "db_connection_failure": build_easy_scenario,
    "cascading_service_timeout": build_medium_scenario,
    "multi_factor_outage": build_hard_scenario,
    "ssl_certificate_expiry": lambda: build_ssl_expiry_scenario(),
    "database_deadlock": lambda: build_database_deadlock_scenario(),
}


def get_scenario(task_name: str) -> ScenarioDef:
    """Load a scenario by task name."""
    builder = SCENARIOS.get(task_name)
    if not builder:
        raise ValueError(f"Unknown task: {task_name}. Available: {list(SCENARIOS.keys())}")
    return builder()


def list_tasks() -> list[dict]:
    """List all available tasks with metadata."""
    return [
        {"task_name": "db_connection_failure", "difficulty": "easy", "description": "Single service database connection failure due to port misconfiguration"},
        {"task_name": "cascading_service_timeout", "difficulty": "medium", "description": "Multi-service cascading timeout caused by memory leak in downstream service"},
        {"task_name": "multi_factor_outage", "difficulty": "hard", "description": "Complex multi-factor outage: routing bug + connection pool exhaustion + traffic spike"},
        {"task_name": "ssl_certificate_expiry", "difficulty": "medium", "description": "Platform-wide HTTPS failure from expired TLS certificate on api-gateway, cert-manager pending renewal"},
        {"task_name": "database_deadlock", "difficulty": "hard", "description": "Cascading deadlocks from lock-order inversion introduced in order-service v2.3.0, combined with analytics contention"},
    ]
