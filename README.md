---
title: Incident Response Environment
emoji: 🚨
colorFrom: red
colorTo: blue
sdk: docker
pinned: false
base_path: /web
---

# Incident Response Environment

**Production Incident Response and Root Cause Analysis — OpenEnv RL Environment**

An AI agent acts as an on-call SRE (Site Reliability Engineer). It receives alerts about production incidents and must investigate logs, metrics, and configurations across multi-service architectures to diagnose root causes, determine blast radius, and execute correct remediations.

## Why This Environment?

Every tech company from Meta to startups has engineers doing incident response at 3 AM. Production outages cost billions annually. Training AI agents to handle incident triage, root cause analysis, and remediation is one of the highest-value applications of RL for real-world tasks.

This environment models the complete incident response workflow: receive alert, triage severity, investigate services (read logs, check metrics, examine configs), trace dependencies (follow the cascade upstream/downstream), identify root causes, and apply remediations (restart, rollback, reconfigure, scale).

Key design properties include multi-step multi-turn episodes (10-30 steps with branching investigation paths), rich partial rewards (not binary, rewards investigation, penalizes waste), multiple valid trajectories (check logs first or metrics first, both work), red herrings (recent deployments that are not the cause, symptomatic services), and difficulty curriculum (easy to medium to hard with increasing service count and root causes).

## Action Space

Actions include: investigate_logs (with optional keyword filter), check_metrics (with optional metric_type), read_config, check_service_health, run_diagnostic, restart_service, update_config (with key and value), rollback_deployment, scale_service (with replicas count), and declare_root_cause (with cause description).

Action format: `{"action_type": "<type>", "service_name": "<name>", "parameters": {...}}`

## Observation Space

Each observation contains: action_result (text output from last action), active_alerts (current alerts with severity), service_statuses (health/error rate/response time for all services), available_services, step_number/max_steps, and incident_summary.

## Reward Function

Investigation of a new relevant service yields +0.02 to +0.10. Correct root cause declaration yields +0.20. Correct remediation yields +0.12 to +0.15. Time bonus of +0.05 to +0.10 for fast resolution. Redundant actions get -0.01. Wrong root cause gets -0.05. Unnecessary remediation gets -0.02. Timeout gets -0.05.

## Grader (Score 0.0 to 1.0)

Root cause identification (40 percent weight), remediation quality (35 percent), investigation thoroughness (15 percent), and efficiency (10 percent).

## Tasks

### Task 1: db_connection_failure (Easy)

Services: user-api, postgres-primary, nginx-lb. Scenario: user-api returning 503 errors due to database port misconfigured as 5433 instead of 5432 after maintenance. Expected score range: 0.5 to 1.0.

### Task 2: cascading_service_timeout (Medium)

Services: payment-service, order-service, inventory-service, orders-db. Scenario: Payment failures cascading through services. Memory leak in inventory-service causing 18-second GC pauses. Red herring: recent payment-service deployment. Expected score range: 0.3 to 0.9.

### Task 3: multi_factor_outage (Hard)

Services: api-gateway, auth-service, product-service, primary-db, redis-cache, search-service. Scenario: 45 percent platform error rate with three simultaneous root causes (routing bug, connection pool exhaustion, traffic spike). Multiple red herrings. Expected score range: 0.1 to 0.8.

## Setup Instructions

### Prerequisites

Python 3.10+, Docker, and a HuggingFace token (free tier works).

### Build and Run

```bash
# Install dependencies
pip install -e .

# Build Docker image
docker build -t incident-response-env .

# Run the environment
docker run -p 7860:7860 incident-response-env

# Web UI at http://localhost:7860/web
# API at http://localhost:7860
```

### Test Manually

```bash
# Reset
curl -X POST http://localhost:7860/reset \
  -H "Content-Type: application/json" \
  -d '{"task_name": "db_connection_failure"}'

# Step
curl -X POST http://localhost:7860/step \
  -H "Content-Type: application/json" \
  -d '{"action_type": "investigate_logs", "service_name": "user-api", "parameters": {"keyword": "error"}}'

# Score
curl http://localhost:7860/score
```

### Run Inference

```bash
export HF_TOKEN=hf_your_token_here
python inference.py
```

### Deploy to HuggingFace Spaces

```bash
pip install openenv-core
openenv push <your-hf-username>/incident-response-env
```

## Baseline Scores

Task db_connection_failure (easy): approximately 0.65, using 8-12 steps. Task cascading_service_timeout (medium): approximately 0.45, using 12-18 steps. Task multi_factor_outage (hard): approximately 0.25, using 18-25 steps. Baseline uses Qwen2.5-72B-Instruct via HF Router.

## Project Structure

```
incident-response-env/
├── Dockerfile              # Container definition
├── openenv.yaml            # OpenEnv metadata
├── pyproject.toml          # Python project config
├── inference.py            # Mandatory inference script (root dir)
├── models.py               # Pydantic models (Action, Observation, State)
├── client.py               # HTTP client library
├── .env.example            # Environment variable template
├── README.md               # This file
├── scenarios/
│   ├── __init__.py
│   └── definitions.py      # Task scenarios (easy/medium/hard)
└── server/
    ├── __init__.py
    ├── app.py              # FastAPI server + Gradio UI
    └── env.py              # Core environment logic
```

## Environment Variables

HF_TOKEN (required): HuggingFace API token. API_BASE_URL (default: https://router.huggingface.co/v1): LLM API endpoint. MODEL_NAME (default: Qwen/Qwen2.5-72B-Instruct): Model for inference. IMAGE_NAME (default: incident-response-env): Docker image name. ENV_URL (default: http://localhost:7860): Environment server URL.

## License

MIT
