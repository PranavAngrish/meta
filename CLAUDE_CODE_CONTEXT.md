# CLAUDE CODE — Incident Response OpenEnv Project Context

## WHAT THIS PROJECT IS
A complete OpenEnv RL environment for the Meta Scaler Hackathon (Round 1).
An AI agent acts as an on-call SRE handling production incidents — investigating
logs, metrics, configs across multi-service architectures to find root causes
and apply remediations.

## PROJECT STATUS: COMPLETE & TESTED
All 3 tasks (easy/medium/hard) pass with scores > 0.85.
All HTTP endpoints (reset/step/state/health/tasks/score) verified working.
The core env logic, scenarios, models, inference script, Dockerfile, openenv.yaml, and README are all written.

## FILE STRUCTURE
```
meta-hackathon/
├── Dockerfile              # Container for HF Spaces deployment
├── openenv.yaml            # OpenEnv metadata spec
├── pyproject.toml          # Python project config
├── inference.py            # MANDATORY inference script (root dir)
├── models.py               # Pydantic: ActionType enum, Action, Observation, State
├── client.py               # HTTP client for env server
├── .env.example            # Env var template
├── README.md               # Documentation
├── CLAUDE_CODE_CONTEXT.md  # This file
├── scenarios/
│   ├── __init__.py
│   └── definitions.py      # 3 scenarios: easy/medium/hard with services, logs, metrics, configs
└── server/
    ├── __init__.py
    ├── app.py              # FastAPI + Gradio web UI
    └── env.py              # Core env: reset/step/state/get_score
```

## HOW THE ENVIRONMENT WORKS

### Action Space (10 actions):
- investigate_logs(service, keyword?) → returns filtered logs
- check_metrics(service, metric_type?) → returns time-series metrics
- read_config(service) → returns current config JSON
- check_service_health(service) → returns health overview
- run_diagnostic(service) → returns diagnostic output
- restart_service(service) → restarts (correct for some scenarios)
- update_config(service, key, value) → changes config
- rollback_deployment(service) → rollback to prev version
- scale_service(service, replicas) → scale horizontally
- declare_root_cause(cause) → submit diagnosis

### Reward Design:
- Investigation: +0.02 to +0.10 for first check of relevant service
- Root cause: +0.20 for correct declaration
- Remediation: +0.12 to +0.15 for correct fix
- Time bonus: +0.05 to +0.10 for fast resolution
- Penalties: -0.01 redundant, -0.05 wrong cause, -0.02 wrong fix

### Grader (0.0 to 1.0):
- Root cause ID: 40% weight
- Remediation: 35% weight  
- Investigation: 15% weight
- Efficiency: 10% weight

## THREE TASKS

### 1. db_connection_failure (EASY)
- 3 services: user-api, postgres-primary, nginx-lb
- Root cause: user-api db_port=5433 instead of 5432
- Fix: update_config user-api db_port 5432
- Max 20 steps

### 2. cascading_service_timeout (MEDIUM)
- 4 services: payment-service, order-service, inventory-service, orders-db
- Root cause: inventory-service memory leak, JVM heap 2GB too small
- Red herring: payment-service recent deployment (unrelated)
- Fix: restart inventory-service + update jvm_heap_max to 4g
- Max 25 steps

### 3. multi_factor_outage (HARD)
- 6 services: api-gateway, auth-service, product-service, primary-db, redis-cache, search-service
- 3 root causes: gateway routing bug + db max_connections + traffic spike
- Red herrings: redis latency, CDN misses
- Fix: rollback gateway + update db max_connections=300 + scale gateway
- Max 30 steps

## REMAINING WORK (for Claude Code)

### Must do before submission:
1. `docker build -t incident-response-env .` — verify Docker builds
2. `docker run -p 7860:7860 incident-response-env` — verify container starts
3. Set HF_TOKEN and run `python inference.py` — verify inference produces valid [START]/[STEP]/[END] output
4. `openenv validate` — verify spec compliance
5. `openenv push <username>/incident-response-env` — deploy to HF Spaces
6. Verify HF Space responds to /health and /reset endpoints

### Nice to have improvements:
- Add more granular partial rewards for the hard task
- Add a 4th bonus task if time permits
- Polish the Gradio web UI with better layout
- Add unit tests
- Fine-tune the LLM system prompt in inference.py for better scores

## KEY TECHNICAL NOTES
- inference.py uses OpenAI client through HF Router (https://router.huggingface.co/v1)
- Default model: Qwen/Qwen2.5-72B-Instruct (free via HF)
- HF_TOKEN is required (free tier works)
- Server runs on port 7860 (HF Spaces default)
- Dockerfile moves server/Dockerfile to root (already done)
- The env is stateful — single instance, reset() clears everything

## HACKATHON EVALUATION CRITERIA
- Real-world utility: 30% (incident response = real and high value)
- Task & grader quality: 25% (3 tasks, deterministic graders, 0-1 range)
- Environment design: 20% (rich rewards, clean state, good action space)
- Code quality & spec: 15% (typed models, Dockerfile, openenv.yaml)
- Creativity & novelty: 10% (novel domain, interesting reward design)
