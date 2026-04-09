# Incident Response Environment

**An OpenEnv-compatible reinforcement learning environment for training AI agents to diagnose and resolve production incidents.**

Built for Meta's Hackathon · v4.0.0

---

## What is this?

Modern cloud systems fail in complex, cascading ways. When they do, an on-call engineer has minutes — not hours — to identify the root cause, assess blast radius, and apply a safe fix. This environment simulates exactly that pressure.

AI agents (or human engineers using the sandbox UI) take on the role of an SRE responding to a live production incident. They must:

1. **Investigate** — query logs, metrics, configs, and service health across a multi-service architecture
2. **Diagnose** — identify the precise root cause from noisy, partially misleading signals
3. **Remediate** — apply the correct targeted fix without causing collateral damage
4. **Work fast** — unresolved failures cascade every 4 steps, spreading degradation across the stack

---

## Key Features

### 6-Dimensional Scoring System

Every episode is graded across six independent dimensions:

| Dimension | Weight | What it measures |
|-----------|--------|-----------------|
| **Root Cause** | 30% | Did the agent identify the correct failure mode? |
| **Remediation** | 25% | Was the right fix applied to the right service? |
| **Investigation** | 15% | Did the agent check enough of the affected services? |
| **Efficiency** | 10% | Were steps used wisely (fewer = higher bonus)? |
| **Safety** | 10% | Were destructive actions avoided on healthy services? |
| **Sequence** | 10% | Did the agent diagnose before applying any fix? |

The final score is clamped to the open interval **(0.01, 0.99)** for OpenEnv validator compliance. Each dimension is independently measurable — enabling fine-grained process supervision for RL training.

### Anti-Reward-Hacking Mechanisms

Three mechanisms prevent agents from gaming the reward signal:

**1. Observation Loop Detection**
Agents that chain 3+ consecutive investigation actions without applying a fix receive:
- Per-occurrence penalty: –0.08 (first), –0.03 (rolling)
- Final score hard-capped at **0.45** while the incident remains unresolved

**2. Diagnosis Gate**
On gated scenarios (`cascading_service_timeout`, `ssl_certificate_expiry`, `database_deadlock`), applying a remediation before investigating the required service yields only **50% remediation credit**. Lucky guessing is penalised; structured investigation is rewarded.

**3. Safety Penalty**
Applying a destructive action (restart, rollback, scale-down) to a service that is currently healthy incurs a **–0.25 per-step penalty**. Collateral damage is tracked and shown in the UI.

### Dynamic System Degradation

Real incidents don't wait. Every 4 steps without resolution:
- Unresolved failures cascade to dependent services
- Error rates increase by +10% per cascade tick
- Response times increase by +40% per cascade tick

This creates genuine time pressure: slow investigation leads to a larger outage, which affects the Investigation and Safety dimensions.

### Failure Type Classification

After each step, the agent is classified into one of five failure types, providing a human-readable process signal:

| Type | Description |
|------|-------------|
| **Efficient Reasoner** | Investigated, diagnosed, fixed in order — high score |
| **Symptom Chaser** | Investigated many services without identifying root cause |
| **Lucky Guesser** | Fixed before properly diagnosing — blind fix |
| **Stuck in Observation Loop** | Too many investigation steps without a fix |
| **Late Corrector** | Fixed the right thing but far too late |

### Procedural Generation

All 5 main tasks support seeded procedural variation via `ScenarioFactory.generate(task_name, seed)`:
- Metric values vary ±12% from baseline
- Random red herring services are injected to test signal-from-noise reasoning
- Investigation hints are varied across seeds
- Same seed always produces the same scenario — reproducible benchmarking

### Per-Step Feedback

Every step returns a human-readable `feedback` string explaining the reward earned or lost in plain language. Agents can use this to self-correct mid-episode; it also makes the training signal interpretable for humans reviewing trajectories.

---

## Scenario Library

### Easy

**`db_connection_failure`** — A single misconfigured `db_port` causes 503 errors on the user-facing API. One root cause, one remediation, 20-step budget. Perfect curriculum learning entry point.

**`alert_triage`** — Classify the severity of three different alert scenarios (P1/P2/P3/P4) using up to 3 investigation actions each. Tests precision: a high error rate does *not* always mean P1 if graceful fallback is active.

### Medium

**`cascading_service_timeout`** — A memory leak in an upstream service causes cascading timeouts down the dependency chain. Multiple services show degradation, but only one is the true root. Red herrings are present.

**`ssl_certificate_expiry`** — An expired TLS certificate causes platform-wide HTTPS failures. The fix (`update_config`) must target the correct service after investigating the certificate expiry indicator.

### Hard

**`multi_factor_outage`** — Three simultaneous root causes: a routing bug, connection pool exhaustion, and a traffic spike. All three must be declared and remediated for full credit. Requires systematic triage.

**`database_deadlock`** — A lock-order inversion bug in the order service causes intermittent deadlocks that eventually become persistent. The only correct fix is a deployment rollback. Requires log analysis to distinguish from a flaky service.

---

## Quick Start

### Native Python

```bash
# Clone and install
git clone <repo-url>
cd meta-hackathon
pip install -e .

# Start the server
python server/app.py
```

The server starts on `http://localhost:7860`.
- **Web UI:** `http://localhost:7860/web`
- **API docs:** `http://localhost:7860/docs`
- **Health check:** `http://localhost:7860/health`

### Docker

```bash
docker build -t incident-response-env .
docker run -p 7860:7860 incident-response-env
```

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `PORT` | `7860` | Server listen port |
| `HF_TOKEN` | — | Hugging Face token (for inference.py) |
| `API_BASE_URL` | — | LLM API base URL |
| `MODEL_NAME` | — | Model identifier for inference |

Copy `.env.example` → `.env` and fill in values.

---

## HTTP API Reference

All endpoints are available at the server root. Interactive docs: `/docs`.

### `GET /health`
Liveness probe.
```json
{"status": "ok", "environment": "incident-response-env", "version": "4.0.0"}
```

### `GET /tasks`
List all available tasks with difficulty and max steps.

### `POST /reset`
Start a new episode.
```json
{"task_name": "db_connection_failure", "seed": 42}
```
Returns the initial `IncidentResponseObservation`.

### `POST /step`
Execute one action.
```json
{
  "action_type": "investigate_logs",
  "service_name": "user-api",
  "parameters": {"keyword": "connection"}
}
```
Returns `{observation, reward, done, info}`.

### `GET /score`
Current 6D score breakdown including `failure_type` and `observation_loop` flag.

### `POST /grader`
Standalone grader compatible with external trajectory evaluators. Returns the full 6D breakdown with weighted components.

### `POST /baseline`
Runs the built-in heuristic agent over all tasks and returns per-task scores. Useful for sanity-checking the reward signal after scenario changes.

### `GET /state`
Full internal environment state snapshot for checkpointing.

---

## Python Client

```python
from client import IncidentResponseClient

client = IncidentResponseClient("http://localhost:7860")
client.wait_for_server()

obs = client.reset(task_name="db_connection_failure", seed=42)
print(obs["observation"]["incident_summary"])

result = client.step(
    action_type="check_service_health",
    service_name="user-api",
)
print(result["reward"], result["observation"]["feedback"])
```

---

## Running Inference

`inference.py` runs a configured LLM against all 6 tasks and prints a per-task score report:

```bash
export HF_TOKEN=your_token
export MODEL_NAME=meta-llama/Llama-3-70b-instruct
python inference.py
```

The inference script:
- Uses structured `[START]` / `[STEP]` / `[END]` logging
- Integrates per-step feedback into the prompt context
- Has an 18-minute timeout guard per task
- Reports success thresholds: >0.5 for alert_triage, >0.3 for standard tasks

---

## Codebase Architecture

```
meta-hackathon/
├── server/
│   ├── app.py              # Entry point: FastAPI + Gradio wiring
│   ├── state.py            # Shared IncidentResponseEnv singletons
│   ├── env.py              # Core RL environment logic
│   ├── env_rubric_patch.py # Optional rubric-based grading extension
│   ├── api/
│   │   ├── routes.py       # All HTTP endpoint handlers
│   │   └── models.py       # Pydantic request/response models
│   └── ui/
│       ├── layout.py       # Gradio Blocks definition
│       ├── callbacks.py    # Event handlers (gr_reset, gr_step, ...)
│       ├── renderers.py    # Pure HTML/Markdown rendering functions
│       ├── constants.py    # Shared display constants
│       └── styles.py       # CSS + header HTML
├── scenarios/
│   ├── definitions.py      # ScenarioDef + 5 scenario builders
│   ├── scenario_classes.py # Service health simulation
│   ├── base_scenario.py    # RubricCheck grader base
│   └── alert_triage.py     # Alert triage task definitions
├── models.py               # IncidentResponseAction/Observation/State
├── graders.py              # Standalone 6D deterministic grader
├── client.py               # HTTP client helper
├── inference.py            # LLM inference runner
├── openenv.yaml            # OpenEnv specification
└── Dockerfile
```

### Design Principles

- **Separation of concerns** — API routes, UI callbacks, rendering, and environment logic live in separate modules. No file has more than one responsibility.
- **Pure renderers** — All `render_*` functions in `renderers.py` take data and return HTML/Markdown strings. They have no side effects and are independently testable.
- **Shared state via `state.py`** — Both API routes and UI callbacks import the same `env` singleton, so browser actions and API calls affect the same environment instance.
- **Episode guards** — The `gr_step` callback checks `env._done` before executing. Once an episode ends, further clicks are rejected gracefully (UI button is disabled, clear message shown) rather than silently appending phantom rows.
- **OpenEnv compliance** — Scores are clamped to open interval (0.01, 0.99). The `/grader` endpoint is compatible with external trajectory evaluators.

---

## Sandbox UI Guide

The web sandbox at `/web` has four tabs:

| Tab | Purpose |
|-----|---------|
| **overview** | Workflow summary, scoring table, scenario list |
| **walkthrough** | Annotated full solve of DB Connection Failure (score 0.89) |
| **faq** | Common questions about scoring mechanics |
| **sandbox** | Interactive episode runner |

### Sandbox layout

**Left column (controls)**
- Episode Setup: task selector, optional seed, Reset button
- Episode State: live metric cards (step, reward, diag streak, collateral)
- Action Controls: action type + target service dropdowns
- Action Parameters: keyword, config key/value, replicas, root cause text, severity

**Right column (displays)**
- Environment Observation: active alerts, service health table, degradation warnings
- Step Detail Card: per-step reward, feedback callout, action output log, service snapshot, warnings
- Action History: table with step, action type badge, service, per-step reward, cumulative reward
- 6D Score: dimension breakdown with progress bars (shown after clicking Grade)

### Episode lifecycle

1. Select a task → click **Reset Environment**
2. Choose an action + target service → click **Execute Action**
3. Read the Step Detail card: feedback tells you what reward was earned/lost and why
4. Repeat until the episode ends (button auto-disables) or click **Grade (6D)** at any point
5. Click **Reset Environment** to start a new episode with a fresh or different task

---

## Reward Signal Design

The reward signal is designed to be **dense** (feedback every step), **honest** (no sparse terminal-only rewards), and **multi-dimensional** (six independently interpretable components):

```
Step reward = investigation_credit      # +0.04 per new service explored
           + root_cause_bonus           # +0.20 per matched root cause
           + remediation_bonus          # +0.15 per correct fix applied
           + efficiency_bonus           # scales with steps remaining
           + time_bonus                 # +0.10 if resolved in <50% of budget
           - safety_penalty             # -0.25 per destructive action on healthy svc
           - observation_loop_penalty   # -0.08 / -0.03 per loop detection
           - blind_fix_penalty          # -0.05 per fix before gated investigation
```

Per-step feedback is generated in plain English and returned in every observation, enabling agents to self-correct without external tooling.

---

## Contributing

1. **New scenarios** — Add a builder function in `scenarios/definitions.py` and register it in `get_scenario()` and `ScenarioFactory.generate()`.
2. **New action types** — Add to `ActionType` enum in `models.py`, handle in `env.py`'s `step()`, and add display metadata in `ui/constants.py`.
3. **New scoring dimensions** — Add computation in `graders.py` and a row in `ui/renderers.py`'s `render_score()`.
4. **UI components** — All rendering is in `ui/renderers.py`. Each `render_*` function is independent and easy to modify.

---

## Version History

| Version | Changes |
|---------|---------|
| **4.0.0** | Rich Gradio UI with step detail cards, action history table with cumulative reward, episode-done guard (button disabled after last step), 6D score panel; alert_triage task; procedural generation; modular server package structure |
| **3.0** | Sequence scoring (6th dimension); observation loop detection + hard cap; diagnosis gate enforcement; failure type classification |
| **2.1** | Per-step feedback strings; dynamic system degradation; service alias normalisation; standalone `graders.py` |
| **2.0** | Service alias normalisation; gated information exposure; multi-service scenarios |
| **1.0** | Initial release: single-service DB failure scenario; basic 5D scoring |

---

## License

Apache 2.0 — see `LICENSE` for details.

Built for the Meta Hackathon · Powered by [OpenEnv](https://openenv.dev)
