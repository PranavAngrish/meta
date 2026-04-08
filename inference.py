"""
Inference Script — Incident Response Environment
=================================================
MANDATORY FORMAT — DO NOT CHANGE STDOUT STRUCTURE

Uses OpenAI Client via HuggingFace Router.
Runs all 6 tasks (alert_triage + easy -> medium -> hard) and reports scores.

v3.0 changes:
  - Added alert_triage task (P1/P2/P3/P4 severity classification, 3-step budget)
  - Dedicated build_alert_triage_prompt() system prompt for triage task
  - task_max_steps routes 3 steps for alert_triage, MAX_STEPS for all others
  - Success threshold: >0.5 for alert_triage, >0.3 for standard tasks

v2.1 changes:
  - Timeout guard: stops gracefully approaching 18-minute HF runner limit
  - Runs all 5 tasks (added ssl_certificate_expiry + database_deadlock)
  - Includes feedback from observation in LLM context

STDOUT FORMAT:
  [START] task=<task_name> env=<benchmark> model=<model_name>
  [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
  [END]   success=<true|false> steps=<n> score=<score> rewards=<r1,r2,...,rn>
"""

import json
import os
import re
import subprocess
import sys
import textwrap
import time
import traceback
from typing import List, Optional

import requests
from openai import OpenAI

# ── Timeout guard ─────────────────────────────────────────────────────────
# HF runner hard-kills at 20 min. Stop at 18 min to emit clean [END] lines.
_SCRIPT_START = time.time()
_MAX_RUNTIME_SECONDS = 18 * 60  # 18 minutes


def _check_timeout():
    """Raise RuntimeError if we're approaching the HF runner time limit."""
    elapsed = time.time() - _SCRIPT_START
    if elapsed > _MAX_RUNTIME_SECONDS:
        raise RuntimeError(
            f"Approaching 20-min HF runner limit (elapsed={elapsed:.0f}s) — stopping early"
        )


# ── Configuration ─────────────────────────────────────────────────────────

IMAGE_NAME = os.getenv("IMAGE_NAME", os.getenv("LOCAL_IMAGE_NAME", "incident-response-env"))
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
BENCHMARK = "incident_response"
MAX_STEPS = 20
TEMPERATURE = 0.7
ENV_URL = os.getenv("ENV_URL", "http://localhost:7860")

# All 6 tasks — easy → medium → hard, plus alert triage
TASKS = [
    "alert_triage",             # easy (triage)
    "db_connection_failure",    # easy
    "cascading_service_timeout",# medium
    "ssl_certificate_expiry",   # medium
    "multi_factor_outage",      # hard
    "database_deadlock",        # hard
]


# ── Environment Client ────────────────────────────────────────────────────

class EnvClient:
    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")

    def wait_ready(self, timeout=120):
        start = time.time()
        while time.time() - start < timeout:
            try:
                r = requests.get(f"{self.base_url}/health", timeout=5)
                if r.status_code == 200:
                    return True
            except Exception:
                pass
            time.sleep(2)
        return False

    def reset(self, task_name: str, seed: Optional[int] = None) -> dict:
        payload: dict = {"task_name": task_name}
        if seed is not None:
            payload["seed"] = seed
        r = requests.post(f"{self.base_url}/reset", json=payload, timeout=30)
        r.raise_for_status()
        return r.json()

    def step(self, action_type: str, service_name: str = None, parameters: dict = None) -> dict:
        payload = {"action_type": action_type, "service_name": service_name, "parameters": parameters or {}}
        r = requests.post(f"{self.base_url}/step", json=payload, timeout=30)
        r.raise_for_status()
        return r.json()

    def get_score(self) -> float:
        r = requests.get(f"{self.base_url}/score", timeout=10)
        r.raise_for_status()
        data = r.json()
        # Score is already clamped to (0.01, 0.99) by the server
        return float(data.get("score", 0.01))


# ── LLM Agent ────────────────────────────────────────────────────────────

def build_system_prompt():
    return textwrap.dedent("""\
    You are an expert SRE (Site Reliability Engineer) responding to a production incident.
    
    You must investigate the incident by examining services, then identify the root cause(s)
    and apply the correct remediations.
    
    WORKFLOW:
    1. Start by checking service health and reading logs for the alerting services
    2. Investigate metrics and configs for suspicious services
    3. Run diagnostics on services that seem problematic
    4. Once you identify the root cause, declare it
    5. Apply the correct remediation (update config, restart, rollback, or scale)
    
    RESPOND WITH EXACTLY ONE JSON ACTION per turn. No extra text. Just the JSON.
    
    Action format:
    {"action_type": "<type>", "service_name": "<n>", "parameters": {<params>}}
    
    Available action_type values:
    - investigate_logs: params={"keyword": "<optional filter>"}
    - check_metrics: params={"metric_type": "all"}
    - read_config: no extra params needed
    - check_service_health: no extra params needed
    - run_diagnostic: no extra params needed
    - restart_service: no extra params needed
    - update_config: params={"key": "<config_key>", "value": "<new_value>"}
    - rollback_deployment: no extra params needed
    - scale_service: params={"replicas": "<count>"}
    - declare_root_cause: service_name not needed, params={"cause": "<description>"}
    
    IMPORTANT RULES:
    - Always respond with ONLY valid JSON, no markdown, no explanation
    - First investigate, then diagnose, then remediate
    - For declare_root_cause, you don't need service_name
    - Be specific in root cause declarations (mention the exact service and issue)
    - The per-step FEEDBACK in the observation tells you what reward you earned last step — use it
    """)


def build_alert_triage_prompt():
    """Specialised system prompt for the alert_triage severity-classification task."""
    return textwrap.dedent("""\
    You are an on-call SRE classifying the severity of a production alert.
    You have at most 3 steps. Use them wisely.

    SEVERITY SCALE:
      P1 - CRITICAL: Complete outage or revenue impact >$1,000/min. Core flow (checkout, login) unavailable.
      P2 - HIGH: Major degradation. Most users affected. Revenue impact present. Core flow degraded but not down.
      P3 - MEDIUM: Partial/minor issue. Graceful fallback active. Limited or zero revenue impact.
      P4 - LOW: Informational. No measurable user or revenue impact.

    KEY INSIGHT - high error rate does NOT automatically mean P1:
      - If a service has graceful fallback AND checkout still works -> P3 or P2, NOT P1.
      - If revenue impact is $0 -> P3 or P4, regardless of error rate.
      - Always ask: Is checkout working? Is login working? What is the revenue impact?

    WORKFLOW (3 steps max):
      Step 1: investigate_logs or check_metrics on the primary alerting service.
      Step 2: check_service_health or check_metrics on a second service if impact is unclear.
      Step 3: submit_severity with your final classification.

    RESPOND WITH EXACTLY ONE JSON ACTION per turn. No extra text. Just the JSON.

    Action format:
    {"action_type": "<type>", "service_name": "<n>", "parameters": {<params>}}

    Available actions (alert_triage only):
      investigate_logs: requires service_name, optional params={"keyword": "<filter>"}
      check_metrics: requires service_name, no params
      check_service_health: requires service_name, no params
      run_diagnostic: requires service_name, no params
      submit_severity: NO service_name, params={"severity": "P1|P2|P3|P4"}

    IMPORTANT: submit_severity does NOT take a service_name. Example:
      {"action_type": "submit_severity", "service_name": null, "parameters": {"severity": "P2"}}

    The per-step FEEDBACK in the observation tells you what reward you earned.
    """)


def parse_llm_action(text: str) -> dict:
    """Extract JSON action from LLM response."""
    text = text.strip()
    text = re.sub(r'```json\s*', '', text)
    text = re.sub(r'```\s*', '', text)
    text = text.strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', text)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    return {"action_type": "check_service_health", "service_name": "user-api", "parameters": {}}


def run_task(task_name: str, client: OpenAI, env: EnvClient) -> tuple:
    """Run a single task. Returns (success, steps, score, rewards)."""
    rewards: List[float] = []
    last_error: Optional[str] = None
    is_alert_triage = (task_name == "alert_triage")
    task_max_steps = 3 if is_alert_triage else MAX_STEPS

    print(f"[START] task={task_name} env={BENCHMARK} model={MODEL_NAME}")

    try:
        _check_timeout()

        reset_resp = env.reset(task_name)
        obs = reset_resp.get("observation", {})
        action_result = obs.get("action_result", "")
        available_services = obs.get("available_services", [])
        done = False

        messages = [
            {"role": "system", "content": build_alert_triage_prompt() if is_alert_triage else build_system_prompt()},
            {"role": "user", "content": f"INCIDENT REPORT:\n{action_result}"},
        ]

        step_num = 0
        while not done and step_num < task_max_steps:
            _check_timeout()
            step_num += 1

            try:
                response = client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=messages,
                    temperature=TEMPERATURE,
                    max_tokens=500,
                )
                llm_text = response.choices[0].message.content or ""
            except Exception as e:
                llm_text = json.dumps({
                    "action_type": "check_service_health",
                    "service_name": available_services[0] if available_services else "user-api",
                    "parameters": {}
                })
                last_error = str(e)

            action = parse_llm_action(llm_text)
            action_type = action.get("action_type", "check_service_health")
            service_name = action.get("service_name")
            parameters = action.get("parameters", {})

            try:
                step_resp = env.step(action_type, service_name, parameters)
                reward = step_resp.get("reward", 0.0)
                done = step_resp.get("done", False)
                step_obs = step_resp.get("observation", {})
                step_info = step_resp.get("info", {})
                last_error = step_info.get("last_action_error")
                action_result = step_obs.get("action_result", "")
                # Include per-step feedback in next LLM context
                step_feedback = step_obs.get("feedback", "")
            except Exception as e:
                reward = 0.0
                done = False
                action_result = f"Error: {e}"
                last_error = str(e)
                step_feedback = ""

            rewards.append(reward)

            action_str = action_type
            if service_name:
                action_str += f"({service_name})"

            error_str = last_error if last_error else "null"
            print(f"[STEP] step={step_num} action={action_str} reward={reward:.2f} done={'true' if done else 'false'} error={error_str}")

            messages.append({"role": "assistant", "content": llm_text})
            obs_msg = f"OBSERVATION:\n{action_result}"
            if step_feedback:
                obs_msg += f"\n\nFEEDBACK: {step_feedback}"
            obs_msg += f"\n\nStep {step_num}/{task_max_steps}. {'Episode done.' if done else 'Choose your next action.'}"
            messages.append({"role": "user", "content": obs_msg})

        score = env.get_score()
        success = score > (0.5 if is_alert_triage else 0.3)

        rewards_str = ",".join(f"{r:.2f}" for r in rewards)
        print(f"[END] success={'true' if success else 'false'} steps={step_num} score={score:.2f} rewards={rewards_str}")

        return success, step_num, score, rewards

    except RuntimeError as e:
        # Timeout — emit clean END line so evaluator can score what we have
        if "runner limit" in str(e) or "Approaching" in str(e):
            print(f"[STEP] step=0 action=timeout reward=0.00 done=true error=timeout", file=sys.stderr)
        print(f"[END] success=false steps=0 score=0.01 rewards=0.00")
        raise  # re-raise so main() can stop the loop

    except Exception as e:
        error_msg = str(e).replace("\n", " ")
        print(f"[STEP] step=1 action=error reward=0.00 done=true error={error_msg}")
        print(f"[END] success=false steps=1 score=0.01 rewards=0.00")
        return False, 1, 0.01, [0.0]


# ── Docker Management ─────────────────────────────────────────────────────

def start_docker():
    subprocess.run(["docker", "rm", "-f", "incident-response-env"], capture_output=True)
    time.sleep(1)
    result = subprocess.run(
        ["docker", "run", "-d", "--name", "incident-response-env",
         "-p", "7860:7860", IMAGE_NAME],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        print(f"Docker start failed: {result.stderr}", file=sys.stderr)
        raise RuntimeError(f"Failed to start Docker: {result.stderr}")
    return True


def stop_docker():
    subprocess.run(["docker", "rm", "-f", "incident-response-env"], capture_output=True)


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    client = OpenAI(api_key=API_KEY, base_url=API_BASE_URL)

    use_docker = os.getenv("USE_DOCKER", "true").lower() == "true"
    if use_docker:
        try:
            start_docker()
        except Exception as e:
            print(f"Warning: Could not start Docker: {e}", file=sys.stderr)

    env = EnvClient(ENV_URL)
    if not env.wait_ready(timeout=120):
        print("ERROR: Environment server not ready", file=sys.stderr)
        for task in TASKS:
            print(f"[START] task={task} env={BENCHMARK} model={MODEL_NAME}")
            print(f"[STEP] step=1 action=error reward=0.00 done=true error=server_not_ready")
            print(f"[END] success=false steps=1 score=0.01 rewards=0.00")
        sys.exit(1)

    results = []
    for task_name in TASKS:
        try:
            success, steps, score, rewards = run_task(task_name, client, env)
            results.append({"task": task_name, "success": success, "steps": steps, "score": score})
        except RuntimeError as e:
            if "Approaching" in str(e) or "runner limit" in str(e):
                print(f"Timeout reached after {task_name} — stopping early", file=sys.stderr)
                break
            raise

    if use_docker:
        stop_docker()

    print(f"\n{'='*60}", file=sys.stderr)
    print(f"SUMMARY", file=sys.stderr)
    print(f"{'='*60}", file=sys.stderr)
    for r in results:
        print(f"  {r['task']}: score={r['score']:.2f} steps={r['steps']} success={r['success']}", file=sys.stderr)
    if results:
        avg_score = sum(r["score"] for r in results) / len(results)
        print(f"  Average: {avg_score:.2f}", file=sys.stderr)


if __name__ == "__main__":
    main()
