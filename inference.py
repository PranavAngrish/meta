"""
Inference Script — Incident Response Environment
=================================================
MANDATORY FORMAT — DO NOT CHANGE STDOUT STRUCTURE

Uses OpenAI Client via HuggingFace Router.
Runs all 3 tasks (easy -> medium -> hard) and reports scores.

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

# ── Configuration ─────────────────────────────────────────────────────────

IMAGE_NAME = os.getenv("IMAGE_NAME", os.getenv("LOCAL_IMAGE_NAME", "incident-response-env"))
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
BENCHMARK = "incident_response"
MAX_STEPS = 20
TEMPERATURE = 0.7
ENV_URL = os.getenv("ENV_URL", "http://localhost:7860")

TASKS = [
    "db_connection_failure",        # easy
    "cascading_service_timeout",    # medium
    "multi_factor_outage",          # hard
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

    def reset(self, task_name: str) -> dict:
        r = requests.post(f"{self.base_url}/reset", json={"task_name": task_name}, timeout=30)
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
        return r.json().get("score", 0.0)


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
    {"action_type": "<type>", "service_name": "<name>", "parameters": {<params>}}
    
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
    """)


def parse_llm_action(text: str) -> dict:
    """Extract JSON action from LLM response."""
    text = text.strip()
    # Try to find JSON in the response
    # Remove markdown code blocks if present
    text = re.sub(r'```json\s*', '', text)
    text = re.sub(r'```\s*', '', text)
    text = text.strip()

    # Try direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Try to find JSON object in text
    match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', text)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    # Fallback: investigate logs of the first available service
    return {"action_type": "check_service_health", "service_name": "user-api", "parameters": {}}


def run_task(task_name: str, client: OpenAI, env: EnvClient):
    """Run a single task and return (success, steps, score, rewards)."""
    rewards: List[float] = []
    last_error: Optional[str] = None

    print(f"[START] task={task_name} env={BENCHMARK} model={MODEL_NAME}")

    try:
        # Reset environment
        reset_resp = env.reset(task_name)
        obs = reset_resp.get("observation", {})
        action_result = obs.get("action_result", "")
        available_services = obs.get("available_services", [])
        done = False

        messages = [
            {"role": "system", "content": build_system_prompt()},
            {"role": "user", "content": f"INCIDENT REPORT:\n{action_result}"},
        ]

        step_num = 0
        while not done and step_num < MAX_STEPS:
            step_num += 1

            # Get LLM action
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

            # Parse action
            action = parse_llm_action(llm_text)
            action_type = action.get("action_type", "check_service_health")
            service_name = action.get("service_name")
            parameters = action.get("parameters", {})

            # Execute step
            try:
                step_resp = env.step(action_type, service_name, parameters)
                reward = step_resp.get("reward", 0.0)
                done = step_resp.get("done", False)
                step_obs = step_resp.get("observation", {})
                step_info = step_resp.get("info", {})
                last_error = step_info.get("last_action_error")
                action_result = step_obs.get("action_result", "")
            except Exception as e:
                reward = 0.0
                done = False
                action_result = f"Error: {e}"
                last_error = str(e)

            rewards.append(reward)

            # Build action string for logging
            action_str = action_type
            if service_name:
                action_str += f"({service_name})"

            error_str = last_error if last_error else "null"
            print(f"[STEP] step={step_num} action={action_str} reward={reward:.2f} done={'true' if done else 'false'} error={error_str}")

            # Update conversation for next turn
            messages.append({"role": "assistant", "content": llm_text})
            messages.append({"role": "user", "content": f"OBSERVATION:\n{action_result}\n\nStep {step_num}/{MAX_STEPS}. {'Episode done.' if done else 'Choose your next action.'}"})

        # Get final score
        score = env.get_score()
        success = score > 0.3  # consider >0.3 a success

        rewards_str = ",".join(f"{r:.2f}" for r in rewards)
        print(f"[END] success={'true' if success else 'false'} steps={step_num} score={score:.2f} rewards={rewards_str}")

        return success, step_num, score, rewards

    except Exception as e:
        error_msg = str(e)
        print(f"[STEP] step=1 action=error reward=0.00 done=true error={error_msg}")
        print(f"[END] success=false steps=1 score=0.00 rewards=0.00")
        return False, 1, 0.0, [0.0]


# ── Docker Management ─────────────────────────────────────────────────────

def start_docker():
    """Start the environment Docker container."""
    # Stop any existing container
    subprocess.run(["docker", "rm", "-f", "incident-response-env"], capture_output=True)
    time.sleep(1)

    # Start container
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
    """Stop the environment Docker container."""
    subprocess.run(["docker", "rm", "-f", "incident-response-env"], capture_output=True)


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    # Initialize OpenAI client (via HF Router)
    client = OpenAI(
        api_key=API_KEY,
        base_url=API_BASE_URL,
    )

    # Start Docker container
    use_docker = os.getenv("USE_DOCKER", "true").lower() == "true"
    if use_docker:
        try:
            start_docker()
        except Exception as e:
            print(f"Warning: Could not start Docker: {e}", file=sys.stderr)

    # Connect to environment
    env = EnvClient(ENV_URL)
    if not env.wait_ready(timeout=120):
        print("ERROR: Environment server not ready", file=sys.stderr)
        # Print required [END] for each task
        for task in TASKS:
            print(f"[START] task={task} env={BENCHMARK} model={MODEL_NAME}")
            print(f"[STEP] step=1 action=error reward=0.00 done=true error=server_not_ready")
            print(f"[END] success=false steps=1 score=0.00 rewards=0.00")
        sys.exit(1)

    # Run all tasks
    results = []
    for task_name in TASKS:
        success, steps, score, rewards = run_task(task_name, client, env)
        results.append({"task": task_name, "success": success, "steps": steps, "score": score})

    # Cleanup
    if use_docker:
        stop_docker()

    # Summary
    print(f"\n{'='*60}", file=sys.stderr)
    print(f"SUMMARY", file=sys.stderr)
    print(f"{'='*60}", file=sys.stderr)
    for r in results:
        print(f"  {r['task']}: score={r['score']:.2f} steps={r['steps']} success={r['success']}", file=sys.stderr)
    avg_score = sum(r["score"] for r in results) / len(results)
    print(f"  Average: {avg_score:.2f}", file=sys.stderr)


if __name__ == "__main__":
    main()
