"""
HTTP client for the Incident Response OpenEnv environment.
"""
from __future__ import annotations
import time
from typing import Any, Dict, Optional
import requests


class IncidentResponseClient:
    def __init__(self, base_url: str = "http://localhost:7860"):
        self.base_url = base_url.rstrip("/")
        self._wait_for_server()

    def _wait_for_server(self, timeout: int = 60):
        start = time.time()
        while time.time() - start < timeout:
            try:
                r = requests.get(f"{self.base_url}/health", timeout=5)
                if r.status_code == 200:
                    return
            except requests.ConnectionError:
                pass
            time.sleep(2)
        raise TimeoutError(f"Server at {self.base_url} not ready after {timeout}s")

    def reset(self, task_name: str = "db_connection_failure") -> dict:
        r = requests.post(f"{self.base_url}/reset", json={"task_name": task_name})
        r.raise_for_status()
        return r.json()

    def step(self, action_type: str, service_name: Optional[str] = None,
             parameters: Optional[Dict[str, Any]] = None) -> dict:
        payload = {"action_type": action_type, "service_name": service_name, "parameters": parameters or {}}
        r = requests.post(f"{self.base_url}/step", json=payload)
        r.raise_for_status()
        return r.json()

    def state(self) -> dict:
        r = requests.get(f"{self.base_url}/state")
        r.raise_for_status()
        return r.json()

    def get_score(self) -> float:
        r = requests.get(f"{self.base_url}/score")
        r.raise_for_status()
        return r.json()["score"]

    def tasks(self) -> list:
        r = requests.get(f"{self.base_url}/tasks")
        r.raise_for_status()
        return r.json()["tasks"]

    def health(self) -> dict:
        r = requests.get(f"{self.base_url}/health")
        r.raise_for_status()
        return r.json()

    def close(self):
        pass
