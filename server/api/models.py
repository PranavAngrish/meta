"""Pydantic request/response models for the Incident Response HTTP API."""
from __future__ import annotations

from typing import Any, Dict, Optional
from pydantic import BaseModel


class ResetRequest(BaseModel):
    task_name: str = "db_connection_failure"
    seed: Optional[int] = None


class StepRequest(BaseModel):
    action_type: str
    service_name: Optional[str] = None
    parameters: Dict[str, Any] = {}


class StepResponse(BaseModel):
    observation: dict
    reward: float
    done: bool
    info: dict
