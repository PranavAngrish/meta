"""
Pydantic models for the Incident Response OpenEnv environment.
Defines typed Action, Observation, and State models following the OpenEnv spec.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class ActionType(str, Enum):
    """All available agent actions."""
    INVESTIGATE_LOGS = "investigate_logs"
    CHECK_METRICS = "check_metrics"
    READ_CONFIG = "read_config"
    CHECK_SERVICE_HEALTH = "check_service_health"
    RUN_DIAGNOSTIC = "run_diagnostic"
    RESTART_SERVICE = "restart_service"
    UPDATE_CONFIG = "update_config"
    ROLLBACK_DEPLOYMENT = "rollback_deployment"
    SCALE_SERVICE = "scale_service"
    DECLARE_ROOT_CAUSE = "declare_root_cause"


class AlertSeverity(str, Enum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    RESOLVED = "resolved"


class Alert(BaseModel):
    """An active alert in the system."""
    alert_id: str
    severity: AlertSeverity
    service: str
    message: str
    timestamp: str


class ServiceStatus(BaseModel):
    """Current status of a service."""
    name: str
    healthy: bool
    response_time_ms: Optional[float] = None
    error_rate: Optional[float] = None
    uptime_seconds: Optional[int] = None


class IncidentResponseAction(BaseModel):
    """An action the agent takes in the environment."""
    action_type: ActionType = Field(..., description="The type of action to perform")
    service_name: Optional[str] = Field(None, description="Target service for the action")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Additional parameters")


class IncidentResponseObservation(BaseModel):
    """What the agent observes after taking an action."""
    action_result: str = Field(..., description="Text result of the last action")
    action_success: bool = Field(True, description="Whether the action succeeded")
    active_alerts: List[Alert] = Field(default_factory=list)
    service_statuses: List[ServiceStatus] = Field(default_factory=list)
    available_services: List[str] = Field(default_factory=list)
    available_actions: List[str] = Field(default_factory=list)
    step_number: int = Field(0)
    max_steps: int = Field(30)
    elapsed_time_minutes: int = Field(0)
    incident_summary: str = Field("")
    task_name: str = Field("")
    task_difficulty: str = Field("")


class IncidentResponseState(BaseModel):
    """Full internal state for checkpointing."""
    task_name: str
    task_difficulty: str
    step_number: int
    elapsed_time_minutes: int
    services: Dict[str, Any]
    alerts: List[Dict[str, Any]]
    root_causes: List[str]
    agent_findings: List[str]
    agent_actions_taken: List[str]
    remediation_applied: List[str]
    correct_remediations: List[str]
    incident_resolved: bool
    cumulative_reward: float
    done: bool
    max_steps: int


class IncidentResponseReward(BaseModel):
    """Reward signal returned after each step."""
    reward: float = Field(..., ge=-1.0, le=1.0)
    cumulative_reward: float = Field(...)
    breakdown: Dict[str, float] = Field(default_factory=dict)
