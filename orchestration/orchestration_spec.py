"""
Orchestration Specification for AI Trading System

This module defines the core data structures and enums for the orchestration system.
It provides the foundation for workflow management, task scheduling, and event handling.
"""

from typing import Dict, List, Optional, Callable
from datetime import datetime
from enum import Enum
from dataclasses import dataclass, field

# Workflow Stages
class WorkflowStage(Enum):
    """Defines the stages of a trading workflow"""
    PRE_MARKET = "pre_market"
    MARKET_OPEN = "market_open"
    INTRADAY = "intraday"
    POST_MARKET = "post_market"
    CONTINUOUS = "continuous"

# Task Status
class TaskStatus(Enum):
    """Defines possible states for workflow tasks"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    RETRYING = "retrying"

# Task Priority
class TaskPriority(Enum):
    """Defines priority levels for task execution"""
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4

# Orchestration Events
class OrchestrationEvent(Enum):
    """Defines system events for monitoring and alerting"""
    WORKFLOW_STARTED = "workflow_started"
    WORKFLOW_COMPLETED = "workflow_completed"
    WORKFLOW_FAILED = "workflow_failed"
    STAGE_STARTED = "stage_started"
    STAGE_COMPLETED = "stage_completed"
    TASK_STARTED = "task_started"
    TASK_COMPLETED = "task_completed"
    TASK_FAILED = "task_failed"
    AGENT_ERROR = "agent_error"
    SYSTEM_ALERT = "system_alert"
    MARKET_OPEN_BELL = "market_open_bell"
    MARKET_CLOSE_BELL = "market_close_bell"

@dataclass
class WorkflowTask:
    """
    Individual task within the workflow
    
    Represents a single unit of work to be executed by an agent
    with support for dependencies, retries, and timeout handling.
    """
    task_id: str
    agent_name: str
    task_type: str
    task_data: Dict
    priority: TaskPriority
    stage: WorkflowStage
    dependencies: List[str]
    timeout_seconds: int
    retry_count: int = 0
    max_retries: int = 3
    status: TaskStatus = TaskStatus.PENDING
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    result: Optional[Dict] = None
    error: Optional[str] = None

@dataclass
class WorkflowExecution:
    """
    Complete workflow execution instance
    
    Tracks the state and progress of an entire workflow execution,
    including all tasks, results, and errors.
    """
    execution_id: str
    workflow_date: datetime
    current_stage: WorkflowStage
    tasks: Dict[str, WorkflowTask]
    execution_state: Dict
    started_at: datetime
    completed_at: Optional[datetime] = None
    status: str = "running"
    errors: List[Dict] = field(default_factory=list)