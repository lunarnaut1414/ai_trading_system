"""
Workflow Engine for AI Trading System

Core workflow execution engine that manages task scheduling, execution,
and inter-agent communication. Handles dependencies, retries, and concurrent execution.
"""

import asyncio
import uuid
import logging
from typing import Dict, List, Any, Callable
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor

from orchestration.orchestration_spec import (
    WorkflowTask, WorkflowExecution, WorkflowStage, TaskStatus, 
    TaskPriority, OrchestrationEvent
)

class WorkflowEngine:
    """
    Core workflow execution engine for agent orchestration
    Manages task scheduling, execution, and inter-agent communication
    """
    
    def __init__(self, agents: Dict[str, Any], config: Any):
        self.agents = agents
        self.config = config
        self.logger = logging.getLogger('workflow_engine')
        
        # Workflow state management
        self.active_executions: Dict[str, WorkflowExecution] = {}
        self.completed_executions: List[WorkflowExecution] = []
        self.event_handlers: Dict[OrchestrationEvent, List[Callable]] = {}
        
        # Execution settings
        self.max_concurrent_tasks = getattr(config, 'MAX_CONCURRENT_TASKS', 5)
        self.default_task_timeout = getattr(config, 'DEFAULT_TASK_TIMEOUT', 300)
        self.workflow_timeout = getattr(config, 'WORKFLOW_TIMEOUT', 3600)
        
        # Task queues by priority
        self.task_queues = {
            TaskPriority.CRITICAL: asyncio.Queue(),
            TaskPriority.HIGH: asyncio.Queue(),
            TaskPriority.MEDIUM: asyncio.Queue(),
            TaskPriority.LOW: asyncio.Queue()
        }
        
        # Running tasks tracking
        self.running_tasks: Dict[str, asyncio.Task] = {}
        self.task_results: Dict[str, Dict] = {}
        
        # Execution thread pool
        self.executor = ThreadPoolExecutor(max_workers=self.max_concurrent_tasks)
        
        self.logger.info(f"✅ Workflow Engine initialized with {len(agents)} agents")
    
    async def create_workflow(self, workflow_definition: Dict) -> str:
        """Create new workflow execution from definition"""
        
        execution_id = f"workflow_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
        
        execution = WorkflowExecution(
            execution_id=execution_id,
            workflow_date=datetime.now(),
            current_stage=WorkflowStage.PRE_MARKET,
            tasks={},
            execution_state={
                'workflow_definition': workflow_definition,
                'agent_outputs': {},
                'stage_results': {},
                'system_alerts': []
            },
            started_at=datetime.now()
        )
        
        # Create tasks from workflow definition
        await self._create_tasks_from_definition(execution, workflow_definition)
        
        # Store execution
        self.active_executions[execution_id] = execution
        
        # Emit workflow started event
        await self._emit_event(OrchestrationEvent.WORKFLOW_STARTED, {
            'execution_id': execution_id,
            'total_tasks': len(execution.tasks)
        })
        
        self.logger.info(f"Created workflow {execution_id} with {len(execution.tasks)} tasks")
        return execution_id
    
    async def execute_workflow(self, execution_id: str) -> Dict:
        """Execute a workflow by ID"""
        
        execution = self.active_executions.get(execution_id)
        if not execution:
            raise ValueError(f"Workflow {execution_id} not found")
        
        self.logger.info(f"Starting execution of workflow {execution_id}")
        
        try:
            # Execute stages in order
            stages = [
                WorkflowStage.PRE_MARKET,
                WorkflowStage.MARKET_OPEN,
                WorkflowStage.INTRADAY,
                WorkflowStage.POST_MARKET
            ]
            
            for stage in stages:
                execution.current_stage = stage
                await self._execute_stage(execution, stage)
            
            # Mark workflow as completed
            execution.status = "completed"
            execution.completed_at = datetime.now()
            
            # Move to completed list
            self.completed_executions.append(execution)
            del self.active_executions[execution_id]
            
            # Generate summary
            summary = await self._generate_workflow_summary(execution)
            
            # Emit workflow completed event
            await self._emit_event(OrchestrationEvent.WORKFLOW_COMPLETED, {
                'execution_id': execution_id,
                'summary': summary
            })
            
            return summary
            
        except Exception as e:
            self.logger.error(f"Workflow {execution_id} failed: {str(e)}")
            execution.status = "failed"
            execution.errors.append({
                'timestamp': datetime.now().isoformat(),
                'error': str(e)
            })
            
            await self._emit_event(OrchestrationEvent.WORKFLOW_FAILED, {
                'execution_id': execution_id,
                'error': str(e)
            })
            
            raise
    
    async def _execute_stage(self, execution: WorkflowExecution, stage: WorkflowStage):
        """Execute all tasks in a stage"""
        
        self.logger.info(f"Executing stage: {stage.value}")
        
        # Get tasks for this stage
        stage_tasks = [
            task for task in execution.tasks.values()
            if task.stage == stage
        ]
        
        if not stage_tasks:
            self.logger.info(f"No tasks for stage {stage.value}")
            return
        
        # Emit stage started event
        await self._emit_event(OrchestrationEvent.STAGE_STARTED, {
            'execution_id': execution.execution_id,
            'stage': stage.value,
            'task_count': len(stage_tasks)
        })
        
        # Order tasks by dependencies
        ordered_batches = self._order_tasks_by_dependencies(stage_tasks)
        
        # Execute task batches
        for batch in ordered_batches:
            await self._execute_task_batch(execution, batch)
        
        # Store stage results
        stage_results = {
            task.task_id: task.result
            for task in stage_tasks
            if task.result is not None
        }
        execution.execution_state['stage_results'][stage.value] = stage_results
        
        # Emit stage completed event
        await self._emit_event(OrchestrationEvent.STAGE_COMPLETED, {
            'execution_id': execution.execution_id,
            'stage': stage.value,
            'results': stage_results
        })
    
    async def _execute_task_batch(self, execution: WorkflowExecution, tasks: List[WorkflowTask]):
        """Execute a batch of tasks concurrently"""
        
        self.logger.info(f"Executing batch of {len(tasks)} tasks")
        
        # Create concurrent task executions
        task_futures = []
        for task in tasks:
            future = asyncio.create_task(self._execute_single_task(execution, task))
            task_futures.append(future)
            self.running_tasks[task.task_id] = future
        
        # Wait for all tasks to complete
        results = await asyncio.gather(*task_futures, return_exceptions=True)
        
        # Process results
        for task, result in zip(tasks, results):
            if isinstance(result, Exception):
                task.status = TaskStatus.FAILED
                task.error = str(result)
                self.logger.error(f"Task {task.task_id} failed: {result}")
            else:
                task.status = TaskStatus.COMPLETED
                task.result = result
            
            # Remove from running tasks
            self.running_tasks.pop(task.task_id, None)
    
    async def _execute_single_task(self, execution: WorkflowExecution, task: WorkflowTask) -> Dict:
        """Execute a single task"""
        
        self.logger.info(f"Executing task {task.task_id} ({task.agent_name}.{task.task_type})")
        
        task.status = TaskStatus.RUNNING
        task.started_at = datetime.now()
        
        # Emit task started event
        await self._emit_event(OrchestrationEvent.TASK_STARTED, {
            'execution_id': execution.execution_id,
            'task_id': task.task_id,
            'agent': task.agent_name,
            'task_type': task.task_type
        })
        
        try:
            # Get agent
            agent = self.agents.get(task.agent_name)
            if not agent:
                raise ValueError(f"Agent {task.agent_name} not found")
            
            # Execute task with timeout
            method_name = f"execute_{task.task_type}"
            if not hasattr(agent, method_name):
                method_name = "analyze"  # Default method
            
            agent_method = getattr(agent, method_name)
            
            # Execute with timeout
            result = await asyncio.wait_for(
                agent_method(**task.task_data),
                timeout=task.timeout_seconds
            )
            
            task.completed_at = datetime.now()
            task.status = TaskStatus.COMPLETED
            task.result = result
            
            # Store agent output
            if task.agent_name not in execution.execution_state['agent_outputs']:
                execution.execution_state['agent_outputs'][task.agent_name] = {}
            execution.execution_state['agent_outputs'][task.agent_name][task.task_type] = result
            
            # Emit task completed event
            await self._emit_event(OrchestrationEvent.TASK_COMPLETED, {
                'execution_id': execution.execution_id,
                'task_id': task.task_id,
                'result': result
            })
            
            return result
            
        except asyncio.TimeoutError:
            error_msg = f"Task {task.task_id} timed out after {task.timeout_seconds}s"
            self.logger.error(error_msg)
            
            # Handle retry logic
            if task.retry_count < task.max_retries:
                task.retry_count += 1
                task.status = TaskStatus.RETRYING
                self.logger.info(f"Retrying task {task.task_id} (attempt {task.retry_count}/{task.max_retries})")
                return await self._execute_single_task(execution, task)
            
            task.status = TaskStatus.FAILED
            task.error = error_msg
            raise
            
        except Exception as e:
            error_msg = f"Task {task.task_id} failed: {str(e)}"
            self.logger.error(error_msg)
            
            # Handle retry logic
            if task.retry_count < task.max_retries:
                task.retry_count += 1
                task.status = TaskStatus.RETRYING
                self.logger.info(f"Retrying task {task.task_id} (attempt {task.retry_count}/{task.max_retries})")
                return await self._execute_single_task(execution, task)
            
            task.status = TaskStatus.FAILED
            task.error = error_msg
            
            await self._emit_event(OrchestrationEvent.TASK_FAILED, {
                'execution_id': execution.execution_id,
                'task_id': task.task_id,
                'error': str(e)
            })
            
            raise
    
    def _order_tasks_by_dependencies(self, tasks: List[WorkflowTask]) -> List[List[WorkflowTask]]:
        """Order tasks into batches based on dependencies"""
        
        # Create task map
        task_map = {task.task_id: task for task in tasks}
        
        # Track remaining tasks
        remaining_tasks = set(task_map.keys())
        completed_tasks = set()
        
        # Build execution batches
        ordered_batches = []
        
        while remaining_tasks:
            # Find tasks with satisfied dependencies
            ready_tasks = []
            for task_id in remaining_tasks:
                task = task_map[task_id]
                if all(dep in completed_tasks or dep not in task_map for dep in task.dependencies):
                    ready_tasks.append(task)
            
            if not ready_tasks:
                # Circular dependency or missing dependency
                self.logger.warning("Circular or missing dependency detected")
                ready_tasks = [task_map[task_id] for task_id in remaining_tasks]
            
            # Sort by priority within batch
            ready_tasks.sort(key=lambda t: t.priority.value, reverse=True)
            ordered_batches.append(ready_tasks)
            
            # Mark as completed for dependency purposes
            for task in ready_tasks:
                remaining_tasks.remove(task.task_id)
                completed_tasks.add(task.task_id)
        
        return ordered_batches
    
    async def _create_tasks_from_definition(self, execution: WorkflowExecution, definition: Dict):
        """Create tasks from workflow definition"""
        
        for stage_name, stage_config in definition.get('stages', {}).items():
            stage = WorkflowStage(stage_name)
            
            for task_config in stage_config.get('tasks', []):
                task_id = f"{stage_name}_{task_config['agent']}_{task_config['task_type']}_{uuid.uuid4().hex[:8]}"
                
                task = WorkflowTask(
                    task_id=task_id,
                    agent_name=task_config['agent'],
                    task_type=task_config['task_type'],
                    task_data=task_config.get('task_data', {}),
                    priority=TaskPriority(task_config.get('priority', TaskPriority.MEDIUM.value)),
                    stage=stage,
                    dependencies=task_config.get('dependencies', []),
                    timeout_seconds=task_config.get('timeout', self.default_task_timeout),
                    max_retries=task_config.get('max_retries', 3)
                )
                
                execution.tasks[task_id] = task
    
    async def _emit_event(self, event: OrchestrationEvent, data: Dict):
        """Emit workflow event to registered handlers"""
        
        handlers = self.event_handlers.get(event, [])
        for handler in handlers:
            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler(event, data)
                else:
                    handler(event, data)
            except Exception as e:
                self.logger.error(f"Event handler failed: {str(e)}")
    
    async def _generate_workflow_summary(self, execution: WorkflowExecution) -> Dict:
        """Generate comprehensive workflow summary"""
        
        total_tasks = len(execution.tasks)
        completed_tasks = sum(1 for task in execution.tasks.values() if task.status == TaskStatus.COMPLETED)
        failed_tasks = sum(1 for task in execution.tasks.values() if task.status == TaskStatus.FAILED)
        
        return {
            'execution_id': execution.execution_id,
            'status': execution.status,
            'workflow_date': execution.workflow_date.isoformat(),
            'duration_minutes': (execution.completed_at - execution.started_at).total_seconds() / 60 if execution.completed_at else None,
            'task_summary': {
                'total_tasks': total_tasks,
                'completed_tasks': completed_tasks,
                'failed_tasks': failed_tasks,
                'success_rate': completed_tasks / total_tasks if total_tasks > 0 else 0
            },
            'stage_results': execution.execution_state.get('stage_results', {}),
            'agent_outputs': execution.execution_state.get('agent_outputs', {}),
            'errors': execution.errors or [],
            'timestamp': datetime.now().isoformat()
        }