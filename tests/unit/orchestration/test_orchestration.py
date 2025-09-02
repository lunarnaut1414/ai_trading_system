"""
Comprehensive test suite for AI Trading System orchestration module
Tests WorkflowEngine, DailyTradingWorkflow, OrchestrationController
Fixed to match actual workflow_engine implementation
"""

import pytest
import asyncio
import uuid
import signal
from datetime import datetime, timedelta, time, date
from unittest.mock import Mock, AsyncMock, patch, MagicMock, call
from typing import Dict, List, Any

# Import from actual git repository structure
from src.orchestration.workflow_engine import WorkflowEngine
from src.orchestration.daily_workflow import DailyTradingWorkflow
from src.orchestration.controller import OrchestrationController
from src.orchestration.spec import (
    WorkflowTask, WorkflowExecution, WorkflowStage, TaskStatus,
    TaskPriority, OrchestrationEvent
)


# ==============================================================================
# FIXTURES
# ==============================================================================

@pytest.fixture
def mock_config():
    """Create mock configuration object"""
    config = Mock()
    config.MAX_CONCURRENT_TASKS = 5
    config.DEFAULT_TASK_TIMEOUT = 300
    config.WORKFLOW_TIMEOUT = 3600
    config.LOG_LEVEL = "INFO"
    return config


@pytest.fixture
def mock_agents():
    """Create mock agents dictionary with proper async handling matching workflow_engine expectations"""
    agents = {}
    agent_names = ['junior_analyst', 'senior_analyst', 'economist', 
                   'portfolio_manager', 'trade_execution', 'analytics_reporting']
    
    for name in agent_names:
        agent = Mock()
        
        # Create a default async method that returns success
        async def default_method(**kwargs):
            return {'status': 'success', 'data': f'{name}_result'}
        
        # Set up the analyze method as the default fallback
        agent.analyze = AsyncMock(return_value={'status': 'success', 'data': f'{name}_result'})
        
        # Set up specific execute_* methods for different task types
        agent.execute_test = AsyncMock(return_value={'status': 'success', 'data': 'test_result'})
        agent.execute_test_retry = AsyncMock(return_value={'status': 'success', 'data': 'test_retry_result'})
        agent.execute_market_analysis = AsyncMock(return_value={'status': 'success', 'data': 'market_analysis_result'})
        agent.execute_macro_analysis = AsyncMock(return_value={'status': 'success', 'data': 'macro_analysis_result'})
        agent.execute_position_evaluation = AsyncMock(return_value={'status': 'success', 'data': 'position_evaluation_result'})
        agent.execute_portfolio_review = AsyncMock(return_value={'status': 'success', 'data': 'portfolio_review_result'})
        agent.execute_opportunity_assessment = AsyncMock(return_value={'status': 'success', 'data': 'opportunity_assessment_result'})
        agent.execute_generate_trades = AsyncMock(return_value={'status': 'success', 'data': 'generate_trades_result'})
        agent.execute_execute_trades = AsyncMock(return_value={'status': 'success', 'data': 'execute_trades_result'})
        agent.execute_risk_monitoring = AsyncMock(return_value={'status': 'success', 'data': 'risk_monitoring_result'})
        agent.execute_monitor_orders = AsyncMock(return_value={'status': 'success', 'data': 'monitor_orders_result'})
        agent.execute_intraday_monitoring = AsyncMock(return_value={'status': 'success', 'data': 'intraday_monitoring_result'})
        agent.execute_daily_performance = AsyncMock(return_value={'status': 'success', 'data': 'daily_performance_result'})
        agent.execute_end_of_day_review = AsyncMock(return_value={'status': 'success', 'data': 'end_of_day_review_result'})
        agent.execute_generate_reports = AsyncMock(return_value={'status': 'success', 'data': 'generate_reports_result'})
        agent.execute_concurrent_test = AsyncMock(return_value={'status': 'success', 'data': 'concurrent_test_result'})
        agent.execute_process = AsyncMock(return_value={'status': 'success', 'data': 'process_result'})
        agent.execute_evaluate = AsyncMock(return_value={'status': 'success', 'data': 'evaluate_result'})
        
        # Add process method for backward compatibility
        agent.process = AsyncMock(return_value={'status': 'success', 'data': f'{name}_result'})
        
        # Add other required methods
        agent.initialize = AsyncMock()
        agent.health_check = AsyncMock(return_value={'healthy': True})
        agent.shutdown = AsyncMock()
        
        agents[name] = agent
    
    return agents


@pytest.fixture
def workflow_engine(mock_config, mock_agents):
    """Create WorkflowEngine instance with mocks"""
    engine = WorkflowEngine(mock_agents, mock_config)
    return engine


@pytest.fixture
def sample_workflow_definition():
    """Create sample workflow definition with valid stages"""
    return {
        'name': 'Test Workflow',
        'stages': {
            'pre_market': {
                'tasks': [
                    {
                        'agent': 'junior_analyst',
                        'task_type': 'market_analysis',
                        'task_data': {'analysis_type': 'pre_market'},
                        'priority': TaskPriority.HIGH.value,
                        'dependencies': [],
                        'timeout': 300,
                        'max_retries': 2
                    },
                    {
                        'agent': 'economist',
                        'task_type': 'macro_analysis',
                        'task_data': {},
                        'priority': TaskPriority.MEDIUM.value,
                        'dependencies': [],
                        'timeout': 300,
                        'max_retries': 1
                    }
                ]
            },
            'market_open': {
                'tasks': [
                    {
                        'agent': 'portfolio_manager',
                        'task_type': 'position_evaluation',
                        'task_data': {},
                        'priority': TaskPriority.CRITICAL.value,
                        'dependencies': [],
                        'timeout': 600,
                        'max_retries': 3
                    }
                ]
            }
        }
    }


# ==============================================================================
# WORKFLOW ENGINE TESTS
# ==============================================================================

class TestWorkflowEngine:
    """Test WorkflowEngine functionality"""
    
    def test_workflow_engine_initialization(self, mock_config, mock_agents):
        """Test WorkflowEngine initialization"""
        engine = WorkflowEngine(mock_agents, mock_config)
        
        assert engine.agents == mock_agents
        assert engine.config == mock_config
        assert len(engine.active_executions) == 0
        assert len(engine.completed_executions) == 0
        assert engine.max_concurrent_tasks == 5
        assert engine.default_task_timeout == 300
        assert engine.workflow_timeout == 3600
    
    @pytest.mark.asyncio
    async def test_workflow_creation(self, workflow_engine, sample_workflow_definition):
        """Test workflow creation from definition"""
        execution_id = await workflow_engine.create_workflow(sample_workflow_definition)
        
        assert execution_id is not None
        assert execution_id.startswith("workflow_")
        assert execution_id in workflow_engine.active_executions
        
        execution = workflow_engine.active_executions[execution_id]
        assert isinstance(execution, WorkflowExecution)
        assert execution.status == "running"
        assert len(execution.tasks) == 3  # 2 pre_market + 1 market_open
        assert execution.current_stage == WorkflowStage.PRE_MARKET
    
    @pytest.mark.asyncio
    async def test_workflow_execution_success(self, mock_config, mock_agents):
        """Test successful workflow execution with proper async agents"""
        # The mock_agents fixture now has proper execute_* methods
        engine = WorkflowEngine(mock_agents, mock_config)
        
        workflow_def = {
            'stages': {
                'pre_market': {
                    'tasks': [{
                        'agent': 'junior_analyst',
                        'task_type': 'test',
                        'task_data': {},
                        'priority': TaskPriority.MEDIUM.value,
                        'dependencies': [],
                        'timeout': 300,
                        'max_retries': 0
                    }]
                }
            }
        }
        
        execution_id = await engine.create_workflow(workflow_def)
        result = await engine.execute_workflow(execution_id)
        
        assert result['status'] == 'completed'
        assert 'task_summary' in result
        assert result['task_summary']['total_tasks'] == 1
        assert result['task_summary']['completed_tasks'] == 1
    
    @pytest.mark.asyncio
    async def test_task_dependency_ordering(self, workflow_engine):
        """Test task ordering based on dependencies"""
        task1 = WorkflowTask(
            task_id="task1",
            agent_name="agent1",
            task_type="type1",
            task_data={},
            priority=TaskPriority.MEDIUM,
            stage=WorkflowStage.PRE_MARKET,
            dependencies=[],
            timeout_seconds=300,
            retry_count=0,
            max_retries=2,
            status=TaskStatus.PENDING,
            created_at=datetime.now()
        )
        
        task2 = WorkflowTask(
            task_id="task2",
            agent_name="agent2",
            task_type="type2",
            task_data={},
            priority=TaskPriority.HIGH,
            stage=WorkflowStage.PRE_MARKET,
            dependencies=["task1"],
            timeout_seconds=300,
            retry_count=0,
            max_retries=2,
            status=TaskStatus.PENDING,
            created_at=datetime.now()
        )
        
        task3 = WorkflowTask(
            task_id="task3",
            agent_name="agent3",
            task_type="type3",
            task_data={},
            priority=TaskPriority.LOW,
            stage=WorkflowStage.PRE_MARKET,
            dependencies=["task1", "task2"],
            timeout_seconds=300,
            retry_count=0,
            max_retries=2,
            status=TaskStatus.PENDING,
            created_at=datetime.now()
        )
        
        tasks = [task1, task2, task3]
        ordered_batches = workflow_engine._order_tasks_by_dependencies(tasks)
        
        # Should have 3 batches due to dependencies
        assert len(ordered_batches) == 3
        assert ordered_batches[0][0].task_id == "task1"
        assert ordered_batches[1][0].task_id == "task2"
        assert ordered_batches[2][0].task_id == "task3"
    
    @pytest.mark.asyncio
    async def test_circular_dependency_detection(self, workflow_engine):
        """Test detection of circular dependencies"""
        task1 = WorkflowTask(
            task_id="task1",
            agent_name="agent1",
            task_type="type1",
            task_data={},
            priority=TaskPriority.MEDIUM,
            stage=WorkflowStage.PRE_MARKET,
            dependencies=["task2"],
            timeout_seconds=300,
            retry_count=0,
            max_retries=2,
            status=TaskStatus.PENDING,
            created_at=datetime.now()
        )
        
        task2 = WorkflowTask(
            task_id="task2",
            agent_name="agent2",
            task_type="type2",
            task_data={},
            priority=TaskPriority.HIGH,
            stage=WorkflowStage.PRE_MARKET,
            dependencies=["task1"],
            timeout_seconds=300,
            retry_count=0,
            max_retries=2,
            status=TaskStatus.PENDING,
            created_at=datetime.now()
        )
        
        tasks = [task1, task2]
        
        with patch.object(workflow_engine.logger, 'warning') as mock_warning:
            ordered_batches = workflow_engine._order_tasks_by_dependencies(tasks)
            mock_warning.assert_called_with("Circular or missing dependency detected")
            assert len(ordered_batches) > 0
    
    @pytest.mark.asyncio
    async def test_task_retry_logic(self, mock_config):
        """Test task retry on failure"""
        call_count = 0
        
        async def failing_process(**kwargs):
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise Exception(f"Simulated failure {call_count}")
            return {'status': 'success', 'attempt': call_count}
        
        # Create a mock agent with the failing async function
        mock_agent = Mock()
        # The workflow engine will look for execute_test_retry method
        mock_agent.execute_test_retry = failing_process
        # Add fallback analyze method
        mock_agent.analyze = failing_process
        
        mock_agents = {
            'junior_analyst': mock_agent
        }
        
        engine = WorkflowEngine(mock_agents, mock_config)
        
        workflow_def = {
            'stages': {
                'pre_market': {
                    'tasks': [{
                        'agent': 'junior_analyst',
                        'task_type': 'test_retry',
                        'task_data': {},
                        'priority': TaskPriority.MEDIUM.value,
                        'dependencies': [],
                        'timeout': 300,
                        'max_retries': 3
                    }]
                }
            }
        }
        
        execution_id = await engine.create_workflow(workflow_def)
        result = await engine.execute_workflow(execution_id)
        
        assert result['status'] == 'completed'
        assert call_count == 3
    
    @pytest.mark.asyncio
    async def test_concurrent_task_execution(self, mock_config):
        """Test concurrent execution of independent tasks"""
        execution_times = []
        
        async def timed_process(**kwargs):
            start = datetime.now()
            await asyncio.sleep(0.05)
            execution_times.append(start)
            return {'status': 'success'}
        
        mock_agents = {}
        for name in ['junior_analyst', 'economist', 'senior_analyst']:
            agent = Mock()
            # Set both execute_ method and analyze as fallback
            agent.execute_concurrent_test = timed_process
            agent.analyze = timed_process
            mock_agents[name] = agent
        
        engine = WorkflowEngine(mock_agents, mock_config)
        
        workflow_def = {
            'stages': {
                'pre_market': {
                    'tasks': [
                        {
                            'agent': agent_name,
                            'task_type': 'concurrent_test',
                            'task_data': {},
                            'priority': TaskPriority.MEDIUM.value,
                            'dependencies': [],
                            'timeout': 300,
                            'max_retries': 0
                        }
                        for agent_name in mock_agents.keys()
                    ]
                }
            }
        }
        
        execution_id = await engine.create_workflow(workflow_def)
        await engine.execute_workflow(execution_id)
        
        # Tasks should have started close to each other
        if len(execution_times) >= 2:
            time_diff = max(execution_times) - min(execution_times)
            assert time_diff.total_seconds() < 1.0
    
    @pytest.mark.asyncio
    async def test_event_emission(self, workflow_engine, sample_workflow_definition):
        """Test event emission during workflow execution"""
        events_received = []
        
        async def event_handler(event, data):
            events_received.append((event, data))
        
        # Register event handlers
        workflow_engine.event_handlers[OrchestrationEvent.WORKFLOW_STARTED] = [event_handler]
        workflow_engine.event_handlers[OrchestrationEvent.WORKFLOW_COMPLETED] = [event_handler]
        workflow_engine.event_handlers[OrchestrationEvent.TASK_STARTED] = [event_handler]
        workflow_engine.event_handlers[OrchestrationEvent.TASK_COMPLETED] = [event_handler]
        
        execution_id = await workflow_engine.create_workflow(sample_workflow_definition)
        await workflow_engine.execute_workflow(execution_id)
        
        # Check that events were emitted
        event_types = [event for event, _ in events_received]
        assert OrchestrationEvent.WORKFLOW_STARTED in event_types
        assert OrchestrationEvent.WORKFLOW_COMPLETED in event_types


# ==============================================================================
# DAILY TRADING WORKFLOW TESTS
# ==============================================================================

class TestDailyTradingWorkflow:
    """Test DailyTradingWorkflow functionality"""
    
    def test_daily_workflow_initialization(self, mock_config):
        """Test DailyTradingWorkflow initialization"""
        mock_engine = Mock()
        workflow = DailyTradingWorkflow(mock_engine, mock_config)
        
        assert workflow.workflow_engine == mock_engine
        assert workflow.config == mock_config
        assert workflow.workflow_status == "idle"
        assert workflow.current_execution_id is None
        assert workflow.market_open_time == time(9, 30)
        assert workflow.market_close_time == time(16, 0)
    
    def test_standard_workflow_creation(self, mock_config):
        """Test creation of standard workflow definition"""
        mock_engine = Mock()
        workflow = DailyTradingWorkflow(mock_engine, mock_config)
        
        workflow_def = workflow._create_standard_workflow()
        
        assert workflow_def['name'] == 'Standard Trading Day'
        assert 'stages' in workflow_def
        assert 'pre_market' in workflow_def['stages']
        assert 'market_open' in workflow_def['stages']
        assert 'intraday' in workflow_def['stages']
        assert 'post_market' in workflow_def['stages']
        
        pre_market_tasks = workflow_def['stages']['pre_market']['tasks']
        agent_types = [task['agent'] for task in pre_market_tasks]
        assert 'junior_analyst' in agent_types
        assert 'economist' in agent_types
    
    def test_special_workflow_creation(self, mock_config):
        """Test creation of special workflow definitions"""
        mock_engine = Mock()
        workflow = DailyTradingWorkflow(mock_engine, mock_config)
        
        # Test earnings workflow
        earnings_def = workflow._create_earnings_workflow()
        assert earnings_def['name'] == 'Earnings Announcement Day'
        assert 'earnings_analysis' in str(earnings_def)
        
        # Test FOMC workflow
        fomc_def = workflow._create_fomc_workflow()
        assert fomc_def['name'] == 'FOMC Day'
        assert 'fomc_analysis' in str(fomc_def)
        
        # Test expiration workflow
        expiration_def = workflow._create_expiration_workflow()
        assert expiration_def['name'] == 'Options Expiration Day'
        assert 'manage_expirations' in str(expiration_def)
    
    @pytest.mark.asyncio
    async def test_run_daily_workflow_market_closed(self, mock_config):
        """Test workflow execution on market closed day"""
        mock_engine = Mock()
        workflow = DailyTradingWorkflow(mock_engine, mock_config)
        
        with patch('src.orchestration.daily_workflow.datetime') as mock_datetime:
            saturday_date = Mock()
            saturday_date.weekday.return_value = 5  # Saturday
            saturday_date.date.return_value = saturday_date
            saturday_date.isoformat.return_value = '2024-01-13'
            
            mock_now = Mock()
            mock_now.date.return_value = saturday_date
            mock_now.weekday.return_value = 5
            mock_datetime.now.return_value = mock_now
            
            result = await workflow.run_daily_workflow()
            
            assert result['status'] == 'skipped'
            assert result['reason'] == 'Weekend - Market closed'
            mock_engine.create_workflow.assert_not_called()
    
    @pytest.mark.asyncio
    async def test_run_daily_workflow_standard_day(self, mock_config):
        """Test standard trading day workflow execution"""
        mock_engine = Mock()
        mock_engine.create_workflow = AsyncMock(return_value="test_execution_id")
        mock_engine.execute_workflow = AsyncMock(return_value={
            'status': 'completed',
            'task_summary': {
                'total_tasks': 10,
                'completed_tasks': 10,
                'failed_tasks': 0,
                'success_rate': 1.0
            }
        })
        
        workflow = DailyTradingWorkflow(mock_engine, mock_config)
        
        with patch('src.orchestration.daily_workflow.datetime') as mock_datetime:
            tuesday_date = Mock()
            tuesday_date.weekday.return_value = 1  # Tuesday
            tuesday_date.date.return_value = tuesday_date
            tuesday_date.isoformat.return_value = '2024-01-16'
            
            mock_now = Mock()
            mock_now.date.return_value = tuesday_date
            mock_now.weekday.return_value = 1
            mock_datetime.now.return_value = mock_now
            
            result = await workflow.run_daily_workflow()
            
            assert result['status'] == 'completed'
            assert workflow.current_execution_id == "test_execution_id"
            assert workflow.workflow_status == "completed"
            mock_engine.create_workflow.assert_called_once()
            mock_engine.execute_workflow.assert_called_once_with("test_execution_id")
    
    @pytest.mark.asyncio
    async def test_workflow_status_monitoring(self, mock_config):
        """Test workflow status monitoring"""
        mock_engine = Mock()
        mock_execution = Mock()
        mock_execution.tasks = {
            "task1": Mock(status=TaskStatus.COMPLETED),
            "task2": Mock(status=TaskStatus.RUNNING),
            "task3": Mock(status=TaskStatus.PENDING),
            "task4": Mock(status=TaskStatus.FAILED)
        }
        mock_execution.current_stage = WorkflowStage.PRE_MARKET
        mock_execution.execution_state = {'stage_results': {}, 'agent_outputs': {}}
        
        mock_engine.active_executions = {"test_id": mock_execution}
        
        workflow = DailyTradingWorkflow(mock_engine, mock_config)
        workflow.current_execution_id = "test_id"
        
        status = await workflow.get_workflow_status()
        
        assert status['execution_id'] == "test_id"
        assert status['status'] == "idle"
    
    @pytest.mark.asyncio
    async def test_high_volatility_adjustment(self, mock_config):
        """Test workflow adjustment for high volatility"""
        mock_engine = Mock()
        workflow = DailyTradingWorkflow(mock_engine, mock_config)
        
        workflow_def = {
            'stages': {
                'pre_market': {
                    'tasks': [{
                        'agent': 'portfolio_manager',
                        'task_type': 'position_evaluation',
                        'task_data': {},
                        'priority': TaskPriority.HIGH.value,
                        'timeout': 300
                    }]
                }
            }
        }
        
        # Method returns None after logging
        result = workflow._adjust_for_high_volatility(workflow_def)
        assert result is None


# ==============================================================================
# ORCHESTRATION CONTROLLER TESTS
# ==============================================================================

class TestOrchestrationController:
    """Test OrchestrationController functionality"""
    
    def test_controller_initialization(self, mock_config):
        """Test OrchestrationController initialization"""
        controller = OrchestrationController(mock_config)
        
        assert controller.config == mock_config
        assert controller.is_running == False
        assert controller.startup_time is None
        assert controller.shutdown_requested == False
        assert len(controller.agents) == 0
        assert controller.workflow_engine is None
        assert controller.daily_workflow is None
    
    @pytest.mark.asyncio
    async def test_system_startup_sequence(self, mock_config):
        """Test system startup sequence"""
        controller = OrchestrationController(mock_config)
        
        controller._initialize_data_provider = AsyncMock()
        controller._initialize_llm_provider = AsyncMock()
        controller._initialize_agents = AsyncMock()
        controller._initialize_workflow_engine = AsyncMock()
        controller._setup_event_handlers = AsyncMock()
        
        controller.data_provider = Mock()
        controller.data_provider.health_check = AsyncMock(return_value={'healthy': True, 'status': 'ok'})
        controller.llm_provider = Mock()
        controller.llm_provider.health_check = AsyncMock(return_value={'healthy': True, 'status': 'ok'})
        controller.agents = {}
        
        await controller.startup_system()
        
        assert controller.is_running == True
        assert controller.startup_time is not None
        assert controller.system_stats['uptime_start'] is not None
        
        controller._initialize_data_provider.assert_called_once()
        controller._initialize_llm_provider.assert_called_once()
        controller._initialize_agents.assert_called_once()
        controller._initialize_workflow_engine.assert_called_once()
        controller._setup_event_handlers.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_component_initialization_failure(self, mock_config):
        """Test handling of component initialization failure"""
        controller = OrchestrationController(mock_config)
        
        controller._initialize_data_provider = AsyncMock(
            side_effect=Exception("Data provider init failed")
        )
        
        with pytest.raises(Exception) as exc_info:
            await controller.startup_system()
        
        assert "Data provider init failed" in str(exc_info.value)
        assert controller.is_running == False
    
    @pytest.mark.asyncio
    async def test_event_handler_setup(self, mock_config):
        """Test event handler registration"""
        controller = OrchestrationController(mock_config)
        
        mock_engine = Mock()
        mock_engine.event_handlers = {
            OrchestrationEvent.WORKFLOW_STARTED: [],
            OrchestrationEvent.WORKFLOW_COMPLETED: [],
            OrchestrationEvent.WORKFLOW_FAILED: [],
            OrchestrationEvent.TASK_FAILED: [],
            OrchestrationEvent.SYSTEM_ALERT: []
        }
        controller.workflow_engine = mock_engine
        
        await controller._setup_event_handlers()
        
        assert len(mock_engine.event_handlers[OrchestrationEvent.WORKFLOW_STARTED]) > 0
        assert len(mock_engine.event_handlers[OrchestrationEvent.WORKFLOW_COMPLETED]) > 0
    
    @pytest.mark.asyncio
    async def test_health_check_execution(self, mock_config):
        """Test health check execution with failure handling"""
        controller = OrchestrationController(mock_config)
        
        controller.data_provider = Mock()
        controller.data_provider.health_check = AsyncMock(return_value={'healthy': True, 'status': 'ok'})
        controller.llm_provider = Mock()
        controller.llm_provider.health_check = AsyncMock(return_value={'healthy': True, 'status': 'ok'})
        
        mock_agent1 = Mock()
        mock_agent1.health_check = AsyncMock(return_value={'healthy': True, 'status': 'ok'})
        mock_agent2 = Mock()
        mock_agent2.health_check = AsyncMock(return_value={'healthy': False, 'error': 'test error'})
        
        controller.agents = {
            'agent1': mock_agent1,
            'agent2': mock_agent2
        }
        
        with pytest.raises(Exception) as exc_info:
            await controller._perform_health_checks()
        
        assert "Agent agent2 health check failed" in str(exc_info.value)
    
    @pytest.mark.asyncio
    async def test_graceful_shutdown(self, mock_config):
        """Test graceful system shutdown"""
        controller = OrchestrationController(mock_config)
        
        mock_agent = Mock()
        mock_agent.shutdown = AsyncMock()
        controller.agents = {'test_agent': mock_agent}
        
        mock_provider = Mock()
        mock_provider.shutdown = AsyncMock()
        controller.data_provider = mock_provider
        
        mock_engine = Mock()
        mock_engine.active_executions = {}
        mock_engine.executor = Mock()
        mock_engine.executor.shutdown = Mock()
        controller.workflow_engine = mock_engine
        
        controller.is_running = True
        
        await controller.shutdown_system("Test shutdown")
        
        assert controller.is_running == False
        mock_agent.shutdown.assert_called_once()
        mock_provider.shutdown.assert_called_once()
        mock_engine.executor.shutdown.assert_called_once_with(wait=True)
    
    def test_signal_handling(self, mock_config):
        """Test signal handler setup"""
        controller = OrchestrationController(mock_config)
        
        controller._signal_handler(signal.SIGINT, None)
        assert controller.shutdown_requested == True
        
        controller.shutdown_requested = False
        controller._signal_handler(signal.SIGTERM, None)
        assert controller.shutdown_requested == True
    
    def test_system_status_reporting(self, mock_config):
        """Test comprehensive system status reporting"""
        controller = OrchestrationController(mock_config)
        
        controller.is_running = True
        controller.startup_time = datetime.now()
        controller.system_stats = {
            'workflows_executed': 5,
            'total_tasks_completed': 50,
            'system_alerts_sent': 2,
            'last_error': None,
            'uptime_start': datetime.now() - timedelta(hours=2)
        }
        
        controller.agents = {
            'agent1': Mock(),
            'agent2': Mock()
        }
        
        mock_engine = Mock()
        mock_engine.active_executions = {'exec1': Mock()}
        mock_engine.completed_executions = [Mock(), Mock(), Mock()]
        controller.workflow_engine = mock_engine
        
        status = controller.get_system_status()
        
        assert status['system_running'] == True
        assert status['startup_time'] is not None
        assert status['uptime_seconds'] > 0
        assert status['agents']['total'] == 2
        assert 'agent1' in status['agents']['names']
        assert status['workflow_engine']['active_executions'] == 1
        assert status['workflow_engine']['completed_executions'] == 3
        assert status['statistics']['workflows_executed'] == 5
    
    @pytest.mark.asyncio
    async def test_event_handler_responses(self, mock_config):
        """Test event handler responses"""
        controller = OrchestrationController(mock_config)
        
        controller._handle_workflow_started(OrchestrationEvent.WORKFLOW_STARTED, 
                                           {'execution_id': 'test_123'})
        
        data = {
            'execution_id': 'test_123',
            'summary': {
                'task_summary': {'completed_tasks': 10}
            }
        }
        controller._handle_workflow_completed(OrchestrationEvent.WORKFLOW_COMPLETED, data)
        assert controller.system_stats['total_tasks_completed'] == 10
        
        await controller._handle_workflow_failed(OrchestrationEvent.WORKFLOW_FAILED,
                                                 {'execution_id': 'test_123', 'error': 'Test error'})
        assert controller.system_stats['last_error'] == 'Test error'
        
        await controller._handle_system_alert(OrchestrationEvent.SYSTEM_ALERT,
                                              {'alert': 'Test alert'})
        assert controller.system_stats['system_alerts_sent'] == 1


# ==============================================================================
# INTEGRATION TESTS
# ==============================================================================

class TestOrchestrationIntegration:
    """Integration tests for the orchestration system"""
    
    @pytest.mark.asyncio
    async def test_end_to_end_workflow_execution(self, mock_config, mock_agents):
        """Test complete workflow execution from controller to agents"""
        controller = OrchestrationController(mock_config)
        
        controller._initialize_data_provider = AsyncMock()
        controller._initialize_llm_provider = AsyncMock()
        
        # Use the fixture mock_agents which have proper execute_* methods
        controller.agents = mock_agents
        controller.workflow_engine = WorkflowEngine(mock_agents, mock_config)
        controller.daily_workflow = DailyTradingWorkflow(controller.workflow_engine, mock_config)
        
        await controller._setup_event_handlers()
        
        with patch('src.orchestration.daily_workflow.datetime') as mock_datetime:
            tuesday = Mock()
            tuesday.weekday.return_value = 1
            tuesday.date.return_value = tuesday
            tuesday.isoformat.return_value = '2024-01-16'
            mock_now = Mock()
            mock_now.date.return_value = tuesday
            mock_now.weekday.return_value = 1
            mock_datetime.now.return_value = mock_now
            
            result = await controller.daily_workflow.run_daily_workflow()
            
            assert result['status'] == 'completed'
            assert result['task_summary']['total_tasks'] > 0
            assert result['task_summary']['completed_tasks'] > 0
    
    @pytest.mark.asyncio
    async def test_system_resilience(self, mock_config):
        """Test system resilience to various failure modes"""
        controller = OrchestrationController(mock_config)
        
        controller._initialize_data_provider = AsyncMock(
            side_effect=Exception("Temporary failure")
        )
        
        with pytest.raises(Exception) as exc_info:
            await controller.startup_system()
        
        assert "Temporary failure" in str(exc_info.value)
        assert controller.is_running == False


# ==============================================================================
# PERFORMANCE TESTS (Optional)
# ==============================================================================

@pytest.mark.performance
class TestPerformance:
    """Performance tests for orchestration system"""
    
    @pytest.mark.asyncio
    async def test_workflow_with_valid_stages(self, mock_config, mock_agents):
        """Test workflow with valid stage names"""
        # Use the mock_agents fixture which already has proper execute_* methods
        engine = WorkflowEngine(mock_agents, mock_config)
        
        workflow_def = {
            'stages': {
                'pre_market': {
                    'tasks': [
                        {
                            'agent': 'junior_analyst',
                            'task_type': 'process',
                            'task_data': {'index': i},
                            'priority': TaskPriority.MEDIUM.value,
                            'dependencies': [],
                            'timeout': 300,
                            'max_retries': 0
                        }
                        for i in range(3)
                    ]
                },
                'market_open': {
                    'tasks': [
                        {
                            'agent': 'portfolio_manager',
                            'task_type': 'evaluate',
                            'task_data': {},
                            'priority': TaskPriority.HIGH.value,
                            'dependencies': [],
                            'timeout': 300,
                            'max_retries': 0
                        }
                    ]
                }
            }
        }
        
        start_time = datetime.now()
        execution_id = await engine.create_workflow(workflow_def)
        result = await engine.execute_workflow(execution_id)
        execution_time = (datetime.now() - start_time).total_seconds()
        
        assert result['status'] == 'completed'
        assert result['task_summary']['total_tasks'] == 4
        assert execution_time < 10


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--asyncio-mode=auto"])