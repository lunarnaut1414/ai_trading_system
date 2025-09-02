"""
Orchestration Controller for AI Trading System

Main orchestration controller that manages system startup, shutdown,
and coordinates all components. Provides centralized control and monitoring
for the entire trading system.
"""

import asyncio
import signal
import logging
from typing import Dict, Any, Optional
from datetime import datetime

from src.orchestration.workflow_engine import WorkflowEngine
from src.orchestration.daily_workflow import DailyTradingWorkflow
from src.orchestration.spec import OrchestrationEvent, TaskStatus

class OrchestrationController:
    """
    Main orchestration controller for the AI trading system
    Manages system startup, shutdown, and coordinates all components
    """
    
    def __init__(self, config: Any):
        self.config = config
        self.logger = logging.getLogger('orchestration_controller')
        
        # System components
        self.agents: Dict[str, Any] = {}
        self.data_provider = None
        self.llm_provider = None
        self.workflow_engine = None
        self.daily_workflow = None
        
        # System state
        self.is_running = False
        self.startup_time = None
        self.shutdown_requested = False
        
        # System statistics
        self.system_stats = {
            'workflows_executed': 0,
            'total_tasks_completed': 0,
            'system_alerts_sent': 0,
            'last_error': None,
            'uptime_start': None
        }
        
        # Setup signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        self.logger.info("✅ Orchestration Controller initialized")
    
    def _signal_handler(self, signum, frame):
        """Handle system signals for graceful shutdown"""
        self.logger.info(f"Received signal {signum}, initiating graceful shutdown...")
        self.shutdown_requested = True
    
    async def startup_system(self):
        """Initialize and start all system components"""
        
        self.logger.info("🚀 Starting AI Trading System...")
        self.startup_time = datetime.now()
        
        try:
            # Initialize data provider
            await self._initialize_data_provider()
            
            # Initialize LLM provider
            await self._initialize_llm_provider()
            
            # Initialize all agents
            await self._initialize_agents()
            
            # Initialize workflow engine
            await self._initialize_workflow_engine()
            
            # Setup event handlers
            await self._setup_event_handlers()
            
            # Perform health checks
            await self._perform_health_checks()
            
            self.is_running = True
            self.system_stats['uptime_start'] = datetime.now()
            
            self.logger.info("✅ System startup complete")
            
        except Exception as e:
            self.logger.error(f"System startup failed: {str(e)}")
            await self.shutdown_system(f"Startup failure: {str(e)}")
            raise
    
    async def _initialize_data_provider(self):
        """Initialize market data provider"""
        
        self.logger.info("🔄 Initializing data provider...")
        
        # Import and create Alpaca provider
        from data_provider.alpaca_provider import AlpacaDataProvider
        
        self.data_provider = AlpacaDataProvider(self.config)
        await self.data_provider.initialize()
        
        self.logger.info("✅ Data provider initialized")
    
    async def _initialize_llm_provider(self):
        """Initialize LLM provider"""
        
        self.logger.info("🔄 Initializing LLM provider...")
        
        # Import and create OpenAI provider
        from src.core.llm_provider import ClaudeLLMProvider
        
        self.llm_provider = ClaudeLLMProvider(self.config)
        await self.llm_provider.initialize()
        
        self.logger.info("✅ LLM provider initialized")
    
    async def _initialize_agents(self):
        """Initialize all trading agents"""
        
        self.logger.info("🔄 Initializing agents...")
        
        # Import all agents
        from agents.junior_analyst import JuniorResearchAnalyst
        from agents.senior_analyst import SeniorResearchAnalyst
        from agents.economist import EconomistAgent
        from agents.portfolio_manager import PortfolioManagerAgent
        from agents.trade_executor import TradeExecutionAgent
        from agents.analytics_reporter import AnalyticsReportingAgent
        
        # Create agent instances
        self.agents = {
            'junior_analyst': JuniorResearchAnalyst(
                self.config, self.data_provider, self.llm_provider
            ),
            'senior_analyst': SeniorResearchAnalyst(
                self.config, self.data_provider, self.llm_provider
            ),
            'economist': EconomistAgent(
                self.config, self.data_provider, self.llm_provider
            ),
            'portfolio_manager': PortfolioManagerAgent(
                self.config, self.data_provider, self.llm_provider
            ),
            'trade_execution': TradeExecutionAgent(
                self.config, self.data_provider
            ),
            'analytics_reporting': AnalyticsReportingAgent(
                self.config, self.data_provider
            )
        }
        
        # Initialize all agents
        for agent_name, agent in self.agents.items():
            await agent.initialize()
            self.logger.info(f"  ✅ {agent_name} initialized")
        
        self.logger.info(f"✅ All {len(self.agents)} agents initialized")
    
    async def _initialize_workflow_engine(self):
        """Initialize workflow orchestration engine"""
        
        self.logger.info("🔄 Initializing workflow engine...")
        
        # Create workflow engine
        self.workflow_engine = WorkflowEngine(self.agents, self.config)
        
        # Create daily workflow manager
        self.daily_workflow = DailyTradingWorkflow(self.workflow_engine, self.config)
        
        self.logger.info("✅ Workflow engine initialized")
    
    async def _setup_event_handlers(self):
        """Setup system event handlers"""
        
        self.logger.info("🔄 Setting up event handlers...")
        
        # Register event handlers
        self.workflow_engine.event_handlers[OrchestrationEvent.WORKFLOW_STARTED] = [
            self._handle_workflow_started
        ]
        self.workflow_engine.event_handlers[OrchestrationEvent.WORKFLOW_COMPLETED] = [
            self._handle_workflow_completed
        ]
        self.workflow_engine.event_handlers[OrchestrationEvent.WORKFLOW_FAILED] = [
            self._handle_workflow_failed
        ]
        self.workflow_engine.event_handlers[OrchestrationEvent.TASK_FAILED] = [
            self._handle_task_failed
        ]
        self.workflow_engine.event_handlers[OrchestrationEvent.SYSTEM_ALERT] = [
            self._handle_system_alert
        ]
        
        self.logger.info("✅ Event handlers configured")
    
    async def _perform_health_checks(self):
        """Perform comprehensive system health checks"""
        
        self.logger.info("🔄 Performing system health checks...")
        
        # Check data provider connection
        data_status = await self.data_provider.health_check()
        if not data_status.get('healthy', False):
            raise Exception(f"Data provider health check failed: {data_status}")
        
        # Check LLM provider
        llm_status = await self.llm_provider.health_check()
        if not llm_status.get('healthy', False):
            raise Exception(f"LLM provider health check failed: {llm_status}")
        
        # Check agent health
        for agent_name, agent in self.agents.items():
            agent_status = await agent.health_check()
            if not agent_status.get('healthy', False):
                raise Exception(f"Agent {agent_name} health check failed: {agent_status}")
        
        self.logger.info("✅ All health checks passed")
    
    async def run_system(self):
        """Run the complete trading system"""
        
        self.logger.info("🚀 Starting AI Trading System execution...")
        
        try:
            # Startup system
            await self.startup_system()
            
            # Main execution loop
            while not self.shutdown_requested:
                current_time = datetime.now()
                
                # Check if it's time to run daily workflow (6:00 AM)
                if (current_time.hour == 6 and current_time.minute == 0 and 
                    current_time.second < 10):  # 10-second window
                    
                    self.logger.info("⏰ Daily workflow trigger time reached")
                    
                    try:
                        # Run daily workflow
                        result = await self.daily_workflow.run_daily_workflow()
                        self.system_stats['workflows_executed'] += 1
                        
                        # Log results
                        if result.get('status') == 'completed':
                            self.logger.info(f"✅ Daily workflow completed successfully")
                        elif result.get('status') == 'skipped':
                            self.logger.info(f"⏭️ Daily workflow skipped: {result.get('reason')}")
                        
                    except Exception as e:
                        self.logger.error(f"Daily workflow failed: {str(e)}")
                        self.system_stats['last_error'] = str(e)
                    
                    # Wait to avoid re-triggering
                    await asyncio.sleep(60)
                
                # Periodic health check (every 15 minutes)
                if current_time.minute % 15 == 0 and current_time.second < 5:
                    await self._perform_health_checks()
                    await asyncio.sleep(5)
                
                # Short sleep to prevent CPU spinning
                await asyncio.sleep(1)
            
            # Graceful shutdown
            await self.shutdown_system("User requested shutdown")
            
        except Exception as e:
            self.logger.error(f"System error: {str(e)}")
            await self.shutdown_system(f"System error: {str(e)}")
            raise
    
    async def shutdown_system(self, reason: str = "Normal shutdown"):
        """Gracefully shutdown all system components"""
        
        self.logger.info(f"🛑 Initiating system shutdown: {reason}")
        
        self.is_running = False
        
        try:
            # Cancel any running workflows
            if self.workflow_engine:
                for execution_id in list(self.workflow_engine.active_executions.keys()):
                    self.logger.info(f"Cancelling workflow {execution_id}")
                    # Mark tasks as cancelled
                    execution = self.workflow_engine.active_executions[execution_id]
                    for task in execution.tasks.values():
                        if task.status in [TaskStatus.PENDING, TaskStatus.RUNNING]:
                            task.status = TaskStatus.CANCELLED
            
            # Shutdown agents
            for agent_name, agent in self.agents.items():
                self.logger.info(f"Shutting down {agent_name}")
                if hasattr(agent, 'shutdown'):
                    await agent.shutdown()
            
            # Shutdown providers
            if self.data_provider and hasattr(self.data_provider, 'shutdown'):
                await self.data_provider.shutdown()
            
            if self.llm_provider and hasattr(self.llm_provider, 'shutdown'):
                await self.llm_provider.shutdown()
            
            # Shutdown executor
            if self.workflow_engine and self.workflow_engine.executor:
                self.workflow_engine.executor.shutdown(wait=True)
            
            self.logger.info("✅ System shutdown complete")
            
        except Exception as e:
            self.logger.error(f"Error during shutdown: {str(e)}")
    
    # Event Handlers
    def _handle_workflow_started(self, event: OrchestrationEvent, data: Dict):
        """Handle workflow started event"""
        self.logger.info(f"📋 Workflow started: {data.get('execution_id')}")
    
    def _handle_workflow_completed(self, event: OrchestrationEvent, data: Dict):
        """Handle workflow completed event"""
        self.logger.info(f"✅ Workflow completed: {data.get('execution_id')}")
        summary = data.get('summary', {})
        self.system_stats['total_tasks_completed'] += summary.get('task_summary', {}).get('completed_tasks', 0)
    
    async def _handle_workflow_failed(self, event: OrchestrationEvent, data: Dict):
        """Handle workflow failed event"""
        self.logger.error(f"❌ Workflow failed: {data.get('execution_id')}")
        self.system_stats['last_error'] = data.get('error')
    
    async def _handle_task_failed(self, event: OrchestrationEvent, data: Dict):
        """Handle task failed event"""
        self.logger.warning(f"⚠️ Task failed: {data.get('task_id')} - {data.get('error')}")
    
    async def _handle_system_alert(self, event: OrchestrationEvent, data: Dict):
        """Handle system alert event"""
        self.logger.warning(f"🚨 System alert: {data}")
        self.system_stats['system_alerts_sent'] += 1
    
    def get_system_status(self) -> Dict:
        """Get comprehensive system status"""
        
        uptime = None
        if self.system_stats.get('uptime_start'):
            uptime = datetime.now() - self.system_stats['uptime_start']
        
        return {
            'system_running': self.is_running,
            'startup_time': self.startup_time.isoformat() if self.startup_time else None,
            'uptime_seconds': uptime.total_seconds() if uptime else None,
            'agents': {
                'total': len(self.agents),
                'names': list(self.agents.keys()),
                'status': {name: 'running' if self.is_running else 'stopped' 
                          for name in self.agents.keys()}
            },
            'workflow_engine': {
                'active_executions': len(self.workflow_engine.active_executions) 
                                    if self.workflow_engine else 0,
                'completed_executions': len(self.workflow_engine.completed_executions) 
                                       if self.workflow_engine else 0
            },
            'statistics': self.system_stats,
            'last_health_check': datetime.now().isoformat()
        }