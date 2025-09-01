# utils/base_agent.py
"""
Base Agent Class - Foundation for All Trading Agents

This abstract base class provides standardized functionality that all
trading agents inherit, ensuring consistent behavior, error handling,
and performance monitoring across the entire system.
"""

import asyncio
import json
import time
import uuid
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union
import logging


class BaseAgent(ABC):
    """
    Abstract base class for all AI trading agents
    
    Provides standardized functionality for:
    - Agent lifecycle management
    - Performance monitoring and metrics
    - Error handling and recovery
    - LLM interaction management
    - Input validation and output formatting
    - Comprehensive logging and debugging
    """
    
    def __init__(self, agent_name: str, llm_provider, config, **kwargs):
        """
        Initialize base agent with core functionality
        
        Args:
            agent_name: Unique identifier for this agent
            llm_provider: LLM provider instance for AI interactions
            config: System configuration object
            **kwargs: Additional agent-specific parameters
        """
        
        # Core agent identity
        self.agent_name = agent_name
        self.agent_id = str(uuid.uuid4())
        self.agent_type = self.__class__.__name__
        
        # Core dependencies
        self.llm = llm_provider
        self.config = config
        
        # Initialize logging
        self.logger = self._setup_agent_logging()
        
        # Performance and health tracking
        self.performance_metrics = {
            "total_tasks": 0,
            "successful_tasks": 0,
            "failed_tasks": 0,
            "average_processing_time": 0.0,
            "total_processing_time": 0.0,
            "last_activity": None,
            "last_error": None,
            "uptime_start": datetime.now(),
            "health_status": "initializing"
        }
        
        # Task execution tracking
        self.task_history: List[Dict] = []
        self.current_task_id: Optional[str] = None
        
        # Agent state management
        self.is_active = False
        self.is_busy = False
        self.last_heartbeat = datetime.now()
        
        # Configuration and validation
        self.required_fields = getattr(self, 'REQUIRED_FIELDS', [])
        self.max_retries = kwargs.get('max_retries', 3)
        self.timeout_seconds = kwargs.get('timeout_seconds', 300)
        
        self.logger.info(f"Agent {self.agent_name} initialized with ID {self.agent_id}")
    
    def _setup_agent_logging(self) -> logging.Logger:
        """
        Setup agent-specific logging with proper formatting and handlers
        
        Returns:
            logging.Logger: Configured logger instance
        """
        
        logger = logging.getLogger(f"agent.{self.agent_name}")
        logger.setLevel(logging.INFO)
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # Console handler
        if not logger.handlers:
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)
        
        return logger
    
    async def start(self) -> bool:
        """
        Start the agent and initialize resources
        
        Returns:
            bool: True if startup successful
        """
        
        try:
            self.logger.info(f"Starting agent {self.agent_name}...")
            
            # Initialize agent resources
            await self._initialize_resources()
            
            # Set agent state
            self.is_active = True
            self.performance_metrics["health_status"] = "healthy"
            self.last_heartbeat = datetime.now()
            
            self.logger.info(f"Agent {self.agent_name} started successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to start agent: {str(e)}")
            self.performance_metrics["health_status"] = "error"
            self.performance_metrics["last_error"] = str(e)
            return False
    
    async def stop(self) -> None:
        """Stop the agent and cleanup resources"""
        
        self.logger.info(f"Stopping agent {self.agent_name}...")
        
        # Wait for current task to complete
        if self.is_busy and self.current_task_id:
            self.logger.info("Waiting for current task to complete...")
            max_wait = 30  # seconds
            waited = 0
            
            while self.is_busy and waited < max_wait:
                await asyncio.sleep(1)
                waited += 1
        
        # Cleanup resources
        await self._cleanup_resources()
        
        # Update state
        self.is_active = False
        self.performance_metrics["health_status"] = "stopped"
        
        self.logger.info(f"Agent {self.agent_name} stopped")
    
    async def process(self, task_data: Dict) -> Dict:
        """
        Main entry point for task processing
        
        Args:
            task_data: Input data for the task
            
        Returns:
            Dict: Processing result with metadata
        """
        
        # Generate task ID
        task_id = str(uuid.uuid4())
        self.current_task_id = task_id
        
        # Initialize result structure
        result = {
            "task_id": task_id,
            "agent_name": self.agent_name,
            "success": False,
            "data": None,
            "error": None,
            "metadata": {
                "start_time": datetime.now().isoformat(),
                "end_time": None,
                "processing_time_seconds": 0
            }
        }
        
        # Check agent status
        if not self.is_active:
            result["error"] = "Agent is not active"
            return result
        
        if self.is_busy:
            result["error"] = "Agent is busy processing another task"
            return result
        
        # Start processing
        self.is_busy = True
        start_time = time.time()
        
        try:
            # Validate input
            validated_data = await self._validate_input(task_data)
            
            # Execute with retry logic
            processed_data = await self._execute_with_retry(validated_data, task_id)
            
            # Format output
            formatted_result = await self._format_output(processed_data)
            
            # Success
            result["success"] = True
            result["data"] = formatted_result
            
            # Update metrics
            self.performance_metrics["successful_tasks"] += 1
            
        except Exception as e:
            # Handle failure
            error_msg = f"Task processing failed: {str(e)}"
            self.logger.error(f"Task {task_id} failed: {str(e)}")
            
            result["error"] = error_msg
            result["data"] = None
            
            # Update metrics
            self.performance_metrics["failed_tasks"] += 1
            self.performance_metrics["last_error"] = error_msg
            
        finally:
            # Calculate processing time
            processing_time = time.time() - start_time
            result["metadata"]["end_time"] = datetime.now().isoformat()
            result["metadata"]["processing_time_seconds"] = round(processing_time, 3)
            
            # Update performance metrics
            self._update_performance_metrics(processing_time, result["success"])
            
            # Store task in history
            self._store_task_result(result)
            
            # Reset agent state
            self.is_busy = False
            self.current_task_id = None
            self.last_heartbeat = datetime.now()
            self.performance_metrics["last_activity"] = datetime.now().isoformat()
        
        return result
    
    async def _validate_input(self, task_data: Dict) -> Dict:
        """
        Validate input data against required fields
        
        Args:
            task_data: Raw input data
            
        Returns:
            Dict: Validated data
            
        Raises:
            ValueError: If validation fails
        """
        
        # Check for required fields
        missing_fields = []
        for field in self.required_fields:
            if field not in task_data:
                missing_fields.append(field)
        
        if missing_fields:
            raise ValueError(f"Missing required fields: {missing_fields}")
        
        # Additional validation can be added by subclasses
        return task_data
    
    async def _execute_with_retry(self, task_data: Dict, task_id: str) -> Dict:
        """
        Execute task with retry logic for transient failures
        
        Args:
            task_data: Validated input data
            task_id: Task identifier
            
        Returns:
            Dict: Task execution result
        """
        
        last_exception = None
        
        for attempt in range(self.max_retries):
            try:
                # Add timeout wrapper
                return await asyncio.wait_for(
                    self._process_internal(task_data),
                    timeout=self.timeout_seconds
                )
                
            except asyncio.TimeoutError:
                last_exception = Exception(f"Task timed out after {self.timeout_seconds} seconds")
                self.logger.warning(f"Task {task_id} attempt {attempt + 1} timed out")
                
            except Exception as e:
                last_exception = e
                self.logger.warning(f"Task {task_id} attempt {attempt + 1} failed: {str(e)}")
                
                # Wait before retry (exponential backoff)
                if attempt < self.max_retries - 1:
                    wait_time = 2 ** attempt
                    await asyncio.sleep(wait_time)
        
        # All retries failed
        raise last_exception
    
    @abstractmethod
    async def _process_internal(self, task_data: Dict) -> Dict:
        """
        Abstract method for agent-specific task processing
        
        Must be implemented by each agent subclass
        
        Args:
            task_data: Validated input data
            
        Returns:
            Dict: Agent-specific processing result
        """
        pass
    
    async def _format_output(self, processed_data: Dict) -> Dict:
        """
        Format the output data for consistency
        
        Args:
            processed_data: Raw processing result
            
        Returns:
            Dict: Formatted output
        """
        
        # Default formatting - can be overridden by subclasses
        return processed_data
    
    async def _initialize_resources(self) -> None:
        """
        Initialize agent-specific resources
        Can be overridden by subclasses
        """
        pass
    
    async def _cleanup_resources(self) -> None:
        """
        Cleanup agent-specific resources
        Can be overridden by subclasses
        """
        pass
    
    def _update_performance_metrics(self, processing_time: float, success: bool) -> None:
        """
        Update agent performance metrics
        
        Args:
            processing_time: Time taken to process task
            success: Whether task succeeded
        """
        
        self.performance_metrics["total_tasks"] += 1
        self.performance_metrics["total_processing_time"] += processing_time
        
        # Update average processing time
        if self.performance_metrics["total_tasks"] > 0:
            self.performance_metrics["average_processing_time"] = (
                self.performance_metrics["total_processing_time"] / 
                self.performance_metrics["total_tasks"]
            )
    
    def _store_task_result(self, result: Dict) -> None:
        """
        Store task result in history
        
        Args:
            result: Task result to store
        """
        
        # Keep only last 100 tasks
        if len(self.task_history) >= 100:
            self.task_history.pop(0)
        
        self.task_history.append({
            "task_id": result["task_id"],
            "timestamp": result["metadata"]["start_time"],
            "success": result["success"],
            "processing_time": result["metadata"]["processing_time_seconds"]
        })
    
    async def get_status(self) -> Dict:
        """
        Get current agent status and health
        
        Returns:
            Dict: Agent status information
        """
        
        # Calculate uptime
        uptime = datetime.now() - self.performance_metrics["uptime_start"]
        
        # Calculate success rate
        total = self.performance_metrics["total_tasks"]
        if total > 0:
            success_rate = self.performance_metrics["successful_tasks"] / total
        else:
            success_rate = 0
        
        # Check if agent is responsive
        time_since_heartbeat = datetime.now() - self.last_heartbeat
        is_responsive = time_since_heartbeat.total_seconds() < 60
        
        # Determine overall status
        if not self.is_active:
            status = "inactive"
        elif not is_responsive:
            status = "unresponsive"
        elif self.is_busy:
            status = "busy"
        else:
            status = "ready"
        
        return {
            "status": status,
            "is_active": self.is_active,
            "is_busy": self.is_busy,
            "is_responsive": is_responsive,
            "last_heartbeat": self.last_heartbeat.isoformat(),
            "current_task": self.current_task_id,
            "uptime_hours": round(uptime.total_seconds() / 3600, 2),
            "performance": {
                "total_tasks": self.performance_metrics["total_tasks"],
                "successful_tasks": self.performance_metrics["successful_tasks"],
                "failed_tasks": self.performance_metrics["failed_tasks"],
                "success_rate": round(success_rate, 3),
                "average_processing_time": round(
                    self.performance_metrics["average_processing_time"], 3
                )
            }
        }


class TradingAgent(BaseAgent):
    """
    Specialized base class for trading-specific agents
    
    Extends BaseAgent with trading-specific functionality:
    - Market data validation
    - Position and portfolio context
    - Risk management integration
    - Trading decision formatting
    """
    
    def __init__(self, agent_name: str, llm_provider, config, **kwargs):
        super().__init__(agent_name, llm_provider, config, **kwargs)
        
        # Trading-specific initialization
        self.position_context: Optional[Dict] = None
        self.portfolio_context: Optional[Dict] = None
        self.market_context: Optional[Dict] = None
        
        # Trading-specific required fields
        if not hasattr(self, 'required_fields'):
            self.required_fields = []
        self.required_fields.extend(['symbol', 'current_price'])
    
    async def set_market_context(self, market_data: Dict) -> None:
        """
        Set current market context for trading decisions
        
        Args:
            market_data: Current market data and conditions
        """
        self.market_context = market_data
        self.logger.debug("Market context updated")
    
    async def set_portfolio_context(self, portfolio_data: Dict) -> None:
        """
        Set current portfolio context for position sizing
        
        Args:
            portfolio_data: Current portfolio positions and metrics
        """
        self.portfolio_context = portfolio_data
        self.logger.debug("Portfolio context updated")
    
    async def _validate_input(self, task_data: Dict) -> Dict:
        """
        Enhanced validation for trading-specific data
        
        Args:
            task_data: Raw trading task data
            
        Returns:
            Dict: Validated trading data
        """
        
        # Perform base validation
        validated_data = await super()._validate_input(task_data)
        
        # Trading-specific validations
        symbol = validated_data.get('symbol')
        if symbol and not isinstance(symbol, str):
            raise ValueError("Symbol must be a string")
        
        price = validated_data.get('current_price')
        if price is not None and (not isinstance(price, (int, float)) or price <= 0):
            raise ValueError("Current price must be a positive number")
        
        return validated_data


class AnalysisAgent(TradingAgent):
    """
    Specialized base class for analysis-focused agents
    
    Extends TradingAgent with analysis-specific functionality:
    - Research data processing
    - Confidence scoring
    - Recommendation formatting
    - Report generation
    """
    
    def __init__(self, agent_name: str, llm_provider, config, **kwargs):
        super().__init__(agent_name, llm_provider, config, **kwargs)
        
        # Analysis-specific tracking
        self.analysis_history: List[Dict] = []
        self.confidence_threshold = kwargs.get('confidence_threshold', 0.7)
    
    def calculate_confidence(self, analysis_data: Dict) -> float:
        """
        Calculate confidence score for analysis
        
        Args:
            analysis_data: Analysis results
            
        Returns:
            float: Confidence score between 0 and 1
        """
        
        # Default implementation - can be overridden
        base_confidence = 0.5
        
        # Adjust based on data quality
        if analysis_data.get('data_quality', 'low') == 'high':
            base_confidence += 0.2
        elif analysis_data.get('data_quality', 'low') == 'medium':
            base_confidence += 0.1
        
        # Adjust based on signal strength
        signal_strength = analysis_data.get('signal_strength', 0)
        base_confidence += signal_strength * 0.3
        
        return min(1.0, max(0.0, base_confidence))


class ExecutionAgent(TradingAgent):
    """
    Specialized base class for execution-focused agents
    
    Extends TradingAgent with execution-specific functionality:
    - Order management and tracking
    - Execution quality monitoring
    - Market timing optimization
    - Slippage and cost analysis
    """
    
    def __init__(self, agent_name: str, llm_provider, config, **kwargs):
        super().__init__(agent_name, llm_provider, config, **kwargs)
        
        # Execution-specific tracking
        self.active_orders: List[Dict] = []
        self.execution_history: List[Dict] = []
        self.execution_quality_metrics: Dict = {
            "total_executions": 0,
            "average_slippage": 0.0,
            "average_execution_time": 0.0,
            "success_rate": 1.0
        }
    
    def track_execution(self, order_data: Dict, execution_result: Dict) -> None:
        """
        Track execution quality and update metrics
        
        Args:
            order_data: Original order parameters
            execution_result: Actual execution results
        """
        
        execution_record = {
            "timestamp": datetime.now().isoformat(),
            "order_data": order_data,
            "execution_result": execution_result,
            "slippage": execution_result.get("slippage", 0.0),
            "execution_time": execution_result.get("execution_time", 0.0)
        }
        
        self.execution_history.append(execution_record)
        
        # Update metrics
        self._update_execution_metrics(execution_record)
    
    def _update_execution_metrics(self, execution_record: Dict) -> None:
        """Update execution quality metrics"""
        
        self.execution_quality_metrics["total_executions"] += 1
        
        # Update average slippage
        slippage = execution_record.get("slippage", 0.0)
        total_execs = self.execution_quality_metrics["total_executions"]
        current_avg_slippage = self.execution_quality_metrics["average_slippage"]
        
        new_avg_slippage = ((current_avg_slippage * (total_execs - 1)) + slippage) / total_execs
        self.execution_quality_metrics["average_slippage"] = round(new_avg_slippage, 4)