# ==============================================================================
# 02 - CORE INFRASTRUCTURE FRAMEWORK - COMPLETE IMPLEMENTATION
# ==============================================================================

"""
Core Infrastructure for AI Trading System

This module provides the foundational components that all trading agents
inherit and use:
- Base Agent Framework
- Claude LLM Provider Management 
- Agent Communication System
- Performance Tracking & Monitoring
- Error Handling & Recovery
"""

# Load environment variables FIRST, before any other imports
try:
    from dotenv import load_dotenv
    import os
    
    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)  # Go up one level from agents/ to project root
    
    # Try to load .env file from multiple locations
    env_paths = [
        os.path.join(project_root, '.env'),  # Project root
        os.path.join(os.getcwd(), '.env'),   # Current working directory
        '.env'                               # Relative to current location
    ]
    
    env_file = None
    for path in env_paths:
        if os.path.exists(path):
            env_file = path
            break
    
    if env_file:
        result = load_dotenv(env_file)
        print(f"✅ Loaded environment variables from: {env_file}")
        print(f"✅ Load result: {result}")
        
        # Debug: Check if the key is actually loaded
        api_key = os.getenv('ANTHROPIC_API_KEY')
        if api_key:
            print(f"✅ ANTHROPIC_API_KEY loaded successfully (length: {len(api_key)})")
        else:
            print("❌ ANTHROPIC_API_KEY not found after loading .env")
    else:
        print("⚠️ No .env file found in any of these locations:")
        for path in env_paths:
            print(f"   - {path}")
        
except ImportError:
    print("⚠️ python-dotenv not installed, skipping .env file loading")

# Now import everything else
import asyncio
import logging
import json
import time
import traceback
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable, Union
from dataclasses import dataclass, asdict
from enum import Enum
import uuid
import threading
from queue import Queue, Empty
import signal
import sys
from pathlib import Path

# External dependencies
import anthropic
from pydantic import BaseModel, ValidationError


# ==============================================================================
# AGENT FRAMEWORK SPECIFICATIONS
# ==============================================================================

@dataclass
class AgentConfig:
    """Configuration for trading agents"""
    agent_name: str
    max_retries: int = 3
    timeout_seconds: int = 30
    llm_provider: str = "claude"  # Primary: Claude
    performance_tracking: bool = True
    debug_mode: bool = False
    required_fields: List[str] = None
    
    def __post_init__(self):
        if self.required_fields is None:
            self.required_fields = []


class TaskStatus(Enum):
    """Task execution status"""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    RETRYING = "retrying"


@dataclass
class TaskResult:
    """Standardized task result format"""
    task_id: str
    agent_name: str
    status: TaskStatus
    data: Dict
    processing_time: float
    confidence: float
    error_message: Optional[str] = None
    timestamp: str = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now().isoformat()
    
    @property
    def success(self) -> bool:
        return self.status == TaskStatus.COMPLETED


@dataclass  
class AgentMessage:
    """Standardized agent communication message"""
    message_id: str
    sender: str
    recipient: str
    message_type: str
    data: Dict
    priority: int = 1
    timestamp: str = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now().isoformat()


# ==============================================================================
# CLAUDE LLM PROVIDER MANAGEMENT SYSTEM
# ==============================================================================

class ClaudeLLMProvider:
    """
    Claude LLM Provider with comprehensive error handling and usage tracking
    """
    
    def __init__(self):
        self.client = None
        self.usage_stats = {
            "requests": 0,
            "tokens": 0,
            "errors": 0,
            "total_cost_estimate": 0.0
        }
        
        # Initialize Claude client
        self._initialize_claude()
        
    def _initialize_claude(self):
        """Initialize Claude client"""
        
        try:
            import os
            claude_key = os.getenv("ANTHROPIC_API_KEY")
            if not claude_key:
                raise ValueError("ANTHROPIC_API_KEY environment variable not set")
            
            # Initialize Claude client with minimal parameters
            self.client = anthropic.Anthropic(
                api_key=claude_key,
                # Removed proxies parameter for compatibility
            )
            logging.info("Claude provider initialized successfully")
            
        except Exception as e:
            logging.error(f"Claude provider initialization failed: {e}")
            raise
    
    async def generate_analysis(self, 
                              prompt: str, 
                              context: Dict = None,
                              max_tokens: int = 2000,
                              temperature: float = 0.1,
                              model: str = "claude-3-sonnet-20240229") -> str:
        """
        Generate analysis using Claude
        
        Args:
            prompt: Analysis prompt
            context: Additional context data
            max_tokens: Maximum response tokens
            temperature: Response creativity (0-1)
            model: Claude model to use
            
        Returns:
            str: Generated analysis
            
        Raises:
            Exception: If Claude API fails
        """
        
        if not self.client:
            raise Exception("Claude client not initialized")
        
        context = context or {}
        
        try:
            # Format context into prompt
            if context:
                context_str = "\n".join([f"{k}: {v}" for k, v in context.items()])
                full_prompt = f"Context:\n{context_str}\n\nTask:\n{prompt}"
            else:
                full_prompt = prompt
            
            # Make API call
            message = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.client.messages.create(
                    model=model,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    messages=[
                        {"role": "user", "content": full_prompt}
                    ]
                )
            )
            
            # Update usage stats
            self.usage_stats["requests"] += 1
            self.usage_stats["tokens"] += message.usage.input_tokens + message.usage.output_tokens
            
            # Estimate cost (rough approximation for Sonnet)
            input_cost = message.usage.input_tokens * 0.000003  # $3 per 1M input tokens
            output_cost = message.usage.output_tokens * 0.000015  # $15 per 1M output tokens
            self.usage_stats["total_cost_estimate"] += input_cost + output_cost
            
            return message.content[0].text
            
        except Exception as e:
            self.usage_stats["errors"] += 1
            logging.error(f"Claude API error: {e}")
            raise Exception(f"Claude API error: {e}")
    
    async def generate_structured_analysis(self,
                                         prompt: str,
                                         structure_template: Dict,
                                         context: Dict = None) -> Dict:
        """
        Generate structured analysis with JSON output
        
        Args:
            prompt: Analysis prompt
            structure_template: Expected JSON structure
            context: Additional context data
            
        Returns:
            Dict: Structured analysis result
        """
        
        # Add structure instructions to prompt
        structure_prompt = f"""
{prompt}

Please provide your response in the following JSON structure:
{json.dumps(structure_template, indent=2)}

Ensure your response is valid JSON that matches this structure exactly.
"""
        
        try:
            response = await self.generate_analysis(structure_prompt, context)
            
            # Try to parse JSON response
            try:
                # Extract JSON from response (in case there's extra text)
                start_idx = response.find('{')
                end_idx = response.rfind('}') + 1
                
                if start_idx != -1 and end_idx != 0:
                    json_str = response[start_idx:end_idx]
                    return json.loads(json_str)
                else:
                    # Fallback: try to parse entire response
                    return json.loads(response)
                    
            except json.JSONDecodeError:
                # Return error structure if JSON parsing fails
                return {
                    "error": "Failed to parse JSON response",
                    "raw_response": response,
                    "success": False
                }
                
        except Exception as e:
            return {
                "error": f"Analysis generation failed: {str(e)}",
                "success": False
            }
    
    def get_usage_stats(self) -> Dict:
        """Get Claude usage statistics"""
        return self.usage_stats.copy()
    
    def reset_usage_stats(self) -> None:
        """Reset usage statistics"""
        self.usage_stats = {
            "requests": 0,
            "tokens": 0,
            "errors": 0,
            "total_cost_estimate": 0.0
        }


# ==============================================================================
# BASE AGENT FRAMEWORK
# ==============================================================================

class BaseAgent(ABC):
    """
    Abstract base class for all trading agents
    
    Provides standardized functionality:
    - Lifecycle management
    - Error handling and recovery
    - Performance tracking
    - Claude LLM integration
    - Input validation
    - Output formatting
    """
    
    def __init__(self, agent_name: str, llm_provider: ClaudeLLMProvider, config: AgentConfig):
        self.agent_name = agent_name
        self.llm_provider = llm_provider
        self.config = config
        
        # Agent state
        self.is_running = False
        self.task_queue = asyncio.Queue()
        self.active_tasks = {}
        
        # Performance tracking
        self.total_tasks = 0
        self.successful_tasks = 0
        self.failed_tasks = 0
        self.total_processing_time = 0.0
        self.performance_history = []
        
        # Setup logging
        self.logger = logging.getLogger(f"agent.{agent_name}")
        
        # Required fields validation
        self.required_fields = config.required_fields or []
        
    async def start(self) -> None:
        """Start the agent"""
        
        try:
            await self._initialize()
            self.is_running = True
            self.logger.info(f"Agent {self.agent_name} started successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to start agent {self.agent_name}: {e}")
            raise
    
    async def stop(self) -> None:
        """Stop the agent gracefully"""
        
        try:
            self.is_running = False
            
            # Wait for active tasks to complete
            while self.active_tasks:
                await asyncio.sleep(0.1)
            
            await self._cleanup()
            self.logger.info(f"Agent {self.agent_name} stopped successfully")
            
        except Exception as e:
            self.logger.error(f"Error stopping agent {self.agent_name}: {e}")
            raise
    
    async def process_task(self, task_data: Dict) -> TaskResult:
        """
        Process a single task with comprehensive error handling
        
        Args:
            task_data: Task input data
            
        Returns:
            TaskResult: Standardized task result
        """
        
        task_id = str(uuid.uuid4())
        start_time = time.time()
        
        try:
            # Track active task
            self.active_tasks[task_id] = {
                "start_time": start_time,
                "data": task_data
            }
            
            # Validate input
            validated_data = await self._validate_input(task_data)
            
            # Assess data quality
            data_quality = await self._assess_data_quality(validated_data)
            
            # Process with retries
            result_data = None
            last_error = None
            
            for attempt in range(self.config.max_retries):
                try:
                    # Set timeout
                    result_data = await asyncio.wait_for(
                        self._process_specific_task(validated_data),
                        timeout=self.config.timeout_seconds
                    )
                    break
                    
                except asyncio.TimeoutError:
                    last_error = f"Task timeout after {self.config.timeout_seconds}s"
                    self.logger.warning(f"Attempt {attempt + 1} timeout for task {task_id}")
                    
                except Exception as e:
                    last_error = str(e)
                    self.logger.warning(f"Attempt {attempt + 1} failed for task {task_id}: {e}")
                    
                    if attempt < self.config.max_retries - 1:
                        await asyncio.sleep(2 ** attempt)  # Exponential backoff
            
            # Calculate metrics
            processing_time = time.time() - start_time
            
            if result_data is not None:
                # Success
                confidence = self._calculate_confidence(result_data, data_quality)
                
                task_result = TaskResult(
                    task_id=task_id,
                    agent_name=self.agent_name,
                    status=TaskStatus.COMPLETED,
                    data=result_data,
                    processing_time=processing_time,
                    confidence=confidence
                )
                
                self.successful_tasks += 1
                
            else:
                # Failure
                task_result = TaskResult(
                    task_id=task_id,
                    agent_name=self.agent_name,
                    status=TaskStatus.FAILED,
                    data={},
                    processing_time=processing_time,
                    confidence=0.0,
                    error_message=last_error
                )
                
                self.failed_tasks += 1
            
            # Update performance metrics
            self.total_tasks += 1
            self.total_processing_time += processing_time
            self._update_performance_metrics(processing_time, task_result.success)
            
            return task_result
            
        except Exception as e:
            # Catastrophic failure
            processing_time = time.time() - start_time
            
            task_result = TaskResult(
                task_id=task_id,
                agent_name=self.agent_name,
                status=TaskStatus.FAILED,
                data={},
                processing_time=processing_time,
                confidence=0.0,
                error_message=f"Catastrophic failure: {str(e)}"
            )
            
            self.failed_tasks += 1
            self.total_tasks += 1
            self.logger.error(f"Catastrophic failure in task {task_id}: {e}")
            
            return task_result
            
        finally:
            # Clean up active task tracking
            if task_id in self.active_tasks:
                del self.active_tasks[task_id]
    
    @abstractmethod
    async def _process_specific_task(self, task_data: Dict) -> Dict:
        """
        Agent-specific task processing logic
        
        Args:
            task_data: Validated input data
            
        Returns:
            Dict: Agent-specific processing result
        """
        pass
    
    async def _initialize(self) -> None:
        """
        Agent-specific initialization logic
        Override in subclasses if needed
        """
        pass
    
    async def _cleanup(self) -> None:
        """
        Agent-specific cleanup logic
        Override in subclasses if needed
        """
        pass
    
    async def _validate_input(self, task_data: Dict) -> Dict:
        """
        Validate and sanitize input data
        
        Args:
            task_data: Raw input data
            
        Returns:
            Dict: Validated and sanitized data
            
        Raises:
            ValueError: If validation fails
        """
        
        if not isinstance(task_data, dict):
            raise ValueError("Task data must be a dictionary")
        
        # Check required fields
        missing_fields = [field for field in self.required_fields if field not in task_data]
        if missing_fields:
            raise ValueError(f"Missing required fields: {missing_fields}")
        
        # Remove None values and empty strings
        cleaned_data = {
            k: v for k, v in task_data.items() 
            if v is not None and v != ""
        }
        
        return cleaned_data
    
    async def _assess_data_quality(self, task_data: Dict) -> str:
        """
        Assess the quality of input data
        
        Args:
            task_data: Validated input data
            
        Returns:
            str: Data quality assessment (excellent/good/fair/poor)
        """
        
        if not task_data:
            return "poor"
        
        # Check completeness
        total_fields = len(self.required_fields) if self.required_fields else len(task_data)
        present_fields = len([f for f in self.required_fields if f in task_data])
        
        if total_fields == 0:
            return "excellent"
        
        completeness = present_fields / total_fields
        
        if completeness >= 0.9:
            return "excellent"
        elif completeness >= 0.7:
            return "good"
        elif completeness >= 0.5:
            return "fair"
        else:
            return "poor"
    
    def _calculate_confidence(self, result_data: Dict, data_quality: str) -> float:
        """
        Calculate confidence score for task result
        
        Args:
            result_data: Task processing result
            data_quality: Input data quality assessment
            
        Returns:
            float: Confidence score (0-1)
        """
        
        # Base confidence from data quality
        quality_scores = {
            "excellent": 1.0,
            "good": 0.8,
            "fair": 0.6,
            "poor": 0.4
        }
        
        base_confidence = quality_scores.get(data_quality, 0.5)
        
        # Adjust based on result completeness
        if isinstance(result_data, dict):
            result_completeness = len(result_data) / max(10, len(result_data))  # Assume 10 fields ideal
            result_completeness = min(1.0, result_completeness)
        else:
            result_completeness = 0.5
        
        # Calculate final confidence
        final_confidence = (base_confidence + result_completeness) / 2
        return round(final_confidence, 3)
    
    def _update_performance_metrics(self, processing_time: float, success: bool) -> None:
        """
        Update agent performance metrics
        
        Args:
            processing_time: Task processing time in seconds
            success: Whether task succeeded
        """
        
        # Add to performance history
        performance_point = {
            "timestamp": datetime.now().isoformat(),
            "processing_time": processing_time,
            "success": success,
            "confidence": self._calculate_confidence({}, "good") if success else 0.0
        }
        
        self.performance_history.append(performance_point)
        
        # Keep only last 1000 entries
        if len(self.performance_history) > 1000:
            self.performance_history = self.performance_history[-1000:]
    
    def get_performance_summary(self) -> Dict:
        """Get agent performance summary"""
        
        if self.total_tasks == 0:
            return {
                "agent_name": self.agent_name,
                "total_tasks": 0,
                "success_rate": 0.0,
                "average_processing_time": 0.0,
                "status": "inactive"
            }
        
        success_rate = self.successful_tasks / self.total_tasks
        avg_processing_time = self.total_processing_time / self.total_tasks
        
        return {
            "agent_name": self.agent_name,
            "total_tasks": self.total_tasks,
            "successful_tasks": self.successful_tasks,
            "failed_tasks": self.failed_tasks,
            "success_rate": round(success_rate, 3),
            "average_processing_time": round(avg_processing_time, 3),
            "status": "running" if self.is_running else "stopped"
        }


# ==============================================================================
# AGENT COMMUNICATION SYSTEM
# ==============================================================================

class AgentCommunicationManager:
    """
    Manages inter-agent communication with message routing and queuing
    """
    
    def __init__(self):
        self.agents = {}
        self.message_queues = {}
        self.message_history = []
        self.max_history = 10000
        
        # Performance tracking
        self.messages_sent = 0
        self.messages_delivered = 0
        self.messages_failed = 0
        
        self.logger = logging.getLogger("communication")
    
    def register_agent(self, agent_name: str, agent: BaseAgent) -> None:
        """Register an agent for communication"""
        
        self.agents[agent_name] = agent
        self.message_queues[agent_name] = asyncio.Queue()
        self.logger.info(f"Registered agent: {agent_name}")
    
    def unregister_agent(self, agent_name: str) -> None:
        """Unregister an agent"""
        
        if agent_name in self.agents:
            del self.agents[agent_name]
            del self.message_queues[agent_name]
            self.logger.info(f"Unregistered agent: {agent_name}")
    
    async def send_message(self, 
                          sender: str, 
                          recipient: str, 
                          message_type: str, 
                          data: Dict,
                          priority: int = 1) -> bool:
        """
        Send message between agents
        
        Args:
            sender: Sending agent name
            recipient: Receiving agent name
            message_type: Type of message
            data: Message data
            priority: Message priority (1=highest, 5=lowest)
            
        Returns:
            bool: True if message sent successfully
        """
        
        try:
            if recipient not in self.agents:
                self.logger.error(f"Recipient agent {recipient} not found")
                self.messages_failed += 1
                return False
            
            # Create message
            message = AgentMessage(
                message_id=str(uuid.uuid4()),
                sender=sender,
                recipient=recipient,
                message_type=message_type,
                data=data,
                priority=priority
            )
            
            # Add to recipient's queue
            await self.message_queues[recipient].put(message)
            
            # Track in history
            self.message_history.append(message)
            if len(self.message_history) > self.max_history:
                self.message_history = self.message_history[-self.max_history:]
            
            self.messages_sent += 1
            self.messages_delivered += 1
            
            self.logger.debug(f"Message sent: {sender} -> {recipient} ({message_type})")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to send message: {e}")
            self.messages_failed += 1
            return False
    
    async def receive_message(self, agent_name: str, timeout: float = 1.0) -> Optional[AgentMessage]:
        """
        Receive message for an agent
        
        Args:
            agent_name: Agent name
            timeout: Timeout in seconds
            
        Returns:
            AgentMessage or None if timeout
        """
        
        try:
            if agent_name not in self.message_queues:
                return None
            
            message = await asyncio.wait_for(
                self.message_queues[agent_name].get(),
                timeout=timeout
            )
            
            self.logger.debug(f"Message received by {agent_name}: {message.message_type}")
            return message
            
        except asyncio.TimeoutError:
            return None
        except Exception as e:
            self.logger.error(f"Error receiving message for {agent_name}: {e}")
            return None
    
    async def broadcast_message(self, 
                              sender: str, 
                              message_type: str, 
                              data: Dict,
                              exclude: List[str] = None) -> int:
        """
        Broadcast message to all agents
        
        Args:
            sender: Sending agent name
            message_type: Type of message
            data: Message data
            exclude: List of agents to exclude
            
        Returns:
            int: Number of agents message sent to
        """
        
        exclude = exclude or [sender]
        sent_count = 0
        
        for agent_name in self.agents:
            if agent_name not in exclude:
                success = await self.send_message(sender, agent_name, message_type, data)
                if success:
                    sent_count += 1
        
        return sent_count
    
    def get_communication_stats(self) -> Dict:
        """Get communication statistics"""
        
        return {
            "registered_agents": len(self.agents),
            "messages_sent": self.messages_sent,
            "messages_delivered": self.messages_delivered,
            "messages_failed": self.messages_failed,
            "delivery_rate": self.messages_delivered / max(1, self.messages_sent),
            "message_history_size": len(self.message_history)
        }


# ==============================================================================
# SYSTEM PERFORMANCE TRACKING
# ==============================================================================

@dataclass
class SystemMetric:
    """System performance metric"""
    metric_name: str
    value: float
    unit: str
    timestamp: str = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now().isoformat()


@dataclass
class AlertData:
    """System alert data"""
    alert_id: str
    alert_type: str
    severity: str  # "info", "warning", "error", "critical"
    message: str
    data: Dict
    timestamp: str = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now().isoformat()


class SystemPerformanceTracker:
    """
    Comprehensive system performance monitoring and alerting
    """
    
    def __init__(self):
        self.metrics_history = []
        self.agent_performance = {}
        self.alerts = []
        self.system_metrics = {}
        
        # Alert thresholds
        self.alert_thresholds = {
            "agent_failure_rate": 0.1,  # 10% failure rate triggers alert
            "system_response_time": 30.0,  # 30 second response time
            "memory_usage": 0.8,  # 80% memory usage
            "error_rate": 0.05  # 5% error rate
        }
        
        self.logger = logging.getLogger("performance")
        
        # Start monitoring
        self._start_system_monitoring()
    
    def _start_system_monitoring(self):
        """Start background system monitoring"""
        
        def monitor_loop():
            while True:
                try:
                    self._collect_system_metrics()
                    self._check_alert_conditions()
                    time.sleep(60)  # Check every minute
                except Exception as e:
                    self.logger.error(f"System monitoring error: {e}")
                    time.sleep(60)
        
        monitor_thread = threading.Thread(target=monitor_loop, daemon=True)
        monitor_thread.start()
    
    def _collect_system_metrics(self):
        """Collect system performance metrics"""
        
        try:
            import psutil
            
            # CPU and Memory
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            
            self.system_metrics.update({
                "cpu_usage_percent": cpu_percent,
                "memory_usage_percent": memory.percent,
                "memory_available_gb": memory.available / (1024**3),
                "timestamp": datetime.now().isoformat()
            })
            
            # Record metrics
            self.record_metric("cpu_usage", cpu_percent, "percent")
            self.record_metric("memory_usage", memory.percent, "percent")
            
        except ImportError:
            # If psutil not available, use basic Python metrics
            self.system_metrics.update({
                "cpu_usage_percent": 0.0,
                "memory_usage_percent": 0.0,
                "memory_available_gb": 0.0,
                "timestamp": datetime.now().isoformat()
            })
        except Exception as e:
            self.logger.warning(f"Failed to collect system metrics: {e}")
    
    def _check_alert_conditions(self):
        """Check for alert conditions"""
        
        # Check agent failure rates
        for agent_name, performance in self.agent_performance.items():
            if performance["total_tasks"] > 10:  # Only alert after sufficient data
                failure_rate = performance["failed_tasks"] / performance["total_tasks"]
                
                if failure_rate > self.alert_thresholds["agent_failure_rate"]:
                    self.create_alert(
                        alert_type="agent_failure_rate",
                        severity="warning",
                        message=f"Agent {agent_name} failure rate {failure_rate:.1%} exceeds threshold",
                        data={"agent": agent_name, "failure_rate": failure_rate}
                    )
        
        # Check system memory usage
        if "memory_usage_percent" in self.system_metrics:
            memory_usage = self.system_metrics["memory_usage_percent"] / 100
            
            if memory_usage > self.alert_thresholds["memory_usage"]:
                self.create_alert(
                    alert_type="memory_usage",
                    severity="warning" if memory_usage < 0.9 else "error",
                    message=f"System memory usage {memory_usage:.1%} exceeds threshold",
                    data={"memory_usage": memory_usage}
                )
    
    def record_metric(self, metric_name: str, value: float, unit: str) -> None:
        """Record a system metric"""
        
        metric = SystemMetric(
            metric_name=metric_name,
            value=value,
            unit=unit
        )
        
        self.metrics_history.append(metric)
        
        # Keep only last 10,000 metrics
        if len(self.metrics_history) > 10000:
            self.metrics_history = self.metrics_history[-10000:]
    
    def record_agent_performance(self, agent_name: str, task_result: TaskResult) -> None:
        """Record agent performance data"""
        
        if agent_name not in self.agent_performance:
            self.agent_performance[agent_name] = {
                "total_tasks": 0,
                "successful_tasks": 0,
                "failed_tasks": 0,
                "total_processing_time": 0.0,
                "performance_history": []
            }
        
        stats = self.agent_performance[agent_name]
        stats["total_tasks"] += 1
        stats["total_processing_time"] += task_result.processing_time
        
        if task_result.success:
            stats["successful_tasks"] += 1
        else:
            stats["failed_tasks"] += 1
        
        # Add to performance history
        performance_point = {
            "timestamp": task_result.timestamp,
            "processing_time": task_result.processing_time,
            "success": task_result.success,
            "confidence": task_result.confidence
        }
        
        stats["performance_history"].append(performance_point)
        
        # Keep only last 1000 entries per agent
        if len(stats["performance_history"]) > 1000:
            stats["performance_history"] = stats["performance_history"][-1000:]
    
    def create_alert(self, 
                    alert_type: str, 
                    severity: str, 
                    message: str, 
                    data: Dict) -> None:
        """Create system alert"""
        
        alert = AlertData(
            alert_id=str(uuid.uuid4()),
            alert_type=alert_type,
            severity=severity,
            message=message,
            data=data
        )
        
        self.alerts.append(alert)
        
        # Keep only last 1000 alerts
        if len(self.alerts) > 1000:
            self.alerts = self.alerts[-1000:]
        
        # Log alert
        log_level = {
            "info": logging.INFO,
            "warning": logging.WARNING,
            "error": logging.ERROR,
            "critical": logging.CRITICAL
        }.get(severity, logging.INFO)
        
        self.logger.log(log_level, f"ALERT [{severity.upper()}] {alert_type}: {message}")
    
    def get_system_health_report(self, hours: int = 24) -> Dict:
        """
        Generate comprehensive system health report
        
        Args:
            hours: Number of hours to include in report
            
        Returns:
            Dict: System health report
        """
        
        cutoff = datetime.now() - timedelta(hours=hours)
        
        # Get metrics for time period
        relevant_metrics = [
            metric for metric in self.metrics_history
            if datetime.fromisoformat(metric.timestamp) > cutoff
        ]
        
        # Get agent performance for time period
        agent_reports = {}
        for agent_name, stats in self.agent_performance.items():
            relevant_history = [
                point for point in stats["performance_history"]
                if datetime.fromisoformat(point["timestamp"]) > cutoff
            ]
            
            if relevant_history:
                total_tasks = len(relevant_history)
                successful_tasks = len([p for p in relevant_history if p["success"]])
                avg_time = sum(p["processing_time"] for p in relevant_history) / total_tasks
                avg_confidence = sum(p["confidence"] for p in relevant_history) / total_tasks
                
                agent_reports[agent_name] = {
                    "total_tasks": total_tasks,
                    "successful_tasks": successful_tasks,
                    "success_rate_percent": round((successful_tasks / total_tasks) * 100, 2),
                    "average_processing_time": round(avg_time, 3),
                    "average_confidence": round(avg_confidence, 3)
                }
        
        # Get alerts for time period
        relevant_alerts = [
            alert for alert in self.alerts
            if datetime.fromisoformat(alert.timestamp) > cutoff
        ]
        
        alert_summary = {}
        for alert in relevant_alerts:
            alert_type = alert.alert_type
            alert_summary[alert_type] = alert_summary.get(alert_type, 0) + 1
        
        return {
            "report_period_hours": hours,
            "total_metrics_collected": len(relevant_metrics),
            "agent_performance": agent_reports,
            "alerts_generated": len(relevant_alerts),
            "alert_breakdown": alert_summary,
            "system_metrics": self.system_metrics.copy()
        }


# ==============================================================================
# REPORTING SYSTEM (LOCAL/BLUESKY READY)
# ==============================================================================

class ReportingManager:
    """
    Manages system reporting with local logging and future Bluesky integration
    """
    
    def __init__(self, performance_tracker: SystemPerformanceTracker):
        self.performance_tracker = performance_tracker
        self.logger = logging.getLogger("reporting")
        
        # Future Bluesky integration placeholder
        self.bluesky_enabled = False
        self.bluesky_client = None
    
    async def generate_daily_report(self) -> str:
        """Generate daily performance report"""
        
        health_report = self.performance_tracker.get_system_health_report(hours=24)
        
        report_lines = [
            "🤖 AI Trading System - Daily Report",
            f"📅 Date: {datetime.now().strftime('%Y-%m-%d')}",
            "",
            "📊 Agent Performance:",
        ]
        
        for agent_name, stats in health_report["agent_performance"].items():
            report_lines.extend([
                f"  • {agent_name}:",
                f"    - Tasks: {stats['total_tasks']} ({stats['success_rate_percent']}% success)",
                f"    - Avg Time: {stats['average_processing_time']}s",
                f"    - Confidence: {stats['average_confidence']:.2f}",
            ])
        
        if health_report["alerts_generated"] > 0:
            report_lines.extend([
                "",
                "⚠️ Alerts:",
                f"  Total alerts: {health_report['alerts_generated']}"
            ])
            
            for alert_type, count in health_report["alert_breakdown"].items():
                report_lines.append(f"  • {alert_type}: {count}")
        
        report_lines.extend([
            "",
            "💻 System Health:",
            f"  • Metrics Collected: {health_report['total_metrics_collected']}",
            f"  • CPU Usage: {health_report['system_metrics'].get('cpu_usage_percent', 0):.1f}%",
            f"  • Memory Usage: {health_report['system_metrics'].get('memory_usage_percent', 0):.1f}%",
        ])
        
        return "\n".join(report_lines)
    
    async def send_alert(self, alert: AlertData) -> bool:
        """
        Send alert (currently logs, future Bluesky integration)
        
        Args:
            alert: Alert data to send
            
        Returns:
            bool: True if sent successfully
        """
        
        try:
            # Log alert locally
            alert_message = f"ALERT: {alert.message}"
            self.logger.warning(alert_message)
            
            # Future: Send to Bluesky if enabled
            if self.bluesky_enabled and self.bluesky_client:
                # Placeholder for future Bluesky integration
                pass
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to send alert: {e}")
            return False
    
    async def send_daily_report(self) -> bool:
        """
        Send daily report (currently logs, future Bluesky integration)
        
        Returns:
            bool: True if sent successfully
        """
        
        try:
            report = await self.generate_daily_report()
            
            # Log report locally
            self.logger.info(f"Daily Report:\n{report}")
            
            # Future: Send to Bluesky if enabled
            if self.bluesky_enabled and self.bluesky_client:
                # Placeholder for future Bluesky integration
                pass
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to send daily report: {e}")
            return False
    
    def enable_bluesky_reporting(self, bluesky_config: Dict) -> bool:
        """
        Enable Bluesky reporting (future implementation)
        
        Args:
            bluesky_config: Bluesky API configuration
            
        Returns:
            bool: True if enabled successfully
        """
        
        # Placeholder for future Bluesky integration
        self.logger.info("Bluesky reporting configuration saved (implementation pending)")
        return True


# ==============================================================================
# TESTING FRAMEWORK
# ==============================================================================

async def run_core_infrastructure_tests() -> Dict:
    """
    Comprehensive testing framework for core infrastructure
    
    Returns:
        Dict: Test results summary
    """
    
    results = {
        "test_suite": "Core Infrastructure",
        "start_time": datetime.now().isoformat(),
        "passed_tests": 0,
        "failed_tests": 0,
        "details": [],
        "overall_status": "UNKNOWN"
    }
    
    # Initialize shared variables
    llm_provider = None
    agent = None
    comm_manager = None
    perf_tracker = None
    
    # Test 1: Claude LLM Provider Initialization
    try:
        llm_provider = get_claude_llm_provider()
        assert hasattr(llm_provider, 'client')
        assert hasattr(llm_provider, 'usage_stats')
        assert llm_provider.client is not None
        
        results["passed_tests"] += 1
        results["details"].append("✅ Claude LLM Provider initialization: PASS")
        
    except Exception as e:
        results["failed_tests"] += 1
        results["details"].append(f"❌ Claude LLM Provider initialization: FAIL - {str(e)}")
    
    # Test 2: Base Agent Framework
    try:
        if llm_provider is None:
            # Skip this test if LLM provider failed
            results["failed_tests"] += 1
            results["details"].append("❌ Base Agent framework: SKIP - LLM Provider not available")
        else:
            class TestAgent(BaseAgent):
                async def _process_specific_task(self, task_data):
                    return {"symbol": task_data["symbol"], "action": "analyzed"}
            
            config = AgentConfig(
                agent_name="test_agent",
                required_fields=["symbol"]
            )
            
            agent = TestAgent("test_agent", llm_provider, config)
            await agent.start()
            
            # Test task processing
            task_result = await agent.process_task({"symbol": "AAPL"})
            assert task_result.success
            assert task_result.data["symbol"] == "AAPL"
            
            await agent.stop()
            
            results["passed_tests"] += 1
            results["details"].append("✅ Base Agent framework: PASS")
        
    except Exception as e:
        results["failed_tests"] += 1
        results["details"].append(f"❌ Base Agent framework: FAIL - {str(e)}")
    
    # Test 3: Agent Communication System
    try:
        if agent is None:
            # Create a minimal agent for testing if needed
            class MinimalTestAgent(BaseAgent):
                async def _process_specific_task(self, task_data):
                    return {"test": "success"}
            
            config = AgentConfig(agent_name="minimal_agent")
            agent = MinimalTestAgent("minimal_agent", llm_provider or get_claude_llm_provider(), config)
        
        comm_manager = get_communication_manager()
        
        # Register test agents
        comm_manager.register_agent("agent1", agent)
        comm_manager.register_agent("agent2", agent)
        
        # Test message sending
        success = await comm_manager.send_message(
            sender="agent1",
            recipient="agent2", 
            message_type="test",
            data={"message": "hello"}
        )
        assert success
        
        # Test message receiving
        message = await comm_manager.receive_message("agent2", timeout=1.0)
        assert message is not None
        assert message.sender == "agent1"
        assert message.data["message"] == "hello"
        
        results["passed_tests"] += 1
        results["details"].append("✅ Agent communication system: PASS")
        
    except Exception as e:
        results["failed_tests"] += 1
        results["details"].append(f"❌ Agent communication system: FAIL - {str(e)}")
    
    # Test 4: Performance Tracking
    try:
        perf_tracker = get_performance_tracker()
        
        # Record test metrics
        perf_tracker.record_metric("test_metric", 95.5, "percent")
        
        # Record test agent performance
        test_task_result = TaskResult(
            task_id="test_task",
            agent_name="test_agent",
            status=TaskStatus.COMPLETED,
            data={"result": "success"},
            processing_time=0.5,
            confidence=0.9
        )
        
        perf_tracker.record_agent_performance("test_agent", test_task_result)
        
        # Get health report
        health_report = perf_tracker.get_system_health_report()
        assert "agent_performance" in health_report
        
        results["passed_tests"] += 1
        results["details"].append("✅ Performance tracking: PASS")
        
    except Exception as e:
        results["failed_tests"] += 1
        results["details"].append(f"❌ Performance tracking: FAIL - {str(e)}")
    
    # Test 5: Error Handling and Recovery
    try:
        if llm_provider is None:
            # Skip this test if LLM provider failed
            results["failed_tests"] += 1
            results["details"].append("❌ Error handling and recovery: SKIP - LLM Provider not available")
        else:
            class FailingAgent(BaseAgent):
                async def _process_specific_task(self, task_data):
                    if task_data.get("should_fail", False):
                        raise Exception("Intentional test failure")
                    return {"status": "success"}
            
            config = AgentConfig(
                agent_name="failing_agent",
                max_retries=2
            )
            
            failing_agent = FailingAgent("failing_agent", llm_provider, config)
            await failing_agent.start()
            
            # Test failure handling
            task_result = await failing_agent.process_task({"should_fail": True})
            assert not task_result.success
            assert task_result.error_message is not None
            
            # Test success after failure
            task_result = await failing_agent.process_task({"should_fail": False})
            assert task_result.success
            
            await failing_agent.stop()
            
            results["passed_tests"] += 1
            results["details"].append("✅ Error handling and recovery: PASS")
        
    except Exception as e:
        results["failed_tests"] += 1
        results["details"].append(f"❌ Error handling and recovery: FAIL - {str(e)}")
    
    # Test 6: Reporting System
    try:
        reporting_manager = get_reporting_manager()
        
        # Test report generation
        daily_report = await reporting_manager.generate_daily_report()
        assert "AI Trading System" in daily_report
        
        # Test alert sending
        test_alert = AlertData(
            alert_id="test_alert",
            alert_type="test",
            severity="info",
            message="Test alert",
            data={}
        )
        
        success = await reporting_manager.send_alert(test_alert)
        assert success
        
        results["passed_tests"] += 1
        results["details"].append("✅ Reporting system: PASS")
        
    except Exception as e:
        results["failed_tests"] += 1
        results["details"].append(f"❌ Reporting system: FAIL - {str(e)}")
    
    # Calculate overall status
    total_tests = results["passed_tests"] + results["failed_tests"]
    success_rate = results["passed_tests"] / total_tests if total_tests > 0 else 0
    
    if success_rate >= 0.95:
        results["overall_status"] = "PASS"
    elif success_rate >= 0.8:
        results["overall_status"] = "PARTIAL"
    else:
        results["overall_status"] = "FAIL"
    
    results["end_time"] = datetime.now().isoformat()
    results["success_rate"] = round(success_rate * 100, 2)
    
    return results


# ==============================================================================
# GLOBAL INSTANCES (Initialize on demand)
# ==============================================================================

# Global instances - will be created when first accessed
_performance_tracker = None
_communication_manager = None
_claude_llm_provider = None
_reporting_manager = None

def get_performance_tracker():
    """Get global performance tracker instance"""
    global _performance_tracker
    if _performance_tracker is None:
        _performance_tracker = SystemPerformanceTracker()
    return _performance_tracker

def get_communication_manager():
    """Get global communication manager instance"""
    global _communication_manager
    if _communication_manager is None:
        _communication_manager = AgentCommunicationManager()
    return _communication_manager

def get_claude_llm_provider():
    """Get global Claude LLM provider instance"""
    global _claude_llm_provider
    if _claude_llm_provider is None:
        _claude_llm_provider = ClaudeLLMProvider()
    return _claude_llm_provider

def get_reporting_manager():
    """Get global reporting manager instance"""
    global _reporting_manager
    if _reporting_manager is None:
        _reporting_manager = ReportingManager(get_performance_tracker())
    return _reporting_manager

# Backward compatibility - these will work but initialize on first access
@property
def performance_tracker():
    return get_performance_tracker()

@property  
def communication_manager():
    return get_communication_manager()

@property
def claude_llm_provider():
    return get_claude_llm_provider()

@property
def reporting_manager():
    return get_reporting_manager()


# ==============================================================================
# MODULE INITIALIZATION AND TESTING
# ==============================================================================

if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Run tests
    print("🚀 Running Core Infrastructure Tests...")
    test_results = asyncio.run(run_core_infrastructure_tests())
    
    # Print results
    print(f"\n📊 TEST RESULTS - {test_results['overall_status']}")
    print(f"Success Rate: {test_results['success_rate']}%")
    print(f"Passed: {test_results['passed_tests']}")
    print(f"Failed: {test_results['failed_tests']}")
    
    print("\nDetailed Results:")
    for detail in test_results["details"]:
        print(f"  {detail}")
    
    # Exit with appropriate code
    exit_code = 0 if test_results["overall_status"] == "PASS" else 1
    print(f"\nExiting with code: {exit_code}")
    sys.exit(exit_code)