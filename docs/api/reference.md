# AI Trading System - API Reference Guide

## Table of Contents

1. [Overview](#overview)
2. [Base Classes](#base-classes)
3. [Agent APIs](#agent-apis)
4. [Core Infrastructure APIs](#core-infrastructure-apis)
5. [Data Provider APIs](#data-provider-apis)
6. [Database APIs](#database-apis)
7. [Orchestration APIs](#orchestration-apis)
8. [Utility APIs](#utility-apis)
9. [Error Codes](#error-codes)
10. [Examples](#examples)

---

## Overview

This document provides a comprehensive reference for all public APIs in the AI Trading System. Each API section includes method signatures, parameters, return types, and usage examples.

### API Conventions

- All async methods are prefixed with `async`
- Optional parameters use `Optional[Type]` or default values
- Return types are explicitly annotated
- Errors raise specific exceptions with error codes

### Import Structure

```python
# Agent imports
from src.agents.junior_analyst import JuniorAnalyst
from src.agents.senior_analyst import SeniorAnalyst
from src.agents.economist import Economist
from src.agents.portfolio_manager import PortfolioManager
from src.agents.trade_executor import TradeExecutor
from src.agents.analytics_reporter import AnalyticsReporter

# Core imports
from src.core.base_agent import BaseAgent
from src.core.infrastructure import SystemInfrastructure
from src.core.llm_provider import LLMProvider

# Data imports
from src.data.alpaca_provider import AlpacaDataProvider

# Database imports
from src.database.manager import DatabaseManager
from src.database.config import DatabaseConfig

# Orchestration imports
from src.orchestration.controller import OrchestrationController
from src.orchestration.workflow_engine import WorkflowEngine
```

---

## Base Classes

### BaseAgent

The foundation class for all trading agents.

```python
class BaseAgent(ABC):
    """
    Abstract base class for all trading agents.
    
    Attributes:
        name (str): Agent identifier
        llm_provider (LLMProvider): LLM integration
        db_manager (DatabaseManager): Database access
        config (Dict): Agent configuration
        is_active (bool): Agent active status
    """
    
    def __init__(
        self,
        name: str,
        llm_provider: LLMProvider,
        db_manager: DatabaseManager,
        config: Optional[Dict] = None
    ) -> None:
        """
        Initialize base agent.
        
        Args:
            name: Unique agent identifier
            llm_provider: LLM provider instance
            db_manager: Database manager instance
            config: Optional configuration dictionary
        """
    
    async def initialize(self) -> bool:
        """
        Initialize agent resources.
        
        Returns:
            bool: True if initialization successful
            
        Raises:
            InitializationError: If initialization fails
        """
    
    @abstractmethod
    async def process(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a task (must be implemented by subclasses).
        
        Args:
            task_data: Task information and parameters
            
        Returns:
            Dict containing task results
            
        Raises:
            ProcessingError: If task processing fails
        """
    
    async def validate_input(
        self, 
        data: Dict[str, Any], 
        schema: Dict[str, type]
    ) -> bool:
        """
        Validate input data against schema.
        
        Args:
            data: Input data to validate
            schema: Expected data schema
            
        Returns:
            bool: True if valid
            
        Raises:
            ValidationError: If validation fails
        """
    
    async def get_status(self) -> Dict[str, Any]:
        """
        Get agent status and health metrics.
        
        Returns:
            Dict containing:
                - active (bool): Agent active status
                - tasks_processed (int): Total tasks processed
                - error_rate (float): Recent error rate
                - last_activity (datetime): Last activity timestamp
                - performance_metrics (Dict): Performance data
        """
    
    async def cleanup(self) -> None:
        """
        Cleanup agent resources.
        
        Raises:
            CleanupError: If cleanup fails
        """
```

---

## Agent APIs

### JuniorAnalyst

Market screening and initial analysis agent.

```python
class JuniorAnalyst(BaseAgent):
    """
    Performs initial market screening and technical analysis.
    """
    
    async def screen_stocks(
        self,
        universe: List[str],
        criteria: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Screen stocks based on criteria.
        
        Args:
            universe: List of stock symbols to screen
            criteria: Screening criteria (optional)
                - min_volume: Minimum daily volume
                - min_price: Minimum stock price
                - max_price: Maximum stock price
                - min_market_cap: Minimum market cap
                
        Returns:
            List of screened stocks with scores:
                [{
                    'symbol': 'AAPL',
                    'score': 0.85,
                    'volume': 50000000,
                    'price': 150.00,
                    'signals': ['breakout', 'momentum']
                }]
                
        Example:
            analyst = JuniorAnalyst(...)
            results = await analyst.screen_stocks(
                universe=['AAPL', 'GOOGL', 'MSFT'],
                criteria={'min_volume': 1000000}
            )
        """
    
    async def analyze_technicals(
        self,
        symbol: str,
        period: str = '1d',
        lookback: int = 30
    ) -> Dict[str, Any]:
        """
        Perform technical analysis on a symbol.
        
        Args:
            symbol: Stock symbol
            period: Time period ('1m', '5m', '1h', '1d')
            lookback: Number of periods to analyze
            
        Returns:
            Dict containing:
                - indicators: Technical indicator values
                - signals: Buy/sell signals
                - strength: Signal strength (0-1)
                - pattern: Detected patterns
        """
    
    async def rank_opportunities(
        self,
        candidates: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Rank investment opportunities.
        
        Args:
            candidates: List of candidate stocks
            
        Returns:
            Ranked list of opportunities
        """
```

### SeniorAnalyst

Deep fundamental and strategic analysis.

```python
class SeniorAnalyst(BaseAgent):
    """
    Performs deep fundamental analysis and generates investment theses.
    """
    
    async def analyze_fundamentals(
        self,
        symbol: str,
        include_competitors: bool = False
    ) -> Dict[str, Any]:
        """
        Analyze company fundamentals.
        
        Args:
            symbol: Stock symbol
            include_competitors: Include competitor analysis
            
        Returns:
            Dict containing:
                - financials: Key financial metrics
                - valuation: Valuation metrics
                - growth: Growth metrics
                - quality: Quality scores
                - thesis: Investment thesis
                - confidence: Confidence score (0-1)
                
        Example:
            analyst = SeniorAnalyst(...)
            analysis = await analyst.analyze_fundamentals(
                'AAPL',
                include_competitors=True
            )
        """
    
    async def generate_thesis(
        self,
        symbol: str,
        analysis_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Generate detailed investment thesis.
        
        Args:
            symbol: Stock symbol
            analysis_data: Analysis results
            
        Returns:
            Dict containing:
                - recommendation: BUY/HOLD/SELL
                - thesis: Detailed investment thesis
                - catalysts: Key catalysts
                - risks: Risk factors
                - target_price: Price target
                - confidence: Confidence level
        """
    
    async def evaluate_entry_points(
        self,
        symbol: str,
        current_price: float
    ) -> Dict[str, Any]:
        """
        Evaluate optimal entry points.
        
        Args:
            symbol: Stock symbol
            current_price: Current market price
            
        Returns:
            Dict with entry point analysis
        """
```

### Economist

Macroeconomic analysis and market regime detection.

```python
class Economist(BaseAgent):
    """
    Analyzes macroeconomic conditions and market regimes.
    """
    
    async def analyze_market_regime(
        self,
        indicators: Optional[Dict[str, float]] = None
    ) -> Dict[str, Any]:
        """
        Determine current market regime.
        
        Args:
            indicators: Optional economic indicators
            
        Returns:
            Dict containing:
                - regime: RISK_ON/RISK_OFF/NEUTRAL/TRANSITION
                - confidence: Confidence score
                - indicators: Key indicator values
                - recommendation: Strategic recommendation
                
        Example:
            economist = Economist(...)
            regime = await economist.analyze_market_regime()
            if regime['regime'] == 'RISK_OFF':
                # Reduce exposure
        """
    
    async def analyze_sectors(self) -> Dict[str, Any]:
        """
        Analyze sector rotation opportunities.
        
        Returns:
            Dict containing:
                - rankings: Sector rankings
                - momentum: Sector momentum scores
                - recommendations: Sector allocations
        """
    
    async def get_risk_parameters(
        self,
        regime: str
    ) -> Dict[str, float]:
        """
        Get risk parameters for market regime.
        
        Args:
            regime: Market regime
            
        Returns:
            Dict containing:
                - max_leverage: Maximum leverage
                - position_size: Base position size
                - stop_loss: Stop loss percentage
                - risk_budget: Risk budget
        """
```

### PortfolioManager

Risk management and position sizing.

```python
class PortfolioManager(BaseAgent):
    """
    Manages portfolio risk and generates trading decisions.
    """
    
    async def evaluate_portfolio(
        self,
        portfolio: Dict[str, Any],
        market_conditions: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Evaluate current portfolio.
        
        Args:
            portfolio: Current portfolio state
            market_conditions: Market conditions
            
        Returns:
            Dict containing:
                - health_score: Portfolio health (0-1)
                - risk_metrics: Risk measurements
                - recommendations: Recommended actions
                - rebalance_needed: Boolean
        """
    
    async def size_position(
        self,
        symbol: str,
        signal_strength: float,
        portfolio_value: float,
        existing_position: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Calculate optimal position size using Kelly Criterion.
        
        Args:
            symbol: Stock symbol
            signal_strength: Signal strength (0-1)
            portfolio_value: Total portfolio value
            existing_position: Current position size
            
        Returns:
            Dict containing:
                - position_size: Recommended position size
                - shares: Number of shares
                - risk_amount: Risk in dollars
                - kelly_fraction: Kelly criterion fraction
                
        Example:
            pm = PortfolioManager(...)
            sizing = await pm.size_position(
                'AAPL',
                signal_strength=0.75,
                portfolio_value=100000
            )
        """
    
    async def generate_trades(
        self,
        recommendations: List[Dict[str, Any]],
        portfolio: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Generate trade orders from recommendations.
        
        Args:
            recommendations: List of recommendations
            portfolio: Current portfolio
            
        Returns:
            List of trade orders:
                [{
                    'symbol': 'AAPL',
                    'action': 'BUY',
                    'quantity': 100,
                    'order_type': 'LIMIT',
                    'limit_price': 150.00,
                    'risk_amount': 500
                }]
        """
    
    async def check_risk_limits(
        self,
        trade: Dict[str, Any],
        portfolio: Dict[str, Any]
    ) -> Tuple[bool, Optional[str]]:
        """
        Check if trade violates risk limits.
        
        Args:
            trade: Proposed trade
            portfolio: Current portfolio
            
        Returns:
            Tuple of (is_valid, error_message)
        """
```

### TradeExecutor

Order execution and management.

```python
class TradeExecutor(BaseAgent):
    """
    Executes trades and manages orders.
    """
    
    async def execute_trade(
        self,
        order: Dict[str, Any],
        strategy: str = 'MARKET'
    ) -> Dict[str, Any]:
        """
        Execute a trade order.
        
        Args:
            order: Order details
            strategy: Execution strategy
                - MARKET: Market order
                - LIMIT: Limit order
                - TWAP: Time-weighted average
                - VWAP: Volume-weighted average
                - ICEBERG: Iceberg order
                
        Returns:
            Dict containing:
                - order_id: Order identifier
                - status: Order status
                - filled_quantity: Filled quantity
                - average_price: Average fill price
                - slippage: Execution slippage
                
        Example:
            executor = TradeExecutor(...)
            result = await executor.execute_trade(
                order={
                    'symbol': 'AAPL',
                    'quantity': 100,
                    'side': 'BUY'
                },
                strategy='TWAP'
            )
        """
    
    async def monitor_order(
        self,
        order_id: str
    ) -> Dict[str, Any]:
        """
        Monitor order status.
        
        Args:
            order_id: Order identifier
            
        Returns:
            Dict with order status and details
        """
    
    async def cancel_order(
        self,
        order_id: str,
        reason: Optional[str] = None
    ) -> bool:
        """
        Cancel an open order.
        
        Args:
            order_id: Order identifier
            reason: Cancellation reason
            
        Returns:
            bool: True if cancelled successfully
        """
    
    async def get_execution_analytics(
        self,
        start_date: datetime,
        end_date: datetime
    ) -> Dict[str, Any]:
        """
        Get execution quality analytics.
        
        Args:
            start_date: Start date
            end_date: End date
            
        Returns:
            Dict with execution metrics
        """
```

### AnalyticsReporter

Performance analytics and reporting.

```python
class AnalyticsReporter(BaseAgent):
    """
    Generates performance analytics and reports.
    """
    
    async def generate_daily_report(
        self,
        date: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """
        Generate daily performance report.
        
        Args:
            date: Report date (defaults to today)
            
        Returns:
            Dict containing:
                - summary: Executive summary
                - performance: Performance metrics
                - trades: Trade summary
                - positions: Position summary
                - alerts: Risk alerts
                
        Example:
            reporter = AnalyticsReporter(...)
            report = await reporter.generate_daily_report()
            print(report['summary'])
        """
    
    async def calculate_metrics(
        self,
        portfolio_history: List[Dict[str, Any]],
        benchmark: Optional[str] = 'SPY'
    ) -> Dict[str, float]:
        """
        Calculate performance metrics.
        
        Args:
            portfolio_history: Historical portfolio values
            benchmark: Benchmark symbol
            
        Returns:
            Dict containing:
                - total_return: Total return percentage
                - sharpe_ratio: Sharpe ratio
                - max_drawdown: Maximum drawdown
                - win_rate: Win rate percentage
                - profit_factor: Profit factor
                - alpha: Alpha vs benchmark
                - beta: Beta vs benchmark
        """
    
    async def generate_alert(
        self,
        alert_type: str,
        severity: str,
        message: str,
        data: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Generate system alert.
        
        Args:
            alert_type: Type of alert
            severity: INFO/WARNING/CRITICAL/EMERGENCY
            message: Alert message
            data: Additional alert data
            
        Returns:
            Dict with alert details
        """
    
    async def export_performance(
        self,
        format: str = 'json',
        period: str = 'all'
    ) -> Union[str, Dict[str, Any]]:
        """
        Export performance data.
        
        Args:
            format: Export format (json/csv/html)
            period: Time period
            
        Returns:
            Exported data in requested format
        """
```

---

## Core Infrastructure APIs

### LLMProvider

Claude AI integration and management.

```python
class LLMProvider:
    """
    Manages LLM (Claude) interactions.
    """
    
    def __init__(
        self,
        api_key: str,
        model: str = 'claude-3-opus-20240229',
        max_retries: int = 3,
        timeout: int = 30
    ) -> None:
        """
        Initialize LLM provider.
        
        Args:
            api_key: Anthropic API key
            model: Model identifier
            max_retries: Maximum retry attempts
            timeout: Request timeout in seconds
        """
    
    async def complete(
        self,
        prompt: str,
        max_tokens: int = 1000,
        temperature: float = 0.7,
        system_prompt: Optional[str] = None
    ) -> str:
        """
        Get completion from LLM.
        
        Args:
            prompt: User prompt
            max_tokens: Maximum response tokens
            temperature: Sampling temperature (0-1)
            system_prompt: System prompt
            
        Returns:
            str: Model response
            
        Raises:
            LLMError: If request fails
            
        Example:
            llm = LLMProvider(api_key)
            response = await llm.complete(
                "Analyze AAPL stock",
                max_tokens=500
            )
        """
    
    async def complete_with_retry(
        self,
        prompt: str,
        **kwargs
    ) -> str:
        """
        Complete with automatic retry logic.
        
        Args:
            prompt: User prompt
            **kwargs: Additional parameters
            
        Returns:
            str: Model response
        """
    
    def estimate_tokens(self, text: str) -> int:
        """
        Estimate token count for text.
        
        Args:
            text: Input text
            
        Returns:
            int: Estimated token count
        """
```

### SystemInfrastructure

Core system infrastructure and utilities.

```python
class SystemInfrastructure:
    """
    Core system infrastructure management.
    """
    
    @staticmethod
    async def initialize_system(
        config: Dict[str, Any]
    ) -> bool:
        """
        Initialize system infrastructure.
        
        Args:
            config: System configuration
            
        Returns:
            bool: Success status
        """
    
    @staticmethod
    async def health_check() -> Dict[str, Any]:
        """
        Perform system health check.
        
        Returns:
            Dict containing:
                - database: Database status
                - api_connections: API connection status
                - agents: Agent status
                - memory_usage: Memory usage
                - cpu_usage: CPU usage
        """
    
    @staticmethod
    async def shutdown_system() -> None:
        """
        Gracefully shutdown system.
        """
```

---

## Data Provider APIs

### AlpacaDataProvider

Market data and trading API integration.

```python
class AlpacaDataProvider:
    """
    Provides market data and trading functionality via Alpaca.
    """
    
    def __init__(
        self,
        api_key: str,
        secret_key: str,
        base_url: str = 'https://paper-api.alpaca.markets',
        data_feed: str = 'iex'
    ) -> None:
        """
        Initialize Alpaca data provider.
        
        Args:
            api_key: Alpaca API key
            secret_key: Alpaca secret key
            base_url: API base URL
            data_feed: Data feed source
        """
    
    async def get_market_data(
        self,
        symbols: List[str],
        timeframe: str = '1Day',
        limit: int = 100
    ) -> Dict[str, pd.DataFrame]:
        """
        Get historical market data.
        
        Args:
            symbols: List of symbols
            timeframe: Bar timeframe
            limit: Number of bars
            
        Returns:
            Dict mapping symbols to DataFrames
            
        Example:
            provider = AlpacaDataProvider(...)
            data = await provider.get_market_data(
                ['AAPL', 'GOOGL'],
                timeframe='1Day',
                limit=30
            )
        """
    
    async def get_latest_price(
        self,
        symbol: str
    ) -> float:
        """
        Get latest price for symbol.
        
        Args:
            symbol: Stock symbol
            
        Returns:
            float: Latest price
        """
    
    async def get_account_info(self) -> Dict[str, Any]:
        """
        Get account information.
        
        Returns:
            Dict containing:
                - buying_power: Available buying power
                - portfolio_value: Total portfolio value
                - cash: Cash balance
                - positions: Number of positions
        """
    
    async def submit_order(
        self,
        symbol: str,
        qty: int,
        side: str,
        order_type: str = 'market',
        limit_price: Optional[float] = None,
        stop_price: Optional[float] = None,
        time_in_force: str = 'day'
    ) -> Dict[str, Any]:
        """
        Submit a trading order.
        
        Args:
            symbol: Stock symbol
            qty: Quantity
            side: 'buy' or 'sell'
            order_type: Order type
            limit_price: Limit price
            stop_price: Stop price
            time_in_force: Time in force
            
        Returns:
            Dict with order details
        """
```

---

## Database APIs

### DatabaseManager

Database operations and connection management.

```python
class DatabaseManager:
    """
    Manages database operations and connections.
    """
    
    def __init__(
        self,
        connection_string: str,
        pool_size: int = 5,
        echo: bool = False
    ) -> None:
        """
        Initialize database manager.
        
        Args:
            connection_string: Database connection string
            pool_size: Connection pool size
            echo: Echo SQL statements
        """
    
    @contextmanager
    def get_session(self) -> Session:
        """
        Get database session context manager.
        
        Yields:
            Session: Database session
            
        Example:
            with db_manager.get_session() as session:
                result = session.query(Trade).all()
        """
    
    async def execute_query(
        self,
        query: str,
        params: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Execute raw SQL query.
        
        Args:
            query: SQL query
            params: Query parameters
            
        Returns:
            List of result dictionaries
        """
    
    async def save_trade(
        self,
        trade_data: Dict[str, Any]
    ) -> int:
        """
        Save trade to database.
        
        Args:
            trade_data: Trade information
            
        Returns:
            int: Trade ID
        """
    
    async def get_portfolio_history(
        self,
        start_date: datetime,
        end_date: datetime
    ) -> List[Dict[str, Any]]:
        """
        Get portfolio history.
        
        Args:
            start_date: Start date
            end_date: End date
            
        Returns:
            List of portfolio snapshots
        """
```

---

## Orchestration APIs

### OrchestrationController

Main system orchestration controller.

```python
class OrchestrationController:
    """
    Controls and coordinates all system components.
    """
    
    def __init__(
        self,
        config: TradingConfig
    ) -> None:
        """
        Initialize orchestration controller.
        
        Args:
            config: System configuration
        """
    
    async def run_daily_workflow(self) -> Dict[str, Any]:
        """
        Run complete daily trading workflow.
        
        Returns:
            Dict containing:
                - status: Workflow status
                - trades_executed: Number of trades
                - performance: Daily performance
                - errors: Any errors encountered
                
        Example:
            controller = OrchestrationController(config)
            result = await controller.run_daily_workflow()
        """
    
    async def process_signal(
        self,
        signal: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Process trading signal through pipeline.
        
        Args:
            signal: Trading signal
            
        Returns:
            Dict with processing result
        """
    
    async def emergency_stop(
        self,
        reason: str
    ) -> None:
        """
        Emergency stop all trading.
        
        Args:
            reason: Stop reason
        """
```

### WorkflowEngine

Workflow execution engine.

```python
class WorkflowEngine:
    """
    Executes defined workflows with dependency management.
    """
    
    def __init__(self) -> None:
        """Initialize workflow engine."""
    
    async def execute_workflow(
        self,
        workflow_definition: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Execute a workflow.
        
        Args:
            workflow_definition: Workflow definition
            context: Execution context
            
        Returns:
            Dict with workflow results
            
        Example:
            engine = WorkflowEngine()
            result = await engine.execute_workflow(
                workflow_definition={
                    'name': 'daily_trading',
                    'tasks': [...],
                    'dependencies': {...}
                }
            )
        """
    
    def register_task(
        self,
        task_name: str,
        task_function: Callable
    ) -> None:
        """
        Register a task function.
        
        Args:
            task_name: Task identifier
            task_function: Task implementation
        """
```

---

## Utility APIs

### Logger

Logging utilities and configuration.

```python
class Logger:
    """
    Centralized logging management.
    """
    
    @staticmethod
    def setup_logging(
        name: str,
        level: str = 'INFO',
        log_file: Optional[str] = None
    ) -> logging.Logger:
        """
        Setup logger instance.
        
        Args:
            name: Logger name
            level: Log level
            log_file: Optional log file path
            
        Returns:
            Logger instance
            
        Example:
            logger = Logger.setup_logging(
                'trading_system',
                level='DEBUG',
                log_file='logs/system.log'
            )
        """
```

### PerformanceTracker

Performance monitoring and tracking.

```python
class PerformanceTracker:
    """
    Tracks system and trading performance.
    """
    
    def __init__(self) -> None:
        """Initialize performance tracker."""
    
    async def record_metric(
        self,
        metric_name: str,
        value: float,
        tags: Optional[Dict[str, str]] = None
    ) -> None:
        """
        Record performance metric.
        
        Args:
            metric_name: Metric identifier
            value: Metric value
            tags: Optional tags
        """
    
    async def get_metrics(
        self,
        metric_name: str,
        start_time: datetime,
        end_time: datetime
    ) -> List[Dict[str, Any]]:
        """
        Retrieve metrics for time period.
        
        Args:
            metric_name: Metric identifier
            start_time: Start time
            end_time: End time
            
        Returns:
            List of metric data points
        """
```

---

## Error Codes

### System Error Codes

| Code | Name | Description |
|------|------|-------------|
| `E001` | `INITIALIZATION_ERROR` | System initialization failed |
| `E002` | `DATABASE_ERROR` | Database operation failed |
| `E003` | `API_CONNECTION_ERROR` | External API connection failed |
| `E004` | `CONFIGURATION_ERROR` | Invalid configuration |
| `E005` | `AUTHENTICATION_ERROR` | Authentication failed |

### Agent Error Codes

| Code | Name | Description |
|------|------|-------------|
| `A001` | `AGENT_INITIALIZATION_ERROR` | Agent initialization failed |
| `A002` | `TASK_PROCESSING_ERROR` | Task processing failed |
| `A003` | `VALIDATION_ERROR` | Input validation failed |
| `A004` | `TIMEOUT_ERROR` | Operation timed out |
| `A005` | `RESOURCE_ERROR` | Resource unavailable |

### Trading Error Codes

| Code | Name | Description |
|------|------|-------------|
| `T001` | `ORDER_REJECTED` | Order rejected by broker |
| `T002` | `INSUFFICIENT_FUNDS` | Insufficient buying power |
| `T003` | `RISK_LIMIT_EXCEEDED` | Risk limit violation |
| `T004` | `MARKET_CLOSED` | Market is closed |
| `T005` | `SYMBOL_NOT_FOUND` | Symbol not found |

---

## Examples

### Complete Trading Workflow

```python
import asyncio
from src.orchestration.controller import OrchestrationController
from config.settings import TradingConfig

async def main():
    # Initialize configuration
    config = TradingConfig()
    
    # Create orchestration controller
    controller = OrchestrationController(config)
    
    # Run daily workflow
    result = await controller.run_daily_workflow()
    
    # Check results
    if result['status'] == 'success':
        print(f"Executed {result['trades_executed']} trades")
        print(f"Daily performance: {result['performance']:.2%}")
    else:
        print(f"Workflow failed: {result.get('error')}")

if __name__ == "__main__":
    asyncio.run(main())
```

### Custom Agent Implementation

```python
from src.core.base_agent import BaseAgent
from typing import Dict, Any

class CustomAgent(BaseAgent):
    """Custom trading agent implementation."""
    
    async def process(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process custom trading logic."""
        
        # Validate input
        await self.validate_input(
            task_data,
            {'symbol': str, 'action': str}
        )
        
        # Custom processing logic
        symbol = task_data['symbol']
        action = task_data['action']
        
        # Use LLM for analysis
        analysis = await self.llm_provider.complete(
            f"Analyze {symbol} for {action} opportunity"
        )
        
        # Return results
        return {
            'symbol': symbol,
            'recommendation': action,
            'analysis': analysis,
            'confidence': 0.75
        }

# Usage
agent = CustomAgent(
    name='custom_agent',
    llm_provider=llm,
    db_manager=db
)

result = await agent.process({
    'symbol': 'AAPL',
    'action': 'BUY'
})
```

### Error Handling

```python
from src.agents.trade_executor import TradeExecutor

async def execute_with_error_handling():
    executor = TradeExecutor(...)
    
    try:
        result = await executor.execute_trade(
            order={'symbol': 'AAPL', 'quantity': 100},
            strategy='LIMIT'
        )
    except OrderRejectedError as e:
        print(f"Order rejected: {e.message}")
        # Handle rejection
    except InsufficientFundsError as e:
        print(f"Insufficient funds: {e.required_amount}")
        # Handle insufficient funds
    except Exception as e:
        print(f"Unexpected error: {e}")
        # Handle general error
```

---

*For more information, see the [Architecture Documentation](../architecture/system_architecture.md) and [Development Guide](../guides/development.md).*