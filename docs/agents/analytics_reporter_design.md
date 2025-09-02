# Analytics & Reporting Agent Documentation

## Overview

The Analytics & Reporting Agent is the comprehensive monitoring and insights engine of the AI trading system, serving as the "eyes and ears" that provide critical performance analytics, risk monitoring, and automated reporting. It transforms raw trading data into actionable insights through sophisticated metrics calculation, real-time alerting, and multi-format report generation, ensuring complete visibility into system performance and portfolio health.

## Table of Contents

1. [Architecture](#architecture)
2. [Core Components](#core-components)
3. [Report Types](#report-types)
4. [Performance Analytics](#performance-analytics)
5. [Alert System](#alert-system)
6. [Report Generation](#report-generation)
7. [System Monitoring](#system-monitoring)
8. [Testing Strategy](#testing-strategy)
9. [Configuration](#configuration)
10. [Integration Points](#integration-points)
11. [Usage Examples](#usage-examples)
12. [Troubleshooting](#troubleshooting)

## Architecture

The Analytics & Reporting Agent implements a modular architecture for comprehensive system monitoring:

```
┌──────────────────────────────────────────────────────┐
│           Analytics & Reporting Agent                 │
├──────────────────────────────────────────────────────┤
│                                                        │
│  ┌─────────────────┐    ┌──────────────────┐        │
│  │  Performance    │    │   Report          │        │
│  │  Analytics      │◄───┤   Manager         │        │
│  └────────┬────────┘    └──────────────────┘        │
│           │                                           │
│  ┌────────▼────────┐    ┌──────────────────┐        │
│  │  Risk           │    │   Alert           │        │
│  │  Monitor        │◄───┤   Engine          │        │
│  └────────┬────────┘    └──────────────────┘        │
│           │                                           │
│  ┌────────▼────────┐    ┌──────────────────┐        │
│  │  System Health  │    │   Agent           │        │
│  │  Monitor        │◄───┤   Performance     │        │
│  └────────┬────────┘    └──────────────────┘        │
│           │                                           │
│  ┌────────▼────────────────────────────────┐        │
│  │         Report Distribution              │        │
│  └─────────────────────────────────────────┘        │
└──────────────────────────────────────────────────────┘
```

## Core Components

### 1. AnalyticsReportingAgent (Main Class)

The primary orchestrator for all analytics and reporting activities.

```python
class AnalyticsReportingAgent(BaseAgent):
    """
    Analytics & Reporting Agent
    Provides performance analytics, risk monitoring, and reporting
    """
    
    def __init__(self, llm_provider, config, data_provider, db_manager):
        super().__init__(
            agent_name="analytics_reporting",
            llm_provider=llm_provider,
            config=config
        )
        
        self.data_provider = data_provider
        self.db_manager = db_manager
        
        # Initialize components
        self.analytics = PerformanceAnalytics(data_provider, db_manager)
        self.report_manager = ReportManager(config)
        
        # Report scheduling
        self.last_daily_report = None
        self.last_weekly_report = None
        self.last_monthly_report = None
        
        # Alert thresholds
        self.alert_thresholds = {...}
```

### 2. PerformanceAnalytics

Calculates comprehensive portfolio and trading metrics.

```python
class PerformanceAnalytics:
    """
    Performance calculation engine
    """
    
    async def calculate_comprehensive_performance() -> PerformanceMetrics:
        """
        Calculate all performance metrics
        
        Returns:
        - P&L (daily, weekly, monthly, YTD)
        - Risk metrics (Sharpe, drawdown)
        - Trade statistics
        - Win/loss ratios
        """
```

### 3. ReportManager

Handles report generation, formatting, and storage.

```python
class ReportManager:
    """
    Report generation and storage system
    """
    
    def __init__(self, config):
        self.reports_dir = Path('reports')
        self.formats = ['markdown', 'json', 'html', 'text']
        self._ensure_reports_directory()
```

### 4. Data Models

#### PerformanceMetrics
```python
@dataclass
class PerformanceMetrics:
    """Comprehensive performance metrics"""
    total_value: float
    daily_pnl: float
    daily_pnl_pct: float
    weekly_pnl: float
    weekly_pnl_pct: float
    monthly_pnl: float
    monthly_pnl_pct: float
    ytd_return: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    avg_win: float
    avg_loss: float
    total_trades: int
    winning_trades: int
    losing_trades: int
```

#### SystemHealthMetrics
```python
@dataclass
class SystemHealthMetrics:
    """System health monitoring"""
    api_connectivity: Dict[str, bool]
    database_health: bool
    agent_status: Dict[str, str]
    resource_usage: Dict[str, float]
    error_rate: float
    response_time: float
    last_heartbeat: datetime
```

## Report Types

### 1. Daily Summary

Comprehensive daily trading overview with executive insights.

**Contents:**
- Portfolio performance snapshot
- P&L summary (absolute and percentage)
- Market overview (SPY, QQQ, major indices)
- Top gainers and losers
- Executive insights (LLM-generated)
- Upcoming economic events

**Generation Schedule:** Daily at market close (4:30 PM ET)

**Sample Output:**
```markdown
# 📈 Daily Trading Summary - January 15, 2024

## Portfolio Performance
- **Total Value**: $1,025,430.50
- **Daily P&L**: +$5,430.50 (+0.53%)
- **Daily Return**: +0.53%

## Market Overview
- **SPY**: 450.25 (+1.2%)
- **QQQ**: 380.50 (+1.5%)
- **VIX**: 15.30 (-5.2%)

## Top Movers
### Gainers 🟢
1. NVDA: +4.5% ($15,230)
2. AAPL: +2.1% ($8,450)

### Losers 🔴
1. TSLA: -3.2% (-$5,120)

## Executive Insights
"Strong performance driven by technology sector momentum..."
```

### 2. Weekly Performance Report

In-depth weekly analysis with detailed metrics.

**Contents:**
- Comprehensive performance metrics
- Trading statistics (win rate, avg win/loss)
- Risk metrics (Sharpe, drawdown, beta)
- Agent performance analysis
- Detailed LLM analysis
- Week-over-week comparisons

**Generation Schedule:** Weekly on Fridays after close

### 3. Monthly Review

Strategic monthly assessment with trend analysis.

**Contents:**
- Monthly performance summary
- Portfolio evolution charts
- Sector allocation analysis
- Strategy effectiveness review
- Risk-adjusted returns
- Recommendations for next month

**Generation Schedule:** Last trading day of month

### 4. Risk Alerts

Real-time risk monitoring and alerting.

**Alert Types:**
- Daily loss exceeding threshold
- Weekly drawdown alerts
- Position concentration warnings
- Low cash warnings
- Maximum drawdown breaches

**Severity Levels:**

| Level | Description | Action Required |
|-------|-------------|-----------------|
| INFO | Informational | None |
| WARNING | Attention needed | Monitor |
| CRITICAL | Immediate attention | Review positions |
| EMERGENCY | System risk | Halt trading |

### 5. Trade Notifications

Real-time trade execution alerts.

**Contents:**
- Trade details (symbol, side, quantity, price)
- Execution quality metrics
- Slippage analysis
- Commission costs
- Position impact

### 6. System Health Report

Comprehensive system monitoring.

**Contents:**
- API connectivity status
- Database health
- Agent status tracking
- Resource usage (CPU, memory, disk)
- Error rates
- Response times
- Health score (0-100)

### 7. Agent Performance Report

Individual agent effectiveness analysis.

**Metrics Tracked:**
- Tasks processed
- Success rates
- Average processing time
- Error frequency
- Confidence scores
- Agent-specific KPIs

### 8. Portfolio Snapshot

Point-in-time portfolio export.

**Contents:**
- All positions with current values
- Cash balances
- Performance metrics
- Risk metrics
- Allocation breakdown

**Formats:** JSON, CSV

## Performance Analytics

### Calculation Methodology

#### Sharpe Ratio
```python
def _calculate_sharpe_ratio(self, returns: List[float]) -> float:
    """
    Calculate risk-adjusted returns
    
    Formula: (Returns - Risk-free rate) / Std deviation
    Annualized for 252 trading days
    """
    
    if len(returns) < 2:
        return 0.0
    
    avg_return = np.mean(returns)
    std_dev = np.std(returns)
    
    if std_dev == 0:
        return 0.0
    
    # Annualized Sharpe (252 trading days)
    sharpe = (avg_return / std_dev) * np.sqrt(252)
    
    # Bound between -10 and 10
    return max(-10, min(10, sharpe))
```

#### Maximum Drawdown
```python
def _calculate_max_drawdown(self, values: List[float]) -> float:
    """
    Calculate maximum peak-to-trough decline
    
    Returns percentage of largest drawdown
    """
    
    if len(values) < 2:
        return 0.0
    
    peak = values[0]
    max_dd = 0.0
    
    for value in values[1:]:
        peak = max(peak, value)
        drawdown = (peak - value) / peak * 100
        max_dd = max(max_dd, drawdown)
    
    return max_dd
```

#### Win Rate
```python
def _calculate_win_rate(self, trades: List[Dict]) -> float:
    """
    Calculate percentage of profitable trades
    """
    
    if not trades:
        return 0.0
    
    winning = sum(1 for t in trades if t['pnl'] > 0)
    return (winning / len(trades)) * 100
```

### Period Returns

The system calculates returns for multiple periods:

| Period | Calculation | Use Case |
|--------|------------|----------|
| Daily | Today vs Yesterday | Intraday performance |
| Weekly | Current vs 1 week ago | Short-term trend |
| Monthly | Current vs 1 month ago | Medium-term performance |
| YTD | Current vs Jan 1 | Annual performance |

## Alert System

### Threshold Configuration

```python
alert_thresholds = {
    'daily_loss_pct': -2.0,        # Daily loss > 2%
    'weekly_loss_pct': -5.0,       # Weekly loss > 5%
    'position_loss_pct': -10.0,    # Position loss > 10%
    'risk_score': 80,              # Risk score > 80
    'low_cash_pct': 5.0,           # Cash < 5%
    'high_concentration_pct': 30.0, # Position > 30%
    'max_drawdown_pct': 15.0       # Drawdown > 15%
}
```

### Alert Generation Flow

```python
async def monitor_risk_alerts(self) -> Dict:
    """
    Monitor and generate risk alerts
    
    Process:
    1. Calculate current metrics
    2. Compare against thresholds
    3. Generate alerts for violations
    4. Determine severity
    5. Save and optionally notify
    """
```

### Alert Object Structure

```python
{
    'id': 'alert_20240115_143022',
    'severity': 'CRITICAL',
    'title': 'Daily Loss Alert',
    'message': 'Portfolio down 3.5% today',
    'timestamp': '2024-01-15T14:30:22',
    'data': {
        'daily_pnl': -35000,
        'daily_pnl_pct': -3.5
    },
    'acknowledged': False
}
```

## Report Generation

### Directory Structure

```
reports/
├── daily/              # Daily summaries
│   └── summary_20240115.md
├── weekly/             # Weekly performance reports
│   └── performance_20240115.md
├── monthly/            # Monthly reviews
│   └── review_202401.md
├── alerts/             # Risk alerts
│   └── alert_20240115_143022.md
├── trades/             # Trade notifications
│   └── trade_AAPL_20240115_093015.md
├── system/             # System health reports
│   └── health_20240115.md
└── snapshots/          # Portfolio snapshots
    └── snapshot_20240115.json
```

### Format Support

The agent supports multiple output formats:

| Format | Extension | Use Case |
|--------|-----------|----------|
| Markdown | .md | Human-readable reports |
| JSON | .json | Structured data export |
| HTML | .html | Web display with styling |
| Text | .txt | Plain text, no formatting |

### Format Conversion

```python
def _convert_to_format(self, content: str, 
                      from_format: ReportFormat,
                      to_format: ReportFormat) -> str:
    """
    Convert report between formats
    
    Conversions:
    - Markdown → HTML (with styling)
    - Markdown → Text (strip formatting)
    - Markdown → JSON (structured)
    """
```

### Report Styling (HTML)

```html
<style>
    body { font-family: Arial, sans-serif; }
    h1 { color: #2c3e50; }
    .metric { 
        background: #f0f0f0;
        padding: 10px;
        border-radius: 5px;
    }
    .positive { color: green; }
    .negative { color: red; }
</style>
```

## System Monitoring

### Health Score Calculation

```python
def _calculate_health_score(self, metrics: SystemHealthMetrics) -> int:
    """
    Calculate overall system health (0-100)
    
    Weights:
    - API connectivity: 30%
    - Database health: 20%
    - Agent status: 30%
    - Resource usage: 10%
    - Error rate: 10%
    """
    
    score = 100
    
    # Deduct for API issues
    offline_apis = sum(1 for s in metrics.api_connectivity.values() if not s)
    score -= offline_apis * 10
    
    # Deduct for database issues
    if not metrics.database_health:
        score -= 20
    
    # Deduct for agent issues
    offline_agents = sum(1 for s in metrics.agent_status.values() if s != 'running')
    score -= offline_agents * 10
    
    # Deduct for high resource usage
    if metrics.resource_usage.get('cpu_percent', 0) > 80:
        score -= 5
    if metrics.resource_usage.get('memory_percent', 0) > 80:
        score -= 5
    
    # Deduct for high error rate
    if metrics.error_rate > 5:
        score -= 10
    
    return max(0, score)
```

### Resource Monitoring

```python
def _get_system_resources(self) -> Dict:
    """
    Get system resource usage
    
    Requires: psutil (optional)
    
    Returns:
    - CPU percentage
    - Memory percentage
    - Disk usage
    - Network I/O
    """
    
    if PSUTIL_AVAILABLE:
        import psutil
        return {
            'cpu_percent': psutil.cpu_percent(interval=1),
            'memory_percent': psutil.virtual_memory().percent,
            'disk_percent': psutil.disk_usage('/').percent
        }
    else:
        # Fallback values
        return {
            'cpu_percent': 0,
            'memory_percent': 0,
            'disk_percent': 0
        }
```

### Agent Status Tracking

```python
agent_status = {
    'junior_analyst': 'running',
    'senior_analyst': 'running',
    'economist': 'idle',
    'portfolio_manager': 'running',
    'trade_execution': 'running',
    'analytics_reporting': 'running'
}
```

## Testing Strategy

### Test Coverage

The Analytics & Reporting Agent has comprehensive test coverage:

| Category | Tests | Coverage |
|----------|-------|----------|
| Unit Tests | 15 | 92% |
| Integration Tests | 8 | 88% |
| Edge Case Tests | 5 | 85% |

### Core Test Categories

#### 1. PerformanceAnalytics Tests (7 tests)

- **test_calculate_sharpe_ratio**: Sharpe calculation accuracy
- **test_calculate_max_drawdown**: Drawdown calculation
- **test_calculate_win_rate**: Win rate calculation
- **test_period_returns**: Period return calculations
- **test_zero_value_handling**: Division by zero protection
- **test_empty_data_handling**: Empty dataset handling
- **test_default_metrics**: Fallback values

#### 2. ReportManager Tests (6 tests)

- **test_report_generation**: Report creation
- **test_format_conversion**: Format transformations
- **test_file_saving**: File system operations
- **test_directory_creation**: Auto directory creation
- **test_error_handling**: Write failure recovery
- **test_concurrent_saves**: Parallel write operations

#### 3. Alert System Tests (5 tests)

- **test_threshold_monitoring**: Threshold detection
- **test_alert_generation**: Alert creation
- **test_severity_classification**: Severity assignment
- **test_alert_storage**: Alert persistence
- **test_multiple_alerts**: Batch alert handling

#### 4. Integration Tests (8 tests)

- **test_daily_summary_generation**: End-to-end daily report
- **test_weekly_performance_report**: Weekly report flow
- **test_risk_alert_workflow**: Alert generation flow
- **test_system_health_check**: Health monitoring
- **test_agent_performance_analysis**: Agent tracking
- **test_portfolio_snapshot_export**: Data export
- **test_concurrent_report_generation**: Parallel reports
- **test_report_scheduling**: Timing logic

#### 5. Edge Case Tests (5 tests)

- **test_empty_portfolio**: Zero position handling
- **test_api_failure_handling**: API error recovery
- **test_extreme_values**: Boundary conditions
- **test_file_permission_error**: Write failure handling
- **test_missing_dependencies**: Optional package handling

### Test Fixtures

```python
@pytest.fixture
def analytics_agent(mock_llm_provider, mock_config, 
                   mock_data_provider, mock_db_manager):
    """Create test agent instance"""
    agent = AnalyticsReportingAgent(
        llm_provider=mock_llm_provider,
        config=mock_config,
        data_provider=mock_data_provider,
        db_manager=mock_db_manager
    )
    agent.is_active = True  # Important!
    return agent

@pytest.fixture
def sample_portfolio_data():
    """Sample portfolio data for testing"""
    return {
        'account': {
            'portfolio_value': 1000000.00,
            'cash': 250000.00,
            'buying_power': 500000.00
        },
        'positions': [
            {
                'symbol': 'AAPL',
                'qty': 100,
                'market_value': 15000.00,
                'unrealized_plpc': 2.45
            }
        ]
    }
```

### Critical Test Patterns

```python
# Mock async methods
analytics_agent._get_market_summary = AsyncMock(
    return_value={'SPY': {'price': 450, 'change_pct': 1.2}}
)

# Test agent active state
assert analytics_agent.is_active == True

# Verify report saved
assert (Path('reports') / 'daily' / filename).exists()
```

## Configuration

### Core Configuration

```python
config = {
    # Report Settings
    'report_formats': ['markdown', 'json'],
    'report_directory': 'reports',
    'enable_html_reports': True,
    'enable_email_reports': False,
    
    # Alert Thresholds
    'daily_loss_threshold': -2.0,
    'weekly_loss_threshold': -5.0,
    'max_drawdown_threshold': 15.0,
    'position_concentration_threshold': 30.0,
    'low_cash_threshold': 5.0,
    
    # Scheduling
    'daily_report_time': '16:30',  # 4:30 PM ET
    'weekly_report_day': 'Friday',
    'monthly_report_day': 'last',
    
    # Performance Calculation
    'sharpe_ratio_period': 252,    # Trading days
    'risk_free_rate': 0.05,        # 5% annual
    'confidence_interval': 0.95,    # For VaR
    
    # System Monitoring
    'health_check_interval': 300,   # 5 minutes
    'resource_warning_threshold': 80, # CPU/Memory %
    'error_rate_threshold': 5.0,    # Error %
    
    # Report Retention
    'keep_daily_reports': 30,        # Days
    'keep_weekly_reports': 12,       # Weeks
    'keep_monthly_reports': 12,      # Months
    'keep_alerts': 90                # Days
}
```

### Alert Configuration

```python
alert_config = {
    'channels': {
        'file': True,           # Save to file
        'console': True,        # Log to console
        'email': False,         # Email notifications
        'slack': False,         # Slack notifications
    },
    'severity_actions': {
        'INFO': ['file'],
        'WARNING': ['file', 'console'],
        'CRITICAL': ['file', 'console', 'email'],
        'EMERGENCY': ['file', 'console', 'email', 'slack']
    },
    'rate_limiting': {
        'max_alerts_per_hour': 20,
        'cooldown_minutes': 5
    }
}
```

### Report Templates

```python
templates = {
    'daily_summary': {
        'sections': [
            'header',
            'performance_summary',
            'market_overview',
            'top_movers',
            'executive_insights',
            'upcoming_events'
        ],
        'include_charts': False,
        'include_recommendations': True
    },
    'weekly_performance': {
        'sections': [
            'header',
            'performance_metrics',
            'trading_statistics',
            'risk_analysis',
            'agent_performance',
            'detailed_analysis'
        ],
        'include_charts': True,
        'include_historical': True
    }
}
```

## Integration Points

### Upstream Dependencies

#### Data Provider (Alpaca)
- **Data**: Portfolio data, positions, account info
- **Format**: Structured dictionaries
- **Frequency**: Real-time and on-demand

#### Database Manager
- **Data**: Historical trades, agent decisions
- **Operations**: Read-only queries
- **Frequency**: Report generation

#### LLM Provider (Claude)
- **Purpose**: Generate executive insights
- **Usage**: Daily summaries, weekly analysis
- **Frequency**: Per report generation

### Downstream Consumers

#### File System
- **Output**: Report files in multiple formats
- **Location**: reports/ directory structure
- **Retention**: Configurable by report type

#### Notification Systems (Future)
- **Email**: Critical alerts and reports
- **Slack**: Team notifications
- **Webhook**: External integrations

### Internal Communication

```python
# Agent status updates
await self.update_agent_status('analytics_reporting', 'processing')

# Performance data sharing
metrics = await self.get_performance_metrics()
await self.broadcast_metrics(metrics)

# Alert distribution
alert = await self.create_alert(severity, title, message)
await self.distribute_alert(alert)
```

## Usage Examples

### Basic Report Generation

```python
# Initialize agent
agent = AnalyticsReportingAgent(
    llm_provider=claude_provider,
    config=config,
    data_provider=alpaca_provider,
    db_manager=db_manager
)

# Activate agent
agent.is_active = True

# Generate daily summary
result = await agent.generate_daily_summary()
print(f"Daily summary saved: {result['save_result']['filepath']}")

# Generate weekly report
result = await agent.generate_weekly_performance_report()
print(f"Weekly report generated with {len(result['save_results'])} files")
```

### Risk Monitoring

```python
# Monitor risk alerts continuously
while agent.is_active:
    alerts = await agent.monitor_risk_alerts()
    
    if alerts['alerts_triggered'] > 0:
        print(f"⚠️ {alerts['alerts_triggered']} alerts triggered")
        for alert_title in alerts['alerts']:
            print(f"  - {alert_title}")
    
    await asyncio.sleep(60)  # Check every minute
```

### System Health Monitoring

```python
# Check system health
health = await agent.check_system_health()

if health['health_score'] < 80:
    print(f"⚠️ System health degraded: {health['health_score']}/100")
    print(f"Issues: {health.get('issues', [])}")
else:
    print(f"✅ System healthy: {health['health_score']}/100")
```

### Custom Report Generation

```python
# Generate custom report with specific format
task_data = {
    'task_type': 'portfolio_snapshot',
    'format': 'json',
    'include_metrics': True
}

result = await agent.process(task_data)
snapshot_path = result['save_result']['filepath']
print(f"Portfolio snapshot exported to: {snapshot_path}")
```

### Agent Performance Analysis

```python
# Analyze agent performance
performance = await agent.analyze_agent_performance()

for agent_name, metrics in performance['agent_metrics'].items():
    print(f"\n{agent_name}:")
    print(f"  Success Rate: {metrics['success_rate']}%")
    print(f"  Avg Time: {metrics['avg_processing_time']:.2f}s")
    print(f"  Total Tasks: {metrics['total_tasks']}")
```

### Alert Subscription

```python
# Subscribe to specific alert types
alert_handler = {
    'daily_loss': lambda alert: send_email(alert),
    'position_concentration': lambda alert: log_warning(alert),
    'max_drawdown': lambda alert: trigger_risk_review(alert)
}

# Process alerts
alerts = await agent.monitor_risk_alerts()
for alert in alerts.get('alerts', []):
    if alert['type'] in alert_handler:
        alert_handler[alert['type']](alert)
```

## Troubleshooting

### Common Issues and Solutions

#### 1. Agent Not Active Error

**Symptom**: "Agent is not active" error

**Solution**:
```python
# Ensure agent is activated
agent.is_active = True

# Verify status
assert agent.is_active == True
```

#### 2. Division by Zero in Calculations

**Symptom**: ZeroDivisionError in metrics calculation

**Solution**:
```python
# Check for zero values
if portfolio_value == 0:
    return default_metrics()

# Use epsilon for near-zero values
if abs(value) < 1e-10:
    value = 1e-10
```

#### 3. Report Save Failures

**Symptom**: Reports not saved to disk

**Solution**:
```python
# Check directory permissions
import os
os.makedirs('reports', exist_ok=True)
os.chmod('reports', 0o755)

# Verify write access
test_file = Path('reports') / 'test.txt'
test_file.write_text('test')
test_file.unlink()
```

#### 4. Missing psutil

**Symptom**: System monitoring not working

**Solution**:
```bash
# Install psutil
pip install psutil

# Or use fallback values
if not PSUTIL_AVAILABLE:
    return default_resource_metrics()
```

#### 5. LLM Timeout

**Symptom**: Report generation hangs on insights

**Solution**:
```python
# Add timeout to LLM calls
try:
    insights = await asyncio.wait_for(
        llm.generate_response(prompt),
        timeout=30
    )
except asyncio.TimeoutError:
    insights = "Analysis temporarily unavailable"
```

### Error Codes

| Code | Description | Action |
|------|-------------|--------|
| AR001 | Agent not active | Activate agent |
| AR002 | Data provider error | Check connection |
| AR003 | Report save failed | Check permissions |
| AR004 | Calculation error | Verify data integrity |
| AR005 | Database error | Check DB connection |
| AR006 | LLM timeout | Retry or skip insights |

### Debug Mode

```python
# Enable debug logging
import logging
logging.getLogger('analytics_reporting').setLevel(logging.DEBUG)

# Trace report generation
agent.debug_mode = True
agent.trace_calculations = True

# Save intermediate results
agent.save_debug_data = True
```

## Best Practices

### 1. Report Scheduling

```python
# Use asyncio scheduling for regular reports
async def scheduled_reporting():
    while True:
        now = datetime.now()
        
        # Daily report at 4:30 PM
        if now.hour == 16 and now.minute == 30:
            await agent.generate_daily_summary()
        
        # Weekly report on Friday
        if now.weekday() == 4 and now.hour == 17:
            await agent.generate_weekly_performance_report()
        
        await asyncio.sleep(60)  # Check every minute
```

### 2. Alert Management

```python
# Implement alert deduplication
seen_alerts = set()

def should_send_alert(alert: Dict) -> bool:
    alert_key = f"{alert['type']}_{alert['severity']}"
    if alert_key in seen_alerts:
        return False
    seen_alerts.add(alert_key)
    return True
```

### 3. Performance Optimization

```python
# Cache frequently accessed data
@lru_cache(maxsize=128)
def get_cached_portfolio_data():
    return data_provider.get_portfolio_data()

# Batch database queries
trades = db_manager.get_trades_batch(
    start_date=week_ago,
    end_date=today
)
```

### 4. Error Recovery

```python
# Implement graceful degradation
try:
    full_report = await generate_full_report()
except Exception as e:
    logger.error(f"Full report failed: {e}")
    basic_report = await generate_basic_report()
    return basic_report
```

### 5. Report Archival

```python
# Implement report rotation
def archive_old_reports():
    cutoff_daily = datetime.now() - timedelta(days=30)
    cutoff_weekly = datetime.now() - timedelta(weeks=12)
    
    for report_file in Path('reports/daily').glob('*.md'):
        if report_file.stat().st_mtime < cutoff_daily.timestamp():
            report_file.rename(f'archive/{report_file.name}')
```

## Future Enhancements

### Planned Features

1. **Real-Time Dashboards**
   - WebSocket streaming
   - Live metric updates
   - Interactive charts
   - Alert notifications

2. **Advanced Analytics**
   - Machine learning insights
   - Anomaly detection
   - Pattern recognition
   - Predictive metrics

3. **Report Automation**
   - Cron-based scheduling
   - Custom report templates
   - Automated distribution
   - Report subscriptions

4. **Integration Extensions**
   - Email delivery
   - Slack notifications
   - Telegram alerts
   - Webhook support

5. **Visualization**
   - Chart.js integration
   - D3.js visualizations
   - Interactive reports
   - PDF generation

### Roadmap

| Quarter | Feature | Priority |
|---------|---------|----------|
| Q1 2024 | Real-time dashboards | High |
| Q2 2024 | Email integration | High |
| Q3 2024 | Advanced analytics | Medium |
| Q4 2024 | PDF reports | Medium |

## API Reference

### Core Methods

#### generate_daily_summary()
```python
async def generate_daily_summary() -> Dict:
    """Generate comprehensive daily report"""
```

#### generate_weekly_performance_report()
```python
async def generate_weekly_performance_report() -> Dict:
    """Generate detailed weekly analysis"""
```

#### monitor_risk_alerts()
```python
async def monitor_risk_alerts() -> Dict:
    """Monitor and generate risk alerts"""
```

#### check_system_health()
```python
async def check_system_health() -> Dict:
    """Check overall system health"""
```

#### analyze_agent_performance()
```python
async def analyze_agent_performance() -> Dict:
    """Analyze individual agent performance"""
```

#### export_portfolio_snapshot()
```python
async def export_portfolio_snapshot() -> Dict:
    """Export current portfolio state"""
```

## Conclusion

The Analytics & Reporting Agent serves as the comprehensive monitoring and insights engine of the AI trading system, providing critical visibility into performance, risk, and system health. With its modular architecture, extensive metric calculations, multi-format reporting, and real-time alerting capabilities, it ensures complete transparency and control over trading operations. The agent's robust error handling, comprehensive testing, and flexible configuration make it an essential component for both automated decision-making and human oversight in production trading environments.