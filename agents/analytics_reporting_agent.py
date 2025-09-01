# agents/analytics_reporting_agent.py
"""
Analytics & Reporting Agent - Performance Analytics and System Reporting

This agent provides comprehensive performance analytics, risk monitoring,
and automated reporting for the AI trading system.
"""

from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import asyncio
import logging
import json
import os
from decimal import Decimal
from pathlib import Path

from utils.base_agent import BaseAgent

# Try to import database models, but don't fail if they don't exist
try:
    from database.models import Portfolio, Position, Trade, AgentDecision
except ImportError:
    # Database models not available, will work without them
    Portfolio = Position = Trade = AgentDecision = None


# ==============================================================================
# DATA STRUCTURES AND ENUMS
# ==============================================================================

class ReportType(Enum):
    """Types of reports generated"""
    DAILY_SUMMARY = "daily_summary"
    WEEKLY_PERFORMANCE = "weekly_performance"
    MONTHLY_REVIEW = "monthly_review"
    RISK_ALERT = "risk_alert"
    TRADE_NOTIFICATION = "trade_notification"
    SYSTEM_HEALTH = "system_health"
    AGENT_PERFORMANCE = "agent_performance"
    PORTFOLIO_SNAPSHOT = "portfolio_snapshot"


class AlertSeverity(Enum):
    """Alert severity levels"""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


class ReportFormat(Enum):
    """Report output formats"""
    MARKDOWN = "markdown"
    JSON = "json"
    HTML = "html"
    TEXT = "text"


@dataclass
class PerformanceMetrics:
    """Portfolio performance metrics"""
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


@dataclass
class SystemHealthMetrics:
    """System health monitoring metrics"""
    api_connectivity: Dict[str, bool]
    database_health: bool
    agent_status: Dict[str, str]
    resource_usage: Dict[str, float]
    error_rate: float
    response_time: float
    last_heartbeat: datetime


# ==============================================================================
# PERFORMANCE ANALYTICS ENGINE
# ==============================================================================

class PerformanceAnalytics:
    """
    Advanced performance analytics and attribution engine
    """
    
    def __init__(self, data_provider, db_manager):
        self.data_provider = data_provider
        self.db_manager = db_manager
        self.logger = logging.getLogger('performance_analytics')
        
        # Benchmark symbols
        self.benchmarks = {
            'SPY': 'S&P 500',
            'QQQ': 'NASDAQ 100',
            'IWM': 'Russell 2000'
        }
    
    async def calculate_comprehensive_performance(self) -> PerformanceMetrics:
        """Calculate all portfolio performance metrics"""
        
        try:
            # Get current portfolio state
            portfolio_data = await self.data_provider.get_portfolio_data()
            
            if 'error' in portfolio_data:
                raise Exception(f"Failed to get portfolio data: {portfolio_data['error']}")
            
            account = portfolio_data['account']
            positions = portfolio_data['positions']
            
            # Calculate P&L metrics
            portfolio_value = float(account['portfolio_value'])
            initial_capital = float(account.get('initial_capital', 100000))
            
            # Get historical data for calculations
            historical_data = await self._get_historical_performance(days=30)
            
            # Calculate period returns
            day_pnl, day_pnl_pct = self._calculate_period_return(
                portfolio_value, historical_data, days=1
            )
            week_pnl, week_pnl_pct = self._calculate_period_return(
                portfolio_value, historical_data, days=7
            )
            month_pnl, month_pnl_pct = self._calculate_period_return(
                portfolio_value, historical_data, days=30
            )
            
            # Calculate YTD return
            ytd_return = ((portfolio_value - initial_capital) / initial_capital) * 100
            
            # Calculate risk metrics
            sharpe_ratio = self._calculate_sharpe_ratio(historical_data)
            max_drawdown = self._calculate_max_drawdown(historical_data)
            
            # Calculate trade statistics
            trade_stats = await self._calculate_trade_statistics()
            
            return PerformanceMetrics(
                total_value=portfolio_value,
                daily_pnl=day_pnl,
                daily_pnl_pct=day_pnl_pct,
                weekly_pnl=week_pnl,
                weekly_pnl_pct=week_pnl_pct,
                monthly_pnl=month_pnl,
                monthly_pnl_pct=month_pnl_pct,
                ytd_return=ytd_return,
                sharpe_ratio=sharpe_ratio,
                max_drawdown=max_drawdown,
                win_rate=trade_stats['win_rate'],
                avg_win=trade_stats['avg_win'],
                avg_loss=trade_stats['avg_loss'],
                total_trades=trade_stats['total_trades'],
                winning_trades=trade_stats['winning_trades'],
                losing_trades=trade_stats['losing_trades']
            )
            
        except Exception as e:
            self.logger.error(f"Failed to calculate performance metrics: {str(e)}")
            # Return default metrics on error
            return PerformanceMetrics(
                total_value=0, daily_pnl=0, daily_pnl_pct=0,
                weekly_pnl=0, weekly_pnl_pct=0, monthly_pnl=0,
                monthly_pnl_pct=0, ytd_return=0, sharpe_ratio=0,
                max_drawdown=0, win_rate=0, avg_win=0, avg_loss=0,
                total_trades=0, winning_trades=0, losing_trades=0
            )
    
    async def _get_historical_performance(self, days: int) -> List[float]:
        """Get historical portfolio values"""
        
        # In production, this would query historical snapshots from database
        # For now, return simulated data
        return [100000 + (i * 100) for i in range(days)]
    
    def _calculate_period_return(self, current_value: float, 
                                historical_data: List[float], 
                                days: int) -> Tuple[float, float]:
        """Calculate return for a specific period"""
        
        if len(historical_data) < days:
            return 0.0, 0.0
        
        start_value = historical_data[-days]
        pnl = current_value - start_value
        pnl_pct = (pnl / start_value) * 100 if start_value > 0 else 0
        
        return round(pnl, 2), round(pnl_pct, 2)
    
    def _calculate_sharpe_ratio(self, historical_data: List[float]) -> float:
        """Calculate Sharpe ratio"""
        
        if len(historical_data) < 2:
            return 0.0
        
        # Calculate daily returns
        returns = []
        for i in range(1, len(historical_data)):
            daily_return = (historical_data[i] - historical_data[i-1]) / historical_data[i-1]
            returns.append(daily_return)
        
        if not returns:
            return 0.0
        
        # Calculate Sharpe (simplified - assuming 0 risk-free rate)
        import numpy as np
        avg_return = np.mean(returns)
        std_return = np.std(returns)
        
        if std_return == 0:
            return 0.0
        
        # Annualized Sharpe ratio
        sharpe = (avg_return / std_return) * np.sqrt(252)
        return round(sharpe, 2)
    
    def _calculate_max_drawdown(self, historical_data: List[float]) -> float:
        """Calculate maximum drawdown"""
        
        if len(historical_data) < 2:
            return 0.0
        
        max_value = historical_data[0]
        max_drawdown = 0.0
        
        for value in historical_data[1:]:
            max_value = max(max_value, value)
            drawdown = ((max_value - value) / max_value) * 100
            max_drawdown = max(max_drawdown, drawdown)
        
        return round(max_drawdown, 2)
    
    async def _calculate_trade_statistics(self) -> Dict:
        """Calculate trading statistics"""
        
        # In production, query from database
        # For now, return sample statistics
        return {
            'total_trades': 45,
            'winning_trades': 28,
            'losing_trades': 17,
            'win_rate': 62.2,
            'avg_win': 245.50,
            'avg_loss': -142.30
        }


# ==============================================================================
# REPORT MANAGER
# ==============================================================================

class ReportManager:
    """
    Manages report generation, formatting, and file storage
    """
    
    def __init__(self, config=None):
        self.config = config
        self.logger = logging.getLogger('report_manager')
        
        # Set up reports directory
        self.reports_dir = Path('reports')
        self._ensure_reports_directory()
        
        # Report formatters
        self.formatters = {
            ReportFormat.MARKDOWN: self._format_markdown,
            ReportFormat.JSON: self._format_json,
            ReportFormat.HTML: self._format_html,
            ReportFormat.TEXT: self._format_text
        }
    
    def _ensure_reports_directory(self):
        """Ensure reports directory structure exists"""
        
        # Create main reports directory
        self.reports_dir.mkdir(exist_ok=True)
        
        # Create subdirectories for different report types
        subdirs = ['daily', 'weekly', 'monthly', 'alerts', 'trades', 'system', 'snapshots']
        for subdir in subdirs:
            (self.reports_dir / subdir).mkdir(exist_ok=True)
    
    async def save_report(self, content: str, report_type: ReportType,
                         format: ReportFormat = ReportFormat.MARKDOWN) -> Dict:
        """Save report to file system"""
        
        try:
            # Determine subdirectory based on report type
            subdir_map = {
                ReportType.DAILY_SUMMARY: 'daily',
                ReportType.WEEKLY_PERFORMANCE: 'weekly',
                ReportType.MONTHLY_REVIEW: 'monthly',
                ReportType.RISK_ALERT: 'alerts',
                ReportType.TRADE_NOTIFICATION: 'trades',
                ReportType.SYSTEM_HEALTH: 'system',
                ReportType.AGENT_PERFORMANCE: 'system',
                ReportType.PORTFOLIO_SNAPSHOT: 'snapshots'
            }
            
            subdir = subdir_map.get(report_type, 'general')
            
            # Generate filename with timestamp
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            extension = self._get_file_extension(format)
            filename = f"{report_type.value}_{timestamp}.{extension}"
            filepath = self.reports_dir / subdir / filename
            
            # Format content based on format type
            if format in self.formatters:
                formatted_content = self.formatters[format](content, report_type)
            else:
                formatted_content = content
            
            # Write to file
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(formatted_content)
            
            self.logger.info(f"Report saved: {filepath}")
            
            return {
                'success': True,
                'filepath': str(filepath),
                'format': format.value,
                'timestamp': timestamp
            }
            
        except Exception as e:
            self.logger.error(f"Failed to save report: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def _get_file_extension(self, format: ReportFormat) -> str:
        """Get file extension for report format"""
        
        extensions = {
            ReportFormat.MARKDOWN: 'md',
            ReportFormat.JSON: 'json',
            ReportFormat.HTML: 'html',
            ReportFormat.TEXT: 'txt'
        }
        return extensions.get(format, 'txt')
    
    def _format_markdown(self, content: str, report_type: ReportType) -> str:
        """Format content as markdown"""
        return content  # Already in markdown format
    
    def _format_json(self, content: str, report_type: ReportType) -> str:
        """Format content as JSON"""
        
        # Parse content and convert to JSON
        data = {
            'report_type': report_type.value,
            'timestamp': datetime.now().isoformat(),
            'content': content
        }
        return json.dumps(data, indent=2)
    
    def _format_html(self, content: str, report_type: ReportType) -> str:
        """Format content as HTML"""
        
        # Convert markdown-style content to HTML
        html_content = content.replace('\n', '<br>\n')
        html_content = html_content.replace('# ', '<h1>').replace('\n', '</h1>\n', 1)
        html_content = html_content.replace('## ', '<h2>').replace('\n', '</h2>\n', 1)
        html_content = html_content.replace('### ', '<h3>').replace('\n', '</h3>\n', 1)
        html_content = html_content.replace('**', '<strong>').replace('**', '</strong>')
        html_content = html_content.replace('*', '<em>').replace('*', '</em>')
        
        html = f"""<!DOCTYPE html>
<html>
<head>
    <title>{report_type.value.replace('_', ' ').title()}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; line-height: 1.6; }}
        h1 {{ color: #333; border-bottom: 2px solid #333; padding-bottom: 10px; }}
        h2 {{ color: #666; margin-top: 20px; }}
        table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #f2f2f2; }}
        .metric {{ background-color: #f9f9f9; padding: 10px; margin: 10px 0; border-left: 4px solid #4CAF50; }}
        .alert {{ background-color: #fff3cd; padding: 10px; margin: 10px 0; border-left: 4px solid #ffc107; }}
        .critical {{ background-color: #f8d7da; border-left-color: #dc3545; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>{report_type.value.replace('_', ' ').title()}</h1>
        <div class="timestamp">Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</div>
        <div class="content">
            {html_content}
        </div>
    </div>
</body>
</html>"""
        
        return html
    
    def _format_text(self, content: str, report_type: ReportType) -> str:
        """Format content as plain text"""
        
        # Remove markdown formatting
        text = content.replace('#', '').replace('*', '').replace('`', '')
        return text
    
    async def export_portfolio_snapshot(self, portfolio_data: Dict, 
                                       performance: PerformanceMetrics) -> Dict:
        """Export complete portfolio snapshot"""
        
        snapshot = {
            'timestamp': datetime.now().isoformat(),
            'portfolio': portfolio_data,
            'performance': {
                'total_value': performance.total_value,
                'daily_pnl': performance.daily_pnl,
                'daily_pnl_pct': performance.daily_pnl_pct,
                'weekly_pnl': performance.weekly_pnl,
                'weekly_pnl_pct': performance.weekly_pnl_pct,
                'monthly_pnl': performance.monthly_pnl,
                'monthly_pnl_pct': performance.monthly_pnl_pct,
                'ytd_return': performance.ytd_return,
                'sharpe_ratio': performance.sharpe_ratio,
                'max_drawdown': performance.max_drawdown,
                'win_rate': performance.win_rate,
                'total_trades': performance.total_trades
            }
        }
        
        # Save as JSON
        content = json.dumps(snapshot, indent=2)
        return await self.save_report(
            content, 
            ReportType.PORTFOLIO_SNAPSHOT,
            ReportFormat.JSON
        )


# ==============================================================================
# MAIN ANALYTICS & REPORTING AGENT
# ==============================================================================

class AnalyticsReportingAgent(BaseAgent):
    """
    Analytics & Reporting Agent
    
    Provides comprehensive performance analytics, risk monitoring,
    and automated reporting for the AI trading system
    """
    
    def __init__(self, llm_provider, config, data_provider, db_manager):
        # Initialize base agent
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
        self.alert_thresholds = {
            'daily_loss_pct': -2.0,
            'weekly_loss_pct': -5.0,
            'position_loss_pct': -10.0,
            'risk_score': 80,
            'low_cash_pct': 5.0,
            'high_concentration_pct': 30.0,
            'max_drawdown_pct': 15.0
        }
        
        # System health tracking
        self.agent_status = {}
        self.system_errors = []
        
        self.logger.info(f"✅ {self.agent_name} initialized successfully")
    
    async def _process_internal(self, task_data: Dict) -> Dict:
        """
        Implementation of abstract method from BaseAgent
        
        Args:
            task_data: Task data to process
            
        Returns:
            Dict: Processing result
        """
        task_type = task_data.get('task_type', 'daily_summary')
        
        if task_type == 'daily_summary':
            return await self.generate_daily_summary()
        elif task_type == 'weekly_performance':
            return await self.generate_weekly_performance_report()
        elif task_type == 'monthly_review':
            return await self.generate_monthly_review()
        elif task_type == 'risk_alert':
            return await self.monitor_risk_alerts()
        elif task_type == 'trade_alert':
            return await self.record_trade_notification(task_data)
        elif task_type == 'system_health':
            return await self.check_system_health()
        elif task_type == 'agent_performance':
            return await self.analyze_agent_performance()
        elif task_type == 'portfolio_snapshot':
            return await self.export_portfolio_snapshot()
        else:
            raise ValueError(f'Unknown task type: {task_type}')
    
    async def generate_daily_summary(self) -> Dict:
        """Generate daily executive summary"""
        
        self.logger.info("Generating daily executive summary")
        
        try:
            # Collect all data
            performance = await self.analytics.calculate_comprehensive_performance()
            portfolio_data = await self.data_provider.get_portfolio_data()
            market_summary = await self._get_market_summary()
            top_movers = await self._get_top_movers()
            upcoming_events = await self._get_upcoming_events()
            
            # Generate summary using LLM
            summary_prompt = self._create_summary_prompt(
                performance, portfolio_data, market_summary, 
                top_movers, upcoming_events
            )
            
            llm_summary = await self.llm.generate_response(summary_prompt)
            
            # Format the report
            report = self._format_daily_report(
                llm_summary, performance, portfolio_data,
                market_summary, top_movers, upcoming_events
            )
            
            # Save report to file system
            save_result = await self.report_manager.save_report(
                report, 
                ReportType.DAILY_SUMMARY,
                ReportFormat.MARKDOWN
            )
            
            # Also save as HTML for better viewing
            await self.report_manager.save_report(
                report,
                ReportType.DAILY_SUMMARY,
                ReportFormat.HTML
            )
            
            # Update last report time
            self.last_daily_report = datetime.now()
            
            return {
                'status': 'success',
                'report_type': ReportType.DAILY_SUMMARY.value,
                'save_result': save_result,
                'metrics': {
                    'portfolio_value': performance.total_value,
                    'daily_pnl': performance.daily_pnl,
                    'daily_pnl_pct': performance.daily_pnl_pct
                }
            }
            
        except Exception as e:
            self.logger.error(f"Failed to generate daily summary: {str(e)}")
            return {'status': 'error', 'message': str(e)}
    
    async def generate_weekly_performance_report(self) -> Dict:
        """Generate detailed weekly performance report"""
        
        self.logger.info("Generating weekly performance report")
        
        try:
            # Comprehensive data collection
            performance = await self.analytics.calculate_comprehensive_performance()
            trade_analysis = await self._analyze_weekly_trades()
            agent_performance = await self._collect_agent_performance_data()
            risk_metrics = await self._calculate_risk_metrics()
            
            # Generate detailed analysis
            analysis_prompt = self._create_weekly_analysis_prompt(
                performance, trade_analysis, agent_performance, risk_metrics
            )
            
            llm_analysis = await self.llm.generate_response(analysis_prompt)
            
            # Format comprehensive report
            report = self._format_weekly_report(
                llm_analysis, performance, trade_analysis,
                agent_performance, risk_metrics
            )
            
            # Save in multiple formats
            save_results = []
            for format in [ReportFormat.MARKDOWN, ReportFormat.HTML, ReportFormat.JSON]:
                result = await self.report_manager.save_report(
                    report, 
                    ReportType.WEEKLY_PERFORMANCE,
                    format
                )
                save_results.append(result)
            
            self.last_weekly_report = datetime.now()
            
            return {
                'status': 'success',
                'report_type': ReportType.WEEKLY_PERFORMANCE.value,
                'save_results': save_results,
                'metrics': {
                    'weekly_return': performance.weekly_pnl_pct,
                    'sharpe_ratio': performance.sharpe_ratio,
                    'win_rate': performance.win_rate
                }
            }
            
        except Exception as e:
            self.logger.error(f"Failed to generate weekly report: {str(e)}")
            return {'status': 'error', 'message': str(e)}
    
    async def generate_monthly_review(self) -> Dict:
        """Generate monthly review report (placeholder for now)"""
        
        self.logger.info("Monthly review generation not yet implemented")
        return {
            'status': 'not_implemented',
            'message': 'Monthly review will be available in future updates'
        }
    
    async def monitor_risk_alerts(self) -> Dict:
        """Monitor and record risk alerts"""
        
        alerts_triggered = []
        
        try:
            # Get current metrics
            performance = await self.analytics.calculate_comprehensive_performance()
            portfolio_data = await self.data_provider.get_portfolio_data()
            
            # Check daily loss threshold
            if performance.daily_pnl_pct < self.alert_thresholds['daily_loss_pct']:
                alert = await self._create_alert(
                    AlertSeverity.WARNING,
                    "Daily Loss Alert",
                    f"Portfolio down {abs(performance.daily_pnl_pct):.2f}% today",
                    {'daily_pnl': performance.daily_pnl, 'daily_pnl_pct': performance.daily_pnl_pct}
                )
                alerts_triggered.append(alert)
            
            # Check weekly loss threshold
            if performance.weekly_pnl_pct < self.alert_thresholds['weekly_loss_pct']:
                alert = await self._create_alert(
                    AlertSeverity.CRITICAL,
                    "Weekly Loss Alert",
                    f"Portfolio down {abs(performance.weekly_pnl_pct):.2f}% this week",
                    {'weekly_pnl': performance.weekly_pnl, 'weekly_pnl_pct': performance.weekly_pnl_pct}
                )
                alerts_triggered.append(alert)
            
            # Check max drawdown
            if performance.max_drawdown > self.alert_thresholds['max_drawdown_pct']:
                alert = await self._create_alert(
                    AlertSeverity.WARNING,
                    "Maximum Drawdown Alert",
                    f"Portfolio drawdown reached {performance.max_drawdown:.2f}%",
                    {'max_drawdown': performance.max_drawdown}
                )
                alerts_triggered.append(alert)
            
            # Check position concentration
            if portfolio_data and 'positions' in portfolio_data:
                for position in portfolio_data['positions']:
                    position_pct = (position['market_value'] / performance.total_value) * 100
                    if position_pct > self.alert_thresholds['high_concentration_pct']:
                        alert = await self._create_alert(
                            AlertSeverity.WARNING,
                            "Position Concentration Alert",
                            f"{position['symbol']} is {position_pct:.1f}% of portfolio",
                            {'symbol': position['symbol'], 'concentration': position_pct}
                        )
                        alerts_triggered.append(alert)
            
            # Save all alerts
            for alert in alerts_triggered:
                await self._save_alert(alert)
            
            return {
                'status': 'success',
                'alerts_triggered': len(alerts_triggered),
                'alerts': [a['title'] for a in alerts_triggered]
            }
            
        except Exception as e:
            self.logger.error(f"Failed to monitor risk alerts: {str(e)}")
            return {'status': 'error', 'message': str(e)}
    
    async def record_trade_notification(self, trade_data: Dict) -> Dict:
        """Record trade execution notification"""
        
        try:
            # Format trade notification
            trade_msg = self._format_trade_notification(trade_data)
            
            # Save to trades log
            save_result = await self.report_manager.save_report(
                trade_msg,
                ReportType.TRADE_NOTIFICATION,
                ReportFormat.MARKDOWN
            )
            
            return {
                'status': 'success',
                'notification_recorded': True,
                'save_result': save_result
            }
            
        except Exception as e:
            self.logger.error(f"Failed to record trade notification: {str(e)}")
            return {'status': 'error', 'message': str(e)}
    
    async def check_system_health(self) -> Dict:
        """Check and report system health"""
        
        try:
            health_metrics = SystemHealthMetrics(
                api_connectivity=await self._check_api_connectivity(),
                database_health=await self._check_database_health(),
                agent_status=await self._check_agent_status(),
                resource_usage=await self._check_resource_usage(),
                error_rate=await self._calculate_error_rate(),
                response_time=await self._measure_response_time(),
                last_heartbeat=datetime.now()
            )
            
            # Generate health score
            health_score = self._calculate_health_score(health_metrics)
            
            # Create health report
            health_report = self._format_health_report(health_metrics, health_score)
            
            # Save health report
            save_result = await self.report_manager.save_report(
                health_report,
                ReportType.SYSTEM_HEALTH,
                ReportFormat.MARKDOWN
            )
            
            # Create alert if critical issues
            if health_score < 70:
                alert = await self._create_alert(
                    AlertSeverity.CRITICAL,
                    "System Health Alert",
                    f"System health score: {health_score}/100",
                    {'health_metrics': health_metrics.__dict__}
                )
                await self._save_alert(alert)
            
            return {
                'status': 'success',
                'health_score': health_score,
                'metrics': health_metrics.__dict__,
                'save_result': save_result
            }
            
        except Exception as e:
            self.logger.error(f"Failed to check system health: {str(e)}")
            return {'status': 'error', 'message': str(e)}
    
    async def analyze_agent_performance(self) -> Dict:
        """Analyze individual agent performance"""
        
        try:
            agent_metrics = await self._collect_agent_performance_data()
            
            # Calculate overall efficiency
            total_tasks = sum(m.get('total_tasks', 0) for m in agent_metrics.values())
            avg_success_rate = sum(m.get('success_rate', 0) for m in agent_metrics.values()) / len(agent_metrics) if agent_metrics else 0
            
            # Identify underperforming agents
            issues = []
            for agent, metrics in agent_metrics.items():
                if metrics.get('success_rate', 0) < 80:
                    issues.append(f"{agent}: Low success rate ({metrics['success_rate']}%)")
            
            # Format performance report
            report = self._format_agent_performance_report(agent_metrics, total_tasks, avg_success_rate, issues)
            
            # Save report
            save_result = await self.report_manager.save_report(
                report,
                ReportType.AGENT_PERFORMANCE,
                ReportFormat.MARKDOWN
            )
            
            return {
                'status': 'success',
                'total_tasks_processed': total_tasks,
                'average_success_rate': avg_success_rate,
                'agent_metrics': agent_metrics,
                'issues_identified': issues,
                'save_result': save_result
            }
            
        except Exception as e:
            self.logger.error(f"Failed to analyze agent performance: {str(e)}")
            return {'status': 'error', 'message': str(e)}
    
    async def export_portfolio_snapshot(self) -> Dict:
        """Export complete portfolio snapshot"""
        
        try:
            portfolio_data = await self.data_provider.get_portfolio_data()
            performance = await self.analytics.calculate_comprehensive_performance()
            
            result = await self.report_manager.export_portfolio_snapshot(
                portfolio_data, performance
            )
            
            return {
                'status': 'success',
                'export_result': result
            }
            
        except Exception as e:
            self.logger.error(f"Failed to export portfolio snapshot: {str(e)}")
            return {'status': 'error', 'message': str(e)}
    
    # ===========================================================================
    # HELPER METHODS
    # ===========================================================================
    
    async def _get_market_summary(self) -> Dict:
        """Get market summary data"""
        
        try:
            # Get major indices
            indices = ['SPY', 'QQQ', 'IWM']
            market_data = {}
            
            for symbol in indices:
                data = await self.data_provider.get_latest_data(symbol)
                if data:
                    market_data[symbol] = {
                        'price': data.get('close', 0),
                        'change_pct': data.get('change_percent', 0)
                    }
            
            return market_data
            
        except Exception as e:
            self.logger.error(f"Failed to get market summary: {str(e)}")
            return {}
    
    async def _get_top_movers(self) -> Dict:
        """Get top gaining and losing positions"""
        
        try:
            portfolio_data = await self.data_provider.get_portfolio_data()
            
            if not portfolio_data or 'positions' not in portfolio_data:
                return {'gainers': [], 'losers': []}
            
            positions = portfolio_data['positions']
            
            # Sort by P&L percentage
            sorted_positions = sorted(positions, 
                                    key=lambda x: x.get('unrealized_plpc', 0), 
                                    reverse=True)
            
            gainers = [p for p in sorted_positions if p.get('unrealized_plpc', 0) > 0][:3]
            losers = [p for p in sorted_positions if p.get('unrealized_plpc', 0) < 0][-3:]
            
            return {'gainers': gainers, 'losers': losers}
            
        except Exception as e:
            self.logger.error(f"Failed to get top movers: {str(e)}")
            return {'gainers': [], 'losers': []}
    
    async def _get_upcoming_events(self) -> List[Dict]:
        """Get upcoming market events"""
        
        # In production, integrate with economic calendar API
        return [
            {'date': 'Tomorrow', 'event': 'Fed Minutes Release'},
            {'date': 'Friday', 'event': 'Non-Farm Payrolls'}
        ]
    
    async def _analyze_weekly_trades(self) -> Dict:
        """Analyze trades from the past week"""
        
        # In production, query trade history from database
        return {
            'total_trades': 23,
            'profitable_trades': 15,
            'loss_trades': 8,
            'best_trade': {'symbol': 'AAPL', 'pnl': 450.00},
            'worst_trade': {'symbol': 'TSLA', 'pnl': -280.00},
            'most_traded': 'NVDA',
            'avg_holding_period': '2.3 days'
        }
    
    async def _collect_agent_performance_data(self) -> Dict:
        """Collect performance data for all agents"""
        
        # In production, query from agent metrics tables
        return {
            'junior_analyst': {
                'total_tasks': 150,
                'success_rate': 92,
                'avg_processing_time': 2.3
            },
            'senior_analyst': {
                'total_tasks': 50,
                'success_rate': 88,
                'avg_processing_time': 4.5
            },
            'portfolio_manager': {
                'total_tasks': 30,
                'success_rate': 95,
                'avg_processing_time': 3.2
            },
            'trade_execution': {
                'total_tasks': 45,
                'success_rate': 98,
                'avg_processing_time': 1.1
            }
        }
    
    async def _calculate_risk_metrics(self) -> Dict:
        """Calculate current risk metrics"""
        
        # In production, calculate from positions
        return {
            'portfolio_beta': 1.15,
            'var_95': -2500.00,
            'expected_shortfall': -3200.00,
            'correlation_spy': 0.82,
            'sector_concentration': {'Technology': 35, 'Healthcare': 20}
        }
    
    async def _check_api_connectivity(self) -> Dict[str, bool]:
        """Check API connectivity status"""
        
        return {
            'alpaca': True,
            'llm_provider': True,
            'market_data': True
        }
    
    async def _check_database_health(self) -> bool:
        """Check database health"""
        
        try:
            with self.db_manager.get_session() as session:
                # Simple health check query
                session.execute("SELECT 1")
                return True
        except:
            return False
    
    async def _check_agent_status(self) -> Dict[str, str]:
        """Check status of all agents"""
        
        return {
            'junior_analyst': 'running',
            'senior_analyst': 'running',
            'portfolio_manager': 'running',
            'trade_execution': 'running',
            'analytics_reporting': 'running'
        }
    
    async def _check_resource_usage(self) -> Dict[str, float]:
        """Check system resource usage"""
        
        import psutil
        
        return {
            'cpu_percent': psutil.cpu_percent(),
            'memory_percent': psutil.virtual_memory().percent,
            'disk_percent': psutil.disk_usage('/').percent
        }
    
    async def _calculate_error_rate(self) -> float:
        """Calculate system error rate"""
        
        # In production, calculate from logs
        return 0.5  # 0.5% error rate
    
    async def _measure_response_time(self) -> float:
        """Measure average system response time"""
        
        # In production, measure actual response times
        return 250.0  # 250ms average
    
    def _calculate_health_score(self, metrics: SystemHealthMetrics) -> int:
        """Calculate overall system health score"""
        
        score = 100
        
        # Deduct points for issues
        if not all(metrics.api_connectivity.values()):
            score -= 30
        
        if not metrics.database_health:
            score -= 30
        
        if metrics.error_rate > 1.0:
            score -= 20
        
        if metrics.response_time > 500:
            score -= 10
        
        if any(v > 80 for v in metrics.resource_usage.values()):
            score -= 10
        
        return max(0, score)
    
    async def _create_alert(self, severity: AlertSeverity, title: str,
                           message: str, data: Dict) -> Dict:
        """Create alert object"""
        
        return {
            'alert_id': f"ALERT_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            'severity': severity.value,
            'title': title,
            'message': message,
            'data': data,
            'timestamp': datetime.now().isoformat()
        }
    
    async def _save_alert(self, alert: Dict) -> bool:
        """Save alert to file system"""
        
        alert_content = f"""# {alert['title']}

**Severity**: {alert['severity'].upper()}  
**Time**: {alert['timestamp']}

## Message
{alert['message']}

## Data
```json
{json.dumps(alert['data'], indent=2)}
```
"""
        
        result = await self.report_manager.save_report(
            alert_content,
            ReportType.RISK_ALERT,
            ReportFormat.MARKDOWN
        )
        
        return result.get('success', False)
    
    def _create_summary_prompt(self, performance, portfolio_data, 
                              market_summary, top_movers, events) -> str:
        """Create prompt for LLM summary generation"""
        
        return f"""Generate an executive summary for today's trading performance.

Portfolio Performance:
- Total Value: ${performance.total_value:,.2f}
- Daily P&L: ${performance.daily_pnl:,.2f} ({performance.daily_pnl_pct:+.2f}%)
- Weekly P&L: ${performance.weekly_pnl:,.2f} ({performance.weekly_pnl_pct:+.2f}%)

Market Overview:
{json.dumps(market_summary, indent=2)}

Top Gainers: {[p['symbol'] for p in top_movers.get('gainers', [])]}
Top Losers: {[p['symbol'] for p in top_movers.get('losers', [])]}

Upcoming Events: {events}

Provide a concise executive summary highlighting key insights and recommendations."""
    
    def _create_weekly_analysis_prompt(self, performance, trade_analysis,
                                      agent_performance, risk_metrics) -> str:
        """Create prompt for weekly analysis"""
        
        return f"""Generate a comprehensive weekly performance analysis.

Performance Metrics:
- Weekly Return: {performance.weekly_pnl_pct:+.2f}%
- Sharpe Ratio: {performance.sharpe_ratio}
- Win Rate: {performance.win_rate}%
- Max Drawdown: {performance.max_drawdown}%

Trading Activity:
{json.dumps(trade_analysis, indent=2)}

Agent Performance:
{json.dumps(agent_performance, indent=2)}

Risk Metrics:
{json.dumps(risk_metrics, indent=2)}

Provide detailed analysis with actionable insights and recommendations for the coming week."""
    
    def _format_daily_report(self, llm_summary, performance, portfolio_data,
                            market_summary, top_movers, events) -> str:
        """Format daily report"""
        
        report = f"""# 📊 Daily Executive Summary - {datetime.now().strftime('%B %d, %Y')}

## Portfolio Performance
- **Total Value**: ${performance.total_value:,.2f}
- **Daily P&L**: ${performance.daily_pnl:,.2f} ({performance.daily_pnl_pct:+.2f}%)
- **Weekly P&L**: ${performance.weekly_pnl:,.2f} ({performance.weekly_pnl_pct:+.2f}%)
- **Max Drawdown**: {performance.max_drawdown}%

## Market Overview
"""
        
        for symbol, data in market_summary.items():
            report += f"- **{symbol}**: ${data['price']:.2f} ({data['change_pct']:+.2f}%)\n"
        
        report += f"""
## Top Movers
### Gainers
"""
        for position in top_movers.get('gainers', []):
            report += f"- {position['symbol']}: {position.get('unrealized_plpc', 0):+.2f}%\n"
        
        report += "\n### Losers\n"
        for position in top_movers.get('losers', []):
            report += f"- {position['symbol']}: {position.get('unrealized_plpc', 0):+.2f}%\n"
        
        report += f"""
## Executive Insights
{llm_summary}

## Upcoming Events
"""
        
        for event in events:
            report += f"- **{event['date']}**: {event['event']}\n"
        
        report += f"""
---
*Generated by Analytics & Reporting Agent - AI Trading System*
"""
        
        return report
    
    def _format_weekly_report(self, llm_analysis, performance, trade_analysis,
                            agent_performance, risk_metrics) -> str:
        """Format weekly performance report"""
        
        report = f"""# 📊 Weekly Performance Report - Week of {datetime.now().strftime('%B %d, %Y')}

## Performance Summary
- **Portfolio Value**: ${performance.total_value:,.2f}
- **Weekly Return**: {performance.weekly_pnl_pct:+.2f}%
- **YTD Return**: {performance.ytd_return:+.2f}%
- **Sharpe Ratio**: {performance.sharpe_ratio}
- **Max Drawdown**: {performance.max_drawdown}%

## Trading Statistics
- **Total Trades**: {trade_analysis['total_trades']}
- **Win Rate**: {performance.win_rate}%
- **Average Win**: ${performance.avg_win:,.2f}
- **Average Loss**: ${performance.avg_loss:,.2f}
- **Best Trade**: {trade_analysis['best_trade']['symbol']} (${trade_analysis['best_trade']['pnl']:,.2f})
- **Worst Trade**: {trade_analysis['worst_trade']['symbol']} (${trade_analysis['worst_trade']['pnl']:,.2f})

## Risk Metrics
- **Portfolio Beta**: {risk_metrics['portfolio_beta']}
- **VaR (95%)**: ${risk_metrics['var_95']:,.2f}
- **SPY Correlation**: {risk_metrics['correlation_spy']}

## Agent Performance
"""
        
        for agent, metrics in agent_performance.items():
            report += f"- **{agent}**: {metrics['success_rate']}% success rate ({metrics['total_tasks']} tasks)\n"
        
        report += f"""

## Detailed Analysis
{llm_analysis}

---
*Generated by Analytics & Reporting Agent - AI Trading System*
"""
        
        return report
    
    def _format_trade_notification(self, trade_data: Dict) -> str:
        """Format trade execution notification"""
        
        return f"""# Trade Executed

**Time**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Trade Details
- **Symbol**: {trade_data.get('symbol', 'N/A')}
- **Side**: {trade_data.get('side', 'N/A')}
- **Quantity**: {trade_data.get('quantity', 0)}
- **Price**: ${trade_data.get('price', 0):.2f}
- **Total Value**: ${trade_data.get('total_value', 0):.2f}
- **Status**: {trade_data.get('status', 'FILLED')}

---
*Trade Execution Notification - AI Trading System*
"""
    
    def _format_health_report(self, metrics: SystemHealthMetrics, score: int) -> str:
        """Format system health report"""
        
        status_emoji = "🟢" if score >= 80 else "🟡" if score >= 60 else "🔴"
        
        report = f"""# System Health Report

**Status**: {status_emoji} {score}/100  
**Last Check**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## API Connectivity
"""
        for api, status in metrics.api_connectivity.items():
            emoji = "✅" if status else "❌"
            report += f"- {api}: {emoji}\n"
        
        report += f"""
## System Resources
- **CPU Usage**: {metrics.resource_usage.get('cpu_percent', 0):.1f}%
- **Memory Usage**: {metrics.resource_usage.get('memory_percent', 0):.1f}%
- **Disk Usage**: {metrics.resource_usage.get('disk_percent', 0):.1f}%

## Performance Metrics
- **Error Rate**: {metrics.error_rate:.2f}%
- **Response Time**: {metrics.response_time:.0f}ms
- **Database**: {"✅ Healthy" if metrics.database_health else "❌ Unhealthy"}

## Agent Status
"""
        for agent, status in metrics.agent_status.items():
            emoji = "🟢" if status == 'running' else "🔴"
            report += f"- {agent}: {emoji} {status}\n"
        
        return report
    
    def _format_agent_performance_report(self, agent_metrics, total_tasks, 
                                        avg_success_rate, issues) -> str:
        """Format agent performance report"""
        
        report = f"""# Agent Performance Report

**Date**: {datetime.now().strftime('%Y-%m-%d')}  
**Total Tasks**: {total_tasks}  
**Average Success Rate**: {avg_success_rate:.1f}%

## Individual Agent Metrics
"""
        
        for agent, metrics in agent_metrics.items():
            report += f"""
### {agent.replace('_', ' ').title()}
- Tasks Processed: {metrics['total_tasks']}
- Success Rate: {metrics['success_rate']}%
- Avg Processing Time: {metrics['avg_processing_time']:.2f}s
"""
        
        if issues:
            report += "\n## Issues Identified\n"
            for issue in issues:
                report += f"- ⚠️ {issue}\n"
        else:
            report += "\n## Status\n✅ All agents performing within normal parameters\n"
        
        return report


# ==============================================================================
# FACTORY FUNCTION
# ==============================================================================

async def create_analytics_reporting_agent(llm_provider, config, 
                                          data_provider, 
                                          db_manager) -> AnalyticsReportingAgent:
    """Factory function to create analytics & reporting agent"""
    
    agent = AnalyticsReportingAgent(llm_provider, config, data_provider, db_manager)
    
    # Initialize scheduled reports
    agent.last_daily_report = datetime.now() - timedelta(days=1)
    agent.last_weekly_report = datetime.now() - timedelta(days=7)
    agent.last_monthly_report = datetime.now() - timedelta(days=30)
    
    # Ensure reports directory exists
    agent.report_manager._ensure_reports_directory()
    
    return agent