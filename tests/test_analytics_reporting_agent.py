# tests/test_analytics_reporting_agent.py
"""
Comprehensive test suite for Analytics & Reporting Agent
Tests all analytics, reporting, and file export functionality
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from unittest.mock import Mock, MagicMock, AsyncMock, patch
from pathlib import Path
import json
import tempfile
import shutil

from agents.analytics_reporting_agent import (
    AnalyticsReportingAgent,
    PerformanceAnalytics,
    ReportManager,
    PerformanceMetrics,
    SystemHealthMetrics,
    ReportType,
    AlertSeverity,
    ReportFormat
)


# ==============================================================================
# FIXTURES
# ==============================================================================

@pytest.fixture
def mock_config():
    """Create mock configuration"""
    config = Mock()
    config.LOG_LEVEL = 'INFO'
    return config


@pytest.fixture
def mock_data_provider():
    """Create mock data provider"""
    provider = Mock()
    
    # Mock portfolio data
    provider.get_portfolio_data = AsyncMock(return_value={
        'account': {
            'portfolio_value': 105000.00,
            'cash': 25000.00,
            'initial_capital': 100000.00
        },
        'positions': [
            {
                'symbol': 'AAPL',
                'qty': 100,
                'market_value': 15000.00,
                'unrealized_pl': 500.00,
                'unrealized_plpc': 3.45
            },
            {
                'symbol': 'GOOGL',
                'qty': 50,
                'market_value': 12000.00,
                'unrealized_pl': -200.00,
                'unrealized_plpc': -1.64
            }
        ],
        'portfolio_metrics': {
            'total_positions': 5,
            'total_value': 105000.00
        }
    })
    
    # Mock market data
    provider.get_latest_data = AsyncMock(return_value={
        'close': 450.00,
        'change_percent': 1.5
    })
    
    return provider


@pytest.fixture
def mock_db_manager():
    """Create mock database manager"""
    db_manager = Mock()
    
    # Mock session
    session = Mock()
    session.execute = Mock()
    session.query = Mock()
    session.commit = Mock()
    
    # Context manager for get_session
    db_manager.get_session = Mock(return_value=Mock(
        __enter__=Mock(return_value=session),
        __exit__=Mock(return_value=None)
    ))
    
    return db_manager


@pytest.fixture
def temp_reports_dir():
    """Create temporary reports directory for testing"""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)


@pytest.fixture
async def analytics_agent(mock_config, mock_data_provider, mock_db_manager, mock_llm_provider, temp_reports_dir):
    """Create analytics & reporting agent instance"""
    
    # Patch the reports directory
    with patch('agents.analytics_reporting_agent.Path') as mock_path:
        mock_path.return_value = Path(temp_reports_dir) / 'reports'
        
        agent = AnalyticsReportingAgent(
            llm_provider=mock_llm_provider,
            config=mock_config,
            data_provider=mock_data_provider,
            db_manager=mock_db_manager
        )
        
        # Override reports directory
        agent.report_manager.reports_dir = Path(temp_reports_dir) / 'reports'
        agent.report_manager._ensure_reports_directory()
        
        return agent


# ==============================================================================
# PERFORMANCE ANALYTICS TESTS
# ==============================================================================

class TestPerformanceAnalytics:
    """Test performance analytics functionality"""
    
    @pytest.mark.asyncio
    async def test_calculate_comprehensive_performance(self, mock_data_provider, mock_db_manager):
        """Test comprehensive performance calculation"""
        
        analytics = PerformanceAnalytics(mock_data_provider, mock_db_manager)
        
        # Mock historical data
        analytics._get_historical_performance = AsyncMock(
            return_value=[100000, 101000, 102000, 103000, 104000, 105000]
        )
        
        # Mock trade statistics
        analytics._calculate_trade_statistics = AsyncMock(return_value={
            'total_trades': 45,
            'winning_trades': 28,
            'losing_trades': 17,
            'win_rate': 62.2,
            'avg_win': 245.50,
            'avg_loss': -142.30
        })
        
        metrics = await analytics.calculate_comprehensive_performance()
        
        assert isinstance(metrics, PerformanceMetrics)
        assert metrics.total_value == 105000.00
        assert metrics.total_trades == 45
        assert metrics.win_rate == 62.2
        assert metrics.sharpe_ratio != 0  # Should calculate Sharpe
    
    def test_calculate_sharpe_ratio(self, mock_data_provider, mock_db_manager):
        """Test Sharpe ratio calculation"""
        
        analytics = PerformanceAnalytics(mock_data_provider, mock_db_manager)
        
        # Test with sample data
        historical_data = [100000, 101000, 99500, 102000, 103500, 105000]
        sharpe = analytics._calculate_sharpe_ratio(historical_data)
        
        assert isinstance(sharpe, float)
        assert sharpe != 0  # Should calculate non-zero Sharpe for volatile data
    
    def test_calculate_max_drawdown(self, mock_data_provider, mock_db_manager):
        """Test maximum drawdown calculation"""
        
        analytics = PerformanceAnalytics(mock_data_provider, mock_db_manager)
        
        # Test with drawdown scenario
        historical_data = [100000, 105000, 103000, 98000, 99000, 102000]
        drawdown = analytics._calculate_max_drawdown(historical_data)
        
        assert isinstance(drawdown, float)
        assert drawdown > 0  # Should detect drawdown
        assert drawdown <= 100  # Percentage should not exceed 100


# ==============================================================================
# REPORT MANAGER TESTS
# ==============================================================================

class TestReportManager:
    """Test report manager functionality"""
    
    @pytest.mark.asyncio
    async def test_save_report_markdown(self, mock_config, temp_reports_dir):
        """Test saving report in markdown format"""
        
        manager = ReportManager(mock_config)
        manager.reports_dir = Path(temp_reports_dir) / 'reports'
        manager._ensure_reports_directory()
        
        content = "# Test Report\nThis is a test report."
        result = await manager.save_report(
            content,
            ReportType.DAILY_SUMMARY,
            ReportFormat.MARKDOWN
        )
        
        assert result['success'] is True
        assert 'filepath' in result
        assert Path(result['filepath']).exists()
        
        # Verify content
        with open(result['filepath'], 'r') as f:
            saved_content = f.read()
        assert "Test Report" in saved_content
    
    @pytest.mark.asyncio
    async def test_save_report_json(self, mock_config, temp_reports_dir):
        """Test saving report in JSON format"""
        
        manager = ReportManager(mock_config)
        manager.reports_dir = Path(temp_reports_dir) / 'reports'
        manager._ensure_reports_directory()
        
        content = "Test JSON report content"
        result = await manager.save_report(
            content,
            ReportType.WEEKLY_PERFORMANCE,
            ReportFormat.JSON
        )
        
        assert result['success'] is True
        assert result['filepath'].endswith('.json')
        
        # Verify JSON structure
        with open(result['filepath'], 'r') as f:
            data = json.load(f)
        assert data['report_type'] == 'weekly_performance'
        assert 'content' in data
    
    @pytest.mark.asyncio
    async def test_save_report_html(self, mock_config, temp_reports_dir):
        """Test saving report in HTML format"""
        
        manager = ReportManager(mock_config)
        manager.reports_dir = Path(temp_reports_dir) / 'reports'
        manager._ensure_reports_directory()
        
        content = "# HTML Report\n**Bold text**"
        result = await manager.save_report(
            content,
            ReportType.RISK_ALERT,
            ReportFormat.HTML
        )
        
        assert result['success'] is True
        assert result['filepath'].endswith('.html')
        
        # Verify HTML structure
        with open(result['filepath'], 'r') as f:
            html = f.read()
        assert '<html>' in html
        assert '<title>' in html
    
    def test_ensure_reports_directory(self, mock_config, temp_reports_dir):
        """Test reports directory creation"""
        
        manager = ReportManager(mock_config)
        manager.reports_dir = Path(temp_reports_dir) / 'test_reports'
        manager._ensure_reports_directory()
        
        # Check all subdirectories created
        expected_dirs = ['daily', 'weekly', 'monthly', 'alerts', 'trades', 'system', 'snapshots']
        for subdir in expected_dirs:
            assert (manager.reports_dir / subdir).exists()
    
    @pytest.mark.asyncio
    async def test_export_portfolio_snapshot(self, mock_config, temp_reports_dir):
        """Test portfolio snapshot export"""
        
        manager = ReportManager(mock_config)
        manager.reports_dir = Path(temp_reports_dir) / 'reports'
        manager._ensure_reports_directory()
        
        portfolio_data = {'positions': [], 'account': {'value': 100000}}
        performance = PerformanceMetrics(
            total_value=100000, daily_pnl=1000, daily_pnl_pct=1.0,
            weekly_pnl=5000, weekly_pnl_pct=5.0, monthly_pnl=10000,
            monthly_pnl_pct=10.0, ytd_return=10.0, sharpe_ratio=1.5,
            max_drawdown=5.0, win_rate=60.0, avg_win=200, avg_loss=-100,
            total_trades=50, winning_trades=30, losing_trades=20
        )
        
        result = await manager.export_portfolio_snapshot(portfolio_data, performance)
        
        assert result['success'] is True
        assert result['format'] == 'json'
        
        # Verify snapshot content
        with open(result['filepath'], 'r') as f:
            snapshot = json.load(f)
        assert 'timestamp' in snapshot
        assert 'portfolio' in snapshot
        assert 'performance' in snapshot


# ==============================================================================
# ANALYTICS & REPORTING AGENT TESTS
# ==============================================================================

class TestAnalyticsReportingAgent:
    """Test main analytics & reporting agent functionality"""
    
    @pytest.mark.asyncio
    async def test_agent_initialization(self, analytics_agent):
        """Test agent initialization"""
        
        assert analytics_agent.agent_name == "analytics_reporting"
        assert analytics_agent.analytics is not None
        assert analytics_agent.report_manager is not None
        assert len(analytics_agent.alert_thresholds) > 0
    
    @pytest.mark.asyncio
    async def test_generate_daily_summary(self, analytics_agent):
        """Test daily summary generation"""
        
        # Mock helper methods
        analytics_agent._get_market_summary = AsyncMock(return_value={
            'SPY': {'price': 450.00, 'change_pct': 1.5}
        })
        analytics_agent._get_top_movers = AsyncMock(return_value={
            'gainers': [{'symbol': 'AAPL', 'unrealized_plpc': 3.45}],
            'losers': [{'symbol': 'GOOGL', 'unrealized_plpc': -1.64}]
        })
        analytics_agent._get_upcoming_events = AsyncMock(return_value=[
            {'date': 'Tomorrow', 'event': 'Fed Minutes'}
        ])
        
        result = await analytics_agent.generate_daily_summary()
        
        assert result['status'] == 'success'
        assert result['report_type'] == ReportType.DAILY_SUMMARY.value
        assert 'save_result' in result
        assert analytics_agent.last_daily_report is not None
        
        # Verify report was saved
        reports_dir = analytics_agent.report_manager.reports_dir / 'daily'
        assert any(reports_dir.glob('*.md'))
    
    @pytest.mark.asyncio
    async def test_generate_weekly_performance_report(self, analytics_agent):
        """Test weekly performance report generation"""
        
        # Mock data collection methods
        analytics_agent._analyze_weekly_trades = AsyncMock(return_value={
            'total_trades': 23,
            'profitable_trades': 15
        })
        analytics_agent._collect_agent_performance_data = AsyncMock(return_value={
            'junior_analyst': {'success_rate': 92}
        })
        analytics_agent._calculate_risk_metrics = AsyncMock(return_value={
            'portfolio_beta': 1.15
        })
        
        result = await analytics_agent.generate_weekly_performance_report()
        
        assert result['status'] == 'success'
        assert result['report_type'] == ReportType.WEEKLY_PERFORMANCE.value
        assert 'save_results' in result
        assert len(result['save_results']) > 0
        assert analytics_agent.last_weekly_report is not None
    
    @pytest.mark.asyncio
    async def test_monitor_risk_alerts(self, analytics_agent):
        """Test risk alert monitoring"""
        
        # Set up test scenario with threshold breach
        analytics_agent.analytics.calculate_comprehensive_performance = AsyncMock(
            return_value=PerformanceMetrics(
                total_value=95000,
                daily_pnl=-5000,
                daily_pnl_pct=-5.0,  # Exceeds threshold
                weekly_pnl=-2000,
                weekly_pnl_pct=-2.0,
                monthly_pnl=1000,
                monthly_pnl_pct=1.0,
                ytd_return=5.0,
                sharpe_ratio=1.2,
                max_drawdown=10.0,
                win_rate=60,
                avg_win=200,
                avg_loss=-150,
                total_trades=20,
                winning_trades=12,
                losing_trades=8
            )
        )
        
        result = await analytics_agent.monitor_risk_alerts()
        
        assert result['status'] == 'success'
        assert result['alerts_triggered'] > 0
        
        # Verify alerts were saved
        alerts_dir = analytics_agent.report_manager.reports_dir / 'alerts'
        assert any(alerts_dir.glob('*.md'))
    
    @pytest.mark.asyncio
    async def test_record_trade_notification(self, analytics_agent):
        """Test trade notification recording"""
        
        trade_data = {
            'symbol': 'AAPL',
            'side': 'BUY',
            'quantity': 100,
            'price': 150.00,
            'total_value': 15000.00,
            'status': 'FILLED'
        }
        
        result = await analytics_agent.record_trade_notification(trade_data)
        
        assert result['status'] == 'success'
        assert result['notification_recorded'] is True
        
        # Verify trade notification was saved
        trades_dir = analytics_agent.report_manager.reports_dir / 'trades'
        assert any(trades_dir.glob('*.md'))
    
    @pytest.mark.asyncio
    async def test_check_system_health(self, analytics_agent):
        """Test system health monitoring"""
        
        # Mock health check methods
        analytics_agent._check_api_connectivity = AsyncMock(return_value={
            'alpaca': True,
            'llm_provider': True
        })
        analytics_agent._check_database_health = AsyncMock(return_value=True)
        analytics_agent._check_agent_status = AsyncMock(return_value={
            'junior_analyst': 'running'
        })
        analytics_agent._check_resource_usage = AsyncMock(return_value={
            'cpu_percent': 45.0
        })
        analytics_agent._calculate_error_rate = AsyncMock(return_value=0.5)
        analytics_agent._measure_response_time = AsyncMock(return_value=200.0)
        
        result = await analytics_agent.check_system_health()
        
        assert result['status'] == 'success'
        assert 'health_score' in result
        assert result['health_score'] >= 0
        assert result['health_score'] <= 100
        
        # Verify health report was saved
        system_dir = analytics_agent.report_manager.reports_dir / 'system'
        assert any(system_dir.glob('*.md'))
    
    @pytest.mark.asyncio
    async def test_analyze_agent_performance(self, analytics_agent):
        """Test agent performance analysis"""
        
        analytics_agent._collect_agent_performance_data = AsyncMock(return_value={
            'junior_analyst': {
                'total_tasks': 100,
                'success_rate': 92,
                'avg_processing_time': 2.3
            },
            'senior_analyst': {
                'total_tasks': 50,
                'success_rate': 88,
                'avg_processing_time': 4.5
            }
        })
        
        result = await analytics_agent.analyze_agent_performance()
        
        assert result['status'] == 'success'
        assert result['total_tasks_processed'] == 150
        assert result['average_success_rate'] == 90
        assert 'agent_metrics' in result
        
        # Verify performance report was saved
        system_dir = analytics_agent.report_manager.reports_dir / 'system'
        assert any(system_dir.glob('agent_performance_*.md'))
    
    @pytest.mark.asyncio
    async def test_export_portfolio_snapshot(self, analytics_agent):
        """Test portfolio snapshot export"""
        
        result = await analytics_agent.export_portfolio_snapshot()
        
        assert result['status'] == 'success'
        assert 'export_result' in result
        
        # Verify snapshot was saved
        snapshots_dir = analytics_agent.report_manager.reports_dir / 'snapshots'
        assert any(snapshots_dir.glob('*.json'))
    
    @pytest.mark.asyncio
    async def test_process_unknown_task(self, analytics_agent):
        """Test handling of unknown task type"""
        
        result = await analytics_agent.process({'task_type': 'unknown_task'})
        
        assert 'error' in result
        assert 'Unknown task type' in result['error']
    
    def test_format_daily_report(self, analytics_agent):
        """Test daily report formatting"""
        
        performance = PerformanceMetrics(
            total_value=105000,
            daily_pnl=1000,
            daily_pnl_pct=0.96,
            weekly_pnl=5000,
            weekly_pnl_pct=5.0,
            monthly_pnl=10000,
            monthly_pnl_pct=10.5,
            ytd_return=5.0,
            sharpe_ratio=1.5,
            max_drawdown=8.0,
            win_rate=65,
            avg_win=250,
            avg_loss=-150,
            total_trades=30,
            winning_trades=20,
            losing_trades=10
        )
        
        report = analytics_agent._format_daily_report(
            "Test summary",
            performance,
            {},
            {'SPY': {'price': 450, 'change_pct': 1.5}},
            {'gainers': [], 'losers': []},
            []
        )
        
        assert "Daily Executive Summary" in report
        assert "$105,000.00" in report
        assert "+0.96%" in report
        assert "Test summary" in report
        assert "Analytics & Reporting Agent" in report
    
    @pytest.mark.asyncio
    async def test_create_alert(self, analytics_agent):
        """Test alert creation"""
        
        alert = await analytics_agent._create_alert(
            AlertSeverity.WARNING,
            "Test Alert",
            "This is a test alert",
            {'value': 100}
        )
        
        assert alert['severity'] == AlertSeverity.WARNING.value
        assert alert['title'] == "Test Alert"
        assert alert['message'] == "This is a test alert"
        assert 'alert_id' in alert
        assert 'timestamp' in alert
    
    def test_calculate_health_score(self, analytics_agent):
        """Test health score calculation"""
        
        metrics = SystemHealthMetrics(
            api_connectivity={'alpaca': True, 'llm': True},
            database_health=True,
            agent_status={'all': 'running'},
            resource_usage={'cpu': 50.0},
            error_rate=0.5,
            response_time=200.0,
            last_heartbeat=datetime.now()
        )
        
        score = analytics_agent._calculate_health_score(metrics)
        
        assert score == 100  # All healthy
        
        # Test with issues
        metrics.database_health = False
        score = analytics_agent._calculate_health_score(metrics)
        assert score < 100


# ==============================================================================
# INTEGRATION TESTS
# ==============================================================================

class TestIntegration:
    """Integration tests for analytics & reporting agent"""
    
    @pytest.mark.asyncio
    async def test_full_daily_workflow(self, analytics_agent):
        """Test complete daily reporting workflow"""
        
        # Mock all dependencies
        analytics_agent._get_market_summary = AsyncMock(return_value={
            'SPY': {'price': 450.00, 'change_pct': 1.5},
            'QQQ': {'price': 380.00, 'change_pct': 2.0}
        })
        analytics_agent._get_top_movers = AsyncMock(return_value={
            'gainers': [
                {'symbol': 'AAPL', 'unrealized_plpc': 5.0},
                {'symbol': 'NVDA', 'unrealized_plpc': 4.5}
            ],
            'losers': [
                {'symbol': 'TSLA', 'unrealized_plpc': -3.0}
            ]
        })
        analytics_agent._get_upcoming_events = AsyncMock(return_value=[
            {'date': 'Tomorrow', 'event': 'CPI Data'},
            {'date': 'Friday', 'event': 'Options Expiry'}
        ])
        
        # Run daily summary
        result = await analytics_agent.generate_daily_summary()
        
        # Verify workflow completed
        assert result['status'] == 'success'
        assert all([
            analytics_agent._get_market_summary.called,
            analytics_agent._get_top_movers.called,
            analytics_agent._get_upcoming_events.called
        ])
        
        # Verify files were created
        daily_dir = analytics_agent.report_manager.reports_dir / 'daily'
        md_files = list(daily_dir.glob('*.md'))
        html_files = list(daily_dir.glob('*.html'))
        assert len(md_files) > 0
        assert len(html_files) > 0
    
    @pytest.mark.asyncio
    async def test_alert_workflow(self, analytics_agent):
        """Test alert generation and storage workflow"""
        
        # Create critical alert
        alert = await analytics_agent._create_alert(
            AlertSeverity.CRITICAL,
            "Critical System Alert",
            "Database connection lost",
            {'component': 'database', 'status': 'down'}
        )
        
        # Save alert
        success = await analytics_agent._save_alert(alert)
        
        assert success is True
        
        # Verify alert was saved
        alerts_dir = analytics_agent.report_manager.reports_dir / 'alerts'
        alert_files = list(alerts_dir.glob('*.md'))
        assert len(alert_files) > 0
        
        # Verify alert content
        with open(alert_files[0], 'r') as f:
            content = f.read()
        assert "Critical System Alert" in content
        assert "CRITICAL" in content.upper()
    
    @pytest.mark.asyncio
    async def test_concurrent_report_generation(self, analytics_agent):
        """Test concurrent report generation"""
        
        # Mock all report dependencies
        analytics_agent._get_market_summary = AsyncMock(return_value={})
        analytics_agent._get_top_movers = AsyncMock(return_value={'gainers': [], 'losers': []})
        analytics_agent._get_upcoming_events = AsyncMock(return_value=[])
        analytics_agent._analyze_weekly_trades = AsyncMock(return_value={})
        analytics_agent._collect_agent_performance_data = AsyncMock(return_value={})
        analytics_agent._calculate_risk_metrics = AsyncMock(return_value={})
        analytics_agent._check_api_connectivity = AsyncMock(return_value={})
        analytics_agent._check_database_health = AsyncMock(return_value=True)
        analytics_agent._check_agent_status = AsyncMock(return_value={})
        analytics_agent._check_resource_usage = AsyncMock(return_value={})
        analytics_agent._calculate_error_rate = AsyncMock(return_value=0.1)
        analytics_agent._measure_response_time = AsyncMock(return_value=100)
        
        # Run multiple reports concurrently
        tasks = [
            analytics_agent.generate_daily_summary(),
            analytics_agent.check_system_health(),
            analytics_agent.export_portfolio_snapshot()
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Verify all completed without exceptions
        for result in results:
            assert not isinstance(result, Exception)
            assert result['status'] == 'success'


# ==============================================================================
# EDGE CASE TESTS
# ==============================================================================

class TestEdgeCases:
    """Test edge cases and boundary conditions"""
    
    @pytest.mark.asyncio
    async def test_empty_portfolio(self, analytics_agent):
        """Test handling of empty portfolio"""
        
        analytics_agent.data_provider.get_portfolio_data = AsyncMock(return_value={
            'account': {
                'portfolio_value': 100000.00,
                'cash': 100000.00,
                'initial_capital': 100000.00
            },
            'positions': [],
            'portfolio_metrics': {
                'total_positions': 0,
                'total_value': 100000.00
            }
        })
        
        analytics_agent._get_market_summary = AsyncMock(return_value={})
        analytics_agent._get_upcoming_events = AsyncMock(return_value=[])
        
        result = await analytics_agent.generate_daily_summary()
        
        assert result['status'] == 'success'
        assert result['metrics']['portfolio_value'] == 100000.00
    
    @pytest.mark.asyncio
    async def test_api_failure_handling(self, analytics_agent):
        """Test handling of API failures"""
        
        # Simulate data provider failure
        analytics_agent.data_provider.get_portfolio_data = AsyncMock(
            return_value={'error': 'API connection failed'}
        )
        
        result = await analytics_agent.generate_daily_summary()
        
        assert result['status'] == 'error'
        assert 'Failed to get portfolio data' in result.get('message', '')
    
    @pytest.mark.asyncio
    async def test_file_write_failure(self, analytics_agent):
        """Test handling of file write failures"""
        
        # Make reports directory read-only
        import os
        reports_dir = analytics_agent.report_manager.reports_dir / 'daily'
        os.chmod(reports_dir, 0o444)
        
        try:
            result = await analytics_agent.report_manager.save_report(
                "Test content",
                ReportType.DAILY_SUMMARY,
                ReportFormat.MARKDOWN
            )
            
            assert result['success'] is False
            assert 'error' in result
        finally:
            # Restore permissions
            os.chmod(reports_dir, 0o755)
    
    def test_extreme_metric_values(self, mock_data_provider, mock_db_manager):
        """Test handling of extreme metric values"""
        
        analytics = PerformanceAnalytics(mock_data_provider, mock_db_manager)
        
        # Test with extreme values
        extreme_data = [1e10, 1e-10, 0, -1e10]
        
        sharpe = analytics._calculate_sharpe_ratio(extreme_data)
        assert isinstance(sharpe, float)
        assert not float('inf') == abs(sharpe)
        
        drawdown = analytics._calculate_max_drawdown(extreme_data)
        assert isinstance(drawdown, float)
        assert 0 <= drawdown <= 100


# ==============================================================================
# TEST RUNNER
# ==============================================================================

def run_all_tests():
    """Run all analytics & reporting agent tests"""
    
    print("=" * 60)
    print("RUNNING ANALYTICS & REPORTING AGENT TEST SUITE")
    print("=" * 60)
    
    # Run pytest with verbose output
    pytest.main([
        __file__,
        '-v',
        '--tb=short',
        '--color=yes',
        '-x'  # Stop on first failure
    ])


if __name__ == "__main__":
    run_all_tests()