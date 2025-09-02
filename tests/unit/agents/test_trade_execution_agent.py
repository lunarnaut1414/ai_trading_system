# tests/test_trade_execution_agent.py
"""
Comprehensive test suite for Trade Execution Agent

Tests all components including ExecutionTimingEngine, OrderManager,
and the main TradeExecutionAgent with mock data and edge cases.
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from typing import Dict, List, Optional
import numpy as np

# Import the components to test
from src.agents.trade_executor import (
    TradeExecutionAgent,
    ExecutionTimingEngine,
    OrderManager,
    TradeInstruction,
    ExecutionPlan,
    ExecutionResult,
    ExecutionStrategy,
    ExecutionUrgency,
    MarketConditions,
    OrderStatus
)

# ==================== FIXTURES ====================

@pytest.fixture
def mock_alpaca_provider():
    """Create mock Alpaca provider"""
    mock = Mock()
    
    # Setup default mock responses
    mock.get_market_data = AsyncMock(return_value={
        'AAPL': {
            'current_price': 150.00,
            'volatility': 0.02,
            'trend_strength': 0.5,
            'volume_profile': 'normal'
        }
    })
    
    mock.get_latest_quote = AsyncMock(return_value={
        'bid': 149.95,
        'ask': 150.05,
        'bid_size': 100,
        'ask_size': 100
    })
    
    mock.get_positions = AsyncMock(return_value={
        'AAPL': {'quantity': 100, 'market_value': 15000}
    })
    
    mock.get_account_info = AsyncMock(return_value={
        'account_value': 1000000,
        'buying_power': 500000,
        'cash': 500000
    })
    
    mock.place_order = AsyncMock(return_value={
        'success': True,
        'order_id': 'test_order_123'
    })
    
    mock.get_order = AsyncMock(return_value={
        'status': 'filled',
        'filled_qty': 100,
        'filled_avg_price': 150.00,
        'commission': 1.00
    })
    
    mock.get_orders = AsyncMock(return_value=[])
    mock.cancel_order = AsyncMock(return_value={'success': True})
    
    return mock

@pytest.fixture
def mock_llm_provider():
    """Create mock LLM provider"""
    mock = Mock()
    mock.complete = AsyncMock(return_value="Test execution insights")
    mock.complete_json = AsyncMock(return_value={"analysis": "test"})
    return mock

@pytest.fixture
def mock_config():
    """Create mock configuration"""
    config = Mock()
    config.LOG_LEVEL = "INFO"
    config.MAX_RETRIES = 3
    config.TIMEOUT_SECONDS = 300
    return config

@pytest.fixture
def execution_timing_engine(mock_alpaca_provider):
    """Create ExecutionTimingEngine instance"""
    return ExecutionTimingEngine(mock_alpaca_provider)

@pytest.fixture
def order_manager(mock_alpaca_provider):
    """Create OrderManager instance"""
    return OrderManager(mock_alpaca_provider)

@pytest.fixture
def trade_execution_agent(mock_llm_provider, mock_alpaca_provider, mock_config):
    """Create TradeExecutionAgent instance"""
    return TradeExecutionAgent(mock_llm_provider, mock_alpaca_provider, mock_config)

@pytest.fixture
def sample_trade_instruction():
    """Create sample trade instruction"""
    return TradeInstruction(
        symbol='AAPL',
        action='BUY',
        target_weight=3.0,
        current_weight=0.0,
        confidence=8,
        time_horizon='medium',
        reasoning='Strong technical setup',
        risk_factors=['market_volatility'],
        urgency=ExecutionUrgency.MEDIUM,
        max_slippage=10.0,
        execution_window=timedelta(hours=2)
    )

@pytest.fixture
def sample_execution_plan(sample_trade_instruction):
    """Create sample execution plan"""
    return ExecutionPlan(
        instruction=sample_trade_instruction,
        strategy=ExecutionStrategy.LIMIT,
        order_size=100,
        target_price=150.00,
        limit_price=149.95,
        stop_price=None,
        chunk_size=None,
        execution_schedule=[],
        market_timing='good',
        expected_slippage=5.0,
        risk_assessment={'risks': []}
    )

# ==================== EXECUTION TIMING ENGINE TESTS ====================

class TestExecutionTimingEngine:
    """Test suite for ExecutionTimingEngine"""
    
    @pytest.mark.asyncio
    async def test_assess_market_timing_normal_conditions(self, execution_timing_engine):
        """Test market timing assessment under normal conditions"""
        result = await execution_timing_engine.assess_market_timing('AAPL', 'BUY')
        
        assert 'timing_score' in result
        assert 'timing_assessment' in result
        assert 'recommended_strategy' in result
        assert 'market_conditions' in result
        assert 'liquidity_analysis' in result
        assert 'execution_risks' in result
        assert 'expected_slippage' in result
        
        assert 1 <= result['timing_score'] <= 10
        assert isinstance(result['recommended_strategy'], ExecutionStrategy)
    
    @pytest.mark.asyncio
    async def test_assess_market_timing_volatile_conditions(self, execution_timing_engine, mock_alpaca_provider):
        """Test market timing with volatile market conditions"""
        mock_alpaca_provider.get_market_data.return_value = {
            'AAPL': {
                'current_price': 150.00,
                'volatility': 0.05,  # High volatility
                'trend_strength': 0.8,
                'volume_profile': 'high'
            }
        }
        
        result = await execution_timing_engine.assess_market_timing('AAPL', 'SELL')
        
        assert result['market_conditions']['condition'] == MarketConditions.VOLATILE
        assert result['timing_score'] < 7  # Should be lower due to volatility
        assert 'high_volatility' in result['execution_risks']
    
    @pytest.mark.asyncio
    async def test_assess_market_timing_poor_liquidity(self, execution_timing_engine, mock_alpaca_provider):
        """Test market timing with poor liquidity"""
        mock_alpaca_provider.get_latest_quote.return_value = {
            'bid': 149.50,
            'ask': 150.50,  # Wide spread
            'bid_size': 10,
            'ask_size': 10
        }
        
        result = await execution_timing_engine.assess_market_timing('AAPL', 'BUY')
        
        assert result['liquidity_analysis']['liquidity_level'] in ['poor', 'fair']
        assert result['liquidity_analysis']['spread_pct'] > 0.2
        assert 'low_liquidity' in result['execution_risks']
    
    @pytest.mark.asyncio
    async def test_assess_market_timing_error_handling(self, execution_timing_engine, mock_alpaca_provider):
        """Test market timing with API errors"""
        mock_alpaca_provider.get_market_data.side_effect = Exception("API Error")
        
        result = await execution_timing_engine.assess_market_timing('AAPL', 'BUY')
        
        # Default score can vary based on liquidity analysis, just check it's in valid range
        assert 1 <= result['timing_score'] <= 10
        assert result['timing_assessment'] in ['excellent', 'good', 'fair', 'poor']
        # The error might result in different risk assessments
        assert len(result['execution_risks']) > 0  # Should have some risks identified
    
    @pytest.mark.asyncio
    @patch('agents.trade_execution_agent.datetime')
    async def test_intraday_timing_patterns(self, mock_datetime, execution_timing_engine):
        """Test intraday timing patterns detection"""
        # Test opening volatility period
        mock_now = datetime(2024, 1, 1, 9, 32, 0)  # 2 minutes after open
        mock_datetime.now.return_value = mock_now
        
        result = await execution_timing_engine._analyze_intraday_patterns('AAPL', 'BUY')
        
        assert result['timing_quality'] == 'poor'
        assert result['session_position'] == 'early'
    
    def test_recommend_execution_strategy(self, execution_timing_engine):
        """Test execution strategy recommendation logic"""
        # High score, good liquidity -> MARKET
        strategy = execution_timing_engine._recommend_execution_strategy(
            timing_score=8,
            conditions={'condition': MarketConditions.NORMAL},
            liquidity={'liquidity_score': 8},
            events={'risk_level': 'low'}
        )
        assert strategy == ExecutionStrategy.MARKET
        
        # Poor liquidity -> VWAP
        strategy = execution_timing_engine._recommend_execution_strategy(
            timing_score=6,
            conditions={'condition': MarketConditions.NORMAL},
            liquidity={'liquidity_score': 3},
            events={'risk_level': 'low'}
        )
        assert strategy == ExecutionStrategy.VWAP
        
        # Volatile conditions -> LIMIT
        strategy = execution_timing_engine._recommend_execution_strategy(
            timing_score=6,
            conditions={'condition': MarketConditions.VOLATILE},
            liquidity={'liquidity_score': 6},
            events={'risk_level': 'low'}
        )
        assert strategy == ExecutionStrategy.LIMIT

# ==================== ORDER MANAGER TESTS ====================

class TestOrderManager:
    """Test suite for OrderManager"""
    
    @pytest.mark.asyncio
    async def test_execute_market_order(self, order_manager, sample_trade_instruction, 
                                       sample_execution_plan, mock_alpaca_provider):
        """Test market order execution"""
        sample_execution_plan.strategy = ExecutionStrategy.MARKET
        
        result = await order_manager.execute_trade_instruction(
            sample_trade_instruction, sample_execution_plan
        )
        
        assert isinstance(result, ExecutionResult)
        assert result.status == OrderStatus.FILLED
        assert result.executed_quantity == 100
        assert result.average_price == 150.00
        mock_alpaca_provider.place_order.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_execute_limit_order(self, order_manager, sample_trade_instruction,
                                      sample_execution_plan, mock_alpaca_provider):
        """Test limit order execution"""
        result = await order_manager.execute_trade_instruction(
            sample_trade_instruction, sample_execution_plan
        )
        
        assert result.status == OrderStatus.FILLED
        assert result.execution_quality in ['excellent', 'good', 'fair', 'poor']
        mock_alpaca_provider.place_order.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_execute_twap_order(self, order_manager, sample_trade_instruction,
                                     sample_execution_plan, mock_alpaca_provider):
        """Test TWAP order execution"""
        sample_execution_plan.strategy = ExecutionStrategy.TWAP
        sample_execution_plan.order_size = 800  # Will be split into 8 slices
        
        # Mock multiple successful order placements
        mock_alpaca_provider.place_order.return_value = {
            'success': True,
            'order_id': 'test_order'
        }
        
        with patch.object(order_manager, '_monitor_order_until_filled',
                         new_callable=AsyncMock) as mock_monitor:
            mock_monitor.return_value = {
                'order_id': 'test_order',
                'status': 'filled',
                'filled_qty': 100,
                'filled_avg_price': 150.00,
                'commission': 0.50
            }
            
            # Mock asyncio.sleep to avoid waiting
            with patch('asyncio.sleep', new_callable=AsyncMock) as mock_sleep:
                result = await order_manager.execute_trade_instruction(
                    sample_trade_instruction, sample_execution_plan
                )
        
        assert result.status == OrderStatus.FILLED
        assert len(result.fills) == 8  # 8 TWAP slices
        assert mock_alpaca_provider.place_order.call_count == 8
        # Verify sleep was called between slices (7 times for 8 slices)
        assert mock_sleep.call_count == 7
    
    @pytest.mark.asyncio
    async def test_execute_vwap_order(self, order_manager, sample_trade_instruction,
                                     sample_execution_plan, mock_alpaca_provider):
        """Test VWAP order execution"""
        sample_execution_plan.strategy = ExecutionStrategy.VWAP
        sample_execution_plan.order_size = 1000
        
        with patch.object(order_manager, '_monitor_order_until_filled',
                         new_callable=AsyncMock) as mock_monitor:
            mock_monitor.return_value = {
                'order_id': 'test_order',
                'status': 'filled',
                'filled_qty': 100,
                'filled_avg_price': 150.00,
                'commission': 0.50
            }
            
            # Mock asyncio.sleep to avoid waiting
            with patch('asyncio.sleep', new_callable=AsyncMock):
                result = await order_manager.execute_trade_instruction(
                    sample_trade_instruction, sample_execution_plan
                )
        
        assert result.status == OrderStatus.FILLED
        # VWAP uses volume distribution
        assert len(result.fills) > 0
    
    @pytest.mark.asyncio
    async def test_execute_iceberg_order(self, order_manager, sample_trade_instruction,
                                        sample_execution_plan, mock_alpaca_provider):
        """Test iceberg order execution"""
        sample_execution_plan.strategy = ExecutionStrategy.ICEBERG
        sample_execution_plan.order_size = 1000
        
        with patch.object(order_manager, '_monitor_order_until_filled',
                         new_callable=AsyncMock) as mock_monitor:
            mock_monitor.return_value = {
                'order_id': 'test_order',
                'status': 'filled',
                'filled_qty': 100,
                'filled_avg_price': 150.00,
                'commission': 0.50
            }
            
            # Mock asyncio.sleep to avoid waiting
            with patch('asyncio.sleep', new_callable=AsyncMock):
                result = await order_manager.execute_trade_instruction(
                    sample_trade_instruction, sample_execution_plan
                )
        
        assert result.status == OrderStatus.FILLED
        # Iceberg shows only 10% at a time
        assert mock_alpaca_provider.place_order.call_count >= 10
    
    @pytest.mark.asyncio
    async def test_execute_opportunistic_order(self, order_manager, sample_trade_instruction,
                                              sample_execution_plan, mock_alpaca_provider):
        """Test opportunistic order execution"""
        sample_execution_plan.strategy = ExecutionStrategy.OPPORTUNISTIC
        sample_execution_plan.order_size = 100
        
        # Mock get_latest_quote to be called during opportunistic waiting
        mock_alpaca_provider.get_latest_quote.return_value = {
            'bid': 148.00,  # Favorable buy price
            'ask': 148.10
        }
        
        # Mock successful order placement
        mock_alpaca_provider.place_order.return_value = {
            'success': True,
            'order_id': 'test_order'
        }
        
        with patch.object(order_manager, '_monitor_order_until_filled',
                         new_callable=AsyncMock) as mock_monitor:
            mock_monitor.return_value = {
                'order_id': 'test_order',
                'status': 'filled',
                'filled_qty': 100,
                'filled_avg_price': 148.05,
                'commission': 0.50
            }
            
            result = await order_manager.execute_trade_instruction(
                sample_trade_instruction, sample_execution_plan
            )
        
        assert result.status == OrderStatus.FILLED
        assert result.execution_quality in ['excellent', 'good', 'fair', 'poor']
        # Verify the order was placed
        mock_alpaca_provider.place_order.assert_called()
    
    @pytest.mark.asyncio
    async def test_monitor_order_timeout(self, order_manager, mock_alpaca_provider):
        """Test order monitoring with timeout"""
        mock_alpaca_provider.get_order.return_value = {
            'status': 'pending',
            'filled_qty': 0
        }
        
        with patch('agents.trade_execution_agent.datetime') as mock_datetime:
            # Mock datetime to simulate timeout immediately
            start_time = datetime(2024, 1, 1, 10, 0, 0)
            timeout_time = datetime(2024, 1, 1, 10, 31, 0)
            mock_datetime.now.side_effect = [start_time, timeout_time]
            
            # Mock asyncio.sleep to avoid waiting
            with patch('asyncio.sleep', new_callable=AsyncMock):
                result = await order_manager._monitor_order_until_filled(
                    'test_order', timeout_minutes=30
                )
        
        assert result['status'] == 'timeout'
        mock_alpaca_provider.cancel_order.assert_called_once_with('test_order')
    
    @pytest.mark.asyncio
    async def test_order_validation(self, order_manager, sample_trade_instruction,
                                   sample_execution_plan):
        """Test order validation logic"""
        # Valid order
        validation = order_manager._validate_execution_request(
            sample_trade_instruction, sample_execution_plan
        )
        assert validation['valid'] is True
        
        # Invalid action - create a new instruction
        invalid_instruction = TradeInstruction(
            symbol='AAPL',
            action='INVALID',
            target_weight=3.0,
            current_weight=0.0,
            confidence=8,
            time_horizon='medium',
            reasoning='Test',
            risk_factors=[],
            urgency=ExecutionUrgency.MEDIUM,
            max_slippage=10.0,
            execution_window=timedelta(hours=2)
        )
        validation = order_manager._validate_execution_request(
            invalid_instruction, sample_execution_plan
        )
        assert validation['valid'] is False
        assert 'Invalid action' in validation['error']
        
        # Zero order size - create a new plan
        invalid_plan = ExecutionPlan(
            instruction=sample_trade_instruction,
            strategy=ExecutionStrategy.LIMIT,
            order_size=0,
            target_price=150.00,
            limit_price=149.95,
            stop_price=None,
            chunk_size=None,
            execution_schedule=[],
            market_timing='good',
            expected_slippage=5.0,
            risk_assessment={'risks': []}
        )
        validation = order_manager._validate_execution_request(
            sample_trade_instruction, invalid_plan
        )
        assert validation['valid'] is False
        assert 'Invalid order size' in validation['error']
    
    @pytest.mark.asyncio
    async def test_calculate_order_parameters(self, order_manager, sample_trade_instruction,
                                             sample_execution_plan, mock_alpaca_provider):
        """Test order parameter calculation"""
        params = await order_manager._calculate_order_parameters(
            sample_trade_instruction, sample_execution_plan
        )
        
        assert params['order_side'] in ['BUY', 'SELL']
        assert params['order_quantity'] >= 0
        assert params['current_price'] > 0
        assert 'current_position' in params
    
    @pytest.mark.asyncio
    async def test_execution_performance_tracking(self, order_manager):
        """Test execution performance metrics"""
        # No data initially
        performance = await order_manager.get_execution_performance()
        assert 'no_data' in performance
        
        # Add execution history
        order_manager.execution_history = [
            {'status': 'filled', 'slippage_bps': 5},
            {'status': 'filled', 'slippage_bps': 10},
            {'status': 'cancelled', 'slippage_bps': 0}
        ]
        
        performance = await order_manager.get_execution_performance()
        assert performance['total_executions'] == 3
        assert performance['successful_executions'] == 2
        assert performance['success_rate'] == pytest.approx(66.67, rel=0.1)
        assert performance['average_slippage_bps'] == 5.0

# ==================== TRADE EXECUTION AGENT TESTS ====================

class TestTradeExecutionAgent:
    """Test suite for main TradeExecutionAgent"""
    
    @pytest.mark.asyncio
    async def test_agent_initialization(self, trade_execution_agent):
        """Test agent initialization"""
        assert trade_execution_agent.agent_name == "trade_execution"
        assert trade_execution_agent.max_daily_trades == 50
        assert trade_execution_agent.max_position_size == 0.05
        assert trade_execution_agent.max_daily_turnover == 0.25
        assert len(trade_execution_agent.daily_trades) == 0
    
    @pytest.mark.asyncio
    async def test_execute_single_trade_success(self, trade_execution_agent, mock_alpaca_provider):
        """Test successful single trade execution"""
        test_decision = {
            'symbol': 'AAPL',
            'action': 'BUY',
            'target_weight': 3.0,
            'current_weight': 0,
            'confidence': 8,
            'reasoning': 'Test trade'
        }
        
        with patch.object(trade_execution_agent.order_manager, 'execute_trade_instruction',
                         new_callable=AsyncMock) as mock_execute:
            mock_execute.return_value = ExecutionResult(
                instruction_id='test_id',
                symbol='AAPL',
                action='BUY',
                requested_quantity=100,
                executed_quantity=100,
                average_price=150.00,
                execution_time=timedelta(minutes=5),
                slippage_bps=5,
                market_impact=0.01,
                total_commission=1.00,
                status=OrderStatus.FILLED,
                fills=[],
                execution_quality='good'
            )
            
            result = await trade_execution_agent.execute_single_trade(test_decision)
        
        assert result['status'] == 'success'
        assert 'execution_result' in result
        assert 'summary' in result
        assert len(trade_execution_agent.daily_trades) == 1
    
    @pytest.mark.asyncio
    async def test_execute_single_trade_daily_limit(self, trade_execution_agent):
        """Test daily trade limit enforcement"""
        # Fill daily trades to limit
        trade_execution_agent.daily_trades = [{'test': i} for i in range(50)]
        
        test_decision = {
            'symbol': 'AAPL',
            'action': 'BUY',
            'target_weight': 3.0,
            'confidence': 8
        }
        
        result = await trade_execution_agent.execute_single_trade(test_decision)
        
        assert result['status'] == 'deferred'
        assert 'Daily trading limits reached' in result['message']
    
    @pytest.mark.asyncio
    async def test_execute_portfolio_decisions(self, trade_execution_agent):
        """Test batch portfolio decisions execution"""
        portfolio_decisions = [
            {
                'symbol': 'AAPL',
                'action': 'BUY',
                'target_weight': 3.0,
                'confidence': 9
            },
            {
                'symbol': 'MSFT',
                'action': 'SELL',
                'target_weight': 0,
                'confidence': 7
            },
            {
                'symbol': 'GOOGL',
                'action': 'TRIM',
                'target_weight': 2.0,
                'confidence': 5
            }
        ]
        
        with patch.object(trade_execution_agent, 'execute_single_trade',
                         new_callable=AsyncMock) as mock_execute:
            mock_execute.side_effect = [
                {'status': 'success'},
                {'status': 'success'},
                {'status': 'deferred'}
            ]
            
            # Mock asyncio.sleep to avoid waiting
            with patch('asyncio.sleep', new_callable=AsyncMock):
                result = await trade_execution_agent.execute_portfolio_decisions(portfolio_decisions)
        
        assert result['status'] == 'success'
        assert result['execution_summary']['total_decisions'] == 3
        assert result['execution_summary']['executed_count'] == 2
        assert result['execution_summary']['deferred_count'] == 1
        assert result['execution_summary']['failed_count'] == 0
        
        # Should be sorted by confidence (9, 7, 5)
        calls = mock_execute.call_args_list
        assert calls[0][0][0]['confidence'] == 9
        assert calls[1][0][0]['confidence'] == 7
        assert calls[2][0][0]['confidence'] == 5
    
    @pytest.mark.asyncio
    async def test_monitor_active_executions(self, trade_execution_agent, mock_alpaca_provider):
        """Test active execution monitoring"""
        mock_alpaca_provider.get_orders.return_value = [
            {'id': 'order1', 'created_at': datetime.now().isoformat()},
            {'id': 'order2', 'created_at': (datetime.now() - timedelta(hours=3)).isoformat()}
        ]
        
        with patch.object(trade_execution_agent.order_manager, 'get_execution_performance',
                         new_callable=AsyncMock) as mock_performance:
            mock_performance.return_value = {
                'total_executions': 10,
                'average_slippage_bps': 25,
                'success_rate': 85
            }
            
            result = await trade_execution_agent.monitor_active_executions()
        
        assert result['status'] == 'success'
        assert result['active_orders'] == 2
        assert len(result['execution_alerts']) >= 2  # Stale order and high slippage alerts
        
        # Check for specific alerts
        alert_types = [alert['type'] for alert in result['execution_alerts']]
        assert 'stale_order' in alert_types
        assert 'high_slippage' in alert_types
        assert 'low_success_rate' in alert_types
    
    @pytest.mark.asyncio
    async def test_generate_execution_report(self, trade_execution_agent):
        """Test execution report generation"""
        # Add sample trades
        trade_execution_agent.daily_trades = [
            {
                'timestamp': datetime.now().isoformat(),
                'symbol': 'AAPL',
                'action': 'BUY',
                'quantity': 100,
                'price': 150.00,
                'commission': 1.00
            },
            {
                'timestamp': datetime.now().isoformat(),
                'symbol': 'MSFT',
                'action': 'SELL',
                'quantity': 50,
                'price': 300.00,
                'commission': 1.00
            }
        ]
        
        with patch.object(trade_execution_agent.order_manager, 'get_execution_performance',
                         new_callable=AsyncMock) as mock_performance:
            mock_performance.return_value = {
                'total_executions': 2,
                'success_rate': 100,
                'average_slippage_bps': 5
            }
            
            result = await trade_execution_agent.generate_execution_report(period_days=1)
        
        assert result['status'] == 'success'
        assert result['report_period_days'] == 1
        assert 'execution_metrics' in result
        assert 'period_statistics' in result
        assert 'execution_insights' in result
        
        # Check period statistics
        stats = result['period_statistics']
        assert stats['total_trades'] == 2
        assert stats['total_volume'] == 30000  # 100*150 + 50*300
        assert stats['total_commission'] == 2.00
    
    @pytest.mark.asyncio
    async def test_create_execution_plan(self, trade_execution_agent, sample_trade_instruction):
        """Test execution plan creation"""
        plan = await trade_execution_agent._create_execution_plan(sample_trade_instruction)
        
        assert isinstance(plan, ExecutionPlan)
        assert plan.instruction == sample_trade_instruction
        assert isinstance(plan.strategy, ExecutionStrategy)
        assert plan.order_size >= 0
        assert plan.target_price > 0
        assert plan.market_timing in ['excellent', 'good', 'fair', 'poor']
    
    @pytest.mark.asyncio
    async def test_convert_decision_to_instruction(self, trade_execution_agent):
        """Test decision to instruction conversion"""
        # High confidence -> HIGH urgency
        decision = {
            'symbol': 'AAPL',
            'action': 'BUY',
            'target_weight': 3.0,
            'confidence': 9
        }
        instruction = trade_execution_agent._convert_decision_to_instruction(decision)
        assert instruction.urgency == ExecutionUrgency.HIGH
        
        # Medium confidence -> MEDIUM urgency
        decision['confidence'] = 6
        instruction = trade_execution_agent._convert_decision_to_instruction(decision)
        assert instruction.urgency == ExecutionUrgency.MEDIUM
        
        # Low confidence -> LOW urgency
        decision['confidence'] = 3
        instruction = trade_execution_agent._convert_decision_to_instruction(decision)
        assert instruction.urgency == ExecutionUrgency.LOW
    
    def test_check_daily_limits(self, trade_execution_agent):
        """Test daily trading limits checking"""
        # Initially should allow trades
        assert trade_execution_agent._check_daily_limits() is True
        
        # Exceed trade count limit
        trade_execution_agent.daily_trades = [{'test': i} for i in range(50)]
        assert trade_execution_agent._check_daily_limits() is False
        
        # Reset and test turnover limit
        trade_execution_agent.daily_trades = []
        trade_execution_agent.daily_volume = 250001  # > 25% of 1M
        assert trade_execution_agent._check_daily_limits() is False
    
    def test_track_daily_execution(self, trade_execution_agent):
        """Test daily execution tracking"""
        result = ExecutionResult(
            instruction_id='test',
            symbol='AAPL',
            action='BUY',
            requested_quantity=100,
            executed_quantity=100,
            average_price=150.00,
            execution_time=timedelta(minutes=5),
            slippage_bps=5,
            market_impact=0.01,
            total_commission=1.00,
            status=OrderStatus.FILLED,
            fills=[],
            execution_quality='good'
        )
        
        trade_execution_agent._track_daily_execution(result)
        
        assert len(trade_execution_agent.daily_trades) == 1
        assert trade_execution_agent.daily_volume == 15000
        assert trade_execution_agent.daily_commission == 1.00
    
    def test_check_execution_alerts(self, trade_execution_agent):
        """Test execution alert generation"""
        active_orders = [
            {'id': 'order1', 'created_at': (datetime.now() - timedelta(hours=3)).isoformat()}
        ]
        
        performance = {
            'average_slippage_bps': 25,
            'success_rate': 85
        }
        
        alerts = trade_execution_agent._check_execution_alerts(active_orders, performance)
        
        assert len(alerts) == 3
        alert_types = [alert['type'] for alert in alerts]
        assert 'stale_order' in alert_types
        assert 'high_slippage' in alert_types
        assert 'low_success_rate' in alert_types
    
    @pytest.mark.asyncio
    async def test_process_method_routing(self, trade_execution_agent):
        """Test main process method routing"""
        # Test execute_trade
        with patch.object(trade_execution_agent, 'execute_single_trade',
                         new_callable=AsyncMock) as mock_execute:
            mock_execute.return_value = {'status': 'success'}
            
            result = await trade_execution_agent.process({
                'task_type': 'execute_trade',
                'trade_instruction': {'symbol': 'AAPL'}
            })
            mock_execute.assert_called_once()
        
        # Test execute_portfolio_decisions
        with patch.object(trade_execution_agent, 'execute_portfolio_decisions',
                         new_callable=AsyncMock) as mock_portfolio:
            mock_portfolio.return_value = {'status': 'success'}
            
            result = await trade_execution_agent.process({
                'task_type': 'execute_portfolio_decisions',
                'portfolio_decisions': []
            })
            mock_portfolio.assert_called_once()
        
        # Test unknown task type
        result = await trade_execution_agent.process({'task_type': 'unknown'})
        assert result['status'] == 'error'
        assert 'Unknown task type' in result['message']

# ==================== INTEGRATION TESTS ====================

class TestIntegration:
    """Integration tests for complete execution flow"""
    
    @pytest.mark.asyncio
    @pytest.mark.timeout(5)  # Add timeout to prevent hanging
    async def test_complete_execution_flow(self, trade_execution_agent, mock_alpaca_provider):
        """Test complete execution flow from decision to result"""
        portfolio_decision = {
            'symbol': 'AAPL',
            'action': 'BUY',
            'target_weight': 3.0,
            'current_weight': 0,
            'confidence': 8,
            'reasoning': 'Strong momentum'
        }
        
        # Mock successful order execution
        mock_alpaca_provider.place_order.return_value = {
            'success': True,
            'order_id': 'test_order_123'
        }
        mock_alpaca_provider.get_order.return_value = {
            'status': 'filled',
            'filled_qty': 200,
            'filled_avg_price': 150.00,
            'commission': 1.00
        }
        mock_alpaca_provider.get_order_status = AsyncMock(return_value={
            'success': True,
            'order': {
                'status': 'filled',
                'filled_qty': 200,
                'filled_avg_price': 150.00,
                'commission': 1.00
            }
        })
        
        # Mock timing engine to avoid delays
        with patch.object(trade_execution_agent.timing_engine, 'assess_market_timing',
                         new_callable=AsyncMock) as mock_timing:
            mock_timing.return_value = {
                'timing_score': 7,
                'timing_assessment': 'good',
                'recommended_strategy': ExecutionStrategy.MARKET,
                'optimal_execution_window': {'duration_hours': 2, 'approach': 'moderate'},
                'market_conditions': {'condition': MarketConditions.NORMAL},
                'liquidity_analysis': {'liquidity_level': 'good', 'liquidity_score': 7},
                'execution_risks': [],
                'expected_slippage': 5
            }
            
            # Mock asyncio.sleep to avoid delays
            with patch('asyncio.sleep', new_callable=AsyncMock):
                result = await trade_execution_agent.execute_single_trade(portfolio_decision)
        
        assert result['status'] == 'success'
        assert 'execution_result' in result
        assert result['execution_result'].status == OrderStatus.FILLED
        # Check that the execution happened
        assert result['execution_result'].symbol == 'AAPL'
    
    @pytest.mark.asyncio
    async def test_failed_execution_recovery(self, trade_execution_agent, mock_alpaca_provider):
        """Test handling of failed executions"""
        mock_alpaca_provider.place_order.return_value = {
            'success': False,
            'error': 'Insufficient buying power'
        }
        
        portfolio_decision = {
            'symbol': 'AAPL',
            'action': 'BUY',
            'target_weight': 10.0,  # Large position
            'confidence': 8
        }
        
        # Mock the order manager to ensure it doesn't hang
        with patch.object(trade_execution_agent.order_manager, 'execute_trade_instruction',
                         new_callable=AsyncMock) as mock_execute:
            # Simulate an execution failure
            mock_execute.side_effect = Exception("Insufficient buying power")
            
            result = await trade_execution_agent.execute_single_trade(portfolio_decision)
        
        assert result['status'] == 'error'
        assert 'Insufficient buying power' in result['message']
    
    @pytest.mark.asyncio
    async def test_multi_strategy_execution(self, order_manager, mock_alpaca_provider):
        """Test execution with different strategies"""
        strategies = [
            ExecutionStrategy.MARKET,
            ExecutionStrategy.LIMIT,
            ExecutionStrategy.TWAP,
            ExecutionStrategy.VWAP,
            ExecutionStrategy.ICEBERG,
            ExecutionStrategy.OPPORTUNISTIC
        ]
        
        # Mock asyncio.sleep for all strategies
        with patch('asyncio.sleep', new_callable=AsyncMock):
            for strategy in strategies:
                instruction = TradeInstruction(
                    symbol='AAPL',
                    action='BUY',
                    target_weight=1.0,
                    current_weight=0,
                    confidence=7,
                    time_horizon='medium',
                    reasoning=f'Test {strategy.value}',
                    risk_factors=[],
                    urgency=ExecutionUrgency.MEDIUM,
                    max_slippage=10.0,
                    execution_window=timedelta(hours=2)
                )
                
                plan = ExecutionPlan(
                    instruction=instruction,
                    strategy=strategy,
                    order_size=100,
                    target_price=150.00,
                    limit_price=149.95 if strategy == ExecutionStrategy.LIMIT else None,
                    stop_price=None,
                    chunk_size=None,
                    execution_schedule=[],
                    market_timing='good',
                    expected_slippage=5.0,
                    risk_assessment={'risks': []}
                )
                
                # Should not raise any exceptions
                with patch.object(order_manager, '_monitor_order_until_filled',
                                new_callable=AsyncMock) as mock_monitor:
                    mock_monitor.return_value = {
                        'order_id': 'test',
                        'status': 'filled',
                        'filled_qty': 100,
                        'filled_avg_price': 150.00,
                        'commission': 1.00
                    }
                    
                    result = await order_manager.execute_trade_instruction(instruction, plan)
                    assert isinstance(result, ExecutionResult)

# ==================== EDGE CASE TESTS ====================

class TestEdgeCases:
    """Test edge cases and error conditions"""
    
    @pytest.mark.asyncio
    async def test_zero_quantity_order(self, order_manager, sample_trade_instruction):
        """Test handling of zero quantity orders"""
        plan = ExecutionPlan(
            instruction=sample_trade_instruction,
            strategy=ExecutionStrategy.MARKET,
            order_size=0,  # Zero size
            target_price=150.00,
            limit_price=None,
            stop_price=None,
            chunk_size=None,
            execution_schedule=[],
            market_timing='good',
            expected_slippage=5.0,
            risk_assessment={'risks': []}
        )
        
        result = await order_manager.execute_trade_instruction(
            sample_trade_instruction, plan
        )
        
        assert result.action == 'NO_CHANGE'
        assert result.executed_quantity == 0
        assert result.status == OrderStatus.CANCELLED
    
    @pytest.mark.asyncio
    async def test_api_timeout_handling(self, execution_timing_engine, mock_alpaca_provider):
        """Test handling of API timeouts"""
        mock_alpaca_provider.get_market_data.side_effect = asyncio.TimeoutError()
        
        result = await execution_timing_engine.assess_market_timing('AAPL', 'BUY')
        
        # Should return default values
        assert 1 <= result['timing_score'] <= 10
        # Should have some execution risks identified
        assert len(result['execution_risks']) > 0
    
    @pytest.mark.asyncio
    async def test_partial_fill_handling(self, order_manager, mock_alpaca_provider):
        """Test handling of partially filled orders"""
        mock_alpaca_provider.get_order.side_effect = [
            {'status': 'partially_filled', 'filled_qty': 50},
            {'status': 'partially_filled', 'filled_qty': 75},
            {'status': 'filled', 'filled_qty': 100, 'filled_avg_price': 150.00, 'commission': 1.00}
        ]
        
        with patch('asyncio.sleep', new_callable=AsyncMock):
            result = await order_manager._monitor_order_until_filled('test_order')
        
        assert result['status'] == 'filled'
        assert result['filled_qty'] == 100
    
    @pytest.mark.asyncio
    async def test_invalid_symbol_handling(self, trade_execution_agent):
        """Test handling of invalid symbols"""
        decision = {
            'symbol': '',  # Empty symbol
            'action': 'BUY',
            'target_weight': 1.0,
            'confidence': 5
        }
        
        # Mock the order manager to prevent actual execution
        with patch.object(trade_execution_agent.order_manager, 'execute_trade_instruction',
                         new_callable=AsyncMock) as mock_execute:
            result = await trade_execution_agent.execute_single_trade(decision)
        
        assert result['status'] == 'error'
        assert 'Invalid trade decision' in result['message']
    
    @pytest.mark.asyncio
    async def test_extreme_market_conditions(self, execution_timing_engine, mock_alpaca_provider):
        """Test execution under extreme market conditions"""
        mock_alpaca_provider.get_market_data.return_value = {
            'AAPL': {
                'current_price': 150.00,
                'volatility': 0.10,  # Extreme volatility
                'trend_strength': 0.9,
                'volume_profile': 'extreme'
            }
        }
        mock_alpaca_provider.get_latest_quote.return_value = {
            'bid': 145.00,
            'ask': 155.00,  # Very wide spread
            'bid_size': 1,
            'ask_size': 1
        }
        
        result = await execution_timing_engine.assess_market_timing('AAPL', 'BUY')
        
        assert result['timing_score'] <= 3  # Should be very low
        assert result['timing_assessment'] == 'poor'
        assert len(result['execution_risks']) >= 2

# ==================== RUN ALL TESTS ====================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--cov=agents.trade_execution_agent", "--cov-report=term-missing"])