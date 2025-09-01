# tests/test_portfolio_manager.py
"""
Comprehensive test suite for Portfolio Manager Agent
Tests all components: market regime analysis, risk management, position evaluation,
opportunity analysis, and decision generation
"""

import pytest
import asyncio
import json
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from datetime import datetime, timedelta
from typing import Dict, List, Any
import numpy as np

# Import the components to test
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.agents.portfolio_manager import (
    PortfolioManagerAgent,
    MarketRegimeAnalyzer,
    PortfolioRiskManager,
    MarketRegime,
    PortfolioPosture,
    ActionType,
    TimeHorizon,
    RiskLimits,
    PortfolioMetrics
)

# ========================================================================================
# FIXTURES
# ========================================================================================

@pytest.fixture
def mock_llm_provider():
    """Create mock LLM provider"""
    mock = Mock()
    
    async def mock_generate_analysis(prompt, context):
        """Generate contextual mock responses based on prompt content"""
        
        if "position evaluation" in prompt.lower():
            return json.dumps({
                "action": "HOLD",
                "confidence": 7,
                "target_weight": 3.5,
                "reasoning": "Position showing strength with positive momentum",
                "risk_factors": ["market_volatility", "sector_rotation"],
                "price_targets": {
                    "upside_target": 165.0,
                    "downside_target": 145.0
                },
                "time_horizon": "medium",
                "conviction_level": "medium"
            })
        
        elif "opportunity analysis" in prompt.lower():
            return json.dumps({
                "actionable": True,
                "recommended_allocation": 3.0,
                "position_sizing_rationale": "High conviction with strong technical setup",
                "opportunity_score": 8,
                "risk_adjusted_allocation": 2.5,
                "timing_assessment": "good",
                "market_timing_rationale": "Breaking key resistance levels",
                "portfolio_fit": "excellent",
                "diversification_impact": "positive",
                "rejection_reason": "",
                "entry_criteria": "Confirm breakout above $150",
                "risk_considerations": ["earnings_volatility", "market_correlation"]
            })
        
        else:
            return "Portfolio decisions reflect balanced risk management approach."
    
    mock.generate_analysis = AsyncMock(side_effect=mock_generate_analysis)
    return mock

@pytest.fixture
def mock_alpaca_provider():
    """Create mock Alpaca provider"""
    mock = Mock()
    
    # Mock account data
    async def mock_get_account():
        account = Mock()
        account.portfolio_value = 100000.0
        account.cash = 30000.0
        account.equity = 100000.0
        account.last_equity = 98000.0
        return account
    
    # Mock positions data
    async def mock_get_positions():
        positions = []
        
        # Create mock position objects
        position1 = Mock()
        position1.symbol = 'AAPL'
        position1.qty = 100.0
        position1.market_value = 15000.0
        position1.cost_basis = 14000.0
        position1.unrealized_pl = 1000.0
        position1.unrealized_plpc = 0.0714
        position1.current_price = 150.0
        position1.avg_entry_price = 140.0
        
        position2 = Mock()
        position2.symbol = 'GOOGL'
        position2.qty = 50.0
        position2.market_value = 12500.0
        position2.cost_basis = 13000.0
        position2.unrealized_pl = -500.0
        position2.unrealized_plpc = -0.0385
        position2.current_price = 125.0
        position2.avg_entry_price = 130.0
        
        positions.extend([position1, position2])
        return positions
    
    # Mock bars data
    async def mock_get_bars(symbol, timeframe='1Day', limit=20):
        bars = []
        base_price = 100.0 if symbol not in ['AAPL', 'GOOGL', 'SPY'] else {
            'AAPL': 150.0,
            'GOOGL': 125.0,
            'SPY': 450.0
        }[symbol]
        
        for i in range(limit):
            bar = Mock()
            # Create trending price data
            bar.c = base_price + (i - limit/2) * 0.5 + np.random.uniform(-2, 2)
            bar.h = bar.c + np.random.uniform(0, 3)
            bar.l = bar.c - np.random.uniform(0, 3)
            bar.o = bar.c + np.random.uniform(-1, 1)
            bar.v = np.random.randint(1000000, 5000000)
            bars.append(bar)
        
        return bars
    
    # Mock news data
    async def mock_get_news(symbol, limit=5):
        news = []
        headlines = [
            f"{symbol} beats earnings expectations, raises guidance",
            f"Analysts upgrade {symbol} on strong fundamentals",
            f"{symbol} announces new product innovation",
            f"Market momentum favors {symbol} stock",
            f"{symbol} expands market share in key segment"
        ]
        
        for i in range(min(limit, len(headlines))):
            article = Mock()
            article.headline = headlines[i]
            article.summary = f"Positive news for {symbol}"
            news.append(article)
        
        return news
    
    mock.get_account = AsyncMock(side_effect=mock_get_account)
    mock.get_positions = AsyncMock(side_effect=mock_get_positions)
    mock.get_bars = AsyncMock(side_effect=mock_get_bars)
    mock.get_news = AsyncMock(side_effect=mock_get_news)
    
    return mock

@pytest.fixture
def mock_config():
    """Create mock configuration"""
    config = Mock()
    config.LOG_LEVEL = "INFO"
    config.MAX_POSITIONS = 20
    config.DEFAULT_RISK_LIMIT = 0.02
    return config

@pytest.fixture
def sample_senior_recommendations():
    """Create sample senior analyst recommendations"""
    return [
        {
            'symbol': 'MSFT',
            'action': 'BUY',
            'confidence_score': 8,
            'expected_return': 15.0,
            'time_horizon': 'medium',
            'risk_level': 'medium',
            'key_catalysts': ['AI growth', 'Cloud expansion', 'Enterprise strength']
        },
        {
            'symbol': 'NVDA',
            'action': 'BUY',
            'confidence_score': 9,
            'expected_return': 20.0,
            'time_horizon': 'long',
            'risk_level': 'high',
            'key_catalysts': ['AI chip dominance', 'Data center growth']
        },
        {
            'symbol': 'JPM',
            'action': 'BUY',
            'confidence_score': 6,
            'expected_return': 10.0,
            'time_horizon': 'medium',
            'risk_level': 'low',
            'key_catalysts': ['Interest rate environment', 'Banking strength']
        }
    ]

@pytest.fixture
def sample_economist_outlook():
    """Create sample economist outlook"""
    return {
        'economic_cycle': 'expansion',
        'growth_outlook': 'moderate_growth',
        'inflation_outlook': 'stable_inflation',
        'policy_outlook': 'neutral',
        'geopolitical_risk': 'moderate',
        'market_regime': 'risk_on',
        'dominant_themes': [
            {'theme_name': 'AI Revolution', 'impact_sectors': ['Technology']},
            {'theme_name': 'Green Transition', 'impact_sectors': ['Energy', 'Utilities']},
            {'theme_name': 'Demographic Shift', 'impact_sectors': ['Healthcare']}
        ],
        'sector_recommendations': {
            'Technology': 'overweight',
            'Financials': 'neutral',
            'Healthcare': 'overweight',
            'Energy': 'underweight',
            'Consumer': 'neutral'
        },
        'asset_allocation': {
            'equities': 70,
            'bonds': 20,
            'commodities': 5,
            'cash': 5
        },
        'risk_scenarios': [
            {
                'scenario': 'inflation_spike',
                'probability': 0.25,
                'impact': 'medium'
            }
        ],
        'confidence_score': 7.5
    }

@pytest.fixture
def portfolio_manager(mock_llm_provider, mock_alpaca_provider, mock_config):
    """Create Portfolio Manager instance"""
    manager = PortfolioManagerAgent(
        agent_name='test_portfolio_manager',
        llm_provider=mock_llm_provider,
        config=mock_config,
        alpaca_provider=mock_alpaca_provider
    )
    return manager

# ========================================================================================
# MARKET REGIME ANALYZER TESTS
# ========================================================================================

class TestMarketRegimeAnalyzer:
    """Test Market Regime Analyzer component"""
    
    @pytest.mark.asyncio
    async def test_analyze_market_regime(self, mock_alpaca_provider):
        """Test market regime analysis"""
        analyzer = MarketRegimeAnalyzer(mock_alpaca_provider)
        
        result = await analyzer.analyze_market_regime()
        
        assert 'regime' in result
        assert 'confidence' in result
        assert 'posture_recommendation' in result
        assert 'regime_characteristics' in result
        assert 'analysis_components' in result
        
        # Check regime is valid enum
        assert result['regime'] in [MarketRegime.RISK_ON, MarketRegime.RISK_OFF, 
                                   MarketRegime.NEUTRAL, MarketRegime.TRANSITION]
        
        # Check confidence is reasonable
        assert 0 <= result['confidence'] <= 100
        
        # Check posture recommendation
        posture = result['posture_recommendation']
        assert 'equity_target' in posture
        assert 'cash_target' in posture
        assert posture['equity_target'] + posture['cash_target'] == 100
    
    @pytest.mark.asyncio
    async def test_trend_analysis(self, mock_alpaca_provider):
        """Test trend analysis component"""
        analyzer = MarketRegimeAnalyzer(mock_alpaca_provider)
        
        # Get market data
        spy_data = await analyzer._get_market_data('SPY')
        
        # Analyze trend
        trend_result = analyzer._analyze_trend(spy_data)
        
        assert 'regime' in trend_result
        assert 'strength' in trend_result
        assert trend_result['regime'] in ['uptrend', 'downtrend', 'sideways']
        assert 0 <= trend_result['strength'] <= 10
    
    @pytest.mark.asyncio
    async def test_volatility_analysis(self, mock_alpaca_provider):
        """Test volatility analysis"""
        analyzer = MarketRegimeAnalyzer(mock_alpaca_provider)
        
        # Mock VIX data
        vix_data = {'current_price': 18.5}
        
        volatility_result = analyzer._analyze_volatility(vix_data)
        
        assert 'regime' in volatility_result
        assert 'level' in volatility_result
        assert volatility_result['regime'] in ['low_volatility', 'normal_volatility', 
                                              'elevated_volatility', 'high_volatility']
    
    def test_regime_determination(self, mock_alpaca_provider):
        """Test regime determination logic"""
        analyzer = MarketRegimeAnalyzer(mock_alpaca_provider)
        
        # Test risk-on scenario
        trend = {'regime': 'uptrend', 'strength': 8}
        volatility = {'regime': 'low_volatility', 'level': 12}
        momentum = {'regime': 'strong_positive', 'score': 8}
        sentiment = {'regime': 'positive', 'score': 7}
        correlation = {'regime': 'normal', 'dispersion': 'moderate'}
        
        regime = analyzer._determine_regime(trend, volatility, momentum, sentiment, correlation)
        assert regime == MarketRegime.RISK_ON
        
        # Test risk-off scenario
        trend = {'regime': 'downtrend', 'strength': 8}
        volatility = {'regime': 'high_volatility', 'level': 35}
        momentum = {'regime': 'strong_negative', 'score': 8}
        
        regime = analyzer._determine_regime(trend, volatility, momentum, sentiment, correlation)
        assert regime == MarketRegime.RISK_OFF
    
    def test_portfolio_posture_recommendation(self, mock_alpaca_provider):
        """Test portfolio posture recommendations"""
        analyzer = MarketRegimeAnalyzer(mock_alpaca_provider)
        
        # Test each regime
        regimes_and_expected = [
            (MarketRegime.RISK_ON, 85, 15),  # Aggressive
            (MarketRegime.RISK_OFF, 55, 45),  # Defensive
            (MarketRegime.NEUTRAL, 70, 30),   # Balanced
            (MarketRegime.TRANSITION, 65, 35) # Balanced-cautious
        ]
        
        for regime, expected_equity, expected_cash in regimes_and_expected:
            posture = analyzer._recommend_portfolio_posture(regime)
            
            assert posture['equity_target'] == expected_equity
            assert posture['cash_target'] == expected_cash
            assert 'style_bias' in posture
            assert 'sector_preference' in posture

# ========================================================================================
# PORTFOLIO RISK MANAGER TESTS
# ========================================================================================

class TestPortfolioRiskManager:
    """Test Portfolio Risk Manager component"""
    
    @pytest.mark.asyncio
    async def test_assess_portfolio_risk(self, mock_alpaca_provider):
        """Test comprehensive portfolio risk assessment"""
        risk_manager = PortfolioRiskManager(mock_alpaca_provider)
        
        # Get mock positions
        positions = [
            {'symbol': 'AAPL', 'market_value': 15000, 'qty': 100},
            {'symbol': 'GOOGL', 'market_value': 12500, 'qty': 50}
        ]
        
        result = await risk_manager.assess_portfolio_risk(positions)
        
        assert 'risk_level' in result
        assert 'overall_risk_score' in result
        assert 'portfolio_metrics' in result
        assert 'limit_violations' in result
        assert 'position_risks' in result
        assert 'concentration_risk' in result
        assert 'correlation_risk' in result
        assert 'recommendations' in result
        
        # Check risk level
        assert result['risk_level'] in ['low', 'medium', 'high', 'critical']
        
        # Check risk score
        assert 0 <= result['overall_risk_score'] <= 100
    
    @pytest.mark.asyncio
    async def test_portfolio_metrics_calculation(self, mock_alpaca_provider):
        """Test portfolio metrics calculation"""
        risk_manager = PortfolioRiskManager(mock_alpaca_provider)
        
        positions = [
            {'symbol': 'AAPL', 'market_value': 15000, 'qty': 100}
        ]
        
        metrics = await risk_manager._calculate_portfolio_metrics(positions)
        
        assert 'total_value' in metrics
        assert 'cash' in metrics
        assert 'cash_percentage' in metrics
        assert 'equity_percentage' in metrics
        assert 'day_pnl' in metrics
        assert 'day_pnl_pct' in metrics
        assert 'position_count' in metrics
        assert 'value_at_risk_1d' in metrics
        
        # Check percentages add up
        assert abs(metrics['cash_percentage'] + metrics['equity_percentage'] - 100) < 0.01
    
    def test_var_calculation(self, mock_alpaca_provider):
        """Test Value at Risk calculation"""
        risk_manager = PortfolioRiskManager(mock_alpaca_provider)
        
        positions = [
            {'symbol': 'AAPL', 'market_value': 50000},
            {'symbol': 'GOOGL', 'market_value': 30000}
        ]
        
        var_1d = risk_manager._calculate_var(positions, 1)
        var_5d = risk_manager._calculate_var(positions, 5)
        
        assert var_1d > 0
        assert var_5d > var_1d  # 5-day VaR should be higher
        assert var_5d < var_1d * 5  # But not linearly higher
    
    def test_risk_violations_detection(self, mock_alpaca_provider):
        """Test risk limit violation detection"""
        risk_manager = PortfolioRiskManager(mock_alpaca_provider)
        
        # Create scenario with violations
        risk_metrics = {
            'total_value': 100000,
            'cash_percentage': 5,  # Below minimum
            'value_at_risk_1d': 3.5  # Above limit
        }
        
        positions = [
            {'symbol': 'AAPL', 'market_value': 60000}  # 60% - too concentrated
        ]
        
        violations = risk_manager._check_risk_violations(risk_metrics, positions)
        
        assert len(violations) > 0
        
        # Check for specific violations
        violation_types = [v['type'] for v in violations]
        assert 'position_size' in violation_types
        assert 'cash_reserve' in violation_types
        assert 'value_at_risk' in violation_types
    
    def test_concentration_risk_assessment(self, mock_alpaca_provider):
        """Test concentration risk calculation"""
        risk_manager = PortfolioRiskManager(mock_alpaca_provider)
        
        # High concentration scenario
        positions = [
            {'symbol': 'AAPL', 'market_value': 80000},
            {'symbol': 'GOOGL', 'market_value': 20000}
        ]
        
        concentration = risk_manager._assess_concentration_risk(positions)
        
        assert concentration['risk_level'] in ['low', 'medium', 'high', 'critical']
        assert 'concentration_score' in concentration
        assert 'hhi' in concentration
        assert concentration['risk_level'] in ['high', 'critical']  # Should be high
    
    def test_sector_exposure_calculation(self, mock_alpaca_provider):
        """Test sector exposure calculation"""
        risk_manager = PortfolioRiskManager(mock_alpaca_provider)
        
        positions = [
            {'symbol': 'AAPL', 'market_value': 30000},
            {'symbol': 'MSFT', 'market_value': 25000},
            {'symbol': 'JPM', 'market_value': 20000},
            {'symbol': 'BAC', 'market_value': 15000}
        ]
        
        sector_exposure = risk_manager._calculate_sector_exposure(positions)
        
        assert 'Technology' in sector_exposure
        assert 'Financials' in sector_exposure
        
        # Check percentages
        total_exposure = sum(sector_exposure.values())
        assert abs(total_exposure - 100) < 0.01  # Should sum to 100%
        
        # Technology should be highest
        assert sector_exposure['Technology'] > sector_exposure['Financials']

# ========================================================================================
# PORTFOLIO MANAGER AGENT TESTS
# ========================================================================================

class TestPortfolioManagerAgent:
    """Test main Portfolio Manager Agent"""
    
    def test_initialization(self, portfolio_manager):
        """Test agent initialization"""
        assert portfolio_manager.agent_name == 'test_portfolio_manager'
        assert portfolio_manager.risk_limits is not None
        assert portfolio_manager.market_analyzer is not None
        assert portfolio_manager.risk_manager is not None
        assert portfolio_manager.total_decisions == 0
        assert portfolio_manager.successful_decisions == 0
    
    @pytest.mark.asyncio
    async def test_daily_portfolio_review(self, portfolio_manager, 
                                         sample_senior_recommendations,
                                         sample_economist_outlook):
        """Test daily portfolio review process"""
        
        result = await portfolio_manager.daily_portfolio_review(
            senior_recommendations=sample_senior_recommendations,
            economist_outlook=sample_economist_outlook
        )
        
        assert result['status'] == 'success'
        assert 'portfolio_decisions' in result
        assert 'allocation_targets' in result
        assert 'market_regime' in result
        assert 'risk_assessment' in result
        assert 'decision_reasoning' in result
        assert 'metrics' in result
        
        # Check decisions
        decisions = result['portfolio_decisions']
        assert isinstance(decisions, list)
        
        for decision in decisions:
            assert 'symbol' in decision
            assert 'action' in decision
            assert 'target_weight' in decision
            assert 'confidence' in decision
            assert 'reasoning' in decision
        
        # Check metrics
        metrics = result['metrics']
        assert 'total_decisions' in metrics
        assert metrics['total_decisions'] == len(decisions)
    
    @pytest.mark.asyncio
    async def test_evaluate_existing_positions(self, portfolio_manager):
        """Test evaluation of existing positions"""
        
        positions = await portfolio_manager._get_current_positions()
        market_conditions = {'regime': MarketRegime.NEUTRAL}
        risk_assessment = {'risk_level': 'medium'}
        
        evaluations = await portfolio_manager._evaluate_existing_positions(
            positions, market_conditions, risk_assessment
        )
        
        assert len(evaluations) == len(positions)
        
        for eval in evaluations:
            assert 'symbol' in eval
            assert 'action' in eval
            assert eval['action'] in ['HOLD', 'TRIM', 'ADD', 'CLOSE']
            assert 'confidence' in eval
            assert 1 <= eval['confidence'] <= 10
            assert 'reasoning' in eval
    
    @pytest.mark.asyncio
    async def test_analyze_new_opportunities(self, portfolio_manager,
                                            sample_senior_recommendations,
                                            sample_economist_outlook):
        """Test new opportunity analysis"""
        
        market_conditions = {'regime': MarketRegime.RISK_ON}
        risk_assessment = {
            'risk_level': 'low',
            'portfolio_metrics': {'cash_percentage': 30}
        }
        
        opportunities = await portfolio_manager._analyze_new_opportunities(
            sample_senior_recommendations,
            sample_economist_outlook,
            market_conditions,
            risk_assessment
        )
        
        assert len(opportunities) > 0
        
        for opp in opportunities:
            assert 'symbol' in opp
            assert 'actionable' in opp
            
            if opp['actionable']:
                assert 'recommended_allocation' in opp
                assert 'opportunity_score' in opp
                assert 'timing_assessment' in opp
                assert 'portfolio_fit' in opp
            else:
                assert 'rejection_reason' in opp
    
    @pytest.mark.asyncio
    async def test_position_sizing_calculation(self, portfolio_manager):
        """Test Kelly Criterion position sizing"""
        
        recommendation = {
            'symbol': 'MSFT',
            'expected_return': 15.0,
            'confidence_score': 8,
            'risk_level': 'medium'
        }
        
        portfolio_context = {
            'total_value': 100000,
            'cash': 30000
        }
        
        result = await portfolio_manager.calculate_position_sizing(
            'MSFT', recommendation, portfolio_context
        )
        
        assert 'symbol' in result
        assert 'recommended_shares' in result
        assert 'position_value' in result
        assert 'portfolio_percentage' in result
        assert 'kelly_fraction' in result
        assert 'sizing_factors' in result
        
        # Check constraints
        assert result['portfolio_percentage'] <= portfolio_manager.risk_limits.max_single_position
        assert result['kelly_fraction'] >= 0
    
    @pytest.mark.asyncio
    async def test_generate_portfolio_decisions(self, portfolio_manager):
        """Test portfolio decision generation"""
        
        position_evaluations = [
            {
                'symbol': 'AAPL',
                'action': 'TRIM',
                'confidence': 7,
                'target_weight': 3.0,
                'current_weight': 5.0,
                'reasoning': 'Taking profits'
            }
        ]
        
        opportunity_analysis = [
            {
                'symbol': 'MSFT',
                'actionable': True,
                'recommended_allocation': 4.0,
                'opportunity_score': 8
            }
        ]
        
        market_conditions = {'regime': MarketRegime.NEUTRAL}
        risk_assessment = {'risk_level': 'medium'}
        economist_outlook = {'economic_cycle': 'expansion'}
        
        decisions = await portfolio_manager._generate_portfolio_decisions(
            position_evaluations,
            opportunity_analysis,
            market_conditions,
            risk_assessment,
            economist_outlook
        )
        
        assert len(decisions) > 0
        
        # Check decision structure
        for decision in decisions:
            assert 'symbol' in decision
            assert 'action' in decision
            assert 'type' in decision
            assert decision['type'] in ['existing_position', 'new_position']
            assert 'target_weight' in decision
            assert 'priority' in decision
    
    def test_validate_decisions_against_risk(self, portfolio_manager):
        """Test risk validation of decisions"""
        
        decisions = [
            {
                'symbol': 'AAPL',
                'action': 'BUY',
                'type': 'new_position',
                'target_weight': 8.0,
                'reasoning': 'Strong opportunity'
            }
        ]
        
        # High risk scenario
        risk_assessment = {
            'risk_level': 'high',
            'limit_violations': [
                {'type': 'position_size', 'symbol': 'GOOGL'}
            ]
        }
        
        validated = portfolio_manager._validate_decisions_against_risk(
            decisions, risk_assessment
        )
        
        # Should reduce allocation due to high risk
        assert validated[0]['target_weight'] < 8.0
        assert 'risk' in validated[0]['reasoning'].lower()
    
    def test_calculate_allocation_targets(self, portfolio_manager,
                                         sample_economist_outlook):
        """Test allocation target calculation"""
        
        market_conditions = {
            'regime': MarketRegime.RISK_ON,
            'posture_recommendation': {
                'equity_target': 85,
                'cash_target': 15,
                'style_bias': 'growth'
            }
        }
        
        targets = portfolio_manager._calculate_allocation_targets(
            market_conditions,
            sample_economist_outlook
        )
        
        assert 'cash_target' in targets
        assert 'equity_target' in targets
        assert 'style_bias' in targets
        assert 'economic_cycle' in targets
        
        # Check allocations sum to 100
        assert targets['cash_target'] + targets['equity_target'] == 100
        
        # Check economic cycle adjustment
        assert targets['economic_cycle'] == 'expansion'
    
    @pytest.mark.asyncio
    async def test_get_symbol_data(self, portfolio_manager):
        """Test symbol data retrieval"""
        
        symbol_data = await portfolio_manager._get_symbol_data('AAPL')
        
        assert 'symbol' in symbol_data
        assert 'current_price' in symbol_data
        assert 'price_metrics' in symbol_data
        assert 'volume_metrics' in symbol_data
        assert 'technical_indicators' in symbol_data
        
        # Check price metrics
        price_metrics = symbol_data['price_metrics']
        assert 'price_change_1d' in price_metrics
        assert 'price_change_5d' in price_metrics
        assert 'price_change_20d' in price_metrics
        
        # Check technical indicators
        tech = symbol_data['technical_indicators']
        assert 'sma_20' in tech
        assert 'rsi' in tech
    
    def test_calculate_rsi(self, portfolio_manager):
        """Test RSI calculation"""
        
        # Test with trending prices
        prices = [100 + i for i in range(15)]  # Uptrend
        rsi = portfolio_manager._calculate_rsi(prices)
        assert rsi > 70  # Should indicate overbought
        
        # Test with declining prices
        prices = [100 - i for i in range(15)]  # Downtrend
        rsi = portfolio_manager._calculate_rsi(prices)
        assert rsi < 30  # Should indicate oversold
        
        # Test with flat prices
        prices = [100] * 15
        rsi = portfolio_manager._calculate_rsi(prices)
        assert rsi == 50  # Should be exactly neutral for flat prices
    
    @pytest.mark.asyncio
    async def test_get_symbol_news_sentiment(self, portfolio_manager):
        """Test news sentiment analysis"""
        
        sentiment = await portfolio_manager._get_symbol_news_sentiment('AAPL')
        
        assert 'sentiment_score' in sentiment
        assert 'article_count' in sentiment
        assert 'key_themes' in sentiment
        
        # Since we have positive mock news
        assert sentiment['sentiment_score'] > 0
        assert sentiment['article_count'] > 0

# ========================================================================================
# INTEGRATION TESTS
# ========================================================================================

class TestIntegration:
    """Integration tests for Portfolio Manager"""
    
    @pytest.mark.asyncio
    async def test_full_portfolio_review_workflow(self, portfolio_manager,
                                                 sample_senior_recommendations,
                                                 sample_economist_outlook):
        """Test complete portfolio review workflow"""
        
        # Run full review
        result = await portfolio_manager.daily_portfolio_review(
            senior_recommendations=sample_senior_recommendations,
            economist_outlook=sample_economist_outlook,
            market_conditions=None  # Will analyze internally
        )
        
        assert result['status'] == 'success'
        
        # Verify all components worked together
        assert len(result['portfolio_decisions']) > 0
        assert result['market_regime'] in ['risk_on', 'risk_off', 'neutral', 'transition']
        assert 'cash_target' in result['allocation_targets']
        
        # Check decision quality
        for decision in result['portfolio_decisions']:
            assert decision['target_weight'] > 0
            assert decision['target_weight'] <= 5.0  # Max position size
    
    @pytest.mark.asyncio
    async def test_risk_constrained_decisions(self, portfolio_manager,
                                             sample_senior_recommendations):
        """Test decision making under high risk conditions"""
        
        # Mock high risk scenario
        with patch.object(portfolio_manager.risk_manager, 'assess_portfolio_risk') as mock_risk:
            mock_risk.return_value = {
                'risk_level': 'critical',
                'overall_risk_score': 85,
                'portfolio_metrics': {'cash_percentage': 5},
                'limit_violations': [
                    {'type': 'value_at_risk', 'severity': 'high'}
                ],
                'can_add_risk': False,
                'should_reduce_risk': True,
                'recommendations': ['Reduce portfolio risk immediately']
            }
            
            result = await portfolio_manager.daily_portfolio_review(
                senior_recommendations=sample_senior_recommendations,
                economist_outlook={'economic_cycle': 'contraction'}
            )
            
            # Should be conservative
            decisions = result['portfolio_decisions']
            new_positions = [d for d in decisions if d['type'] == 'new_position']
            
            # Should limit new positions under critical risk
            for pos in new_positions:
                assert pos['target_weight'] < 3.0  # Reduced sizing
    
    @pytest.mark.asyncio
    async def test_error_handling(self, portfolio_manager):
        """Test error handling in various scenarios"""
        
        # Test with invalid task type
        result = await portfolio_manager.process({'task_type': 'invalid'})
        assert 'error' in result
        
        # Test with missing data
        result = await portfolio_manager.daily_portfolio_review(
            senior_recommendations=[],
            economist_outlook={}
        )
        
        # Should still return valid structure
        assert result['status'] in ['success', 'error']
        if result['status'] == 'success':
            assert 'portfolio_decisions' in result

# ========================================================================================
# PERFORMANCE TESTS
# ========================================================================================

class TestPerformance:
    """Performance and stress tests"""
    
    @pytest.mark.asyncio
    async def test_large_portfolio_handling(self, mock_llm_provider, 
                                           mock_alpaca_provider, 
                                           mock_config):
        """Test handling of large portfolio"""
        
        # Create many positions
        large_positions = []
        for i in range(50):
            pos = Mock()
            pos.symbol = f'STOCK{i}'
            pos.qty = 100
            pos.market_value = 10000
            pos.cost_basis = 9500
            pos.unrealized_pl = 500
            pos.unrealized_plpc = 0.05
            pos.current_price = 100
            pos.avg_entry_price = 95
            large_positions.append(pos)
        
        mock_alpaca_provider.get_positions = AsyncMock(return_value=large_positions)
        
        manager = PortfolioManagerAgent(
            'test_large',
            mock_llm_provider,
            mock_config,
            mock_alpaca_provider
        )
        
        # Should handle without issues
        positions = await manager._get_current_positions()
        assert len(positions) == 50
    
    @pytest.mark.asyncio
    async def test_concurrent_analysis(self, portfolio_manager):
        """Test concurrent analysis of multiple symbols"""
        
        symbols = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA']
        
        # Run concurrent analysis
        tasks = [portfolio_manager._get_symbol_data(symbol) for symbol in symbols]
        results = await asyncio.gather(*tasks)
        
        assert len(results) == len(symbols)
        for result in results:
            assert 'current_price' in result
    
    def test_decision_priority_sorting(self, portfolio_manager):
        """Test decision prioritization with many decisions"""
        
        decisions = []
        for i in range(20):
            decisions.append({
                'symbol': f'STOCK{i}',
                'action': 'BUY' if i % 2 == 0 else 'SELL',
                'type': 'new_position',
                'target_weight': 2.0,
                'confidence': i % 10,
                'priority': i % 10,
                'reasoning': 'Test'
            })
        
        # Apply constraints
        constrained = portfolio_manager._apply_portfolio_constraints(
            decisions,
            {'portfolio_metrics': {'cash_percentage': 20}}
        )
        
        # Verify constraints were applied
        assert len(constrained) == len(decisions)
        
        # Check that new position allocations were scaled if needed
        total_new_allocation = sum(d['target_weight'] for d in constrained 
                                  if d['type'] == 'new_position')
        
        # Should not exceed available cash buffer (80% of 20% = 16%)
        assert total_new_allocation <= 16.1  # Small tolerance for rounding

# ========================================================================================
# MOCK DATA GENERATION
# ========================================================================================

def generate_mock_market_data(symbol: str, days: int = 20) -> List[Dict]:
    """Generate realistic mock market data"""
    data = []
    base_price = 100.0
    
    for i in range(days):
        daily_return = np.random.normal(0.001, 0.02)  # 0.1% mean, 2% std
        base_price *= (1 + daily_return)
        
        data.append({
            'date': datetime.now() - timedelta(days=days-i),
            'close': base_price,
            'open': base_price * (1 + np.random.uniform(-0.01, 0.01)),
            'high': base_price * (1 + np.random.uniform(0, 0.02)),
            'low': base_price * (1 - np.random.uniform(0, 0.02)),
            'volume': np.random.randint(1000000, 10000000)
        })
    
    return data

def generate_mock_positions(count: int = 10) -> List[Dict]:
    """Generate mock portfolio positions"""
    positions = []
    symbols = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA', 
               'META', 'NVDA', 'JPM', 'BAC', 'JNJ']
    
    for i in range(min(count, len(symbols))):
        cost_basis = np.random.uniform(50, 200)
        current_price = cost_basis * (1 + np.random.uniform(-0.3, 0.5))
        qty = np.random.randint(10, 200)
        
        positions.append({
            'symbol': symbols[i],
            'qty': qty,
            'cost_basis': cost_basis * qty,
            'market_value': current_price * qty,
            'current_price': current_price,
            'avg_entry_price': cost_basis,
            'unrealized_pl': (current_price - cost_basis) * qty,
            'unrealized_plpc': (current_price - cost_basis) / cost_basis
        })
    
    return positions

# ========================================================================================
# RUN CONFIGURATION
# ========================================================================================

if __name__ == "__main__":
    # Run tests with coverage
    pytest.main([
        __file__,
        '-v',
        '--cov=agents.portfolio_manager',
        '--cov-report=term-missing',
        '--cov-report=html',
        '-x',  # Stop on first failure
        '--tb=short'  # Short traceback format
    ])