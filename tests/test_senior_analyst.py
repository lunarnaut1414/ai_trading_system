# tests/test_senior_analyst.py
"""
Senior Research Analyst Test Suite
Comprehensive testing for the Senior Research Analyst Agent using pytest

Run tests:
    pytest tests/test_senior_analyst.py -v                    # All tests with verbose
    pytest tests/test_senior_analyst.py -v -m unit           # Unit tests only
    pytest tests/test_senior_analyst.py -v -m integration    # Integration tests only
    pytest tests/test_senior_analyst.py -v -m stress         # Stress tests only
    pytest tests/test_senior_analyst.py -v -k "ranking"      # Specific test by name
    pytest tests/test_senior_analyst.py --cov=agents         # With coverage report
"""

import asyncio
import json
import pytest
from datetime import datetime, timedelta
from typing import Dict, List
from unittest.mock import Mock, AsyncMock, MagicMock
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.senior_research_analyst import (
    SeniorResearchAnalyst,
    StrategicAnalysisEngine,
    MarketContextAnalyzer,
    OpportunityRanking,
    PortfolioTheme
)


# ==============================================================================
# FIXTURES
# ==============================================================================

@pytest.fixture
def mock_llm_provider():
    """Mock LLM provider for testing"""
    provider = MagicMock()
    
    async def generate_analysis_mock(prompt: str, context: Dict) -> Dict:
        await asyncio.sleep(0.01)  # Simulate API delay
        return {
            "executive_summary": "Strong opportunities identified in technology sector with favorable risk-reward profiles.",
            "key_decisions": [
                "Initiate positions in AAPL and MSFT with 4% allocation each",
                "Reduce exposure to energy sector",
                "Maintain 15% cash reserve for volatility"
            ],
            "positioning_advice": "Favor quality growth names with strong fundamentals",
            "risk_priorities": [
                "Monitor sector concentration",
                "Set tight stop losses on new positions"
            ],
            "time_horizon_strategy": "Balance 40% short-term momentum with 60% medium-term positions"
        }
    
    provider.generate_analysis = generate_analysis_mock
    return provider


@pytest.fixture
def mock_alpaca_provider():
    """Mock Alpaca provider for testing"""
    provider = MagicMock()
    
    async def get_market_data_mock(symbols: List[str], timeframe: str = '1Day', limit: int = 20) -> Dict:
        await asyncio.sleep(0.01)  # Simulate API delay
        
        mock_data = {}
        base_prices = {
            'SPY': 450, 'QQQ': 380, 'IWM': 200, 'VIX': 18,
            'XLK': 170, 'XLF': 40, 'XLV': 140, 'XLE': 80
        }
        
        for symbol in symbols:
            base_price = base_prices.get(symbol, 100)
            prices = []
            
            for i in range(limit):
                price_variation = 1 + (0.02 * (i % 5 - 2))
                prices.append({
                    'timestamp': (datetime.now() - timedelta(days=limit-i)).isoformat(),
                    'open': base_price * price_variation,
                    'high': base_price * price_variation * 1.01,
                    'low': base_price * price_variation * 0.99,
                    'close': base_price * price_variation * 1.005,
                    'volume': 1000000 * (1 + i % 3)
                })
            
            mock_data[symbol] = prices
        
        return mock_data
    
    provider.get_market_data = get_market_data_mock
    return provider


@pytest.fixture
def mock_config():
    """Mock configuration for testing"""
    config = MagicMock()
    config.MAX_POSITIONS = 10
    config.MAX_POSITION_SIZE = 0.05
    config.MAX_SECTOR_EXPOSURE = 0.25
    config.RISK_TOLERANCE = "moderate"
    config.LOG_LEVEL = "INFO"
    return config


@pytest.fixture
def sample_junior_reports():
    """Sample junior analyst reports for testing"""
    return [
        {
            'ticker': 'AAPL',
            'recommendation': 'BUY',
            'confidence': 8,
            'analysis_status': 'success',
            'target_upside_percent': 15,
            'stop_loss_percent': 8,
            'time_horizon': 'medium',
            'sector': 'Technology',
            'market_cap': 'Large',
            'risk_assessment': {'risk_level': 'medium'},
            'technical_analysis': {
                'technical_score': 7.5,
                'volume_ratio': 1.2,
                'rsi': 55,
                'macd_signal': 'bullish'
            },
            'catalysts': ['earnings_beat', 'product_launch', 'buyback_program'],
            'thesis': 'Strong fundamentals with upcoming product cycle driving growth.',
            'key_risks': ['Valuation concerns', 'China exposure']
        },
        {
            'ticker': 'MSFT',
            'recommendation': 'BUY',
            'confidence': 9,
            'analysis_status': 'success',
            'target_upside_percent': 12,
            'stop_loss_percent': 6,
            'time_horizon': 'long',
            'sector': 'Technology',
            'market_cap': 'Large',
            'risk_assessment': {'risk_level': 'low'},
            'technical_analysis': {
                'technical_score': 8.2,
                'volume_ratio': 1.5,
                'rsi': 58,
                'macd_signal': 'bullish'
            },
            'catalysts': ['cloud_growth', 'ai_expansion'],
            'thesis': 'Dominant cloud position with AI tailwinds.',
            'key_risks': ['Competition from AWS']
        },
        {
            'ticker': 'JPM',
            'recommendation': 'BUY',
            'confidence': 7,
            'analysis_status': 'success',
            'target_upside_percent': 10,
            'stop_loss_percent': 7,
            'time_horizon': 'short',
            'sector': 'Financials',
            'market_cap': 'Large',
            'risk_assessment': {'risk_level': 'medium'},
            'technical_analysis': {
                'technical_score': 6.8,
                'volume_ratio': 0.9,
                'rsi': 52,
                'macd_signal': 'neutral'
            },
            'catalysts': ['interest_rates', 'earnings'],
            'thesis': 'Benefiting from rising interest rates.',
            'key_risks': ['Credit losses']
        }
    ]


@pytest.fixture
def sample_portfolio():
    """Sample portfolio context for testing"""
    return {
        'positions': [
            {'ticker': 'SPY', 'sector': 'Index', 'value': 50000, 'weight': 0.20},
            {'ticker': 'QQQ', 'sector': 'Index', 'value': 30000, 'weight': 0.12}
        ],
        'cash': 145000,
        'total_value': 250000,
        'sectors': {
            'Technology': 0.22,
            'Index': 0.32,
            'Cash': 0.58
        }
    }


@pytest.fixture
def strategic_engine():
    """Strategic Analysis Engine instance"""
    return StrategicAnalysisEngine()


@pytest.fixture
def market_analyzer(mock_alpaca_provider):
    """Market Context Analyzer instance"""
    return MarketContextAnalyzer(mock_alpaca_provider)


@pytest.fixture
async def senior_analyst(mock_llm_provider, mock_alpaca_provider, mock_config):
    """Senior Research Analyst instance"""
    return SeniorResearchAnalyst(mock_llm_provider, mock_alpaca_provider, mock_config)


# ==============================================================================
# UNIT TESTS - Strategic Analysis Engine
# ==============================================================================

@pytest.mark.unit
class TestStrategicAnalysisEngine:
    """Test suite for Strategic Analysis Engine"""
    
    def test_engine_initialization(self, strategic_engine):
        """Test engine initialization with correct parameters"""
        assert strategic_engine is not None
        assert strategic_engine.scoring_weights is not None
        assert strategic_engine.risk_thresholds is not None
        assert sum(strategic_engine.scoring_weights.values()) == 1.0
    
    def test_opportunity_ranking(self, strategic_engine, sample_junior_reports, sample_portfolio):
        """Test opportunity ranking logic"""
        market_context = {'regime': 'risk_on', 'sector_performance': {}}
        
        result = strategic_engine.synthesize_junior_reports(
            sample_junior_reports, market_context, sample_portfolio
        )
        
        assert 'ranked_opportunities' in result
        opportunities = result['ranked_opportunities']
        assert len(opportunities) > 0
        
        # Verify ranking order (descending by score)
        for i in range(len(opportunities) - 1):
            assert opportunities[i].risk_adjusted_score >= opportunities[i+1].risk_adjusted_score
    
    def test_theme_identification(self, strategic_engine, sample_junior_reports):
        """Test strategic theme identification"""
        market_context = {'regime': 'risk_on', 'sector_performance': {}}
        portfolio_context = {}
        
        result = strategic_engine.synthesize_junior_reports(
            sample_junior_reports, market_context, portfolio_context
        )
        
        assert 'strategic_themes' in result
        themes = result['strategic_themes']
        
        # Should identify technology concentration
        tech_theme_found = any('Technology' in theme.theme_name for theme in themes)
        assert tech_theme_found
    
    def test_risk_assessment(self, strategic_engine, sample_junior_reports, sample_portfolio):
        """Test portfolio risk assessment"""
        market_context = {'regime': 'neutral', 'sector_performance': {}}
        
        result = strategic_engine.synthesize_junior_reports(
            sample_junior_reports, market_context, sample_portfolio
        )
        
        assert 'risk_assessment' in result
        risk = result['risk_assessment']
        
        assert 'risk_level' in risk
        assert 'risk_score' in risk
        assert 1 <= risk['risk_score'] <= 10
        assert risk['risk_level'] in ['low', 'medium', 'high', 'unknown']
    
    def test_time_horizon_balance(self, strategic_engine, sample_junior_reports):
        """Test time horizon balancing"""
        market_context = {'regime': 'neutral', 'sector_performance': {}}
        portfolio_context = {}
        
        result = strategic_engine.synthesize_junior_reports(
            sample_junior_reports, market_context, portfolio_context
        )
        
        assert 'time_horizon_allocation' in result
        allocation = result['time_horizon_allocation']
        
        if 'percentages' in allocation:
            total = sum(allocation['percentages'].values())
            assert 0.99 <= total <= 1.01  # Allow for rounding
    
    def test_correlation_analysis(self, strategic_engine, sample_junior_reports):
        """Test correlation analysis between opportunities"""
        market_context = {'regime': 'neutral', 'sector_performance': {}}
        portfolio_context = {}
        
        result = strategic_engine.synthesize_junior_reports(
            sample_junior_reports, market_context, portfolio_context
        )
        
        assert 'correlation_analysis' in result
        correlation = result['correlation_analysis']
        
        assert 'average_correlation' in correlation
        assert 0 <= correlation['average_correlation'] <= 1
        assert 'risk_level' in correlation
    
    def test_empty_reports_handling(self, strategic_engine):
        """Test handling of empty report list"""
        result = strategic_engine.synthesize_junior_reports([], {}, {})
        
        assert 'ranked_opportunities' in result
        assert len(result['ranked_opportunities']) == 0
        assert 'recommendation_summary' in result
    
    def test_invalid_report_filtering(self, strategic_engine):
        """Test filtering of invalid reports"""
        invalid_reports = [
            {'ticker': 'TEST1'},  # Missing required fields
            {'ticker': 'TEST2', 'recommendation': 'BUY'},  # Missing confidence
            {'ticker': 'TEST3', 'recommendation': 'BUY', 'confidence': 8, 'analysis_status': 'failed'}
        ]
        
        result = strategic_engine.synthesize_junior_reports(invalid_reports, {}, {})
        
        assert 'ranked_opportunities' in result
        assert len(result['ranked_opportunities']) == 0


# ==============================================================================
# UNIT TESTS - Market Context Analyzer
# ==============================================================================

@pytest.mark.unit
class TestMarketContextAnalyzer:
    """Test suite for Market Context Analyzer"""
    
    @pytest.mark.asyncio
    async def test_market_context_analysis(self, market_analyzer):
        """Test comprehensive market context analysis"""
        context = await market_analyzer.get_market_context()
        
        assert context is not None
        assert 'market_momentum' in context
        assert 'volatility_regime' in context
        assert 'regime_classification' in context
        assert 'positioning_recommendations' in context
        
        # Verify regime classification
        regime = context['regime_classification']
        assert 'regime' in regime
        assert 'confidence' in regime
        assert regime['regime'] in ['trending_bull', 'volatile_bull', 'sideways_market', 
                                    'correction', 'crisis']
    
    @pytest.mark.asyncio
    async def test_sector_rotation_analysis(self, market_analyzer):
        """Test sector rotation analysis"""
        context = await market_analyzer.get_market_context()
        
        assert 'sector_rotation' in context
        rotation = context['sector_rotation']
        
        if rotation.get('rotation_active'):
            assert 'leading_sectors' in rotation
            assert 'lagging_sectors' in rotation
            assert isinstance(rotation['leading_sectors'], list)
            assert isinstance(rotation['lagging_sectors'], list)
    
    @pytest.mark.asyncio
    async def test_risk_sentiment_assessment(self, market_analyzer):
        """Test risk sentiment assessment"""
        context = await market_analyzer.get_market_context()
        
        assert 'risk_sentiment' in context
        sentiment = context['risk_sentiment']
        
        assert 'sentiment' in sentiment
        assert 'risk_score' in sentiment
        assert sentiment['sentiment'] in ['risk_on', 'risk_off', 'neutral', 'extreme_fear']
        assert 1 <= sentiment['risk_score'] <= 10
    
    @pytest.mark.asyncio
    async def test_positioning_recommendations(self, market_analyzer):
        """Test positioning recommendations"""
        context = await market_analyzer.get_market_context()
        
        recommendations = context['positioning_recommendations']
        
        assert 'overall_posture' in recommendations
        assert 'cash_allocation' in recommendations
        assert 'risk_level' in recommendations
        
        assert 0 <= recommendations['cash_allocation'] <= 100
        assert recommendations['risk_level'] in ['low', 'medium-low', 'medium', 
                                                 'medium-high', 'high']


# ==============================================================================
# UNIT TESTS - Senior Research Analyst
# ==============================================================================

@pytest.mark.unit
class TestSeniorResearchAnalyst:
    """Test suite for Senior Research Analyst Agent"""
    
    @pytest.mark.asyncio
    async def test_agent_initialization(self, senior_analyst):
        """Test agent initialization"""
        analyst = await senior_analyst
        
        assert analyst.agent_name == "senior_research_analyst"
        assert analyst.agent_id is not None
        assert analyst.strategic_engine is not None
        assert analyst.market_analyzer is not None
    
    @pytest.mark.asyncio
    async def test_synthesize_reports_success(self, senior_analyst, sample_junior_reports, sample_portfolio):
        """Test successful report synthesis"""
        analyst = await senior_analyst
        
        result = await analyst.synthesize_reports(sample_junior_reports, sample_portfolio)
        
        assert result['status'] == 'success'
        assert 'strategic_analysis' in result
        assert 'markdown_report' in result
        assert 'metadata' in result
        
        # Verify analysis components
        analysis = result['strategic_analysis']
        assert 'ranked_opportunities' in analysis
        assert 'strategic_themes' in analysis
        assert 'risk_assessment' in analysis
    
    @pytest.mark.asyncio
    async def test_llm_enhancement(self, senior_analyst, sample_junior_reports):
        """Test LLM enhancement of analysis"""
        analyst = await senior_analyst
        
        result = await analyst.synthesize_reports(sample_junior_reports)
        
        analysis = result['strategic_analysis']
        
        # Should have LLM-enhanced content
        assert 'executive_summary' in analysis
        assert len(analysis['executive_summary']) > 0
        
        if 'llm_insights' in analysis:
            insights = analysis['llm_insights']
            assert 'key_decisions' in insights
            assert isinstance(insights['key_decisions'], list)
    
    @pytest.mark.asyncio
    async def test_markdown_report_generation(self, senior_analyst, sample_junior_reports):
        """Test markdown report generation"""
        analyst = await senior_analyst
        
        result = await analyst.synthesize_reports(sample_junior_reports)
        
        assert 'markdown_report' in result
        report = result['markdown_report']
        
        # Verify report structure
        assert '# Senior Research Analyst Report' in report
        assert '## Executive Summary' in report
        assert '## Top Opportunities' in report
        assert '## Strategic Themes' in report
        assert '## Risk Assessment' in report
        assert '| Ticker |' in report  # Table formatting
    
    @pytest.mark.asyncio
    async def test_error_handling_empty_reports(self, senior_analyst):
        """Test error handling with empty reports"""
        analyst = await senior_analyst
        
        result = await analyst.synthesize_reports([])
        
        assert result['status'] == 'error'
        assert 'error' in result
        assert 'No junior analyst reports' in result['error']
    
    @pytest.mark.asyncio
    async def test_performance_metrics_tracking(self, senior_analyst, sample_junior_reports):
        """Test performance metrics tracking"""
        analyst = await senior_analyst
        
        # Initial metrics
        initial_metrics = analyst.get_performance_metrics()
        assert initial_metrics['total_syntheses'] == 0
        
        # Run synthesis
        await analyst.synthesize_reports(sample_junior_reports)
        
        # Updated metrics
        updated_metrics = analyst.get_performance_metrics()
        assert updated_metrics['total_syntheses'] == 1
        assert updated_metrics['successful_syntheses'] == 1
        assert updated_metrics['average_processing_time'] > 0
    
    @pytest.mark.asyncio
    async def test_caching_behavior(self, senior_analyst, sample_junior_reports):
        """Test synthesis caching behavior"""
        analyst = await senior_analyst
        
        # First synthesis
        result1 = await analyst.synthesize_reports(sample_junior_reports)
        time1 = result1['metadata']['processing_time']
        
        # Second synthesis with same data (should be different due to market context)
        result2 = await analyst.synthesize_reports(sample_junior_reports)
        time2 = result2['metadata']['processing_time']
        
        # Both should succeed
        assert result1['status'] == 'success'
        assert result2['status'] == 'success'


# ==============================================================================
# INTEGRATION TESTS
# ==============================================================================

@pytest.mark.integration
class TestIntegration:
    """Integration tests for complete workflow"""
    
    @pytest.mark.asyncio
    async def test_full_synthesis_workflow(self, senior_analyst, sample_junior_reports, sample_portfolio):
        """Test complete synthesis workflow"""
        analyst = await senior_analyst
        
        result = await analyst.synthesize_reports(sample_junior_reports, sample_portfolio)
        
        assert result['status'] == 'success'
        
        # Verify complete output
        analysis = result['strategic_analysis']
        
        # Check all major components present
        assert len(analysis['ranked_opportunities']) > 0
        assert len(analysis['strategic_themes']) > 0
        assert analysis['risk_assessment']['risk_level'] in ['low', 'medium', 'high']
        assert 'time_horizon_allocation' in analysis
        assert 'correlation_analysis' in analysis
        assert 'executive_summary' in analysis
        
        # Verify markdown report
        report = result['markdown_report']
        assert len(report) > 500  # Substantial report
        
        # Verify metadata
        metadata = result['metadata']
        assert metadata['reports_synthesized'] == len(sample_junior_reports)
        assert metadata['processing_time'] > 0
    
    @pytest.mark.asyncio
    async def test_multiple_synthesis_consistency(self, senior_analyst, sample_junior_reports):
        """Test consistency across multiple synthesis runs"""
        analyst = await senior_analyst
        
        results = []
        for _ in range(3):
            result = await analyst.synthesize_reports(sample_junior_reports)
            results.append(result)
        
        # All should succeed
        assert all(r['status'] == 'success' for r in results)
        
        # Top opportunity should be consistent
        top_tickers = []
        for r in results:
            if r['strategic_analysis']['ranked_opportunities']:
                top_tickers.append(r['strategic_analysis']['ranked_opportunities'][0].ticker)
        
        # Most common top pick should appear multiple times
        if top_tickers:
            from collections import Counter
            most_common = Counter(top_tickers).most_common(1)[0]
            assert most_common[1] >= 2  # Should appear at least twice


# ==============================================================================
# STRESS TESTS
# ==============================================================================

@pytest.mark.stress
class TestStress:
    """Stress tests for performance validation"""
    
    @pytest.mark.asyncio
    async def test_large_report_batch(self, senior_analyst, sample_junior_reports):
        """Test with large batch of reports"""
        analyst = await senior_analyst
        
        # Create 50 reports
        large_batch = sample_junior_reports * 17  # ~51 reports
        
        start_time = datetime.now()
        result = await analyst.synthesize_reports(large_batch)
        processing_time = (datetime.now() - start_time).total_seconds()
        
        assert result['status'] == 'success'
        assert processing_time < 10  # Should complete within 10 seconds
        
        analysis = result['strategic_analysis']
        assert len(analysis['ranked_opportunities']) <= 10  # Should limit to top 10
    
    @pytest.mark.asyncio
    async def test_concurrent_synthesis(self, mock_llm_provider, mock_alpaca_provider, mock_config, sample_junior_reports):
        """Test concurrent synthesis operations"""
        
        # Create multiple analysts
        analysts = []
        for _ in range(3):
            analyst = SeniorResearchAnalyst(mock_llm_provider, mock_alpaca_provider, mock_config)
            analysts.append(analyst)
        
        # Run concurrent synthesis
        tasks = [analyst.synthesize_reports(sample_junior_reports) for analyst in analysts]
        results = await asyncio.gather(*tasks)
        
        # All should succeed
        assert all(r['status'] == 'success' for r in results)
    
    @pytest.mark.asyncio
    async def test_memory_efficiency(self, senior_analyst, sample_junior_reports):
        """Test memory efficiency with repeated operations"""
        analyst = await senior_analyst
        
        # Run multiple times to check for memory leaks
        for i in range(10):
            result = await analyst.synthesize_reports(sample_junior_reports)
            assert result['status'] == 'success'
            
            # Clear any internal caches periodically
            if i % 5 == 0:
                analyst.synthesis_cache.clear()


# ==============================================================================
# PARAMETRIZED TESTS
# ==============================================================================

@pytest.mark.parametrize("market_regime,expected_posture", [
    ("trending_bull", "aggressive"),
    ("crisis", "defensive"),
    ("correction", "cautious"),
    ("sideways_market", "neutral")
])
@pytest.mark.unit
def test_regime_positioning(strategic_engine, sample_junior_reports, market_regime, expected_posture):
    """Test positioning recommendations for different market regimes"""
    market_context = {
        'regime_classification': {'regime': market_regime},
        'sector_performance': {}
    }
    
    analyzer = MarketContextAnalyzer(Mock())
    recommendations = analyzer._get_positioning_recommendations({'regime': market_regime})
    
    assert recommendations['overall_posture'] == expected_posture


@pytest.mark.parametrize("confidence_level,expected_ranking", [
    (9, 0),  # Highest confidence should rank first
    (8, 1),
    (7, 2)
])
@pytest.mark.unit 
def test_confidence_based_ranking(strategic_engine, confidence_level, expected_ranking):
    """Test that higher confidence reports rank higher"""
    reports = [
        {'ticker': f'TEST{i}', 'recommendation': 'BUY', 'confidence': conf, 
         'analysis_status': 'success', 'target_upside_percent': 10,
         'stop_loss_percent': 5, 'sector': 'Tech', 'risk_assessment': {'risk_level': 'medium'}}
        for i, conf in enumerate([7, 8, 9])
    ]
    
    result = strategic_engine.synthesize_junior_reports(reports, {}, {})
    opportunities = result['ranked_opportunities']
    
    # Find the position of report with given confidence
    for idx, opp in enumerate(opportunities):
        if opp.ticker == f'TEST{[7, 8, 9].index(confidence_level)}':
            assert idx == expected_ranking


# ==============================================================================
# CONFTEST CONTENT (if needed separately)
# ==============================================================================

def pytest_configure(config):
    """Configure pytest with custom markers"""
    config.addinivalue_line("markers", "unit: mark test as a unit test")
    config.addinivalue_line("markers", "integration: mark test as an integration test")
    config.addinivalue_line("markers", "stress: mark test as a stress test")
    config.addinivalue_line("markers", "asyncio: mark test as async")


# ==============================================================================
# TEST RUNNER (if running directly)
# ==============================================================================

if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))