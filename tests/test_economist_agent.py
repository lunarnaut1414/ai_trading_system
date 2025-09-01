# tests/test_economist_agent.py
"""
Comprehensive test suite for the Economist Agent
Tests all major functionality including economic analysis, theme identification,
cross-asset analysis, and macro outlook generation
"""

import pytest
import asyncio
import json
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from datetime import datetime, timedelta
from typing import Dict, List

# Import the economist agent and related classes
from src.agents.economist import (
    EconomistAgent,
    EconomicDataAnalyzer,
    MacroThemeIdentifier,
    CrossAssetAnalyzer,
    EconomicIndicator,
    MacroTheme,
    MacroOutlook,
    EconomicCycle,
    PolicyStance,
    GeopoliticalRisk,
    MarketRegime,
    AllocationStrategy
)

# Import shared components
from src.agents.junior_analyst import (
    MarketContextManager,
    IntelligentCacheManager,
    AnalysisMetadataTracker
)


# ==============================================================================
# FIXTURES
# ==============================================================================

@pytest.fixture
def mock_llm_provider():
    """Mock LLM provider for testing"""
    provider = Mock()
    provider.generate_analysis = AsyncMock(return_value={
        'summary': 'Test economic outlook suggests cautious optimism',
        'recommendations': [
            'Increase equity allocation',
            'Add inflation hedges',
            'Focus on quality growth'
        ],
        'risks': ['Recession risk', 'Policy uncertainty'],
        'opportunities': ['AI revolution', 'Green transition']
    })
    return provider


@pytest.fixture
def mock_alpaca_provider():
    """Mock Alpaca provider for testing"""
    provider = Mock()
    
    # Mock market data methods
    provider.get_latest_quote = AsyncMock(return_value={
        'symbol': 'SPY',
        'price': 450.0,
        'volume': 1000000
    })
    
    provider.get_market_status = AsyncMock(return_value='open')
    
    provider.get_historical_data = AsyncMock(return_value=[
        {'close': 445.0, 'volume': 900000},
        {'close': 448.0, 'volume': 950000},
        {'close': 450.0, 'volume': 1000000}
    ])
    
    # Add missing methods for MarketContextManager
    provider.get_market_indicators = AsyncMock(return_value={
        'vix': 15.5,
        'dxy': 103.2,
        'breadth': 0.65
    })
    
    provider.get_sector_performance = AsyncMock(return_value={
        'XLK': 5.0,
        'XLF': 3.0,
        'XLE': -2.0
    })
    
    return provider


@pytest.fixture
def mock_config():
    """Mock configuration for testing"""
    config = Mock()
    config.cache_duration = 60
    config.max_retries = 3
    config.timeout = 30
    return config


@pytest.fixture
async def economist_agent(mock_llm_provider, mock_alpaca_provider, mock_config):
    """Create economist agent instance for testing"""
    agent = EconomistAgent(
        agent_name='test_economist',
        llm_provider=mock_llm_provider,
        config=mock_config,
        alpaca_provider=mock_alpaca_provider
    )
    return agent


@pytest.fixture
def economic_data_analyzer(mock_alpaca_provider):
    """Create economic data analyzer for testing"""
    return EconomicDataAnalyzer(mock_alpaca_provider)


@pytest.fixture
def theme_identifier():
    """Create theme identifier for testing"""
    return MacroThemeIdentifier()


@pytest.fixture
def cross_asset_analyzer(mock_alpaca_provider):
    """Create cross-asset analyzer for testing"""
    return CrossAssetAnalyzer(mock_alpaca_provider)


@pytest.fixture
def sample_economic_data():
    """Sample economic data for testing"""
    return {
        'indicators': {
            'gdp': {
                'current_growth': 2.1,
                'previous_growth': 2.3,
                'trend': 'slowing',
                'forecast': 1.8
            },
            'inflation': {
                'cpi': 3.2,
                'core_cpi': 2.8,
                'pce': 2.9,
                'trend': 'moderating'
            },
            'employment': {
                'unemployment_rate': 3.8,
                'job_growth': 187000,
                'wage_growth': 4.1,
                'trend': 'softening'
            },
            'monetary_policy': {
                'fed_funds_rate': 5.25,
                'stance': 'hawkish',
                'next_move': 'pause'
            },
            'yield_curve': {
                '2y_yield': 4.85,
                '10y_yield': 4.25,
                'spread': -0.60,
                'shape': 'inverted'
            }
        },
        'health_score': 5.5,
        'trend': 'deteriorating',
        'risks': ['inverted_yield_curve', 'growth_slowdown'],
        'opportunities': ['disinflation_beneficiaries']
    }


@pytest.fixture
def sample_market_context():
    """Sample market context for testing"""
    return {
        'regime': 'risk_off',
        'volatility': {'vix': 22.5, 'regime': 'elevated'},
        'breadth': 0.45,
        'momentum': {'trend': 'weakening'},
        'sector_rotation': {
            'leading_sectors': ['XLP', 'XLU', 'XLV'],
            'lagging_sectors': ['XLK', 'XLY', 'XLF']
        }
    }


# ==============================================================================
# UNIT TESTS - Economic Data Analyzer
# ==============================================================================

@pytest.mark.unit
class TestEconomicDataAnalyzer:
    """Test suite for Economic Data Analyzer"""
    
    @pytest.mark.asyncio
    async def test_analyze_economic_indicators(self, economic_data_analyzer):
        """Test economic indicator analysis"""
        result = await economic_data_analyzer.analyze_economic_indicators()
        
        assert 'indicators' in result
        assert 'health_score' in result
        assert 'trend' in result
        assert 'risks' in result
        assert 'opportunities' in result
        
        # Check health score range
        assert 0 <= result['health_score'] <= 10
        
        # Check trend values
        assert result['trend'] in ['improving', 'deteriorating', 'mixed', 'neutral']
    
    @pytest.mark.asyncio
    async def test_gdp_analysis(self, economic_data_analyzer):
        """Test GDP analysis component"""
        gdp_data = await economic_data_analyzer._analyze_gdp()
        
        assert 'current_growth' in gdp_data
        assert 'previous_growth' in gdp_data
        assert 'trend' in gdp_data
        assert 'forecast' in gdp_data
        assert 'components' in gdp_data
    
    @pytest.mark.asyncio
    async def test_inflation_analysis(self, economic_data_analyzer):
        """Test inflation analysis component"""
        inflation_data = await economic_data_analyzer._analyze_inflation()
        
        assert 'cpi' in inflation_data
        assert 'core_cpi' in inflation_data
        assert 'pce' in inflation_data
        assert 'trend' in inflation_data
        assert 'expectations' in inflation_data
    
    @pytest.mark.asyncio
    async def test_employment_analysis(self, economic_data_analyzer):
        """Test employment analysis component"""
        employment_data = await economic_data_analyzer._analyze_employment()
        
        assert 'unemployment_rate' in employment_data
        assert 'job_growth' in employment_data
        assert 'wage_growth' in employment_data
        assert 'participation_rate' in employment_data
        assert 'trend' in employment_data
        assert 'sectors' in employment_data
    
    @pytest.mark.asyncio
    async def test_yield_curve_analysis(self, economic_data_analyzer):
        """Test yield curve analysis"""
        yield_data = await economic_data_analyzer._analyze_yield_curve()
        
        assert '2y_yield' in yield_data
        assert '10y_yield' in yield_data
        assert 'spread' in yield_data
        assert 'shape' in yield_data
        assert 'recession_signal' in yield_data
    
    def test_economic_health_score_calculation(self, economic_data_analyzer, sample_economic_data):
        """Test economic health score calculation"""
        score = economic_data_analyzer._calculate_economic_health_score(
            sample_economic_data['indicators']
        )
        
        assert isinstance(score, float)
        assert 0 <= score <= 10
    
    def test_economic_trend_determination(self, economic_data_analyzer, sample_economic_data):
        """Test economic trend determination"""
        trend = economic_data_analyzer._determine_economic_trend(
            sample_economic_data['indicators']
        )
        
        assert trend in ['improving', 'deteriorating', 'mixed']
    
    def test_risk_identification(self, economic_data_analyzer, sample_economic_data):
        """Test economic risk identification"""
        risks = economic_data_analyzer._identify_economic_risks(
            sample_economic_data['indicators']
        )
        
        assert isinstance(risks, list)
        # Should identify inverted yield curve risk
        assert 'inverted_yield_curve_recession_risk' in risks
    
    @pytest.mark.asyncio
    async def test_error_handling(self, economic_data_analyzer):
        """Test error handling in economic analysis"""
        # Mock a failure
        with patch.object(economic_data_analyzer, '_analyze_gdp', 
                         side_effect=Exception("API Error")):
            result = await economic_data_analyzer.analyze_economic_indicators()
            
            # Should return default indicators
            assert result['health_score'] == 5.0
            assert result['trend'] == 'neutral'


# ==============================================================================
# UNIT TESTS - Macro Theme Identifier
# ==============================================================================

@pytest.mark.unit
class TestMacroThemeIdentifier:
    """Test suite for Macro Theme Identifier"""
    
    def test_identify_macro_themes(self, theme_identifier, sample_economic_data, sample_market_context):
        """Test macro theme identification"""
        themes = theme_identifier.identify_macro_themes(
            sample_economic_data, 
            sample_market_context
        )
        
        assert isinstance(themes, list)
        assert len(themes) <= 5  # Should return top 5 themes
        
        if themes:
            theme = themes[0]
            assert isinstance(theme, MacroTheme)
            assert hasattr(theme, 'theme_name')
            assert hasattr(theme, 'confidence')
            assert 0 <= theme.confidence <= 10
    
    def test_stagflation_detection(self, theme_identifier):
        """Test stagflation theme detection"""
        data = {
            'indicators': {
                'gdp': {'current_growth': 1.0},
                'inflation': {'cpi': 5.0}
            }
        }
        
        assert theme_identifier._check_stagflation(data) == True
        
        # Test non-stagflation
        data['indicators']['gdp']['current_growth'] = 3.0
        assert theme_identifier._check_stagflation(data) == False
    
    def test_disinflation_detection(self, theme_identifier):
        """Test disinflation theme detection"""
        data = {
            'indicators': {
                'inflation': {
                    'trend': 'moderating',
                    'cpi': 3.0
                }
            }
        }
        
        assert theme_identifier._check_disinflation(data) == True
        
        # Test non-disinflation
        data['indicators']['inflation']['trend'] = 'accelerating'
        assert theme_identifier._check_disinflation(data) == False
    
    def test_recession_detection(self, theme_identifier):
        """Test recession theme detection"""
        data = {
            'indicators': {
                'yield_curve': {'shape': 'inverted'},
                'employment': {'trend': 'softening'}
            }
        }
        
        assert theme_identifier._check_recession(data) == True
    
    def test_theme_creation(self, theme_identifier):
        """Test theme creation methods"""
        stagflation_theme = theme_identifier._create_stagflation_theme()
        
        assert stagflation_theme.theme_name == "Stagflation Risk"
        assert len(stagflation_theme.beneficiaries) > 0
        assert len(stagflation_theme.victims) > 0
        assert len(stagflation_theme.action_items) > 0
        assert stagflation_theme.confidence > 0


# ==============================================================================
# UNIT TESTS - Cross Asset Analyzer
# ==============================================================================

@pytest.mark.unit
class TestCrossAssetAnalyzer:
    """Test suite for Cross Asset Analyzer"""
    
    @pytest.mark.asyncio
    async def test_analyze_cross_asset_dynamics(self, cross_asset_analyzer):
        """Test cross-asset dynamics analysis"""
        result = await cross_asset_analyzer.analyze_cross_asset_dynamics()
        
        assert 'correlations' in result
        assert 'divergences' in result
        assert 'allocation_signals' in result
        assert 'risk_on_score' in result
        assert 'sector_rotation' in result
        
        # Check risk-on score range
        assert 0 <= result['risk_on_score'] <= 10
    
    @pytest.mark.asyncio
    async def test_equity_market_data(self, cross_asset_analyzer):
        """Test equity market data retrieval"""
        equity_data = await cross_asset_analyzer._get_equity_market_data()
        
        assert 'spy_price' in equity_data
        assert 'vix' in equity_data
        assert 'breadth' in equity_data
        assert 'sector_performance' in equity_data
    
    @pytest.mark.asyncio
    async def test_bond_market_data(self, cross_asset_analyzer):
        """Test bond market data retrieval"""
        bond_data = await cross_asset_analyzer._get_bond_market_data()
        
        assert '10y_yield' in bond_data
        assert '2y_yield' in bond_data
        assert 'credit_spreads' in bond_data
    
    def test_correlation_calculation(self, cross_asset_analyzer):
        """Test cross-asset correlation calculation"""
        equity = {'vix': 25}
        bond = {}
        commodity = {}
        currency = {}
        
        correlations = cross_asset_analyzer._calculate_cross_asset_correlations(
            equity, bond, commodity, currency
        )
        
        assert 'equity_bond' in correlations
        assert 'equity_commodity' in correlations
        assert 'equity_dollar' in correlations
        
        # High VIX should strengthen negative equity-bond correlation
        assert correlations['equity_bond'] < -0.4
    
    def test_divergence_identification(self, cross_asset_analyzer):
        """Test divergence identification"""
        correlations = {
            'equity_bond': -0.1,  # Breakdown
            'commodity_dollar': -0.2  # Divergence
        }
        
        divergences = cross_asset_analyzer._identify_divergences(correlations)
        
        assert 'equity_bond_correlation_breakdown' in divergences
        assert 'commodity_dollar_divergence' in divergences
    
    def test_allocation_signal_generation(self, cross_asset_analyzer):
        """Test allocation signal generation"""
        equity = {'vix': 15, 'breadth': 0.7}
        bond = {'10y_yield': 4.8}
        commodity = {'commodity_index': 240}
        
        signals = cross_asset_analyzer._generate_allocation_signals(
            equity, bond, commodity
        )
        
        assert signals['equity'] == 'overweight'  # Low VIX, good breadth
        assert signals['bonds'] == 'overweight'  # High yield
        assert signals['commodities'] == 'overweight'  # Low index
    
    def test_risk_on_score_calculation(self, cross_asset_analyzer):
        """Test risk-on score calculation"""
        equity = {'vix': 12, 'breadth': 0.8}
        bond = {'credit_spreads': 0.8}
        
        score = cross_asset_analyzer._calculate_risk_on_score(equity, bond)
        
        assert score > 5  # Should be risk-on
        assert score <= 10
    
    def test_sector_rotation_analysis(self, cross_asset_analyzer):
        """Test sector rotation analysis"""
        equity = {
            'sector_performance': {
                'XLK': 5.0,  # Tech
                'XLF': 3.0,  # Financials
                'XLE': -2.0,  # Energy
                'XLP': 0.5,  # Staples
                'XLU': -1.0  # Utilities
            }
        }
        
        rotation = cross_asset_analyzer._analyze_sector_rotation(equity)
        
        assert 'leading_sectors' in rotation
        assert 'lagging_sectors' in rotation
        assert 'rotation_pattern' in rotation
        assert 'XLK' in rotation['leading_sectors']
        assert 'XLE' in rotation['lagging_sectors']


# ==============================================================================
# INTEGRATION TESTS - Economist Agent
# ==============================================================================

@pytest.mark.integration
class TestEconomistAgent:
    """Integration tests for Economist Agent"""
    
    @pytest.mark.asyncio
    async def test_full_macro_analysis(self, economist_agent):
        """Test complete macro economic analysis"""
        agent = await economist_agent
        result = await agent.analyze_macro_environment('full')
        
        # Check all required fields
        assert 'timestamp' in result
        assert 'economic_cycle' in result
        assert 'growth_outlook' in result
        assert 'inflation_outlook' in result
        assert 'policy_outlook' in result
        assert 'geopolitical_risk' in result
        assert 'dominant_themes' in result
        assert 'sector_recommendations' in result
        assert 'asset_allocation' in result
        assert 'risk_scenarios' in result
        assert 'confidence_score' in result
        assert 'market_regime' in result
        
        # Validate ranges
        assert 0 <= result['confidence_score'] <= 10
        
        # Validate asset allocation
        allocation = result['asset_allocation']
        total = sum(allocation.values())
        assert 99 <= total <= 101  # Allow for rounding
    
    @pytest.mark.asyncio
    async def test_economic_cycle_determination(self, economist_agent):
        """Test economic cycle phase determination"""
        agent = await economist_agent
        
        # Test expansion detection
        economic_data = {
            'indicators': {
                'gdp': {'trend': 'accelerating'},
                'employment': {'trend': 'improving'}
            }
        }
        market_data = {}
        
        cycle = agent._determine_economic_cycle(economic_data, market_data)
        assert cycle == EconomicCycle.EXPANSION.value
        
        # Test contraction detection
        economic_data['indicators']['gdp']['trend'] = 'slowing'
        economic_data['indicators']['employment']['trend'] = 'softening'
        
        cycle = agent._determine_economic_cycle(economic_data, market_data)
        assert cycle == EconomicCycle.CONTRACTION.value
    
    @pytest.mark.asyncio
    async def test_sector_recommendations(self, economist_agent):
        """Test sector recommendation generation"""
        agent = await economist_agent
        
        themes = [
            MacroTheme(
                theme_name="Test Theme",
                description="Test",
                impact_sectors=[],
                beneficiaries=['Technology', 'Healthcare'],
                victims=['Energy', 'Materials'],
                time_horizon='medium',
                confidence=7,
                action_items=[]
            )
        ]
        
        recommendations = agent._generate_sector_recommendations(
            themes, 
            EconomicCycle.EXPANSION.value,
            {}
        )
        
        assert 'Technology' in recommendations
        assert recommendations['Technology'] == 'overweight'
        assert 'Energy' in recommendations
        assert recommendations['Energy'] == 'underweight'
    
    @pytest.mark.asyncio
    async def test_asset_allocation_generation(self, economist_agent):
        """Test asset allocation generation"""
        agent = await economist_agent
        
        # Test aggressive allocation
        allocation = agent._generate_asset_allocation(
            EconomicCycle.EXPANSION.value,
            {'risk_on_score': 8},
            7
        )
        
        assert allocation['equities'] == 80
        assert allocation['cash'] == 5
        
        # Test defensive allocation
        allocation = agent._generate_asset_allocation(
            EconomicCycle.CONTRACTION.value,
            {'risk_on_score': 2},
            3
        )
        
        assert allocation['equities'] <= 40
        assert allocation['cash'] >= 15
    
    @pytest.mark.asyncio
    async def test_risk_scenario_identification(self, economist_agent):
        """Test risk scenario identification"""
        agent = await economist_agent
        
        economic_data = {
            'indicators': {
                'yield_curve': {'shape': 'inverted'},
                'inflation': {'trend': 'accelerating'},
                'monetary_policy': {'stance': 'hawkish'}
            }
        }
        
        scenarios = agent._identify_risk_scenarios(economic_data, {}, [])
        
        assert len(scenarios) > 0
        
        # Check for recession scenario
        recession_scenario = next(
            (s for s in scenarios if s['scenario'] == 'recession'), 
            None
        )
        assert recession_scenario is not None
        assert 'probability' in recession_scenario
        assert 'hedges' in recession_scenario
    
    @pytest.mark.asyncio
    async def test_market_regime_determination(self, economist_agent):
        """Test market regime determination"""
        agent = await economist_agent
        
        # Test risk-on regime
        regime = agent._determine_market_regime(
            {'risk_on_score': 8},
            {}
        )
        assert regime == MarketRegime.RISK_ON.value
        
        # Test risk-off regime
        regime = agent._determine_market_regime(
            {'risk_on_score': 2},
            {}
        )
        assert regime == MarketRegime.RISK_OFF.value
    
    @pytest.mark.asyncio
    async def test_caching_mechanism(self, economist_agent):
        """Test caching functionality"""
        agent = await economist_agent
        
        # First call should not be cached
        result1 = await agent.analyze_macro_environment('full')
        assert agent.cache_hits == 0
        
        # Second call should be cached
        result2 = await agent.analyze_macro_environment('full')
        assert agent.cache_hits == 1
        
        # Results should be identical
        assert result1['economic_cycle'] == result2['economic_cycle']
        assert result1['confidence_score'] == result2['confidence_score']
    
    @pytest.mark.asyncio
    async def test_llm_integration(self, economist_agent, mock_llm_provider):
        """Test LLM integration for insights"""
        agent = await economist_agent
        
        result = await agent.analyze_macro_environment('full')
        
        assert 'ai_insights' in result
        assert 'summary' in result['ai_insights']
        assert 'recommendations' in result['ai_insights']
        
        # Verify LLM was called
        mock_llm_provider.generate_analysis.assert_called()
    
    @pytest.mark.asyncio
    async def test_error_recovery(self, economist_agent):
        """Test error handling and fallback mechanisms"""
        agent = await economist_agent
        
        # Mock a failure in economic analysis
        with patch.object(agent.economic_analyzer, 'analyze_economic_indicators',
                         side_effect=Exception("API Error")):
            result = await agent.analyze_macro_environment('full')
            
            # Should return fallback outlook
            assert result['confidence_score'] == 3.0
            assert result['economic_cycle'] == EconomicCycle.PEAK.value
            assert result['market_regime'] == MarketRegime.NEUTRAL.value
    
    @pytest.mark.asyncio
    async def test_performance_metrics(self, economist_agent):
        """Test performance metric tracking"""
        agent = await economist_agent
        
        # Run analysis
        await agent.analyze_macro_environment('full')
        
        metrics = agent.get_performance_metrics()
        
        assert metrics['total_analyses'] == 1
        assert metrics['successful_analyses'] == 1
        assert metrics['success_rate'] == 1.0
        assert 'cache_hit_rate' in metrics


# ==============================================================================
# PARAMETRIZED TESTS
# ==============================================================================

@pytest.mark.parametrize("growth,expected_outlook", [
    (3.0, 'strong_growth'),
    (2.0, 'moderate_growth'),
    (0.5, 'slow_growth'),
    (-1.0, 'contraction')
])
@pytest.mark.asyncio
async def test_growth_outlook_assessment(economist_agent, growth, expected_outlook):
    """Test growth outlook assessment with various inputs"""
    agent = await economist_agent
    
    data = {
        'indicators': {
            'gdp': {'current_growth': growth}
        }
    }
    
    outlook = agent._assess_growth_outlook(data)
    assert outlook == expected_outlook


@pytest.mark.parametrize("cpi,trend,expected", [
    (5.0, 'accelerating', 'rising_inflation'),
    (3.0, 'moderating', 'moderating_inflation'),
    (2.0, 'stable', 'stable_inflation')
])
@pytest.mark.asyncio
async def test_inflation_outlook_assessment(economist_agent, cpi, trend, expected):
    """Test inflation outlook assessment"""
    agent = await economist_agent
    
    data = {
        'indicators': {
            'inflation': {
                'cpi': cpi,
                'trend': trend
            }
        }
    }
    
    outlook = agent._assess_inflation_outlook(data)
    assert outlook == expected


@pytest.mark.parametrize("cycle,risk_score,expected_allocation", [
    (EconomicCycle.EXPANSION.value, 8, 80),  # Aggressive
    (EconomicCycle.RECOVERY.value, 6, 60),   # Moderate
    (EconomicCycle.CONTRACTION.value, 3, 40),  # Conservative
    (EconomicCycle.STAGFLATION.value, 2, 20)  # Defensive
])
@pytest.mark.asyncio
async def test_allocation_strategies(economist_agent, cycle, risk_score, expected_allocation):
    """Test asset allocation under different conditions"""
    agent = await economist_agent
    
    allocation = agent._generate_asset_allocation(
        cycle,
        {'risk_on_score': risk_score},
        5
    )
    
    assert allocation['equities'] == expected_allocation


# ==============================================================================
# PERFORMANCE TESTS
# ==============================================================================

@pytest.mark.performance
class TestPerformance:
    """Performance tests for Economist Agent"""
    
    @pytest.mark.asyncio
    async def test_analysis_speed(self, economist_agent):
        """Test that analysis completes within reasonable time"""
        agent = await economist_agent
        
        import time
        start = time.time()
        
        await agent.analyze_macro_environment('full')
        
        elapsed = time.time() - start
        assert elapsed < 5.0  # Should complete within 5 seconds
    
    @pytest.mark.asyncio
    async def test_concurrent_requests(self, economist_agent):
        """Test handling of concurrent analysis requests"""
        agent = await economist_agent
        
        # Run multiple analyses concurrently
        tasks = [
            agent.analyze_macro_environment('full'),
            agent.analyze_macro_environment('economic'),
            agent.analyze_macro_environment('market')
        ]
        
        results = await asyncio.gather(*tasks)
        
        assert len(results) == 3
        assert all('economic_cycle' in r for r in results)
    
    @pytest.mark.asyncio
    async def test_memory_efficiency(self, economist_agent):
        """Test memory usage remains reasonable"""
        agent = await economist_agent
        
        import sys
        
        # Run multiple analyses
        for _ in range(10):
            await agent.analyze_macro_environment('full')
        
        # Check that cache doesn't grow unbounded
        cache_size = sys.getsizeof(agent.cache_manager.cache)
        assert cache_size < 1_000_000  # Less than 1MB


# ==============================================================================
# END-TO-END TESTS
# ==============================================================================

@pytest.mark.e2e
class TestEndToEnd:
    """End-to-end tests simulating real usage"""
    
    @pytest.mark.asyncio
    async def test_complete_workflow(self, economist_agent):
        """Test complete analysis workflow"""
        agent = await economist_agent
        
        # Simulate portfolio manager requesting macro context
        outlook = await agent.analyze_macro_environment('full')
        
        # Verify outlook can be used for decisions
        assert outlook['market_regime'] in ['risk_on', 'risk_off', 'neutral', 'transition']
        
        # Check themes are actionable
        themes = outlook['dominant_themes']
        if themes:
            theme = themes[0]
            assert len(theme['action_items']) > 0
        
        # Verify sector recommendations
        sectors = outlook['sector_recommendations']
        assert all(w in ['overweight', 'neutral', 'underweight'] 
                  for w in sectors.values())
        
        # Check asset allocation totals 100%
        allocation = outlook['asset_allocation']
        total = sum(allocation.values())
        assert 99 <= total <= 101
    
    @pytest.mark.asyncio
    async def test_integration_with_market_context(self, economist_agent):
        """Test integration with market context manager"""
        agent = await economist_agent
        
        # This tests that the agent properly integrates with
        # the shared MarketContextManager from junior_research_analyst
        result = await agent.analyze_macro_environment('full')
        
        assert 'market_regime' in result
        assert 'economic_indicators' in result
        assert 'cross_asset_analysis' in result


# ==============================================================================
# PYTEST CONFIGURATION
# ==============================================================================

# Create pytest.ini if running this file directly
def create_pytest_config():
    """Create pytest.ini with custom markers"""
    import os
    if not os.path.exists('pytest.ini'):
        with open('pytest.ini', 'w') as f:
            f.write("""[pytest]
markers =
    unit: Unit tests
    integration: Integration tests
    performance: Performance tests
    e2e: End-to-end tests
    asyncio: Async tests
""")


# ==============================================================================
# RUN TESTS
# ==============================================================================

if __name__ == "__main__":
    create_pytest_config()
    pytest.main([__file__, "-v", "--tb=short"])