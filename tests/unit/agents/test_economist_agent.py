# tests/unit/agents/test_economist_agent.py
"""
Complete test suite for Economist Agent
Tests all components and functionality with 48 tests
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from datetime import datetime, timedelta
import time
import sys
import numpy as np

# Import the modules to test
from src.agents.economist import (
    EconomistAgent,
    EconomicDataAnalyzer,
    MacroThemeIdentifier,
    CrossAssetAnalyzer,
    EconomicCycle,
    PolicyStance,
    MacroTheme,
    GeopoliticalRisk
)

# Import from junior_analyst (without MarketRegime)
from src.agents.junior_analyst import (
    ConvictionLevel,
    TimeHorizon,
    RiskLevel,
    MarketContextManager,
    IntelligentCacheManager,
    AnalysisMetadataTracker
)

# Import MarketRegime from senior_analyst where it's actually defined
from src.agents.senior_analyst import MarketRegime


# ==============================================================================
# FIXTURES
# ==============================================================================

@pytest.fixture
def mock_llm_provider():
    """Mock LLM provider for testing"""
    provider = Mock()
    provider.generate_analysis = AsyncMock(return_value={
        'summary': 'Economic outlook remains positive with moderate growth expectations',
        'recommendations': [
            'Maintain balanced portfolio allocation',
            'Consider defensive positioning'
        ],
        'risks': ['Inflation persistence', 'Policy overtightening'],
        'opportunities': ['Technology sector recovery', 'Bond yields attractive']
    })
    return provider


@pytest.fixture
def mock_alpaca_provider():
    """Mock Alpaca market data provider"""
    provider = Mock()
    
    # Mock methods
    provider.get_latest_quote = AsyncMock(return_value={
        'symbol': 'SPY',
        'price': 450.0,
        'volume': 1000000
    })
    
    provider.get_market_indicators = AsyncMock(return_value={
        'vix': 15.5,
        'dxy': 103.5,
        'tlt': 95.0
    })
    
    provider.get_sector_performance = AsyncMock(return_value={
        'technology': 0.025,
        'financials': 0.015,
        'healthcare': -0.005
    })
    
    return provider


@pytest.fixture
def mock_config():
    """Mock configuration object"""
    config = Mock()
    config.get = Mock(return_value={
        'cache_ttl': 300,
        'max_retries': 3,
        'timeout': 30
    })
    return config


@pytest.fixture
async def economist_agent(mock_llm_provider, mock_alpaca_provider, mock_config):
    """Create test Economist Agent instance"""
    agent = EconomistAgent(
        agent_name='test_economist',
        llm_provider=mock_llm_provider,
        config=mock_config,
        alpaca_provider=mock_alpaca_provider
    )
    return agent


# ==============================================================================
# UNIT TESTS - ECONOMIC DATA ANALYZER
# ==============================================================================

class TestEconomicDataAnalyzer:
    """Test EconomicDataAnalyzer component"""
    
    @pytest.mark.asyncio
    async def test_analyze_economic_indicators(self, mock_alpaca_provider):
        """Test economic indicator analysis"""
        analyzer = EconomicDataAnalyzer(mock_alpaca_provider)
        
        result = await analyzer.analyze_economic_indicators()
        
        assert 'indicators' in result
        assert 'health_score' in result
        assert 'trend' in result
        assert 'risks' in result
        assert 'timestamp' in result
        
        # Check health score is in valid range
        assert 0 <= result['health_score'] <= 10
    
    @pytest.mark.asyncio
    async def test_gdp_analysis(self, mock_alpaca_provider):
        """Test GDP analysis"""
        analyzer = EconomicDataAnalyzer(mock_alpaca_provider)
        
        gdp_data = await analyzer._analyze_gdp()
        
        assert 'current_growth' in gdp_data
        assert 'trend' in gdp_data
        assert 'forecast' in gdp_data
        assert gdp_data['trend'] in ['expanding', 'contracting', 'stable']
    
    @pytest.mark.asyncio
    async def test_inflation_analysis(self, mock_alpaca_provider):
        """Test inflation analysis"""
        analyzer = EconomicDataAnalyzer(mock_alpaca_provider)
        
        inflation_data = await analyzer._analyze_inflation()
        
        assert 'cpi' in inflation_data
        assert 'core_cpi' in inflation_data
        assert 'pce' in inflation_data
        assert 'trend' in inflation_data
        assert inflation_data['trend'] in ['accelerating', 'moderating', 'stable', 'declining']
    
    @pytest.mark.asyncio
    async def test_employment_analysis(self, mock_alpaca_provider):
        """Test employment data analysis"""
        analyzer = EconomicDataAnalyzer(mock_alpaca_provider)
        
        employment_data = await analyzer._analyze_employment()
        
        assert 'unemployment_rate' in employment_data
        assert 'job_growth' in employment_data
        assert 'wage_growth' in employment_data
        assert employment_data['unemployment_rate'] > 0
    
    @pytest.mark.asyncio
    async def test_yield_curve_analysis(self, mock_alpaca_provider):
        """Test yield curve analysis"""
        analyzer = EconomicDataAnalyzer(mock_alpaca_provider)
        
        yield_data = await analyzer._analyze_yield_curve()
        
        assert 'shape' in yield_data
        assert 'spread_2_10' in yield_data
        assert yield_data['shape'] in ['normal', 'inverted', 'flat']
    
    def test_economic_health_score_calculation(self, mock_alpaca_provider):
        """Test economic health score calculation"""
        analyzer = EconomicDataAnalyzer(mock_alpaca_provider)
        
        indicators = {
            'gdp': {'current_growth': 2.5},
            'employment': {'unemployment_rate': 3.8},
            'inflation': {'cpi': 2.2},
            'manufacturing': {'pmi': 52}
        }
        
        score = analyzer._calculate_economic_health_score(indicators)
        
        assert isinstance(score, float)
        assert 0 <= score <= 10
    
    def test_economic_trend_determination(self, mock_alpaca_provider):
        """Test economic trend determination"""
        analyzer = EconomicDataAnalyzer(mock_alpaca_provider)
        
        indicators = {
            'gdp': {'trend': 'expanding'},
            'employment': {'unemployment_rate': 3.5},
            'manufacturing': {'pmi': 53}
        }
        
        trend = analyzer._determine_economic_trend(indicators)
        
        assert trend in ['improving', 'deteriorating', 'mixed', 'neutral']
        assert trend == 'improving'  # Based on positive indicators
    
    def test_risk_identification(self, mock_alpaca_provider):
        """Test economic risk identification"""
        analyzer = EconomicDataAnalyzer(mock_alpaca_provider)
        
        indicators = {
            'yield_curve': {'shape': 'inverted'},
            'inflation': {'cpi': 5.0},
            'employment': {'trend': 'weakening'},
            'manufacturing': {'pmi': 48}
        }
        
        risks = analyzer._identify_economic_risks(indicators)
        
        assert isinstance(risks, list)
        assert 'yield_curve_inversion' in risks
        assert 'elevated_inflation' in risks
        assert 'manufacturing_contraction' in risks
    
    @pytest.mark.asyncio
    async def test_error_handling(self, mock_alpaca_provider):
        """Test error handling in economic analysis"""
        analyzer = EconomicDataAnalyzer(mock_alpaca_provider)
        
        # Force an error by mocking a method to raise
        with patch.object(analyzer, '_analyze_gdp', side_effect=Exception("Test error")):
            result = await analyzer.analyze_economic_indicators()
            
            # Should return default indicators on error
            assert result['health_score'] == 5.0
            assert result['trend'] == 'neutral'
            assert result['risks'] == []


# ==============================================================================
# UNIT TESTS - MACRO THEME IDENTIFIER
# ==============================================================================

class TestMacroThemeIdentifier:
    """Test MacroThemeIdentifier component"""
    
    def test_identify_macro_themes(self):
        """Test macro theme identification"""
        identifier = MacroThemeIdentifier()
        
        economic_data = {
            'indicators': {
                'inflation': {'cpi': 5.0, 'trend': 'accelerating'},
                'gdp': {'current_growth': 0.5},
                'yield_curve': {'shape': 'inverted'},
                'manufacturing': {'pmi': 47}
            }
        }
        
        themes = identifier.identify_macro_themes(economic_data)
        
        assert isinstance(themes, list)
        assert len(themes) > 0
        assert all(isinstance(t, MacroTheme) for t in themes)
        
        # Should identify stagflation and recession themes
        theme_names = [t.theme_name for t in themes]
        assert 'Stagflation Concerns' in theme_names
        assert 'Recession Risk' in theme_names
    
    def test_stagflation_detection(self):
        """Test stagflation theme detection"""
        identifier = MacroThemeIdentifier()
        
        indicators = {
            'inflation': {'cpi': 5.0},
            'gdp': {'current_growth': 0.5}
        }
        
        is_stagflation = identifier._check_stagflation(indicators)
        assert is_stagflation == True
        
        theme = identifier._create_stagflation_theme()
        assert theme.theme_name == 'Stagflation Concerns'
        assert 'commodities' in theme.beneficiaries
        assert 'technology' in theme.victims
    
    def test_disinflation_detection(self):
        """Test disinflation theme detection"""
        identifier = MacroThemeIdentifier()
        
        indicators = {
            'inflation': {'trend': 'moderating'}
        }
        
        is_disinflation = identifier._check_disinflation(indicators)
        assert is_disinflation == True
        
        theme = identifier._create_disinflation_theme()
        assert theme.theme_name == 'Disinflation Trend'
        assert 'technology' in theme.beneficiaries
    
    def test_recession_detection(self):
        """Test recession theme detection"""
        identifier = MacroThemeIdentifier()
        
        indicators = {
            'yield_curve': {'shape': 'inverted'},
            'manufacturing': {'pmi': 45}
        }
        
        is_recession = identifier._check_recession(indicators)
        assert is_recession == True
        
        theme = identifier._create_recession_theme()
        assert theme.theme_name == 'Recession Risk'
        assert 'utilities' in theme.beneficiaries
        assert 'financials' in theme.victims
    
    def test_theme_creation(self):
        """Test theme creation with proper structure"""
        identifier = MacroThemeIdentifier()
        
        theme = identifier._create_goldilocks_theme()
        
        assert isinstance(theme, MacroTheme)
        assert theme.theme_name == 'Goldilocks Economy'
        assert isinstance(theme.confidence, float)
        assert 0 <= theme.confidence <= 1
        assert isinstance(theme.action_items, list)
        assert len(theme.action_items) > 0


# ==============================================================================
# UNIT TESTS - CROSS ASSET ANALYZER
# ==============================================================================

class TestCrossAssetAnalyzer:
    """Test CrossAssetAnalyzer component"""
    
    @pytest.mark.asyncio
    async def test_analyze_cross_asset_dynamics(self, mock_alpaca_provider):
        """Test cross-asset analysis"""
        analyzer = CrossAssetAnalyzer(mock_alpaca_provider)
        
        result = await analyzer.analyze_cross_asset_dynamics()
        
        assert 'equity_metrics' in result
        assert 'bond_metrics' in result
        assert 'correlations' in result
        assert 'risk_on_score' in result
        assert 'sector_rotation' in result
        
        # Check risk-on score is in valid range
        assert 0 <= result['risk_on_score'] <= 10
    
    @pytest.mark.asyncio
    async def test_equity_market_data(self, mock_alpaca_provider):
        """Test equity market data gathering"""
        analyzer = CrossAssetAnalyzer(mock_alpaca_provider)
        
        equity_data = await analyzer._get_equity_market_data()
        
        assert 'sp500_level' in equity_data
        assert 'vix' in equity_data
        assert 'pe_ratio' in equity_data
        assert 'breadth' in equity_data
    
    @pytest.mark.asyncio
    async def test_bond_market_data(self, mock_alpaca_provider):
        """Test bond market data gathering"""
        analyzer = CrossAssetAnalyzer(mock_alpaca_provider)
        
        bond_data = await analyzer._get_bond_market_data()
        
        assert '10y_yield' in bond_data
        assert '2y_yield' in bond_data
        assert 'credit_spreads' in bond_data
    
    def test_correlation_calculation(self, mock_alpaca_provider):
        """Test cross-asset correlation calculation"""
        analyzer = CrossAssetAnalyzer(mock_alpaca_provider)
        
        correlations = analyzer._calculate_correlations(
            {'sp500_level': 4500}, 
            {'10y_yield': 4.25},
            {'oil_price': 85}, 
            {'dxy_level': 103}
        )
        
        assert 'equity_bond' in correlations
        assert 'equity_vix' in correlations
        assert -1 <= correlations['equity_bond'] <= 1
        assert -1 <= correlations['equity_vix'] <= 1
    
    def test_divergence_identification(self, mock_alpaca_provider):
        """Test divergence identification"""
        analyzer = CrossAssetAnalyzer(mock_alpaca_provider)
        
        correlations = {
            'equity_bond': -0.1,  # Breakdown
            'equity_vix': -0.5   # Divergence
        }
        
        divergences = analyzer._identify_divergences(correlations)
        
        assert isinstance(divergences, list)
        assert 'equity_bond_correlation_breakdown' in divergences
        assert 'vix_equity_divergence' in divergences
    
    def test_allocation_signal_generation(self, mock_alpaca_provider):
        """Test allocation signal generation"""
        analyzer = CrossAssetAnalyzer(mock_alpaca_provider)
        
        signals = analyzer._generate_allocation_signals(
            {'pe_ratio': 16, 'vix': 18},
            {'10y_yield': 4.8},
            {'commodity_trend': 'upward'}
        )
        
        assert 'equity' in signals
        assert 'bonds' in signals
        assert 'commodities' in signals
        assert signals['equity'] in ['overweight', 'neutral', 'underweight']
        assert signals['bonds'] == 'overweight'  # High yield
        assert signals['commodities'] == 'overweight'  # Upward trend
    
    def test_risk_on_score_calculation(self, mock_alpaca_provider):
        """Test risk-on/risk-off score calculation"""
        analyzer = CrossAssetAnalyzer(mock_alpaca_provider)
        
        score = analyzer._calculate_risk_on_score(
            {'trend': 'upward', 'vix': 12},
            {'credit_spreads': 0.8},
            {'emerging_markets': 'strong'}
        )
        
        assert isinstance(score, float)
        assert 0 <= score <= 10
        assert score > 5  # Should be risk-on given inputs
    
    def test_sector_rotation_analysis(self, mock_alpaca_provider):
        """Test sector rotation analysis"""
        analyzer = CrossAssetAnalyzer(mock_alpaca_provider)
        
        rotation = analyzer._analyze_sector_rotation()
        
        assert isinstance(rotation, dict)
        assert 'technology' in rotation
        assert 'financials' in rotation
        assert 'healthcare' in rotation
        
        # Check valid rotation signals
        valid_signals = ['momentum_positive', 'momentum_negative', 'neutral', 
                        'defensive_bid', 'weakening', 'rate_sensitive', 
                        'commodity_linked', 'growth_sensitive']
        for sector, signal in rotation.items():
            assert signal in valid_signals


# ==============================================================================
# INTEGRATION TESTS - ECONOMIST AGENT
# ==============================================================================

class TestEconomistAgent:
    """Integration tests for the complete Economist Agent"""
    
    @pytest.mark.asyncio
    async def test_full_macro_analysis(self, economist_agent):
        """Test complete macro analysis workflow"""
        agent = await economist_agent
        
        result = await agent.analyze_macro_environment('full')
        
        # Check all required fields are present
        assert 'economic_cycle' in result
        assert 'growth_outlook' in result
        assert 'inflation_outlook' in result
        assert 'policy_outlook' in result
        assert 'dominant_themes' in result
        assert 'sector_recommendations' in result
        assert 'asset_allocation' in result
        assert 'risk_scenarios' in result
        assert 'confidence_score' in result
        assert 'market_regime' in result
        
        # Validate data types
        assert isinstance(result['economic_cycle'], str)
        assert isinstance(result['dominant_themes'], list)
        assert isinstance(result['sector_recommendations'], dict)
        assert isinstance(result['asset_allocation'], dict)
        assert isinstance(result['confidence_score'], float)
        
        # Check allocations sum to 100
        total_allocation = sum(result['asset_allocation'].values())
        assert abs(total_allocation - 100) < 0.01
    
    @pytest.mark.asyncio
    async def test_economic_cycle_determination(self, economist_agent):
        """Test economic cycle phase determination"""
        agent = await economist_agent
        
        # Test with expansion conditions
        economic_data = {
            'indicators': {
                'gdp': {'current_growth': 3.5, 'trend': 'expanding'},
                'employment': {'unemployment_rate': 3.5},
                'inflation': {'cpi': 2.5},
                'manufacturing': {'pmi': 55}
            }
        }
        
        cycle = agent._determine_economic_cycle(economic_data)
        assert cycle == EconomicCycle.EXPANSION.value
        
        # Test with contraction conditions
        economic_data['indicators']['gdp']['current_growth'] = -0.5
        economic_data['indicators']['manufacturing']['pmi'] = 45
        
        cycle = agent._determine_economic_cycle(economic_data)
        assert cycle == EconomicCycle.CONTRACTION.value
    
    @pytest.mark.asyncio
    async def test_sector_recommendations(self, economist_agent):
        """Test sector recommendation generation"""
        agent = await economist_agent
        
        result = await agent.analyze_macro_environment('full')
        
        recommendations = result['sector_recommendations']
        
        # Check all major sectors are covered
        expected_sectors = ['technology', 'financials', 'healthcare', 
                          'consumer_discretionary', 'consumer_staples',
                          'industrials', 'energy', 'utilities', 'real_estate']
        
        for sector in expected_sectors:
            assert sector in recommendations
            assert recommendations[sector] in ['overweight', 'neutral', 'underweight']
    
    @pytest.mark.asyncio
    async def test_asset_allocation_generation(self, economist_agent):
        """Test asset allocation generation"""
        agent = await economist_agent
        
        # Test allocation for different cycles
        allocations = agent._generate_asset_allocation(
            EconomicCycle.EXPANSION.value,
            {'risk_on_score': 7},
            2.5  # inflation
        )
        
        assert 'equities' in allocations
        assert 'bonds' in allocations
        assert 'commodities' in allocations
        assert 'cash' in allocations
        
        # Check reasonable allocation ranges
        assert 20 <= allocations['equities'] <= 80
        assert 10 <= allocations['bonds'] <= 50
        assert 0 <= allocations['commodities'] <= 30
        assert 5 <= allocations['cash'] <= 30
        
        # Check sum to 100
        total = sum(allocations.values())
        assert abs(total - 100) < 0.01
    
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
        """Test caching functionality with proper state management"""
        agent = await economist_agent
        
        # Reset cache hits counter to ensure clean state
        agent.cache_hits = 0
        
        # Clear any existing cache to ensure fresh start
        agent._cache_dict.clear()
        if hasattr(agent.cache_manager, 'cache'):
            agent.cache_manager.cache.clear()
        
        # First call should not be cached
        result1 = await agent.analyze_macro_environment('full')
        initial_cache_hits = agent.cache_hits
        assert initial_cache_hits == 0, "First call should not be a cache hit"
        
        # Verify result was stored in cache
        today = datetime.now().strftime('%Y%m%d')
        cache_key = f"macro_full_{today}"
        
        # Check that the result is now in cache
        cached_value = agent._cache_get(cache_key)
        assert cached_value is not None, "Result should be cached after first call"
        
        # Second call with same parameters should be cached
        result2 = await agent.analyze_macro_environment('full')
        assert agent.cache_hits == 1, f"Second call should be a cache hit (got {agent.cache_hits})"
        
        # Results should be identical (same object reference since cached)
        assert result1['economic_cycle'] == result2['economic_cycle']
        assert result1['confidence_score'] == result2['confidence_score']
        
        # Verify that the cached result is actually being returned
        assert result2 == cached_value, "Cached value should be returned"
        
        # Test cache with different request type (should not hit cache)
        agent.cache_hits = 0  # Reset counter
        result3 = await agent.analyze_macro_environment('economic')
        assert agent.cache_hits == 0, "Different request type should not hit cache"
        
        # But calling same type again should hit cache
        result4 = await agent.analyze_macro_environment('economic')
        assert agent.cache_hits == 1, "Same request type should hit cache"
    
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
        """Test error recovery and fallback mechanisms"""
        agent = await economist_agent
        
        # Force an error in economic analyzer
        with patch.object(agent.economic_analyzer, 'analyze_economic_indicators', 
                         side_effect=Exception("Test error")):
            result = await agent.analyze_macro_environment('full')
            
            # Should return fallback outlook
            assert result is not None
            assert 'economic_cycle' in result
            assert 'asset_allocation' in result
            assert result['confidence_score'] == 5.0  # Default fallback score
    
    @pytest.mark.asyncio
    async def test_performance_metrics(self, economist_agent):
        """Test performance metrics tracking"""
        agent = await economist_agent
        
        # Mock the market context manager to avoid errors
        agent.market_context_manager.get_current_context = AsyncMock(return_value={
            'regime': 'neutral',
            'volatility': 15.0,
            'trend': 'sideways'
        })
        
        # Clear any existing cache to ensure clean test
        agent._cache_dict.clear()
        
        # Reset counters
        agent.total_analyses = 0
        agent.successful_analyses = 0
        agent.cache_hits = 0
        
        # Run some analyses
        result1 = await agent.analyze_macro_environment('full')
        assert result1 is not None  # Verify first analysis succeeded
        
        result2 = await agent.analyze_macro_environment('full')  # Should be cached
        assert result2 is not None  # Verify cached result returned
        
        result3 = await agent.analyze_macro_environment('economic')
        assert result3 is not None  # Verify third analysis succeeded
        
        metrics = agent.get_performance_metrics()
        
        assert metrics['total_analyses'] == 3
        assert metrics['successful_analyses'] == 3  # All should be successful now
        assert metrics['cache_hits'] == 1
        assert metrics['success_rate'] == 100.0
        assert metrics['cache_hit_rate'] > 0


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
    """Test growth outlook assessment for different GDP levels"""
    agent = await economist_agent
    
    economic_data = {
        'indicators': {
            'gdp': {'current_growth': growth}
        }
    }
    
    outlook = agent._assess_growth_outlook(economic_data)
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
    
    economic_data = {
        'indicators': {
            'inflation': {
                'cpi': cpi,
                'trend': trend
            }
        }
    }
    
    outlook = agent._assess_inflation_outlook(economic_data)
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
        2.5  # Normal inflation
    )
    
    # Check equity allocation is close to expected
    # Allow some variance due to adjustments
    assert abs(allocation['equities'] - expected_allocation) <= 10


# ==============================================================================
# PERFORMANCE TESTS
# ==============================================================================

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
        cache_size = sys.getsizeof(agent._cache_dict)
        assert cache_size < 1_000_000  # Less than 1MB


# ==============================================================================
# END-TO-END TESTS
# ==============================================================================

class TestEndToEnd:
    """End-to-end tests for complete workflows"""
    
    @pytest.mark.asyncio
    async def test_complete_workflow(self, economist_agent):
        """Test complete analysis workflow from start to finish"""
        agent = await economist_agent
        
        # Run full analysis
        result = await agent.analyze_macro_environment('full')
        
        # Verify comprehensive output
        assert result is not None
        assert result['confidence_score'] >= 5.0
        assert len(result['dominant_themes']) <= 3
        assert len(result['risk_scenarios']) <= 5
        
        # Verify all sectors have recommendations
        all_sectors = ['technology', 'financials', 'healthcare', 
                      'consumer_discretionary', 'consumer_staples',
                      'industrials', 'energy', 'utilities', 'real_estate',
                      'materials', 'communication']
        
        for sector in all_sectors:
            assert sector in result['sector_recommendations']
        
        # Verify asset allocation is complete
        assert sum(result['asset_allocation'].values()) == 100
        
        # Verify economic indicators were analyzed
        assert 'economic_indicators' in result
        assert 'cross_asset_analysis' in result
    
    @pytest.mark.asyncio
    async def test_integration_with_market_context(self, economist_agent):
        """Test integration with market context manager"""
        agent = await economist_agent
        
        # Mock market context response - use get_current_context instead of get_market_context
        with patch.object(agent.market_context_manager, 'get_current_context',
                         AsyncMock(return_value={
                             'market_state': 'open',
                             'volatility': 15.5,
                             'trend': 'bullish',
                             'regime': 'risk_on'
                         })):
            
            result = await agent.analyze_macro_environment('full')
            
            assert result is not None
            assert 'market_regime' in result
            assert result['market_regime'] in [
                MarketRegime.RISK_ON.value,
                MarketRegime.RISK_OFF.value,
                MarketRegime.NEUTRAL.value,
                MarketRegime.TRANSITION.value  # Changed from TRANSITIONAL to TRANSITION
            ]