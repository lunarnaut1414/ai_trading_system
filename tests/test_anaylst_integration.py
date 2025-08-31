# tests/test_analyst_integration.py
"""
Fixed Integration Tests for Junior and Senior Research Analysts
Corrected fixture issues and async handling
"""

import asyncio
import pytest
from datetime import datetime
from typing import Dict, List
import json

# Note: Update these imports based on your actual project structure
try:
    from agents.junior_research_analyst import (
        JuniorResearchAnalyst,
        JuniorAnalystPool,
        MarketContextManager,
        UnifiedRiskAssessment,
        AnalysisType,
        TimeHorizon,
        RiskLevel,
        create_junior_analyst
    )

    from agents.senior_research_analyst import (
        SeniorResearchAnalyst,
        StrategicAnalysisEngine,
        MarketContextAnalyzer,
        create_senior_analyst
    )
except ImportError:
    # Fallback imports if modules are structured differently
    pass


# ========================================================================================
# FIXTURES - FIXED TO NOT BE ASYNC
# ========================================================================================

@pytest.fixture
def mock_llm_provider():
    """Mock LLM provider for testing"""
    class MockLLMProvider:
        async def generate(self, prompt: str) -> Dict:
            return {
                'summary': 'Strategic analysis indicates strong opportunities in technology sector.',
                'recommendations': [
                    'Increase technology allocation',
                    'Focus on high-conviction opportunities',
                    'Maintain risk management discipline'
                ],
                'risks': ['Market volatility', 'Sector concentration']
            }
    
    return MockLLMProvider()


@pytest.fixture
def mock_alpaca_provider():
    """Mock Alpaca provider for testing"""
    class MockAlpacaProvider:
        async def get_bars(self, symbol: str, timeframe: str, limit: int) -> List[Dict]:
            # Return mock market data
            return [
                {'close': 100 + i, 'high': 101 + i, 'low': 99 + i, 'volume': 1000000}
                for i in range(limit)
            ]
        
        async def get_latest_quote(self, symbol: str) -> Dict:
            return {'price': 20.5, 'bid': 20.4, 'ask': 20.6}
        
        async def get_news(self, symbol: str) -> List[Dict]:
            return [
                {'headline': f'Positive news for {symbol}', 'summary': 'Growth expected'}
            ]
    
    return MockAlpacaProvider()


@pytest.fixture
def mock_config():
    """Mock configuration"""
    class MockConfig:
        ANTHROPIC_API_KEY = "test_key"
        ALPACA_API_KEY = "test_key"
        ALPACA_SECRET_KEY = "test_secret"
        DATABASE_URL = "sqlite:///:memory:"
        CACHE_TTL = 300
        MAX_POSITIONS = 20
        MAX_POSITION_SIZE = 0.05
    
    return MockConfig()


@pytest.fixture
def junior_analyst(mock_llm_provider, mock_alpaca_provider, mock_config):
    """Create Junior Analyst instance"""
    return JuniorResearchAnalyst(mock_llm_provider, mock_alpaca_provider, mock_config)


@pytest.fixture
def senior_analyst(mock_llm_provider, mock_alpaca_provider, mock_config):
    """Create Senior Analyst instance"""
    return SeniorResearchAnalyst(mock_llm_provider, mock_alpaca_provider, mock_config)


@pytest.fixture
def junior_analyst_pool(mock_llm_provider, mock_alpaca_provider, mock_config):
    """Create Junior Analyst Pool"""
    return JuniorAnalystPool(mock_llm_provider, mock_alpaca_provider, mock_config, pool_size=3)


# ========================================================================================
# INTEGRATION TESTS - Full Pipeline
# ========================================================================================

@pytest.mark.integration
class TestFullPipeline:
    """Test complete analysis pipeline with enhanced features"""
    
    @pytest.mark.asyncio
    async def test_complete_analysis_pipeline(self, junior_analyst, senior_analyst):
        """Test complete pipeline from Junior analysis to Senior synthesis"""
        
        # Step 1: Junior analyzes multiple stocks
        tickers = ["AAPL", "GOOGL", "MSFT", "AMZN", "TSLA"]
        junior_reports = []
        
        for ticker in tickers:
            task_data = {
                "task_type": AnalysisType.NEW_OPPORTUNITY.value,
                "ticker": ticker,
                "technical_signal": {
                    "pattern": "breakout",
                    "score": 7 + (tickers.index(ticker) % 3),
                    "resistance_level": 105,
                    "support_level": 95
                }
            }
            
            report = await junior_analyst.analyze_stock(task_data)
            junior_reports.append(report)
        
        # Verify Junior reports have all required fields
        for report in junior_reports:
            assert report['analysis_status'] == 'success'
            assert 'ticker' in report
            assert 'conviction_level' in report
            assert 'risk_assessment' in report
            assert 'position_weight_percent' in report
            assert 'market_context' in report
            assert 'liquidity_score' in report
            assert 'catalyst_strength' in report
        
        # Step 2: Senior synthesizes reports
        synthesis_result = await senior_analyst.synthesize_reports(junior_reports)
        
        # Verify synthesis
        assert synthesis_result['status'] == 'success'
        assert 'strategic_analysis' in synthesis_result
        assert 'markdown_report' in synthesis_result
        
        analysis = synthesis_result['strategic_analysis']
        
        # Check all major components
        assert len(analysis['ranked_opportunities']) > 0
        assert len(analysis['strategic_themes']) > 0
        assert 'risk_assessment' in analysis
        assert 'time_horizon_allocation' in analysis
        assert 'correlation_analysis' in analysis
        assert 'execution_plan' in analysis
        
        # Verify data flow
        top_opportunity = analysis['ranked_opportunities'][0]
        assert top_opportunity.ticker in tickers
        assert top_opportunity.conviction_score > 0
        assert top_opportunity.junior_analyst_id != ''
        
        # Check execution plan
        execution = analysis['execution_plan']
        assert len(execution['actions']) > 0
        assert execution['total_capital_required'] > 0
    
    @pytest.mark.asyncio
    async def test_shared_market_context(self, junior_analyst, senior_analyst):
        """Test that market context is shared between agents"""
        
        # Both agents should use the same market context manager
        junior_context = await junior_analyst.market_context_manager.get_current_context()
        senior_context = await senior_analyst.market_context_manager.get_current_context()
        
        # Verify context has required fields
        assert 'regime' in junior_context
        assert 'vix_level' in junior_context
        assert 'sector_performance' in junior_context
        
        # Context should be consistent (cached)
        assert junior_context['timestamp'] == senior_context['timestamp']
    
    @pytest.mark.asyncio
    async def test_unified_risk_assessment(self):
        """Test unified risk assessment system"""
        
        analysis_data = {
            'volatility': 0.25,
            'average_volume': 5000000,
            'sector': 'Technology',
            'correlation_to_spy': 0.65,
            'catalysts': ['Earnings', 'Product launch'],
            'technical_score': 7
        }
        
        risk_assessment = UnifiedRiskAssessment.calculate_risk_score(analysis_data)
        
        assert 'overall_risk_score' in risk_assessment
        assert 'risk_level' in risk_assessment
        assert 'risk_factors' in risk_assessment
        assert 'key_risks' in risk_assessment
        
        assert 1 <= risk_assessment['overall_risk_score'] <= 10
        assert risk_assessment['risk_level'] in ['low', 'medium', 'high', 'very_high']


# ========================================================================================
# INTEGRATION TESTS - Feedback Mechanism
# ========================================================================================

@pytest.mark.integration
class TestFeedbackMechanism:
    """Test feedback loop between Senior and Junior Analysts"""
    
    @pytest.mark.asyncio
    async def test_feedback_loop(self, junior_analyst, senior_analyst):
        """Test feedback mechanism between agents"""
        
        # Junior analysis
        analysis = await junior_analyst.analyze_stock({
            "task_type": AnalysisType.NEW_OPPORTUNITY.value,
            "ticker": "AAPL",
            "technical_signal": {"pattern": "breakout", "score": 8}
        })
        
        analysis_id = analysis['analysis_id']
        
        # Senior provides feedback
        feedback_result = await senior_analyst.provide_junior_feedback(
            analysis_id,
            {
                "type": "quality",
                "score": 7,
                "improvements": ["Include more fundamental data", "Enhance catalyst analysis"],
                "strengths": ["Good technical analysis", "Clear risk assessment"]
            }
        )
        
        assert feedback_result['status'] == 'feedback_recorded'
        
        # Junior processes feedback
        await junior_analyst.process_feedback({
            'analysis_id': analysis_id,
            'performance_score': 7,
            'improvements_needed': ["Include more fundamental data"],
            'strengths': ["Good technical analysis"]
        })
        
        # Verify feedback was processed
        assert junior_analyst.performance_metrics['feedback_received_count'] > 0
        assert junior_analyst.performance_metrics['performance_score'] != 5.0  # Changed from default
    
    @pytest.mark.asyncio
    async def test_quality_scoring(self, senior_analyst):
        """Test quality scoring of Junior reports by Senior"""
        
        # Create reports with varying quality
        high_quality_report = {
            'ticker': 'AAPL',
            'analysis_id': 'test_1',
            'recommendation': 'BUY',
            'confidence': 8,
            'conviction_level': 4,
            'thesis': 'Strong growth momentum with solid fundamentals',
            'catalysts': ['iPhone launch', 'Services growth'],
            'risk_assessment': {'overall_risk_score': 4},
            'technical_signals': {'rsi': 55, 'macd': 'bullish'},
            'expected_return': 0.15,
            'position_weight_percent': 4,
            'liquidity_score': 9,
            'catalyst_strength': 8,
            'technical_score': 8,
            'analysis_status': 'success'
        }
        
        low_quality_report = {
            'ticker': 'XYZ',
            'analysis_id': 'test_2',
            'recommendation': 'HOLD',
            'confidence': 3,
            'conviction_level': 2,
            'expected_return': 0.05,
            'position_weight_percent': 2,
            'liquidity_score': 5,
            'catalyst_strength': 3,
            'technical_score': 4,
            'analysis_status': 'success',
            'risk_assessment': {}
        }
        
        # Synthesize and evaluate
        synthesis = await senior_analyst.synthesize_reports([high_quality_report, low_quality_report])
        
        # Check feedback queue
        assert len(senior_analyst.feedback_queue) > 0
        
        # High quality report should rank higher
        opportunities = synthesis['strategic_analysis']['ranked_opportunities']
        assert opportunities[0].ticker == 'AAPL'


# ========================================================================================
# INTEGRATION TESTS - Parallel Processing
# ========================================================================================

@pytest.mark.integration
class TestParallelProcessing:
    """Test parallel processing capabilities"""
    
    @pytest.mark.asyncio
    async def test_junior_analyst_pool(self, junior_analyst_pool):
        """Test Junior Analyst pool for batch processing"""
        
        tickers = ["AAPL", "GOOGL", "MSFT", "AMZN", "TSLA", "META", "NVDA", "AMD"]
        
        # Process batch in parallel
        start_time = datetime.now()
        results = await junior_analyst_pool.analyze_batch(
            tickers, 
            AnalysisType.NEW_OPPORTUNITY.value
        )
        processing_time = (datetime.now() - start_time).total_seconds()
        
        # Verify results
        assert len(results) == len(tickers)
        
        for result in results:
            assert result['analysis_status'] == 'success'
            assert result['ticker'] in tickers
        
        # Check pool metrics
        metrics = junior_analyst_pool.get_pool_metrics()
        assert metrics['pool_size'] == 3
        assert metrics['total_analyses'] == len(tickers)
        
        print(f"Batch processing time: {processing_time:.2f}s for {len(tickers)} stocks")
    
    @pytest.mark.asyncio
    async def test_concurrent_synthesis(self, mock_llm_provider, mock_alpaca_provider, mock_config):
        """Test concurrent synthesis operations"""
        
        # Create multiple Senior Analysts
        analysts = [
            SeniorResearchAnalyst(mock_llm_provider, mock_alpaca_provider, mock_config)
            for _ in range(3)
        ]
        
        # Create sample reports
        sample_reports = [
            {
                'ticker': f'STOCK{i}',
                'analysis_id': f'id_{i}',
                'recommendation': 'BUY',
                'confidence': 7,
                'conviction_level': 4,
                'expected_return': 0.12,
                'risk_assessment': {'overall_risk_score': 5},
                'position_weight_percent': 3,
                'liquidity_score': 7,
                'catalyst_strength': 6,
                'technical_score': 7,
                'analysis_status': 'success'
            }
            for i in range(5)
        ]
        
        # Run concurrent synthesis
        tasks = [
            analyst.synthesize_reports(sample_reports) 
            for analyst in analysts
        ]
        results = await asyncio.gather(*tasks)
        
        # All should succeed
        assert all(r['status'] == 'success' for r in results)


# ========================================================================================
# INTEGRATION TESTS - Caching and Performance
# ========================================================================================

@pytest.mark.integration
class TestCachingAndPerformance:
    """Test intelligent caching and performance optimizations"""
    
    @pytest.mark.asyncio
    async def test_intelligent_caching(self, junior_analyst):
        """Test intelligent cache manager"""
        
        # First analysis (cache miss)
        task_data = {
            "task_type": AnalysisType.NEW_OPPORTUNITY.value,
            "ticker": "AAPL",
            "technical_signal": {"pattern": "breakout", "score": 8}
        }
        
        start_time = datetime.now()
        result1 = await junior_analyst.analyze_stock(task_data)
        time1 = (datetime.now() - start_time).total_seconds()
        
        # Second analysis (cache hit)
        start_time = datetime.now()
        result2 = await junior_analyst.analyze_stock(task_data)
        time2 = (datetime.now() - start_time).total_seconds()
        
        # Cache hit should be faster
        assert time2 < time1
        assert junior_analyst.performance_metrics['cache_hits'] > 0
        
        # Results should be identical
        assert result1['ticker'] == result2['ticker']
        assert result1['recommendation'] == result2['recommendation']
    
    @pytest.mark.asyncio
    async def test_cache_eviction(self, junior_analyst):
        """Test LRU cache eviction"""
        
        # Fill cache beyond capacity
        cache_size = 100  # Default cache size
        
        for i in range(cache_size + 20):
            task_data = {
                "task_type": AnalysisType.NEW_OPPORTUNITY.value,
                "ticker": f"STOCK{i}",
                "technical_signal": {"pattern": "test", "score": 5}
            }
            await junior_analyst.analyze_stock(task_data)
        
        # Cache should not exceed max size
        assert len(junior_analyst.cache_manager.cache) <= cache_size
    
    @pytest.mark.asyncio
    async def test_metadata_tracking(self, junior_analyst, senior_analyst):
        """Test analysis chain metadata tracking"""
        
        # Junior analysis
        task_data = {
            "task_type": AnalysisType.NEW_OPPORTUNITY.value,
            "ticker": "GOOGL",
            "technical_signal": {"pattern": "ascending_triangle", "score": 9}
        }
        
        junior_result = await junior_analyst.analyze_stock(task_data)
        
        # Check metadata
        assert 'analysis_chain' in junior_result
        chain_summary = junior_result['analysis_chain']
        assert chain_summary['status'] == 'success'
        assert chain_summary['num_steps'] > 0
        
        # Senior synthesis
        senior_result = await senior_analyst.synthesize_reports([junior_result])
        
        # Check synthesis metadata
        assert 'analysis_chain' in senior_result
        assert senior_result['metadata']['processing_time'] > 0


# ========================================================================================
# INTEGRATION TESTS - Market Regime Adaptation
# ========================================================================================

@pytest.mark.integration
class TestMarketRegimeAdaptation:
    """Test market regime-based adaptations"""
    
    @pytest.mark.asyncio
    async def test_risk_on_regime(self, senior_analyst):
        """Test behavior in risk-on market regime"""
        
        # Create growth-oriented reports
        growth_reports = [
            {
                'ticker': ticker,
                'analysis_id': f'growth_{i}',
                'recommendation': 'BUY',
                'confidence': 8,
                'conviction_level': 4,
                'sector': 'Technology',
                'expected_return': 0.20,
                'risk_assessment': {'overall_risk_score': 6},
                'position_weight_percent': 4,
                'liquidity_score': 8,
                'catalyst_strength': 8,
                'technical_score': 8,
                'time_horizon': TimeHorizon.MEDIUM_TERM.value,
                'analysis_status': 'success'
            }
            for i, ticker in enumerate(['NVDA', 'AMD', 'MSFT'])
        ]
        
        # Mock risk-on context
        async def mock_get_context():
            return {
                'regime': 'risk_on',
                'vix_level': {'level': 15, 'interpretation': 'low'},
                'sector_performance': {'XLK': 3.5},
                'timestamp': datetime.now().isoformat()
            }
        
        senior_analyst.market_context_manager.get_current_context = mock_get_context
        
        synthesis = await senior_analyst.synthesize_reports(growth_reports)
        analysis = synthesis['strategic_analysis']
        
        # Should identify growth theme
        themes = analysis['strategic_themes']
        assert any('Growth' in theme.theme_name for theme in themes)
        
        # Time horizon should favor shorter terms
        allocation = analysis['time_horizon_allocation']
        assert allocation['target_allocation']['short'] >= 0.3
    
    @pytest.mark.asyncio
    async def test_risk_off_regime(self, senior_analyst):
        """Test behavior in risk-off market regime"""
        
        # Create defensive reports
        defensive_reports = [
            {
                'ticker': ticker,
                'analysis_id': f'defensive_{i}',
                'recommendation': 'BUY',
                'confidence': 7,
                'conviction_level': 3,
                'sector': sector,
                'expected_return': 0.08,
                'risk_assessment': {'overall_risk_score': 3},
                'position_weight_percent': 3,
                'liquidity_score': 9,
                'catalyst_strength': 5,
                'technical_score': 6,
                'time_horizon': TimeHorizon.LONG_TERM.value,
                'analysis_status': 'success'
            }
            for i, (ticker, sector) in enumerate([
                ('XLU', 'Utilities'),
                ('PG', 'Consumer Staples'),
                ('JNJ', 'Healthcare')
            ])
        ]
        
        # Mock risk-off context
        async def mock_get_context():
            return {
                'regime': 'risk_off',
                'vix_level': {'level': 35, 'interpretation': 'high'},
                'sector_performance': {'XLU': 1.2, 'XLP': 0.8},
                'timestamp': datetime.now().isoformat()
            }
        
        senior_analyst.market_context_manager.get_current_context = mock_get_context
        
        synthesis = await senior_analyst.synthesize_reports(defensive_reports)
        analysis = synthesis['strategic_analysis']
        
        # Should identify defensive theme
        themes = analysis['strategic_themes']
        assert any('Defensive' in theme.theme_name for theme in themes)
        
        # Risk assessment should be lower
        risk = analysis['risk_assessment']
        assert risk.overall_risk_score < 5


# ========================================================================================
# TEST RUNNER
# ========================================================================================

if __name__ == "__main__":
    import sys
    
    # Run with coverage if requested
    if "--coverage" in sys.argv:
        sys.argv.remove("--coverage")
        sys.exit(pytest.main([
            __file__, 
            "--cov=agents", 
            "--cov-report=html", 
            "--cov-report=term-missing",
            "-v"
        ]))
    else:
        # Run all tests with verbose output
        sys.exit(pytest.main([__file__, "-v", "-s"]))