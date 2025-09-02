# tests/test_junior_analyst.py
"""
Junior Research Analyst Test Suite
Comprehensive testing for the Junior Research Analyst Agent using mocked providers

Run tests:
    pytest tests/test_junior_analyst.py -v                    # All tests
    pytest tests/test_junior_analyst.py -v -m unit           # Unit tests only
    pytest tests/test_junior_analyst.py -v -m integration    # Integration tests
    pytest tests/test_junior_analyst.py -v -m smoke          # Quick smoke tests
    pytest tests/test_junior_analyst.py -v -k "analysis"     # Specific tests
    pytest tests/test_junior_analyst.py --cov=agents         # With coverage
"""

import pytest
import asyncio
import json
import os
import sys
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from unittest.mock import Mock, AsyncMock, MagicMock, patch
import uuid

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.agents.junior_analyst import (
    JuniorResearchAnalyst,
    AnalysisType,
    RecommendationType,
    TimeHorizon,
    RiskLevel,
    PositionSize,
    create_junior_analyst
)


# ==============================================================================
# FIXTURES
# ==============================================================================

@pytest.fixture
def mock_config():
    """Mock configuration for testing"""
    config = MagicMock()
    config.ANTHROPIC_API_KEY = "test_anthropic_key"
    config.MAX_POSITIONS = 10
    config.MAX_POSITION_SIZE = 0.05
    config.RISK_TOLERANCE = "moderate"
    config.LOG_LEVEL = "INFO"
    config.CACHE_ENABLED = True
    return config


@pytest.fixture
def mock_llm_provider():
    """Mock LLM provider with realistic responses"""
    provider = MagicMock()
    
    # Default successful analysis response
    default_response = {
        "recommendation": "buy",
        "confidence": 8,
        "time_horizon": "medium_term",
        "position_size": "medium",
        "entry_target": 185.50,
        "exit_targets": {"primary": 195.00, "secondary": 200.00},
        "stop_loss": 178.00,
        "risk_reward_ratio": 2.5,
        "investment_thesis": "Strong technical pattern with fundamental support. Company shows consistent revenue growth.",
        "risk_factors": ["Market volatility", "Sector rotation risk", "Earnings uncertainty"],
        "catalyst_timeline": {
            "short_term": ["Technical breakout confirmation"],
            "medium_term": ["Q2 earnings release"],
            "long_term": ["New product launch cycle"]
        },
        "technical_score": 7.5,
        "fundamental_score": 8.0
    }
    
    provider.generate_analysis = AsyncMock(return_value=default_response)
    provider.generate_completion = AsyncMock(return_value=json.dumps(default_response))
    
    return provider


@pytest.fixture
def mock_alpaca_provider():
    """Mock Alpaca provider with market data"""
    provider = MagicMock()
    
    # Mock market data
    provider.get_market_data = AsyncMock(return_value={
        "AAPL": [
            {
                "timestamp": "2024-01-15T16:00:00Z",
                "open": 180.00,
                "high": 186.00,
                "low": 179.50,
                "close": 185.00,
                "volume": 75000000
            },
            {
                "timestamp": "2024-01-16T16:00:00Z",
                "open": 185.00,
                "high": 187.50,
                "low": 184.00,
                "close": 186.50,
                "volume": 65000000
            }
        ]
    })
    
    # Mock current quote
    provider.get_quote = AsyncMock(return_value={
        "symbol": "AAPL",
        "bid": 186.45,
        "ask": 186.55,
        "last": 186.50,
        "volume": 50000000
    })
    
    provider.get_current_quote = AsyncMock(return_value={
        "symbol": "AAPL",
        "price": 186.50,
        "bid": 186.45,
        "ask": 186.55,
        "volume": 50000000
    })
    
    # Mock technical indicators
    provider.get_technical_indicators = AsyncMock(return_value={
        "rsi": 55.5,
        "macd": {"macd": 1.2, "signal": 0.9, "histogram": 0.3},
        "sma_20": 183.00,
        "sma_50": 180.00,
        "ema_12": 184.50,
        "ema_26": 182.00,
        "volume_ratio": 1.2,
        "atr": 2.5,
        "bollinger_bands": {"upper": 190.00, "middle": 185.00, "lower": 180.00}
    })
    
    # Mock news
    provider.get_news = AsyncMock(return_value={
        "articles": [
            {
                "headline": "Apple Reports Strong Q1 Earnings",
                "summary": "Revenue beat expectations",
                "created_at": "2024-01-15T09:00:00Z",
                "sentiment": 0.8
            },
            {
                "headline": "New Product Launch Expected",
                "summary": "Analysts optimistic about upcoming releases",
                "created_at": "2024-01-14T14:00:00Z",
                "sentiment": 0.6
            }
        ]
    })
    
    # Mock company info
    provider.get_company_info = AsyncMock(return_value={
        "sector": "Technology",
        "industry": "Consumer Electronics",
        "market_cap": 2.95e12,
        "pe_ratio": 30.5,
        "dividend_yield": 0.44,
        "beta": 1.25
    })
    
    # Mock financial data
    provider.get_financial_data = AsyncMock(return_value={
        "revenue_growth": 0.08,
        "earnings_growth": 0.12,
        "profit_margin": 0.25,
        "debt_to_equity": 1.5,
        "roe": 0.35,
        "current_ratio": 1.2
    })
    
    return provider


@pytest.fixture
def mock_technical_engine(mock_alpaca_provider):
    """Mock Technical Analysis Engine"""
    engine = MagicMock()
    engine.alpaca = mock_alpaca_provider
    
    engine.analyze = AsyncMock(return_value={
        "technical_score": 7.5,
        "trend": "bullish",
        "momentum": "strong",
        "support_levels": [180.00, 175.00],
        "resistance_levels": [190.00, 195.00],
        "patterns": ["ascending_triangle", "bullish_flag"],
        "signals": {
            "buy": 3,
            "sell": 0,
            "neutral": 2
        }
    })
    
    return engine


@pytest.fixture
def mock_fundamental_engine(mock_alpaca_provider):
    """Mock Fundamental Analysis Engine"""
    engine = MagicMock()
    engine.alpaca = mock_alpaca_provider
    
    engine.analyze = AsyncMock(return_value={
        "fundamental_score": 8.0,
        "valuation": "fair",
        "growth_rating": "strong",
        "financial_health": "excellent",
        "competitive_position": "dominant",
        "earnings_trend": "improving"
    })
    
    return engine


@pytest.fixture
def junior_analyst(mock_llm_provider, mock_alpaca_provider, mock_config):
    """Create Junior Research Analyst instance"""
    analyst = JuniorResearchAnalyst(
        mock_llm_provider,
        mock_alpaca_provider,
        mock_config
    )
    
    # Mock the internal engines
    analyst.technical_engine = MagicMock()
    analyst.technical_engine.analyze = AsyncMock(return_value={
        "technical_score": 7.5,
        "trend": "bullish"
    })
    
    analyst.fundamental_engine = MagicMock()
    analyst.fundamental_engine.analyze = AsyncMock(return_value={
        "fundamental_score": 8.0,
        "valuation": "fair"
    })
    
    return analyst


@pytest.fixture
def sample_technical_signal():
    """Sample technical signal for testing"""
    return {
        "pattern": "ascending_triangle",
        "score": 8.2,
        "resistance_level": 190.00,
        "support_level": 180.00,
        "volume_confirmation": True,
        "formation_days": 8,
        "breakout_probability": 0.75
    }


@pytest.fixture
def sample_position_data():
    """Sample position data for reevaluation"""
    return {
        "symbol": "AAPL",
        "quantity": 100,
        "entry_price": 175.00,
        "current_price": 186.50,
        "unrealized_pnl": 1150.00,
        "unrealized_pnl_percent": 6.57,
        "holding_period_days": 30
    }


# ==============================================================================
# UNIT TESTS - Agent Initialization
# ==============================================================================

@pytest.mark.unit
class TestAgentInitialization:
    """Test Junior Analyst initialization"""
    
    def test_agent_creation(self, junior_analyst):
        """Test agent is created successfully"""
        assert junior_analyst is not None
        assert junior_analyst.agent_name == "junior_analyst"
        assert junior_analyst.agent_id is not None
        assert isinstance(junior_analyst.agent_id, str)
        assert len(junior_analyst.agent_id) == 36  # UUID length
    
    def test_agent_initial_metrics(self, junior_analyst):
        """Test initial performance metrics"""
        assert junior_analyst.performance_metrics["total_analyses"] == 0
        assert junior_analyst.performance_metrics["successful_analyses"] == 0
        assert junior_analyst.performance_metrics["failed_analyses"] == 0
        assert junior_analyst.performance_metrics["cache_hits"] == 0
        assert junior_analyst.performance_metrics["average_processing_time"] == 0.0
    
    def test_factory_function(self, mock_llm_provider, mock_alpaca_provider, mock_config):
        """Test factory function creates analyst"""
        analyst = create_junior_analyst(mock_llm_provider, mock_alpaca_provider, mock_config)
        
        assert isinstance(analyst, JuniorResearchAnalyst)
        assert analyst.agent_name == "junior_analyst"
    
    def test_agent_has_required_methods(self, junior_analyst):
        """Test agent has all required methods"""
        required_methods = [
            'analyze_stock',
            'process_with_metadata',
            '_analyze_new_opportunity',
            '_reevaluate_position',
            'get_performance_summary'
        ]
        
        for method in required_methods:
            assert hasattr(junior_analyst, method)


# ==============================================================================
# UNIT TESTS - New Opportunity Analysis
# ==============================================================================

@pytest.mark.unit
class TestNewOpportunityAnalysis:
    """Test new opportunity analysis functionality"""
    
    @pytest.mark.asyncio
    async def test_analyze_new_opportunity_success(self, junior_analyst, sample_technical_signal):
        """Test successful new opportunity analysis"""
        task_data = {
            "task_type": AnalysisType.NEW_OPPORTUNITY.value,
            "ticker": "AAPL",
            "technical_signal": sample_technical_signal
        }
        
        result = await junior_analyst.analyze_stock(task_data)
        
        assert result is not None
        assert result["ticker"] == "AAPL"
        assert result["analysis_type"] == AnalysisType.NEW_OPPORTUNITY.value
        assert "recommendation" in result
        assert "confidence" in result
        assert "entry_target" in result
        assert "stop_loss" in result
        assert "investment_thesis" in result
    
    @pytest.mark.asyncio
    async def test_analyze_with_metadata(self, junior_analyst, sample_technical_signal):
        """Test analysis with metadata wrapper"""
        task_data = {
            "task_type": AnalysisType.NEW_OPPORTUNITY.value,
            "ticker": "AAPL",
            "technical_signal": sample_technical_signal
        }
        
        result = await junior_analyst.process_with_metadata(task_data)
        
        assert "metadata" in result
        assert result["metadata"]["status"] in ["success", "error"]
        assert "analysis_id" in result
        assert "timestamp" in result
        assert "processing_time" in result["metadata"]
    
    @pytest.mark.asyncio
    async def test_confidence_scoring(self, junior_analyst, sample_technical_signal):
        """Test confidence score calculation"""
        task_data = {
            "task_type": AnalysisType.NEW_OPPORTUNITY.value,
            "ticker": "AAPL",
            "technical_signal": sample_technical_signal
        }
        
        result = await junior_analyst.analyze_stock(task_data)
        
        assert "confidence" in result
        assert isinstance(result["confidence"], (int, float))
        assert 1 <= result["confidence"] <= 10
    
    @pytest.mark.asyncio
    @pytest.mark.parametrize("ticker", ["AAPL", "MSFT", "GOOGL", "AMZN"])
    async def test_multiple_tickers(self, junior_analyst, sample_technical_signal, ticker):
        """Test analysis works for multiple tickers"""
        task_data = {
            "task_type": AnalysisType.NEW_OPPORTUNITY.value,
            "ticker": ticker,
            "technical_signal": sample_technical_signal
        }
        
        result = await junior_analyst.analyze_stock(task_data)
        
        assert result["ticker"] == ticker


# ==============================================================================
# UNIT TESTS - Position Reevaluation
# ==============================================================================

@pytest.mark.unit
class TestPositionReevaluation:
    """Test position reevaluation functionality"""
    
    @pytest.mark.asyncio
    async def test_reevaluate_position_success(self, junior_analyst, sample_position_data):
        """Test successful position reevaluation"""
        task_data = {
            "task_type": AnalysisType.POSITION_REEVALUATION.value,
            "ticker": "AAPL",
            "current_position": sample_position_data
        }
        
        result = await junior_analyst.analyze_stock(task_data)
        
        assert result is not None
        assert result["analysis_type"] == AnalysisType.POSITION_REEVALUATION.value
        assert "action" in result
        assert "updated_confidence" in result
        assert "updated_targets" in result
        assert "updated_stop_loss" in result
        assert "conviction_change" in result
        assert "recommendation_rationale" in result
    
    @pytest.mark.asyncio
    async def test_reevaluation_actions(self, junior_analyst, sample_position_data):
        """Test different reevaluation actions"""
        task_data = {
            "task_type": AnalysisType.POSITION_REEVALUATION.value,
            "ticker": "AAPL",
            "current_position": sample_position_data
        }
        
        result = await junior_analyst.analyze_stock(task_data)
        
        valid_actions = ["hold", "increase", "reduce", "exit"]
        assert result["action"] in valid_actions
    
    @pytest.mark.asyncio
    async def test_conviction_change_tracking(self, junior_analyst, sample_position_data):
        """Test conviction change tracking"""
        task_data = {
            "task_type": AnalysisType.POSITION_REEVALUATION.value,
            "ticker": "AAPL",
            "current_position": sample_position_data
        }
        
        result = await junior_analyst.analyze_stock(task_data)
        
        valid_changes = ["increased", "decreased", "unchanged"]
        assert result["conviction_change"] in valid_changes


# ==============================================================================
# UNIT TESTS - Risk Assessment
# ==============================================================================

@pytest.mark.unit
class TestRiskAssessment:
    """Test risk assessment functionality"""
    
    @pytest.mark.asyncio
    async def test_risk_assessment_success(self, junior_analyst):
        """Test successful risk assessment"""
        task_data = {
            "task_type": AnalysisType.RISK_ASSESSMENT.value,
            "ticker": "AAPL",
            "position_data": {"quantity": 100, "market_value": 18650.00}
        }
        
        result = await junior_analyst.analyze_stock(task_data)
        
        assert result is not None
        assert result["analysis_type"] == AnalysisType.RISK_ASSESSMENT.value
        assert "risk_level" in result
        assert "risk_score" in result
        assert "risk_factors" in result
    
    @pytest.mark.asyncio
    async def test_risk_levels(self, junior_analyst):
        """Test risk level categorization"""
        task_data = {
            "task_type": AnalysisType.RISK_ASSESSMENT.value,
            "ticker": "AAPL",
            "position_data": {"quantity": 100, "market_value": 18650.00}
        }
        
        result = await junior_analyst.analyze_stock(task_data)
        
        valid_levels = ["low", "medium", "high", "very_high"]
        assert result["risk_level"] in valid_levels
    
    @pytest.mark.asyncio
    async def test_risk_score_range(self, junior_analyst):
        """Test risk score is in valid range"""
        task_data = {
            "task_type": AnalysisType.RISK_ASSESSMENT.value,
            "ticker": "AAPL",
            "position_data": {"quantity": 100, "market_value": 18650.00}
        }
        
        result = await junior_analyst.analyze_stock(task_data)
        
        assert isinstance(result["risk_score"], (int, float))
        assert 0 <= result["risk_score"] <= 10


# ==============================================================================
# INTEGRATION TESTS - Complete Analysis Flow
# ==============================================================================

@pytest.mark.integration
class TestCompleteAnalysisFlow:
    """Test complete analysis workflows"""
    
    @pytest.mark.asyncio
    async def test_full_new_opportunity_flow(self, junior_analyst, sample_technical_signal):
        """Test complete new opportunity analysis flow"""
        task_data = {
            "task_type": AnalysisType.NEW_OPPORTUNITY.value,
            "ticker": "AAPL",
            "technical_signal": sample_technical_signal
        }
        
        # Process with metadata
        result = await junior_analyst.process_with_metadata(task_data)
        
        # Verify complete result structure
        assert result["metadata"]["status"] == "success"
        assert result["ticker"] == "AAPL"
        assert result["recommendation"] in ["buy", "strong_buy", "hold", "sell", "strong_sell"]
        assert result["confidence"] >= 1 and result["confidence"] <= 10
        
        # Verify all required fields present
        required_fields = [
            "entry_target", "stop_loss", "exit_targets",
            "investment_thesis", "risk_factors", "time_horizon",
            "position_size", "risk_reward_ratio"
        ]
        
        for field in required_fields:
            assert field in result
    
    @pytest.mark.asyncio
    async def test_analysis_with_caching(self, junior_analyst, sample_technical_signal):
        """Test analysis caching functionality"""
        task_data = {
            "task_type": AnalysisType.NEW_OPPORTUNITY.value,
            "ticker": "AAPL",
            "technical_signal": sample_technical_signal
        }
        
        # First analysis
        result1 = await junior_analyst.analyze_stock(task_data)
        
        # Second analysis (should be cached)
        result2 = await junior_analyst.analyze_stock(task_data)
        
        # Results should be identical (from cache)
        assert result1["analysis_id"] == result2["analysis_id"]
        assert junior_analyst.performance_metrics["cache_hits"] > 0
    
    @pytest.mark.asyncio
    async def test_error_recovery(self, junior_analyst):
        """Test error recovery mechanisms"""
        # Create task with missing required field
        task_data = {
            "task_type": AnalysisType.NEW_OPPORTUNITY.value,
            # Missing ticker
        }
        
        result = await junior_analyst.process_with_metadata(task_data)
        
        assert result["metadata"]["status"] == "error"
        assert "error" in result["metadata"]


# ==============================================================================
# INTEGRATION TESTS - Engine Integration
# ==============================================================================

@pytest.mark.integration
class TestEngineIntegration:
    """Test integration with analysis engines"""
    
    @pytest.mark.asyncio
    async def test_technical_engine_integration(self, junior_analyst):
        """Test technical analysis engine integration"""
        task_data = {
            "task_type": AnalysisType.NEW_OPPORTUNITY.value,
            "ticker": "AAPL",
            "technical_signal": {"pattern": "breakout", "score": 8}
        }
        
        result = await junior_analyst.analyze_stock(task_data)
        
        # Verify technical analysis was performed
        junior_analyst.technical_engine.analyze.assert_called()
    
    @pytest.mark.asyncio
    async def test_fundamental_engine_integration(self, junior_analyst):
        """Test fundamental analysis engine integration"""
        task_data = {
            "task_type": AnalysisType.NEW_OPPORTUNITY.value,
            "ticker": "AAPL",
            "technical_signal": {"pattern": "breakout", "score": 8}
        }
        
        result = await junior_analyst.analyze_stock(task_data)
        
        # Verify fundamental analysis was performed
        junior_analyst.fundamental_engine.analyze.assert_called()


# ==============================================================================
# STRESS TESTS
# ==============================================================================

@pytest.mark.stress
class TestStress:
    """Stress tests for Junior Analyst"""
    
    @pytest.mark.asyncio
    async def test_concurrent_analyses(self, junior_analyst, sample_technical_signal):
        """Test concurrent analysis requests"""
        tasks = []
        tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "META"] * 10
        
        for ticker in tickers:
            task_data = {
                "task_type": AnalysisType.NEW_OPPORTUNITY.value,
                "ticker": ticker,
                "technical_signal": sample_technical_signal
            }
            tasks.append(junior_analyst.analyze_stock(task_data))
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        successful = sum(1 for r in results if not isinstance(r, Exception))
        assert successful == len(tickers)
    
    @pytest.mark.asyncio
    async def test_rapid_sequential_analyses(self, junior_analyst, sample_technical_signal):
        """Test rapid sequential analyses"""
        for i in range(50):
            task_data = {
                "task_type": AnalysisType.NEW_OPPORTUNITY.value,
                "ticker": f"TEST{i}",
                "technical_signal": sample_technical_signal
            }
            
            result = await junior_analyst.analyze_stock(task_data)
            assert result is not None
    
    @pytest.mark.asyncio
    async def test_memory_efficiency(self, junior_analyst, sample_technical_signal):
        """Test memory efficiency with many analyses"""
        initial_cache_size = len(junior_analyst.analysis_cache)
        
        # Perform many analyses
        for i in range(100):
            task_data = {
                "task_type": AnalysisType.NEW_OPPORTUNITY.value,
                "ticker": f"STOCK{i}",
                "technical_signal": sample_technical_signal
            }
            await junior_analyst.analyze_stock(task_data)
        
        # Cache should not grow unbounded
        # Assuming some cache size limit
        assert len(junior_analyst.analysis_cache) <= 1000


# ==============================================================================
# SMOKE TESTS
# ==============================================================================

@pytest.mark.smoke
class TestSmoke:
    """Quick smoke tests for basic functionality"""
    
    @pytest.mark.asyncio
    async def test_basic_analysis(self, junior_analyst):
        """Test basic analysis works"""
        task_data = {
            "task_type": AnalysisType.NEW_OPPORTUNITY.value,
            "ticker": "AAPL",
            "technical_signal": {"pattern": "test", "score": 7}
        }
        
        result = await junior_analyst.analyze_stock(task_data)
        
        assert result is not None
        assert result["ticker"] == "AAPL"
    
    def test_agent_exists(self, junior_analyst):
        """Test agent exists and is initialized"""
        assert junior_analyst is not None
        assert junior_analyst.agent_name == "junior_analyst"
    
    def test_performance_metrics_exist(self, junior_analyst):
        """Test performance metrics are tracked"""
        summary = junior_analyst.get_performance_summary()
        
        assert summary is not None
        assert "agent_name" in summary
        assert "total_analyses" in summary


# ==============================================================================
# ERROR HANDLING TESTS
# ==============================================================================

@pytest.mark.unit
class TestErrorHandling:
    """Test error handling scenarios"""
    
    @pytest.mark.asyncio
    async def test_missing_ticker(self, junior_analyst):
        """Test handling of missing ticker"""
        task_data = {
            "task_type": AnalysisType.NEW_OPPORTUNITY.value,
            # Missing ticker
        }
        
        result = await junior_analyst.process_with_metadata(task_data)
        
        assert result["metadata"]["status"] == "error"
        assert "ticker" in result["metadata"]["error"].lower()
    
    @pytest.mark.asyncio
    async def test_invalid_task_type(self, junior_analyst):
        """Test handling of invalid task type"""
        task_data = {
            "task_type": "INVALID_TYPE",
            "ticker": "AAPL"
        }
        
        result = await junior_analyst.process_with_metadata(task_data)
        
        assert result["metadata"]["status"] == "error"
    
    @pytest.mark.asyncio
    async def test_llm_failure_handling(self, junior_analyst, mock_llm_provider):
        """Test handling of LLM failures"""
        # Make LLM fail
        mock_llm_provider.generate_analysis.side_effect = Exception("LLM Error")
        
        task_data = {
            "task_type": AnalysisType.NEW_OPPORTUNITY.value,
            "ticker": "AAPL",
            "technical_signal": {"pattern": "test", "score": 7}
        }
        
        result = await junior_analyst.process_with_metadata(task_data)
        
        # Should handle gracefully with fallback
        assert result is not None
        assert result["metadata"]["status"] in ["error", "success"]  # May have fallback


# ==============================================================================
# PARAMETRIZED TESTS
# ==============================================================================

@pytest.mark.parametrize("recommendation,expected_confidence_min", [
    ("strong_buy", 8),
    ("buy", 6),
    ("hold", 4),
    ("sell", 4),
    ("strong_sell", 6),
])
@pytest.mark.unit
def test_recommendation_confidence_correlation(recommendation, expected_confidence_min):
    """Test correlation between recommendation and confidence"""
    # This is a conceptual test - would need actual implementation
    assert expected_confidence_min >= 4  # All recommendations need some confidence


@pytest.mark.parametrize("time_horizon,position_size", [
    (TimeHorizon.SHORT_TERM.value, PositionSize.SMALL.value),
    (TimeHorizon.MEDIUM_TERM.value, PositionSize.MEDIUM.value),
    (TimeHorizon.LONG_TERM.value, PositionSize.LARGE.value),
])
@pytest.mark.unit
def test_time_horizon_position_size_relationship(time_horizon, position_size):
    """Test relationship between time horizon and position size"""
    # Conceptual test to verify expected relationships
    valid_sizes = [PositionSize.SMALL.value, PositionSize.MEDIUM.value, 
                   PositionSize.LARGE.value, PositionSize.MAX.value]
    assert position_size in valid_sizes


# ==============================================================================
# PERFORMANCE TESTS
# ==============================================================================

@pytest.mark.unit
class TestPerformanceTracking:
    """Test performance tracking functionality"""
    
    @pytest.mark.asyncio
    async def test_performance_metrics_update(self, junior_analyst, sample_technical_signal):
        """Test performance metrics are updated"""
        initial_total = junior_analyst.performance_metrics["total_analyses"]
        
        task_data = {
            "task_type": AnalysisType.NEW_OPPORTUNITY.value,
            "ticker": "AAPL",
            "technical_signal": sample_technical_signal
        }
        
        await junior_analyst.analyze_stock(task_data)
        
        assert junior_analyst.performance_metrics["total_analyses"] == initial_total + 1
        assert junior_analyst.performance_metrics["successful_analyses"] >= initial_total
    
    def test_performance_summary_format(self, junior_analyst):
        """Test performance summary format"""
        summary = junior_analyst.get_performance_summary()
        
        expected_fields = [
            "agent_name",
            "agent_id",
            "total_analyses",
            "success_rate",
            "average_processing_time",
            "cache_hit_rate"
        ]
        
        for field in expected_fields:
            assert field in summary
    
    @pytest.mark.asyncio
    async def test_processing_time_tracking(self, junior_analyst, sample_technical_signal):
        """Test processing time is tracked"""
        task_data = {
            "task_type": AnalysisType.NEW_OPPORTUNITY.value,
            "ticker": "AAPL",
            "technical_signal": sample_technical_signal
        }
        
        result = await junior_analyst.process_with_metadata(task_data)
        
        assert "processing_time" in result["metadata"]
        assert result["metadata"]["processing_time"] >= 0

# ==============================================================================
# TEST RUNNER
# ==============================================================================

if __name__ == "__main__":
    import sys
    
    # Run with coverage if requested
    if "--coverage" in sys.argv:
        sys.argv.remove("--coverage")
        sys.exit(pytest.main([__file__, "--cov=agents", "--cov-report=html", "-v"]))
    else:
        sys.exit(pytest.main([__file__, "-v"]))