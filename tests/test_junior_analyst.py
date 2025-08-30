# test_junior_analyst.py
"""
Junior Research Analyst Test Suite
Comprehensive testing for the Junior Research Analyst Agent
Optimized for macOS M2 Max development
"""

import pytest
import asyncio
import json
import tempfile
import os
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch
from pathlib import Path

# Import the agent components
from agents.junior_research_analyst import (
    JuniorResearchAnalyst, AnalysisType, RecommendationType,
    TimeHorizon, RiskLevel, PositionSize, TechnicalAnalysisEngine,
    FundamentalAnalysisEngine, create_junior_analyst
)


class TestJuniorResearchAnalyst:
    """Test suite for Junior Research Analyst functionality"""
    
    @pytest.fixture
    def mock_config(self):
        """Mock configuration object"""
        config = Mock()
        config.ANTHROPIC_API_KEY = "test_anthropic_key"
        config.MAX_POSITIONS = 10
        config.LOG_LEVEL = "INFO"
        return config
    
    @pytest.fixture
    def mock_llm_provider(self):
        """Mock LLM provider with realistic responses"""
        llm = Mock()
        
        # Mock successful JSON response for new opportunity
        successful_response = {
            "recommendation": "buy",
            "confidence": 8,
            "time_horizon": "medium_term",
            "position_size": "medium",
            "entry_target": 185.50,
            "exit_targets": {"primary": 195.00, "secondary": 200.00},
            "stop_loss": 178.00,
            "risk_reward_ratio": 2.5,
            "investment_thesis": "Strong technical pattern with fundamental support",
            "risk_factors": ["Market volatility", "Sector headwinds"],
            "catalyst_timeline": {
                "short_term": ["Technical breakout"],
                "medium_term": ["Earnings beat"],
                "long_term": ["Market expansion"]
            }
        }
        
        llm.generate_completion = AsyncMock(return_value=json.dumps(successful_response))
        return llm
    
    @pytest.fixture
    def mock_alpaca_provider(self):
        """Mock Alpaca provider with realistic market data"""
        alpaca = Mock()
        
        # Mock market data
        alpaca.get_market_data = AsyncMock(return_value={
            "AAPL": [
                {
                    "timestamp": "2024-01-15T16:00:00Z",
                    "open": 183.0,
                    "high": 186.0,
                    "low": 182.0,
                    "close": 184.5,
                    "volume": 1200000
                }
            ]
        })
        
        # Mock current quote
        alpaca.get_current_quote = AsyncMock(return_value={
            "symbol": "AAPL",
            "price": 184.50,
            "bid": 184.45,
            "ask": 184.55,
            "volume": 1250000
        })
        
        # Mock technical indicators
        alpaca.get_technical_indicators = AsyncMock(return_value={
            "rsi": 65.5,
            "moving_averages": {"sma_20": 182.30},
            "trend": "bullish"
        })
        
        # Mock news
        alpaca.get_news = AsyncMock(return_value=[
            {
                "headline": "Strong earnings reported",
                "created_at": "2024-01-15T09:00:00Z"
            }
        ])
        
        # Mock company info
        alpaca.get_company_info = AsyncMock(return_value={
            "sector": "Technology",
            "market_cap": 2500000000000
        })
        
        # Mock financial data
        alpaca.get_financial_data = AsyncMock(return_value={
            "pe_ratio": 28.5,
            "debt_to_equity": 0.25
        })
        
        return alpaca
    
    @pytest.fixture
    def analyst(self, mock_llm_provider, mock_alpaca_provider, mock_config):
        """Create analyst instance for testing"""
        return JuniorResearchAnalyst(mock_llm_provider, mock_alpaca_provider, mock_config)
    
    def test_agent_initialization(self, analyst):
        """Test agent initialization"""
        assert analyst.agent_name == "junior_research_analyst"
        assert analyst.agent_id is not None
        assert len(analyst.agent_id) == 36  # UUID length
        assert analyst.performance_metrics["total_analyses"] == 0
        assert analyst.performance_metrics["successful_analyses"] == 0
        assert analyst.performance_metrics["failed_analyses"] == 0
    
    def test_factory_function(self, mock_llm_provider, mock_alpaca_provider, mock_config):
        """Test factory function"""
        analyst = create_junior_analyst(mock_llm_provider, mock_alpaca_provider, mock_config)
        assert isinstance(analyst, JuniorResearchAnalyst)
        assert analyst.agent_name == "junior_research_analyst"
    
    @pytest.mark.asyncio
    async def test_new_opportunity_analysis_success(self, analyst):
        """Test successful new opportunity analysis"""
        
        task_data = {
            "task_type": AnalysisType.NEW_OPPORTUNITY.value,
            "ticker": "AAPL",
            "technical_signal": {
                "pattern": "ascending_triangle",
                "score": 8.2,
                "resistance_level": 185.00
            }
        }
        
        result = await analyst.analyze_stock(task_data)
        
        # Verify successful result structure
        assert result["metadata"]["status"] == "success"
        assert result["ticker"] == "AAPL"
        assert result["analysis_type"] == AnalysisType.NEW_OPPORTUNITY.value
        assert "recommendation" in result
        assert "confidence" in result
        assert "entry_target" in result
        assert "exit_targets" in result
        assert "stop_loss" in result
        assert "investment_thesis" in result
        assert "risk_factors" in result
        
        # Verify confidence is valid range
        assert 1 <= result["confidence"] <= 10
        
        # Verify targets are numerical
        assert isinstance(result["entry_target"], (int, float))
        assert isinstance(result["stop_loss"], (int, float))
        
        # Verify performance metrics updated
        assert analyst.performance_metrics["total_analyses"] == 1
        assert analyst.performance_metrics["successful_analyses"] == 1
        assert analyst.performance_metrics["failed_analyses"] == 0
    
    @pytest.mark.asyncio
    async def test_position_reevaluation_success(self, analyst, mock_llm_provider):
        """Test successful position reevaluation"""
        
        # Mock reevaluation response
        reevaluation_response = {
            "action": "increase",
            "confidence": 7,
            "targets": {"primary": 190.00, "secondary": 195.00},
            "stop_loss": 180.00,
            "conviction_change": "increased",
            "new_developments": "Positive earnings surprise",
            "risk_assessment": "Risk decreased due to strong fundamentals",
            "rationale": "Strong momentum supports position increase"
        }
        
        mock_llm_provider.generate_completion = AsyncMock(
            return_value=json.dumps(reevaluation_response)
        )
        
        task_data = {
            "task_type": AnalysisType.POSITION_REEVALUATION.value,
            "ticker": "AAPL",
            "current_position": {
                "quantity": 100,
                "entry_price": 180.00,
                "current_price": 184.50
            }
        }
        
        result = await analyst.analyze_stock(task_data)
        
        # Verify successful reevaluation structure
        assert result["metadata"]["status"] == "success"
        assert result["ticker"] == "AAPL"
        assert result["analysis_type"] == AnalysisType.POSITION_REEVALUATION.value
        assert result["action"] == "increase"
        assert result["updated_confidence"] == 7
        assert "updated_targets" in result
        assert "updated_stop_loss" in result
        assert "new_developments" in result
        assert "recommendation_rationale" in result
    
    @pytest.mark.asyncio
    async def test_risk_assessment_success(self, analyst, mock_llm_provider):
        """Test successful risk assessment"""
        
        # Mock risk assessment response
        risk_response = {
            "risk_level": "medium",
            "risk_score": 6,
            "risk_factors": ["Market volatility", "Sector rotation"],
            "volatility_assessment": "Moderate volatility expected",
            "downside_scenarios": ["Support breakdown", "Market correction"],
            "risk_mitigation": ["Stop loss", "Position sizing"]
        }
        
        mock_llm_provider.generate_completion = AsyncMock(
            return_value=json.dumps(risk_response)
        )
        
        task_data = {
            "task_type": AnalysisType.RISK_ASSESSMENT.value,
            "ticker": "AAPL",
            "position_data": {
                "quantity": 100,
                "market_value": 18450.00
            }
        }
        
        result = await analyst.analyze_stock(task_data)
        
        # Verify successful risk assessment structure
        assert result["metadata"]["status"] == "success"
        assert result["ticker"] == "AAPL"
        assert result["analysis_type"] == AnalysisType.RISK_ASSESSMENT.value
        assert result["risk_level"] == "medium"
        assert result["risk_score"] == 6
        assert isinstance(result["risk_factors"], list)
        assert isinstance(result["downside_scenarios"], list)
        assert isinstance(result["risk_mitigation"], list)
    
    @pytest.mark.asyncio
    async def test_invalid_task_type(self, analyst):
        """Test handling of invalid task type"""
        
        task_data = {
            "task_type": "invalid_type",
            "ticker": "AAPL"
        }
        
        result = await analyst.analyze_stock(task_data)
        
        assert result["metadata"]["status"] == "error"
        assert "Invalid task_type" in result["metadata"]["error"]
    
    @pytest.mark.asyncio
    async def test_missing_ticker(self, analyst):
        """Test handling of missing ticker"""
        
        task_data = {
            "task_type": AnalysisType.NEW_OPPORTUNITY.value
        }
        
        result = await analyst.analyze_stock(task_data)
        
        assert result["metadata"]["status"] == "error"
        assert "Missing required field: ticker" in result["metadata"]["error"]
    
    @pytest.mark.asyncio
    async def test_empty_ticker(self, analyst):
        """Test handling of empty ticker"""
        
        task_data = {
            "task_type": AnalysisType.NEW_OPPORTUNITY.value,
            "ticker": ""
        }
        
        result = await analyst.analyze_stock(task_data)
        
        assert result["metadata"]["status"] == "error"
        assert "Ticker must be a non-empty string" in result["metadata"]["error"]
    
    @pytest.mark.asyncio
    async def test_llm_failure_fallback(self, analyst, mock_llm_provider):
        """Test fallback behavior when LLM fails"""
        
        # Mock LLM failure
        mock_llm_provider.generate_completion = AsyncMock(
            side_effect=Exception("LLM API Error")
        )
        
        task_data = {
            "task_type": AnalysisType.NEW_OPPORTUNITY.value,
            "ticker": "AAPL",
            "technical_signal": {"pattern": "breakout", "score": 8}
        }
        
        result = await analyst.analyze_stock(task_data)
        
        # Should return fallback analysis
        assert result["metadata"]["status"] == "fallback"
        assert result["ticker"] == "AAPL"
        assert "recommendation" in result
        assert "investment_thesis" in result
        assert "fallback_reason" in result["metadata"]
    
    @pytest.mark.asyncio
    async def test_market_data_failure(self, analyst, mock_alpaca_provider):
        """Test handling of market data failure"""
        
        # Mock market data failure
        mock_alpaca_provider.get_market_data = AsyncMock(
            side_effect=Exception("Market data API error")
        )
        
        task_data = {
            "task_type": AnalysisType.NEW_OPPORTUNITY.value,
            "ticker": "AAPL",
            "technical_signal": {"pattern": "breakout", "score": 8}
        }
        
        result = await analyst.analyze_stock(task_data)
        
        # Should return error due to market data failure
        assert result["metadata"]["status"] == "error"
        assert "Failed to gather market data" in result["metadata"]["error"]
    
    @pytest.mark.asyncio
    async def test_caching_mechanism(self, analyst):
        """Test analysis result caching"""
        
        task_data = {
            "task_type": AnalysisType.NEW_OPPORTUNITY.value,
            "ticker": "AAPL",
            "technical_signal": {"pattern": "breakout", "score": 8}
        }
        
        # First analysis
        result1 = await analyst.analyze_stock(task_data)
        first_analysis_id = result1["analysis_id"]
        
        # Second analysis (should be cached)
        result2 = await analyst.analyze_stock(task_data)
        second_analysis_id = result2["analysis_id"]
        
        # Should return same cached result
        assert first_analysis_id == second_analysis_id
        assert analyst.performance_metrics["cache_hits"] == 1
    
    def test_performance_summary(self, analyst):
        """Test performance summary generation"""
        
        # Manually update some metrics for testing
        analyst.performance_metrics["total_analyses"] = 5
        analyst.performance_metrics["successful_analyses"] = 4
        analyst.performance_metrics["failed_analyses"] = 1
        analyst.performance_metrics["cache_hits"] = 2
        analyst.performance_metrics["average_processing_time"] = 1.5
        analyst.performance_metrics["last_activity"] = datetime.now()
        
        summary = analyst.get_performance_summary()
        
        assert summary["agent_name"] == "junior_research_analyst"
        assert summary["total_analyses"] == 5
        assert summary["success_rate"] == "80.0%"
        assert summary["average_processing_time"] == "1.50s"
        assert summary["cache_hit_rate"] == "40.0%"
        assert summary["last_activity"] is not None
    
    def test_markdown_report_generation(self, analyst):
        """Test markdown report generation"""
        
        # Sample analysis result
        analysis_result = {
            "analysis_id": "test-123",
            "ticker": "AAPL",
            "timestamp": "2024-01-15T10:00:00Z",
            "analysis_type": AnalysisType.NEW_OPPORTUNITY.value,
            "recommendation": "buy",
            "confidence": 8,
            "time_horizon": "medium_term",
            "position_size": "medium",
            "entry_target": 185.50,
            "exit_targets": {"primary": 195.00, "secondary": 200.00},
            "stop_loss": 178.00,
            "risk_reward_ratio": 2.5,
            "investment_thesis": "Strong technical pattern with fundamental support",
            "risk_factors": ["Market volatility", "Sector headwinds"],
            "catalyst_timeline": {
                "short_term": ["Technical breakout"],
                "medium_term": ["Earnings beat"],
                "long_term": ["Market expansion"]
            },
            "technical_summary": "Bullish trend confirmed",
            "fundamental_summary": "Strong fundamentals",
            "market_context": {"current_price": 184.50, "rsi": 65.5, "trend": "bullish"},
            "metadata": {"status": "success", "data_quality": "good", "analysis_version": "1.0"}
        }
        
        markdown_report = analyst.generate_markdown_report(analysis_result)
        
        # Verify markdown structure
        assert "# Stock Analysis Report: AAPL" in markdown_report
        assert "**Recommendation:** BUY" in markdown_report
        assert "**Confidence:** 8/10" in markdown_report
        assert "## Investment Thesis" in markdown_report
        assert "## Risk Factors" in markdown_report
        assert "Strong technical pattern with fundamental support" in markdown_report
        assert "- Market volatility" in markdown_report
        assert "- Sector headwinds" in markdown_report
    
    def test_markdown_report_error_case(self, analyst):
        """Test markdown report generation for error cases"""
        
        error_result = {
            "analysis_id": "error-123",
            "ticker": "INVALID",
            "timestamp": "2024-01-15T10:00:00Z",
            "metadata": {
                "status": "error",
                "error": "Invalid ticker symbol"
            }
        }
        
        markdown_report = analyst.generate_markdown_report(error_result)
        
        assert "# Analysis Error Report" in markdown_report
        assert "**Ticker:** INVALID" in markdown_report
        assert "Invalid ticker symbol" in markdown_report
    
    @pytest.mark.asyncio
    async def test_technical_analysis_engine(self, mock_alpaca_provider):
        """Test technical analysis engine"""
        
        engine = TechnicalAnalysisEngine(mock_alpaca_provider)
        
        market_data = {
            "price_data": [
                {"close": 180.0, "volume": 1000000},
                {"close": 182.0, "volume": 1100000},
                {"close": 184.0, "volume": 1200000},
                {"close": 186.0, "volume": 1300000},
                {"close": 185.0, "volume": 1250000}
            ],
            "technical_indicators": {"rsi": 65.5}
        }
        
        result = await engine.analyze("AAPL", market_data)
        
        assert "trend" in result
        assert "current_price" in result
        assert "summary" in result
        assert result["current_price"] == 185.0
    
    @pytest.mark.asyncio
    async def test_fundamental_analysis_engine(self, mock_alpaca_provider):
        """Test fundamental analysis engine"""
        
        engine = FundamentalAnalysisEngine(mock_alpaca_provider)
        
        result = await engine.analyze("AAPL")
        
        assert "sector" in result
        assert "summary" in result
        # Should use mocked data
        assert "Technology" in result["summary"]


class TestIntegration:
    """Integration tests for the complete agent workflow"""
    
    @pytest.mark.asyncio
    async def test_end_to_end_workflow(self):
        """Test complete end-to-end analysis workflow"""
        
        # Setup components (you would use real implementations in production)
        config = Mock()
        config.ANTHROPIC_API_KEY = "test_key"
        
        # Mock LLM with realistic response
        llm = Mock()
        llm.generate_completion = AsyncMock(return_value=json.dumps({
            "recommendation": "buy",
            "confidence": 8,
            "time_horizon": "medium_term",
            "position_size": "medium",
            "entry_target": 185.50,
            "exit_targets": {"primary": 195.00, "secondary": 200.00},
            "stop_loss": 178.00,
            "risk_reward_ratio": 2.5,
            "investment_thesis": "Comprehensive analysis supports buy recommendation",
            "risk_factors": ["Market volatility", "Earnings risk"],
            "catalyst_timeline": {
                "short_term": ["Technical breakout"],
                "medium_term": ["Earnings release"],
                "long_term": ["Product cycle"]
            }
        }))
        
        # Mock Alpaca with market data
        alpaca = Mock()
        alpaca.get_market_data = AsyncMock(return_value={
            "AAPL": [{"close": 184.5, "volume": 1000000, "timestamp": "2024-01-15T16:00:00Z"}]
        })
        alpaca.get_current_quote = AsyncMock(return_value={"price": 184.50})
        alpaca.get_technical_indicators = AsyncMock(return_value={"rsi": 65.5})
        alpaca.get_news = AsyncMock(return_value=[])
        alpaca.get_company_info = AsyncMock(return_value={"sector": "Technology"})
        alpaca.get_financial_data = AsyncMock(return_value={"pe_ratio": 28.5})
        
        # Create analyst
        analyst = create_junior_analyst(llm, alpaca, config)
        
        # Test complete workflow
        task_data = {
            "task_type": AnalysisType.NEW_OPPORTUNITY.value,
            "ticker": "AAPL",
            "technical_signal": {
                "pattern": "ascending_triangle",
                "score": 8.2,
                "resistance_level": 185.00
            }
        }
        
        # Perform analysis
        result = await analyst.analyze_stock(task_data)
        
        # Verify complete workflow
        assert result["metadata"]["status"] == "success"
        assert result["ticker"] == "AAPL"
        assert result["recommendation"] == "buy"
        assert result["confidence"] == 8
        
        # Generate markdown report
        markdown_report = analyst.generate_markdown_report(result)
        assert len(markdown_report) > 100  # Substantial report generated
        
        # Verify performance tracking
        performance = analyst.get_performance_summary()
        assert performance["total_analyses"] == 1
        assert performance["success_rate"] == "100.0%"
        
        print("✅ End-to-end integration test passed!")


def run_tests():
    """Run all tests with detailed output"""
    
    print("🧪 JUNIOR RESEARCH ANALYST - TEST SUITE")
    print("=" * 60)
    
    # Run pytest with verbose output
    pytest_args = [
        __file__,
        "-v",  # Verbose output
        "-s",  # Don't capture stdout
        "--tb=short",  # Short traceback format
        "--durations=10"  # Show 10 slowest tests
    ]
    
    exit_code = pytest.main(pytest_args)
    
    if exit_code == 0:
        print("\n✅ ALL TESTS PASSED!")
        print("The Junior Research Analyst Agent is ready for deployment.")
    else:
        print("\n❌ SOME TESTS FAILED!")
        print("Please review the test output and fix any issues.")
    
    return exit_code


if __name__ == "__main__":
    # Run the test suite
    exit_code = run_tests()
    exit(exit_code)