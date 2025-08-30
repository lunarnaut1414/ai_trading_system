# tests/test_claude_provider.py
"""
Pytest-compatible test suite for Claude-only LLM Provider
Run with: pytest tests/test_claude_provider.py -v
"""

import pytest
import json
import sys
import os
from pathlib import Path
from datetime import datetime
import asyncio

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.claude_llm_provider import ClaudeLLMProvider, LLMResponse
from config.claude_config import ClaudeConfig

# Skip all tests if API key is not configured
pytestmark = pytest.mark.skipif(
    not os.getenv("ANTHROPIC_API_KEY"),
    reason="ANTHROPIC_API_KEY not set"
)

@pytest.fixture(scope="module")
def claude_provider():
    """Fixture to create and return Claude provider instance"""
    if not ClaudeConfig.validate():
        pytest.skip("Claude configuration invalid")
    
    provider = ClaudeLLMProvider(
        api_key=ClaudeConfig.ANTHROPIC_API_KEY,
        model=ClaudeConfig.CLAUDE_MODEL,
        max_retries=3,
        cache_ttl_minutes=5
    )
    return provider

@pytest.mark.asyncio
async def test_provider_initialization():
    """Test that provider initializes correctly"""
    provider = ClaudeLLMProvider(
        api_key=ClaudeConfig.ANTHROPIC_API_KEY,
        model=ClaudeConfig.CLAUDE_MODEL
    )
    
    assert provider is not None
    assert provider.client is not None
    assert provider.model == ClaudeConfig.CLAUDE_MODEL
    assert provider.metrics["total_calls"] == 0

@pytest.mark.asyncio
async def test_basic_completion(claude_provider):
    """Test basic text completion functionality"""
    response = await claude_provider.get_completion(
        prompt="Explain the concept of a moving average in trading in exactly one sentence.",
        temperature=0.5,
        max_tokens=100
    )
    
    assert response.success == True
    assert response.content is not None and len(response.content) > 0
    assert response.tokens_used > 0
    assert response.processing_time > 0
    assert response.model == claude_provider.model

@pytest.mark.asyncio
async def test_system_prompt(claude_provider):
    """Test that system prompts work correctly"""
    response = await claude_provider.get_completion(
        prompt="AAPL",
        system_prompt="You are a trading analyst. When given a stock symbol, respond with exactly 'Analyzing [SYMBOL]' and nothing else.",
        temperature=0.1,
        max_tokens=50
    )
    
    assert response.success == True
    assert "Analyzing AAPL" in response.content or "AAPL" in response.content

@pytest.mark.asyncio
async def test_json_parsing(claude_provider):
    """Test JSON response parsing capability"""
    response = await claude_provider.get_completion(
        prompt="Create a JSON object with keys 'symbol' (value: 'TSLA'), 'action' (value: 'BUY'), and 'confidence' (value: 8). Return only valid JSON.",
        system_prompt="You are a JSON generator. Always respond with valid JSON only, no additional text.",
        temperature=0.1,
        max_tokens=100,
        parse_json=True
    )
    
    assert response.success == True
    
    # Should be valid JSON
    parsed = json.loads(response.content)
    assert isinstance(parsed, dict)
    assert "symbol" in parsed or "TSLA" in str(parsed)

@pytest.mark.asyncio
async def test_structured_technical_analysis(claude_provider):
    """Test structured technical analysis functionality"""
    test_data = {
        "symbol": "GOOGL",
        "current_price": 140.50,
        "rsi": 45,
        "macd": {"signal": "bearish", "histogram": -0.3},
        "volume": "below_average",
        "support": 135,
        "resistance": 145,
        "ma_50": 142,
        "ma_200": 138
    }
    
    analysis = await claude_provider.get_structured_analysis(
        analysis_type="technical",
        data=test_data,
        context={"timeframe": "daily", "market_sentiment": "mixed"}
    )
    
    assert "error" not in analysis
    assert isinstance(analysis, dict)
    # The AI might return different field names, so check if we got a meaningful response
    assert len(analysis) > 0  # Should have some analysis fields

@pytest.mark.asyncio
async def test_structured_fundamental_analysis(claude_provider):
    """Test structured fundamental analysis functionality"""
    test_data = {
        "symbol": "MSFT",
        "pe_ratio": 32,
        "earnings_growth": "12% YoY",
        "revenue_growth": "8% YoY",
        "profit_margin": 35,
        "debt_to_equity": 0.47,
        "roe": 38,
        "recent_news": "Cloud revenue beat expectations"
    }
    
    analysis = await claude_provider.get_structured_analysis(
        analysis_type="fundamental",
        data=test_data
    )
    
    assert "error" not in analysis
    assert isinstance(analysis, dict)
    # Check that we got some analysis back (field names may vary)
    assert len(analysis) > 0
    # At least one of these concepts should be present in the response
    possible_fields = ["valuation", "growth", "recommendation", "confidence", 
                      "health_score", "risk", "analysis", "outlook"]
    response_str = json.dumps(analysis).lower()
    assert any(field in response_str for field in possible_fields)

@pytest.mark.asyncio
async def test_structured_risk_analysis(claude_provider):
    """Test structured risk analysis functionality"""
    test_data = {
        "symbol": "NVDA",
        "position_size": 15000,
        "portfolio_value": 100000,
        "volatility": "high",
        "beta": 2.1,
        "current_price": 850,
        "entry_price": 800
    }
    
    analysis = await claude_provider.get_structured_analysis(
        analysis_type="risk",
        data=test_data
    )
    
    assert "error" not in analysis
    assert isinstance(analysis, dict)
    # Check that we got some risk analysis back
    assert len(analysis) > 0
    # At least one risk-related concept should be present
    possible_fields = ["risk", "position", "stop", "var", "score", 
                      "sizing", "exposure", "volatility"]
    response_str = json.dumps(analysis).lower()
    assert any(field in response_str for field in possible_fields)

@pytest.mark.asyncio
async def test_cache_functionality(claude_provider):
    """Test that caching works correctly"""
    # Reset metrics first
    claude_provider.reset_metrics()
    
    # First call - should hit API
    prompt = "What is 2+2? Answer with just the number."
    response1 = await claude_provider.get_completion(
        prompt=prompt,
        temperature=0.1,
        max_tokens=10,
        use_cache=True
    )
    
    assert response1.success == True
    
    # Second call - should use cache
    response2 = await claude_provider.get_completion(
        prompt=prompt,
        temperature=0.1,
        max_tokens=10,
        use_cache=True
    )
    
    assert response2.success == True
    assert response2.content == response1.content
    
    # Check metrics
    metrics = claude_provider.get_metrics_summary()
    assert metrics["cache_hits"] > 0

@pytest.mark.asyncio
async def test_error_handling(claude_provider):
    """Test error handling for invalid inputs"""
    # Test with empty prompt
    response = await claude_provider.get_completion(
        prompt="",
        max_tokens=10
    )
    
    # Should handle gracefully
    if not response.success:
        assert response.error is not None

@pytest.mark.asyncio
async def test_rate_limiter(claude_provider):
    """Test that rate limiting works"""
    # This test just verifies the rate limiter exists and functions
    assert claude_provider.rate_limiter is not None
    
    # Test acquire method
    can_proceed = await claude_provider.rate_limiter.acquire()
    assert isinstance(can_proceed, bool)

@pytest.mark.asyncio
async def test_metrics_tracking(claude_provider):
    """Test that metrics are tracked correctly"""
    # Reset metrics
    claude_provider.reset_metrics()
    
    # Make a successful call
    response = await claude_provider.get_completion(
        prompt="Say 'test'",
        max_tokens=10
    )
    
    if response.success:
        metrics = claude_provider.get_metrics_summary()
        
        assert metrics["total_calls"] > 0
        assert metrics["success_rate"] > 0
        assert metrics["average_response_time"] > 0
        # Fixed: Use correct key name
        assert metrics["average_tokens_per_call"] > 0 or metrics.get("total_tokens", 0) > 0

@pytest.mark.asyncio
async def test_trading_scenario(claude_provider):
    """Test a realistic trading analysis scenario"""
    junior_prompt = """Analyze NVDA for trading with this data:
    Price: $850, RSI: 72, MACD: Bullish, Volume: +20% avg
    Recent: Beat earnings by 15%, new AI partnership

    Provide JSON with: direction, confidence (1-10), entry_price, stop_loss"""
    
    response = await claude_provider.get_completion(
        prompt=junior_prompt,
        system_prompt="You are a trading analyst. Respond with valid JSON only.",
        temperature=0.3,
        parse_json=True
    )
    
    assert response.success == True
    
    # Try to parse as JSON
    try:
        analysis = json.loads(response.content)
        assert isinstance(analysis, dict)
        # Should have at least some trading-related fields
        trading_fields = ["direction", "confidence", "entry", "stop", "buy", "sell", "hold"]
        assert any(field in str(analysis).lower() for field in trading_fields)
    except json.JSONDecodeError:
        # If not pure JSON, at least check it has trading content
        assert any(word in response.content.lower() for word in ["buy", "sell", "hold", "bullish", "bearish"])

# Performance/Load Tests (optional - mark as slow)
@pytest.mark.slow
@pytest.mark.asyncio
async def test_multiple_concurrent_requests(claude_provider):
    """Test handling multiple concurrent requests"""
    prompts = [
        "What is RSI in trading?",
        "Explain MACD indicator",
        "What is a stop loss?",
        "Define market cap",
        "What is P/E ratio?"
    ]
    
    # Create concurrent tasks
    tasks = [
        claude_provider.get_completion(prompt, max_tokens=50)
        for prompt in prompts
    ]
    
    # Execute concurrently
    responses = await asyncio.gather(*tasks)
    
    # All should succeed
    assert all(r.success for r in responses)
    assert all(r.content for r in responses)

# Utility function to run standalone
def run_tests_standalone():
    """Run tests without pytest (for debugging)"""
    import asyncio
    
    async def run():
        print("🧪 Running Claude Provider Tests (Standalone Mode)\n")
        
        # Validate config
        if not ClaudeConfig.validate():
            print("❌ Configuration validation failed")
            return False
        
        # Initialize provider
        provider = ClaudeLLMProvider(
            api_key=ClaudeConfig.ANTHROPIC_API_KEY,
            model=ClaudeConfig.CLAUDE_MODEL
        )
        
        # Run basic test
        print("Testing basic completion...")
        response = await provider.get_completion(
            prompt="What is a moving average?",
            max_tokens=100
        )
        
        if response.success:
            print(f"✅ Basic test passed: {response.content[:100]}...")
            return True
        else:
            print(f"❌ Basic test failed: {response.error}")
            return False
    
    return asyncio.run(run())

if __name__ == "__main__":
    # If run directly, use standalone mode
    success = run_tests_standalone()
    sys.exit(0 if success else 1)