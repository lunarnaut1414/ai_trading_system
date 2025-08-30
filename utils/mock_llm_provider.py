# utils/mock_llm_provider.py
"""
Mock Claude LLM Provider for testing without API credits
Use this for development and testing when you don't have API credits
"""

import asyncio
import json
import random
from typing import Dict, Optional
from datetime import datetime
from utils.llm_provider import LLMResponse

class MockClaudeLLMProvider:
    """
    Mock Claude Provider that simulates API responses
    Use for testing without consuming API credits
    """
    
    def __init__(self, **kwargs):
        """Initialize mock provider"""
        self.model = kwargs.get("model", "claude-3-5-sonnet-20241022")
        self.metrics = {
            "total_calls": 0,
            "successful_calls": 0,
            "failed_calls": 0,
            "cache_hits": 0,
            "total_tokens": 0,
            "total_cost": 0.0,
            "average_response_time": 0.0,
            "errors_by_type": {}
        }
        print("🎭 Mock Claude Provider initialized (no API calls will be made)")
    
    async def get_completion(self, 
                           prompt: str,
                           system_prompt: Optional[str] = None,
                           max_tokens: int = 4000,
                           temperature: float = 0.7,
                           use_cache: bool = True,
                           parse_json: bool = False) -> LLMResponse:
        """Mock completion that returns realistic responses"""
        
        # Simulate processing delay
        await asyncio.sleep(random.uniform(0.1, 0.5))
        
        # Update metrics
        self.metrics["total_calls"] += 1
        self.metrics["successful_calls"] += 1
        
        # Generate mock response based on prompt content
        if parse_json or "json" in prompt.lower():
            # Return mock JSON for JSON requests
            content = self._generate_mock_json(prompt)
        else:
            # Return mock text analysis
            content = self._generate_mock_analysis(prompt, system_prompt)
        
        tokens_used = len(content.split()) * 2  # Rough token estimate
        self.metrics["total_tokens"] += tokens_used
        
        return LLMResponse(
            content=content,
            model=self.model,
            tokens_used=tokens_used,
            processing_time=0.25,
            success=True,
            timestamp=datetime.now().isoformat(),
            usage_details={
                "input_tokens": len(prompt.split()) * 2,
                "output_tokens": tokens_used,
                "stop_reason": "end_turn"
            }
        )
    
    async def get_structured_analysis(self,
                                     analysis_type: str,
                                     data: Dict,
                                     context: Optional[Dict] = None) -> Dict:
        """Mock structured analysis"""
        
        await asyncio.sleep(random.uniform(0.1, 0.3))
        
        if analysis_type == "technical":
            return {
                "trend_direction": random.choice(["bullish", "bearish", "neutral"]),
                "strength": random.randint(4, 8),
                "support_levels": [data.get("support", 100) - i*5 for i in range(3)],
                "resistance_levels": [data.get("resistance", 110) + i*5 for i in range(3)],
                "entry_points": [data.get("current_price", 105) - 2],
                "stop_loss": data.get("current_price", 105) - 5,
                "take_profit": [data.get("current_price", 105) + i*3 for i in range(1, 4)],
                "confidence": random.randint(6, 9)
            }
        
        elif analysis_type == "fundamental":
            return {
                "valuation": random.choice(["undervalued", "fair", "overvalued"]),
                "growth_prospects": random.choice(["low", "medium", "high"]),
                "risk_factors": ["Market volatility", "Competition", "Regulatory changes"],
                "catalysts": ["Earnings release", "Product launch", "Market expansion"],
                "recommendation": random.choice(["buy", "hold", "sell"]),
                "target_price": data.get("current_price", 100) * random.uniform(0.9, 1.2),
                "confidence": random.randint(5, 8)
            }
        
        elif analysis_type == "risk":
            return {
                "risk_level": random.choice(["low", "medium", "high"]),
                "position_size": random.uniform(2, 5),
                "stop_loss": data.get("entry_price", 100) * 0.95,
                "risk_reward_ratio": random.uniform(1.5, 3.0),
                "correlation_risk": "moderate",
                "max_loss": data.get("position_size", 10000) * 0.05,
                "confidence": random.randint(6, 8)
            }
        
        return {"error": f"Unknown analysis type: {analysis_type}"}
    
    def _generate_mock_json(self, prompt: str) -> str:
        """Generate mock JSON response"""
        
        # Check what kind of JSON is requested
        if "TSLA" in prompt or "symbol" in prompt.lower():
            return json.dumps({
                "symbol": "TSLA",
                "action": "BUY",
                "confidence": 8
            })
        
        elif "trading" in prompt.lower() or "NVDA" in prompt:
            return json.dumps({
                "direction": "BULLISH",
                "confidence": 7,
                "entry_price": 845,
                "stop_loss": 820,
                "take_profit_targets": [870, 890, 920],
                "time_horizon": "2-4 weeks",
                "rationale": "Strong technical momentum with positive fundamental catalysts"
            })
        
        # Default JSON
        return json.dumps({
            "status": "success",
            "data": "mock response",
            "timestamp": datetime.now().isoformat()
        })
    
    def _generate_mock_analysis(self, prompt: str, system_prompt: str = None) -> str:
        """Generate mock text analysis"""
        
        if "moving average" in prompt.lower():
            return "A moving average is a technical indicator that smooths price data by calculating the average price over a specific number of periods."
        
        elif "AAPL" in prompt:
            return "Analyzing AAPL"
        
        elif "2+2" in prompt:
            return "4"
        
        elif "RSI" in prompt.lower():
            return "RSI (Relative Strength Index) is a momentum oscillator that measures the speed and magnitude of price changes, typically used to identify overbought or oversold conditions."
        
        # Default response
        return f"Mock analysis response for: {prompt[:50]}... Based on the data provided, the analysis suggests moderate opportunity with standard risk parameters."
    
    def get_metrics_summary(self) -> Dict:
        """Get mock metrics"""
        return {
            "total_calls": self.metrics["total_calls"],
            "success_rate": 1.0,  # Always successful in mock
            "cache_hit_rate": 0.2,  # Simulate some cache hits
            "average_response_time": 0.25,
            "total_tokens_used": self.metrics["total_tokens"],
            "estimated_cost": 0.0,  # No cost for mock
            "errors": {}
        }
    
    def reset_metrics(self):
        """Reset metrics"""
        self.metrics = {
            "total_calls": 0,
            "successful_calls": 0,
            "failed_calls": 0,
            "cache_hits": 0,
            "total_tokens": 0,
            "total_cost": 0.0,
            "average_response_time": 0.0,
            "errors_by_type": {}
        }


# Test configuration for switching between real and mock
class LLMProviderFactory:
    """Factory to create appropriate LLM provider based on configuration"""
    
    @staticmethod
    def create_provider(use_mock: bool = False, **kwargs):
        """
        Create LLM provider instance
        
        Args:
            use_mock: If True, return mock provider (no API calls)
            **kwargs: Additional arguments for provider
        
        Returns:
            LLM Provider instance (real or mock)
        """
        
        # Check environment variable for mock mode
        import os
        if use_mock or os.getenv("USE_MOCK_LLM", "false").lower() == "true":
            print("🎭 Using Mock LLM Provider (no API credits required)")
            return MockClaudeLLMProvider(**kwargs)
        else:
            from utils.llm_provider import ClaudeLLMProvider
            from config.claude_config import ClaudeConfig
            
            return ClaudeLLMProvider(
                api_key=kwargs.get("api_key", ClaudeConfig.ANTHROPIC_API_KEY),
                model=kwargs.get("model", ClaudeConfig.CLAUDE_MODEL),
                **kwargs
            )