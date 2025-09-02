# utils/llm_provider.py
"""
Claude-Only LLM Provider for AI Trading System
Optimized for macOS M2 Max with comprehensive error handling and retry logic
"""

import asyncio
import json
import logging
import time
from typing import Dict, Optional, List, Any, Union
from datetime import datetime, timedelta
from dataclasses import dataclass
from anthropic import Anthropic, APIError, APIConnectionError, RateLimitError
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_sleep_log
)

@dataclass
class LLMResponse:
    """Standardized LLM response format"""
    content: str
    model: str
    tokens_used: int
    processing_time: float
    success: bool
    timestamp: str
    error: Optional[str] = None
    usage_details: Optional[Dict] = None

class ClaudeLLMProvider:
    """
    Claude-Only LLM Provider for AI Trading System
    
    Features:
    - Automatic retry with exponential backoff
    - Rate limiting and quota management
    - Response caching for identical requests
    - JSON parsing and validation
    - Comprehensive error handling
    - Performance monitoring
    """
    
    def __init__(self, 
                 api_key: str,
                 model: str = "claude-sonnet-4-20250514",  # Updated to current model
                 max_retries: int = 3,
                 cache_ttl_minutes: int = 5):
        """
        Initialize Claude LLM Provider
        
        Args:
            api_key: Anthropic API key
            model: Claude model to use (opus, sonnet, or haiku)
            max_retries: Maximum retry attempts for failed requests
            cache_ttl_minutes: Cache time-to-live in minutes
        """
        
        self.api_key = api_key
        self.model = model
        self.max_retries = max_retries
        self.cache_ttl = timedelta(minutes=cache_ttl_minutes)
        
        # Initialize Claude client
        self.client = Anthropic(api_key=api_key)
        
        # Setup logging
        self.logger = logging.getLogger("claude_llm_provider")
        
        # Rate limiting
        self.rate_limiter = RateLimiter(
            calls_per_minute=50,  # Claude's default rate limit
            calls_per_day=10000
        )
        
        # Response cache
        self.response_cache: Dict[str, CachedResponse] = {}
        
        # Performance metrics
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
        
        self.logger.info(f"Claude LLM Provider initialized with model: {model}")
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=60),
        retry=retry_if_exception_type((APIConnectionError, RateLimitError)),
        before_sleep=before_sleep_log(logging.getLogger("claude_llm_provider"), logging.WARNING)
    )
    async def get_completion(self,
                           prompt: str,
                           system_prompt: Optional[str] = None,
                           max_tokens: int = 4000,
                           temperature: float = 0.7,
                           use_cache: bool = True,
                           parse_json: bool = False) -> LLMResponse:
        """
        Get completion from Claude API
        
        Args:
            prompt: User prompt
            system_prompt: System instructions
            max_tokens: Maximum response tokens
            temperature: Response randomness (0-1)
            use_cache: Whether to use response cache
            parse_json: Whether to parse response as JSON
            
        Returns:
            LLMResponse with generated content
        """
        
        start_time = time.time()
        
        # Check cache first
        if use_cache:
            cache_key = self._generate_cache_key(prompt, system_prompt, temperature)
            cached = self._get_cached_response(cache_key)
            if cached:
                self.metrics["cache_hits"] += 1
                self.logger.debug(f"Cache hit for prompt: {prompt[:50]}...")
                return cached
        
        # Check rate limits
        if not await self.rate_limiter.acquire():
            error_msg = "Rate limit exceeded. Please wait before making another request."
            return LLMResponse(
                content="",
                model=self.model,
                tokens_used=0,
                processing_time=time.time() - start_time,
                success=False,
                timestamp=datetime.now().isoformat(),
                error=error_msg
            )
        
        try:
            # Prepare messages
            messages = [{"role": "user", "content": prompt}]
            
            # Make API call with proper parameters
            self.logger.debug(f"Calling Claude API with prompt: {prompt[:50]}...")
            
            # Build kwargs for API call
            api_kwargs = {
                "model": self.model,
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature
            }
            
            # Add system prompt if provided (as string, not list)
            if system_prompt:
                api_kwargs["system"] = system_prompt
            
            response = await asyncio.to_thread(
                self.client.messages.create,
                **api_kwargs
            )
            
            # Extract content
            content = response.content[0].text
            
            # Parse JSON if requested
            if parse_json:
                try:
                    parsed = json.loads(content)
                    content = json.dumps(parsed)  # Ensure it's properly formatted
                except json.JSONDecodeError as e:
                    self.logger.warning(f"Failed to parse JSON response: {e}")
                    # Try to extract JSON from the response
                    content = self._extract_json(content)
            
            # Calculate metrics
            tokens_used = response.usage.input_tokens + response.usage.output_tokens
            processing_time = time.time() - start_time
            
            # Update metrics
            self.metrics["total_calls"] += 1
            self.metrics["successful_calls"] += 1
            self.metrics["total_tokens"] += tokens_used
            self._update_average_response_time(processing_time)
            
            # Create response
            result = LLMResponse(
                content=content,
                model=self.model,
                tokens_used=tokens_used,
                processing_time=processing_time,
                success=True,
                timestamp=datetime.now().isoformat(),
                usage_details={
                    "input_tokens": response.usage.input_tokens,
                    "output_tokens": response.usage.output_tokens
                }
            )
            
            # Cache successful response
            if use_cache:
                self._cache_response(cache_key, result)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Claude API error: {str(e)}")
            self.metrics["total_calls"] += 1
            self.metrics["failed_calls"] += 1
            
            # Track error type
            error_type = type(e).__name__
            self.metrics["errors_by_type"][error_type] = self.metrics["errors_by_type"].get(error_type, 0) + 1
            
            return LLMResponse(
                content="",
                model=self.model,
                tokens_used=0,
                processing_time=time.time() - start_time,
                success=False,
                timestamp=datetime.now().isoformat(),
                error=str(e)
            )
    
    async def get_structured_analysis(self,
                                    analysis_type: str,
                                    data: Dict,
                                    context: Optional[Dict] = None) -> Dict:
        """
        Get structured analysis for specific trading scenarios
        
        Args:
            analysis_type: Type of analysis (technical/fundamental/risk)
            data: Market data for analysis
            context: Additional context
            
        Returns:
            Structured analysis result
        """
        
        # Build specialized prompts based on analysis type
        prompts = {
            "technical": self._build_technical_prompt,
            "fundamental": self._build_fundamental_prompt,
            "risk": self._build_risk_prompt,
            "portfolio": self._build_portfolio_prompt,
            "execution": self._build_execution_prompt
        }
        
        if analysis_type not in prompts:
            return {"error": f"Unknown analysis type: {analysis_type}"}
        
        prompt = prompts[analysis_type](data, context)
        
        response = await self.get_completion(
            prompt=prompt,
            system_prompt="You are an expert trading analyst. Provide detailed JSON analysis.",
            temperature=0.3,
            parse_json=True
        )
        
        if response.success:
            try:
                return json.loads(response.content)
            except:
                return {"error": "Failed to parse analysis", "raw": response.content}
        else:
            return {"error": response.error}
    
    def _build_technical_prompt(self, data: Dict, context: Optional[Dict]) -> str:
        """Build technical analysis prompt"""
        return f"""
        Analyze the following technical indicators for {data.get('symbol', 'UNKNOWN')}:
        
        Current Price: ${data.get('current_price', 0)}
        RSI: {data.get('rsi', 'N/A')}
        MACD: {data.get('macd', 'N/A')}
        Volume: {data.get('volume', 'N/A')}
        Support: ${data.get('support', 0)}
        Resistance: ${data.get('resistance', 0)}
        MA(50): ${data.get('ma_50', 0)}
        MA(200): ${data.get('ma_200', 0)}
        
        Context: {json.dumps(context) if context else 'None'}
        
        Provide a JSON response with:
        - signal: BUY/SELL/HOLD
        - confidence: 1-10
        - entry_price: suggested entry
        - stop_loss: stop loss level
        - take_profit: profit target
        - reasoning: brief explanation
        """
    
    def _build_fundamental_prompt(self, data: Dict, context: Optional[Dict]) -> str:
        """Build fundamental analysis prompt"""
        return f"""
        Analyze the fundamental metrics for {data.get('symbol', 'UNKNOWN')}:
        
        P/E Ratio: {data.get('pe_ratio', 'N/A')}
        Earnings Growth: {data.get('earnings_growth', 'N/A')}
        Revenue Growth: {data.get('revenue_growth', 'N/A')}
        Profit Margin: {data.get('profit_margin', 'N/A')}%
        Debt/Equity: {data.get('debt_to_equity', 'N/A')}
        ROE: {data.get('roe', 'N/A')}%
        Recent News: {data.get('recent_news', 'None')}
        
        Provide a JSON response with:
        - valuation: UNDERVALUED/FAIR/OVERVALUED
        - health_score: 1-10
        - growth_potential: LOW/MEDIUM/HIGH
        - risks: list of key risks
        - catalysts: list of potential catalysts
        - recommendation: investment recommendation
        """
    
    def _build_risk_prompt(self, data: Dict, context: Optional[Dict]) -> str:
        """Build risk analysis prompt"""
        return f"""
        Assess risk for position in {data.get('symbol', 'UNKNOWN')}:
        
        Position Size: ${data.get('position_size', 0)}
        Portfolio Value: ${data.get('portfolio_value', 0)}
        Volatility: {data.get('volatility', 'N/A')}
        Beta: {data.get('beta', 'N/A')}
        Current Price: ${data.get('current_price', 0)}
        Entry Price: ${data.get('entry_price', 0)}
        
        Provide a JSON response with:
        - risk_score: 1-10
        - position_sizing: APPROPRIATE/REDUCE/INCREASE
        - var_95: Value at Risk (95% confidence)
        - max_drawdown: expected maximum drawdown
        - correlation_risk: portfolio correlation assessment
        - hedging_recommendation: suggested hedges
        """
    
    def _build_portfolio_prompt(self, data: Dict, context: Optional[Dict]) -> str:
        """Build portfolio optimization prompt"""
        return f"""
        Optimize portfolio allocation with these holdings:
        {json.dumps(data.get('holdings', []), indent=2)}
        
        Total Capital: ${data.get('total_capital', 0)}
        Risk Tolerance: {data.get('risk_tolerance', 'MODERATE')}
        
        Provide a JSON response with optimal allocations and rebalancing recommendations.
        """
    
    def _build_execution_prompt(self, data: Dict, context: Optional[Dict]) -> str:
        """Build trade execution prompt"""
        return f"""
        Plan execution for {data.get('action', 'BUY')} order:
        
        Symbol: {data.get('symbol', 'UNKNOWN')}
        Quantity: {data.get('quantity', 0)}
        Current Bid: ${data.get('bid', 0)}
        Current Ask: ${data.get('ask', 0)}
        Volume: {data.get('volume', 0)}
        Volatility: {data.get('volatility', 'N/A')}
        
        Provide a JSON response with:
        - order_type: MARKET/LIMIT/STOP_LIMIT
        - timing: IMMEDIATE/STAGED/WAIT
        - price_target: execution price
        - slicing: order slicing recommendation
        - expected_slippage: estimated slippage
        """
    
    def _extract_json(self, text: str) -> str:
        """Extract JSON from text that might contain other content"""
        import re
        
        # Try to find JSON-like content
        json_patterns = [
            r'\{[^{}]*\}',  # Simple JSON object
            r'\{.*\}',       # Any JSON object
            r'\[.*\]'        # JSON array
        ]
        
        for pattern in json_patterns:
            matches = re.findall(pattern, text, re.DOTALL)
            for match in matches:
                try:
                    json.loads(match)
                    return match
                except:
                    continue
        
        # If no valid JSON found, return original
        return text
    
    def _generate_cache_key(self, prompt: str, system_prompt: Optional[str], temperature: float) -> str:
        """Generate cache key for request"""
        import hashlib
        
        key_data = f"{prompt}|{system_prompt}|{temperature}|{self.model}"
        return hashlib.md5(key_data.encode()).hexdigest()
    
    def _get_cached_response(self, cache_key: str) -> Optional[LLMResponse]:
        """Get cached response if available and not expired"""
        if cache_key in self.response_cache:
            cached = self.response_cache[cache_key]
            if datetime.now() - cached.timestamp < self.cache_ttl:
                return cached.response
            else:
                del self.response_cache[cache_key]
        return None
    
    def _cache_response(self, cache_key: str, response: LLMResponse):
        """Cache successful response"""
        self.response_cache[cache_key] = CachedResponse(
            response=response,
            timestamp=datetime.now()
        )
    
    def _update_average_response_time(self, new_time: float):
        """Update rolling average response time"""
        current_avg = self.metrics["average_response_time"]
        total_calls = self.metrics["successful_calls"]
        
        if total_calls == 1:
            self.metrics["average_response_time"] = new_time
        else:
            self.metrics["average_response_time"] = (
                (current_avg * (total_calls - 1) + new_time) / total_calls
            )
    
    def get_metrics_summary(self) -> Dict:
        """Get performance metrics summary"""
        total = self.metrics["total_calls"]
        if total == 0:
            return self.metrics
        
        return {
            **self.metrics,
            "success_rate": self.metrics["successful_calls"] / total,
            "cache_hit_rate": self.metrics["cache_hits"] / total,
            "average_tokens_per_call": self.metrics["total_tokens"] / total if total > 0 else 0,
            "estimated_cost": self._estimate_cost()
        }
    
    def _estimate_cost(self) -> float:
        """Estimate total cost based on token usage"""
        # Claude pricing (approximate)
        pricing = {
            "claude-sonnet-4-20250514": {"input": 0.003, "output": 0.015},  # per 1K tokens
            "claude-opus-4-1-20250805": {"input": 0.015, "output": 0.075},
            "claude-3-5-haiku-20241022": {"input": 0.00025, "output": 0.00125}
        }
        
        model_pricing = pricing.get(self.model, pricing["claude-sonnet-4-20250514"])
        
        # Rough estimate (assuming 50/50 input/output split)
        total_tokens_k = self.metrics["total_tokens"] / 1000
        estimated_cost = total_tokens_k * (model_pricing["input"] + model_pricing["output"]) / 2
        
        return round(estimated_cost, 4)
    
    def reset_metrics(self):
        """Reset all metrics"""
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
        self.response_cache.clear()

# Supporting Classes

@dataclass
class CachedResponse:
    """Cached response with timestamp"""
    response: LLMResponse
    timestamp: datetime

class RateLimiter:
    """Rate limiter for API calls"""
    
    def __init__(self, calls_per_minute: int = 50, calls_per_day: int = 10000):
        self.calls_per_minute = calls_per_minute
        self.calls_per_day = calls_per_day
        self.minute_calls: List[datetime] = []
        self.day_calls: List[datetime] = []
    
    async def acquire(self) -> bool:
        """Check if we can make another API call"""
        now = datetime.now()
        
        # Clean old entries
        self.minute_calls = [t for t in self.minute_calls if now - t < timedelta(minutes=1)]
        self.day_calls = [t for t in self.day_calls if now - t < timedelta(days=1)]
        
        # Check limits
        if len(self.minute_calls) >= self.calls_per_minute:
            return False
        if len(self.day_calls) >= self.calls_per_day:
            return False
        
        # Record this call
        self.minute_calls.append(now)
        self.day_calls.append(now)
        
        return True