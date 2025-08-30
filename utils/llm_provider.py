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
                 model: str = "claude-3-5-haiku-20241022",
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
            raise RateLimitError("Rate limit exceeded. Please wait before making another request.")
        
        try:
            # Prepare messages
            messages = [{"role": "user", "content": prompt}]
            
            # Make API call
            self.logger.debug(f"Calling Claude API with prompt: {prompt[:50]}...")
            
            response = await asyncio.to_thread(
                self.client.messages.create,
                model=self.model,
                system=system_prompt if system_prompt else None,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature
            )
            
            # Extract content
            content = response.content[0].text
            
            # Parse JSON if requested
            if parse_json:
                try:
                    content = json.loads(content)
                    content = json.dumps(content)  # Ensure it's properly formatted
                except json.JSONDecodeError as e:
                    self.logger.warning(f"Failed to parse JSON response: {e}")
                    # Try to extract JSON from the response
                    content = self._extract_json(content)
            
            # Calculate metrics
            processing_time = time.time() - start_time
            tokens_used = response.usage.input_tokens + response.usage.output_tokens
            
            # Create response
            llm_response = LLMResponse(
                content=content,
                model=self.model,
                tokens_used=tokens_used,
                processing_time=processing_time,
                success=True,
                timestamp=datetime.now().isoformat(),
                usage_details={
                    "input_tokens": response.usage.input_tokens,
                    "output_tokens": response.usage.output_tokens,
                    "stop_reason": response.stop_reason
                }
            )
            
            # Update metrics
            self._update_metrics(llm_response)
            
            # Cache successful response
            if use_cache and llm_response.success:
                self._cache_response(cache_key, llm_response)
            
            return llm_response
            
        except APIError as e:
            self.logger.error(f"Claude API error: {str(e)}")
            self.metrics["failed_calls"] += 1
            self.metrics["errors_by_type"][type(e).__name__] = \
                self.metrics["errors_by_type"].get(type(e).__name__, 0) + 1
            
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
        Get structured analysis for trading decisions
        
        Args:
            analysis_type: Type of analysis (technical, fundamental, risk)
            data: Input data for analysis
            context: Additional context
            
        Returns:
            Structured analysis result
        """
        
        # Define system prompts for different analysis types
        system_prompts = {
            "technical": """You are a technical analysis expert. Analyze the provided market data and return a JSON response with:
                - trend_direction: bullish/bearish/neutral
                - strength: 1-10
                - support_levels: list of prices
                - resistance_levels: list of prices
                - entry_points: list of recommended entry prices
                - stop_loss: recommended stop loss price
                - take_profit: list of take profit targets
                - confidence: 1-10""",
            
            "fundamental": """You are a fundamental analysis expert. Analyze the provided company data and return a JSON response with:
                - valuation: undervalued/fair/overvalued
                - growth_prospects: low/medium/high
                - risk_factors: list of key risks
                - catalysts: list of potential catalysts
                - recommendation: strong_buy/buy/hold/sell/strong_sell
                - target_price: price target
                - confidence: 1-10""",
            
            "risk": """You are a risk management expert. Analyze the position and return a JSON response with:
                - risk_level: low/medium/high/extreme
                - position_size: recommended position size as percentage
                - stop_loss: recommended stop loss
                - risk_reward_ratio: numerical ratio
                - correlation_risk: assessment of portfolio correlation
                - max_loss: maximum potential loss
                - confidence: 1-10"""
        }
        
        system_prompt = system_prompts.get(analysis_type, "You are a trading analyst. Provide analysis in JSON format.")
        
        # Format the prompt
        prompt = f"""Analyze the following data and provide structured output:

Data: {json.dumps(data, indent=2)}

{f"Additional Context: {json.dumps(context, indent=2)}" if context else ""}

Respond with valid JSON only."""
        
        # Get completion with JSON parsing
        response = await self.get_completion(
            prompt=prompt,
            system_prompt=system_prompt,
            temperature=0.3,  # Lower temperature for more consistent structured output
            parse_json=True
        )
        
        if response.success:
            try:
                return json.loads(response.content)
            except json.JSONDecodeError:
                self.logger.error("Failed to parse structured response")
                return {"error": "Failed to parse analysis", "raw": response.content}
        else:
            return {"error": response.error}
    
    def _generate_cache_key(self, prompt: str, system_prompt: Optional[str], temperature: float) -> str:
        """Generate cache key for request"""
        import hashlib
        
        key_parts = [prompt, system_prompt or "", str(temperature), self.model]
        key_string = "|".join(key_parts)
        return hashlib.sha256(key_string.encode()).hexdigest()
    
    def _get_cached_response(self, cache_key: str) -> Optional[LLMResponse]:
        """Get cached response if available and not expired"""
        
        if cache_key in self.response_cache:
            cached = self.response_cache[cache_key]
            if datetime.now() - cached.timestamp < self.cache_ttl:
                return cached.response
            else:
                # Remove expired cache entry
                del self.response_cache[cache_key]
        
        return None
    
    def _cache_response(self, cache_key: str, response: LLMResponse):
        """Cache a response"""
        
        self.response_cache[cache_key] = CachedResponse(
            response=response,
            timestamp=datetime.now()
        )
        
        # Clean old cache entries
        self._clean_cache()
    
    def _clean_cache(self):
        """Remove expired cache entries"""
        
        current_time = datetime.now()
        expired_keys = [
            key for key, cached in self.response_cache.items()
            if current_time - cached.timestamp > self.cache_ttl
        ]
        
        for key in expired_keys:
            del self.response_cache[key]
    
    def _extract_json(self, text: str) -> str:
        """Extract JSON from text that may contain other content"""
        
        import re
        
        # Try to find JSON blocks
        json_patterns = [
            r'\{[^{}]*\}',  # Simple JSON object
            r'\{.*?\}(?=\s*$)',  # JSON at end of text
            r'```json\s*(.*?)\s*```',  # JSON in code blocks
            r'\{(?:[^{}]|(?:\{[^{}]*\}))*\}'  # Nested JSON
        ]
        
        for pattern in json_patterns:
            matches = re.findall(pattern, text, re.DOTALL)
            if matches:
                try:
                    # Try to parse the first match
                    json_str = matches[0] if isinstance(matches[0], str) else matches[0][0]
                    parsed = json.loads(json_str)
                    return json.dumps(parsed)
                except json.JSONDecodeError:
                    continue
        
        # If no valid JSON found, return original text
        return text
    
    def _update_metrics(self, response: LLMResponse):
        """Update performance metrics"""
        
        self.metrics["total_calls"] += 1
        
        if response.success:
            self.metrics["successful_calls"] += 1
            self.metrics["total_tokens"] += response.tokens_used
            
            # Estimate cost (Claude-3 Opus pricing as of 2024)
            input_cost = response.usage_details.get("input_tokens", 0) * 0.015 / 1000
            output_cost = response.usage_details.get("output_tokens", 0) * 0.075 / 1000
            self.metrics["total_cost"] += input_cost + output_cost
            
            # Update average response time
            n = self.metrics["successful_calls"]
            avg = self.metrics["average_response_time"]
            self.metrics["average_response_time"] = (avg * (n - 1) + response.processing_time) / n
        else:
            self.metrics["failed_calls"] += 1
    
    def get_metrics_summary(self) -> Dict:
        """Get summary of provider metrics"""
        
        return {
            "total_calls": self.metrics["total_calls"],
            "success_rate": self.metrics["successful_calls"] / max(self.metrics["total_calls"], 1),
            "cache_hit_rate": self.metrics["cache_hits"] / max(self.metrics["total_calls"], 1),
            "average_response_time": round(self.metrics["average_response_time"], 2),
            "total_tokens_used": self.metrics["total_tokens"],
            "estimated_cost": round(self.metrics["total_cost"], 2),
            "errors": self.metrics["errors_by_type"]
        }
    
    def reset_metrics(self):
        """Reset performance metrics"""
        
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
        
        self.logger.info("Metrics reset")

class RateLimiter:
    """Rate limiter for API calls"""
    
    def __init__(self, calls_per_minute: int = 50, calls_per_day: int = 10000):
        self.calls_per_minute = calls_per_minute
        self.calls_per_day = calls_per_day
        self.minute_calls: List[datetime] = []
        self.daily_calls: List[datetime] = []
    
    async def acquire(self) -> bool:
        """Check if we can make another API call"""
        
        now = datetime.now()
        
        # Clean old calls
        minute_ago = now - timedelta(minutes=1)
        day_ago = now - timedelta(days=1)
        
        self.minute_calls = [t for t in self.minute_calls if t > minute_ago]
        self.daily_calls = [t for t in self.daily_calls if t > day_ago]
        
        # Check limits
        if len(self.minute_calls) >= self.calls_per_minute:
            wait_time = (self.minute_calls[0] - minute_ago).total_seconds()
            await asyncio.sleep(wait_time)
            return await self.acquire()  # Retry after waiting
        
        if len(self.daily_calls) >= self.calls_per_day:
            return False
        
        # Record the call
        self.minute_calls.append(now)
        self.daily_calls.append(now)
        
        return True

@dataclass
class CachedResponse:
    """Cached response with timestamp"""
    response: LLMResponse
    timestamp: datetime