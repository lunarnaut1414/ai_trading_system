# config/claude_config.py
"""
Claude Configuration for AI Trading System
"""

import os
from typing import Optional

class ClaudeConfig:
    """Claude-specific configuration"""
    
    # API Configuration
    ANTHROPIC_API_KEY: str = os.getenv("ANTHROPIC_API_KEY", "")
    
    # Model Selection - Updated to use current models
    # Options: 
    # - claude-sonnet-4-20250514 (newest, most capable)
    # - claude-opus-4-1-20250805 (most powerful)
    # - claude-3-5-haiku-20241022 (fastest, most cost-effective)
    CLAUDE_MODEL: str = os.getenv("CLAUDE_MODEL", "claude-sonnet-4-20250514")
    
    # Rate Limits
    RATE_LIMIT_PER_MINUTE: int = 50
    RATE_LIMIT_PER_DAY: int = 10000
    
    # Retry Configuration
    MAX_RETRIES: int = 3
    RETRY_DELAY_SECONDS: float = 1.0
    EXPONENTIAL_BACKOFF: bool = True
    
    # Cache Configuration
    ENABLE_CACHE: bool = True
    CACHE_TTL_MINUTES: int = 5
    
    # Performance Settings
    DEFAULT_MAX_TOKENS: int = 4000
    DEFAULT_TEMPERATURE: float = 0.7
    
    # Cost Management
    TRACK_COSTS: bool = True
    DAILY_COST_LIMIT: float = 100.0  # USD
    ALERT_COST_THRESHOLD: float = 50.0  # Alert when daily cost exceeds this
    
    @classmethod
    def validate(cls) -> bool:
        """Validate configuration"""
        if not cls.ANTHROPIC_API_KEY:
            print("❌ ANTHROPIC_API_KEY not set in environment variables")
            print("   Please set: export ANTHROPIC_API_KEY='your-api-key'")
            return False
        
        # Check if model is valid
        valid_models = [
            "claude-sonnet-4-20250514",
            "claude-opus-4-1-20250805", 
            "claude-3-5-haiku-20241022"
        ]
        
        if cls.CLAUDE_MODEL not in valid_models:
            print(f"⚠️ Invalid model: {cls.CLAUDE_MODEL}")
            print(f"   Valid models: {', '.join(valid_models)}")
            print(f"   Defaulting to: claude-sonnet-4-20250514")
            cls.CLAUDE_MODEL = "claude-sonnet-4-20250514"
        
        # Warn about deprecated models
        if cls.CLAUDE_MODEL == "claude-3-opus-20240229":
            print(f"⚠️ Warning: {cls.CLAUDE_MODEL} will be deprecated on July 21, 2025")
            print("   Consider switching to claude-3-5-sonnet-20241022")
        
        print(f"✅ Claude configuration valid")
        print(f"   Model: {cls.CLAUDE_MODEL}")
        print(f"   Rate limits: {cls.RATE_LIMIT_PER_MINUTE}/min, {cls.RATE_LIMIT_PER_DAY}/day")
        
        return True
    
    @classmethod
    def get_model_info(cls, model: Optional[str] = None) -> dict:
        """Get information about a specific model"""
        model = model or cls.CLAUDE_MODEL
        
        model_info = {
            "claude-sonnet-4-20250514": {
                "name": "Claude Sonnet 4",
                "context_window": 200000,
                "max_output": 8192,
                "cost_per_1k_input": 0.003,
                "cost_per_1k_output": 0.015,
                "strengths": "Newest model, best performance and capabilities",
                "best_for": "Complex trading analysis, research, and decision making"
            },
            "claude-opus-4-1-20250805": {
                "name": "Claude Opus 4.1",
                "context_window": 200000,
                "max_output": 8192,
                "cost_per_1k_input": 0.015,
                "cost_per_1k_output": 0.075,
                "strengths": "Most powerful and sophisticated model",
                "best_for": "Complex reasoning and advanced analysis tasks"
            },
            "claude-3-5-haiku-20241022": {
                "name": "Claude 3.5 Haiku",
                "context_window": 200000,
                "max_output": 8192,
                "cost_per_1k_input": 0.00025,
                "cost_per_1k_output": 0.00125,
                "strengths": "Fast and cost-effective, good for simple tasks",
                "best_for": "Quick analysis, data processing, and high-volume operations"
            }
        }
        
        return model_info.get(model, {
            "error": f"Unknown model: {model}",
            "available_models": list(model_info.keys())
        })

# Quick validation when module is imported
if __name__ == "__main__":
    if ClaudeConfig.validate():
        info = ClaudeConfig.get_model_info()
        print(f"\n📊 Model Information:")
        for key, value in info.items():
            if key != "error":
                print(f"   {key}: {value}")