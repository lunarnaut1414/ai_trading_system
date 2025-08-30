# config/claude_config.py
"""
Configuration for Claude-only implementation
"""

import os
from typing import Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class ClaudeConfig:
    """Configuration for Claude LLM Provider"""
    
    # API Configuration
    ANTHROPIC_API_KEY: str = os.getenv("ANTHROPIC_API_KEY", "")
    
    # Model Selection (in order of preference)
    # - claude-3-opus-20240229: Most capable, best for complex analysis
    # - claude-3-sonnet-20240229: Balanced performance and cost
    # - claude-3-haiku-20240307: Fastest and most cost-effective
    CLAUDE_MODEL: str = os.getenv("CLAUDE_MODEL", "claude-3-sonnet-20240229")
    
    # Rate Limiting
    CLAUDE_CALLS_PER_MINUTE: int = 50
    CLAUDE_CALLS_PER_DAY: int = 10000
    
    # Retry Configuration
    MAX_RETRIES: int = 3
    RETRY_DELAY_BASE: int = 4  # Base delay in seconds for exponential backoff
    RETRY_DELAY_MAX: int = 60  # Maximum delay in seconds
    
    # Cache Configuration
    CACHE_TTL_MINUTES: int = 5  # Cache time-to-live
    ENABLE_CACHE: bool = True
    
    # Response Configuration
    DEFAULT_MAX_TOKENS: int = 4000
    DEFAULT_TEMPERATURE: float = 0.7
    
    # For structured analysis (lower temperature for consistency)
    ANALYSIS_TEMPERATURE: float = 0.3
    ANALYSIS_MAX_TOKENS: int = 2000
    
    @classmethod
    def validate(cls) -> bool:
        """Validate configuration"""
        
        if not cls.ANTHROPIC_API_KEY:
            print("❌ ANTHROPIC_API_KEY not set in environment")
            return False
        
        if not cls.ANTHROPIC_API_KEY.startswith("sk-ant-"):
            print("⚠️ ANTHROPIC_API_KEY format may be incorrect")
            
        valid_models = [
            "claude-3-opus-20240229",
            "claude-3-sonnet-20240229", 
            "claude-3-haiku-20240307"
        ]
        
        if cls.CLAUDE_MODEL not in valid_models:
            print(f"⚠️ Unknown Claude model: {cls.CLAUDE_MODEL}")
            print(f"   Valid models: {', '.join(valid_models)}")
        
        print(f"✅ Claude configuration valid")
        print(f"   Model: {cls.CLAUDE_MODEL}")
        print(f"   Rate limits: {cls.CLAUDE_CALLS_PER_MINUTE}/min, {cls.CLAUDE_CALLS_PER_DAY}/day")
        
        return True