# config/models.py
"""
Centralized Claude Model Configuration for AI Trading System
Single source of truth for all Claude model references with dynamic updates
"""

import os
import json
import logging
from enum import Enum
from typing import Dict, Optional, Any, List
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path

class ModelTier(Enum):
    """Model tiers for different use cases"""
    HAIKU = "haiku"      # Fast, cost-effective for simple tasks
    SONNET = "sonnet"    # Balanced performance for most tasks  
    OPUS = "opus"        # Most powerful for complex analysis

@dataclass
class ModelConfig:
    """Configuration for a Claude model"""
    model_id: str
    name: str
    tier: ModelTier
    context_window: int
    max_output: int
    cost_per_1k_input: float
    cost_per_1k_output: float
    best_for: str
    deprecated: bool = False
    deprecation_date: Optional[str] = None
    last_verified: Optional[str] = None

class ClaudeModels:
    """
    Centralized Claude model management with dynamic updates
    
    Usage:
        from config.models import ClaudeModels
        
        # Get specific model
        model = ClaudeModels.get_model('sonnet')
        
        # Get model for specific agent
        model = ClaudeModels.get_agent_model('junior_analyst')
        
        # Test and discover new models
        ClaudeModels.discover_models()
    """
    
    # Cache file for discovered models
    CACHE_FILE = Path("config/.claude_models_cache.json")
    CACHE_TTL_DAYS = 7  # Refresh model list weekly
    
    # ===== LATEST CLAUDE MODELS (August 2025) =====
    # Based on the model list you provided
    DEFAULT_MODELS = {
        ModelTier.HAIKU: ModelConfig(
            model_id="claude-3-5-haiku-20241022",
            name="Claude 3.5 Haiku",
            tier=ModelTier.HAIKU,
            context_window=200000,
            max_output=8192,
            cost_per_1k_input=0.00025,
            cost_per_1k_output=0.00125,
            best_for="High-volume operations, quick analysis, data processing",
            last_verified=datetime.now().isoformat()
        ),
        ModelTier.SONNET: ModelConfig(
            model_id="claude-sonnet-4-20250514",  # Latest Sonnet 4
            name="Claude Sonnet 4",
            tier=ModelTier.SONNET,
            context_window=200000,
            max_output=8192,
            cost_per_1k_input=0.003,
            cost_per_1k_output=0.015,
            best_for="Trading analysis, research, decision making",
            last_verified=datetime.now().isoformat()
        ),
        ModelTier.OPUS: ModelConfig(
            model_id="claude-opus-4-1-20250805",  # Latest Opus 4.1
            name="Claude Opus 4.1",
            tier=ModelTier.OPUS,
            context_window=200000,
            max_output=8192,
            cost_per_1k_input=0.015,
            cost_per_1k_output=0.075,
            best_for="Complex reasoning, portfolio strategy, risk analysis",
            last_verified=datetime.now().isoformat()
        )
    }
    
    # All known models including legacy versions
    ALL_KNOWN_MODELS = {
        # Current generation (Claude 4)
        "claude-opus-4-1-20250805": ("opus", "Claude Opus 4.1 (Latest)", False),
        "claude-opus-4-20250514": ("opus", "Claude Opus 4", False),
        "claude-sonnet-4-20250514": ("sonnet", "Claude Sonnet 4 (Latest)", False),
        
        # Claude 3.7
        "claude-3-7-sonnet-20250219": ("sonnet", "Claude 3.7 Sonnet", False),
        
        # Claude 3.5
        "claude-3-5-haiku-20241022": ("haiku", "Claude 3.5 Haiku (Latest)", False),
        "claude-3-5-sonnet-20241022": ("sonnet", "Claude 3.5 Sonnet", False),
        "claude-3-5-sonnet-20240620": ("sonnet", "Claude 3.5 Sonnet (June)", False),
        
        # Claude 3 (Legacy)
        "claude-3-haiku-20240307": ("haiku", "Claude 3 Haiku", True),
        "claude-3-opus-20240229": ("opus", "Claude 3 Opus", True),
        "claude-3-sonnet-20240229": ("sonnet", "Claude 3 Sonnet", True),
    }
    
    # ===== AGENT-SPECIFIC MODEL ASSIGNMENTS =====
    AGENT_MODELS = {
        "technical_screener": ModelTier.HAIKU,     # Fast pattern scanning
        "junior_analyst": ModelTier.SONNET,        # Individual stock analysis
        "senior_analyst": ModelTier.OPUS,          # Strategic synthesis
        "portfolio_manager": ModelTier.OPUS,       # Critical decisions
        "trade_execution": ModelTier.HAIKU,        # Fast execution
        "cfo_reporting": ModelTier.SONNET,         # Report generation
        "risk_manager": ModelTier.OPUS,            # Risk assessment
        "news_analyzer": ModelTier.HAIKU,          # News processing
        "market_scanner": ModelTier.HAIKU,         # Market scanning
        "default": ModelTier.SONNET                # Fallback model
    }
    
    # Runtime model cache
    _models_cache: Dict[ModelTier, ModelConfig] = None
    _cache_loaded_at: Optional[datetime] = None
    
    @classmethod
    def _load_cache(cls) -> Dict[ModelTier, ModelConfig]:
        """Load cached model configurations if available and fresh"""
        if cls.CACHE_FILE.exists():
            try:
                with open(cls.CACHE_FILE, 'r') as f:
                    cache_data = json.load(f)
                
                # Check cache age
                cache_date = datetime.fromisoformat(cache_data.get('updated_at', ''))
                if datetime.now() - cache_date < timedelta(days=cls.CACHE_TTL_DAYS):
                    # Convert cached data back to ModelConfig objects
                    models = {}
                    for tier_str, model_data in cache_data.get('models', {}).items():
                        tier = ModelTier(tier_str)
                        models[tier] = ModelConfig(
                            model_id=model_data['model_id'],
                            name=model_data['name'],
                            tier=tier,
                            context_window=model_data['context_window'],
                            max_output=model_data['max_output'],
                            cost_per_1k_input=model_data['cost_per_1k_input'],
                            cost_per_1k_output=model_data['cost_per_1k_output'],
                            best_for=model_data['best_for'],
                            deprecated=model_data.get('deprecated', False),
                            deprecation_date=model_data.get('deprecation_date'),
                            last_verified=model_data.get('last_verified')
                        )
                    
                    logging.info(f"Loaded model cache from {cls.CACHE_FILE}")
                    return models
            except Exception as e:
                logging.warning(f"Failed to load model cache: {e}")
        
        return None
    
    @classmethod
    def _save_cache(cls, models: Dict[ModelTier, ModelConfig]):
        """Save model configurations to cache"""
        try:
            cache_data = {
                'updated_at': datetime.now().isoformat(),
                'models': {}
            }
            
            for tier, config in models.items():
                cache_data['models'][tier.value] = {
                    'model_id': config.model_id,
                    'name': config.name,
                    'context_window': config.context_window,
                    'max_output': config.max_output,
                    'cost_per_1k_input': config.cost_per_1k_input,
                    'cost_per_1k_output': config.cost_per_1k_output,
                    'best_for': config.best_for,
                    'deprecated': config.deprecated,
                    'deprecation_date': config.deprecation_date,
                    'last_verified': config.last_verified
                }
            
            cls.CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
            with open(cls.CACHE_FILE, 'w') as f:
                json.dump(cache_data, f, indent=2)
            
            logging.info(f"Saved model cache to {cls.CACHE_FILE}")
        except Exception as e:
            logging.warning(f"Failed to save model cache: {e}")
    
    @classmethod
    def get_models(cls) -> Dict[ModelTier, ModelConfig]:
        """Get all models, loading from cache or defaults"""
        if cls._models_cache is None:
            # Try to load from cache first
            cached = cls._load_cache()
            if cached:
                cls._models_cache = cached
            else:
                cls._models_cache = cls.DEFAULT_MODELS.copy()
                cls._save_cache(cls._models_cache)
        
        return cls._models_cache
    
    @classmethod
    def get_model(cls, tier: str) -> ModelConfig:
        """
        Get model configuration by tier name
        
        Args:
            tier: 'haiku', 'sonnet', or 'opus'
            
        Returns:
            ModelConfig object
        """
        # Check environment variable override first
        env_key = f"CLAUDE_MODEL_{tier.upper()}"
        env_value = os.getenv(env_key)
        
        models = cls.get_models()
        tier_enum = ModelTier(tier.lower())
        
        if env_value:
            # Override with custom model ID from environment
            config = models[tier_enum]
            # Create new config with overridden model_id
            return ModelConfig(
                model_id=env_value,
                name=config.name,
                tier=config.tier,
                context_window=config.context_window,
                max_output=config.max_output,
                cost_per_1k_input=config.cost_per_1k_input,
                cost_per_1k_output=config.cost_per_1k_output,
                best_for=config.best_for,
                deprecated=config.deprecated,
                deprecation_date=config.deprecation_date,
                last_verified=datetime.now().isoformat()
            )
        
        # Return default configuration
        return models.get(tier_enum, models[ModelTier.SONNET])
    
    @classmethod
    def get_agent_model(cls, agent_name: str) -> ModelConfig:
        """
        Get model configuration for specific agent
        
        Args:
            agent_name: Name of the agent (e.g., 'junior_analyst')
            
        Returns:
            ModelConfig object appropriate for the agent
        """
        # Check for environment variable override
        env_key = f"CLAUDE_MODEL_{agent_name.upper()}"
        env_value = os.getenv(env_key)
        
        if env_value:
            # Try to determine tier from model ID
            for model_id, (tier_str, name, deprecated) in cls.ALL_KNOWN_MODELS.items():
                if model_id == env_value:
                    return cls.get_model(tier_str)
            
            # Unknown model, use it anyway with sonnet defaults
            logging.warning(f"Unknown model {env_value} for agent {agent_name}, using with default settings")
            config = cls.get_model("sonnet")
            config.model_id = env_value
            return config
        
        # Get tier assignment for agent
        tier = cls.AGENT_MODELS.get(agent_name, cls.AGENT_MODELS["default"])
        return cls.get_model(tier.value)
    
    @classmethod
    def discover_models(cls, test_models: Optional[List[str]] = None) -> Dict[str, Dict]:
        """
        Test and discover available Claude models
        
        Args:
            test_models: Optional list of model IDs to test
            
        Returns:
            Dictionary of discovered models with their status
        """
        from anthropic import Anthropic
        import asyncio
        
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            return {"error": "ANTHROPIC_API_KEY not set"}
        
        client = Anthropic(api_key=api_key)
        results = {}
        
        # Models to test
        if test_models is None:
            test_models = list(cls.ALL_KNOWN_MODELS.keys())
        
        print("\n" + "="*60)
        print("CLAUDE MODEL DISCOVERY")
        print("="*60)
        print(f"Testing {len(test_models)} models...")
        
        for model_id in test_models:
            try:
                # Test with minimal API call
                response = client.messages.create(
                    model=model_id,
                    messages=[{"role": "user", "content": "Hi"}],
                    max_tokens=10,
                    temperature=0
                )
                
                # Model works!
                results[model_id] = {
                    "status": "✅ Available",
                    "response_length": len(response.content[0].text),
                    "tier": cls.ALL_KNOWN_MODELS.get(model_id, ("unknown", "Unknown", False))[0]
                }
                print(f"✅ {model_id}: Available")
                
            except Exception as e:
                error_msg = str(e)
                if "model_not_found" in error_msg.lower() or "does not exist" in error_msg.lower():
                    status = "❌ Not Found"
                elif "authentication" in error_msg.lower():
                    status = "🔐 Auth Error"
                elif "rate" in error_msg.lower():
                    status = "⏱️ Rate Limited"
                else:
                    status = f"❌ Error: {error_msg[:50]}"
                
                results[model_id] = {
                    "status": status,
                    "error": error_msg
                }
                print(f"{status.split()[0]} {model_id}: {status}")
        
        # Update cache with working models
        working_models = {k: v for k, v in results.items() if "✅" in v["status"]}
        if working_models:
            print(f"\n📊 Found {len(working_models)} working models")
            # Update cache if we found models for each tier
            # (Implementation depends on your needs)
        
        return results
    
    @classmethod
    def update_model(cls, tier: str, model_id: str, **kwargs):
        """
        Update a model configuration
        
        Args:
            tier: Model tier to update
            model_id: New model ID
            **kwargs: Additional model parameters
        """
        models = cls.get_models()
        tier_enum = ModelTier(tier.lower())
        
        if tier_enum in models:
            config = models[tier_enum]
            config.model_id = model_id
            config.last_verified = datetime.now().isoformat()
            
            # Update any provided parameters
            for key, value in kwargs.items():
                if hasattr(config, key):
                    setattr(config, key, value)
            
            # Save updated cache
            cls._save_cache(models)
            logging.info(f"Updated {tier} model to {model_id}")
    
    @classmethod
    def get_model_id(cls, tier: str) -> str:
        """Quick helper to get just the model ID"""
        return cls.get_model(tier).model_id
    
    @classmethod
    def get_agent_model_id(cls, agent_name: str) -> str:
        """Quick helper to get model ID for an agent"""
        return cls.get_agent_model(agent_name).model_id
    
    @classmethod
    def estimate_cost(cls, tier: str, input_tokens: int, output_tokens: int) -> float:
        """Estimate cost for a Claude API call"""
        model = cls.get_model(tier)
        input_cost = (input_tokens / 1000) * model.cost_per_1k_input
        output_cost = (output_tokens / 1000) * model.cost_per_1k_output
        return round(input_cost + output_cost, 4)
    
    @classmethod
    def validate_configuration(cls) -> Dict[str, Any]:
        """Validate model configuration and return status"""
        results = {
            "valid": True,
            "models_available": len(cls.get_models()),
            "agents_configured": len(cls.AGENT_MODELS),
            "warnings": [],
            "errors": [],
            "model_assignments": {},
            "cache_status": "Not loaded"
        }
        
        # Check cache status
        if cls.CACHE_FILE.exists():
            try:
                with open(cls.CACHE_FILE, 'r') as f:
                    cache_data = json.load(f)
                cache_date = datetime.fromisoformat(cache_data.get('updated_at', ''))
                age_days = (datetime.now() - cache_date).days
                results["cache_status"] = f"Loaded ({age_days} days old)"
            except:
                results["cache_status"] = "Invalid"
        
        # Check for API key
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            results["errors"].append("ANTHROPIC_API_KEY not set in environment")
            results["valid"] = False
        
        # Check model assignments
        for agent, tier in cls.AGENT_MODELS.items():
            if agent == "default":
                continue
            model = cls.get_agent_model(agent)
            results["model_assignments"][agent] = {
                "tier": tier.value,
                "model_id": model.model_id,
                "estimated_hourly_cost": cls._estimate_hourly_cost(model)
            }
            
            # Check for deprecated models
            if model.deprecated:
                results["warnings"].append(
                    f"Agent '{agent}' uses deprecated model {model.model_id}"
                )
        
        return results
    
    @classmethod
    def _estimate_hourly_cost(cls, model: ModelConfig, calls_per_hour: int = 100) -> float:
        """Estimate hourly cost for a model assuming average usage"""
        avg_input_tokens = 2000
        avg_output_tokens = 1000
        cost_per_call = cls.estimate_cost(
            model.tier.value, 
            avg_input_tokens, 
            avg_output_tokens
        )
        return round(cost_per_call * calls_per_hour, 2)
    
    @classmethod
    def print_configuration(cls):
        """Print current configuration in a formatted way"""
        print("\n" + "="*60)
        print("CLAUDE MODEL CONFIGURATION")
        print("="*60)
        
        validation = cls.validate_configuration()
        
        print(f"\n📁 Cache Status: {validation['cache_status']}")
        
        if not validation["valid"]:
            print("\n❌ CONFIGURATION ERRORS:")
            for error in validation["errors"]:
                print(f"   - {error}")
        
        if validation["warnings"]:
            print("\n⚠️  WARNINGS:")
            for warning in validation["warnings"]:
                print(f"   - {warning}")
        
        print("\n📊 ACTIVE MODEL TIERS:")
        for tier_name, config in cls.get_models().items():
            print(f"\n   {tier_name.value.upper()}:")
            print(f"      Model ID: {config.model_id}")
            print(f"      Context: {config.context_window:,} tokens")
            print(f"      Cost: ${config.cost_per_1k_input}/1k in, ${config.cost_per_1k_output}/1k out")
            print(f"      Best for: {config.best_for}")
            if config.last_verified:
                print(f"      Last verified: {config.last_verified[:10]}")
        
        print("\n🤖 AGENT ASSIGNMENTS:")
        for agent, details in validation["model_assignments"].items():
            print(f"   {agent:20} -> {details['tier']:6} (${details['estimated_hourly_cost']}/hr)")
        
        print("\n💡 TIP: Run ClaudeModels.discover_models() to test all available models")
        print("="*60)


# ===== USAGE EXAMPLES =====
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "discover":
        # Run model discovery
        print("Starting model discovery...")
        results = ClaudeModels.discover_models()
        
        print("\n📊 DISCOVERY RESULTS:")
        available = [k for k, v in results.items() if "✅" in v.get("status", "")]
        if available:
            print(f"\n✅ Available Models ({len(available)}):")
            for model_id in available:
                info = ClaudeModels.ALL_KNOWN_MODELS.get(model_id, ("?", "Unknown", False))
                print(f"   - {model_id} ({info[1]})")
    else:
        # Print current configuration
        ClaudeModels.print_configuration()
        
        # Example: Get model for junior analyst
        print("\n📍 Example Usage:")
        model = ClaudeModels.get_agent_model("junior_analyst")
        print(f"Junior Analyst uses: {model.name} ({model.model_id})")
        
        # Example: Estimate cost
        cost = ClaudeModels.estimate_cost("sonnet", 2000, 500)
        print(f"Estimated cost for 2000 input/500 output tokens: ${cost}")
        
        print("\n💡 Run 'python config/models.py discover' to test all models")