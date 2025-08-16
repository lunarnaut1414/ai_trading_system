# config/settings.py - Simple fix for existing setup
"""
AI Trading System Configuration Management
Compatible with your existing .env file structure
"""

import os
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

@dataclass
class TradingConfig:
    """Main configuration class for the trading system"""
    
    # ===== API CREDENTIALS =====
    ALPACA_API_KEY: str = os.getenv("ALPACA_API_KEY", "")
    ALPACA_SECRET_KEY: str = os.getenv("ALPACA_SECRET_KEY", "")
    ALPACA_BASE_URL: str = os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")
    ALPACA_PAPER: bool = os.getenv("ALPACA_PAPER", "true").lower() == "true"
    
    # ===== AI/LLM CONFIGURATION =====
    ANTHROPIC_API_KEY: str = os.getenv("ANTHROPIC_API_KEY", "")
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "anthropic")
    
    # ===== RISK MANAGEMENT (with your existing env var names) =====
    MAX_POSITIONS: int = int(os.getenv("MAX_POSITIONS", "5"))
    MAX_POSITION_SIZE: float = float(os.getenv("MAX_POSITION_SIZE", "0.05"))
    MAX_SECTOR_EXPOSURE: float = float(os.getenv("MAX_SECTOR_EXPOSURE", "0.25"))
    DAILY_LOSS_LIMIT: float = float(os.getenv("DAILY_LOSS_LIMIT", "0.02"))
    MIN_CASH_RESERVE: float = float(os.getenv("MIN_CASH_RESERVE", "0.10"))
    RISK_TOLERANCE: str = os.getenv("RISK_TOLERANCE", "moderate")
    
    # ===== SYSTEM SETTINGS =====
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    DATABASE_URL: str = os.getenv("DATABASE_URL", "sqlite:///./trading_system.db")
    TESTING_MODE: bool = os.getenv("TESTING_MODE", "false").lower() == "true"
    DRY_RUN: bool = os.getenv("DRY_RUN", "false").lower() == "true"
    
    # ===== NOTIFICATION SETTINGS (with your existing env var names) =====
    NOTIFICATION_PROVIDER: str = os.getenv("NOTIFICATION_PROVIDER", "disabled")
    ENABLE_NOTIFICATIONS: bool = os.getenv("ENABLE_NOTIFICATIONS", "false").lower() == "true"
    
    # BlueSky Configuration
    BLUESKY_HANDLE: str = os.getenv("BLUESKY_HANDLE", "")
    BLUESKY_APP_PASSWORD: str = os.getenv("BLUESKY_APP_PASSWORD", "")
    
    # Email Configuration  
    SMTP_SERVER: str = os.getenv("SMTP_SERVER", "smtp.gmail.com")
    SMTP_PORT: int = int(os.getenv("SMTP_PORT", "587"))
    SMTP_USERNAME: str = os.getenv("SMTP_USERNAME", "")
    SMTP_PASSWORD: str = os.getenv("SMTP_PASSWORD", "")
    NOTIFICATION_EMAIL: str = os.getenv("NOTIFICATION_EMAIL", "")
    
    # Webhook Configuration
    WEBHOOK_URL: str = os.getenv("WEBHOOK_URL", "")
    
    # ===== DERIVED PROPERTIES FOR COMPATIBILITY =====
    @property
    def ENVIRONMENT(self) -> str:
        return "development" if self.TESTING_MODE else "production"
    
    @property
    def DEBUG(self) -> bool:
        return self.TESTING_MODE
    
    @property
    def ALPACA_RATE_LIMIT(self) -> int:
        return 200
    
    @property 
    def CLAUDE_RATE_LIMIT(self) -> int:
        return 50
    
    # ===== METHODS =====
    def get_trading_params(self) -> Dict[str, Any]:
        """Get trading parameters as dictionary"""
        return {
            "max_positions": self.MAX_POSITIONS,
            "max_position_size": self.MAX_POSITION_SIZE,
            "max_sector_exposure": self.MAX_SECTOR_EXPOSURE,
            "daily_loss_limit": self.DAILY_LOSS_LIMIT,
            "min_cash_reserve": self.MIN_CASH_RESERVE,
            "risk_tolerance": self.RISK_TOLERANCE
        }
    
    def get_alpaca_config(self) -> Dict[str, Any]:
        """Get Alpaca configuration"""
        return {
            'api_key': self.ALPACA_API_KEY,
            'secret_key': self.ALPACA_SECRET_KEY,
            'paper': self.ALPACA_PAPER,
            'base_url': self.ALPACA_BASE_URL
        }
    
    def get_claude_config(self) -> Dict[str, Any]:
        """Get Claude AI configuration"""
        return {
            'api_key': self.ANTHROPIC_API_KEY,
            'model': 'claude-3-5-sonnet-20241022',
            'max_tokens': 4000,
            'temperature': 0.1,
            'timeout': 30
        }
    
    def get_trading_symbols(self) -> List[str]:
        """Get list of symbols to trade (S&P 500 subset for demo)"""
        return [
            'AAPL', 'MSFT', 'AMZN', 'NVDA', 'GOOGL', 'TSLA', 'GOOG', 'META',
            'UNH', 'XOM', 'LLY', 'JNJ', 'JPM', 'V', 'PG', 'MA', 'AVGO', 'HD',
            'CVX', 'MRK', 'ABBV', 'PEP', 'KO', 'COST', 'WMT', 'BAC', 'TMO',
            'ACN', 'MCD', 'CSCO', 'ABT', 'LIN', 'DHR', 'VZ', 'ADBE', 'TXN',
            'WFC', 'CRM', 'BX', 'PM', 'BMY', 'AMGN', 'RTX', 'SPGI', 'T',
            'LOW', 'HON', 'UPS', 'INTU', 'GS'
        ]
    
    def validate_config(self) -> Dict[str, bool]:
        """Validate configuration parameters"""
        validations = {
            "alpaca_credentials": bool(self.ALPACA_API_KEY and self.ALPACA_SECRET_KEY),
            "llm_credentials": bool(self.ANTHROPIC_API_KEY),
            "risk_parameters": (
                0 < self.MAX_POSITION_SIZE <= 1.0 and
                0 < self.MAX_SECTOR_EXPOSURE <= 1.0 and
                0 < self.DAILY_LOSS_LIMIT <= 1.0 and
                0 <= self.MIN_CASH_RESERVE <= 1.0 and
                self.MAX_POSITIONS > 0
            ),
            "database_config": bool(self.DATABASE_URL),
            "notification_config": True  # Always valid for simplified setup
        }
        return validations

# Global configuration instance (for backward compatibility)
config = TradingConfig()


# Configuration validation function
def validate_config() -> bool:
    """Validate all configuration settings"""
    try:
        config = TradingConfig()
        validations = config.validate_config()
        
        all_valid = all(validations.values())
        
        if all_valid:
            print("✅ Configuration validation passed")
        else:
            print("❌ Configuration validation failed:")
            for key, valid in validations.items():
                status = "✅" if valid else "❌"
                print(f"  {status} {key}")
        
        return all_valid
        
    except Exception as e:
        print(f"❌ Configuration validation failed: {str(e)}")
        return False


if __name__ == "__main__":
    # Test configuration
    if validate_config():
        config = TradingConfig()
        print("\n📋 Trading System Configuration")
        print(f"Environment: {config.ENVIRONMENT}")
        print(f"Paper Trading: {config.ALPACA_PAPER}")
        print(f"LLM Provider: {config.LLM_PROVIDER}")
        print(f"Max Position Size: {config.MAX_POSITION_SIZE * 100}%")
        print(f"Trading Symbols: {len(config.get_trading_symbols())} symbols")
    else:
        print("❌ Configuration setup failed")