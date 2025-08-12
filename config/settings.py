"""
AI Trading System Configuration Management
Centralized configuration with validation and type safety
"""

import os
from typing import Dict, Any, Optional
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
    LLM_PROVIDER: str = "anthropic"  # Fixed to Claude/Anthropic only
    
    # ===== RISK MANAGEMENT =====
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
    
    # ===== NOTIFICATION SETTINGS =====
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
    
    def validate_config(self) -> Dict[str, bool]:
        """Validate configuration parameters"""
        validations = {
            "alpaca_credentials": bool(self.ALPACA_API_KEY and self.ALPACA_SECRET_KEY),
            "llm_credentials": bool(self.ANTHROPIC_API_KEY or self.OPENAI_API_KEY),
            "risk_parameters": (
                0 < self.MAX_POSITION_SIZE <= 1.0 and
                0 < self.MAX_SECTOR_EXPOSURE <= 1.0 and
                0 < self.DAILY_LOSS_LIMIT <= 1.0 and
                0 <= self.MIN_CASH_RESERVE <= 1.0 and
                self.MAX_POSITIONS > 0
            ),
            "database_config": bool(self.DATABASE_URL),
            "notification_config": (
                not self.ENABLE_NOTIFICATIONS or 
                self.NOTIFICATION_PROVIDER == "disabled" or
                (self.NOTIFICATION_PROVIDER == "bluesky" and bool(self.BLUESKY_HANDLE and self.BLUESKY_APP_PASSWORD)) or
                (self.NOTIFICATION_PROVIDER == "email" and bool(self.SMTP_USERNAME and self.SMTP_PASSWORD)) or
                (self.NOTIFICATION_PROVIDER == "webhook" and bool(self.WEBHOOK_URL))
            ),
        }
        return validations

# Global configuration instance
config = TradingConfig()