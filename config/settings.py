"""
AI Trading System Configuration
Centralized configuration management with environment variable support
"""

import os
from typing import Dict, List, Any, Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class TradingConfig:
    """Trading system configuration with environment variable support"""
    
    def __init__(self):
        # ===== ENVIRONMENT CONFIGURATION =====
        self.ENVIRONMENT = os.getenv('ENVIRONMENT', 'development')
        self.DEBUG = os.getenv('DEBUG', 'true').lower() == 'true'
        
        # ===== API CONFIGURATION =====
        self.ALPACA_API_KEY = os.getenv('ALPACA_API_KEY', '')
        self.ALPACA_SECRET_KEY = os.getenv('ALPACA_SECRET_KEY', '')
        self.ALPACA_BASE_URL = os.getenv('ALPACA_BASE_URL', 'https://paper-api.alpaca.markets')
        self.ALPACA_PAPER = os.getenv('ALPACA_PAPER', 'true').lower() == 'true'
        
        # ===== AI/LLM CONFIGURATION =====
        self.LLM_PROVIDER = os.getenv('LLM_PROVIDER', 'anthropic')  # Use Claude
        self.ANTHROPIC_API_KEY = os.getenv('ANTHROPIC_API_KEY', '')
        self.ANTHROPIC_MODEL = os.getenv('ANTHROPIC_MODEL', 'claude-3-sonnet-20240229')
        
        # ===== TRADING PARAMETERS =====
        self.MAX_POSITIONS = int(os.getenv('MAX_POSITIONS', '10'))
        self.MAX_POSITION_SIZE = float(os.getenv('MAX_POSITION_SIZE', '0.05'))  # 5% max per position
        self.MAX_SECTOR_EXPOSURE = float(os.getenv('MAX_SECTOR_EXPOSURE', '0.25'))  # 25% max per sector
        self.DAILY_LOSS_LIMIT = float(os.getenv('DAILY_LOSS_LIMIT', '0.02'))  # 2% daily loss limit
        self.MIN_CASH_RESERVE = float(os.getenv('MIN_CASH_RESERVE', '0.10'))  # 10% cash reserve
        self.RISK_TOLERANCE = os.getenv('RISK_TOLERANCE', 'moderate')
        
        # ===== DATABASE CONFIGURATION =====
        self.DATABASE_URL = os.getenv('DATABASE_URL', 'sqlite:///trading_system.db')
        
        # ===== LOGGING CONFIGURATION =====
        self.LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
        self.LOG_FILE = os.getenv('LOG_FILE', 'logs/trading_system.log')
        
        # ===== TRADING SYMBOLS =====
        default_symbols = 'AAPL,MSFT,GOOGL,AMZN,TSLA,NVDA,META,NFLX,CRM,ADBE'
        self.TRADING_SYMBOLS = os.getenv('TRADING_SYMBOLS', default_symbols).split(',')
        
        # ===== REPORTING CONFIGURATION =====
        self.REPORT_OUTPUT_DIR = os.getenv('REPORT_OUTPUT_DIR', 'reports')
        self.ENABLE_NOTIFICATIONS = os.getenv('ENABLE_NOTIFICATIONS', 'false').lower() == 'true'
    
    def get_alpaca_config(self) -> Dict[str, Any]:
        """Get Alpaca API configuration"""
        return {
            'api_key': self.ALPACA_API_KEY,
            'secret_key': self.ALPACA_SECRET_KEY,
            'base_url': self.ALPACA_BASE_URL,
            'paper': self.ALPACA_PAPER
        }
    
    def get_llm_config(self) -> Dict[str, Any]:
        """Get LLM configuration for Claude"""
        return {
            'provider': self.LLM_PROVIDER,
            'api_key': self.ANTHROPIC_API_KEY,
            'model': self.ANTHROPIC_MODEL
        }
    
    def get_trading_params(self) -> Dict[str, Any]:
        """Get trading parameters"""
        return {
            'max_positions': self.MAX_POSITIONS,
            'max_position_size': self.MAX_POSITION_SIZE,
            'max_sector_exposure': self.MAX_SECTOR_EXPOSURE,
            'daily_loss_limit': self.DAILY_LOSS_LIMIT,
            'min_cash_reserve': self.MIN_CASH_RESERVE,
            'risk_tolerance': self.RISK_TOLERANCE
        }
    
    def get_trading_symbols(self) -> List[str]:
        """Get list of trading symbols"""
        return [symbol.strip().upper() for symbol in self.TRADING_SYMBOLS]
    
    def validate_config(self) -> Dict[str, bool]:
        """Validate configuration settings"""
        validations = {
            "alpaca_api": bool(self.ALPACA_API_KEY and self.ALPACA_SECRET_KEY),
            "llm_api": bool(self.ANTHROPIC_API_KEY),
            "risk_parameters": (
                0 < self.MAX_POSITION_SIZE <= 1.0 and
                0 < self.MAX_SECTOR_EXPOSURE <= 1.0 and
                0 < self.DAILY_LOSS_LIMIT <= 1.0 and
                0 <= self.MIN_CASH_RESERVE <= 1.0 and
                self.MAX_POSITIONS > 0
            ),
            "database_config": bool(self.DATABASE_URL),
            "trading_symbols": len(self.get_trading_symbols()) > 0
        }
        return validations
    
    def is_paper_trading(self) -> bool:
        """Check if we're in paper trading mode"""
        return self.ALPACA_PAPER
    
    def __str__(self) -> str:
        """String representation of configuration"""
        return f"TradingConfig(env={self.ENVIRONMENT}, paper={self.ALPACA_PAPER}, symbols={len(self.TRADING_SYMBOLS)})"

# Global configuration instance
config = TradingConfig()

# Validation function
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
        print(f"Max Positions: {config.MAX_POSITIONS}")
        print(f"Max Position Size: {config.MAX_POSITION_SIZE * 100}%")
        print(f"Trading Symbols: {len(config.get_trading_symbols())} symbols")
    else:
        print("❌ Configuration setup failed")