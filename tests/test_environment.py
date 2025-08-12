"""
Environment validation tests
Ensures proper setup and configuration
"""

import pytest
import os
from config.settings import TradingConfig
from config.validator import ConfigurationValidator

class TestEnvironmentSetup:
    """Test environment setup and configuration"""
    
    def test_configuration_loading(self):
        """Test configuration loads properly"""
        config = TradingConfig()
        assert config is not None
        assert hasattr(config, 'ALPACA_API_KEY')
        assert hasattr(config, 'MAX_POSITIONS')
    
    def test_risk_parameters_valid(self):
        """Test risk parameters are within valid ranges"""
        config = TradingConfig()
        
        assert 0 < config.MAX_POSITION_SIZE <= 1.0
        assert 0 < config.MAX_SECTOR_EXPOSURE <= 1.0
        assert 0 < config.DAILY_LOSS_LIMIT <= 1.0
        assert 0 <= config.MIN_CASH_RESERVE <= 1.0
        assert config.MAX_POSITIONS > 0
    
    def test_trading_params_dict(self):
        """Test trading parameters dictionary format"""
        config = TradingConfig()
        params = config.get_trading_params()
        
        required_keys = [
            'max_positions', 'max_position_size', 'max_sector_exposure',
            'daily_loss_limit', 'min_cash_reserve', 'risk_tolerance'
        ]
        
        for key in required_keys:
            assert key in params
    
    def test_configuration_validation(self):
        """Test configuration validation method"""
        config = TradingConfig()
        validations = config.validate_config()
        
        assert isinstance(validations, dict)
        assert 'risk_parameters' in validations
        assert validations['risk_parameters'] is True

class TestDependencies:
    """Test required dependencies are available"""
    
    def test_critical_imports(self):
        """Test critical package imports"""
        import pandas as pd
        import sqlalchemy
        import fastapi
        import pydantic
        
        assert pd.__version__ is not None
        assert sqlalchemy.__version__ is not None
    
    def test_alpaca_import(self):
        """Test Alpaca API import"""
        import alpaca_trade_api as tradeapi
        assert tradeapi is not None
    
    @pytest.mark.skipif(
        not os.getenv("ANTHROPIC_API_KEY"),
        reason="No Claude/Anthropic API key available"
    )
    def test_claude_import(self):
        """Test Claude/Anthropic library import"""
        import anthropic
        assert anthropic is not None

if __name__ == "__main__":
    pytest.main([__file__, "-v"])