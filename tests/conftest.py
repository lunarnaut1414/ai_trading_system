"""
pytest configuration for AI Trading System
Shared fixtures and test configuration
"""

import pytest
import asyncio
import os
from unittest.mock import Mock
from config.settings import TradingConfig

@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()

@pytest.fixture
def trading_config():
    """Provide test trading configuration"""
    return TradingConfig()

@pytest.fixture
def mock_alpaca_api():
    """Mock Alpaca API for testing"""
    mock_api = Mock()
    mock_api.get_account.return_value = Mock(
        equity=100000,
        buying_power=50000,
        cash=25000
    )
    return mock_api

@pytest.fixture
def sample_market_data():
    """Sample market data for testing"""
    return {
        'AAPL': {
            'price': 150.00,
            'volume': 1000000,
            'change': 2.5,
            'change_percent': 1.67
        },
        'MSFT': {
            'price': 300.00,
            'volume': 800000,
            'change': -1.5,
            'change_percent': -0.50
        }
    }