"""
pytest configuration for AI Trading System
Shared fixtures and test configuration
"""

import pytest
import asyncio
import os
import sys
from unittest.mock import Mock

# Fix Python path - add project root to sys.path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Now we can import config
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
    # Set test environment variables
    os.environ.update({
        'ALPACA_API_KEY': 'test_api_key',
        'ALPACA_SECRET_KEY': 'test_secret_key',
        'ANTHROPIC_API_KEY': 'test_anthropic_key',
        'ALPACA_PAPER': 'true',
        'ENVIRONMENT': 'testing'
    })
    return TradingConfig()

@pytest.fixture
def mock_alpaca_api():
    """Mock Alpaca API for testing"""
    mock_api = Mock()
    mock_api.get_account.return_value = Mock(
        equity=100000,
        buying_power=50000,
        cash=25000,
        status='ACTIVE',
        trading_blocked=False
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

@pytest.fixture
def mock_claude_client():
    """Mock Claude/Anthropic client for testing"""
    mock = Mock()
    mock.messages.create.return_value = Mock(
        content=[Mock(text="Test analysis response")]
    )
    return mock