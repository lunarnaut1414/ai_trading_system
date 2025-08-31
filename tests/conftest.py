# tests/conftest.py
"""
Pytest configuration and shared fixtures for AI Trading System tests

This file is automatically loaded by pytest and provides:
- Shared fixtures available to all tests
- Test configuration and setup
- Custom hooks and plugins
"""

import asyncio
import logging
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any
from unittest.mock import Mock, AsyncMock, MagicMock

import pytest
import pytest_asyncio

# Add project root to Python path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# ==============================================================================
# PYTEST CONFIGURATION
# ==============================================================================

def pytest_configure(config):
    """Configure pytest with custom settings and markers"""
    
    # Register custom markers
    config.addinivalue_line("markers", "unit: Unit tests - fast, isolated component tests")
    config.addinivalue_line("markers", "integration: Integration tests - test component interactions")
    config.addinivalue_line("markers", "stress: Stress tests - performance and load testing")
    config.addinivalue_line("markers", "slow: Slow running tests (>5 seconds)")
    config.addinivalue_line("markers", "requires_api: Tests requiring external API access")
    config.addinivalue_line("markers", "smoke: Smoke tests - basic functionality verification")
    
    # Set up test environment
    os.environ['TESTING'] = 'true'
    os.environ['LOG_LEVEL'] = 'DEBUG'


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers based on test names"""
    
    for item in items:
        # Auto-mark async tests
        if asyncio.iscoroutinefunction(item.function):
            item.add_marker(pytest.mark.asyncio)
        
        # Auto-mark stress tests
        if 'stress' in item.nodeid or 'performance' in item.nodeid:
            item.add_marker(pytest.mark.stress)
        
        # Auto-mark integration tests
        if 'integration' in item.nodeid:
            item.add_marker(pytest.mark.integration)


# ==============================================================================
# ASYNC CONFIGURATION
# ==============================================================================

@pytest.fixture(scope="session")
def event_loop():
    """Create an event loop for the test session"""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


# ==============================================================================
# MOCK PROVIDERS
# ==============================================================================

@pytest.fixture
def mock_llm_provider():
    """Mock Claude LLM provider for testing"""
    provider = MagicMock()
    
    async def generate_analysis(prompt: str, context: Dict) -> Dict:
        """Mock LLM analysis generation"""
        await asyncio.sleep(0.01)  # Simulate API delay
        
        return {
            "executive_summary": "Test analysis summary based on provided context.",
            "key_decisions": [
                "Decision 1 based on analysis",
                "Decision 2 based on analysis",
                "Decision 3 based on analysis"
            ],
            "positioning_advice": "Test positioning advice",
            "risk_priorities": ["Risk 1", "Risk 2"],
            "time_horizon_strategy": "Balanced approach across time horizons",
            "confidence_score": 0.85
        }
    
    provider.generate_analysis = generate_analysis
    provider.usage_stats = {"requests": 0, "tokens": 0, "errors": 0}
    
    return provider


@pytest.fixture
def mock_alpaca_provider():
    """Mock Alpaca market data provider"""
    provider = MagicMock()
    
    async def get_market_data(symbols: List[str], timeframe: str = '1Day', limit: int = 20) -> Dict:
        """Mock market data retrieval"""
        await asyncio.sleep(0.01)  # Simulate API delay
        
        data = {}
        for symbol in symbols:
            base_price = 100 + hash(symbol) % 400  # Deterministic but varied prices
            bars = []
            
            for i in range(limit):
                date = datetime.now() - timedelta(days=limit-i)
                variation = 1 + (0.02 * ((i + hash(symbol)) % 5 - 2))
                
                bars.append({
                    'timestamp': date.isoformat(),
                    'open': base_price * variation,
                    'high': base_price * variation * 1.02,
                    'low': base_price * variation * 0.98,
                    'close': base_price * variation * 1.01,
                    'volume': 1000000 + (i * 10000)
                })
            
            data[symbol] = bars
        
        return data
    
    async def get_account() -> Dict:
        """Mock account information"""
        return {
            'buying_power': 100000.00,
            'portfolio_value': 250000.00,
            'cash': 100000.00,
            'positions_value': 150000.00
        }
    
    provider.get_market_data = get_market_data
    provider.get_account = get_account
    
    return provider


# ==============================================================================
# TEST CONFIGURATIONS
# ==============================================================================

@pytest.fixture
def test_config():
    """Test configuration object"""
    config = MagicMock()
    
    # Trading parameters
    config.MAX_POSITIONS = 10
    config.MAX_POSITION_SIZE = 0.05
    config.MAX_SECTOR_EXPOSURE = 0.25
    config.DAILY_LOSS_LIMIT = 0.02
    config.MIN_CASH_RESERVE = 0.10
    
    # Risk parameters
    config.RISK_TOLERANCE = "moderate"
    config.STOP_LOSS_PERCENTAGE = 0.08
    
    # System parameters
    config.LOG_LEVEL = "DEBUG"
    config.ENVIRONMENT = "testing"
    config.CACHE_ENABLED = False
    
    # API keys (mock)
    config.ANTHROPIC_API_KEY = "test_anthropic_key"
    config.ALPACA_API_KEY = "test_alpaca_key"
    config.ALPACA_SECRET_KEY = "test_alpaca_secret"
    
    return config


# ==============================================================================
# TEST DATA FIXTURES
# ==============================================================================

@pytest.fixture
def sample_stock_data():
    """Sample stock market data"""
    return {
        'AAPL': {
            'current_price': 185.50,
            'volume': 75000000,
            'market_cap': 2.95e12,
            'pe_ratio': 30.5,
            'dividend_yield': 0.0044,
            'beta': 1.25
        },
        'MSFT': {
            'current_price': 380.25,
            'volume': 25000000,
            'market_cap': 2.83e12,
            'pe_ratio': 35.2,
            'dividend_yield': 0.0072,
            'beta': 0.93
        },
        'JPM': {
            'current_price': 155.75,
            'volume': 12000000,
            'market_cap': 4.5e11,
            'pe_ratio': 11.8,
            'dividend_yield': 0.026,
            'beta': 1.15
        }
    }


@pytest.fixture
def sample_junior_reports():
    """Sample junior analyst reports"""
    return [
        {
            'ticker': 'AAPL',
            'recommendation': 'BUY',
            'confidence': 8,
            'analysis_status': 'success',
            'target_upside_percent': 15,
            'stop_loss_percent': 8,
            'time_horizon': 'medium',
            'sector': 'Technology',
            'market_cap': 'Large',
            'risk_assessment': {
                'risk_level': 'medium',
                'risk_score': 5.5,
                'key_risks': ['Market volatility', 'Competition']
            },
            'technical_analysis': {
                'technical_score': 7.5,
                'volume_ratio': 1.2,
                'rsi': 55,
                'macd_signal': 'bullish',
                'support_level': 180.00,
                'resistance_level': 195.00
            },
            'catalysts': ['Product launch', 'Earnings beat', 'Share buyback'],
            'thesis': 'Strong product cycle with services growth acceleration.',
            'timestamp': datetime.now().isoformat()
        },
        {
            'ticker': 'MSFT',
            'recommendation': 'BUY',
            'confidence': 9,
            'analysis_status': 'success',
            'target_upside_percent': 12,
            'stop_loss_percent': 6,
            'time_horizon': 'long',
            'sector': 'Technology',
            'market_cap': 'Large',
            'risk_assessment': {
                'risk_level': 'low',
                'risk_score': 3.5,
                'key_risks': ['Regulatory', 'Cloud competition']
            },
            'technical_analysis': {
                'technical_score': 8.2,
                'volume_ratio': 1.5,
                'rsi': 58,
                'macd_signal': 'bullish',
                'support_level': 370.00,
                'resistance_level': 400.00
            },
            'catalysts': ['AI adoption', 'Cloud growth', 'Enterprise wins'],
            'thesis': 'AI leadership position with strong cloud momentum.',
            'timestamp': datetime.now().isoformat()
        }
    ]


@pytest.fixture
def sample_portfolio():
    """Sample portfolio state"""
    return {
        'positions': [
            {
                'ticker': 'SPY',
                'quantity': 100,
                'entry_price': 440.00,
                'current_price': 450.00,
                'value': 45000.00,
                'weight': 0.18,
                'sector': 'Index',
                'unrealized_pnl': 1000.00
            },
            {
                'ticker': 'QQQ',
                'quantity': 75,
                'entry_price': 370.00,
                'current_price': 380.00,
                'value': 28500.00,
                'weight': 0.114,
                'sector': 'Index',
                'unrealized_pnl': 750.00
            }
        ],
        'cash': 176500.00,
        'total_value': 250000.00,
        'buying_power': 176500.00,
        'sectors': {
            'Index': 0.294,
            'Cash': 0.706
        },
        'metrics': {
            'total_return': 0.085,
            'daily_return': 0.0012,
            'sharpe_ratio': 1.45,
            'max_drawdown': 0.045
        }
    }


# ==============================================================================
# LOGGING FIXTURES
# ==============================================================================

@pytest.fixture
def test_logger():
    """Configured logger for tests"""
    logger = logging.getLogger('test')
    logger.setLevel(logging.DEBUG)
    
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    
    return logger


# ==============================================================================
# UTILITY FIXTURES
# ==============================================================================

@pytest.fixture
def temp_cache_dir(tmp_path):
    """Temporary directory for cache during tests"""
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    return cache_dir


@pytest.fixture
def mock_database():
    """Mock database connection"""
    db = MagicMock()
    
    db.save = AsyncMock(return_value=True)
    db.get = AsyncMock(return_value=None)
    db.update = AsyncMock(return_value=True)
    db.delete = AsyncMock(return_value=True)
    db.query = AsyncMock(return_value=[])
    
    return db


# ==============================================================================
# PERFORMANCE FIXTURES
# ==============================================================================

@pytest.fixture
def performance_timer():
    """Timer for performance testing"""
    class Timer:
        def __init__(self):
            self.start_time = None
            self.end_time = None
        
        def start(self):
            self.start_time = datetime.now()
        
        def stop(self):
            self.end_time = datetime.now()
            return self.elapsed()
        
        def elapsed(self):
            if self.start_time and self.end_time:
                return (self.end_time - self.start_time).total_seconds()
            return 0
    
    return Timer()


# ==============================================================================
# CLEANUP
# ==============================================================================

@pytest.fixture(autouse=True)
def cleanup_after_test():
    """Cleanup after each test"""
    yield
    # Add any cleanup code here if needed
    pass


def pytest_sessionfinish(session, exitstatus):
    """Called after whole test run finished"""
    print(f"\n{'='*60}")
    print(f"Test session finished with exit status: {exitstatus}")
    print(f"Total tests run: {session.testscollected}")
    print(f"{'='*60}\n")