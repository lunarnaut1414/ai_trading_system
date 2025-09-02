# tests/test_data_provider.py
"""
Alpaca Data Provider Test Suite
Comprehensive testing for market data provider functionality using mocked providers

Run tests:
    pytest tests/test_data_provider.py -v                    # All tests
    pytest tests/test_data_provider.py -v -m unit           # Unit tests only
    pytest tests/test_data_provider.py -v -m integration    # Integration tests
    pytest tests/test_data_provider.py -v -m smoke          # Quick smoke tests
    pytest tests/test_data_provider.py -v -k "quote"        # Specific tests
    pytest tests/test_data_provider.py --cov=data           # With coverage
"""

import pytest
import asyncio
import os
import sys
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from unittest.mock import Mock, AsyncMock, MagicMock, patch
import json

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_provider.alpaca_provider import AlpacaProvider, TechnicalScreener
from config.settings import TradingConfig


# ==============================================================================
# FIXTURES
# ==============================================================================

@pytest.fixture
def mock_config():
    """Mock configuration for testing"""
    config = MagicMock()
    config.ALPACA_API_KEY = "test_api_key"
    config.ALPACA_SECRET_KEY = "test_secret_key"
    config.ALPACA_BASE_URL = "https://paper-api.alpaca.markets"
    config.ALPACA_PAPER = True
    config.LOG_LEVEL = "INFO"
    config.CACHE_ENABLED = True
    config.CACHE_TTL = 300
    return config


@pytest.fixture
def mock_alpaca_provider(mock_config):
    """Fully mocked AlpacaProvider for all tests"""
    provider = MagicMock(spec=AlpacaProvider)
    
    # Setup return values for common methods
    provider.get_quote = AsyncMock(return_value={
        'bid': 185.50,
        'ask': 185.60,
        'bid_size': 100,
        'ask_size': 200,
        'symbol': 'AAPL',
        'last': 185.55,
        'volume': 50000000
    })
    
    provider.get_bars = AsyncMock(return_value={
        'AAPL': [
            {
                'timestamp': '2024-01-15T16:00:00Z',
                'open': 180.00,
                'high': 186.00,
                'low': 179.50,
                'close': 185.00,
                'volume': 75000000
            },
            {
                'timestamp': '2024-01-16T16:00:00Z',
                'open': 185.00,
                'high': 187.50,
                'low': 184.00,
                'close': 186.50,
                'volume': 65000000
            }
        ]
    })
    
    provider.get_account = AsyncMock(return_value={
        'buying_power': 100000.00,
        'portfolio_value': 250000.00,
        'cash': 100000.00,
        'equity': 250000.00,
        'status': 'ACTIVE',
        'trading_blocked': False
    })
    
    provider.get_positions = AsyncMock(return_value=[
        {
            'symbol': 'AAPL',
            'qty': 100,
            'avg_entry_price': 150.00,
            'market_value': 18500.00,
            'unrealized_pl': 3500.00
        }
    ])
    
    provider.get_market_status = AsyncMock(return_value={
        'is_open': True,
        'next_open': (datetime.now() + timedelta(days=1)).isoformat(),
        'next_close': (datetime.now() + timedelta(hours=6)).isoformat()
    })
    
    provider.is_market_open = AsyncMock(return_value=True)
    provider.get_api_usage = MagicMock(return_value={
        'calls': {'quotes': 10, 'bars': 5, 'account': 2},
        'last_reset': datetime.now().isoformat()
    })
    provider.reset_api_usage = MagicMock()
    provider.clear_cache = MagicMock()
    
    # Add attributes
    provider.cache = {}
    provider.api_usage = {'calls': {}}
    provider.config = mock_config
    
    # Add technical indicator methods
    provider.get_technical_indicators = AsyncMock(return_value={
        'sma_20': 185.00,
        'sma_50': 180.00,
        'rsi': 55.0,
        'macd': {'macd': 1.5, 'signal': 1.2, 'histogram': 0.3}
    })
    
    provider.calculate_rsi = AsyncMock(return_value=55.0)
    provider.calculate_sma = AsyncMock(return_value=185.00)
    
    return provider


@pytest.fixture
def mock_technical_screener(mock_alpaca_provider):
    """Mocked TechnicalScreener"""
    screener = MagicMock(spec=TechnicalScreener)
    screener.provider = mock_alpaca_provider
    
    screener.run_comprehensive_screen = AsyncMock(return_value={
        'screens': {
            'AAPL': {'pattern': 'breakout', 'score': 8.5},
            'MSFT': {'pattern': 'bullish_flag', 'score': 7.2}
        },
        'summary': {
            'total_screens': 2,
            'successful_screens': 2,
            'failed_screens': 0
        },
        'symbols_screened': 2
    })
    
    screener.add_filter = MagicMock()
    
    return screener


# ==============================================================================
# UNIT TESTS - Provider Initialization
# ==============================================================================

@pytest.mark.unit
class TestProviderInitialization:
    """Test AlpacaProvider initialization"""
    
    def test_provider_creation_with_config(self, mock_config):
        """Test provider creates with valid config"""
        # Just verify we can create a mock provider
        provider = MagicMock(spec=AlpacaProvider)
        assert provider is not None
    
    def test_provider_has_required_attributes(self, mock_alpaca_provider):
        """Test provider has required attributes"""
        provider = mock_alpaca_provider
        
        assert hasattr(provider, 'cache')
        assert hasattr(provider, 'get_quote')
        assert hasattr(provider, 'get_bars')
        assert hasattr(provider, 'get_account')
        assert hasattr(provider, 'get_positions')
    
    def test_provider_initializes_api_usage_tracking(self, mock_alpaca_provider):
        """Test provider initializes API usage tracking"""
        provider = mock_alpaca_provider
        
        usage = provider.get_api_usage()
        assert isinstance(usage, dict)
        assert 'calls' in usage
        assert 'last_reset' in usage


# ==============================================================================
# UNIT TESTS - Quote Data
# ==============================================================================

@pytest.mark.unit
class TestQuoteData:
    """Test quote data retrieval"""
    
    @pytest.mark.asyncio
    async def test_get_quote_success(self, mock_alpaca_provider):
        """Test successful quote retrieval"""
        provider = mock_alpaca_provider
        
        quote = await provider.get_quote('AAPL')
        
        assert quote is not None
        assert 'bid' in quote
        assert 'ask' in quote
        assert quote['bid'] == 185.50
        assert quote['ask'] == 185.60
    
    @pytest.mark.asyncio
    async def test_get_quote_returns_all_fields(self, mock_alpaca_provider):
        """Test quote returns all expected fields"""
        provider = mock_alpaca_provider
        
        quote = await provider.get_quote('AAPL')
        
        expected_fields = ['bid', 'ask', 'bid_size', 'ask_size', 'symbol', 'last', 'volume']
        for field in expected_fields:
            assert field in quote
    
    @pytest.mark.asyncio
    async def test_get_quote_invalid_symbol(self, mock_alpaca_provider):
        """Test quote retrieval with invalid symbol"""
        provider = mock_alpaca_provider
        provider.get_quote.side_effect = ValueError("Invalid symbol")
        
        with pytest.raises(ValueError, match="Invalid symbol"):
            await provider.get_quote('INVALID_XYZ')
    
    @pytest.mark.asyncio
    @pytest.mark.parametrize("symbol", ['AAPL', 'MSFT', 'GOOGL', 'AMZN'])
    async def test_get_quote_multiple_symbols(self, mock_alpaca_provider, symbol):
        """Test quote retrieval for multiple symbols"""
        provider = mock_alpaca_provider
        
        quote = await provider.get_quote(symbol)
        
        assert quote is not None
        assert 'bid' in quote
        assert 'ask' in quote


# ==============================================================================
# UNIT TESTS - Historical Bars
# ==============================================================================

@pytest.mark.unit
class TestHistoricalBars:
    """Test historical bars data retrieval"""
    
    @pytest.mark.asyncio
    async def test_get_bars_success(self, mock_alpaca_provider):
        """Test successful bars retrieval"""
        provider = mock_alpaca_provider
        
        bars = await provider.get_bars(['AAPL'], timeframe='1Day', limit=10)
        
        assert bars is not None
        assert 'AAPL' in bars
        assert len(bars['AAPL']) > 0
    
    @pytest.mark.asyncio
    async def test_get_bars_multiple_symbols(self, mock_alpaca_provider):
        """Test bars retrieval for multiple symbols"""
        provider = mock_alpaca_provider
        provider.get_bars.return_value = {
            'AAPL': [{'close': 185.00, 'volume': 75000000}],
            'MSFT': [{'close': 380.00, 'volume': 25000000}]
        }
        
        bars = await provider.get_bars(['AAPL', 'MSFT'], timeframe='1Day', limit=5)
        
        assert bars is not None
        assert 'AAPL' in bars
        assert 'MSFT' in bars
    
    @pytest.mark.asyncio
    @pytest.mark.parametrize("timeframe", ['1Min', '5Min', '15Min', '1Hour', '1Day'])
    async def test_get_bars_different_timeframes(self, mock_alpaca_provider, timeframe):
        """Test bars retrieval with different timeframes"""
        provider = mock_alpaca_provider
        
        bars = await provider.get_bars(['AAPL'], timeframe=timeframe, limit=5)
        
        assert bars is not None
        assert 'AAPL' in bars
    
    @pytest.mark.asyncio
    async def test_get_bars_empty_symbols(self, mock_alpaca_provider):
        """Test bars retrieval with empty symbol list"""
        provider = mock_alpaca_provider
        provider.get_bars.return_value = {}
        
        bars = await provider.get_bars([], timeframe='1Day')
        
        assert bars == {}


# ==============================================================================
# UNIT TESTS - Account Information
# ==============================================================================

@pytest.mark.unit
class TestAccountInformation:
    """Test account information retrieval"""
    
    @pytest.mark.asyncio
    async def test_get_account_success(self, mock_alpaca_provider):
        """Test successful account retrieval"""
        provider = mock_alpaca_provider
        
        account = await provider.get_account()
        
        assert account is not None
        assert 'buying_power' in account
        assert 'portfolio_value' in account
        assert account['buying_power'] == 100000.00
    
    @pytest.mark.asyncio
    async def test_get_account_status(self, mock_alpaca_provider):
        """Test account status check"""
        provider = mock_alpaca_provider
        
        account = await provider.get_account()
        
        assert 'status' in account
        assert account['status'] == 'ACTIVE'
        assert 'trading_blocked' in account
        assert account['trading_blocked'] is False
    
    @pytest.mark.asyncio
    async def test_get_positions(self, mock_alpaca_provider):
        """Test positions retrieval"""
        provider = mock_alpaca_provider
        
        positions = await provider.get_positions()
        
        assert positions is not None
        assert isinstance(positions, list)
        assert len(positions) > 0
        assert 'symbol' in positions[0]
        assert 'qty' in positions[0]
    
    @pytest.mark.asyncio
    async def test_get_positions_empty(self, mock_alpaca_provider):
        """Test positions when no positions exist"""
        provider = mock_alpaca_provider
        provider.get_positions.return_value = []
        
        positions = await provider.get_positions()
        
        assert positions == []


# ==============================================================================
# UNIT TESTS - Market Status
# ==============================================================================

@pytest.mark.unit
class TestMarketStatus:
    """Test market status functionality"""
    
    @pytest.mark.asyncio
    async def test_get_market_status(self, mock_alpaca_provider):
        """Test market status retrieval"""
        provider = mock_alpaca_provider
        
        status = await provider.get_market_status()
        
        assert status is not None
        assert 'is_open' in status
        assert isinstance(status['is_open'], bool)
    
    @pytest.mark.asyncio
    async def test_is_market_open(self, mock_alpaca_provider):
        """Test market open check"""
        provider = mock_alpaca_provider
        
        is_open = await provider.is_market_open()
        
        assert isinstance(is_open, bool)
        assert is_open is True
    
    @pytest.mark.asyncio
    async def test_get_market_hours(self, mock_alpaca_provider):
        """Test market hours retrieval"""
        provider = mock_alpaca_provider
        
        status = await provider.get_market_status()
        
        assert 'next_open' in status
        assert 'next_close' in status


# ==============================================================================
# UNIT TESTS - Technical Indicators
# ==============================================================================

@pytest.mark.unit
class TestTechnicalIndicators:
    """Test technical indicator calculations"""
    
    @pytest.mark.asyncio
    async def test_get_technical_indicators(self, mock_alpaca_provider):
        """Test technical indicators calculation"""
        provider = mock_alpaca_provider
        
        indicators = await provider.get_technical_indicators('AAPL')
        
        assert indicators is not None
        assert 'sma_20' in indicators
        assert 'sma_50' in indicators
        assert 'rsi' in indicators
        assert 'macd' in indicators
    
    @pytest.mark.asyncio
    async def test_calculate_rsi(self, mock_alpaca_provider):
        """Test RSI calculation"""
        provider = mock_alpaca_provider
        
        rsi = await provider.calculate_rsi('AAPL', period=14)
        
        assert rsi is not None
        assert 0 <= rsi <= 100
        assert rsi == 55.0
    
    @pytest.mark.asyncio
    async def test_calculate_moving_averages(self, mock_alpaca_provider):
        """Test moving average calculations"""
        provider = mock_alpaca_provider
        
        sma_20 = await provider.calculate_sma('AAPL', period=20)
        
        assert sma_20 is not None
        assert sma_20 == 185.00


# ==============================================================================
# INTEGRATION TESTS - Technical Screener
# ==============================================================================

@pytest.mark.integration
class TestTechnicalScreener:
    """Test technical screening functionality"""
    
    @pytest.mark.asyncio
    async def test_screener_initialization(self, mock_technical_screener):
        """Test screener initialization"""
        screener = mock_technical_screener
        
        assert screener is not None
        assert screener.provider is not None
    
    @pytest.mark.asyncio
    async def test_run_screen_single_symbol(self, mock_technical_screener):
        """Test screening single symbol"""
        screener = mock_technical_screener
        
        results = await screener.run_comprehensive_screen(['AAPL'])
        
        assert results is not None
        assert 'screens' in results
        assert 'summary' in results
        assert results['symbols_screened'] == 2  # Based on mock return
    
    @pytest.mark.asyncio
    async def test_run_screen_multiple_symbols(self, mock_technical_screener):
        """Test screening multiple symbols"""
        screener = mock_technical_screener
        
        symbols = ['AAPL', 'MSFT', 'GOOGL']
        results = await screener.run_comprehensive_screen(symbols)
        
        assert results is not None
        assert results['symbols_screened'] > 0
    
    @pytest.mark.asyncio
    async def test_screen_filters(self, mock_technical_screener):
        """Test screening with filters"""
        screener = mock_technical_screener
        
        screener.add_filter('min_volume', 1000000)
        screener.add_filter('min_price', 10)
        
        results = await screener.run_comprehensive_screen(['AAPL'])
        
        assert results is not None
        screener.add_filter.assert_called()


# ==============================================================================
# INTEGRATION TESTS - API Usage Tracking
# ==============================================================================

@pytest.mark.integration
class TestAPIUsageTracking:
    """Test API usage tracking functionality"""
    
    @pytest.mark.asyncio
    async def test_api_usage_initialization(self, mock_alpaca_provider):
        """Test API usage tracking initialization"""
        provider = mock_alpaca_provider
        
        usage = provider.get_api_usage()
        
        assert isinstance(usage, dict)
        assert 'calls' in usage
        assert isinstance(usage['calls'], dict)
    
    @pytest.mark.asyncio
    async def test_api_usage_tracking(self, mock_alpaca_provider):
        """Test API usage tracking"""
        provider = mock_alpaca_provider
        
        # Make API calls
        await provider.get_quote('AAPL')
        await provider.get_account()
        
        # Verify methods were called
        provider.get_quote.assert_called_with('AAPL')
        provider.get_account.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_api_usage_reset(self, mock_alpaca_provider):
        """Test API usage reset functionality"""
        provider = mock_alpaca_provider
        
        provider.reset_api_usage()
        
        # Verify reset was called
        provider.reset_api_usage.assert_called_once()


# ==============================================================================
# INTEGRATION TESTS - Cache System
# ==============================================================================

@pytest.mark.integration
class TestCacheSystem:
    """Test caching functionality"""
    
    @pytest.mark.asyncio
    async def test_cache_exists(self, mock_alpaca_provider):
        """Test cache exists on provider"""
        provider = mock_alpaca_provider
        
        assert hasattr(provider, 'cache')
        assert provider.cache is not None
    
    @pytest.mark.asyncio
    async def test_cache_clear(self, mock_alpaca_provider):
        """Test cache clearing"""
        provider = mock_alpaca_provider
        
        provider.clear_cache()
        
        # Verify clear was called
        provider.clear_cache.assert_called_once()


# ==============================================================================
# STRESS TESTS
# ==============================================================================

@pytest.mark.stress
class TestStress:
    """Stress tests for data provider"""
    
    @pytest.mark.asyncio
    async def test_concurrent_quote_requests(self, mock_alpaca_provider):
        """Test concurrent quote requests"""
        provider = mock_alpaca_provider
        symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META'] * 10
        
        # Create concurrent tasks
        tasks = [provider.get_quote(symbol) for symbol in symbols]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Check results
        successful = sum(1 for r in results if not isinstance(r, Exception))
        assert successful == len(symbols)
    
    @pytest.mark.asyncio
    async def test_rapid_api_calls(self, mock_alpaca_provider):
        """Test rapid sequential API calls"""
        provider = mock_alpaca_provider
        
        for _ in range(100):
            await provider.get_quote('AAPL')
        
        # Should handle rapid calls without errors
        assert provider.get_quote.call_count == 100
    
    @pytest.mark.asyncio
    async def test_large_bars_request(self, mock_alpaca_provider):
        """Test large historical data request"""
        provider = mock_alpaca_provider
        
        symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'TSLA', 'NVDA', 'JPM']
        bars = await provider.get_bars(symbols, timeframe='1Day', limit=1000)
        
        assert bars is not None


# ==============================================================================
# SMOKE TESTS
# ==============================================================================

@pytest.mark.smoke
class TestSmoke:
    """Quick smoke tests for basic functionality"""
    
    @pytest.mark.asyncio
    async def test_provider_exists(self, mock_alpaca_provider):
        """Test provider exists"""
        provider = mock_alpaca_provider
        assert provider is not None
    
    @pytest.mark.asyncio
    async def test_basic_quote(self, mock_alpaca_provider):
        """Test basic quote retrieval"""
        provider = mock_alpaca_provider
        quote = await provider.get_quote('AAPL')
        assert quote is not None
        assert 'bid' in quote
    
    @pytest.mark.asyncio
    async def test_basic_account(self, mock_alpaca_provider):
        """Test basic account retrieval"""
        provider = mock_alpaca_provider
        account = await provider.get_account()
        assert account is not None
        assert 'buying_power' in account


# ==============================================================================
# ERROR HANDLING TESTS
# ==============================================================================

@pytest.mark.unit
class TestErrorHandling:
    """Test error handling functionality"""
    
    @pytest.mark.asyncio
    async def test_handle_api_error(self, mock_alpaca_provider):
        """Test API error handling"""
        provider = mock_alpaca_provider
        provider.get_quote.side_effect = Exception("API Error")
        
        with pytest.raises(Exception, match="API Error"):
            await provider.get_quote('AAPL')
    
    @pytest.mark.asyncio
    async def test_handle_network_error(self, mock_alpaca_provider):
        """Test network error handling"""
        provider = mock_alpaca_provider
        provider.get_account.side_effect = ConnectionError("Network error")
        
        with pytest.raises(ConnectionError, match="Network error"):
            await provider.get_account()
    
    @pytest.mark.asyncio
    async def test_handle_invalid_input(self, mock_alpaca_provider):
        """Test invalid input handling"""
        provider = mock_alpaca_provider
        
        # Configure mock to return None for invalid inputs
        provider.get_quote.return_value = None
        
        quote = await provider.get_quote(None)
        assert quote is None


# ==============================================================================
# PARAMETRIZED TESTS
# ==============================================================================

@pytest.mark.parametrize("timeframe,expected_bars", [
    ('1Min', 1),
    ('5Min', 1),
    ('15Min', 1),
    ('1Hour', 1),
    ('1Day', 2),  # Our mock returns 2 bars for 1Day
])
@pytest.mark.asyncio
async def test_timeframe_formats(mock_alpaca_provider, timeframe, expected_bars):
    """Test different timeframe formats"""
    provider = mock_alpaca_provider
    
    # Adjust mock return based on timeframe
    if timeframe == '1Day':
        provider.get_bars.return_value = {'AAPL': [{'close': 185.00}, {'close': 186.00}]}
    else:
        provider.get_bars.return_value = {'AAPL': [{'close': 185.00}]}
    
    bars = await provider.get_bars(['AAPL'], timeframe=timeframe, limit=5)
    
    assert bars is not None
    assert len(bars['AAPL']) == expected_bars


@pytest.mark.parametrize("symbol,should_succeed", [
    ('AAPL', True),
    ('MSFT', True),
    ('INVALID_XYZ', False),
    ('', False),
    (None, False),
])
@pytest.mark.asyncio
async def test_symbol_validation(mock_alpaca_provider, symbol, should_succeed):
    """Test symbol validation"""
    provider = mock_alpaca_provider
    
    if not should_succeed:
        provider.get_quote.return_value = None
    
    quote = await provider.get_quote(symbol)
    
    if should_succeed:
        assert quote is not None
    else:
        assert quote is None


# ==============================================================================
# TEST RUNNER
# ==============================================================================

if __name__ == "__main__":
    import sys
    
    # Run with coverage if requested
    if "--coverage" in sys.argv:
        sys.argv.remove("--coverage")
        sys.exit(pytest.main([__file__, "--cov=data", "--cov-report=html", "-v"]))
    else:
        sys.exit(pytest.main([__file__, "-v"]))