# tests/test_technical_screener.py
"""
Test suite for Technical Screener
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch, MagicMock
import pandas as pd
import numpy as np

from screener.technical_screener import TechnicalScreener
from screener.pattern_recognition import (
    PatternRecognitionEngine, PatternType, SignalDirection, 
    TechnicalSignal, SupportResistanceLevel
)
from screener.screening_scheduler import ScreeningScheduler
from screener.screener_integration import ScreenerIntegration

class TestPatternRecognitionEngine:
    """Test pattern recognition engine"""
    
    @pytest.fixture
    def engine(self):
        return PatternRecognitionEngine()
    
    @pytest.fixture
    def sample_price_data(self):
        """Generate sample price data"""
        dates = pd.date_range(start='2024-01-01', periods=100, freq='D')
        
        # Generate realistic price movement
        np.random.seed(42)
        prices = 100 + np.cumsum(np.random.randn(100) * 2)
        
        data = pd.DataFrame({
            'open': prices + np.random.randn(100) * 0.5,
            'high': prices + abs(np.random.randn(100)) * 2,
            'low': prices - abs(np.random.randn(100)) * 2,
            'close': prices,
            'volume': np.random.randint(1000000, 5000000, 100)
        }, index=dates)
        
        return data
    
    def test_engine_initialization(self, engine):
        """Test pattern recognition engine initialization"""
        assert engine.min_pattern_bars == 10
        assert engine.max_pattern_bars == 60
        assert engine.confidence_threshold == 6.0
        assert engine.volume_threshold == 1.5
    
    def test_support_resistance_detection(self, engine, sample_price_data):
        """Test support and resistance level detection"""
        sr_levels = engine._detect_support_resistance(sample_price_data)
        
        assert isinstance(sr_levels, list)
        
        # Check if we have both support and resistance levels
        resistance_levels = [l for l in sr_levels if l.level_type == 'resistance']
        support_levels = [l for l in sr_levels if l.level_type == 'support']
        
        assert len(resistance_levels) <= 5  # Max 5 resistance levels
        assert len(support_levels) <= 5  # Max 5 support levels
        
        # Check level properties
        for level in sr_levels:
            assert isinstance(level.level, float)
            assert 1 <= level.strength <= 10
            assert level.touch_count >= 3  # Minimum touches
            assert level.level_type in ['support', 'resistance']
    
    def test_triangle_pattern_detection(self, engine, sample_price_data):
        """Test triangle pattern detection"""
        signals = engine._detect_triangle_patterns(
            'TEST', sample_price_data, sample_price_data[['volume']]
        )
        
        assert isinstance(signals, list)
        
        for signal in signals:
            assert signal.pattern_type in [
                PatternType.ASCENDING_TRIANGLE,
                PatternType.DESCENDING_TRIANGLE,
                PatternType.SYMMETRICAL_TRIANGLE
            ]
            assert signal.direction in [SignalDirection.BULLISH, SignalDirection.BEARISH]
            assert signal.risk_reward_ratio >= 1.5
    
    def test_breakout_pattern_detection(self, engine, sample_price_data):
        """Test breakout pattern detection"""
        # Create mock S/R levels
        sr_levels = [
            SupportResistanceLevel(
                level=105.0,
                strength=8.0,
                touch_count=5,
                last_touch=datetime.now(),
                level_type='resistance',
                volume_at_level=2000000
            )
        ]
        
        signals = engine._detect_breakout_patterns(
            'TEST', sample_price_data, sample_price_data[['volume']], sr_levels
        )
        
        assert isinstance(signals, list)
        
        for signal in signals:
            assert signal.pattern_type in [
                PatternType.RESISTANCE_BREAKOUT,
                PatternType.SUPPORT_BREAKOUT
            ]
    
    def test_indicator_calculation(self, engine, sample_price_data):
        """Test technical indicator calculations"""
        indicators = engine._calculate_indicators(
            sample_price_data, sample_price_data[['volume']]
        )
        
        assert 'rsi' in indicators
        assert 'macd' in indicators
        assert 'sma_20' in indicators
        assert 'volume_ratio' in indicators
        assert 'bollinger' in indicators
        assert 'atr' in indicators
        
        # Check RSI range
        assert 0 <= indicators['rsi'] <= 100
        
        # Check MACD components
        assert 'trend' in indicators['macd']
        assert indicators['macd']['trend'] in ['bullish', 'bearish', 'neutral']
    
    def test_signal_filtering(self, engine):
        """Test signal quality filtering"""
        # Create mix of good and bad signals
        signals = [
            TechnicalSignal(
                ticker='GOOD1',
                pattern_type=PatternType.ASCENDING_TRIANGLE,
                direction=SignalDirection.BULLISH,
                confidence=8.0,  # Good confidence
                entry_price=100,
                target_price=110,
                stop_loss_price=95,
                risk_reward_ratio=2.0,  # Good R/R
                pattern_completion=80.0,  # Good completion
                volume_confirmation=True,
                time_horizon='medium',
                current_price=100,
                avg_volume=1000000,
                market_cap=None,
                sector=None,
                rsi=None,
                macd_signal=None,
                moving_avg_position=None,
                detected_at=datetime.now(),
                pattern_start_date=datetime.now() - timedelta(days=10),
                estimated_duration=timedelta(days=5)
            ),
            TechnicalSignal(
                ticker='BAD1',
                pattern_type=PatternType.BULL_FLAG,
                direction=SignalDirection.BULLISH,
                confidence=4.0,  # Too low confidence
                entry_price=100,
                target_price=105,
                stop_loss_price=95,
                risk_reward_ratio=1.0,  # Too low R/R
                pattern_completion=50.0,  # Too low completion
                volume_confirmation=False,
                time_horizon='short',
                current_price=100,
                avg_volume=500000,
                market_cap=None,
                sector=None,
                rsi=None,
                macd_signal=None,
                moving_avg_position=None,
                detected_at=datetime.now(),
                pattern_start_date=datetime.now() - timedelta(days=5),
                estimated_duration=timedelta(days=2)
            )
        ]
        
        filtered = engine._filter_signals(signals)
        
        assert len(filtered) == 1
        assert filtered[0].ticker == 'GOOD1'
        assert filtered[0].confidence >= 6.0
        assert filtered[0].risk_reward_ratio >= 1.5

class TestTechnicalScreener:
    """Test main technical screener"""
    
    @pytest.fixture
    def mock_data_provider(self):
        provider = Mock()
        
        # Create a more comprehensive mock
        async def mock_get_bars(symbols, timeframe='1Day', limit=100):
            """Mock get_bars that returns correct format"""
            result = {}
            for symbol in symbols:
                result[symbol] = [
                    {'timestamp': datetime.now() - timedelta(days=i),
                     'open': 100 + i * 0.5,
                     'high': 102 + i * 0.5,
                     'low': 98 + i * 0.5,
                     'close': 100 + i * 0.5,
                     'volume': 1000000 + i * 10000}
                    for i in range(min(limit, 100))
                ]
            return result
        
        provider.get_bars = mock_get_bars
        return provider
    
    @pytest.fixture
    def screener(self, mock_data_provider):
        return TechnicalScreener(mock_data_provider)
    
    @pytest.mark.asyncio
    async def test_screener_initialization(self, screener):
        """Test screener initialization"""
        assert screener.min_avg_volume == 500_000
        assert screener.min_market_cap == 1_000_000_000
        assert screener.max_symbols_per_scan == 500
        assert screener.min_confidence == 6.0
        assert screener.min_risk_reward == 1.5
    
    @pytest.mark.asyncio
    async def test_universe_initialization(self, screener):
        """Test universe initialization"""
        await screener.initialize_universe()
        
        # The universe should be initialized even if filtering doesn't work
        assert len(screener.sp500_symbols) > 0
        assert len(screener.nasdaq_symbols) > 0
        assert len(screener.screening_universe) > 0
        
        # Check that screening universe contains expected symbols
        expected_symbols = {'AAPL', 'MSFT', 'GOOGL'}
        assert expected_symbols.issubset(screener.screening_universe)
    
    @pytest.mark.asyncio
    async def test_single_symbol_analysis(self, screener):
        """Test analyzing a single symbol"""
        signals = await screener._analyze_single_symbol('AAPL')
        
        assert isinstance(signals, list)
    
    @pytest.mark.asyncio
    async def test_signal_ranking(self, screener):
        """Test signal ranking logic"""
        # Create test signals with different quality scores
        signals = [
            TechnicalSignal(
                ticker='AAPL',
                pattern_type=PatternType.ASCENDING_TRIANGLE,
                direction=SignalDirection.BULLISH,
                confidence=9.0,
                entry_price=150,
                target_price=160,
                stop_loss_price=145,
                risk_reward_ratio=2.0,
                pattern_completion=90.0,
                volume_confirmation=True,
                time_horizon='medium',
                current_price=150,
                avg_volume=50000000,
                market_cap=None,
                sector=None,
                rsi=None,
                macd_signal=None,
                moving_avg_position=None,
                detected_at=datetime.now(),
                pattern_start_date=datetime.now() - timedelta(days=10),
                estimated_duration=timedelta(days=5)
            ),
            TechnicalSignal(
                ticker='MSFT',
                pattern_type=PatternType.BULL_FLAG,
                direction=SignalDirection.BULLISH,
                confidence=7.0,
                entry_price=400,
                target_price=420,
                stop_loss_price=390,
                risk_reward_ratio=2.0,
                pattern_completion=75.0,
                volume_confirmation=False,
                time_horizon='short',
                current_price=400,
                avg_volume=30000000,
                market_cap=None,
                sector=None,
                rsi=None,
                macd_signal=None,
                moving_avg_position=None,
                detected_at=datetime.now(),
                pattern_start_date=datetime.now() - timedelta(days=5),
                estimated_duration=timedelta(days=3)
            )
        ]
        
        ranked = screener._rank_signals(signals)
        
        assert len(ranked) <= len(signals)
        assert ranked[0].confidence >= ranked[-1].confidence if ranked else True
        
        # Check quality score calculation
        for signal in ranked:
            assert hasattr(signal, 'quality_score')
            assert signal.quality_score > 0
    
    @pytest.mark.asyncio
    async def test_opportunity_formatting(self, screener):
        """Test opportunity formatting for agents"""
        signal = TechnicalSignal(
            ticker='NVDA',
            pattern_type=PatternType.RESISTANCE_BREAKOUT,
            direction=SignalDirection.BULLISH,
            confidence=8.5,
            entry_price=500,
            target_price=550,
            stop_loss_price=480,
            risk_reward_ratio=2.5,
            pattern_completion=100.0,
            volume_confirmation=True,
            time_horizon='short',
            current_price=500,
            avg_volume=40000000,
            market_cap=None,
            sector=None,
            rsi=65.0,
            macd_signal='bullish',
            moving_avg_position='above_all',
            detected_at=datetime.now(),
            pattern_start_date=datetime.now() - timedelta(days=5),
            estimated_duration=timedelta(days=3)
        )
        
        signal.quality_score = 8.5
        
        opportunities = screener._format_opportunities([signal])
        
        assert len(opportunities) == 1
        opp = opportunities[0]
        
        assert opp['ticker'] == 'NVDA'
        assert opp['pattern'] == 'resistance_breakout'
        assert opp['direction'] == 'bullish'
        assert opp['confidence'] == 8.5
        assert opp['priority'] == 'HIGH'
        assert 'technical_summary' in opp
        assert opp['technical_summary']['rsi'] == 65.0
    
    @pytest.mark.asyncio
    async def test_cache_functionality(self, screener):
        """Test caching mechanism"""
        # Run scan to populate cache
        screener.screening_cache['latest_scan'] = {
            'opportunities': [{'ticker': 'TEST', 'confidence': 8.0}]
        }
        screener.screening_cache['cache_time'] = datetime.now()
        
        # Get from cache
        opportunities1 = await screener.get_top_opportunities(limit=5)
        
        # Should return cached results
        opportunities2 = await screener.get_top_opportunities(limit=5)
        
        assert len(opportunities1) == len(opportunities2)
    
    @pytest.mark.asyncio
    async def test_error_handling(self, screener):
        """Test error handling in screening"""
        # Test with invalid symbols
        signals = await screener.screen_specific_symbols(['INVALID', 'FAKE'])
        assert isinstance(signals, list)  # Should return empty list, not error
        
        # Test universe info with empty universe
        screener.screening_universe = set()
        info = screener.get_universe_info()
        assert info['total_symbols'] == 0

class TestScreeningScheduler:
    """Test screening scheduler"""
    
    @pytest.fixture
    def screener(self):
        mock_provider = Mock()
        mock_provider.get_bars = AsyncMock(return_value=[])
        return TechnicalScreener(mock_provider)
    
    @pytest.fixture
    def scheduler(self, screener):
        return ScreeningScheduler(screener)
    
    def test_scheduler_initialization(self, scheduler):
        """Test scheduler initialization"""
        assert scheduler.post_market_time == "16:30"
        assert scheduler.pre_market_time == "08:00"
        assert scheduler.is_running == False
        assert scheduler.last_scan_time is None
    
    def test_scheduler_start_stop(self, scheduler):
        """Test scheduler start/stop"""
        scheduler.start_daily_scanning()
        assert scheduler.is_running == True
        
        scheduler.stop_scanning()
        assert scheduler.is_running == False
    
    def test_callback_registration(self, scheduler):
        """Test callback registration"""
        on_complete = Mock()
        on_error = Mock()
        
        scheduler.register_callbacks(on_complete, on_error)
        
        assert scheduler.on_scan_complete == on_complete
        assert scheduler.on_scan_error == on_error

class TestScreenerIntegration:
    """Test screener integration layer"""
    
    @pytest.fixture
    def screener(self):
        mock_provider = Mock()
        mock_provider.get_bars = AsyncMock(return_value=[])
        return TechnicalScreener(mock_provider)
    
    @pytest.fixture
    def integration(self, screener):
        return ScreenerIntegration(screener)
    
    @pytest.mark.asyncio
    async def test_junior_analyst_opportunities(self, integration):
        """Test getting opportunities for Junior Analyst"""
        # Mock screener opportunities
        integration.screener.get_top_opportunities = AsyncMock(return_value=[
            {
                'ticker': 'AAPL',
                'pattern': 'ascending_triangle',
                'direction': 'bullish',
                'confidence': 8.0,
                'entry_price': 150,
                'target_price': 160,
                'stop_loss': 145,
                'risk_reward_ratio': 2.0,
                'quality_score': 8.0,
                'time_horizon': 'medium',
                'priority': 'HIGH',
                'detected_at': datetime.now(),
                'technical_summary': {'rsi': 65}
            }
        ])
        
        opportunities = await integration.get_opportunities_for_junior_analyst(limit=5)
        
        assert len(opportunities) > 0
        opp = opportunities[0]
        
        assert opp['ticker'] == 'AAPL'
        assert opp['signal_type'] == 'technical_pattern'
        assert opp['requires_fundamental_validation'] == True
        assert 'entry_point' in opp
        assert 'targets' in opp
        assert 'risk_metrics' in opp
    
    @pytest.mark.asyncio
    async def test_market_breadth_analysis(self, integration):
        """Test market breadth analysis"""
        # Mock screening statistics
        integration.screener.screening_cache = {
            'latest_scan': {
                'symbols_scanned': 100,
                'statistics': {
                    'bullish_count': 60,
                    'bearish_count': 30,
                    'neutral_count': 10,
                    'patterns_by_type': {'ascending_triangle': 10},
                    'avg_confidence': 7.5,
                    'avg_risk_reward': 2.0
                }
            }
        }
        
        integration.screener.get_screening_statistics = Mock(return_value={
            'universe_size': 500
        })
        
        breadth = await integration.get_market_breadth_analysis()
        
        assert 'market_sentiment' in breadth
        assert 'breadth_metrics' in breadth
        assert breadth['breadth_metrics']['bullish_signals'] == 60
        assert breadth['market_sentiment'] in [
            'strongly_bullish', 'bullish', 'neutral', 'bearish', 'strongly_bearish'
        ]
    
    @pytest.mark.asyncio
    async def test_portfolio_screening(self, integration):
        """Test screening portfolio holdings"""
        holdings = ['AAPL', 'MSFT', 'GOOGL']
        
        # Mock screener response
        mock_signal = TechnicalSignal(
            ticker='AAPL',
            pattern_type=PatternType.BULL_FLAG,
            direction=SignalDirection.BULLISH,
            confidence=8.0,
            entry_price=150,
            target_price=160,
            stop_loss_price=145,
            risk_reward_ratio=2.0,
            pattern_completion=85.0,
            volume_confirmation=True,
            time_horizon='short',
            current_price=150,
            avg_volume=50000000,
            market_cap=None,
            sector=None,
            rsi=None,
            macd_signal=None,
            moving_avg_position=None,
            detected_at=datetime.now(),
            pattern_start_date=datetime.now() - timedelta(days=5),
            estimated_duration=timedelta(days=3)
        )
        
        integration.screener.screen_specific_symbols = AsyncMock(
            return_value=[mock_signal]
        )
        
        results = await integration.screen_portfolio_holdings(holdings)
        
        assert 'holdings_analyzed' in results
        assert 'signals_by_ticker' in results
        assert results['holdings_analyzed'] == 3
        
        if 'AAPL' in results['signals_by_ticker']:
            aapl_signals = results['signals_by_ticker']['AAPL']
            assert len(aapl_signals) > 0
            assert aapl_signals[0]['pattern'] == 'bull_flag'

# Integration test
@pytest.mark.asyncio
async def test_complete_screening_workflow():
    """Test complete screening workflow"""
    
    # Mock data provider with correct format
    mock_provider = Mock()
    
    async def mock_get_bars(symbols, timeframe='1Day', limit=100):
        """Mock get_bars that returns correct format"""
        result = {}
        for symbol in symbols:
            result[symbol] = [
                {'timestamp': datetime.now() - timedelta(days=i),
                 'open': 100 + i * 0.5,
                 'high': 102 + i * 0.5,
                 'low': 98 + i * 0.5,
                 'close': 100 + i * 0.5,
                 'volume': 1000000 + i * 10000}
                for i in range(limit)
            ]
        return result
    
    mock_provider.get_bars = mock_get_bars
    
    # Create screener
    screener = TechnicalScreener(mock_provider)
    
    # Initialize universe
    await screener.initialize_universe()
    assert len(screener.screening_universe) > 0
    
    # Run scan (with mocked data)
    screener.screening_universe = {'AAPL', 'MSFT', 'GOOGL'}  # Limit for test
    results = await screener.run_daily_scan()
    
    assert 'scan_date' in results
    assert 'symbols_scanned' in results
    assert results['symbols_scanned'] <= 3
    
    # Test integration layer
    integration = ScreenerIntegration(screener)
    
    # Get opportunities for analyst
    opportunities = await integration.get_opportunities_for_junior_analyst()
    assert isinstance(opportunities, list)
    
    # Get market breadth
    breadth = await integration.get_market_breadth_analysis()
    assert 'market_sentiment' in breadth
    
    print("✅ Complete screening workflow test passed")

if __name__ == "__main__":
    asyncio.run(test_complete_screening_workflow())