# screener/pattern_recognition.py
"""
Advanced Pattern Recognition Engine for Technical Screening
"""

from dataclasses import dataclass
from enum import Enum
from typing import List, Dict, Optional, Tuple
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import logging

class PatternType(Enum):
    """Supported technical pattern types"""
    # Triangular patterns
    ASCENDING_TRIANGLE = "ascending_triangle"
    DESCENDING_TRIANGLE = "descending_triangle"
    SYMMETRICAL_TRIANGLE = "symmetrical_triangle"
    
    # Flag and pennant patterns
    BULL_FLAG = "bull_flag"
    BEAR_FLAG = "bear_flag"
    PENNANT = "pennant"
    
    # Breakout patterns
    RESISTANCE_BREAKOUT = "resistance_breakout"
    SUPPORT_BREAKOUT = "support_breakout"
    VOLUME_BREAKOUT = "volume_breakout"
    
    # Reversal patterns
    DOUBLE_TOP = "double_top"
    DOUBLE_BOTTOM = "double_bottom"
    HEAD_SHOULDERS = "head_shoulders"
    INVERSE_HEAD_SHOULDERS = "inverse_head_shoulders"
    
    # Channel patterns
    UPTREND_CHANNEL = "uptrend_channel"
    DOWNTREND_CHANNEL = "downtrend_channel"
    SIDEWAYS_CHANNEL = "sideways_channel"
    
    # Cup and handle
    CUP_AND_HANDLE = "cup_and_handle"
    
    # Wedge patterns
    RISING_WEDGE = "rising_wedge"
    FALLING_WEDGE = "falling_wedge"

class SignalDirection(Enum):
    """Signal direction"""
    BULLISH = "bullish"
    BEARISH = "bearish"
    NEUTRAL = "neutral"

@dataclass
class SupportResistanceLevel:
    """Support or resistance level detection"""
    level: float
    strength: float          # 1-10 strength score
    touch_count: int         # Number of times price touched level
    last_touch: datetime     # Last time level was touched
    level_type: str          # "support" or "resistance"
    volume_at_level: float   # Average volume when touching level

@dataclass
class TechnicalSignal:
    """Individual technical signal from pattern recognition"""
    ticker: str
    pattern_type: PatternType
    direction: SignalDirection
    confidence: float        # 0-10 confidence score
    entry_price: float
    target_price: float
    stop_loss_price: float
    risk_reward_ratio: float
    pattern_completion: float  # % of pattern completed
    volume_confirmation: bool
    time_horizon: str        # "short", "medium", "long"
    current_price: float
    avg_volume: float
    market_cap: Optional[float]
    sector: Optional[str]
    rsi: Optional[float]
    macd_signal: Optional[str]
    moving_avg_position: Optional[str]
    detected_at: datetime
    pattern_start_date: datetime
    estimated_duration: timedelta

class PatternRecognitionEngine:
    """
    Core pattern recognition engine for technical analysis
    """
    
    def __init__(self):
        self.logger = logging.getLogger('pattern_recognition')
        
        # Pattern detection parameters
        self.min_pattern_bars = 10
        self.max_pattern_bars = 60
        self.volume_threshold = 1.5  # Volume must be 1.5x average
        self.confidence_threshold = 6.0
        
        # Support/Resistance parameters
        self.sr_lookback = 100
        self.sr_touch_threshold = 0.02  # 2% threshold for touching level
        self.min_touches = 3
        
    def analyze_ticker(self, ticker: str, price_data: pd.DataFrame, 
                       volume_data: pd.DataFrame) -> Dict:
        """
        Comprehensive technical analysis for a single ticker
        
        Args:
            ticker: Stock symbol
            price_data: DataFrame with OHLC data
            volume_data: DataFrame with volume data
            
        Returns:
            Dict with detected patterns and signals
        """
        
        try:
            # Detect support/resistance levels
            sr_levels = self._detect_support_resistance(price_data)
            
            # Detect various pattern types
            signals = []
            
            # Triangle patterns
            triangle_signals = self._detect_triangle_patterns(ticker, price_data, volume_data)
            signals.extend(triangle_signals)
            
            # Flag patterns
            flag_signals = self._detect_flag_patterns(ticker, price_data, volume_data)
            signals.extend(flag_signals)
            
            # Breakout patterns
            breakout_signals = self._detect_breakout_patterns(ticker, price_data, volume_data, sr_levels)
            signals.extend(breakout_signals)
            
            # Reversal patterns
            reversal_signals = self._detect_reversal_patterns(ticker, price_data)
            signals.extend(reversal_signals)
            
            # Channel patterns
            channel_signals = self._detect_channel_patterns(ticker, price_data)
            signals.extend(channel_signals)
            
            # Cup and handle
            cup_handle_signals = self._detect_cup_handle(ticker, price_data, volume_data)
            signals.extend(cup_handle_signals)
            
            # Wedge patterns
            wedge_signals = self._detect_wedge_patterns(ticker, price_data)
            signals.extend(wedge_signals)
            
            # Calculate technical indicators
            indicators = self._calculate_indicators(price_data, volume_data)
            
            # Filter and rank signals
            filtered_signals = self._filter_signals(signals)
            
            return {
                'ticker': ticker,
                'signals': filtered_signals,
                'support_resistance': sr_levels,
                'indicators': indicators,
                'analysis_timestamp': datetime.now()
            }
            
        except Exception as e:
            self.logger.error(f"Pattern analysis failed for {ticker}: {str(e)}")
            return {
                'ticker': ticker,
                'signals': [],
                'error': str(e)
            }
    
    def _detect_support_resistance(self, price_data: pd.DataFrame) -> List[SupportResistanceLevel]:
        """Detect support and resistance levels"""
        
        levels = []
        
        try:
            # Use recent highs and lows
            recent_data = price_data.tail(self.sr_lookback)
            
            # Find local maxima (resistance)
            highs = recent_data['high'].values
            for i in range(2, len(highs) - 2):
                if (highs[i] > highs[i-1] and highs[i] > highs[i-2] and
                    highs[i] > highs[i+1] and highs[i] > highs[i+2]):
                    
                    level = highs[i]
                    touches = self._count_level_touches(price_data, level, 'resistance')
                    
                    if touches >= self.min_touches:
                        levels.append(SupportResistanceLevel(
                            level=level,
                            strength=min(10, touches * 2),
                            touch_count=touches,
                            last_touch=recent_data.index[i],
                            level_type='resistance',
                            volume_at_level=recent_data['volume'].iloc[i]
                        ))
            
            # Find local minima (support)
            lows = recent_data['low'].values
            for i in range(2, len(lows) - 2):
                if (lows[i] < lows[i-1] and lows[i] < lows[i-2] and
                    lows[i] < lows[i+1] and lows[i] < lows[i+2]):
                    
                    level = lows[i]
                    touches = self._count_level_touches(price_data, level, 'support')
                    
                    if touches >= self.min_touches:
                        levels.append(SupportResistanceLevel(
                            level=level,
                            strength=min(10, touches * 2),
                            touch_count=touches,
                            last_touch=recent_data.index[i],
                            level_type='support',
                            volume_at_level=recent_data['volume'].iloc[i]
                        ))
            
            # Sort by strength
            levels.sort(key=lambda x: x.strength, reverse=True)
            
            # Keep top 5 of each type
            resistance_levels = [l for l in levels if l.level_type == 'resistance'][:5]
            support_levels = [l for l in levels if l.level_type == 'support'][:5]
            
            return resistance_levels + support_levels
            
        except Exception as e:
            self.logger.error(f"Support/resistance detection failed: {str(e)}")
            return []
    
    def _count_level_touches(self, price_data: pd.DataFrame, level: float, 
                            level_type: str) -> int:
        """Count how many times price touched a level"""
        
        touches = 0
        threshold = level * self.sr_touch_threshold
        
        for _, row in price_data.iterrows():
            if level_type == 'resistance':
                if abs(row['high'] - level) <= threshold:
                    touches += 1
            else:  # support
                if abs(row['low'] - level) <= threshold:
                    touches += 1
        
        return touches
    
    def _detect_triangle_patterns(self, ticker: str, price_data: pd.DataFrame, 
                                 volume_data: pd.DataFrame) -> List[TechnicalSignal]:
        """Detect triangle patterns"""
        
        signals = []
        
        try:
            # Look for triangle patterns in recent data
            recent_data = price_data.tail(40)  # Last 40 periods
            
            if len(recent_data) < 20:
                return signals
            
            highs = recent_data['high'].values
            lows = recent_data['low'].values
            
            # Ascending triangle: horizontal resistance + rising support
            resistance_level = max(highs[-10:])  # Recent high as resistance
            if self._is_horizontal_line(highs[-15:], resistance_level):
                if self._is_rising_trendline(lows[-15:]):
                    signal = self._create_triangle_signal(
                        ticker, recent_data, PatternType.ASCENDING_TRIANGLE,
                        SignalDirection.BULLISH, resistance_level
                    )
                    if signal:
                        signals.append(signal)
            
            # Descending triangle: horizontal support + falling resistance
            support_level = min(lows[-10:])  # Recent low as support
            if self._is_horizontal_line(lows[-15:], support_level):
                if self._is_falling_trendline(highs[-15:]):
                    signal = self._create_triangle_signal(
                        ticker, recent_data, PatternType.DESCENDING_TRIANGLE,
                        SignalDirection.BEARISH, support_level
                    )
                    if signal:
                        signals.append(signal)
            
            # Symmetrical triangle: converging trendlines
            if (self._is_falling_trendline(highs[-15:]) and 
                self._is_rising_trendline(lows[-15:])):
                
                # Determine direction based on prior trend
                prior_trend = self._determine_prior_trend(price_data.tail(60))
                direction = SignalDirection.BULLISH if prior_trend == 'up' else SignalDirection.BEARISH
                
                signal = self._create_triangle_signal(
                    ticker, recent_data, PatternType.SYMMETRICAL_TRIANGLE,
                    direction, (resistance_level + support_level) / 2
                )
                if signal:
                    signals.append(signal)
            
            return signals
            
        except Exception as e:
            self.logger.error(f"Triangle pattern detection failed for {ticker}: {str(e)}")
            return []
    
    def _is_horizontal_line(self, values: np.ndarray, level: float, 
                           tolerance: float = 0.02) -> bool:
        """Check if values form a horizontal line"""
        return np.std(values) / level < tolerance
    
    def _is_rising_trendline(self, values: np.ndarray) -> bool:
        """Check if values form a rising trendline"""
        x = np.arange(len(values))
        slope = np.polyfit(x, values, 1)[0]
        return slope > 0
    
    def _is_falling_trendline(self, values: np.ndarray) -> bool:
        """Check if values form a falling trendline"""
        x = np.arange(len(values))
        slope = np.polyfit(x, values, 1)[0]
        return slope < 0
    
    def _determine_prior_trend(self, price_data: pd.DataFrame) -> str:
        """Determine the prior trend direction"""
        if len(price_data) < 20:
            return 'neutral'
        
        sma20 = price_data['close'].rolling(window=20).mean()
        current_price = price_data['close'].iloc[-1]
        
        if current_price > sma20.iloc[-1]:
            return 'up'
        else:
            return 'down'
    
    def _create_triangle_signal(self, ticker: str, price_data: pd.DataFrame,
                               pattern_type: PatternType, direction: SignalDirection,
                               breakout_level: float) -> Optional[TechnicalSignal]:
        """Create a triangle pattern signal"""
        
        current_price = price_data['close'].iloc[-1]
        
        # Calculate target and stop loss
        pattern_height = price_data['high'].max() - price_data['low'].min()
        
        if direction == SignalDirection.BULLISH:
            target_price = breakout_level + pattern_height
            stop_loss = price_data['low'].min()
        else:
            target_price = breakout_level - pattern_height
            stop_loss = price_data['high'].max()
        
        risk_reward = abs(target_price - current_price) / abs(stop_loss - current_price) if abs(stop_loss - current_price) > 0 else 0
        
        if risk_reward < 1.5:  # Minimum risk/reward ratio
            return None
        
        return TechnicalSignal(
            ticker=ticker,
            pattern_type=pattern_type,
            direction=direction,
            confidence=7.5,
            entry_price=current_price,
            target_price=target_price,
            stop_loss_price=stop_loss,
            risk_reward_ratio=risk_reward,
            pattern_completion=85.0,
            volume_confirmation=False,  # Will be checked separately
            time_horizon='medium',
            current_price=current_price,
            avg_volume=price_data['volume'].mean(),
            market_cap=None,
            sector=None,
            rsi=None,
            macd_signal=None,
            moving_avg_position=None,
            detected_at=datetime.now(),
            pattern_start_date=price_data.index[0],
            estimated_duration=timedelta(days=10)
        )
    
    def _detect_flag_patterns(self, ticker: str, price_data: pd.DataFrame, 
                             volume_data: pd.DataFrame) -> List[TechnicalSignal]:
        """Detect flag and pennant patterns"""
        
        signals = []
        
        try:
            # Look for flag patterns after strong moves
            recent_data = price_data.tail(30)
            
            if len(recent_data) < 15:
                return signals
            
            # Check for strong prior move (flagpole)
            flagpole_start = -20
            flagpole_end = -10
            
            if len(recent_data) < abs(flagpole_start):
                return signals
            
            flagpole_change = (recent_data.iloc[flagpole_end]['close'] - 
                              recent_data.iloc[flagpole_start]['close']) / recent_data.iloc[flagpole_start]['close']
            
            # Bull flag: strong up move + consolidation
            if flagpole_change > 0.08:  # 8%+ move up
                flag_data = recent_data.tail(10)
                if self._is_consolidation_pattern(flag_data):
                    signal = self._create_flag_signal(
                        ticker, recent_data, PatternType.BULL_FLAG,
                        SignalDirection.BULLISH
                    )
                    if signal:
                        signals.append(signal)
            
            # Bear flag: strong down move + consolidation
            elif flagpole_change < -0.08:  # 8%+ move down
                flag_data = recent_data.tail(10)
                if self._is_consolidation_pattern(flag_data):
                    signal = self._create_flag_signal(
                        ticker, recent_data, PatternType.BEAR_FLAG,
                        SignalDirection.BEARISH
                    )
                    if signal:
                        signals.append(signal)
            
            return signals
            
        except Exception as e:
            self.logger.error(f"Flag pattern detection failed for {ticker}: {str(e)}")
            return []
    
    def _is_consolidation_pattern(self, price_data: pd.DataFrame) -> bool:
        """Check if price data shows consolidation"""
        price_range = price_data['high'].max() - price_data['low'].min()
        avg_price = price_data['close'].mean()
        
        # Consolidation if range is less than 5% of average price
        return (price_range / avg_price) < 0.05
    
    def _create_flag_signal(self, ticker: str, price_data: pd.DataFrame,
                           pattern_type: PatternType, direction: SignalDirection) -> Optional[TechnicalSignal]:
        """Create a flag pattern signal"""
        
        current_price = price_data['close'].iloc[-1]
        pattern_height = abs(price_data['high'].max() - price_data['low'].min())
        
        if direction == SignalDirection.BULLISH:
            target_price = current_price + pattern_height
            stop_loss = price_data['low'].min()
        else:
            target_price = current_price - pattern_height
            stop_loss = price_data['high'].max()
        
        risk_reward = abs(target_price - current_price) / abs(stop_loss - current_price) if abs(stop_loss - current_price) > 0 else 0
        
        if risk_reward < 1.5:
            return None
        
        return TechnicalSignal(
            ticker=ticker,
            pattern_type=pattern_type,
            direction=direction,
            confidence=7.0,
            entry_price=current_price,
            target_price=target_price,
            stop_loss_price=stop_loss,
            risk_reward_ratio=risk_reward,
            pattern_completion=90.0,
            volume_confirmation=False,
            time_horizon='short',
            current_price=current_price,
            avg_volume=price_data['volume'].mean(),
            market_cap=None,
            sector=None,
            rsi=None,
            macd_signal=None,
            moving_avg_position=None,
            detected_at=datetime.now(),
            pattern_start_date=price_data.index[0],
            estimated_duration=timedelta(days=5)
        )
    
    def _detect_breakout_patterns(self, ticker: str, price_data: pd.DataFrame, 
                                 volume_data: pd.DataFrame, 
                                 sr_levels: List[SupportResistanceLevel]) -> List[TechnicalSignal]:
        """Detect breakout patterns from support/resistance levels"""
        
        signals = []
        
        try:
            current_price = price_data['close'].iloc[-1]
            recent_volume = volume_data['volume'].tail(5).mean()
            avg_volume = volume_data['volume'].tail(20).mean()
            
            # Check for resistance breakouts
            for level in sr_levels:
                if level.level_type == 'resistance' and current_price > level.level * 1.01:
                    # Volume confirmation
                    volume_confirmed = recent_volume > avg_volume * self.volume_threshold
                    
                    signal = TechnicalSignal(
                        ticker=ticker,
                        pattern_type=PatternType.RESISTANCE_BREAKOUT,
                        direction=SignalDirection.BULLISH,
                        confidence=min(8.0, level.strength + (2 if volume_confirmed else 0)),
                        entry_price=current_price,
                        target_price=level.level * 1.05,  # 5% above resistance
                        stop_loss_price=level.level * 0.98,  # 2% below resistance
                        risk_reward_ratio=2.5,
                        pattern_completion=100.0,
                        volume_confirmation=volume_confirmed,
                        time_horizon='short',
                        current_price=current_price,
                        avg_volume=avg_volume,
                        market_cap=None,
                        sector=None,
                        rsi=None,
                        macd_signal=None,
                        moving_avg_position=None,
                        detected_at=datetime.now(),
                        pattern_start_date=level.last_touch,
                        estimated_duration=timedelta(days=5)
                    )
                    signals.append(signal)
            
            return signals
            
        except Exception as e:
            self.logger.error(f"Breakout pattern detection failed for {ticker}: {str(e)}")
            return []
    
    def _detect_reversal_patterns(self, ticker: str, price_data: pd.DataFrame) -> List[TechnicalSignal]:
        """Detect reversal patterns like double tops/bottoms"""
        # Implementation would continue here
        return []
    
    def _detect_channel_patterns(self, ticker: str, price_data: pd.DataFrame) -> List[TechnicalSignal]:
        """Detect channel patterns"""
        # Implementation would continue here
        return []
    
    def _detect_cup_handle(self, ticker: str, price_data: pd.DataFrame, 
                          volume_data: pd.DataFrame) -> List[TechnicalSignal]:
        """Detect cup and handle patterns"""
        # Implementation would continue here
        return []
    
    def _detect_wedge_patterns(self, ticker: str, price_data: pd.DataFrame) -> List[TechnicalSignal]:
        """Detect wedge patterns"""
        # Implementation would continue here
        return []
    
    def _calculate_indicators(self, price_data: pd.DataFrame, 
                            volume_data: pd.DataFrame) -> Dict:
        """Calculate technical indicators"""
        
        indicators = {}
        
        try:
            # RSI
            indicators['rsi'] = self._calculate_rsi(price_data['close'])
            
            # MACD
            indicators['macd'] = self._calculate_macd(price_data['close'])
            
            # Moving averages
            indicators['sma_20'] = price_data['close'].rolling(window=20).mean().iloc[-1]
            indicators['sma_50'] = price_data['close'].rolling(window=50).mean().iloc[-1]
            indicators['sma_200'] = price_data['close'].rolling(window=200).mean().iloc[-1]
            
            # Volume indicators
            indicators['volume_ratio'] = volume_data['volume'].tail(5).mean() / volume_data['volume'].tail(20).mean()
            
            # Bollinger Bands
            indicators['bollinger'] = self._calculate_bollinger(price_data['close'])
            
            # ATR (Average True Range)
            indicators['atr'] = self._calculate_atr(price_data)
            
            return indicators
            
        except Exception as e:
            self.logger.error(f"Indicator calculation failed: {str(e)}")
            return {}
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> float:
        """Calculate RSI"""
        try:
            delta = prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            
            return rsi.iloc[-1]
        except:
            return 50.0
    
    def _calculate_macd(self, prices: pd.Series) -> Dict:
        """Calculate MACD"""
        try:
            exp1 = prices.ewm(span=12, adjust=False).mean()
            exp2 = prices.ewm(span=26, adjust=False).mean()
            macd = exp1 - exp2
            signal = macd.ewm(span=9, adjust=False).mean()
            
            return {
                'macd': macd.iloc[-1],
                'signal': signal.iloc[-1],
                'histogram': macd.iloc[-1] - signal.iloc[-1],
                'trend': 'bullish' if macd.iloc[-1] > signal.iloc[-1] else 'bearish'
            }
        except:
            return {'macd': 0, 'signal': 0, 'histogram': 0, 'trend': 'neutral'}
    
    def _calculate_bollinger(self, prices: pd.Series, period: int = 20) -> Dict:
        """Calculate Bollinger Bands"""
        try:
            sma = prices.rolling(window=period).mean()
            std = prices.rolling(window=period).std()
            
            upper = sma + (std * 2)
            lower = sma - (std * 2)
            
            current_price = prices.iloc[-1]
            
            return {
                'upper': upper.iloc[-1],
                'middle': sma.iloc[-1],
                'lower': lower.iloc[-1],
                'position': (current_price - lower.iloc[-1]) / (upper.iloc[-1] - lower.iloc[-1])
            }
        except:
            return {}
    
    def _calculate_atr(self, price_data: pd.DataFrame, period: int = 14) -> float:
        """Calculate Average True Range"""
        try:
            high = price_data['high']
            low = price_data['low']
            close = price_data['close']
            
            tr1 = high - low
            tr2 = abs(high - close.shift())
            tr3 = abs(low - close.shift())
            
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            atr = tr.rolling(window=period).mean()
            
            return atr.iloc[-1]
        except:
            return 0.0
    
    def _filter_signals(self, signals: List[TechnicalSignal]) -> List[TechnicalSignal]:
        """Filter signals by quality criteria"""
        
        filtered_signals = []
        
        for signal in signals:
            # Quality filters
            if (signal.confidence >= self.confidence_threshold and
                signal.risk_reward_ratio >= 1.5 and
                signal.pattern_completion >= 60.0):
                
                filtered_signals.append(signal)
        
        # Sort by confidence
        filtered_signals.sort(key=lambda x: x.confidence, reverse=True)
        
        return filtered_signals