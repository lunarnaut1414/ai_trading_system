# screener/technical_screener.py
"""
Technical Screener for Pattern Recognition and Opportunity Identification
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Set
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd
import numpy as np

from .pattern_recognition import PatternRecognitionEngine, TechnicalSignal
from utils.performance_tracker import PerformanceTracker

class TechnicalScreener:
    """
    Main technical screener for pattern recognition and opportunity identification
    """
    
    def __init__(self, alpaca_provider, config=None):
        """
        Initialize Technical Screener
        
        Args:
            alpaca_provider: AlpacaDataProvider instance from data.alpaca_provider
            config: Optional TradingConfig instance
        """
        self.alpaca = alpaca_provider
        self.config = config
        self.pattern_engine = PatternRecognitionEngine()
        self.performance_tracker = PerformanceTracker("technical_screener")
        
        # Initialize database if available
        self.db_manager = None
        try:
            from database.database_config import DatabaseManager
            if config and hasattr(config, 'DATABASE_URL'):
                self.db_manager = DatabaseManager(config.DATABASE_URL)
        except ImportError:
            pass  # Database is optional
        
        # Universe management
        self.sp500_symbols = set()
        self.nasdaq_symbols = set()
        self.screening_universe = set()
        
        # Screening parameters
        self.max_concurrent_requests = 10
        self.min_avg_volume = 500_000  # Minimum average volume
        self.min_market_cap = 1_000_000_000  # $1B minimum market cap
        self.max_symbols_per_scan = 500  # Limit per scan to manage resources
        
        # Results caching
        self.screening_cache = {}
        self.cache_duration = timedelta(hours=4)  # Cache results for 4 hours
        
        # Quality thresholds
        self.min_confidence = 6.0
        self.min_risk_reward = 1.5
        self.max_signals_per_ticker = 3
        
        self.logger = logging.getLogger('technical_screener')
        self.logger.info("Technical Screener initialized")
    
    async def initialize_universe(self):
        """Initialize the screening universe with S&P 500 and NASDAQ symbols"""
        
        try:
            # Get S&P 500 symbols
            self.sp500_symbols = await self._get_sp500_symbols()
            self.logger.info(f"Loaded {len(self.sp500_symbols)} S&P 500 symbols")
            
            # Get NASDAQ symbols
            self.nasdaq_symbols = await self._get_nasdaq_symbols()
            self.logger.info(f"Loaded {len(self.nasdaq_symbols)} NASDAQ symbols")
            
            # Combine universes
            combined_universe = self.sp500_symbols.union(self.nasdaq_symbols)
            
            # Try to apply liquidity filters, but if it fails or returns empty, use unfiltered
            try:
                if hasattr(self.alpaca, 'get_bars'):
                    filtered = await self._filter_by_liquidity(combined_universe)
                    if filtered:  # Only use filtered if it's not empty
                        self.screening_universe = filtered
                    else:
                        self.screening_universe = combined_universe
                else:
                    self.screening_universe = combined_universe
            except Exception:
                # If filtering fails, use unfiltered universe
                self.screening_universe = combined_universe
            
            self.logger.info(f"Final screening universe: {len(self.screening_universe)} symbols")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize universe: {str(e)}")
            # Use a default list if initialization fails
            self.screening_universe = self._get_default_symbols()
    
    async def _get_sp500_symbols(self) -> Set[str]:
        """Get S&P 500 constituent symbols"""
        
        # In production, this would fetch from a data provider
        # For now, using a representative sample
        sp500_sample = {
            'AAPL', 'MSFT', 'AMZN', 'NVDA', 'GOOGL', 'META', 'TSLA', 'BRK.B',
            'JPM', 'JNJ', 'V', 'PG', 'UNH', 'HD', 'MA', 'DIS', 'BAC', 'ADBE',
            'CRM', 'NFLX', 'KO', 'PEP', 'TMO', 'CSCO', 'ABT', 'CVX', 'NKE',
            'WMT', 'MRK', 'LLY', 'ABBV', 'AVGO', 'ACN', 'COST', 'TXN', 'MCD',
            'NEE', 'DHR', 'VZ', 'T', 'LOW', 'ORCL', 'PM', 'UPS', 'MS', 'RTX',
            'INTC', 'AMD', 'GS', 'HON', 'IBM', 'CAT', 'AMGN', 'QCOM', 'GE'
        }
        
        return sp500_sample
    
    async def _get_nasdaq_symbols(self) -> Set[str]:
        """Get NASDAQ-100 constituent symbols"""
        
        # In production, this would fetch from a data provider
        nasdaq_sample = {
            'AAPL', 'MSFT', 'AMZN', 'NVDA', 'GOOGL', 'META', 'TSLA', 'GOOG',
            'AVGO', 'PEP', 'COST', 'ADBE', 'CSCO', 'CMCSA', 'NFLX', 'TMUS',
            'TXN', 'INTC', 'AMD', 'QCOM', 'INTU', 'AMGN', 'AMAT', 'ISRG',
            'BKNG', 'ADP', 'MDLZ', 'GILD', 'REGN', 'VRTX', 'ADI', 'LRCX',
            'PANW', 'ASML', 'MU', 'SNPS', 'CDNS', 'KLAC', 'MAR', 'MRVL',
            'ORLY', 'CTAS', 'FTNT', 'ADSK', 'MELI', 'WDAY', 'ROST', 'ODFL'
        }
        
        return nasdaq_sample
    
    async def _filter_by_liquidity(self, symbols: Set[str]) -> Set[str]:
        """Filter symbols by liquidity requirements"""
        
        filtered_symbols = set()
        
        for symbol in symbols:
            try:
                # Get recent volume data
                bars = await self.data_provider.get_bars(
                    symbol=symbol,
                    timeframe='1Day',
                    limit=20
                )
                
                if bars and len(bars) > 0:
                    avg_volume = sum(bar['v'] for bar in bars) / len(bars)
                    
                    if avg_volume >= self.min_avg_volume:
                        filtered_symbols.add(symbol)
                        
            except Exception as e:
                self.logger.debug(f"Failed to get volume for {symbol}: {str(e)}")
                continue
        
        return filtered_symbols
    
    def _get_default_symbols(self) -> Set[str]:
        """Get default high-liquidity symbols as fallback"""
        
        return {
            'AAPL', 'MSFT', 'AMZN', 'NVDA', 'GOOGL', 'META', 'TSLA',
            'JPM', 'V', 'UNH', 'MA', 'HD', 'PG', 'JNJ', 'BAC',
            'DIS', 'ADBE', 'NFLX', 'CRM', 'WMT', 'KO', 'PEP'
        }
    
    async def run_daily_scan(self) -> Dict:
        """
        Run daily technical screening scan
        
        Returns:
            Dict with screening results and statistics
        """
        
        start_time = datetime.now()
        self.logger.info("Starting daily technical scan")
        
        try:
            # Initialize universe if not already done
            if not self.screening_universe:
                await self.initialize_universe()
            
            # Prepare symbols for scanning
            symbols_to_scan = list(self.screening_universe)[:self.max_symbols_per_scan]
            
            # Run pattern detection in parallel
            all_signals = await self._parallel_pattern_detection(symbols_to_scan)
            
            # Rank and filter signals
            top_signals = self._rank_signals(all_signals)
            
            # Generate opportunities for agents
            opportunities = self._format_opportunities(top_signals)
            
            # Save to database
            await self._save_screening_results(opportunities)
            
            # Calculate statistics
            scan_duration = (datetime.now() - start_time).total_seconds()
            statistics = self._calculate_scan_statistics(all_signals, top_signals)
            
            # Update performance metrics
            self.performance_tracker.record_success(scan_duration)
            
            result = {
                'scan_date': datetime.now(),
                'symbols_scanned': len(symbols_to_scan),
                'total_signals': len(all_signals),
                'top_signals': top_signals[:20],  # Top 20 opportunities
                'opportunities': opportunities[:10],  # Top 10 for agents
                'statistics': statistics,
                'scan_duration': scan_duration
            }
            
            # Cache results
            self.screening_cache['latest_scan'] = result
            self.screening_cache['cache_time'] = datetime.now()
            
            self.logger.info(f"Daily scan completed: {len(top_signals)} quality signals found")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Daily scan failed: {str(e)}")
            self.performance_tracker.record_failure(str(e))
            return {
                'error': str(e),
                'scan_date': datetime.now()
            }
    
    async def _parallel_pattern_detection(self, symbols: List[str]) -> List[TechnicalSignal]:
        """Run pattern detection in parallel for multiple symbols"""
        
        all_signals = []
        
        with ThreadPoolExecutor(max_workers=self.max_concurrent_requests) as executor:
            # Submit all tasks
            future_to_symbol = {}
            
            for symbol in symbols:
                future = executor.submit(
                    asyncio.run,
                    self._analyze_single_symbol(symbol)
                )
                future_to_symbol[future] = symbol
            
            # Collect results
            for future in as_completed(future_to_symbol):
                symbol = future_to_symbol[future]
                
                try:
                    signals = future.result()
                    if signals:
                        all_signals.extend(signals)
                        
                except Exception as e:
                    self.logger.error(f"Pattern detection failed for {symbol}: {str(e)}")
        
        return all_signals
    
    async def _analyze_single_symbol(self, symbol: str) -> List[TechnicalSignal]:
        """Analyze a single symbol for patterns"""
        
        try:
            # Get historical data
            bars = await self.data_provider.get_bars(
                symbol=symbol,
                timeframe='1Day',
                limit=100
            )
            
            if not bars or len(bars) < 50:
                return []
            
            # Convert to DataFrames
            price_data = pd.DataFrame(bars)
            price_data.columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
            price_data.set_index('timestamp', inplace=True)
            
            volume_data = price_data[['volume']].copy()
            
            # Run pattern recognition
            analysis = self.pattern_engine.analyze_ticker(
                ticker=symbol,
                price_data=price_data,
                volume_data=volume_data
            )
            
            return analysis.get('signals', [])
            
        except Exception as e:
            self.logger.error(f"Failed to analyze {symbol}: {str(e)}")
            return []
    
    def _rank_signals(self, signals: List[TechnicalSignal]) -> List[TechnicalSignal]:
        """Rank signals by quality score"""
        
        # Filter by quality thresholds
        quality_signals = [
            s for s in signals
            if s.confidence >= self.min_confidence
            and s.risk_reward_ratio >= self.min_risk_reward
        ]
        
        # Calculate composite score for each signal
        for signal in quality_signals:
            # Composite score based on multiple factors
            signal.quality_score = (
                signal.confidence * 0.4 +
                min(signal.risk_reward_ratio, 5) * 2 +  # Cap at 5 for scoring
                (10 if signal.volume_confirmation else 5) * 0.2 +
                signal.pattern_completion / 10 * 0.2
            )
        
        # Sort by quality score
        quality_signals.sort(key=lambda x: x.quality_score, reverse=True)
        
        # Limit signals per ticker
        seen_tickers = {}
        final_signals = []
        
        for signal in quality_signals:
            count = seen_tickers.get(signal.ticker, 0)
            if count < self.max_signals_per_ticker:
                final_signals.append(signal)
                seen_tickers[signal.ticker] = count + 1
        
        return final_signals
    
    def _format_opportunities(self, signals: List[TechnicalSignal]) -> List[Dict]:
        """Format signals as opportunities for agent consumption"""
        
        opportunities = []
        
        for signal in signals[:20]:  # Top 20 signals
            opportunity = {
                'ticker': signal.ticker,
                'pattern': signal.pattern_type.value,
                'direction': signal.direction.value,
                'confidence': signal.confidence,
                'entry_price': signal.entry_price,
                'target_price': signal.target_price,
                'stop_loss': signal.stop_loss_price,
                'risk_reward_ratio': signal.risk_reward_ratio,
                'quality_score': getattr(signal, 'quality_score', signal.confidence),
                'time_horizon': signal.time_horizon,
                'technical_summary': {
                    'pattern_completion': signal.pattern_completion,
                    'volume_confirmation': signal.volume_confirmation,
                    'rsi': signal.rsi,
                    'macd_signal': signal.macd_signal,
                    'moving_avg_position': signal.moving_avg_position
                },
                'detected_at': signal.detected_at,
                'priority': self._calculate_priority(signal)
            }
            
            opportunities.append(opportunity)
        
        return opportunities
    
    def _calculate_priority(self, signal: TechnicalSignal) -> str:
        """Calculate priority level for opportunity"""
        
        quality_score = getattr(signal, 'quality_score', signal.confidence)
        
        if quality_score >= 8.5:
            return 'HIGH'
        elif quality_score >= 7.0:
            return 'MEDIUM'
        else:
            return 'LOW'
    
    async def _save_screening_results(self, opportunities: List[Dict]):
        """Save screening results to database"""
        
        if not self.db_manager:
            return  # Skip if no database configured
        
        try:
            with self.db_manager.get_session() as session:
                # Create screening result record
                from database.models.screening import ScreeningResult, PatternDetection
                
                screening_result = ScreeningResult(
                    scan_date=datetime.now(),
                    symbols_scanned=len(self.screening_universe),
                    patterns_detected=len(opportunities),
                    top_opportunities=opportunities[:10],  # Store top 10
                    scan_duration=0  # Will be updated
                )
                
                session.add(screening_result)
                session.flush()  # Get the ID
                
                # Create pattern detection records
                for opp in opportunities[:20]:
                    pattern = PatternDetection(
                        ticker=opp['ticker'],
                        pattern_type=opp['pattern'],
                        confidence=opp['confidence'],
                        entry_price=opp['entry_price'],
                        target_price=opp['target_price'],
                        stop_loss=opp['stop_loss'],
                        risk_reward_ratio=opp['risk_reward_ratio'],
                        detected_at=opp['detected_at'],
                        screening_result_id=screening_result.id
                    )
                    session.add(pattern)
                
                session.commit()
                
        except Exception as e:
            self.logger.error(f"Failed to save screening results: {str(e)}")
    
    def _calculate_scan_statistics(self, all_signals: List, 
                                  top_signals: List) -> Dict:
        """Calculate scanning statistics"""
        
        if not all_signals:
            return {
                'patterns_by_type': {},
                'bullish_bearish_ratio': 0,
                'avg_confidence': 0,
                'avg_risk_reward': 0
            }
        
        # Pattern distribution
        pattern_counts = {}
        for signal in all_signals:
            pattern_type = signal.pattern_type.value
            pattern_counts[pattern_type] = pattern_counts.get(pattern_type, 0) + 1
        
        # Direction distribution
        bullish = sum(1 for s in all_signals if s.direction.value == 'bullish')
        bearish = sum(1 for s in all_signals if s.direction.value == 'bearish')
        
        # Quality metrics
        avg_confidence = sum(s.confidence for s in top_signals) / len(top_signals) if top_signals else 0
        avg_risk_reward = sum(s.risk_reward_ratio for s in top_signals) / len(top_signals) if top_signals else 0
        
        return {
            'patterns_by_type': pattern_counts,
            'bullish_bearish_ratio': bullish / bearish if bearish > 0 else bullish,
            'bullish_count': bullish,
            'bearish_count': bearish,
            'avg_confidence': avg_confidence,
            'avg_risk_reward': avg_risk_reward,
            'top_patterns': list(pattern_counts.keys())[:5]
        }
    
    async def screen_specific_symbols(self, symbols: List[str]) -> List[TechnicalSignal]:
        """Screen specific symbols on demand"""
        
        all_signals = []
        
        for symbol in symbols:
            try:
                signals = await self._analyze_single_symbol(symbol)
                all_signals.extend(signals)
                
            except Exception as e:
                self.logger.error(f"Failed to screen {symbol}: {str(e)}")
        
        return self._rank_signals(all_signals)
    
    async def get_top_opportunities(self, limit: int = 10) -> List[Dict]:
        """Get top opportunities from cache or run new scan"""
        
        # Check cache
        if 'latest_scan' in self.screening_cache:
            cache_time = self.screening_cache.get('cache_time')
            if cache_time and (datetime.now() - cache_time) < self.cache_duration:
                cached_results = self.screening_cache['latest_scan']
                return cached_results.get('opportunities', [])[:limit]
        
        # Run new scan if cache miss
        results = await self.run_daily_scan()
        return results.get('opportunities', [])[:limit]
    
    def get_screening_statistics(self) -> Dict:
        """Get screening performance statistics"""
        
        return {
            'universe_size': len(self.screening_universe),
            'sp500_symbols': len(self.sp500_symbols),
            'nasdaq_symbols': len(self.nasdaq_symbols),
            'cache_status': 'valid' if self._is_cache_valid() else 'expired',
            'performance_metrics': self.performance_tracker.get_daily_summary()
        }
    
    def _is_cache_valid(self) -> bool:
        """Check if cache is still valid"""
        
        if 'cache_time' not in self.screening_cache:
            return False
        
        cache_time = self.screening_cache['cache_time']
        return (datetime.now() - cache_time) < self.cache_duration
    
    def get_universe_info(self) -> Dict:
        """Get information about screening universe"""
        
        return {
            'total_symbols': len(self.screening_universe),
            'sp500_count': len(self.sp500_symbols),
            'nasdaq_count': len(self.nasdaq_symbols),
            'min_volume_filter': self.min_avg_volume,
            'min_market_cap_filter': self.min_market_cap,
            'last_update': self.screening_cache.get('cache_time', 'Never')
        }