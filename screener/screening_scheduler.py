# screener/screening_scheduler.py
"""
Automated Scheduling for Technical Screener
"""

import asyncio
import schedule
import time
from datetime import datetime, time as datetime_time
from typing import Optional, Callable
import logging
import threading

from .technical_screener import TechnicalScreener

class ScreeningScheduler:
    """
    Scheduler for automated technical screening runs
    """
    
    def __init__(self, screener: TechnicalScreener):
        self.screener = screener
        self.logger = logging.getLogger('screening_scheduler')
        
        # Scheduling configuration
        self.post_market_time = "16:30"  # 4:30 PM ET
        self.pre_market_time = "08:00"   # 8:00 AM ET
        
        # Scheduler state
        self.is_running = False
        self.scheduler_thread = None
        self.last_scan_time = None
        self.last_scan_results = None
        
        # Callback handlers
        self.on_scan_complete = None
        self.on_scan_error = None
        
    def start_daily_scanning(self):
        """Start automated daily scanning schedule"""
        
        if self.is_running:
            self.logger.warning("Scheduler already running")
            return
        
        self.is_running = True
        
        # Schedule daily scans
        schedule.every().day.at(self.post_market_time).do(self._run_post_market_scan)
        schedule.every().day.at(self.pre_market_time).do(self._run_pre_market_scan)
        
        # Start scheduler thread
        self.scheduler_thread = threading.Thread(target=self._run_scheduler)
        self.scheduler_thread.daemon = True
        self.scheduler_thread.start()
        
        self.logger.info(f"Screening scheduler started - Post-market: {self.post_market_time}, Pre-market: {self.pre_market_time}")
    
    def stop_scanning(self):
        """Stop automated scanning"""
        
        self.is_running = False
        schedule.clear()
        
        if self.scheduler_thread:
            self.scheduler_thread.join(timeout=5)
        
        self.logger.info("Screening scheduler stopped")
    
    def _run_scheduler(self):
        """Run the scheduler loop"""
        
        while self.is_running:
            schedule.run_pending()
            time.sleep(60)  # Check every minute
    
    def _run_post_market_scan(self):
        """Run post-market technical scan"""
        
        self.logger.info("Starting post-market technical scan")
        
        try:
            # Run async scan in sync context
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            results = loop.run_until_complete(self.screener.run_daily_scan())
            
            self.last_scan_time = datetime.now()
            self.last_scan_results = results
            
            # Process results
            self._process_scan_results(results, 'post_market')
            
            # Trigger callback if set
            if self.on_scan_complete:
                self.on_scan_complete(results)
            
            self.logger.info(f"Post-market scan completed: {len(results.get('top_signals', []))} signals found")
            
        except Exception as e:
            self.logger.error(f"Post-market scan failed: {str(e)}")
            
            if self.on_scan_error:
                self.on_scan_error(str(e))
    
    def _run_pre_market_scan(self):
        """Run pre-market quick scan"""
        
        self.logger.info("Starting pre-market quick scan")
        
        try:
            # Focus on high-priority symbols only
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            # Get top movers or specific watchlist
            priority_symbols = self._get_priority_symbols()
            
            results = loop.run_until_complete(
                self.screener.screen_specific_symbols(priority_symbols)
            )
            
            self._process_scan_results({'top_signals': results}, 'pre_market')
            
            self.logger.info(f"Pre-market scan completed: {len(results)} signals found")
            
        except Exception as e:
            self.logger.error(f"Pre-market scan failed: {str(e)}")
    
    def _get_priority_symbols(self) -> list:
        """Get priority symbols for pre-market scan"""
        
        # In production, this would fetch pre-market movers
        # For now, return top tech stocks
        return ['AAPL', 'MSFT', 'NVDA', 'AMZN', 'GOOGL', 'META', 'TSLA', 
                'AMD', 'AVGO', 'ORCL', 'CRM', 'ADBE', 'NFLX', 'INTC', 'QCOM']
    
    def _process_scan_results(self, results: dict, scan_type: str):
        """Process and distribute scan results"""
        
        if not results or 'error' in results:
            return
        
        top_signals = results.get('top_signals', [])
        opportunities = results.get('opportunities', [])
        
        # Log summary
        self.logger.info(f"""
        {scan_type.upper()} SCAN SUMMARY:
        - Symbols Scanned: {results.get('symbols_scanned', 0)}
        - Total Signals: {results.get('total_signals', 0)}
        - Quality Signals: {len(top_signals)}
        - Top Opportunities: {len(opportunities)}
        - Scan Duration: {results.get('scan_duration', 0):.2f}s
        """)
        
        # Store for agent consumption
        self._store_opportunities_for_agents(opportunities)
    
    def _store_opportunities_for_agents(self, opportunities: list):
        """Store opportunities for agent consumption"""
        
        # This would integrate with the agent communication system
        # For now, just log
        if opportunities:
            self.logger.info(f"Stored {len(opportunities)} opportunities for agent analysis")
    
    def register_callbacks(self, on_complete: Optional[Callable] = None,
                          on_error: Optional[Callable] = None):
        """Register callback functions for scan events"""
        
        if on_complete:
            self.on_scan_complete = on_complete
        
        if on_error:
            self.on_scan_error = on_error


# screener/screener_integration.py
"""
Integration layer for Technical Screener with other agents
"""

from typing import Dict, List, Optional
import logging
from datetime import datetime

class ScreenerIntegration:
    """
    Integration interface for Technical Screener with other system components
    """
    
    def __init__(self, screener: TechnicalScreener):
        self.screener = screener
        self.logger = logging.getLogger('screener_integration')
        
    async def get_opportunities_for_junior_analyst(self, 
                                                  limit: int = 10,
                                                  min_confidence: float = 7.0) -> List[Dict]:
        """
        Get formatted opportunities for Junior Research Analyst
        
        Args:
            limit: Maximum number of opportunities
            min_confidence: Minimum confidence threshold
            
        Returns:
            List of opportunities formatted for analyst consumption
        """
        
        try:
            # Get top opportunities
            opportunities = await self.screener.get_top_opportunities(limit=limit * 2)
            
            # Filter by confidence
            filtered = [
                opp for opp in opportunities
                if opp.get('confidence', 0) >= min_confidence
            ]
            
            # Format for Junior Analyst
            formatted_opportunities = []
            
            for opp in filtered[:limit]:
                formatted = {
                    'ticker': opp['ticker'],
                    'signal_type': 'technical_pattern',
                    'pattern': opp['pattern'],
                    'direction': opp['direction'],
                    'confidence': opp['confidence'],
                    'entry_point': {
                        'price': opp['entry_price'],
                        'timing': 'immediate' if opp['time_horizon'] == 'short' else 'patient'
                    },
                    'targets': {
                        'primary': opp['target_price'],
                        'stop_loss': opp['stop_loss']
                    },
                    'risk_metrics': {
                        'risk_reward_ratio': opp['risk_reward_ratio'],
                        'quality_score': opp['quality_score']
                    },
                    'technical_context': opp.get('technical_summary', {}),
                    'priority': opp['priority'],
                    'timestamp': opp['detected_at'],
                    'requires_fundamental_validation': True
                }
                
                formatted_opportunities.append(formatted)
            
            self.logger.info(f"Prepared {len(formatted_opportunities)} opportunities for Junior Analyst")
            
            return formatted_opportunities
            
        except Exception as e:
            self.logger.error(f"Failed to get opportunities for analyst: {str(e)}")
            return []
    
    async def get_market_breadth_analysis(self) -> Dict:
        """
        Get market breadth analysis from screening results
        
        Returns:
            Market breadth metrics and sentiment
        """
        
        try:
            # Get latest screening statistics
            stats = self.screener.get_screening_statistics()
            
            # Get cached scan results
            latest_scan = self.screener.screening_cache.get('latest_scan', {})
            scan_stats = latest_scan.get('statistics', {})
            
            # Calculate market breadth
            bullish_count = scan_stats.get('bullish_count', 0)
            bearish_count = scan_stats.get('bearish_count', 0)
            total_signals = bullish_count + bearish_count
            
            breadth_analysis = {
                'timestamp': datetime.now(),
                'market_sentiment': self._calculate_market_sentiment(bullish_count, bearish_count),
                'breadth_metrics': {
                    'bullish_signals': bullish_count,
                    'bearish_signals': bearish_count,
                    'neutral_signals': scan_stats.get('neutral_count', 0),
                    'bullish_percentage': (bullish_count / total_signals * 100) if total_signals > 0 else 0
                },
                'pattern_distribution': scan_stats.get('patterns_by_type', {}),
                'quality_metrics': {
                    'avg_confidence': scan_stats.get('avg_confidence', 0),
                    'avg_risk_reward': scan_stats.get('avg_risk_reward', 0)
                },
                'universe_coverage': {
                    'symbols_analyzed': latest_scan.get('symbols_scanned', 0),
                    'total_universe': stats.get('universe_size', 0),
                    'coverage_percentage': (latest_scan.get('symbols_scanned', 0) / stats.get('universe_size', 1) * 100)
                }
            }
            
            return breadth_analysis
            
        except Exception as e:
            self.logger.error(f"Failed to generate market breadth analysis: {str(e)}")
            return {}
    
    def _calculate_market_sentiment(self, bullish: int, bearish: int) -> str:
        """Calculate overall market sentiment"""
        
        if bullish + bearish == 0:
            return 'neutral'
        
        bullish_ratio = bullish / (bullish + bearish)
        
        if bullish_ratio >= 0.7:
            return 'strongly_bullish'
        elif bullish_ratio >= 0.55:
            return 'bullish'
        elif bullish_ratio >= 0.45:
            return 'neutral'
        elif bullish_ratio >= 0.3:
            return 'bearish'
        else:
            return 'strongly_bearish'
    
    async def screen_portfolio_holdings(self, holdings: List[str]) -> Dict:
        """
        Screen current portfolio holdings for technical signals
        
        Args:
            holdings: List of ticker symbols in portfolio
            
        Returns:
            Technical signals for portfolio holdings
        """
        
        try:
            # Screen specific holdings
            signals = await self.screener.screen_specific_symbols(holdings)
            
            # Group by ticker
            signals_by_ticker = {}
            for signal in signals:
                ticker = signal.ticker
                if ticker not in signals_by_ticker:
                    signals_by_ticker[ticker] = []
                signals_by_ticker[ticker].append({
                    'pattern': signal.pattern_type.value,
                    'direction': signal.direction.value,
                    'confidence': signal.confidence,
                    'action_required': self._determine_action(signal)
                })
            
            return {
                'timestamp': datetime.now(),
                'holdings_analyzed': len(holdings),
                'holdings_with_signals': len(signals_by_ticker),
                'signals_by_ticker': signals_by_ticker,
                'summary': self._generate_holdings_summary(signals_by_ticker)
            }
            
        except Exception as e:
            self.logger.error(f"Failed to screen portfolio holdings: {str(e)}")
            return {}
    
    def _determine_action(self, signal) -> str:
        """Determine recommended action based on signal"""
        
        if signal.confidence >= 8.0:
            if signal.direction.value == 'bullish':
                return 'consider_adding'
            else:
                return 'consider_reducing'
        elif signal.confidence >= 6.5:
            return 'monitor_closely'
        else:
            return 'no_action'
    
    def _generate_holdings_summary(self, signals_by_ticker: Dict) -> Dict:
        """Generate summary of holdings analysis"""
        
        bullish_holdings = 0
        bearish_holdings = 0
        neutral_holdings = 0
        
        for ticker, signals in signals_by_ticker.items():
            sentiment = self._aggregate_ticker_sentiment(signals)
            if sentiment == 'bullish':
                bullish_holdings += 1
            elif sentiment == 'bearish':
                bearish_holdings += 1
            else:
                neutral_holdings += 1
        
        return {
            'bullish_holdings': bullish_holdings,
            'bearish_holdings': bearish_holdings,
            'neutral_holdings': neutral_holdings,
            'recommended_actions': self._generate_recommendations(signals_by_ticker)
        }
    
    def _aggregate_ticker_sentiment(self, signals: List[Dict]) -> str:
        """Aggregate multiple signals for a ticker into overall sentiment"""
        
        if not signals:
            return 'neutral'
        
        bullish = sum(1 for s in signals if s['direction'] == 'bullish')
        bearish = sum(1 for s in signals if s['direction'] == 'bearish')
        
        if bullish > bearish:
            return 'bullish'
        elif bearish > bullish:
            return 'bearish'
        else:
            return 'neutral'
    
    def _generate_recommendations(self, signals_by_ticker: Dict) -> List[Dict]:
        """Generate actionable recommendations from signals"""
        
        recommendations = []
        
        for ticker, signals in signals_by_ticker.items():
            # Find highest confidence signal
            if signals:
                best_signal = max(signals, key=lambda x: x['confidence'])
                
                if best_signal['confidence'] >= 7.5:
                    recommendations.append({
                        'ticker': ticker,
                        'action': best_signal['action_required'],
                        'pattern': best_signal['pattern'],
                        'confidence': best_signal['confidence'],
                        'priority': 'high' if best_signal['confidence'] >= 8.5 else 'medium'
                    })
        
        # Sort by confidence
        recommendations.sort(key=lambda x: x['confidence'], reverse=True)
        
        return recommendations[:5]  # Top 5 recommendations