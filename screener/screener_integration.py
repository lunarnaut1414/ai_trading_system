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
    
    def __init__(self, screener):
        """
        Initialize integration layer
        
        Args:
            screener: TechnicalScreener instance
        """
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
                },
                'top_patterns': scan_stats.get('top_patterns', [])
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
    
    async def get_pre_market_signals(self, watchlist: Optional[List[str]] = None) -> List[Dict]:
        """
        Get pre-market technical signals for daily planning
        
        Args:
            watchlist: Optional list of symbols to focus on
            
        Returns:
            List of pre-market signals
        """
        
        try:
            if watchlist:
                # Screen specific watchlist
                signals = await self.screener.screen_specific_symbols(watchlist)
            else:
                # Get top opportunities from cache
                opportunities = await self.screener.get_top_opportunities(limit=20)
                
                # Convert opportunities to signals format
                signals = []
                for opp in opportunities:
                    if opp.get('priority') in ['HIGH', 'MEDIUM']:
                        signals.append({
                            'ticker': opp['ticker'],
                            'pattern': opp['pattern'],
                            'direction': opp['direction'],
                            'confidence': opp['confidence'],
                            'entry_price': opp['entry_price'],
                            'priority': opp['priority']
                        })
            
            return signals
            
        except Exception as e:
            self.logger.error(f"Failed to get pre-market signals: {str(e)}")
            return []
    
    async def get_intraday_alerts(self, positions: List[str]) -> List[Dict]:
        """
        Get intraday alerts for existing positions
        
        Args:
            positions: List of current position symbols
            
        Returns:
            List of technical alerts
        """
        
        alerts = []
        
        try:
            # Screen current positions
            position_signals = await self.screener.screen_specific_symbols(positions)
            
            for signal in position_signals:
                # Check for significant patterns
                if signal.confidence >= 7.0:
                    alert_type = self._determine_alert_type(signal)
                    
                    if alert_type:
                        alerts.append({
                            'ticker': signal.ticker,
                            'alert_type': alert_type,
                            'pattern': signal.pattern_type.value,
                            'confidence': signal.confidence,
                            'message': self._generate_alert_message(signal, alert_type),
                            'severity': self._determine_severity(signal),
                            'timestamp': datetime.now()
                        })
            
            return alerts
            
        except Exception as e:
            self.logger.error(f"Failed to get intraday alerts: {str(e)}")
            return []
    
    def _determine_alert_type(self, signal) -> Optional[str]:
        """Determine alert type based on pattern"""
        
        pattern = signal.pattern_type.value
        
        if 'breakout' in pattern:
            return 'breakout'
        elif 'reversal' in pattern or 'double' in pattern:
            return 'reversal'
        elif 'flag' in pattern or 'pennant' in pattern:
            return 'continuation'
        elif signal.confidence >= 8.0:
            return 'high_confidence'
        else:
            return None
    
    def _generate_alert_message(self, signal, alert_type: str) -> str:
        """Generate alert message"""
        
        messages = {
            'breakout': f"{signal.ticker}: {signal.pattern_type.value} detected - potential breakout opportunity",
            'reversal': f"{signal.ticker}: Reversal pattern forming - consider position adjustment",
            'continuation': f"{signal.ticker}: Continuation pattern detected - trend likely to persist",
            'high_confidence': f"{signal.ticker}: High confidence {signal.pattern_type.value} signal"
        }
        
        return messages.get(alert_type, f"{signal.ticker}: Technical signal detected")
    
    def _determine_severity(self, signal) -> str:
        """Determine alert severity"""
        
        if signal.confidence >= 8.5:
            return 'critical'
        elif signal.confidence >= 7.5:
            return 'warning'
        else:
            return 'info'
    
    async def generate_daily_technical_report(self) -> Dict:
        """
        Generate comprehensive daily technical report
        
        Returns:
            Daily technical analysis report
        """
        
        try:
            # Get latest scan results
            latest_scan = self.screener.screening_cache.get('latest_scan', {})
            
            # Get market breadth
            breadth = await self.get_market_breadth_analysis()
            
            # Get top opportunities
            opportunities = await self.screener.get_top_opportunities(limit=10)
            
            report = {
                'report_date': datetime.now(),
                'market_overview': {
                    'sentiment': breadth.get('market_sentiment', 'neutral'),
                    'breadth_metrics': breadth.get('breadth_metrics', {}),
                    'pattern_distribution': breadth.get('pattern_distribution', {})
                },
                'screening_summary': {
                    'symbols_analyzed': latest_scan.get('symbols_scanned', 0),
                    'patterns_detected': latest_scan.get('total_signals', 0),
                    'high_quality_signals': len([o for o in opportunities if o.get('confidence', 0) >= 7.5]),
                    'scan_duration': latest_scan.get('scan_duration', 0)
                },
                'top_opportunities': opportunities[:5],
                'pattern_insights': self._generate_pattern_insights(latest_scan.get('statistics', {})),
                'recommendations': {
                    'new_positions': [o for o in opportunities if o.get('priority') == 'HIGH'][:3],
                    'monitor_list': [o for o in opportunities if o.get('priority') == 'MEDIUM'][:5]
                }
            }
            
            self.logger.info("Generated daily technical report")
            return report
            
        except Exception as e:
            self.logger.error(f"Failed to generate daily report: {str(e)}")
            return {}
    
    def _generate_pattern_insights(self, statistics: Dict) -> List[str]:
        """Generate insights from pattern statistics"""
        
        insights = []
        
        # Market sentiment insight
        bullish_ratio = statistics.get('bullish_count', 0) / max(statistics.get('bullish_count', 0) + statistics.get('bearish_count', 0), 1)
        if bullish_ratio > 0.6:
            insights.append("Market showing bullish technical bias with majority of patterns favoring upside")
        elif bullish_ratio < 0.4:
            insights.append("Technical patterns suggesting bearish market conditions")
        
        # Pattern type insights
        pattern_types = statistics.get('patterns_by_type', {})
        if pattern_types:
            most_common = max(pattern_types.items(), key=lambda x: x[1])
            insights.append(f"Most common pattern: {most_common[0]} ({most_common[1]} occurrences)")
        
        # Quality insights
        avg_confidence = statistics.get('avg_confidence', 0)
        if avg_confidence >= 7.5:
            insights.append("High quality signals detected with strong pattern confirmation")
        elif avg_confidence <= 6.0:
            insights.append("Pattern quality below average - exercise caution")
        
        return insights