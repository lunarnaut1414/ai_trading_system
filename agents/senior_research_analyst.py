# agents/senior_research_analyst.py
"""
Senior Research Analyst Agent - Complete Implementation
Designed for macOS M2 Max with Claude AI integration and markdown reporting

This agent synthesizes multiple junior analyst reports into strategic portfolio
recommendations, ranking opportunities and identifying market themes.
"""

import asyncio
import json
import logging
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
from enum import Enum
from dataclasses import dataclass
import numpy as np
import statistics
from collections import defaultdict
import uuid

# Enums for analysis categorization
class ConvictionLevel(Enum):
    LOW = 1
    MEDIUM_LOW = 2
    MEDIUM = 3
    MEDIUM_HIGH = 4
    HIGH = 5

class TimeHorizon(Enum):
    SHORT = "short"      # 1-10 days
    MEDIUM = "medium"    # 2-8 weeks  
    LONG = "long"        # 2-6 months

class MarketRegime(Enum):
    RISK_ON = "risk_on"
    RISK_OFF = "risk_off"
    NEUTRAL = "neutral"
    TRANSITION = "transition"

class SectorTheme(Enum):
    GROWTH = "growth"
    VALUE = "value"
    DEFENSIVE = "defensive"
    CYCLICAL = "cyclical"
    MOMENTUM = "momentum"

@dataclass
class OpportunityRanking:
    """Data class for ranked opportunities"""
    ticker: str
    conviction_score: float
    risk_adjusted_score: float
    time_horizon: str
    expected_return: float
    risk_level: str
    sector: str
    market_cap: str
    correlation_risk: float
    liquidity_score: float
    catalyst_strength: float
    thesis_summary: str
    key_risks: List[str]

@dataclass
class PortfolioTheme:
    """Data class for portfolio themes"""
    theme_name: str
    confidence: float
    supporting_tickers: List[str]
    time_horizon: str
    risk_factors: List[str]
    allocation_suggestion: float


class StrategicAnalysisEngine:
    """
    Strategic analysis engine for the Senior Research Analyst
    Synthesizes multiple junior analyst reports into portfolio strategy
    """
    
    def __init__(self):
        self.logger = logging.getLogger('strategic_analysis')
        
        # Scoring weights for opportunity ranking
        self.scoring_weights = {
            'conviction': 0.25,
            'risk_reward': 0.20,
            'catalyst_strength': 0.15,
            'technical_score': 0.15,
            'liquidity': 0.10,
            'correlation_bonus': 0.10,
            'sector_momentum': 0.05
        }
        
        # Risk thresholds
        self.risk_thresholds = {
            'max_correlation': 0.70,
            'max_sector_concentration': 0.25,
            'min_liquidity_score': 6.0,
            'max_portfolio_beta': 1.30
        }
    
    def synthesize_junior_reports(self, junior_reports: List[Dict], 
                                 market_context: Dict, 
                                 portfolio_context: Dict) -> Dict:
        """Synthesize multiple junior analyst reports into strategic recommendations"""
        
        try:
            if not junior_reports:
                return self._create_empty_analysis("No junior reports provided")
            
            # Filter and validate reports
            valid_reports = self._filter_valid_reports(junior_reports)
            if not valid_reports:
                return self._create_empty_analysis("No valid junior reports")
            
            # Calculate opportunity rankings
            ranked_opportunities = self._rank_opportunities(valid_reports, market_context, portfolio_context)
            
            # Identify strategic themes
            strategic_themes = self._identify_strategic_themes(valid_reports, market_context)
            
            # Assess portfolio-level risk
            risk_assessment = self._assess_portfolio_risk(ranked_opportunities, portfolio_context)
            
            # Balance time horizons
            time_horizon_allocation = self._balance_time_horizons(ranked_opportunities)
            
            # Generate correlation matrix
            correlation_analysis = self._analyze_correlations(ranked_opportunities)
            
            return {
                'synthesis_timestamp': datetime.now().isoformat(),
                'ranked_opportunities': ranked_opportunities,
                'strategic_themes': strategic_themes,
                'risk_assessment': risk_assessment,
                'time_horizon_allocation': time_horizon_allocation,
                'correlation_analysis': correlation_analysis,
                'market_regime': market_context.get('regime', MarketRegime.NEUTRAL.value),
                'recommendation_summary': self._create_recommendation_summary(
                    ranked_opportunities, strategic_themes, risk_assessment
                )
            }
            
        except Exception as e:
            self.logger.error(f"Synthesis failed: {str(e)}")
            return self._create_empty_analysis(f"Synthesis error: {str(e)}")
    
    def _filter_valid_reports(self, reports: List[Dict]) -> List[Dict]:
        """Filter out invalid or incomplete reports"""
        valid_reports = []
        
        for report in reports:
            if self._is_valid_report(report):
                valid_reports.append(report)
            else:
                self.logger.warning(f"Filtered out invalid report for {report.get('ticker', 'Unknown')}")
        
        return valid_reports
    
    def _is_valid_report(self, report: Dict) -> bool:
        """Check if a junior analyst report is valid"""
        required_fields = ['ticker', 'recommendation', 'confidence', 'analysis_status']
        return all(field in report for field in required_fields) and report['analysis_status'] == 'success'
    
    def _rank_opportunities(self, reports: List[Dict], 
                          market_context: Dict, 
                          portfolio_context: Dict) -> List[OpportunityRanking]:
        """Rank opportunities using multi-factor scoring"""
        
        rankings = []
        
        for report in reports:
            score = self._calculate_opportunity_score(report, market_context, portfolio_context)
            
            ranking = OpportunityRanking(
                ticker=report['ticker'],
                conviction_score=score['conviction_score'],
                risk_adjusted_score=score['risk_adjusted_score'],
                time_horizon=report.get('time_horizon', 'medium'),
                expected_return=report.get('target_upside_percent', 0),
                risk_level=report.get('risk_assessment', {}).get('risk_level', 'medium'),
                sector=report.get('sector', 'Unknown'),
                market_cap=report.get('market_cap', 'Unknown'),
                correlation_risk=score.get('correlation_risk', 0.5),
                liquidity_score=score.get('liquidity_score', 5.0),
                catalyst_strength=score.get('catalyst_strength', 0.5),
                thesis_summary=report.get('thesis', '')[:200],
                key_risks=report.get('key_risks', [])
            )
            
            rankings.append(ranking)
        
        # Sort by risk-adjusted score (descending)
        rankings.sort(key=lambda x: x.risk_adjusted_score, reverse=True)
        
        return rankings[:10]  # Return top 10 opportunities
    
    def _calculate_opportunity_score(self, report: Dict, 
                                   market_context: Dict, 
                                   portfolio_context: Dict) -> Dict:
        """Calculate multi-factor opportunity score"""
        
        scores = {}
        
        # Conviction score (from junior analyst)
        scores['conviction'] = report.get('confidence', 5) / 10.0
        
        # Risk-reward score
        upside = report.get('target_upside_percent', 0)
        downside = report.get('stop_loss_percent', 0)
        risk_reward = upside / max(downside, 1) if downside > 0 else upside / 10
        scores['risk_reward'] = min(risk_reward / 5, 1.0)  # Normalize to 0-1
        
        # Catalyst strength
        catalysts = report.get('catalysts', [])
        scores['catalyst_strength'] = min(len(catalysts) / 3, 1.0)
        
        # Technical score
        tech_analysis = report.get('technical_analysis', {})
        scores['technical_score'] = tech_analysis.get('technical_score', 5) / 10.0
        
        # Liquidity score
        volume_ratio = tech_analysis.get('volume_ratio', 1.0)
        scores['liquidity'] = min(volume_ratio, 2.0) / 2.0
        
        # Correlation bonus (favor uncorrelated opportunities)
        scores['correlation_bonus'] = self._calculate_correlation_bonus(
            report['ticker'], portfolio_context
        )
        
        # Sector momentum
        scores['sector_momentum'] = self._calculate_sector_momentum(
            report.get('sector'), market_context
        )
        
        # Calculate weighted scores
        conviction_score = sum(
            scores[factor] * self.scoring_weights.get(factor, 0)
            for factor in scores
        )
        
        # Risk adjustment
        risk_multiplier = self._get_risk_multiplier(report.get('risk_assessment', {}))
        risk_adjusted_score = conviction_score * risk_multiplier
        
        return {
            'conviction_score': round(conviction_score, 3),
            'risk_adjusted_score': round(risk_adjusted_score, 3),
            'correlation_risk': 1 - scores['correlation_bonus'],
            'liquidity_score': scores['liquidity'] * 10,
            'catalyst_strength': scores['catalyst_strength']
        }
    
    def _calculate_correlation_bonus(self, ticker: str, portfolio_context: Dict) -> float:
        """Calculate bonus for uncorrelated opportunities"""
        # Simplified correlation calculation
        existing_positions = portfolio_context.get('positions', [])
        
        if not existing_positions:
            return 1.0  # Maximum bonus for first position
        
        # In production, would use actual correlation matrix
        # For now, use sector diversification as proxy
        existing_sectors = [pos.get('sector') for pos in existing_positions]
        ticker_sector = ticker[:3]  # Simplified sector detection
        
        if ticker_sector not in existing_sectors:
            return 0.8
        else:
            return 0.3
    
    def _calculate_sector_momentum(self, sector: str, market_context: Dict) -> float:
        """Calculate sector momentum score"""
        sector_performance = market_context.get('sector_performance', {})
        
        if sector in sector_performance:
            momentum = sector_performance[sector].get('momentum', 0)
            return (momentum + 100) / 200  # Normalize -100 to 100 => 0 to 1
        
        return 0.5  # Neutral if no data
    
    def _get_risk_multiplier(self, risk_assessment: Dict) -> float:
        """Get risk adjustment multiplier"""
        risk_level = risk_assessment.get('risk_level', 'medium')
        
        multipliers = {
            'low': 1.2,
            'medium': 1.0,
            'high': 0.8,
            'very_high': 0.6
        }
        
        return multipliers.get(risk_level, 1.0)
    
    def _identify_strategic_themes(self, reports: List[Dict], market_context: Dict) -> List[PortfolioTheme]:
        """Identify strategic themes from analyst reports"""
        
        themes = []
        
        # Sector concentration analysis
        sector_counts = defaultdict(list)
        for report in reports:
            if report.get('recommendation') in ['BUY', 'STRONG_BUY']:
                sector = report.get('sector', 'Unknown')
                sector_counts[sector].append(report['ticker'])
        
        # Create themes from concentrated sectors
        for sector, tickers in sector_counts.items():
            if len(tickers) >= 2:  # At least 2 stocks in sector
                theme = PortfolioTheme(
                    theme_name=f"{sector} Opportunity",
                    confidence=min(len(tickers) / 5, 1.0),
                    supporting_tickers=tickers[:5],
                    time_horizon='medium',
                    risk_factors=[f"{sector} sector rotation risk"],
                    allocation_suggestion=min(len(tickers) * 0.05, 0.25)
                )
                themes.append(theme)
        
        # Market regime themes
        regime = market_context.get('regime', 'neutral')
        if regime == 'risk_on':
            themes.append(PortfolioTheme(
                theme_name="Growth Momentum",
                confidence=0.7,
                supporting_tickers=self._get_growth_tickers(reports),
                time_horizon='short',
                risk_factors=["Sentiment reversal", "Valuation concerns"],
                allocation_suggestion=0.3
            ))
        elif regime == 'risk_off':
            themes.append(PortfolioTheme(
                theme_name="Defensive Quality",
                confidence=0.8,
                supporting_tickers=self._get_defensive_tickers(reports),
                time_horizon='medium',
                risk_factors=["Opportunity cost in recovery"],
                allocation_suggestion=0.4
            ))
        
        return themes[:3]  # Return top 3 themes
    
    def _get_growth_tickers(self, reports: List[Dict]) -> List[str]:
        """Get growth-oriented tickers"""
        growth_tickers = []
        for report in reports:
            if report.get('growth_score', 0) > 7:
                growth_tickers.append(report['ticker'])
        return growth_tickers[:5]
    
    def _get_defensive_tickers(self, reports: List[Dict]) -> List[str]:
        """Get defensive tickers"""
        defensive_sectors = ['Utilities', 'Consumer Staples', 'Healthcare']
        defensive_tickers = []
        for report in reports:
            if report.get('sector') in defensive_sectors:
                defensive_tickers.append(report['ticker'])
        return defensive_tickers[:5]
    
    def _assess_portfolio_risk(self, opportunities: List[OpportunityRanking], 
                              portfolio_context: Dict) -> Dict:
        """Assess portfolio-level risk metrics"""
        
        risk_assessment = {
            'timestamp': datetime.now().isoformat(),
            'risk_level': 'medium',
            'risk_factors': [],
            'risk_score': 5.0,
            'recommendations': []
        }
        
        # Concentration risk
        sector_concentration = self._calculate_sector_concentration(opportunities)
        if sector_concentration > self.risk_thresholds['max_sector_concentration']:
            risk_assessment['risk_factors'].append('High sector concentration')
            risk_assessment['recommendations'].append('Diversify across sectors')
        
        # Correlation risk
        avg_correlation = np.mean([opp.correlation_risk for opp in opportunities])
        if avg_correlation > self.risk_thresholds['max_correlation']:
            risk_assessment['risk_factors'].append('High correlation among positions')
            risk_assessment['recommendations'].append('Seek uncorrelated opportunities')
        
        # Liquidity risk
        avg_liquidity = np.mean([opp.liquidity_score for opp in opportunities])
        if avg_liquidity < self.risk_thresholds['min_liquidity_score']:
            risk_assessment['risk_factors'].append('Low liquidity in positions')
            risk_assessment['recommendations'].append('Focus on more liquid securities')
        
        # Calculate overall risk score
        risk_score = self._calculate_risk_score(
            sector_concentration, avg_correlation, avg_liquidity
        )
        risk_assessment['risk_score'] = round(risk_score, 1)
        
        # Determine risk level
        if risk_score < 3:
            risk_assessment['risk_level'] = 'low'
        elif risk_score < 7:
            risk_assessment['risk_level'] = 'medium'
        else:
            risk_assessment['risk_level'] = 'high'
        
        return risk_assessment
    
    def _calculate_sector_concentration(self, opportunities: List[OpportunityRanking]) -> float:
        """Calculate sector concentration"""
        if not opportunities:
            return 0.0
        
        sector_counts = defaultdict(int)
        for opp in opportunities:
            sector_counts[opp.sector] += 1
        
        max_concentration = max(sector_counts.values()) / len(opportunities)
        return max_concentration
    
    def _calculate_risk_score(self, sector_conc: float, avg_corr: float, avg_liq: float) -> float:
        """Calculate overall portfolio risk score (1-10)"""
        
        # Weighted risk calculation
        sector_risk = sector_conc * 10 * 0.3
        correlation_risk = avg_corr * 10 * 0.4
        liquidity_risk = (1 - avg_liq/10) * 10 * 0.3
        
        total_risk = sector_risk + correlation_risk + liquidity_risk
        return min(max(total_risk, 1), 10)
    
    def _balance_time_horizons(self, opportunities: List[OpportunityRanking]) -> Dict:
        """Balance opportunities across time horizons"""
        
        horizon_allocation = {
            'short': [],
            'medium': [],
            'long': []
        }
        
        for opp in opportunities:
            horizon = opp.time_horizon
            if horizon in horizon_allocation:
                horizon_allocation[horizon].append(opp.ticker)
        
        # Calculate percentages
        total = len(opportunities)
        if total > 0:
            percentages = {
                'short': len(horizon_allocation['short']) / total,
                'medium': len(horizon_allocation['medium']) / total,
                'long': len(horizon_allocation['long']) / total
            }
        else:
            percentages = {'short': 0, 'medium': 0, 'long': 0}
        
        # Recommendations
        recommendations = []
        if percentages['short'] > 0.5:
            recommendations.append("Consider adding longer-term positions for stability")
        if percentages['long'] > 0.5:
            recommendations.append("Consider adding shorter-term positions for flexibility")
        
        return {
            'allocation': horizon_allocation,
            'percentages': percentages,
            'recommendations': recommendations,
            'balance_score': self._calculate_balance_score(percentages)
        }
    
    def _calculate_balance_score(self, percentages: Dict) -> float:
        """Calculate time horizon balance score"""
        ideal = {'short': 0.3, 'medium': 0.4, 'long': 0.3}
        
        deviation = sum(abs(percentages[h] - ideal[h]) for h in ideal)
        score = max(0, 10 - (deviation * 10))
        
        return round(score, 1)
    
    def _analyze_correlations(self, opportunities: List[OpportunityRanking]) -> Dict:
        """Analyze correlations between opportunities"""
        
        if len(opportunities) < 2:
            return {'correlation_matrix': {}, 'average_correlation': 0, 'risk_level': 'low'}
        
        # Simplified correlation analysis
        tickers = [opp.ticker for opp in opportunities[:5]]  # Top 5
        
        # In production, would calculate actual correlation matrix
        # For now, use sector-based approximation
        correlation_matrix = {}
        for i, ticker1 in enumerate(tickers):
            for j, ticker2 in enumerate(tickers):
                if i < j:
                    sector1 = opportunities[i].sector
                    sector2 = opportunities[j].sector
                    
                    if sector1 == sector2:
                        correlation = 0.7  # High correlation in same sector
                    else:
                        correlation = 0.3  # Lower correlation across sectors
                    
                    correlation_matrix[f"{ticker1}-{ticker2}"] = correlation
        
        # Calculate average correlation
        if correlation_matrix:
            avg_correlation = np.mean(list(correlation_matrix.values()))
        else:
            avg_correlation = 0
        
        # Determine risk level
        if avg_correlation < 0.4:
            risk_level = 'low'
        elif avg_correlation < 0.6:
            risk_level = 'medium'
        else:
            risk_level = 'high'
        
        return {
            'correlation_matrix': correlation_matrix,
            'average_correlation': round(avg_correlation, 3),
            'risk_level': risk_level,
            'diversification_score': round((1 - avg_correlation) * 10, 1)
        }
    
    def _create_recommendation_summary(self, opportunities: List[OpportunityRanking],
                                      themes: List[PortfolioTheme],
                                      risk_assessment: Dict) -> str:
        """Create executive summary of recommendations"""
        
        if not opportunities:
            return "No actionable opportunities identified at this time."
        
        top_picks = opportunities[:3]
        picks_str = ", ".join([opp.ticker for opp in top_picks])
        
        theme_str = ""
        if themes:
            theme_str = f" Key themes: {themes[0].theme_name}"
        
        risk_str = f"Portfolio risk: {risk_assessment['risk_level']}"
        
        summary = (
            f"Top opportunities: {picks_str}. "
            f"{theme_str}. "
            f"{risk_str}. "
            f"Recommended action: Proceed with top-ranked positions while maintaining diversification."
        )
        
        return summary
    
    def _create_empty_analysis(self, reason: str) -> Dict:
        """Create empty analysis structure"""
        return {
            'synthesis_timestamp': datetime.now().isoformat(),
            'ranked_opportunities': [],
            'strategic_themes': [],
            'risk_assessment': {'risk_level': 'unknown', 'reason': reason},
            'time_horizon_allocation': {},
            'correlation_analysis': {},
            'market_regime': 'unknown',
            'recommendation_summary': reason
        }


class MarketContextAnalyzer:
    """Analyzes market context for strategic positioning"""
    
    def __init__(self, alpaca_provider):
        self.alpaca = alpaca_provider
        self.logger = logging.getLogger('market_context')
    
    async def get_market_context(self) -> Dict:
        """Get comprehensive market context"""
        
        try:
            # Get market indices
            indices = await self._get_market_indices()
            
            # Analyze momentum
            momentum = self._analyze_market_momentum(indices)
            
            # Analyze volatility
            volatility = self._analyze_volatility(indices)
            
            # Analyze sector rotation
            sector_rotation = await self._analyze_sector_rotation()
            
            # Determine regime
            regime = self._classify_market_regime(momentum, volatility)
            
            return {
                'timestamp': datetime.now().isoformat(),
                'market_momentum': momentum,
                'volatility_regime': volatility,
                'sector_rotation': sector_rotation,
                'risk_sentiment': self._assess_risk_sentiment(volatility),
                'regime_classification': regime,
                'positioning_recommendations': self._get_positioning_recommendations(regime)
            }
            
        except Exception as e:
            self.logger.error(f"Market context analysis failed: {str(e)}")
            return self._create_neutral_context()
    
    async def _get_market_indices(self) -> Dict:
        """Get major market indices data"""
        
        indices = {}
        symbols = ['SPY', 'QQQ', 'IWM', 'VIX']  # S&P, NASDAQ, Russell, Volatility
        
        for symbol in symbols:
            try:
                data = await self.alpaca.get_market_data([symbol], timeframe='1Day', limit=20)
                if symbol in data:
                    indices[symbol] = data[symbol]
            except Exception as e:
                self.logger.warning(f"Failed to get data for {symbol}: {str(e)}")
        
        return indices
    
    def _analyze_market_momentum(self, indices: Dict) -> Dict:
        """Analyze overall market momentum"""
        
        momentum = {
            'trend_direction': 'neutral',
            'momentum_strength': 'moderate',
            'breadth': 'mixed'
        }
        
        if 'SPY' in indices and len(indices['SPY']) > 10:
            spy_data = indices['SPY']
            
            # Calculate short-term momentum (5-day)
            recent_return = (spy_data[-1]['close'] - spy_data[-5]['close']) / spy_data[-5]['close']
            
            # Calculate medium-term momentum (20-day)
            if len(spy_data) >= 20:
                medium_return = (spy_data[-1]['close'] - spy_data[-20]['close']) / spy_data[-20]['close']
            else:
                medium_return = recent_return
            
            # Determine trend
            if recent_return > 0.02 and medium_return > 0.03:
                momentum['trend_direction'] = 'bullish'
                momentum['momentum_strength'] = 'strong'
            elif recent_return > 0.01:
                momentum['trend_direction'] = 'bullish'
                momentum['momentum_strength'] = 'moderate'
            elif recent_return < -0.02 and medium_return < -0.03:
                momentum['trend_direction'] = 'bearish'
                momentum['momentum_strength'] = 'strong'
            elif recent_return < -0.01:
                momentum['trend_direction'] = 'bearish'
                momentum['momentum_strength'] = 'moderate'
        
        return momentum
    
    def _analyze_volatility(self, indices: Dict) -> Dict:
        """Analyze market volatility regime"""
        
        volatility = {
            'regime': 'normal',
            'trend': 'stable',
            'vix_level': 20
        }
        
        if 'VIX' in indices and indices['VIX']:
            vix_data = indices['VIX']
            current_vix = vix_data[-1]['close']
            
            volatility['vix_level'] = current_vix
            
            # Classify volatility regime
            if current_vix < 15:
                volatility['regime'] = 'low'
            elif current_vix < 20:
                volatility['regime'] = 'normal'
            elif current_vix < 30:
                volatility['regime'] = 'elevated'
            elif current_vix < 40:
                volatility['regime'] = 'high'
            else:
                volatility['regime'] = 'extreme'
            
            # Analyze trend
            if len(vix_data) > 5:
                vix_change = (current_vix - vix_data[-5]['close']) / vix_data[-5]['close']
                
                if vix_change > 0.2:
                    volatility['trend'] = 'rising'
                elif vix_change < -0.2:
                    volatility['trend'] = 'falling'
                else:
                    volatility['trend'] = 'stable'
        
        return volatility
    
    async def _analyze_sector_rotation(self) -> Dict:
        """Analyze sector rotation patterns"""
        
        sectors = {
            'XLK': 'Technology',
            'XLF': 'Financials',
            'XLV': 'Healthcare',
            'XLE': 'Energy',
            'XLI': 'Industrials',
            'XLY': 'Consumer Discretionary',
            'XLP': 'Consumer Staples',
            'XLU': 'Utilities'
        }
        
        sector_performance = {}
        
        for symbol, name in sectors.items():
            try:
                data = await self.alpaca.get_market_data([symbol], timeframe='1Day', limit=20)
                if symbol in data and len(data[symbol]) > 5:
                    recent_perf = (data[symbol][-1]['close'] - data[symbol][-5]['close']) / data[symbol][-5]['close']
                    sector_performance[name] = {
                        'performance': round(recent_perf * 100, 2),
                        'momentum': 'positive' if recent_perf > 0 else 'negative'
                    }
            except Exception:
                pass
        
        # Identify rotation
        if sector_performance:
            sorted_sectors = sorted(sector_performance.items(), 
                                  key=lambda x: x[1]['performance'], 
                                  reverse=True)
            
            return {
                'leading_sectors': [s[0] for s in sorted_sectors[:3]],
                'lagging_sectors': [s[0] for s in sorted_sectors[-3:]],
                'rotation_active': True if len(sector_performance) > 5 else False,
                'sector_performance': sector_performance
            }
        
        return {'rotation_active': False}
    
    def _assess_risk_sentiment(self, volatility: Dict) -> Dict:
        """Assess overall risk sentiment"""
        
        vix_level = volatility.get('vix_level', 20)
        
        if vix_level < 15:
            sentiment = 'risk_on'
            confidence = 8
            risk_score = 3
        elif vix_level < 25:
            sentiment = 'neutral'
            confidence = 6
            risk_score = 5
        elif vix_level < 35:
            sentiment = 'risk_off'
            confidence = 7
            risk_score = 7
        else:
            sentiment = 'extreme_fear'
            confidence = 9
            risk_score = 9
        
        return {
            'sentiment': sentiment,
            'confidence': confidence,
            'risk_score': risk_score
        }
    
    def _classify_market_regime(self, momentum: Dict, volatility: Dict) -> Dict:
        """Classify current market regime"""
        
        trend = momentum.get('trend_direction', 'neutral')
        vol_regime = volatility.get('regime', 'normal')
        
        # Determine regime
        if trend == 'bullish' and vol_regime in ['low', 'normal']:
            regime = 'trending_bull'
            confidence = 8
        elif trend == 'bullish' and vol_regime in ['elevated', 'high']:
            regime = 'volatile_bull'
            confidence = 6
        elif trend == 'bearish' and vol_regime in ['elevated', 'high', 'extreme']:
            regime = 'crisis'
            confidence = 8
        elif trend == 'bearish':
            regime = 'correction'
            confidence = 7
        else:
            regime = 'sideways_market'
            confidence = 5
        
        return {
            'regime': regime,
            'confidence': confidence,
            'drivers': self._identify_regime_drivers(momentum, volatility, 
                                                    self._assess_risk_sentiment(volatility))
        }
    
    def _get_positioning_recommendations(self, regime: Dict) -> Dict:
        """Get positioning recommendations based on regime"""
        
        regime_type = regime.get('regime', 'sideways_market')
        
        recommendations = {
            'overall_posture': 'neutral',
            'cash_allocation': 20,
            'risk_level': 'medium',
            'position_sizing': 'normal',
            'sector_focus': []
        }
        
        if regime_type == 'trending_bull':
            recommendations.update({
                'overall_posture': 'aggressive',
                'cash_allocation': 10,
                'risk_level': 'medium-high',
                'position_sizing': 'increased',
                'sector_focus': ['Technology', 'Consumer Discretionary']
            })
        elif regime_type == 'crisis':
            recommendations.update({
                'overall_posture': 'defensive',
                'cash_allocation': 40,
                'risk_level': 'low',
                'position_sizing': 'reduced',
                'sector_focus': ['Utilities', 'Consumer Staples', 'Healthcare']
            })
        elif regime_type == 'correction':
            recommendations.update({
                'overall_posture': 'cautious',
                'cash_allocation': 30,
                'risk_level': 'medium-low',
                'position_sizing': 'normal',
                'sector_focus': ['Healthcare', 'Consumer Staples']
            })
        
        return recommendations
    
    def _identify_regime_drivers(self, momentum: Dict, volatility: Dict, 
                                risk_sentiment: Dict) -> List[str]:
        """Identify key drivers of current market regime"""
        
        drivers = []
        
        if momentum['momentum_strength'] == 'strong':
            drivers.append(f"strong_{momentum['trend_direction']}_momentum")
        
        if volatility['regime'] in ['low', 'high', 'extreme']:
            drivers.append(f"{volatility['regime']}_volatility")
        
        if risk_sentiment['confidence'] >= 7:
            drivers.append(f"clear_{risk_sentiment['sentiment']}_sentiment")
        
        return drivers
    
    def _create_neutral_context(self) -> Dict:
        """Create neutral market context for error cases"""
        
        return {
            'timestamp': datetime.now().isoformat(),
            'market_momentum': {'trend_direction': 'neutral', 'momentum_strength': 'moderate'},
            'volatility_regime': {'regime': 'normal', 'trend': 'stable'},
            'sector_rotation': {'rotation_active': False},
            'risk_sentiment': {'sentiment': 'neutral', 'risk_score': 5},
            'regime_classification': {'regime': 'sideways_market', 'confidence': 3},
            'positioning_recommendations': {
                'overall_posture': 'neutral',
                'cash_allocation': 20,
                'risk_level': 'medium'
            }
        }


class SeniorResearchAnalyst:
    """
    Senior Research Analyst Agent
    
    Synthesizes multiple junior analyst reports into strategic portfolio recommendations.
    Operates at the portfolio level, ranking opportunities and balancing risk across time horizons.
    """
    
    def __init__(self, llm_provider, alpaca_provider, config):
        """Initialize the Senior Research Analyst"""
        
        self.agent_name = "senior_research_analyst"
        self.agent_id = str(uuid.uuid4())
        
        # Core dependencies
        self.llm = llm_provider
        self.alpaca = alpaca_provider
        self.config = config
        
        # Setup logging
        self.logger = self._setup_logging()
        
        # Initialize analysis engines
        self.strategic_engine = StrategicAnalysisEngine()
        self.market_analyzer = MarketContextAnalyzer(alpaca_provider)
        
        # Performance tracking
        self.synthesis_cache = {}
        self.performance_metrics = {
            "total_syntheses": 0,
            "successful_syntheses": 0,
            "failed_syntheses": 0,
            "average_processing_time": 0.0,
            "cache_hits": 0,
            "last_activity": None
        }
        
        self.logger.info(f"✅ Senior Research Analyst initialized with ID: {self.agent_id}")
    
    def _setup_logging(self) -> logging.Logger:
        """Setup agent-specific logging"""
        logger = logging.getLogger(f"agent.{self.agent_name}")
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    async def synthesize_reports(self, junior_reports: List[Dict], 
                                current_portfolio: Optional[Dict] = None) -> Dict:
        """
        Synthesize multiple junior analyst reports into strategic recommendations
        
        Args:
            junior_reports: List of junior analyst analysis reports
            current_portfolio: Current portfolio positions and metrics
        
        Returns:
            Dict with strategic analysis and ranked opportunities
        """
        
        start_time = datetime.now()
        
        try:
            # Input validation
            if not junior_reports:
                return self._create_error_response("No junior analyst reports provided")
            
            self.logger.info(f"Synthesizing {len(junior_reports)} junior analyst reports")
            
            # Get market context
            market_context = await self.market_analyzer.get_market_context()
            
            # Prepare portfolio context
            portfolio_context = current_portfolio or {}
            
            # Perform strategic analysis
            strategic_analysis = self.strategic_engine.synthesize_junior_reports(
                junior_reports, market_context, portfolio_context
            )
            
            # Enhance with LLM insights
            enhanced_analysis = await self._enhance_with_llm_insights(
                strategic_analysis, junior_reports, market_context
            )
            
            # Generate markdown report
            markdown_report = self._generate_markdown_report(enhanced_analysis)
            
            # Track performance
            processing_time = (datetime.now() - start_time).total_seconds()
            self._update_performance_metrics(True, processing_time)
            
            return {
                "status": "success",
                "timestamp": datetime.now().isoformat(),
                "strategic_analysis": enhanced_analysis,
                "markdown_report": markdown_report,
                "metadata": {
                    "reports_synthesized": len(junior_reports),
                    "processing_time": round(processing_time, 2),
                    "market_regime": market_context.get('regime_classification', {}).get('regime', 'unknown'),
                    "agent_id": self.agent_id
                }
            }
            
        except Exception as e:
            self.logger.error(f"Synthesis failed: {str(e)}")
            self._update_performance_metrics(False, 0)
            return self._create_error_response(str(e))
    
    async def _enhance_with_llm_insights(self, strategic_analysis: Dict, 
                                        junior_reports: List[Dict], 
                                        market_context: Dict) -> Dict:
        """Enhance strategic analysis with Claude AI insights"""
        
        try:
            # Prepare context for LLM
            llm_context = {
                "ranked_opportunities": strategic_analysis.get('ranked_opportunities', [])[:5],
                "strategic_themes": strategic_analysis.get('strategic_themes', []),
                "risk_assessment": strategic_analysis.get('risk_assessment', {}),
                "market_regime": market_context.get('regime_classification', {}),
                "junior_report_summary": self._summarize_junior_reports(junior_reports)
            }
            
            # Create prompt for strategic insights
            prompt = self._create_strategic_prompt(llm_context)
            
            # Get LLM response
            llm_response = await self.llm.generate_analysis(prompt, llm_context)
            
            # Parse and integrate LLM insights
            if isinstance(llm_response, dict) and not llm_response.get('error'):
                strategic_analysis['llm_insights'] = llm_response
                strategic_analysis['executive_summary'] = llm_response.get(
                    'executive_summary', 
                    strategic_analysis.get('recommendation_summary', '')
                )
                strategic_analysis['key_decisions'] = llm_response.get('key_decisions', [])
            
            return strategic_analysis
            
        except Exception as e:
            self.logger.warning(f"LLM enhancement failed: {str(e)}")
            return strategic_analysis
    
    def _create_strategic_prompt(self, context: Dict) -> str:
        """Create prompt for strategic LLM analysis"""
        
        return """
        As a Senior Research Analyst, synthesize the following analysis into strategic portfolio recommendations:
        
        CONTEXT:
        - Market Regime: {market_regime}
        - Top Opportunities: {opportunities}
        - Strategic Themes: {themes}
        - Risk Assessment: {risk}
        
        Provide:
        1. Executive summary (2-3 sentences)
        2. Key portfolio decisions (top 3-5 actions)
        3. Strategic positioning advice
        4. Risk management priorities
        5. Time horizon recommendations
        
        Format as structured JSON with keys:
        - executive_summary
        - key_decisions (list)
        - positioning_advice
        - risk_priorities (list)
        - time_horizon_strategy
        """.format(
            market_regime=context.get('market_regime', {}),
            opportunities=context.get('ranked_opportunities', [])[:3],
            themes=context.get('strategic_themes', []),
            risk=context.get('risk_assessment', {})
        )
    
    def _summarize_junior_reports(self, reports: List[Dict]) -> Dict:
        """Summarize junior analyst reports"""
        
        summary = {
            "total_reports": len(reports),
            "bullish_count": sum(1 for r in reports if r.get('recommendation') in ['BUY', 'STRONG_BUY']),
            "bearish_count": sum(1 for r in reports if r.get('recommendation') in ['SELL', 'STRONG_SELL']),
            "average_confidence": np.mean([r.get('confidence', 5) for r in reports]),
            "sectors_covered": list(set(r.get('sector', 'Unknown') for r in reports))
        }
        
        return summary
    
    def _generate_markdown_report(self, analysis: Dict) -> str:
        """Generate comprehensive markdown report"""
        
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        report = f"""# Senior Research Analyst Report
Generated: {timestamp}

## Executive Summary
{analysis.get('executive_summary', 'No executive summary available.')}

## Market Context
- **Market Regime**: {analysis.get('market_regime', 'Unknown')}
- **Risk Level**: {analysis.get('risk_assessment', {}).get('risk_level', 'Unknown')}
- **Recommendation**: {analysis.get('recommendation_summary', 'No recommendations available.')}

## Top Opportunities

"""
        
        # Add ranked opportunities
        opportunities = analysis.get('ranked_opportunities', [])
        if opportunities:
            report += "| Ticker | Score | Expected Return | Risk | Time Horizon | Thesis |\n"
            report += "|--------|-------|-----------------|------|--------------|--------|\n"
            
            for opp in opportunities[:5]:
                report += f"| **{opp.ticker}** | {opp.risk_adjusted_score:.2f} | {opp.expected_return:.1f}% | {opp.risk_level} | {opp.time_horizon} | {opp.thesis_summary[:50]}... |\n"
        else:
            report += "*No opportunities identified*\n"
        
        # Add strategic themes
        report += "\n## Strategic Themes\n\n"
        themes = analysis.get('strategic_themes', [])
        if themes:
            for i, theme in enumerate(themes[:3], 1):
                report += f"### {i}. {theme.theme_name}\n"
                report += f"- **Confidence**: {theme.confidence:.1%}\n"
                report += f"- **Tickers**: {', '.join(theme.supporting_tickers)}\n"
                report += f"- **Allocation Suggestion**: {theme.allocation_suggestion:.1%}\n\n"
        else:
            report += "*No strategic themes identified*\n"
        
        # Add risk assessment
        report += "\n## Risk Assessment\n\n"
        risk = analysis.get('risk_assessment', {})
        report += f"- **Overall Risk Level**: {risk.get('risk_level', 'Unknown')}\n"
        report += f"- **Risk Score**: {risk.get('risk_score', 0):.1f}/10\n"
        
        if risk.get('risk_factors'):
            report += "\n### Risk Factors\n"
            for factor in risk['risk_factors']:
                report += f"- {factor}\n"
        
        if risk.get('recommendations'):
            report += "\n### Risk Management Recommendations\n"
            for rec in risk['recommendations']:
                report += f"- {rec}\n"
        
        # Add time horizon allocation
        report += "\n## Time Horizon Allocation\n\n"
        time_allocation = analysis.get('time_horizon_allocation', {})
        if time_allocation.get('percentages'):
            report += "| Horizon | Allocation | Tickers |\n"
            report += "|---------|------------|----------|\n"
            
            for horizon in ['short', 'medium', 'long']:
                pct = time_allocation['percentages'].get(horizon, 0)
                tickers = ', '.join(time_allocation.get('allocation', {}).get(horizon, []))
                report += f"| {horizon.capitalize()} | {pct:.1%} | {tickers or 'None'} |\n"
        
        # Add LLM insights if available
        if analysis.get('llm_insights'):
            report += "\n## Strategic Insights\n\n"
            insights = analysis['llm_insights']
            
            if insights.get('key_decisions'):
                report += "### Key Portfolio Decisions\n"
                for decision in insights['key_decisions']:
                    report += f"- {decision}\n"
            
            if insights.get('positioning_advice'):
                report += f"\n### Positioning Advice\n{insights['positioning_advice']}\n"
        
        report += "\n---\n*Report generated by Senior Research Analyst Agent*"
        
        return report
    
    def _update_performance_metrics(self, success: bool, processing_time: float):
        """Update agent performance metrics"""
        
        self.performance_metrics['total_syntheses'] += 1
        
        if success:
            self.performance_metrics['successful_syntheses'] += 1
        else:
            self.performance_metrics['failed_syntheses'] += 1
        
        # Update average processing time
        total = self.performance_metrics['total_syntheses']
        current_avg = self.performance_metrics['average_processing_time']
        new_avg = ((current_avg * (total - 1)) + processing_time) / total
        self.performance_metrics['average_processing_time'] = round(new_avg, 2)
        
        self.performance_metrics['last_activity'] = datetime.now().isoformat()
    
    def _create_error_response(self, error_message: str) -> Dict:
        """Create standardized error response"""
        
        return {
            "status": "error",
            "timestamp": datetime.now().isoformat(),
            "error": error_message,
            "strategic_analysis": {},
            "markdown_report": f"# Error\n\nAnalysis failed: {error_message}",
            "metadata": {
                "agent_id": self.agent_id,
                "error_timestamp": datetime.now().isoformat()
            }
        }
    
    def get_performance_metrics(self) -> Dict:
        """Get current performance metrics"""
        return self.performance_metrics.copy()


# Factory function for easy initialization
def create_senior_analyst(llm_provider, alpaca_provider, config):
    """Factory function to create Senior Research Analyst"""
    return SeniorResearchAnalyst(llm_provider, alpaca_provider, config)