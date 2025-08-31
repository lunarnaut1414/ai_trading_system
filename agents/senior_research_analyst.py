# agents/senior_research_analyst.py
"""
Enhanced Senior Research Analyst Agent - Complete Refactored Implementation
Optimized for macOS M2 Max with Claude AI integration

This refactored version includes:
- Better integration with Junior Analyst output format
- Shared market context management
- Unified risk assessment
- Feedback mechanism for Junior Analysts
- Enhanced caching and metadata tracking
- Parallel processing capabilities
"""

import hashlib
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

# Import shared components from junior_research_analyst
from agents.junior_research_analyst import (
    MarketContextManager,
    UnifiedRiskAssessment,
    IntelligentCacheManager,
    AnalysisMetadataTracker,
    ConvictionLevel,
    TimeHorizon,
    RiskLevel,
    PositionSize
)


# ========================================================================================
# ENUMS AND DATA CLASSES
# ========================================================================================

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
    ROTATION = "rotation"


class AllocationStrategy(Enum):
    AGGRESSIVE = "aggressive"      # 80% equity, high risk
    MODERATE = "moderate"          # 60% equity, balanced
    CONSERVATIVE = "conservative"  # 40% equity, low risk
    DEFENSIVE = "defensive"        # 20% equity, capital preservation


@dataclass
class OpportunityRanking:
    """Enhanced data class for ranked opportunities"""
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
    position_weight: float
    execution_priority: int
    junior_analyst_id: str
    analysis_chain_id: str


@dataclass
class PortfolioTheme:
    """Enhanced data class for portfolio themes"""
    theme_name: str
    theme_type: str
    confidence: float
    supporting_tickers: List[str]
    time_horizon: str
    risk_factors: List[str]
    allocation_suggestion: float
    expected_impact: str
    market_conditions: List[str]


@dataclass
class RiskAssessment:
    """Portfolio-level risk assessment"""
    overall_risk_score: float
    risk_level: str
    concentration_risk: float
    correlation_risk: float
    market_risk: float
    liquidity_risk: float
    key_risk_factors: List[str]
    risk_mitigation: List[str]


# ========================================================================================
# STRATEGIC ANALYSIS ENGINE
# ========================================================================================

class StrategicAnalysisEngine:
    """
    Enhanced strategic analysis engine for the Senior Research Analyst
    Synthesizes multiple junior analyst reports into portfolio strategy
    """
    
    def __init__(self):
        self.logger = logging.getLogger('strategic_analysis')
        
        # Enhanced scoring weights
        self.scoring_weights = {
            'conviction': 0.20,
            'risk_reward': 0.15,
            'catalyst_strength': 0.15,
            'technical_score': 0.10,
            'liquidity': 0.10,
            'correlation_bonus': 0.10,
            'sector_momentum': 0.05,
            'market_alignment': 0.10,
            'time_horizon_fit': 0.05
        }
        
        # Risk thresholds
        self.risk_thresholds = {
            'max_correlation': 0.70,
            'max_sector_concentration': 0.25,
            'min_liquidity_score': 6.0,
            'max_portfolio_beta': 1.30,
            'max_single_position': 0.05,
            'min_diversification': 10
        }
        
        # Time horizon allocation targets
        self.time_horizon_targets = {
            'aggressive': {'short': 0.4, 'medium': 0.4, 'long': 0.2},
            'moderate': {'short': 0.2, 'medium': 0.5, 'long': 0.3},
            'conservative': {'short': 0.1, 'medium': 0.3, 'long': 0.6}
        }
    
    def synthesize_junior_reports(self, junior_reports: List[Dict], 
                                 market_context: Dict, 
                                 portfolio_context: Dict) -> Dict:
        """Enhanced synthesis of multiple junior analyst reports"""
        
        try:
            if not junior_reports:
                return self._create_empty_analysis("No junior reports provided")
            
            # Filter and validate reports with enhanced validation
            valid_reports = self._filter_and_validate_reports(junior_reports)
            if not valid_reports:
                return self._create_empty_analysis("No valid junior reports")
            
            # Calculate opportunity rankings with enhanced scoring
            ranked_opportunities = self._rank_opportunities_enhanced(
                valid_reports, market_context, portfolio_context
            )
            
            # Identify strategic themes with market context
            strategic_themes = self._identify_strategic_themes_enhanced(
                valid_reports, market_context
            )
            
            # Assess portfolio-level risk with correlations
            risk_assessment = self._assess_portfolio_risk_enhanced(
                ranked_opportunities, portfolio_context
            )
            
            # Balance time horizons based on market regime
            time_horizon_allocation = self._balance_time_horizons_enhanced(
                ranked_opportunities, market_context
            )
            
            # Perform correlation analysis
            correlation_analysis = self._analyze_correlations_enhanced(
                ranked_opportunities, portfolio_context
            )
            
            # Generate execution priorities
            execution_plan = self._generate_execution_plan(
                ranked_opportunities, risk_assessment, portfolio_context
            )
            
            return {
                'status': 'success',
                'timestamp': datetime.now().isoformat(),
                'ranked_opportunities': ranked_opportunities,
                'strategic_themes': strategic_themes,
                'risk_assessment': risk_assessment,
                'time_horizon_allocation': time_horizon_allocation,
                'correlation_analysis': correlation_analysis,
                'execution_plan': execution_plan,
                'market_regime': market_context.get('regime', 'neutral'),
                'reports_processed': len(valid_reports),
                'confidence_score': self._calculate_overall_confidence(ranked_opportunities)
            }
            
        except Exception as e:
            self.logger.error(f"Synthesis failed: {str(e)}")
            return self._create_empty_analysis(str(e))
    
    def _filter_and_validate_reports(self, reports: List[Dict]) -> List[Dict]:
        """Enhanced validation of junior analyst reports"""
        valid_reports = []
        
        for report in reports:
            # Check for required fields from enhanced Junior Analyst
            required_fields = [
                'ticker', 'recommendation', 'confidence', 'conviction_level',
                'expected_return', 'risk_assessment', 'position_weight_percent',
                'liquidity_score', 'catalyst_strength', 'technical_score'
            ]
            
            if all(field in report for field in required_fields):
                # Additional validation
                if report.get('analysis_status') == 'success':
                    if report.get('confidence', 0) >= 3:  # Minimum confidence threshold
                        valid_reports.append(report)
                    else:
                        self.logger.debug(f"Skipping {report.get('ticker')} - low confidence")
                else:
                    self.logger.debug(f"Skipping {report.get('ticker')} - analysis failed")
            else:
                missing = [f for f in required_fields if f not in report]
                self.logger.warning(f"Report missing fields: {missing}")
        
        return valid_reports
    
    def _rank_opportunities_enhanced(self, reports: List[Dict], 
                                    market_context: Dict, 
                                    portfolio_context: Dict) -> List[OpportunityRanking]:
        """Enhanced opportunity ranking with multiple factors"""
        
        rankings = []
        market_regime = market_context.get('regime', 'neutral')
        
        for report in reports:
            # Calculate multi-factor score
            score_components = {
                'conviction': report.get('conviction_level', 3) / 5.0 * 10,
                'risk_reward': min(report.get('risk_reward_ratio', 1.0) / 3.0 * 10, 10),
                'catalyst_strength': report.get('catalyst_strength', 5),
                'technical_score': report.get('technical_score', 5),
                'liquidity': report.get('liquidity_score', 5),
                'correlation_bonus': self._calculate_correlation_bonus(
                    report, portfolio_context
                ),
                'sector_momentum': self._calculate_sector_momentum(
                    report, market_context
                ),
                'market_alignment': self._calculate_market_alignment(
                    report, market_regime
                ),
                'time_horizon_fit': self._calculate_time_horizon_fit(
                    report, portfolio_context
                )
            }
            
            # Calculate weighted score
            total_score = sum(
                score_components[factor] * self.scoring_weights[factor]
                for factor in score_components
            )
            
            # Adjust for risk
            risk_adjustment = self._calculate_risk_adjustment(report)
            risk_adjusted_score = total_score * risk_adjustment
            
            ranking = OpportunityRanking(
                ticker=report['ticker'],
                conviction_score=report.get('conviction_level', 3),
                risk_adjusted_score=round(risk_adjusted_score, 2),
                time_horizon=report.get('time_horizon', 'medium'),
                expected_return=report.get('expected_return', 0),
                risk_level=report.get('risk_level', 'medium'),
                sector=report.get('sector', 'Unknown'),
                market_cap=report.get('market_cap', 'Unknown'),
                correlation_risk=self._assess_correlation_risk(report, portfolio_context),
                liquidity_score=report.get('liquidity_score', 5),
                catalyst_strength=report.get('catalyst_strength', 5),
                thesis_summary=report.get('thesis_summary', ''),
                key_risks=report.get('key_risks', []),
                position_weight=report.get('position_weight_percent', 3.5),
                execution_priority=0,  # Will be set later
                junior_analyst_id=report.get('agent_id', ''),
                analysis_chain_id=report.get('analysis_chain', {}).get('chain_id', '')
            )
            
            rankings.append(ranking)
        
        # Sort by risk-adjusted score
        rankings.sort(key=lambda x: x.risk_adjusted_score, reverse=True)
        
        # Set execution priorities
        for i, ranking in enumerate(rankings):
            ranking.execution_priority = i + 1
        
        # Return top 10
        return rankings[:10]
    
    def _identify_strategic_themes_enhanced(self, reports: List[Dict], 
                                           market_context: Dict) -> List[PortfolioTheme]:
        """Enhanced theme identification with market context"""
        
        themes = []
        
        # Sector concentration analysis
        sector_counts = defaultdict(list)
        for report in reports:
            sector = report.get('sector', 'Unknown')
            if report.get('recommendation') in ['BUY', 'STRONG_BUY']:
                sector_counts[sector].append(report['ticker'])
        
        # Identify sector themes
        for sector, tickers in sector_counts.items():
            if len(tickers) >= 2:  # At least 2 stocks in sector
                theme = PortfolioTheme(
                    theme_name=f"{sector} Concentration",
                    theme_type=SectorTheme.MOMENTUM.value,
                    confidence=min(len(tickers) / 3.0 * 10, 10),
                    supporting_tickers=tickers[:5],
                    time_horizon=TimeHorizon.MEDIUM_TERM.value,
                    risk_factors=[f"Sector concentration in {sector}"],
                    allocation_suggestion=min(len(tickers) * 5, 25),
                    expected_impact="Potential sector outperformance",
                    market_conditions=[market_context.get('regime', 'neutral')]
                )
                themes.append(theme)
        
        # Market regime themes
        regime = market_context.get('regime', 'neutral')
        if regime == 'risk_on':
            growth_stocks = [
                r['ticker'] for r in reports 
                if r.get('sector') in ['Technology', 'Consumer Discretionary']
                and r.get('recommendation') in ['BUY', 'STRONG_BUY']
            ]
            
            if growth_stocks:
                themes.append(PortfolioTheme(
                    theme_name="Growth Momentum",
                    theme_type=SectorTheme.GROWTH.value,
                    confidence=8.0,
                    supporting_tickers=growth_stocks[:5],
                    time_horizon=TimeHorizon.MEDIUM_TERM.value,
                    risk_factors=["Growth stock volatility"],
                    allocation_suggestion=30,
                    expected_impact="Capture risk-on momentum",
                    market_conditions=["risk_on", "bull_market"]
                ))
        
        elif regime == 'risk_off':
            defensive_stocks = [
                r['ticker'] for r in reports 
                if r.get('sector') in ['Utilities', 'Consumer Staples', 'Healthcare']
                and r.get('recommendation') in ['BUY', 'HOLD']
            ]
            
            if defensive_stocks:
                themes.append(PortfolioTheme(
                    theme_name="Defensive Positioning",
                    theme_type=SectorTheme.DEFENSIVE.value,
                    confidence=8.0,
                    supporting_tickers=defensive_stocks[:5],
                    time_horizon=TimeHorizon.SHORT_TERM.value,
                    risk_factors=["Potential underperformance in recovery"],
                    allocation_suggestion=40,
                    expected_impact="Capital preservation",
                    market_conditions=["risk_off", "bear_market"]
                ))
        
        return themes
    
    def _assess_portfolio_risk_enhanced(self, opportunities: List[OpportunityRanking],
                                       portfolio_context: Dict) -> RiskAssessment:
        """Enhanced portfolio risk assessment"""
        
        if not opportunities:
            return self._create_default_risk_assessment()
        
        # Calculate various risk metrics
        concentration_risk = self._calculate_concentration_risk(opportunities)
        correlation_risk = self._calculate_portfolio_correlation_risk(opportunities)
        market_risk = self._calculate_market_risk(opportunities)
        liquidity_risk = self._calculate_liquidity_risk(opportunities)
        
        # Overall risk score (weighted average)
        risk_weights = {
            'concentration': 0.25,
            'correlation': 0.30,
            'market': 0.25,
            'liquidity': 0.20
        }
        
        overall_score = (
            concentration_risk * risk_weights['concentration'] +
            correlation_risk * risk_weights['correlation'] +
            market_risk * risk_weights['market'] +
            liquidity_risk * risk_weights['liquidity']
        )
        
        # Determine risk level
        if overall_score <= 3:
            risk_level = RiskLevel.LOW.value
        elif overall_score <= 5:
            risk_level = RiskLevel.MEDIUM.value
        elif overall_score <= 7:
            risk_level = RiskLevel.HIGH.value
        else:
            risk_level = RiskLevel.VERY_HIGH.value
        
        # Identify key risk factors
        key_risks = []
        if concentration_risk > 6:
            key_risks.append("High concentration risk")
        if correlation_risk > 6:
            key_risks.append("High correlation among positions")
        if market_risk > 6:
            key_risks.append("High market sensitivity")
        if liquidity_risk > 6:
            key_risks.append("Liquidity concerns")
        
        # Risk mitigation suggestions
        mitigation = []
        if concentration_risk > 6:
            mitigation.append("Diversify across more sectors")
        if correlation_risk > 6:
            mitigation.append("Add uncorrelated assets")
        if market_risk > 6:
            mitigation.append("Consider hedging strategies")
        if liquidity_risk > 6:
            mitigation.append("Focus on more liquid positions")
        
        return RiskAssessment(
            overall_risk_score=round(overall_score, 2),
            risk_level=risk_level,
            concentration_risk=round(concentration_risk, 2),
            correlation_risk=round(correlation_risk, 2),
            market_risk=round(market_risk, 2),
            liquidity_risk=round(liquidity_risk, 2),
            key_risk_factors=key_risks,
            risk_mitigation=mitigation
        )
    
    def _balance_time_horizons_enhanced(self, opportunities: List[OpportunityRanking],
                                       market_context: Dict) -> Dict:
        """Enhanced time horizon balancing based on market regime"""
        
        if not opportunities:
            return {'allocation': {}, 'recommendations': []}
        
        # Count opportunities by time horizon
        horizon_counts = defaultdict(list)
        for opp in opportunities:
            horizon_counts[opp.time_horizon].append(opp.ticker)
        
        # Calculate current allocation
        total = len(opportunities)
        current_allocation = {
            TimeHorizon.SHORT_TERM.value: len(horizon_counts.get(TimeHorizon.SHORT_TERM.value, [])) / total,
            TimeHorizon.MEDIUM_TERM.value: len(horizon_counts.get(TimeHorizon.MEDIUM_TERM.value, [])) / total,
            TimeHorizon.LONG_TERM.value: len(horizon_counts.get(TimeHorizon.LONG_TERM.value, [])) / total
        }
        
        # Determine target allocation based on market regime
        regime = market_context.get('regime', 'neutral')
        if regime == 'risk_on':
            target_allocation = self.time_horizon_targets['aggressive']
        elif regime == 'risk_off':
            target_allocation = self.time_horizon_targets['conservative']
        else:
            target_allocation = self.time_horizon_targets['moderate']
        
        # Generate recommendations
        recommendations = []
        for horizon, target_pct in target_allocation.items():
            current_pct = current_allocation.get(horizon, 0)
            if current_pct < target_pct - 0.1:
                recommendations.append(f"Increase {horizon} exposure by {(target_pct - current_pct)*100:.0f}%")
            elif current_pct > target_pct + 0.1:
                recommendations.append(f"Reduce {horizon} exposure by {(current_pct - target_pct)*100:.0f}%")
        
        return {
            'current_allocation': current_allocation,
            'target_allocation': target_allocation,
            'recommendations': recommendations,
            'market_regime': regime
        }
    
    def _analyze_correlations_enhanced(self, opportunities: List[OpportunityRanking],
                                      portfolio_context: Dict) -> Dict:
        """Enhanced correlation analysis"""
        
        correlations = {}
        high_correlation_pairs = []
        
        # Analyze pairwise correlations
        for i, opp1 in enumerate(opportunities):
            for opp2 in opportunities[i+1:]:
                correlation = self._estimate_correlation(opp1, opp2)
                pair_key = f"{opp1.ticker}-{opp2.ticker}"
                correlations[pair_key] = correlation
                
                if correlation > self.risk_thresholds['max_correlation']:
                    high_correlation_pairs.append({
                        'pair': pair_key,
                        'correlation': correlation,
                        'risk': 'High correlation risk'
                    })
        
        # Portfolio correlation
        avg_correlation = np.mean(list(correlations.values())) if correlations else 0
        
        return {
            'average_correlation': round(avg_correlation, 3),
            'max_correlation': round(max(correlations.values()), 3) if correlations else 0,
            'high_correlation_pairs': high_correlation_pairs,
            'diversification_score': round(10 * (1 - avg_correlation), 1),
            'recommendation': self._get_correlation_recommendation(avg_correlation)
        }
    
    def _generate_execution_plan(self, opportunities: List[OpportunityRanking],
                                risk_assessment: RiskAssessment,
                                portfolio_context: Dict) -> Dict:
        """Generate detailed execution plan"""
        
        if not opportunities:
            return {'actions': [], 'total_capital': 0}
        
        available_capital = portfolio_context.get('cash', 100000)
        actions = []
        total_allocation = 0
        
        for opp in opportunities[:5]:  # Top 5 opportunities
            # Calculate position size based on conviction and risk
            base_allocation = opp.position_weight / 100 * available_capital
            
            # Adjust for risk
            if risk_assessment.risk_level == RiskLevel.HIGH.value:
                allocation = base_allocation * 0.7
            elif risk_assessment.risk_level == RiskLevel.VERY_HIGH.value:
                allocation = base_allocation * 0.5
            else:
                allocation = base_allocation
            
            actions.append({
                'ticker': opp.ticker,
                'action': 'BUY',
                'allocation': round(allocation, 2),
                'priority': opp.execution_priority,
                'timing': self._determine_execution_timing(opp),
                'entry_strategy': self._determine_entry_strategy(opp),
                'risk_controls': {
                    'stop_loss': f"{opp.expected_return * -0.5:.1%}",
                    'position_limit': f"{opp.position_weight}%"
                }
            })
            
            total_allocation += allocation
        
        return {
            'actions': actions,
            'total_capital_required': round(total_allocation, 2),
            'available_capital': available_capital,
            'utilization_rate': round(total_allocation / available_capital * 100, 1),
            'execution_timeline': self._create_execution_timeline(actions)
        }
    
    # Helper methods
    def _calculate_correlation_bonus(self, report: Dict, portfolio_context: Dict) -> float:
        """Calculate bonus for low correlation to existing portfolio"""
        correlation = report.get('correlation_to_spy', 0.5)
        if abs(correlation) < 0.3:
            return 8
        elif abs(correlation) < 0.5:
            return 5
        else:
            return 2
    
    def _calculate_sector_momentum(self, report: Dict, market_context: Dict) -> float:
        """Calculate sector momentum score"""
        sector = report.get('sector', 'Unknown')
        sector_performance = market_context.get('sector_performance', {})
        
        # Map sector to ETF
        sector_etf_map = {
            'Technology': 'XLK',
            'Financials': 'XLF',
            'Healthcare': 'XLV',
            'Energy': 'XLE'
        }
        
        etf = sector_etf_map.get(sector)
        if etf and etf in sector_performance:
            performance = sector_performance[etf]
            if performance > 2:
                return 8
            elif performance > 0:
                return 5
            else:
                return 3
        
        return 5  # Neutral
    
    def _calculate_market_alignment(self, report: Dict, market_regime: str) -> float:
        """Calculate alignment with current market regime"""
        sector = report.get('sector', 'Unknown')
        
        if market_regime == 'risk_on':
            if sector in ['Technology', 'Consumer Discretionary']:
                return 9
            elif sector in ['Financials', 'Industrials']:
                return 7
            else:
                return 4
        
        elif market_regime == 'risk_off':
            if sector in ['Utilities', 'Consumer Staples', 'Healthcare']:
                return 9
            elif sector in ['Real Estate', 'Bonds']:
                return 7
            else:
                return 4
        
        return 5  # Neutral
    
    def _calculate_time_horizon_fit(self, report: Dict, portfolio_context: Dict) -> float:
        """Calculate fit with portfolio time horizon needs"""
        # Simplified - could be enhanced with actual portfolio analysis
        return 6
    
    def _calculate_risk_adjustment(self, report: Dict) -> float:
        """Calculate risk adjustment factor"""
        risk_score = report.get('risk_assessment', {}).get('overall_risk_score', 5)
        
        # Convert risk score to adjustment factor (0.5 to 1.5)
        adjustment = 1.5 - (risk_score / 10)
        return max(0.5, min(1.5, adjustment))
    
    def _assess_correlation_risk(self, report: Dict, portfolio_context: Dict) -> float:
        """Assess correlation risk for specific opportunity"""
        return report.get('correlation_to_spy', 0.5) * 10
    
    def _calculate_concentration_risk(self, opportunities: List[OpportunityRanking]) -> float:
        """Calculate concentration risk"""
        if not opportunities:
            return 0
        
        # Sector concentration
        sectors = defaultdict(int)
        for opp in opportunities:
            sectors[opp.sector] += 1
        
        max_concentration = max(sectors.values()) / len(opportunities)
        return max_concentration * 10
    
    def _calculate_portfolio_correlation_risk(self, opportunities: List[OpportunityRanking]) -> float:
        """Calculate portfolio correlation risk"""
        if not opportunities:
            return 0
        
        avg_correlation = np.mean([opp.correlation_risk for opp in opportunities])
        return min(avg_correlation, 10)
    
    def _calculate_market_risk(self, opportunities: List[OpportunityRanking]) -> float:
        """Calculate market risk"""
        if not opportunities:
            return 0
        
        # Average of risk levels
        risk_scores = []
        for opp in opportunities:
            if opp.risk_level == RiskLevel.LOW.value:
                risk_scores.append(2)
            elif opp.risk_level == RiskLevel.MEDIUM.value:
                risk_scores.append(5)
            elif opp.risk_level == RiskLevel.HIGH.value:
                risk_scores.append(8)
            else:
                risk_scores.append(10)
        
        return np.mean(risk_scores)
    
    def _calculate_liquidity_risk(self, opportunities: List[OpportunityRanking]) -> float:
        """Calculate liquidity risk"""
        if not opportunities:
            return 0
        
        avg_liquidity = np.mean([opp.liquidity_score for opp in opportunities])
        return 10 - avg_liquidity  # Inverse relationship
    
    def _estimate_correlation(self, opp1: OpportunityRanking, opp2: OpportunityRanking) -> float:
        """Estimate correlation between two opportunities"""
        # Simplified correlation estimation
        if opp1.sector == opp2.sector:
            return 0.7
        elif opp1.market_cap == opp2.market_cap:
            return 0.4
        else:
            return 0.2
    
    def _get_correlation_recommendation(self, avg_correlation: float) -> str:
        """Get recommendation based on correlation"""
        if avg_correlation > 0.7:
            return "High correlation - seek diversification"
        elif avg_correlation > 0.5:
            return "Moderate correlation - monitor closely"
        else:
            return "Good diversification"
    
    def _determine_execution_timing(self, opp: OpportunityRanking) -> str:
        """Determine optimal execution timing"""
        if opp.time_horizon == TimeHorizon.SHORT_TERM.value:
            return "Immediate"
        elif opp.time_horizon == TimeHorizon.MEDIUM_TERM.value:
            return "Within 2-3 days"
        else:
            return "Within 1 week"
    
    def _determine_entry_strategy(self, opp: OpportunityRanking) -> str:
        """Determine entry strategy"""
        if opp.liquidity_score > 8:
            return "Market order"
        elif opp.liquidity_score > 5:
            return "Limit order at mid"
        else:
            return "Scale in over multiple days"
    
    def _create_execution_timeline(self, actions: List[Dict]) -> List[Dict]:
        """Create execution timeline"""
        timeline = []
        
        immediate = [a for a in actions if 'Immediate' in a['timing']]
        short_term = [a for a in actions if '2-3 days' in a['timing']]
        medium_term = [a for a in actions if '1 week' in a['timing']]
        
        if immediate:
            timeline.append({
                'day': 'Day 1',
                'actions': [a['ticker'] for a in immediate],
                'capital': sum(a['allocation'] for a in immediate)
            })
        
        if short_term:
            timeline.append({
                'day': 'Day 2-3',
                'actions': [a['ticker'] for a in short_term],
                'capital': sum(a['allocation'] for a in short_term)
            })
        
        if medium_term:
            timeline.append({
                'day': 'Day 4-7',
                'actions': [a['ticker'] for a in medium_term],
                'capital': sum(a['allocation'] for a in medium_term)
            })
        
        return timeline
    
    def _calculate_overall_confidence(self, opportunities: List[OpportunityRanking]) -> float:
        """Calculate overall confidence score"""
        if not opportunities:
            return 0
        
        confidences = [opp.conviction_score for opp in opportunities[:5]]
        return round(np.mean(confidences) * 2, 1)  # Scale to 10
    
    def _create_empty_analysis(self, reason: str) -> Dict:
        """Create empty analysis result"""
        return {
            'status': 'error',
            'error': reason,
            'ranked_opportunities': [],
            'strategic_themes': [],
            'risk_assessment': self._create_default_risk_assessment(),
            'time_horizon_allocation': {},
            'correlation_analysis': {},
            'execution_plan': {'actions': [], 'total_capital': 0}
        }
    
    def _create_default_risk_assessment(self) -> RiskAssessment:
        """Create default risk assessment"""
        return RiskAssessment(
            overall_risk_score=5.0,
            risk_level=RiskLevel.MEDIUM.value,
            concentration_risk=5.0,
            correlation_risk=5.0,
            market_risk=5.0,
            liquidity_risk=5.0,
            key_risk_factors=[],
            risk_mitigation=[]
        )


# ========================================================================================
# MARKET CONTEXT ANALYZER (Enhanced)
# ========================================================================================

class MarketContextAnalyzer:
    """Enhanced market context analyzer"""
    
    def __init__(self, alpaca_provider):
        self.alpaca = alpaca_provider
        self.logger = logging.getLogger('market_context')
        
    async def analyze_market_regime(self) -> Dict:
        """Analyze current market regime with multiple indicators"""
        
        try:
            # Get market data
            spy_data = await self.alpaca.get_bars('SPY', timeframe='1Day', limit=50)
            vix_data = await self.alpaca.get_latest_quote('VIXY')
            
            # Calculate indicators
            regime_indicators = {
                'trend': self._calculate_trend(spy_data),
                'volatility': self._assess_volatility(vix_data),
                'breadth': await self._calculate_breadth(),
                'momentum': self._calculate_momentum(spy_data)
            }
            
            # Determine regime
            regime = self._determine_regime(regime_indicators)
            
            return {
                'regime': regime,
                'indicators': regime_indicators,
                'confidence': self._calculate_regime_confidence(regime_indicators),
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Market regime analysis failed: {str(e)}")
            return {
                'regime': MarketRegime.NEUTRAL.value,
                'indicators': {},
                'confidence': 0.5,
                'timestamp': datetime.now().isoformat()
            }
    
    def _calculate_trend(self, spy_data: List[Dict]) -> str:
        """Calculate market trend"""
        if not spy_data or len(spy_data) < 20:
            return 'neutral'
        
        closes = [bar['close'] for bar in spy_data]
        sma_20 = np.mean(closes[-20:])
        sma_50 = np.mean(closes) if len(closes) >= 50 else sma_20
        
        current = closes[-1]
        
        if current > sma_20 * 1.02 and sma_20 > sma_50:
            return 'strong_uptrend'
        elif current > sma_20:
            return 'uptrend'
        elif current < sma_20 * 0.98 and sma_20 < sma_50:
            return 'strong_downtrend'
        elif current < sma_20:
            return 'downtrend'
        else:
            return 'neutral'
    
    def _assess_volatility(self, vix_data: Dict) -> str:
        """Assess volatility level"""
        vix = vix_data.get('price', 20)
        
        if vix < 12:
            return 'very_low'
        elif vix < 20:
            return 'low'
        elif vix < 30:
            return 'moderate'
        elif vix < 40:
            return 'high'
        else:
            return 'extreme'
    
    async def _calculate_breadth(self) -> float:
        """Calculate market breadth (simplified)"""
        # In production, this would calculate advance/decline ratio
        return 0.55  # Placeholder
    
    def _calculate_momentum(self, spy_data: List[Dict]) -> float:
        """Calculate momentum indicator"""
        if not spy_data or len(spy_data) < 10:
            return 0
        
        closes = [bar['close'] for bar in spy_data]
        returns = [(closes[i] - closes[i-1]) / closes[i-1] for i in range(1, len(closes))]
        
        return np.mean(returns[-10:]) * 100  # 10-day average return
    
    def _determine_regime(self, indicators: Dict) -> str:
        """Determine market regime from indicators"""
        trend = indicators.get('trend', 'neutral')
        volatility = indicators.get('volatility', 'moderate')
        momentum = indicators.get('momentum', 0)
        
        if 'uptrend' in trend and volatility in ['very_low', 'low'] and momentum > 0.5:
            return MarketRegime.RISK_ON.value
        elif 'downtrend' in trend and volatility in ['high', 'extreme']:
            return MarketRegime.RISK_OFF.value
        elif volatility in ['high', 'extreme']:
            return MarketRegime.TRANSITION.value
        else:
            return MarketRegime.NEUTRAL.value
    
    def _calculate_regime_confidence(self, indicators: Dict) -> float:
        """Calculate confidence in regime determination"""
        # Simplified confidence calculation
        return 0.75


# ========================================================================================
# ENHANCED SENIOR RESEARCH ANALYST
# ========================================================================================

class SeniorResearchAnalyst:
    """
    Enhanced Senior Research Analyst Agent
    
    Synthesizes junior analyst reports into strategic portfolio recommendations
    with enhanced market context awareness and feedback mechanisms.
    """
    
    def __init__(self, llm_provider, alpaca_provider, config):
        """Initialize Enhanced Senior Research Analyst"""
        
        self.agent_name = "senior_research_analyst"
        self.agent_id = str(uuid.uuid4())
        
        # Core dependencies
        self.llm = llm_provider
        self.alpaca = alpaca_provider
        self.config = config
        
        # Setup logging
        self.logger = self._setup_logging()
        
        # Shared components
        self.market_context_manager = MarketContextManager(alpaca_provider)
        self.cache_manager = IntelligentCacheManager(max_size=50, ttl_seconds=600)
        self.metadata_tracker = AnalysisMetadataTracker()
        
        # Analysis engines
        self.strategic_engine = StrategicAnalysisEngine()
        self.market_analyzer = MarketContextAnalyzer(alpaca_provider)
        
        # Performance tracking
        self.performance_metrics = {
            "total_syntheses": 0,
            "successful_syntheses": 0,
            "failed_syntheses": 0,
            "average_processing_time": 0.0,
            "cache_hits": 0,
            "feedback_provided": 0,
            "last_activity": None
        }
        
        # Feedback storage for Junior Analysts
        self.feedback_queue = []
        
        self.logger.info(f"✅ Enhanced Senior Research Analyst initialized with ID: {self.agent_id}")
    
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
                                portfolio_context: Optional[Dict] = None) -> Dict:
        """Main entry point for synthesizing junior analyst reports"""
        
        start_time = datetime.now()
        
        # Create analysis chain
        chain_id = self.metadata_tracker.create_analysis_chain(
            f"synthesis_{len(junior_reports)}_reports"
        )
        
        try:
            # Check cache
            cache_key = self._generate_cache_key(junior_reports)
            cached_result = self.cache_manager.get(cache_key)
            
            if cached_result:
                self.performance_metrics['cache_hits'] += 1
                self.logger.info("Cache hit for synthesis")
                return cached_result
            
            # Get shared market context
            market_context = await self.market_context_manager.get_current_context()
            
            # Get enhanced market regime analysis
            regime_analysis = await self.market_analyzer.analyze_market_regime()
            market_context['regime_analysis'] = regime_analysis
            
            # Get portfolio context if not provided
            if not portfolio_context:
                portfolio_context = await self._get_portfolio_context()
            
            # Perform strategic synthesis
            synthesis_result = self.strategic_engine.synthesize_junior_reports(
                junior_reports, market_context, portfolio_context
            )
            
            # Enhance with LLM insights
            enhanced_result = await self._enhance_with_llm(synthesis_result, market_context)
            
            # Generate markdown report
            markdown_report = self._generate_markdown_report(enhanced_result)
            
            # Prepare feedback for Junior Analysts (only if synthesis succeeded)
            if synthesis_result.get('status') == 'success':
                self._prepare_junior_feedback(junior_reports, enhanced_result)
            
            # Create final result
            final_result = {
                'status': 'success',
                'timestamp': datetime.now().isoformat(),
                'strategic_analysis': enhanced_result,
                'markdown_report': markdown_report,
                'metadata': {
                    'agent_id': self.agent_id,
                    'reports_synthesized': len(junior_reports),
                    'processing_time': (datetime.now() - start_time).total_seconds(),
                    'market_regime': market_context.get('regime', 'neutral'),
                    'confidence_score': enhanced_result.get('confidence_score', 0)
                },
                'analysis_chain': self.metadata_tracker.get_chain_summary(chain_id)
            }
            
            # Cache result
            self.cache_manager.put(cache_key, final_result)
            
            # Update metrics
            self._update_metrics(True, (datetime.now() - start_time).total_seconds())
            
            # Complete chain
            self.metadata_tracker.complete_chain(chain_id)
            
            return final_result
            
        except Exception as e:
            self.logger.error(f"Synthesis failed: {str(e)}")
            self._update_metrics(False, (datetime.now() - start_time).total_seconds())
            self.metadata_tracker.complete_chain(chain_id, "failed")
            
            return {
                'status': 'error',
                'error': str(e),
                'strategic_analysis': self.strategic_engine._create_empty_analysis(str(e)),
                'markdown_report': '',
                'metadata': {
                    'agent_id': self.agent_id,
                    'processing_time': (datetime.now() - start_time).total_seconds()
                }
            }
    
    async def provide_junior_feedback(self, analysis_id: str, feedback: Dict) -> Dict:
        """Provide feedback to Junior Analyst for learning"""
        
        feedback_data = {
            "analysis_id": analysis_id,
            "timestamp": datetime.now().isoformat(),
            "feedback_type": feedback.get('type', 'general'),
            "performance_score": feedback.get('score', 5),
            "improvements_needed": feedback.get('improvements', []),
            "strengths": feedback.get('strengths', []),
            "context_missing": feedback.get('missing_context', [])
        }
        
        # Add to feedback queue
        self.feedback_queue.append(feedback_data)
        self.performance_metrics['feedback_provided'] += 1
        
        self.logger.info(f"Feedback prepared for analysis {analysis_id}")
        
        return {
            "status": "feedback_recorded",
            "feedback_id": str(uuid.uuid4()),
            "analysis_id": analysis_id
        }
    
    async def rank_opportunities(self, opportunities: List[Dict], 
                                market_context: Optional[Dict] = None) -> List[Dict]:
        """Rank investment opportunities"""
        
        if not market_context:
            market_context = await self.market_context_manager.get_current_context()
        
        # Use strategic engine for ranking
        ranked = self.strategic_engine._rank_opportunities_enhanced(
            opportunities, market_context, {}
        )
        
        # Convert to dict format
        return [self._opportunity_to_dict(opp) for opp in ranked]
    
    async def assess_portfolio_risk(self, portfolio: List[Dict]) -> Dict:
        """Assess portfolio-level risk"""
        
        # Convert to OpportunityRanking format
        opportunities = []
        for position in portfolio:
            opp = OpportunityRanking(
                ticker=position.get('ticker'),
                conviction_score=position.get('conviction', 5),
                risk_adjusted_score=5,
                time_horizon=TimeHorizon.MEDIUM_TERM.value,
                expected_return=0.1,
                risk_level=RiskLevel.MEDIUM.value,
                sector=position.get('sector', 'Unknown'),
                market_cap='Large',
                correlation_risk=5,
                liquidity_score=7,
                catalyst_strength=5,
                thesis_summary='',
                key_risks=[],
                position_weight=position.get('weight', 3),
                execution_priority=1,
                junior_analyst_id='',
                analysis_chain_id=''
            )
            opportunities.append(opp)
        
        # Assess risk
        risk_assessment = self.strategic_engine._assess_portfolio_risk_enhanced(
            opportunities, {}
        )
        
        return {
            'overall_risk_score': risk_assessment.overall_risk_score,
            'risk_level': risk_assessment.risk_level,
            'concentration_risk': risk_assessment.concentration_risk,
            'correlation_risk': risk_assessment.correlation_risk,
            'key_risk_factors': risk_assessment.key_risk_factors,
            'risk_mitigation': risk_assessment.risk_mitigation
        }
    
    async def identify_market_themes(self, junior_reports: List[Dict]) -> List[Dict]:
        """Identify strategic market themes"""
        
        market_context = await self.market_context_manager.get_current_context()
        themes = self.strategic_engine._identify_strategic_themes_enhanced(
            junior_reports, market_context
        )
        
        return [self._theme_to_dict(theme) for theme in themes]
    
    def get_performance_metrics(self) -> Dict:
        """Get performance metrics"""
        
        total = self.performance_metrics['total_syntheses']
        successful = self.performance_metrics['successful_syntheses']
        
        return {
            "agent_name": self.agent_name,
            "agent_id": self.agent_id,
            "total_syntheses": total,
            "success_rate": (successful / total * 100) if total > 0 else 0,
            "average_processing_time": self.performance_metrics['average_processing_time'],
            "cache_hit_rate": (
                self.performance_metrics['cache_hits'] / total * 100
            ) if total > 0 else 0,
            "feedback_provided": self.performance_metrics['feedback_provided']
        }
    
    # Private helper methods
    async def _get_portfolio_context(self) -> Dict:
        """Get current portfolio context"""
        # Simplified - would connect to portfolio database in production
        return {
            'positions': [],
            'cash': 100000,
            'total_value': 100000,
            'sectors': {}
        }
    
    async def _enhance_with_llm(self, synthesis_result: Dict, market_context: Dict) -> Dict:
        """Enhance synthesis with LLM insights"""
        
        # Prepare prompt for LLM
        prompt = self._create_llm_prompt(synthesis_result, market_context)
        
        try:
            # Get LLM insights
            llm_response = await self.llm.generate(prompt)
            
            # Parse and add insights
            synthesis_result['executive_summary'] = llm_response.get('summary', '')
            synthesis_result['strategic_recommendations'] = llm_response.get('recommendations', [])
            synthesis_result['risk_considerations'] = llm_response.get('risks', [])
            
        except Exception as e:
            self.logger.error(f"LLM enhancement failed: {str(e)}")
            synthesis_result['executive_summary'] = "Strategic analysis complete."
        
        return synthesis_result
    
    def _create_llm_prompt(self, synthesis: Dict, market_context: Dict) -> str:
        """Create prompt for LLM enhancement"""
        
        opportunities = synthesis.get('ranked_opportunities', [])[:3]
        themes = synthesis.get('strategic_themes', [])
        risk = synthesis.get('risk_assessment')
        
        # Handle both RiskAssessment dataclass and dict
        if risk:
            if hasattr(risk, 'risk_level'):
                risk_level = risk.risk_level
                risk_factors = risk.key_risk_factors
            else:
                risk_level = risk.get('risk_level', 'medium')
                risk_factors = risk.get('key_risk_factors', [])
        else:
            risk_level = 'medium'
            risk_factors = []
        
        prompt = f"""
        As a Senior Research Analyst, provide strategic insights for this portfolio analysis:
        
        Market Regime: {market_context.get('regime', 'neutral')}
        
        Top Opportunities:
        {self._format_opportunities_for_prompt(opportunities)}
        
        Strategic Themes:
        {self._format_themes_for_prompt(themes)}
        
        Risk Assessment:
        - Overall Risk: {risk_level}
        - Key Risks: {', '.join(risk_factors)}
        
        Provide:
        1. Executive summary (2-3 sentences)
        2. Top 3 strategic recommendations
        3. Key risk considerations
        """
        
        return prompt
    
    def _format_opportunities_for_prompt(self, opportunities: List) -> str:
        """Format opportunities for LLM prompt"""
        formatted = []
        for opp in opportunities[:3]:
            formatted.append(f"- {opp.ticker}: Score {opp.risk_adjusted_score}, {opp.time_horizon}")
        return '\n'.join(formatted)
    
    def _format_themes_for_prompt(self, themes: List) -> str:
        """Format themes for LLM prompt"""
        formatted = []
        for theme in themes[:3]:
            formatted.append(f"- {theme.theme_name}: {theme.confidence}/10 confidence")
        return '\n'.join(formatted)
    
    def _generate_markdown_report(self, analysis: Dict) -> str:
        """Generate comprehensive markdown report"""
        
        report = f"""
# Strategic Portfolio Analysis Report
**Date**: {datetime.now().strftime('%Y-%m-%d %H:%M')}
**Market Regime**: {analysis.get('market_regime', 'Neutral')}
**Confidence Score**: {analysis.get('confidence_score', 0)}/10

## Executive Summary
{analysis.get('executive_summary', 'Strategic analysis complete.')}

## Top Investment Opportunities

"""
        
        # Add opportunities
        opportunities = analysis.get('ranked_opportunities', [])
        for i, opp in enumerate(opportunities[:5], 1):
            report += f"""
### {i}. {opp.ticker}
- **Risk-Adjusted Score**: {opp.risk_adjusted_score}/10
- **Expected Return**: {opp.expected_return:.1%}
- **Time Horizon**: {opp.time_horizon}
- **Risk Level**: {opp.risk_level}
- **Position Weight**: {opp.position_weight}%
- **Key Risks**: {', '.join(opp.key_risks[:2]) if opp.key_risks else 'None identified'}
"""
        
        # Add themes
        report += "\n## Strategic Themes\n"
        themes = analysis.get('strategic_themes', [])
        for theme in themes[:3]:
            report += f"""
### {theme.theme_name}
- **Confidence**: {theme.confidence}/10
- **Allocation Suggestion**: {theme.allocation_suggestion}%
- **Supporting Stocks**: {', '.join(theme.supporting_tickers[:3])}
"""
        
        # Add risk assessment - Handle RiskAssessment dataclass properly
        risk = analysis.get('risk_assessment')
        if risk:
            # Check if it's a RiskAssessment dataclass or a dict
            if hasattr(risk, 'overall_risk_score'):
                # It's a dataclass
                report += f"""
## Risk Assessment
- **Overall Risk Score**: {risk.overall_risk_score}/10
- **Risk Level**: {risk.risk_level}
- **Key Risk Factors**: {', '.join(risk.key_risk_factors[:3]) if risk.key_risk_factors else 'None identified'}

### Risk Mitigation Strategies
"""
                for strategy in (risk.risk_mitigation or [])[:3]:
                    report += f"- {strategy}\n"
            else:
                # It's a dict (fallback)
                report += f"""
## Risk Assessment
- **Overall Risk Score**: {risk.get('overall_risk_score', 5)}/10
- **Risk Level**: {risk.get('risk_level', 'Medium')}
- **Key Risk Factors**: {', '.join(risk.get('key_risk_factors', [])[:3])}

### Risk Mitigation Strategies
"""
                for strategy in risk.get('risk_mitigation', [])[:3]:
                    report += f"- {strategy}\n"
        
        # Add execution plan
        execution = analysis.get('execution_plan', {})
        if execution.get('actions'):
            report += "\n## Execution Plan\n"
            report += f"**Total Capital Required**: ${execution.get('total_capital_required', 0):,.2f}\n\n"
            
            for action in execution.get('actions', [])[:3]:
                report += f"- **{action['ticker']}**: ${action['allocation']:,.2f} ({action['timing']})\n"
        
        return report
    
    def _prepare_junior_feedback(self, junior_reports: List[Dict], synthesis: Dict):
        """Prepare feedback for Junior Analysts"""
        
        # Analyze quality of junior reports
        for report in junior_reports:
            analysis_id = report.get('analysis_id')
            if not analysis_id:
                continue
            
            # Evaluate report quality
            score = self._evaluate_report_quality(report, synthesis)
            
            # Prepare feedback
            feedback = {
                'type': 'quality_assessment',
                'score': score,
                'improvements': [],
                'strengths': []
            }
            
            # Identify strengths and improvements
            if report.get('confidence', 0) >= 7:
                feedback['strengths'].append('High conviction analysis')
            
            if not report.get('catalysts'):
                feedback['improvements'].append('Include catalyst analysis')
            
            if not report.get('risk_assessment'):
                feedback['improvements'].append('Enhance risk assessment')
            
            # Add to feedback queue
            self.feedback_queue.append({
                'analysis_id': analysis_id,
                'feedback': feedback,
                'timestamp': datetime.now().isoformat()
            })
    
    def _evaluate_report_quality(self, report: Dict, synthesis: Dict) -> float:
        """Evaluate quality of junior report"""
        
        score = 5.0  # Base score
        
        # Check completeness
        if report.get('thesis'):
            score += 1
        if report.get('catalysts'):
            score += 1
        if report.get('risk_assessment'):
            score += 1
        if report.get('technical_signals'):
            score += 0.5
        
        # Check if recommendation was included in top opportunities
        ranked_tickers = [opp.ticker for opp in synthesis.get('ranked_opportunities', [])]
        if report.get('ticker') in ranked_tickers[:5]:
            score += 1.5
        
        return min(score, 10)
    
    def _generate_cache_key(self, reports: List[Dict]) -> str:
        """Generate cache key for reports"""
        
        # Create unique key from report tickers and timestamps
        key_parts = []
        for report in sorted(reports, key=lambda x: x.get('ticker', '')):
            key_parts.append(f"{report.get('ticker')}_{report.get('confidence', 0)}")
        
        key_string = '_'.join(key_parts)
        return hashlib.md5(key_string.encode()).hexdigest()
    
    def _update_metrics(self, success: bool, processing_time: float):
        """Update performance metrics"""
        
        self.performance_metrics['total_syntheses'] += 1
        
        if success:
            self.performance_metrics['successful_syntheses'] += 1
        else:
            self.performance_metrics['failed_syntheses'] += 1
        
        # Update average processing time
        current_avg = self.performance_metrics['average_processing_time']
        total = self.performance_metrics['total_syntheses']
        
        self.performance_metrics['average_processing_time'] = (
            (current_avg * (total - 1) + processing_time) / total
        )
        
        self.performance_metrics['last_activity'] = datetime.now()
    
    def _opportunity_to_dict(self, opp: OpportunityRanking) -> Dict:
        """Convert OpportunityRanking to dict"""
        return {
            'ticker': opp.ticker,
            'conviction_score': opp.conviction_score,
            'risk_adjusted_score': opp.risk_adjusted_score,
            'time_horizon': opp.time_horizon,
            'expected_return': opp.expected_return,
            'risk_level': opp.risk_level,
            'position_weight': opp.position_weight,
            'execution_priority': opp.execution_priority
        }
    
    def _theme_to_dict(self, theme: PortfolioTheme) -> Dict:
        """Convert PortfolioTheme to dict"""
        return {
            'theme_name': theme.theme_name,
            'theme_type': theme.theme_type,
            'confidence': theme.confidence,
            'supporting_tickers': theme.supporting_tickers,
            'allocation_suggestion': theme.allocation_suggestion,
            'expected_impact': theme.expected_impact
        }


# ========================================================================================
# FACTORY FUNCTION
# ========================================================================================

def create_senior_analyst(llm_provider, alpaca_provider, config) -> SeniorResearchAnalyst:
    """Factory function to create Senior Research Analyst"""
    return SeniorResearchAnalyst(llm_provider, alpaca_provider, config)