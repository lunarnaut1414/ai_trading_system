# agents/portfolio_manager.py
"""
Portfolio Manager Agent - Complete Implementation
Makes strategic portfolio decisions based on research recommendations and market conditions

Integrates with:
- Senior Research Analyst recommendations
- Economist Agent macro insights
- Market regime analysis
- Risk management framework
"""

import asyncio
import json
import logging
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
from enum import Enum
from dataclasses import dataclass, asdict
import numpy as np
import pandas as pd
from collections import defaultdict

# ========================================================================================
# ENUMS AND DATA CLASSES
# ========================================================================================

class MarketRegime(Enum):
    """Market regime classification"""
    RISK_ON = "risk_on"
    RISK_OFF = "risk_off"
    NEUTRAL = "neutral"
    TRANSITION = "transition"

class PortfolioPosture(Enum):
    """Portfolio positioning stance"""
    AGGRESSIVE = "aggressive"    # 85% equity, 15% cash
    BALANCED = "balanced"       # 70% equity, 30% cash
    DEFENSIVE = "defensive"     # 55% equity, 45% cash
    CASH_HEAVY = "cash_heavy"   # 30% equity, 70% cash

class ActionType(Enum):
    """Portfolio action types"""
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"
    TRIM = "TRIM"
    ADD = "ADD"
    CLOSE = "CLOSE"

class TimeHorizon(Enum):
    """Investment time horizons"""
    SHORT = "short"      # 1-10 days
    MEDIUM = "medium"    # 2-8 weeks
    LONG = "long"        # 2-6 months

@dataclass
class PortfolioAction:
    """Individual portfolio action with full context"""
    symbol: str
    action: ActionType
    target_weight: float
    current_weight: float
    sizing_rationale: str
    risk_assessment: str
    time_horizon: TimeHorizon
    confidence: int  # 1-10 scale
    expected_return: float
    max_loss: float
    correlation_impact: float

@dataclass
class RiskLimits:
    """Portfolio risk limits and constraints"""
    max_single_position: float = 5.0    # % of portfolio
    max_sector_exposure: float = 25.0   # % of portfolio
    max_correlation: float = 0.7        # Between major positions
    max_daily_var: float = 2.0          # Daily Value at Risk %
    max_drawdown: float = 15.0          # Maximum portfolio drawdown %
    min_cash_reserve: float = 10.0      # Minimum cash %
    max_leverage: float = 1.0           # No leverage for now

@dataclass
class PortfolioMetrics:
    """Current portfolio performance metrics"""
    total_value: float
    cash_percentage: float
    equity_percentage: float
    day_pnl: float
    day_pnl_pct: float
    ytd_return: float
    sharpe_ratio: float
    max_drawdown: float
    current_drawdown: float
    var_95: float
    beta: float
    correlation_matrix: Dict

# ========================================================================================
# MARKET REGIME ANALYZER
# ========================================================================================

class MarketRegimeAnalyzer:
    """Analyzes market conditions to determine regime"""
    
    def __init__(self, alpaca_provider):
        self.alpaca = alpaca_provider
        self.logger = logging.getLogger('market_regime_analyzer')
        
    async def analyze_market_regime(self) -> Dict:
        """Comprehensive market regime analysis"""
        
        try:
            # Get market data for major indices
            spy_data = await self._get_market_data('SPY')
            vix_data = await self._get_market_data('VIX')
            
            # Analyze multiple dimensions
            trend_analysis = self._analyze_trend(spy_data)
            volatility_analysis = self._analyze_volatility(vix_data)
            momentum_analysis = self._analyze_momentum(spy_data)
            sentiment_analysis = await self._analyze_sentiment()
            correlation_analysis = await self._analyze_correlations()
            
            # Determine overall regime
            regime = self._determine_regime(
                trend_analysis,
                volatility_analysis,
                momentum_analysis,
                sentiment_analysis,
                correlation_analysis
            )
            
            # Generate posture recommendation
            posture_rec = self._recommend_portfolio_posture(regime)
            
            return {
                'timestamp': datetime.now().isoformat(),
                'regime': regime,
                'confidence': self._calculate_regime_confidence(regime),
                'posture_recommendation': posture_rec,
                'regime_characteristics': self._get_regime_characteristics(regime),
                'risk_indicators': self._identify_risk_indicators(regime),
                'opportunity_indicators': self._identify_opportunities(regime),
                'analysis_components': {
                    'trend': trend_analysis,
                    'volatility': volatility_analysis,
                    'momentum': momentum_analysis,
                    'sentiment': sentiment_analysis,
                    'correlation': correlation_analysis
                }
            }
            
        except Exception as e:
            self.logger.error(f"Market regime analysis error: {str(e)}")
            return self._get_default_regime()
    
    async def _get_market_data(self, symbol: str) -> Dict:
        """Get market data for analysis"""
        
        try:
            # Get bars data
            bars = await self.alpaca.get_bars(
                symbol,
                timeframe='1Day',
                limit=50
            )
            
            if not bars or len(bars) == 0:
                return {}
            
            # Calculate returns and metrics
            closes = [bar.c for bar in bars]
            returns = [(closes[i] - closes[i-1])/closes[i-1] for i in range(1, len(closes))]
            
            return {
                'symbol': symbol,
                'current_price': closes[-1],
                'returns': returns,
                'volatility': np.std(returns) * np.sqrt(252) if returns else 0,
                'trend': self._calculate_trend(closes),
                'momentum': self._calculate_momentum(closes)
            }
            
        except Exception as e:
            self.logger.error(f"Error getting market data for {symbol}: {str(e)}")
            return {}
    
    def _analyze_trend(self, spy_data: Dict) -> Dict:
        """Analyze market trend"""
        
        if not spy_data or 'trend' not in spy_data:
            return {'regime': 'neutral', 'strength': 0}
        
        trend = spy_data['trend']
        
        if trend > 0.02:
            return {'regime': 'uptrend', 'strength': min(trend * 50, 10)}
        elif trend < -0.02:
            return {'regime': 'downtrend', 'strength': min(abs(trend) * 50, 10)}
        else:
            return {'regime': 'sideways', 'strength': 5}
    
    def _analyze_volatility(self, vix_data: Dict) -> Dict:
        """Analyze market volatility"""
        
        if not vix_data or 'current_price' not in vix_data:
            return {'regime': 'normal_volatility', 'level': 20}
        
        vix_level = vix_data.get('current_price', 20)
        
        if vix_level < 15:
            return {'regime': 'low_volatility', 'level': vix_level}
        elif vix_level < 25:
            return {'regime': 'normal_volatility', 'level': vix_level}
        elif vix_level < 35:
            return {'regime': 'elevated_volatility', 'level': vix_level}
        else:
            return {'regime': 'high_volatility', 'level': vix_level}
    
    def _analyze_momentum(self, spy_data: Dict) -> Dict:
        """Analyze market momentum"""
        
        if not spy_data or 'momentum' not in spy_data:
            return {'regime': 'neutral', 'score': 0}
        
        momentum = spy_data['momentum']
        
        if momentum > 0.05:
            return {'regime': 'strong_positive', 'score': min(momentum * 100, 10)}
        elif momentum > 0:
            return {'regime': 'positive', 'score': momentum * 100}
        elif momentum > -0.05:
            return {'regime': 'negative', 'score': abs(momentum) * 100}
        else:
            return {'regime': 'strong_negative', 'score': min(abs(momentum) * 100, 10)}
    
    async def _analyze_sentiment(self) -> Dict:
        """Analyze market sentiment"""
        
        # Simplified sentiment analysis
        # In production, would integrate news sentiment, put/call ratios, etc.
        return {'regime': 'neutral', 'score': 5}
    
    async def _analyze_correlations(self) -> Dict:
        """Analyze asset correlations"""
        
        # Simplified correlation analysis
        # In production, would calculate actual correlations
        return {'regime': 'normal', 'dispersion': 'moderate'}
    
    def _determine_regime(self, trend: Dict, volatility: Dict, 
                         momentum: Dict, sentiment: Dict, correlation: Dict) -> MarketRegime:
        """Determine overall market regime"""
        
        # Score each component
        risk_on_score = 0
        risk_off_score = 0
        
        # Trend contribution
        if trend.get('regime') == 'uptrend':
            risk_on_score += 2
        elif trend.get('regime') == 'downtrend':
            risk_off_score += 2
        
        # Volatility contribution
        if volatility.get('regime') == 'low_volatility':
            risk_on_score += 1
        elif volatility.get('regime') in ['elevated_volatility', 'high_volatility']:
            risk_off_score += 2
        
        # Momentum contribution
        if momentum.get('regime') in ['strong_positive', 'positive']:
            risk_on_score += 1
        elif momentum.get('regime') in ['strong_negative', 'negative']:
            risk_off_score += 1
        
        # Determine regime
        if risk_on_score > risk_off_score + 2:
            return MarketRegime.RISK_ON
        elif risk_off_score > risk_on_score + 2:
            return MarketRegime.RISK_OFF
        elif abs(risk_on_score - risk_off_score) > 1:
            return MarketRegime.TRANSITION
        else:
            return MarketRegime.NEUTRAL
    
    def _recommend_portfolio_posture(self, regime: MarketRegime) -> Dict:
        """Recommend portfolio posture based on regime"""
        
        posture_map = {
            MarketRegime.RISK_ON: {
                'posture': PortfolioPosture.AGGRESSIVE,
                'equity_target': 85,
                'cash_target': 15,
                'style_bias': 'growth',
                'sector_preference': 'technology_consumer'
            },
            MarketRegime.RISK_OFF: {
                'posture': PortfolioPosture.DEFENSIVE,
                'equity_target': 55,
                'cash_target': 45,
                'style_bias': 'value_defensive',
                'sector_preference': 'utilities_staples'
            },
            MarketRegime.NEUTRAL: {
                'posture': PortfolioPosture.BALANCED,
                'equity_target': 70,
                'cash_target': 30,
                'style_bias': 'balanced',
                'sector_preference': 'diversified'
            },
            MarketRegime.TRANSITION: {
                'posture': PortfolioPosture.BALANCED,
                'equity_target': 65,
                'cash_target': 35,
                'style_bias': 'quality',
                'sector_preference': 'selective'
            }
        }
        
        return posture_map.get(regime, posture_map[MarketRegime.NEUTRAL])
    
    def _calculate_regime_confidence(self, regime: MarketRegime) -> float:
        """Calculate confidence in regime determination"""
        
        # Simplified confidence calculation
        base_confidence = 70.0
        
        if regime in [MarketRegime.RISK_ON, MarketRegime.RISK_OFF]:
            return base_confidence + 15
        elif regime == MarketRegime.TRANSITION:
            return base_confidence - 10
        else:
            return base_confidence
    
    def _get_regime_characteristics(self, regime: MarketRegime) -> List[str]:
        """Get characteristics of current regime"""
        
        characteristics = {
            MarketRegime.RISK_ON: [
                "Positive market momentum",
                "Low volatility environment",
                "Strong economic indicators",
                "Favorable for growth assets"
            ],
            MarketRegime.RISK_OFF: [
                "Negative market sentiment",
                "Elevated volatility",
                "Economic uncertainty",
                "Flight to quality behavior"
            ],
            MarketRegime.NEUTRAL: [
                "Mixed market signals",
                "Normal volatility levels",
                "Balanced risk/reward",
                "Sector rotation opportunities"
            ],
            MarketRegime.TRANSITION: [
                "Regime change in progress",
                "Increased uncertainty",
                "Diverging indicators",
                "Positioning adjustments needed"
            ]
        }
        
        return characteristics.get(regime, [])
    
    def _identify_risk_indicators(self, regime: MarketRegime) -> List[str]:
        """Identify current risk indicators"""
        
        risks = []
        
        if regime == MarketRegime.RISK_OFF:
            risks.extend([
                "Market drawdown risk elevated",
                "Correlation breakdown possible",
                "Liquidity concerns rising"
            ])
        elif regime == MarketRegime.TRANSITION:
            risks.extend([
                "Regime uncertainty high",
                "False signals possible",
                "Whipsaw risk elevated"
            ])
        
        return risks
    
    def _identify_opportunities(self, regime: MarketRegime) -> List[str]:
        """Identify regime-specific opportunities"""
        
        opportunities = []
        
        if regime == MarketRegime.RISK_ON:
            opportunities.extend([
                "Growth stock outperformance",
                "Small-cap opportunities",
                "Emerging market potential"
            ])
        elif regime == MarketRegime.RISK_OFF:
            opportunities.extend([
                "Value stock opportunities",
                "Defensive sector strength",
                "Bond market opportunities"
            ])
        
        return opportunities
    
    def _calculate_trend(self, prices: List[float]) -> float:
        """Calculate price trend"""
        
        if len(prices) < 20:
            return 0
        
        # Simple linear regression slope
        x = list(range(len(prices)))
        y = prices
        
        n = len(x)
        xy_sum = sum(x[i] * y[i] for i in range(n))
        x_sum = sum(x)
        y_sum = sum(y)
        x_squared_sum = sum(x[i]**2 for i in range(n))
        
        slope = (n * xy_sum - x_sum * y_sum) / (n * x_squared_sum - x_sum**2)
        
        # Normalize by average price
        avg_price = sum(prices) / len(prices)
        normalized_slope = slope / avg_price if avg_price > 0 else 0
        
        return normalized_slope
    
    def _calculate_momentum(self, prices: List[float]) -> float:
        """Calculate price momentum"""
        
        if len(prices) < 20:
            return 0
        
        # Rate of change over 20 periods
        roc = (prices[-1] - prices[-20]) / prices[-20] if prices[-20] > 0 else 0
        
        return roc
    
    def _get_default_regime(self) -> Dict:
        """Get default regime for error cases"""
        
        return {
            'timestamp': datetime.now().isoformat(),
            'regime': MarketRegime.NEUTRAL,
            'confidence': 50.0,
            'posture_recommendation': {
                'posture': PortfolioPosture.BALANCED,
                'equity_target': 60,
                'cash_target': 40,
                'style_bias': 'balanced',
                'sector_preference': 'diversified'
            },
            'regime_characteristics': ['Default conservative positioning'],
            'risk_indicators': ['Data unavailable - using conservative defaults'],
            'opportunity_indicators': [],
            'analysis_components': {
                'trend': {'regime': 'neutral', 'error': 'data_unavailable'},
                'volatility': {'regime': 'normal_volatility', 'error': 'data_unavailable'},
                'momentum': {'regime': 'mixed', 'error': 'data_unavailable'},
                'sentiment': {'regime': 'neutral', 'error': 'data_unavailable'},
                'correlation': {'regime': 'normal', 'error': 'data_unavailable'}
            },
            'error': 'Using default conservative regime due to data issues'
        }

# ========================================================================================
# PORTFOLIO RISK MANAGER
# ========================================================================================

class PortfolioRiskManager:
    """Comprehensive portfolio risk management"""
    
    def __init__(self, alpaca_provider, risk_limits: RiskLimits = None):
        self.alpaca = alpaca_provider
        self.risk_limits = risk_limits or RiskLimits()
        self.logger = logging.getLogger('portfolio_risk_manager')
        
    async def assess_portfolio_risk(self, positions: List[Dict], 
                                   new_trades: List[Dict] = None) -> Dict:
        """Comprehensive portfolio risk assessment"""
        
        try:
            # Get current portfolio metrics
            portfolio_metrics = await self._calculate_portfolio_metrics(positions)
            
            # Check risk limit violations
            violations = self._check_risk_violations(portfolio_metrics, positions)
            
            # Calculate position-level risks
            position_risks = await self._calculate_position_risks(positions)
            
            # Assess concentration risk
            concentration_risk = self._assess_concentration_risk(positions)
            
            # Calculate correlation risk
            correlation_risk = await self._assess_correlation_risk(positions)
            
            # Determine overall risk level
            risk_level = self._determine_risk_level(
                violations, 
                concentration_risk, 
                correlation_risk
            )
            
            # Generate risk recommendations
            recommendations = self._generate_risk_recommendations(
                risk_level, 
                violations, 
                portfolio_metrics
            )
            
            return {
                'timestamp': datetime.now().isoformat(),
                'risk_level': risk_level,
                'overall_risk_score': self._calculate_risk_score(risk_level),
                'portfolio_metrics': portfolio_metrics,
                'limit_violations': violations,
                'position_risks': position_risks,
                'concentration_risk': concentration_risk,
                'correlation_risk': correlation_risk,
                'recommendations': recommendations,
                'can_add_risk': risk_level in ['low', 'medium'],
                'should_reduce_risk': risk_level in ['high', 'critical']
            }
            
        except Exception as e:
            self.logger.error(f"Risk assessment error: {str(e)}")
            return self._get_default_risk_assessment()
    
    async def _calculate_portfolio_metrics(self, positions: List[Dict]) -> Dict:
        """Calculate key portfolio metrics"""
        
        try:
            account = await self.alpaca.get_account()
            
            total_value = float(account.portfolio_value)
            cash = float(account.cash)
            equity_value = total_value - cash
            
            # Calculate returns
            day_pnl = float(account.equity) - float(account.last_equity)
            day_pnl_pct = (day_pnl / float(account.last_equity)) * 100 if float(account.last_equity) > 0 else 0
            
            # Calculate risk metrics
            position_values = [pos.get('market_value', 0) for pos in positions if 'error' not in pos]
            
            return {
                'total_value': total_value,
                'cash': cash,
                'cash_percentage': (cash / total_value) * 100 if total_value > 0 else 100,
                'equity_value': equity_value,
                'equity_percentage': (equity_value / total_value) * 100 if total_value > 0 else 0,
                'day_pnl': day_pnl,
                'day_pnl_pct': day_pnl_pct,
                'position_count': len(positions),
                'average_position_size': sum(position_values) / len(position_values) if position_values else 0,
                'largest_position': max(position_values) if position_values else 0,
                'value_at_risk_1d': self._calculate_var(positions, 1),
                'value_at_risk_5d': self._calculate_var(positions, 5)
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating portfolio metrics: {str(e)}")
            return self._get_default_portfolio_metrics()
    
    def _calculate_var(self, positions: List[Dict], days: int) -> float:
        """Calculate Value at Risk"""
        
        # Simplified VaR calculation
        # In production, would use historical simulation or Monte Carlo
        
        if not positions:
            return 0
        
        # Assume 95% confidence level
        z_score = 1.645
        
        # Estimate portfolio volatility (simplified)
        portfolio_vol = 0.02 * np.sqrt(days)  # 2% daily vol assumption
        
        total_value = sum(abs(pos.get('market_value', 0)) for pos in positions if 'error' not in pos)
        
        var = total_value * portfolio_vol * z_score
        
        return (var / total_value) * 100 if total_value > 0 else 0
    
    async def _calculate_position_risks(self, positions: List[Dict]) -> List[Dict]:
        """Calculate individual position risks"""
        
        position_risks = []
        
        for position in positions:
            if 'error' in position:
                continue
                
            symbol = position['symbol']
            
            # Get recent price data
            try:
                bars = await self.alpaca.get_bars(symbol, timeframe='1Day', limit=20)
                
                if bars and len(bars) > 1:
                    closes = [bar.c for bar in bars]
                    returns = [(closes[i] - closes[i-1])/closes[i-1] for i in range(1, len(closes))]
                    
                    volatility = np.std(returns) * np.sqrt(252) if returns else 0
                    max_drawdown = self._calculate_max_drawdown(closes)
                    
                    position_risks.append({
                        'symbol': symbol,
                        'volatility': volatility,
                        'max_drawdown': max_drawdown,
                        'position_size_pct': (abs(position.get('market_value', 0)) / 
                                            position.get('portfolio_value', 1)) * 100,
                        'risk_contribution': volatility * position.get('qty', 0)
                    })
                    
            except Exception as e:
                self.logger.error(f"Error calculating risk for {symbol}: {str(e)}")
                continue
        
        return position_risks
    
    def _calculate_max_drawdown(self, prices: List[float]) -> float:
        """Calculate maximum drawdown"""
        
        if len(prices) < 2:
            return 0
        
        peak = prices[0]
        max_dd = 0
        
        for price in prices[1:]:
            if price > peak:
                peak = price
            else:
                dd = (peak - price) / peak
                max_dd = max(max_dd, dd)
        
        return max_dd * 100
    
    def _check_risk_violations(self, risk_metrics: Dict, positions: List[Dict]) -> List[Dict]:
        """Check for risk limit violations"""
        
        violations = []
        
        # Check position size limits
        for pos in positions:
            if 'error' in pos:
                continue
                
            pos_size_pct = (abs(pos.get('market_value', 0)) / 
                          risk_metrics.get('total_value', 1)) * 100
            
            if pos_size_pct > self.risk_limits.max_single_position:
                violations.append({
                    'type': 'position_size',
                    'symbol': pos['symbol'],
                    'severity': 'high',
                    'current_value': pos_size_pct,
                    'limit': self.risk_limits.max_single_position,
                    'message': f"{pos['symbol']} exceeds max position size ({pos_size_pct:.1f}% > {self.risk_limits.max_single_position}%)"
                })
        
        # Check sector exposure
        sector_exposure = self._calculate_sector_exposure(positions)
        for sector, exposure in sector_exposure.items():
            if exposure > self.risk_limits.max_sector_exposure:
                violations.append({
                    'type': 'sector_concentration',
                    'sector': sector,
                    'severity': 'medium',
                    'current_value': exposure,
                    'limit': self.risk_limits.max_sector_exposure,
                    'message': f"{sector} exceeds max sector exposure ({exposure:.1f}% > {self.risk_limits.max_sector_exposure}%)"
                })
        
        # Check VaR limits
        var_1d = risk_metrics.get('value_at_risk_1d', 0)
        if var_1d > self.risk_limits.max_daily_var:
            violations.append({
                'type': 'value_at_risk',
                'severity': 'high',
                'current_value': var_1d,
                'limit': self.risk_limits.max_daily_var,
                'message': f'Daily VaR ({var_1d:.1f}%) exceeds limit ({self.risk_limits.max_daily_var}%)'
            })
        
        # Check cash reserves
        cash_pct = risk_metrics.get('cash_percentage', 0)
        if cash_pct < self.risk_limits.min_cash_reserve:
            violations.append({
                'type': 'cash_reserve',
                'severity': 'medium',
                'current_value': cash_pct,
                'limit': self.risk_limits.min_cash_reserve,
                'message': f'Cash reserve ({cash_pct:.1f}%) below minimum ({self.risk_limits.min_cash_reserve}%)'
            })
        
        return violations
    
    def _calculate_sector_exposure(self, positions: List[Dict]) -> Dict[str, float]:
        """Calculate sector exposure percentages"""
        
        # Get total portfolio value
        total_value = sum(abs(pos.get('market_value', 0)) for pos in positions if 'error' not in pos)
        
        if total_value == 0:
            return {}
        
        # Simplified sector mapping
        sector_map = {
            'AAPL': 'Technology', 'MSFT': 'Technology', 'GOOGL': 'Technology', 'AMZN': 'Technology',
            'TSLA': 'Technology', 'META': 'Technology', 'NVDA': 'Technology', 'NFLX': 'Technology',
            'JPM': 'Financials', 'BAC': 'Financials', 'WFC': 'Financials', 'GS': 'Financials',
            'JNJ': 'Healthcare', 'PFE': 'Healthcare', 'UNH': 'Healthcare', 'MRK': 'Healthcare',
            'XOM': 'Energy', 'CVX': 'Energy', 'COP': 'Energy',
            'KO': 'Consumer Staples', 'PG': 'Consumer Staples', 'WMT': 'Consumer Staples'
        }
        
        sector_exposure = {}
        
        for pos in positions:
            if 'error' not in pos:
                symbol = pos['symbol']
                market_value = abs(pos.get('market_value', 0))
                sector = sector_map.get(symbol, 'Other')
                
                exposure_pct = (market_value / total_value) * 100
                sector_exposure[sector] = sector_exposure.get(sector, 0) + exposure_pct
        
        return sector_exposure
    
    def _assess_concentration_risk(self, positions: List[Dict]) -> Dict:
        """Assess portfolio concentration risk"""
        
        if not positions:
            return {'risk_level': 'low', 'concentration_score': 0}
        
        # Calculate Herfindahl-Hirschman Index (HHI) for concentration
        total_value = sum(abs(pos.get('market_value', 0)) for pos in positions if 'error' not in pos)
        
        if total_value == 0:
            return {'risk_level': 'low', 'concentration_score': 0}
        
        hhi = 0
        for pos in positions:
            if 'error' not in pos:
                weight = abs(pos.get('market_value', 0)) / total_value
                hhi += weight ** 2
        
        # Convert to 0-100 scale
        concentration_score = hhi * 100
        
        # Determine risk level
        if concentration_score < 10:
            risk_level = 'low'
        elif concentration_score < 20:
            risk_level = 'medium'
        elif concentration_score < 30:
            risk_level = 'high'
        else:
            risk_level = 'critical'
        
        return {
            'risk_level': risk_level,
            'concentration_score': concentration_score,
            'hhi': hhi,
            'position_count': len(positions),
            'largest_position_weight': max([abs(pos.get('market_value', 0))/total_value 
                                          for pos in positions if 'error' not in pos]) * 100
        }
    
    async def _assess_correlation_risk(self, positions: List[Dict]) -> Dict:
        """Assess correlation risk between positions"""
        
        # Simplified correlation assessment
        # In production, would calculate actual correlation matrix
        
        if len(positions) < 2:
            return {'risk_level': 'low', 'avg_correlation': 0}
        
        # For now, use sector-based correlation proxy
        sector_exposure = self._calculate_sector_exposure(positions)
        
        # High concentration in single sector implies high correlation
        max_sector_exposure = max(sector_exposure.values()) if sector_exposure else 0
        
        if max_sector_exposure > 50:
            return {'risk_level': 'high', 'avg_correlation': 0.7, 'max_sector_exposure': max_sector_exposure}
        elif max_sector_exposure > 30:
            return {'risk_level': 'medium', 'avg_correlation': 0.5, 'max_sector_exposure': max_sector_exposure}
        else:
            return {'risk_level': 'low', 'avg_correlation': 0.3, 'max_sector_exposure': max_sector_exposure}
    
    def _determine_risk_level(self, violations: List[Dict], 
                             concentration_risk: Dict, 
                             correlation_risk: Dict) -> str:
        """Determine overall portfolio risk level"""
        
        # Count severe violations
        high_severity_count = sum(1 for v in violations if v['severity'] == 'high')
        medium_severity_count = sum(1 for v in violations if v['severity'] == 'medium')
        
        # Aggregate risk scores
        risk_score = 0
        
        # Violations contribution
        risk_score += high_severity_count * 30
        risk_score += medium_severity_count * 15
        
        # Concentration contribution
        if concentration_risk['risk_level'] == 'critical':
            risk_score += 30
        elif concentration_risk['risk_level'] == 'high':
            risk_score += 20
        elif concentration_risk['risk_level'] == 'medium':
            risk_score += 10
        
        # Correlation contribution
        if correlation_risk['risk_level'] == 'high':
            risk_score += 20
        elif correlation_risk['risk_level'] == 'medium':
            risk_score += 10
        
        # Determine overall risk level
        if risk_score >= 70:
            return 'critical'
        elif risk_score >= 50:
            return 'high'
        elif risk_score >= 30:
            return 'medium'
        else:
            return 'low'
    
    def _calculate_risk_score(self, risk_level: str) -> float:
        """Convert risk level to numeric score"""
        
        risk_scores = {
            'low': 25,
            'medium': 50,
            'high': 75,
            'critical': 90
        }
        
        return risk_scores.get(risk_level, 50)
    
    def _generate_risk_recommendations(self, risk_level: str, 
                                      violations: List[Dict], 
                                      portfolio_metrics: Dict) -> List[str]:
        """Generate risk management recommendations"""
        
        recommendations = []
        
        if risk_level == 'critical':
            recommendations.append("⚠️ URGENT: Immediately reduce portfolio risk exposure")
            recommendations.append("Consider closing or trimming largest positions")
            recommendations.append("Increase cash allocation to defensive levels")
        
        elif risk_level == 'high':
            recommendations.append("Reduce position sizes in concentrated holdings")
            recommendations.append("Avoid new risk-on positions until risk normalizes")
            recommendations.append("Consider taking profits in winning positions")
        
        # Specific violation recommendations
        for violation in violations:
            if violation['type'] == 'position_size':
                recommendations.append(f"Trim {violation['symbol']} to max {self.risk_limits.max_single_position}% of portfolio")
            elif violation['type'] == 'sector_concentration':
                recommendations.append(f"Reduce {violation['sector']} exposure below {self.risk_limits.max_sector_exposure}%")
            elif violation['type'] == 'cash_reserve':
                recommendations.append(f"Increase cash reserves to minimum {self.risk_limits.min_cash_reserve}%")
        
        # Portfolio metrics recommendations
        if portfolio_metrics.get('value_at_risk_1d', 0) > self.risk_limits.max_daily_var * 0.8:
            recommendations.append("VaR approaching limits - consider hedging strategies")
        
        return recommendations[:5]  # Limit to top 5 recommendations
    
    def _get_default_risk_assessment(self) -> Dict:
        """Get default risk assessment for error cases"""
        
        return {
            'timestamp': datetime.now().isoformat(),
            'risk_level': 'medium',
            'overall_risk_score': 50,
            'portfolio_metrics': self._get_default_portfolio_metrics(),
            'limit_violations': [],
            'position_risks': [],
            'concentration_risk': {'risk_level': 'medium', 'concentration_score': 20},
            'correlation_risk': {'risk_level': 'medium', 'avg_correlation': 0.5},
            'recommendations': ['Unable to assess risk - using conservative defaults'],
            'can_add_risk': False,
            'should_reduce_risk': False,
            'error': 'Using default risk assessment due to data issues'
        }
    
    def _get_default_portfolio_metrics(self) -> Dict:
        """Get default portfolio metrics for error cases"""
        
        return {
            'total_value': 0,
            'cash': 0,
            'cash_percentage': 100,
            'equity_value': 0,
            'equity_percentage': 0,
            'day_pnl': 0,
            'day_pnl_pct': 0,
            'position_count': 0,
            'average_position_size': 0,
            'largest_position': 0,
            'value_at_risk_1d': 0,
            'value_at_risk_5d': 0
        }

# ========================================================================================
# MAIN PORTFOLIO MANAGER AGENT
# ========================================================================================

class PortfolioManagerAgent:
    """
    Portfolio Manager Agent
    Makes strategic portfolio decisions based on research and market conditions
    """
    
    def __init__(self, agent_name: str, llm_provider, config, alpaca_provider):
        """Initialize Portfolio Manager Agent"""
        
        self.agent_name = agent_name
        self.llm = llm_provider
        self.config = config
        self.alpaca = alpaca_provider
        
        # Initialize components
        self.market_analyzer = MarketRegimeAnalyzer(alpaca_provider)
        self.risk_manager = PortfolioRiskManager(alpaca_provider)
        
        # Portfolio management settings
        self.risk_limits = RiskLimits()
        
        # Performance tracking
        self.total_decisions = 0
        self.successful_decisions = 0
        
        # Logging
        self.logger = logging.getLogger(f'portfolio_manager_{agent_name}')
        self.logger.info(f"Portfolio Manager Agent {agent_name} initialized")
    
    async def process(self, task_data: Dict) -> Dict:
        """Process portfolio management request"""
        
        task_type = task_data.get('task_type')
        
        if task_type == 'daily_portfolio_review':
            return await self.daily_portfolio_review(
                task_data.get('senior_analyst_recommendations', []),
                task_data.get('economist_outlook', {}),
                task_data.get('market_conditions', {})
            )
        elif task_type == 'position_sizing':
            return await self.calculate_position_sizing(
                task_data.get('symbol'),
                task_data.get('recommendation', {}),
                task_data.get('portfolio_context', {})
            )
        elif task_type == 'risk_assessment':
            return await self.assess_portfolio_risk()
        else:
            return {'error': f'Unknown task type: {task_type}'}
    
    async def daily_portfolio_review(self, senior_recommendations: List[Dict],
                                    economist_outlook: Dict,
                                    market_conditions: Dict = None) -> Dict:
        """Conduct daily portfolio review and generate trading decisions"""
        
        try:
            self.total_decisions += 1
            
            # Get current portfolio state
            positions = await self._get_current_positions()
            account = await self.alpaca.get_account()
            
            # Analyze market regime if not provided
            if not market_conditions:
                market_conditions = await self.market_analyzer.analyze_market_regime()
            
            # Assess current portfolio risk
            risk_assessment = await self.risk_manager.assess_portfolio_risk(positions)
            
            # Evaluate existing positions
            position_evaluations = await self._evaluate_existing_positions(
                positions, 
                market_conditions, 
                risk_assessment
            )
            
            # Evaluate new opportunities from senior analyst
            opportunity_analysis = await self._analyze_new_opportunities(
                senior_recommendations,
                economist_outlook,
                market_conditions,
                risk_assessment
            )
            
            # Generate portfolio decisions
            portfolio_decisions = await self._generate_portfolio_decisions(
                position_evaluations,
                opportunity_analysis,
                market_conditions,
                risk_assessment,
                economist_outlook
            )
            
            # Validate decisions against risk limits
            validated_decisions = self._validate_decisions_against_risk(
                portfolio_decisions,
                risk_assessment
            )
            
            # Calculate allocation targets
            allocation_targets = self._calculate_allocation_targets(
                market_conditions,
                economist_outlook
            )
            
            # Generate decision reasoning
            decision_reasoning = await self._generate_decision_reasoning(
                validated_decisions,
                market_conditions,
                risk_assessment,
                economist_outlook
            )
            
            self.successful_decisions += 1
            
            # Get market regime value safely
            market_regime_value = 'neutral'
            if 'regime' in market_conditions:
                regime = market_conditions['regime']
                if isinstance(regime, MarketRegime):
                    market_regime_value = regime.value
                elif isinstance(regime, dict) and 'value' in regime:
                    market_regime_value = regime['value']
                elif isinstance(regime, str):
                    market_regime_value = regime
            
            return {
                'timestamp': datetime.now().isoformat(),
                'status': 'success',
                'portfolio_decisions': validated_decisions,
                'allocation_targets': allocation_targets,
                'market_regime': market_regime_value,
                'risk_assessment': risk_assessment,
                'decision_reasoning': decision_reasoning,
                'position_evaluations': position_evaluations,
                'opportunity_analysis': opportunity_analysis,
                'metrics': {
                    'total_decisions': len(validated_decisions),
                    'buy_decisions': sum(1 for d in validated_decisions if d['action'] == 'BUY'),
                    'sell_decisions': sum(1 for d in validated_decisions if d['action'] == 'SELL'),
                    'hold_decisions': sum(1 for d in validated_decisions if d['action'] == 'HOLD')
                }
            }
            
        except Exception as e:
            self.logger.error(f"Daily portfolio review error: {str(e)}")
            import traceback
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            return {
                'status': 'error',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    async def _get_current_positions(self) -> List[Dict]:
        """Get current portfolio positions"""
        
        try:
            positions = await self.alpaca.get_positions()
            
            if not positions:
                return []
            
            formatted_positions = []
            for pos in positions:
                formatted_positions.append({
                    'symbol': pos.symbol,
                    'qty': float(pos.qty),
                    'market_value': float(pos.market_value),
                    'cost_basis': float(pos.cost_basis),
                    'unrealized_pl': float(pos.unrealized_pl),
                    'unrealized_plpc': float(pos.unrealized_plpc),
                    'current_price': float(pos.current_price),
                    'avg_entry_price': float(pos.avg_entry_price)
                })
            
            return formatted_positions
            
        except Exception as e:
            self.logger.error(f"Error getting positions: {str(e)}")
            return []
    
    async def _evaluate_existing_positions(self, positions: List[Dict],
                                          market_conditions: Dict,
                                          risk_assessment: Dict) -> List[Dict]:
        """Evaluate existing positions for hold/trim/add/close decisions"""
        
        evaluations = []
        
        for position in positions:
            try:
                # Get recent data for position
                symbol = position['symbol']
                symbol_data = await self._get_symbol_data(symbol)
                symbol_news = await self._get_symbol_news_sentiment(symbol)
                
                # Generate evaluation using LLM
                evaluation = await self._evaluate_position_with_llm(
                    position,
                    symbol_data,
                    symbol_news,
                    market_conditions,
                    risk_assessment
                )
                
                evaluations.append(evaluation)
                
            except Exception as e:
                self.logger.error(f"Error evaluating position {position.get('symbol')}: {str(e)}")
                evaluations.append({
                    'symbol': position.get('symbol'),
                    'action': 'HOLD',
                    'confidence': 5,
                    'reasoning': 'Unable to evaluate - defaulting to hold',
                    'error': str(e)
                })
        
        return evaluations
    
    async def _get_symbol_data(self, symbol: str) -> Dict:
        """Get recent market data for symbol"""
        
        try:
            bars = await self.alpaca.get_bars(symbol, timeframe='1Day', limit=20)
            
            if not bars:
                return {'error': 'No data available'}
            
            closes = [bar.c for bar in bars]
            volumes = [bar.v for bar in bars]
            
            # Calculate basic metrics
            current_price = closes[-1]
            price_change_1d = ((closes[-1] - closes[-2]) / closes[-2]) * 100 if len(closes) > 1 else 0
            price_change_5d = ((closes[-1] - closes[-5]) / closes[-5]) * 100 if len(closes) > 5 else 0
            price_change_20d = ((closes[-1] - closes[0]) / closes[0]) * 100 if len(closes) == 20 else 0
            
            # Volume analysis
            avg_volume = sum(volumes) / len(volumes)
            volume_ratio = volumes[-1] / avg_volume if avg_volume > 0 else 1
            
            # Simple technical indicators
            sma_20 = sum(closes) / len(closes)
            rsi = self._calculate_rsi(closes)
            
            return {
                'symbol': symbol,
                'current_price': current_price,
                'price_metrics': {
                    'price_change_1d': price_change_1d,
                    'price_change_5d': price_change_5d,
                    'price_change_20d': price_change_20d
                },
                'volume_metrics': {
                    'current_volume': volumes[-1],
                    'avg_volume': avg_volume,
                    'volume_ratio': volume_ratio
                },
                'technical_indicators': {
                    'sma_20': sma_20,
                    'price_vs_sma': ((current_price - sma_20) / sma_20) * 100,
                    'rsi': rsi
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error getting symbol data for {symbol}: {str(e)}")
            return {'error': str(e)}
    
    def _calculate_rsi(self, prices: List[float], period: int = 14) -> float:
        """Calculate RSI indicator"""
        
        if len(prices) < period + 1:
            return 50  # Default neutral RSI
        
        gains = []
        losses = []
        
        for i in range(1, len(prices)):
            change = prices[i] - prices[i-1]
            if change > 0:
                gains.append(change)
                losses.append(0)
            else:
                gains.append(0)
                losses.append(abs(change))
        
        avg_gain = sum(gains[-period:]) / period
        avg_loss = sum(losses[-period:]) / period
        
        if avg_loss == 0:
            if avg_gain == 0:
                return 50  # No movement, neutral RSI
            else:
                return 100  # Only gains, max RSI
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    async def _get_symbol_news_sentiment(self, symbol: str) -> Dict:
        """Get news sentiment for symbol"""
        
        try:
            # Get news from Alpaca
            news = await self.alpaca.get_news(symbol, limit=5)
            
            if not news:
                return {'sentiment_score': 0, 'article_count': 0, 'key_themes': []}
            
            # Simple sentiment scoring (in production, would use NLP)
            positive_keywords = ['upgrade', 'beat', 'strong', 'growth', 'positive', 'outperform']
            negative_keywords = ['downgrade', 'miss', 'weak', 'decline', 'negative', 'underperform']
            
            sentiment_score = 0
            key_themes = []
            
            for article in news:
                headline = article.headline.lower()
                
                # Count positive/negative keywords
                pos_count = sum(1 for keyword in positive_keywords if keyword in headline)
                neg_count = sum(1 for keyword in negative_keywords if keyword in headline)
                
                sentiment_score += (pos_count - neg_count)
                
                # Extract themes (simplified)
                if 'earnings' in headline:
                    key_themes.append('earnings')
                if 'analyst' in headline:
                    key_themes.append('analyst_coverage')
                if 'product' in headline or 'launch' in headline:
                    key_themes.append('product_news')
            
            return {
                'sentiment_score': sentiment_score,
                'article_count': len(news),
                'key_themes': list(set(key_themes))[:3]
            }
            
        except Exception as e:
            self.logger.error(f"Error getting news sentiment for {symbol}: {str(e)}")
            return {'sentiment_score': 0, 'article_count': 0, 'key_themes': [], 'error': str(e)}
    
    async def _evaluate_position_with_llm(self, position: Dict, symbol_data: Dict,
                                         symbol_news: Dict, market_conditions: Dict,
                                         risk_assessment: Dict) -> Dict:
        """Use LLM to evaluate existing position"""
        
        # Calculate position metrics
        unrealized_pnl_pct = position.get('unrealized_plpc', 0) * 100
        current_price = position.get('current_price', 0)
        avg_cost = position.get('avg_entry_price', 0)
        portfolio_value = risk_assessment.get('portfolio_metrics', {}).get('total_value', 100000)
        weight_pct = (abs(position.get('market_value', 0)) / portfolio_value) * 100
        
        # Get market regime value safely
        market_regime_str = 'neutral'
        if 'regime' in market_conditions:
            regime = market_conditions['regime']
            if isinstance(regime, MarketRegime):
                market_regime_str = regime.value
            elif isinstance(regime, dict) and 'value' in regime:
                market_regime_str = regime['value']
            elif isinstance(regime, str):
                market_regime_str = regime
        
        prompt = f"""You are a Portfolio Manager evaluating an existing position.

POSITION DETAILS:
- Symbol: {position.get('symbol')}
- Current Price: ${current_price:.2f}
- Average Cost: ${avg_cost:.2f}
- Unrealized P&L: {unrealized_pnl_pct:+.2f}%
- Portfolio Weight: {weight_pct:.1f}%
- Shares: {position.get('qty', 0)}

RECENT PERFORMANCE:
- 1-Day Change: {symbol_data.get('price_metrics', {}).get('price_change_1d', 0):+.2f}%
- 5-Day Change: {symbol_data.get('price_metrics', {}).get('price_change_5d', 0):+.2f}%
- 20-Day Change: {symbol_data.get('price_metrics', {}).get('price_change_20d', 0):+.2f}%

TECHNICAL INDICATORS:
- RSI: {symbol_data.get('technical_indicators', {}).get('rsi', 50):.1f}
- Price vs SMA20: {symbol_data.get('technical_indicators', {}).get('price_vs_sma', 0):+.2f}%
- Volume Ratio: {symbol_data.get('volume_metrics', {}).get('volume_ratio', 1):.2f}x

NEWS SENTIMENT:
- Sentiment Score: {symbol_news.get('sentiment_score', 0):.2f}
- Recent Articles: {symbol_news.get('article_count', 0)}
- Key Themes: {', '.join(symbol_news.get('key_themes', [])[:3])}

MARKET CONDITIONS:
- Market Regime: {market_regime_str}
- Portfolio Risk Level: {risk_assessment.get('risk_level', 'medium')}

Provide position evaluation in this EXACT JSON format:

{{
    "action": "HOLD/TRIM/ADD/CLOSE",
    "confidence": 1-10,
    "target_weight": float (target weight % for position),
    "reasoning": "Detailed explanation for the recommended action",
    "risk_factors": ["risk1", "risk2"],
    "price_targets": {{
        "upside_target": float,
        "downside_target": float
    }},
    "time_horizon": "short/medium/long",
    "conviction_level": "low/medium/high"
}}

EVALUATION CRITERIA:
- Consider position size relative to conviction
- Assess recent price action and momentum
- Factor in news sentiment and developments
- Evaluate risk/reward at current levels
- Account for broader market conditions

POSITION ACTIONS:
- HOLD: Maintain current position
- TRIM: Reduce position by 25-50%
- ADD: Increase position by 25-50%
- CLOSE: Exit entire position"""

        try:
            response = await self.llm.generate_analysis(prompt, "position_evaluation")
            
            # Parse response
            evaluation = json.loads(response)
            
            # Add symbol and metadata
            evaluation['symbol'] = position.get('symbol')
            evaluation['current_weight'] = weight_pct
            evaluation['unrealized_pnl_pct'] = unrealized_pnl_pct
            
            return evaluation
            
        except Exception as e:
            self.logger.error(f"LLM evaluation error for {position.get('symbol')}: {str(e)}")
            return {
                'symbol': position.get('symbol'),
                'action': 'HOLD',
                'confidence': 5,
                'target_weight': weight_pct,
                'reasoning': 'Unable to evaluate - defaulting to hold',
                'error': str(e)
            }
    
    async def _analyze_new_opportunities(self, senior_recommendations: List[Dict],
                                        economist_outlook: Dict,
                                        market_conditions: Dict,
                                        risk_assessment: Dict) -> List[Dict]:
        """Analyze new opportunities from senior analyst"""
        
        opportunities = []
        
        # Get available cash
        portfolio_metrics = risk_assessment.get('portfolio_metrics', {})
        available_cash = portfolio_metrics.get('cash_percentage', 30)
        
        for recommendation in senior_recommendations[:5]:  # Limit to top 5
            try:
                # Get current market data
                symbol = recommendation.get('symbol')
                symbol_data = await self._get_symbol_data(symbol)
                
                # Analyze opportunity with LLM
                opportunity = await self._analyze_opportunity_with_llm(
                    recommendation,
                    symbol_data,
                    market_conditions,
                    risk_assessment,
                    economist_outlook,
                    available_cash
                )
                
                opportunities.append(opportunity)
                
            except Exception as e:
                self.logger.error(f"Error analyzing opportunity {recommendation.get('symbol')}: {str(e)}")
                opportunities.append({
                    'symbol': recommendation.get('symbol'),
                    'actionable': False,
                    'rejection_reason': f'Analysis error: {str(e)}'
                })
        
        return opportunities
    
    async def _analyze_opportunity_with_llm(self, recommendation: Dict,
                                           symbol_data: Dict,
                                           market_conditions: Dict,
                                           risk_assessment: Dict,
                                           economist_outlook: Dict,
                                           available_cash: float) -> Dict:
        """Use LLM to analyze new opportunity"""
        
        symbol = recommendation.get('symbol')
        
        # Get market regime value safely
        market_regime_str = 'neutral'
        if 'regime' in market_conditions:
            regime = market_conditions['regime']
            if isinstance(regime, MarketRegime):
                market_regime_str = regime.value
            elif isinstance(regime, dict) and 'value' in regime:
                market_regime_str = regime['value']
            elif isinstance(regime, str):
                market_regime_str = regime
        
        prompt = f"""You are a Portfolio Manager evaluating a new opportunity from the Senior Research Analyst.

SENIOR ANALYST RECOMMENDATION:
- Symbol: {symbol}
- Action: {recommendation.get('action', 'BUY')}
- Confidence Score: {recommendation.get('confidence_score', 0)}/10
- Expected Return: {recommendation.get('expected_return', 0):.1f}%
- Time Horizon: {recommendation.get('time_horizon', 'medium')}
- Risk Level: {recommendation.get('risk_level', 'medium')}
- Key Catalysts: {', '.join(recommendation.get('key_catalysts', [])[:3])}

CURRENT MARKET DATA:
- Current Price: ${symbol_data.get('current_price', 0):.2f}
- 5-Day Change: {symbol_data.get('price_metrics', {}).get('price_change_5d', 0):+.2f}%
- RSI: {symbol_data.get('technical_indicators', {}).get('rsi', 50):.1f}
- Volume Ratio: {symbol_data.get('volume_metrics', {}).get('volume_ratio', 1):.2f}x

ECONOMIST OUTLOOK:
- Economic Cycle: {economist_outlook.get('economic_cycle', 'expansion')}
- Market Regime: {market_regime_str}
- Sector Recommendations: {json.dumps(economist_outlook.get('sector_recommendations', {}))}

PORTFOLIO CONTEXT:
- Current Risk Level: {risk_assessment.get('risk_level', 'medium')}
- Available Cash: {available_cash:.1f}%
- Risk Violations: {len(risk_assessment.get('limit_violations', []))}

Provide opportunity analysis in this EXACT JSON format:

{{
    "actionable": true/false,
    "recommended_allocation": float (% of portfolio, 0 if not actionable),
    "position_sizing_rationale": "Detailed reasoning for allocation size",
    "opportunity_score": 1-10,
    "risk_adjusted_allocation": float (allocation adjusted for risk),
    "timing_assessment": "excellent/good/fair/poor",
    "market_timing_rationale": "Why now is good/bad time",
    "portfolio_fit": "excellent/good/fair/poor",
    "diversification_impact": "positive/neutral/negative",
    "rejection_reason": "string (if not actionable)",
    "entry_criteria": "Conditions for position initiation",
    "risk_considerations": ["risk1", "risk2"]
}}

ALLOCATION GUIDELINES:
- High conviction (8-10): 3-5% allocation
- Medium conviction (5-7): 1.5-3% allocation
- Low conviction (1-4): 0.5-1.5% allocation
- Adjust for market regime and portfolio risk

REJECT OPPORTUNITY IF:
- Portfolio risk is already elevated
- Insufficient differentiation from existing positions
- Poor market timing for the opportunity
- Allocation would exceed risk limits"""

        try:
            response = await self.llm.generate_analysis(prompt, "opportunity_analysis")
            
            # Parse response
            opportunity = json.loads(response)
            
            # Add metadata
            opportunity['symbol'] = symbol
            opportunity['senior_confidence'] = recommendation.get('confidence_score', 0)
            opportunity['expected_return'] = recommendation.get('expected_return', 0)
            
            return opportunity
            
        except Exception as e:
            self.logger.error(f"LLM opportunity analysis error for {symbol}: {str(e)}")
            return {
                'symbol': symbol,
                'actionable': False,
                'rejection_reason': f'Analysis error: {str(e)}',
                'error': str(e)
            }
    
    async def _generate_portfolio_decisions(self, position_evaluations: List[Dict],
                                           opportunity_analysis: List[Dict],
                                           market_conditions: Dict,
                                           risk_assessment: Dict,
                                           economist_outlook: Dict) -> List[Dict]:
        """Generate final portfolio decisions"""
        
        decisions = []
        
        # Process existing position decisions
        for evaluation in position_evaluations:
            if evaluation.get('action') != 'HOLD':
                decisions.append({
                    'symbol': evaluation['symbol'],
                    'action': evaluation['action'],
                    'type': 'existing_position',
                    'target_weight': evaluation.get('target_weight', 0),
                    'current_weight': evaluation.get('current_weight', 0),
                    'confidence': evaluation.get('confidence', 5),
                    'reasoning': evaluation.get('reasoning', ''),
                    'time_horizon': evaluation.get('time_horizon', 'medium'),
                    'priority': self._calculate_decision_priority(evaluation)
                })
        
        # Process new opportunities
        for opportunity in opportunity_analysis:
            if opportunity.get('actionable'):
                decisions.append({
                    'symbol': opportunity['symbol'],
                    'action': 'BUY',
                    'type': 'new_position',
                    'target_weight': opportunity.get('recommended_allocation', 0),
                    'current_weight': 0,
                    'confidence': opportunity.get('opportunity_score', 5),
                    'reasoning': opportunity.get('position_sizing_rationale', ''),
                    'time_horizon': 'medium',
                    'entry_criteria': opportunity.get('entry_criteria', ''),
                    'priority': opportunity.get('opportunity_score', 5)
                })
        
        # Sort by priority
        decisions.sort(key=lambda x: x['priority'], reverse=True)
        
        # Apply portfolio constraints
        decisions = self._apply_portfolio_constraints(decisions, risk_assessment)
        
        return decisions
    
    def _calculate_decision_priority(self, evaluation: Dict) -> float:
        """Calculate priority score for decision"""
        
        base_priority = evaluation.get('confidence', 5)
        
        # Adjust for action type
        if evaluation.get('action') == 'CLOSE':
            base_priority += 2  # Prioritize exits
        elif evaluation.get('action') == 'TRIM':
            base_priority += 1  # Prioritize risk reduction
        
        # Adjust for P&L
        if evaluation.get('unrealized_pnl_pct', 0) < -10:
            base_priority += 1  # Prioritize losing positions
        
        return base_priority
    
    def _apply_portfolio_constraints(self, decisions: List[Dict],
                                    risk_assessment: Dict) -> List[Dict]:
        """Apply portfolio constraints to decisions"""
        
        constrained_decisions = []
        portfolio_metrics = risk_assessment.get('portfolio_metrics', {})
        
        # Calculate total allocation for new positions
        total_new_allocation = sum(d['target_weight'] for d in decisions 
                                  if d['type'] == 'new_position')
        
        # Get available capital
        available_cash = portfolio_metrics.get('cash_percentage', 30)
        
        # Scale down allocations if exceeding available cash
        if total_new_allocation > available_cash * 0.8:  # Keep 20% cash buffer
            scale_factor = (available_cash * 0.8) / total_new_allocation
            
            for decision in decisions:
                if decision['type'] == 'new_position':
                    decision['target_weight'] *= scale_factor
                    decision['reasoning'] += f" (Scaled down due to cash constraints)"
        
        # Apply position size limits
        for decision in decisions:
            if decision['target_weight'] > self.risk_limits.max_single_position:
                decision['target_weight'] = self.risk_limits.max_single_position
                decision['reasoning'] += f" (Capped at max position size)"
        
        return decisions
    
    def _validate_decisions_against_risk(self, decisions: List[Dict],
                                        risk_assessment: Dict) -> List[Dict]:
        """Validate decisions against risk limits"""
        
        validated_decisions = []
        current_risk_level = risk_assessment.get('risk_level', 'medium')
        violations = risk_assessment.get('limit_violations', [])
        
        for decision in decisions:
            # Skip or modify decisions based on risk level
            if current_risk_level in ['high', 'critical']:
                if decision['type'] == 'new_position':
                    # Reduce allocation for new positions
                    decision['target_weight'] *= 0.7
                    decision['reasoning'] += f" (Reduced due to elevated portfolio risk: {current_risk_level})"
                elif decision['action'] == 'ADD':
                    # Convert ADD to HOLD if risk is high
                    decision['action'] = 'HOLD'
                    decision['reasoning'] += f" (Changed from ADD to HOLD due to risk level: {current_risk_level})"
            
            # Check for specific violations
            for violation in violations:
                if violation['type'] == 'position_size' and decision['action'] in ['ADD', 'BUY']:
                    # Reduce target weight if position size limits are being violated
                    decision['target_weight'] = min(decision['target_weight'], 
                                                   self.risk_limits.max_single_position)
                    decision['reasoning'] += " (Capped due to position size limits)"
            
            validated_decisions.append(decision)
        
        return validated_decisions
    
    def _calculate_allocation_targets(self, market_conditions: Dict,
                                     economist_outlook: Dict) -> Dict:
        """Calculate optimal portfolio allocation targets"""
        
        # Get base allocation from market regime
        posture_rec = market_conditions.get('posture_recommendation', {})
        
        # Adjust based on economist outlook
        economic_cycle = economist_outlook.get('economic_cycle', 'expansion')
        asset_allocation = economist_outlook.get('asset_allocation', {})
        
        # Synthesize allocations
        if economic_cycle in ['contraction', 'trough']:
            # More defensive in contraction
            equity_target = min(posture_rec.get('equity_target', 70) - 10, 60)
            cash_target = 100 - equity_target
        elif economic_cycle == 'peak':
            # Moderate at peak
            equity_target = posture_rec.get('equity_target', 70)
            cash_target = posture_rec.get('cash_target', 30)
        else:  # expansion, recovery
            # More aggressive in expansion
            equity_target = min(posture_rec.get('equity_target', 70) + 5, 85)
            cash_target = 100 - equity_target
        
        return {
            'cash_target': cash_target,
            'equity_target': equity_target,
            'style_bias': posture_rec.get('style_bias', 'balanced'),
            'sector_preference': economist_outlook.get('sector_recommendations', {}),
            'market_regime': market_conditions.get('regime', MarketRegime.NEUTRAL).value,
            'economic_cycle': economic_cycle,
            'last_updated': datetime.now().isoformat()
        }
    
    async def _generate_decision_reasoning(self, decisions: List[Dict],
                                          market_conditions: Dict,
                                          risk_assessment: Dict,
                                          economist_outlook: Dict) -> str:
        """Generate comprehensive reasoning for portfolio decisions"""
        
        # Get market regime value safely
        market_regime_str = 'neutral'
        if 'regime' in market_conditions:
            regime = market_conditions['regime']
            if isinstance(regime, MarketRegime):
                market_regime_str = regime.value
            elif isinstance(regime, dict) and 'value' in regime:
                market_regime_str = regime['value']
            elif isinstance(regime, str):
                market_regime_str = regime
        
        # Get posture value safely
        posture_str = 'balanced'
        posture_rec = market_conditions.get('posture_recommendation', {})
        if 'posture' in posture_rec:
            posture = posture_rec['posture']
            if hasattr(posture, 'value'):
                posture_str = posture.value
            elif isinstance(posture, str):
                posture_str = posture
        
        # Format decisions for prompt
        decisions_summary = []
        for d in decisions[:5]:
            decisions_summary.append({
                'symbol': d.get('symbol'),
                'action': d.get('action'),
                'target_weight': d.get('target_weight'),
                'confidence': d.get('confidence')
            })
        
        # Format themes safely
        themes_list = []
        for theme in economist_outlook.get('dominant_themes', [])[:3]:
            if isinstance(theme, dict):
                themes_list.append(theme.get('theme_name', ''))
            else:
                themes_list.append(str(theme))
        
        prompt = f"""You are a Portfolio Manager explaining today's portfolio decisions.

MARKET CONDITIONS:
- Regime: {market_regime_str}
- Confidence: {market_conditions.get('confidence', 50):.1f}%
- Posture: {posture_str}

ECONOMIST OUTLOOK:
- Economic Cycle: {economist_outlook.get('economic_cycle', 'expansion')}
- Growth Outlook: {economist_outlook.get('growth_outlook', 'moderate')}
- Inflation Outlook: {economist_outlook.get('inflation_outlook', 'stable')}
- Key Themes: {', '.join(themes_list)}

RISK ASSESSMENT:
- Risk Level: {risk_assessment.get('risk_level', 'medium')}
- Risk Score: {risk_assessment.get('overall_risk_score', 50)}/100
- Violations: {len(risk_assessment.get('limit_violations', []))}

PORTFOLIO DECISIONS ({len(decisions)}):
{json.dumps(decisions_summary, indent=2)}

Provide a comprehensive but concise explanation (3-4 paragraphs) covering:

1. Market Context: How current market conditions and economic outlook influenced decisions
2. Risk Management: How portfolio risk considerations shaped position sizing
3. Key Actions: Rationale for the most important buy/sell/trim decisions
4. Forward Outlook: What we're watching for potential adjustments

Format as clear, professional commentary suitable for stakeholders."""

        try:
            reasoning = await self.llm.generate_analysis(prompt, "decision_reasoning")
            return reasoning
        except Exception as e:
            self.logger.error(f"Error generating decision reasoning: {str(e)}")
            return self._generate_fallback_reasoning(decisions, market_conditions, risk_assessment)
    
    def _generate_fallback_reasoning(self, decisions: List[Dict],
                                    market_conditions: Dict,
                                    risk_assessment: Dict) -> str:
        """Generate fallback reasoning if LLM fails"""
        
        regime = market_conditions.get('regime', MarketRegime.NEUTRAL).value
        risk_level = risk_assessment.get('risk_level', 'medium')
        
        buy_count = sum(1 for d in decisions if d['action'] == 'BUY')
        sell_count = sum(1 for d in decisions if d['action'] in ['SELL', 'CLOSE'])
        trim_count = sum(1 for d in decisions if d['action'] == 'TRIM')
        
        reasoning = f"""Today's portfolio decisions reflect a {regime} market regime with {risk_level} portfolio risk levels.

We are executing {len(decisions)} total actions: {buy_count} new positions, {sell_count} exits, and {trim_count} position reductions. 
These decisions balance opportunity capture with risk management given current market conditions.

Position sizing has been calibrated to maintain appropriate diversification while respecting our risk limits. 
We continue to monitor market developments and will adjust positioning as conditions evolve."""
        
        return reasoning
    
    async def calculate_position_sizing(self, symbol: str, recommendation: Dict,
                                       portfolio_context: Dict) -> Dict:
        """Calculate optimal position size using Kelly Criterion variant"""
        
        try:
            # Get current portfolio metrics
            portfolio_value = portfolio_context.get('total_value', 100000)
            available_cash = portfolio_context.get('cash', 30000)
            
            # Extract recommendation parameters
            expected_return = recommendation.get('expected_return', 0) / 100
            confidence = recommendation.get('confidence_score', 5) / 10
            risk_level = recommendation.get('risk_level', 'medium')
            
            # Get symbol volatility
            symbol_data = await self._get_symbol_data(symbol)
            
            # Calculate base Kelly fraction
            win_probability = 0.5 + (confidence * 0.3)  # Convert confidence to probability
            loss_probability = 1 - win_probability
            
            # Simplified Kelly: f = (p*b - q) / b
            # where p = win probability, q = loss probability, b = win/loss ratio
            win_loss_ratio = 2.0  # Assume 2:1 reward/risk
            kelly_fraction = (win_probability * win_loss_ratio - loss_probability) / win_loss_ratio
            
            # Apply safety factor (use 25% of Kelly)
            safe_kelly = kelly_fraction * 0.25
            
            # Apply risk-based adjustment
            risk_multiplier = {
                'low': 1.2,
                'medium': 1.0,
                'high': 0.7,
                'very_high': 0.5
            }.get(risk_level, 1.0)
            
            adjusted_fraction = safe_kelly * risk_multiplier
            
            # Apply portfolio constraints
            max_position = self.risk_limits.max_single_position / 100
            final_fraction = min(adjusted_fraction, max_position)
            
            # Calculate position size
            position_value = portfolio_value * final_fraction
            current_price = symbol_data.get('current_price', 100)
            shares = int(position_value / current_price)
            
            return {
                'symbol': symbol,
                'recommended_shares': shares,
                'position_value': position_value,
                'portfolio_percentage': final_fraction * 100,
                'kelly_fraction': kelly_fraction,
                'adjusted_fraction': final_fraction,
                'sizing_factors': {
                    'confidence': confidence,
                    'expected_return': expected_return,
                    'risk_level': risk_level,
                    'win_probability': win_probability,
                    'safety_factor': 0.25
                },
                'constraints_applied': [
                    f"Max position size: {self.risk_limits.max_single_position}%",
                    f"Kelly safety factor: 25%",
                    f"Risk adjustment: {risk_multiplier}x"
                ]
            }
            
        except Exception as e:
            self.logger.error(f"Position sizing error for {symbol}: {str(e)}")
            return {
                'symbol': symbol,
                'error': str(e),
                'recommended_shares': 0,
                'position_value': 0,
                'portfolio_percentage': 0
            }
    
    async def assess_portfolio_risk(self) -> Dict:
        """Assess current portfolio risk"""
        
        try:
            positions = await self._get_current_positions()
            risk_assessment = await self.risk_manager.assess_portfolio_risk(positions)
            
            return risk_assessment
            
        except Exception as e:
            self.logger.error(f"Portfolio risk assessment error: {str(e)}")
            return {
                'status': 'error',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def get_performance_metrics(self) -> Dict:
        """Get agent performance metrics"""
        
        return {
            'agent_name': self.agent_name,
            'total_decisions': self.total_decisions,
            'successful_decisions': self.successful_decisions,
            'success_rate': self.successful_decisions / max(1, self.total_decisions),
            'last_update': datetime.now().isoformat()
        }

# ========================================================================================
# TESTING AND DEMONSTRATION
# ========================================================================================

async def test_portfolio_manager():
    """Test the Portfolio Manager Agent functionality"""
    
    # Mock providers
    class MockLLMProvider:
        async def generate_analysis(self, prompt, context):
            # Return mock JSON responses
            if "position evaluation" in prompt:
                return json.dumps({
                    "action": "HOLD",
                    "confidence": 7,
                    "target_weight": 3.5,
                    "reasoning": "Position showing strength",
                    "risk_factors": ["market_volatility"],
                    "price_targets": {"upside_target": 150, "downside_target": 120},
                    "time_horizon": "medium",
                    "conviction_level": "medium"
                })
            elif "opportunity analysis" in prompt:
                return json.dumps({
                    "actionable": True,
                    "recommended_allocation": 3.0,
                    "position_sizing_rationale": "High conviction opportunity",
                    "opportunity_score": 8,
                    "risk_adjusted_allocation": 2.5,
                    "timing_assessment": "good",
                    "market_timing_rationale": "Technical setup favorable",
                    "portfolio_fit": "good",
                    "diversification_impact": "positive",
                    "entry_criteria": "Break above resistance",
                    "risk_considerations": ["earnings_risk"]
                })
            else:
                return "Portfolio decisions reflect balanced approach to current market conditions."
    
    class MockAlpacaProvider:
        async def get_account(self):
            class Account:
                portfolio_value = 100000
                cash = 30000
                equity = 100000
                last_equity = 98000
            return Account()
        
        async def get_positions(self):
            return [
                type('Position', (), {
                    'symbol': 'AAPL',
                    'qty': 100,
                    'market_value': 15000,
                    'cost_basis': 14000,
                    'unrealized_pl': 1000,
                    'unrealized_plpc': 0.0714,
                    'current_price': 150,
                    'avg_entry_price': 140
                })()
            ]
        
        async def get_bars(self, symbol, timeframe='1Day', limit=20):
            import random
            base_price = 100
            bars = []
            for i in range(limit):
                bars.append(type('Bar', (), {
                    'c': base_price + random.uniform(-5, 5),
                    'v': random.randint(1000000, 5000000)
                })())
            return bars
        
        async def get_news(self, symbol, limit=5):
            return [
                type('News', (), {
                    'headline': f'Positive outlook for {symbol} amid strong earnings'
                })()
            ]
    
    # Create test instance
    config = type('Config', (), {})()
    llm_provider = MockLLMProvider()
    alpaca_provider = MockAlpacaProvider()
    
    portfolio_manager = PortfolioManagerAgent(
        agent_name='pm_test',
        llm_provider=llm_provider,
        config=config,
        alpaca_provider=alpaca_provider
    )
    
    print("Testing Portfolio Manager Agent...")
    
    # Test daily portfolio review
    test_recommendations = [
        {
            'symbol': 'MSFT',
            'action': 'BUY',
            'confidence_score': 8,
            'expected_return': 15,
            'time_horizon': 'medium',
            'risk_level': 'medium',
            'key_catalysts': ['AI growth', 'Cloud expansion']
        }
    ]
    
    test_economist_outlook = {
        'economic_cycle': 'expansion',
        'growth_outlook': 'moderate_growth',
        'inflation_outlook': 'stable_inflation',
        'market_regime': 'risk_on',
        'dominant_themes': [
            {'theme_name': 'AI Revolution'},
            {'theme_name': 'Green Transition'}
        ],
        'sector_recommendations': {
            'Technology': 'overweight',
            'Financials': 'neutral'
        },
        'asset_allocation': {
            'equities': 70,
            'bonds': 20,
            'cash': 10
        }
    }
    
    result = await portfolio_manager.daily_portfolio_review(
        senior_recommendations=test_recommendations,
        economist_outlook=test_economist_outlook
    )
    
    print(f"\n✅ Daily Portfolio Review Complete")
    print(f"Status: {result['status']}")
    print(f"Market Regime: {result.get('market_regime', 'unknown')}")
    print(f"Total Decisions: {result.get('metrics', {}).get('total_decisions', 0)}")
    
    print(f"\nPortfolio Decisions:")
    for decision in result.get('portfolio_decisions', [])[:3]:
        print(f"  - {decision['action']} {decision['symbol']} "
              f"(Target: {decision['target_weight']:.1f}%, "
              f"Confidence: {decision['confidence']}/10)")
    
    print(f"\nAllocation Targets:")
    targets = result.get('allocation_targets', {})
    print(f"  Cash: {targets.get('cash_target', 0):.1f}%")
    print(f"  Equity: {targets.get('equity_target', 0):.1f}%")
    print(f"  Economic Cycle: {targets.get('economic_cycle', 'unknown')}")
    
    print("\n✅ Portfolio Manager test completed successfully!")

if __name__ == "__main__":
    import asyncio
    asyncio.run(test_portfolio_manager())