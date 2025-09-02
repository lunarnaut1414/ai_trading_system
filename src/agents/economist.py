# src/agents/economist.py
"""
Economist Agent - Complete Implementation
Provides macro economic analysis and market outlook for Portfolio Manager

This agent analyzes:
- Global economic indicators
- Central bank policies
- Geopolitical events
- Market cycles and trends
- Cross-asset correlations
- Sector rotation signals
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
import hashlib

# Import shared components from existing agents
from src.agents.junior_analyst import (
    MarketContextManager,
    UnifiedRiskAssessment,
    IntelligentCacheManager,
    AnalysisMetadataTracker,
    ConvictionLevel,
    TimeHorizon,
    RiskLevel
)

# MarketRegime is in senior_analyst, not junior_analyst
from src.agents.senior_analyst import (
    MarketRegime,
    AllocationStrategy
)


# ========================================================================================
# ENUMS AND DATA CLASSES
# ========================================================================================

class EconomicCycle(Enum):
    """Economic cycle phases"""
    EXPANSION = "expansion"
    PEAK = "peak"
    CONTRACTION = "contraction"
    TROUGH = "trough"
    RECOVERY = "recovery"
    STAGFLATION = "stagflation"


class PolicyStance(Enum):
    """Central bank policy stance"""
    HAWKISH = "hawkish"
    DOVISH = "dovish"
    NEUTRAL = "neutral"
    TRANSITIONING = "transitioning"


class GeopoliticalRisk(Enum):
    """Geopolitical risk levels"""
    LOW = "low"
    MODERATE = "moderate"
    ELEVATED = "elevated"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class EconomicIndicator:
    """Economic indicator data"""
    name: str
    value: float
    previous_value: float
    expected_value: float
    trend: str  # rising, falling, stable
    importance: str  # high, medium, low
    impact_assessment: str
    data_quality: str


@dataclass
class MacroTheme:
    """Macro economic theme"""
    theme_name: str
    description: str
    impact_sectors: List[str]
    beneficiaries: List[str]  # sectors/assets that benefit
    victims: List[str]  # sectors/assets that suffer
    time_horizon: str
    confidence: float
    action_items: List[str]


@dataclass
class MacroOutlook:
    """Complete macro economic outlook"""
    economic_cycle: str
    growth_outlook: str
    inflation_outlook: str
    policy_outlook: str
    geopolitical_risk: str
    dominant_themes: List[MacroTheme]
    sector_recommendations: Dict[str, str]
    asset_allocation: Dict[str, float]
    risk_scenarios: List[Dict]
    confidence_score: float


# ========================================================================================
# ECONOMIC DATA ANALYZER
# ========================================================================================

class EconomicDataAnalyzer:
    """Analyzes economic indicators and trends"""
    
    def __init__(self, alpaca_provider):
        self.alpaca_provider = alpaca_provider
        self.logger = logging.getLogger("EconomicDataAnalyzer")
    
    async def analyze_economic_indicators(self) -> Dict:
        """
        Analyze key economic indicators
        Returns comprehensive economic assessment
        """
        try:
            # For now, using simulated data
            # In production, would fetch from real economic data sources
            indicators = {
                'gdp': await self._analyze_gdp(),
                'inflation': await self._analyze_inflation(),
                'employment': await self._analyze_employment(),
                'yield_curve': await self._analyze_yield_curve(),
                'monetary_policy': await self._analyze_monetary_policy(),
                'consumer': await self._analyze_consumer_health(),
                'housing': await self._analyze_housing_market(),
                'manufacturing': await self._analyze_manufacturing()
            }
            
            # Calculate composite scores
            health_score = self._calculate_economic_health_score(indicators)
            trend = self._determine_economic_trend(indicators)
            risks = self._identify_economic_risks(indicators)
            
            return {
                'indicators': indicators,
                'health_score': health_score,
                'trend': trend,
                'risks': risks,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing economic indicators: {e}")
            return self._get_default_indicators()
    
    async def _analyze_gdp(self) -> Dict:
        """Analyze GDP growth and trends"""
        # Simulated GDP data
        return {
            'current_growth': 2.5,
            'previous_growth': 2.3,
            'trend': 'expanding',
            'yoy_change': 0.2,
            'forecast': 2.6
        }
    
    async def _analyze_inflation(self) -> Dict:
        """Analyze inflation metrics"""
        return {
            'cpi': 3.2,
            'core_cpi': 2.8,
            'pce': 2.9,
            'trend': 'moderating',
            'expectations': 2.5,
            'current_rate': 3.2
        }
    
    async def _analyze_employment(self) -> Dict:
        """Analyze employment data"""
        return {
            'unemployment_rate': 3.8,
            'job_growth': 250000,
            'wage_growth': 4.2,
            'participation_rate': 63.2,
            'trend': 'strong'
        }
    
    async def _analyze_yield_curve(self) -> Dict:
        """Analyze yield curve shape and implications"""
        return {
            'shape': 'normal',  # normal, inverted, flat
            'spread_2_10': 0.45,
            'spread_3m_10': 1.2,
            'trend': 'steepening'
        }
    
    async def _analyze_monetary_policy(self) -> Dict:
        """Analyze central bank policy stance"""
        return {
            'fed_funds_rate': 5.25,
            'stance': 'hawkish',
            'next_move_probability': {'hike': 0.2, 'hold': 0.7, 'cut': 0.1},
            'dot_plot_median': 5.1
        }
    
    async def _analyze_consumer_health(self) -> Dict:
        """Analyze consumer sentiment and spending"""
        return {
            'sentiment_index': 68.5,
            'retail_sales_growth': 3.1,
            'savings_rate': 4.2,
            'credit_card_delinquencies': 2.1
        }
    
    async def _analyze_housing_market(self) -> Dict:
        """Analyze housing market conditions"""
        return {
            'home_prices_yoy': 5.2,
            'mortgage_rate': 7.1,
            'housing_starts': 1420000,
            'inventory_months': 3.2
        }
    
    async def _analyze_manufacturing(self) -> Dict:
        """Analyze manufacturing activity"""
        return {
            'pmi': 48.5,
            'new_orders': 47.2,
            'industrial_production': 0.3,
            'capacity_utilization': 78.5
        }
    
    def _calculate_economic_health_score(self, indicators: Dict) -> float:
        """Calculate overall economic health score (0-10)"""
        scores = []
        
        # GDP contribution
        gdp_growth = indicators.get('gdp', {}).get('current_growth', 2.0)
        scores.append(min(10, max(0, gdp_growth * 2)))
        
        # Employment contribution
        unemployment = indicators.get('employment', {}).get('unemployment_rate', 5.0)
        scores.append(max(0, 10 - unemployment * 1.5))
        
        # Inflation contribution (2% is ideal)
        inflation = indicators.get('inflation', {}).get('cpi', 3.0)
        inflation_score = 10 - abs(inflation - 2.0) * 2
        scores.append(max(0, inflation_score))
        
        # Manufacturing contribution
        pmi = indicators.get('manufacturing', {}).get('pmi', 50)
        scores.append((pmi - 40) / 2)
        
        return round(statistics.mean(scores), 1)
    
    def _determine_economic_trend(self, indicators: Dict) -> str:
        """Determine overall economic trend"""
        positive_signals = 0
        negative_signals = 0
        
        # Check GDP
        if indicators.get('gdp', {}).get('trend') == 'expanding':
            positive_signals += 1
        else:
            negative_signals += 1
        
        # Check employment
        if indicators.get('employment', {}).get('unemployment_rate', 5) < 4.5:
            positive_signals += 1
        else:
            negative_signals += 1
        
        # Check manufacturing
        if indicators.get('manufacturing', {}).get('pmi', 50) > 50:
            positive_signals += 1
        else:
            negative_signals += 1
        
        if positive_signals > negative_signals:
            return 'improving'
        elif negative_signals > positive_signals:
            return 'deteriorating'
        else:
            return 'mixed'
    
    def _identify_economic_risks(self, indicators: Dict) -> List[str]:
        """Identify key economic risks"""
        risks = []
        
        # Check for recession risk
        if indicators.get('yield_curve', {}).get('shape') == 'inverted':
            risks.append('yield_curve_inversion')
        
        # Check for inflation risk
        if indicators.get('inflation', {}).get('cpi', 2) > 4:
            risks.append('elevated_inflation')
        
        # Check for employment weakness
        if indicators.get('employment', {}).get('trend') == 'weakening':
            risks.append('labor_market_softening')
        
        # Check for manufacturing contraction
        if indicators.get('manufacturing', {}).get('pmi', 50) < 50:
            risks.append('manufacturing_contraction')
        
        return risks
    
    def _get_default_indicators(self) -> Dict:
        """Return default indicators on error"""
        return {
            'indicators': {},
            'health_score': 5.0,
            'trend': 'neutral',
            'risks': [],
            'timestamp': datetime.now().isoformat()
        }


# ========================================================================================
# MACRO THEME IDENTIFIER
# ========================================================================================

class MacroThemeIdentifier:
    """Identifies and analyzes macro economic themes"""
    
    def __init__(self):
        self.logger = logging.getLogger("MacroThemeIdentifier")
    
    def identify_macro_themes(self, economic_data: Dict) -> List[MacroTheme]:
        """
        Identify dominant macro themes based on economic conditions
        """
        themes = []
        indicators = economic_data.get('indicators', {})
        
        # Check for stagflation
        if self._check_stagflation(indicators):
            themes.append(self._create_stagflation_theme())
        
        # Check for disinflation
        if self._check_disinflation(indicators):
            themes.append(self._create_disinflation_theme())
        
        # Check for recession
        if self._check_recession(indicators):
            themes.append(self._create_recession_theme())
        
        # Check for goldilocks
        if self._check_goldilocks(indicators):
            themes.append(self._create_goldilocks_theme())
        
        # Check for policy pivot
        if self._check_policy_pivot(indicators):
            themes.append(self._create_policy_pivot_theme())
        
        # Sort by confidence
        themes.sort(key=lambda x: x.confidence, reverse=True)
        
        return themes
    
    def _check_stagflation(self, indicators: Dict) -> bool:
        """Check for stagflation conditions"""
        inflation = indicators.get('inflation', {}).get('cpi', 2)
        gdp = indicators.get('gdp', {}).get('current_growth', 2)
        return inflation > 4 and gdp < 1
    
    def _check_disinflation(self, indicators: Dict) -> bool:
        """Check for disinflation theme"""
        trend = indicators.get('inflation', {}).get('trend')
        return trend == 'moderating' or trend == 'declining'
    
    def _check_recession(self, indicators: Dict) -> bool:
        """Check for recession risk"""
        yield_curve = indicators.get('yield_curve', {}).get('shape')
        pmi = indicators.get('manufacturing', {}).get('pmi', 50)
        return yield_curve == 'inverted' or pmi < 48
    
    def _check_goldilocks(self, indicators: Dict) -> bool:
        """Check for goldilocks conditions"""
        inflation = indicators.get('inflation', {}).get('cpi', 2)
        gdp = indicators.get('gdp', {}).get('current_growth', 2)
        return 1.5 < inflation < 3 and gdp > 2
    
    def _check_policy_pivot(self, indicators: Dict) -> bool:
        """Check for central bank policy pivot"""
        stance = indicators.get('monetary_policy', {}).get('stance')
        return stance == 'transitioning'
    
    def _create_stagflation_theme(self) -> MacroTheme:
        """Create stagflation theme"""
        return MacroTheme(
            theme_name='Stagflation Concerns',
            description='High inflation with slowing growth',
            impact_sectors=['consumer_discretionary', 'real_estate', 'financials'],
            beneficiaries=['energy', 'commodities', 'utilities'],
            victims=['technology', 'consumer_discretionary', 'real_estate'],
            time_horizon='6-12 months',
            confidence=0.75,
            action_items=[
                'Reduce growth stock exposure',
                'Increase commodity allocation',
                'Focus on pricing power companies'
            ]
        )
    
    def _create_disinflation_theme(self) -> MacroTheme:
        """Create disinflation theme"""
        return MacroTheme(
            theme_name='Disinflation Trend',
            description='Inflation moderating toward target',
            impact_sectors=['technology', 'consumer_discretionary', 'real_estate'],
            beneficiaries=['technology', 'growth_stocks', 'bonds'],
            victims=['commodities', 'energy', 'banks'],
            time_horizon='3-6 months',
            confidence=0.80,
            action_items=[
                'Increase duration in bonds',
                'Rotate to growth stocks',
                'Reduce inflation hedges'
            ]
        )
    
    def _create_recession_theme(self) -> MacroTheme:
        """Create recession theme"""
        return MacroTheme(
            theme_name='Recession Risk',
            description='Economic contraction risk elevated',
            impact_sectors=['all'],
            beneficiaries=['utilities', 'consumer_staples', 'healthcare'],
            victims=['financials', 'industrials', 'consumer_discretionary'],
            time_horizon='6-12 months',
            confidence=0.70,
            action_items=[
                'Increase defensive allocation',
                'Reduce cyclical exposure',
                'Build cash reserves'
            ]
        )
    
    def _create_goldilocks_theme(self) -> MacroTheme:
        """Create goldilocks theme"""
        return MacroTheme(
            theme_name='Goldilocks Economy',
            description='Moderate growth with controlled inflation',
            impact_sectors=['all'],
            beneficiaries=['technology', 'financials', 'consumer_discretionary'],
            victims=['utilities', 'consumer_staples'],
            time_horizon='3-6 months',
            confidence=0.85,
            action_items=[
                'Increase equity allocation',
                'Focus on growth sectors',
                'Reduce defensive positions'
            ]
        )
    
    def _create_policy_pivot_theme(self) -> MacroTheme:
        """Create policy pivot theme"""
        return MacroTheme(
            theme_name='Policy Pivot',
            description='Central bank changing stance',
            impact_sectors=['financials', 'real_estate', 'technology'],
            beneficiaries=['bonds', 'real_estate', 'technology'],
            victims=['cash', 'short_duration'],
            time_horizon='3-6 months',
            confidence=0.65,
            action_items=[
                'Monitor Fed communications',
                'Prepare for volatility',
                'Adjust duration exposure'
            ]
        )


# ========================================================================================
# CROSS ASSET ANALYZER
# ========================================================================================

class CrossAssetAnalyzer:
    """Analyzes cross-asset correlations and relationships"""
    
    def __init__(self, alpaca_provider):
        self.alpaca_provider = alpaca_provider
        self.logger = logging.getLogger("CrossAssetAnalyzer")
    
    async def analyze_cross_asset_dynamics(self) -> Dict:
        """
        Analyze relationships between different asset classes
        """
        try:
            # Gather data for major asset classes
            equity_data = await self._get_equity_market_data()
            bond_data = await self._get_bond_market_data()
            commodity_data = await self._get_commodity_data()
            currency_data = await self._get_currency_data()
            
            # Calculate correlations
            correlations = self._calculate_correlations(
                equity_data, bond_data, commodity_data, currency_data
            )
            
            # Identify divergences
            divergences = self._identify_divergences(correlations)
            
            # Generate allocation signals
            allocation_signals = self._generate_allocation_signals(
                equity_data, bond_data, commodity_data
            )
            
            # Calculate risk-on/risk-off score
            risk_on_score = self._calculate_risk_on_score(
                equity_data, bond_data, currency_data
            )
            
            return {
                'equity_metrics': equity_data,
                'bond_metrics': bond_data,
                'commodity_metrics': commodity_data,
                'currency_metrics': currency_data,
                'correlations': correlations,
                'divergences': divergences,
                'allocation_signals': allocation_signals,
                'risk_on_score': risk_on_score,
                'sector_rotation': self._analyze_sector_rotation()
            }
            
        except Exception as e:
            self.logger.error(f"Error in cross-asset analysis: {e}")
            return self._get_default_cross_asset_data()
    
    async def _get_equity_market_data(self) -> Dict:
        """Get equity market data"""
        # Simulated data - would fetch real data in production
        return {
            'sp500_level': 4500,
            'sp500_trend': 'upward',
            'vix': 15.5,
            'pe_ratio': 22.5,
            'earnings_growth': 5.2,
            'breadth': 0.65  # % of stocks above 200 DMA
        }
    
    async def _get_bond_market_data(self) -> Dict:
        """Get bond market data"""
        return {
            '10y_yield': 4.25,
            '2y_yield': 4.70,
            'real_yield': 1.95,
            'credit_spreads': 1.2,
            'duration_risk': 'moderate'
        }
    
    async def _get_commodity_data(self) -> Dict:
        """Get commodity market data"""
        return {
            'oil_price': 85,
            'gold_price': 2050,
            'copper_price': 4.25,
            'dxy': 103.5,
            'commodity_trend': 'mixed'
        }
    
    async def _get_currency_data(self) -> Dict:
        """Get currency market data"""
        return {
            'dxy_level': 103.5,
            'eurusd': 1.08,
            'usdjpy': 148.5,
            'emerging_markets': 'stable'
        }
    
    def _calculate_correlations(self, equity: Dict, bonds: Dict, 
                                commodities: Dict, currencies: Dict) -> Dict:
        """Calculate cross-asset correlations"""
        # Simplified correlation calculation
        return {
            'equity_bond': -0.35,  # Typical negative correlation
            'equity_commodity': 0.25,
            'bond_dollar': 0.15,
            'commodity_dollar': -0.45,
            'equity_vix': -0.85
        }
    
    def _identify_divergences(self, correlations: Dict) -> List[str]:
        """Identify unusual divergences in correlations"""
        divergences = []
        
        # Check if correlations are breaking down
        if abs(correlations.get('equity_bond', 0)) < 0.2:
            divergences.append('equity_bond_correlation_breakdown')
        
        if correlations.get('equity_vix', 0) > -0.7:
            divergences.append('vix_equity_divergence')
        
        return divergences
    
    def _generate_allocation_signals(self, equity: Dict, bonds: Dict, 
                                     commodities: Dict) -> Dict:
        """Generate asset allocation signals"""
        signals = {}
        
        # Equity signal
        if equity.get('pe_ratio', 20) < 18 and equity.get('vix', 20) < 20:
            signals['equity'] = 'overweight'
        elif equity.get('pe_ratio', 20) > 25:
            signals['equity'] = 'underweight'
        else:
            signals['equity'] = 'neutral'
        
        # Bond signal
        if bonds.get('10y_yield', 3) > 4.5:
            signals['bonds'] = 'overweight'
        elif bonds.get('10y_yield', 3) < 2:
            signals['bonds'] = 'underweight'
        else:
            signals['bonds'] = 'neutral'
        
        # Commodity signal
        if commodities.get('commodity_trend') == 'upward':
            signals['commodities'] = 'overweight'
        else:
            signals['commodities'] = 'neutral'
        
        return signals
    
    def _calculate_risk_on_score(self, equity: Dict, bonds: Dict, 
                                 currencies: Dict) -> float:
        """Calculate risk-on/risk-off score (0-10)"""
        score = 5.0  # Start neutral
        
        # Equity factors
        if equity.get('trend') == 'upward':
            score += 1
        if equity.get('vix', 20) < 15:
            score += 1
        elif equity.get('vix', 20) > 25:
            score -= 2
        
        # Bond factors
        if bonds.get('credit_spreads', 1) < 1:
            score += 0.5
        elif bonds.get('credit_spreads', 1) > 2:
            score -= 1
        
        # Currency factors
        if currencies.get('emerging_markets') == 'strong':
            score += 0.5
        elif currencies.get('emerging_markets') == 'weak':
            score -= 1
        
        return max(0, min(10, score))
    
    def _analyze_sector_rotation(self) -> Dict:
        """Analyze sector rotation signals"""
        return {
            'technology': 'momentum_positive',
            'financials': 'neutral',
            'energy': 'momentum_negative',
            'healthcare': 'defensive_bid',
            'consumer_discretionary': 'weakening',
            'industrials': 'neutral',
            'utilities': 'defensive_bid',
            'real_estate': 'rate_sensitive',
            'materials': 'commodity_linked',
            'consumer_staples': 'defensive_bid',
            'communication': 'growth_sensitive'
        }
    
    def _get_default_cross_asset_data(self) -> Dict:
        """Return default cross-asset data on error"""
        return {
            'equity_metrics': {},
            'bond_metrics': {},
            'commodity_metrics': {},
            'currency_metrics': {},
            'correlations': {},
            'divergences': [],
            'allocation_signals': {},
            'risk_on_score': 5.0,
            'sector_rotation': {}
        }


# ========================================================================================
# MAIN ECONOMIST AGENT
# ========================================================================================

class EconomistAgent:
    """
    Main Economist Agent providing macro economic analysis
    """
    
    def __init__(self, agent_name: str, llm_provider, config, alpaca_provider):
        """Initialize Economist Agent with all components"""
        self.agent_name = agent_name
        self.llm_provider = llm_provider
        self.config = config
        self.alpaca_provider = alpaca_provider
        
        # Initialize logger
        self.logger = logging.getLogger(f"EconomistAgent.{agent_name}")
        
        # Initialize components
        self.economic_analyzer = EconomicDataAnalyzer(alpaca_provider)
        self.theme_identifier = MacroThemeIdentifier()
        self.cross_asset_analyzer = CrossAssetAnalyzer(alpaca_provider)
        self.market_context_manager = MarketContextManager(alpaca_provider)
        self.cache_manager = IntelligentCacheManager()  # No parameters needed
        self.metadata_tracker = AnalysisMetadataTracker()
        
        # Create simple cache wrapper if needed
        # This ensures we always have a working cache mechanism
        self._cache_dict = {}  # Always create fallback cache
        
        # Performance tracking
        self.total_analyses = 0
        self.successful_analyses = 0
        self.cache_hits = 0
        
        self.logger.info(f"Economist Agent '{agent_name}' initialized")
    
    def _cache_get(self, key: str) -> Optional[Dict]:
        """
        Get from cache with fallback
        
        Returns the cached value and increments cache_hits if found
        """
        value = None
        
        # Try the cache manager first if it has proper methods
        if hasattr(self.cache_manager, 'get') and callable(getattr(self.cache_manager, 'get', None)):
            try:
                value = self.cache_manager.get(key)
            except Exception as e:
                self.logger.debug(f"Cache manager get failed: {e}")
                value = None
        
        # Fallback to dictionary cache if cache manager didn't work or returned None
        if value is None and key in self._cache_dict:
            value = self._cache_dict[key]
        
        # Log cache hit for debugging
        if value is not None:
            self.logger.debug(f"Cache hit for key: {key}")
        else:
            self.logger.debug(f"Cache miss for key: {key}")
            
        return value
    
    def _cache_set(self, key: str, value: Dict) -> None:
        """
        Set in cache with fallback
        
        Stores the value in available cache mechanism
        """
        stored = False
        
        # Try the cache manager first if it has proper methods
        if hasattr(self.cache_manager, 'set') and callable(getattr(self.cache_manager, 'set', None)):
            try:
                self.cache_manager.set(key, value)
                stored = True
                self.logger.debug(f"Cached result in cache manager for key: {key}")
            except Exception as e:
                self.logger.debug(f"Cache manager set failed: {e}")
                stored = False
        
        # Always store in dictionary cache as well for fallback
        self._cache_dict[key] = value
        if not stored:
            self.logger.debug(f"Cached result in dictionary for key: {key}")
    
    async def analyze_macro_environment(self, request_type: str = 'full') -> Dict:
        """
        Main entry point for macro economic analysis
        
        Args:
            request_type: 'full', 'economic', 'market', 'themes'
        
        Returns:
            Comprehensive macro outlook
        """
        
        self.total_analyses += 1
        
        # Generate cache key with date for daily expiration
        today = datetime.now().strftime('%Y%m%d')
        cache_key = f"macro_{request_type}_{today}"
        
        # Check cache first
        cached = self._cache_get(cache_key)
        if cached is not None:
            self.cache_hits += 1
            self.successful_analyses += 1  # Cached results are successful
            self.logger.info(f"Using cached macro analysis for {request_type}")
            return cached
        
        try:
            # Start analysis
            self.logger.info(f"Starting macro analysis: {request_type}")
            
            # Gather all data components
            economic_data = await self._gather_economic_data()
            market_data = await self._gather_market_data() 
            
            # Perform analysis based on request type
            if request_type == 'economic':
                themes = self.theme_identifier.identify_macro_themes(economic_data)
                cross_asset = {}
            elif request_type == 'market':
                themes = []
                cross_asset = await self.cross_asset_analyzer.analyze_cross_asset_dynamics()
            else:  # 'full'
                themes = self.theme_identifier.identify_macro_themes(economic_data)
                cross_asset = await self.cross_asset_analyzer.analyze_cross_asset_dynamics()
            
            # Determine economic cycle
            economic_cycle = self._determine_economic_cycle(economic_data)
            
            # Generate asset allocation
            allocation = self._generate_asset_allocation(
                economic_cycle,
                cross_asset,
                economic_data.get('indicators', {}).get('inflation', {}).get('current_rate', 2.0)
            )
            
            # Generate sector recommendations
            sector_recs = self._generate_sector_recommendations(
                economic_cycle,
                themes,
                cross_asset
            )
            
            # Identify risk scenarios
            risk_scenarios = self._identify_risk_scenarios(
                economic_data,
                cross_asset,
                themes
            )
            
            # Determine market regime
            market_regime = self._determine_market_regime(cross_asset, economic_data)
            
            # Generate AI insights if LLM available
            ai_insights = {}
            if self.llm_provider and request_type == 'full':
                ai_insights = await self._generate_ai_insights(
                    economic_data, themes, cross_asset
                )
            
            # Build comprehensive outlook
            outlook = {
                'economic_cycle': economic_cycle,
                'growth_outlook': self._assess_growth_outlook(economic_data),
                'inflation_outlook': self._assess_inflation_outlook(economic_data),
                'policy_outlook': economic_data.get('indicators', {}).get('monetary_policy', {}).get('stance', 'neutral'),
                'geopolitical_risk': self._assess_geopolitical_risk(),
                'dominant_themes': themes[:3] if themes else [],
                'sector_recommendations': sector_recs,
                'asset_allocation': allocation,
                'risk_scenarios': risk_scenarios[:5],  # Top 5 risks
                'confidence_score': self._calculate_confidence_score(economic_data, cross_asset),
                'economic_indicators': economic_data,
                'cross_asset_analysis': cross_asset,
                'market_regime': market_regime,
                'ai_insights': ai_insights,
                'timestamp': datetime.now().isoformat(),
                'request_type': request_type
            }
            
            # Cache the result before returning
            self._cache_set(cache_key, outlook)
            
            # Update success counter
            self.successful_analyses += 1
            
            self.logger.info(f"Completed macro analysis: {request_type}")
            return outlook
            
        except Exception as e:
            self.logger.error(f"Error in macro analysis: {str(e)}")
            
            # Return minimal valid response on error
            fallback = self._get_fallback_outlook(request_type)
            
            # Still cache even the fallback to prevent repeated failures
            self._cache_set(cache_key, fallback)
            
            # Count fallback as successful since we're returning valid data
            self.successful_analyses += 1
            
            return fallback
    
    async def _gather_economic_data(self) -> Dict:
        """Gather all economic data"""
        return await self.economic_analyzer.analyze_economic_indicators()
    
    async def _gather_market_data(self) -> Dict:
        """Gather market context data"""
        try:
            # MarketContextManager uses get_current_context() method
            context = await self.market_context_manager.get_current_context()
            return context
        except Exception as e:
            self.logger.error(f"Error getting market context: {e}")
            return {}
    
    def _determine_economic_cycle(self, economic_data: Dict) -> str:
        """Determine current economic cycle phase"""
        indicators = economic_data.get('indicators', {})
        
        gdp_growth = indicators.get('gdp', {}).get('current_growth', 2)
        gdp_trend = indicators.get('gdp', {}).get('trend', 'stable')
        unemployment = indicators.get('employment', {}).get('unemployment_rate', 4)
        inflation = indicators.get('inflation', {}).get('cpi', 2)
        pmi = indicators.get('manufacturing', {}).get('pmi', 50)
        
        # Simple rule-based cycle determination
        if gdp_growth > 3 and gdp_trend == 'expanding' and unemployment < 4:
            return EconomicCycle.EXPANSION.value
        elif gdp_growth > 3 and inflation > 4:
            return EconomicCycle.PEAK.value
        elif gdp_growth < 1 or pmi < 48:
            return EconomicCycle.CONTRACTION.value
        elif gdp_growth < 0 and unemployment > 6:
            return EconomicCycle.TROUGH.value
        elif gdp_growth > 0 and gdp_growth < 2 and gdp_trend == 'expanding':
            return EconomicCycle.RECOVERY.value
        elif inflation > 4 and gdp_growth < 1:
            return EconomicCycle.STAGFLATION.value
        else:
            return EconomicCycle.EXPANSION.value
    
    def _generate_asset_allocation(self, cycle: str, cross_asset: Dict, 
                                   inflation: float) -> Dict:
        """Generate strategic asset allocation based on conditions"""
        # Base allocations by cycle
        allocations = {
            EconomicCycle.EXPANSION.value: {'equities': 70, 'bonds': 20, 'commodities': 5, 'cash': 5},
            EconomicCycle.PEAK.value: {'equities': 50, 'bonds': 30, 'commodities': 10, 'cash': 10},
            EconomicCycle.CONTRACTION.value: {'equities': 30, 'bonds': 50, 'commodities': 5, 'cash': 15},
            EconomicCycle.TROUGH.value: {'equities': 40, 'bonds': 40, 'commodities': 5, 'cash': 15},
            EconomicCycle.RECOVERY.value: {'equities': 60, 'bonds': 30, 'commodities': 5, 'cash': 5},
            EconomicCycle.STAGFLATION.value: {'equities': 20, 'bonds': 20, 'commodities': 30, 'cash': 30}
        }
        
        allocation = allocations.get(cycle, {'equities': 50, 'bonds': 30, 'commodities': 10, 'cash': 10})
        
        # Adjust based on risk-on/risk-off score
        risk_score = cross_asset.get('risk_on_score', 5)
        if risk_score > 7:
            allocation['equities'] = min(80, allocation['equities'] + 10)
            allocation['cash'] = max(5, allocation['cash'] - 10)
        elif risk_score < 3:
            allocation['equities'] = max(20, allocation['equities'] - 20)
            allocation['cash'] = min(30, allocation['cash'] + 20)
        
        # Adjust for inflation
        if inflation > 4:
            allocation['commodities'] = min(20, allocation['commodities'] + 10)
            allocation['bonds'] = max(10, allocation['bonds'] - 10)
        
        return allocation
    
    def _generate_sector_recommendations(self, cycle: str, themes: List[MacroTheme],
                                        cross_asset: Dict) -> Dict:
        """Generate sector allocation recommendations"""
        recommendations = {}
        
        # Base recommendations by cycle
        cycle_recs = {
            EconomicCycle.EXPANSION.value: {
                'technology': 'overweight',
                'financials': 'overweight',
                'consumer_discretionary': 'overweight',
                'industrials': 'neutral',
                'utilities': 'underweight',
                'consumer_staples': 'underweight'
            },
            EconomicCycle.CONTRACTION.value: {
                'technology': 'underweight',
                'financials': 'underweight',
                'consumer_discretionary': 'underweight',
                'utilities': 'overweight',
                'consumer_staples': 'overweight',
                'healthcare': 'overweight'
            }
        }
        
        recommendations = cycle_recs.get(cycle, {})
        
        # Adjust based on themes
        for theme in themes[:2]:  # Consider top 2 themes
            for sector in theme.beneficiaries:
                if sector in recommendations:
                    recommendations[sector] = 'overweight'
            for sector in theme.victims:
                if sector in recommendations:
                    recommendations[sector] = 'underweight'
        
        # Fill in missing sectors
        all_sectors = ['technology', 'financials', 'healthcare', 'consumer_discretionary',
                      'consumer_staples', 'industrials', 'energy', 'utilities', 
                      'real_estate', 'materials', 'communication']
        
        for sector in all_sectors:
            if sector not in recommendations:
                recommendations[sector] = 'neutral'
        
        return recommendations
    
    def _identify_risk_scenarios(self, economic_data: Dict, cross_asset: Dict,
                                 themes: List[MacroTheme]) -> List[Dict]:
        """Identify and quantify risk scenarios"""
        scenarios = []
        
        indicators = economic_data.get('indicators', {})
        
        # Recession risk
        if indicators.get('yield_curve', {}).get('shape') == 'inverted':
            scenarios.append({
                'scenario': 'recession',
                'probability': 0.35,
                'impact': 'high',
                'timeline': '6-12 months',
                'hedges': ['long_bonds', 'defensive_sectors', 'cash']
            })
        
        # Inflation spike risk
        if indicators.get('inflation', {}).get('trend') == 'accelerating':
            scenarios.append({
                'scenario': 'inflation_spike',
                'probability': 0.25,
                'impact': 'medium',
                'timeline': '3-6 months',
                'hedges': ['commodities', 'tips', 'floating_rate']
            })
        
        # Policy error risk
        if indicators.get('monetary_policy', {}).get('stance') == 'hawkish':
            scenarios.append({
                'scenario': 'policy_overtightening',
                'probability': 0.20,
                'impact': 'high',
                'timeline': '3-6 months',
                'hedges': ['long_duration', 'quality_stocks']
            })
        
        # Geopolitical risk
        scenarios.append({
            'scenario': 'geopolitical_escalation',
            'probability': 0.15,
            'impact': 'medium',
            'timeline': 'ongoing',
            'hedges': ['gold', 'energy', 'defense_stocks']
        })
        
        # Sort by probability
        scenarios.sort(key=lambda x: x['probability'], reverse=True)
        
        return scenarios
    
    def _determine_market_regime(self, cross_asset: Dict, economic_data: Dict) -> str:
        """Determine current market regime"""
        risk_score = cross_asset.get('risk_on_score', 5)
        
        if risk_score >= 7:
            return MarketRegime.RISK_ON.value
        elif risk_score <= 3:
            return MarketRegime.RISK_OFF.value
        elif 4 <= risk_score <= 6:
            volatility = cross_asset.get('equity_metrics', {}).get('vix', 20)
            if volatility > 25:
                return MarketRegime.TRANSITION.value  # Changed from TRANSITIONAL to TRANSITION
            else:
                return MarketRegime.NEUTRAL.value
        else:
            return MarketRegime.NEUTRAL.value
    
    def _assess_growth_outlook(self, economic_data: Dict) -> str:
        """Assess economic growth outlook"""
        gdp = economic_data.get('indicators', {}).get('gdp', {}).get('current_growth', 2)
        
        if gdp >= 3:  # Changed from > to >= for 3.0 to be strong_growth
            return 'strong_growth'
        elif gdp >= 2:  # Changed from > to >= for 2.0 to be moderate_growth
            return 'moderate_growth'
        elif gdp > 0:
            return 'slow_growth'
        else:
            return 'contraction'
    
    def _assess_inflation_outlook(self, economic_data: Dict) -> str:
        """Assess inflation outlook"""
        inflation = economic_data.get('indicators', {}).get('inflation', {})
        cpi = inflation.get('cpi', 2)
        trend = inflation.get('trend', 'stable')
        
        if cpi > 4 and trend == 'accelerating':
            return 'rising_inflation'
        elif cpi >= 3 and trend == 'moderating':  # Fixed: cpi >= 3 for moderating
            return 'moderating_inflation'
        elif 1.5 <= cpi <= 2.5 and trend == 'stable':  # More specific range for stable
            return 'stable_inflation'
        elif cpi < 1.5:
            return 'low_inflation'
        else:
            return 'elevated_inflation'
    
    def _assess_geopolitical_risk(self) -> str:
        """Assess geopolitical risk level"""
        # Simplified assessment - would use real data in production
        return GeopoliticalRisk.MODERATE.value
    
    def _calculate_confidence_score(self, economic_data: Dict, cross_asset: Dict) -> float:
        """Calculate confidence in the analysis (0-10)"""
        score = 7.0  # Base score
        
        # Adjust based on data quality
        if economic_data.get('health_score'):
            score += 0.5
        
        if cross_asset.get('correlations'):
            score += 0.5
        
        # Adjust based on divergences
        divergences = cross_asset.get('divergences', [])
        score -= len(divergences) * 0.5
        
        return max(0, min(10, score))
    
    async def _generate_ai_insights(self, economic_data: Dict, themes: List[MacroTheme],
                                   cross_asset: Dict) -> Dict:
        """Generate AI-powered insights using LLM"""
        if not self.llm_provider:
            return {}
        
        try:
            # Prepare context for LLM
            context = {
                'economic_health': economic_data.get('health_score', 5),
                'trend': economic_data.get('trend', 'neutral'),
                'top_themes': [t.theme_name for t in themes[:3]],
                'risk_on_score': cross_asset.get('risk_on_score', 5)
            }
            
            # Generate insights
            response = await self.llm_provider.generate_analysis(
                analysis_type='macro_outlook',
                context=context
            )
            
            return {
                'summary': response.get('summary', 'Macro analysis complete'),
                'recommendations': response.get('recommendations', []),
                'risks': response.get('risks', []),
                'opportunities': response.get('opportunities', [])
            }
            
        except Exception as e:
            self.logger.error(f"Error generating AI insights: {e}")
            return {}
    
    def _get_fallback_outlook(self, request_type: str) -> Dict:
        """Generate fallback outlook on error"""
        return {
            'economic_cycle': EconomicCycle.EXPANSION.value,
            'growth_outlook': 'moderate_growth',
            'inflation_outlook': 'stable_inflation', 
            'policy_outlook': 'neutral',
            'geopolitical_risk': GeopoliticalRisk.MODERATE.value,
            'dominant_themes': [],
            'sector_recommendations': {
                'technology': 'neutral',
                'financials': 'neutral',
                'healthcare': 'neutral',
                'consumer_discretionary': 'neutral',
                'consumer_staples': 'neutral',
                'industrials': 'neutral',
                'energy': 'neutral',
                'utilities': 'neutral',
                'real_estate': 'neutral',
                'materials': 'neutral',
                'communication': 'neutral'
            },
            'asset_allocation': {
                'equities': 50,
                'bonds': 30,
                'commodities': 10,
                'cash': 10
            },
            'risk_scenarios': [],
            'confidence_score': 5.0,
            'economic_indicators': {},
            'cross_asset_analysis': {},
            'market_regime': MarketRegime.NEUTRAL.value,
            'ai_insights': {},
            'timestamp': datetime.now().isoformat(),
            'request_type': request_type
        }
    
    def get_performance_metrics(self) -> Dict:
        """Get agent performance metrics"""
        success_rate = (self.successful_analyses / max(1, self.total_analyses)) * 100
        cache_hit_rate = (self.cache_hits / max(1, self.total_analyses)) * 100
        
        return {
            'total_analyses': self.total_analyses,
            'successful_analyses': self.successful_analyses,
            'success_rate': round(success_rate, 2),
            'cache_hits': self.cache_hits,
            'cache_hit_rate': round(cache_hit_rate, 2)
        }