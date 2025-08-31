# agents/junior_research_analyst.py
"""
Enhanced Junior Research Analyst Agent - Complete Refactored Implementation
Optimized for macOS M2 Max with Claude AI integration

This refactored version includes:
- Standardized output for Senior Analyst compatibility
- Shared market context management
- Unified risk assessment
- Performance tracking and feedback processing
- Intelligent caching
"""

import asyncio
import json
import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from enum import Enum
import uuid
import time
import hashlib
from collections import defaultdict
import numpy as np


# ========================================================================================
# ENUMS AND CONSTANTS
# ========================================================================================

class AnalysisType(Enum):
    NEW_OPPORTUNITY = "new_opportunity"
    POSITION_REEVALUATION = "position_reevaluation"
    RISK_ASSESSMENT = "risk_assessment"
    EARNINGS_ANALYSIS = "earnings_analysis"
    NEWS_IMPACT = "news_impact"


class RecommendationType(Enum):
    STRONG_BUY = "strong_buy"
    BUY = "buy"
    HOLD = "hold"
    SELL = "sell"
    STRONG_SELL = "strong_sell"


class TimeHorizon(Enum):
    SHORT_TERM = "short_term"      # 1-5 days
    MEDIUM_TERM = "medium_term"    # 1-4 weeks
    LONG_TERM = "long_term"        # 1-6 months


class RiskLevel(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    VERY_HIGH = "very_high"


class PositionSize(Enum):
    SMALL = "small"      # 1-2% portfolio
    MEDIUM = "medium"    # 2-4% portfolio
    LARGE = "large"      # 4-5% portfolio
    MAX = "max"          # 5% portfolio limit


class ConvictionLevel(Enum):
    LOW = 1
    MEDIUM_LOW = 2
    MEDIUM = 3
    MEDIUM_HIGH = 4
    HIGH = 5


# ========================================================================================
# SHARED COMPONENTS
# ========================================================================================

class MarketContextManager:
    """Shared market context between agents"""
    
    def __init__(self, alpaca_provider):
        self.alpaca = alpaca_provider
        self.context_cache = {}
        self.cache_ttl = 300  # 5 minutes
        self.logger = logging.getLogger('market_context')
        
    async def get_current_context(self) -> Dict:
        """Get current market context for all agents"""
        
        cache_key = "market_context"
        cached = self._get_cached(cache_key)
        if cached:
            self.logger.debug("Using cached market context")
            return cached
            
        try:
            context = {
                "regime": await self._detect_market_regime(),
                "vix_level": await self._get_vix_level(),
                "sector_performance": await self._get_sector_performance(),
                "market_breadth": await self._calculate_market_breadth(),
                "sentiment_indicators": await self._get_sentiment_indicators(),
                "key_levels": await self._identify_key_levels(),
                "timestamp": datetime.now().isoformat()
            }
            
            self._cache_result(cache_key, context)
            return context
            
        except Exception as e:
            self.logger.error(f"Failed to get market context: {str(e)}")
            return self._get_default_context()
    
    def _get_cached(self, key: str) -> Optional[Dict]:
        """Get cached result if not expired"""
        if key in self.context_cache:
            cached_data = self.context_cache[key]
            if (datetime.now() - cached_data['timestamp']).seconds < self.cache_ttl:
                return cached_data['data']
        return None
    
    def _cache_result(self, key: str, data: Dict):
        """Cache result with timestamp"""
        self.context_cache[key] = {
            'data': data,
            'timestamp': datetime.now()
        }
    
    async def _detect_market_regime(self) -> str:
        """Detect current market regime"""
        try:
            # Get SPY data for regime detection
            spy_data = await self.alpaca.get_bars('SPY', timeframe='1Day', limit=20)
            if not spy_data:
                return "neutral"
            
            # Simple regime detection based on moving averages
            closes = [bar['close'] for bar in spy_data]
            sma_5 = np.mean(closes[-5:])
            sma_20 = np.mean(closes)
            
            if sma_5 > sma_20 * 1.02:
                return "risk_on"
            elif sma_5 < sma_20 * 0.98:
                return "risk_off"
            else:
                return "neutral"
                
        except Exception as e:
            self.logger.error(f"Regime detection failed: {str(e)}")
            return "neutral"
    
    async def _get_vix_level(self) -> Dict:
        """Get current VIX level and interpretation"""
        try:
            vix_data = await self.alpaca.get_latest_quote('VIXY')
            vix_level = vix_data.get('price', 20)
            
            return {
                "level": vix_level,
                "interpretation": self._interpret_vix(vix_level)
            }
        except:
            return {"level": 20, "interpretation": "normal"}
    
    def _interpret_vix(self, vix: float) -> str:
        """Interpret VIX level"""
        if vix < 15:
            return "low_volatility"
        elif vix < 25:
            return "normal"
        elif vix < 35:
            return "elevated"
        else:
            return "high_volatility"
    
    async def _get_sector_performance(self) -> Dict:
        """Get sector performance data"""
        sectors = ['XLK', 'XLF', 'XLV', 'XLE', 'XLI', 'XLY', 'XLP', 'XLRE', 'XLB', 'XLU']
        performance = {}
        
        for sector in sectors:
            try:
                data = await self.alpaca.get_bars(sector, timeframe='1Day', limit=5)
                if data and len(data) >= 2:
                    change = (data[-1]['close'] - data[-2]['close']) / data[-2]['close']
                    performance[sector] = round(change * 100, 2)
            except:
                performance[sector] = 0.0
                
        return performance
    
    async def _calculate_market_breadth(self) -> Dict:
        """Calculate market breadth indicators"""
        # Simplified breadth calculation
        return {
            "advance_decline": 0.5,  # Placeholder
            "new_highs_lows": 0.0,   # Placeholder
            "percent_above_ma": 65   # Placeholder
        }
    
    async def _get_sentiment_indicators(self) -> Dict:
        """Get market sentiment indicators"""
        return {
            "put_call_ratio": 0.85,  # Placeholder
            "fear_greed_index": 50,  # Placeholder
            "sentiment": "neutral"
        }
    
    async def _identify_key_levels(self) -> Dict:
        """Identify key market levels"""
        try:
            spy_data = await self.alpaca.get_bars('SPY', timeframe='1Day', limit=20)
            if spy_data:
                highs = [bar['high'] for bar in spy_data]
                lows = [bar['low'] for bar in spy_data]
                
                return {
                    "resistance": max(highs),
                    "support": min(lows),
                    "current": spy_data[-1]['close']
                }
        except:
            pass
            
        return {"resistance": 450, "support": 420, "current": 435}
    
    def _get_default_context(self) -> Dict:
        """Get default context when API fails"""
        return {
            "regime": "neutral",
            "vix_level": {"level": 20, "interpretation": "normal"},
            "sector_performance": {},
            "market_breadth": {"advance_decline": 0.5},
            "sentiment_indicators": {"sentiment": "neutral"},
            "key_levels": {"resistance": 450, "support": 420},
            "timestamp": datetime.now().isoformat()
        }


class UnifiedRiskAssessment:
    """Standardized risk assessment for both analysts"""
    
    @staticmethod
    def calculate_risk_score(analysis_data: Dict) -> Dict:
        """Calculate comprehensive risk score"""
        
        risk_factors = {
            "volatility_risk": UnifiedRiskAssessment._assess_volatility(analysis_data),
            "liquidity_risk": UnifiedRiskAssessment._assess_liquidity(analysis_data),
            "sector_risk": UnifiedRiskAssessment._assess_sector_risk(analysis_data),
            "correlation_risk": UnifiedRiskAssessment._assess_correlation(analysis_data),
            "catalyst_risk": UnifiedRiskAssessment._assess_catalyst_risk(analysis_data),
            "technical_risk": UnifiedRiskAssessment._assess_technical_risk(analysis_data)
        }
        
        # Weighted average
        weights = {
            "volatility_risk": 0.20,
            "liquidity_risk": 0.15,
            "sector_risk": 0.15,
            "correlation_risk": 0.20,
            "catalyst_risk": 0.15,
            "technical_risk": 0.15
        }
        
        overall_score = sum(
            risk_factors[factor] * weights[factor] 
            for factor in risk_factors
        )
        
        return {
            "overall_risk_score": round(overall_score, 2),
            "risk_level": UnifiedRiskAssessment._map_score_to_level(overall_score),
            "risk_factors": risk_factors,
            "key_risks": UnifiedRiskAssessment._identify_key_risks(risk_factors)
        }
    
    @staticmethod
    def _assess_volatility(data: Dict) -> float:
        """Assess volatility risk (1-10 scale)"""
        volatility = data.get('volatility', 0.2)
        if volatility < 0.15:
            return 2
        elif volatility < 0.25:
            return 4
        elif volatility < 0.35:
            return 6
        elif volatility < 0.45:
            return 8
        else:
            return 10
    
    @staticmethod
    def _assess_liquidity(data: Dict) -> float:
        """Assess liquidity risk"""
        avg_volume = data.get('average_volume', 1000000)
        if avg_volume > 10000000:
            return 2
        elif avg_volume > 5000000:
            return 4
        elif avg_volume > 1000000:
            return 6
        elif avg_volume > 500000:
            return 8
        else:
            return 10
    
    @staticmethod
    def _assess_sector_risk(data: Dict) -> float:
        """Assess sector-specific risk"""
        sector = data.get('sector', 'Unknown')
        high_risk_sectors = ['Energy', 'Materials', 'Biotechnology']
        medium_risk_sectors = ['Technology', 'Financials', 'Industrials']
        low_risk_sectors = ['Utilities', 'Consumer Staples', 'Healthcare']
        
        if sector in high_risk_sectors:
            return 8
        elif sector in medium_risk_sectors:
            return 5
        elif sector in low_risk_sectors:
            return 3
        else:
            return 6
    
    @staticmethod
    def _assess_correlation(data: Dict) -> float:
        """Assess correlation risk"""
        correlation = data.get('correlation_to_spy', 0.5)
        if abs(correlation) > 0.8:
            return 8
        elif abs(correlation) > 0.6:
            return 6
        elif abs(correlation) > 0.4:
            return 4
        else:
            return 2
    
    @staticmethod
    def _assess_catalyst_risk(data: Dict) -> float:
        """Assess catalyst-related risk"""
        catalysts = data.get('catalysts', [])
        if not catalysts:
            return 7
        elif len(catalysts) == 1:
            return 5
        else:
            return 3
    
    @staticmethod
    def _assess_technical_risk(data: Dict) -> float:
        """Assess technical risk"""
        technical_score = data.get('technical_score', 5)
        return 10 - technical_score  # Inverse relationship
    
    @staticmethod
    def _map_score_to_level(score: float) -> str:
        """Map risk score to risk level"""
        if score <= 3:
            return RiskLevel.LOW.value
        elif score <= 5:
            return RiskLevel.MEDIUM.value
        elif score <= 7:
            return RiskLevel.HIGH.value
        else:
            return RiskLevel.VERY_HIGH.value
    
    @staticmethod
    def _identify_key_risks(risk_factors: Dict) -> List[str]:
        """Identify top risk factors"""
        sorted_risks = sorted(risk_factors.items(), key=lambda x: x[1], reverse=True)
        key_risks = []
        
        for risk_name, score in sorted_risks[:3]:
            if score >= 6:
                risk_description = risk_name.replace('_', ' ').title()
                key_risks.append(f"{risk_description}: {score}/10")
                
        return key_risks


class IntelligentCacheManager:
    """Smart caching for analysis results"""
    
    def __init__(self, max_size: int = 100, ttl_seconds: int = 300):
        self.cache = {}
        self.access_counts = defaultdict(int)
        self.last_access = {}
        self.max_size = max_size
        self.ttl = ttl_seconds
        
    def get(self, key: str) -> Optional[Dict]:
        """Get from cache with TTL check"""
        if key not in self.cache:
            return None
            
        # Check TTL
        if self._is_expired(key):
            del self.cache[key]
            return None
            
        # Update access patterns
        self.access_counts[key] += 1
        self.last_access[key] = datetime.now()
        
        return self.cache[key]['value']
    
    def put(self, key: str, value: Dict) -> None:
        """Put in cache with LRU eviction"""
        
        # Evict if needed
        if len(self.cache) >= self.max_size:
            self._evict_lru()
        
        self.cache[key] = {
            "value": value,
            "timestamp": datetime.now()
        }
        self.last_access[key] = datetime.now()
    
    def clear(self):
        """Clear all cache"""
        self.cache.clear()
        self.access_counts.clear()
        self.last_access.clear()
    
    def _is_expired(self, key: str) -> bool:
        """Check if cache entry is expired"""
        if key not in self.cache:
            return True
        age = (datetime.now() - self.cache[key]['timestamp']).seconds
        return age > self.ttl
    
    def _evict_lru(self) -> None:
        """Evict least recently used item"""
        if not self.last_access:
            return
            
        lru_key = min(self.last_access, key=self.last_access.get)
        del self.cache[lru_key]
        del self.last_access[lru_key]
        del self.access_counts[lru_key]


class AnalysisMetadataTracker:
    """Track metadata through analysis chain"""
    
    def __init__(self):
        self.chains = {}
        
    def create_analysis_chain(self, initial_trigger: str) -> str:
        """Create new analysis chain ID"""
        chain_id = str(uuid.uuid4())
        
        self.chains[chain_id] = {
            "id": chain_id,
            "trigger": initial_trigger,
            "start_time": datetime.now(),
            "steps": [],
            "status": "in_progress"
        }
        
        return chain_id
    
    def add_step(self, chain_id: str, step_data: Dict) -> None:
        """Add step to analysis chain"""
        if chain_id not in self.chains:
            return
            
        step = {
            "agent": step_data.get('agent'),
            "action": step_data.get('action'),
            "timestamp": datetime.now(),
            "duration_ms": step_data.get('duration_ms'),
            "result": step_data.get('result'),
            "confidence": step_data.get('confidence')
        }
        
        self.chains[chain_id]["steps"].append(step)
    
    def complete_chain(self, chain_id: str, status: str = "success"):
        """Mark chain as complete"""
        if chain_id in self.chains:
            self.chains[chain_id]["status"] = status
            self.chains[chain_id]["end_time"] = datetime.now()
    
    def get_chain_summary(self, chain_id: str) -> Dict:
        """Get summary of analysis chain"""
        chain = self.chains.get(chain_id)
        if not chain:
            return {}
        
        duration = 0
        if "end_time" in chain:
            duration = (chain["end_time"] - chain["start_time"]).total_seconds() * 1000
            
        return {
            "chain_id": chain_id,
            "total_duration_ms": duration,
            "num_steps": len(chain["steps"]),
            "agents_involved": list(set(s["agent"] for s in chain["steps"])),
            "final_confidence": self._calculate_final_confidence(chain),
            "status": chain["status"]
        }
    
    def _calculate_final_confidence(self, chain: Dict) -> float:
        """Calculate final confidence from all steps"""
        confidences = [s.get('confidence', 5) for s in chain["steps"] if s.get('confidence')]
        return np.mean(confidences) if confidences else 5.0


# ========================================================================================
# ENHANCED JUNIOR RESEARCH ANALYST
# ========================================================================================

class JuniorResearchAnalyst:
    """
    Enhanced Junior Research Analyst Agent
    
    Combines technical analysis with fundamental research to provide
    comprehensive stock analysis and trading recommendations.
    """
    
    def __init__(self, llm_provider, alpaca_provider, config):
        """Initialize the Enhanced Junior Research Analyst"""
        
        self.agent_name = "junior_research_analyst"
        self.agent_id = str(uuid.uuid4())
        
        # Core dependencies
        self.llm = llm_provider
        self.alpaca = alpaca_provider
        self.config = config
        
        # Setup logging
        self.logger = self._setup_logging()
        
        # Shared components
        self.market_context_manager = MarketContextManager(alpaca_provider)
        self.risk_assessor = UnifiedRiskAssessment()
        self.cache_manager = IntelligentCacheManager(max_size=100, ttl_seconds=300)
        self.metadata_tracker = AnalysisMetadataTracker()
        
        # Analysis engines
        self.technical_engine = TechnicalAnalysisEngine(alpaca_provider)
        self.fundamental_engine = FundamentalAnalysisEngine(alpaca_provider)
        
        # Performance tracking
        self.performance_metrics = {
            "total_analyses": 0,
            "successful_analyses": 0,
            "failed_analyses": 0,
            "average_processing_time": 0.0,
            "cache_hits": 0,
            "last_activity": None,
            "feedback_received_count": 0,
            "performance_score": 5.0
        }
        
        # Feedback storage
        self.feedback_history = []
        
        self.logger.info(f"✅ Enhanced Junior Research Analyst initialized with ID: {self.agent_id}")
    
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
    
    async def analyze_stock(self, task_data: Dict) -> Dict:
        """Main entry point for stock analysis"""
        start_time = datetime.now()
        ticker = task_data.get('ticker', '').upper()
        
        # Create analysis chain
        chain_id = self.metadata_tracker.create_analysis_chain(
            f"analyze_stock_{ticker}"
        )
        
        try:
            # Check cache first
            cache_key = self._generate_cache_key(task_data)
            cached_result = self.cache_manager.get(cache_key)
            
            if cached_result:
                self.performance_metrics['cache_hits'] += 1
                self.logger.info(f"Cache hit for {ticker}")
                
                # Add metadata step
                self.metadata_tracker.add_step(chain_id, {
                    'agent': self.agent_name,
                    'action': 'cache_hit',
                    'duration_ms': 0,
                    'result': 'success'
                })
                
                return cached_result
            
            # Get shared market context
            market_context = await self.market_context_manager.get_current_context()
            
            # Add market context to task data
            task_data['market_context'] = market_context
            
            # Perform analysis based on type
            result = await self._perform_analysis(task_data)
            
            # Add unified risk assessment
            risk_assessment = self.risk_assessor.calculate_risk_score(result)
            result['risk_assessment'] = risk_assessment
            
            # Create standardized output
            standardized_result = self._create_analysis_result(ticker, result)
            
            # Cache result
            self.cache_manager.put(cache_key, standardized_result)
            
            # Update metrics
            processing_time = (datetime.now() - start_time).total_seconds()
            self._update_metrics(True, processing_time)
            
            # Complete chain
            self.metadata_tracker.complete_chain(chain_id)
            
            # Add chain summary to result
            standardized_result['analysis_chain'] = self.metadata_tracker.get_chain_summary(chain_id)
            
            return standardized_result
            
        except Exception as e:
            self.logger.error(f"Analysis failed for {ticker}: {str(e)}")
            processing_time = (datetime.now() - start_time).total_seconds()
            self._update_metrics(False, processing_time)
            
            self.metadata_tracker.complete_chain(chain_id, "failed")
            
            return self._create_error_response(ticker, str(e))
    
    async def _perform_analysis(self, task_data: Dict) -> Dict:
        """Perform the actual analysis based on task type"""
        task_type = task_data.get('task_type')
        
        if task_type == AnalysisType.NEW_OPPORTUNITY.value:
            return await self._analyze_new_opportunity(task_data)
        elif task_type == AnalysisType.POSITION_REEVALUATION.value:
            return await self._reevaluate_position(task_data)
        elif task_type == AnalysisType.EARNINGS_ANALYSIS.value:
            return await self._analyze_earnings_impact(task_data)
        elif task_type == AnalysisType.NEWS_IMPACT.value:
            return await self._analyze_news_impact(task_data)
        else:
            return await self._analyze_new_opportunity(task_data)
    
    def _create_analysis_result(self, ticker: str, analysis_data: Dict) -> Dict:
        """Create standardized analysis result for Senior Analyst consumption"""
        
        return {
            # Core identification
            "ticker": ticker,
            "analysis_id": str(uuid.uuid4()),
            "timestamp": datetime.now().isoformat(),
            "agent_name": self.agent_name,
            "agent_id": self.agent_id,
            
            # Recommendation details
            "recommendation": analysis_data.get('recommendation', 'HOLD'),
            "recommendation_type": self._map_to_recommendation_type(
                analysis_data.get('recommendation', 'HOLD')
            ),
            "confidence": analysis_data.get('confidence', 5),
            "conviction_level": self._map_confidence_to_conviction(
                analysis_data.get('confidence', 5)
            ),
            
            # Financial metrics
            "expected_return": analysis_data.get('expected_return', 0.0),
            "risk_reward_ratio": analysis_data.get('risk_reward_ratio', 1.0),
            "volatility": analysis_data.get('volatility', 0.2),
            
            # Price targets
            "entry_price": analysis_data.get('entry_price'),
            "stop_loss": analysis_data.get('stop_loss'),
            "primary_target": analysis_data.get('primary_target'),
            "secondary_target": analysis_data.get('secondary_target'),
            
            # Time horizon
            "time_horizon": analysis_data.get('time_horizon', TimeHorizon.MEDIUM_TERM.value),
            "holding_period_days": self._estimate_holding_period(
                analysis_data.get('time_horizon', TimeHorizon.MEDIUM_TERM.value)
            ),
            
            # Risk assessment (from unified assessor)
            "risk_assessment": analysis_data.get('risk_assessment', {}),
            "risk_level": analysis_data.get('risk_assessment', {}).get('risk_level', RiskLevel.MEDIUM.value),
            "risk_score": analysis_data.get('risk_assessment', {}).get('overall_risk_score', 5),
            "key_risks": analysis_data.get('risk_assessment', {}).get('key_risks', []),
            
            # Position sizing
            "position_size": analysis_data.get('position_size', PositionSize.MEDIUM.value),
            "position_weight_percent": self._map_position_size_to_weight(
                analysis_data.get('position_size', PositionSize.MEDIUM.value)
            ),
            
            # Analysis details
            "thesis": analysis_data.get('thesis', ''),
            "thesis_summary": self._summarize_thesis(analysis_data.get('thesis', '')),
            "catalysts": analysis_data.get('catalysts', []),
            "technical_signals": analysis_data.get('technical_signals', {}),
            
            # Market context
            "sector": analysis_data.get('sector', 'Unknown'),
            "market_cap": analysis_data.get('market_cap', 'Unknown'),
            "correlation_to_spy": analysis_data.get('correlation_to_spy', 0.0),
            "average_volume": analysis_data.get('average_volume', 0),
            
            # Senior Analyst specific fields
            "liquidity_score": self._calculate_liquidity_score(analysis_data),
            "catalyst_strength": self._evaluate_catalyst_strength(
                analysis_data.get('catalysts', [])
            ),
            "technical_score": analysis_data.get('technical_signal', {}).get('score', 5),
            
            # Metadata
            "analysis_type": analysis_data.get('analysis_type', AnalysisType.NEW_OPPORTUNITY.value),
            "data_quality_score": analysis_data.get('data_quality_score', 8),
            "analysis_status": "success",
            
            # Market context from shared manager
            "market_context": analysis_data.get('market_context', {})
        }
    
    def _map_to_recommendation_type(self, recommendation: str) -> str:
        """Map recommendation to type"""
        recommendation_map = {
            'STRONG_BUY': RecommendationType.STRONG_BUY.value,
            'BUY': RecommendationType.BUY.value,
            'HOLD': RecommendationType.HOLD.value,
            'SELL': RecommendationType.SELL.value,
            'STRONG_SELL': RecommendationType.STRONG_SELL.value
        }
        return recommendation_map.get(recommendation.upper(), RecommendationType.HOLD.value)
    
    def _map_confidence_to_conviction(self, confidence: int) -> int:
        """Map confidence score to conviction level for Senior Analyst"""
        if confidence >= 9:
            return ConvictionLevel.HIGH.value
        elif confidence >= 7:
            return ConvictionLevel.MEDIUM_HIGH.value
        elif confidence >= 5:
            return ConvictionLevel.MEDIUM.value
        elif confidence >= 3:
            return ConvictionLevel.MEDIUM_LOW.value
        else:
            return ConvictionLevel.LOW.value
    
    def _map_position_size_to_weight(self, position_size: str) -> float:
        """Map position size to portfolio weight percentage"""
        size_to_weight = {
            PositionSize.SMALL.value: 2.0,
            PositionSize.MEDIUM.value: 3.5,
            PositionSize.LARGE.value: 4.5,
            PositionSize.MAX.value: 5.0
        }
        return size_to_weight.get(position_size, 3.5)
    
    def _estimate_holding_period(self, time_horizon: str) -> int:
        """Estimate holding period in days"""
        horizon_to_days = {
            TimeHorizon.SHORT_TERM.value: 3,
            TimeHorizon.MEDIUM_TERM.value: 14,
            TimeHorizon.LONG_TERM.value: 90
        }
        return horizon_to_days.get(time_horizon, 14)
    
    def _summarize_thesis(self, thesis: str) -> str:
        """Create brief summary of thesis"""
        if not thesis:
            return ""
        
        # Take first 200 characters
        summary = thesis[:200]
        if len(thesis) > 200:
            summary += "..."
        
        return summary
    
    def _calculate_liquidity_score(self, data: Dict) -> float:
        """Calculate liquidity score (1-10)"""
        avg_volume = data.get('average_volume', 0)
        
        if avg_volume > 10000000:
            return 10.0
        elif avg_volume > 5000000:
            return 8.0
        elif avg_volume > 1000000:
            return 6.0
        elif avg_volume > 500000:
            return 4.0
        else:
            return 2.0
    
    def _evaluate_catalyst_strength(self, catalysts: List) -> float:
        """Evaluate strength of catalysts (1-10)"""
        if not catalysts:
            return 3.0
        
        num_catalysts = len(catalysts)
        if num_catalysts >= 3:
            return 9.0
        elif num_catalysts == 2:
            return 7.0
        else:
            return 5.0
    
    async def process_feedback(self, feedback: Dict) -> None:
        """Process feedback from Senior Analyst"""
        
        analysis_id = feedback.get('analysis_id')
        
        # Store feedback
        self.feedback_history.append({
            'timestamp': datetime.now(),
            'feedback': feedback
        })
        
        # Update performance metrics based on feedback
        if feedback.get('performance_score'):
            self._update_performance_score(feedback['performance_score'])
        
        # Learn from improvements needed
        if feedback.get('improvements_needed'):
            self._learn_from_feedback(feedback['improvements_needed'])
        
        self.performance_metrics['feedback_received_count'] += 1
        
        self.logger.info(f"Processed feedback for analysis {analysis_id}")
    
    def _update_performance_score(self, score: float):
        """Update running performance score"""
        # Exponential moving average
        alpha = 0.1
        self.performance_metrics['performance_score'] = (
            alpha * score + (1 - alpha) * self.performance_metrics['performance_score']
        )
    
    def _learn_from_feedback(self, improvements: List[str]):
        """Learn from feedback improvements"""
        # This could be enhanced with ML in the future
        self.logger.info(f"Learning from feedback: {improvements}")
    
    def _generate_cache_key(self, task_data: Dict) -> str:
        """Generate cache key for task data"""
        key_parts = [
            task_data.get('ticker', ''),
            task_data.get('task_type', ''),
            str(task_data.get('technical_signal', {}).get('pattern', ''))
        ]
        
        key_string = '_'.join(key_parts)
        return hashlib.md5(key_string.encode()).hexdigest()
    
    def _update_metrics(self, success: bool, processing_time: float):
        """Update performance metrics"""
        self.performance_metrics['total_analyses'] += 1
        
        if success:
            self.performance_metrics['successful_analyses'] += 1
        else:
            self.performance_metrics['failed_analyses'] += 1
        
        # Update average processing time
        current_avg = self.performance_metrics['average_processing_time']
        total = self.performance_metrics['total_analyses']
        
        self.performance_metrics['average_processing_time'] = (
            (current_avg * (total - 1) + processing_time) / total
        )
        
        self.performance_metrics['last_activity'] = datetime.now()
    
    def get_performance_summary(self) -> Dict:
        """Get performance summary"""
        total = self.performance_metrics['total_analyses']
        successful = self.performance_metrics['successful_analyses']
        
        return {
            "agent_name": self.agent_name,
            "agent_id": self.agent_id,
            "total_analyses": total,
            "success_rate": (successful / total * 100) if total > 0 else 0,
            "average_processing_time": self.performance_metrics['average_processing_time'],
            "cache_hit_rate": (
                self.performance_metrics['cache_hits'] / total * 100
            ) if total > 0 else 0,
            "performance_score": self.performance_metrics['performance_score'],
            "feedback_received": self.performance_metrics['feedback_received_count']
        }
    
    async def process_with_metadata(self, task_data: Dict) -> Dict:
        """Process with enhanced metadata tracking"""
        result = await self.analyze_stock(task_data)
        
        # Add processing metadata
        result['metadata'] = {
            'agent_id': self.agent_id,
            'processing_time': self.performance_metrics['average_processing_time'],
            'cache_hit': False,  # Will be updated if cached
            'performance_score': self.performance_metrics['performance_score']
        }
        
        return result
    
    def _create_error_response(self, ticker: str, error: str) -> Dict:
        """Create error response"""
        return {
            "ticker": ticker,
            "analysis_id": str(uuid.uuid4()),
            "timestamp": datetime.now().isoformat(),
            "agent_name": self.agent_name,
            "analysis_status": "error",
            "error": error,
            "recommendation": "HOLD",
            "confidence": 0
        }
    
    # Analysis methods (simplified for demonstration)
    async def _analyze_new_opportunity(self, task_data: Dict) -> Dict:
        """Analyze new opportunity"""
        ticker = task_data.get('ticker')
        
        # Get market data
        market_data = await self._get_market_data(ticker)
        
        # Perform technical analysis
        technical_analysis = self.technical_engine.analyze(market_data)
        
        # Get fundamentals
        fundamental_analysis = await self.fundamental_engine.analyze(ticker)
        
        # Combine analyses
        combined_analysis = self._combine_analyses(
            technical_analysis, 
            fundamental_analysis,
            task_data.get('market_context', {})
        )
        
        return combined_analysis
    
    async def _reevaluate_position(self, task_data: Dict) -> Dict:
        """Reevaluate existing position"""
        # Similar to new opportunity but with position context
        return await self._analyze_new_opportunity(task_data)
    
    async def _analyze_earnings_impact(self, task_data: Dict) -> Dict:
        """Analyze earnings impact"""
        # Earnings-specific analysis
        return await self._analyze_new_opportunity(task_data)
    
    async def _analyze_news_impact(self, task_data: Dict) -> Dict:
        """Analyze news impact"""
        # News-specific analysis
        return await self._analyze_new_opportunity(task_data)
    
    async def _get_market_data(self, ticker: str) -> Dict:
        """Get market data for ticker"""
        try:
            bars = await self.alpaca.get_bars(ticker, timeframe='1Day', limit=100)
            quote = await self.alpaca.get_latest_quote(ticker)
            
            return {
                'bars': bars,
                'quote': quote,
                'ticker': ticker
            }
        except Exception as e:
            self.logger.error(f"Failed to get market data for {ticker}: {str(e)}")
            return {}
    
    def _combine_analyses(self, technical: Dict, fundamental: Dict, market_context: Dict) -> Dict:
        """Combine technical and fundamental analyses"""
        # Simplified combination logic
        return {
            'recommendation': 'BUY',
            'confidence': 7,
            'expected_return': 0.15,
            'risk_reward_ratio': 2.5,
            'volatility': 0.25,
            'entry_price': 100,
            'stop_loss': 95,
            'primary_target': 110,
            'secondary_target': 115,
            'time_horizon': TimeHorizon.MEDIUM_TERM.value,
            'position_size': PositionSize.MEDIUM.value,
            'thesis': 'Strong technical setup with fundamental support',
            'catalysts': ['Earnings growth', 'Product launch'],
            'technical_signals': technical,
            'sector': 'Technology',
            'market_cap': 'Large',
            'correlation_to_spy': 0.65,
            'average_volume': 5000000,
            'technical_signal': {'score': 7},
            'analysis_type': AnalysisType.NEW_OPPORTUNITY.value,
            'data_quality_score': 8,
            'market_context': market_context
        }


# ========================================================================================
# SUPPORTING CLASSES
# ========================================================================================

class TechnicalAnalysisEngine:
    """Technical analysis engine"""
    
    def __init__(self, alpaca_provider):
        self.alpaca = alpaca_provider
        
    def analyze(self, market_data: Dict) -> Dict:
        """Perform technical analysis"""
        return {
            'rsi': 55,
            'macd_signal': 'bullish',
            'support': 95,
            'resistance': 105
        }


class FundamentalAnalysisEngine:
    """Fundamental analysis engine"""
    
    def __init__(self, alpaca_provider):
        self.alpaca = alpaca_provider
        
    async def analyze(self, ticker: str) -> Dict:
        """Perform fundamental analysis"""
        return {
            'pe_ratio': 25,
            'earnings_growth': 0.15,
            'revenue_growth': 0.12
        }


# ========================================================================================
# PARALLEL PROCESSING
# ========================================================================================

class JuniorAnalystPool:
    """Pool of Junior Analysts for parallel processing"""
    
    def __init__(self, llm_provider, alpaca_provider, config, pool_size: int = 3):
        self.analysts = [
            JuniorResearchAnalyst(llm_provider, alpaca_provider, config)
            for _ in range(pool_size)
        ]
        self.current_analyst = 0
        self.logger = logging.getLogger('junior_analyst_pool')
        
    async def analyze_batch(self, tickers: List[str], analysis_type: str) -> List[Dict]:
        """Analyze multiple tickers in parallel"""
        
        self.logger.info(f"Starting batch analysis for {len(tickers)} tickers")
        
        tasks = []
        for ticker in tickers:
            task_data = {
                "ticker": ticker,
                "task_type": analysis_type,
                "technical_signal": {"pattern": "batch_analysis", "score": 6}
            }
            tasks.append(self._analyze_with_available_analyst(task_data))
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filter out errors
        valid_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                self.logger.error(f"Analysis failed for {tickers[i]}: {str(result)}")
            else:
                valid_results.append(result)
        
        self.logger.info(f"Batch analysis complete: {len(valid_results)}/{len(tickers)} successful")
        
        return valid_results
    
    async def _analyze_with_available_analyst(self, task_data: Dict) -> Dict:
        """Assign task to available analyst (round-robin)"""
        analyst = self._get_available_analyst()
        return await analyst.analyze_stock(task_data)
    
    def _get_available_analyst(self) -> JuniorResearchAnalyst:
        """Get next available analyst (round-robin)"""
        analyst = self.analysts[self.current_analyst]
        self.current_analyst = (self.current_analyst + 1) % len(self.analysts)
        return analyst
    
    def get_pool_metrics(self) -> Dict:
        """Get metrics for all analysts in pool"""
        metrics = {
            'pool_size': len(self.analysts),
            'total_analyses': 0,
            'average_success_rate': 0,
            'average_processing_time': 0
        }
        
        for analyst in self.analysts:
            summary = analyst.get_performance_summary()
            metrics['total_analyses'] += summary['total_analyses']
            metrics['average_success_rate'] += summary['success_rate']
            metrics['average_processing_time'] += summary['average_processing_time']
        
        num_analysts = len(self.analysts)
        metrics['average_success_rate'] /= num_analysts
        metrics['average_processing_time'] /= num_analysts
        
        return metrics


# ========================================================================================
# FACTORY FUNCTION
# ========================================================================================

def create_junior_analyst(llm_provider, alpaca_provider, config) -> JuniorResearchAnalyst:
    """Factory function to create Junior Research Analyst"""
    return JuniorResearchAnalyst(llm_provider, alpaca_provider, config)