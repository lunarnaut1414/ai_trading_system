# agents/junior_research_analyst.py
"""
Junior Research Analyst Agent - Complete Implementation
Designed for macOS M2 Max with Claude AI integration and markdown reporting

This agent performs individual stock analysis by combining technical signals
with fundamental research to generate actionable trading recommendations.
"""

import asyncio
import json
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from enum import Enum
import uuid

class AnalysisType(Enum):
    NEW_OPPORTUNITY = "new_opportunity"
    POSITION_REEVALUATION = "position_reevaluation"
    RISK_ASSESSMENT = "risk_assessment"

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

class JuniorResearchAnalyst:
    """
    Junior Research Analyst Agent
    
    Combines technical analysis with fundamental research to provide
    comprehensive stock analysis and trading recommendations.
    """
    
    def __init__(self, llm_provider, alpaca_provider, config):
        """Initialize the Junior Research Analyst"""
        
        self.agent_name = "junior_research_analyst"
        self.agent_id = str(uuid.uuid4())
        
        # Core dependencies
        self.llm = llm_provider
        self.alpaca = alpaca_provider
        self.config = config
        
        # Setup logging
        self.logger = self._setup_logging()
        
        # Analysis engines
        self.technical_engine = TechnicalAnalysisEngine(alpaca_provider)
        self.fundamental_engine = FundamentalAnalysisEngine(alpaca_provider)
        
        # Performance tracking
        self.analysis_cache = {}
        self.performance_metrics = {
            "total_analyses": 0,
            "successful_analyses": 0,
            "failed_analyses": 0,
            "average_processing_time": 0.0,
            "cache_hits": 0,
            "last_activity": None
        }
        
        self.logger.info(f"✅ Junior Research Analyst initialized with ID: {self.agent_id}")
    
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
        """
        Main entry point for stock analysis
        
        Args:
            task_data: Dictionary containing analysis parameters
                - task_type: AnalysisType enum value
                - ticker: Stock symbol to analyze
                - technical_signal: Optional technical signal data
                - current_position: Optional current position data
        
        Returns:
            Dict: Complete analysis results with recommendations
        """
        
        start_time = datetime.now()
        analysis_id = str(uuid.uuid4())
        
        try:
            self.logger.info(f"🔍 Starting analysis {analysis_id} for {task_data.get('ticker', 'UNKNOWN')}")
            
            # Validate input
            validation_result = self._validate_task_data(task_data)
            if not validation_result["valid"]:
                return self._create_error_response(validation_result["error"], analysis_id)
            
            # Check cache first
            cache_key = self._generate_cache_key(task_data)
            if cache_key in self.analysis_cache:
                cached_result = self.analysis_cache[cache_key]
                if self._is_cache_fresh(cached_result["timestamp"]):
                    self.performance_metrics["cache_hits"] += 1
                    self.logger.info(f"📋 Using cached analysis for {task_data['ticker']}")
                    return cached_result["data"]
            
            # Perform analysis based on type
            task_type = AnalysisType(task_data["task_type"])
            
            if task_type == AnalysisType.NEW_OPPORTUNITY:
                result = await self._analyze_new_opportunity(task_data, analysis_id)
            elif task_type == AnalysisType.POSITION_REEVALUATION:
                result = await self._reevaluate_position(task_data, analysis_id)
            elif task_type == AnalysisType.RISK_ASSESSMENT:
                result = await self._assess_risk(task_data, analysis_id)
            else:
                return self._create_error_response(f"Unknown task type: {task_type}", analysis_id)
            
            # Cache successful result
            if result.get("metadata", {}).get("status") == "success":
                self.analysis_cache[cache_key] = {
                    "data": result,
                    "timestamp": datetime.now()
                }
            
            # Update performance metrics
            processing_time = (datetime.now() - start_time).total_seconds()
            self._update_performance_metrics(processing_time, True)
            
            self.logger.info(f"✅ Analysis {analysis_id} completed in {processing_time:.2f}s")
            return result
            
        except Exception as e:
            # Update performance metrics for failure
            processing_time = (datetime.now() - start_time).total_seconds()
            self._update_performance_metrics(processing_time, False)
            
            self.logger.error(f"❌ Analysis {analysis_id} failed: {str(e)}")
            return self._create_error_response(str(e), analysis_id)
    
    async def _analyze_new_opportunity(self, task_data: Dict, analysis_id: str) -> Dict:
        """Analyze a new stock opportunity"""
        
        ticker = task_data["ticker"]
        technical_signal = task_data.get("technical_signal", {})
        
        self.logger.info(f"📊 Analyzing new opportunity: {ticker}")
        
        # Gather market data
        market_data = await self._gather_market_data(ticker)
        if not market_data["success"]:
            return self._create_error_response(f"Failed to gather market data: {market_data['error']}", analysis_id)
        
        # Perform technical analysis
        technical_analysis = await self.technical_engine.analyze(ticker, market_data["data"])
        
        # Perform fundamental analysis
        fundamental_analysis = await self.fundamental_engine.analyze(ticker)
        
        # Generate AI-powered synthesis
        synthesis = await self._generate_ai_synthesis({
            "ticker": ticker,
            "technical_signal": technical_signal,
            "technical_analysis": technical_analysis,
            "fundamental_analysis": fundamental_analysis,
            "market_data": market_data["data"],
            "analysis_type": "new_opportunity"
        })
        
        if not synthesis["success"]:
            return self._create_fallback_analysis(ticker, technical_signal, analysis_id)
        
        # Create comprehensive result
        result = {
            "analysis_id": analysis_id,
            "ticker": ticker,
            "timestamp": datetime.now().isoformat(),
            "analysis_type": AnalysisType.NEW_OPPORTUNITY.value,
            
            # Core recommendation
            "recommendation": synthesis["recommendation"],
            "confidence": synthesis["confidence"],
            "time_horizon": synthesis["time_horizon"],
            "position_size": synthesis["position_size"],
            
            # Targets and risk management
            "entry_target": synthesis["entry_target"],
            "exit_targets": synthesis["exit_targets"],
            "stop_loss": synthesis["stop_loss"],
            "risk_reward_ratio": synthesis["risk_reward_ratio"],
            
            # Analysis details
            "investment_thesis": synthesis["investment_thesis"],
            "risk_factors": synthesis["risk_factors"],
            "catalyst_timeline": synthesis["catalyst_timeline"],
            
            # Supporting data
            "technical_summary": technical_analysis.get("summary", "Not available"),
            "fundamental_summary": fundamental_analysis.get("summary", "Not available"),
            "market_context": self._extract_market_context(market_data["data"]),
            
            # Metadata
            "metadata": {
                "status": "success",
                "processing_time": 0,  # Will be updated by caller
                "data_quality": self._assess_data_quality(market_data["data"], fundamental_analysis),
                "analysis_version": "1.0"
            }
        }
        
        return result
    
    async def _reevaluate_position(self, task_data: Dict, analysis_id: str) -> Dict:
        """Reevaluate an existing position"""
        
        ticker = task_data["ticker"]
        current_position = task_data.get("current_position", {})
        
        self.logger.info(f"🔄 Reevaluating position: {ticker}")
        
        # Gather updated market data
        market_data = await self._gather_market_data(ticker)
        if not market_data["success"]:
            return self._create_error_response(f"Failed to gather market data: {market_data['error']}", analysis_id)
        
        # Get updated analysis
        technical_analysis = await self.technical_engine.analyze(ticker, market_data["data"])
        fundamental_analysis = await self.fundamental_engine.analyze(ticker)
        
        # Generate AI-powered reevaluation
        reevaluation = await self._generate_ai_synthesis({
            "ticker": ticker,
            "current_position": current_position,
            "technical_analysis": technical_analysis,
            "fundamental_analysis": fundamental_analysis,
            "market_data": market_data["data"],
            "analysis_type": "position_reevaluation"
        })
        
        if not reevaluation["success"]:
            return self._create_fallback_reevaluation(ticker, current_position, analysis_id)
        
        # Create reevaluation result
        result = {
            "analysis_id": analysis_id,
            "ticker": ticker,
            "timestamp": datetime.now().isoformat(),
            "analysis_type": AnalysisType.POSITION_REEVALUATION.value,
            
            # Updated recommendation
            "action": reevaluation["action"],  # hold, increase, decrease, close
            "updated_confidence": reevaluation["confidence"],
            "updated_targets": reevaluation["targets"],
            "updated_stop_loss": reevaluation["stop_loss"],
            
            # Change analysis
            "conviction_change": reevaluation["conviction_change"],
            "new_developments": reevaluation["new_developments"],
            "risk_assessment": reevaluation["risk_assessment"],
            "recommendation_rationale": reevaluation["rationale"],
            
            # Context
            "original_position": current_position,
            "market_developments": self._identify_market_developments(market_data["data"]),
            
            # Metadata
            "metadata": {
                "status": "success",
                "processing_time": 0,
                "data_quality": self._assess_data_quality(market_data["data"], fundamental_analysis),
                "analysis_version": "1.0"
            }
        }
        
        return result
    
    async def _assess_risk(self, task_data: Dict, analysis_id: str) -> Dict:
        """Assess risk for a position or opportunity"""
        
        ticker = task_data["ticker"]
        position_data = task_data.get("position_data", {})
        
        self.logger.info(f"⚠️ Assessing risk for: {ticker}")
        
        # Gather risk-relevant data
        market_data = await self._gather_market_data(ticker)
        if not market_data["success"]:
            return self._create_error_response(f"Failed to gather market data: {market_data['error']}", analysis_id)
        
        # Perform risk-focused analysis
        risk_analysis = await self._generate_ai_synthesis({
            "ticker": ticker,
            "position_data": position_data,
            "market_data": market_data["data"],
            "analysis_type": "risk_assessment"
        })
        
        if not risk_analysis["success"]:
            return self._create_fallback_risk_assessment(ticker, analysis_id)
        
        result = {
            "analysis_id": analysis_id,
            "ticker": ticker,
            "timestamp": datetime.now().isoformat(),
            "analysis_type": AnalysisType.RISK_ASSESSMENT.value,
            
            "risk_level": risk_analysis["risk_level"],
            "risk_factors": risk_analysis["risk_factors"],
            "risk_score": risk_analysis["risk_score"],
            "volatility_assessment": risk_analysis["volatility_assessment"],
            "downside_scenarios": risk_analysis["downside_scenarios"],
            "risk_mitigation": risk_analysis["risk_mitigation"],
            
            "metadata": {
                "status": "success",
                "processing_time": 0,
                "analysis_version": "1.0"
            }
        }
        
        return result
    
    async def _gather_market_data(self, ticker: str) -> Dict:
        """Gather comprehensive market data for analysis"""
        
        try:
            # Get historical price data (30 days)
            end_date = datetime.now()
            start_date = end_date - timedelta(days=30)
            
            price_data = await self.alpaca.get_market_data(
                symbols=[ticker],
                timeframe="1Day",
                start=start_date.isoformat(),
                end=end_date.isoformat()
            )
            
            # Get current quote
            current_quote = await self.alpaca.get_current_quote(ticker)
            
            # Get technical indicators
            technical_indicators = await self.alpaca.get_technical_indicators(ticker)
            
            # Get news
            news_data = await self.alpaca.get_news(symbol=ticker, limit=10)
            
            return {
                "success": True,
                "data": {
                    "price_data": price_data.get(ticker, []),
                    "current_quote": current_quote,
                    "technical_indicators": technical_indicators,
                    "news": news_data,
                    "timestamp": datetime.now().isoformat()
                }
            }
            
        except Exception as e:
            self.logger.error(f"Failed to gather market data for {ticker}: {str(e)}")
            return {"success": False, "error": str(e)}
    
    async def _generate_ai_synthesis(self, analysis_data: Dict) -> Dict:
        """Generate AI-powered analysis synthesis using Claude"""
        
        try:
            # Create comprehensive prompt
            prompt = self._build_analysis_prompt(analysis_data)
            
            # Get Claude's analysis
            response = await self.llm.generate_completion(
                prompt=prompt,
                max_tokens=2000,
                temperature=0.3,
                response_format="json"
            )
            
            # Parse and validate response
            parsed_response = json.loads(response)
            validated_response = self._validate_ai_response(parsed_response, analysis_data["analysis_type"])
            
            if validated_response["valid"]:
                return {"success": True, **validated_response["data"]}
            else:
                self.logger.warning(f"AI response validation failed: {validated_response['error']}")
                return {"success": False, "error": validated_response["error"]}
                
        except Exception as e:
            self.logger.error(f"AI synthesis failed: {str(e)}")
            return {"success": False, "error": str(e)}
    
    def _build_analysis_prompt(self, analysis_data: Dict) -> str:
        """Build comprehensive analysis prompt for Claude"""
        
        analysis_type = analysis_data["analysis_type"]
        ticker = analysis_data["ticker"]
        
        if analysis_type == "new_opportunity":
            return f"""
You are a Junior Research Analyst for an institutional trading firm. Analyze {ticker} for a potential new position.

TECHNICAL SIGNAL:
{json.dumps(analysis_data.get('technical_signal', {}), indent=2)}

TECHNICAL ANALYSIS:
{json.dumps(analysis_data.get('technical_analysis', {}), indent=2)}

FUNDAMENTAL ANALYSIS:
{json.dumps(analysis_data.get('fundamental_analysis', {}), indent=2)}

MARKET DATA:
Current Price: {analysis_data.get('market_data', {}).get('current_quote', {}).get('price', 'N/A')}
Recent News: {len(analysis_data.get('market_data', {}).get('news', []))} articles available

Provide a comprehensive analysis in JSON format with these exact fields:
{{
    "recommendation": "strong_buy|buy|hold|sell|strong_sell",
    "confidence": 1-10,
    "time_horizon": "short_term|medium_term|long_term",
    "position_size": "small|medium|large|max",
    "entry_target": number,
    "exit_targets": {{"primary": number, "secondary": number}},
    "stop_loss": number,
    "risk_reward_ratio": number,
    "investment_thesis": "detailed thesis string",
    "risk_factors": ["risk1", "risk2", "risk3"],
    "catalyst_timeline": {{"short_term": ["catalyst1"], "medium_term": ["catalyst2"], "long_term": ["catalyst3"]}}
}}

Focus on actionable insights with specific price targets and risk management parameters.
"""
        
        elif analysis_type == "position_reevaluation":
            return f"""
You are a Junior Research Analyst reviewing an existing position in {ticker}.

CURRENT POSITION:
{json.dumps(analysis_data.get('current_position', {}), indent=2)}

UPDATED TECHNICAL ANALYSIS:
{json.dumps(analysis_data.get('technical_analysis', {}), indent=2)}

UPDATED FUNDAMENTAL ANALYSIS:
{json.dumps(analysis_data.get('fundamental_analysis', {}), indent=2)}

CURRENT MARKET DATA:
{json.dumps(analysis_data.get('market_data', {}), indent=2)}

Provide a position reevaluation in JSON format:
{{
    "action": "hold|increase|decrease|close",
    "confidence": 1-10,
    "targets": {{"primary": number, "secondary": number}},
    "stop_loss": number,
    "conviction_change": "increased|decreased|unchanged",
    "new_developments": "string describing new information",
    "risk_assessment": "updated risk assessment string",
    "rationale": "detailed rationale for recommendation"
}}

Focus on what has changed since the original analysis and how it affects the position.
"""
        
        elif analysis_type == "risk_assessment":
            return f"""
You are a Junior Research Analyst conducting a comprehensive risk assessment for {ticker}.

POSITION DATA:
{json.dumps(analysis_data.get('position_data', {}), indent=2)}

MARKET DATA:
{json.dumps(analysis_data.get('market_data', {}), indent=2)}

Provide a risk assessment in JSON format:
{{
    "risk_level": "low|medium|high|very_high",
    "risk_score": 1-10,
    "risk_factors": ["factor1", "factor2", "factor3"],
    "volatility_assessment": "assessment string",
    "downside_scenarios": ["scenario1", "scenario2"],
    "risk_mitigation": ["strategy1", "strategy2"]
}}

Focus on identifying and quantifying specific risks and mitigation strategies.
"""
        
        return ""
    
    def _validate_ai_response(self, response: Dict, analysis_type: str) -> Dict:
        """Validate AI response structure and content"""
        
        try:
            if analysis_type == "new_opportunity":
                required_fields = [
                    "recommendation", "confidence", "time_horizon", "position_size",
                    "entry_target", "exit_targets", "stop_loss", "risk_reward_ratio",
                    "investment_thesis", "risk_factors", "catalyst_timeline"
                ]
            elif analysis_type == "position_reevaluation":
                required_fields = [
                    "action", "confidence", "targets", "stop_loss",
                    "conviction_change", "new_developments", "risk_assessment", "rationale"
                ]
            elif analysis_type == "risk_assessment":
                required_fields = [
                    "risk_level", "risk_score", "risk_factors", "volatility_assessment",
                    "downside_scenarios", "risk_mitigation"
                ]
            else:
                return {"valid": False, "error": f"Unknown analysis type: {analysis_type}"}
            
            # Check required fields
            for field in required_fields:
                if field not in response:
                    return {"valid": False, "error": f"Missing required field: {field}"}
            
            # Validate confidence scores
            if "confidence" in response:
                confidence = response["confidence"]
                if not isinstance(confidence, (int, float)) or confidence < 1 or confidence > 10:
                    return {"valid": False, "error": "Confidence must be between 1 and 10"}
            
            # Validate numerical fields
            numerical_fields = ["entry_target", "stop_loss", "risk_reward_ratio", "risk_score"]
            for field in numerical_fields:
                if field in response and not isinstance(response[field], (int, float)):
                    return {"valid": False, "error": f"{field} must be a number"}
            
            return {"valid": True, "data": response}
            
        except Exception as e:
            return {"valid": False, "error": f"Validation error: {str(e)}"}
    
    def _create_fallback_analysis(self, ticker: str, technical_signal: Dict, analysis_id: str) -> Dict:
        """Create fallback analysis when AI synthesis fails"""
        
        # Extract basic info from technical signal
        signal_score = technical_signal.get("score", 5)
        pattern = technical_signal.get("pattern", "unknown")
        
        # Create conservative fallback recommendation
        if signal_score >= 8:
            recommendation = RecommendationType.BUY.value
            confidence = 6
        elif signal_score >= 6:
            recommendation = RecommendationType.HOLD.value
            confidence = 5
        else:
            recommendation = RecommendationType.HOLD.value
            confidence = 4
        
        return {
            "analysis_id": analysis_id,
            "ticker": ticker,
            "timestamp": datetime.now().isoformat(),
            "analysis_type": AnalysisType.NEW_OPPORTUNITY.value,
            
            "recommendation": recommendation,
            "confidence": confidence,
            "time_horizon": TimeHorizon.MEDIUM_TERM.value,
            "position_size": PositionSize.SMALL.value,
            
            "entry_target": 0,
            "exit_targets": {"primary": 0, "secondary": 0},
            "stop_loss": 0,
            "risk_reward_ratio": 2.0,
            
            "investment_thesis": f"Fallback analysis based on technical pattern: {pattern}",
            "risk_factors": ["Analysis system unavailable", "Limited data quality"],
            "catalyst_timeline": {"short_term": [], "medium_term": [], "long_term": []},
            
            "technical_summary": f"Technical signal detected: {pattern} (score: {signal_score})",
            "fundamental_summary": "Fundamental analysis unavailable",
            "market_context": {"status": "unavailable"},
            
            "metadata": {
                "status": "fallback",
                "processing_time": 0,
                "data_quality": "limited",
                "analysis_version": "1.0",
                "fallback_reason": "AI synthesis failed"
            }
        }
    
    def _create_fallback_reevaluation(self, ticker: str, current_position: Dict, analysis_id: str) -> Dict:
        """Create fallback reevaluation when AI synthesis fails"""
        
        return {
            "analysis_id": analysis_id,
            "ticker": ticker,
            "timestamp": datetime.now().isoformat(),
            "analysis_type": AnalysisType.POSITION_REEVALUATION.value,
            
            "action": "hold",
            "updated_confidence": 5,
            "updated_targets": {"primary": 0, "secondary": 0},
            "updated_stop_loss": 0,
            "conviction_change": "unchanged",
            "new_developments": "Unable to assess due to system limitations",
            "risk_assessment": "Cannot determine current risk level",
            "recommendation_rationale": "Fallback hold recommendation due to analysis failure",
            
            "original_position": current_position,
            "market_developments": {"status": "unavailable"},
            
            "metadata": {
                "status": "fallback",
                "processing_time": 0,
                "data_quality": "limited",
                "analysis_version": "1.0",
                "fallback_reason": "AI synthesis failed"
            }
        }
    
    def _create_fallback_risk_assessment(self, ticker: str, analysis_id: str) -> Dict:
        """Create fallback risk assessment when AI synthesis fails"""
        
        return {
            "analysis_id": analysis_id,
            "ticker": ticker,
            "timestamp": datetime.now().isoformat(),
            "analysis_type": AnalysisType.RISK_ASSESSMENT.value,
            
            "risk_level": RiskLevel.MEDIUM.value,
            "risk_score": 5,
            "risk_factors": ["System analysis unavailable", "Limited data quality"],
            "volatility_assessment": "Cannot assess volatility due to system limitations",
            "downside_scenarios": ["Analysis system failure", "Data quality issues"],
            "risk_mitigation": ["Manual review required", "Reduce position size"],
            
            "metadata": {
                "status": "fallback",
                "processing_time": 0,
                "analysis_version": "1.0",
                "fallback_reason": "AI synthesis failed"
            }
        }
    
    def _validate_task_data(self, task_data: Dict) -> Dict:
        """Validate input task data"""
        
        if not isinstance(task_data, dict):
            return {"valid": False, "error": "Task data must be a dictionary"}
        
        if "task_type" not in task_data:
            return {"valid": False, "error": "Missing required field: task_type"}
        
        if "ticker" not in task_data:
            return {"valid": False, "error": "Missing required field: ticker"}
        
        try:
            AnalysisType(task_data["task_type"])
        except ValueError:
            return {"valid": False, "error": f"Invalid task_type: {task_data['task_type']}"}
        
        ticker = task_data["ticker"]
        if not isinstance(ticker, str) or len(ticker) == 0:
            return {"valid": False, "error": "Ticker must be a non-empty string"}
        
        return {"valid": True}
    
    def _generate_cache_key(self, task_data: Dict) -> str:
        """Generate cache key for analysis results"""
        
        ticker = task_data["ticker"]
        task_type = task_data["task_type"]
        
        # Include date for daily cache invalidation
        date_str = datetime.now().strftime("%Y-%m-%d")
        
        return f"{ticker}_{task_type}_{date_str}"
    
    def _is_cache_fresh(self, timestamp: datetime, max_age_hours: int = 2) -> bool:
        """Check if cached result is still fresh"""
        
        age = datetime.now() - timestamp
        return age.total_seconds() < (max_age_hours * 3600)
    
    def _extract_market_context(self, market_data: Dict) -> Dict:
        """Extract relevant market context from data"""
        
        try:
            current_quote = market_data.get("current_quote", {})
            technical_indicators = market_data.get("technical_indicators", {})
            
            return {
                "current_price": current_quote.get("price", 0),
                "volume": current_quote.get("volume", 0),
                "rsi": technical_indicators.get("rsi", 50),
                "moving_averages": technical_indicators.get("moving_averages", {}),
                "trend": technical_indicators.get("trend", "neutral")
            }
        except Exception:
            return {"status": "unavailable"}
    
    def _identify_market_developments(self, market_data: Dict) -> Dict:
        """Identify recent market developments"""
        
        try:
            news = market_data.get("news", [])
            recent_news_count = len([n for n in news if n.get("created_at", "")[:10] == datetime.now().strftime("%Y-%m-%d")])
            
            return {
                "recent_news_count": recent_news_count,
                "notable_developments": "Analysis of developments would require more sophisticated processing",
                "sentiment_shift": "neutral"
            }
        except Exception:
            return {"status": "unavailable"}
    
    def _assess_data_quality(self, market_data: Dict, fundamental_data: Dict) -> str:
        """Assess overall data quality for analysis"""
        
        try:
            quality_score = 0
            max_score = 4
            
            # Check price data availability
            if market_data.get("price_data") and len(market_data["price_data"]) > 0:
                quality_score += 1
            
            # Check current quote
            if market_data.get("current_quote", {}).get("price"):
                quality_score += 1
            
            # Check technical indicators
            if market_data.get("technical_indicators"):
                quality_score += 1
            
            # Check fundamental data
            if fundamental_data and len(fundamental_data) > 0:
                quality_score += 1
            
            quality_ratio = quality_score / max_score
            
            if quality_ratio >= 0.8:
                return "excellent"
            elif quality_ratio >= 0.6:
                return "good"
            elif quality_ratio >= 0.4:
                return "fair"
            else:
                return "poor"
                
        except Exception:
            return "unknown"
    
    def _update_performance_metrics(self, processing_time: float, success: bool):
        """Update agent performance metrics"""
        
        self.performance_metrics["total_analyses"] += 1
        self.performance_metrics["last_activity"] = datetime.now()
        
        if success:
            self.performance_metrics["successful_analyses"] += 1
        else:
            self.performance_metrics["failed_analyses"] += 1
        
        # Update average processing time
        total_time = self.performance_metrics["average_processing_time"] * (self.performance_metrics["total_analyses"] - 1)
        total_time += processing_time
        self.performance_metrics["average_processing_time"] = total_time / self.performance_metrics["total_analyses"]
    
    def _create_error_response(self, error_message: str, analysis_id: str) -> Dict:
        """Create standardized error response"""
        
        return {
            "analysis_id": analysis_id,
            "timestamp": datetime.now().isoformat(),
            "metadata": {
                "status": "error",
                "error": error_message,
                "analysis_version": "1.0"
            }
        }
    
    def get_performance_summary(self) -> Dict:
        """Get agent performance summary"""
        
        total = self.performance_metrics["total_analyses"]
        success_rate = (self.performance_metrics["successful_analyses"] / total * 100) if total > 0 else 0
        
        return {
            "agent_name": self.agent_name,
            "agent_id": self.agent_id,
            "total_analyses": total,
            "success_rate": f"{success_rate:.1f}%",
            "average_processing_time": f"{self.performance_metrics['average_processing_time']:.2f}s",
            "cache_hit_rate": f"{(self.performance_metrics['cache_hits'] / total * 100) if total > 0 else 0:.1f}%",
            "last_activity": self.performance_metrics["last_activity"].isoformat() if self.performance_metrics["last_activity"] else None
        }
    
    def generate_markdown_report(self, analysis_result: Dict) -> str:
        """Generate markdown report for analysis result"""
        
        if analysis_result.get("metadata", {}).get("status") == "error":
            return f"""# Analysis Error Report

**Ticker:** {analysis_result.get('ticker', 'N/A')}
**Analysis ID:** {analysis_result.get('analysis_id', 'N/A')}
**Timestamp:** {analysis_result.get('timestamp', 'N/A')}

## Error Details
{analysis_result.get('metadata', {}).get('error', 'Unknown error')}
"""
        
        result = analysis_result
        analysis_type = result.get("analysis_type", "unknown")
        
        if analysis_type == AnalysisType.NEW_OPPORTUNITY.value:
            return self._generate_opportunity_report(result)
        elif analysis_type == AnalysisType.POSITION_REEVALUATION.value:
            return self._generate_reevaluation_report(result)
        elif analysis_type == AnalysisType.RISK_ASSESSMENT.value:
            return self._generate_risk_report(result)
        else:
            return self._generate_generic_report(result)
    
    def _generate_opportunity_report(self, result: Dict) -> str:
        """Generate markdown report for new opportunity analysis"""
        
        return f"""# Stock Analysis Report: {result['ticker']}

**Analysis ID:** {result['analysis_id']}
**Timestamp:** {result['timestamp']}
**Analyst:** Junior Research Analyst

## Executive Summary

**Recommendation:** {result['recommendation'].upper()}
**Confidence:** {result['confidence']}/10
**Time Horizon:** {result['time_horizon'].replace('_', ' ').title()}
**Suggested Position Size:** {result['position_size'].replace('_', ' ').title()}

## Price Targets & Risk Management

| Metric | Value |
|--------|--------|
| Entry Target | ${result['entry_target']:.2f} |
| Primary Exit | ${result['exit_targets']['primary']:.2f} |
| Secondary Exit | ${result['exit_targets']['secondary']:.2f} |
| Stop Loss | ${result['stop_loss']:.2f} |
| Risk/Reward Ratio | {result['risk_reward_ratio']:.1f}:1 |

## Investment Thesis

{result['investment_thesis']}

## Risk Factors

{chr(10).join([f"- {risk}" for risk in result['risk_factors']])}

## Catalyst Timeline

### Short Term
{chr(10).join([f"- {catalyst}" for catalyst in result.get('catalyst_timeline', {}).get('short_term', [])])}

### Medium Term
{chr(10).join([f"- {catalyst}" for catalyst in result.get('catalyst_timeline', {}).get('medium_term', [])])}

### Long Term
{chr(10).join([f"- {catalyst}" for catalyst in result.get('catalyst_timeline', {}).get('long_term', [])])}

## Technical Analysis Summary

{result['technical_summary']}

## Fundamental Analysis Summary

{result['fundamental_summary']}

## Market Context

**Current Price:** ${result.get('market_context', {}).get('current_price', 0):.2f}
**RSI:** {result.get('market_context', {}).get('rsi', 50):.1f}
**Trend:** {result.get('market_context', {}).get('trend', 'Neutral').title()}

## Metadata

**Data Quality:** {result.get('metadata', {}).get('data_quality', 'Unknown').title()}
**Analysis Status:** {result.get('metadata', {}).get('status', 'Unknown').title()}
**Version:** {result.get('metadata', {}).get('analysis_version', '1.0')}

---
*Generated by AI Trading System - Junior Research Analyst*
"""
    
    def _generate_reevaluation_report(self, result: Dict) -> str:
        """Generate markdown report for position reevaluation"""
        
        return f"""# Position Reevaluation Report: {result['ticker']}

**Analysis ID:** {result['analysis_id']}
**Timestamp:** {result['timestamp']}
**Analyst:** Junior Research Analyst

## Recommendation

**Action:** {result['action'].upper()}
**Updated Confidence:** {result['updated_confidence']}/10
**Conviction Change:** {result['conviction_change'].replace('_', ' ').title()}

## Updated Targets

| Metric | Value |
|--------|--------|
| Primary Target | ${result['updated_targets']['primary']:.2f} |
| Secondary Target | ${result['updated_targets']['secondary']:.2f} |
| Updated Stop Loss | ${result['updated_stop_loss']:.2f} |

## Analysis Summary

### New Developments
{result['new_developments']}

### Risk Assessment
{result['risk_assessment']}

### Recommendation Rationale
{result['recommendation_rationale']}

## Original Position Details

{chr(10).join([f"**{k.replace('_', ' ').title()}:** {v}" for k, v in result.get('original_position', {}).items()])}

## Market Developments

**Status:** {result.get('market_developments', {}).get('status', 'Available')}

---
*Generated by AI Trading System - Junior Research Analyst*
"""
    
    def _generate_risk_report(self, result: Dict) -> str:
        """Generate markdown report for risk assessment"""
        
        return f"""# Risk Assessment Report: {result['ticker']}

**Analysis ID:** {result['analysis_id']}
**Timestamp:** {result['timestamp']}
**Analyst:** Junior Research Analyst

## Risk Overview

**Risk Level:** {result['risk_level'].upper()}
**Risk Score:** {result['risk_score']}/10

## Risk Factors

{chr(10).join([f"- {factor}" for factor in result['risk_factors']])}

## Volatility Assessment

{result['volatility_assessment']}

## Downside Scenarios

{chr(10).join([f"- {scenario}" for scenario in result['downside_scenarios']])}

## Risk Mitigation Strategies

{chr(10).join([f"- {strategy}" for strategy in result['risk_mitigation']])}

---
*Generated by AI Trading System - Junior Research Analyst*
"""
    
    def _generate_generic_report(self, result: Dict) -> str:
        """Generate generic markdown report"""
        
        return f"""# Analysis Report: {result.get('ticker', 'N/A')}

**Analysis ID:** {result.get('analysis_id', 'N/A')}
**Timestamp:** {result.get('timestamp', 'N/A')}
**Type:** {result.get('analysis_type', 'Unknown').replace('_', ' ').title()}

## Results

{chr(10).join([f"**{k.replace('_', ' ').title()}:** {v}" for k, v in result.items() if k not in ['analysis_id', 'timestamp', 'ticker', 'analysis_type', 'metadata']])}

---
*Generated by AI Trading System - Junior Research Analyst*
"""


# Supporting Analysis Engines

class TechnicalAnalysisEngine:
    """Technical analysis engine for the Junior Research Analyst"""
    
    def __init__(self, alpaca_provider):
        self.alpaca = alpaca_provider
        self.logger = logging.getLogger("technical_analysis_engine")
    
    async def analyze(self, ticker: str, market_data: Dict) -> Dict:
        """Perform technical analysis on stock data"""
        
        try:
            price_data = market_data.get("price_data", [])
            technical_indicators = market_data.get("technical_indicators", {})
            
            if not price_data:
                return {"summary": "Insufficient price data for technical analysis"}
            
            # Calculate basic technical metrics
            recent_prices = [float(bar.get("close", 0)) for bar in price_data[-10:]]
            current_price = recent_prices[-1] if recent_prices else 0
            
            # Simple trend analysis
            if len(recent_prices) >= 5:
                sma_5 = sum(recent_prices[-5:]) / 5
                trend = "bullish" if current_price > sma_5 else "bearish"
            else:
                trend = "neutral"
            
            # Volume analysis
            recent_volumes = [float(bar.get("volume", 0)) for bar in price_data[-5:]]
            avg_volume = sum(recent_volumes) / len(recent_volumes) if recent_volumes else 0
            
            return {
                "trend": trend,
                "current_price": current_price,
                "sma_5": sma_5 if 'sma_5' in locals() else current_price,
                "average_volume": avg_volume,
                "rsi": technical_indicators.get("rsi", 50),
                "summary": f"Technical analysis shows {trend} trend with RSI at {technical_indicators.get('rsi', 50):.1f}"
            }
            
        except Exception as e:
            self.logger.error(f"Technical analysis failed for {ticker}: {str(e)}")
            return {"summary": f"Technical analysis failed: {str(e)}"}


class FundamentalAnalysisEngine:
    """Fundamental analysis engine for the Junior Research Analyst"""
    
    def __init__(self, alpaca_provider):
        self.alpaca = alpaca_provider
        self.logger = logging.getLogger("fundamental_analysis_engine")
    
    async def analyze(self, ticker: str) -> Dict:
        """Perform fundamental analysis on stock"""
        
        try:
            # Get company information
            company_info = await self.alpaca.get_company_info(ticker)
            
            # Get financial data
            financial_data = await self.alpaca.get_financial_data(ticker)
            
            if not company_info and not financial_data:
                return {"summary": "Fundamental data not available"}
            
            # Basic fundamental metrics
            sector = company_info.get("sector", "Unknown")
            market_cap = company_info.get("market_cap", 0)
            
            return {
                "sector": sector,
                "market_cap": market_cap,
                "pe_ratio": financial_data.get("pe_ratio"),
                "debt_to_equity": financial_data.get("debt_to_equity"),
                "summary": f"Company in {sector} sector with market cap of ${market_cap/1e9:.1f}B"
            }
            
        except Exception as e:
            self.logger.error(f"Fundamental analysis failed for {ticker}: {str(e)}")
            return {"summary": f"Fundamental analysis failed: {str(e)}"}


# Factory function for easy initialization
def create_junior_analyst(llm_provider, alpaca_provider, config) -> JuniorResearchAnalyst:
    """Factory function to create Junior Research Analyst"""
    return JuniorResearchAnalyst(llm_provider, alpaca_provider, config)