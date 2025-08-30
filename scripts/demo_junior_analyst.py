# demo_junior_analyst.py
"""
Junior Research Analyst Demo Script - CORRECTED VERSION
Demonstrates the complete functionality of the Junior Analyst Agent
Optimized for macOS M2 Max with Claude AI integration
"""

import asyncio
import json
import os
import sys
from datetime import datetime
from pathlib import Path

# Fix import path
project_root = Path(__file__).parent.parent if Path(__file__).parent.name == "scripts" else Path(__file__).parent
sys.path.insert(0, str(project_root))

# Import the agent (with error handling)
try:
    from agents.junior_research_analyst import JuniorResearchAnalyst, AnalysisType, create_junior_analyst
except ImportError as e:
    print(f"❌ Import Error: {e}")
    print("Please ensure:")
    print("1. agents/junior_research_analyst.py exists")
    print("2. You're running from the project root")
    print("3. Virtual environment is activated")
    sys.exit(1)


class MockLLMProvider:
    """Mock LLM provider for demonstration purposes"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.call_count = 0
    
    async def generate_completion(self, prompt: str, max_tokens: int = 1000, 
                                temperature: float = 0.3, response_format: str = "text") -> str:
        """Generate a mock completion for demo purposes"""
        
        self.call_count += 1
        
        # Add small delay to simulate API call
        await asyncio.sleep(0.1)
        
        if "new_opportunity" in prompt and "AAPL" in prompt:
            return json.dumps({
                "recommendation": "buy",
                "confidence": 8,
                "time_horizon": "medium_term",
                "position_size": "medium",
                "entry_target": 185.50,
                "exit_targets": {"primary": 195.00, "secondary": 200.00},
                "stop_loss": 178.00,
                "risk_reward_ratio": 2.5,
                "investment_thesis": "Strong technical breakout pattern combined with positive earnings momentum and sector rotation tailwinds. The ascending triangle formation suggests institutional accumulation with a high probability breakout above resistance at $185.",
                "risk_factors": ["Market volatility from Fed policy uncertainty", "Sector rotation risk from tech", "Earnings disappointment risk"],
                "catalyst_timeline": {
                    "short_term": ["Technical breakout confirmation above $185", "Volume surge on breakout"],
                    "medium_term": ["Q1 2024 earnings beat expectations", "Product launch announcement in March"],
                    "long_term": ["Market share expansion in services", "New AI technology adoption cycle"]
                }
            })
        
        elif "position_reevaluation" in prompt and "AAPL" in prompt:
            return json.dumps({
                "action": "increase",
                "confidence": 7,
                "targets": {"primary": 190.00, "secondary": 195.00},
                "stop_loss": 180.00,
                "conviction_change": "increased",
                "new_developments": "Recent earnings beat expectations by 15% with raised guidance for next quarter. Technical momentum has strengthened with volume confirmation above key resistance levels. Management commentary on AI integration driving services growth.",
                "risk_assessment": "Risk profile has improved due to strong fundamental performance and technical confirmation. Stop loss can be moved up to $180 to protect gains while maintaining upside exposure.",
                "rationale": "The combination of fundamental outperformance and technical strength supports increasing the position size while the setup remains favorable. Strong institutional buying flow supports the thesis."
            })
        
        elif "risk_assessment" in prompt and "AAPL" in prompt:
            return json.dumps({
                "risk_level": "medium",
                "risk_score": 6,
                "risk_factors": ["Market volatility from macro uncertainty", "Technology sector rotation risk", "Technical support breakdown below $175"],
                "volatility_assessment": "Moderate volatility expected with 15-20% annual volatility range based on historical patterns. Recent options flow suggests increased uncertainty around upcoming earnings, but implied volatility remains reasonable.",
                "downside_scenarios": ["Break below $175 support triggers technical selling", "Broader technology sector selloff on rate fears", "Market-wide correction impacts all risk assets"],
                "risk_mitigation": ["Maintain tight stop loss at $178 level", "Keep position sizing at 3% of portfolio maximum", "Consider hedging with technology sector ETF puts if correlation increases"]
            })
        
        # Fallback response
        return json.dumps({
            "status": "mock_response",
            "message": "This is a mock response for testing purposes"
        })


class MockAlpacaProvider:
    """Mock Alpaca provider for demonstration purposes"""
    
    def __init__(self, config):
        self.config = config
        self.call_count = 0
    
    async def get_market_data(self, symbols: list, timeframe: str = "1Day", 
                            start: str = None, end: str = None) -> dict:
        """Mock market data"""
        self.call_count += 1
        await asyncio.sleep(0.05)  # Simulate API delay
        
        symbol = symbols[0] if symbols else "AAPL"
        
        # Generate realistic mock price data
        mock_data = []
        base_price = 180.0
        
        for i in range(30):
            # Simulate realistic price movement
            daily_change = (i % 7 - 3) * 0.3  # Weekly pattern
            trend_change = i * 0.08  # Upward trend
            noise = (hash(f"{symbol}_{i}") % 100 - 50) * 0.02  # Pseudo-random noise
            
            close_price = base_price + daily_change + trend_change + noise
            
            mock_data.append({
                "timestamp": f"2024-01-{i+1:02d}T16:00:00Z",
                "open": round(close_price - 0.5, 2),
                "high": round(close_price + 1.2, 2),
                "low": round(close_price - 1.1, 2),
                "close": round(close_price, 2),
                "volume": 1000000 + (i * 15000) + (hash(f"{symbol}_vol_{i}") % 500000)
            })
        
        return {symbol: mock_data}
    
    async def get_current_quote(self, symbol: str) -> dict:
        """Mock current quote"""
        self.call_count += 1
        await asyncio.sleep(0.02)
        
        base_prices = {"AAPL": 184.50, "MSFT": 378.20, "GOOGL": 142.30}
        price = base_prices.get(symbol, 150.00)
        
        return {
            "symbol": symbol,
            "price": price,
            "bid": round(price - 0.05, 2),
            "ask": round(price + 0.05, 2),
            "volume": 1250000,
            "timestamp": datetime.now().isoformat()
        }
    
    async def get_technical_indicators(self, symbol: str) -> dict:
        """Mock technical indicators"""
        self.call_count += 1
        await asyncio.sleep(0.03)
        
        base_rsi = {"AAPL": 65.5, "MSFT": 58.2, "GOOGL": 72.1}
        rsi = base_rsi.get(symbol, 60.0)
        
        return {
            "rsi": rsi,
            "moving_averages": {
                "sma_20": 182.30 if symbol == "AAPL" else 150.00,
                "sma_50": 179.80 if symbol == "AAPL" else 145.00,
                "ema_12": 183.60 if symbol == "AAPL" else 152.00
            },
            "trend": "bullish" if rsi < 70 else "overbought",
            "support_levels": [180.00, 175.50] if symbol == "AAPL" else [145.00, 140.00],
            "resistance_levels": [185.00, 188.50] if symbol == "AAPL" else [155.00, 160.00]
        }
    
    async def get_news(self, symbol: str, limit: int = 10) -> list:
        """Mock news data"""
        self.call_count += 1
        await asyncio.sleep(0.05)
        
        news_templates = {
            "AAPL": [
                "Company reports strong Q4 earnings with 15% revenue growth",
                "Investment firm upgrades stock to Buy with $200 price target", 
                "New iPhone Pro Max receives positive early reviews",
                "Services revenue hits new record high in latest quarter"
            ],
            "MSFT": [
                "Cloud revenue grows 25% year-over-year exceeding estimates",
                "Major enterprise AI deal announced with Fortune 500 company",
                "Office 365 subscriptions reach new milestone",
                "Azure market share gains against competitors"
            ]
        }
        
        templates = news_templates.get(symbol, ["General market news for " + symbol])
        
        return [
            {
                "headline": f"{templates[i % len(templates)]}",
                "summary": f"Detailed analysis of {templates[i % len(templates)].lower()}",
                "created_at": f"2024-01-{15-i:02d}T{9+i:02d}:30:00Z",
                "source": ["Financial Times", "Reuters", "Bloomberg", "MarketWatch"][i % 4]
            }
            for i in range(min(limit, len(templates)))
        ]
    
    async def get_company_info(self, symbol: str) -> dict:
        """Mock company information"""
        self.call_count += 1
        await asyncio.sleep(0.02)
        
        company_data = {
            "AAPL": {
                "symbol": symbol,
                "name": "Apple Inc.",
                "sector": "Technology",
                "industry": "Consumer Electronics",
                "market_cap": 2.9e12,  # $2.9T
                "employees": 164000,
                "description": "Designs, manufactures, and markets smartphones, personal computers, tablets, wearables, and accessories"
            },
            "MSFT": {
                "symbol": symbol,
                "name": "Microsoft Corporation",
                "sector": "Technology", 
                "industry": "Software",
                "market_cap": 2.8e12,
                "employees": 221000,
                "description": "Develops, licenses, and supports software, services, devices, and solutions worldwide"
            }
        }
        
        return company_data.get(symbol, {
            "symbol": symbol,
            "name": f"{symbol} Inc.",
            "sector": "Technology",
            "industry": "Software",
            "market_cap": 1.0e12,
            "employees": 100000,
            "description": f"Mock company data for {symbol}"
        })
    
    async def get_financial_data(self, symbol: str) -> dict:
        """Mock financial data"""
        self.call_count += 1
        await asyncio.sleep(0.03)
        
        financial_data = {
            "AAPL": {
                "pe_ratio": 28.5,
                "debt_to_equity": 0.31,
                "roe": 0.175,
                "profit_margin": 0.233,
                "revenue_growth": 0.08,
                "earnings_growth": 0.11
            },
            "MSFT": {
                "pe_ratio": 32.1,
                "debt_to_equity": 0.27,
                "roe": 0.189,
                "profit_margin": 0.314,
                "revenue_growth": 0.12,
                "earnings_growth": 0.15
            }
        }
        
        return financial_data.get(symbol, {
            "pe_ratio": 25.0,
            "debt_to_equity": 0.25,
            "roe": 0.15,
            "profit_margin": 0.20,
            "revenue_growth": 0.10,
            "earnings_growth": 0.12
        })


class MockConfig:
    """Mock configuration for demonstration"""
    
    def __init__(self):
        self.ANTHROPIC_API_KEY = "mock_anthropic_key"
        self.MAX_POSITIONS = 10
        self.LOG_LEVEL = "INFO"
        self.CACHE_EXPIRY_HOURS = 2
        self.MAX_CONCURRENT_ANALYSES = 3
        self.REQUEST_TIMEOUT_SECONDS = 30


async def demo_new_opportunity_analysis():
    """Demonstrate new opportunity analysis"""
    
    print("🔍 DEMO: New Opportunity Analysis")
    print("=" * 60)
    
    # Setup mock components
    config = MockConfig()
    llm = MockLLMProvider(config.ANTHROPIC_API_KEY)
    alpaca = MockAlpacaProvider(config)
    
    # Create analyst
    analyst = create_junior_analyst(llm, alpaca, config)
    
    # Example technical signal from screener
    technical_signal = {
        "pattern": "ascending_triangle",
        "score": 8.2,
        "resistance_level": 185.00,
        "support_level": 180.00,
        "volume_confirmation": True,
        "formation_days": 8,
        "breakout_probability": 0.75,
        "risk_reward_ratio": 2.5
    }
    
    # Task data for new opportunity
    task_data = {
        "task_type": AnalysisType.NEW_OPPORTUNITY.value,
        "ticker": "AAPL",
        "technical_signal": technical_signal
    }
    
    print(f"📊 Analyzing {task_data['ticker']} for new position opportunity:")
    print(f"   • Pattern: {technical_signal['pattern']}")
    print(f"   • Signal Score: {technical_signal['score']}/10")
    print(f"   • Resistance: ${technical_signal['resistance_level']}")
    print(f"   • Support: ${technical_signal['support_level']}")
    print()
    
    # Perform analysis
    result = await analyst.analyze_stock(task_data)
    
    if result.get("metadata", {}).get("status") == "success":
        print("✅ Analysis Complete! Here's the summary:")
        print()
        print(f"📈 RECOMMENDATION: {result['recommendation'].upper()}")
        print(f"🎯 CONFIDENCE: {result['confidence']}/10")
        print(f"⏰ TIME HORIZON: {result['time_horizon'].replace('_', ' ').title()}")
        print(f"💰 POSITION SIZE: {result['position_size'].replace('_', ' ').title()}")
        print()
        print("🎯 PRICE TARGETS:")
        print(f"   • Entry Target: ${result['entry_target']:.2f}")
        print(f"   • Primary Exit: ${result['exit_targets']['primary']:.2f}")
        print(f"   • Secondary Exit: ${result['exit_targets']['secondary']:.2f}")
        print(f"   • Stop Loss: ${result['stop_loss']:.2f}")
        print(f"   • Risk/Reward: {result['risk_reward_ratio']:.1f}:1")
        print()
        print("📝 INVESTMENT THESIS:")
        thesis_lines = result['investment_thesis'].split('. ')
        for line in thesis_lines[:2]:  # Show first two sentences
            print(f"   {line.strip()}.")
        if len(thesis_lines) > 2:
            print(f"   [Analysis continues...]")
        print()
        print("⚠️ KEY RISKS:")
        for i, risk in enumerate(result['risk_factors'][:3], 1):  # Show top 3 risks
            print(f"   {i}. {risk}")
        
        # Generate and save markdown report
        markdown_report = analyst.generate_markdown_report(result)
        
        # Save report to file
        reports_dir = Path("reports")
        reports_dir.mkdir(exist_ok=True)
        
        report_filename = f"analysis_{result['ticker']}_{result['analysis_id'][:8]}.md"
        report_path = reports_dir / report_filename
        
        with open(report_path, 'w') as f:
            f.write(markdown_report)
        
        print()
        print(f"📄 Detailed markdown report saved to: {report_path}")
        
    else:
        print(f"❌ Analysis failed: {result.get('metadata', {}).get('error', 'Unknown error')}")


async def demo_position_reevaluation():
    """Demonstrate position reevaluation"""
    
    print("\n🔄 DEMO: Position Reevaluation")
    print("=" * 60)
    
    # Setup mock components
    config = MockConfig()
    llm = MockLLMProvider(config.ANTHROPIC_API_KEY)
    alpaca = MockAlpacaProvider(config)
    
    # Create analyst
    analyst = create_junior_analyst(llm, alpaca, config)
    
    # Mock current position
    current_position = {
        "ticker": "AAPL",
        "quantity": 100,
        "entry_price": 180.00,
        "current_price": 184.50,
        "unrealized_pnl": 450.00,
        "entry_date": "2024-01-10",
        "stop_loss": 175.00,
        "target_price": 190.00,
        "days_held": 5
    }
    
    # Task data for reevaluation
    task_data = {
        "task_type": AnalysisType.POSITION_REEVALUATION.value,
        "ticker": "AAPL",
        "current_position": current_position
    }
    
    print(f"📊 Reevaluating existing position in {task_data['ticker']}:")
    print(f"   • Position Size: {current_position['quantity']} shares")
    print(f"   • Entry Price: ${current_position['entry_price']:.2f}")
    print(f"   • Current Price: ${current_position['current_price']:.2f}")
    print(f"   • Unrealized P&L: ${current_position['unrealized_pnl']:.2f}")
    print(f"   • Days Held: {current_position['days_held']}")
    print()
    
    # Perform reevaluation
    result = await analyst.analyze_stock(task_data)
    
    if result.get("metadata", {}).get("status") == "success":
        print("✅ Reevaluation Complete! Here's the update:")
        print()
        print(f"🎯 RECOMMENDED ACTION: {result['action'].upper()}")
        print(f"📊 UPDATED CONFIDENCE: {result['updated_confidence']}/10")
        print(f"📈 CONVICTION CHANGE: {result['conviction_change'].replace('_', ' ').title()}")
        print()
        print("🎯 UPDATED TARGETS:")
        print(f"   • Primary Target: ${result['updated_targets']['primary']:.2f}")
        print(f"   • Secondary Target: ${result['updated_targets']['secondary']:.2f}")
        print(f"   • Updated Stop Loss: ${result['updated_stop_loss']:.2f}")
        print()
        print("📰 NEW DEVELOPMENTS:")
        developments = result['new_developments'][:150] + "..." if len(result['new_developments']) > 150 else result['new_developments']
        print(f"   {developments}")
        print()
        print("🔍 RATIONALE:")
        rationale = result['recommendation_rationale'][:150] + "..." if len(result['recommendation_rationale']) > 150 else result['recommendation_rationale']
        print(f"   {rationale}")
        
        # Generate and save markdown report
        markdown_report = analyst.generate_markdown_report(result)
        
        # Save report to file
        reports_dir = Path("reports")
        reports_dir.mkdir(exist_ok=True)
        
        report_filename = f"reevaluation_{result['ticker']}_{result['analysis_id'][:8]}.md"
        report_path = reports_dir / report_filename
        
        with open(report_path, 'w') as f:
            f.write(markdown_report)
        
        print()
        print(f"📄 Detailed markdown report saved to: {report_path}")
        
    else:
        print(f"❌ Reevaluation failed: {result.get('metadata', {}).get('error', 'Unknown error')}")


async def demo_risk_assessment():
    """Demonstrate risk assessment"""
    
    print("\n⚠️ DEMO: Risk Assessment")
    print("=" * 60)
    
    # Setup mock components
    config = MockConfig()
    llm = MockLLMProvider(config.ANTHROPIC_API_KEY)
    alpaca = MockAlpacaProvider(config)
    
    # Create analyst
    analyst = create_junior_analyst(llm, alpaca, config)
    
    # Mock position data for risk assessment
    position_data = {
        "ticker": "AAPL",
        "quantity": 200,
        "market_value": 36900.00,
        "portfolio_weight": 0.04,  # 4% of portfolio
        "beta": 1.2,
        "volatility": 0.25,
        "var_1day": 1850.00  # 1-day Value at Risk
    }
    
    # Task data for risk assessment
    task_data = {
        "task_type": AnalysisType.RISK_ASSESSMENT.value,
        "ticker": "AAPL",
        "position_data": position_data
    }
    
    print(f"📊 Assessing risk for position in {task_data['ticker']}:")
    print(f"   • Position Value: ${position_data['market_value']:,.2f}")
    print(f"   • Portfolio Weight: {position_data['portfolio_weight']*100:.1f}%")
    print(f"   • Beta: {position_data['beta']}")
    print(f"   • 1-Day VaR: ${position_data['var_1day']:,.2f}")
    print()
    
    # Perform risk assessment
    result = await analyst.analyze_stock(task_data)
    
    if result.get("metadata", {}).get("status") == "success":
        print("✅ Risk Assessment Complete!")
        print()
        print(f"🚨 RISK LEVEL: {result['risk_level'].upper()}")
        print(f"📊 RISK SCORE: {result['risk_score']}/10")
        print()
        print("⚠️ IDENTIFIED RISKS:")
        for i, risk in enumerate(result['risk_factors'][:3], 1):
            print(f"   {i}. {risk}")
        print()
        print("📉 DOWNSIDE SCENARIOS:")
        for i, scenario in enumerate(result['downside_scenarios'][:2], 1):
            print(f"   {i}. {scenario}")
        print()
        print("🛡️ RISK MITIGATION:")
        for i, strategy in enumerate(result['risk_mitigation'][:3], 1):
            print(f"   {i}. {strategy}")
        print()
        print("📊 VOLATILITY ASSESSMENT:")
        volatility_summary = result['volatility_assessment'][:100] + "..." if len(result['volatility_assessment']) > 100 else result['volatility_assessment']
        print(f"   {volatility_summary}")
        
        # Generate and save markdown report
        markdown_report = analyst.generate_markdown_report(result)
        
        # Save report to file
        reports_dir = Path("reports")
        reports_dir.mkdir(exist_ok=True)
        
        report_filename = f"risk_assessment_{result['ticker']}_{result['analysis_id'][:8]}.md"
        report_path = reports_dir / report_filename
        
        with open(report_path, 'w') as f:
            f.write(markdown_report)
        
        print()
        print(f"📄 Detailed markdown report saved to: {report_path}")
        
    else:
        print(f"❌ Risk assessment failed: {result.get('metadata', {}).get('error', 'Unknown error')}")


async def demo_performance_metrics():
    """Demonstrate performance tracking"""
    
    print("\n📊 DEMO: Performance Metrics")
    print("=" * 60)
    
    # Setup mock components
    config = MockConfig()
    llm = MockLLMProvider(config.ANTHROPIC_API_KEY)
    alpaca = MockAlpacaProvider(config)
    
    # Create analyst
    analyst = create_junior_analyst(llm, alpaca, config)
    
    # Run a few analyses to generate performance data
    print("Running sample analyses to generate performance metrics...")
    
    tickers = ["AAPL", "MSFT", "GOOGL"]
    for i, ticker in enumerate(tickers):
        task_data = {
            "task_type": AnalysisType.NEW_OPPORTUNITY.value,
            "ticker": ticker,
            "technical_signal": {"pattern": "breakout", "score": 7 + i}
        }
        
        result = await analyst.analyze_stock(task_data)
        status = result.get("metadata", {}).get("status", "unknown")
        print(f"   ✅ Analyzed {ticker} (Status: {status})")
    
    # Get performance summary
    performance = analyst.get_performance_summary()
    
    print("\n📈 AGENT PERFORMANCE SUMMARY:")
    print(f"   • Agent Name: {performance['agent_name']}")
    print(f"   • Agent ID: {performance['agent_id'][:8]}...")
    print(f"   • Total Analyses: {performance['total_analyses']}")
    print(f"   • Success Rate: {performance['success_rate']}")
    print(f"   • Avg Processing Time: {performance['average_processing_time']}")
    print(f"   • Cache Hit Rate: {performance['cache_hit_rate']}")
    if performance['last_activity']:
        print(f"   • Last Activity: {performance['last_activity']}")
    
    # Show API call counts
    print(f"\n🔌 API USAGE METRICS:")
    print(f"   • LLM Calls: {llm.call_count}")
    print(f"   • Market Data Calls: {alpaca.call_count}")


async def demo_error_handling():
    """Demonstrate error handling capabilities"""
    
    print("\n🚨 DEMO: Error Handling")
    print("=" * 60)
    
    # Setup mock components
    config = MockConfig()
    llm = MockLLMProvider(config.ANTHROPIC_API_KEY)
    alpaca = MockAlpacaProvider(config)
    
    # Create analyst
    analyst = create_junior_analyst(llm, alpaca, config)
    
    print("Testing various error scenarios...")
    
    # Test 1: Invalid task type
    print("\n🔍 Test 1: Invalid task type")
    task_data = {
        "task_type": "invalid_type",
        "ticker": "AAPL"
    }
    
    result = await analyst.analyze_stock(task_data)
    print(f"   Result: {result.get('metadata', {}).get('status')}")
    error_msg = result.get('metadata', {}).get('error', 'None')
    print(f"   Error: {error_msg[:50]}..." if len(error_msg) > 50 else f"   Error: {error_msg}")
    
    # Test 2: Missing required fields
    print("\n🔍 Test 2: Missing ticker field")
    task_data = {
        "task_type": AnalysisType.NEW_OPPORTUNITY.value
    }
    
    result = await analyst.analyze_stock(task_data)
    print(f"   Result: {result.get('metadata', {}).get('status')}")
    error_msg = result.get('metadata', {}).get('error', 'None')
    print(f"   Error: {error_msg[:50]}..." if len(error_msg) > 50 else f"   Error: {error_msg}")
    
    # Test 3: Empty ticker
    print("\n🔍 Test 3: Empty ticker")
    task_data = {
        "task_type": AnalysisType.NEW_OPPORTUNITY.value,
        "ticker": ""
    }
    
    result = await analyst.analyze_stock(task_data)
    print(f"   Result: {result.get('metadata', {}).get('status')}")
    error_msg = result.get('metadata', {}).get('error', 'None')
    print(f"   Error: {error_msg[:50]}..." if len(error_msg) > 50 else f"   Error: {error_msg}")
    
    # Test 4: Invalid data types
    print("\n🔍 Test 4: Invalid data types")
    task_data = {
        "task_type": AnalysisType.NEW_OPPORTUNITY.value,
        "ticker": 123  # Should be string
    }
    
    result = await analyst.analyze_stock(task_data)
    print(f"   Result: {result.get('metadata', {}).get('status')}")
    error_msg = result.get('metadata', {}).get('error', 'None')
    print(f"   Error: {error_msg[:50]}..." if len(error_msg) > 50 else f"   Error: {error_msg}")
    
    print("\n✅ Error handling tests completed!")


async def demo_caching_mechanism():
    """Demonstrate caching functionality"""
    
    print("\n💾 DEMO: Caching Mechanism")
    print("=" * 60)
    
    # Setup mock components
    config = MockConfig()
    llm = MockLLMProvider(config.ANTHROPIC_API_KEY)
    alpaca = MockAlpacaProvider(config)
    
    # Create analyst
    analyst = create_junior_analyst(llm, alpaca, config)
    
    task_data = {
        "task_type": AnalysisType.NEW_OPPORTUNITY.value,
        "ticker": "AAPL",
        "technical_signal": {"pattern": "breakout", "score": 8}
    }
    
    print("🔍 First analysis (will be cached):")
    start_time = datetime.now()
    result1 = await analyst.analyze_stock(task_data)
    time1 = (datetime.now() - start_time).total_seconds()
    print(f"   Time taken: {time1:.3f} seconds")
    print(f"   Analysis ID: {result1.get('analysis_id', 'N/A')[:8]}...")
    print(f"   Status: {result1.get('metadata', {}).get('status')}")
    
    print("\n🔍 Second analysis (should use cache):")
    start_time = datetime.now()
    result2 = await analyst.analyze_stock(task_data)
    time2 = (datetime.now() - start_time).total_seconds()
    print(f"   Time taken: {time2:.3f} seconds")
    print(f"   Analysis ID: {result2.get('analysis_id', 'N/A')[:8]}...")
    print(f"   Status: {result2.get('metadata', {}).get('status')}")
    
    # Check if results are identical (cached)
    if result1.get('analysis_id') == result2.get('analysis_id'):
        print("   ✅ Cache hit! Same analysis returned")
        print(f"   Speed improvement: {((time1 - time2) / time1 * 100):.1f}%")
    else:
        print("   ℹ️ New analysis performed (cache miss or disabled)")
    
    # Show cache statistics
    performance = analyst.get_performance_summary()
    print(f"\n📊 Cache Statistics:")
    print(f"   Cache Hit Rate: {performance['cache_hit_rate']}")


async def main():
    """Main demo function"""
    
    print("🤖 JUNIOR RESEARCH ANALYST - COMPLETE DEMO")
    print("=" * 80)
    print("This demo showcases the complete capabilities of the Junior Research Analyst Agent")
    print("optimized for macOS M2 Max with Claude AI integration and markdown reporting.")
    print("Note: This demo uses mock data for demonstration purposes.")
    print()
    
    try:
        # Run all demo scenarios
        await demo_new_opportunity_analysis()
        await demo_position_reevaluation()
        await demo_risk_assessment()
        await demo_performance_metrics()
        await demo_error_handling()
        await demo_caching_mechanism()
        
        print("\n" + "=" * 80)
        print("🎉 ALL DEMOS COMPLETED SUCCESSFULLY!")
        print("\nThe Junior Research Analyst Agent provides:")
        print("✅ Comprehensive stock analysis (technical + fundamental + AI)")
        print("✅ Position reevaluation with updated recommendations")
        print("✅ Detailed risk assessment and mitigation strategies")
        print("✅ Markdown report generation for easy sharing")
        print("✅ Performance tracking and metrics")
        print("✅ Robust error handling and fallback mechanisms")
        print("✅ Intelligent caching for efficiency")
        print("✅ Claude AI integration for sophisticated analysis")
        print("\n📄 All reports have been saved to the './reports/' directory")
        print("🚀 Ready for integration with your trading system!")
        
    except KeyboardInterrupt:
        print("\n\n⏹️ Demo interrupted by user")
    except Exception as e:
        print(f"\n❌ Demo failed with error: {str(e)}")
        import traceback
        traceback.print_exc()
        
        print(f"\n🔧 Troubleshooting Tips:")
        print("1. Ensure agents/junior_research_analyst.py exists")
        print("2. Check that virtual environment is activated")
        print("3. Verify all dependencies are installed")
        print("4. Run from the project root directory")


def setup_demo_environment():
    """Setup the demo environment"""
    
    print("🔧 Setting up demo environment...")
    
    # Create necessary directories
    os.makedirs("reports", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    
    # Setup basic logging
    import logging
    log_level = getattr(logging, "INFO", logging.INFO)
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('logs/junior_analyst_demo.log'),
            logging.StreamHandler()
        ]
    )
    
    print("✅ Demo environment ready!")


if __name__ == "__main__":
    # Setup and run demo
    setup_demo_environment()
    
    print("Starting Junior Research Analyst Demo...")
    print("Note: This demo uses mock data and providers for demonstration.")
    print("In production, replace mock providers with real Alpaca and Claude APIs.")
    print()
    
    # Run the complete demo
    asyncio.run(main())