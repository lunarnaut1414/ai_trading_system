# agents/base_agent.py
"""
Example: Base Agent using centralized Claude configuration
Shows how all your agents should use the new model configuration
"""

import os
import logging
from typing import Dict, Optional, Any
from anthropic import Anthropic
from config.models import ClaudeModels, ModelTier

class BaseAgent:
    """
    Base class for all AI agents using Claude
    Demonstrates proper usage of centralized model configuration
    """
    
    def __init__(self, agent_name: str, custom_tier: Optional[str] = None):
        """
        Initialize agent with appropriate Claude model
        
        Args:
            agent_name: Name of the agent (e.g., 'junior_analyst')
            custom_tier: Optional override ('haiku', 'sonnet', 'opus')
        """
        self.agent_name = agent_name
        self.logger = logging.getLogger(agent_name)
        
        # Get model configuration from centralized config
        if custom_tier:
            # Allow manual override
            self.model_config = ClaudeModels.get_model(custom_tier)
        else:
            # Use agent-specific assignment
            self.model_config = ClaudeModels.get_agent_model(agent_name)
        
        # Initialize Claude client
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY not set in environment")
        
        self.client = Anthropic(api_key=api_key)
        
        # Log configuration
        self.logger.info(
            f"Initialized {agent_name} with model: {self.model_config.name} "
            f"({self.model_config.model_id})"
        )
        
        # Track usage for cost monitoring
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_cost = 0.0
    
    def analyze(self, prompt: str, system_prompt: Optional[str] = None) -> Dict[str, Any]:
        """
        Send analysis request to Claude
        
        Args:
            prompt: User prompt
            system_prompt: Optional system prompt
            
        Returns:
            Analysis results with metadata
        """
        try:
            # Build messages
            messages = [{"role": "user", "content": prompt}]
            
            # Make API call using model from centralized config
            response = self.client.messages.create(
                model=self.model_config.model_id,  # Use centralized model ID
                messages=messages,
                system=system_prompt if system_prompt else self._get_default_system_prompt(),
                max_tokens=self.model_config.max_output,
                temperature=0.7
            )
            
            # Track usage
            input_tokens = response.usage.input_tokens
            output_tokens = response.usage.output_tokens
            
            self.total_input_tokens += input_tokens
            self.total_output_tokens += output_tokens
            
            # Calculate cost using centralized pricing
            cost = ClaudeModels.estimate_cost(
                self.model_config.tier.value,
                input_tokens,
                output_tokens
            )
            self.total_cost += cost
            
            return {
                "success": True,
                "content": response.content[0].text,
                "model": self.model_config.model_id,
                "tokens": {
                    "input": input_tokens,
                    "output": output_tokens,
                    "total": input_tokens + output_tokens
                },
                "cost": cost,
                "agent": self.agent_name
            }
            
        except Exception as e:
            self.logger.error(f"Analysis failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "agent": self.agent_name
            }
    
    def _get_default_system_prompt(self) -> str:
        """Get default system prompt based on agent type"""
        prompts = {
            "junior_analyst": "You are a junior financial analyst focused on individual stock analysis.",
            "senior_analyst": "You are a senior analyst synthesizing multiple research inputs.",
            "portfolio_manager": "You are a portfolio manager making strategic allocation decisions.",
            "trade_execution": "You are a trade execution specialist optimizing order timing.",
            "cfo_reporting": "You are a CFO creating executive reports and insights."
        }
        return prompts.get(self.agent_name, "You are a financial analysis assistant.")
    
    def get_usage_summary(self) -> Dict[str, Any]:
        """Get usage summary for this agent"""
        return {
            "agent": self.agent_name,
            "model": self.model_config.name,
            "model_id": self.model_config.model_id,
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_cost": round(self.total_cost, 4),
            "model_tier": self.model_config.tier.value,
            "cost_per_1k_input": self.model_config.cost_per_1k_input,
            "cost_per_1k_output": self.model_config.cost_per_1k_output
        }


# ===== SPECIFIC AGENT IMPLEMENTATIONS =====

class JuniorAnalyst(BaseAgent):
    """Junior Analyst - uses Sonnet by default"""
    
    def __init__(self):
        super().__init__("junior_analyst")
    
    def analyze_stock(self, symbol: str, data: Dict) -> Dict[str, Any]:
        """Analyze individual stock"""
        prompt = f"""
        Analyze {symbol} with the following data:
        Price: ${data.get('price')}
        Volume: {data.get('volume')}
        RSI: {data.get('rsi')}
        
        Provide buy/sell/hold recommendation with confidence score.
        """
        return self.analyze(prompt)


class SeniorAnalyst(BaseAgent):
    """Senior Analyst - uses Opus for complex synthesis"""
    
    def __init__(self):
        super().__init__("senior_analyst")
    
    def synthesize_research(self, reports: list) -> Dict[str, Any]:
        """Synthesize multiple research reports"""
        prompt = f"""
        Synthesize the following {len(reports)} research reports into 
        strategic recommendations. Identify top 3 opportunities.
        
        Reports: {reports}
        """
        return self.analyze(prompt)


class PortfolioManager(BaseAgent):
    """Portfolio Manager - uses Opus for critical decisions"""
    
    def __init__(self):
        super().__init__("portfolio_manager")
    
    def make_allocation_decision(self, opportunities: list, risk_metrics: Dict) -> Dict[str, Any]:
        """Make portfolio allocation decisions"""
        prompt = f"""
        Given these opportunities: {opportunities}
        And risk metrics: {risk_metrics}
        
        Determine optimal portfolio allocation maintaining:
        - Max 5% per position
        - Max 25% per sector
        - Target Sharpe ratio > 1.5
        """
        return self.analyze(prompt)


class TradeExecutionAgent(BaseAgent):
    """Trade Execution - uses Haiku for speed"""
    
    def __init__(self):
        super().__init__("trade_execution")
    
    def optimize_execution(self, order: Dict) -> Dict[str, Any]:
        """Optimize trade execution timing"""
        prompt = f"""
        Optimize execution for order: {order}
        Determine: immediate, TWAP, or wait.
        Response in JSON format.
        """
        return self.analyze(prompt)


# ===== USAGE EXAMPLE =====
if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Validate configuration
    print("Validating Claude configuration...")
    validation = ClaudeModels.validate_configuration()
    
    if not validation["valid"]:
        print("Configuration errors:", validation["errors"])
        exit(1)
    
    # Show current configuration
    ClaudeModels.print_configuration()
    
    # Example: Create agents
    print("\n" + "="*60)
    print("CREATING AGENTS")
    print("="*60)
    
    try:
        # Each agent automatically uses the right model
        junior = JuniorAnalyst()
        print(f"✅ Junior Analyst using: {junior.model_config.name}")
        
        senior = SeniorAnalyst()
        print(f"✅ Senior Analyst using: {senior.model_config.name}")
        
        pm = PortfolioManager()
        print(f"✅ Portfolio Manager using: {pm.model_config.name}")
        
        trader = TradeExecutionAgent()
        print(f"✅ Trade Execution using: {trader.model_config.name}")
        
        # Example: Override model for specific use case
        fast_analyst = BaseAgent("custom_analyst", custom_tier="haiku")
        print(f"✅ Custom Analyst using: {fast_analyst.model_config.name}")
        
    except Exception as e:
        print(f"❌ Error creating agents: {e}")
        print("Make sure ANTHROPIC_API_KEY is set in your environment")