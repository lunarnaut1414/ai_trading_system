#!/usr/bin/env python3
"""
AI Trading System - Production Launch Script

This is the main entry point for running the complete AI trading system.
It initializes all components and starts the automated workflow.

Usage:
    python run_trading_system.py                 # Show help
    python run_trading_system.py --start         # Start full system
    python run_trading_system.py --test          # Run system tests
    python run_trading_system.py --status        # Check system status
    python run_trading_system.py --workflow      # Run manual workflow
"""

import asyncio
import argparse
import sys
import os
import logging
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from orchestration.orchestration_controller import OrchestrationController
from orchestration.workflow_engine import WorkflowEngine
from orchestration.daily_workflow import DailyTradingWorkflow
from config.settings import TradingConfig

class TradingSystemLauncher:
    """Production launcher for the AI trading system"""
    
    def __init__(self):
        self.config = TradingConfig()
        self.controller = None
    
    async def start_system(self):
        """Start the complete trading system"""
        
        print("🚀 AI TRADING SYSTEM")
        print("=" * 50)
        print(f"Environment: {'PAPER TRADING' if self.config.ALPACA_PAPER else 'LIVE TRADING'}")
        print(f"Log Level: {self.config.LOG_LEVEL}")
        print(f"Market Hours Only: {getattr(self.config, 'MARKET_HOURS_ONLY', True)}")
        print("=" * 50)
        
        # Create orchestration controller
        self.controller = OrchestrationController(self.config)
        
        # Run the system
        await self.controller.run_system()
    
    async def run_tests(self):
        """Run system tests"""
        
        print("🧪 RUNNING SYSTEM TESTS")
        print("=" * 50)
        
        try:
            # Basic configuration test
            print("Testing configuration...")
            config = TradingConfig()
            print("✅ Configuration loaded successfully")
            
            # Test component imports
            print("\nTesting agent imports...")
            from agents.junior_research_analyst import JuniorResearchAnalyst
            from agents.senior_research_analyst import SeniorResearchAnalyst
            from agents.portfolio_manager import PortfolioManagerAgent
            from agents.trade_execution_agent import TradeExecutionAgent
            from agents.analytics_reporting_agent import AnalyticsReportingAgent
            from agents.economist_agent import EconomistAgent
            print("✅ All agent imports successful")
            
            print("\nTesting orchestration imports...")
            from orchestration.workflow_engine import WorkflowEngine
            from orchestration.daily_workflow import DailyTradingWorkflow
            from orchestration.orchestration_controller import OrchestrationController
            print("✅ All orchestration imports successful")
            
            print("\nTesting data provider imports...")
            from data.alpaca_provider import AlpacaDataProvider
            print("✅ Data provider imports successful")
            
            print("\nTesting LLM provider imports...")
            from llm_providers.claude_llm_provider import ClaudeLLMProvider
            print("✅ LLM provider imports successful")
            
            # Test basic workflow creation
            print("\nTesting workflow engine creation...")
            test_agents = {'test_agent': type('Agent', (), {
                'analyze': lambda self, **kwargs: {'result': 'test'},
                'initialize': lambda self: None,
                'health_check': lambda self: {'healthy': True}
            })()}
            engine = WorkflowEngine(test_agents, config)
            print("✅ Workflow engine creation successful")
            
            # Test workflow manager creation
            print("\nTesting daily workflow creation...")
            workflow = DailyTradingWorkflow(engine, config)
            print("✅ Daily workflow creation successful")
            
            # Test controller creation
            print("\nTesting orchestration controller creation...")
            controller = OrchestrationController(config)
            print("✅ Orchestration controller creation successful")
            
            print("\n" + "=" * 50)
            print("✅ ALL SYSTEM TESTS PASSED")
            print("=" * 50)
            
        except Exception as e:
            print(f"\n❌ System tests failed: {str(e)}")
            import traceback
            traceback.print_exc()
            sys.exit(1)
    
    async def check_status(self):
        """Check system status"""
        
        print("📊 SYSTEM STATUS CHECK")
        print("=" * 50)
        
        try:
            # Create controller for status check
            self.controller = OrchestrationController(self.config)
            
            # Get system status
            status = self.controller.get_system_status()
            
            print(f"System Running: {status['system_running']}")
            print(f"Total Agents: {status['agents']['total']}")
            
            if status['agents']['names']:
                print(f"Agent Names: {', '.join(status['agents']['names'])}")
            
            if status.get('statistics'):
                stats = status['statistics']
                print("\nStatistics:")
                print(f"  Workflows Executed: {stats.get('workflows_executed', 0)}")
                print(f"  Tasks Completed: {stats.get('total_tasks_completed', 0)}")
                print(f"  System Alerts: {stats.get('system_alerts_sent', 0)}")
                
                if stats.get('last_error'):
                    print(f"  Last Error: {stats['last_error']}")
            
            if status.get('workflow_engine'):
                we = status['workflow_engine']
                print("\nWorkflow Engine:")
                print(f"  Active Executions: {we.get('active_executions', 0)}")
                print(f"  Completed Executions: {we.get('completed_executions', 0)}")
            
            print("\n✅ Status check completed")
            
        except Exception as e:
            print(f"❌ Status check failed: {str(e)}")
    
    async def run_manual_workflow(self):
        """Run a manual workflow execution"""
        
        print("⚙️ MANUAL WORKFLOW EXECUTION")
        print("=" * 50)
        
        try:
            # Create and start controller
            self.controller = OrchestrationController(self.config)
            
            print("Initializing system components...")
            await self.controller.startup_system()
            
            print("\nRunning daily workflow...")
            # Run daily workflow manually
            result = await self.controller.daily_workflow.run_daily_workflow()
            
            print("\n" + "=" * 50)
            print("WORKFLOW RESULTS")
            print("=" * 50)
            print(f"Status: {result.get('status')}")
            
            if result.get('task_summary'):
                summary = result['task_summary']
                print(f"\nTask Summary:")
                print(f"  Total Tasks: {summary.get('total_tasks')}")
                print(f"  Completed: {summary.get('completed_tasks')}")
                print(f"  Failed: {summary.get('failed_tasks')}")
                print(f"  Success Rate: {summary.get('success_rate', 0):.2%}")
            
            if result.get('errors'):
                print(f"\nErrors:")
                for error in result['errors']:
                    print(f"  - {error}")
            
            # Shutdown
            print("\nShutting down system...")
            await self.controller.shutdown_system("Manual workflow completed")
            
            print("\n✅ Manual workflow completed")
            
        except Exception as e:
            print(f"\n❌ Manual workflow failed: {str(e)}")
            if self.controller:
                await self.controller.shutdown_system(f"Error: {str(e)}")

async def main():
    """Main entry point"""
    
    parser = argparse.ArgumentParser(
        description='AI Trading System - Orchestration and Execution',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_trading_system.py --start      # Start the trading system
  python run_trading_system.py --test       # Run system tests
  python run_trading_system.py --status     # Check system status
  python run_trading_system.py --workflow   # Run manual workflow
        """
    )
    
    parser.add_argument('--start', action='store_true', 
                       help='Start the complete trading system')
    parser.add_argument('--test', action='store_true', 
                       help='Run system tests')
    parser.add_argument('--status', action='store_true', 
                       help='Check system status')
    parser.add_argument('--workflow', action='store_true', 
                       help='Run manual workflow')
    parser.add_argument('--log-level', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'], 
                       default='INFO', help='Set logging level')
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(f'logs/orchestration_{datetime.now().strftime("%Y%m%d")}.log')
        ]
    )
    
    # Create logs directory if it doesn't exist
    os.makedirs('logs', exist_ok=True)
    
    launcher = TradingSystemLauncher()
    
    try:
        if args.test:
            await launcher.run_tests()
        elif args.status:
            await launcher.check_status()
        elif args.workflow:
            await launcher.run_manual_workflow()
        elif args.start:
            await launcher.start_system()
        else:
            # Show help if no arguments provided
            parser.print_help()
    
    except KeyboardInterrupt:
        print("\n🛑 System interrupted by user")
    except Exception as e:
        print(f"\n❌ System failed: {str(e)}")
        logging.error(f"System failed: {str(e)}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())