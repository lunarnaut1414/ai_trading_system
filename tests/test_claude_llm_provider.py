#!/usr/bin/env python3
"""
Claude Model Tester and Auto-Configuration Tool
Discovers available models and updates configuration automatically
"""

import os
import sys
import json
import asyncio
import argparse
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from pathlib import Path
from anthropic import Anthropic, APIError

class ClaudeModelTester:
    """Test Claude models and auto-configure the best available ones"""
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize the model tester"""
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError("ANTHROPIC_API_KEY not found in environment")
        
        self.client = Anthropic(api_key=self.api_key)
        self.results = {}
        
    def test_model(self, model_id: str, verbose: bool = True) -> Dict:
        """
        Test a single model
        
        Returns:
            Dictionary with test results
        """
        try:
            if verbose:
                print(f"Testing {model_id}...", end=" ")
            
            # Test with minimal prompt
            start_time = datetime.now()
            response = self.client.messages.create(
                model=model_id,
                messages=[{"role": "user", "content": "Reply with just 'ok'"}],
                max_tokens=10,
                temperature=0
            )
            response_time = (datetime.now() - start_time).total_seconds()
            
            # Model works!
            result = {
                "status": "available",
                "response_time": response_time,
                "content": response.content[0].text,
                "tested_at": datetime.now().isoformat()
            }
            
            if verbose:
                print(f"✅ Available ({response_time:.2f}s)")
            
            return result
            
        except APIError as e:
            error_msg = str(e)
            
            # Categorize error
            if "model_not_found" in error_msg.lower() or "does not exist" in error_msg.lower():
                status = "not_found"
                symbol = "❌"
            elif "authentication" in error_msg.lower():
                status = "auth_error"
                symbol = "🔐"
            elif "rate" in error_msg.lower():
                status = "rate_limited"
                symbol = "⏱️"
            elif "permission" in error_msg.lower():
                status = "no_permission"
                symbol = "🚫"
            else:
                status = "error"
                symbol = "❌"
            
            if verbose:
                print(f"{symbol} {status}")
            
            return {
                "status": status,
                "error": error_msg[:200],
                "tested_at": datetime.now().isoformat()
            }
        
        except Exception as e:
            if verbose:
                print(f"❌ Error: {str(e)[:50]}")
            
            return {
                "status": "error",
                "error": str(e)[:200],
                "tested_at": datetime.now().isoformat()
            }
    
    def test_model_capabilities(self, model_id: str) -> Dict:
        """
        Test model capabilities in detail
        
        Returns:
            Dictionary with capability test results
        """
        print(f"\n🔬 Testing capabilities for {model_id}...")
        capabilities = {}
        
        # Test 1: Basic reasoning
        try:
            response = self.client.messages.create(
                model=model_id,
                messages=[{"role": "user", "content": "What is 15 * 17? Reply with just the number."}],
                max_tokens=10,
                temperature=0
            )
            capabilities["basic_math"] = "255" in response.content[0].text
            print(f"   Basic Math: {'✅' if capabilities['basic_math'] else '❌'}")
        except:
            capabilities["basic_math"] = False
            print(f"   Basic Math: ❌")
        
        # Test 2: JSON generation
        try:
            response = self.client.messages.create(
                model=model_id,
                messages=[{"role": "user", "content": 'Generate JSON: {"status": "ok", "value": 42}'}],
                max_tokens=50,
                temperature=0
            )
            json.loads(response.content[0].text)
            capabilities["json_generation"] = True
            print(f"   JSON Generation: ✅")
        except:
            capabilities["json_generation"] = False
            print(f"   JSON Generation: ❌")
        
        # Test 3: Context length (test with longer prompt)
        try:
            long_prompt = "Summarize this: " + ("test " * 1000)
            response = self.client.messages.create(
                model=model_id,
                messages=[{"role": "user", "content": long_prompt}],
                max_tokens=50,
                temperature=0
            )
            capabilities["handles_long_context"] = True
            print(f"   Long Context: ✅")
        except:
            capabilities["handles_long_context"] = False
            print(f"   Long Context: ❌")
        
        # Test 4: Response speed
        times = []
        for _ in range(3):
            start = datetime.now()
            try:
                self.client.messages.create(
                    model=model_id,
                    messages=[{"role": "user", "content": "Hi"}],
                    max_tokens=10,
                    temperature=0
                )
                times.append((datetime.now() - start).total_seconds())
            except:
                pass
        
        if times:
            avg_time = sum(times) / len(times)
            capabilities["avg_response_time"] = avg_time
            print(f"   Avg Response Time: {avg_time:.2f}s")
        
        return capabilities
    
    def discover_all_models(self) -> Dict[str, Dict]:
        """
        Test all known Claude models
        
        Returns:
            Dictionary of all test results
        """
        # Comprehensive list of Claude models to test
        all_models = [
            # Claude 4 series (Latest)
            "claude-opus-4-1-20250805",
            "claude-opus-4-20250514",
            "claude-sonnet-4-20250514",
            
            # Claude 3.7 series
            "claude-3-7-sonnet-20250219",
            
            # Claude 3.5 series
            "claude-3-5-haiku-20241022",
            "claude-3-5-sonnet-20241022",
            "claude-3-5-sonnet-20240620",
            
            # Claude 3 series (Legacy)
            "claude-3-opus-20240229",
            "claude-3-sonnet-20240229",
            "claude-3-haiku-20240307",
            
            # Claude 2 series (Very old, likely deprecated)
            "claude-2.1",
            "claude-2.0",
            
            # Instant models (if they exist)
            "claude-instant-1.2",
        ]
        
        print("\n" + "="*60)
        print("CLAUDE MODEL DISCOVERY")
        print("="*60)
        print(f"Testing {len(all_models)} models...\n")
        
        results = {}
        for model_id in all_models:
            results[model_id] = self.test_model(model_id)
        
        # Categorize results
        available = {k: v for k, v in results.items() if v["status"] == "available"}
        not_found = {k: v for k, v in results.items() if v["status"] == "not_found"}
        errors = {k: v for k, v in results.items() if v["status"] not in ["available", "not_found"]}
        
        print("\n" + "="*60)
        print("DISCOVERY RESULTS")
        print("="*60)
        
        if available:
            print(f"\n✅ Available Models ({len(available)}):")
            for model_id, info in available.items():
                print(f"   {model_id} - Response time: {info['response_time']:.2f}s")
        
        if errors:
            print(f"\n⚠️  Models with Errors ({len(errors)}):")
            for model_id, info in errors.items():
                print(f"   {model_id} - {info['status']}")
        
        if not_found:
            print(f"\n❌ Not Found ({len(not_found)}):")
            for model_id in not_found:
                print(f"   {model_id}")
        
        self.results = results
        return results
    
    def recommend_configuration(self, results: Optional[Dict] = None) -> Dict[str, str]:
        """
        Recommend the best model for each tier based on test results
        
        Returns:
            Dictionary with recommended models for each tier
        """
        results = results or self.results
        available = {k: v for k, v in results.items() if v.get("status") == "available"}
        
        if not available:
            print("\n⚠️  No models available!")
            return {}
        
        # Categorize available models by tier
        haiku_models = []
        sonnet_models = []
        opus_models = []
        
        for model_id in available:
            if "haiku" in model_id.lower():
                haiku_models.append(model_id)
            elif "sonnet" in model_id.lower():
                sonnet_models.append(model_id)
            elif "opus" in model_id.lower():
                opus_models.append(model_id)
        
        # Sort by version (newer versions have higher numbers)
        def sort_key(model_id):
            # Extract date from model ID (e.g., 20241022)
            import re
            match = re.search(r'(\d{8})', model_id)
            return match.group(1) if match else "0"
        
        recommendations = {}
        
        if haiku_models:
            haiku_models.sort(key=sort_key, reverse=True)
            recommendations["haiku"] = haiku_models[0]
        
        if sonnet_models:
            sonnet_models.sort(key=sort_key, reverse=True)
            recommendations["sonnet"] = sonnet_models[0]
        
        if opus_models:
            opus_models.sort(key=sort_key, reverse=True)
            recommendations["opus"] = opus_models[0]
        
        print("\n" + "="*60)
        print("RECOMMENDED CONFIGURATION")
        print("="*60)
        
        for tier, model_id in recommendations.items():
            response_time = available[model_id].get("response_time", 0)
            print(f"\n{tier.upper()}:")
            print(f"   Model: {model_id}")
            print(f"   Response Time: {response_time:.2f}s")
        
        return recommendations
    
    def save_results(self, filepath: str = "claude_models_test_results.json"):
        """Save test results to file"""
        output = {
            "tested_at": datetime.now().isoformat(),
            "api_key_prefix": self.api_key[:10] + "..." if self.api_key else "None",
            "results": self.results,
            "recommendations": self.recommend_configuration()
        }
        
        with open(filepath, 'w') as f:
            json.dump(output, f, indent=2)
        
        print(f"\n💾 Results saved to {filepath}")
    
    def generate_env_config(self) -> str:
        """Generate environment variable configuration"""
        recommendations = self.recommend_configuration()
        
        if not recommendations:
            return "# No models available"
        
        config = [
            "# Claude Model Configuration",
            f"# Generated: {datetime.now().isoformat()}",
            ""
        ]
        
        if "haiku" in recommendations:
            config.append(f"CLAUDE_MODEL_HAIKU={recommendations['haiku']}")
        if "sonnet" in recommendations:
            config.append(f"CLAUDE_MODEL_SONNET={recommendations['sonnet']}")
        if "opus" in recommendations:
            config.append(f"CLAUDE_MODEL_OPUS={recommendations['opus']}")
        
        config.extend([
            "",
            "# Agent-specific overrides (optional)",
            f"# CLAUDE_MODEL_JUNIOR_ANALYST={recommendations.get('sonnet', '')}",
            f"# CLAUDE_MODEL_SENIOR_ANALYST={recommendations.get('opus', '')}",
            f"# CLAUDE_MODEL_PORTFOLIO_MANAGER={recommendations.get('opus', '')}",
            f"# CLAUDE_MODEL_TRADE_EXECUTION={recommendations.get('haiku', '')}",
        ])
        
        return "\n".join(config)


def main():
    parser = argparse.ArgumentParser(
        description="Test Claude models and generate optimal configuration"
    )
    parser.add_argument(
        "--detailed",
        action="store_true",
        help="Run detailed capability tests on available models"
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="Save results to JSON file"
    )
    parser.add_argument(
        "--env",
        action="store_true",
        help="Generate environment variable configuration"
    )
    parser.add_argument(
        "--test-model",
        type=str,
        help="Test a specific model ID"
    )
    
    args = parser.parse_args()
    
    try:
        tester = ClaudeModelTester()
        
        if args.test_model:
            # Test specific model
            print(f"Testing {args.test_model}...")
            result = tester.test_model(args.test_model)
            print(f"Result: {json.dumps(result, indent=2)}")
            
            if result["status"] == "available" and args.detailed:
                capabilities = tester.test_model_capabilities(args.test_model)
                print(f"\nCapabilities: {json.dumps(capabilities, indent=2)}")
        else:
            # Discover all models
            results = tester.discover_all_models()
            recommendations = tester.recommend_configuration(results)
            
            # Run detailed tests if requested
            if args.detailed and recommendations:
                print("\n" + "="*60)
                print("DETAILED CAPABILITY TESTING")
                print("="*60)
                for tier, model_id in recommendations.items():
                    capabilities = tester.test_model_capabilities(model_id)
                    results[model_id]["capabilities"] = capabilities
            
            # Save results if requested
            if args.save:
                tester.save_results()
            
            # Generate env config if requested
            if args.env:
                print("\n" + "="*60)
                print("ENVIRONMENT CONFIGURATION")
                print("="*60)
                print(tester.generate_env_config())
                
                # Optionally save to file
                env_file = Path(".env.claude")
                with open(env_file, 'w') as f:
                    f.write(tester.generate_env_config())
                print(f"\n💾 Environment config saved to {env_file}")
    
    except Exception as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()