"""
Environment and Configuration Validator
Comprehensive validation for trading system setup
"""

import sys
import os
import asyncio
from typing import List, Tuple, Any
from pathlib import Path

# Fix import path - add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Load environment variables first
try:
    from dotenv import load_dotenv
    
    # Try to find and load .env file
    env_file = project_root / '.env'
    if env_file.exists():
        load_dotenv(env_file)
        print(f"✅ Loaded .env from: {env_file}")
    else:
        print(f"⚠️ No .env file found at {env_file}")
        
except ImportError:
    print("⚠️ python-dotenv not installed")

# Now import config after environment is loaded
try:
    from config.settings import TradingConfig
except ImportError as e:
    print(f"❌ Could not import TradingConfig: {e}")
    print(f"Creating minimal config from environment variables...")
    
    # Fallback: Create a minimal config class
    class TradingConfig:
        def __init__(self):
            # API Configuration
            self.ALPACA_API_KEY = os.getenv('ALPACA_API_KEY', '')
            self.ALPACA_SECRET_KEY = os.getenv('ALPACA_SECRET_KEY', '')
            self.ANTHROPIC_API_KEY = os.getenv('ANTHROPIC_API_KEY', '')
            
            # Trading Parameters
            self.MAX_POSITIONS = int(os.getenv('MAX_POSITIONS', '10'))
            self.MAX_POSITION_SIZE = float(os.getenv('MAX_POSITION_SIZE', '0.05'))
            self.MAX_SECTOR_EXPOSURE = float(os.getenv('MAX_SECTOR_EXPOSURE', '0.25'))
            self.DAILY_LOSS_LIMIT = float(os.getenv('DAILY_LOSS_LIMIT', '0.02'))
            self.MIN_CASH_RESERVE = float(os.getenv('MIN_CASH_RESERVE', '0.10'))
            self.RISK_TOLERANCE = os.getenv('RISK_TOLERANCE', 'moderate')
            
            # Notification Settings
            self.NOTIFICATION_PROVIDER = os.getenv('NOTIFICATION_PROVIDER', 'disabled')
            self.ENABLE_NOTIFICATIONS = os.getenv('ENABLE_NOTIFICATIONS', 'false').lower() == 'true'
            self.BLUESKY_HANDLE = os.getenv('BLUESKY_HANDLE', '')
            self.BLUESKY_APP_PASSWORD = os.getenv('BLUESKY_APP_PASSWORD', '')
            self.SMTP_USERNAME = os.getenv('SMTP_USERNAME', '')
            self.SMTP_PASSWORD = os.getenv('SMTP_PASSWORD', '')
            self.WEBHOOK_URL = os.getenv('WEBHOOK_URL', '')
            
        def get_trading_params(self):
            return {
                'max_positions': self.MAX_POSITIONS,
                'max_position_size': self.MAX_POSITION_SIZE,
                'max_sector_exposure': self.MAX_SECTOR_EXPOSURE,
                'daily_loss_limit': self.DAILY_LOSS_LIMIT,
                'min_cash_reserve': self.MIN_CASH_RESERVE,
                'risk_tolerance': self.RISK_TOLERANCE
            }

class ConfigurationValidator:
    """Validates system configuration and dependencies"""
    
    def __init__(self):
        self.config = TradingConfig()
        
    def validate_environment(self) -> bool:
        """Run comprehensive environment validation"""
        print("🔍 Validating AI Trading System Environment...")
        print("=" * 50)
        
        validation_results: List[Tuple[str, bool, str]] = []
        
        # ===== PYTHON ENVIRONMENT =====
        python_version = sys.version_info
        python_valid = python_version >= (3, 9)
        validation_results.append((
            "Python Version", 
            python_valid, 
            f"v{python_version.major}.{python_version.minor}.{python_version.micro}"
        ))
        
        # ===== REQUIRED IMPORTS =====
        imports_to_test = [
            ("alpaca_trade_api", "Alpaca Trading API"),
            ("anthropic", "Claude/Anthropic AI"),
            ("pandas", "Data Processing"),
            ("sqlalchemy", "Database ORM"),
            ("fastapi", "Web Framework"),
            ("pydantic", "Data Validation"),
        ]
        
        for module_name, description in imports_to_test:
            try:
                __import__(module_name)
                validation_results.append((description, True, "✅ Available"))
            except ImportError:
                validation_results.append((description, False, "❌ Missing"))
        
        # ===== API CREDENTIALS =====
        credential_checks = [
            ("Alpaca API Key", bool(self.config.ALPACA_API_KEY), "Required for trading"),
            ("Alpaca Secret", bool(self.config.ALPACA_SECRET_KEY), "Required for trading"),
            ("Claude API Key", bool(self.config.ANTHROPIC_API_KEY), "Required for AI analysis"),
        ]
        
        validation_results.extend(credential_checks)
        
        # ===== NOTIFICATION SETTINGS =====
        notification_provider = getattr(self.config, 'NOTIFICATION_PROVIDER', 'disabled')
        enable_notifications = getattr(self.config, 'ENABLE_NOTIFICATIONS', False)
        
        notification_checks = [
            ("Notification Config", True, f"Provider: {notification_provider}"),
        ]
        
        if enable_notifications and notification_provider != "disabled":
            if notification_provider == "bluesky":
                bluesky_handle = getattr(self.config, 'BLUESKY_HANDLE', '')
                bluesky_password = getattr(self.config, 'BLUESKY_APP_PASSWORD', '')
                notification_checks.append((
                    "BlueSky Credentials", 
                    bool(bluesky_handle and bluesky_password),
                    "Required for BlueSky notifications"
                ))
            elif notification_provider == "email":
                smtp_username = getattr(self.config, 'SMTP_USERNAME', '')
                smtp_password = getattr(self.config, 'SMTP_PASSWORD', '')
                notification_checks.append((
                    "Email Credentials",
                    bool(smtp_username and smtp_password),
                    "Required for email notifications"
                ))
            elif notification_provider == "webhook":
                webhook_url = getattr(self.config, 'WEBHOOK_URL', '')
                notification_checks.append((
                    "Webhook URL",
                    bool(webhook_url),
                    "Required for webhook notifications"
                ))
        
        validation_results.extend(notification_checks)
        
        # ===== RISK PARAMETERS =====
        param_validations = [
            ("Max Positions", self.config.MAX_POSITIONS > 0, f"Current: {self.config.MAX_POSITIONS}"),
            ("Position Size", 0 < self.config.MAX_POSITION_SIZE <= 1.0, f"Current: {self.config.MAX_POSITION_SIZE*100:.1f}%"),
            ("Sector Exposure", 0 < self.config.MAX_SECTOR_EXPOSURE <= 1.0, f"Current: {self.config.MAX_SECTOR_EXPOSURE*100:.1f}%"),
            ("Daily Loss Limit", 0 < self.config.DAILY_LOSS_LIMIT <= 1.0, f"Current: {self.config.DAILY_LOSS_LIMIT*100:.1f}%"),
            ("Cash Reserve", 0 <= self.config.MIN_CASH_RESERVE <= 1.0, f"Current: {self.config.MIN_CASH_RESERVE*100:.1f}%"),
        ]
        
        validation_results.extend(param_validations)
        
        # ===== DISPLAY RESULTS =====
        all_passed = True
        
        print("\n📋 Configuration Status:")
        print("-" * 50)
        
        for check_name, passed, detail in validation_results:
            status = "✅ PASS" if passed else "❌ FAIL"
            print(f"{check_name:20} {status:10} {detail}")
            if not passed:
                all_passed = False
        
        # ===== TRADING PARAMETERS SUMMARY =====
        print(f"\n⚙️ Trading Parameters:")
        print("-" * 30)
        params = self.config.get_trading_params()
        print(f"Max Positions:     {params['max_positions']}")
        print(f"Position Size:     {params['max_position_size']*100:.1f}%")
        print(f"Sector Exposure:   {params['max_sector_exposure']*100:.1f}%")
        print(f"Daily Loss Limit:  {params['daily_loss_limit']*100:.1f}%")
        print(f"Cash Reserve:      {params['min_cash_reserve']*100:.1f}%")
        print(f"Risk Tolerance:    {params['risk_tolerance']}")
        
        # ===== NOTIFICATION STATUS =====
        print(f"\n📱 Notification Settings:")
        print("-" * 30)
        notification_provider = getattr(self.config, 'NOTIFICATION_PROVIDER', 'disabled')
        enable_notifications = getattr(self.config, 'ENABLE_NOTIFICATIONS', False)
        
        print(f"Provider:          {notification_provider}")
        print(f"Enabled:           {enable_notifications}")
        
        if notification_provider == "bluesky":
            bluesky_handle = getattr(self.config, 'BLUESKY_HANDLE', '')
            if bluesky_handle:
                print(f"BlueSky Handle:    {bluesky_handle}")
        elif notification_provider == "email":
            smtp_username = getattr(self.config, 'SMTP_USERNAME', '')
            if smtp_username:
                print(f"Email:             {smtp_username}")
        elif notification_provider == "webhook":
            webhook_url = getattr(self.config, 'WEBHOOK_URL', '')
            if webhook_url:
                print(f"Webhook:           {webhook_url}")
        else:
            print(f"Status:            Disabled - Ready for future development")
        
        # ===== ENVIRONMENT FILE CHECK =====
        print(f"\n📄 Environment File Status:")
        print("-" * 30)
        env_file = project_root / '.env'
        if env_file.exists():
            print(f"✅ .env file found: {env_file}")
            # Check if key variables are in the file
            try:
                with open(env_file, 'r') as f:
                    content = f.read()
                    if 'ANTHROPIC_API_KEY' in content:
                        print("✅ ANTHROPIC_API_KEY present in .env file")
                    else:
                        print("❌ ANTHROPIC_API_KEY not found in .env file")
            except Exception as e:
                print(f"⚠️ Could not read .env file: {e}")
        else:
            print(f"❌ No .env file found at {env_file}")
            print(f"💡 Create .env file with your API keys")
        
        # ===== API KEY VERIFICATION =====
        print(f"\n🔑 API Key Status:")
        print("-" * 30)
        
        # Direct environment check
        anthropic_key_env = os.getenv('ANTHROPIC_API_KEY')
        anthropic_key_config = self.config.ANTHROPIC_API_KEY
        
        if anthropic_key_env:
            print(f"✅ ANTHROPIC_API_KEY in environment (length: {len(anthropic_key_env)})")
        else:
            print(f"❌ ANTHROPIC_API_KEY not in environment")
            
        if anthropic_key_config:
            print(f"✅ ANTHROPIC_API_KEY in config (length: {len(anthropic_key_config)})")
        else:
            print(f"❌ ANTHROPIC_API_KEY not in config")
        
        # ===== FINAL STATUS =====
        if all_passed:
            print(f"\n🎉 Environment validation successful!")
            print(f"✅ Ready to proceed to Core Infrastructure setup")
            print(f"✅ Ready to run tests:")
            print(f"   pytest tests/unit/llm_provider/test_claude_llm_provider.py -v")
            return True
        else:
            print(f"\n❌ Environment validation failed!")
            print(f"🔧 Please update your .env file with missing values")
            print(f"📖 Refer to .env.example for required parameters")
            return False

def validate_environment() -> bool:
    """Main validation function"""
    validator = ConfigurationValidator()
    return validator.validate_environment()

if __name__ == "__main__":
    print(f"🏗️ Project root: {project_root}")
    print(f"🐍 Python path: {sys.path[0]}")
    print()
    
    success = validate_environment()
    sys.exit(0 if success else 1)