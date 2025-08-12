"""
Environment and Configuration Validator
Comprehensive validation for trading system setup
"""

import sys
import os
import asyncio
from typing import List, Tuple, Any
from config.settings import TradingConfig

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
        # ===== API CREDENTIALS =====
        credential_checks = [
            ("Alpaca API Key", bool(self.config.ALPACA_API_KEY), "Required for trading"),
            ("Alpaca Secret", bool(self.config.ALPACA_SECRET_KEY), "Required for trading"),
            ("Claude API Key", bool(self.config.ANTHROPIC_API_KEY), "Required for AI analysis"),
        ]
        
        validation_results.extend(credential_checks)
        
        # ===== NOTIFICATION SETTINGS =====
        notification_checks = [
            ("Notification Config", True, f"Provider: {self.config.NOTIFICATION_PROVIDER}"),
        ]
        
        if self.config.ENABLE_NOTIFICATIONS and self.config.NOTIFICATION_PROVIDER != "disabled":
            if self.config.NOTIFICATION_PROVIDER == "bluesky":
                notification_checks.append((
                    "BlueSky Credentials", 
                    bool(self.config.BLUESKY_HANDLE and self.config.BLUESKY_APP_PASSWORD),
                    "Required for BlueSky notifications"
                ))
            elif self.config.NOTIFICATION_PROVIDER == "email":
                notification_checks.append((
                    "Email Credentials",
                    bool(self.config.SMTP_USERNAME and self.config.SMTP_PASSWORD),
                    "Required for email notifications"
                ))
            elif self.config.NOTIFICATION_PROVIDER == "webhook":
                notification_checks.append((
                    "Webhook URL",
                    bool(self.config.WEBHOOK_URL),
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
        print(f"Provider:          {self.config.NOTIFICATION_PROVIDER}")
        print(f"Enabled:           {self.config.ENABLE_NOTIFICATIONS}")
        if self.config.NOTIFICATION_PROVIDER == "bluesky" and self.config.BLUESKY_HANDLE:
            print(f"BlueSky Handle:    {self.config.BLUESKY_HANDLE}")
        elif self.config.NOTIFICATION_PROVIDER == "email" and self.config.SMTP_USERNAME:
            print(f"Email:             {self.config.SMTP_USERNAME}")
        elif self.config.NOTIFICATION_PROVIDER == "webhook" and self.config.WEBHOOK_URL:
            print(f"Webhook:           {self.config.WEBHOOK_URL}")
        else:
            print(f"Status:            Disabled - Ready for future development")
        
        # ===== FINAL STATUS =====
        if all_passed:
            print(f"\n🎉 Environment validation successful!")
            print(f"✅ Ready to proceed to Core Infrastructure setup")
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
    success = validate_environment()
    sys.exit(0 if success else 1)