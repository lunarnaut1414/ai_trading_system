# AI Trading System - Complete Environment Setup Guide

## 🚀 Getting Started

This guide will walk you through setting up your institutional-grade AI Trading System from scratch. Follow each step carefully to ensure a robust foundation for your multi-agent trading platform.

## Prerequisites Checklist

Before starting, ensure you have:
- [ ] Python 3.9+ installed (`python --version`)
- [ ] Git version control (`git --version`)
- [ ] Code editor (VS Code recommended)
- [ ] Alpaca Trading account (paper trading enabled)
- [ ] Anthropic Claude API key ([Get it here](https://console.anthropic.com))
- [ ] Terminal/Command line access

## Step 1: Project Structure Creation

### Create Project Directory
```bash
# Create main project directory
mkdir ai_trading_system
cd ai_trading_system

# Create complete directory structure
mkdir -p agents data trading utils config tests logs scripts docs
mkdir -p database/models database/migrations
mkdir -p orchestration screener deployment monitoring backtest

# Create essential Python files
touch __init__.py main.py requirements.txt README.md
touch .env .env.example .gitignore

# Create module __init__.py files
touch agents/__init__.py data/__init__.py trading/__init__.py
touch utils/__init__.py config/__init__.py tests/__init__.py
```

### Verify Structure
```bash
# Check directory structure
tree -d -L 2
```

Expected output:
```
ai_trading_system/
├── agents/
├── backtest/
├── config/
├── data/
├── database/
│   ├── migrations/
│   └── models/
├── deployment/
├── docs/
├── logs/
├── monitoring/
├── orchestration/
├── screener/
├── scripts/
├── tests/
├── trading/
└── utils/
```

## Step 2: Virtual Environment Setup

### Create and Activate Virtual Environment
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Verify activation
which python  # Should show path to venv
pip --version  # Should show pip from venv

# Upgrade pip
pip install --upgrade pip
```

## Step 3: Dependencies Installation

### Create Requirements File
Create `requirements.txt`:
```txt
# ===== CORE TRADING LIBRARIES =====
alpaca-trade-api==3.0.2
yfinance==0.2.22

# ===== AI/ML LIBRARIES =====
anthropic==0.7.0

# ===== WEB FRAMEWORK =====
fastapi==0.104.1
uvicorn[standard]==0.24.0

# ===== DATABASE =====
sqlalchemy==2.0.23
alembic==1.12.1

# ===== DATA PROCESSING =====
pandas==2.1.3
numpy==1.25.2
scipy==1.11.4
matplotlib==3.8.2
seaborn==0.13.0

# ===== ASYNC SUPPORT =====
# Note: aiohttp version constrained by alpaca-trade-api
aiohttp==3.8.2
aiofiles==23.2.0

# ===== UTILITIES =====
python-dotenv==1.0.0
schedule==1.2.0
requests==2.31.0
pydantic==2.5.0
python-dateutil==2.8.2

# ===== DEVELOPMENT TOOLS =====
pytest==7.4.3
pytest-asyncio==0.21.1
pytest-cov==4.1.0
black==23.11.0
flake8==6.1.0
mypy==1.7.1

# ===== MONITORING =====
prometheus-client==0.19.0
structlog==23.2.0

# ===== TECHNICAL ANALYSIS =====
# Note: TA-Lib requires special installation on macOS - see installation instructions
# Alternative: pandas-ta (easier to install)
pandas-ta==0.3.14b
```

### Install Dependencies

**Important**: Due to dependency conflicts between packages, follow this specific installation order:

```bash
# Step 1: Clean any failed installations (if needed)
# If you had dependency conflicts, clean first:
# pip uninstall -y -r requirements.txt
# Or recreate virtual environment if needed:
# deactivate && rm -rf venv && python -m venv venv && source venv/bin/activate

# Step 2: Install core trading library first (this constrains aiohttp version)
pip install alpaca-trade-api==3.0.2

# Step 3: Install remaining dependencies
pip install yfinance==0.2.22
pip install anthropic==0.7.0
pip install fastapi==0.104.1
pip install uvicorn[standard]==0.24.0
pip install sqlalchemy==2.0.23
pip install alembic==1.12.1
pip install pandas==2.1.3
pip install numpy==1.25.2
pip install scipy==1.11.4
pip install matplotlib==3.8.2
pip install seaborn==0.13.0
pip install aiofiles==23.2.0
pip install python-dotenv==1.0.0
pip install schedule==1.2.0
pip install requests==2.31.0
pip install pydantic==2.5.0
pip install python-dateutil==2.8.2
pip install pytest==7.4.3
pip install pytest-asyncio==0.21.1
pip install pytest-cov==4.1.0
pip install black==23.11.0
pip install flake8==6.1.0
pip install structlog==23.2.0

# Optional: BlueSky notifications support
# pip install atproto

# Optional: pandas-ta for technical analysis
pip install pandas-ta==0.3.14b

# Step 4: Verify critical installations
pip list | grep -E "(alpaca|anthropic|fastapi|pandas|sqlalchemy)"

# Step 5: Test key imports
python -c "
import alpaca_trade_api
import anthropic
import pandas as pd
import sqlalchemy
import fastapi
print('✅ All critical imports successful')
"
```

**Note about TA-Lib**: Technical Analysis Library requires special installation on macOS:
```bash
# Option 1: Install TA-Lib (requires Homebrew)
brew install ta-lib
pip install TA-Lib

# Option 2: Use pandas-ta instead (easier alternative)
pip install pandas-ta
```

## Step 4: Configuration Setup

### Create Environment Template
Create `.env.example`:
```env
# ===== ALPACA TRADING API =====
ALPACA_API_KEY=your_alpaca_api_key_here
ALPACA_SECRET_KEY=your_alpaca_secret_key_here
ALPACA_BASE_URL=https://paper-api.alpaca.markets
ALPACA_PAPER=true

# ===== AI/LLM PROVIDERS =====
ANTHROPIC_API_KEY=your_anthropic_api_key_here
LLM_PROVIDER=anthropic

# ===== RISK MANAGEMENT =====
MAX_POSITIONS=5
MAX_POSITION_SIZE=0.05
MAX_SECTOR_EXPOSURE=0.25
DAILY_LOSS_LIMIT=0.02
MIN_CASH_RESERVE=0.10
RISK_TOLERANCE=moderate

# ===== SYSTEM CONFIGURATION =====
LOG_LEVEL=INFO
DATABASE_URL=sqlite:///./trading_system.db
TESTING_MODE=false
DRY_RUN=false

# ===== NOTIFICATION SETTINGS =====
# Options: bluesky, email, webhook, disabled
NOTIFICATION_PROVIDER=disabled
ENABLE_NOTIFICATIONS=false

# BlueSky Configuration (if using bluesky provider)
BLUESKY_HANDLE=your.bluesky.handle
BLUESKY_APP_PASSWORD=your_bluesky_app_password

# Email Configuration (if using email provider)
SMTP_SERVER=smtp.gmail.com
SMTP_PORT=587
SMTP_USERNAME=your_email@gmail.com
SMTP_PASSWORD=your_email_app_password
NOTIFICATION_EMAIL=your_notification_email@gmail.com

# Webhook Configuration (if using webhook provider)
WEBHOOK_URL=https://your-webhook-endpoint.com/alerts
```

### Create Your Environment File
```bash
# Copy template to create your environment file
cp .env.example .env

# Edit with your actual API keys
nano .env  # or use your preferred editor
```

### Create Configuration Module
First, create `config/__init__.py`:
```python
"""
Configuration package for AI Trading System
"""
```

Then create `config/settings.py`:
```python
"""
AI Trading System Configuration Management
Centralized configuration with validation and type safety
"""

import os
from typing import Dict, Any, Optional
from dataclasses import dataclass
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

@dataclass
class TradingConfig:
    """Main configuration class for the trading system"""
    
    # ===== API CREDENTIALS =====
    ALPACA_API_KEY: str = os.getenv("ALPACA_API_KEY", "")
    ALPACA_SECRET_KEY: str = os.getenv("ALPACA_SECRET_KEY", "")
    ALPACA_BASE_URL: str = os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")
    ALPACA_PAPER: bool = os.getenv("ALPACA_PAPER", "true").lower() == "true"
    
    # ===== AI/LLM CONFIGURATION =====
    ANTHROPIC_API_KEY: str = os.getenv("ANTHROPIC_API_KEY", "")
    LLM_PROVIDER: str = "anthropic"  # Fixed to Claude/Anthropic only
    
    # ===== RISK MANAGEMENT =====
    MAX_POSITIONS: int = int(os.getenv("MAX_POSITIONS", "5"))
    MAX_POSITION_SIZE: float = float(os.getenv("MAX_POSITION_SIZE", "0.05"))
    MAX_SECTOR_EXPOSURE: float = float(os.getenv("MAX_SECTOR_EXPOSURE", "0.25"))
    DAILY_LOSS_LIMIT: float = float(os.getenv("DAILY_LOSS_LIMIT", "0.02"))
    MIN_CASH_RESERVE: float = float(os.getenv("MIN_CASH_RESERVE", "0.10"))
    RISK_TOLERANCE: str = os.getenv("RISK_TOLERANCE", "moderate")
    
    # ===== SYSTEM SETTINGS =====
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    DATABASE_URL: str = os.getenv("DATABASE_URL", "sqlite:///./trading_system.db")
    TESTING_MODE: bool = os.getenv("TESTING_MODE", "false").lower() == "true"
    DRY_RUN: bool = os.getenv("DRY_RUN", "false").lower() == "true"
    
    # ===== NOTIFICATION SETTINGS =====
    NOTIFICATION_PROVIDER: str = os.getenv("NOTIFICATION_PROVIDER", "disabled")
    ENABLE_NOTIFICATIONS: bool = os.getenv("ENABLE_NOTIFICATIONS", "false").lower() == "true"
    
    # BlueSky Configuration
    BLUESKY_HANDLE: str = os.getenv("BLUESKY_HANDLE", "")
    BLUESKY_APP_PASSWORD: str = os.getenv("BLUESKY_APP_PASSWORD", "")
    
    # Email Configuration  
    SMTP_SERVER: str = os.getenv("SMTP_SERVER", "smtp.gmail.com")
    SMTP_PORT: int = int(os.getenv("SMTP_PORT", "587"))
    SMTP_USERNAME: str = os.getenv("SMTP_USERNAME", "")
    SMTP_PASSWORD: str = os.getenv("SMTP_PASSWORD", "")
    NOTIFICATION_EMAIL: str = os.getenv("NOTIFICATION_EMAIL", "")
    
    # Webhook Configuration
    WEBHOOK_URL: str = os.getenv("WEBHOOK_URL", "")
    
    def get_trading_params(self) -> Dict[str, Any]:
        """Get trading parameters as dictionary"""
        return {
            "max_positions": self.MAX_POSITIONS,
            "max_position_size": self.MAX_POSITION_SIZE,
            "max_sector_exposure": self.MAX_SECTOR_EXPOSURE,
            "daily_loss_limit": self.DAILY_LOSS_LIMIT,
            "min_cash_reserve": self.MIN_CASH_RESERVE,
            "risk_tolerance": self.RISK_TOLERANCE
        }
    
    def validate_config(self) -> Dict[str, bool]:
        """Validate configuration parameters"""
        validations = {
            "alpaca_credentials": bool(self.ALPACA_API_KEY and self.ALPACA_SECRET_KEY),
            "llm_credentials": bool(self.ANTHROPIC_API_KEY or self.OPENAI_API_KEY),
            "risk_parameters": (
                0 < self.MAX_POSITION_SIZE <= 1.0 and
                0 < self.MAX_SECTOR_EXPOSURE <= 1.0 and
                0 < self.DAILY_LOSS_LIMIT <= 1.0 and
                0 <= self.MIN_CASH_RESERVE <= 1.0 and
                self.MAX_POSITIONS > 0
            ),
            "database_config": bool(self.DATABASE_URL),
            "notification_config": (
                not self.ENABLE_NOTIFICATIONS or 
                self.NOTIFICATION_PROVIDER == "disabled" or
                (self.NOTIFICATION_PROVIDER == "bluesky" and bool(self.BLUESKY_HANDLE and self.BLUESKY_APP_PASSWORD)) or
                (self.NOTIFICATION_PROVIDER == "email" and bool(self.SMTP_USERNAME and self.SMTP_PASSWORD)) or
                (self.NOTIFICATION_PROVIDER == "webhook" and bool(self.WEBHOOK_URL))
            ),
        }
        return validations

# Global configuration instance
config = TradingConfig()
```

### Create Configuration Validator
Create `config/validator.py`:
```python
"""
Environment and Configuration Validator
Comprehensive validation for trading system setup
"""

import sys
import os
from typing import List, Tuple, Any

# Add the parent directory to Python path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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
```

## Step 5: Git Repository Setup

### Create .gitignore
Create `.gitignore`:
```gitignore
# ===== ENVIRONMENT & SECRETS =====
.env
.env.local
.env.production
.env.staging
*.key
*.pem

# ===== PYTHON =====
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg
MANIFEST

# ===== VIRTUAL ENVIRONMENTS =====
venv/
env/
ENV/
.venv/

# ===== IDE & EDITORS =====
.vscode/
.idea/
*.swp
*.swo
*~

# ===== LOGS =====
logs/*.log
*.log

# ===== DATABASE =====
*.db
*.sqlite
*.sqlite3
database/

# ===== CACHE =====
.cache/
.pytest_cache/
.mypy_cache/
.coverage
htmlcov/

# ===== OS GENERATED =====
.DS_Store
Thumbs.db

# ===== TRADING SPECIFIC =====
backtest_results/
trade_logs/
performance_reports/
market_data_cache/
```

### Initialize Git Repository
```bash
# Initialize git repository
git init

# Add files
git add .gitignore .env.example requirements.txt
git add config/ README.md

# Initial commit
git commit -m "Initial commit: Environment setup and configuration"

# Create development branch
git checkout -b development
```

## Step 6: Testing Framework

### Create Test Configuration
Create `tests/conftest.py`:
```python
"""
pytest configuration for AI Trading System
Shared fixtures and test configuration
"""

import pytest
import asyncio
import os
from unittest.mock import Mock
from config.settings import TradingConfig

@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()

@pytest.fixture
def trading_config():
    """Provide test trading configuration"""
    return TradingConfig()

@pytest.fixture
def mock_alpaca_api():
    """Mock Alpaca API for testing"""
    mock_api = Mock()
    mock_api.get_account.return_value = Mock(
        equity=100000,
        buying_power=50000,
        cash=25000
    )
    return mock_api

@pytest.fixture
def sample_market_data():
    """Sample market data for testing"""
    return {
        'AAPL': {
            'price': 150.00,
            'volume': 1000000,
            'change': 2.5,
            'change_percent': 1.67
        },
        'MSFT': {
            'price': 300.00,
            'volume': 800000,
            'change': -1.5,
            'change_percent': -0.50
        }
    }
```

### Create Environment Test
Create `tests/test_environment.py`:
```python
"""
Environment validation tests
Ensures proper setup and configuration
"""

import pytest
import os
from config.settings import TradingConfig
from config.validator import ConfigurationValidator

class TestEnvironmentSetup:
    """Test environment setup and configuration"""
    
    def test_configuration_loading(self):
        """Test configuration loads properly"""
        config = TradingConfig()
        assert config is not None
        assert hasattr(config, 'ALPACA_API_KEY')
        assert hasattr(config, 'MAX_POSITIONS')
    
    def test_risk_parameters_valid(self):
        """Test risk parameters are within valid ranges"""
        config = TradingConfig()
        
        assert 0 < config.MAX_POSITION_SIZE <= 1.0
        assert 0 < config.MAX_SECTOR_EXPOSURE <= 1.0
        assert 0 < config.DAILY_LOSS_LIMIT <= 1.0
        assert 0 <= config.MIN_CASH_RESERVE <= 1.0
        assert config.MAX_POSITIONS > 0
    
    def test_trading_params_dict(self):
        """Test trading parameters dictionary format"""
        config = TradingConfig()
        params = config.get_trading_params()
        
        required_keys = [
            'max_positions', 'max_position_size', 'max_sector_exposure',
            'daily_loss_limit', 'min_cash_reserve', 'risk_tolerance'
        ]
        
        for key in required_keys:
            assert key in params
    
    def test_configuration_validation(self):
        """Test configuration validation method"""
        config = TradingConfig()
        validations = config.validate_config()
        
        assert isinstance(validations, dict)
        assert 'risk_parameters' in validations
        assert validations['risk_parameters'] is True

class TestDependencies:
    """Test required dependencies are available"""
    
    def test_critical_imports(self):
        """Test critical package imports"""
        import pandas as pd
        import sqlalchemy
        import fastapi
        import pydantic
        
        assert pd.__version__ is not None
        assert sqlalchemy.__version__ is not None
    
    def test_alpaca_import(self):
        """Test Alpaca API import"""
        import alpaca_trade_api as tradeapi
        assert tradeapi is not None
    
    @pytest.mark.skipif(
        not os.getenv("ANTHROPIC_API_KEY"),
        reason="No Claude/Anthropic API key available"
    )
    def test_claude_import(self):
        """Test Claude/Anthropic library import"""
        import anthropic
        assert anthropic is not None

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
```

## Step 7: Main Application Entry Point

Create `main.py`:
```python
"""
AI Trading System - Main Application Entry Point
Multi-Agent Trading System with Risk Management
"""

import asyncio
import logging
import sys
import os
from datetime import datetime

# Add current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config.settings import TradingConfig
from config.validator import validate_environment

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/trading_system.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

class TradingSystemManager:
    """Main trading system orchestrator"""
    
    def __init__(self):
        self.config = TradingConfig()
        self.is_running = False
        
    async def startup(self):
        """Initialize trading system"""
        logger.info("🚀 Starting AI Trading System...")
        
        # Validate environment
        if not validate_environment():
            logger.error("❌ Environment validation failed")
            return False
            
        # Initialize components (placeholder for now)
        logger.info("✅ Environment validated successfully")
        logger.info("📊 Trading Parameters:")
        logger.info(f"   Max Positions: {self.config.MAX_POSITIONS}")
        logger.info(f"   Position Size: {self.config.MAX_POSITION_SIZE*100:.1f}%")
        logger.info(f"   Risk Tolerance: {self.config.RISK_TOLERANCE}")
        logger.info(f"   AI Provider: Claude/Anthropic")
        
        self.is_running = True
        return True
    
    async def shutdown(self):
        """Gracefully shutdown trading system"""
        logger.info("🛑 Shutting down AI Trading System...")
        self.is_running = False
        
    async def run(self):
        """Main trading system loop"""
        if not await self.startup():
            return
            
        try:
            logger.info("🔄 Trading system is running...")
            logger.info("📝 Note: Core infrastructure not yet implemented")
            logger.info("➡️  Next: Run 'python config/validator.py' to verify setup")
            
            # Main loop placeholder
            while self.is_running:
                await asyncio.sleep(1)
                
        except KeyboardInterrupt:
            logger.info("🔴 Received shutdown signal")
        except Exception as e:
            logger.error(f"💥 System error: {str(e)}")
        finally:
            await self.shutdown()

async def main():
    """Application entry point"""
    system = TradingSystemManager()
    await system.run()

if __name__ == "__main__":
    asyncio.run(main())
```

## Step 8: Validation and Testing

### Run Environment Validation
```bash
# Validate your environment
python config/validator.py
```

### Run Basic Tests
```bash
# Run environment tests
pytest tests/test_environment.py -v

# Run all tests with coverage
pytest --cov=. --cov-report=html
```

### Test Main Application
```bash
# Test main application
python main.py
```

## ✅ Validation Checklist

Before proceeding to the next phase, ensure:

- [ ] **Virtual environment** created and activated successfully
- [ ] **All dependencies** installed without errors (`pip list` shows all packages)
- [ ] **Configuration files** created (`.env` from `.env.example`)
- [ ] **Environment validation** passes (`python config/validator.py`)
- [ ] **API keys** properly configured in `.env` file
- [ ] **Git repository** initialized with proper `.gitignore`
- [ ] **Basic tests** pass (`pytest tests/test_environment.py`)
- [ ] **Main application** runs without errors (`python main.py`)

## 🔗 Next Steps

Once environment setup is complete and validated, you'll be ready for:

**Phase 2: Core Infrastructure** - This will include:
- Base Agent Framework and communication protocols
- LLM Provider Integration with retry logic
- Logging & Monitoring infrastructure  
- Database Foundation with SQLAlchemy models

## 🚨 Troubleshooting

### Common Issues

**Dependency Conflict Errors**
```bash
# If you get dependency conflicts, clean and reinstall:
pip uninstall -y -r requirements.txt

# Or recreate virtual environment completely:
deactivate
rm -rf venv
python -m venv venv
source venv/bin/activate  # macOS/Linux
# venv\Scripts\activate  # Windows
pip install --upgrade pip

# Follow the step-by-step installation above
```

**Missing API Keys**
```bash
# Check .env file
cat .env | grep -E "(ALPACA|ANTHROPIC|OPENAI)"
```

**Import Errors**
```bash
# Test specific problematic imports
python -c "import alpaca_trade_api; print('Alpaca: OK')"
python -c "import anthropic; print('Claude/Anthropic: OK')"
python -c "import aiohttp; print('aiohttp: OK')"

# Check aiohttp version (should be 3.8.2 for alpaca compatibility)
pip show aiohttp | grep Version
```

**Virtual Environment Issues**
```bash
# Verify virtual environment is activated
which python  # Should show venv path
pip --version  # Should show pip from venv

# If not activated properly:
source venv/bin/activate  # macOS/Linux
# venv\Scripts\activate  # Windows
```

**TA-Lib Installation Issues (macOS)**
```bash
# If TA-Lib fails to install:
# 1. Install via Homebrew first:
brew install ta-lib

# 2. Then install Python wrapper:
pip install TA-Lib

# Alternative: Use pandas-ta instead
pip install pandas-ta
```

## 📞 Support

If you encounter issues:
1. Run the validator: `python config/validator.py`
2. Check logs in `logs/` directory
3. Verify `.env` file has all required values
4. Ensure virtual environment is activated

Your foundation is now ready for building the multi-agent trading system! 🎉