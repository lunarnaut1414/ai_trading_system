# AI Trading System - Environment Setup Guide

Complete guide for setting up the institutional-grade AI Trading System development environment.

---

## 📋 Prerequisites

Before beginning setup, ensure you have the following:

### Required Software
- **Python 3.9 or higher** (`python --version` to check)
- **Git version control** (`git --version` to check)
- **Code editor** (VS Code recommended)
- **Terminal/Command line** access

### Required Accounts & API Keys
- **Alpaca Trading Account** - [Sign up here](https://alpaca.markets/)
  - Enable paper trading for testing
  - Generate API keys from dashboard
- **Anthropic Claude API Key** - [Get it here](https://console.anthropic.com/)
  - Create account and generate API key

### Optional (for Production)
- **PostgreSQL Database** (for production deployments)
- **Email Account** (for notifications)

---

## 🚀 Quick Start Setup

### 1. Clone the Repository

```bash
# Clone the repository
git clone https://github.com/yourusername/ai-trading-system.git
cd ai-trading-system

# Or if downloaded as ZIP
unzip ai-trading-system.zip
cd ai-trading-system
```

### 2. Create Virtual Environment

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On macOS/Linux:
source venv/bin/activate
# On Windows:
venv\Scripts\activate

# Verify activation (should show path to venv)
which python

# Upgrade pip to latest version
pip install --upgrade pip
```

### 3. Install Dependencies

```bash
# Install all required packages
pip install -r requirements.txt

# For PostgreSQL support (optional)
pip install psycopg2-binary
```

### 4. Environment Configuration

```bash
# Copy example environment file
cp .env.example .env

# Edit with your credentials
nano .env
# or
code .env
```

**Configure your `.env` file with your actual credentials:**

```env
# ===== ALPACA TRADING API =====
ALPACA_API_KEY=your_alpaca_api_key_here
ALPACA_SECRET_KEY=your_alpaca_secret_key_here
ALPACA_BASE_URL=https://paper-api.alpaca.markets
ALPACA_PAPER=true

# ===== AI PROVIDER =====
ANTHROPIC_API_KEY=your_anthropic_api_key_here
LLM_PROVIDER=anthropic

# ===== DATABASE =====
# Option 1: SQLite (recommended for development)
DATABASE_URL=sqlite:///./trading_system.db

# Option 2: PostgreSQL (for production)
# DB_HOST=localhost
# DB_PORT=5432
# DB_NAME=trading_system_dev
# DB_USER=postgres
# DB_PASSWORD=your_password

# ===== SYSTEM CONFIGURATION =====
ENVIRONMENT=development
LOG_LEVEL=INFO

# ===== TRADING CONFIGURATION =====
MAX_POSITIONS=10
MAX_POSITION_SIZE=0.05
MAX_SECTOR_EXPOSURE=0.25
DAILY_LOSS_LIMIT=0.02
MIN_CASH_RESERVE=0.10
RISK_TOLERANCE=moderate
```

### 5. Initialize Database

```bash
# Initialize database and create tables
python scripts/setup_environment.py --init-db
```

### 6. Validate Setup

```bash
# Validate environment configuration
python scripts/setup_environment.py --validate

# Run system tests
pytest tests/unit -v
```

---

## 🗄️ Database Setup Options

### Option 1: SQLite (Recommended for Development)

**Advantages:**
- ✅ No separate database server needed
- ✅ File-based - easy to backup/restore  
- ✅ No additional dependencies
- ✅ Perfect for development and testing
- ✅ Easy to reset/recreate

**Configuration:**
```env
DATABASE_URL=sqlite:///./trading_system.db
```

**Initialize:**
```bash
python scripts/setup_environment.py --init-db
```

### Option 2: PostgreSQL (Production Ready)

**Advantages:**
- ✅ Production-grade performance
- ✅ Advanced query capabilities
- ✅ Better concurrent access handling
- ✅ Robust backup and recovery

**macOS Setup:**
```bash
# Install PostgreSQL
brew install postgresql

# Start PostgreSQL service
brew services start postgresql

# Create database
createdb trading_system_dev

# Install Python driver
pip install psycopg2-binary
```

**Linux Setup:**
```bash
# Ubuntu/Debian
sudo apt update
sudo apt install postgresql postgresql-contrib

# Start service
sudo systemctl start postgresql
sudo systemctl enable postgresql

# Create database and user
sudo -u postgres createdb trading_system_dev
sudo -u postgres createuser --interactive
```

**Configuration:**
```env
DB_HOST=localhost
DB_PORT=5432
DB_NAME=trading_system_dev
DB_USER=postgres
DB_PASSWORD=your_password
```

---

## 🔧 Setup Script Usage

The `setup_environment.py` script provides comprehensive system setup functionality:

### Basic Commands

```bash
# Initialize database
python scripts/setup_environment.py --init-db

# Validate environment
python scripts/setup_environment.py --validate

# Check dependencies
python scripts/setup_environment.py --check-deps

# Complete setup (all steps)
python scripts/setup_environment.py --full-setup

# Show system information
python scripts/setup_environment.py --info
```

### Advanced Commands

```bash
# Reset database (WARNING: deletes all data)
python scripts/setup_environment.py --reset-db

# Create sample test data
python scripts/setup_environment.py --sample-data

# Async database initialization
python scripts/setup_environment.py --init-db-async

# Verbose logging
python scripts/setup_environment.py --init-db --verbose
```

---

## 🧪 Testing Your Setup

### 1. Environment Validation

```bash
# Validate all configuration
python config/validator.py
```

**Expected Output:**
```
🔍 Validating Configuration...
✅ Alpaca API credentials format valid
✅ Anthropic API key format valid  
✅ Risk management parameters valid
✅ Database configuration valid
✅ Ready to proceed to Core Infrastructure setup
```

### 2. Run Unit Tests

```bash
# Run all unit tests
pytest tests/unit -v

# Run specific test categories
pytest tests/unit/database -v          # Database tests
pytest tests/unit/agents -v            # Agent tests
pytest tests/unit/config -v            # Configuration tests

# Run with coverage report
pytest tests/unit --cov=src --cov-report=html
```

### 3. System Health Check

```bash
# Check system status
python scripts/setup_environment.py --info
```

---

## 🎮 Running the System

### Development Mode

```bash
# Start system in paper trading mode
python scripts/run_system.py --start

# Run with debug logging
python scripts/run_system.py --start --log-level DEBUG

# Run single workflow execution
python scripts/run_system.py --workflow

# Check system status
python scripts/run_system.py --status
```

---

## 🔍 Troubleshooting

### Common Issues

#### 1. Import Errors
```
❌ ModuleNotFoundError: No module named 'src'
```

**Solution:**
```bash
# Ensure you're in the project root directory
pwd  # Should show ai-trading-system path

# Verify Python path
python -c "import sys; print(sys.path)"
```

#### 2. Database Connection Errors

**SQLite Errors:**
```bash
# Check file permissions
ls -la trading_system.db

# Remove and recreate database
rm trading_system.db
python scripts/setup_environment.py --init-db
```

**PostgreSQL Errors:**
```bash
# Check PostgreSQL is running
brew services list | grep postgresql

# Test connection
psql -h localhost -p 5432 -U postgres -d trading_system_dev

# Restart PostgreSQL
brew services restart postgresql
```

#### 3. API Key Issues
```
❌ Invalid API key format
```

**Solution:**
- Verify API keys are correctly copied (no extra spaces)
- Check API key permissions in respective dashboards
- Ensure keys are active and not expired

#### 4. Missing Dependencies
```
❌ No module named 'anthropic'
```

**Solution:**
```bash
# Reinstall requirements
pip install -r requirements.txt

# Or install specific package
pip install anthropic
```

### Environment-Specific Issues

#### macOS
```bash
# Install Xcode command line tools if needed
xcode-select --install

# Update Homebrew
brew update && brew upgrade
```

#### Linux
```bash
# Install build essentials
sudo apt-get install build-essential

# Install Python development headers
sudo apt-get install python3-dev
```

#### Windows
```bash
# Use Windows Subsystem for Linux (WSL) for best compatibility
# Or ensure Visual Studio Build Tools are installed
```

---

## 📁 Project Structure

After successful setup, your project structure should look like:

```
ai-trading-system/
├── src/                          # Source code
│   ├── agents/                  # AI trading agents
│   │   ├── junior_analyst.py   # Market screening agent
│   │   ├── senior_analyst.py   # Deep analysis agent  
│   │   ├── economist.py        # Macro analysis agent
│   │   ├── portfolio_manager.py # Risk & position management
│   │   ├── trade_executor.py   # Order execution agent
│   │   └── analytics_agent.py  # Reporting & analytics
│   │
│   ├── data/                    # Data providers
│   │   └── alpaca_provider.py  # Alpaca integration
│   │
│   ├── database/                # Database layer
│   │   ├── config.py           # DB configuration
│   │   ├── manager.py          # DB connection management
│   │   ├── models/             # SQLAlchemy models
│   │   └── repositories/       # Data access layer
│   │
│   ├── orchestration/           # System coordination
│   │   ├── controller.py       # Main orchestrator
│   │   ├── workflow_engine.py  # Task management
│   │   └── daily_workflow.py   # Trading workflow
│   │
│   └── utils/                   # Utilities
│       ├── logger.py           # Logging setup
│       └── performance_tracker.py # Metrics tracking
│
├── config/                      # Configuration
│   ├── settings.py             # System settings
│   └── validator.py            # Config validation
│
├── scripts/                     # Executable scripts  
│   ├── setup_environment.py    # Environment setup
│   └── run_system.py          # System launcher
│
├── tests/                       # Test suite
│   ├── unit/                   # Unit tests
│   ├── integration/            # Integration tests
│   └── fixtures/               # Test data
│
├── docs/                        # Documentation
├── logs/                        # Log files
├── .env                        # Environment variables
├── .env.example               # Environment template
├── requirements.txt           # Python dependencies
└── trading_system.db          # SQLite database (if using SQLite)
```

---

## 🚨 Security Notes

### Environment Variables
- **Never commit `.env` files** to version control
- Use **strong, unique passwords** for database access
- **Rotate API keys** regularly
- Use **paper trading** for development and testing

### API Key Security
- Store API keys securely in environment variables
- Use minimum required permissions
- Monitor API usage for unusual activity
- Revoke unused or compromised keys immediately

---

## 📈 Next Steps

After successful setup:

1. **Review Configuration**: `python config/validator.py`
2. **Run Tests**: `pytest tests/unit -v`
3. **Explore Agents**: Check individual agent functionality
4. **Start System**: `python scripts/run_system.py --start`
5. **Monitor Logs**: Check `logs/` directory for system activity
6. **Review Documentation**: Explore `docs/` for detailed guides

---

## 🆘 Getting Help

### Documentation
- **Architecture Guide**: `docs/architecture/`
- **Agent Documentation**: `docs/agents/`
- **API Reference**: `docs/api/`

### Support Channels
- **Issues**: Create GitHub issues for bugs or feature requests
- **Discussions**: Use GitHub discussions for questions
- **Documentation**: Check inline code documentation

### Debug Information Collection
When reporting issues, include:

```bash
# System information
python scripts/setup_environment.py --info

# Environment validation
python config/validator.py

# Test results
pytest tests/unit -v --tb=short

# Log files from logs/ directory
```

---

*Last Updated: September 2025*  
*Version: 1.0.0*