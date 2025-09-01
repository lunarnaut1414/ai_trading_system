# 🤖 AI Trading System

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

An institutional-grade, multi-agent AI trading system powered by Claude (Anthropic) that orchestrates intelligent trading decisions through specialized AI agents working in concert.

## 🌟 Features

- **Multi-Agent Architecture**: Six specialized AI agents working together
- **Risk Management**: Built-in position sizing, sector limits, and drawdown protection
- **Real-Time Trading**: Integration with Alpaca Markets for live/paper trading
- **AI-Powered Analysis**: Claude AI for market analysis and decision-making
- **Comprehensive Reporting**: Automated performance analytics and alerts
- **Production Ready**: Full test coverage, error handling, and monitoring

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Orchestration Controller                  │
└─────────────┬───────────────────────────────────┬───────────┘
              │                                   │
    ┌─────────▼──────────┐              ┌────────▼──────────┐
    │   Market Analysis  │              │  Risk Management  │
    ├────────────────────┤              ├───────────────────┤
    │  • Junior Analyst  │              │ • Portfolio Mgr   │
    │  • Senior Analyst  │              │ • Risk Controls   │
    │  • Economist       │              │ • Position Sizing │
    └─────────┬──────────┘              └────────┬──────────┘
              │                                   │
    ┌─────────▼───────────────────────────────────▼──────────┐
    │                   Trade Execution                       │
    ├─────────────────────────────────────────────────────────┤
    │  • Order Management  • Execution Strategies            │
    │  • Slippage Control  • Market/Limit/TWAP/VWAP         │
    └─────────┬───────────────────────────────────────────────┘
              │
    ┌─────────▼───────────────────────────────────────────────┐
    │              Analytics & Reporting                      │
    ├─────────────────────────────────────────────────────────┤
    │  • Performance Metrics  • Risk Alerts  • Daily Reports │
    └─────────────────────────────────────────────────────────┘
```

## 📁 Project Structure

```
ai_trading_system/
├── src/                          # Source code
│   ├── agents/                   # AI Trading Agents
│   │   ├── junior_analyst.py    # Market screening & initial analysis
│   │   ├── senior_analyst.py    # Deep analysis & recommendations
│   │   ├── economist.py         # Macro analysis & market regime
│   │   ├── portfolio_manager.py # Risk management & position sizing
│   │   ├── trade_executor.py    # Order execution & management
│   │   └── analytics_reporter.py # Performance tracking & reporting
│   │
│   ├── core/                     # Core infrastructure
│   │   ├── base_agent.py        # Base agent framework
│   │   ├── infrastructure.py    # System infrastructure
│   │   └── llm_provider.py      # Claude AI integration
│   │
│   ├── data/                     # Data providers
│   │   └── alpaca_provider.py   # Market data integration
│   │
│   ├── database/                 # Database layer
│   │   ├── config.py            # Database configuration
│   │   ├── manager.py           # Database operations
│   │   ├── models/              # SQLAlchemy models
│   │   └── repositories/        # Data repositories
│   │
│   ├── orchestration/            # System orchestration
│   │   ├── controller.py        # Main controller
│   │   ├── workflow_engine.py   # Workflow management
│   │   └── daily_workflow.py    # Daily trading workflow
│   │
│   └── utils/                    # Utilities
│       ├── logger.py            # Logging utilities
│       └── performance_tracker.py # Performance metrics
│
├── config/                       # Configuration
│   ├── settings.py              # System settings
│   └── validator.py             # Config validation
│
├── tests/                        # Test suite
│   ├── unit/                    # Unit tests
│   ├── integration/             # Integration tests
│   └── fixtures/                # Test fixtures
│
├── scripts/                      # Executable scripts
│   ├── run_system.py            # Main entry point
│   └── setup_environment.py     # Environment setup
│
├── docs/                         # Documentation
│   ├── architecture/            # System design docs
│   ├── guides/                  # User guides
│   └── api/                     # API reference
│
└── requirements.txt             # Python dependencies
```

## 🚀 Quick Start

### Prerequisites

- Python 3.9 or higher
- Alpaca Trading Account ([Sign up here](https://alpaca.markets/))
- Anthropic API Key ([Get it here](https://console.anthropic.com/))

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/ai-trading-system.git
cd ai-trading-system
```

### 2. Set Up Virtual Environment

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On macOS/Linux:
source venv/bin/activate
# On Windows:
venv\Scripts\activate

# Upgrade pip
pip install --upgrade pip
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Environment

Create a `.env` file in the project root:

```bash
# Copy example environment file
cp .env.example .env

# Edit with your credentials
nano .env
```

Add your credentials to `.env`:

```env
# Alpaca API Credentials
ALPACA_API_KEY=your_alpaca_api_key
ALPACA_SECRET_KEY=your_alpaca_secret_key
ALPACA_BASE_URL=https://paper-api.alpaca.markets
ALPACA_PAPER=true

# AI Provider
ANTHROPIC_API_KEY=your_anthropic_api_key
LLM_PROVIDER=anthropic

# Database
DATABASE_URL=sqlite:///trading_system.db

# Trading Configuration
MAX_POSITIONS=10
MAX_POSITION_SIZE=0.05
MAX_SECTOR_EXPOSURE=0.25
DAILY_LOSS_LIMIT=0.02
MIN_CASH_RESERVE=0.10
RISK_TOLERANCE=moderate

# System Settings
ENVIRONMENT=development
LOG_LEVEL=INFO
```

### 5. Initialize Database

```bash
# Run database migrations
python scripts/setup_environment.py --init-db
```

### 6. Validate Setup

```bash
# Run configuration validation
python config/validator.py

# Run system tests
pytest tests/unit -v
```

## 🎮 Running the System

### Start the Trading System

```bash
# Run in paper trading mode (recommended for testing)
python scripts/run_system.py --start

# Run with specific log level
python scripts/run_system.py --start --log-level DEBUG
```

### Run Manual Workflow

```bash
# Execute a single trading workflow
python scripts/run_system.py --workflow
```

### Check System Status

```bash
# View system health and status
python scripts/run_system.py --status
```

## 📊 Agent Capabilities

### 1. **Junior Analyst** (`junior_analyst.py`)
- Screens market for opportunities
- Performs initial technical analysis
- Filters stocks based on criteria
- Generates preliminary recommendations

### 2. **Senior Analyst** (`senior_analyst.py`)
- Conducts deep fundamental analysis
- Evaluates company financials
- Analyzes competitive positioning
- Provides detailed investment thesis

### 3. **Economist** (`economist.py`)
- Analyzes macroeconomic conditions
- Determines market regime (Risk-On/Off/Neutral)
- Evaluates sector rotations
- Provides economic context

### 4. **Portfolio Manager** (`portfolio_manager.py`)
- Manages risk allocation
- Sizes positions using Kelly Criterion
- Monitors portfolio exposure
- Generates trading decisions

### 5. **Trade Executor** (`trade_executor.py`)
- Executes orders efficiently
- Implements various execution strategies (TWAP, VWAP, etc.)
- Manages order lifecycle
- Minimizes slippage

### 6. **Analytics Reporter** (`analytics_reporter.py`)
- Tracks performance metrics
- Generates daily/weekly/monthly reports
- Monitors risk alerts
- Provides system diagnostics

## 🧪 Testing

### Run All Tests

```bash
# Run complete test suite
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html
```

### Run Specific Test Categories

```bash
# Unit tests only
pytest tests/unit -v

# Integration tests only
pytest tests/integration -v

# Run tests with markers
pytest -m "not slow"  # Skip slow tests
pytest -m "smoke"      # Quick smoke tests
```

## 📈 Monitoring & Reports

### View Trading Logs

```bash
# Today's logs
tail -f logs/trading_system_$(date +%Y%m%d).log

# All logs
ls -la logs/
```

### Access Reports

Reports are automatically generated in the `reports/` directory:

```bash
reports/
├── daily/         # Daily trading summaries
├── weekly/        # Weekly performance reports
├── monthly/       # Monthly reviews
├── alerts/        # Risk alerts and warnings
├── trades/        # Trade execution reports
└── snapshots/     # Portfolio snapshots
```

## 🔧 Configuration

### Risk Management Settings

Edit `config/settings.py` to adjust risk parameters:

```python
MAX_POSITIONS = 10              # Maximum number of positions
MAX_POSITION_SIZE = 0.05        # Max 5% per position
MAX_SECTOR_EXPOSURE = 0.25      # Max 25% in one sector
DAILY_LOSS_LIMIT = 0.02         # 2% daily loss limit
MIN_CASH_RESERVE = 0.10         # Keep 10% cash
```

### Execution Strategies

Available execution strategies in `trade_executor.py`:
- `MARKET`: Immediate market orders
- `LIMIT`: Limit orders with price targets
- `TWAP`: Time-weighted average price
- `VWAP`: Volume-weighted average price
- `ICEBERG`: Large orders split into smaller chunks
- `OPPORTUNISTIC`: Smart routing for best execution

## 🚀 Production Deployment

### Using Docker

```bash
# Build Docker image
docker build -t ai-trading-system .

# Run container
docker-compose up -d
```

### Using Systemd (Linux)

```bash
# Copy service file
sudo cp deployment/ai-trading-system.service /etc/systemd/system/

# Enable and start service
sudo systemctl enable ai-trading-system
sudo systemctl start ai-trading-system

# Check status
sudo systemctl status ai-trading-system
```

### Scheduled Execution

The system includes a scheduler for automated trading:

```bash
# Run scheduled trading (runs daily at market close)
python scripts/schedule_daily_runs.py
```

## 📚 Documentation

- **[Setup Guide](docs/guides/setup.md)** - Detailed setup instructions
- **[Architecture Overview](docs/architecture/overview.md)** - System design details
- **[API Reference](docs/api/reference.md)** - Code API documentation
- **[Testing Guide](docs/guides/testing.md)** - Testing strategies and examples

## 🛡️ Security

- Never commit `.env` files or API keys
- Use environment variables for sensitive data
- Regularly rotate API keys
- Monitor for unusual trading activity
- Set up alerts for risk limits

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## ⚠️ Disclaimer

**IMPORTANT**: This software is for educational and research purposes only. 

- Not financial advice
- Past performance doesn't guarantee future results
- Trading involves substantial risk of loss
- Always do your own research
- Start with paper trading before using real money

## 🙏 Acknowledgments

- [Alpaca Markets](https://alpaca.markets/) for trading API
- [Anthropic](https://www.anthropic.com/) for Claude AI
- [SQLAlchemy](https://www.sqlalchemy.org/) for database ORM
- All open-source contributors

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/yourusername/ai-trading-system/issues)
- **Discussions**: [GitHub Discussions](https://github.com/yourusername/ai-trading-system/discussions)
- **Documentation**: [Full Docs](docs/)

---

**Built with ❤️ by Julian Wang **

