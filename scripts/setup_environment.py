#!/usr/bin/env python3
"""
AI Trading System - Environment Setup and Database Initialization Script

This script handles:
- Database initialization and migration
- Environment validation
- System dependency checks
- Initial configuration setup

Usage:
    python scripts/setup_environment.py --init-db            # Initialize database
    python scripts/setup_environment.py --validate          # Validate environment
    python scripts/setup_environment.py --check-deps        # Check dependencies
    python scripts/setup_environment.py --full-setup        # Complete setup
    python scripts/setup_environment.py --help              # Show help

Author: AI Trading System
Version: 1.0.0
"""

import sys
import os
import argparse
import asyncio
import logging
from pathlib import Path
from typing import Optional, List, Dict, Any
from datetime import datetime
import importlib.util

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import project modules
try:
    from src.database.manager import get_db_manager, DatabaseManager
    from src.database.config import get_database_config, DatabaseEnvironment
    from src.database.models import Base
    from config.settings import TradingConfig
    from config.validator import ConfigurationValidator
except ImportError as e:
    print(f"❌ Failed to import required modules: {e}")
    print(f"💡 Make sure you're running from the project root directory")
    sys.exit(1)


class EnvironmentSetup:
    """Main setup and initialization class."""
    
    def __init__(self):
        """Initialize setup manager."""
        self.logger = self._setup_logging()
        self.config = None
        self.db_manager = None
        
    def _setup_logging(self) -> logging.Logger:
        """Setup logging for setup process."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        return logging.getLogger(__name__)

    def validate_environment(self) -> bool:
        """Validate environment configuration."""
        print("\n🔍 Validating Environment Configuration...")
        print("=" * 60)
        
        try:
            # Load configuration
            self.config = TradingConfig()
            
            # Validate using configuration validator
            validator = ConfigurationValidator()
            is_valid = validator.validate_environment()
            
            if is_valid:
                print("✅ Environment validation successful!")
                return True
            else:
                print("❌ Environment validation failed!")
                return False
                
        except Exception as e:
            self.logger.error(f"Environment validation error: {e}")
            print(f"❌ Environment validation failed: {e}")
            return False

    def check_dependencies(self) -> bool:
        """Check system dependencies."""
        print("\n📦 Checking System Dependencies...")
        print("=" * 60)
        
        required_packages = [
            'anthropic',
            'alpaca_trade_api', 
            'sqlalchemy',
            'pandas',
            'numpy',
            'pytest',
            'python-dotenv',
            'pydantic'
        ]
        
        missing_packages = []
        
        for package in required_packages:
            try:
                if package == 'alpaca_trade_api':
                    import alpaca_trade_api
                    version = getattr(alpaca_trade_api, '__version__', 'unknown')
                    print(f"✅ {package} (version: {version})")
                else:
                    spec = importlib.util.find_spec(package.replace('-', '_'))
                    if spec is not None:
                        module = importlib.import_module(package.replace('-', '_'))
                        version = getattr(module, '__version__', 'unknown')
                        print(f"✅ {package} (version: {version})")
                    else:
                        missing_packages.append(package)
                        print(f"❌ {package} - NOT FOUND")
            except ImportError:
                missing_packages.append(package)
                print(f"❌ {package} - NOT FOUND")
        
        if missing_packages:
            print(f"\n❌ Missing packages: {', '.join(missing_packages)}")
            print(f"💡 Install missing packages with: pip install {' '.join(missing_packages)}")
            return False
        
        print("\n✅ All required dependencies are installed!")
        return True

    def init_database(self) -> bool:
        """Initialize database and create tables."""
        print("\n🗄️  Initializing Database...")
        print("=" * 60)
        
        try:
            # Get database configuration
            db_config = get_database_config()
            print(f"📊 Database: {db_config.database}")
            print(f"🌐 Host: {db_config.host}:{db_config.port}")
            print(f"👤 User: {db_config.username}")
            
            # Initialize database manager
            self.db_manager = get_db_manager()
            
            # Create tables synchronously
            print("🔨 Creating database tables...")
            self.db_manager.create_tables()
            
            # Verify table creation
            print("🔍 Verifying table creation...")
            with self.db_manager.get_session() as session:
                # Try to query a simple count to verify connectivity
                result = session.execute("SELECT 1")
                if result:
                    print("✅ Database connection verified!")
                else:
                    raise Exception("Database connection test failed")
            
            print("✅ Database initialization completed successfully!")
            return True
            
        except Exception as e:
            self.logger.error(f"Database initialization error: {e}")
            print(f"❌ Database initialization failed: {e}")
            
            # Provide troubleshooting guidance
            self._print_db_troubleshooting()
            return False

    async def init_database_async(self) -> bool:
        """Initialize database asynchronously."""
        print("\n🗄️  Initializing Database (Async)...")
        print("=" * 60)
        
        try:
            # Get database configuration
            db_config = get_database_config()
            print(f"📊 Database: {db_config.database}")
            print(f"🌐 Host: {db_config.host}:{db_config.port}")
            print(f"👤 User: {db_config.username}")
            
            # Initialize async database manager
            from src.database.manager import AsyncDatabaseManager
            async_db_manager = AsyncDatabaseManager()
            
            # Create tables asynchronously
            print("🔨 Creating database tables...")
            await async_db_manager.create_tables()
            
            print("✅ Async database initialization completed successfully!")
            return True
            
        except Exception as e:
            self.logger.error(f"Async database initialization error: {e}")
            print(f"❌ Async database initialization failed: {e}")
            return False

    def _print_db_troubleshooting(self):
        """Print database troubleshooting information."""
        print("\n🔧 Database Troubleshooting:")
        print("=" * 40)
        print("1. Check your .env file contains valid database credentials")
        print("2. Ensure database server is running")
        print("3. Verify database exists and is accessible")
        print("4. Check network connectivity to database host")
        print("5. For SQLite: ensure directory has write permissions")
        print("\nExample .env configuration:")
        print("DATABASE_URL=sqlite:///./trading_system.db")
        print("# OR for PostgreSQL:")
        print("DB_HOST=localhost")
        print("DB_PORT=5432") 
        print("DB_NAME=trading_system_dev")
        print("DB_USER=postgres")
        print("DB_PASSWORD=your_password")

    def create_sample_data(self) -> bool:
        """Create sample data for testing."""
        print("\n📝 Creating Sample Data...")
        print("=" * 60)
        
        try:
            # Import required models
            from src.database.models.trade import Trade, TradeStatus, TradeType
            from src.database.models.position import Position, PositionStatus
            from datetime import datetime, timezone
            
            with self.db_manager.get_session() as session:
                # Create sample trade
                sample_trade = Trade(
                    symbol="AAPL",
                    trade_type=TradeType.BUY,
                    quantity=10,
                    entry_price=150.00,
                    status=TradeStatus.FILLED,
                    timestamp=datetime.now(timezone.utc),
                    strategy="sample_strategy"
                )
                
                session.add(sample_trade)
                session.commit()
                
                print("✅ Sample data created successfully!")
                return True
                
        except Exception as e:
            self.logger.error(f"Sample data creation error: {e}")
            print(f"❌ Sample data creation failed: {e}")
            return False

    def run_full_setup(self) -> bool:
        """Run complete system setup."""
        print("\n🚀 Running Full System Setup...")
        print("=" * 60)
        print(f"⏰ Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        setup_steps = [
            ("Checking Dependencies", self.check_dependencies),
            ("Validating Environment", self.validate_environment),
            ("Initializing Database", self.init_database),
        ]
        
        for step_name, step_function in setup_steps:
            print(f"\n🔄 {step_name}...")
            success = step_function()
            
            if not success:
                print(f"❌ Setup failed at step: {step_name}")
                return False
            
            print(f"✅ {step_name} completed successfully!")
        
        print("\n🎉 Full system setup completed successfully!")
        print("=" * 60)
        print("Next steps:")
        print("1. Run: python config/validator.py")
        print("2. Run tests: pytest tests/unit -v") 
        print("3. Start system: python scripts/run_system.py --start")
        
        return True

    def reset_database(self) -> bool:
        """Reset database by dropping and recreating tables."""
        print("\n🔄 Resetting Database...")
        print("=" * 60)
        
        confirm = input("⚠️  This will DELETE all data. Continue? (yes/no): ")
        if confirm.lower() not in ['yes', 'y']:
            print("❌ Database reset cancelled.")
            return False
        
        try:
            if not self.db_manager:
                self.db_manager = get_db_manager()
            
            # Drop and recreate tables
            print("🗑️  Dropping existing tables...")
            self.db_manager.reset_database()
            
            print("✅ Database reset completed successfully!")
            return True
            
        except Exception as e:
            self.logger.error(f"Database reset error: {e}")
            print(f"❌ Database reset failed: {e}")
            return False

    def show_system_info(self):
        """Show system information."""
        print("\n📊 System Information")
        print("=" * 60)
        
        # Python version
        print(f"🐍 Python: {sys.version}")
        
        # Project root
        print(f"📁 Project Root: {project_root}")
        
        # Environment
        env = os.getenv('ENVIRONMENT', 'development')
        print(f"🌍 Environment: {env}")
        
        # Database info
        try:
            db_config = get_database_config()
            print(f"🗄️  Database: {db_config.database}")
            print(f"🌐 DB Host: {db_config.host}:{db_config.port}")
        except Exception as e:
            print(f"🗄️  Database: Error loading config - {e}")
        
        # Config info
        try:
            config = TradingConfig()
            print(f"📈 Max Positions: {config.MAX_POSITIONS}")
            print(f"📊 Risk Tolerance: {config.RISK_TOLERANCE}")
        except Exception as e:
            print(f"⚙️  Config: Error loading - {e}")


def main():
    """Main entry point for setup script."""
    parser = argparse.ArgumentParser(
        description="AI Trading System - Environment Setup Script",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/setup_environment.py --init-db      # Initialize database
  python scripts/setup_environment.py --validate     # Validate environment  
  python scripts/setup_environment.py --full-setup   # Complete setup
  python scripts/setup_environment.py --reset-db     # Reset database
        """
    )
    
    parser.add_argument('--init-db', action='store_true',
                       help='Initialize database and create tables')
    parser.add_argument('--init-db-async', action='store_true',
                       help='Initialize database asynchronously')
    parser.add_argument('--validate', action='store_true',
                       help='Validate environment configuration')
    parser.add_argument('--check-deps', action='store_true',
                       help='Check system dependencies')
    parser.add_argument('--full-setup', action='store_true',
                       help='Run complete system setup')
    parser.add_argument('--reset-db', action='store_true',
                       help='Reset database (WARNING: deletes all data)')
    parser.add_argument('--sample-data', action='store_true',
                       help='Create sample data for testing')
    parser.add_argument('--info', action='store_true',
                       help='Show system information')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose logging')
    
    args = parser.parse_args()
    
    # Show help if no arguments
    if len(sys.argv) == 1:
        parser.print_help()
        return
    
    # Setup logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Initialize setup manager
    setup = EnvironmentSetup()
    
    success = True
    
    try:
        # Execute requested operations
        if args.info:
            setup.show_system_info()
        
        if args.check_deps:
            success &= setup.check_dependencies()
        
        if args.validate:
            success &= setup.validate_environment()
        
        if args.init_db:
            success &= setup.init_database()
            
        if args.init_db_async:
            success &= asyncio.run(setup.init_database_async())
        
        if args.reset_db:
            success &= setup.reset_database()
        
        if args.sample_data:
            success &= setup.create_sample_data()
        
        if args.full_setup:
            success &= setup.run_full_setup()
        
        # Print final status
        if success:
            print(f"\n✅ Setup operations completed successfully!")
            sys.exit(0)
        else:
            print(f"\n❌ Some setup operations failed!")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print(f"\n⚠️  Setup interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 Setup failed with error: {e}")
        logging.exception("Setup error details:")
        sys.exit(1)


if __name__ == "__main__":
    main()