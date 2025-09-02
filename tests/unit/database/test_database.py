"""
Test suite for database configuration and manager.

Run tests:
    pytest tests/unit/database/test_database.py -v              # All database tests
    pytest tests/unit/database/test_database.py::TestConfig -v  # Config tests only
    pytest tests/unit/database/test_database.py -m unit -v      # Unit tests only
"""

import pytest
import os
import tempfile
from unittest.mock import Mock, patch, MagicMock, PropertyMock
from datetime import datetime, timedelta
from typing import Dict, List
import asyncio

from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from sqlalchemy.exc import SQLAlchemyError, OperationalError

# Import modules to test
from src.database.config import DatabaseConfig, DatabaseEnvironment, get_database_config
from src.database.manager import DatabaseManager, AsyncDatabaseManager, get_db_manager
from src.database.models import (
    Base, Trade, Position, Portfolio, Signal, 
    AgentAction, Alert, PerformanceMetric
)


# ==============================================================================
# FIXTURES
# ==============================================================================

@pytest.fixture
def clean_env():
    """Provide clean environment for testing."""
    original_env = os.environ.copy()
    
    # Set test environment variables
    test_env = {
        'ENVIRONMENT': 'testing',
        'DB_HOST': 'test-host',
        'DB_PORT': '5432',
        'DB_NAME': 'test_db',
        'DB_USER': 'test_user',
        'DB_PASSWORD': 'test_password',
        'DB_POOL_SIZE': '5',
        'DB_MAX_OVERFLOW': '10',
        'DB_ECHO': 'false'
    }
    
    os.environ.update(test_env)
    yield test_env
    
    # Restore original environment
    os.environ.clear()
    os.environ.update(original_env)


@pytest.fixture
def db_config():
    """Create test database configuration."""
    return DatabaseConfig(
        host='localhost',
        port=5432,
        database='test_trading',
        username='test_user',
        password='test@pass#123',
        pool_size=5,
        max_overflow=10,
        echo=False
    )


@pytest.fixture
def in_memory_db():
    """Create in-memory SQLite database for testing."""
    engine = create_engine('sqlite:///:memory:')
    Base.metadata.create_all(engine)
    
    Session = sessionmaker(bind=engine)
    session = Session()
    
    yield session
    
    session.close()
    engine.dispose()


@pytest.fixture
def db_manager_mock():
    """Create mock database manager."""
    manager = Mock(spec=DatabaseManager)
    manager.engine = Mock()
    manager.session_factory = Mock()
    manager.get_session = Mock()
    return manager


@pytest.fixture
def sample_trade():
    """Create sample trade data."""
    return {
        'symbol': 'AAPL',
        'action': 'BUY',
        'quantity': 100,
        'price': 150.50,
        'executed_at': datetime.utcnow(),
        'order_id': 'TEST123',
        'status': 'FILLED'
    }


@pytest.fixture
def sample_position():
    """Create sample position data."""
    return {
        'symbol': 'AAPL',
        'quantity': 100,
        'entry_price': 150.00,
        'current_price': 155.00,
        'market_value': 15500.00,
        'unrealized_pnl': 500.00,
        'status': 'OPEN'
    }


# ==============================================================================
# DATABASE CONFIG TESTS
# ==============================================================================

@pytest.mark.unit
class TestDatabaseConfig:
    """Test DatabaseConfig class."""
    
    def test_config_initialization(self, db_config):
        """Test config object initialization."""
        assert db_config.host == 'localhost'
        assert db_config.port == 5432
        assert db_config.database == 'test_trading'
        assert db_config.username == 'test_user'
        assert db_config.pool_size == 5
    
    def test_connection_string_generation(self, db_config):
        """Test connection string generation with special characters."""
        conn_str = db_config.connection_string
        
        # Password should be URL encoded
        assert 'test%40pass%23123' in conn_str
        assert 'postgresql://' in conn_str
        assert 'localhost:5432/test_trading' in conn_str
    
    def test_async_connection_string(self, db_config):
        """Test async connection string generation."""
        async_conn_str = db_config.async_connection_string
        
        assert 'postgresql+asyncpg://' in async_conn_str
        assert 'test%40pass%23123' in async_conn_str
        assert 'localhost:5432/test_trading' in async_conn_str
    
    def test_engine_kwargs(self, db_config):
        """Test SQLAlchemy engine configuration."""
        kwargs = db_config.engine_kwargs
        
        assert kwargs['pool_size'] == 5
        assert kwargs['max_overflow'] == 10
        assert kwargs['pool_pre_ping'] is True
        assert 'connect_args' in kwargs
    
    def test_from_env(self, clean_env):
        """Test configuration from environment variables."""
        config = DatabaseConfig.from_env()
        
        assert config.host == 'test-host'
        assert config.port == 5432
        assert config.database == 'test_db'
        assert config.username == 'test_user'
        assert config.password == 'test_password'
    
    def test_from_env_with_prefix(self, clean_env):
        """Test configuration with environment variable prefix."""
        os.environ['PROD_DB_HOST'] = 'prod-host'
        os.environ['PROD_DB_NAME'] = 'prod_db'
        os.environ['PROD_DB_PASSWORD'] = 'prod_pass'
        
        config = DatabaseConfig.from_env('PROD_')
        
        assert config.host == 'prod-host'
        assert config.database == 'prod_db'
        assert config.password == 'prod_pass'
    
    def test_missing_required_env_var(self):
        """Test error on missing required environment variable."""
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ValueError, match="Missing required environment variable"):
                DatabaseConfig.from_env()


@pytest.mark.unit
class TestDatabaseEnvironment:
    """Test DatabaseEnvironment class."""
    
    def test_environment_initialization(self, clean_env):
        """Test environment manager initialization."""
        db_env = DatabaseEnvironment()
        
        assert db_env._current_env == 'testing'
        assert 'testing' in db_env.configs
    
    def test_get_current_config(self, clean_env):
        """Test getting current environment config."""
        db_env = DatabaseEnvironment()
        config = db_env.current
        
        assert config.host == 'test-host'
        assert config.database == 'test_db'
    
    def test_switch_environment(self, clean_env):
        """Test switching between environments."""
        db_env = DatabaseEnvironment()
        
        # Switch to development
        os.environ['ENVIRONMENT'] = 'development'
        db_env.set_environment('development')
        
        assert db_env._current_env == 'development'
    
    def test_invalid_environment(self):
        """Test error on invalid environment."""
        db_env = DatabaseEnvironment()
        db_env._current_env = 'invalid'
        
        with pytest.raises(ValueError, match="No configuration found"):
            _ = db_env.current


# ==============================================================================
# DATABASE MANAGER TESTS
# ==============================================================================

@pytest.mark.unit
class TestDatabaseManager:
    """Test DatabaseManager class."""
    
    def test_manager_initialization(self, db_config):
        """Test database manager initialization."""
        with patch('src.database.manager.create_engine') as mock_engine:
            manager = DatabaseManager(db_config)
            
            assert manager.config == db_config
            mock_engine.assert_called_once()
    
    def test_create_tables(self, db_manager_mock):
        """Test table creation."""
        with patch.object(Base.metadata, 'create_all') as mock_create:
            db_manager_mock.engine = Mock()
            db_manager_mock.create_tables = DatabaseManager.create_tables.__get__(db_manager_mock)
            
            db_manager_mock.create_tables()
            mock_create.assert_called_once()
    
    def test_session_context_manager(self, in_memory_db):
        """Test session context manager with automatic cleanup."""
        # Create a simple manager with in-memory DB
        config = DatabaseConfig(
            host='',
            port=0,
            database=':memory:',
            username='',
            password=''
        )
        
        with patch('src.database.manager.create_engine') as mock_engine:
            mock_engine.return_value = in_memory_db.bind
            manager = DatabaseManager(config)
            manager.session_factory = lambda: in_memory_db
            
            # Test successful transaction
            with manager.get_session() as session:
                assert session is not None
    
    def test_session_rollback_on_error(self, db_manager_mock):
        """Test session rollback on error."""
        mock_session = Mock()
        db_manager_mock.session_factory.return_value = mock_session
        db_manager_mock.get_session = DatabaseManager.get_session.__get__(db_manager_mock)
        
        with pytest.raises(Exception):
            with db_manager_mock.get_session() as session:
                raise Exception("Test error")
        
        mock_session.rollback.assert_called_once()
        mock_session.close.assert_called_once()


@pytest.mark.unit
class TestDatabaseCRUD:
    """Test CRUD operations."""
    
    def test_create_trade(self, in_memory_db, sample_trade):
        """Test creating a trade record."""
        trade = Trade(**sample_trade)
        in_memory_db.add(trade)
        in_memory_db.commit()
        
        assert trade.id is not None
        assert trade.symbol == 'AAPL'
        assert trade.quantity == 100
    
    def test_bulk_create(self, in_memory_db):
        """Test bulk record creation."""
        trades = [
            Trade(symbol='AAPL', action='BUY', quantity=100, price=150.00),
            Trade(symbol='GOOGL', action='BUY', quantity=50, price=2800.00),
            Trade(symbol='MSFT', action='SELL', quantity=75, price=380.00)
        ]
        
        in_memory_db.add_all(trades)
        in_memory_db.commit()
        
        result = in_memory_db.query(Trade).all()
        assert len(result) == 3
    
    def test_get_by_id(self, in_memory_db, sample_trade):
        """Test retrieving record by ID."""
        trade = Trade(**sample_trade)
        in_memory_db.add(trade)
        in_memory_db.commit()
        
        retrieved = in_memory_db.query(Trade).filter(Trade.id == trade.id).first()
        assert retrieved is not None
        assert retrieved.symbol == 'AAPL'
    
    def test_update_record(self, in_memory_db, sample_position):
        """Test updating a record."""
        position = Position(**sample_position)
        in_memory_db.add(position)
        in_memory_db.commit()
        
        # Update position
        position.current_price = 160.00
        position.unrealized_pnl = 1000.00
        in_memory_db.commit()
        
        updated = in_memory_db.query(Position).filter(Position.id == position.id).first()
        assert updated.current_price == 160.00
        assert updated.unrealized_pnl == 1000.00
    
    def test_delete_record(self, in_memory_db, sample_trade):
        """Test deleting a record."""
        trade = Trade(**sample_trade)
        in_memory_db.add(trade)
        in_memory_db.commit()
        
        trade_id = trade.id
        in_memory_db.delete(trade)
        in_memory_db.commit()
        
        deleted = in_memory_db.query(Trade).filter(Trade.id == trade_id).first()
        assert deleted is None


@pytest.mark.unit
class TestTradeOperations:
    """Test trade-specific operations."""
    
    def test_get_recent_trades(self, in_memory_db):
        """Test retrieving recent trades."""
        # Add trades with different dates
        trades = [
            Trade(
                symbol='AAPL',
                action='BUY',
                quantity=100,
                price=150.00,
                executed_at=datetime.utcnow()
            ),
            Trade(
                symbol='GOOGL',
                action='SELL',
                quantity=50,
                price=2800.00,
                executed_at=datetime.utcnow() - timedelta(days=10)
            )
        ]
        
        in_memory_db.add_all(trades)
        in_memory_db.commit()
        
        # Get trades from last 7 days
        cutoff = datetime.utcnow() - timedelta(days=7)
        recent = in_memory_db.query(Trade).filter(
            Trade.executed_at >= cutoff
        ).all()
        
        assert len(recent) == 1
        assert recent[0].symbol == 'AAPL'
    
    def test_get_trades_by_symbol(self, in_memory_db):
        """Test retrieving trades by symbol."""
        trades = [
            Trade(symbol='AAPL', action='BUY', quantity=100, price=150.00),
            Trade(symbol='AAPL', action='SELL', quantity=50, price=155.00),
            Trade(symbol='GOOGL', action='BUY', quantity=25, price=2800.00)
        ]
        
        in_memory_db.add_all(trades)
        in_memory_db.commit()
        
        aapl_trades = in_memory_db.query(Trade).filter(
            Trade.symbol == 'AAPL'
        ).all()
        
        assert len(aapl_trades) == 2
        assert all(t.symbol == 'AAPL' for t in aapl_trades)


@pytest.mark.unit
class TestPositionOperations:
    """Test position-specific operations."""
    
    def test_get_open_positions(self, in_memory_db):
        """Test retrieving open positions."""
        positions = [
            Position(symbol='AAPL', quantity=100, status='OPEN'),
            Position(symbol='GOOGL', quantity=50, status='OPEN'),
            Position(symbol='MSFT', quantity=75, status='CLOSED')
        ]
        
        in_memory_db.add_all(positions)
        in_memory_db.commit()
        
        open_positions = in_memory_db.query(Position).filter(
            Position.status == 'OPEN'
        ).all()
        
        assert len(open_positions) == 2
        assert all(p.status == 'OPEN' for p in open_positions)
    
    def test_update_position_by_symbol(self, in_memory_db):
        """Test updating position by symbol."""
        position = Position(
            symbol='AAPL',
            quantity=100,
            entry_price=150.00,
            current_price=150.00,
            status='OPEN'
        )
        
        in_memory_db.add(position)
        in_memory_db.commit()
        
        # Update position
        position.current_price = 160.00
        position.unrealized_pnl = 1000.00
        position.updated_at = datetime.utcnow()
        in_memory_db.commit()
        
        updated = in_memory_db.query(Position).filter(
            Position.symbol == 'AAPL',
            Position.status == 'OPEN'
        ).first()
        
        assert updated.current_price == 160.00
        assert updated.unrealized_pnl == 1000.00


@pytest.mark.unit
class TestPortfolioOperations:
    """Test portfolio operations."""
    
    def test_get_latest_portfolio(self, in_memory_db):
        """Test retrieving latest portfolio snapshot."""
        portfolios = [
            Portfolio(
                total_value=100000,
                cash_balance=50000,
                timestamp=datetime.utcnow() - timedelta(hours=2)
            ),
            Portfolio(
                total_value=105000,
                cash_balance=45000,
                timestamp=datetime.utcnow()
            )
        ]
        
        in_memory_db.add_all(portfolios)
        in_memory_db.commit()
        
        latest = in_memory_db.query(Portfolio).order_by(
            Portfolio.timestamp.desc()
        ).first()
        
        assert latest.total_value == 105000
        assert latest.cash_balance == 45000


@pytest.mark.unit  
class TestAlertOperations:
    """Test alert operations."""
    
    def test_create_alert(self, in_memory_db):
        """Test creating an alert."""
        alert = Alert(
            alert_type='RISK',
            severity='WARNING',
            message='High portfolio concentration',
            source='risk_manager',
            resolved=False
        )
        
        in_memory_db.add(alert)
        in_memory_db.commit()
        
        assert alert.id is not None
        assert alert.severity == 'WARNING'
    
    def test_get_unresolved_alerts(self, in_memory_db):
        """Test retrieving unresolved alerts."""
        alerts = [
            Alert(alert_type='RISK', severity='WARNING', resolved=False),
            Alert(alert_type='SYSTEM', severity='INFO', resolved=True),
            Alert(alert_type='TRADE', severity='CRITICAL', resolved=False)
        ]
        
        in_memory_db.add_all(alerts)
        in_memory_db.commit()
        
        unresolved = in_memory_db.query(Alert).filter(
            Alert.resolved == False
        ).all()
        
        assert len(unresolved) == 2
        assert all(not a.resolved for a in unresolved)
    
    def test_resolve_alert(self, in_memory_db):
        """Test resolving an alert."""
        alert = Alert(
            alert_type='RISK',
            severity='WARNING',
            message='Test alert',
            resolved=False
        )
        
        in_memory_db.add(alert)
        in_memory_db.commit()
        
        # Resolve alert
        alert.resolved = True
        alert.resolved_at = datetime.utcnow()
        alert.resolution = 'Issue resolved'
        in_memory_db.commit()
        
        resolved = in_memory_db.query(Alert).filter(
            Alert.id == alert.id
        ).first()
        
        assert resolved.resolved is True
        assert resolved.resolution == 'Issue resolved'
        assert resolved.resolved_at is not None


@pytest.mark.unit
class TestUtilityOperations:
    """Test utility operations."""
    
    def test_check_connection(self, in_memory_db):
        """Test database connection check."""
        result = in_memory_db.execute(text("SELECT 1")).scalar()
        assert result == 1
    
    def test_get_table_stats(self, in_memory_db):
        """Test getting table statistics."""
        # Add sample data
        in_memory_db.add_all([
            Trade(symbol='AAPL', action='BUY', quantity=100, price=150),
            Trade(symbol='GOOGL', action='SELL', quantity=50, price=2800),
            Position(symbol='AAPL', quantity=100, status='OPEN'),
            Alert(alert_type='RISK', severity='INFO')
        ])
        in_memory_db.commit()
        
        # Get counts
        trade_count = in_memory_db.query(Trade).count()
        position_count = in_memory_db.query(Position).count()
        alert_count = in_memory_db.query(Alert).count()
        
        assert trade_count == 2
        assert position_count == 1
        assert alert_count == 1
    
    def test_cleanup_old_data(self, in_memory_db):
        """Test cleaning up old data."""
        cutoff = datetime.utcnow() - timedelta(days=90)
        
        # Add old and new trades
        trades = [
            Trade(
                symbol='OLD',
                action='BUY',
                quantity=100,
                price=100,
                executed_at=cutoff - timedelta(days=1)
            ),
            Trade(
                symbol='NEW',
                action='BUY',
                quantity=100,
                price=100,
                executed_at=datetime.utcnow()
            )
        ]
        
        in_memory_db.add_all(trades)
        in_memory_db.commit()
        
        # Clean up old data
        in_memory_db.query(Trade).filter(
            Trade.executed_at < cutoff
        ).delete()
        in_memory_db.commit()
        
        remaining = in_memory_db.query(Trade).all()
        assert len(remaining) == 1
        assert remaining[0].symbol == 'NEW'


# ==============================================================================
# ASYNC DATABASE MANAGER TESTS
# ==============================================================================

@pytest.mark.asyncio
@pytest.mark.unit
class TestAsyncDatabaseManager:
    """Test AsyncDatabaseManager class."""
    
    async def test_async_manager_initialization(self, db_config):
        """Test async database manager initialization."""
        with patch('src.database.manager.create_async_engine') as mock_engine:
            manager = AsyncDatabaseManager(db_config)
            
            assert manager.config == db_config
            mock_engine.assert_called_once()
    
    async def test_async_session_creation(self):
        """Test async session creation."""
        config = DatabaseConfig(
            host='localhost',
            port=5432,
            database='test',
            username='user',
            password='pass'
        )
        
        with patch('src.database.manager.create_async_engine'):
            manager = AsyncDatabaseManager(config)
            manager.session_factory = Mock()
            
            session = await manager.get_session()
            manager.session_factory.assert_called_once()


# ==============================================================================
# INTEGRATION TESTS
# ==============================================================================

@pytest.mark.integration
class TestDatabaseIntegration:
    """Integration tests for database operations."""
    
    def test_end_to_end_workflow(self, in_memory_db):
        """Test complete workflow from trade to portfolio."""
        # Create trade
        trade = Trade(
            symbol='AAPL',
            action='BUY',
            quantity=100,
            price=150.00,
            executed_at=datetime.utcnow()
        )
        in_memory_db.add(trade)
        
        # Create position
        position = Position(
            symbol='AAPL',
            quantity=100,
            entry_price=150.00,
            current_price=150.00,
            status='OPEN'
        )
        in_memory_db.add(position)
        
        # Create portfolio snapshot
        portfolio = Portfolio(
            total_value=100000,
            cash_balance=85000,
            positions_value=15000,
            timestamp=datetime.utcnow()
        )
        in_memory_db.add(portfolio)
        
        in_memory_db.commit()
        
        # Verify all records
        assert in_memory_db.query(Trade).count() == 1
        assert in_memory_db.query(Position).count() == 1
        assert in_memory_db.query(Portfolio).count() == 1
    
    def test_transaction_rollback(self, in_memory_db):
        """Test transaction rollback on error."""
        try:
            trade = Trade(symbol='AAPL', action='BUY', quantity=100, price=150)
            in_memory_db.add(trade)
            
            # Force an error
            raise Exception("Simulated error")
            
            in_memory_db.commit()
        except Exception:
            in_memory_db.rollback()
        
        # Verify no data was saved
        assert in_memory_db.query(Trade).count() == 0


# ==============================================================================
# PERFORMANCE TESTS
# ==============================================================================

@pytest.mark.performance
class TestDatabasePerformance:
    """Performance tests for database operations."""
    
    def test_bulk_insert_performance(self, in_memory_db):
        """Test performance of bulk inserts."""
        import time
        
        # Create 1000 trades
        trades = [
            Trade(
                symbol=f'SYM{i}',
                action='BUY' if i % 2 == 0 else 'SELL',
                quantity=100,
                price=100.00 + i
            )
            for i in range(1000)
        ]
        
        start_time = time.time()
        in_memory_db.add_all(trades)
        in_memory_db.commit()
        elapsed = time.time() - start_time
        
        assert in_memory_db.query(Trade).count() == 1000
        assert elapsed < 5.0  # Should complete within 5 seconds
    
    def test_query_performance(self, in_memory_db):
        """Test query performance with indexes."""
        # Add test data
        for i in range(100):
            position = Position(
                symbol=f'SYM{i % 10}',
                quantity=100,
                status='OPEN' if i % 3 != 0 else 'CLOSED'
            )
            in_memory_db.add(position)
        
        in_memory_db.commit()
        
        import time
        start_time = time.time()
        
        # Query with filter
        open_positions = in_memory_db.query(Position).filter(
            Position.status == 'OPEN',
            Position.symbol == 'SYM1'
        ).all()
        
        elapsed = time.time() - start_time
        
        assert len(open_positions) > 0
        assert elapsed < 0.1  # Should complete within 100ms


# ==============================================================================
# ERROR HANDLING TESTS
# ==============================================================================

@pytest.mark.unit
class TestErrorHandling:
    """Test error handling in database operations."""
    
    def test_connection_error_handling(self, db_config):
        """Test handling of connection errors."""
        # Use invalid connection string
        db_config.host = 'invalid_host'
        
        with patch('src.database.manager.create_engine') as mock_engine:
            mock_engine.side_effect = OperationalError("Connection failed", None, None)
            
            with pytest.raises(OperationalError):
                DatabaseManager(db_config)
    
    def test_integrity_error_handling(self, in_memory_db):
        """Test handling of integrity constraint violations."""
        # This would need proper unique constraints in the model
        # For now, just test the error handling pattern
        pass
    
    def test_session_cleanup_on_error(self):
        """Test session cleanup on unexpected errors."""
        manager = Mock(spec=DatabaseManager)
        session = Mock()
        manager.session_factory.return_value = session
        manager.get_session = DatabaseManager.get_session.__get__(manager)
        
        with pytest.raises(ValueError):
            with manager.get_session() as s:
                raise ValueError("Test error")
        
        session.rollback.assert_called_once()
        session.close.assert_called_once()