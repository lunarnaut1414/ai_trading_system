# Database Implementation Documentation

## Overview

The AI Trading System database layer provides a robust, scalable foundation for storing and managing trading data, agent decisions, portfolio state, and system metrics. Built on SQLAlchemy 2.0 with support for both synchronous and asynchronous operations, the implementation ensures data integrity, performance, and maintainability.

## Architecture

### Core Components

```
src/database/
├── config.py           # Database configuration and environment management
├── manager.py          # Database connection and session management
└── models/
    ├── __init__.py     # Model exports and initialization
    ├── base.py         # Base model and mixins
    ├── trades.py       # Trade execution models
    ├── portfolio.py    # Portfolio and position models
    ├── signals.py      # Trading signal models
    ├── agents.py       # Agent action and decision models
    ├── alerts.py       # System alert models
    └── performance.py  # Performance metric models
```

## Configuration

### DatabaseConfig Class

Manages database connection parameters with support for multiple environments:

```python
from src.database.config import DatabaseConfig

# Create configuration from environment variables
config = DatabaseConfig.from_env()

# Or create manually
config = DatabaseConfig(
    host='localhost',
    port=5432,
    database='trading_system',
    username='trader',
    password='secure_password',
    pool_size=10,
    max_overflow=20
)
```

### Environment Management

The `DatabaseEnvironment` class handles multiple database configurations:

- **Development**: Local PostgreSQL with verbose logging
- **Testing**: In-memory SQLite for fast test execution
- **Production**: PostgreSQL with connection pooling and optimizations

```python
from src.database.config import DatabaseEnvironment

db_env = DatabaseEnvironment()
db_env.set_environment('production')
config = db_env.current
```

### Connection Strings

Automatic URL encoding for special characters in passwords:

```python
# PostgreSQL
postgresql://user:encoded%40password@host:5432/database

# Async PostgreSQL with asyncpg
postgresql+asyncpg://user:encoded%40password@host:5432/database
```

## Database Models

### Base Model and Mixins

All models inherit from a common base with automatic timestamp tracking:

```python
from sqlalchemy.orm import declarative_base
from sqlalchemy import Column, DateTime, func

Base = declarative_base()

class TimestampMixin:
    """Automatic created_at and updated_at timestamps"""
    created_at = Column(DateTime, default=func.now(), nullable=False)
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now(), nullable=False)
```

### Trade Model

Tracks all executed trades with complete audit trail:

```python
class Trade(Base, TimestampMixin):
    __tablename__ = 'trades'
    
    id = Column(Integer, primary_key=True)
    symbol = Column(String(10), nullable=False)
    action = Column(String(10), nullable=False)  # BUY/SELL
    quantity = Column(Integer, nullable=False)
    price = Column(Float, nullable=False)
    executed_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    order_id = Column(String(100), unique=True)
    status = Column(Enum(TradeStatus), default=TradeStatus.PENDING)
```

### Position Model

Represents current holdings with P&L tracking:

```python
class Position(Base, TimestampMixin):
    __tablename__ = 'positions'
    
    id = Column(Integer, primary_key=True)
    symbol = Column(String(10), nullable=False)
    quantity = Column(Integer, nullable=False)
    entry_price = Column(Float, nullable=False, default=0.0)
    current_price = Column(Float, default=0.0)
    market_value = Column(Float, default=0.0)
    unrealized_pnl = Column(Float, default=0.0)
    status = Column(Enum(PositionStatus), default=PositionStatus.OPEN)
```

### Portfolio Model

Snapshots of overall portfolio state:

```python
class Portfolio(Base, TimestampMixin):
    __tablename__ = 'portfolios'
    
    id = Column(Integer, primary_key=True)
    total_value = Column(Float, nullable=False)
    cash_balance = Column(Float, nullable=False)
    timestamp = Column(DateTime, nullable=False, default=datetime.utcnow)
```

### Alert Model

System alerts and risk warnings:

```python
class Alert(Base, TimestampMixin):
    __tablename__ = 'alerts'
    
    id = Column(Integer, primary_key=True)
    alert_type = Column(String(50), nullable=False)
    severity = Column(String(20), nullable=False)  # INFO/WARNING/CRITICAL/EMERGENCY
    message = Column(Text, nullable=False, default='')
    source = Column(String(50))
    resolved = Column(Boolean, default=False)
    timestamp = Column(DateTime, nullable=False, default=datetime.utcnow)
```

## Database Manager

### Synchronous Operations

The `DatabaseManager` class provides high-level database operations:

```python
from src.database.manager import DatabaseManager

manager = DatabaseManager(config)

# Session management with automatic cleanup
with manager.get_session() as session:
    trade = Trade(symbol='AAPL', action='BUY', quantity=100, price=150.00)
    session.add(trade)
    # Automatically commits on success, rolls back on error

# CRUD operations
trade = manager.create(trade_obj)
trades = manager.get_all(Trade, limit=100)
trade = manager.get(Trade, trade_id)
manager.update(trade_obj)
manager.delete(trade_obj)
```

### Asynchronous Operations

The `AsyncDatabaseManager` supports async/await patterns:

```python
from src.database.manager import AsyncDatabaseManager

async_manager = AsyncDatabaseManager(config)

async with async_manager.get_session() as session:
    result = await session.execute(select(Trade).where(Trade.symbol == 'AAPL'))
    trades = result.scalars().all()
```

### Specialized Operations

Domain-specific methods for common queries:

```python
# Trade operations
recent_trades = manager.get_recent_trades(limit=100, days=7)
symbol_trades = manager.get_trades_by_symbol('AAPL')

# Position operations
open_positions = manager.get_open_positions()
position = manager.get_position_by_symbol('AAPL')
manager.update_position('AAPL', {'current_price': 155.00})

# Portfolio operations
latest_portfolio = manager.get_latest_portfolio()
portfolio_history = manager.get_portfolio_history(days=30)

# Alert operations
unresolved_alerts = manager.get_unresolved_alerts()
manager.resolve_alert(alert_id, resolution='Issue fixed')
```

## Testing

### Test Structure

```
tests/unit/database/
└── test_database.py    # Comprehensive database test suite
```

### Test Categories

#### Configuration Tests
- Environment variable parsing
- Connection string generation
- URL encoding for special characters
- Multi-environment support

#### CRUD Operations Tests
- Create, read, update, delete operations
- Bulk operations
- Transaction management
- Session cleanup

#### Domain-Specific Tests
- Trade operations
- Position management
- Portfolio snapshots
- Alert handling

#### Performance Tests
- Bulk insert performance (1000+ records)
- Query optimization
- Index effectiveness

#### Error Handling Tests
- Connection failures
- Constraint violations
- Transaction rollback
- Session cleanup on errors

### Test Fixtures

```python
@pytest.fixture
def in_memory_db():
    """In-memory SQLite database for testing"""
    engine = create_engine('sqlite:///:memory:')
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    session = Session()
    yield session
    session.close()
    engine.dispose()

@pytest.fixture
def sample_trade():
    """Sample trade data for testing"""
    return {
        'symbol': 'AAPL',
        'action': 'BUY',
        'quantity': 100,
        'price': 150.50,
        'executed_at': datetime.utcnow()
    }
```

### Running Tests

```bash
# Run all database tests
pytest tests/unit/database/test_database.py -v

# Run specific test class
pytest tests/unit/database/test_database.py::TestDatabaseCRUD -v

# Run with coverage
pytest tests/unit/database/test_database.py --cov=src.database --cov-report=html

# Run performance tests only
pytest tests/unit/database/test_database.py -m slow
```

### Test Results

Current test coverage: **40/40 tests passing** ✅

- Configuration: 11 tests
- Manager operations: 4 tests
- CRUD operations: 5 tests
- Trade operations: 2 tests
- Position operations: 2 tests
- Portfolio operations: 1 test
- Alert operations: 3 tests
- Utility operations: 3 tests
- Async operations: 2 tests
- Integration tests: 2 tests
- Performance tests: 2 tests
- Error handling: 3 tests

## Best Practices

### 1. Session Management
Always use context managers for automatic cleanup:
```python
with manager.get_session() as session:
    # Operations here
```

### 2. Enum Usage
Use enums for status fields to ensure data integrity:
```python
from src.database.models.portfolio import PositionStatus
position.status = PositionStatus.OPEN  # Not 'OPEN' string
```

### 3. Default Values
Define sensible defaults for nullable fields:
```python
entry_price = Column(Float, nullable=False, default=0.0)
timestamp = Column(DateTime, nullable=False, default=datetime.utcnow)
```

### 4. Transaction Patterns
Group related operations in transactions:
```python
with manager.get_session() as session:
    trade = Trade(...)
    position = Position(...)
    session.add_all([trade, position])
    # Both committed together or rolled back
```

### 5. Query Optimization
Use indexes and eager loading for performance:
```python
# Filter with indexed columns
trades = session.query(Trade).filter(
    Trade.symbol == 'AAPL',
    Trade.executed_at >= cutoff_date
).all()
```

## Migration Strategy

### Schema Versioning
Use Alembic for database migrations:

```bash
# Initialize Alembic
alembic init migrations

# Create migration
alembic revision --autogenerate -m "Add new field"

# Apply migrations
alembic upgrade head
```

### Backward Compatibility
- Add new columns as nullable initially
- Use default values for new required fields
- Create indexes concurrently in production

## Performance Considerations

### Connection Pooling
```python
config = DatabaseConfig(
    pool_size=10,        # Number of persistent connections
    max_overflow=20,     # Maximum overflow connections
    pool_timeout=30,     # Timeout for getting connection
    pool_recycle=3600    # Recycle connections after 1 hour
)
```

### Query Optimization
- Use appropriate indexes on frequently queried columns
- Implement pagination for large result sets
- Use bulk operations for multiple inserts/updates
- Consider read replicas for reporting queries

### Monitoring
Track key metrics:
- Connection pool usage
- Query execution time
- Transaction duration
- Lock contention

## Security

### Connection Security
- Use SSL/TLS for database connections
- Store credentials in environment variables
- Rotate passwords regularly
- Use connection encryption in production

### SQL Injection Prevention
SQLAlchemy's ORM provides automatic parameterization:
```python
# Safe - uses parameterized queries
session.query(Trade).filter(Trade.symbol == user_input)

# Never use string formatting for queries
# BAD: f"SELECT * FROM trades WHERE symbol = '{user_input}'"
```

### Access Control
- Use database roles with minimum required privileges
- Separate read/write connections where appropriate
- Audit database access and modifications

## Troubleshooting

### Common Issues

1. **Import Error: Cannot import name 'Base'**
   - Ensure `src/database/models/__init__.py` exports Base
   - Check that `base.py` defines Base correctly

2. **NOT NULL constraint failed**
   - Verify all required fields have values or defaults
   - Check model definitions match database schema

3. **Enum comparison failures**
   - Use enum values, not strings: `PositionStatus.OPEN`
   - Import enums from model modules

4. **Connection pool exhausted**
   - Ensure sessions are properly closed
   - Increase pool_size if necessary
   - Check for connection leaks

## Future Enhancements

### Planned Features
- [ ] Time-series optimized storage for tick data
- [ ] Partitioning for historical data
- [ ] Read replica support for analytics
- [ ] Database sharding for scale
- [ ] Real-time change data capture (CDC)
- [ ] Automated backup and recovery

### Performance Optimizations
- [ ] Query result caching with Redis
- [ ] Materialized views for complex aggregations
- [ ] Columnar storage for analytics
- [ ] Database connection multiplexing

## References

- [SQLAlchemy 2.0 Documentation](https://docs.sqlalchemy.org/en/20/)
- [PostgreSQL Best Practices](https://www.postgresql.org/docs/current/index.html)
- [Alembic Migration Guide](https://alembic.sqlalchemy.org/en/latest/)
- [Python Database Testing](https://docs.pytest.org/en/stable/)

---

*Last Updated: September 2025*
*Version: 1.0.0*