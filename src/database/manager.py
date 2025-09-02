"""
Database manager for the AI Trading System.

This module provides high-level database operations, session management,
and common database utilities.
"""

import logging
from contextlib import contextmanager
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Type, TypeVar, Union
from sqlalchemy import create_engine, text, and_, or_, func, desc
from sqlalchemy.orm import Session, sessionmaker, scoped_session
from sqlalchemy.pool import NullPool, QueuePool
from sqlalchemy.exc import SQLAlchemyError, IntegrityError, OperationalError
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
import asyncio

from src.database.config import get_database_config, get_connection_string, get_async_connection_string
from src.database.models import (
    Base, Trade, Position, Portfolio, Signal, 
    AgentAction, Alert, PerformanceMetric
)

logger = logging.getLogger(__name__)

T = TypeVar('T')


class DatabaseManager:
    """Manages database connections and operations."""
    
    def __init__(self, config=None):
        """Initialize database manager.
        
        Args:
            config: Optional DatabaseConfig instance
        """
        self.config = config or get_database_config()
        self.engine = None
        self.session_factory = None
        self.scoped_session = None
        self._initialize_engine()
    
    def _initialize_engine(self):
        """Initialize SQLAlchemy engine and session factory."""
        try:
            # Create engine with configuration
            self.engine = create_engine(
                self.config.connection_string,
                **self.config.engine_kwargs
            )
            
            # Create session factory
            self.session_factory = sessionmaker(
                bind=self.engine,
                expire_on_commit=False
            )
            
            # Create scoped session for thread-local sessions
            self.scoped_session = scoped_session(self.session_factory)
            
            logger.info("Database engine initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize database engine: {e}")
            raise
    
    def create_tables(self):
        """Create all database tables."""
        try:
            Base.metadata.create_all(self.engine)
            logger.info("Database tables created successfully")
        except Exception as e:
            logger.error(f"Failed to create database tables: {e}")
            raise
    
    def drop_tables(self):
        """Drop all database tables."""
        try:
            Base.metadata.drop_all(self.engine)
            logger.info("Database tables dropped successfully")
        except Exception as e:
            logger.error(f"Failed to drop database tables: {e}")
            raise
    
    def reset_database(self):
        """Reset database by dropping and recreating all tables."""
        self.drop_tables()
        self.create_tables()
        logger.info("Database reset completed")
    
    @contextmanager
    def get_session(self):
        """Get a database session with automatic cleanup.
        
        Yields:
            Session instance
        """
        session = self.session_factory()
        try:
            yield session
            session.commit()
        except Exception as e:
            session.rollback()
            logger.error(f"Database session error: {e}")
            raise
        finally:
            session.close()
    
    def get_scoped_session(self) -> Session:
        """Get a thread-local scoped session.
        
        Returns:
            Scoped session instance
        """
        return self.scoped_session()
    
    def close_scoped_session(self):
        """Close and remove the current scoped session."""
        self.scoped_session.remove()
    
    # ============= CRUD Operations =============
    
    def create(self, obj: Base) -> Base:
        """Create a new database record.
        
        Args:
            obj: SQLAlchemy model instance
        
        Returns:
            Created object with ID populated
        """
        with self.get_session() as session:
            session.add(obj)
            session.flush()
            session.refresh(obj)
            return obj
    
    def bulk_create(self, objects: List[Base]) -> List[Base]:
        """Create multiple database records.
        
        Args:
            objects: List of SQLAlchemy model instances
        
        Returns:
            List of created objects
        """
        with self.get_session() as session:
            session.add_all(objects)
            session.flush()
            for obj in objects:
                session.refresh(obj)
            return objects
    
    def get(self, model: Type[T], id: Any) -> Optional[T]:
        """Get a record by ID.
        
        Args:
            model: SQLAlchemy model class
            id: Primary key value
        
        Returns:
            Model instance or None
        """
        with self.get_session() as session:
            return session.query(model).filter(model.id == id).first()
    
    def get_all(self, model: Type[T], limit: Optional[int] = None) -> List[T]:
        """Get all records of a model type.
        
        Args:
            model: SQLAlchemy model class
            limit: Optional limit on number of records
        
        Returns:
            List of model instances
        """
        with self.get_session() as session:
            query = session.query(model)
            if limit:
                query = query.limit(limit)
            return query.all()
    
    def update(self, obj: Base) -> Base:
        """Update an existing database record.
        
        Args:
            obj: SQLAlchemy model instance with updates
        
        Returns:
            Updated object
        """
        with self.get_session() as session:
            session.merge(obj)
            session.flush()
            session.refresh(obj)
            return obj
    
    def delete(self, obj: Base) -> bool:
        """Delete a database record.
        
        Args:
            obj: SQLAlchemy model instance to delete
        
        Returns:
            True if successful
        """
        with self.get_session() as session:
            session.delete(obj)
            return True
    
    # ============= Trade Operations =============
    
    def create_trade(self, trade_data: Dict) -> Trade:
        """Create a new trade record.
        
        Args:
            trade_data: Dictionary with trade information
        
        Returns:
            Created Trade instance
        """
        trade = Trade(**trade_data)
        return self.create(trade)
    
    def get_recent_trades(self, limit: int = 100, days: int = 7) -> List[Trade]:
        """Get recent trades.
        
        Args:
            limit: Maximum number of trades to return
            days: Number of days to look back
        
        Returns:
            List of Trade instances
        """
        cutoff_date = datetime.utcnow() - timedelta(days=days)
        
        with self.get_session() as session:
            return session.query(Trade).filter(
                Trade.executed_at >= cutoff_date
            ).order_by(desc(Trade.executed_at)).limit(limit).all()
    
    def get_trades_by_symbol(self, symbol: str, limit: Optional[int] = None) -> List[Trade]:
        """Get trades for a specific symbol.
        
        Args:
            symbol: Stock symbol
            limit: Optional limit on number of trades
        
        Returns:
            List of Trade instances
        """
        with self.get_session() as session:
            query = session.query(Trade).filter(Trade.symbol == symbol)
            query = query.order_by(desc(Trade.executed_at))
            if limit:
                query = query.limit(limit)
            return query.all()
    
    # ============= Position Operations =============
    
    def get_open_positions(self) -> List[Position]:
        """Get all open positions.
        
        Returns:
            List of open Position instances
        """
        with self.get_session() as session:
            return session.query(Position).filter(
                Position.status == 'OPEN'
            ).all()
    
    def get_position_by_symbol(self, symbol: str) -> Optional[Position]:
        """Get position for a specific symbol.
        
        Args:
            symbol: Stock symbol
        
        Returns:
            Position instance or None
        """
        with self.get_session() as session:
            return session.query(Position).filter(
                and_(Position.symbol == symbol, Position.status == 'OPEN')
            ).first()
    
    def update_position(self, symbol: str, updates: Dict) -> Optional[Position]:
        """Update a position.
        
        Args:
            symbol: Stock symbol
            updates: Dictionary with fields to update
        
        Returns:
            Updated Position instance or None
        """
        with self.get_session() as session:
            position = session.query(Position).filter(
                and_(Position.symbol == symbol, Position.status == 'OPEN')
            ).first()
            
            if position:
                for key, value in updates.items():
                    setattr(position, key, value)
                position.updated_at = datetime.utcnow()
                session.flush()
                session.refresh(position)
                return position
            return None
    
    # ============= Portfolio Operations =============
    
    def get_latest_portfolio(self) -> Optional[Portfolio]:
        """Get the most recent portfolio snapshot.
        
        Returns:
            Latest Portfolio instance or None
        """
        with self.get_session() as session:
            return session.query(Portfolio).order_by(
                desc(Portfolio.timestamp)
            ).first()
    
    def create_portfolio_snapshot(self, portfolio_data: Dict) -> Portfolio:
        """Create a new portfolio snapshot.
        
        Args:
            portfolio_data: Dictionary with portfolio information
        
        Returns:
            Created Portfolio instance
        """
        portfolio = Portfolio(**portfolio_data)
        return self.create(portfolio)
    
    def get_portfolio_history(self, days: int = 30) -> List[Portfolio]:
        """Get portfolio history.
        
        Args:
            days: Number of days to look back
        
        Returns:
            List of Portfolio instances
        """
        cutoff_date = datetime.utcnow() - timedelta(days=days)
        
        with self.get_session() as session:
            return session.query(Portfolio).filter(
                Portfolio.timestamp >= cutoff_date
            ).order_by(Portfolio.timestamp).all()
    
    # ============= Signal Operations =============
    
    def create_signal(self, signal_data: Dict) -> Signal:
        """Create a new trading signal.
        
        Args:
            signal_data: Dictionary with signal information
        
        Returns:
            Created Signal instance
        """
        signal = Signal(**signal_data)
        return self.create(signal)
    
    def get_recent_signals(self, source: Optional[str] = None, limit: int = 50) -> List[Signal]:
        """Get recent trading signals.
        
        Args:
            source: Optional filter by signal source
            limit: Maximum number of signals to return
        
        Returns:
            List of Signal instances
        """
        with self.get_session() as session:
            query = session.query(Signal)
            if source:
                query = query.filter(Signal.source == source)
            return query.order_by(desc(Signal.timestamp)).limit(limit).all()
    
    # ============= Agent Action Operations =============
    
    def log_agent_action(self, action_data: Dict) -> AgentAction:
        """Log an agent action.
        
        Args:
            action_data: Dictionary with action information
        
        Returns:
            Created AgentAction instance
        """
        action = AgentAction(**action_data)
        return self.create(action)
    
    def get_agent_actions(self, agent_name: str, limit: int = 100) -> List[AgentAction]:
        """Get actions for a specific agent.
        
        Args:
            agent_name: Name of the agent
            limit: Maximum number of actions to return
        
        Returns:
            List of AgentAction instances
        """
        with self.get_session() as session:
            return session.query(AgentAction).filter(
                AgentAction.agent_name == agent_name
            ).order_by(desc(AgentAction.timestamp)).limit(limit).all()
    
    # ============= Alert Operations =============
    
    def create_alert(self, alert_data: Dict) -> Alert:
        """Create a new alert.
        
        Args:
            alert_data: Dictionary with alert information
        
        Returns:
            Created Alert instance
        """
        alert = Alert(**alert_data)
        return self.create(alert)
    
    def get_unresolved_alerts(self, severity: Optional[str] = None) -> List[Alert]:
        """Get unresolved alerts.
        
        Args:
            severity: Optional filter by severity level
        
        Returns:
            List of Alert instances
        """
        with self.get_session() as session:
            query = session.query(Alert).filter(Alert.resolved == False)
            if severity:
                query = query.filter(Alert.severity == severity)
            return query.order_by(desc(Alert.created_at)).all()
    
    def resolve_alert(self, alert_id: int, resolution: str) -> Optional[Alert]:
        """Resolve an alert.
        
        Args:
            alert_id: Alert ID
            resolution: Resolution description
        
        Returns:
            Updated Alert instance or None
        """
        with self.get_session() as session:
            alert = session.query(Alert).filter(Alert.id == alert_id).first()
            if alert:
                alert.resolved = True
                alert.resolved_at = datetime.utcnow()
                alert.resolution = resolution
                session.flush()
                session.refresh(alert)
                return alert
            return None
    
    # ============= Performance Operations =============
    
    def record_performance_metric(self, metric_data: Dict) -> PerformanceMetric:
        """Record a performance metric.
        
        Args:
            metric_data: Dictionary with metric information
        
        Returns:
            Created PerformanceMetric instance
        """
        metric = PerformanceMetric(**metric_data)
        return self.create(metric)
    
    def get_performance_metrics(self, metric_name: Optional[str] = None, 
                               days: int = 30) -> List[PerformanceMetric]:
        """Get performance metrics.
        
        Args:
            metric_name: Optional filter by metric name
            days: Number of days to look back
        
        Returns:
            List of PerformanceMetric instances
        """
        cutoff_date = datetime.utcnow() - timedelta(days=days)
        
        with self.get_session() as session:
            query = session.query(PerformanceMetric).filter(
                PerformanceMetric.timestamp >= cutoff_date
            )
            if metric_name:
                query = query.filter(PerformanceMetric.metric_name == metric_name)
            return query.order_by(PerformanceMetric.timestamp).all()
    
    # ============= Utility Operations =============
    
    def execute_raw_sql(self, sql: str, params: Optional[Dict] = None) -> Any:
        """Execute raw SQL query.
        
        Args:
            sql: SQL query string
            params: Optional parameters for the query
        
        Returns:
            Query result
        """
        with self.get_session() as session:
            result = session.execute(text(sql), params or {})
            return result.fetchall()
    
    def check_connection(self) -> bool:
        """Check database connection.
        
        Returns:
            True if connection is successful
        """
        try:
            with self.get_session() as session:
                session.execute(text("SELECT 1"))
            return True
        except Exception as e:
            logger.error(f"Database connection check failed: {e}")
            return False
    
    def get_table_stats(self) -> Dict[str, int]:
        """Get row counts for all tables.
        
        Returns:
            Dictionary with table names and row counts
        """
        stats = {}
        tables = [Trade, Position, Portfolio, Signal, AgentAction, Alert, PerformanceMetric]
        
        with self.get_session() as session:
            for table in tables:
                count = session.query(func.count(table.id)).scalar()
                stats[table.__tablename__] = count
        
        return stats
    
    def cleanup_old_data(self, days: int = 90):
        """Clean up old data from tables.
        
        Args:
            days: Number of days of data to keep
        """
        cutoff_date = datetime.utcnow() - timedelta(days=days)
        
        with self.get_session() as session:
            # Clean up old trades
            session.query(Trade).filter(Trade.executed_at < cutoff_date).delete()
            
            # Clean up old signals
            session.query(Signal).filter(Signal.timestamp < cutoff_date).delete()
            
            # Clean up old agent actions
            session.query(AgentAction).filter(AgentAction.timestamp < cutoff_date).delete()
            
            # Clean up old resolved alerts
            session.query(Alert).filter(
                and_(Alert.resolved == True, Alert.resolved_at < cutoff_date)
            ).delete()
            
            # Clean up old performance metrics
            session.query(PerformanceMetric).filter(
                PerformanceMetric.timestamp < cutoff_date
            ).delete()
            
            session.commit()
            logger.info(f"Cleaned up data older than {days} days")
    
    def close(self):
        """Close database connections."""
        if self.scoped_session:
            self.scoped_session.remove()
        if self.engine:
            self.engine.dispose()
        logger.info("Database connections closed")


class AsyncDatabaseManager:
    """Async database manager for concurrent operations."""
    
    def __init__(self, config=None):
        """Initialize async database manager.
        
        Args:
            config: Optional DatabaseConfig instance
        """
        self.config = config or get_database_config()
        self.engine = None
        self.session_factory = None
        self._initialize_engine()
    
    def _initialize_engine(self):
        """Initialize async SQLAlchemy engine."""
        self.engine = create_async_engine(
            get_async_connection_string(),
            echo=self.config.echo,
            pool_size=self.config.pool_size,
            max_overflow=self.config.max_overflow
        )
        
        self.session_factory = async_sessionmaker(
            self.engine,
            class_=AsyncSession,
            expire_on_commit=False
        )
    
    async def get_session(self) -> AsyncSession:
        """Get an async database session.
        
        Returns:
            AsyncSession instance
        """
        async with self.session_factory() as session:
            return session
    
    async def create_tables(self):
        """Create all database tables asynchronously."""
        async with self.engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
    
    async def drop_tables(self):
        """Drop all database tables asynchronously."""
        async with self.engine.begin() as conn:
            await conn.run_sync(Base.metadata.drop_all)
    
    async def close(self):
        """Close async database connections."""
        await self.engine.dispose()


# Global database manager instance
_db_manager: Optional[DatabaseManager] = None


def get_db_manager() -> DatabaseManager:
    """Get or create global database manager instance.
    
    Returns:
        DatabaseManager instance
    """
    global _db_manager
    if _db_manager is None:
        _db_manager = DatabaseManager()
    return _db_manager


def init_database():
    """Initialize database and create tables."""
    manager = get_db_manager()
    manager.create_tables()
    logger.info("Database initialized successfully")


def reset_database():
    """Reset database by dropping and recreating all tables."""
    manager = get_db_manager()
    manager.reset_database()
    logger.info("Database reset successfully")