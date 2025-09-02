"""
Portfolio and Position models for the database.
"""

from sqlalchemy import Column, Integer, String, Float, DateTime, Enum
from .base import Base, TimestampMixin
from datetime import datetime
import enum


class PositionStatus(enum.Enum):
    OPEN = "OPEN"
    CLOSED = "CLOSED"


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


class Portfolio(Base, TimestampMixin):
    __tablename__ = 'portfolios'
    
    id = Column(Integer, primary_key=True)
    total_value = Column(Float, nullable=False)
    cash_balance = Column(Float, nullable=False)
    timestamp = Column(DateTime, nullable=False, default=datetime.utcnow)
