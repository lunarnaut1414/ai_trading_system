"""
Trade model for the database.
"""

from sqlalchemy import Column, Integer, String, Float, DateTime, Enum
from .base import Base, TimestampMixin
from datetime import datetime
import enum


class TradeStatus(enum.Enum):
    PENDING = "PENDING"
    FILLED = "FILLED"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"


class Trade(Base, TimestampMixin):
    __tablename__ = 'trades'
    
    id = Column(Integer, primary_key=True)
    symbol = Column(String(10), nullable=False)
    action = Column(String(10), nullable=False)
    quantity = Column(Integer, nullable=False)
    price = Column(Float, nullable=False)
    executed_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    order_id = Column(String(100), unique=True)
    status = Column(Enum(TradeStatus), default=TradeStatus.PENDING)
