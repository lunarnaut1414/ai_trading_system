"""
Performance metric model for the database.
"""

from sqlalchemy import Column, Integer, String, Float, DateTime
from .base import Base, TimestampMixin
from datetime import datetime


class PerformanceMetric(Base, TimestampMixin):
    __tablename__ = 'performance_metrics'
    
    id = Column(Integer, primary_key=True)
    metric_name = Column(String(50), nullable=False)
    metric_value = Column(Float, nullable=False)
    category = Column(String(50))
    timestamp = Column(DateTime, nullable=False, default=datetime.utcnow)
