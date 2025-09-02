"""
Alert model for the database.
"""

from sqlalchemy import Column, Integer, String, Text, Boolean, DateTime
from .base import Base, TimestampMixin
from datetime import datetime


class Alert(Base, TimestampMixin):
    __tablename__ = 'alerts'
    
    id = Column(Integer, primary_key=True)
    alert_type = Column(String(50), nullable=False)
    severity = Column(String(20), nullable=False)
    message = Column(Text, nullable=False, default='')
    source = Column(String(50))
    resolved = Column(Boolean, default=False)
    timestamp = Column(DateTime, nullable=False, default=datetime.utcnow)
