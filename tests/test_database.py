# tests/test_database.py
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

@pytest.mark.unit
class TestDatabase:
    """Test database operations"""
    
    @pytest.fixture
    def test_db_session(self):
        """Create test database session with rollback"""
        engine = create_engine('sqlite:///:memory:')
        Session = sessionmaker(bind=engine)
        session = Session()
        
        yield session
        
        session.rollback()
        session.close()
    
    def test_portfolio_crud(self, test_db_session):
        """Test portfolio CRUD operations"""
        # Test implementation
        pass