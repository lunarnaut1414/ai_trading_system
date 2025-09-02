"""
Database configuration for the AI Trading System.

This module handles database connection settings, pooling configuration,
and environment-specific database parameters.
"""

import os
from typing import Dict, Optional
from dataclasses import dataclass
from urllib.parse import quote_plus
import logging

logger = logging.getLogger(__name__)


@dataclass
class DatabaseConfig:
    """Database configuration settings."""
    
    # Connection parameters
    host: str
    port: int
    database: str
    username: str
    password: str
    
    # Connection pool settings
    pool_size: int = 10
    max_overflow: int = 20
    pool_timeout: int = 30
    pool_recycle: int = 3600
    
    # Query settings
    echo: bool = False
    echo_pool: bool = False
    
    # Performance settings
    connect_timeout: int = 10
    command_timeout: int = 60
    
    @property
    def connection_string(self) -> str:
        """Generate SQLAlchemy connection string."""
        # URL encode password to handle special characters
        encoded_password = quote_plus(self.password)
        
        # Build connection string with options
        conn_str = (
            f"postgresql://{self.username}:{encoded_password}@"
            f"{self.host}:{self.port}/{self.database}"
        )
        
        # Add connection parameters
        options = []
        if self.connect_timeout:
            options.append(f"connect_timeout={self.connect_timeout}")
        if self.command_timeout:
            options.append(f"command_timeout={self.command_timeout}")
        
        if options:
            conn_str += "?" + "&".join(options)
        
        return conn_str
    
    @property
    def async_connection_string(self) -> str:
        """Generate async SQLAlchemy connection string."""
        # URL encode password to handle special characters
        encoded_password = quote_plus(self.password)
        
        # Use asyncpg driver for async connections
        conn_str = (
            f"postgresql+asyncpg://{self.username}:{encoded_password}@"
            f"{self.host}:{self.port}/{self.database}"
        )
        
        return conn_str
    
    @property
    def engine_kwargs(self) -> Dict:
        """Get SQLAlchemy engine configuration."""
        return {
            "pool_size": self.pool_size,
            "max_overflow": self.max_overflow,
            "pool_timeout": self.pool_timeout,
            "pool_recycle": self.pool_recycle,
            "echo": self.echo,
            "echo_pool": self.echo_pool,
            "pool_pre_ping": True,  # Verify connections before using
            "connect_args": {
                "server_settings": {
                    "application_name": "ai_trading_system",
                    "jit": "off"  # Disable JIT for more predictable performance
                },
                "command_timeout": self.command_timeout,
                "timeout": self.connect_timeout,
            }
        }
    
    @classmethod
    def from_env(cls, env_prefix: str = "") -> "DatabaseConfig":
        """Create configuration from environment variables.
        
        Args:
            env_prefix: Prefix for environment variables (e.g., "TEST_", "PROD_")
        
        Returns:
            DatabaseConfig instance
        """
        def get_env(key: str, default: Optional[str] = None) -> str:
            env_key = f"{env_prefix}{key}" if env_prefix else key
            value = os.getenv(env_key, default)
            if value is None:
                raise ValueError(f"Missing required environment variable: {env_key}")
            return value
        
        def get_env_int(key: str, default: int) -> int:
            env_key = f"{env_prefix}{key}" if env_prefix else key
            value = os.getenv(env_key)
            return int(value) if value else default
        
        def get_env_bool(key: str, default: bool) -> bool:
            env_key = f"{env_prefix}{key}" if env_prefix else key
            value = os.getenv(env_key)
            if value is None:
                return default
            return value.lower() in ("true", "1", "yes", "on")
        
        return cls(
            host=get_env("DB_HOST", "localhost"),
            port=get_env_int("DB_PORT", 5432),
            database=get_env("DB_NAME", "trading_system"),
            username=get_env("DB_USER", "postgres"),
            password=get_env("DB_PASSWORD"),
            pool_size=get_env_int("DB_POOL_SIZE", 10),
            max_overflow=get_env_int("DB_MAX_OVERFLOW", 20),
            pool_timeout=get_env_int("DB_POOL_TIMEOUT", 30),
            pool_recycle=get_env_int("DB_POOL_RECYCLE", 3600),
            echo=get_env_bool("DB_ECHO", False),
            echo_pool=get_env_bool("DB_ECHO_POOL", False),
            connect_timeout=get_env_int("DB_CONNECT_TIMEOUT", 10),
            command_timeout=get_env_int("DB_COMMAND_TIMEOUT", 60),
        )


class DatabaseEnvironment:
    """Manage database configurations for different environments."""
    
    def __init__(self):
        """Initialize database environment manager."""
        self.configs: Dict[str, DatabaseConfig] = {}
        self._current_env = os.getenv("ENVIRONMENT", "development")
        
        # Load configurations
        self._load_configs()
    
    def _load_configs(self):
        """Load database configurations for all environments."""
        # Development configuration
        if self._current_env == "development":
            self.configs["development"] = DatabaseConfig(
                host=os.getenv("DB_HOST", "localhost"),
                port=int(os.getenv("DB_PORT", 5432)),
                database=os.getenv("DB_NAME", "trading_system_dev"),
                username=os.getenv("DB_USER", "postgres"),
                password=os.getenv("DB_PASSWORD", "postgres"),
                echo=True,
                pool_size=5,
                max_overflow=10
            )
        
        # Testing configuration
        elif self._current_env == "testing":
            self.configs["testing"] = DatabaseConfig(
                host=os.getenv("TEST_DB_HOST", "localhost"),
                port=int(os.getenv("TEST_DB_PORT", 5432)),
                database=os.getenv("TEST_DB_NAME", "trading_system_test"),
                username=os.getenv("TEST_DB_USER", "postgres"),
                password=os.getenv("TEST_DB_PASSWORD", "postgres"),
                echo=False,
                pool_size=2,
                max_overflow=5
            )
        
        # Production configuration
        elif self._current_env == "production":
            self.configs["production"] = DatabaseConfig.from_env("PROD_")
        
        # Default to loading from environment
        else:
            self.configs[self._current_env] = DatabaseConfig.from_env()
    
    @property
    def current(self) -> DatabaseConfig:
        """Get current environment's database configuration."""
        if self._current_env not in self.configs:
            raise ValueError(f"No configuration found for environment: {self._current_env}")
        return self.configs[self._current_env]
    
    def get(self, environment: str) -> DatabaseConfig:
        """Get database configuration for specific environment.
        
        Args:
            environment: Environment name
        
        Returns:
            DatabaseConfig for the specified environment
        """
        if environment not in self.configs:
            raise ValueError(f"No configuration found for environment: {environment}")
        return self.configs[environment]
    
    def set_environment(self, environment: str):
        """Switch to a different environment.
        
        Args:
            environment: Environment name to switch to
        """
        if environment not in self.configs:
            # Try to load it dynamically
            self._current_env = environment
            self._load_configs()
        else:
            self._current_env = environment
        
        logger.info(f"Switched to database environment: {environment}")


# Global database environment instance
db_env = DatabaseEnvironment()


def get_database_config() -> DatabaseConfig:
    """Get current database configuration.
    
    Returns:
        Current DatabaseConfig instance
    """
    return db_env.current


def get_connection_string() -> str:
    """Get current database connection string.
    
    Returns:
        SQLAlchemy connection string
    """
    return db_env.current.connection_string


def get_async_connection_string() -> str:
    """Get current async database connection string.
    
    Returns:
        SQLAlchemy async connection string
    """
    return db_env.current.async_connection_string