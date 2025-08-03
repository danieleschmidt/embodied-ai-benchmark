"""Database connection management."""

import os
import logging
from typing import Optional, Dict, Any
from contextlib import contextmanager
import sqlite3
from pathlib import Path

try:
    import psycopg2
    from psycopg2.extras import RealDictCursor
    POSTGRES_AVAILABLE = True
except ImportError:
    POSTGRES_AVAILABLE = False

try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

logger = logging.getLogger(__name__)


class DatabaseConnection:
    """Database connection manager supporting SQLite and PostgreSQL."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize database connection.
        
        Args:
            config: Database configuration dictionary
        """
        self.config = config or self._load_config_from_env()
        self.db_type = self.config.get("type", "sqlite")
        self._connection = None
        self._setup_database()
    
    def _load_config_from_env(self) -> Dict[str, Any]:
        """Load database configuration from environment variables."""
        database_url = os.getenv("DATABASE_URL")
        
        if database_url:
            # Parse DATABASE_URL
            if database_url.startswith("postgresql://") or database_url.startswith("postgres://"):
                return {
                    "type": "postgresql",
                    "url": database_url
                }
            elif database_url.startswith("sqlite://"):
                return {
                    "type": "sqlite",
                    "path": database_url.replace("sqlite://", "")
                }
        
        # Fallback to individual environment variables
        return {
            "type": os.getenv("DATABASE_TYPE", "sqlite"),
            "host": os.getenv("DATABASE_HOST", "localhost"),
            "port": int(os.getenv("DATABASE_PORT", "5432")),
            "name": os.getenv("DATABASE_NAME", "embodied_ai_benchmark"),
            "user": os.getenv("DATABASE_USER", "postgres"),
            "password": os.getenv("DATABASE_PASSWORD", ""),
            "path": os.getenv("DATABASE_PATH", "./data/benchmark.db")
        }
    
    def _setup_database(self):
        """Setup database and create tables if needed."""
        if self.db_type == "sqlite":
            self._setup_sqlite()
        elif self.db_type == "postgresql" and POSTGRES_AVAILABLE:
            self._setup_postgresql()
        else:
            logger.warning(f"Database type {self.db_type} not supported or dependencies missing")
            self._setup_sqlite()  # Fallback to SQLite
    
    def _setup_sqlite(self):
        """Setup SQLite database."""
        db_path = self.config.get("path", "./data/benchmark.db")
        
        # Create directory if it doesn't exist
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        
        self._connection = sqlite3.connect(db_path, check_same_thread=False)
        self._connection.row_factory = sqlite3.Row  # Enable dict-like access
        
        # Enable foreign keys
        self._connection.execute("PRAGMA foreign_keys = ON")
        
        logger.info(f"Connected to SQLite database at {db_path}")
        self._create_tables()
    
    def _setup_postgresql(self):
        """Setup PostgreSQL database connection."""
        try:
            if "url" in self.config:
                self._connection = psycopg2.connect(
                    self.config["url"],
                    cursor_factory=RealDictCursor
                )
            else:
                self._connection = psycopg2.connect(
                    host=self.config["host"],
                    port=self.config["port"],
                    database=self.config["name"],
                    user=self.config["user"],
                    password=self.config["password"],
                    cursor_factory=RealDictCursor
                )
            
            self._connection.autocommit = True
            logger.info(f"Connected to PostgreSQL database at {self.config.get('host', 'unknown')}")
            self._create_tables()
            
        except Exception as e:
            logger.error(f"Failed to connect to PostgreSQL: {e}")
            logger.info("Falling back to SQLite")
            self._setup_sqlite()
    
    def _create_tables(self):
        """Create database tables if they don't exist."""
        tables_sql = [
            """
            CREATE TABLE IF NOT EXISTS experiments (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                description TEXT,
                config TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS benchmark_runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                experiment_id INTEGER,
                agent_name TEXT NOT NULL,
                task_name TEXT NOT NULL,
                num_episodes INTEGER,
                success_rate REAL,
                avg_reward REAL,
                avg_steps REAL,
                total_time REAL,
                config TEXT,
                results TEXT,
                started_at TIMESTAMP,
                completed_at TIMESTAMP,
                FOREIGN KEY (experiment_id) REFERENCES experiments (id)
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS episodes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id INTEGER,
                episode_id INTEGER,
                success BOOLEAN,
                total_steps INTEGER,
                total_reward REAL,
                completion_time REAL,
                trajectory_data TEXT,
                metrics TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (run_id) REFERENCES benchmark_runs (id)
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS agent_performance (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                agent_name TEXT NOT NULL,
                task_name TEXT NOT NULL,
                metric_name TEXT NOT NULL,
                metric_value REAL,
                measurement_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                metadata TEXT
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS task_metadata (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                task_name TEXT UNIQUE NOT NULL,
                task_type TEXT,
                difficulty TEXT,
                description TEXT,
                requirements TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        ]
        
        for sql in tables_sql:
            try:
                if self.db_type == "sqlite":
                    self._connection.execute(sql)
                elif self.db_type == "postgresql":
                    # Convert SQLite-specific SQL to PostgreSQL
                    pg_sql = sql.replace("INTEGER PRIMARY KEY AUTOINCREMENT", "SERIAL PRIMARY KEY")
                    pg_sql = pg_sql.replace("TIMESTAMP DEFAULT CURRENT_TIMESTAMP", "TIMESTAMP DEFAULT NOW()")
                    
                    with self._connection.cursor() as cursor:
                        cursor.execute(pg_sql)
                
            except Exception as e:
                logger.error(f"Error creating table: {e}")
        
        if self.db_type == "sqlite":
            self._connection.commit()
    
    @contextmanager
    def get_cursor(self):
        """Get database cursor with automatic cleanup."""
        if self.db_type == "sqlite":
            cursor = self._connection.cursor()
            try:
                yield cursor
            finally:
                cursor.close()
        elif self.db_type == "postgresql":
            with self._connection.cursor() as cursor:
                yield cursor
    
    def execute_query(self, query: str, params: tuple = None) -> list:
        """Execute a SELECT query and return results.
        
        Args:
            query: SQL query string
            params: Query parameters
            
        Returns:
            List of result rows
        """
        with self.get_cursor() as cursor:
            cursor.execute(query, params or ())
            return cursor.fetchall()
    
    def execute_insert(self, query: str, params: tuple = None) -> int:
        """Execute an INSERT query and return the last row ID.
        
        Args:
            query: SQL query string
            params: Query parameters
            
        Returns:
            Last inserted row ID
        """
        with self.get_cursor() as cursor:
            cursor.execute(query, params or ())
            
            if self.db_type == "sqlite":
                self._connection.commit()
                return cursor.lastrowid
            elif self.db_type == "postgresql":
                return cursor.fetchone()[0] if cursor.rowcount > 0 else None
    
    def execute_update(self, query: str, params: tuple = None) -> int:
        """Execute an UPDATE/DELETE query and return affected rows.
        
        Args:
            query: SQL query string
            params: Query parameters
            
        Returns:
            Number of affected rows
        """
        with self.get_cursor() as cursor:
            cursor.execute(query, params or ())
            
            if self.db_type == "sqlite":
                self._connection.commit()
            
            return cursor.rowcount
    
    def close(self):
        """Close database connection."""
        if self._connection:
            self._connection.close()
            self._connection = None
            logger.info("Database connection closed")
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


class CacheManager:
    """Redis cache manager for benchmark results."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize cache manager.
        
        Args:
            config: Redis configuration dictionary
        """
        self.config = config or self._load_config_from_env()
        self._redis_client = None
        self._setup_cache()
    
    def _load_config_from_env(self) -> Dict[str, Any]:
        """Load cache configuration from environment variables."""
        redis_url = os.getenv("REDIS_URL")
        
        if redis_url:
            return {"url": redis_url}
        
        return {
            "host": os.getenv("REDIS_HOST", "localhost"),
            "port": int(os.getenv("REDIS_PORT", "6379")),
            "db": int(os.getenv("REDIS_DB", "0")),
            "password": os.getenv("REDIS_PASSWORD")
        }
    
    def _setup_cache(self):
        """Setup Redis connection."""
        if not REDIS_AVAILABLE:
            logger.warning("Redis not available, caching disabled")
            return
        
        try:
            if "url" in self.config:
                self._redis_client = redis.from_url(self.config["url"])
            else:
                self._redis_client = redis.Redis(
                    host=self.config["host"],
                    port=self.config["port"],
                    db=self.config["db"],
                    password=self.config.get("password"),
                    decode_responses=True
                )
            
            # Test connection
            self._redis_client.ping()
            logger.info("Connected to Redis cache")
            
        except Exception as e:
            logger.warning(f"Failed to connect to Redis: {e}")
            self._redis_client = None
    
    def get(self, key: str) -> Optional[str]:
        """Get value from cache.
        
        Args:
            key: Cache key
            
        Returns:
            Cached value or None
        """
        if not self._redis_client:
            return None
        
        try:
            return self._redis_client.get(key)
        except Exception as e:
            logger.error(f"Cache get error: {e}")
            return None
    
    def set(self, key: str, value: str, ttl: int = 3600):
        """Set value in cache.
        
        Args:
            key: Cache key
            value: Value to cache
            ttl: Time to live in seconds
        """
        if not self._redis_client:
            return
        
        try:
            self._redis_client.setex(key, ttl, value)
        except Exception as e:
            logger.error(f"Cache set error: {e}")
    
    def delete(self, key: str):
        """Delete key from cache.
        
        Args:
            key: Cache key to delete
        """
        if not self._redis_client:
            return
        
        try:
            self._redis_client.delete(key)
        except Exception as e:
            logger.error(f"Cache delete error: {e}")
    
    def clear(self):
        """Clear all cache entries."""
        if not self._redis_client:
            return
        
        try:
            self._redis_client.flushdb()
        except Exception as e:
            logger.error(f"Cache clear error: {e}")


# Global database connection instance
_db_connection = None
_cache_manager = None


def get_database() -> DatabaseConnection:
    """Get global database connection instance."""
    global _db_connection
    if _db_connection is None:
        _db_connection = DatabaseConnection()
    return _db_connection


def get_cache() -> CacheManager:
    """Get global cache manager instance."""
    global _cache_manager
    if _cache_manager is None:
        _cache_manager = CacheManager()
    return _cache_manager


def close_connections():
    """Close all database connections."""
    global _db_connection, _cache_manager
    
    if _db_connection:
        _db_connection.close()
        _db_connection = None
    
    if _cache_manager and _cache_manager._redis_client:
        _cache_manager._redis_client.close()
        _cache_manager = None