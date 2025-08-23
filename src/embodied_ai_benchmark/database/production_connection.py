"""Production database connection with PostgreSQL and MongoDB support."""

import os
import json
import logging
import time
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
import psycopg2
from psycopg2.extras import RealDictCursor
import pymongo
from pymongo import MongoClient


@dataclass
class ExperimentResult:
    """Structured experiment result for database storage."""
    experiment_id: str
    task_name: str
    agent_name: str
    episode_id: str
    success: bool
    reward: float
    steps: int
    duration: float
    observations: List[Dict[str, Any]]
    actions: List[Dict[str, Any]]
    metrics: Dict[str, float]
    metadata: Dict[str, Any]
    timestamp: datetime


@dataclass
class ModelPerformance:
    """Model performance tracking."""
    model_id: str
    task_type: str
    success_rate: float
    avg_reward: float
    avg_steps: float
    total_episodes: int
    last_updated: datetime
    performance_trend: List[float]


class ProductionDatabase:
    """Production database interface with PostgreSQL and MongoDB support."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize database connections.
        
        Args:
            config: Database configuration
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Connection parameters
        self.pg_config = config.get("postgresql", {})
        self.mongo_config = config.get("mongodb", {})
        
        # Connection pools
        self.pg_conn = None
        self.mongo_client = None
        self.mongo_db = None
        
        # Performance tracking
        self.query_times = []
        self.connection_errors = 0
        
        self._initialize_connections()
    
    def _initialize_connections(self):
        """Initialize database connections."""
        try:
            # PostgreSQL for structured data
            if self.pg_config:
                self.pg_conn = psycopg2.connect(
                    host=self.pg_config.get("host", "localhost"),
                    port=self.pg_config.get("port", 5432),
                    database=self.pg_config.get("database", "embodied_ai"),
                    user=self.pg_config.get("user", os.getenv("DB_USER", "postgres")),
                    password=self.pg_config.get("password", os.getenv("DB_PASSWORD", "")),
                    cursor_factory=RealDictCursor
                )
                self._initialize_pg_schema()
                self.logger.info("PostgreSQL connection established")
            
            # MongoDB for unstructured data
            if self.mongo_config:
                mongo_uri = self.mongo_config.get("uri", "mongodb://localhost:27017/")
                self.mongo_client = MongoClient(mongo_uri)
                self.mongo_db = self.mongo_client[self.mongo_config.get("database", "embodied_ai")]
                
                # Test connection
                self.mongo_client.admin.command('ping')
                self.logger.info("MongoDB connection established")
            
        except Exception as e:
            self.logger.error(f"Database initialization failed: {e}")
            self.connection_errors += 1
            
            # Use SQLite fallback
            self._initialize_sqlite_fallback()
    
    def _initialize_sqlite_fallback(self):
        """Initialize SQLite fallback database."""
        import sqlite3
        
        try:
            db_path = self.config.get("sqlite_path", "/tmp/embodied_ai.db")
            self.sqlite_conn = sqlite3.connect(db_path, check_same_thread=False)
            self.sqlite_conn.row_factory = sqlite3.Row
            
            # Create tables
            self._create_sqlite_tables()
            self.logger.info(f"SQLite fallback initialized at {db_path}")
            
        except Exception as e:
            self.logger.error(f"SQLite fallback failed: {e}")
    
    def _initialize_pg_schema(self):
        """Initialize PostgreSQL schema."""
        
        schema_sql = [
            """
            CREATE TABLE IF NOT EXISTS experiments (
                id SERIAL PRIMARY KEY,
                experiment_id VARCHAR(255) UNIQUE NOT NULL,
                name VARCHAR(255) NOT NULL,
                description TEXT,
                config JSONB,
                created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                status VARCHAR(50) DEFAULT 'running',
                results_summary JSONB
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS episodes (
                id SERIAL PRIMARY KEY,
                experiment_id VARCHAR(255) REFERENCES experiments(experiment_id),
                episode_id VARCHAR(255) NOT NULL,
                task_name VARCHAR(255) NOT NULL,
                agent_name VARCHAR(255) NOT NULL,
                success BOOLEAN NOT NULL,
                reward REAL NOT NULL,
                steps INTEGER NOT NULL,
                duration REAL NOT NULL,
                metrics JSONB,
                metadata JSONB,
                created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                UNIQUE(experiment_id, episode_id)
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS model_performance (
                id SERIAL PRIMARY KEY,
                model_id VARCHAR(255) NOT NULL,
                task_type VARCHAR(255) NOT NULL,
                success_rate REAL NOT NULL,
                avg_reward REAL NOT NULL,
                avg_steps REAL NOT NULL,
                total_episodes INTEGER NOT NULL,
                performance_trend JSONB,
                last_updated TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                UNIQUE(model_id, task_type)
            )
            """,
            """
            CREATE INDEX IF NOT EXISTS idx_episodes_experiment ON episodes(experiment_id);
            CREATE INDEX IF NOT EXISTS idx_episodes_task ON episodes(task_name);
            CREATE INDEX IF NOT EXISTS idx_episodes_success ON episodes(success);
            CREATE INDEX IF NOT EXISTS idx_episodes_created ON episodes(created_at);
            CREATE INDEX IF NOT EXISTS idx_model_perf_model ON model_performance(model_id);
            """
        ]
        
        with self.pg_conn.cursor() as cursor:
            for sql in schema_sql:
                cursor.execute(sql)
        self.pg_conn.commit()
    
    def _create_sqlite_tables(self):
        """Create SQLite tables for fallback."""
        
        tables_sql = [
            """
            CREATE TABLE IF NOT EXISTS experiments (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                experiment_id TEXT UNIQUE NOT NULL,
                name TEXT NOT NULL,
                description TEXT,
                config TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                status TEXT DEFAULT 'running',
                results_summary TEXT
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS episodes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                experiment_id TEXT,
                episode_id TEXT NOT NULL,
                task_name TEXT NOT NULL,
                agent_name TEXT NOT NULL,
                success BOOLEAN NOT NULL,
                reward REAL NOT NULL,
                steps INTEGER NOT NULL,
                duration REAL NOT NULL,
                metrics TEXT,
                metadata TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS model_performance (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                model_id TEXT NOT NULL,
                task_type TEXT NOT NULL,
                success_rate REAL NOT NULL,
                avg_reward REAL NOT NULL,
                avg_steps REAL NOT NULL,
                total_episodes INTEGER NOT NULL,
                performance_trend TEXT,
                last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        ]
        
        for sql in tables_sql:
            self.sqlite_conn.execute(sql)
        self.sqlite_conn.commit()
    
    def store_experiment_result(self, result: ExperimentResult) -> bool:
        """Store experiment result in database."""
        
        start_time = time.time()
        
        try:
            if self.pg_conn:
                return self._store_pg_result(result)
            elif hasattr(self, 'sqlite_conn'):
                return self._store_sqlite_result(result)
            else:
                self.logger.error("No database connection available")
                return False
                
        except Exception as e:
            self.logger.error(f"Failed to store experiment result: {e}")
            return False
        
        finally:
            query_time = time.time() - start_time
            self.query_times.append(query_time)
    
    def _store_pg_result(self, result: ExperimentResult) -> bool:
        """Store result in PostgreSQL."""
        
        sql = """
        INSERT INTO episodes (
            experiment_id, episode_id, task_name, agent_name,
            success, reward, steps, duration, metrics, metadata
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (experiment_id, episode_id) 
        DO UPDATE SET
            success = EXCLUDED.success,
            reward = EXCLUDED.reward,
            steps = EXCLUDED.steps,
            duration = EXCLUDED.duration,
            metrics = EXCLUDED.metrics,
            metadata = EXCLUDED.metadata
        """
        
        with self.pg_conn.cursor() as cursor:
            cursor.execute(sql, (
                result.experiment_id,
                result.episode_id,
                result.task_name,
                result.agent_name,
                result.success,
                result.reward,
                result.steps,
                result.duration,
                json.dumps(result.metrics),
                json.dumps(result.metadata)
            ))
        
        self.pg_conn.commit()
        
        # Store detailed observations/actions in MongoDB if available
        if self.mongo_db:
            self._store_mongo_trajectory(result)
        
        return True
    
    def _store_sqlite_result(self, result: ExperimentResult) -> bool:
        """Store result in SQLite."""
        
        sql = """
        INSERT OR REPLACE INTO episodes (
            experiment_id, episode_id, task_name, agent_name,
            success, reward, steps, duration, metrics, metadata
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        
        self.sqlite_conn.execute(sql, (
            result.experiment_id,
            result.episode_id,
            result.task_name,
            result.agent_name,
            result.success,
            result.reward,
            result.steps,
            result.duration,
            json.dumps(result.metrics),
            json.dumps(result.metadata)
        ))
        
        self.sqlite_conn.commit()
        return True
    
    def _store_mongo_trajectory(self, result: ExperimentResult):
        """Store detailed trajectory data in MongoDB."""
        
        try:
            trajectory_doc = {
                "experiment_id": result.experiment_id,
                "episode_id": result.episode_id,
                "task_name": result.task_name,
                "agent_name": result.agent_name,
                "observations": result.observations,
                "actions": result.actions,
                "timestamp": result.timestamp,
                "metadata": result.metadata
            }
            
            collection = self.mongo_db["trajectories"]
            collection.replace_one(
                {"experiment_id": result.experiment_id, "episode_id": result.episode_id},
                trajectory_doc,
                upsert=True
            )
            
        except Exception as e:
            self.logger.warning(f"Failed to store trajectory in MongoDB: {e}")
    
    def get_experiment_results(self, 
                             experiment_id: str, 
                             limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Get experiment results from database."""
        
        try:
            if self.pg_conn:
                return self._get_pg_results(experiment_id, limit)
            elif hasattr(self, 'sqlite_conn'):
                return self._get_sqlite_results(experiment_id, limit)
            else:
                return []
                
        except Exception as e:
            self.logger.error(f"Failed to get experiment results: {e}")
            return []
    
    def _get_pg_results(self, experiment_id: str, limit: Optional[int]) -> List[Dict[str, Any]]:
        """Get results from PostgreSQL."""
        
        sql = """
        SELECT * FROM episodes 
        WHERE experiment_id = %s 
        ORDER BY created_at DESC
        """
        
        if limit:
            sql += f" LIMIT {limit}"
        
        with self.pg_conn.cursor() as cursor:
            cursor.execute(sql, (experiment_id,))
            results = cursor.fetchall()
        
        return [dict(row) for row in results]
    
    def _get_sqlite_results(self, experiment_id: str, limit: Optional[int]) -> List[Dict[str, Any]]:
        """Get results from SQLite."""
        
        sql = """
        SELECT * FROM episodes 
        WHERE experiment_id = ? 
        ORDER BY created_at DESC
        """
        
        if limit:
            sql += f" LIMIT {limit}"
        
        cursor = self.sqlite_conn.execute(sql, (experiment_id,))
        results = cursor.fetchall()
        
        return [dict(row) for row in results]
    
    def update_model_performance(self, performance: ModelPerformance) -> bool:
        """Update model performance metrics."""
        
        try:
            if self.pg_conn:
                return self._update_pg_performance(performance)
            elif hasattr(self, 'sqlite_conn'):
                return self._update_sqlite_performance(performance)
            else:
                return False
                
        except Exception as e:
            self.logger.error(f"Failed to update model performance: {e}")
            return False
    
    def _update_pg_performance(self, performance: ModelPerformance) -> bool:
        """Update performance in PostgreSQL."""
        
        sql = """
        INSERT INTO model_performance (
            model_id, task_type, success_rate, avg_reward, avg_steps,
            total_episodes, performance_trend
        ) VALUES (%s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (model_id, task_type)
        DO UPDATE SET
            success_rate = EXCLUDED.success_rate,
            avg_reward = EXCLUDED.avg_reward,
            avg_steps = EXCLUDED.avg_steps,
            total_episodes = EXCLUDED.total_episodes,
            performance_trend = EXCLUDED.performance_trend,
            last_updated = NOW()
        """
        
        with self.pg_conn.cursor() as cursor:
            cursor.execute(sql, (
                performance.model_id,
                performance.task_type,
                performance.success_rate,
                performance.avg_reward,
                performance.avg_steps,
                performance.total_episodes,
                json.dumps(performance.performance_trend)
            ))
        
        self.pg_conn.commit()
        return True
    
    def _update_sqlite_performance(self, performance: ModelPerformance) -> bool:
        """Update performance in SQLite."""
        
        sql = """
        INSERT OR REPLACE INTO model_performance (
            model_id, task_type, success_rate, avg_reward, avg_steps,
            total_episodes, performance_trend
        ) VALUES (?, ?, ?, ?, ?, ?, ?)
        """
        
        self.sqlite_conn.execute(sql, (
            performance.model_id,
            performance.task_type,
            performance.success_rate,
            performance.avg_reward,
            performance.avg_steps,
            performance.total_episodes,
            json.dumps(performance.performance_trend)
        ))
        
        self.sqlite_conn.commit()
        return True
    
    def get_model_performance(self, model_id: str, task_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get model performance data."""
        
        try:
            if self.pg_conn:
                return self._get_pg_performance(model_id, task_type)
            elif hasattr(self, 'sqlite_conn'):
                return self._get_sqlite_performance(model_id, task_type)
            else:
                return []
                
        except Exception as e:
            self.logger.error(f"Failed to get model performance: {e}")
            return []
    
    def _get_pg_performance(self, model_id: str, task_type: Optional[str]) -> List[Dict[str, Any]]:
        """Get performance from PostgreSQL."""
        
        if task_type:
            sql = "SELECT * FROM model_performance WHERE model_id = %s AND task_type = %s"
            params = (model_id, task_type)
        else:
            sql = "SELECT * FROM model_performance WHERE model_id = %s"
            params = (model_id,)
        
        with self.pg_conn.cursor() as cursor:
            cursor.execute(sql, params)
            results = cursor.fetchall()
        
        return [dict(row) for row in results]
    
    def _get_sqlite_performance(self, model_id: str, task_type: Optional[str]) -> List[Dict[str, Any]]:
        """Get performance from SQLite."""
        
        if task_type:
            sql = "SELECT * FROM model_performance WHERE model_id = ? AND task_type = ?"
            params = (model_id, task_type)
        else:
            sql = "SELECT * FROM model_performance WHERE model_id = ?"
            params = (model_id,)
        
        cursor = self.sqlite_conn.execute(sql, params)
        results = cursor.fetchall()
        
        return [dict(row) for row in results]
    
    def get_task_statistics(self, task_name: str, days: int = 7) -> Dict[str, Any]:
        """Get task performance statistics."""
        
        try:
            if self.pg_conn:
                return self._get_pg_task_stats(task_name, days)
            elif hasattr(self, 'sqlite_conn'):
                return self._get_sqlite_task_stats(task_name, days)
            else:
                return {}
                
        except Exception as e:
            self.logger.error(f"Failed to get task statistics: {e}")
            return {}
    
    def _get_pg_task_stats(self, task_name: str, days: int) -> Dict[str, Any]:
        """Get task statistics from PostgreSQL."""
        
        sql = """
        SELECT 
            COUNT(*) as total_episodes,
            AVG(CASE WHEN success THEN 1.0 ELSE 0.0 END) as success_rate,
            AVG(reward) as avg_reward,
            AVG(steps) as avg_steps,
            AVG(duration) as avg_duration,
            COUNT(DISTINCT agent_name) as unique_agents
        FROM episodes 
        WHERE task_name = %s 
        AND created_at >= NOW() - INTERVAL '%s days'
        """
        
        with self.pg_conn.cursor() as cursor:
            cursor.execute(sql, (task_name, days))
            result = cursor.fetchone()
        
        return dict(result) if result else {}
    
    def _get_sqlite_task_stats(self, task_name: str, days: int) -> Dict[str, Any]:
        """Get task statistics from SQLite."""
        
        sql = """
        SELECT 
            COUNT(*) as total_episodes,
            AVG(CASE WHEN success THEN 1.0 ELSE 0.0 END) as success_rate,
            AVG(reward) as avg_reward,
            AVG(steps) as avg_steps,
            AVG(duration) as avg_duration,
            COUNT(DISTINCT agent_name) as unique_agents
        FROM episodes 
        WHERE task_name = ? 
        AND created_at >= datetime('now', '-{} days')
        """.format(days)
        
        cursor = self.sqlite_conn.execute(sql, (task_name,))
        result = cursor.fetchone()
        
        return dict(result) if result else {}
    
    def health_check(self) -> Dict[str, Any]:
        """Check database health."""
        
        health = {
            "postgresql": {"status": "disconnected", "latency": None},
            "mongodb": {"status": "disconnected", "latency": None},
            "sqlite": {"status": "disconnected", "latency": None},
            "query_performance": {
                "avg_query_time": np.mean(self.query_times) if self.query_times else 0,
                "connection_errors": self.connection_errors
            }
        }
        
        # Test PostgreSQL
        if self.pg_conn:
            try:
                start_time = time.time()
                with self.pg_conn.cursor() as cursor:
                    cursor.execute("SELECT 1")
                latency = time.time() - start_time
                health["postgresql"] = {"status": "connected", "latency": latency}
            except Exception as e:
                health["postgresql"] = {"status": "error", "error": str(e)}
        
        # Test MongoDB
        if self.mongo_client:
            try:
                start_time = time.time()
                self.mongo_client.admin.command('ping')
                latency = time.time() - start_time
                health["mongodb"] = {"status": "connected", "latency": latency}
            except Exception as e:
                health["mongodb"] = {"status": "error", "error": str(e)}
        
        # Test SQLite
        if hasattr(self, 'sqlite_conn'):
            try:
                start_time = time.time()
                self.sqlite_conn.execute("SELECT 1")
                latency = time.time() - start_time
                health["sqlite"] = {"status": "connected", "latency": latency}
            except Exception as e:
                health["sqlite"] = {"status": "error", "error": str(e)}
        
        return health
    
    def close(self):
        """Close database connections."""
        
        try:
            if self.pg_conn:
                self.pg_conn.close()
                self.logger.info("PostgreSQL connection closed")
            
            if self.mongo_client:
                self.mongo_client.close()
                self.logger.info("MongoDB connection closed")
            
            if hasattr(self, 'sqlite_conn'):
                self.sqlite_conn.close()
                self.logger.info("SQLite connection closed")
                
        except Exception as e:
            self.logger.error(f"Error closing database connections: {e}")