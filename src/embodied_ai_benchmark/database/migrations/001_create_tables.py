"""Initial database schema migration."""

from typing import List
from ..connection import DatabaseConnection


class Migration001:
    """Create initial database tables."""
    
    def __init__(self, db: DatabaseConnection):
        self.db = db
    
    def up(self):
        """Apply migration."""
        self._create_experiments_table()
        self._create_benchmark_runs_table()
        self._create_episodes_table()
        self._create_agent_performance_table()
        self._create_task_metadata_table()
        self._create_indices()
    
    def down(self):
        """Rollback migration."""
        tables = [
            "agent_performance",
            "episodes", 
            "benchmark_runs",
            "experiments",
            "task_metadata"
        ]
        
        for table in tables:
            self.db.execute_update(f"DROP TABLE IF EXISTS {table}")
    
    def _create_experiments_table(self):
        """Create experiments table."""
        if self.db.db_type == "sqlite":
            sql = """
            CREATE TABLE IF NOT EXISTS experiments (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL UNIQUE,
                description TEXT,
                config TEXT,
                status TEXT DEFAULT 'pending',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                completed_at TIMESTAMP NULL
            )
            """
        else:  # PostgreSQL
            sql = """
            CREATE TABLE IF NOT EXISTS experiments (
                id SERIAL PRIMARY KEY,
                name VARCHAR(255) NOT NULL UNIQUE,
                description TEXT,
                config JSONB,
                status VARCHAR(50) DEFAULT 'pending',
                created_at TIMESTAMP DEFAULT NOW(),
                updated_at TIMESTAMP DEFAULT NOW(),
                completed_at TIMESTAMP NULL
            )
            """
        
        self.db.execute_update(sql)
    
    def _create_benchmark_runs_table(self):
        """Create benchmark_runs table."""
        if self.db.db_type == "sqlite":
            sql = """
            CREATE TABLE IF NOT EXISTS benchmark_runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                experiment_id INTEGER,
                agent_name TEXT NOT NULL,
                agent_config TEXT,
                task_name TEXT NOT NULL,
                task_config TEXT,
                num_episodes INTEGER DEFAULT 0,
                success_rate REAL DEFAULT 0.0,
                avg_reward REAL DEFAULT 0.0,
                avg_steps REAL DEFAULT 0.0,
                total_time REAL DEFAULT 0.0,
                status TEXT DEFAULT 'pending',
                error_message TEXT NULL,
                results TEXT,
                started_at TIMESTAMP NULL,
                completed_at TIMESTAMP NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (experiment_id) REFERENCES experiments (id) ON DELETE CASCADE
            )
            """
        else:  # PostgreSQL
            sql = """
            CREATE TABLE IF NOT EXISTS benchmark_runs (
                id SERIAL PRIMARY KEY,
                experiment_id INTEGER,
                agent_name VARCHAR(255) NOT NULL,
                agent_config JSONB,
                task_name VARCHAR(255) NOT NULL,
                task_config JSONB,
                num_episodes INTEGER DEFAULT 0,
                success_rate REAL DEFAULT 0.0,
                avg_reward REAL DEFAULT 0.0,
                avg_steps REAL DEFAULT 0.0,
                total_time REAL DEFAULT 0.0,
                status VARCHAR(50) DEFAULT 'pending',
                error_message TEXT NULL,
                results JSONB,
                started_at TIMESTAMP NULL,
                completed_at TIMESTAMP NULL,
                created_at TIMESTAMP DEFAULT NOW(),
                FOREIGN KEY (experiment_id) REFERENCES experiments (id) ON DELETE CASCADE
            )
            """
        
        self.db.execute_update(sql)
    
    def _create_episodes_table(self):
        """Create episodes table."""
        if self.db.db_type == "sqlite":
            sql = """
            CREATE TABLE IF NOT EXISTS episodes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id INTEGER NOT NULL,
                episode_id INTEGER NOT NULL,
                success BOOLEAN DEFAULT FALSE,
                total_steps INTEGER DEFAULT 0,
                total_reward REAL DEFAULT 0.0,
                completion_time REAL DEFAULT 0.0,
                safety_violations INTEGER DEFAULT 0,
                trajectory_data TEXT NULL,
                metrics TEXT NULL,
                error_message TEXT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (run_id) REFERENCES benchmark_runs (id) ON DELETE CASCADE,
                UNIQUE(run_id, episode_id)
            )
            """
        else:  # PostgreSQL
            sql = """
            CREATE TABLE IF NOT EXISTS episodes (
                id SERIAL PRIMARY KEY,
                run_id INTEGER NOT NULL,
                episode_id INTEGER NOT NULL,
                success BOOLEAN DEFAULT FALSE,
                total_steps INTEGER DEFAULT 0,
                total_reward REAL DEFAULT 0.0,
                completion_time REAL DEFAULT 0.0,
                safety_violations INTEGER DEFAULT 0,
                trajectory_data JSONB NULL,
                metrics JSONB NULL,
                error_message TEXT NULL,
                created_at TIMESTAMP DEFAULT NOW(),
                FOREIGN KEY (run_id) REFERENCES benchmark_runs (id) ON DELETE CASCADE,
                UNIQUE(run_id, episode_id)
            )
            """
        
        self.db.execute_update(sql)
    
    def _create_agent_performance_table(self):
        """Create agent_performance table."""
        if self.db.db_type == "sqlite":
            sql = """
            CREATE TABLE IF NOT EXISTS agent_performance (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                agent_name TEXT NOT NULL,
                task_name TEXT NOT NULL,
                metric_name TEXT NOT NULL,
                metric_value REAL NOT NULL,
                measurement_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                metadata TEXT NULL,
                session_id TEXT NULL
            )
            """
        else:  # PostgreSQL
            sql = """
            CREATE TABLE IF NOT EXISTS agent_performance (
                id SERIAL PRIMARY KEY,
                agent_name VARCHAR(255) NOT NULL,
                task_name VARCHAR(255) NOT NULL,
                metric_name VARCHAR(255) NOT NULL,
                metric_value REAL NOT NULL,
                measurement_time TIMESTAMP DEFAULT NOW(),
                metadata JSONB NULL,
                session_id VARCHAR(255) NULL
            )
            """
        
        self.db.execute_update(sql)
    
    def _create_task_metadata_table(self):
        """Create task_metadata table."""
        if self.db.db_type == "sqlite":
            sql = """
            CREATE TABLE IF NOT EXISTS task_metadata (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                task_name TEXT UNIQUE NOT NULL,
                task_type TEXT NOT NULL,
                difficulty TEXT DEFAULT 'medium',
                description TEXT,
                requirements TEXT,
                success_criteria TEXT,
                time_limit INTEGER DEFAULT 300,
                max_steps INTEGER DEFAULT 1000,
                multi_agent BOOLEAN DEFAULT FALSE,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        else:  # PostgreSQL
            sql = """
            CREATE TABLE IF NOT EXISTS task_metadata (
                id SERIAL PRIMARY KEY,
                task_name VARCHAR(255) UNIQUE NOT NULL,
                task_type VARCHAR(100) NOT NULL,
                difficulty VARCHAR(50) DEFAULT 'medium',
                description TEXT,
                requirements JSONB,
                success_criteria TEXT,
                time_limit INTEGER DEFAULT 300,
                max_steps INTEGER DEFAULT 1000,
                multi_agent BOOLEAN DEFAULT FALSE,
                created_at TIMESTAMP DEFAULT NOW(),
                updated_at TIMESTAMP DEFAULT NOW()
            )
            """
        
        self.db.execute_update(sql)
    
    def _create_indices(self):
        """Create database indices for performance."""
        indices = [
            "CREATE INDEX IF NOT EXISTS idx_benchmark_runs_experiment ON benchmark_runs(experiment_id)",
            "CREATE INDEX IF NOT EXISTS idx_benchmark_runs_agent ON benchmark_runs(agent_name)",
            "CREATE INDEX IF NOT EXISTS idx_benchmark_runs_task ON benchmark_runs(task_name)",
            "CREATE INDEX IF NOT EXISTS idx_benchmark_runs_status ON benchmark_runs(status)",
            "CREATE INDEX IF NOT EXISTS idx_episodes_run ON episodes(run_id)",
            "CREATE INDEX IF NOT EXISTS idx_episodes_success ON episodes(success)",
            "CREATE INDEX IF NOT EXISTS idx_agent_performance_agent ON agent_performance(agent_name)",
            "CREATE INDEX IF NOT EXISTS idx_agent_performance_task ON agent_performance(task_name)",
            "CREATE INDEX IF NOT EXISTS idx_agent_performance_metric ON agent_performance(metric_name)",
            "CREATE INDEX IF NOT EXISTS idx_agent_performance_time ON agent_performance(measurement_time)",
            "CREATE INDEX IF NOT EXISTS idx_task_metadata_type ON task_metadata(task_type)",
            "CREATE INDEX IF NOT EXISTS idx_task_metadata_difficulty ON task_metadata(difficulty)"
        ]
        
        for index_sql in indices:
            try:
                self.db.execute_update(index_sql)
            except Exception as e:
                # Index might already exist, continue
                pass


def run_migration(db: DatabaseConnection):
    """Run the migration."""
    migration = Migration001(db)
    migration.up()
    return True


def rollback_migration(db: DatabaseConnection):
    """Rollback the migration."""
    migration = Migration001(db)
    migration.down()
    return True