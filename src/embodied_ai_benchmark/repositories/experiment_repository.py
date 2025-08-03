"""Repository for experiment data management."""

import json
from typing import Any, Dict, List, Optional
from datetime import datetime

from .base_repository import BaseRepository


class ExperimentRepository(BaseRepository):
    """Repository for managing experiment records."""
    
    def _get_table_name(self) -> str:
        return "experiments"
    
    def create_experiment(self, 
                         name: str, 
                         description: str, 
                         config: Dict[str, Any]) -> Optional[int]:
        """Create a new experiment.
        
        Args:
            name: Experiment name
            description: Experiment description
            config: Experiment configuration
            
        Returns:
            Experiment ID or None if failed
        """
        data = {
            "name": name,
            "description": description,
            "config": json.dumps(config) if self.db.db_type == "sqlite" else config,
            "status": "pending",
            "created_at": datetime.now().isoformat()
        }
        
        return self.create(data)
    
    def find_by_name(self, name: str) -> Optional[Dict[str, Any]]:
        """Find experiment by name.
        
        Args:
            name: Experiment name
            
        Returns:
            Experiment record or None if not found
        """
        experiments = self.find_by_field("name", name)
        return experiments[0] if experiments else None
    
    def find_by_status(self, status: str) -> List[Dict[str, Any]]:
        """Find experiments by status.
        
        Args:
            status: Experiment status
            
        Returns:
            List of matching experiments
        """
        return self.find_by_field("status", status)
    
    def update_status(self, experiment_id: int, status: str) -> bool:
        """Update experiment status.
        
        Args:
            experiment_id: Experiment ID
            status: New status
            
        Returns:
            True if updated successfully
        """
        data = {"status": status}
        
        if status == "completed":
            data["completed_at"] = datetime.now().isoformat()
        
        return self.update(experiment_id, data)
    
    def get_experiment_summary(self, experiment_id: int) -> Optional[Dict[str, Any]]:
        """Get experiment summary with run statistics.
        
        Args:
            experiment_id: Experiment ID
            
        Returns:
            Experiment summary or None if not found
        """
        # Get experiment details
        experiment = self.find_by_id(experiment_id)
        if not experiment:
            return None
        
        # Get run statistics
        stats_query = """
        SELECT 
            COUNT(*) as total_runs,
            COUNT(CASE WHEN status = 'completed' THEN 1 END) as completed_runs,
            COUNT(CASE WHEN status = 'failed' THEN 1 END) as failed_runs,
            AVG(CASE WHEN success_rate IS NOT NULL THEN success_rate END) as avg_success_rate,
            AVG(CASE WHEN total_time IS NOT NULL THEN total_time END) as avg_time
        FROM benchmark_runs 
        WHERE experiment_id = ?
        """
        
        if self.db.db_type == "postgresql":
            stats_query = stats_query.replace("?", "%s")
        
        stats = self.db.execute_query(stats_query, (experiment_id,))
        
        if stats:
            stats_dict = dict(stats[0])
            experiment.update(stats_dict)
        
        # Parse config if it's a JSON string
        if isinstance(experiment.get("config"), str):
            try:
                experiment["config"] = json.loads(experiment["config"])
            except json.JSONDecodeError:
                pass
        
        return experiment
    
    def find_recent_experiments(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Find most recent experiments.
        
        Args:
            limit: Maximum number of experiments to return
            
        Returns:
            List of recent experiments
        """
        query = f"SELECT * FROM {self.table_name} ORDER BY created_at DESC LIMIT {limit}"
        rows = self.db.execute_query(query)
        
        experiments = []
        for row in rows:
            exp = dict(row)
            # Parse config if it's a JSON string
            if isinstance(exp.get("config"), str):
                try:
                    exp["config"] = json.loads(exp["config"])
                except json.JSONDecodeError:
                    pass
            experiments.append(exp)
        
        return experiments
    
    def get_experiments_by_date_range(self, 
                                     start_date: datetime, 
                                     end_date: datetime) -> List[Dict[str, Any]]:
        """Get experiments within date range.
        
        Args:
            start_date: Start date
            end_date: End date
            
        Returns:
            List of experiments in date range
        """
        if self.db.db_type == "sqlite":
            query = """
            SELECT * FROM experiments 
            WHERE created_at >= ? AND created_at <= ?
            ORDER BY created_at DESC
            """
            params = (start_date.isoformat(), end_date.isoformat())
        else:  # PostgreSQL
            query = """
            SELECT * FROM experiments 
            WHERE created_at >= %s AND created_at <= %s
            ORDER BY created_at DESC
            """
            params = (start_date, end_date)
        
        rows = self.db.execute_query(query, params)
        return [dict(row) for row in rows]


class BenchmarkRunRepository(BaseRepository):
    """Repository for managing benchmark run records."""
    
    def _get_table_name(self) -> str:
        return "benchmark_runs"
    
    def create_run(self,
                   experiment_id: int,
                   agent_name: str,
                   task_name: str,
                   config: Dict[str, Any]) -> Optional[int]:
        """Create a new benchmark run.
        
        Args:
            experiment_id: Parent experiment ID
            agent_name: Name of the agent
            task_name: Name of the task
            config: Run configuration
            
        Returns:
            Run ID or None if failed
        """
        data = {
            "experiment_id": experiment_id,
            "agent_name": agent_name,
            "task_name": task_name,
            "agent_config": json.dumps(config.get("agent", {})) if self.db.db_type == "sqlite" else config.get("agent", {}),
            "task_config": json.dumps(config.get("task", {})) if self.db.db_type == "sqlite" else config.get("task", {}),
            "status": "pending",
            "started_at": datetime.now().isoformat()
        }
        
        return self.create(data)
    
    def update_run_results(self,
                          run_id: int,
                          results: Dict[str, Any]) -> bool:
        """Update run with results.
        
        Args:
            run_id: Run ID
            results: Run results
            
        Returns:
            True if updated successfully
        """
        data = {
            "num_episodes": results.get("num_episodes", 0),
            "success_rate": results.get("success_rate", 0.0),
            "avg_reward": results.get("avg_reward", 0.0),
            "avg_steps": results.get("avg_steps", 0.0),
            "total_time": results.get("total_time", 0.0),
            "results": json.dumps(results) if self.db.db_type == "sqlite" else results,
            "status": "completed",
            "completed_at": datetime.now().isoformat()
        }
        
        return self.update(run_id, data)
    
    def mark_run_failed(self, run_id: int, error_message: str) -> bool:
        """Mark run as failed with error message.
        
        Args:
            run_id: Run ID
            error_message: Error description
            
        Returns:
            True if updated successfully
        """
        data = {
            "status": "failed",
            "error_message": error_message,
            "completed_at": datetime.now().isoformat()
        }
        
        return self.update(run_id, data)
    
    def find_by_experiment(self, experiment_id: int) -> List[Dict[str, Any]]:
        """Find all runs for an experiment.
        
        Args:
            experiment_id: Experiment ID
            
        Returns:
            List of runs
        """
        return self.find_by_field("experiment_id", experiment_id)
    
    def find_by_agent_and_task(self, agent_name: str, task_name: str) -> List[Dict[str, Any]]:
        """Find runs by agent and task.
        
        Args:
            agent_name: Agent name
            task_name: Task name
            
        Returns:
            List of matching runs
        """
        return self.find_by_fields({
            "agent_name": agent_name,
            "task_name": task_name
        })
    
    def get_agent_performance_summary(self, agent_name: str) -> Dict[str, Any]:
        """Get performance summary for an agent across all tasks.
        
        Args:
            agent_name: Agent name
            
        Returns:
            Performance summary
        """
        if self.db.db_type == "sqlite":
            query = """
            SELECT 
                task_name,
                COUNT(*) as total_runs,
                AVG(success_rate) as avg_success_rate,
                AVG(avg_reward) as avg_reward,
                AVG(avg_steps) as avg_steps,
                AVG(total_time) as avg_time
            FROM benchmark_runs 
            WHERE agent_name = ? AND status = 'completed'
            GROUP BY task_name
            """
            params = (agent_name,)
        else:  # PostgreSQL
            query = """
            SELECT 
                task_name,
                COUNT(*) as total_runs,
                AVG(success_rate) as avg_success_rate,
                AVG(avg_reward) as avg_reward,
                AVG(avg_steps) as avg_steps,
                AVG(total_time) as avg_time
            FROM benchmark_runs 
            WHERE agent_name = %s AND status = 'completed'
            GROUP BY task_name
            """
            params = (agent_name,)
        
        rows = self.db.execute_query(query, params)
        
        summary = {
            "agent_name": agent_name,
            "task_performance": [dict(row) for row in rows]
        }
        
        # Calculate overall averages
        if rows:
            summary["overall"] = {
                "total_runs": sum(row["total_runs"] for row in rows),
                "avg_success_rate": sum(row["avg_success_rate"] or 0 for row in rows) / len(rows),
                "avg_reward": sum(row["avg_reward"] or 0 for row in rows) / len(rows),
                "avg_steps": sum(row["avg_steps"] or 0 for row in rows) / len(rows),
                "avg_time": sum(row["avg_time"] or 0 for row in rows) / len(rows)
            }
        
        return summary
    
    def get_task_leaderboard(self, task_name: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Get leaderboard for a specific task.
        
        Args:
            task_name: Task name
            limit: Maximum number of entries
            
        Returns:
            List of top performing agents
        """
        if self.db.db_type == "sqlite":
            query = """
            SELECT 
                agent_name,
                AVG(success_rate) as avg_success_rate,
                AVG(avg_reward) as avg_reward,
                AVG(avg_steps) as avg_steps,
                COUNT(*) as num_runs
            FROM benchmark_runs 
            WHERE task_name = ? AND status = 'completed'
            GROUP BY agent_name
            ORDER BY avg_success_rate DESC, avg_reward DESC
            LIMIT ?
            """
            params = (task_name, limit)
        else:  # PostgreSQL
            query = """
            SELECT 
                agent_name,
                AVG(success_rate) as avg_success_rate,
                AVG(avg_reward) as avg_reward,
                AVG(avg_steps) as avg_steps,
                COUNT(*) as num_runs
            FROM benchmark_runs 
            WHERE task_name = %s AND status = 'completed'
            GROUP BY agent_name
            ORDER BY avg_success_rate DESC, avg_reward DESC
            LIMIT %s
            """
            params = (task_name, limit)
        
        rows = self.db.execute_query(query, params)
        
        leaderboard = []
        for i, row in enumerate(rows):
            entry = dict(row)
            entry["rank"] = i + 1
            leaderboard.append(entry)
        
        return leaderboard