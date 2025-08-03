"""Test data factories for creating test objects."""

import random
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional
from faker import Faker

fake = Faker()


class ExperimentFactory:
    """Factory for creating experiment test data."""
    
    @staticmethod
    def create(name: Optional[str] = None,
               description: Optional[str] = None,
               config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Create experiment data.
        
        Args:
            name: Experiment name
            description: Experiment description
            config: Experiment configuration
            
        Returns:
            Experiment data dictionary
        """
        return {
            "name": name or f"experiment_{fake.uuid4()[:8]}",
            "description": description or fake.text(max_nb_chars=200),
            "config": config or {
                "agent_type": random.choice(["random", "scripted", "rl"]),
                "num_episodes": random.randint(10, 100),
                "tasks": random.sample([
                    "FurnitureAssembly-v0",
                    "PointGoal-v0", 
                    "CooperativeFurnitureAssembly-v0"
                ], k=random.randint(1, 3)),
                "metrics": ["success_rate", "efficiency", "safety"]
            },
            "status": random.choice(["pending", "running", "completed", "failed"]),
            "created_at": fake.date_time_between(start_date="-30d", end_date="now").isoformat(),
            "updated_at": fake.date_time_between(start_date="-7d", end_date="now").isoformat()
        }
    
    @staticmethod
    def create_batch(count: int = 5) -> List[Dict[str, Any]]:
        """Create batch of experiments.
        
        Args:
            count: Number of experiments to create
            
        Returns:
            List of experiment data dictionaries
        """
        return [ExperimentFactory.create() for _ in range(count)]


class BenchmarkRunFactory:
    """Factory for creating benchmark run test data."""
    
    @staticmethod
    def create(experiment_id: Optional[int] = None,
               agent_name: Optional[str] = None,
               task_name: Optional[str] = None,
               status: Optional[str] = None) -> Dict[str, Any]:
        """Create benchmark run data.
        
        Args:
            experiment_id: Parent experiment ID
            agent_name: Agent name
            task_name: Task name
            status: Run status
            
        Returns:
            Benchmark run data dictionary
        """
        task_names = ["FurnitureAssembly-v0", "PointGoal-v0", "CooperativeFurnitureAssembly-v0"]
        agent_names = ["RandomAgent", "ScriptedAgent", "RLAgent", "GPTAgent"]
        
        run_status = status or random.choice(["pending", "running", "completed", "failed"])
        
        base_data = {
            "experiment_id": experiment_id or random.randint(1, 100),
            "agent_name": agent_name or random.choice(agent_names),
            "task_name": task_name or random.choice(task_names),
            "status": run_status,
            "started_at": fake.date_time_between(start_date="-7d", end_date="now").isoformat()
        }
        
        # Add results if completed
        if run_status == "completed":
            base_data.update({
                "num_episodes": random.randint(10, 200),
                "success_rate": round(random.uniform(0.0, 1.0), 3),
                "avg_reward": round(random.uniform(-10.0, 100.0), 2),
                "avg_steps": round(random.uniform(50.0, 500.0), 1),
                "total_time": round(random.uniform(60.0, 3600.0), 2),
                "completed_at": fake.date_time_between(start_date="-1d", end_date="now").isoformat(),
                "results": {
                    "metrics": {
                        "success_rate": {
                            "mean": round(random.uniform(0.0, 1.0), 3),
                            "std": round(random.uniform(0.0, 0.3), 3)
                        },
                        "efficiency": {
                            "mean": round(random.uniform(0.0, 1.0), 3),
                            "std": round(random.uniform(0.0, 0.3), 3)
                        }
                    }
                }
            })
        elif run_status == "failed":
            base_data["error_message"] = fake.sentence()
        
        return base_data
    
    @staticmethod
    def create_batch(count: int = 10, experiment_id: Optional[int] = None) -> List[Dict[str, Any]]:
        """Create batch of benchmark runs.
        
        Args:
            count: Number of runs to create
            experiment_id: Parent experiment ID for all runs
            
        Returns:
            List of benchmark run data dictionaries
        """
        return [BenchmarkRunFactory.create(experiment_id=experiment_id) for _ in range(count)]


class EpisodeFactory:
    """Factory for creating episode test data."""
    
    @staticmethod
    def create(run_id: Optional[int] = None,
               episode_id: Optional[int] = None,
               success: Optional[bool] = None) -> Dict[str, Any]:
        """Create episode data.
        
        Args:
            run_id: Parent run ID
            episode_id: Episode number
            success: Episode success status
            
        Returns:
            Episode data dictionary
        """
        is_success = success if success is not None else random.choice([True, False])
        
        return {
            "run_id": run_id or random.randint(1, 100),
            "episode_id": episode_id or random.randint(0, 199),
            "success": is_success,
            "total_steps": random.randint(50, 500),
            "total_reward": round(random.uniform(-10.0, 100.0), 2),
            "completion_time": round(random.uniform(5.0, 300.0), 2),
            "trajectory_data": {
                "states": [f"state_{i}" for i in range(random.randint(5, 20))],
                "actions": [f"action_{i}" for i in range(random.randint(5, 20))],
                "rewards": [round(random.uniform(-1, 5), 2) for _ in range(random.randint(5, 20))]
            },
            "metrics": {
                "success_rate": 1.0 if is_success else 0.0,
                "efficiency": round(random.uniform(0.0, 1.0), 3),
                "safety": round(random.uniform(0.8, 1.0), 3)
            },
            "created_at": fake.date_time_between(start_date="-1d", end_date="now").isoformat()
        }
    
    @staticmethod
    def create_batch(count: int = 20, run_id: Optional[int] = None) -> List[Dict[str, Any]]:
        """Create batch of episodes.
        
        Args:
            count: Number of episodes to create
            run_id: Parent run ID for all episodes
            
        Returns:
            List of episode data dictionaries
        """
        return [EpisodeFactory.create(run_id=run_id, episode_id=i) for i in range(count)]


class AgentPerformanceFactory:
    """Factory for creating agent performance test data."""
    
    @staticmethod
    def create(agent_name: Optional[str] = None,
               task_name: Optional[str] = None,
               metric_name: Optional[str] = None) -> Dict[str, Any]:
        """Create agent performance data.
        
        Args:
            agent_name: Agent name
            task_name: Task name
            metric_name: Metric name
            
        Returns:
            Agent performance data dictionary
        """
        agent_names = ["RandomAgent", "ScriptedAgent", "RLAgent", "GPTAgent"]
        task_names = ["FurnitureAssembly-v0", "PointGoal-v0", "CooperativeFurnitureAssembly-v0"]
        metric_names = ["success_rate", "efficiency", "safety", "collaboration"]
        
        return {
            "agent_name": agent_name or random.choice(agent_names),
            "task_name": task_name or random.choice(task_names),
            "metric_name": metric_name or random.choice(metric_names),
            "metric_value": round(random.uniform(0.0, 1.0), 3),
            "measurement_time": fake.date_time_between(start_date="-7d", end_date="now").isoformat(),
            "metadata": {
                "num_episodes": random.randint(10, 100),
                "confidence_interval": [round(random.uniform(0.0, 0.5), 3), round(random.uniform(0.5, 1.0), 3)],
                "standard_deviation": round(random.uniform(0.0, 0.3), 3)
            }
        }
    
    @staticmethod
    def create_batch(count: int = 50) -> List[Dict[str, Any]]:
        """Create batch of agent performance data.
        
        Args:
            count: Number of performance records to create
            
        Returns:
            List of agent performance data dictionaries
        """
        return [AgentPerformanceFactory.create() for _ in range(count)]


class TaskMetadataFactory:
    """Factory for creating task metadata test data."""
    
    @staticmethod
    def create(task_name: Optional[str] = None,
               task_type: Optional[str] = None,
               difficulty: Optional[str] = None) -> Dict[str, Any]:
        """Create task metadata.
        
        Args:
            task_name: Task name
            task_type: Task type
            difficulty: Difficulty level
            
        Returns:
            Task metadata dictionary
        """
        task_names = ["FurnitureAssembly-v0", "PointGoal-v0", "CooperativeFurnitureAssembly-v0"]
        task_types = ["manipulation", "navigation", "multiagent"]
        difficulties = ["easy", "medium", "hard"]
        
        return {
            "task_name": task_name or random.choice(task_names),
            "task_type": task_type or random.choice(task_types),
            "difficulty": difficulty or random.choice(difficulties),
            "description": fake.text(max_nb_chars=300),
            "requirements": {
                "min_agents": random.randint(1, 4),
                "max_agents": random.randint(1, 8),
                "time_limit": random.randint(60, 600),
                "success_criteria": fake.sentence()
            },
            "created_at": fake.date_time_between(start_date="-30d", end_date="now").isoformat()
        }
    
    @staticmethod
    def create_batch(count: int = 10) -> List[Dict[str, Any]]:
        """Create batch of task metadata.
        
        Args:
            count: Number of task metadata records to create
            
        Returns:
            List of task metadata dictionaries
        """
        return [TaskMetadataFactory.create() for _ in range(count)]


class ConfigFactory:
    """Factory for creating configuration test data."""
    
    @staticmethod
    def create_agent_config(agent_type: str = "random") -> Dict[str, Any]:
        """Create agent configuration.
        
        Args:
            agent_type: Type of agent
            
        Returns:
            Agent configuration dictionary
        """
        base_config = {
            "agent_id": f"{agent_type}_agent_{fake.uuid4()[:8]}",
            "type": agent_type,
            "action_space": {
                "type": "continuous",
                "shape": (7,),
                "low": [-1.0] * 7,
                "high": [1.0] * 7
            },
            "observation_space": {
                "type": "dict",
                "spaces": {
                    "rgb": {"shape": (480, 640, 3), "dtype": "uint8"},
                    "depth": {"shape": (480, 640), "dtype": "float32"},
                    "joints": {"shape": (7,), "dtype": "float32"}
                }
            }
        }
        
        if agent_type == "rl":
            base_config["model"] = {
                "algorithm": random.choice(["PPO", "SAC", "TD3"]),
                "learning_rate": random.uniform(1e-5, 1e-3),
                "batch_size": random.choice([32, 64, 128, 256])
            }
        elif agent_type == "scripted":
            base_config["script"] = {
                "strategy": random.choice(["conservative", "aggressive", "balanced"]),
                "safety_margin": random.uniform(0.1, 0.5)
            }
        
        return base_config
    
    @staticmethod
    def create_task_config(task_type: str = "manipulation") -> Dict[str, Any]:
        """Create task configuration.
        
        Args:
            task_type: Type of task
            
        Returns:
            Task configuration dictionary
        """
        base_config = {
            "name": f"{task_type}_task_{fake.uuid4()[:8]}",
            "type": task_type,
            "max_steps": random.randint(100, 1000),
            "time_limit": random.randint(60, 600),
            "success_threshold": random.uniform(0.8, 1.0)
        }
        
        if task_type == "manipulation":
            base_config["objects"] = random.sample([
                "chair", "table", "shelf", "cabinet", "stool"
            ], k=random.randint(1, 3))
            base_config["tools"] = random.sample([
                "screwdriver", "wrench", "hammer", "drill"
            ], k=random.randint(0, 2))
        elif task_type == "navigation":
            base_config["map_size"] = (random.randint(10, 50), random.randint(10, 50))
            base_config["obstacles"] = random.randint(0, 20)
            base_config["goal_tolerance"] = random.uniform(0.1, 1.0)
        elif task_type == "multiagent":
            base_config["num_agents"] = random.randint(2, 8)
            base_config["cooperation_required"] = random.choice([True, False])
            base_config["communication_enabled"] = random.choice([True, False])
        
        return base_config
    
    @staticmethod
    def create_benchmark_config() -> Dict[str, Any]:
        """Create benchmark configuration.
        
        Returns:
            Benchmark configuration dictionary
        """
        return {
            "num_episodes": random.randint(10, 200),
            "max_steps_per_episode": random.randint(100, 1000),
            "timeout": random.randint(60, 300),
            "parallel": random.choice([True, False]),
            "num_workers": random.randint(1, 8),
            "metrics": random.sample([
                "success_rate", "efficiency", "safety", "collaboration", "diversity"
            ], k=random.randint(2, 5)),
            "record_trajectories": random.choice([True, False]),
            "save_results": True,
            "results_format": random.choice(["json", "hdf5", "pickle"])
        }
