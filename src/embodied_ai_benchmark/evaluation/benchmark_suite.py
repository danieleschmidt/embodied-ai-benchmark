"""Main benchmark suite for coordinating evaluations."""

import time
from typing import Any, Dict, List, Optional, Union
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

from ..core.base_task import BaseTask
from ..core.base_env import BaseEnv
from ..core.base_agent import BaseAgent, RandomAgent
from ..core.base_metric import BaseMetric, SuccessMetric, EfficiencyMetric, SafetyMetric
from .evaluator import Evaluator
from ..utils.logging_config import get_logger
from ..utils.validation import InputValidator, ValidationError, SecurityValidator
from ..utils.monitoring import performance_monitor, health_checker, BenchmarkMetrics

logger = get_logger(__name__)


class BenchmarkSuite:
    """Main benchmark suite for evaluating embodied AI agents."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize benchmark suite.
        
        Args:
            config: Suite configuration dictionary
        """
        self.config = config or {}
        self.tasks = {}
        self.metrics = {}
        self.evaluator = Evaluator(self.config.get("evaluator", {}))
        self._results_history = []
        
        # Setup default metrics
        self._setup_default_metrics()
    
    def _setup_default_metrics(self):
        """Setup default evaluation metrics."""
        default_metrics = [
            SuccessMetric({"name": "success_rate", "weight": 1.0}),
            EfficiencyMetric({"name": "efficiency", "weight": 0.5}),
            SafetyMetric({"name": "safety", "weight": 0.8})
        ]
        
        for metric in default_metrics:
            self.add_metric(metric.get_name(), metric)
    
    def add_task(self, task_name: str, task: BaseTask):
        """Add task to benchmark suite.
        
        Args:
            task_name: Unique task identifier
            task: Task instance
        """
        self.tasks[task_name] = task
    
    def add_metric(self, metric_name: str, metric: BaseMetric):
        """Add metric to benchmark suite.
        
        Args:
            metric_name: Unique metric identifier
            metric: Metric instance
        """
        self.metrics[metric_name] = metric
    
    def get_tasks(self) -> Dict[str, BaseTask]:
        """Get all registered tasks.
        
        Returns:
            Dictionary of task names to task instances
        """
        return self.tasks.copy()
    
    def get_metrics(self) -> Dict[str, BaseMetric]:
        """Get all registered metrics.
        
        Returns:
            Dictionary of metric names to metric instances
        """
        return self.metrics.copy()
    
    def evaluate(self, 
                 env: BaseEnv,
                 agent: BaseAgent,
                 num_episodes: int = 100,
                 max_steps_per_episode: int = 1000,
                 seed: Optional[int] = None,
                 parallel: bool = False,
                 num_workers: int = 4) -> Dict[str, Any]:
        """Evaluate agent on benchmark environment with comprehensive monitoring.
        
        Args:
            env: Environment to evaluate on
            agent: Agent to evaluate
            num_episodes: Number of evaluation episodes
            max_steps_per_episode: Maximum steps per episode
            seed: Random seed for reproducibility
            parallel: Whether to run episodes in parallel
            num_workers: Number of parallel workers
            
        Returns:
            Dictionary of evaluation results
            
        Raises:
            ValidationError: If input parameters are invalid
            RuntimeError: If evaluation fails
        """
        # Input validation
        try:
            self._validate_evaluation_inputs(env, agent, num_episodes, max_steps_per_episode, seed)
        except ValidationError as e:
            logger.error(f"Evaluation input validation failed: {e}")
            raise
        
        # Pre-evaluation health check
        health_status = health_checker.get_overall_health()
        if not health_status["overall_healthy"]:
            logger.warning(f"System health check failed: {health_status['health_percentage']:.1f}% healthy")
            # Continue with warning but log the issues
            for check_name, result in health_status["details"].items():
                if not result["healthy"]:
                    logger.warning(f"Health check '{check_name}' failed: {result['message']}")
        
        # Start performance monitoring
        if not performance_monitor._monitoring_active:
            performance_monitor.start_monitoring()
        
        # Log evaluation start
        agent_name = getattr(agent, 'agent_id', agent.__class__.__name__)
        task_name = getattr(env, 'task_id', 'unknown_task')
        
        logger.info(f"Starting evaluation: {agent_name} on {task_name}")
        logger.info(f"Episodes: {num_episodes}, Max steps: {max_steps_per_episode}, Parallel: {parallel}")
        
        try:
            if seed is not None:
                np.random.seed(seed)
            
            start_time = time.time()
            
            # Execute evaluation with error handling
            if parallel and num_episodes > 1:
                results = self._evaluate_parallel(
                    env, agent, num_episodes, max_steps_per_episode, num_workers
                )
            else:
                results = self._evaluate_sequential(
                    env, agent, num_episodes, max_steps_per_episode
                )
            
            total_time = time.time() - start_time
            
            # Validate results
            if not results:
                raise RuntimeError("Evaluation produced no results")
            
            # Aggregate results with error handling
            try:
                aggregated_results = self._aggregate_results(results, total_time)
            except Exception as e:
                logger.error(f"Result aggregation failed: {e}")
                raise RuntimeError(f"Failed to aggregate evaluation results: {e}") from e
            
            # Add evaluation metadata
            aggregated_results.update({
                "evaluation_id": f"{agent_name}_{task_name}_{int(time.time())}",
                "agent_name": agent_name,
                "task_name": task_name,
                "evaluation_start_time": datetime.fromtimestamp(start_time).isoformat(),
                "evaluation_end_time": datetime.now().isoformat(),
                "seed": seed,
                "parallel": parallel,
                "system_health": health_status
            })
            
            self._results_history.append(aggregated_results)
            
            # Log successful completion
            logger.info(f"Evaluation completed successfully in {total_time:.2f}s")
            logger.info(f"Success rate: {aggregated_results.get('success_rate', 0):.2%}")
            
            return aggregated_results
            
        except Exception as e:
            logger.error(f"Evaluation failed: {e}")
            # Log failure details
            failure_info = {
                "agent_name": agent_name,
                "task_name": task_name,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
            logger.error(f"Evaluation failure details: {failure_info}")
            raise RuntimeError(f"Evaluation failed for {agent_name} on {task_name}: {e}") from e
    
    def _validate_evaluation_inputs(self, env, agent, num_episodes, max_steps_per_episode, seed):
        """Validate evaluation input parameters."""
        if not isinstance(env, BaseEnv):
            raise ValidationError(f"Environment must be BaseEnv instance, got {type(env)}")
        
        if not isinstance(agent, BaseAgent):
            raise ValidationError(f"Agent must be BaseAgent instance, got {type(agent)}")
        
        if not isinstance(num_episodes, int) or num_episodes <= 0:
            raise ValidationError(f"num_episodes must be positive integer, got {num_episodes}")
        
        if not isinstance(max_steps_per_episode, int) or max_steps_per_episode <= 0:
            raise ValidationError(f"max_steps_per_episode must be positive integer, got {max_steps_per_episode}")
        
        if seed is not None and (not isinstance(seed, int) or seed < 0):
            raise ValidationError(f"seed must be non-negative integer or None, got {seed}")
        
        # Resource checks
        try:
            SecurityValidator.check_resource_limits(
                memory_mb=None,  # Will be checked during execution
                cpu_percent=None,
                gpu_memory_mb=None
            )
        except ValidationError as e:
            logger.warning(f"Resource limit warning: {e}")
            # Don't fail evaluation, just warn
    
    def _evaluate_sequential(self, 
                           env: BaseEnv, 
                           agent: BaseAgent, 
                           num_episodes: int, 
                           max_steps_per_episode: int) -> List[Dict[str, Any]]:
        """Run sequential evaluation episodes.
        
        Args:
            env: Environment instance
            agent: Agent instance
            num_episodes: Number of episodes
            max_steps_per_episode: Max steps per episode
            
        Returns:
            List of episode results
        """
        episode_results = []
        
        for episode in range(num_episodes):
            episode_result = self.evaluator.run_episode(
                env, agent, max_steps_per_episode, episode_id=episode
            )
            episode_results.append(episode_result)
            
            # Compute metrics for this episode
            episode_metrics = {}
            for metric_name, metric in self.metrics.items():
                metric.reset()
                
                # Update metric with episode data
                for step_data in episode_result["steps"]:
                    metric.update(
                        step_data["observation"],
                        step_data["action"],
                        step_data["reward"],
                        step_data["next_observation"],
                        step_data["done"],
                        step_data["info"]
                    )
                
                episode_metrics[metric_name] = metric.compute()
            
            episode_result["metrics"] = episode_metrics
        
        return episode_results
    
    def _evaluate_parallel(self, 
                          env: BaseEnv, 
                          agent: BaseAgent, 
                          num_episodes: int, 
                          max_steps_per_episode: int,
                          num_workers: int) -> List[Dict[str, Any]]:
        """Run parallel evaluation episodes.
        
        Args:
            env: Environment instance
            agent: Agent instance
            num_episodes: Number of episodes
            max_steps_per_episode: Max steps per episode
            num_workers: Number of parallel workers
            
        Returns:
            List of episode results
        """
        episode_results = []
        
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            # Submit all episodes
            future_to_episode = {
                executor.submit(
                    self.evaluator.run_episode,
                    env, agent, max_steps_per_episode, episode
                ): episode for episode in range(num_episodes)
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_episode):
                episode_id = future_to_episode[future]
                try:
                    episode_result = future.result()
                    
                    # Compute metrics for this episode
                    episode_metrics = {}
                    for metric_name, metric in self.metrics.items():
                        metric.reset()
                        
                        for step_data in episode_result["steps"]:
                            metric.update(
                                step_data["observation"],
                                step_data["action"],
                                step_data["reward"],
                                step_data["next_observation"],
                                step_data["done"],
                                step_data["info"]
                            )
                        
                        episode_metrics[metric_name] = metric.compute()
                    
                    episode_result["metrics"] = episode_metrics
                    episode_results.append(episode_result)
                    
                except Exception as exc:
                    print(f"Episode {episode_id} generated exception: {exc}")
        
        # Sort results by episode ID
        episode_results.sort(key=lambda x: x["episode_id"])
        return episode_results
    
    def _aggregate_results(self, 
                          episode_results: List[Dict[str, Any]], 
                          total_time: float) -> Dict[str, Any]:
        """Aggregate results across all episodes.
        
        Args:
            episode_results: List of episode results
            total_time: Total evaluation time
            
        Returns:
            Aggregated results dictionary
        """
        if not episode_results:
            return {"error": "No episode results to aggregate"}
        
        # Aggregate metrics
        aggregated_metrics = {}
        for metric_name in self.metrics.keys():
            metric_values = [ep["metrics"][metric_name] for ep in episode_results 
                           if metric_name in ep["metrics"]]
            
            if metric_values:
                aggregated_metrics[metric_name] = {
                    "mean": np.mean(metric_values),
                    "std": np.std(metric_values),
                    "min": np.min(metric_values),
                    "max": np.max(metric_values),
                    "values": metric_values
                }
        
        # Aggregate episode statistics
        episode_lengths = [ep["total_steps"] for ep in episode_results]
        episode_rewards = [ep["total_reward"] for ep in episode_results]
        success_count = sum(1 for ep in episode_results if ep.get("success", False))
        
        return {
            "num_episodes": len(episode_results),
            "total_time": total_time,
            "avg_episode_time": total_time / len(episode_results),
            "success_rate": success_count / len(episode_results),
            "avg_steps": np.mean(episode_lengths),
            "std_steps": np.std(episode_lengths),
            "avg_reward": np.mean(episode_rewards),
            "std_reward": np.std(episode_rewards),
            "metrics": aggregated_metrics,
            "episodes": episode_results
        }
    
    def get_training_tasks(self, 
                          difficulty_range: tuple = (0.1, 0.8),
                          num_tasks: int = 100) -> List[BaseTask]:
        """Get tasks suitable for training.
        
        Args:
            difficulty_range: Min and max difficulty levels
            num_tasks: Number of tasks to return
            
        Returns:
            List of training tasks
        """
        training_tasks = []
        available_tasks = list(self.tasks.values())
        
        # Filter by difficulty if tasks have difficulty attribute
        filtered_tasks = []
        for task in available_tasks:
            if hasattr(task, 'difficulty'):
                if isinstance(task.difficulty, str):
                    # Convert string difficulty to numeric
                    diff_map = {"easy": 0.3, "medium": 0.6, "hard": 0.9}
                    diff_val = diff_map.get(task.difficulty, 0.5)
                else:
                    diff_val = task.difficulty
                
                if difficulty_range[0] <= diff_val <= difficulty_range[1]:
                    filtered_tasks.append(task)
            else:
                filtered_tasks.append(task)
        
        # Sample tasks up to num_tasks
        if len(filtered_tasks) > num_tasks:
            indices = np.random.choice(len(filtered_tasks), num_tasks, replace=False)
            training_tasks = [filtered_tasks[i] for i in indices]
        else:
            training_tasks = filtered_tasks
        
        return training_tasks
    
    def get_eval_tasks(self) -> List[BaseTask]:
        """Get tasks suitable for evaluation.
        
        Returns:
            List of evaluation tasks
        """
        return list(self.tasks.values())
    
    def get_all_tasks(self) -> List[BaseTask]:
        """Get all available tasks.
        
        Returns:
            List of all tasks
        """
        return list(self.tasks.values())
    
    def get_results_history(self) -> List[Dict[str, Any]]:
        """Get history of evaluation results.
        
        Returns:
            List of historical results
        """
        return self._results_history.copy()
    
    def save_results(self, filepath: str, results: Optional[Dict[str, Any]] = None):
        """Save evaluation results to file.
        
        Args:
            filepath: Path to save results
            results: Results to save (uses latest if None)
        """
        import json
        
        if results is None:
            if self._results_history:
                results = self._results_history[-1]
            else:
                raise ValueError("No results to save")
        
        # Convert numpy arrays to lists for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, dict):
                return {key: convert_numpy(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(item) for item in obj]
            else:
                return obj
        
        serializable_results = convert_numpy(results)
        
        with open(filepath, 'w') as f:
            json.dump(serializable_results, f, indent=2)