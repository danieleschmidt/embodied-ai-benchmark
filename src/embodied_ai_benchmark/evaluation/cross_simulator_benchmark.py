"""Cross-simulator generalization benchmark for multi-modal embodied AI systems."""

import time
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
import logging

from ..core.base_env import BaseEnv
from ..core.base_agent import BaseAgent
from ..utils.logging_config import get_logger

logger = get_logger(__name__)


class CrossSimulatorBenchmark:
    """Novel benchmark for cross-simulator generalization evaluation.
    
    This benchmark evaluates agents' ability to transfer knowledge and skills
    across different simulation platforms and sensory modalities.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize cross-simulator benchmark.
        
        Args:
            config: Benchmark configuration
        """
        self.config = config or {}
        self.simulators = ['habitat', 'maniskill3', 'isaac_sim', 'genesis']
        self.modalities = ['vision', 'tactile', 'audio', 'proprioception']
        self.adaptation_methods = ['domain_adversarial', 'meta_learning', 'continual_adaptation']
        # Import BenchmarkSuite locally to avoid circular imports
        from .benchmark_suite import BenchmarkSuite
        self.benchmark_suite = BenchmarkSuite(config)
        self._evaluation_history = []
    
    def evaluate_cross_modal_transfer(
        self, 
        agent: BaseAgent, 
        source_env: BaseEnv,
        target_env: BaseEnv,
        modality_pairs: List[Tuple[str, str]],
        num_episodes: int = 50,
        adaptation_episodes: int = 10
    ) -> Dict[str, Any]:
        """Evaluate transfer across simulators and modalities.
        
        Args:
            agent: Agent to evaluate
            source_env: Source environment for training
            target_env: Target environment for evaluation
            modality_pairs: List of (source_modality, target_modality) pairs
            num_episodes: Number of evaluation episodes
            adaptation_episodes: Number of episodes for adaptation
            
        Returns:
            Dictionary containing transfer evaluation results
        """
        logger.info(f"Starting cross-modal transfer evaluation")
        logger.info(f"Source: {source_env.simulator_name}, Target: {target_env.simulator_name}")
        
        results = {
            "source_simulator": source_env.simulator_name,
            "target_simulator": target_env.simulator_name,
            "modality_pairs": modality_pairs,
            "num_episodes": num_episodes,
            "adaptation_episodes": adaptation_episodes,
            "transfer_results": {},
            "baseline_performance": {},
            "adapted_performance": {},
            "transfer_efficiency": {},
            "evaluation_timestamp": datetime.now().isoformat()
        }
        
        # Evaluate baseline performance on source environment
        logger.info("Evaluating baseline performance on source environment")
        baseline_source = self.benchmark_suite.evaluate(
            source_env, agent, num_episodes=num_episodes
        )
        results["baseline_performance"]["source"] = baseline_source
        
        # Evaluate zero-shot transfer to target environment
        logger.info("Evaluating zero-shot transfer to target environment")
        zero_shot_target = self.benchmark_suite.evaluate(
            target_env, agent, num_episodes=num_episodes
        )
        results["baseline_performance"]["target_zero_shot"] = zero_shot_target
        
        # Perform adaptation on target environment
        if adaptation_episodes > 0:
            logger.info(f"Performing adaptation on target environment ({adaptation_episodes} episodes)")
            adaptation_results = self._perform_adaptation(
                agent, target_env, adaptation_episodes
            )
            
            # Evaluate post-adaptation performance
            adapted_target = self.benchmark_suite.evaluate(
                target_env, agent, num_episodes=num_episodes
            )
            results["adapted_performance"]["target"] = adapted_target
            results["adaptation_results"] = adaptation_results
        
        # Calculate transfer metrics
        results["transfer_metrics"] = self._calculate_transfer_metrics(
            baseline_source, zero_shot_target, 
            results.get("adapted_performance", {}).get("target")
        )
        
        # Evaluate modality-specific transfer
        for source_mod, target_mod in modality_pairs:
            mod_results = self._evaluate_modality_transfer(
                agent, source_env, target_env, source_mod, target_mod, num_episodes
            )
            results["transfer_results"][f"{source_mod}_to_{target_mod}"] = mod_results
        
        self._evaluation_history.append(results)
        logger.info("Cross-modal transfer evaluation completed")
        
        return results
    
    def _perform_adaptation(
        self, 
        agent: BaseAgent, 
        target_env: BaseEnv, 
        num_episodes: int
    ) -> Dict[str, Any]:
        """Perform adaptation on target environment.
        
        Args:
            agent: Agent to adapt
            target_env: Target environment
            num_episodes: Number of adaptation episodes
            
        Returns:
            Adaptation results
        """
        adaptation_results = {
            "episodes": [],
            "learning_curve": [],
            "adaptation_efficiency": 0.0
        }
        
        performance_history = []
        
        for episode in range(num_episodes):
            # Run adaptation episode
            episode_result = self._run_adaptation_episode(agent, target_env, episode)
            adaptation_results["episodes"].append(episode_result)
            
            # Track performance improvement
            performance = episode_result.get("total_reward", 0)
            performance_history.append(performance)
            
            # Calculate learning curve
            if len(performance_history) >= 5:
                recent_avg = np.mean(performance_history[-5:])
                adaptation_results["learning_curve"].append(recent_avg)
        
        # Calculate adaptation efficiency
        if len(performance_history) > 1:
            initial_performance = np.mean(performance_history[:3])
            final_performance = np.mean(performance_history[-3:])
            adaptation_results["adaptation_efficiency"] = (
                final_performance - initial_performance
            ) / max(abs(initial_performance), 1e-6)
        
        return adaptation_results
    
    def _run_adaptation_episode(
        self, 
        agent: BaseAgent, 
        env: BaseEnv, 
        episode_id: int
    ) -> Dict[str, Any]:
        """Run single adaptation episode.
        
        Args:
            agent: Agent to run
            env: Environment
            episode_id: Episode identifier
            
        Returns:
            Episode results
        """
        obs = env.reset()
        total_reward = 0
        steps = 0
        done = False
        
        episode_data = {
            "episode_id": episode_id,
            "steps": [],
            "total_reward": 0,
            "total_steps": 0,
            "success": False
        }
        
        while not done and steps < 1000:
            action = agent.act(obs)
            next_obs, reward, done, info = env.step(action)
            
            # Store step data
            step_data = {
                "step": steps,
                "observation": obs,
                "action": action,
                "reward": reward,
                "next_observation": next_obs,
                "done": done,
                "info": info
            }
            episode_data["steps"].append(step_data)
            
            # Update agent (if it supports online learning)
            if hasattr(agent, 'update'):
                agent.update(obs, action, reward, next_obs, done)
            
            obs = next_obs
            total_reward += reward
            steps += 1
        
        episode_data["total_reward"] = total_reward
        episode_data["total_steps"] = steps
        episode_data["success"] = info.get("success", False)
        
        return episode_data
    
    def _evaluate_modality_transfer(
        self, 
        agent: BaseAgent,
        source_env: BaseEnv,
        target_env: BaseEnv,
        source_modality: str,
        target_modality: str,
        num_episodes: int
    ) -> Dict[str, Any]:
        """Evaluate transfer for specific modality pair.
        
        Args:
            agent: Agent to evaluate
            source_env: Source environment
            target_env: Target environment
            source_modality: Source sensory modality
            target_modality: Target sensory modality
            num_episodes: Number of evaluation episodes
            
        Returns:
            Modality-specific transfer results
        """
        logger.info(f"Evaluating {source_modality} to {target_modality} transfer")
        
        # Configure environments for specific modalities
        source_env_config = self._configure_modality(source_env, source_modality)
        target_env_config = self._configure_modality(target_env, target_modality)
        
        # Evaluate performance with modality constraints
        source_performance = self.benchmark_suite.evaluate(
            source_env_config, agent, num_episodes=num_episodes // 2
        )
        
        target_performance = self.benchmark_suite.evaluate(
            target_env_config, agent, num_episodes=num_episodes // 2
        )
        
        # Calculate modality-specific transfer metrics
        transfer_score = self._calculate_modality_transfer_score(
            source_performance, target_performance, source_modality, target_modality
        )
        
        return {
            "source_modality": source_modality,
            "target_modality": target_modality,
            "source_performance": source_performance,
            "target_performance": target_performance,
            "transfer_score": transfer_score,
            "modality_compatibility": self._assess_modality_compatibility(
                source_modality, target_modality
            )
        }
    
    def _configure_modality(
        self, 
        env: BaseEnv, 
        modality: str
    ) -> BaseEnv:
        """Configure environment for specific modality.
        
        Args:
            env: Environment to configure
            modality: Target modality
            
        Returns:
            Configured environment
        """
        # Create modality-specific environment configuration
        # This would typically involve masking or enhancing specific sensor inputs
        if hasattr(env, 'configure_modality'):
            return env.configure_modality(modality)
        else:
            logger.warning(f"Environment {env.simulator_name} doesn't support modality configuration")
            return env
    
    def _calculate_transfer_metrics(
        self, 
        source_results: Dict[str, Any],
        target_zero_shot: Dict[str, Any],
        target_adapted: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Calculate transfer learning metrics.
        
        Args:
            source_results: Results on source environment
            target_zero_shot: Zero-shot results on target environment
            target_adapted: Post-adaptation results on target environment
            
        Returns:
            Transfer metrics
        """
        source_performance = source_results.get("success_rate", 0)
        target_zero_shot_performance = target_zero_shot.get("success_rate", 0)
        
        # Transfer ratio: how much performance is retained
        transfer_ratio = target_zero_shot_performance / max(source_performance, 1e-6)
        
        metrics = {
            "transfer_ratio": transfer_ratio,
            "zero_shot_drop": source_performance - target_zero_shot_performance,
            "relative_performance_drop": 1 - transfer_ratio
        }
        
        if target_adapted:
            target_adapted_performance = target_adapted.get("success_rate", 0)
            adaptation_gain = target_adapted_performance - target_zero_shot_performance
            
            metrics.update({
                "adaptation_gain": adaptation_gain,
                "final_transfer_ratio": target_adapted_performance / max(source_performance, 1e-6),
                "adaptation_effectiveness": adaptation_gain / max(abs(source_performance - target_zero_shot_performance), 1e-6)
            })
        
        return metrics
    
    def _calculate_modality_transfer_score(
        self, 
        source_results: Dict[str, Any],
        target_results: Dict[str, Any],
        source_modality: str,
        target_modality: str
    ) -> float:
        """Calculate transfer score for specific modality pair.
        
        Args:
            source_results: Source modality results
            target_results: Target modality results
            source_modality: Source modality name
            target_modality: Target modality name
            
        Returns:
            Transfer score (0-1)
        """
        source_performance = source_results.get("success_rate", 0)
        target_performance = target_results.get("success_rate", 0)
        
        # Base transfer score
        base_score = target_performance / max(source_performance, 1e-6)
        
        # Modality compatibility adjustment
        compatibility = self._assess_modality_compatibility(source_modality, target_modality)
        
        # Adjusted transfer score
        transfer_score = base_score * compatibility
        
        return min(transfer_score, 1.0)
    
    def _assess_modality_compatibility(
        self, 
        source_modality: str, 
        target_modality: str
    ) -> float:
        """Assess compatibility between modalities.
        
        Args:
            source_modality: Source modality
            target_modality: Target modality
            
        Returns:
            Compatibility score (0-1)
        """
        # Define modality similarity matrix
        similarity_matrix = {
            ('vision', 'vision'): 1.0,
            ('vision', 'tactile'): 0.3,
            ('vision', 'audio'): 0.2,
            ('vision', 'proprioception'): 0.4,
            ('tactile', 'vision'): 0.3,
            ('tactile', 'tactile'): 1.0,
            ('tactile', 'proprioception'): 0.7,
            ('tactile', 'audio'): 0.1,
            ('audio', 'vision'): 0.2,
            ('audio', 'tactile'): 0.1,
            ('audio', 'audio'): 1.0,
            ('audio', 'proprioception'): 0.3,
            ('proprioception', 'vision'): 0.4,
            ('proprioception', 'tactile'): 0.7,
            ('proprioception', 'audio'): 0.3,
            ('proprioception', 'proprioception'): 1.0
        }
        
        return similarity_matrix.get((source_modality, target_modality), 0.5)
    
    def generate_transfer_report(self, results: Dict[str, Any]) -> str:
        """Generate comprehensive transfer evaluation report.
        
        Args:
            results: Transfer evaluation results
            
        Returns:
            Formatted report string
        """
        report = []
        report.append("\n" + "="*60)
        report.append("CROSS-SIMULATOR TRANSFER EVALUATION REPORT")
        report.append("="*60)
        
        # Basic info
        report.append(f"\nEvaluation Timestamp: {results['evaluation_timestamp']}")
        report.append(f"Source Simulator: {results['source_simulator']}")
        report.append(f"Target Simulator: {results['target_simulator']}")
        report.append(f"Episodes: {results['num_episodes']}")
        report.append(f"Adaptation Episodes: {results['adaptation_episodes']}")
        
        # Transfer metrics
        if 'transfer_metrics' in results:
            metrics = results['transfer_metrics']
            report.append("\nTRANSFER METRICS:")
            report.append(f"  Transfer Ratio: {metrics.get('transfer_ratio', 0):.3f}")
            report.append(f"  Zero-shot Performance Drop: {metrics.get('zero_shot_drop', 0):.3f}")
            
            if 'adaptation_gain' in metrics:
                report.append(f"  Adaptation Gain: {metrics.get('adaptation_gain', 0):.3f}")
                report.append(f"  Final Transfer Ratio: {metrics.get('final_transfer_ratio', 0):.3f}")
        
        # Modality-specific results
        if 'transfer_results' in results:
            report.append("\nMODALITY-SPECIFIC TRANSFER:")
            for modality_pair, mod_results in results['transfer_results'].items():
                report.append(f"  {modality_pair}:")
                report.append(f"    Transfer Score: {mod_results.get('transfer_score', 0):.3f}")
                report.append(f"    Compatibility: {mod_results.get('modality_compatibility', 0):.3f}")
        
        report.append("\n" + "="*60)
        
        return "\n".join(report)
    
    def get_evaluation_history(self) -> List[Dict[str, Any]]:
        """Get history of cross-simulator evaluations.
        
        Returns:
            List of evaluation results
        """
        return self._evaluation_history.copy()
    
    def save_results(self, results: Dict[str, Any], filepath: str):
        """Save cross-simulator evaluation results.
        
        Args:
            results: Results to save
            filepath: Output file path
        """
        import json
        
        # Convert numpy arrays for JSON serialization
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
        
        logger.info(f"Cross-simulator evaluation results saved to {filepath}")
