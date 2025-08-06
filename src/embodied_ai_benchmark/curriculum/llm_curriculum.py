"""LLM-guided curriculum learning for adaptive task generation."""

import json
import time
import numpy as np
from typing import Any, Dict, List, Optional, Tuple, Union
from datetime import datetime
from dataclasses import dataclass, asdict

from ..core.base_task import BaseTask
from ..core.base_agent import BaseAgent
from ..utils.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class PerformanceAnalysis:
    """Analysis of agent performance for curriculum adaptation."""
    success_rate: float
    average_steps: float
    efficiency_score: float
    learning_velocity: float
    failure_modes: List[str]
    skill_gaps: List[str]
    mastery_indicators: List[str]
    recommended_difficulty: str
    confidence_level: float


class LLMCurriculum:
    """LLM-guided curriculum learning system."""
    
    def __init__(self, 
                 llm_model: str = "gpt-4",
                 student_model: Optional[BaseAgent] = None,
                 domain: str = "manipulation",
                 adaptation_threshold: float = 0.2):
        """Initialize LLM curriculum system.
        
        Args:
            llm_model: LLM model identifier for curriculum decisions
            student_model: Agent being trained
            domain: Task domain (manipulation, navigation, etc.)
            adaptation_threshold: Performance change threshold for adaptation
        """
        self.llm_model = llm_model
        self.student_model = student_model
        self.domain = domain
        self.adaptation_threshold = adaptation_threshold
        
        # Curriculum state
        self.current_level = 1
        self.performance_history = []
        self.task_progression = []
        self.adaptation_log = []
        
        # Learning parameters
        self.min_episodes_per_task = 10
        self.max_episodes_per_task = 100
        self.target_success_rate = 0.75
        
        # LLM interaction tracking
        self.llm_queries = 0
        self.llm_decisions = []
        
    def analyze_performance(self, episode_results: List[Dict[str, Any]]) -> PerformanceAnalysis:
        """Analyze agent performance using LLM-guided analysis.
        
        Args:
            episode_results: List of episode result dictionaries
            
        Returns:
            Performance analysis with curriculum recommendations
        """
        if not episode_results:
            return PerformanceAnalysis(
                success_rate=0.0, average_steps=0.0, efficiency_score=0.0,
                learning_velocity=0.0, failure_modes=[], skill_gaps=[],
                mastery_indicators=[], recommended_difficulty="easy",
                confidence_level=0.0
            )
        
        # Compute basic metrics
        success_rate = np.mean([ep.get("success", False) for ep in episode_results])
        average_steps = np.mean([ep.get("total_steps", 0) for ep in episode_results])
        rewards = [ep.get("total_reward", 0) for ep in episode_results]
        efficiency_score = np.mean(rewards) / max(average_steps, 1)
        
        # Compute learning velocity (improvement over time)
        learning_velocity = 0.0
        if len(episode_results) > 1:
            early_rewards = np.mean(rewards[:len(rewards)//2])
            late_rewards = np.mean(rewards[len(rewards)//2:])
            learning_velocity = (late_rewards - early_rewards) / max(abs(early_rewards), 1)
        
        # Analyze failure modes
        failure_modes = self._analyze_failure_modes(episode_results)
        
        # Identify skill gaps
        skill_gaps = self._identify_skill_gaps(episode_results, success_rate)
        
        # Detect mastery indicators
        mastery_indicators = self._detect_mastery_indicators(
            episode_results, success_rate, efficiency_score
        )
        
        # LLM-guided difficulty recommendation
        difficulty_recommendation = self._llm_recommend_difficulty(
            success_rate, efficiency_score, learning_velocity, failure_modes, skill_gaps
        )
        
        # Compute confidence in analysis
        confidence = min(1.0, len(episode_results) / 20.0)  # More episodes = higher confidence
        
        return PerformanceAnalysis(
            success_rate=success_rate,
            average_steps=average_steps,
            efficiency_score=efficiency_score,
            learning_velocity=learning_velocity,
            failure_modes=failure_modes,
            skill_gaps=skill_gaps,
            mastery_indicators=mastery_indicators,
            recommended_difficulty=difficulty_recommendation,
            confidence_level=confidence
        )
    
    def _analyze_failure_modes(self, episode_results: List[Dict[str, Any]]) -> List[str]:
        """Analyze common failure patterns."""
        failure_modes = []
        failed_episodes = [ep for ep in episode_results if not ep.get("success", False)]
        
        if not failed_episodes:
            return failure_modes
        
        # Analyze failure reasons
        timeout_failures = sum(1 for ep in failed_episodes 
                             if ep.get("total_steps", 0) >= ep.get("max_steps", 1000))
        collision_failures = sum(1 for ep in failed_episodes 
                               if ep.get("info", {}).get("collision", False))
        precision_failures = sum(1 for ep in failed_episodes 
                               if "precision" in str(ep.get("info", {})).lower())
        
        if timeout_failures / len(failed_episodes) > 0.5:
            failure_modes.append("timeout_exceeded")
        if collision_failures / len(failed_episodes) > 0.3:
            failure_modes.append("collision_prone")
        if precision_failures / len(failed_episodes) > 0.4:
            failure_modes.append("precision_issues")
        
        # Check for low-reward episodes
        low_reward_episodes = sum(1 for ep in episode_results 
                                if ep.get("total_reward", 0) < -10)
        if low_reward_episodes / len(episode_results) > 0.3:
            failure_modes.append("poor_reward_accumulation")
        
        return failure_modes
    
    def _identify_skill_gaps(self, episode_results: List[Dict[str, Any]], success_rate: float) -> List[str]:
        """Identify specific skill gaps based on performance patterns."""
        skill_gaps = []
        
        if success_rate < 0.3:
            skill_gaps.append("fundamental_task_understanding")
        elif success_rate < 0.6:
            skill_gaps.append("execution_consistency")
        
        # Analyze step efficiency
        steps_data = [ep.get("total_steps", 0) for ep in episode_results]
        if np.std(steps_data) / (np.mean(steps_data) + 1) > 0.5:
            skill_gaps.append("strategy_inconsistency")
        
        # Check for learning plateau
        if len(episode_results) >= 10:
            recent_success = np.mean([ep.get("success", False) 
                                    for ep in episode_results[-5:]])
            earlier_success = np.mean([ep.get("success", False) 
                                     for ep in episode_results[-10:-5]])
            if abs(recent_success - earlier_success) < 0.1:
                skill_gaps.append("learning_plateau")
        
        return skill_gaps
    
    def _detect_mastery_indicators(self, 
                                 episode_results: List[Dict[str, Any]], 
                                 success_rate: float, 
                                 efficiency_score: float) -> List[str]:
        """Detect indicators of skill mastery."""
        mastery_indicators = []
        
        if success_rate > 0.8:
            mastery_indicators.append("high_success_rate")
        
        if efficiency_score > 0.5:
            mastery_indicators.append("efficient_execution")
        
        # Check for consistent performance
        success_data = [ep.get("success", False) for ep in episode_results[-10:]]
        if len(success_data) >= 5 and np.std(success_data) < 0.3:
            mastery_indicators.append("consistent_performance")
        
        # Fast completion indicator
        steps_data = [ep.get("total_steps", 0) for ep in episode_results[-5:]]
        if steps_data and np.mean(steps_data) < np.mean([ep.get("max_steps", 1000) 
                                                        for ep in episode_results]) * 0.6:
            mastery_indicators.append("fast_completion")
        
        return mastery_indicators
    
    def _llm_recommend_difficulty(self, 
                                success_rate: float, 
                                efficiency_score: float,
                                learning_velocity: float,
                                failure_modes: List[str],
                                skill_gaps: List[str]) -> str:
        """Use LLM to recommend difficulty adjustment."""
        self.llm_queries += 1
        
        # Create context for LLM
        context = {
            "success_rate": success_rate,
            "efficiency_score": efficiency_score,
            "learning_velocity": learning_velocity,
            "failure_modes": failure_modes,
            "skill_gaps": skill_gaps,
            "current_level": self.current_level,
            "domain": self.domain
        }
        
        # Simulate LLM reasoning (in production, this would call actual LLM)
        recommendation = self._simulate_llm_reasoning(context)
        
        # Log LLM decision
        self.llm_decisions.append({
            "timestamp": datetime.now().isoformat(),
            "context": context,
            "recommendation": recommendation,
            "reasoning": self._get_reasoning(context, recommendation)
        })
        
        return recommendation
    
    def _simulate_llm_reasoning(self, context: Dict[str, Any]) -> str:
        """Simulate LLM reasoning for curriculum decisions."""
        success_rate = context["success_rate"]
        efficiency = context["efficiency_score"]
        learning_vel = context["learning_velocity"]
        failure_modes = context["failure_modes"]
        skill_gaps = context["skill_gaps"]
        
        # Decision logic based on multiple factors
        if success_rate > 0.85 and efficiency > 0.6 and learning_vel > 0.1:
            return "increase_difficulty"
        elif success_rate < 0.4 or "fundamental_task_understanding" in skill_gaps:
            return "decrease_difficulty"
        elif success_rate > 0.75 and "fast_completion" in context.get("mastery_indicators", []):
            return "increase_complexity"
        elif "learning_plateau" in skill_gaps:
            return "vary_task_type"
        elif len(failure_modes) > 2:
            return "focus_on_weaknesses"
        else:
            return "maintain_level"
    
    def _get_reasoning(self, context: Dict[str, Any], recommendation: str) -> str:
        """Generate reasoning explanation for the recommendation."""
        success_rate = context["success_rate"]
        efficiency = context["efficiency_score"]
        
        reasoning_map = {
            "increase_difficulty": f"High success rate ({success_rate:.2f}) and efficiency ({efficiency:.2f}) indicate readiness for greater challenge",
            "decrease_difficulty": f"Low success rate ({success_rate:.2f}) suggests current level is too challenging",
            "increase_complexity": "Consistent mastery indicators suggest readiness for more complex scenarios",
            "vary_task_type": "Learning plateau detected - introducing task variety to stimulate learning",
            "focus_on_weaknesses": f"Multiple failure modes detected: {context['failure_modes']} - targeted practice needed",
            "maintain_level": "Performance metrics suggest current level is appropriately challenging"
        }
        
        return reasoning_map.get(recommendation, "Standard curriculum progression")
    
    def generate_next_task(self, 
                          current_task: BaseTask, 
                          performance_analysis: PerformanceAnalysis) -> Dict[str, Any]:
        """Generate next task configuration based on performance analysis.
        
        Args:
            current_task: Current task being practiced
            performance_analysis: Analysis of recent performance
            
        Returns:
            Configuration dictionary for next task
        """
        recommendation = performance_analysis.recommended_difficulty
        base_config = current_task.config.copy()
        
        # Adjust task parameters based on LLM recommendation
        if recommendation == "increase_difficulty":
            return self._increase_difficulty(base_config)
        elif recommendation == "decrease_difficulty":
            return self._decrease_difficulty(base_config)
        elif recommendation == "increase_complexity":
            return self._increase_complexity(base_config)
        elif recommendation == "vary_task_type":
            return self._vary_task_type(base_config)
        elif recommendation == "focus_on_weaknesses":
            return self._focus_on_weaknesses(base_config, performance_analysis.failure_modes)
        else:  # maintain_level
            return self._add_minor_variation(base_config)
    
    def _increase_difficulty(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Increase task difficulty."""
        new_config = config.copy()
        
        # Reduce time and step limits
        new_config["max_steps"] = int(config.get("max_steps", 1000) * 0.8)
        new_config["time_limit"] = int(config.get("time_limit", 300) * 0.9)
        
        # Increase precision requirements
        if "precision_tolerance" in config:
            new_config["precision_tolerance"] = config["precision_tolerance"] * 0.8
        
        # Adjust curriculum level
        new_config["curriculum_level"] = config.get("curriculum_level", 1) + 1
        
        logger.info(f"Increased difficulty: steps {config.get('max_steps')} -> {new_config['max_steps']}")
        
        return new_config
    
    def _decrease_difficulty(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Decrease task difficulty."""
        new_config = config.copy()
        
        # Increase time and step limits
        new_config["max_steps"] = int(config.get("max_steps", 1000) * 1.3)
        new_config["time_limit"] = int(config.get("time_limit", 300) * 1.2)
        
        # Relax precision requirements
        if "precision_tolerance" in config:
            new_config["precision_tolerance"] = config["precision_tolerance"] * 1.5
        
        # Reduce curriculum level
        new_config["curriculum_level"] = max(1, config.get("curriculum_level", 1) - 1)
        
        logger.info(f"Decreased difficulty: steps {config.get('max_steps')} -> {new_config['max_steps']}")
        
        return new_config
    
    def _increase_complexity(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Increase task complexity without changing basic difficulty."""
        new_config = config.copy()
        
        # Add more objects or constraints
        if "num_objects" in config:
            new_config["num_objects"] = config["num_objects"] + 1
        
        # Add distractors
        new_config["distractors"] = config.get("distractors", 0) + 2
        
        # Enable physics complications
        new_config["physics_complexity"] = "high"
        
        logger.info("Increased task complexity with additional objects and constraints")
        
        return new_config
    
    def _vary_task_type(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Introduce task variation to break learning plateau."""
        new_config = config.copy()
        
        # Change task variant
        furniture_types = ["table", "chair", "shelf", "desk"]
        if config.get("furniture_type") in furniture_types:
            current_idx = furniture_types.index(config["furniture_type"])
            new_idx = (current_idx + 1) % len(furniture_types)
            new_config["furniture_type"] = furniture_types[new_idx]
        
        # Add environmental variation
        new_config["environment_variation"] = True
        new_config["lighting_conditions"] = np.random.choice(["bright", "dim", "variable"])
        
        logger.info(f"Varied task type: {config.get('furniture_type')} -> {new_config.get('furniture_type')}")
        
        return new_config
    
    def _focus_on_weaknesses(self, config: Dict[str, Any], failure_modes: List[str]) -> Dict[str, Any]:
        """Create task that focuses on addressing specific weaknesses."""
        new_config = config.copy()
        
        if "collision_prone" in failure_modes:
            new_config["collision_penalty"] = -10.0
            new_config["collision_detection"] = "strict"
        
        if "timeout_exceeded" in failure_modes:
            new_config["time_pressure"] = True
            new_config["efficiency_bonus"] = 5.0
        
        if "precision_issues" in failure_modes:
            new_config["precision_training"] = True
            new_config["alignment_guidance"] = True
        
        logger.info(f"Focusing on weaknesses: {failure_modes}")
        
        return new_config
    
    def _add_minor_variation(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Add minor variations while maintaining difficulty level."""
        new_config = config.copy()
        
        # Small random variations
        if "max_steps" in config:
            variation = np.random.randint(-50, 51)
            new_config["max_steps"] = max(100, config["max_steps"] + variation)
        
        # Add small environmental changes
        new_config["surface_friction"] = np.random.uniform(0.8, 1.2)
        
        return new_config
    
    def should_progress(self, performance_history: List[PerformanceAnalysis]) -> bool:
        """Determine if agent should progress to next curriculum level."""
        if len(performance_history) < 3:
            return False
        
        # Check recent performance
        recent_performance = performance_history[-3:]
        avg_success = np.mean([p.success_rate for p in recent_performance])
        avg_efficiency = np.mean([p.efficiency_score for p in recent_performance])
        
        # Progress criteria
        return (avg_success >= self.target_success_rate and 
                avg_efficiency > 0.4 and
                all(p.confidence_level > 0.5 for p in recent_performance))
    
    def get_curriculum_state(self) -> Dict[str, Any]:
        """Get current curriculum learning state."""
        return {
            "current_level": self.current_level,
            "domain": self.domain,
            "performance_history_length": len(self.performance_history),
            "task_progression_length": len(self.task_progression),
            "llm_queries_made": self.llm_queries,
            "adaptation_threshold": self.adaptation_threshold,
            "target_success_rate": self.target_success_rate,
            "recent_adaptations": self.adaptation_log[-5:] if self.adaptation_log else []
        }
    
    def save_curriculum_state(self, filepath: str):
        """Save curriculum state to file."""
        state = {
            "curriculum_config": {
                "llm_model": self.llm_model,
                "domain": self.domain,
                "current_level": self.current_level,
                "adaptation_threshold": self.adaptation_threshold
            },
            "performance_history": [asdict(p) for p in self.performance_history],
            "task_progression": self.task_progression,
            "adaptation_log": self.adaptation_log,
            "llm_decisions": self.llm_decisions
        }
        
        with open(filepath, 'w') as f:
            json.dump(state, f, indent=2)
        
        logger.info(f"Saved curriculum state to {filepath}")
    
    def load_curriculum_state(self, filepath: str):
        """Load curriculum state from file."""
        with open(filepath, 'r') as f:
            state = json.load(f)
        
        # Restore configuration
        config = state["curriculum_config"]
        self.current_level = config["current_level"]
        self.adaptation_threshold = config["adaptation_threshold"]
        
        # Restore history
        self.performance_history = [
            PerformanceAnalysis(**p) for p in state["performance_history"]
        ]
        self.task_progression = state["task_progression"]
        self.adaptation_log = state["adaptation_log"]
        self.llm_decisions = state["llm_decisions"]
        self.llm_queries = len(self.llm_decisions)
        
        logger.info(f"Loaded curriculum state from {filepath}")


class CurriculumTrainer:
    """High-level interface for curriculum-based training."""
    
    def __init__(self, curriculum: LLMCurriculum, benchmark_suite):
        """Initialize curriculum trainer.
        
        Args:
            curriculum: LLM curriculum instance
            benchmark_suite: Benchmark suite for evaluation
        """
        self.curriculum = curriculum
        self.benchmark_suite = benchmark_suite
        self.training_history = []
    
    def train(self, 
              initial_task: BaseTask,
              num_tasks: int = 100,
              episodes_per_task: int = 20,
              save_interval: int = 10) -> Dict[str, Any]:
        """Run curriculum-based training.
        
        Args:
            initial_task: Starting task
            num_tasks: Number of curriculum tasks to run
            episodes_per_task: Episodes per task
            save_interval: Interval for saving progress
            
        Returns:
            Training results summary
        """
        current_task = initial_task
        training_results = []
        
        logger.info(f"Starting curriculum training with {num_tasks} tasks")
        
        for task_idx in range(num_tasks):
            logger.info(f"Task {task_idx + 1}/{num_tasks}: {current_task.name}")
            
            # Run episodes on current task
            episode_results = []
            for episode in range(episodes_per_task):
                # In a real implementation, this would run the agent
                # For now, we simulate episode results
                episode_result = self._simulate_episode(current_task)
                episode_results.append(episode_result)
            
            # Analyze performance
            analysis = self.curriculum.analyze_performance(episode_results)
            self.curriculum.performance_history.append(analysis)
            
            # Generate next task
            next_config = self.curriculum.generate_next_task(current_task, analysis)
            
            # Create next task (simplified - would use task factory)
            next_task = type(current_task)(next_config)
            
            # Log progress
            training_results.append({
                "task_idx": task_idx,
                "task_name": current_task.name,
                "success_rate": analysis.success_rate,
                "efficiency": analysis.efficiency_score,
                "curriculum_level": next_config.get("curriculum_level", 1),
                "recommendation": analysis.recommended_difficulty
            })
            
            # Save progress periodically
            if (task_idx + 1) % save_interval == 0:
                self.curriculum.save_curriculum_state(
                    f"curriculum_checkpoint_{task_idx + 1}.json"
                )
            
            current_task = next_task
        
        logger.info("Curriculum training completed")
        
        return {
            "num_tasks_completed": num_tasks,
            "total_episodes": num_tasks * episodes_per_task,
            "final_curriculum_level": self.curriculum.current_level,
            "final_performance": self.curriculum.performance_history[-1] if self.curriculum.performance_history else None,
            "training_progression": training_results
        }
    
    def _simulate_episode(self, task: BaseTask) -> Dict[str, Any]:
        """Simulate episode results for testing."""
        # Simulate realistic performance based on curriculum level
        base_success_rate = min(0.9, 0.3 + 0.1 * task.curriculum_level)
        success = np.random.random() < base_success_rate
        
        steps = np.random.randint(50, task.max_steps)
        reward = np.random.uniform(-10, 100) if success else np.random.uniform(-50, 10)
        
        return {
            "success": success,
            "total_steps": steps,
            "total_reward": reward,
            "max_steps": task.max_steps,
            "info": {
                "collision": np.random.random() < 0.1,
                "timeout": steps >= task.max_steps
            }
        }