"""Emergent multi-agent curriculum learning system."""

import time
from typing import Any, Dict, List, Optional, Tuple, Set
import numpy as np
from datetime import datetime
import logging
from collections import defaultdict, deque

from ..core.base_task import BaseTask
from ..core.base_agent import BaseAgent
from ..multiagent.coordination_protocols import CommunicationProtocol
from ..utils.logging_config import get_logger
from .llm_curriculum import LLMCurriculum

logger = get_logger(__name__)


class BehaviorPattern:
    """Represents an emergent behavior pattern from multi-agent interactions."""
    
    def __init__(self, pattern_id: str, agents_involved: List[str], 
                 interaction_sequence: List[Dict[str, Any]], 
                 success_rate: float, complexity_score: float):
        self.pattern_id = pattern_id
        self.agents_involved = agents_involved
        self.interaction_sequence = interaction_sequence
        self.success_rate = success_rate
        self.complexity_score = complexity_score
        self.discovered_at = datetime.now()
        self.usage_count = 0
        self.effectiveness_history = deque(maxlen=100)
    
    def update_effectiveness(self, outcome: float):
        """Update pattern effectiveness based on recent outcomes."""
        self.effectiveness_history.append(outcome)
        self.usage_count += 1
    
    def get_current_effectiveness(self) -> float:
        """Get current pattern effectiveness."""
        if not self.effectiveness_history:
            return self.success_rate
        return np.mean(list(self.effectiveness_history))
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert pattern to dictionary representation."""
        return {
            "pattern_id": self.pattern_id,
            "agents_involved": self.agents_involved,
            "interaction_sequence": self.interaction_sequence,
            "success_rate": self.success_rate,
            "complexity_score": self.complexity_score,
            "discovered_at": self.discovered_at.isoformat(),
            "usage_count": self.usage_count,
            "current_effectiveness": self.get_current_effectiveness()
        }


class BehaviorAnalyzer:
    """Analyzes multi-agent interactions to identify emergent behavior patterns."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.min_pattern_length = self.config.get("min_pattern_length", 3)
        self.min_success_rate = self.config.get("min_success_rate", 0.6)
        self.complexity_threshold = self.config.get("complexity_threshold", 0.5)
        self._interaction_history = defaultdict(list)
        self._discovered_patterns = {}
    
    def analyze_episode_interactions(
        self, 
        episode_data: Dict[str, Any]
    ) -> List[BehaviorPattern]:
        """Analyze single episode for emergent behavior patterns.
        
        Args:
            episode_data: Episode data containing multi-agent interactions
            
        Returns:
            List of discovered behavior patterns
        """
        interactions = self._extract_interactions(episode_data)
        patterns = self._identify_patterns(interactions)
        
        new_patterns = []
        for pattern_data in patterns:
            pattern = self._create_behavior_pattern(pattern_data, episode_data)
            if self._is_novel_pattern(pattern):
                self._discovered_patterns[pattern.pattern_id] = pattern
                new_patterns.append(pattern)
                logger.info(f"Discovered new behavior pattern: {pattern.pattern_id}")
        
        return new_patterns
    
    def _extract_interactions(
        self, 
        episode_data: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Extract interaction sequences from episode data.
        
        Args:
            episode_data: Episode data
            
        Returns:
            List of interaction events
        """
        interactions = []
        
        if "steps" not in episode_data:
            return interactions
        
        for step_data in episode_data["steps"]:
            # Extract multi-agent interactions
            step_interactions = self._extract_step_interactions(step_data)
            interactions.extend(step_interactions)
        
        return interactions
    
    def _extract_step_interactions(
        self, 
        step_data: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Extract interactions from a single step.
        
        Args:
            step_data: Single step data
            
        Returns:
            List of interactions in this step
        """
        interactions = []
        
        # Extract communication events
        if "communication" in step_data:
            for comm_event in step_data["communication"]:
                interactions.append({
                    "type": "communication",
                    "timestamp": step_data.get("timestamp", time.time()),
                    "sender": comm_event.get("sender"),
                    "receiver": comm_event.get("receiver"),
                    "message": comm_event.get("message"),
                    "step": step_data.get("step", 0)
                })
        
        # Extract coordination events
        if "coordination" in step_data:
            for coord_event in step_data["coordination"]:
                interactions.append({
                    "type": "coordination",
                    "timestamp": step_data.get("timestamp", time.time()),
                    "agents": coord_event.get("agents", []),
                    "action_type": coord_event.get("action_type"),
                    "synchronization": coord_event.get("synchronization", False),
                    "step": step_data.get("step", 0)
                })
        
        # Extract spatial interactions (proximity-based)
        if "agent_positions" in step_data:
            spatial_interactions = self._detect_spatial_interactions(
                step_data["agent_positions"], step_data.get("step", 0)
            )
            interactions.extend(spatial_interactions)
        
        return interactions
    
    def _detect_spatial_interactions(
        self, 
        agent_positions: Dict[str, Any], 
        step: int
    ) -> List[Dict[str, Any]]:
        """Detect spatial interactions between agents.
        
        Args:
            agent_positions: Dictionary of agent positions
            step: Current step number
            
        Returns:
            List of spatial interaction events
        """
        interactions = []
        agent_ids = list(agent_positions.keys())
        proximity_threshold = self.config.get("proximity_threshold", 2.0)
        
        for i, agent1 in enumerate(agent_ids):
            for j, agent2 in enumerate(agent_ids[i+1:], i+1):
                pos1 = np.array(agent_positions[agent1].get("position", [0, 0, 0]))
                pos2 = np.array(agent_positions[agent2].get("position", [0, 0, 0]))
                
                distance = np.linalg.norm(pos1 - pos2)
                
                if distance < proximity_threshold:
                    interactions.append({
                        "type": "spatial_proximity",
                        "timestamp": time.time(),
                        "agents": [agent1, agent2],
                        "distance": distance,
                        "positions": {agent1: pos1.tolist(), agent2: pos2.tolist()},
                        "step": step
                    })
        
        return interactions
    
    def _identify_patterns(
        self, 
        interactions: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Identify recurring patterns in interaction sequences.
        
        Args:
            interactions: List of interaction events
            
        Returns:
            List of identified patterns
        """
        if len(interactions) < self.min_pattern_length:
            return []
        
        patterns = []
        
        # Use sliding window to find recurring subsequences
        for window_size in range(self.min_pattern_length, len(interactions) + 1):
            pattern_counts = defaultdict(int)
            pattern_sequences = defaultdict(list)
            
            for i in range(len(interactions) - window_size + 1):
                window = interactions[i:i + window_size]
                pattern_key = self._create_pattern_key(window)
                pattern_counts[pattern_key] += 1
                pattern_sequences[pattern_key].append(window)
            
            # Find patterns that occur multiple times
            for pattern_key, count in pattern_counts.items():
                if count >= 2:  # Pattern occurs at least twice
                    patterns.append({
                        "pattern_key": pattern_key,
                        "occurrences": count,
                        "sequences": pattern_sequences[pattern_key],
                        "window_size": window_size
                    })
        
        # Rank patterns by frequency and complexity
        ranked_patterns = sorted(patterns, key=lambda p: (p["occurrences"], p["window_size"]), reverse=True)
        
        return ranked_patterns[:10]  # Return top 10 patterns
    
    def _create_pattern_key(self, interaction_sequence: List[Dict[str, Any]]) -> str:
        """Create a key to identify similar interaction patterns.
        
        Args:
            interaction_sequence: Sequence of interactions
            
        Returns:
            Pattern key string
        """
        key_parts = []
        
        for interaction in interaction_sequence:
            interaction_type = interaction.get("type", "unknown")
            
            if interaction_type == "communication":
                key_parts.append(f"comm_{interaction.get('sender')}_{interaction.get('receiver')}")
            elif interaction_type == "coordination":
                agents = sorted(interaction.get("agents", []))
                action = interaction.get("action_type", "unknown")
                key_parts.append(f"coord_{'_'.join(agents)}_{action}")
            elif interaction_type == "spatial_proximity":
                agents = sorted(interaction.get("agents", []))
                key_parts.append(f"spatial_{'_'.join(agents)}")
            else:
                key_parts.append(f"{interaction_type}")
        
        return "|".join(key_parts)
    
    def _create_behavior_pattern(
        self, 
        pattern_data: Dict[str, Any], 
        episode_data: Dict[str, Any]
    ) -> BehaviorPattern:
        """Create BehaviorPattern object from pattern data.
        
        Args:
            pattern_data: Raw pattern data
            episode_data: Episode context data
            
        Returns:
            BehaviorPattern instance
        """
        pattern_id = f"pattern_{hash(pattern_data['pattern_key']) % 10000:04d}"
        
        # Extract involved agents
        agents_involved = set()
        for sequence in pattern_data["sequences"]:
            for interaction in sequence:
                if interaction.get("type") == "communication":
                    agents_involved.add(interaction.get("sender", ""))
                    agents_involved.add(interaction.get("receiver", ""))
                elif "agents" in interaction:
                    agents_involved.update(interaction.get("agents", []))
        
        # Calculate success rate based on episode outcome
        success_rate = 1.0 if episode_data.get("success", False) else 0.0
        
        # Calculate complexity score
        complexity_score = self._calculate_pattern_complexity(pattern_data)
        
        return BehaviorPattern(
            pattern_id=pattern_id,
            agents_involved=list(agents_involved),
            interaction_sequence=pattern_data["sequences"][0],  # Use first occurrence as template
            success_rate=success_rate,
            complexity_score=complexity_score
        )
    
    def _calculate_pattern_complexity(self, pattern_data: Dict[str, Any]) -> float:
        """Calculate complexity score for a pattern.
        
        Args:
            pattern_data: Pattern data
            
        Returns:
            Complexity score (0-1)
        """
        # Factors contributing to complexity:
        # 1. Number of agents involved
        # 2. Length of interaction sequence
        # 3. Variety of interaction types
        # 4. Temporal coordination requirements
        
        agents_involved = set()
        interaction_types = set()
        sequence_length = pattern_data["window_size"]
        
        for sequence in pattern_data["sequences"]:
            for interaction in sequence:
                interaction_types.add(interaction.get("type", "unknown"))
                if "agents" in interaction:
                    agents_involved.update(interaction.get("agents", []))
                if interaction.get("type") == "communication":
                    agents_involved.add(interaction.get("sender", ""))
                    agents_involved.add(interaction.get("receiver", ""))
        
        # Normalize factors
        agent_complexity = min(len(agents_involved) / 8.0, 1.0)  # Max 8 agents
        type_complexity = min(len(interaction_types) / 4.0, 1.0)  # Max 4 types
        length_complexity = min(sequence_length / 10.0, 1.0)  # Max 10 steps
        
        # Weighted combination
        complexity = (
            0.4 * agent_complexity + 
            0.3 * type_complexity + 
            0.3 * length_complexity
        )
        
        return complexity
    
    def _is_novel_pattern(self, pattern: BehaviorPattern) -> bool:
        """Check if pattern is novel (not already discovered).
        
        Args:
            pattern: Behavior pattern to check
            
        Returns:
            True if pattern is novel
        """
        # Check against existing patterns
        for existing_id, existing_pattern in self._discovered_patterns.items():
            if self._patterns_similar(pattern, existing_pattern):
                return False
        
        return True
    
    def _patterns_similar(
        self, 
        pattern1: BehaviorPattern, 
        pattern2: BehaviorPattern, 
        similarity_threshold: float = 0.8
    ) -> bool:
        """Check if two patterns are similar.
        
        Args:
            pattern1: First pattern
            pattern2: Second pattern
            similarity_threshold: Similarity threshold
            
        Returns:
            True if patterns are similar
        """
        # Compare agents involved
        agents1 = set(pattern1.agents_involved)
        agents2 = set(pattern2.agents_involved)
        
        if not agents1 or not agents2:
            return False
        
        agent_similarity = len(agents1.intersection(agents2)) / len(agents1.union(agents2))
        
        # Compare interaction sequences (simplified)
        seq1_types = [i.get("type") for i in pattern1.interaction_sequence]
        seq2_types = [i.get("type") for i in pattern2.interaction_sequence]
        
        if len(seq1_types) != len(seq2_types):
            return False
        
        type_matches = sum(1 for t1, t2 in zip(seq1_types, seq2_types) if t1 == t2)
        type_similarity = type_matches / len(seq1_types) if seq1_types else 0
        
        overall_similarity = 0.6 * agent_similarity + 0.4 * type_similarity
        
        return overall_similarity >= similarity_threshold
    
    def get_discovered_patterns(self) -> Dict[str, BehaviorPattern]:
        """Get all discovered behavior patterns.
        
        Returns:
            Dictionary of pattern IDs to BehaviorPattern objects
        """
        return self._discovered_patterns.copy()
    
    def get_pattern_statistics(self) -> Dict[str, Any]:
        """Get statistics about discovered patterns.
        
        Returns:
            Pattern statistics
        """
        if not self._discovered_patterns:
            return {"total_patterns": 0}
        
        patterns = list(self._discovered_patterns.values())
        
        return {
            "total_patterns": len(patterns),
            "avg_complexity": np.mean([p.complexity_score for p in patterns]),
            "avg_success_rate": np.mean([p.success_rate for p in patterns]),
            "avg_effectiveness": np.mean([p.get_current_effectiveness() for p in patterns]),
            "most_used_pattern": max(patterns, key=lambda p: p.usage_count).pattern_id if patterns else None,
            "highest_complexity": max(patterns, key=lambda p: p.complexity_score).pattern_id if patterns else None
        }


class EmergentCurriculumGenerator:
    """Generate curricula based on emergent multi-agent behaviors."""
    
    def __init__(self, llm_model: str = "gpt-4", config: Optional[Dict[str, Any]] = None):
        """Initialize emergent curriculum generator.
        
        Args:
            llm_model: LLM model to use for task generation
            config: Configuration dictionary
        """
        self.config = config or {}
        self.llm_curriculum = LLMCurriculum({
            "llm_model": llm_model,
            **self.config
        })
        self.behavior_analyzer = BehaviorAnalyzer(config.get("behavior_analyzer", {}))
        self._generated_tasks = []
        self._pattern_usage_history = defaultdict(list)
    
    def generate_emergent_tasks(
        self, 
        multi_agent_interactions: List[Dict[str, Any]],
        base_task: BaseTask,
        num_tasks: int = 5
    ) -> List[Dict[str, Any]]:
        """Analyze emergent behaviors to create novel task variations.
        
        Args:
            multi_agent_interactions: List of multi-agent episode data
            base_task: Base task to create variations of
            num_tasks: Number of tasks to generate
            
        Returns:
            List of generated task specifications
        """
        logger.info(f"Generating {num_tasks} emergent tasks from {len(multi_agent_interactions)} episodes")
        
        # Analyze interactions to discover patterns
        all_patterns = []
        for episode_data in multi_agent_interactions:
            episode_patterns = self.behavior_analyzer.analyze_episode_interactions(episode_data)
            all_patterns.extend(episode_patterns)
        
        if not all_patterns:
            logger.warning("No emergent patterns discovered, falling back to random task generation")
            return self._generate_random_task_variations(base_task, num_tasks)
        
        # Select most promising patterns
        selected_patterns = self._select_patterns_for_curriculum(all_patterns, num_tasks)
        
        # Generate tasks based on selected patterns
        generated_tasks = []
        for pattern in selected_patterns:
            task_spec = self._generate_task_from_pattern(pattern, base_task)
            if task_spec:
                generated_tasks.append(task_spec)
                self._pattern_usage_history[pattern.pattern_id].append({
                    "timestamp": datetime.now().isoformat(),
                    "task_spec": task_spec
                })
        
        # Fill remaining slots with variations
        while len(generated_tasks) < num_tasks:
            variation = self._generate_pattern_variation(selected_patterns, base_task)
            if variation and variation not in generated_tasks:
                generated_tasks.append(variation)
        
        self._generated_tasks.extend(generated_tasks)
        
        logger.info(f"Successfully generated {len(generated_tasks)} emergent tasks")
        return generated_tasks
    
    def _select_patterns_for_curriculum(
        self, 
        patterns: List[BehaviorPattern], 
        num_patterns: int
    ) -> List[BehaviorPattern]:
        """Select most promising patterns for curriculum generation.
        
        Args:
            patterns: List of discovered patterns
            num_patterns: Number of patterns to select
            
        Returns:
            Selected patterns
        """
        if len(patterns) <= num_patterns:
            return patterns
        
        # Score patterns based on multiple criteria
        pattern_scores = []
        for pattern in patterns:
            score = self._calculate_pattern_curriculum_score(pattern)
            pattern_scores.append((pattern, score))
        
        # Sort by score and select top patterns
        pattern_scores.sort(key=lambda x: x[1], reverse=True)
        selected = [pattern for pattern, score in pattern_scores[:num_patterns]]
        
        return selected
    
    def _calculate_pattern_curriculum_score(self, pattern: BehaviorPattern) -> float:
        """Calculate curriculum generation score for a pattern.
        
        Args:
            pattern: Behavior pattern
            
        Returns:
            Curriculum score
        """
        # Factors:
        # 1. Pattern complexity (higher is better for learning)
        # 2. Success rate (moderate success is ideal for learning)
        # 3. Novelty (less used patterns are preferred)
        # 4. Current effectiveness
        
        complexity_score = pattern.complexity_score
        
        # Optimal success rate is around 0.6-0.8 (challenging but achievable)
        success_score = 1.0 - abs(pattern.success_rate - 0.7) / 0.7
        
        # Novelty based on usage
        usage_penalty = min(pattern.usage_count / 10.0, 0.5)  # Cap penalty at 50%
        novelty_score = 1.0 - usage_penalty
        
        effectiveness_score = pattern.get_current_effectiveness()
        
        # Weighted combination
        total_score = (
            0.3 * complexity_score +
            0.3 * success_score +
            0.2 * novelty_score +
            0.2 * effectiveness_score
        )
        
        return total_score
    
    def _generate_task_from_pattern(
        self, 
        pattern: BehaviorPattern, 
        base_task: BaseTask
    ) -> Optional[Dict[str, Any]]:
        """Generate task specification from behavior pattern.
        
        Args:
            pattern: Behavior pattern
            base_task: Base task to modify
            
        Returns:
            Task specification dictionary
        """
        try:
            # Create pattern description for LLM
            pattern_description = self._create_pattern_description(pattern)
            
            # Generate task using LLM
            task_prompt = f"""
            Based on the following emergent behavior pattern observed in multi-agent interactions, 
            create a new task variation that encourages and evaluates this specific behavior:
            
            Pattern Description: {pattern_description}
            
            Base Task: {getattr(base_task, 'description', 'Multi-agent coordination task')}
            
            Generate a task specification that:
            1. Requires agents to exhibit the discovered behavior pattern
            2. Has clear success criteria
            3. Is appropriately challenging
            4. Builds on the base task structure
            
            Return the task specification as a JSON object with fields:
            - name: task name
            - description: detailed task description
            - objectives: list of objectives
            - success_criteria: success evaluation criteria
            - difficulty_level: estimated difficulty (0.0-1.0)
            - pattern_requirements: specific pattern behaviors required
            """
            
            # Use LLM curriculum to generate task
            task_spec = self._query_llm_for_task(task_prompt, pattern)
            
            if task_spec:
                # Add pattern metadata
                task_spec["source_pattern_id"] = pattern.pattern_id
                task_spec["pattern_complexity"] = pattern.complexity_score
                task_spec["agents_required"] = len(pattern.agents_involved)
                task_spec["generation_timestamp"] = datetime.now().isoformat()
            
            return task_spec
            
        except Exception as e:
            logger.error(f"Failed to generate task from pattern {pattern.pattern_id}: {e}")
            return None
    
    def _create_pattern_description(self, pattern: BehaviorPattern) -> str:
        """Create human-readable description of behavior pattern.
        
        Args:
            pattern: Behavior pattern
            
        Returns:
            Pattern description string
        """
        description_parts = [
            f"Pattern ID: {pattern.pattern_id}",
            f"Agents Involved: {len(pattern.agents_involved)} ({', '.join(pattern.agents_involved)})",
            f"Complexity Score: {pattern.complexity_score:.3f}",
            f"Success Rate: {pattern.success_rate:.3f}",
            f"Current Effectiveness: {pattern.get_current_effectiveness():.3f}",
            "\nInteraction Sequence:"
        ]
        
        for i, interaction in enumerate(pattern.interaction_sequence):
            interaction_type = interaction.get("type", "unknown")
            step = interaction.get("step", i)
            
            if interaction_type == "communication":
                sender = interaction.get("sender", "unknown")
                receiver = interaction.get("receiver", "unknown")
                description_parts.append(f"  Step {step}: {sender} communicates with {receiver}")
            elif interaction_type == "coordination":
                agents = interaction.get("agents", [])
                action = interaction.get("action_type", "unknown")
                description_parts.append(f"  Step {step}: Agents {', '.join(agents)} coordinate {action}")
            elif interaction_type == "spatial_proximity":
                agents = interaction.get("agents", [])
                distance = interaction.get("distance", "unknown")
                description_parts.append(f"  Step {step}: Agents {', '.join(agents)} in proximity (distance: {distance})")
            else:
                description_parts.append(f"  Step {step}: {interaction_type} interaction")
        
        return "\n".join(description_parts)
    
    def _query_llm_for_task(
        self, 
        prompt: str, 
        pattern: BehaviorPattern
    ) -> Optional[Dict[str, Any]]:
        """Query LLM to generate task specification.
        
        Args:
            prompt: Task generation prompt
            pattern: Source behavior pattern
            
        Returns:
            Task specification or None if generation failed
        """
        try:
            # This would interface with the actual LLM
            # For now, create a template-based response
            task_spec = {
                "name": f"Emergent Pattern Task - {pattern.pattern_id}",
                "description": f"Multi-agent task designed to elicit behavior pattern {pattern.pattern_id}",
                "objectives": [
                    "Agents must coordinate using discovered interaction patterns",
                    "Demonstrate emergent behaviors observed in training",
                    "Achieve task completion through collaborative strategies"
                ],
                "success_criteria": {
                    "pattern_exhibited": True,
                    "task_completed": True,
                    "coordination_quality": "> 0.7"
                },
                "difficulty_level": min(0.5 + pattern.complexity_score * 0.4, 0.9),
                "pattern_requirements": {
                    "agents_involved": pattern.agents_involved,
                    "interaction_types": list(set(i.get("type") for i in pattern.interaction_sequence)),
                    "min_coordination_events": len(pattern.interaction_sequence)
                }
            }
            
            return task_spec
            
        except Exception as e:
            logger.error(f"LLM query failed for pattern {pattern.pattern_id}: {e}")
            return None
    
    def _generate_random_task_variations(
        self, 
        base_task: BaseTask, 
        num_tasks: int
    ) -> List[Dict[str, Any]]:
        """Generate random task variations as fallback.
        
        Args:
            base_task: Base task
            num_tasks: Number of tasks to generate
            
        Returns:
            List of task specifications
        """
        variations = []
        
        for i in range(num_tasks):
            variation = {
                "name": f"Random Variation {i+1}",
                "description": f"Random variation of {getattr(base_task, 'name', 'base task')}",
                "objectives": ["Complete the task using any strategy"],
                "success_criteria": {"task_completed": True},
                "difficulty_level": np.random.uniform(0.3, 0.8),
                "generation_method": "random_fallback"
            }
            variations.append(variation)
        
        return variations
    
    def _generate_pattern_variation(
        self, 
        patterns: List[BehaviorPattern], 
        base_task: BaseTask
    ) -> Optional[Dict[str, Any]]:
        """Generate variation by combining multiple patterns.
        
        Args:
            patterns: List of behavior patterns
            base_task: Base task
            
        Returns:
            Task specification or None
        """
        if len(patterns) < 2:
            return None
        
        # Select 2-3 patterns to combine
        num_combine = min(3, len(patterns))
        selected_patterns = np.random.choice(patterns, num_combine, replace=False)
        
        combined_agents = set()
        combined_complexity = 0
        combined_success = 0
        
        for pattern in selected_patterns:
            combined_agents.update(pattern.agents_involved)
            combined_complexity += pattern.complexity_score
            combined_success += pattern.success_rate
        
        avg_complexity = combined_complexity / len(selected_patterns)
        avg_success = combined_success / len(selected_patterns)
        
        variation = {
            "name": f"Combined Pattern Task - {'+'.join([p.pattern_id for p in selected_patterns])}",
            "description": f"Task combining {len(selected_patterns)} emergent behavior patterns",
            "objectives": [
                "Exhibit multiple coordinated behavior patterns",
                "Demonstrate complex multi-agent interactions",
                "Achieve task success through pattern combination"
            ],
            "success_criteria": {
                "multiple_patterns_exhibited": True,
                "task_completed": True,
                "interaction_richness": "> 0.8"
            },
            "difficulty_level": min(0.6 + avg_complexity * 0.3, 0.95),
            "pattern_requirements": {
                "agents_required": len(combined_agents),
                "source_patterns": [p.pattern_id for p in selected_patterns],
                "expected_complexity": avg_complexity
            },
            "generation_method": "pattern_combination"
        }
        
        return variation
    
    def update_pattern_effectiveness(
        self, 
        pattern_id: str, 
        task_outcome: float
    ):
        """Update pattern effectiveness based on task outcomes.
        
        Args:
            pattern_id: Pattern identifier
            task_outcome: Task outcome score (0-1)
        """
        patterns = self.behavior_analyzer.get_discovered_patterns()
        if pattern_id in patterns:
            patterns[pattern_id].update_effectiveness(task_outcome)
            logger.debug(f"Updated effectiveness for pattern {pattern_id}: {task_outcome}")
    
    def get_curriculum_statistics(self) -> Dict[str, Any]:
        """Get statistics about the emergent curriculum.
        
        Returns:
            Curriculum statistics
        """
        pattern_stats = self.behavior_analyzer.get_pattern_statistics()
        
        return {
            "total_tasks_generated": len(self._generated_tasks),
            "patterns_discovered": pattern_stats.get("total_patterns", 0),
            "avg_pattern_complexity": pattern_stats.get("avg_complexity", 0),
            "avg_pattern_effectiveness": pattern_stats.get("avg_effectiveness", 0),
            "patterns_used_for_generation": len(self._pattern_usage_history),
            "generation_methods": self._get_generation_method_stats()
        }
    
    def _get_generation_method_stats(self) -> Dict[str, int]:
        """Get statistics about task generation methods.
        
        Returns:
            Generation method counts
        """
        method_counts = defaultdict(int)
        
        for task in self._generated_tasks:
            method = task.get("generation_method", "pattern_based")
            method_counts[method] += 1
        
        return dict(method_counts)
    
    def save_curriculum_data(self, filepath: str):
        """Save curriculum data to file.
        
        Args:
            filepath: Output file path
        """
        import json
        
        curriculum_data = {
            "generated_tasks": self._generated_tasks,
            "pattern_usage_history": dict(self._pattern_usage_history),
            "discovered_patterns": {
                pid: pattern.to_dict() 
                for pid, pattern in self.behavior_analyzer.get_discovered_patterns().items()
            },
            "statistics": self.get_curriculum_statistics(),
            "export_timestamp": datetime.now().isoformat()
        }
        
        with open(filepath, 'w') as f:
            json.dump(curriculum_data, f, indent=2, default=str)
        
        logger.info(f"Curriculum data saved to {filepath}")
