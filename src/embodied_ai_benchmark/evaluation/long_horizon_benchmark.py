"""Long-horizon multi-agent task planning and evaluation benchmark."""

import time
from typing import Any, Dict, List, Optional, Tuple, Set, Callable
import numpy as np
from datetime import datetime, timedelta
import logging
from collections import defaultdict, deque
from enum import Enum
import threading
import asyncio

from ..core.base_task import BaseTask
from ..core.base_agent import BaseAgent
from ..core.base_env import BaseEnv
from ..utils.logging_config import get_logger
from ..utils.error_handling import SafeExecutor, ErrorRecoveryManager
from ..utils.monitoring import performance_monitor
from .benchmark_suite import BenchmarkSuite

logger = get_logger(__name__)


class TaskPhase(Enum):
    """Enumeration of task phases for long-horizon planning."""
    PLANNING = "planning"
    COORDINATION = "coordination"
    EXECUTION = "execution"
    ADAPTATION = "adaptation"
    COMPLETION = "completion"


class CoordinationEvent:
    """Represents a coordination event between agents."""
    
    def __init__(self, event_id: str, timestamp: float, agents_involved: List[str],
                 event_type: str, data: Dict[str, Any], priority: int = 1):
        self.event_id = event_id
        self.timestamp = timestamp
        self.agents_involved = agents_involved
        self.event_type = event_type
        self.data = data
        self.priority = priority
        self.completed = False
        self.result = None
        self.execution_time = None
    
    def complete(self, result: Any, execution_time: float):
        """Mark event as completed with result."""
        self.completed = True
        self.result = result
        self.execution_time = execution_time
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "event_id": self.event_id,
            "timestamp": self.timestamp,
            "agents_involved": self.agents_involved,
            "event_type": self.event_type,
            "data": self.data,
            "priority": self.priority,
            "completed": self.completed,
            "result": self.result,
            "execution_time": self.execution_time
        }


class HierarchicalTaskPlan:
    """Represents a hierarchical task plan for long-horizon execution."""
    
    def __init__(self, task_id: str, description: str, max_horizon: int = 100):
        self.task_id = task_id
        self.description = description
        self.max_horizon = max_horizon
        self.phases = []
        self.dependencies = defaultdict(list)  # phase_id -> list of prerequisite phase_ids
        self.agent_assignments = defaultdict(list)  # agent_id -> list of assigned phase_ids
        self.coordination_events = []
        self.created_at = datetime.now()
        self.status = "created"
        self.current_phase = None
        self.execution_history = []
        
    def add_phase(self, phase_id: str, description: str, phase_type: TaskPhase,
                  estimated_duration: float, required_agents: List[str]) -> Dict[str, Any]:
        """Add a phase to the task plan.
        
        Args:
            phase_id: Unique phase identifier
            description: Phase description
            phase_type: Type of phase
            estimated_duration: Estimated execution time
            required_agents: List of required agent IDs
            
        Returns:
            Phase specification dictionary
        """
        phase = {
            "phase_id": phase_id,
            "description": description,
            "phase_type": phase_type,
            "estimated_duration": estimated_duration,
            "required_agents": required_agents,
            "status": "pending",
            "start_time": None,
            "end_time": None,
            "actual_duration": None,
            "success": None,
            "coordination_events": [],
            "error_count": 0,
            "recovery_attempts": 0
        }
        
        self.phases.append(phase)
        
        # Assign agents to this phase
        for agent_id in required_agents:
            self.agent_assignments[agent_id].append(phase_id)
        
        return phase
    
    def add_dependency(self, phase_id: str, prerequisite_phase_id: str):
        """Add dependency between phases.
        
        Args:
            phase_id: Phase that depends on prerequisite
            prerequisite_phase_id: Phase that must complete first
        """
        self.dependencies[phase_id].append(prerequisite_phase_id)
    
    def get_next_executable_phases(self) -> List[Dict[str, Any]]:
        """Get phases that are ready for execution.
        
        Returns:
            List of executable phases
        """
        executable = []
        
        for phase in self.phases:
            if phase["status"] == "pending":
                # Check if all dependencies are satisfied
                dependencies_met = True
                for prereq_id in self.dependencies.get(phase["phase_id"], []):
                    prereq_phase = self.get_phase_by_id(prereq_id)
                    if not prereq_phase or prereq_phase["status"] != "completed":
                        dependencies_met = False
                        break
                
                if dependencies_met:
                    executable.append(phase)
        
        return executable
    
    def get_phase_by_id(self, phase_id: str) -> Optional[Dict[str, Any]]:
        """Get phase by ID.
        
        Args:
            phase_id: Phase identifier
            
        Returns:
            Phase dictionary or None
        """
        for phase in self.phases:
            if phase["phase_id"] == phase_id:
                return phase
        return None
    
    def update_phase_status(self, phase_id: str, status: str, **kwargs):
        """Update phase status and metadata.
        
        Args:
            phase_id: Phase identifier
            status: New status
            **kwargs: Additional metadata to update
        """
        phase = self.get_phase_by_id(phase_id)
        if phase:
            phase["status"] = status
            for key, value in kwargs.items():
                if key in phase:
                    phase[key] = value
    
    def get_execution_statistics(self) -> Dict[str, Any]:
        """Get execution statistics for the plan.
        
        Returns:
            Execution statistics
        """
        total_phases = len(self.phases)
        completed_phases = len([p for p in self.phases if p["status"] == "completed"])
        failed_phases = len([p for p in self.phases if p["status"] == "failed"])
        
        total_estimated = sum(p["estimated_duration"] for p in self.phases)
        total_actual = sum(p["actual_duration"] or 0 for p in self.phases 
                          if p["actual_duration"] is not None)
        
        return {
            "total_phases": total_phases,
            "completed_phases": completed_phases,
            "failed_phases": failed_phases,
            "success_rate": completed_phases / total_phases if total_phases > 0 else 0,
            "total_estimated_duration": total_estimated,
            "total_actual_duration": total_actual,
            "time_efficiency": total_estimated / max(total_actual, 1e-6) if total_actual > 0 else 0,
            "coordination_events": len(self.coordination_events),
            "total_errors": sum(p["error_count"] for p in self.phases),
            "total_recoveries": sum(p["recovery_attempts"] for p in self.phases)
        }


class LongHorizonMultiAgentBenchmark:
    """Benchmark for long-horizon multi-agent task planning and execution."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize long-horizon benchmark.
        
        Args:
            config: Benchmark configuration
        """
        self.config = config or {}
        self.task_types = [
            "furniture_assembly_complex",  # 20+ step assembly
            "collaborative_cooking",       # 15+ step meal preparation  
            "disaster_response_scenario",  # 30+ step search & rescue
            "construction_project",        # 50+ step building task
            "warehouse_logistics",         # 40+ step inventory management
            "scientific_experiment"        # 25+ step lab protocol
        ]
        
        self.benchmark_suite = BenchmarkSuite(config)
        self.safe_executor = SafeExecutor()
        self.error_recovery = ErrorRecoveryManager()
        self._evaluation_history = []
        self._active_plans = {}
        self._coordination_lock = threading.Lock()
    
    def evaluate_hierarchical_planning(
        self, 
        multi_agent_system: Dict[str, BaseAgent], 
        env: BaseEnv,
        task_type: str = "furniture_assembly_complex",
        max_horizon: int = 100,
        num_episodes: int = 5
    ) -> Dict[str, Any]:
        """Evaluate planning and coordination over long horizons.
        
        Args:
            multi_agent_system: Dictionary of agent_id -> BaseAgent
            env: Environment for evaluation
            task_type: Type of long-horizon task
            max_horizon: Maximum planning horizon
            num_episodes: Number of evaluation episodes
            
        Returns:
            Evaluation results
        """
        logger.info(f"Starting long-horizon evaluation: {task_type} with {len(multi_agent_system)} agents")
        logger.info(f"Max horizon: {max_horizon}, Episodes: {num_episodes}")
        
        evaluation_start = time.time()
        
        try:
            # Generate task specifications
            task_specs = self._generate_long_horizon_tasks(task_type, max_horizon, num_episodes)
            
            episode_results = []
            
            for episode_idx, task_spec in enumerate(task_specs):
                logger.info(f"Executing episode {episode_idx + 1}/{num_episodes}: {task_spec['name']}")
                
                # Execute episode with comprehensive error handling
                episode_result = self.safe_executor.execute_with_recovery(
                    self._execute_long_horizon_episode,
                    multi_agent_system, env, task_spec, episode_idx,
                    max_retries=3,
                    timeout=task_spec.get("timeout", 3600)  # 1 hour default timeout
                )
                
                if episode_result.success:
                    episode_results.append(episode_result.result)
                    logger.info(f"Episode {episode_idx + 1} completed successfully")
                else:
                    logger.error(f"Episode {episode_idx + 1} failed: {episode_result.error}")
                    # Add failed episode with error information
                    failed_result = {
                        "episode_id": episode_idx,
                        "task_spec": task_spec,
                        "success": False,
                        "error": str(episode_result.error),
                        "execution_time": episode_result.execution_time,
                        "recovery_attempts": episode_result.recovery_attempts
                    }
                    episode_results.append(failed_result)
            
            # Aggregate results
            aggregated_results = self._aggregate_long_horizon_results(
                episode_results, task_type, max_horizon
            )
            
            # Add evaluation metadata
            aggregated_results.update({
                "evaluation_id": f"long_horizon_{task_type}_{int(time.time())}",
                "task_type": task_type,
                "max_horizon": max_horizon,
                "num_agents": len(multi_agent_system),
                "num_episodes": num_episodes,
                "total_evaluation_time": time.time() - evaluation_start,
                "evaluation_timestamp": datetime.now().isoformat(),
                "agent_ids": list(multi_agent_system.keys())
            })
            
            self._evaluation_history.append(aggregated_results)
            
            logger.info(f"Long-horizon evaluation completed successfully")
            logger.info(f"Overall success rate: {aggregated_results.get('overall_success_rate', 0):.2%}")
            
            return aggregated_results
            
        except Exception as e:
            logger.error(f"Long-horizon evaluation failed: {e}")
            error_result = {
                "success": False,
                "error": str(e),
                "task_type": task_type,
                "max_horizon": max_horizon,
                "evaluation_time": time.time() - evaluation_start,
                "timestamp": datetime.now().isoformat()
            }
            self._evaluation_history.append(error_result)
            raise
    
    def _generate_long_horizon_tasks(
        self, 
        task_type: str, 
        max_horizon: int, 
        num_tasks: int
    ) -> List[Dict[str, Any]]:
        """Generate long-horizon task specifications.
        
        Args:
            task_type: Type of task to generate
            max_horizon: Maximum planning horizon
            num_tasks: Number of tasks to generate
            
        Returns:
            List of task specifications
        """
        task_generators = {
            "furniture_assembly_complex": self._generate_furniture_assembly_task,
            "collaborative_cooking": self._generate_cooking_task,
            "disaster_response_scenario": self._generate_disaster_response_task,
            "construction_project": self._generate_construction_task,
            "warehouse_logistics": self._generate_warehouse_task,
            "scientific_experiment": self._generate_experiment_task
        }
        
        generator = task_generators.get(task_type, self._generate_generic_task)
        
        tasks = []
        for i in range(num_tasks):
            task_spec = generator(f"{task_type}_{i}", max_horizon, i)
            tasks.append(task_spec)
        
        return tasks
    
    def _generate_furniture_assembly_task(
        self, 
        task_id: str, 
        max_horizon: int, 
        episode_idx: int
    ) -> Dict[str, Any]:
        """Generate complex furniture assembly task.
        
        Args:
            task_id: Task identifier
            max_horizon: Maximum horizon
            episode_idx: Episode index for variation
            
        Returns:
            Task specification
        """
        furniture_types = ["bookshelf", "dining_table", "wardrobe", "office_desk", "kitchen_cabinet"]
        furniture_type = furniture_types[episode_idx % len(furniture_types)]
        
        # Generate hierarchical plan
        plan = HierarchicalTaskPlan(task_id, f"Assemble {furniture_type}", max_horizon)
        
        # Planning phase
        plan.add_phase("plan_assembly", "Plan assembly sequence and assign roles", 
                      TaskPhase.PLANNING, 5.0, ["agent_0"])
        
        # Preparation phases
        plan.add_phase("sort_parts", "Sort and organize all parts", 
                      TaskPhase.COORDINATION, 10.0, ["agent_0", "agent_1"])
        plan.add_phase("prepare_tools", "Prepare required tools", 
                      TaskPhase.COORDINATION, 8.0, ["agent_1"])
        
        # Assembly phases (furniture-specific)
        if furniture_type == "bookshelf":
            phases = [
                ("assemble_frame", "Assemble main frame", 15.0, ["agent_0", "agent_1"]),
                ("attach_shelves", "Attach shelf boards", 20.0, ["agent_0", "agent_1"]),
                ("install_back_panel", "Install back panel", 12.0, ["agent_0"]),
                ("final_assembly", "Final assembly and adjustment", 10.0, ["agent_0", "agent_1"])
            ]
        elif furniture_type == "dining_table":
            phases = [
                ("assemble_legs", "Assemble table legs", 18.0, ["agent_0", "agent_1"]),
                ("attach_tabletop", "Attach tabletop to legs", 25.0, ["agent_0", "agent_1"]),
                ("install_supports", "Install support braces", 15.0, ["agent_0"]),
                ("quality_check", "Quality check and stabilization", 8.0, ["agent_0", "agent_1"])
            ]
        else:
            # Generic assembly phases
            phases = [
                ("base_assembly", "Assemble base structure", 20.0, ["agent_0", "agent_1"]),
                ("main_assembly", "Main assembly process", 30.0, ["agent_0", "agent_1"]),
                ("finishing", "Finishing and adjustments", 15.0, ["agent_0"])
            ]
        
        for phase_id, description, duration, agents in phases:
            plan.add_phase(phase_id, description, TaskPhase.EXECUTION, duration, agents)
        
        # Add dependencies
        plan.add_dependency("sort_parts", "plan_assembly")
        plan.add_dependency("prepare_tools", "plan_assembly")
        
        prev_phase = "sort_parts"
        for phase_id, _, _, _ in phases:
            if prev_phase in ["sort_parts", "prepare_tools"]:
                plan.add_dependency(phase_id, "sort_parts")
                plan.add_dependency(phase_id, "prepare_tools")
            else:
                plan.add_dependency(phase_id, prev_phase)
            prev_phase = phase_id
        
        return {
            "task_id": task_id,
            "name": f"Complex {furniture_type.title()} Assembly",
            "description": f"Multi-agent collaborative assembly of {furniture_type}",
            "furniture_type": furniture_type,
            "plan": plan,
            "success_criteria": {
                "all_phases_completed": True,
                "assembly_stable": True,
                "quality_score": "> 0.8",
                "time_within_budget": True
            },
            "timeout": max_horizon * 10,  # 10 seconds per planning step
            "complexity_score": 0.8 + (episode_idx % 3) * 0.1,
            "required_coordination_events": len(phases) * 2
        }
    
    def _generate_cooking_task(
        self, 
        task_id: str, 
        max_horizon: int, 
        episode_idx: int
    ) -> Dict[str, Any]:
        """Generate collaborative cooking task."""
        recipes = ["pasta_with_sauce", "stir_fry_vegetables", "pizza_from_scratch", "soup_and_bread"]
        recipe = recipes[episode_idx % len(recipes)]
        
        plan = HierarchicalTaskPlan(task_id, f"Cook {recipe}", max_horizon)
        
        # Standard cooking phases
        plan.add_phase("menu_planning", "Plan menu and assign cooking roles", 
                      TaskPhase.PLANNING, 3.0, ["agent_0"])
        plan.add_phase("ingredient_prep", "Prepare and organize ingredients", 
                      TaskPhase.COORDINATION, 8.0, ["agent_0", "agent_1"])
        plan.add_phase("cooking_main", "Cook main dish", 
                      TaskPhase.EXECUTION, 20.0, ["agent_0"])
        plan.add_phase("cooking_side", "Prepare side dishes", 
                      TaskPhase.EXECUTION, 15.0, ["agent_1"])
        plan.add_phase("plating", "Plate and present food", 
                      TaskPhase.COORDINATION, 5.0, ["agent_0", "agent_1"])
        
        # Add dependencies
        plan.add_dependency("ingredient_prep", "menu_planning")
        plan.add_dependency("cooking_main", "ingredient_prep")
        plan.add_dependency("cooking_side", "ingredient_prep")
        plan.add_dependency("plating", "cooking_main")
        plan.add_dependency("plating", "cooking_side")
        
        return {
            "task_id": task_id,
            "name": f"Collaborative Cooking - {recipe.replace('_', ' ').title()}",
            "description": f"Multi-agent cooking of {recipe}",
            "recipe": recipe,
            "plan": plan,
            "success_criteria": {
                "dish_completed": True,
                "food_quality": "> 0.7",
                "kitchen_clean": True,
                "coordination_smooth": True
            },
            "timeout": max_horizon * 8,
            "complexity_score": 0.6 + (episode_idx % 4) * 0.1
        }
    
    def _generate_disaster_response_task(
        self, 
        task_id: str, 
        max_horizon: int, 
        episode_idx: int
    ) -> Dict[str, Any]:
        """Generate disaster response scenario."""
        disaster_types = ["building_collapse", "flood_rescue", "fire_evacuation", "earthquake_response"]
        disaster_type = disaster_types[episode_idx % len(disaster_types)]
        
        plan = HierarchicalTaskPlan(task_id, f"Respond to {disaster_type}", max_horizon)
        
        # Emergency response phases
        plan.add_phase("situation_assessment", "Assess disaster situation", 
                      TaskPhase.PLANNING, 2.0, ["agent_0"])
        plan.add_phase("resource_mobilization", "Mobilize rescue resources", 
                      TaskPhase.COORDINATION, 5.0, ["agent_0", "agent_1", "agent_2"])
        plan.add_phase("area_search", "Search affected areas", 
                      TaskPhase.EXECUTION, 25.0, ["agent_0", "agent_1"])
        plan.add_phase("victim_rescue", "Rescue identified victims", 
                      TaskPhase.EXECUTION, 30.0, ["agent_0", "agent_1", "agent_2"])
        plan.add_phase("medical_triage", "Provide medical triage", 
                      TaskPhase.EXECUTION, 20.0, ["agent_2"])
        plan.add_phase("evacuation", "Evacuate victims to safety", 
                      TaskPhase.COORDINATION, 15.0, ["agent_0", "agent_1", "agent_2"])
        
        # Critical dependencies
        plan.add_dependency("resource_mobilization", "situation_assessment")
        plan.add_dependency("area_search", "resource_mobilization")
        plan.add_dependency("victim_rescue", "area_search")
        plan.add_dependency("medical_triage", "victim_rescue")
        plan.add_dependency("evacuation", "victim_rescue")
        
        return {
            "task_id": task_id,
            "name": f"Emergency Response - {disaster_type.replace('_', ' ').title()}",
            "description": f"Multi-agent response to {disaster_type}",
            "disaster_type": disaster_type,
            "plan": plan,
            "success_criteria": {
                "all_victims_found": True,
                "rescue_time_optimal": True,
                "no_agent_casualties": True,
                "coordination_effective": True
            },
            "timeout": max_horizon * 6,  # Time-critical scenario
            "complexity_score": 0.9,  # High complexity
            "required_agents": 3
        }
    
    def _generate_construction_task(
        self, 
        task_id: str, 
        max_horizon: int, 
        episode_idx: int
    ) -> Dict[str, Any]:
        """Generate construction project task."""
        projects = ["small_house", "bridge", "workshop", "playground"]
        project = projects[episode_idx % len(projects)]
        
        plan = HierarchicalTaskPlan(task_id, f"Build {project}", max_horizon)
        
        # Construction phases
        phases = [
            ("site_preparation", "Prepare construction site", TaskPhase.COORDINATION, 8.0, ["agent_0", "agent_1"]),
            ("foundation", "Lay foundation", TaskPhase.EXECUTION, 20.0, ["agent_0", "agent_1", "agent_2"]),
            ("frame_construction", "Build frame structure", TaskPhase.EXECUTION, 35.0, ["agent_0", "agent_1", "agent_2"]),
            ("utilities", "Install utilities", TaskPhase.EXECUTION, 25.0, ["agent_1", "agent_2"]),
            ("finishing", "Finishing work", TaskPhase.EXECUTION, 20.0, ["agent_0", "agent_1"]),
            ("inspection", "Final inspection", TaskPhase.COMPLETION, 5.0, ["agent_0"])
        ]
        
        for phase_id, description, phase_type, duration, agents in phases:
            plan.add_phase(phase_id, description, phase_type, duration, agents)
        
        # Sequential dependencies
        prev_phase = None
        for phase_id, _, _, _, _ in phases:
            if prev_phase:
                plan.add_dependency(phase_id, prev_phase)
            prev_phase = phase_id
        
        return {
            "task_id": task_id,
            "name": f"Construction Project - {project.replace('_', ' ').title()}",
            "description": f"Multi-agent construction of {project}",
            "project_type": project,
            "plan": plan,
            "success_criteria": {
                "structure_completed": True,
                "safety_standards_met": True,
                "quality_inspection_passed": True,
                "within_time_budget": True
            },
            "timeout": max_horizon * 12,
            "complexity_score": 0.85 + (episode_idx % 2) * 0.1,
            "required_agents": 3
        }
    
    def _generate_warehouse_task(
        self, 
        task_id: str, 
        max_horizon: int, 
        episode_idx: int
    ) -> Dict[str, Any]:
        """Generate warehouse logistics task."""
        operations = ["inventory_audit", "order_fulfillment", "stock_reorganization", "shipment_processing"]
        operation = operations[episode_idx % len(operations)]
        
        plan = HierarchicalTaskPlan(task_id, f"Warehouse {operation}", max_horizon)
        
        # Warehouse phases
        plan.add_phase("system_check", "Check warehouse systems", 
                      TaskPhase.PLANNING, 3.0, ["agent_0"])
        plan.add_phase("task_assignment", "Assign tasks to agents", 
                      TaskPhase.COORDINATION, 5.0, ["agent_0"])
        plan.add_phase("inventory_scan", "Scan inventory areas", 
                      TaskPhase.EXECUTION, 20.0, ["agent_1", "agent_2", "agent_3"])
        plan.add_phase("item_processing", "Process items according to operation", 
                      TaskPhase.EXECUTION, 30.0, ["agent_1", "agent_2", "agent_3"])
        plan.add_phase("quality_control", "Quality control check", 
                      TaskPhase.EXECUTION, 10.0, ["agent_0"])
        plan.add_phase("data_update", "Update warehouse management system", 
                      TaskPhase.COMPLETION, 5.0, ["agent_0"])
        
        # Dependencies
        plan.add_dependency("task_assignment", "system_check")
        plan.add_dependency("inventory_scan", "task_assignment")
        plan.add_dependency("item_processing", "inventory_scan")
        plan.add_dependency("quality_control", "item_processing")
        plan.add_dependency("data_update", "quality_control")
        
        return {
            "task_id": task_id,
            "name": f"Warehouse {operation.replace('_', ' ').title()}",
            "description": f"Multi-agent warehouse {operation}",
            "operation_type": operation,
            "plan": plan,
            "success_criteria": {
                "operation_completed": True,
                "accuracy_high": "> 0.95",
                "efficiency_good": "> 0.8",
                "system_updated": True
            },
            "timeout": max_horizon * 10,
            "complexity_score": 0.7 + (episode_idx % 3) * 0.1,
            "required_agents": 4
        }
    
    def _generate_experiment_task(
        self, 
        task_id: str, 
        max_horizon: int, 
        episode_idx: int
    ) -> Dict[str, Any]:
        """Generate scientific experiment task."""
        experiments = ["chemical_synthesis", "biological_assay", "materials_testing", "data_collection"]
        experiment = experiments[episode_idx % len(experiments)]
        
        plan = HierarchicalTaskPlan(task_id, f"Conduct {experiment}", max_horizon)
        
        # Scientific method phases
        plan.add_phase("protocol_review", "Review experimental protocol", 
                      TaskPhase.PLANNING, 5.0, ["agent_0"])
        plan.add_phase("equipment_setup", "Set up experimental equipment", 
                      TaskPhase.COORDINATION, 15.0, ["agent_0", "agent_1"])
        plan.add_phase("sample_preparation", "Prepare samples", 
                      TaskPhase.EXECUTION, 20.0, ["agent_1"])
        plan.add_phase("experiment_execution", "Execute experiment", 
                      TaskPhase.EXECUTION, 25.0, ["agent_0", "agent_1"])
        plan.add_phase("data_collection", "Collect and record data", 
                      TaskPhase.EXECUTION, 15.0, ["agent_0"])
        plan.add_phase("cleanup", "Clean up equipment and workspace", 
                      TaskPhase.COMPLETION, 10.0, ["agent_0", "agent_1"])
        
        # Scientific dependencies
        plan.add_dependency("equipment_setup", "protocol_review")
        plan.add_dependency("sample_preparation", "equipment_setup")
        plan.add_dependency("experiment_execution", "sample_preparation")
        plan.add_dependency("data_collection", "experiment_execution")
        plan.add_dependency("cleanup", "data_collection")
        
        return {
            "task_id": task_id,
            "name": f"Scientific Experiment - {experiment.replace('_', ' ').title()}",
            "description": f"Multi-agent {experiment} experiment",
            "experiment_type": experiment,
            "plan": plan,
            "success_criteria": {
                "protocol_followed": True,
                "data_quality_high": "> 0.9",
                "safety_maintained": True,
                "experiment_completed": True
            },
            "timeout": max_horizon * 9,
            "complexity_score": 0.75 + (episode_idx % 2) * 0.15,
            "required_agents": 2
        }
    
    def _generate_generic_task(
        self, 
        task_id: str, 
        max_horizon: int, 
        episode_idx: int
    ) -> Dict[str, Any]:
        """Generate generic long-horizon task."""
        plan = HierarchicalTaskPlan(task_id, "Generic long-horizon task", max_horizon)
        
        # Generic phases
        plan.add_phase("planning", "Plan task execution", TaskPhase.PLANNING, 5.0, ["agent_0"])
        plan.add_phase("preparation", "Prepare for execution", TaskPhase.COORDINATION, 10.0, ["agent_0", "agent_1"])
        plan.add_phase("execution", "Execute main task", TaskPhase.EXECUTION, 30.0, ["agent_0", "agent_1"])
        plan.add_phase("completion", "Complete and verify task", TaskPhase.COMPLETION, 8.0, ["agent_0"])
        
        plan.add_dependency("preparation", "planning")
        plan.add_dependency("execution", "preparation")
        plan.add_dependency("completion", "execution")
        
        return {
            "task_id": task_id,
            "name": "Generic Long-Horizon Task",
            "description": "Generic multi-agent long-horizon task",
            "plan": plan,
            "success_criteria": {"task_completed": True},
            "timeout": max_horizon * 10,
            "complexity_score": 0.5
        }
    
    def _execute_long_horizon_episode(
        self, 
        multi_agent_system: Dict[str, BaseAgent],
        env: BaseEnv,
        task_spec: Dict[str, Any],
        episode_id: int
    ) -> Dict[str, Any]:
        """Execute a single long-horizon episode.
        
        Args:
            multi_agent_system: Dictionary of agents
            env: Environment
            task_spec: Task specification
            episode_id: Episode identifier
            
        Returns:
            Episode execution results
        """
        episode_start = time.time()
        plan = task_spec["plan"]
        
        # Initialize episode tracking
        episode_result = {
            "episode_id": episode_id,
            "task_spec": task_spec,
            "start_time": episode_start,
            "plan_execution": [],
            "coordination_events": [],
            "errors": [],
            "recovery_actions": [],
            "agent_performance": defaultdict(list),
            "phase_results": [],
            "success": False,
            "completion_reason": "unknown"
        }
        
        # Store active plan for coordination
        plan_id = f"{task_spec['task_id']}_{episode_id}"
        self._active_plans[plan_id] = plan
        
        try:
            # Reset environment
            obs = env.reset()
            
            # Execute plan phases
            while True:
                # Get next executable phases
                executable_phases = plan.get_next_executable_phases()
                
                if not executable_phases:
                    # Check if all phases completed
                    all_completed = all(p["status"] == "completed" for p in plan.phases)
                    if all_completed:
                        episode_result["success"] = True
                        episode_result["completion_reason"] = "all_phases_completed"
                        break
                    else:
                        # Check for failed phases or deadlock
                        failed_phases = [p for p in plan.phases if p["status"] == "failed"]
                        if failed_phases:
                            episode_result["completion_reason"] = "phase_failed"
                            break
                        else:
                            episode_result["completion_reason"] = "execution_deadlock"
                            break
                
                # Execute phases (can be parallel if they don't conflict)
                for phase in executable_phases[:2]:  # Execute up to 2 phases in parallel
                    phase_result = self._execute_phase(
                        phase, multi_agent_system, env, obs, episode_result
                    )
                    episode_result["phase_results"].append(phase_result)
                    
                    # Update phase status based on result
                    if phase_result["success"]:
                        plan.update_phase_status(
                            phase["phase_id"], "completed",
                            actual_duration=phase_result["execution_time"],
                            end_time=time.time()
                        )
                    else:
                        plan.update_phase_status(
                            phase["phase_id"], "failed",
                            error_count=phase["error_count"] + 1
                        )
                
                # Check timeout
                if time.time() - episode_start > task_spec.get("timeout", 3600):
                    episode_result["completion_reason"] = "timeout"
                    break
            
            # Final episode statistics
            episode_result.update({
                "end_time": time.time(),
                "execution_time": time.time() - episode_start,
                "plan_statistics": plan.get_execution_statistics(),
                "final_plan_status": {p["phase_id"]: p["status"] for p in plan.phases}
            })
            
        except Exception as e:
            logger.error(f"Episode {episode_id} execution failed: {e}")
            episode_result.update({
                "success": False,
                "error": str(e),
                "completion_reason": "execution_error",
                "execution_time": time.time() - episode_start
            })
            episode_result["errors"].append({
                "timestamp": time.time(),
                "error": str(e),
                "phase": "episode_execution"
            })
        
        finally:
            # Clean up active plan
            if plan_id in self._active_plans:
                del self._active_plans[plan_id]
        
        return episode_result
    
    def _execute_phase(
        self, 
        phase: Dict[str, Any], 
        multi_agent_system: Dict[str, BaseAgent],
        env: BaseEnv,
        current_obs: Dict[str, Any],
        episode_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute a single phase of the plan.
        
        Args:
            phase: Phase specification
            multi_agent_system: Dictionary of agents
            env: Environment
            current_obs: Current observations
            episode_context: Episode context for coordination
            
        Returns:
            Phase execution result
        """
        phase_start = time.time()
        phase_id = phase["phase_id"]
        required_agents = phase["required_agents"]
        
        logger.debug(f"Executing phase {phase_id} with agents {required_agents}")
        
        phase_result = {
            "phase_id": phase_id,
            "start_time": phase_start,
            "required_agents": required_agents,
            "steps": [],
            "coordination_events": [],
            "success": False,
            "error": None
        }
        
        # Update phase status
        phase["status"] = "executing"
        phase["start_time"] = phase_start
        
        try:
            # Execute phase-specific logic
            max_phase_steps = int(phase["estimated_duration"] * 2)  # 2 steps per time unit
            
            for step in range(max_phase_steps):
                # Get actions from required agents
                agent_actions = {}
                for agent_id in required_agents:
                    if agent_id in multi_agent_system:
                        # Add phase context to observations
                        agent_obs = current_obs.copy()
                        agent_obs["phase_info"] = {
                            "phase_id": phase_id,
                            "phase_type": phase["phase_type"].value,
                            "description": phase["description"],
                            "step": step,
                            "max_steps": max_phase_steps
                        }
                        
                        action = multi_agent_system[agent_id].act(agent_obs)
                        agent_actions[agent_id] = action
                
                # Execute joint action
                if agent_actions:
                    next_obs, rewards, done, info = env.step(agent_actions)
                    
                    # Record step
                    step_data = {
                        "step": step,
                        "timestamp": time.time(),
                        "agent_actions": agent_actions,
                        "observations": current_obs,
                        "rewards": rewards,
                        "done": done,
                        "info": info
                    }
                    phase_result["steps"].append(step_data)
                    
                    # Check for phase completion criteria
                    if self._check_phase_completion(phase, step_data, episode_context):
                        phase_result["success"] = True
                        break
                    
                    current_obs = next_obs
                    
                    # Check for early termination
                    if done:
                        break
                else:
                    logger.warning(f"No agents available for phase {phase_id}")
                    break
            
            # Check final success if not already determined
            if not phase_result["success"]:
                phase_result["success"] = self._evaluate_phase_success(phase, phase_result)
            
        except Exception as e:
            logger.error(f"Phase {phase_id} execution failed: {e}")
            phase_result["error"] = str(e)
            phase_result["success"] = False
        
        finally:
            phase_result["execution_time"] = time.time() - phase_start
            phase_result["end_time"] = time.time()
            
            logger.debug(f"Phase {phase_id} completed. Success: {phase_result['success']}")
        
        return phase_result
    
    def _check_phase_completion(
        self, 
        phase: Dict[str, Any], 
        step_data: Dict[str, Any],
        episode_context: Dict[str, Any]
    ) -> bool:
        """Check if phase completion criteria are met.
        
        Args:
            phase: Phase specification
            step_data: Current step data
            episode_context: Episode context
            
        Returns:
            True if phase should be completed
        """
        # Phase-type specific completion logic
        phase_type = phase["phase_type"]
        info = step_data.get("info", {})
        
        if phase_type == TaskPhase.PLANNING:
            # Planning phases complete when plan is ready
            return info.get("plan_ready", False) or step_data["step"] >= 5
        
        elif phase_type == TaskPhase.COORDINATION:
            # Coordination phases complete when agents are synchronized
            return info.get("agents_synchronized", False) or step_data["step"] >= 10
        
        elif phase_type == TaskPhase.EXECUTION:
            # Execution phases complete when task objective is achieved
            return (
                info.get("objective_completed", False) or 
                info.get("phase_success", False) or
                step_data["step"] >= 30  # Timeout
            )
        
        elif phase_type == TaskPhase.COMPLETION:
            # Completion phases are typically short verification steps
            return info.get("verification_passed", True) or step_data["step"] >= 3
        
        else:
            # Default completion after reasonable number of steps
            return step_data["step"] >= 15
    
    def _evaluate_phase_success(
        self, 
        phase: Dict[str, Any], 
        phase_result: Dict[str, Any]
    ) -> bool:
        """Evaluate overall phase success.
        
        Args:
            phase: Phase specification
            phase_result: Phase execution result
            
        Returns:
            True if phase was successful
        """
        # Check if phase had any steps
        if not phase_result["steps"]:
            return False
        
        # Check final step info
        final_step = phase_result["steps"][-1]
        final_info = final_step.get("info", {})
        
        # Phase-specific success criteria
        phase_type = phase["phase_type"]
        
        if phase_type == TaskPhase.PLANNING:
            return len(phase_result["steps"]) >= 2  # Minimum planning effort
        elif phase_type == TaskPhase.COORDINATION:
            return final_info.get("coordination_quality", 0.5) > 0.6
        elif phase_type == TaskPhase.EXECUTION:
            return final_info.get("task_progress", 0.0) > 0.7
        else:
            return len(phase_result["steps"]) > 0  # At least attempted
    
    def _aggregate_long_horizon_results(
        self, 
        episode_results: List[Dict[str, Any]], 
        task_type: str,
        max_horizon: int
    ) -> Dict[str, Any]:
        """Aggregate results across long-horizon episodes.
        
        Args:
            episode_results: List of episode results
            task_type: Type of task
            max_horizon: Maximum horizon used
            
        Returns:
            Aggregated results
        """
        if not episode_results:
            return {"error": "No episodes to aggregate"}
        
        successful_episodes = [ep for ep in episode_results if ep.get("success", False)]
        
        # Basic success metrics
        overall_success_rate = len(successful_episodes) / len(episode_results)
        
        # Execution time statistics
        execution_times = [ep.get("execution_time", 0) for ep in episode_results]
        avg_execution_time = np.mean(execution_times) if execution_times else 0
        
        # Phase completion statistics
        all_phase_results = []
        for ep in episode_results:
            all_phase_results.extend(ep.get("phase_results", []))
        
        phase_success_rate = (
            len([p for p in all_phase_results if p.get("success", False)]) / 
            max(len(all_phase_results), 1)
        )
        
        # Coordination quality
        all_coordination_events = []
        for ep in episode_results:
            all_coordination_events.extend(ep.get("coordination_events", []))
        
        coordination_quality = np.mean([
            event.get("quality_score", 0.5) for event in all_coordination_events
        ]) if all_coordination_events else 0.5
        
        # Error analysis
        total_errors = sum(len(ep.get("errors", [])) for ep in episode_results)
        error_rate = total_errors / max(len(episode_results), 1)
        
        # Planning horizon utilization
        plan_statistics = [ep.get("plan_statistics", {}) for ep in successful_episodes]
        avg_phases_completed = np.mean([
            stats.get("completed_phases", 0) for stats in plan_statistics
        ]) if plan_statistics else 0
        
        avg_time_efficiency = np.mean([
            stats.get("time_efficiency", 0) for stats in plan_statistics
        ]) if plan_statistics else 0
        
        return {
            "overall_success_rate": overall_success_rate,
            "avg_execution_time": avg_execution_time,
            "phase_success_rate": phase_success_rate,
            "coordination_quality": coordination_quality,
            "error_rate": error_rate,
            "avg_phases_completed": avg_phases_completed,
            "avg_time_efficiency": avg_time_efficiency,
            "total_episodes": len(episode_results),
            "successful_episodes": len(successful_episodes),
            "total_phases_executed": len(all_phase_results),
            "total_coordination_events": len(all_coordination_events),
            "horizon_utilization": avg_phases_completed / max_horizon if max_horizon > 0 else 0,
            "detailed_episode_results": episode_results,
            "task_type_performance": {
                task_type: {
                    "success_rate": overall_success_rate,
                    "avg_execution_time": avg_execution_time,
                    "coordination_quality": coordination_quality
                }
            }
        }
    
    def generate_long_horizon_report(self, results: Dict[str, Any]) -> str:
        """Generate comprehensive long-horizon evaluation report.
        
        Args:
            results: Evaluation results
            
        Returns:
            Formatted report string
        """
        report = []
        report.append("\n" + "="*70)
        report.append("LONG-HORIZON MULTI-AGENT BENCHMARK REPORT")
        report.append("="*70)
        
        # Basic information
        report.append(f"\nEvaluation ID: {results.get('evaluation_id', 'unknown')}")
        report.append(f"Task Type: {results.get('task_type', 'unknown')}")
        report.append(f"Max Horizon: {results.get('max_horizon', 'unknown')}")
        report.append(f"Number of Agents: {results.get('num_agents', 'unknown')}")
        report.append(f"Number of Episodes: {results.get('num_episodes', 'unknown')}")
        report.append(f"Total Evaluation Time: {results.get('total_evaluation_time', 0):.2f}s")
        
        # Success metrics
        report.append("\nSUCCESS METRICS:")
        report.append(f"  Overall Success Rate: {results.get('overall_success_rate', 0):.2%}")
        report.append(f"  Phase Success Rate: {results.get('phase_success_rate', 0):.2%}")
        report.append(f"  Average Execution Time: {results.get('avg_execution_time', 0):.1f}s")
        report.append(f"  Horizon Utilization: {results.get('horizon_utilization', 0):.2%}")
        
        # Coordination metrics
        report.append("\nCOORDINATION METRICS:")
        report.append(f"  Coordination Quality: {results.get('coordination_quality', 0):.3f}")
        report.append(f"  Total Coordination Events: {results.get('total_coordination_events', 0)}")
        report.append(f"  Average Phases Completed: {results.get('avg_phases_completed', 0):.1f}")
        
        # Performance metrics
        report.append("\nPERFORMANCE METRICS:")
        report.append(f"  Time Efficiency: {results.get('avg_time_efficiency', 0):.3f}")
        report.append(f"  Error Rate: {results.get('error_rate', 0):.2f} errors/episode")
        report.append(f"  Total Phases Executed: {results.get('total_phases_executed', 0)}")
        
        # Task-specific performance
        task_performance = results.get('task_type_performance', {})
        if task_performance:
            report.append("\nTASK-SPECIFIC PERFORMANCE:")
            for task_type, metrics in task_performance.items():
                report.append(f"  {task_type.replace('_', ' ').title()}:")
                report.append(f"    Success Rate: {metrics.get('success_rate', 0):.2%}")
                report.append(f"    Avg Time: {metrics.get('avg_execution_time', 0):.1f}s")
                report.append(f"    Coordination: {metrics.get('coordination_quality', 0):.3f}")
        
        report.append("\n" + "="*70)
        
        return "\n".join(report)
    
    def get_evaluation_history(self) -> List[Dict[str, Any]]:
        """Get history of long-horizon evaluations.
        
        Returns:
            List of evaluation results
        """
        return self._evaluation_history.copy()
    
    def save_results(self, results: Dict[str, Any], filepath: str):
        """Save long-horizon evaluation results.
        
        Args:
            results: Results to save
            filepath: Output file path
        """
        import json
        
        # Convert numpy arrays and other non-serializable objects
        def convert_for_json(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, datetime):
                return obj.isoformat()
            elif isinstance(obj, TaskPhase):
                return obj.value
            elif isinstance(obj, dict):
                return {key: convert_for_json(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_for_json(item) for item in obj]
            else:
                return obj
        
        serializable_results = convert_for_json(results)
        
        with open(filepath, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        logger.info(f"Long-horizon evaluation results saved to {filepath}")
