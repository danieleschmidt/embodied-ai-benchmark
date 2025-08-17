"""Advanced Hierarchical Task Decomposition with Neural-Symbolic Reasoning.

Novel contributions:
1. Attention-based task hierarchy discovery
2. Neural-symbolic reasoning for goal decomposition  
3. Dynamic task graph construction and optimization
4. Multi-agent coordination through hierarchical planning
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GraphConv, global_mean_pool
from torch_geometric.data import Data, Batch
import numpy as np
from typing import Any, Dict, List, Optional, Tuple, Union, Set
from dataclasses import dataclass, field
from enum import Enum
import networkx as nx
from collections import defaultdict, deque
import time
import math
from abc import ABC, abstractmethod

from ..utils.logging_config import get_logger

logger = get_logger(__name__)


class TaskType(Enum):
    """Types of hierarchical tasks."""
    PRIMITIVE = "primitive"
    COMPOSITE = "composite"
    ABSTRACT = "abstract"
    COORDINATION = "coordination"


class GoalStatus(Enum):
    """Status of goal achievement."""
    PENDING = "pending"
    ACTIVE = "active"
    COMPLETED = "completed"
    FAILED = "failed"
    SUSPENDED = "suspended"


@dataclass
class SubGoal:
    """Individual sub-goal in task hierarchy."""
    goal_id: str
    description: str
    task_type: TaskType
    preconditions: List[str] = field(default_factory=list)
    effects: List[str] = field(default_factory=list)
    required_skills: List[str] = field(default_factory=list)
    estimated_duration: float = 1.0
    complexity_score: float = 0.5
    priority: float = 0.5
    status: GoalStatus = GoalStatus.PENDING
    parent_goals: List[str] = field(default_factory=list)
    child_goals: List[str] = field(default_factory=list)
    resource_requirements: Dict[str, float] = field(default_factory=dict)
    success_criteria: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TaskDecompositionResult:
    """Result of hierarchical task decomposition."""
    goal_hierarchy: Dict[str, SubGoal]
    execution_order: List[str]
    dependency_graph: nx.DiGraph
    coordination_requirements: Dict[str, List[str]]
    estimated_completion_time: float
    complexity_analysis: Dict[str, float]


class SymbolicReasoner:
    """Neural-symbolic reasoning engine for goal decomposition."""
    
    def __init__(self, knowledge_base: Optional[Dict[str, Any]] = None):
        self.knowledge_base = knowledge_base or {}
        self.rule_base = []
        self.logical_constraints = []
        
        # Initialize with basic embodied AI rules
        self._initialize_embodied_ai_rules()
    
    def _initialize_embodied_ai_rules(self):
        """Initialize with domain-specific rules for embodied AI."""
        # Basic manipulation rules
        self.add_rule("manipulation", {
            "prerequisites": ["object_detection", "grasp_planning"],
            "decomposition": ["approach_object", "grasp_object", "manipulate_object"],
            "constraints": ["no_collision", "stable_grasp"]
        })
        
        # Navigation rules
        self.add_rule("navigation", {
            "prerequisites": ["path_planning", "localization"],
            "decomposition": ["generate_path", "follow_path", "avoid_obstacles"],
            "constraints": ["safety_bounds", "kinematic_limits"]
        })
        
        # Assembly rules
        self.add_rule("assembly", {
            "prerequisites": ["part_identification", "assembly_sequence"],
            "decomposition": ["align_parts", "insert_part", "verify_assembly"],
            "constraints": ["force_limits", "precision_requirements"]
        })
        
        # Multi-agent coordination
        self.add_rule("coordination", {
            "prerequisites": ["communication", "role_assignment"],
            "decomposition": ["coordinate_actions", "synchronize_timing", "resolve_conflicts"],
            "constraints": ["no_interference", "resource_sharing"]
        })
    
    def add_rule(self, rule_name: str, rule_definition: Dict[str, Any]):
        """Add a reasoning rule to the rule base."""
        self.rule_base.append({
            "name": rule_name,
            "definition": rule_definition
        })
    
    def decompose_goal(self, 
                      goal_description: str,
                      context: Dict[str, Any],
                      depth_limit: int = 5) -> List[SubGoal]:
        """Decompose high-level goal using symbolic reasoning."""
        # Parse goal description and extract task type
        task_type = self._infer_task_type(goal_description, context)
        
        # Find applicable rules
        applicable_rules = self._find_applicable_rules(task_type, context)
        
        # Generate sub-goals using rules
        sub_goals = []
        for rule in applicable_rules:
            rule_subgoals = self._apply_rule(rule, goal_description, context, depth_limit)
            sub_goals.extend(rule_subgoals)
        
        # Remove duplicates and refine
        sub_goals = self._refine_subgoals(sub_goals, context)
        
        return sub_goals
    
    def _infer_task_type(self, goal_description: str, context: Dict[str, Any]) -> str:
        """Infer task type from goal description."""
        description_lower = goal_description.lower()
        
        # Simple keyword-based classification (in practice would use NLP)
        if any(word in description_lower for word in ["move", "navigate", "go to"]):
            return "navigation"
        elif any(word in description_lower for word in ["pick", "grasp", "manipulate", "hold"]):
            return "manipulation"
        elif any(word in description_lower for word in ["assemble", "build", "construct"]):
            return "assembly"
        elif any(word in description_lower for word in ["coordinate", "collaborate", "together"]):
            return "coordination"
        else:
            return "general"
    
    def _find_applicable_rules(self, task_type: str, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Find rules applicable to the given task type."""
        applicable_rules = []
        
        for rule in self.rule_base:
            if rule["name"] == task_type or "general" in rule["name"]:
                # Check if prerequisites are met
                prerequisites = rule["definition"].get("prerequisites", [])
                if self._check_prerequisites(prerequisites, context):
                    applicable_rules.append(rule)
        
        return applicable_rules
    
    def _check_prerequisites(self, prerequisites: List[str], context: Dict[str, Any]) -> bool:
        """Check if prerequisites are satisfied in the current context."""
        available_capabilities = context.get("capabilities", [])
        return all(prereq in available_capabilities for prereq in prerequisites)
    
    def _apply_rule(self, 
                   rule: Dict[str, Any], 
                   goal_description: str,
                   context: Dict[str, Any],
                   depth_limit: int) -> List[SubGoal]:
        """Apply a specific rule to generate sub-goals."""
        decomposition = rule["definition"].get("decomposition", [])
        constraints = rule["definition"].get("constraints", [])
        
        sub_goals = []
        for i, sub_task in enumerate(decomposition):
            sub_goal = SubGoal(
                goal_id=f"{goal_description}_{sub_task}_{i}",
                description=f"{sub_task} for {goal_description}",
                task_type=TaskType.COMPOSITE if depth_limit > 1 else TaskType.PRIMITIVE,
                required_skills=[sub_task],
                estimated_duration=context.get("duration_estimates", {}).get(sub_task, 1.0),
                complexity_score=context.get("complexity_scores", {}).get(sub_task, 0.5),
                priority=1.0 / (i + 1),  # Earlier tasks have higher priority
                preconditions=constraints,
                effects=[f"completed_{sub_task}"]
            )
            sub_goals.append(sub_goal)
        
        return sub_goals
    
    def _refine_subgoals(self, sub_goals: List[SubGoal], context: Dict[str, Any]) -> List[SubGoal]:
        """Refine and deduplicate sub-goals."""
        # Remove duplicates based on description similarity
        unique_goals = []
        seen_descriptions = set()
        
        for goal in sub_goals:
            if goal.description not in seen_descriptions:
                unique_goals.append(goal)
                seen_descriptions.add(goal.description)
        
        return unique_goals


class TaskHierarchyEncoder(nn.Module):
    """Neural network for encoding task hierarchies."""
    
    def __init__(self, 
                 vocab_size: int = 10000,
                 embedding_dim: int = 256,
                 hidden_dim: int = 512,
                 num_layers: int = 3):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        
        # Text embedding for goal descriptions
        self.text_embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # Task feature encoder
        self.task_encoder = nn.Sequential(
            nn.Linear(embedding_dim + 10, hidden_dim),  # +10 for numerical features
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embedding_dim)
        )
        
        # Hierarchical attention mechanism
        self.hierarchy_attention = nn.MultiheadAttention(
            embed_dim=embedding_dim,
            num_heads=8,
            batch_first=True
        )
        
        # Graph neural network for task dependencies
        self.gcn_layers = nn.ModuleList([
            GCNConv(embedding_dim, embedding_dim) for _ in range(num_layers)
        ])
        
        # Task decomposition predictor
        self.decomposition_predictor = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, vocab_size)  # Predict next sub-task
        )
        
    def forward(self, 
                task_descriptions: torch.Tensor,
                task_features: torch.Tensor,
                dependency_graph: Optional[Data] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode task hierarchy and predict decomposition.
        
        Args:
            task_descriptions: [batch_size, seq_len] tokenized descriptions
            task_features: [batch_size, feature_dim] numerical task features
            dependency_graph: PyTorch Geometric graph of task dependencies
            
        Returns:
            task_embeddings: [batch_size, embedding_dim] encoded task representations
            decomposition_logits: [batch_size, vocab_size] next sub-task predictions
        """
        batch_size = task_descriptions.size(0)
        
        # Embed task descriptions
        text_embeds = self.text_embedding(task_descriptions).mean(dim=1)  # Average pooling
        
        # Combine with numerical features
        combined_features = torch.cat([text_embeds, task_features], dim=-1)
        task_embeddings = self.task_encoder(combined_features)
        
        # Apply hierarchical attention
        attended_embeddings, attention_weights = self.hierarchy_attention(
            task_embeddings.unsqueeze(1),
            task_embeddings.unsqueeze(1),
            task_embeddings.unsqueeze(1)
        )
        attended_embeddings = attended_embeddings.squeeze(1)
        
        # Apply graph neural network if dependency graph provided
        if dependency_graph is not None:
            graph_embeddings = attended_embeddings
            for gcn_layer in self.gcn_layers:
                graph_embeddings = F.relu(gcn_layer(graph_embeddings, dependency_graph.edge_index))
            attended_embeddings = graph_embeddings + attended_embeddings  # Residual connection
        
        # Predict decomposition
        decomposition_logits = self.decomposition_predictor(attended_embeddings)
        
        return attended_embeddings, decomposition_logits


class DynamicTaskGraph:
    """Dynamic task graph that evolves during execution."""
    
    def __init__(self):
        self.graph = nx.DiGraph()
        self.execution_history = []
        self.resource_allocation = {}
        self.timing_constraints = {}
        
    def add_task_node(self, sub_goal: SubGoal):
        """Add a task node to the graph."""
        self.graph.add_node(
            sub_goal.goal_id,
            goal=sub_goal,
            status=sub_goal.status,
            start_time=None,
            end_time=None,
            actual_duration=None
        )
    
    def add_dependency(self, parent_id: str, child_id: str, dependency_type: str = "sequential"):
        """Add dependency edge between tasks."""
        self.graph.add_edge(parent_id, child_id, type=dependency_type)
    
    def update_task_status(self, task_id: str, new_status: GoalStatus, timestamp: float):
        """Update task status and record timing."""
        if task_id in self.graph.nodes:
            self.graph.nodes[task_id]['status'] = new_status
            
            if new_status == GoalStatus.ACTIVE:
                self.graph.nodes[task_id]['start_time'] = timestamp
            elif new_status in [GoalStatus.COMPLETED, GoalStatus.FAILED]:
                self.graph.nodes[task_id]['end_time'] = timestamp
                start_time = self.graph.nodes[task_id]['start_time']
                if start_time is not None:
                    self.graph.nodes[task_id]['actual_duration'] = timestamp - start_time
            
            # Record in execution history
            self.execution_history.append({
                'task_id': task_id,
                'status': new_status,
                'timestamp': timestamp
            })
    
    def get_ready_tasks(self) -> List[str]:
        """Get tasks that are ready to execute (all dependencies completed)."""
        ready_tasks = []
        
        for node_id in self.graph.nodes:
            node_data = self.graph.nodes[node_id]
            if node_data['status'] == GoalStatus.PENDING:
                # Check if all dependencies are completed
                dependencies_completed = True
                for pred in self.graph.predecessors(node_id):
                    pred_status = self.graph.nodes[pred]['status']
                    if pred_status != GoalStatus.COMPLETED:
                        dependencies_completed = False
                        break
                
                if dependencies_completed:
                    ready_tasks.append(node_id)
        
        return ready_tasks
    
    def get_critical_path(self) -> List[str]:
        """Compute critical path through task graph."""
        # Use topological sort to find longest path
        try:
            topo_order = list(nx.topological_sort(self.graph))
            
            # Calculate longest path (critical path)
            distances = {node: 0 for node in self.graph.nodes}
            predecessors = {node: None for node in self.graph.nodes}
            
            for node in topo_order:
                goal = self.graph.nodes[node]['goal']
                node_duration = goal.estimated_duration
                
                for successor in self.graph.successors(node):
                    new_distance = distances[node] + node_duration
                    if new_distance > distances[successor]:
                        distances[successor] = new_distance
                        predecessors[successor] = node
            
            # Reconstruct critical path
            max_node = max(distances, key=distances.get)
            critical_path = []
            current = max_node
            
            while current is not None:
                critical_path.append(current)
                current = predecessors[current]
            
            return list(reversed(critical_path))
            
        except nx.NetworkXError:
            return []  # Graph has cycles
    
    def optimize_execution_order(self) -> List[str]:
        """Optimize task execution order considering dependencies and priorities."""
        ready_tasks = self.get_ready_tasks()
        execution_order = []
        
        while ready_tasks:
            # Sort by priority and complexity
            task_scores = []
            for task_id in ready_tasks:
                goal = self.graph.nodes[task_id]['goal']
                # Score combines priority, inverse complexity, and resource availability
                score = goal.priority * (1.0 - goal.complexity_score) * self._get_resource_availability(goal)
                task_scores.append((task_id, score))
            
            # Select highest scoring task
            task_scores.sort(key=lambda x: x[1], reverse=True)
            selected_task = task_scores[0][0]
            
            execution_order.append(selected_task)
            ready_tasks.remove(selected_task)
            
            # Update ready tasks (may have new tasks ready after completing dependencies)
            self.update_task_status(selected_task, GoalStatus.COMPLETED, time.time())
            new_ready = self.get_ready_tasks()
            for task in new_ready:
                if task not in ready_tasks and task not in execution_order:
                    ready_tasks.append(task)
        
        return execution_order
    
    def _get_resource_availability(self, goal: SubGoal) -> float:
        """Estimate resource availability for a goal."""
        # Simplified resource availability calculation
        required_resources = goal.resource_requirements
        availability_score = 1.0
        
        for resource, amount in required_resources.items():
            allocated = self.resource_allocation.get(resource, 0.0)
            available = max(0.0, 1.0 - allocated)
            if amount > available:
                availability_score *= available / amount
        
        return availability_score
    
    def detect_conflicts(self) -> List[Dict[str, Any]]:
        """Detect resource conflicts and timing conflicts."""
        conflicts = []
        
        # Resource conflicts
        resource_usage = defaultdict(list)
        for node_id in self.graph.nodes:
            goal = self.graph.nodes[node_id]['goal']
            for resource, amount in goal.resource_requirements.items():
                resource_usage[resource].append((node_id, amount))
        
        for resource, usage_list in resource_usage.items():
            total_usage = sum(amount for _, amount in usage_list)
            if total_usage > 1.0:  # Resource overallocation
                conflicts.append({
                    'type': 'resource_conflict',
                    'resource': resource,
                    'total_usage': total_usage,
                    'tasks': [task_id for task_id, _ in usage_list]
                })
        
        # Timing conflicts (simplified)
        active_tasks = [node_id for node_id in self.graph.nodes 
                       if self.graph.nodes[node_id]['status'] == GoalStatus.ACTIVE]
        
        if len(active_tasks) > 1:
            conflicts.append({
                'type': 'timing_conflict',
                'conflicting_tasks': active_tasks,
                'description': 'Multiple tasks active simultaneously'
            })
        
        return conflicts


class HierarchicalTaskDecomposer:
    """Main hierarchical task decomposition system."""
    
    def __init__(self,
                 vocab_size: int = 10000,
                 embedding_dim: int = 256,
                 max_depth: int = 5,
                 use_symbolic_reasoning: bool = True):
        """
        Initialize hierarchical task decomposer.
        
        Args:
            vocab_size: Size of task vocabulary
            embedding_dim: Embedding dimension for neural components
            max_depth: Maximum decomposition depth
            use_symbolic_reasoning: Whether to use symbolic reasoning
        """
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.max_depth = max_depth
        self.use_symbolic_reasoning = use_symbolic_reasoning
        
        # Initialize components
        self.task_encoder = TaskHierarchyEncoder(vocab_size, embedding_dim)
        
        if use_symbolic_reasoning:
            self.symbolic_reasoner = SymbolicReasoner()
        else:
            self.symbolic_reasoner = None
        
        self.task_graph = DynamicTaskGraph()
        
        # Task vocabulary (simple tokenization)
        self.vocab = self._build_vocabulary()
        self.token_to_id = {token: i for i, token in enumerate(self.vocab)}
        self.id_to_token = {i: token for token, i in self.token_to_id.items()}
        
        # Performance tracking
        self.decomposition_stats = {
            'total_decompositions': 0,
            'avg_depth': 0.0,
            'avg_branching_factor': 0.0,
            'success_rate': 0.0
        }
    
    def _build_vocabulary(self) -> List[str]:
        """Build task vocabulary for tokenization."""
        base_vocab = [
            # Basic actions
            'move', 'pick', 'place', 'grasp', 'release', 'navigate', 'turn', 'approach',
            'manipulate', 'assemble', 'disassemble', 'insert', 'remove', 'rotate',
            'push', 'pull', 'lift', 'lower', 'align', 'connect', 'disconnect',
            
            # Objects
            'object', 'tool', 'part', 'component', 'surface', 'container', 'obstacle',
            'target', 'goal', 'location', 'position', 'orientation', 'chair', 'table',
            'screw', 'bolt', 'plate', 'beam', 'joint', 'connection',
            
            # Properties
            'left', 'right', 'top', 'bottom', 'front', 'back', 'near', 'far',
            'heavy', 'light', 'large', 'small', 'precise', 'rough', 'smooth',
            'stable', 'unstable', 'secure', 'loose', 'tight',
            
            # Actions
            'coordinate', 'synchronize', 'communicate', 'collaborate', 'share',
            'allocate', 'distribute', 'optimize', 'plan', 'execute', 'monitor',
            'verify', 'check', 'validate', 'test', 'measure', 'calibrate',
            
            # Special tokens
            '<start>', '<end>', '<unk>', '<pad>'
        ]
        
        # Extend to vocab_size with numbered variants
        extended_vocab = base_vocab[:]
        for i in range(len(base_vocab), self.vocab_size):
            extended_vocab.append(f'<token_{i}>')
        
        return extended_vocab[:self.vocab_size]
    
    def decompose_task(self,
                      main_goal: str,
                      context: Dict[str, Any],
                      agent_capabilities: List[str],
                      resource_constraints: Dict[str, float]) -> TaskDecompositionResult:
        """
        Perform hierarchical task decomposition.
        
        Args:
            main_goal: High-level goal description
            context: Task context and environment information
            agent_capabilities: Available agent capabilities
            resource_constraints: Resource availability constraints
            
        Returns:
            Complete task decomposition result
        """
        logger.info(f"Decomposing task: {main_goal}")
        
        # Initialize task graph
        self.task_graph = DynamicTaskGraph()
        
        # Enhanced context with capabilities and constraints
        enhanced_context = {
            **context,
            'capabilities': agent_capabilities,
            'resource_constraints': resource_constraints,
            'main_goal': main_goal
        }
        
        # Recursive decomposition
        all_goals = self._recursive_decompose(main_goal, enhanced_context, depth=0)
        
        # Build task graph
        for goal in all_goals:
            self.task_graph.add_task_node(goal)
        
        # Add dependencies
        self._build_dependencies(all_goals)
        
        # Optimize execution order
        execution_order = self.task_graph.optimize_execution_order()
        
        # Analyze coordination requirements
        coordination_reqs = self._analyze_coordination_requirements(all_goals)
        
        # Estimate completion time
        completion_time = self._estimate_completion_time(all_goals)
        
        # Complexity analysis
        complexity_analysis = self._analyze_complexity(all_goals)
        
        result = TaskDecompositionResult(
            goal_hierarchy={goal.goal_id: goal for goal in all_goals},
            execution_order=execution_order,
            dependency_graph=self.task_graph.graph,
            coordination_requirements=coordination_reqs,
            estimated_completion_time=completion_time,
            complexity_analysis=complexity_analysis
        )
        
        # Update statistics
        self._update_decomposition_stats(result)
        
        return result
    
    def _recursive_decompose(self,
                           goal_description: str,
                           context: Dict[str, Any],
                           depth: int,
                           parent_id: Optional[str] = None) -> List[SubGoal]:
        """Recursively decompose goals."""
        if depth >= self.max_depth:
            # Create primitive goal at max depth
            primitive_goal = SubGoal(
                goal_id=f"primitive_{goal_description}_{depth}",
                description=goal_description,
                task_type=TaskType.PRIMITIVE,
                complexity_score=min(1.0, depth / self.max_depth),
                parent_goals=[parent_id] if parent_id else []
            )
            return [primitive_goal]
        
        all_sub_goals = []
        
        # Use symbolic reasoning if available
        if self.symbolic_reasoner:
            symbolic_goals = self.symbolic_reasoner.decompose_goal(goal_description, context, depth)
            all_sub_goals.extend(symbolic_goals)
        
        # Use neural decomposition as well
        neural_goals = self._neural_decompose(goal_description, context, depth)
        all_sub_goals.extend(neural_goals)
        
        # Remove duplicates and refine
        refined_goals = self._merge_and_refine_goals(all_sub_goals, parent_id)
        
        # Recursively decompose composite goals
        final_goals = []
        for goal in refined_goals:
            final_goals.append(goal)
            
            if goal.task_type == TaskType.COMPOSITE and depth < self.max_depth - 1:
                child_goals = self._recursive_decompose(
                    goal.description, context, depth + 1, goal.goal_id
                )
                
                # Update parent-child relationships
                goal.child_goals = [child.goal_id for child in child_goals]
                for child in child_goals:
                    child.parent_goals.append(goal.goal_id)
                
                final_goals.extend(child_goals)
        
        return final_goals
    
    def _neural_decompose(self,
                         goal_description: str,
                         context: Dict[str, Any],
                         depth: int) -> List[SubGoal]:
        """Use neural network for goal decomposition."""
        # Tokenize goal description
        tokens = self._tokenize(goal_description)
        token_ids = torch.tensor([self.token_to_id.get(token, self.token_to_id['<unk>']) for token in tokens])
        
        # Prepare features
        features = self._extract_features(context)
        
        # Neural prediction
        with torch.no_grad():
            task_embedding, decomposition_logits = self.task_encoder(
                token_ids.unsqueeze(0),  # Add batch dimension
                features.unsqueeze(0)
            )
        
        # Decode top predictions into sub-goals
        top_k = min(5, decomposition_logits.size(1))
        top_predictions = torch.topk(decomposition_logits[0], top_k)
        
        sub_goals = []
        for i, (score, token_id) in enumerate(zip(top_predictions.values, top_predictions.indices)):
            if score > 0.1:  # Threshold for meaningful predictions
                token = self.id_to_token.get(token_id.item(), '<unk>')
                if token not in ['<start>', '<end>', '<unk>', '<pad>']:
                    sub_goal = SubGoal(
                        goal_id=f"neural_{token}_{goal_description}_{i}",
                        description=f"{token} for {goal_description}",
                        task_type=TaskType.COMPOSITE if depth < self.max_depth - 2 else TaskType.PRIMITIVE,
                        complexity_score=min(1.0, (depth + 1) / self.max_depth),
                        priority=score.item(),
                        required_skills=[token]
                    )
                    sub_goals.append(sub_goal)
        
        return sub_goals
    
    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization of text."""
        # Basic tokenization (in practice would use sophisticated NLP)
        tokens = text.lower().replace(',', '').replace('.', '').split()
        return tokens
    
    def _extract_features(self, context: Dict[str, Any]) -> torch.Tensor:
        """Extract numerical features from context."""
        features = [
            len(context.get('capabilities', [])) / 20.0,  # Normalize capability count
            len(context.get('resource_constraints', {})) / 10.0,  # Resource constraints
            context.get('time_limit', 300.0) / 300.0,  # Normalize time limit
            context.get('complexity_preference', 0.5),  # User complexity preference
            context.get('multi_agent', False),  # Boolean to float
            context.get('real_time', False),  # Boolean to float
            len(context.get('objects', [])) / 50.0,  # Object count
            context.get('safety_critical', False),  # Boolean to float
            context.get('precision_required', 0.5),  # Precision requirement
            context.get('force_limits', 1.0)  # Force limits
        ]
        
        return torch.tensor(features, dtype=torch.float32)
    
    def _merge_and_refine_goals(self, 
                               goals: List[SubGoal], 
                               parent_id: Optional[str]) -> List[SubGoal]:
        """Merge similar goals and refine the goal set."""
        if not goals:
            return []
        
        # Remove duplicates based on similarity
        unique_goals = []
        for goal in goals:
            is_duplicate = False
            for existing in unique_goals:
                if self._goals_similar(goal, existing):
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                if parent_id:
                    goal.parent_goals = [parent_id]
                unique_goals.append(goal)
        
        # Sort by priority
        unique_goals.sort(key=lambda g: g.priority, reverse=True)
        
        return unique_goals
    
    def _goals_similar(self, goal1: SubGoal, goal2: SubGoal) -> bool:
        """Check if two goals are similar (simple string similarity)."""
        desc1_words = set(goal1.description.lower().split())
        desc2_words = set(goal2.description.lower().split())
        
        if not desc1_words or not desc2_words:
            return False
        
        intersection = desc1_words.intersection(desc2_words)
        union = desc1_words.union(desc2_words)
        
        similarity = len(intersection) / len(union)
        return similarity > 0.7
    
    def _build_dependencies(self, goals: List[SubGoal]):
        """Build dependency edges in the task graph."""
        for goal in goals:
            # Add parent-child dependencies
            for child_id in goal.child_goals:
                self.task_graph.add_dependency(goal.goal_id, child_id, "hierarchical")
            
            # Add sequential dependencies based on preconditions
            for other_goal in goals:
                if goal.goal_id != other_goal.goal_id:
                    # Check if other_goal's effects satisfy goal's preconditions
                    for precondition in goal.preconditions:
                        if any(effect in precondition for effect in other_goal.effects):
                            self.task_graph.add_dependency(other_goal.goal_id, goal.goal_id, "causal")
    
    def _analyze_coordination_requirements(self, goals: List[SubGoal]) -> Dict[str, List[str]]:
        """Analyze multi-agent coordination requirements."""
        coordination_reqs = {}
        
        for goal in goals:
            if goal.task_type == TaskType.COORDINATION:
                # Find goals that need coordination
                coordinated_goals = []
                for other_goal in goals:
                    if (other_goal.goal_id != goal.goal_id and 
                        any(req in other_goal.required_skills for req in goal.required_skills)):
                        coordinated_goals.append(other_goal.goal_id)
                
                if coordinated_goals:
                    coordination_reqs[goal.goal_id] = coordinated_goals
        
        return coordination_reqs
    
    def _estimate_completion_time(self, goals: List[SubGoal]) -> float:
        """Estimate total completion time considering dependencies."""
        # Use critical path method
        critical_path = self.task_graph.get_critical_path()
        
        total_time = 0.0
        for goal_id in critical_path:
            goal = next((g for g in goals if g.goal_id == goal_id), None)
            if goal:
                total_time += goal.estimated_duration
        
        return total_time
    
    def _analyze_complexity(self, goals: List[SubGoal]) -> Dict[str, float]:
        """Analyze complexity of the decomposed task."""
        if not goals:
            return {'total_complexity': 0.0, 'avg_complexity': 0.0, 'max_complexity': 0.0}
        
        complexities = [goal.complexity_score for goal in goals]
        
        return {
            'total_complexity': sum(complexities),
            'avg_complexity': np.mean(complexities),
            'max_complexity': max(complexities),
            'complexity_variance': np.var(complexities),
            'num_goals': len(goals),
            'primitive_ratio': len([g for g in goals if g.task_type == TaskType.PRIMITIVE]) / len(goals)
        }
    
    def _update_decomposition_stats(self, result: TaskDecompositionResult):
        """Update decomposition statistics."""
        self.decomposition_stats['total_decompositions'] += 1
        
        # Calculate depth
        max_depth = 0
        for goal in result.goal_hierarchy.values():
            depth = len(goal.parent_goals)
            max_depth = max(max_depth, depth)
        
        # Update running averages
        n = self.decomposition_stats['total_decompositions']
        self.decomposition_stats['avg_depth'] = (
            (self.decomposition_stats['avg_depth'] * (n - 1) + max_depth) / n
        )
        
        # Calculate branching factor
        total_children = sum(len(goal.child_goals) for goal in result.goal_hierarchy.values())
        non_leaf_goals = len([goal for goal in result.goal_hierarchy.values() if goal.child_goals])
        avg_branching = total_children / max(non_leaf_goals, 1)
        
        self.decomposition_stats['avg_branching_factor'] = (
            (self.decomposition_stats['avg_branching_factor'] * (n - 1) + avg_branching) / n
        )
    
    def execute_hierarchical_plan(self,
                                 decomposition_result: TaskDecompositionResult,
                                 execution_callback: callable,
                                 monitoring_interval: float = 1.0) -> Dict[str, Any]:
        """Execute hierarchical plan with monitoring and adaptation."""
        execution_log = []
        start_time = time.time()
        
        for task_id in decomposition_result.execution_order:
            goal = decomposition_result.goal_hierarchy[task_id]
            
            logger.info(f"Executing task: {goal.description}")
            
            # Update task status to active
            self.task_graph.update_task_status(task_id, GoalStatus.ACTIVE, time.time())
            
            # Execute task using callback
            try:
                task_start = time.time()
                execution_result = execution_callback(goal)
                task_end = time.time()
                
                # Update task status based on result
                if execution_result.get('success', False):
                    self.task_graph.update_task_status(task_id, GoalStatus.COMPLETED, task_end)
                    status = GoalStatus.COMPLETED
                else:
                    self.task_graph.update_task_status(task_id, GoalStatus.FAILED, task_end)
                    status = GoalStatus.FAILED
                
                execution_log.append({
                    'task_id': task_id,
                    'goal_description': goal.description,
                    'status': status,
                    'execution_time': task_end - task_start,
                    'result': execution_result
                })
                
                # Early termination on critical failure
                if status == GoalStatus.FAILED and goal.task_type == TaskType.PRIMITIVE:
                    logger.warning(f"Critical task failed: {goal.description}")
                    break
                    
            except Exception as e:
                logger.error(f"Task execution failed: {e}")
                self.task_graph.update_task_status(task_id, GoalStatus.FAILED, time.time())
                execution_log.append({
                    'task_id': task_id,
                    'goal_description': goal.description,
                    'status': GoalStatus.FAILED,
                    'error': str(e)
                })
        
        total_time = time.time() - start_time
        
        # Calculate success metrics
        completed_tasks = len([log for log in execution_log if log.get('status') == GoalStatus.COMPLETED])
        total_tasks = len(execution_log)
        success_rate = completed_tasks / max(total_tasks, 1)
        
        return {
            'execution_log': execution_log,
            'total_execution_time': total_time,
            'success_rate': success_rate,
            'completed_tasks': completed_tasks,
            'total_tasks': total_tasks,
            'critical_path_time': decomposition_result.estimated_completion_time,
            'efficiency': decomposition_result.estimated_completion_time / max(total_time, 1e-6)
        }
    
    def get_decomposition_metrics(self) -> Dict[str, Any]:
        """Get comprehensive decomposition performance metrics."""
        return {
            'decomposition_statistics': self.decomposition_stats,
            'graph_metrics': {
                'num_nodes': self.task_graph.graph.number_of_nodes(),
                'num_edges': self.task_graph.graph.number_of_edges(),
                'is_dag': nx.is_directed_acyclic_graph(self.task_graph.graph),
                'density': nx.density(self.task_graph.graph)
            },
            'conflicts': self.task_graph.detect_conflicts(),
            'critical_path_length': len(self.task_graph.get_critical_path())
        }
    
    def save_decomposer(self, filepath: str):
        """Save task decomposer state."""
        state = {
            'task_encoder': self.task_encoder.state_dict(),
            'vocab': self.vocab,
            'token_to_id': self.token_to_id,
            'decomposition_stats': self.decomposition_stats,
            'config': {
                'vocab_size': self.vocab_size,
                'embedding_dim': self.embedding_dim,
                'max_depth': self.max_depth,
                'use_symbolic_reasoning': self.use_symbolic_reasoning
            }
        }
        
        if self.symbolic_reasoner:
            state['symbolic_reasoner'] = {
                'rule_base': self.symbolic_reasoner.rule_base,
                'knowledge_base': self.symbolic_reasoner.knowledge_base
            }
        
        torch.save(state, filepath)
        logger.info(f"Saved task decomposer to {filepath}")
    
    def load_decomposer(self, filepath: str):
        """Load task decomposer state."""
        state = torch.load(filepath)
        
        self.task_encoder.load_state_dict(state['task_encoder'])
        self.vocab = state['vocab']
        self.token_to_id = state['token_to_id']
        self.id_to_token = {i: token for token, i in self.token_to_id.items()}
        self.decomposition_stats = state['decomposition_stats']
        
        if 'symbolic_reasoner' in state and self.symbolic_reasoner:
            self.symbolic_reasoner.rule_base = state['symbolic_reasoner']['rule_base']
            self.symbolic_reasoner.knowledge_base = state['symbolic_reasoner']['knowledge_base']
        
        logger.info(f"Loaded task decomposer from {filepath}")