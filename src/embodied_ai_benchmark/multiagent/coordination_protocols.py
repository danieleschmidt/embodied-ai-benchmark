"""Advanced multi-agent coordination protocols and communication systems."""

import time
import numpy as np
from typing import Any, Dict, List, Optional, Tuple, Set, Callable
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque
from datetime import datetime
import threading
import queue

from ..core.base_agent import BaseAgent
from ..utils.logging_config import get_logger

logger = get_logger(__name__)


class MessageType(Enum):
    """Types of messages in multi-agent communication."""
    REQUEST = "request"
    INFORM = "inform"
    CONFIRM = "confirm"
    COORDINATE = "coordinate"
    NEGOTIATE = "negotiate"
    EMERGENCY = "emergency"


@dataclass
class Message:
    """Communication message between agents."""
    sender: str
    recipient: str
    message_type: MessageType
    content: Dict[str, Any]
    timestamp: float = field(default_factory=time.time)
    priority: int = 1  # Higher values = higher priority
    ttl: float = 10.0  # Time to live in seconds
    requires_ack: bool = False
    message_id: str = field(default_factory=lambda: f"msg_{int(time.time() * 1000000)}")


@dataclass
class CoordinationTask:
    """Multi-agent coordination task specification."""
    task_id: str
    task_type: str
    required_agents: List[str]
    coordination_pattern: str  # 'sequential', 'parallel', 'hierarchical'
    dependencies: Dict[str, List[str]]
    timing_constraints: Dict[str, float]
    success_criteria: Dict[str, Any]
    created_at: float = field(default_factory=time.time)


class CommunicationProtocol:
    """Advanced communication protocol for multi-agent systems."""
    
    def __init__(self,
                 bandwidth_limit: int = 100,  # messages per second
                 latency: float = 0.1,  # seconds
                 packet_loss: float = 0.01,  # percentage
                 max_message_size: int = 1024):  # bytes
        """Initialize communication protocol.
        
        Args:
            bandwidth_limit: Maximum messages per second
            latency: Network latency in seconds
            packet_loss: Packet loss rate (0-1)
            max_message_size: Maximum message size in bytes
        """
        self.bandwidth_limit = bandwidth_limit
        self.latency = latency
        self.packet_loss = packet_loss
        self.max_message_size = max_message_size
        
        # Message queues and tracking
        self.message_queues = defaultdict(queue.PriorityQueue)
        self.sent_messages = defaultdict(list)
        self.received_messages = defaultdict(list)
        self.acknowledgments = {}
        
        # Protocol statistics
        self.stats = {
            "messages_sent": 0,
            "messages_received": 0,
            "messages_lost": 0,
            "bandwidth_usage": 0.0,
            "average_latency": 0.0
        }
        
        # Message handlers
        self.message_handlers = {}
        
        # Thread-safe lock
        self._lock = threading.Lock()
        
    def register_handler(self, message_type: MessageType, handler: Callable):
        """Register message handler for specific message type.
        
        Args:
            message_type: Type of message to handle
            handler: Handler function
        """
        self.message_handlers[message_type] = handler
        logger.info(f"Registered handler for {message_type}")
    
    def send_message(self, message: Message) -> bool:
        """Send message with protocol constraints.
        
        Args:
            message: Message to send
            
        Returns:
            True if message was queued successfully
        """
        with self._lock:
            # Check bandwidth limits
            current_time = time.time()
            recent_messages = [
                msg for msg in self.sent_messages[message.sender]
                if current_time - msg.timestamp < 1.0
            ]
            
            if len(recent_messages) >= self.bandwidth_limit:
                logger.warning(f"Bandwidth limit exceeded for {message.sender}")
                return False
            
            # Simulate packet loss
            if np.random.random() < self.packet_loss:
                self.stats["messages_lost"] += 1
                logger.debug(f"Message lost: {message.message_id}")
                return False
            
            # Add latency
            delivery_time = current_time + self.latency
            
            # Queue message for delivery
            priority = -message.priority  # Negative for max-heap behavior
            self.message_queues[message.recipient].put(
                (priority, delivery_time, message)
            )
            
            # Track sent message
            self.sent_messages[message.sender].append(message)
            self.stats["messages_sent"] += 1
            
            logger.debug(f"Message queued: {message.sender} -> {message.recipient}")
            return True
    
    def receive_messages(self, agent_id: str) -> List[Message]:
        """Receive queued messages for agent.
        
        Args:
            agent_id: Agent identifier
            
        Returns:
            List of messages ready for delivery
        """
        messages = []
        current_time = time.time()
        
        with self._lock:
            message_queue = self.message_queues[agent_id]
            
            # Process messages ready for delivery
            ready_messages = []
            while not message_queue.empty():
                priority, delivery_time, message = message_queue.get()
                
                # Check if message has expired
                if current_time - message.timestamp > message.ttl:
                    logger.debug(f"Message expired: {message.message_id}")
                    continue
                
                # Check if message is ready for delivery
                if delivery_time <= current_time:
                    messages.append(message)
                    self.received_messages[agent_id].append(message)
                    self.stats["messages_received"] += 1
                    
                    # Send acknowledgment if required
                    if message.requires_ack:
                        self._send_acknowledgment(message)
                else:
                    # Put back in queue if not ready
                    ready_messages.append((priority, delivery_time, message))
            
            # Put back undelivered messages
            for item in ready_messages:
                message_queue.put(item)
        
        return messages
    
    def _send_acknowledgment(self, original_message: Message):
        """Send acknowledgment for message.
        
        Args:
            original_message: Message to acknowledge
        """
        ack_message = Message(
            sender=original_message.recipient,
            recipient=original_message.sender,
            message_type=MessageType.CONFIRM,
            content={"ack_for": original_message.message_id},
            priority=5  # High priority for acks
        )
        self.send_message(ack_message)
    
    def handle_message(self, message: Message, agent: BaseAgent) -> Optional[Dict[str, Any]]:
        """Handle incoming message with registered handlers.
        
        Args:
            message: Message to handle
            agent: Agent receiving the message
            
        Returns:
            Handler response if any
        """
        handler = self.message_handlers.get(message.message_type)
        if handler:
            try:
                return handler(message, agent)
            except Exception as e:
                logger.error(f"Message handler error: {e}")
                return None
        else:
            logger.warning(f"No handler for message type: {message.message_type}")
            return None
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get communication protocol statistics.
        
        Returns:
            Statistics dictionary
        """
        with self._lock:
            return self.stats.copy()


class DynamicRoleAssignment:
    """Dynamic role assignment system for multi-agent coordination."""
    
    def __init__(self, roles: List[str], assignment_method: str = "capability_based"):
        """Initialize role assignment system.
        
        Args:
            roles: Available roles
            assignment_method: Method for role assignment
        """
        self.roles = roles
        self.assignment_method = assignment_method
        self.current_assignments = {}
        self.role_history = []
        self.assignment_callbacks = []
        
        # Role requirements and preferences
        self.role_requirements = {}
        self.agent_capabilities = {}
        self.role_performance = defaultdict(dict)
        
    def register_assignment_callback(self, callback: Callable):
        """Register callback for role assignment changes.
        
        Args:
            callback: Callback function
        """
        self.assignment_callbacks.append(callback)
    
    def update_agent_capabilities(self, agent_id: str, capabilities: Dict[str, float]):
        """Update agent capability scores.
        
        Args:
            agent_id: Agent identifier
            capabilities: Dictionary of capability name to score (0-1)
        """
        self.agent_capabilities[agent_id] = capabilities.copy()
        logger.debug(f"Updated capabilities for {agent_id}: {capabilities}")
    
    def set_role_requirements(self, role: str, requirements: Dict[str, float]):
        """Set requirements for a specific role.
        
        Args:
            role: Role identifier
            requirements: Dictionary of required capabilities and minimum scores
        """
        self.role_requirements[role] = requirements.copy()
        logger.info(f"Set requirements for role {role}: {requirements}")
    
    def assign_roles(self, 
                    agents: List[str], 
                    task_context: Optional[Dict[str, Any]] = None) -> Dict[str, str]:
        """Assign roles to agents based on current method.
        
        Args:
            agents: List of agent identifiers
            task_context: Optional task context for assignment
            
        Returns:
            Dictionary mapping agent_id to assigned role
        """
        if self.assignment_method == "capability_based":
            assignments = self._assign_capability_based(agents, task_context)
        elif self.assignment_method == "performance_based":
            assignments = self._assign_performance_based(agents, task_context)
        elif self.assignment_method == "round_robin":
            assignments = self._assign_round_robin(agents)
        else:
            assignments = self._assign_random(agents)
        
        # Update current assignments
        self.current_assignments.update(assignments)
        
        # Record assignment in history
        self.role_history.append({
            "timestamp": datetime.now().isoformat(),
            "assignments": assignments.copy(),
            "method": self.assignment_method,
            "context": task_context
        })
        
        # Notify callbacks
        for callback in self.assignment_callbacks:
            try:
                callback(assignments, task_context)
            except Exception as e:
                logger.error(f"Role assignment callback error: {e}")
        
        logger.info(f"Assigned roles: {assignments}")
        return assignments
    
    def _assign_capability_based(self, 
                                agents: List[str], 
                                task_context: Optional[Dict[str, Any]]) -> Dict[str, str]:
        """Assign roles based on agent capabilities.
        
        Args:
            agents: List of agents
            task_context: Task context
            
        Returns:
            Role assignments
        """
        assignments = {}
        available_roles = self.roles.copy()
        
        # Calculate capability matches for each agent-role pair
        matches = []
        for agent in agents:
            agent_caps = self.agent_capabilities.get(agent, {})
            
            for role in available_roles:
                role_reqs = self.role_requirements.get(role, {})
                
                # Calculate match score
                match_score = self._calculate_match_score(agent_caps, role_reqs)
                matches.append((match_score, agent, role))
        
        # Sort by match score (descending)
        matches.sort(reverse=True)
        
        # Assign roles greedily
        assigned_agents = set()
        assigned_roles = set()
        
        for score, agent, role in matches:
            if agent not in assigned_agents and role not in assigned_roles:
                assignments[agent] = role
                assigned_agents.add(agent)
                assigned_roles.add(role)
        
        # Assign remaining agents to remaining roles
        unassigned_agents = [a for a in agents if a not in assigned_agents]
        unassigned_roles = [r for r in available_roles if r not in assigned_roles]
        
        for i, agent in enumerate(unassigned_agents):
            if i < len(unassigned_roles):
                assignments[agent] = unassigned_roles[i]
            else:
                # Default role if more agents than roles
                assignments[agent] = "general"
        
        return assignments
    
    def _calculate_match_score(self, 
                             agent_capabilities: Dict[str, float], 
                             role_requirements: Dict[str, float]) -> float:
        """Calculate capability match score between agent and role.
        
        Args:
            agent_capabilities: Agent's capability scores
            role_requirements: Role's requirement scores
            
        Returns:
            Match score (higher is better)
        """
        if not role_requirements:
            return 0.5  # Neutral score for roles without requirements
        
        total_score = 0.0
        total_weight = 0.0
        
        for capability, required_score in role_requirements.items():
            agent_score = agent_capabilities.get(capability, 0.0)
            
            # Score is how well agent meets requirement
            if agent_score >= required_score:
                score = 1.0 + (agent_score - required_score)  # Bonus for exceeding
            else:
                score = agent_score / required_score if required_score > 0 else 0.0
            
            total_score += score * required_score  # Weight by importance
            total_weight += required_score
        
        return total_score / total_weight if total_weight > 0 else 0.0
    
    def _assign_performance_based(self, 
                                agents: List[str], 
                                task_context: Optional[Dict[str, Any]]) -> Dict[str, str]:
        """Assign roles based on historical performance.
        
        Args:
            agents: List of agents
            task_context: Task context
            
        Returns:
            Role assignments
        """
        assignments = {}
        available_roles = self.roles.copy()
        
        # Calculate performance scores for each agent-role pair
        performance_matches = []
        for agent in agents:
            for role in available_roles:
                perf_score = self.role_performance[agent].get(role, 0.5)
                performance_matches.append((perf_score, agent, role))
        
        # Sort by performance (descending)
        performance_matches.sort(reverse=True)
        
        # Assign roles greedily
        assigned_agents = set()
        assigned_roles = set()
        
        for score, agent, role in performance_matches:
            if agent not in assigned_agents and role not in assigned_roles:
                assignments[agent] = role
                assigned_agents.add(agent)
                assigned_roles.add(role)
        
        return assignments
    
    def _assign_round_robin(self, agents: List[str]) -> Dict[str, str]:
        """Assign roles in round-robin fashion.
        
        Args:
            agents: List of agents
            
        Returns:
            Role assignments
        """
        assignments = {}
        for i, agent in enumerate(agents):
            role_idx = i % len(self.roles)
            assignments[agent] = self.roles[role_idx]
        
        return assignments
    
    def _assign_random(self, agents: List[str]) -> Dict[str, str]:
        """Assign roles randomly.
        
        Args:
            agents: List of agents
            
        Returns:
            Role assignments
        """
        assignments = {}
        for agent in agents:
            role = np.random.choice(self.roles)
            assignments[agent] = role
        
        return assignments
    
    def update_performance(self, agent_id: str, role: str, performance_score: float):
        """Update performance score for agent in specific role.
        
        Args:
            agent_id: Agent identifier
            role: Role identifier
            performance_score: Performance score (0-1)
        """
        self.role_performance[agent_id][role] = performance_score
        logger.debug(f"Updated performance for {agent_id} in {role}: {performance_score}")
    
    def get_role_assignment_history(self) -> List[Dict[str, Any]]:
        """Get history of role assignments.
        
        Returns:
            List of assignment history entries
        """
        return self.role_history.copy()


class CoordinationOrchestrator:
    """Orchestrates multi-agent coordination tasks."""
    
    def __init__(self, 
                 communication_protocol: CommunicationProtocol,
                 role_assignment: DynamicRoleAssignment):
        """Initialize coordination orchestrator.
        
        Args:
            communication_protocol: Communication protocol instance
            role_assignment: Role assignment system
        """
        self.comm_protocol = communication_protocol
        self.role_assignment = role_assignment
        
        # Task management
        self.active_tasks = {}
        self.task_history = []
        self.coordination_patterns = {
            "sequential": self._coordinate_sequential,
            "parallel": self._coordinate_parallel,
            "hierarchical": self._coordinate_hierarchical
        }
        
        # Synchronization primitives
        self.synchronization_points = {}
        self.barrier_states = {}
        
    def create_coordination_task(self, task_spec: CoordinationTask) -> bool:
        """Create new coordination task.
        
        Args:
            task_spec: Task specification
            
        Returns:
            True if task was created successfully
        """
        if task_spec.task_id in self.active_tasks:
            logger.warning(f"Task {task_spec.task_id} already exists")
            return False
        
        # Validate task specification
        if not self._validate_task_spec(task_spec):
            logger.error(f"Invalid task specification: {task_spec.task_id}")
            return False
        
        # Assign roles for this task
        role_assignments = self.role_assignment.assign_roles(
            task_spec.required_agents,
            {"task_type": task_spec.task_type, "task_id": task_spec.task_id}
        )
        
        # Store task with assignments
        self.active_tasks[task_spec.task_id] = {
            "spec": task_spec,
            "role_assignments": role_assignments,
            "status": "created",
            "start_time": None,
            "coordination_state": {},
            "agent_states": {agent: "ready" for agent in task_spec.required_agents}
        }
        
        logger.info(f"Created coordination task: {task_spec.task_id}")
        return True
    
    def start_coordination_task(self, task_id: str) -> bool:
        """Start executing coordination task.
        
        Args:
            task_id: Task identifier
            
        Returns:
            True if task started successfully
        """
        if task_id not in self.active_tasks:
            logger.error(f"Task {task_id} not found")
            return False
        
        task_info = self.active_tasks[task_id]
        task_spec = task_info["spec"]
        
        # Check if all required agents are ready
        if not all(state == "ready" for state in task_info["agent_states"].values()):
            logger.warning(f"Not all agents ready for task {task_id}")
            return False
        
        # Update task status
        task_info["status"] = "running"
        task_info["start_time"] = time.time()
        
        # Start coordination based on pattern
        coordination_pattern = task_spec.coordination_pattern
        if coordination_pattern in self.coordination_patterns:
            self.coordination_patterns[coordination_pattern](task_id)
            logger.info(f"Started coordination task {task_id} with pattern {coordination_pattern}")
            return True
        else:
            logger.error(f"Unknown coordination pattern: {coordination_pattern}")
            return False
    
    def _validate_task_spec(self, task_spec: CoordinationTask) -> bool:
        """Validate task specification.
        
        Args:
            task_spec: Task specification to validate
            
        Returns:
            True if valid
        """
        # Check required fields
        if not task_spec.task_id or not task_spec.required_agents:
            return False
        
        # Check dependencies are valid
        for agent, deps in task_spec.dependencies.items():
            if agent not in task_spec.required_agents:
                return False
            for dep in deps:
                if dep not in task_spec.required_agents:
                    return False
        
        return True
    
    def _coordinate_sequential(self, task_id: str):
        """Coordinate task with sequential execution pattern.
        
        Args:
            task_id: Task identifier
        """
        task_info = self.active_tasks[task_id]
        task_spec = task_info["spec"]
        
        # Create execution order based on dependencies
        execution_order = self._topological_sort(
            task_spec.required_agents,
            task_spec.dependencies
        )
        
        # Send coordination messages
        for i, agent in enumerate(execution_order):
            message = Message(
                sender="orchestrator",
                recipient=agent,
                message_type=MessageType.COORDINATE,
                content={
                    "task_id": task_id,
                    "coordination_pattern": "sequential",
                    "execution_order": i,
                    "total_agents": len(execution_order),
                    "role": task_info["role_assignments"].get(agent, "general"),
                    "dependencies": task_spec.dependencies.get(agent, [])
                },
                priority=3,
                requires_ack=True
            )
            
            self.comm_protocol.send_message(message)
        
        logger.info(f"Initiated sequential coordination for task {task_id}")
    
    def _coordinate_parallel(self, task_id: str):
        """Coordinate task with parallel execution pattern.
        
        Args:
            task_id: Task identifier
        """
        task_info = self.active_tasks[task_id]
        task_spec = task_info["spec"]
        
        # Create synchronization barrier
        barrier_id = f"{task_id}_barrier"
        self.barrier_states[barrier_id] = {
            "required_agents": set(task_spec.required_agents),
            "ready_agents": set(),
            "completed": False
        }
        
        # Send coordination messages to all agents
        for agent in task_spec.required_agents:
            message = Message(
                sender="orchestrator",
                recipient=agent,
                message_type=MessageType.COORDINATE,
                content={
                    "task_id": task_id,
                    "coordination_pattern": "parallel",
                    "barrier_id": barrier_id,
                    "role": task_info["role_assignments"].get(agent, "general"),
                    "sync_required": True
                },
                priority=3,
                requires_ack=True
            )
            
            self.comm_protocol.send_message(message)
        
        logger.info(f"Initiated parallel coordination for task {task_id}")
    
    def _coordinate_hierarchical(self, task_id: str):
        """Coordinate task with hierarchical execution pattern.
        
        Args:
            task_id: Task identifier
        """
        task_info = self.active_tasks[task_id]
        task_spec = task_info["spec"]
        role_assignments = task_info["role_assignments"]
        
        # Find leader (highest priority role or first agent)
        leader = None
        for agent, role in role_assignments.items():
            if role == "leader":
                leader = agent
                break
        
        if leader is None:
            leader = task_spec.required_agents[0]  # Default to first agent
            role_assignments[leader] = "leader"
        
        # Send leader coordination message
        leader_message = Message(
            sender="orchestrator",
            recipient=leader,
            message_type=MessageType.COORDINATE,
            content={
                "task_id": task_id,
                "coordination_pattern": "hierarchical",
                "role": "leader",
                "subordinates": [a for a in task_spec.required_agents if a != leader],
                "task_spec": task_spec.__dict__
            },
            priority=4,
            requires_ack=True
        )
        
        self.comm_protocol.send_message(leader_message)
        
        # Send subordinate coordination messages
        for agent in task_spec.required_agents:
            if agent != leader:
                subordinate_message = Message(
                    sender="orchestrator",
                    recipient=agent,
                    message_type=MessageType.COORDINATE,
                    content={
                        "task_id": task_id,
                        "coordination_pattern": "hierarchical",
                        "role": role_assignments.get(agent, "subordinate"),
                        "leader": leader
                    },
                    priority=3,
                    requires_ack=True
                )
                
                self.comm_protocol.send_message(subordinate_message)
        
        logger.info(f"Initiated hierarchical coordination for task {task_id} with leader {leader}")
    
    def _topological_sort(self, agents: List[str], dependencies: Dict[str, List[str]]) -> List[str]:
        """Perform topological sort on agents based on dependencies.
        
        Args:
            agents: List of agents
            dependencies: Agent dependencies
            
        Returns:
            Sorted list of agents
        """
        # Build dependency graph
        in_degree = {agent: 0 for agent in agents}
        graph = defaultdict(list)
        
        for agent, deps in dependencies.items():
            for dep in deps:
                graph[dep].append(agent)
                in_degree[agent] += 1
        
        # Topological sort using Kahn's algorithm
        queue = deque([agent for agent in agents if in_degree[agent] == 0])
        result = []
        
        while queue:
            agent = queue.popleft()
            result.append(agent)
            
            for neighbor in graph[agent]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)
        
        # Check for cycles
        if len(result) != len(agents):
            logger.warning("Circular dependencies detected, using original order")
            return agents
        
        return result
    
    def handle_coordination_message(self, message: Message) -> Dict[str, Any]:
        """Handle coordination-related messages.
        
        Args:
            message: Coordination message
            
        Returns:
            Response dictionary
        """
        content = message.content
        
        if "barrier_id" in content:
            # Handle barrier synchronization
            return self._handle_barrier_sync(message)
        elif "task_complete" in content:
            # Handle task completion notification
            return self._handle_task_completion(message)
        else:
            # Generic coordination response
            return {"status": "acknowledged", "message_id": message.message_id}
    
    def _handle_barrier_sync(self, message: Message) -> Dict[str, Any]:
        """Handle barrier synchronization message.
        
        Args:
            message: Barrier sync message
            
        Returns:
            Response dictionary
        """
        barrier_id = message.content["barrier_id"]
        agent_id = message.sender
        
        if barrier_id not in self.barrier_states:
            return {"status": "error", "message": "Unknown barrier"}
        
        barrier = self.barrier_states[barrier_id]
        barrier["ready_agents"].add(agent_id)
        
        # Check if all agents are ready
        if barrier["ready_agents"] == barrier["required_agents"]:
            barrier["completed"] = True
            
            # Notify all agents that barrier is complete
            for agent in barrier["required_agents"]:
                notification = Message(
                    sender="orchestrator",
                    recipient=agent,
                    message_type=MessageType.INFORM,
                    content={
                        "barrier_id": barrier_id,
                        "status": "complete",
                        "all_agents_ready": True
                    },
                    priority=4
                )
                self.comm_protocol.send_message(notification)
            
            logger.info(f"Barrier {barrier_id} synchronized")
        
        return {
            "status": "acknowledged",
            "barrier_status": "complete" if barrier["completed"] else "waiting"
        }
    
    def _handle_task_completion(self, message: Message) -> Dict[str, Any]:
        """Handle task completion notification.
        
        Args:
            message: Task completion message
            
        Returns:
            Response dictionary
        """
        task_id = message.content["task_id"]
        agent_id = message.sender
        
        if task_id not in self.active_tasks:
            return {"status": "error", "message": "Unknown task"}
        
        task_info = self.active_tasks[task_id]
        task_info["agent_states"][agent_id] = "completed"
        
        # Check if all agents completed
        if all(state == "completed" for state in task_info["agent_states"].values()):
            task_info["status"] = "completed"
            task_info["end_time"] = time.time()
            
            # Move to history
            self.task_history.append(task_info)
            del self.active_tasks[task_id]
            
            logger.info(f"Task {task_id} completed successfully")
            
            return {"status": "task_complete", "all_agents_finished": True}
        
        return {"status": "acknowledged", "task_status": "in_progress"}
    
    def get_coordination_statistics(self) -> Dict[str, Any]:
        """Get coordination system statistics.
        
        Returns:
            Statistics dictionary
        """
        return {
            "active_tasks": len(self.active_tasks),
            "completed_tasks": len(self.task_history),
            "active_barriers": len(self.barrier_states),
            "communication_stats": self.comm_protocol.get_statistics(),
            "role_assignment_history": len(self.role_assignment.get_role_assignment_history())
        }