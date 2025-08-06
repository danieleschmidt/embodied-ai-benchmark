"""Tests for multi-agent coordination protocols."""

import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock
import time
from collections import defaultdict

from embodied_ai_benchmark.multiagent.coordination_protocols import (
    CommunicationProtocol,
    DynamicRoleAssignment, 
    CoordinationOrchestrator,
    Message,
    MessageType,
    CoordinationTask
)


class TestMessage:
    """Test message system functionality."""
    
    def test_message_creation(self):
        """Test message creation and properties."""
        message = Message(
            sender_id="agent_1",
            receiver_id="agent_2", 
            message_type=MessageType.TASK_ASSIGNMENT,
            content={"task": "navigate_to_goal"},
            priority=2
        )
        
        assert message.sender_id == "agent_1"
        assert message.receiver_id == "agent_2"
        assert message.message_type == MessageType.TASK_ASSIGNMENT
        assert message.content["task"] == "navigate_to_goal"
        assert message.priority == 2
        assert message.timestamp > 0
        assert isinstance(message.message_id, str)
        assert len(message.message_id) > 0
    
    def test_message_serialization(self):
        """Test message to_dict and from_dict methods."""
        original = Message(
            sender_id="test_sender",
            receiver_id="test_receiver",
            message_type=MessageType.STATUS_UPDATE,
            content={"status": "completed"},
            priority=1
        )
        
        # Serialize
        message_dict = original.to_dict()
        assert isinstance(message_dict, dict)
        assert message_dict["sender_id"] == "test_sender"
        assert message_dict["message_type"] == "STATUS_UPDATE"
        
        # Deserialize
        restored = Message.from_dict(message_dict)
        assert restored.sender_id == original.sender_id
        assert restored.receiver_id == original.receiver_id
        assert restored.message_type == original.message_type
        assert restored.content == original.content
        assert restored.priority == original.priority
    
    def test_message_comparison(self):
        """Test message priority comparison for queue ordering."""
        high_priority = Message("a1", "a2", MessageType.EMERGENCY, {}, priority=3)
        low_priority = Message("a1", "a2", MessageType.COORDINATION, {}, priority=1)
        
        # Higher priority should be "less than" for priority queue
        assert high_priority < low_priority
        assert not (low_priority < high_priority)


class TestCommunicationProtocol:
    """Test communication protocol functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.protocol = CommunicationProtocol(max_queue_size=100)
    
    def test_initialization(self):
        """Test protocol initialization."""
        assert self.protocol.max_queue_size == 100
        assert len(self.protocol.message_queues) == 0
        assert len(self.protocol.subscribers) == 0
    
    def test_agent_registration(self):
        """Test agent registration."""
        self.protocol.register_agent("agent_1")
        
        assert "agent_1" in self.protocol.message_queues
        assert "agent_1" in self.protocol.subscribers
        assert self.protocol.message_queues["agent_1"].empty()
    
    def test_message_sending_and_receiving(self):
        """Test basic message sending and receiving."""
        # Register agents
        self.protocol.register_agent("sender")
        self.protocol.register_agent("receiver")
        
        # Send message
        message = Message(
            sender_id="sender",
            receiver_id="receiver",
            message_type=MessageType.TASK_ASSIGNMENT,
            content={"task": "test_task"}
        )
        
        success = self.protocol.send_message(message)
        assert success
        
        # Receive message
        received = self.protocol.receive_message("receiver")
        assert received is not None
        assert received.sender_id == "sender"
        assert received.content["task"] == "test_task"
    
    def test_broadcast_message(self):
        """Test message broadcasting to all agents."""
        agents = ["agent_1", "agent_2", "agent_3"]
        for agent in agents:
            self.protocol.register_agent(agent)
        
        broadcast_msg = Message(
            sender_id="coordinator",
            receiver_id="*",  # Broadcast
            message_type=MessageType.COORDINATION,
            content={"command": "start_mission"}
        )
        
        success = self.protocol.send_message(broadcast_msg)
        assert success
        
        # All agents should receive the message
        for agent in agents:
            received = self.protocol.receive_message(agent)
            assert received is not None
            assert received.content["command"] == "start_mission"
    
    def test_message_priority_ordering(self):
        """Test priority-based message ordering."""
        self.protocol.register_agent("receiver")
        
        # Send messages with different priorities
        low_msg = Message("sender", "receiver", MessageType.STATUS_UPDATE, {"priority": "low"}, priority=1)
        high_msg = Message("sender", "receiver", MessageType.EMERGENCY, {"priority": "high"}, priority=3)
        medium_msg = Message("sender", "receiver", MessageType.TASK_ASSIGNMENT, {"priority": "medium"}, priority=2)
        
        # Send in random order
        self.protocol.send_message(low_msg)
        self.protocol.send_message(high_msg)
        self.protocol.send_message(medium_msg)
        
        # Should receive in priority order (high, medium, low)
        first = self.protocol.receive_message("receiver")
        second = self.protocol.receive_message("receiver")
        third = self.protocol.receive_message("receiver")
        
        assert first.content["priority"] == "high"
        assert second.content["priority"] == "medium"
        assert third.content["priority"] == "low"
    
    def test_queue_overflow_protection(self):
        """Test protection against message queue overflow."""
        protocol = CommunicationProtocol(max_queue_size=2)
        protocol.register_agent("receiver")
        
        # Fill queue to capacity
        msg1 = Message("s", "receiver", MessageType.COORDINATION, {"id": 1})
        msg2 = Message("s", "receiver", MessageType.COORDINATION, {"id": 2})
        msg3 = Message("s", "receiver", MessageType.COORDINATION, {"id": 3})
        
        assert protocol.send_message(msg1)
        assert protocol.send_message(msg2)
        assert not protocol.send_message(msg3)  # Should fail due to full queue
    
    def test_message_subscription(self):
        """Test message type subscription filtering."""
        self.protocol.register_agent("subscriber")
        
        # Subscribe only to task assignments
        self.protocol.subscribe_to_message_type("subscriber", MessageType.TASK_ASSIGNMENT)
        
        # Send different types of messages
        task_msg = Message("sender", "subscriber", MessageType.TASK_ASSIGNMENT, {"task": "navigate"})
        status_msg = Message("sender", "subscriber", MessageType.STATUS_UPDATE, {"status": "running"})
        
        self.protocol.send_message(task_msg)
        self.protocol.send_message(status_msg)
        
        # Should only receive the task assignment
        received = self.protocol.receive_message("subscriber")
        assert received is not None
        assert received.message_type == MessageType.TASK_ASSIGNMENT
        
        # No more messages should be available
        assert self.protocol.receive_message("subscriber") is None
    
    def test_get_message_stats(self):
        """Test message statistics collection."""
        self.protocol.register_agent("agent_1")
        self.protocol.register_agent("agent_2")
        
        # Send some messages
        for i in range(5):
            msg = Message(f"sender_{i}", "agent_1", MessageType.COORDINATION, {"msg": i})
            self.protocol.send_message(msg)
        
        stats = self.protocol.get_message_stats()
        
        assert isinstance(stats, dict)
        assert "total_messages_sent" in stats
        assert "messages_per_agent" in stats
        assert "queue_sizes" in stats
        
        assert stats["total_messages_sent"] == 5
        assert stats["queue_sizes"]["agent_1"] == 5


class TestDynamicRoleAssignment:
    """Test dynamic role assignment system."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.role_system = DynamicRoleAssignment()
    
    def test_initialization(self):
        """Test role system initialization."""
        assert len(self.role_system.agents) == 0
        assert len(self.role_system.roles) == 0
        assert isinstance(self.role_system.role_history, list)
    
    def test_agent_registration(self):
        """Test agent registration with capabilities."""
        capabilities = ["navigation", "manipulation", "perception"]
        
        self.role_system.register_agent(
            agent_id="agent_1",
            capabilities=capabilities,
            performance_history={"navigation": 0.8, "manipulation": 0.6}
        )
        
        assert "agent_1" in self.role_system.agents
        agent = self.role_system.agents["agent_1"]
        assert agent["capabilities"] == capabilities
        assert agent["performance_history"]["navigation"] == 0.8
    
    def test_role_definition(self):
        """Test role definition and requirements."""
        role_requirements = {
            "required_capabilities": ["navigation", "communication"],
            "preferred_capabilities": ["mapping"],
            "min_performance": {"navigation": 0.7},
            "max_agents": 2
        }
        
        self.role_system.define_role("scout", role_requirements)
        
        assert "scout" in self.role_system.roles
        role = self.role_system.roles["scout"]
        assert role["required_capabilities"] == ["navigation", "communication"]
        assert role["max_agents"] == 2
    
    def test_automatic_role_assignment(self):
        """Test automatic role assignment based on capabilities."""
        # Register agents with different capabilities
        self.role_system.register_agent(
            "navigator", 
            capabilities=["navigation", "communication"],
            performance_history={"navigation": 0.9, "communication": 0.8}
        )
        
        self.role_system.register_agent(
            "manipulator",
            capabilities=["manipulation", "perception"], 
            performance_history={"manipulation": 0.85, "perception": 0.7}
        )
        
        # Define roles
        self.role_system.define_role("scout", {
            "required_capabilities": ["navigation"],
            "min_performance": {"navigation": 0.7},
            "max_agents": 1
        })
        
        self.role_system.define_role("worker", {
            "required_capabilities": ["manipulation"],
            "min_performance": {"manipulation": 0.8},
            "max_agents": 1
        })
        
        # Assign roles
        assignments = self.role_system.assign_roles()
        
        assert "scout" in assignments
        assert "worker" in assignments
        assert assignments["scout"] == ["navigator"]
        assert assignments["worker"] == ["manipulator"]
    
    def test_role_reassignment_on_performance_change(self):
        """Test role reassignment when agent performance changes."""
        # Register agents
        self.role_system.register_agent(
            "agent_1",
            capabilities=["navigation", "manipulation"],
            performance_history={"navigation": 0.9, "manipulation": 0.5}
        )
        
        self.role_system.register_agent(
            "agent_2",
            capabilities=["navigation", "manipulation"],
            performance_history={"navigation": 0.6, "manipulation": 0.9}
        )
        
        # Define role requiring high navigation performance
        self.role_system.define_role("lead_navigator", {
            "required_capabilities": ["navigation"],
            "min_performance": {"navigation": 0.8},
            "max_agents": 1
        })
        
        # Initial assignment
        initial_assignments = self.role_system.assign_roles()
        assert initial_assignments["lead_navigator"] == ["agent_1"]
        
        # Update agent_2's navigation performance
        self.role_system.update_agent_performance("agent_2", {"navigation": 0.95})
        
        # Reassign roles
        new_assignments = self.role_system.assign_roles()
        assert new_assignments["lead_navigator"] == ["agent_2"]  # Should switch to better performer
    
    def test_workload_balancing(self):
        """Test workload balancing across agents."""
        # Register multiple capable agents
        for i in range(3):
            self.role_system.register_agent(
                f"agent_{i}",
                capabilities=["navigation", "communication"],
                performance_history={"navigation": 0.8, "communication": 0.8}
            )
        
        # Define role that can have multiple agents
        self.role_system.define_role("scout_team", {
            "required_capabilities": ["navigation"],
            "max_agents": 3
        })
        
        # Simulate different current workloads
        self.role_system.agents["agent_0"]["current_tasks"] = 5
        self.role_system.agents["agent_1"]["current_tasks"] = 2
        self.role_system.agents["agent_2"]["current_tasks"] = 3
        
        assignments = self.role_system.assign_roles(balance_workload=True)
        
        # Should prefer agent with lower workload
        scout_agents = assignments["scout_team"]
        assert "agent_1" in scout_agents  # Lowest workload should be included
    
    def test_role_history_tracking(self):
        """Test tracking of role assignment history."""
        self.role_system.register_agent("agent_1", capabilities=["navigation"])
        self.role_system.define_role("explorer", {"required_capabilities": ["navigation"]})
        
        # Make assignment
        assignments = self.role_system.assign_roles()
        
        # Check history was recorded
        assert len(self.role_system.role_history) > 0
        
        last_assignment = self.role_system.role_history[-1]
        assert "timestamp" in last_assignment
        assert "assignments" in last_assignment
        assert last_assignment["assignments"]["explorer"] == ["agent_1"]


class TestCoordinationOrchestrator:
    """Test coordination orchestrator functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.orchestrator = CoordinationOrchestrator()
    
    @pytest.mark.asyncio
    async def test_task_decomposition(self):
        """Test complex task decomposition into subtasks."""
        complex_task = CoordinationTask(
            task_id="furniture_assembly",
            description="Assemble a chair with multiple agents",
            required_capabilities=["manipulation", "perception", "planning"],
            estimated_duration=300,  # 5 minutes
            dependencies=[]
        )
        
        subtasks = await self.orchestrator.decompose_task(complex_task)
        
        assert isinstance(subtasks, list)
        assert len(subtasks) > 1
        
        # Check that subtasks have required fields
        for subtask in subtasks:
            assert hasattr(subtask, 'task_id')
            assert hasattr(subtask, 'required_capabilities')
            assert hasattr(subtask, 'dependencies')
    
    @pytest.mark.asyncio
    async def test_task_allocation_optimization(self):
        """Test optimal task allocation to agents."""
        # Create agents with different capabilities
        agents = {
            "manipulator_1": {"capabilities": ["manipulation"], "availability": True},
            "navigator_1": {"capabilities": ["navigation"], "availability": True},
            "generalist_1": {"capabilities": ["manipulation", "navigation"], "availability": True}
        }
        
        # Create tasks requiring different capabilities
        tasks = [
            CoordinationTask("move_object", "Move object to location", ["manipulation"], 60),
            CoordinationTask("explore_area", "Explore unknown area", ["navigation"], 90),
            CoordinationTask("fetch_tool", "Get tool and bring back", ["manipulation", "navigation"], 120)
        ]
        
        allocation = await self.orchestrator.allocate_tasks(tasks, agents)
        
        assert isinstance(allocation, dict)
        
        # Verify allocation is valid
        for agent_id, assigned_tasks in allocation.items():
            assert agent_id in agents
            for task in assigned_tasks:
                # Agent should have required capabilities
                agent_caps = set(agents[agent_id]["capabilities"])
                task_caps = set(task.required_capabilities)
                assert task_caps.issubset(agent_caps)
    
    @pytest.mark.asyncio 
    async def test_dependency_ordering(self):
        """Test proper ordering of tasks with dependencies."""
        # Create tasks with dependencies
        task_a = CoordinationTask("prepare_workspace", "Clear work area", ["manipulation"], 30)
        task_b = CoordinationTask("fetch_materials", "Get assembly materials", ["navigation"], 60, dependencies=["prepare_workspace"])
        task_c = CoordinationTask("assemble", "Put parts together", ["manipulation"], 120, dependencies=["fetch_materials"])
        
        tasks = [task_c, task_a, task_b]  # Intentionally out of order
        
        ordered_tasks = await self.orchestrator.order_tasks_by_dependencies(tasks)
        
        # Should be ordered by dependencies
        task_names = [t.task_id for t in ordered_tasks]
        assert task_names.index("prepare_workspace") < task_names.index("fetch_materials")
        assert task_names.index("fetch_materials") < task_names.index("assemble")
    
    @pytest.mark.asyncio
    async def test_real_time_coordination(self):
        """Test real-time coordination during task execution."""
        agents = {
            "agent_1": {"status": "idle", "current_task": None},
            "agent_2": {"status": "busy", "current_task": "existing_task"}
        }
        
        task = CoordinationTask("urgent_task", "Handle urgent situation", ["any"], 30)
        
        # Mock message protocol
        mock_protocol = Mock()
        mock_protocol.send_message = Mock(return_value=True)
        self.orchestrator.communication = mock_protocol
        
        coordination_plan = await self.orchestrator.coordinate_execution(task, agents)
        
        assert isinstance(coordination_plan, dict)
        assert "assigned_agent" in coordination_plan
        assert "execution_plan" in coordination_plan
        
        # Should assign to idle agent
        assert coordination_plan["assigned_agent"] == "agent_1"
    
    @pytest.mark.asyncio
    async def test_failure_recovery_coordination(self):
        """Test coordination when agents fail during task execution."""
        # Simulate agent failure scenario
        failed_task = CoordinationTask("failed_task", "Task that failed", ["manipulation"], 60)
        
        agents = {
            "primary_agent": {"status": "failed", "current_task": "failed_task"},
            "backup_agent": {"status": "idle", "current_task": None}
        }
        
        recovery_plan = await self.orchestrator.handle_agent_failure("primary_agent", failed_task, agents)
        
        assert isinstance(recovery_plan, dict)
        assert "recovery_action" in recovery_plan
        assert "backup_assignment" in recovery_plan
        
        # Should reassign to backup agent
        assert recovery_plan["backup_assignment"]["agent_id"] == "backup_agent"
    
    @pytest.mark.asyncio
    async def test_performance_monitoring_integration(self):
        """Test integration with performance monitoring."""
        task = CoordinationTask("monitored_task", "Task with monitoring", ["navigation"], 90)
        
        agent_performance = {
            "agent_1": {"success_rate": 0.95, "avg_completion_time": 85},
            "agent_2": {"success_rate": 0.80, "avg_completion_time": 110}
        }
        
        optimal_agent = await self.orchestrator.select_optimal_agent(task, agent_performance)
        
        # Should select agent with better performance
        assert optimal_agent == "agent_1"
    
    def test_coordination_metrics_collection(self):
        """Test collection of coordination metrics."""
        # Simulate some coordination activities
        self.orchestrator.record_task_completion("task_1", "agent_1", 45, success=True)
        self.orchestrator.record_task_completion("task_2", "agent_2", 120, success=False)
        self.orchestrator.record_task_completion("task_3", "agent_1", 60, success=True)
        
        metrics = self.orchestrator.get_coordination_metrics()
        
        assert isinstance(metrics, dict)
        assert "total_tasks" in metrics
        assert "success_rate" in metrics
        assert "average_completion_time" in metrics
        assert "agent_performance" in metrics
        
        assert metrics["total_tasks"] == 3
        assert metrics["success_rate"] == 2/3  # 2 successes out of 3 tasks
        
        # Agent-specific metrics
        agent_1_metrics = metrics["agent_performance"]["agent_1"]
        assert agent_1_metrics["tasks_completed"] == 2
        assert agent_1_metrics["success_rate"] == 1.0  # Both tasks succeeded