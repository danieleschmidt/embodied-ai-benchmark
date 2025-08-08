"""
Project Orchestration System

Intelligent system for managing the complete software development lifecycle,
including sprint planning, task decomposition, progress tracking, and delivery coordination.
"""

import json
import uuid
from typing import Dict, List, Tuple, Optional, Any, Set
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
import logging
from pathlib import Path

from .requirements_engine import Requirement, RequirementType, Priority
from .code_generator import GeneratedCode, Language


class TaskStatus(Enum):
    """Task execution status"""
    BACKLOG = "backlog"
    TODO = "todo"
    IN_PROGRESS = "in_progress"
    CODE_REVIEW = "code_review"
    TESTING = "testing"
    DONE = "done"
    BLOCKED = "blocked"


class TaskType(Enum):
    """Types of development tasks"""
    ANALYSIS = "analysis"
    DESIGN = "design"
    IMPLEMENTATION = "implementation"
    TESTING = "testing"
    DOCUMENTATION = "documentation"
    DEPLOYMENT = "deployment"
    MAINTENANCE = "maintenance"


@dataclass
class DevelopmentTask:
    """Individual development task"""
    id: str
    title: str
    description: str
    type: TaskType
    status: TaskStatus
    requirement_id: Optional[str] = None
    assignee: Optional[str] = "AI_Agent"
    story_points: int = 1
    priority: Priority = Priority.MEDIUM
    dependencies: List[str] = field(default_factory=list)
    acceptance_criteria: List[str] = field(default_factory=list)
    estimated_hours: Optional[float] = None
    actual_hours: Optional[float] = None
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    blocked_reason: Optional[str] = None
    tags: Set[str] = field(default_factory=set)
    artifacts: List[str] = field(default_factory=list)  # Generated file paths
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation"""
        return {
            'id': self.id,
            'title': self.title,
            'description': self.description,
            'type': self.type.value,
            'status': self.status.value,
            'requirement_id': self.requirement_id,
            'assignee': self.assignee,
            'story_points': self.story_points,
            'priority': self.priority.value,
            'dependencies': self.dependencies,
            'acceptance_criteria': self.acceptance_criteria,
            'estimated_hours': self.estimated_hours,
            'actual_hours': self.actual_hours,
            'created_at': self.created_at.isoformat(),
            'started_at': self.started_at.isoformat() if self.started_at else None,
            'completed_at': self.completed_at.isoformat() if self.completed_at else None,
            'blocked_reason': self.blocked_reason,
            'tags': list(self.tags),
            'artifacts': self.artifacts
        }
    
    def start_work(self) -> None:
        """Mark task as started"""
        self.status = TaskStatus.IN_PROGRESS
        self.started_at = datetime.now()
    
    def complete_work(self, artifacts: List[str] = None) -> None:
        """Mark task as completed"""
        self.status = TaskStatus.DONE
        self.completed_at = datetime.now()
        if artifacts:
            self.artifacts.extend(artifacts)
    
    def block_task(self, reason: str) -> None:
        """Block task with reason"""
        self.status = TaskStatus.BLOCKED
        self.blocked_reason = reason
    
    def unblock_task(self) -> None:
        """Unblock task"""
        if self.status == TaskStatus.BLOCKED:
            self.status = TaskStatus.TODO
            self.blocked_reason = None


@dataclass
class Sprint:
    """Agile sprint representation"""
    id: str
    name: str
    goal: str
    start_date: datetime
    end_date: datetime
    capacity_hours: float
    tasks: List[DevelopmentTask] = field(default_factory=list)
    status: str = "planned"  # planned, active, completed, cancelled
    
    @property
    def duration_days(self) -> int:
        """Get sprint duration in days"""
        return (self.end_date - self.start_date).days
    
    @property
    def total_story_points(self) -> int:
        """Get total story points in sprint"""
        return sum(task.story_points for task in self.tasks)
    
    @property
    def completed_story_points(self) -> int:
        """Get completed story points"""
        return sum(task.story_points for task in self.tasks if task.status == TaskStatus.DONE)
    
    @property
    def burndown_data(self) -> List[Dict[str, Any]]:
        """Get burndown chart data"""
        # Simplified burndown calculation
        total_points = self.total_story_points
        completed_points = self.completed_story_points
        remaining_points = total_points - completed_points
        
        return [
            {'day': 0, 'remaining': total_points, 'ideal': total_points},
            {'day': self.duration_days, 'remaining': remaining_points, 'ideal': 0}
        ]
    
    def add_task(self, task: DevelopmentTask) -> None:
        """Add task to sprint"""
        self.tasks.append(task)
    
    def remove_task(self, task_id: str) -> bool:
        """Remove task from sprint"""
        initial_count = len(self.tasks)
        self.tasks = [task for task in self.tasks if task.id != task_id]
        return len(self.tasks) < initial_count


class TaskDecomposer:
    """Breaks down requirements into development tasks"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def decompose_requirement(self, requirement: Requirement) -> List[DevelopmentTask]:
        """Break requirement into development tasks"""
        tasks = []
        
        # Always start with analysis
        analysis_task = self._create_analysis_task(requirement)
        tasks.append(analysis_task)
        
        # Generate task type based on requirement type
        if requirement.type == RequirementType.FUNCTIONAL:
            tasks.extend(self._decompose_functional_requirement(requirement))
        elif requirement.type == RequirementType.TECHNICAL:
            tasks.extend(self._decompose_technical_requirement(requirement))
        elif requirement.type in [RequirementType.SECURITY, RequirementType.PERFORMANCE]:
            tasks.extend(self._decompose_nonfunctional_requirement(requirement))
        else:
            tasks.extend(self._decompose_generic_requirement(requirement))
        
        # Always end with testing and documentation
        testing_task = self._create_testing_task(requirement)
        tasks.append(testing_task)
        
        doc_task = self._create_documentation_task(requirement)
        tasks.append(doc_task)
        
        # Set up dependencies
        self._setup_task_dependencies(tasks)
        
        self.logger.info(f"Decomposed requirement {requirement.id} into {len(tasks)} tasks")
        return tasks
    
    def _create_analysis_task(self, requirement: Requirement) -> DevelopmentTask:
        """Create analysis task for requirement"""
        return DevelopmentTask(
            id=f"TASK-{uuid.uuid4().hex[:8].upper()}",
            title=f"Analyze {requirement.title}",
            description=f"Analyze and design solution for: {requirement.description}",
            type=TaskType.ANALYSIS,
            status=TaskStatus.BACKLOG,
            requirement_id=requirement.id,
            story_points=2,
            priority=requirement.priority,
            acceptance_criteria=[
                "Requirements are fully understood",
                "Solution approach is documented",
                "Technical risks are identified",
                "Architecture decisions are made"
            ],
            estimated_hours=4.0
        )
    
    def _decompose_functional_requirement(self, requirement: Requirement) -> List[DevelopmentTask]:
        """Decompose functional requirement"""
        tasks = []
        
        # Design task
        design_task = DevelopmentTask(
            id=f"TASK-{uuid.uuid4().hex[:8].upper()}",
            title=f"Design {requirement.title}",
            description=f"Create detailed design for {requirement.description}",
            type=TaskType.DESIGN,
            status=TaskStatus.BACKLOG,
            requirement_id=requirement.id,
            story_points=3,
            priority=requirement.priority,
            estimated_hours=6.0
        )
        tasks.append(design_task)
        
        # Implementation task
        impl_task = DevelopmentTask(
            id=f"TASK-{uuid.uuid4().hex[:8].upper()}",
            title=f"Implement {requirement.title}",
            description=f"Implement functionality: {requirement.description}",
            type=TaskType.IMPLEMENTATION,
            status=TaskStatus.BACKLOG,
            requirement_id=requirement.id,
            story_points=requirement.effort_estimate or 5,
            priority=requirement.priority,
            acceptance_criteria=requirement.acceptance_criteria.copy(),
            estimated_hours=(requirement.effort_estimate or 5) * 2.0
        )
        tasks.append(impl_task)
        
        # Database task if needed
        if 'database' in requirement.description.lower() or 'data' in requirement.description.lower():
            db_task = DevelopmentTask(
                id=f"TASK-{uuid.uuid4().hex[:8].upper()}",
                title=f"Database schema for {requirement.title}",
                description=f"Create database schema and models for {requirement.description}",
                type=TaskType.IMPLEMENTATION,
                status=TaskStatus.BACKLOG,
                requirement_id=requirement.id,
                story_points=3,
                priority=requirement.priority,
                tags={"database"},
                estimated_hours=4.0
            )
            tasks.append(db_task)
        
        # API task if needed
        if 'api' in requirement.description.lower() or 'endpoint' in requirement.description.lower():
            api_task = DevelopmentTask(
                id=f"TASK-{uuid.uuid4().hex[:8].upper()}",
                title=f"API endpoint for {requirement.title}",
                description=f"Create API endpoint for {requirement.description}",
                type=TaskType.IMPLEMENTATION,
                status=TaskStatus.BACKLOG,
                requirement_id=requirement.id,
                story_points=4,
                priority=requirement.priority,
                tags={"api"},
                estimated_hours=6.0
            )
            tasks.append(api_task)
        
        return tasks
    
    def _decompose_technical_requirement(self, requirement: Requirement) -> List[DevelopmentTask]:
        """Decompose technical requirement"""
        tasks = []
        
        # Technical implementation task
        tech_task = DevelopmentTask(
            id=f"TASK-{uuid.uuid4().hex[:8].upper()}",
            title=f"Technical implementation: {requirement.title}",
            description=requirement.description,
            type=TaskType.IMPLEMENTATION,
            status=TaskStatus.BACKLOG,
            requirement_id=requirement.id,
            story_points=requirement.effort_estimate or 6,
            priority=requirement.priority,
            tags={"technical"},
            estimated_hours=(requirement.effort_estimate or 6) * 1.5
        )
        tasks.append(tech_task)
        
        return tasks
    
    def _decompose_nonfunctional_requirement(self, requirement: Requirement) -> List[DevelopmentTask]:
        """Decompose non-functional requirement"""
        tasks = []
        
        # Non-functional implementation
        nf_task = DevelopmentTask(
            id=f"TASK-{uuid.uuid4().hex[:8].upper()}",
            title=f"Implement {requirement.type.value.replace('_', ' ').title()}: {requirement.title}",
            description=requirement.description,
            type=TaskType.IMPLEMENTATION,
            status=TaskStatus.BACKLOG,
            requirement_id=requirement.id,
            story_points=requirement.effort_estimate or 4,
            priority=requirement.priority,
            tags={requirement.type.value},
            estimated_hours=(requirement.effort_estimate or 4) * 1.8
        )
        tasks.append(nf_task)
        
        # Performance/security often needs monitoring
        if requirement.type in [RequirementType.PERFORMANCE, RequirementType.SECURITY]:
            monitoring_task = DevelopmentTask(
                id=f"TASK-{uuid.uuid4().hex[:8].upper()}",
                title=f"Monitoring for {requirement.title}",
                description=f"Set up monitoring and alerting for {requirement.description}",
                type=TaskType.IMPLEMENTATION,
                status=TaskStatus.BACKLOG,
                requirement_id=requirement.id,
                story_points=2,
                priority=requirement.priority,
                tags={"monitoring"},
                estimated_hours=3.0
            )
            tasks.append(monitoring_task)
        
        return tasks
    
    def _decompose_generic_requirement(self, requirement: Requirement) -> List[DevelopmentTask]:
        """Decompose generic/business requirement"""
        tasks = []
        
        # Generic implementation task
        generic_task = DevelopmentTask(
            id=f"TASK-{uuid.uuid4().hex[:8].upper()}",
            title=f"Implement {requirement.title}",
            description=requirement.description,
            type=TaskType.IMPLEMENTATION,
            status=TaskStatus.BACKLOG,
            requirement_id=requirement.id,
            story_points=requirement.effort_estimate or 3,
            priority=requirement.priority,
            estimated_hours=(requirement.effort_estimate or 3) * 2.0
        )
        tasks.append(generic_task)
        
        return tasks
    
    def _create_testing_task(self, requirement: Requirement) -> DevelopmentTask:
        """Create testing task for requirement"""
        return DevelopmentTask(
            id=f"TASK-{uuid.uuid4().hex[:8].upper()}",
            title=f"Test {requirement.title}",
            description=f"Create and execute tests for {requirement.description}",
            type=TaskType.TESTING,
            status=TaskStatus.BACKLOG,
            requirement_id=requirement.id,
            story_points=2,
            priority=requirement.priority,
            acceptance_criteria=[
                "Unit tests written and passing",
                "Integration tests implemented",
                "Test coverage > 80%",
                "All acceptance criteria verified"
            ],
            estimated_hours=4.0
        )
    
    def _create_documentation_task(self, requirement: Requirement) -> DevelopmentTask:
        """Create documentation task for requirement"""
        return DevelopmentTask(
            id=f"TASK-{uuid.uuid4().hex[:8].upper()}",
            title=f"Document {requirement.title}",
            description=f"Create documentation for {requirement.description}",
            type=TaskType.DOCUMENTATION,
            status=TaskStatus.BACKLOG,
            requirement_id=requirement.id,
            story_points=1,
            priority=Priority.LOW,  # Documentation typically lower priority
            acceptance_criteria=[
                "API documentation updated",
                "User guide sections written",
                "Code comments added",
                "Architecture documentation updated"
            ],
            estimated_hours=2.0
        )
    
    def _setup_task_dependencies(self, tasks: List[DevelopmentTask]) -> None:
        """Set up dependencies between tasks"""
        if len(tasks) < 2:
            return
        
        # Basic dependency chain: Analysis -> Design/Implementation -> Testing -> Documentation
        task_types = {task.type: task for task in tasks}
        
        if TaskType.ANALYSIS in task_types and TaskType.DESIGN in task_types:
            task_types[TaskType.DESIGN].dependencies.append(task_types[TaskType.ANALYSIS].id)
        
        if TaskType.DESIGN in task_types:
            for task in tasks:
                if task.type == TaskType.IMPLEMENTATION and TaskType.DESIGN in task_types:
                    task.dependencies.append(task_types[TaskType.DESIGN].id)
        
        if TaskType.TESTING in task_types:
            for task in tasks:
                if task.type == TaskType.IMPLEMENTATION:
                    task_types[TaskType.TESTING].dependencies.append(task.id)
        
        if TaskType.DOCUMENTATION in task_types:
            for task in tasks:
                if task.type in [TaskType.IMPLEMENTATION, TaskType.TESTING]:
                    task_types[TaskType.DOCUMENTATION].dependencies.append(task.id)


class SprintPlanner:
    """Plans and manages sprints"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def create_sprint_plan(self, 
                          tasks: List[DevelopmentTask],
                          sprint_capacity_hours: float = 80.0,
                          sprint_duration_days: int = 14) -> List[Sprint]:
        """Create sprint plan from tasks"""
        sprints = []
        remaining_tasks = tasks.copy()
        sprint_number = 1
        
        while remaining_tasks:
            sprint = self._create_sprint(
                sprint_number,
                remaining_tasks,
                sprint_capacity_hours,
                sprint_duration_days
            )
            
            # Remove tasks assigned to sprint
            task_ids_in_sprint = {task.id for task in sprint.tasks}
            remaining_tasks = [task for task in remaining_tasks if task.id not in task_ids_in_sprint]
            
            sprints.append(sprint)
            sprint_number += 1
            
            # Safety check to avoid infinite loop
            if sprint_number > 20:
                self.logger.warning("Too many sprints generated, stopping")
                break
        
        self.logger.info(f"Created {len(sprints)} sprints for {len(tasks)} tasks")
        return sprints
    
    def _create_sprint(self, 
                      sprint_number: int,
                      available_tasks: List[DevelopmentTask],
                      capacity_hours: float,
                      duration_days: int) -> Sprint:
        """Create individual sprint"""
        start_date = datetime.now() + timedelta(days=(sprint_number - 1) * duration_days)
        end_date = start_date + timedelta(days=duration_days)
        
        sprint = Sprint(
            id=f"SPRINT-{sprint_number:02d}",
            name=f"Sprint {sprint_number}",
            goal=self._determine_sprint_goal(available_tasks),
            start_date=start_date,
            end_date=end_date,
            capacity_hours=capacity_hours
        )
        
        # Select tasks for sprint using capacity and dependencies
        selected_tasks = self._select_tasks_for_sprint(
            available_tasks, capacity_hours
        )
        
        for task in selected_tasks:
            sprint.add_task(task)
        
        return sprint
    
    def _determine_sprint_goal(self, tasks: List[DevelopmentTask]) -> str:
        """Determine sprint goal from available tasks"""
        if not tasks:
            return "Complete remaining tasks"
        
        # Group tasks by requirement
        req_tasks = {}
        for task in tasks[:10]:  # Look at first 10 tasks
            req_id = task.requirement_id
            if req_id:
                if req_id not in req_tasks:
                    req_tasks[req_id] = []
                req_tasks[req_id].append(task)
        
        if req_tasks:
            # Find the requirement with most tasks
            main_req = max(req_tasks.keys(), key=lambda k: len(req_tasks[k]))
            first_task = req_tasks[main_req][0]
            return f"Implement {first_task.title} and related functionality"
        
        return f"Complete {tasks[0].title} and related tasks"
    
    def _select_tasks_for_sprint(self, 
                                available_tasks: List[DevelopmentTask],
                                capacity_hours: float) -> List[DevelopmentTask]:
        """Select tasks for sprint based on capacity and dependencies"""
        selected = []
        used_capacity = 0.0
        
        # Sort tasks by priority and dependencies
        sorted_tasks = self._sort_tasks_for_selection(available_tasks)
        
        for task in sorted_tasks:
            if not task.estimated_hours:
                continue
            
            # Check if we have capacity
            if used_capacity + task.estimated_hours <= capacity_hours:
                # Check if dependencies are satisfied
                if self._dependencies_satisfied(task, selected, available_tasks):
                    selected.append(task)
                    used_capacity += task.estimated_hours
        
        return selected
    
    def _sort_tasks_for_selection(self, tasks: List[DevelopmentTask]) -> List[DevelopmentTask]:
        """Sort tasks for sprint selection"""
        return sorted(tasks, key=lambda t: (
            -t.priority.value,  # Higher priority first
            len(t.dependencies),  # Tasks with fewer dependencies first
            t.estimated_hours or 0  # Smaller tasks first
        ))
    
    def _dependencies_satisfied(self, 
                              task: DevelopmentTask,
                              selected_tasks: List[DevelopmentTask],
                              all_available: List[DevelopmentTask]) -> bool:
        """Check if task dependencies are satisfied"""
        if not task.dependencies:
            return True
        
        selected_ids = {t.id for t in selected_tasks}
        
        for dep_id in task.dependencies:
            # Check if dependency is in selected tasks
            if dep_id in selected_ids:
                continue
            
            # Check if dependency is already completed
            dep_task = next((t for t in all_available if t.id == dep_id), None)
            if dep_task and dep_task.status == TaskStatus.DONE:
                continue
            
            # Dependency not satisfied
            return False
        
        return True


class ProjectOrchestrator:
    """Main orchestrator for project management"""
    
    def __init__(self):
        self.task_decomposer = TaskDecomposer()
        self.sprint_planner = SprintPlanner()
        self.tasks: Dict[str, DevelopmentTask] = {}
        self.sprints: Dict[str, Sprint] = {}
        self.logger = logging.getLogger(__name__)
    
    def create_project_plan(self, requirements: List[Requirement]) -> Dict[str, Any]:
        """Create complete project plan from requirements"""
        self.logger.info(f"Creating project plan for {len(requirements)} requirements")
        
        # Decompose all requirements into tasks
        all_tasks = []
        for requirement in requirements:
            tasks = self.task_decomposer.decompose_requirement(requirement)
            all_tasks.extend(tasks)
            
            # Store tasks
            for task in tasks:
                self.tasks[task.id] = task
        
        # Create sprint plan
        sprints = self.sprint_planner.create_sprint_plan(all_tasks)
        
        # Store sprints
        for sprint in sprints:
            self.sprints[sprint.id] = sprint
        
        # Generate project summary
        project_plan = {
            'summary': {
                'total_requirements': len(requirements),
                'total_tasks': len(all_tasks),
                'total_sprints': len(sprints),
                'estimated_duration_days': len(sprints) * 14,
                'total_story_points': sum(task.story_points for task in all_tasks),
                'total_estimated_hours': sum(task.estimated_hours or 0 for task in all_tasks)
            },
            'requirements': [req.to_dict() for req in requirements],
            'tasks': [task.to_dict() for task in all_tasks],
            'sprints': [sprint.__dict__ for sprint in sprints],
            'task_breakdown_by_type': self._analyze_task_breakdown(all_tasks),
            'risk_analysis': self._analyze_project_risks(requirements, all_tasks)
        }
        
        self.logger.info(f"Created project plan: {project_plan['summary']}")
        return project_plan
    
    def execute_next_task(self) -> Optional[DevelopmentTask]:
        """Execute the next available task"""
        # Find highest priority task that's ready to start
        ready_tasks = [
            task for task in self.tasks.values()
            if task.status == TaskStatus.TODO and self._task_ready_to_start(task)
        ]
        
        if not ready_tasks:
            self.logger.info("No tasks ready to execute")
            return None
        
        # Sort by priority and select highest
        ready_tasks.sort(key=lambda t: t.priority.value, reverse=True)
        task = ready_tasks[0]
        
        # Start the task
        task.start_work()
        self.logger.info(f"Started task: {task.title}")
        
        return task
    
    def complete_task(self, task_id: str, artifacts: List[str] = None) -> bool:
        """Mark task as completed"""
        if task_id not in self.tasks:
            return False
        
        task = self.tasks[task_id]
        task.complete_work(artifacts or [])
        
        self.logger.info(f"Completed task: {task.title}")
        
        # Update dependent tasks
        self._update_dependent_tasks(task_id)
        
        return True
    
    def _task_ready_to_start(self, task: DevelopmentTask) -> bool:
        """Check if task is ready to start"""
        if task.status != TaskStatus.TODO:
            return False
        
        # Check dependencies
        for dep_id in task.dependencies:
            if dep_id in self.tasks:
                dep_task = self.tasks[dep_id]
                if dep_task.status != TaskStatus.DONE:
                    return False
        
        return True
    
    def _update_dependent_tasks(self, completed_task_id: str) -> None:
        """Update tasks that depend on completed task"""
        for task in self.tasks.values():
            if completed_task_id in task.dependencies and task.status == TaskStatus.BACKLOG:
                # Check if all dependencies are now complete
                all_deps_complete = all(
                    self.tasks.get(dep_id, {}).get('status') == TaskStatus.DONE
                    for dep_id in task.dependencies
                )
                
                if all_deps_complete:
                    task.status = TaskStatus.TODO
                    self.logger.info(f"Task {task.title} is now ready to start")
    
    def _analyze_task_breakdown(self, tasks: List[DevelopmentTask]) -> Dict[str, int]:
        """Analyze task breakdown by type"""
        breakdown = {}
        for task in tasks:
            task_type = task.type.value
            breakdown[task_type] = breakdown.get(task_type, 0) + 1
        return breakdown
    
    def _analyze_project_risks(self, requirements: List[Requirement], tasks: List[DevelopmentTask]) -> List[Dict[str, str]]:
        """Analyze project risks"""
        risks = []
        
        # High complexity requirements
        high_effort_reqs = [req for req in requirements if (req.effort_estimate or 0) > 8]
        if high_effort_reqs:
            risks.append({
                'type': 'complexity',
                'description': f"{len(high_effort_reqs)} high-complexity requirements may cause delays",
                'impact': 'high',
                'mitigation': 'Break down complex requirements further, add buffer time'
            })
        
        # Security/compliance requirements
        security_reqs = [req for req in requirements if req.type in [RequirementType.SECURITY, RequirementType.COMPLIANCE]]
        if security_reqs:
            risks.append({
                'type': 'security',
                'description': f"{len(security_reqs)} security/compliance requirements need special attention",
                'impact': 'medium',
                'mitigation': 'Involve security experts, add extra review cycles'
            })
        
        # Task dependencies
        complex_deps = [task for task in tasks if len(task.dependencies) > 2]
        if complex_deps:
            risks.append({
                'type': 'dependencies',
                'description': f"{len(complex_deps)} tasks have complex dependencies",
                'impact': 'medium',
                'mitigation': 'Monitor dependency chains carefully, have contingency plans'
            })
        
        return risks
    
    def get_project_status(self) -> Dict[str, Any]:
        """Get current project status"""
        total_tasks = len(self.tasks)
        completed_tasks = len([t for t in self.tasks.values() if t.status == TaskStatus.DONE])
        in_progress_tasks = len([t for t in self.tasks.values() if t.status == TaskStatus.IN_PROGRESS])
        blocked_tasks = len([t for t in self.tasks.values() if t.status == TaskStatus.BLOCKED])
        
        return {
            'total_tasks': total_tasks,
            'completed_tasks': completed_tasks,
            'in_progress_tasks': in_progress_tasks,
            'blocked_tasks': blocked_tasks,
            'completion_percentage': (completed_tasks / total_tasks * 100) if total_tasks > 0 else 0,
            'active_sprint': self._get_active_sprint(),
            'next_ready_tasks': self._get_next_ready_tasks(5)
        }
    
    def _get_active_sprint(self) -> Optional[Dict[str, Any]]:
        """Get currently active sprint"""
        now = datetime.now()
        for sprint in self.sprints.values():
            if sprint.start_date <= now <= sprint.end_date:
                return {
                    'id': sprint.id,
                    'name': sprint.name,
                    'goal': sprint.goal,
                    'progress': sprint.completed_story_points / sprint.total_story_points * 100 if sprint.total_story_points > 0 else 0,
                    'days_remaining': (sprint.end_date - now).days
                }
        return None
    
    def _get_next_ready_tasks(self, limit: int = 5) -> List[Dict[str, str]]:
        """Get next ready tasks"""
        ready_tasks = [
            task for task in self.tasks.values()
            if task.status == TaskStatus.TODO and self._task_ready_to_start(task)
        ]
        
        ready_tasks.sort(key=lambda t: t.priority.value, reverse=True)
        
        return [
            {
                'id': task.id,
                'title': task.title,
                'type': task.type.value,
                'priority': task.priority.name,
                'estimated_hours': task.estimated_hours or 0
            }
            for task in ready_tasks[:limit]
        ]
    
    def export_project_plan(self, format: str = 'json') -> str:
        """Export project plan in various formats"""
        project_data = {
            'tasks': {task_id: task.to_dict() for task_id, task in self.tasks.items()},
            'sprints': {sprint_id: sprint.__dict__ for sprint_id, sprint in self.sprints.items()},
            'status': self.get_project_status(),
            'exported_at': datetime.now().isoformat()
        }
        
        if format.lower() == 'json':
            return json.dumps(project_data, indent=2, default=str)
        else:
            raise ValueError(f"Unsupported export format: {format}")
    
    def simulate_project_execution(self) -> Dict[str, Any]:
        """Simulate project execution to identify potential issues"""
        simulation_results = {
            'estimated_completion_date': None,
            'critical_path': [],
            'resource_conflicts': [],
            'recommendations': []
        }
        
        # Calculate critical path (simplified)
        task_graph = self._build_task_graph()
        critical_path = self._find_critical_path(task_graph)
        simulation_results['critical_path'] = critical_path
        
        # Estimate completion date based on critical path
        total_hours = sum(self.tasks[task_id].estimated_hours or 0 for task_id in critical_path)
        working_hours_per_day = 8
        estimated_days = total_hours / working_hours_per_day
        simulation_results['estimated_completion_date'] = (
            datetime.now() + timedelta(days=estimated_days)
        ).isoformat()
        
        # Generate recommendations
        if len(critical_path) > len(self.tasks) * 0.8:
            simulation_results['recommendations'].append(
                "Many tasks are on critical path. Consider parallelizing work."
            )
        
        if total_hours > 500:
            simulation_results['recommendations'].append(
                "Project is large. Consider breaking into phases."
            )
        
        return simulation_results
    
    def _build_task_graph(self) -> Dict[str, List[str]]:
        """Build task dependency graph"""
        graph = {}
        for task_id, task in self.tasks.items():
            graph[task_id] = task.dependencies.copy()
        return graph
    
    def _find_critical_path(self, graph: Dict[str, List[str]]) -> List[str]:
        """Find critical path through tasks (simplified algorithm)"""
        # This is a simplified version - a real implementation would use
        # more sophisticated critical path method (CPM) algorithms
        
        # For now, just find the longest path
        def get_path_length(task_id: str, visited: Set[str]) -> Tuple[int, List[str]]:
            if task_id in visited:
                return 0, []
            
            visited.add(task_id)
            task = self.tasks.get(task_id)
            if not task:
                return 0, []
            
            task_duration = task.estimated_hours or 0
            
            if not task.dependencies:
                return task_duration, [task_id]
            
            max_dep_length = 0
            longest_path = []
            
            for dep_id in task.dependencies:
                dep_length, dep_path = get_path_length(dep_id, visited.copy())
                if dep_length > max_dep_length:
                    max_dep_length = dep_length
                    longest_path = dep_path
            
            return max_dep_length + task_duration, longest_path + [task_id]
        
        longest_duration = 0
        critical_path = []
        
        for task_id in self.tasks:
            duration, path = get_path_length(task_id, set())
            if duration > longest_duration:
                longest_duration = duration
                critical_path = path
        
        return critical_path