"""
Requirements Generation Engine

Intelligent system for analyzing user needs, generating requirements,
and managing the requirements lifecycle autonomously.
"""

import re
import json
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
import hashlib
import logging

from ..utils.error_handling import ErrorHandler
from ..utils.i18n import gettext as _


class RequirementType(Enum):
    """Types of requirements in software development"""
    FUNCTIONAL = "functional"
    NON_FUNCTIONAL = "non_functional"
    BUSINESS = "business"
    TECHNICAL = "technical"
    SECURITY = "security"
    PERFORMANCE = "performance"
    USABILITY = "usability"
    COMPLIANCE = "compliance"


class Priority(Enum):
    """Requirement priority levels"""
    CRITICAL = 5
    HIGH = 4
    MEDIUM = 3
    LOW = 2
    NICE_TO_HAVE = 1


@dataclass
class Requirement:
    """Individual software requirement"""
    id: str
    title: str
    description: str
    type: RequirementType
    priority: Priority
    stakeholder: str
    acceptance_criteria: List[str]
    dependencies: List[str] = field(default_factory=list)
    effort_estimate: Optional[int] = None  # story points
    business_value: Optional[int] = None
    risk_level: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    tags: Set[str] = field(default_factory=set)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary representation"""
        return {
            'id': self.id,
            'title': self.title,
            'description': self.description,
            'type': self.type.value,
            'priority': self.priority.value,
            'stakeholder': self.stakeholder,
            'acceptance_criteria': self.acceptance_criteria,
            'dependencies': self.dependencies,
            'effort_estimate': self.effort_estimate,
            'business_value': self.business_value,
            'risk_level': self.risk_level,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat(),
            'tags': list(self.tags)
        }


@dataclass
class Stakeholder:
    """Project stakeholder representation"""
    name: str
    role: str
    influence: int  # 1-10 scale
    interest: int   # 1-10 scale
    communication_preference: str
    availability: str
    expertise_areas: List[str]
    pain_points: List[str] = field(default_factory=list)
    goals: List[str] = field(default_factory=list)


class StakeholderAnalyzer:
    """Analyzes and models project stakeholders"""
    
    def __init__(self):
        self.stakeholders: Dict[str, Stakeholder] = {}
        self.logger = logging.getLogger(__name__)
    
    def add_stakeholder(self, stakeholder: Stakeholder) -> None:
        """Add stakeholder to analysis"""
        self.stakeholders[stakeholder.name] = stakeholder
        self.logger.info(f"Added stakeholder: {stakeholder.name} ({stakeholder.role})")
    
    def analyze_stakeholder_needs(self, stakeholder_name: str) -> List[str]:
        """Generate needs analysis for a stakeholder"""
        if stakeholder_name not in self.stakeholders:
            return []
        
        stakeholder = self.stakeholders[stakeholder_name]
        needs = []
        
        # Map pain points to needs
        for pain_point in stakeholder.pain_points:
            need = self._pain_point_to_need(pain_point, stakeholder.role)
            if need:
                needs.append(need)
        
        # Map goals to needs
        for goal in stakeholder.goals:
            need = self._goal_to_need(goal, stakeholder.role)
            if need:
                needs.append(need)
        
        return needs
    
    def _pain_point_to_need(self, pain_point: str, role: str) -> Optional[str]:
        """Convert pain point to requirement need"""
        pain_patterns = {
            r"slow|performance|speed": f"System must have fast response times for {role} tasks",
            r"manual|repetitive|tedious": f"System must automate repetitive {role} workflows",
            r"error|mistake|bug": f"System must have robust error handling for {role} operations",
            r"difficult|complex|confusing": f"System must have intuitive interface for {role} users",
            r"integration|compatibility": f"System must integrate seamlessly with existing {role} tools"
        }
        
        for pattern, need_template in pain_patterns.items():
            if re.search(pattern, pain_point.lower()):
                return need_template
        
        return f"System must address: {pain_point}"
    
    def _goal_to_need(self, goal: str, role: str) -> Optional[str]:
        """Convert goal to requirement need"""
        goal_patterns = {
            r"increase|improve|enhance": f"System must support {role} productivity improvements",
            r"reduce|decrease|minimize": f"System must help {role} optimize resource usage",
            r"automate|streamline": f"System must provide workflow automation for {role}",
            r"analyze|report|track": f"System must provide analytics and reporting for {role}",
            r"scale|grow|expand": f"System must support scalable operations for {role}"
        }
        
        for pattern, need_template in goal_patterns.items():
            if re.search(pattern, goal.lower()):
                return need_template
        
        return f"System must enable: {goal}"
    
    def prioritize_stakeholders(self) -> List[Stakeholder]:
        """Prioritize stakeholders by influence and interest"""
        return sorted(
            self.stakeholders.values(),
            key=lambda s: (s.influence * s.interest, s.influence),
            reverse=True
        )


class RequirementPrioritizer:
    """Prioritizes requirements using multiple criteria"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def prioritize_requirements(self, 
                              requirements: List[Requirement],
                              stakeholders: Dict[str, Stakeholder]) -> List[Requirement]:
        """Prioritize requirements using MoSCoW and weighted scoring"""
        scored_reqs = []
        
        for req in requirements:
            score = self._calculate_priority_score(req, stakeholders)
            req.priority = self._score_to_priority(score)
            scored_reqs.append((req, score))
        
        # Sort by score descending
        scored_reqs.sort(key=lambda x: x[1], reverse=True)
        return [req for req, _ in scored_reqs]
    
    def _calculate_priority_score(self, 
                                requirement: Requirement,
                                stakeholders: Dict[str, Stakeholder]) -> float:
        """Calculate composite priority score"""
        score = 0.0
        
        # Business value weight
        if requirement.business_value:
            score += requirement.business_value * 0.3
        
        # Stakeholder influence weight  
        if requirement.stakeholder in stakeholders:
            stakeholder = stakeholders[requirement.stakeholder]
            score += (stakeholder.influence * stakeholder.interest) * 0.25
        
        # Type-based priority
        type_weights = {
            RequirementType.SECURITY: 0.9,
            RequirementType.COMPLIANCE: 0.85,
            RequirementType.FUNCTIONAL: 0.8,
            RequirementType.PERFORMANCE: 0.7,
            RequirementType.BUSINESS: 0.65,
            RequirementType.TECHNICAL: 0.6,
            RequirementType.USABILITY: 0.55,
            RequirementType.NON_FUNCTIONAL: 0.5
        }
        score += type_weights.get(requirement.type, 0.5) * 10 * 0.2
        
        # Risk adjustment
        risk_multipliers = {
            "high": 1.2,
            "medium": 1.0, 
            "low": 0.9
        }
        if requirement.risk_level:
            score *= risk_multipliers.get(requirement.risk_level, 1.0)
        
        # Effort vs value ratio
        if requirement.effort_estimate and requirement.business_value:
            efficiency_ratio = requirement.business_value / max(requirement.effort_estimate, 1)
            score += efficiency_ratio * 0.15
        
        return score
    
    def _score_to_priority(self, score: float) -> Priority:
        """Convert numeric score to priority enum"""
        if score >= 8.0:
            return Priority.CRITICAL
        elif score >= 6.0:
            return Priority.HIGH
        elif score >= 4.0:
            return Priority.MEDIUM
        elif score >= 2.0:
            return Priority.LOW
        else:
            return Priority.NICE_TO_HAVE


class RequirementsEngine:
    """Main requirements generation and management engine"""
    
    def __init__(self):
        self.requirements: Dict[str, Requirement] = {}
        self.stakeholder_analyzer = StakeholderAnalyzer()
        self.prioritizer = RequirementPrioritizer()
        self.error_handler = ErrorHandler()
        self.logger = logging.getLogger(__name__)
    
    def analyze_user_input(self, user_input: str, context: Dict = None) -> List[Requirement]:
        """Generate requirements from natural language user input"""
        try:
            requirements = []
            
            # Extract key information from input
            extracted_info = self._extract_requirements_info(user_input)
            
            # Generate requirements from extracted information
            for info in extracted_info:
                req = self._create_requirement_from_info(info, context or {})
                if req:
                    requirements.append(req)
                    self.requirements[req.id] = req
            
            self.logger.info(f"Generated {len(requirements)} requirements from user input")
            return requirements
            
        except Exception as e:
            self.error_handler.handle_error(e, "requirements_analysis")
            return []
    
    def _extract_requirements_info(self, text: str) -> List[Dict]:
        """Extract requirement information from text using NLP patterns"""
        requirements_info = []
        
        # Requirement indicators
        functional_patterns = [
            r"system (must|should|shall) (.+?)(?=\.|$)",
            r"user (can|should be able to) (.+?)(?=\.|$)", 
            r"application (needs to|will) (.+?)(?=\.|$)",
            r"feature (for|to) (.+?)(?=\.|$)"
        ]
        
        non_functional_patterns = [
            r"performance (.+?)(?=\.|$)",
            r"security (.+?)(?=\.|$)",
            r"scalability (.+?)(?=\.|$)",
            r"availability (.+?)(?=\.|$)"
        ]
        
        # Extract functional requirements
        for pattern in functional_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                if isinstance(match, tuple):
                    modal_verb, description = match
                    requirements_info.append({
                        'type': RequirementType.FUNCTIONAL,
                        'title': f"System {modal_verb} {description[:50]}...",
                        'description': description,
                        'priority': self._infer_priority_from_modal(modal_verb)
                    })
        
        # Extract non-functional requirements
        for pattern in non_functional_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                req_type = pattern.split()[0].replace(r"\b", "").replace(r"(", "")
                requirements_info.append({
                    'type': RequirementType.NON_FUNCTIONAL,
                    'title': f"{req_type.title()} requirement",
                    'description': match,
                    'priority': Priority.MEDIUM
                })
        
        # If no specific patterns found, create generic requirement
        if not requirements_info:
            requirements_info.append({
                'type': RequirementType.BUSINESS,
                'title': "General system requirement",
                'description': text,
                'priority': Priority.MEDIUM
            })
        
        return requirements_info
    
    def _infer_priority_from_modal(self, modal_verb: str) -> Priority:
        """Infer priority from modal verbs"""
        modal_priorities = {
            'must': Priority.CRITICAL,
            'shall': Priority.HIGH,
            'should': Priority.MEDIUM,
            'may': Priority.LOW,
            'could': Priority.NICE_TO_HAVE
        }
        return modal_priorities.get(modal_verb.lower(), Priority.MEDIUM)
    
    def _create_requirement_from_info(self, info: Dict, context: Dict) -> Optional[Requirement]:
        """Create requirement object from extracted information"""
        try:
            # Generate unique ID
            content = f"{info['title']}{info['description']}"
            req_id = f"REQ-{hashlib.md5(content.encode()).hexdigest()[:8].upper()}"
            
            # Generate acceptance criteria
            acceptance_criteria = self._generate_acceptance_criteria(info)
            
            requirement = Requirement(
                id=req_id,
                title=info['title'],
                description=info['description'],
                type=info['type'],
                priority=info['priority'],
                stakeholder=context.get('stakeholder', 'System User'),
                acceptance_criteria=acceptance_criteria,
                business_value=self._estimate_business_value(info),
                effort_estimate=self._estimate_effort(info),
                risk_level=self._assess_risk(info)
            )
            
            return requirement
            
        except Exception as e:
            self.logger.error(f"Error creating requirement: {e}")
            return None
    
    def _generate_acceptance_criteria(self, info: Dict) -> List[str]:
        """Generate acceptance criteria for requirement"""
        criteria = []
        
        if info['type'] == RequirementType.FUNCTIONAL:
            criteria.extend([
                f"Given a user with appropriate permissions",
                f"When they {info['description'].lower()}",
                f"Then the system responds appropriately",
                f"And all business rules are enforced"
            ])
        elif info['type'] == RequirementType.PERFORMANCE:
            criteria.extend([
                f"Given normal system load",
                f"When {info['description'].lower()}",
                f"Then response time is within acceptable limits",
                f"And system resources are efficiently utilized"
            ])
        elif info['type'] == RequirementType.SECURITY:
            criteria.extend([
                f"Given security requirements",
                f"When {info['description'].lower()}",
                f"Then all data is protected",
                f"And access controls are enforced"
            ])
        else:
            criteria.extend([
                f"Given the requirement for {info['title'].lower()}",
                f"When implemented",
                f"Then {info['description'].lower()}",
                f"And quality standards are met"
            ])
        
        return criteria
    
    def _estimate_business_value(self, info: Dict) -> int:
        """Estimate business value (1-10 scale)"""
        value_keywords = {
            'revenue': 9, 'profit': 9, 'cost': 8, 'efficiency': 7,
            'productivity': 7, 'customer': 8, 'user': 6, 'automation': 6,
            'security': 8, 'compliance': 8, 'performance': 6, 'scalability': 7
        }
        
        description_lower = info['description'].lower()
        max_value = 5  # default
        
        for keyword, value in value_keywords.items():
            if keyword in description_lower:
                max_value = max(max_value, value)
        
        return max_value
    
    def _estimate_effort(self, info: Dict) -> int:
        """Estimate effort in story points"""
        complexity_indicators = {
            'integration': 8, 'api': 6, 'database': 5, 'algorithm': 8,
            'ui': 4, 'report': 3, 'simple': 2, 'complex': 10,
            'migration': 8, 'security': 6, 'performance': 7
        }
        
        description_lower = info['description'].lower()
        base_effort = 3  # default
        
        for indicator, effort in complexity_indicators.items():
            if indicator in description_lower:
                base_effort = max(base_effort, effort)
        
        # Type-based adjustment
        type_multipliers = {
            RequirementType.TECHNICAL: 1.2,
            RequirementType.SECURITY: 1.3,
            RequirementType.PERFORMANCE: 1.4,
            RequirementType.COMPLIANCE: 1.1,
            RequirementType.FUNCTIONAL: 1.0,
            RequirementType.USABILITY: 0.8
        }
        
        multiplier = type_multipliers.get(info['type'], 1.0)
        return int(base_effort * multiplier)
    
    def _assess_risk(self, info: Dict) -> str:
        """Assess implementation risk level"""
        high_risk_keywords = ['integration', 'migration', 'security', 'performance', 'complex']
        medium_risk_keywords = ['new', 'change', 'update', 'modify', 'extend']
        
        description_lower = info['description'].lower()
        
        if any(keyword in description_lower for keyword in high_risk_keywords):
            return "high"
        elif any(keyword in description_lower for keyword in medium_risk_keywords):
            return "medium"
        else:
            return "low"
    
    def generate_requirements_document(self) -> str:
        """Generate formatted requirements document"""
        doc = []
        doc.append("# Software Requirements Specification\n")
        doc.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        doc.append("---\n")
        
        # Group requirements by type
        req_by_type = {}
        for req in self.requirements.values():
            req_type = req.type.value
            if req_type not in req_by_type:
                req_by_type[req_type] = []
            req_by_type[req_type].append(req)
        
        # Sort each type by priority
        for req_type in req_by_type:
            req_by_type[req_type].sort(key=lambda r: r.priority.value, reverse=True)
        
        # Generate document sections
        for req_type, reqs in req_by_type.items():
            doc.append(f"\n## {req_type.replace('_', ' ').title()} Requirements\n")
            
            for req in reqs:
                doc.append(f"### {req.id}: {req.title}\n")
                doc.append(f"**Priority:** {req.priority.name}\n")
                doc.append(f"**Stakeholder:** {req.stakeholder}\n")
                doc.append(f"**Description:** {req.description}\n")
                
                if req.acceptance_criteria:
                    doc.append("**Acceptance Criteria:**\n")
                    for criteria in req.acceptance_criteria:
                        doc.append(f"- {criteria}\n")
                
                if req.dependencies:
                    doc.append(f"**Dependencies:** {', '.join(req.dependencies)}\n")
                
                if req.effort_estimate:
                    doc.append(f"**Effort Estimate:** {req.effort_estimate} story points\n")
                
                if req.business_value:
                    doc.append(f"**Business Value:** {req.business_value}/10\n")
                
                if req.risk_level:
                    doc.append(f"**Risk Level:** {req.risk_level.title()}\n")
                
                doc.append("\n")
        
        return "".join(doc)
    
    def export_requirements(self, format: str = 'json') -> str:
        """Export requirements in various formats"""
        if format.lower() == 'json':
            req_dict = {req_id: req.to_dict() for req_id, req in self.requirements.items()}
            return json.dumps(req_dict, indent=2, default=str)
        elif format.lower() == 'markdown':
            return self.generate_requirements_document()
        else:
            raise ValueError(f"Unsupported export format: {format}")
    
    def validate_requirements(self) -> List[Dict]:
        """Validate requirements for completeness and consistency"""
        issues = []
        
        for req_id, req in self.requirements.items():
            # Check for completeness
            if not req.title.strip():
                issues.append({
                    'id': req_id,
                    'type': 'error',
                    'message': 'Missing requirement title'
                })
            
            if not req.description.strip():
                issues.append({
                    'id': req_id,
                    'type': 'error', 
                    'message': 'Missing requirement description'
                })
            
            if not req.acceptance_criteria:
                issues.append({
                    'id': req_id,
                    'type': 'warning',
                    'message': 'Missing acceptance criteria'
                })
            
            # Check for circular dependencies
            if req_id in req.dependencies:
                issues.append({
                    'id': req_id,
                    'type': 'error',
                    'message': 'Circular dependency detected'
                })
            
            # Check for orphaned dependencies
            for dep_id in req.dependencies:
                if dep_id not in self.requirements:
                    issues.append({
                        'id': req_id,
                        'type': 'warning',
                        'message': f'Dependency {dep_id} not found'
                    })
        
        return issues
    
    def get_requirements_by_priority(self, priority: Priority) -> List[Requirement]:
        """Get requirements filtered by priority"""
        return [req for req in self.requirements.values() if req.priority == priority]
    
    def get_requirements_by_stakeholder(self, stakeholder: str) -> List[Requirement]:
        """Get requirements filtered by stakeholder"""
        return [req for req in self.requirements.values() if req.stakeholder == stakeholder]
    
    def update_requirement(self, req_id: str, updates: Dict) -> bool:
        """Update an existing requirement"""
        if req_id not in self.requirements:
            return False
        
        req = self.requirements[req_id]
        
        for field, value in updates.items():
            if hasattr(req, field):
                setattr(req, field, value)
        
        req.updated_at = datetime.now()
        self.logger.info(f"Updated requirement {req_id}")
        return True