"""Natural language interface for task specification and guidance."""

import re
import json
import time
import numpy as np
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from enum import Enum

from ..core.base_task import BaseTask
from ..utils.logging_config import get_logger

logger = get_logger(__name__)


class InstructionType(Enum):
    """Types of instructions in natural language."""
    COMMAND = "command"
    DESCRIPTION = "description" 
    CONSTRAINT = "constraint"
    GOAL = "goal"
    SEQUENCE = "sequence"


@dataclass
class ParsedInstruction:
    """Parsed instruction from natural language."""
    instruction_type: InstructionType
    action: str
    objects: List[str]
    parameters: Dict[str, Any]
    constraints: List[str]
    priority: int = 1
    temporal_order: int = 0


class TaskParser:
    """Parses natural language task descriptions into structured representations."""
    
    def __init__(self, language: str = "en"):
        """Initialize task parser.
        
        Args:
            language: Language code for parsing
        """
        self.language = language
        self.action_patterns = self._load_action_patterns()
        self.object_patterns = self._load_object_patterns()
        self.constraint_patterns = self._load_constraint_patterns()
        
    def _load_action_patterns(self) -> Dict[str, List[str]]:
        """Load action recognition patterns."""
        return {
            "manipulation": [
                r"pick\s+up|grab|grasp|take|lift",
                r"place|put|set|position|drop",
                r"move|relocate|transfer|shift",
                r"assemble|build|construct|attach|connect",
                r"rotate|turn|spin|orient",
                r"push|pull|drag|slide"
            ],
            "navigation": [
                r"go\s+to|navigate\s+to|move\s+to|walk\s+to",
                r"follow|pursue|track",
                r"explore|search|scan|patrol",
                r"avoid|evade|dodge|circumvent"
            ],
            "observation": [
                r"look\s+at|observe|inspect|examine|check",
                r"find|locate|identify|detect|spot",
                r"monitor|watch|track|survey"
            ],
            "communication": [
                r"tell|inform|notify|report|communicate",
                r"ask|request|query|inquire",
                r"coordinate|synchronize|collaborate"
            ]
        }
    
    def _load_object_patterns(self) -> Dict[str, List[str]]:
        """Load object recognition patterns."""
        return {
            "furniture": [
                r"table|desk|chair|shelf|cabinet|drawer",
                r"bed|couch|sofa|bench|stool"
            ],
            "tools": [
                r"screwdriver|hammer|wrench|pliers|drill",
                r"knife|scissors|saw|ruler|measuring\s+tape"
            ],
            "containers": [
                r"box|container|bin|basket|bag|pouch",
                r"cup|glass|bowl|plate|tray"
            ],
            "parts": [
                r"screw|bolt|nut|washer|nail|pin",
                r"leg|top|side|back|front|base",
                r"component|piece|part|element"
            ],
            "locations": [
                r"kitchen|bedroom|living\s+room|office|garage",
                r"corner|center|edge|surface|floor|ceiling"
            ]
        }
    
    def _load_constraint_patterns(self) -> List[str]:
        """Load constraint recognition patterns."""
        return [
            r"carefully|gently|slowly|precisely|exactly",
            r"quickly|fast|rapidly|immediately|urgently",
            r"without\s+touching|avoiding|not\s+damaging",
            r"making\s+sure|ensuring|verifying|checking",
            r"before|after|while|during|simultaneously",
            r"stable|secure|tight|loose|aligned"
        ]
    
    def parse_task_description(self, description: str) -> List[ParsedInstruction]:
        """Parse natural language task description into structured instructions.
        
        Args:
            description: Natural language task description
            
        Returns:
            List of parsed instructions
        """
        # Preprocess text
        text = description.lower().strip()
        sentences = self._split_sentences(text)
        
        instructions = []
        for i, sentence in enumerate(sentences):
            instruction = self._parse_sentence(sentence, i)
            if instruction:
                instructions.append(instruction)
        
        # Post-process: resolve temporal relationships
        instructions = self._resolve_temporal_order(instructions)
        
        logger.info(f"Parsed {len(instructions)} instructions from task description")
        return instructions
    
    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        # Simple sentence splitting (could be enhanced with NLP library)
        sentences = re.split(r'[.!?]+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def _parse_sentence(self, sentence: str, order: int) -> Optional[ParsedInstruction]:
        """Parse individual sentence into instruction.
        
        Args:
            sentence: Sentence to parse
            order: Sentence order in text
            
        Returns:
            Parsed instruction or None
        """
        # Extract action
        action = self._extract_action(sentence)
        if not action:
            return None
        
        # Extract objects
        objects = self._extract_objects(sentence)
        
        # Extract parameters (numbers, measurements, etc.)
        parameters = self._extract_parameters(sentence)
        
        # Extract constraints
        constraints = self._extract_constraints(sentence)
        
        # Determine instruction type
        instruction_type = self._classify_instruction_type(sentence, action)
        
        # Determine priority based on keywords
        priority = self._determine_priority(sentence)
        
        return ParsedInstruction(
            instruction_type=instruction_type,
            action=action,
            objects=objects,
            parameters=parameters,
            constraints=constraints,
            priority=priority,
            temporal_order=order
        )
    
    def _extract_action(self, sentence: str) -> Optional[str]:
        """Extract action verb from sentence."""
        for action_category, patterns in self.action_patterns.items():
            for pattern in patterns:
                match = re.search(pattern, sentence)
                if match:
                    return match.group(0)
        return None
    
    def _extract_objects(self, sentence: str) -> List[str]:
        """Extract object references from sentence."""
        objects = []
        for object_category, patterns in self.object_patterns.items():
            for pattern in patterns:
                matches = re.finditer(pattern, sentence)
                for match in matches:
                    objects.append(match.group(0))
        
        # Also extract color/size adjectives with objects
        adjective_object_pattern = r'(red|blue|green|yellow|black|white|small|large|big|tiny)\s+(\w+)'
        matches = re.finditer(adjective_object_pattern, sentence)
        for match in matches:
            objects.append(f"{match.group(1)} {match.group(2)}")
        
        return list(set(objects))  # Remove duplicates
    
    def _extract_parameters(self, sentence: str) -> Dict[str, Any]:
        """Extract numerical parameters from sentence."""
        parameters = {}
        
        # Extract measurements
        measurement_pattern = r'(\d+(?:\.\d+)?)\s*(cm|mm|m|inch|ft|degrees?|kg|g|lbs?)'
        matches = re.finditer(measurement_pattern, sentence)
        for match in matches:
            value = float(match.group(1))
            unit = match.group(2)
            parameters[f"measurement_{unit}"] = value
        
        # Extract quantities
        quantity_pattern = r'(\d+)\s*(times?|pieces?|parts?|items?)'
        matches = re.finditer(quantity_pattern, sentence)
        for match in matches:
            quantity = int(match.group(1))
            parameters["quantity"] = quantity
        
        # Extract force/pressure specifications
        force_pattern = r'(\d+(?:\.\d+)?)\s*(newtons?|n|pounds?|force)'
        matches = re.finditer(force_pattern, sentence)
        for match in matches:
            force = float(match.group(1))
            parameters["force"] = force
        
        return parameters
    
    def _extract_constraints(self, sentence: str) -> List[str]:
        """Extract constraints from sentence."""
        constraints = []
        for pattern in self.constraint_patterns:
            matches = re.finditer(pattern, sentence)
            for match in matches:
                constraints.append(match.group(0))
        
        return constraints
    
    def _classify_instruction_type(self, sentence: str, action: str) -> InstructionType:
        """Classify the type of instruction."""
        if any(word in sentence for word in ["must", "should", "ensure", "make sure"]):
            return InstructionType.CONSTRAINT
        elif any(word in sentence for word in ["goal", "objective", "target", "achieve"]):
            return InstructionType.GOAL
        elif any(word in sentence for word in ["first", "then", "next", "finally", "after", "before"]):
            return InstructionType.SEQUENCE
        elif action:
            return InstructionType.COMMAND
        else:
            return InstructionType.DESCRIPTION
    
    def _determine_priority(self, sentence: str) -> int:
        """Determine instruction priority based on keywords."""
        high_priority_words = ["urgent", "critical", "important", "must", "immediately"]
        medium_priority_words = ["should", "need", "require", "preferably"]
        
        if any(word in sentence for word in high_priority_words):
            return 3
        elif any(word in sentence for word in medium_priority_words):
            return 2
        else:
            return 1
    
    def _resolve_temporal_order(self, instructions: List[ParsedInstruction]) -> List[ParsedInstruction]:
        """Resolve temporal relationships between instructions."""
        # Simple temporal ordering based on keywords
        sequence_keywords = {
            "first": 0, "initially": 0, "start": 0, "begin": 0,
            "then": 1, "next": 1, "after": 1, "subsequently": 1,
            "finally": 2, "last": 2, "end": 2, "complete": 2
        }
        
        for instruction in instructions:
            for constraint in instruction.constraints:
                for keyword, order_modifier in sequence_keywords.items():
                    if keyword in constraint:
                        instruction.temporal_order = instruction.temporal_order * 10 + order_modifier
                        break
        
        return sorted(instructions, key=lambda x: x.temporal_order)


class InstructionGenerator:
    """Generates natural language guidance and feedback for agents."""
    
    def __init__(self, language: str = "en"):
        """Initialize instruction generator.
        
        Args:
            language: Language code for generation
        """
        self.language = language
        self.templates = self._load_templates()
        
    def _load_templates(self) -> Dict[str, Dict[str, List[str]]]:
        """Load instruction templates for different contexts."""
        return {
            "guidance": {
                "manipulation": [
                    "Pick up the {object} carefully",
                    "Place the {object} on the {target}",
                    "Rotate the {object} to align with {target}",
                    "Connect the {object1} to the {object2}",
                    "Apply {force}N of force to secure the connection"
                ],
                "navigation": [
                    "Navigate to the {location}",
                    "Move towards the {target} while avoiding obstacles",
                    "Follow the path to reach {destination}",
                    "Explore the area to find {object}"
                ],
                "coordination": [
                    "Wait for {agent} to complete their task",
                    "Coordinate with {agent} to lift the {object}",
                    "Communicate your status to the team",
                    "Synchronize your actions with {agent}"
                ]
            },
            "feedback": {
                "success": [
                    "Excellent! The task was completed successfully.",
                    "Great job! All objectives have been achieved.",
                    "Perfect execution! The {object} is now properly {action}."
                ],
                "progress": [
                    "Good progress! You've completed {percentage}% of the task.",
                    "You're on the right track. Continue with the {next_action}.",
                    "The {object} is now positioned correctly. Proceed to the next step."
                ],
                "correction": [
                    "Careful! The {object} is not properly aligned.",
                    "You need to adjust the {parameter} to {value}.",
                    "Try approaching the {object} from a different angle.",
                    "The force applied is too {high_low}. Adjust to {target_force}N."
                ],
                "error": [
                    "There was an issue with {object}. Please retry the {action}.",
                    "Collision detected! Move away from the {obstacle}.",
                    "The {object} appears to be stuck. Try a different approach."
                ]
            }
        }
    
    def generate_step_guidance(self, 
                             current_state: Dict[str, Any],
                             next_action: str,
                             context: Dict[str, Any]) -> str:
        """Generate step-by-step guidance for current situation.
        
        Args:
            current_state: Current environment state
            next_action: Recommended next action
            context: Task context information
            
        Returns:
            Natural language guidance string
        """
        # Determine action category
        action_category = self._categorize_action(next_action)
        
        # Select appropriate template
        templates = self.templates["guidance"].get(action_category, ["Continue with the task"])
        template = np.random.choice(templates)
        
        # Fill template with context
        guidance = self._fill_template(template, context, current_state)
        
        return guidance
    
    def generate_feedback(self, 
                         performance_data: Dict[str, Any],
                         feedback_type: str = "progress") -> str:
        """Generate performance feedback.
        
        Args:
            performance_data: Performance metrics and data
            feedback_type: Type of feedback (success, progress, correction, error)
            
        Returns:
            Natural language feedback string
        """
        templates = self.templates["feedback"].get(feedback_type, ["Keep going!"])
        template = np.random.choice(templates)
        
        # Fill template with performance data
        feedback = self._fill_template(template, performance_data, {})
        
        return feedback
    
    def _categorize_action(self, action: str) -> str:
        """Categorize action for template selection."""
        action_lower = action.lower()
        
        if any(word in action_lower for word in ["pick", "place", "move", "rotate", "connect"]):
            return "manipulation"
        elif any(word in action_lower for word in ["navigate", "go", "walk", "follow"]):
            return "navigation"
        elif any(word in action_lower for word in ["coordinate", "communicate", "wait", "sync"]):
            return "coordination"
        else:
            return "general"
    
    def _fill_template(self, 
                      template: str, 
                      context: Dict[str, Any], 
                      state: Dict[str, Any]) -> str:
        """Fill template with context variables.
        
        Args:
            template: Template string with placeholders
            context: Context variables
            state: Current state variables
            
        Returns:
            Filled template string
        """
        # Combine context and state
        variables = {**context, **state}
        
        # Fill placeholders
        filled_template = template
        for key, value in variables.items():
            placeholder = "{" + key + "}"
            if placeholder in filled_template:
                filled_template = filled_template.replace(placeholder, str(value))
        
        # Handle any remaining unfilled placeholders with defaults
        remaining_placeholders = re.findall(r'\{(\w+)\}', filled_template)
        for placeholder in remaining_placeholders:
            default_value = self._get_default_value(placeholder)
            filled_template = filled_template.replace(f"{{{placeholder}}}", default_value)
        
        return filled_template
    
    def _get_default_value(self, placeholder: str) -> str:
        """Get default value for unfilled placeholder."""
        defaults = {
            "object": "item",
            "target": "destination", 
            "location": "position",
            "agent": "teammate",
            "action": "action",
            "force": "10",
            "percentage": "50",
            "next_action": "next step"
        }
        return defaults.get(placeholder, placeholder)


class LanguageTaskInterface:
    """Main interface for natural language task interaction."""
    
    def __init__(self, language: str = "en"):
        """Initialize language task interface.
        
        Args:
            language: Language code
        """
        self.language = language
        self.parser = TaskParser(language)
        self.generator = InstructionGenerator(language)
        self.conversation_history = []
        
    def parse_task(self, 
                   task_description: str,
                   available_objects: Optional[List[str]] = None,
                   available_tools: Optional[List[str]] = None) -> Dict[str, Any]:
        """Parse natural language task description into executable format.
        
        Args:
            task_description: Natural language task description
            available_objects: List of available objects in environment
            available_tools: List of available tools
            
        Returns:
            Parsed task configuration dictionary
        """
        # Parse instructions
        instructions = self.parser.parse_task_description(task_description)
        
        # Validate against available objects/tools
        if available_objects or available_tools:
            instructions = self._validate_instructions(
                instructions, available_objects or [], available_tools or []
            )
        
        # Convert to task configuration
        task_config = self._instructions_to_config(instructions)
        
        # Store in conversation history
        self.conversation_history.append({
            "timestamp": time.time(),
            "type": "task_parse",
            "input": task_description,
            "output": task_config,
            "instructions": [self._instruction_to_dict(inst) for inst in instructions]
        })
        
        return task_config
    
    def get_guidance(self, 
                    task: BaseTask,
                    current_state: Dict[str, Any],
                    agent_capabilities: Optional[Dict[str, Any]] = None) -> str:
        """Get natural language guidance for current situation.
        
        Args:
            task: Current task instance
            current_state: Current environment state
            agent_capabilities: Agent capabilities for personalization
            
        Returns:
            Natural language guidance string
        """
        # Analyze current state and task progress
        progress_analysis = self._analyze_progress(task, current_state)
        
        # Get quantum-inspired plan if available
        next_actions = []
        if hasattr(task, 'get_quantum_plan'):
            next_actions = task.get_quantum_plan()
        
        # Generate context-aware guidance
        if next_actions:
            next_action = next_actions[0]
        else:
            next_action = "continue"
        
        context = {
            "task_name": task.name,
            "progress": progress_analysis["completion_percentage"],
            "next_action": next_action,
            "objects": progress_analysis.get("visible_objects", []),
            "agent": agent_capabilities.get("name", "agent") if agent_capabilities else "agent"
        }
        
        guidance = self.generator.generate_step_guidance(
            current_state, next_action, context
        )
        
        # Store in conversation history
        self.conversation_history.append({
            "timestamp": time.time(),
            "type": "guidance",
            "context": context,
            "guidance": guidance
        })
        
        return guidance
    
    def provide_feedback(self, 
                        performance_data: Dict[str, Any],
                        task_status: str = "in_progress") -> str:
        """Provide natural language performance feedback.
        
        Args:
            performance_data: Performance metrics and data
            task_status: Current task status
            
        Returns:
            Natural language feedback string
        """
        # Determine feedback type based on performance and status
        feedback_type = self._determine_feedback_type(performance_data, task_status)
        
        # Generate contextual feedback
        feedback = self.generator.generate_feedback(performance_data, feedback_type)
        
        # Store in conversation history
        self.conversation_history.append({
            "timestamp": time.time(),
            "type": "feedback",
            "performance_data": performance_data,
            "task_status": task_status,
            "feedback": feedback
        })
        
        return feedback
    
    def _validate_instructions(self, 
                             instructions: List[ParsedInstruction],
                             available_objects: List[str],
                             available_tools: List[str]) -> List[ParsedInstruction]:
        """Validate instructions against available objects and tools."""
        validated_instructions = []
        
        for instruction in instructions:
            # Check if required objects are available
            missing_objects = []
            for obj in instruction.objects:
                if obj not in available_objects and obj not in available_tools:
                    missing_objects.append(obj)
            
            if missing_objects:
                logger.warning(f"Instruction references unavailable objects: {missing_objects}")
                # Could substitute similar objects or mark as warnings
                
            validated_instructions.append(instruction)
        
        return validated_instructions
    
    def _instructions_to_config(self, instructions: List[ParsedInstruction]) -> Dict[str, Any]:
        """Convert parsed instructions to task configuration."""
        config = {
            "name": "language_specified_task",
            "instruction_count": len(instructions),
            "actions": [],
            "objects": set(),
            "constraints": [],
            "priorities": [],
            "sequence": True if any(i.instruction_type == InstructionType.SEQUENCE for i in instructions) else False
        }
        
        for instruction in instructions:
            config["actions"].append({
                "action": instruction.action,
                "type": instruction.instruction_type.value,
                "objects": instruction.objects,
                "parameters": instruction.parameters,
                "constraints": instruction.constraints,
                "priority": instruction.priority,
                "order": instruction.temporal_order
            })
            
            config["objects"].update(instruction.objects)
            config["constraints"].extend(instruction.constraints)
            config["priorities"].append(instruction.priority)
        
        # Convert set to list for JSON serialization
        config["objects"] = list(config["objects"])
        
        return config
    
    def _instruction_to_dict(self, instruction: ParsedInstruction) -> Dict[str, Any]:
        """Convert instruction to dictionary for serialization."""
        return {
            "type": instruction.instruction_type.value,
            "action": instruction.action,
            "objects": instruction.objects,
            "parameters": instruction.parameters,
            "constraints": instruction.constraints,
            "priority": instruction.priority,
            "temporal_order": instruction.temporal_order
        }
    
    def _analyze_progress(self, task: BaseTask, current_state: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze task progress from current state."""
        analysis = {
            "completion_percentage": 0,
            "visible_objects": [],
            "current_step": getattr(task, 'current_step', 0),
            "total_steps": getattr(task, 'max_steps', 1000)
        }
        
        # Calculate completion percentage
        if hasattr(task, 'current_step') and hasattr(task, 'max_steps'):
            analysis["completion_percentage"] = int(
                (task.current_step / task.max_steps) * 100
            )
        
        # Extract visible objects from state
        if "objects" in current_state:
            analysis["visible_objects"] = list(current_state["objects"].keys())
        elif "parts" in current_state:
            analysis["visible_objects"] = list(current_state["parts"].keys())
        
        return analysis
    
    def _determine_feedback_type(self, 
                                performance_data: Dict[str, Any], 
                                task_status: str) -> str:
        """Determine appropriate feedback type."""
        if task_status == "completed" and performance_data.get("success", False):
            return "success"
        elif "error" in performance_data or "collision" in performance_data:
            return "error"
        elif "precision_error" in performance_data or "alignment_error" in performance_data:
            return "correction"
        else:
            return "progress"
    
    def get_conversation_history(self) -> List[Dict[str, Any]]:
        """Get conversation history."""
        return self.conversation_history.copy()
    
    def clear_conversation_history(self):
        """Clear conversation history."""
        self.conversation_history = []