"""Real-time Adaptive Physics Engine with Neural Acceleration.

Novel contributions:
1. GPU-accelerated parallel contact resolution
2. Learned physics approximations for real-time performance  
3. Adaptive level-of-detail based on importance and error
4. Multi-fidelity simulation with automatic switching
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
import numpy as np
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import time
import math
from collections import deque, defaultdict
from concurrent.futures import ThreadPoolExecutor
import threading
from abc import ABC, abstractmethod

from ..utils.logging_config import get_logger

logger = get_logger(__name__)


class SimulationFidelity(Enum):
    """Simulation fidelity levels."""
    ULTRA_LOW = "ultra_low"    # Neural approximation only
    LOW = "low"                # Simplified physics + neural
    MEDIUM = "medium"          # Standard physics with shortcuts  
    HIGH = "high"              # Full physics simulation
    ULTRA_HIGH = "ultra_high"  # Maximum accuracy physics


@dataclass
class PhysicsObject:
    """Physics object representation."""
    object_id: str
    position: torch.Tensor      # [3] position
    velocity: torch.Tensor      # [3] velocity
    orientation: torch.Tensor   # [4] quaternion
    angular_velocity: torch.Tensor  # [3] angular velocity
    mass: float
    inertia_tensor: torch.Tensor    # [3, 3] inertia tensor
    bounding_box: torch.Tensor      # [6] min_x, max_x, min_y, max_y, min_z, max_z
    material_properties: Dict[str, float] = field(default_factory=dict)
    importance_score: float = 1.0
    last_update_time: float = 0.0
    prediction_error: float = 0.0
    
    
@dataclass 
class ContactConstraint:
    """Contact constraint between objects."""
    object_a_id: str
    object_b_id: str
    contact_point: torch.Tensor     # [3] world space contact point
    contact_normal: torch.Tensor    # [3] contact normal (from A to B)
    penetration_depth: float
    friction_coefficient: float
    restitution_coefficient: float
    contact_impulse: torch.Tensor = field(default_factory=lambda: torch.zeros(3))


@dataclass
class SimulationState:
    """Complete physics simulation state."""
    objects: Dict[str, PhysicsObject]
    constraints: List[ContactConstraint]
    global_forces: torch.Tensor     # [3] gravity, etc.
    simulation_time: float
    fidelity_level: SimulationFidelity
    performance_metrics: Dict[str, float] = field(default_factory=dict)


class NeuralPhysicsPredictor(nn.Module):
    """Neural network for fast physics prediction."""
    
    def __init__(self,
                 max_objects: int = 100,
                 state_dim: int = 13,  # pos(3) + vel(3) + quat(4) + angvel(3)
                 hidden_dim: int = 512,
                 num_layers: int = 4,
                 use_attention: bool = True):
        super().__init__()
        self.max_objects = max_objects
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        self.use_attention = use_attention
        
        # Object state encoder
        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim + 6, hidden_dim),  # +6 for bounding box
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Inter-object attention mechanism
        if use_attention:
            self.object_attention = nn.MultiheadAttention(
                embed_dim=hidden_dim,
                num_heads=8,
                batch_first=True,
                dropout=0.1
            )
        
        # Physics dynamics predictor
        self.dynamics_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, state_dim)  # Predict state change
        )
        
        # Contact detection network
        self.contact_detector = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 4)  # contact_prob, normal(3)
        )
        
        # Uncertainty estimation
        self.uncertainty_estimator = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, 1),
            nn.Sigmoid()
        )
        
    def forward(self, 
                object_states: torch.Tensor,
                dt: float,
                global_forces: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Predict physics state evolution.
        
        Args:
            object_states: [batch_size, num_objects, state_dim + 6] object states
            dt: Timestep size
            global_forces: [3] global forces (gravity, etc.)
            
        Returns:
            state_changes: [batch_size, num_objects, state_dim] predicted state changes
            contact_predictions: [batch_size, num_objects, num_objects, 4] contact predictions
            uncertainties: [batch_size, num_objects, 1] prediction uncertainties
        """
        batch_size, num_objects, _ = object_states.shape
        
        # Encode object states
        encoded_states = self.state_encoder(object_states)  # [batch, num_objects, hidden_dim]
        
        # Apply inter-object attention
        if self.use_attention:
            attended_states, attention_weights = self.object_attention(
                encoded_states, encoded_states, encoded_states
            )
            # Residual connection
            encoded_states = encoded_states + attended_states
        
        # Predict state changes
        state_changes = self.dynamics_predictor(encoded_states)
        
        # Scale by timestep
        state_changes = state_changes * dt
        
        # Add global forces to velocity changes
        gravity_effect = global_forces.unsqueeze(0).unsqueeze(0).expand(batch_size, num_objects, 3)
        state_changes[:, :, 3:6] += gravity_effect * dt  # Velocity components
        
        # Predict pairwise contacts
        contact_predictions = torch.zeros(batch_size, num_objects, num_objects, 4)
        for i in range(num_objects):
            for j in range(i + 1, num_objects):
                # Concatenate features for object pair
                pair_features = torch.cat([encoded_states[:, i], encoded_states[:, j]], dim=-1)
                contact_pred = self.contact_detector(pair_features)
                contact_predictions[:, i, j] = contact_pred
                contact_predictions[:, j, i] = contact_pred  # Symmetric
        
        # Estimate uncertainties
        uncertainties = self.uncertainty_estimator(encoded_states)
        
        return state_changes, contact_predictions, uncertainties


class AdaptiveLevelOfDetail:
    """Adaptive level-of-detail system for physics simulation."""
    
    def __init__(self,
                 importance_threshold: float = 0.5,
                 error_threshold: float = 0.1,
                 update_frequency: int = 10):
        self.importance_threshold = importance_threshold
        self.error_threshold = error_threshold
        self.update_frequency = update_frequency
        self.update_counter = 0
        
        # Tracking
        self.object_importance_history = defaultdict(deque)
        self.prediction_error_history = defaultdict(deque)
        self.fidelity_assignments = {}
        
    def update_importance_scores(self, 
                                objects: Dict[str, PhysicsObject],
                                camera_position: torch.Tensor,
                                agent_focus_objects: List[str]) -> Dict[str, float]:
        """Update importance scores for all objects."""
        importance_scores = {}
        
        for obj_id, obj in objects.items():
            # Distance-based importance
            distance = torch.norm(obj.position - camera_position)
            distance_importance = 1.0 / (1.0 + distance)
            
            # Agent focus importance
            agent_importance = 2.0 if obj_id in agent_focus_objects else 1.0
            
            # Velocity-based importance (fast-moving objects are more important)
            velocity_magnitude = torch.norm(obj.velocity)
            velocity_importance = 1.0 + torch.tanh(velocity_magnitude)
            
            # Interaction importance (objects with recent contacts)
            interaction_importance = 1.0 + obj.prediction_error
            
            # Combined importance
            total_importance = (
                distance_importance * 0.3 +
                agent_importance * 0.4 +
                velocity_importance * 0.2 +
                interaction_importance * 0.1
            )
            
            importance_scores[obj_id] = float(total_importance)
            
            # Update history
            self.object_importance_history[obj_id].append(total_importance)
            if len(self.object_importance_history[obj_id]) > 100:
                self.object_importance_history[obj_id].popleft()
        
        return importance_scores
    
    def assign_fidelity_levels(self, 
                              objects: Dict[str, PhysicsObject],
                              importance_scores: Dict[str, float],
                              computational_budget: float) -> Dict[str, SimulationFidelity]:
        """Assign fidelity levels based on importance and computational budget."""
        # Sort objects by importance
        sorted_objects = sorted(importance_scores.items(), key=lambda x: x[1], reverse=True)
        
        fidelity_assignments = {}
        remaining_budget = computational_budget
        
        # Fidelity costs (relative)
        fidelity_costs = {
            SimulationFidelity.ULTRA_LOW: 0.1,
            SimulationFidelity.LOW: 0.3,
            SimulationFidelity.MEDIUM: 1.0,
            SimulationFidelity.HIGH: 3.0,
            SimulationFidelity.ULTRA_HIGH: 10.0
        }
        
        for obj_id, importance in sorted_objects:
            obj = objects[obj_id]
            
            # Determine appropriate fidelity based on importance and error
            if importance > 0.8 and obj.prediction_error > self.error_threshold:
                desired_fidelity = SimulationFidelity.HIGH
            elif importance > 0.6:
                desired_fidelity = SimulationFidelity.MEDIUM
            elif importance > 0.3:
                desired_fidelity = SimulationFidelity.LOW
            else:
                desired_fidelity = SimulationFidelity.ULTRA_LOW
            
            # Check if we can afford this fidelity
            cost = fidelity_costs[desired_fidelity]
            if remaining_budget >= cost:
                fidelity_assignments[obj_id] = desired_fidelity
                remaining_budget -= cost
            else:
                # Assign lower fidelity that we can afford
                for fidelity in [SimulationFidelity.MEDIUM, SimulationFidelity.LOW, SimulationFidelity.ULTRA_LOW]:
                    if remaining_budget >= fidelity_costs[fidelity]:
                        fidelity_assignments[obj_id] = fidelity
                        remaining_budget -= fidelity_costs[fidelity]
                        break
                else:
                    fidelity_assignments[obj_id] = SimulationFidelity.ULTRA_LOW
        
        self.fidelity_assignments = fidelity_assignments
        return fidelity_assignments
    
    def update_prediction_errors(self, 
                                actual_states: Dict[str, PhysicsObject],
                                predicted_states: Dict[str, PhysicsObject]):
        """Update prediction errors for each object."""
        for obj_id in actual_states:
            if obj_id in predicted_states:
                actual_pos = actual_states[obj_id].position
                predicted_pos = predicted_states[obj_id].position
                
                position_error = torch.norm(actual_pos - predicted_pos).item()
                
                # Update object's error
                actual_states[obj_id].prediction_error = position_error
                
                # Update history
                self.prediction_error_history[obj_id].append(position_error)
                if len(self.prediction_error_history[obj_id]) > 50:
                    self.prediction_error_history[obj_id].popleft()


class ParallelContactResolver:
    """GPU-accelerated parallel contact resolution."""
    
    def __init__(self, device: str = "cuda"):
        self.device = device
        self.max_iterations = 10
        self.convergence_threshold = 1e-6
        
    def resolve_contacts_parallel(self,
                                 objects: Dict[str, PhysicsObject],
                                 contacts: List[ContactConstraint],
                                 dt: float) -> Dict[str, PhysicsObject]:
        """Resolve all contacts in parallel using GPU acceleration."""
        if not contacts:
            return objects
        
        # Convert to tensors for parallel processing
        object_ids = list(objects.keys())
        id_to_idx = {obj_id: i for i, obj_id in enumerate(object_ids)}
        
        # Pack object data
        positions = torch.stack([objects[obj_id].position for obj_id in object_ids]).to(self.device)
        velocities = torch.stack([objects[obj_id].velocity for obj_id in object_ids]).to(self.device)
        masses = torch.tensor([objects[obj_id].mass for obj_id in object_ids]).to(self.device)
        
        # Pack contact data
        contact_pairs = torch.tensor(
            [[id_to_idx[c.object_a_id], id_to_idx[c.object_b_id]] for c in contacts]
        ).to(self.device)
        contact_points = torch.stack([c.contact_point for c in contacts]).to(self.device)
        contact_normals = torch.stack([c.contact_normal for c in contacts]).to(self.device)
        penetration_depths = torch.tensor([c.penetration_depth for c in contacts]).to(self.device)
        friction_coeffs = torch.tensor([c.friction_coefficient for c in contacts]).to(self.device)
        restitution_coeffs = torch.tensor([c.restitution_coefficient for c in contacts]).to(self.device)
        
        # Parallel contact resolution iterations
        impulses = torch.zeros(len(contacts), 3).to(self.device)
        
        for iteration in range(self.max_iterations):
            old_impulses = impulses.clone()
            
            # Calculate relative velocities at contact points
            obj_a_indices = contact_pairs[:, 0]
            obj_b_indices = contact_pairs[:, 1]
            
            vel_a = velocities[obj_a_indices]
            vel_b = velocities[obj_b_indices]
            rel_velocity = vel_a - vel_b
            
            # Project onto contact normal
            normal_velocity = torch.sum(rel_velocity * contact_normals, dim=1)
            
            # Calculate impulse magnitude for separation
            mass_a = masses[obj_a_indices]
            mass_b = masses[obj_b_indices]
            reduced_mass = (mass_a * mass_b) / (mass_a + mass_b + 1e-8)
            
            # Impulse to resolve penetration and velocity
            target_velocity = -restitution_coeffs * normal_velocity
            velocity_change = target_velocity - normal_velocity
            
            impulse_magnitude = reduced_mass * velocity_change / dt
            
            # Apply position correction for penetration
            position_correction = 0.2 * penetration_depths  # Baumgarte stabilization
            impulse_magnitude += reduced_mass * position_correction / (dt * dt)
            
            # Update impulses
            impulses = impulse_magnitude.unsqueeze(1) * contact_normals
            
            # Apply impulses to velocities
            velocity_changes_a = impulses / mass_a.unsqueeze(1)
            velocity_changes_b = -impulses / mass_b.unsqueeze(1)
            
            # Scatter updates to velocity tensor
            velocities = velocities.scatter_add(0, obj_a_indices.unsqueeze(1).expand(-1, 3), velocity_changes_a)
            velocities = velocities.scatter_add(0, obj_b_indices.unsqueeze(1).expand(-1, 3), velocity_changes_b)
            
            # Check convergence
            impulse_change = torch.norm(impulses - old_impulses)
            if impulse_change < self.convergence_threshold:
                break
        
        # Apply friction impulses
        tangential_velocities = rel_velocity - normal_velocity.unsqueeze(1) * contact_normals
        tangential_magnitude = torch.norm(tangential_velocities, dim=1)
        
        friction_impulses = torch.zeros_like(impulses)
        valid_friction = tangential_magnitude > 1e-6
        
        if valid_friction.any():
            friction_directions = tangential_velocities[valid_friction] / tangential_magnitude[valid_friction].unsqueeze(1)
            max_friction_impulse = friction_coeffs[valid_friction] * torch.norm(impulses[valid_friction], dim=1)
            friction_impulse_magnitude = torch.clamp(
                reduced_mass[valid_friction] * tangential_magnitude[valid_friction] / dt,
                max=max_friction_impulse
            )
            friction_impulses[valid_friction] = -friction_impulse_magnitude.unsqueeze(1) * friction_directions
            
            # Apply friction impulses
            friction_changes_a = friction_impulses / mass_a.unsqueeze(1)
            friction_changes_b = -friction_impulses / mass_b.unsqueeze(1)
            
            velocities = velocities.scatter_add(0, obj_a_indices.unsqueeze(1).expand(-1, 3), friction_changes_a)
            velocities = velocities.scatter_add(0, obj_b_indices.unsqueeze(1).expand(-1, 3), friction_changes_b)
        
        # Update object velocities
        updated_objects = {}
        for i, obj_id in enumerate(object_ids):
            updated_obj = objects[obj_id]
            updated_obj.velocity = velocities[i].cpu()
            updated_objects[obj_id] = updated_obj
        
        return updated_objects


class RealTimeAdaptivePhysicsEngine:
    """Main real-time adaptive physics engine."""
    
    def __init__(self,
                 target_fps: float = 60.0,
                 use_neural_acceleration: bool = True,
                 device: str = "cuda" if torch.cuda.is_available() else "cpu",
                 adaptive_lod: bool = True):
        """
        Initialize real-time adaptive physics engine.
        
        Args:
            target_fps: Target simulation frame rate
            use_neural_acceleration: Whether to use neural physics prediction
            device: Computation device (cuda/cpu)
            adaptive_lod: Whether to use adaptive level of detail
        """
        self.target_fps = target_fps
        self.target_dt = 1.0 / target_fps
        self.use_neural_acceleration = use_neural_acceleration
        self.device = device
        self.adaptive_lod = adaptive_lod
        
        # Initialize components
        if use_neural_acceleration:
            self.neural_predictor = NeuralPhysicsPredictor().to(device)
            self.scaler = GradScaler()  # For mixed precision
        
        if adaptive_lod:
            self.lod_system = AdaptiveLevelOfDetail()
        
        self.contact_resolver = ParallelContactResolver(device)
        
        # State management
        self.current_state = SimulationState(
            objects={},
            constraints=[],
            global_forces=torch.tensor([0.0, 0.0, -9.81]),  # Gravity
            simulation_time=0.0,
            fidelity_level=SimulationFidelity.MEDIUM
        )
        
        # Performance tracking
        self.frame_times = deque(maxlen=60)  # Track last 60 frames
        self.prediction_accuracies = deque(maxlen=100)
        self.computational_budget = 1.0  # Available computation per frame
        
        # Multi-threading
        self.executor = ThreadPoolExecutor(max_workers=4)
        self.neural_prediction_cache = {}
        self.cache_lock = threading.Lock()
        
        # Adaptive timestep
        self.adaptive_timestep = True
        self.min_dt = 1.0 / 1000.0  # 1ms minimum
        self.max_dt = 1.0 / 30.0    # 33ms maximum
        
    def add_object(self, physics_object: PhysicsObject):
        """Add object to simulation."""
        self.current_state.objects[physics_object.object_id] = physics_object
        logger.debug(f"Added object {physics_object.object_id} to simulation")
    
    def remove_object(self, object_id: str):
        """Remove object from simulation."""
        if object_id in self.current_state.objects:
            del self.current_state.objects[object_id]
            # Remove related constraints
            self.current_state.constraints = [
                c for c in self.current_state.constraints
                if c.object_a_id != object_id and c.object_b_id != object_id
            ]
            logger.debug(f"Removed object {object_id} from simulation")
    
    def step(self, 
             dt: Optional[float] = None,
             external_forces: Optional[Dict[str, torch.Tensor]] = None) -> SimulationState:
        """
        Advance simulation by one timestep.
        
        Args:
            dt: Timestep size (auto-calculated if None)
            external_forces: External forces to apply {object_id: force_vector}
            
        Returns:
            Updated simulation state
        """
        frame_start_time = time.time()
        
        # Determine timestep
        if dt is None:
            dt = self._calculate_adaptive_timestep()
        else:
            dt = max(self.min_dt, min(self.max_dt, dt))
        
        # Update importance scores and fidelity assignments
        if self.adaptive_lod:
            self._update_adaptive_lod()
        
        # Detect contacts
        contacts = self._detect_contacts()
        self.current_state.constraints = contacts
        
        # Choose simulation method based on performance requirements
        if self._should_use_neural_prediction():
            # Use neural prediction for speed
            updated_objects = self._neural_physics_step(dt, external_forces)
        else:
            # Use traditional physics for accuracy
            updated_objects = self._traditional_physics_step(dt, external_forces)
        
        # Resolve contacts
        if contacts:
            updated_objects = self.contact_resolver.resolve_contacts_parallel(
                updated_objects, contacts, dt
            )
        
        # Update simulation state
        self.current_state.objects = updated_objects
        self.current_state.simulation_time += dt
        
        # Performance tracking
        frame_time = time.time() - frame_start_time
        self.frame_times.append(frame_time)
        
        # Update computational budget
        self._update_computational_budget(frame_time)
        
        # Update performance metrics
        self.current_state.performance_metrics = {
            'frame_time': frame_time,
            'fps': 1.0 / frame_time if frame_time > 0 else float('inf'),
            'avg_frame_time': np.mean(self.frame_times) if self.frame_times else 0.0,
            'avg_fps': 1.0 / np.mean(self.frame_times) if self.frame_times and np.mean(self.frame_times) > 0 else 0.0,
            'num_objects': len(self.current_state.objects),
            'num_contacts': len(contacts),
            'computational_budget': self.computational_budget,
            'fidelity_level': self.current_state.fidelity_level.value
        }
        
        return self.current_state
    
    def _calculate_adaptive_timestep(self) -> float:
        """Calculate adaptive timestep based on performance."""
        if not self.adaptive_timestep or not self.frame_times:
            return self.target_dt
        
        avg_frame_time = np.mean(self.frame_times)
        
        # If we're running faster than target, we can afford larger timesteps
        if avg_frame_time < self.target_dt * 0.8:
            dt_multiplier = 1.1
        # If we're running slower, reduce timestep
        elif avg_frame_time > self.target_dt * 1.2:
            dt_multiplier = 0.9
        else:
            dt_multiplier = 1.0
        
        current_dt = self.target_dt
        adaptive_dt = current_dt * dt_multiplier
        
        return max(self.min_dt, min(self.max_dt, adaptive_dt))
    
    def _update_adaptive_lod(self):
        """Update adaptive level of detail assignments."""
        if not self.adaptive_lod or not self.current_state.objects:
            return
        
        # Dummy camera position (would come from renderer)
        camera_position = torch.tensor([0.0, 0.0, 5.0])
        
        # Dummy agent focus (would come from task/agent)
        agent_focus_objects = list(self.current_state.objects.keys())[:3]
        
        # Update importance scores
        importance_scores = self.lod_system.update_importance_scores(
            self.current_state.objects, camera_position, agent_focus_objects
        )
        
        # Assign fidelity levels
        fidelity_assignments = self.lod_system.assign_fidelity_levels(
            self.current_state.objects, importance_scores, self.computational_budget
        )
        
        # Update object importance scores
        for obj_id, score in importance_scores.items():
            if obj_id in self.current_state.objects:
                self.current_state.objects[obj_id].importance_score = score
    
    def _should_use_neural_prediction(self) -> bool:
        """Decide whether to use neural prediction or traditional physics."""
        if not self.use_neural_acceleration:
            return False
        
        # Use neural prediction if we're behind target framerate
        if self.frame_times:
            avg_frame_time = np.mean(self.frame_times)
            return avg_frame_time > self.target_dt * 1.1
        
        return False
    
    def _neural_physics_step(self, 
                            dt: float,
                            external_forces: Optional[Dict[str, torch.Tensor]] = None) -> Dict[str, PhysicsObject]:
        """Perform physics step using neural prediction."""
        if not self.current_state.objects:
            return {}
        
        # Prepare input data
        object_ids = list(self.current_state.objects.keys())
        object_states = []
        
        for obj_id in object_ids:
            obj = self.current_state.objects[obj_id]
            state_vector = torch.cat([
                obj.position,
                obj.velocity,
                obj.orientation,
                obj.angular_velocity,
                obj.bounding_box
            ])
            object_states.append(state_vector)
        
        if not object_states:
            return {}
        
        object_states_tensor = torch.stack(object_states).unsqueeze(0).to(self.device)
        global_forces = self.current_state.global_forces.to(self.device)
        
        # Neural prediction with mixed precision
        with autocast():
            with torch.no_grad():
                state_changes, contact_predictions, uncertainties = self.neural_predictor(
                    object_states_tensor, dt, global_forces
                )
        
        # Apply predictions
        updated_objects = {}
        for i, obj_id in enumerate(object_ids):
            obj = self.current_state.objects[obj_id]
            state_change = state_changes[0, i].cpu()
            uncertainty = uncertainties[0, i].cpu().item()
            
            # Apply state changes
            new_obj = PhysicsObject(
                object_id=obj.object_id,
                position=obj.position + state_change[:3],
                velocity=obj.velocity + state_change[3:6],
                orientation=obj.orientation + state_change[6:10],
                angular_velocity=obj.angular_velocity + state_change[10:13],
                mass=obj.mass,
                inertia_tensor=obj.inertia_tensor,
                bounding_box=obj.bounding_box,
                material_properties=obj.material_properties,
                importance_score=obj.importance_score,
                last_update_time=time.time(),
                prediction_error=uncertainty
            )
            
            # Normalize quaternion
            new_obj.orientation = F.normalize(new_obj.orientation, dim=0)
            
            # Apply external forces if provided
            if external_forces and obj_id in external_forces:
                force = external_forces[obj_id]
                acceleration = force / obj.mass
                new_obj.velocity += acceleration * dt
            
            updated_objects[obj_id] = new_obj
        
        return updated_objects
    
    def _traditional_physics_step(self,
                                 dt: float,
                                 external_forces: Optional[Dict[str, torch.Tensor]] = None) -> Dict[str, PhysicsObject]:
        """Perform physics step using traditional methods."""
        updated_objects = {}
        
        for obj_id, obj in self.current_state.objects.items():
            # Apply external forces
            total_force = self.current_state.global_forces.clone()
            if external_forces and obj_id in external_forces:
                total_force += external_forces[obj_id]
            
            # Integrate acceleration
            acceleration = total_force / obj.mass
            new_velocity = obj.velocity + acceleration * dt
            
            # Integrate velocity  
            new_position = obj.position + new_velocity * dt
            
            # Simple angular integration (should use proper quaternion integration)
            new_angular_velocity = obj.angular_velocity * 0.99  # Damping
            new_orientation = obj.orientation  # Simplified
            
            updated_obj = PhysicsObject(
                object_id=obj.object_id,
                position=new_position,
                velocity=new_velocity,
                orientation=new_orientation,
                angular_velocity=new_angular_velocity,
                mass=obj.mass,
                inertia_tensor=obj.inertia_tensor,
                bounding_box=obj.bounding_box,
                material_properties=obj.material_properties,
                importance_score=obj.importance_score,
                last_update_time=time.time(),
                prediction_error=0.0  # Traditional physics has no prediction error
            )
            
            updated_objects[obj_id] = updated_obj
        
        return updated_objects
    
    def _detect_contacts(self) -> List[ContactConstraint]:
        """Detect contacts between objects."""
        contacts = []
        object_list = list(self.current_state.objects.values())
        
        for i in range(len(object_list)):
            for j in range(i + 1, len(object_list)):
                obj_a, obj_b = object_list[i], object_list[j]
                
                # Simple bounding box intersection test
                if self._bounding_boxes_intersect(obj_a, obj_b):
                    contact = self._compute_contact_constraint(obj_a, obj_b)
                    if contact:
                        contacts.append(contact)
        
        return contacts
    
    def _bounding_boxes_intersect(self, obj_a: PhysicsObject, obj_b: PhysicsObject) -> bool:
        """Check if bounding boxes of two objects intersect."""
        # Extract bounding box coordinates
        a_min = obj_a.bounding_box[:3] + obj_a.position
        a_max = obj_a.bounding_box[3:] + obj_a.position
        b_min = obj_b.bounding_box[:3] + obj_b.position
        b_max = obj_b.bounding_box[3:] + obj_b.position
        
        # Check for separation along each axis
        for i in range(3):
            if a_max[i] < b_min[i] or b_max[i] < a_min[i]:
                return False
        
        return True
    
    def _compute_contact_constraint(self, obj_a: PhysicsObject, obj_b: PhysicsObject) -> Optional[ContactConstraint]:
        """Compute contact constraint between two objects."""
        # Simplified contact computation (in practice would use detailed geometry)
        center_to_center = obj_b.position - obj_a.position
        distance = torch.norm(center_to_center)
        
        # Assume sphere collision for simplicity
        radius_a = torch.norm(obj_a.bounding_box[3:] - obj_a.bounding_box[:3]) / 2
        radius_b = torch.norm(obj_b.bounding_box[3:] - obj_b.bounding_box[:3]) / 2
        
        penetration = (radius_a + radius_b) - distance
        
        if penetration > 0:
            contact_normal = center_to_center / distance if distance > 1e-6 else torch.tensor([0.0, 0.0, 1.0])
            contact_point = obj_a.position + contact_normal * radius_a
            
            # Material properties
            friction = 0.5  # Default friction
            restitution = 0.3  # Default restitution
            
            return ContactConstraint(
                object_a_id=obj_a.object_id,
                object_b_id=obj_b.object_id,
                contact_point=contact_point,
                contact_normal=contact_normal,
                penetration_depth=float(penetration),
                friction_coefficient=friction,
                restitution_coefficient=restitution
            )
        
        return None
    
    def _update_computational_budget(self, frame_time: float):
        """Update available computational budget based on performance."""
        target_frame_time = self.target_dt
        
        if frame_time < target_frame_time * 0.8:
            # We have spare time, increase budget
            self.computational_budget = min(2.0, self.computational_budget * 1.05)
        elif frame_time > target_frame_time * 1.2:
            # We're over budget, reduce
            self.computational_budget = max(0.1, self.computational_budget * 0.95)
    
    def train_neural_predictor(self,
                              training_data: List[Tuple[torch.Tensor, torch.Tensor]],
                              num_epochs: int = 100,
                              learning_rate: float = 0.001):
        """Train neural physics predictor on collected data."""
        if not self.use_neural_acceleration or not training_data:
            return
        
        optimizer = torch.optim.Adam(self.neural_predictor.parameters(), lr=learning_rate)
        criterion = nn.MSELoss()
        
        self.neural_predictor.train()
        
        for epoch in range(num_epochs):
            total_loss = 0.0
            
            for input_states, target_changes in training_data:
                input_states = input_states.to(self.device)
                target_changes = target_changes.to(self.device)
                
                optimizer.zero_grad()
                
                with autocast():
                    predicted_changes, _, _ = self.neural_predictor(
                        input_states, self.target_dt, self.current_state.global_forces.to(self.device)
                    )
                    
                    loss = criterion(predicted_changes, target_changes)
                
                self.scaler.scale(loss).backward()
                self.scaler.step(optimizer)
                self.scaler.update()
                
                total_loss += loss.item()
            
            if epoch % 20 == 0:
                avg_loss = total_loss / len(training_data)
                logger.info(f"Neural predictor training epoch {epoch}: loss = {avg_loss:.6f}")
        
        self.neural_predictor.eval()
        logger.info("Neural predictor training completed")
    
    def benchmark_performance(self, 
                             num_steps: int = 1000,
                             varying_complexity: bool = True) -> Dict[str, Any]:
        """Benchmark physics engine performance."""
        logger.info(f"Benchmarking performance for {num_steps} steps")
        
        # Add test objects
        original_objects = self.current_state.objects.copy()
        
        for i in range(20):  # 20 test objects
            test_obj = PhysicsObject(
                object_id=f"test_obj_{i}",
                position=torch.randn(3) * 2.0,
                velocity=torch.randn(3) * 0.5,
                orientation=F.normalize(torch.randn(4), dim=0),
                angular_velocity=torch.randn(3) * 0.1,
                mass=1.0 + torch.rand(1).item(),
                inertia_tensor=torch.eye(3),
                bounding_box=torch.tensor([-0.5, -0.5, -0.5, 0.5, 0.5, 0.5])
            )
            self.add_object(test_obj)
        
        # Benchmark
        frame_times = []
        neural_usage = 0
        traditional_usage = 0
        
        for step in range(num_steps):
            start_time = time.time()
            
            # Vary complexity during benchmark
            if varying_complexity and step % 100 == 0:
                # Add/remove objects to vary complexity
                if len(self.current_state.objects) < 50 and step < num_steps // 2:
                    new_obj = PhysicsObject(
                        object_id=f"dynamic_obj_{step}",
                        position=torch.randn(3) * 3.0,
                        velocity=torch.randn(3) * 1.0,
                        orientation=F.normalize(torch.randn(4), dim=0),
                        angular_velocity=torch.randn(3) * 0.2,
                        mass=0.5 + torch.rand(1).item(),
                        inertia_tensor=torch.eye(3),
                        bounding_box=torch.tensor([-0.3, -0.3, -0.3, 0.3, 0.3, 0.3])
                    )
                    self.add_object(new_obj)
            
            # Simulate step
            was_neural = self._should_use_neural_prediction()
            self.step()
            
            frame_time = time.time() - start_time
            frame_times.append(frame_time)
            
            if was_neural:
                neural_usage += 1
            else:
                traditional_usage += 1
        
        # Calculate statistics
        frame_times = np.array(frame_times)
        
        results = {
            'avg_frame_time': float(np.mean(frame_times)),
            'std_frame_time': float(np.std(frame_times)),
            'min_frame_time': float(np.min(frame_times)),
            'max_frame_time': float(np.max(frame_times)),
            'avg_fps': float(1.0 / np.mean(frame_times)),
            'neural_usage_ratio': neural_usage / num_steps,
            'traditional_usage_ratio': traditional_usage / num_steps,
            'total_objects_simulated': len(self.current_state.objects),
            'performance_score': min(60.0, 1.0 / np.mean(frame_times))  # Score capped at 60 FPS
        }
        
        # Restore original state
        self.current_state.objects = original_objects
        
        logger.info(f"Benchmark completed: {results['avg_fps']:.1f} FPS average")
        
        return results
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get comprehensive performance metrics."""
        if not self.frame_times:
            return {'status': 'no_data'}
        
        frame_times = np.array(self.frame_times)
        
        return {
            'real_time_performance': {
                'current_fps': 1.0 / frame_times[-1] if frame_times[-1] > 0 else 0.0,
                'avg_fps': 1.0 / np.mean(frame_times),
                'min_fps': 1.0 / np.max(frame_times),
                'max_fps': 1.0 / np.min(frame_times),
                'fps_stability': 1.0 / np.std(frame_times) if np.std(frame_times) > 0 else float('inf')
            },
            'computational_efficiency': {
                'budget_utilization': self.computational_budget,
                'neural_acceleration_active': self.use_neural_acceleration,
                'adaptive_lod_active': self.adaptive_lod,
                'avg_objects_per_frame': len(self.current_state.objects),
                'avg_contacts_per_frame': len(self.current_state.constraints)
            },
            'simulation_quality': {
                'fidelity_level': self.current_state.fidelity_level.value,
                'prediction_accuracy': np.mean(self.prediction_accuracies) if self.prediction_accuracies else 0.0,
                'physics_stability': 1.0 - np.var([obj.prediction_error for obj in self.current_state.objects.values()])
            }
        }
    
    def save_engine_state(self, filepath: str):
        """Save physics engine state."""
        state = {
            'current_state': {
                'objects': {obj_id: {
                    'position': obj.position.tolist(),
                    'velocity': obj.velocity.tolist(),
                    'orientation': obj.orientation.tolist(),
                    'angular_velocity': obj.angular_velocity.tolist(),
                    'mass': obj.mass,
                    'bounding_box': obj.bounding_box.tolist(),
                    'importance_score': obj.importance_score
                } for obj_id, obj in self.current_state.objects.items()},
                'simulation_time': self.current_state.simulation_time,
                'fidelity_level': self.current_state.fidelity_level.value
            },
            'config': {
                'target_fps': self.target_fps,
                'use_neural_acceleration': self.use_neural_acceleration,
                'adaptive_lod': self.adaptive_lod,
                'computational_budget': self.computational_budget
            },
            'performance_metrics': self.get_performance_metrics()
        }
        
        if self.use_neural_acceleration:
            state['neural_predictor'] = self.neural_predictor.state_dict()
        
        torch.save(state, filepath)
        logger.info(f"Saved physics engine state to {filepath}")
    
    def load_engine_state(self, filepath: str):
        """Load physics engine state."""
        state = torch.load(filepath, map_location=self.device)
        
        # Restore objects
        self.current_state.objects = {}
        for obj_id, obj_data in state['current_state']['objects'].items():
            physics_obj = PhysicsObject(
                object_id=obj_id,
                position=torch.tensor(obj_data['position']),
                velocity=torch.tensor(obj_data['velocity']),
                orientation=torch.tensor(obj_data['orientation']),
                angular_velocity=torch.tensor(obj_data['angular_velocity']),
                mass=obj_data['mass'],
                inertia_tensor=torch.eye(3),  # Default
                bounding_box=torch.tensor(obj_data['bounding_box']),
                importance_score=obj_data['importance_score']
            )
            self.current_state.objects[obj_id] = physics_obj
        
        self.current_state.simulation_time = state['current_state']['simulation_time']
        self.computational_budget = state['config']['computational_budget']
        
        if 'neural_predictor' in state and self.use_neural_acceleration:
            self.neural_predictor.load_state_dict(state['neural_predictor'])
        
        logger.info(f"Loaded physics engine state from {filepath}")