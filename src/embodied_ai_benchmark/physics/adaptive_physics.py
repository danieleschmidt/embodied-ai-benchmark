"""Adaptive physics parameter learning for sim-to-real transfer."""

import time
from typing import Any, Dict, List, Optional, Tuple, Callable
import numpy as np
from datetime import datetime
import logging
from collections import deque, defaultdict
import json

from ..utils.logging_config import get_logger
from .physics_config import PhysicsConfig

logger = get_logger(__name__)


class ParameterDistribution:
    """Represents a learnable physics parameter with uncertainty."""
    
    def __init__(self, initial_value: float, min_value: float, max_value: float, 
                 uncertainty: float = 0.1, name: str = "param"):
        """Initialize parameter distribution.
        
        Args:
            initial_value: Initial parameter value
            min_value: Minimum allowed value
            max_value: Maximum allowed value
            uncertainty: Initial uncertainty (standard deviation)
            name: Parameter name for logging
        """
        self.name = name
        self.mean = initial_value
        self.std = uncertainty
        self.min_value = min_value
        self.max_value = max_value
        self.update_history = deque(maxlen=1000)
        self.initial_value = initial_value
    
    def sample(self) -> float:
        """Sample a value from the parameter distribution.
        
        Returns:
            Sampled parameter value
        """
        value = np.random.normal(self.mean, self.std)
        return np.clip(value, self.min_value, self.max_value)
    
    def update(self, gradient: float, learning_rate: float = 0.01):
        """Update parameter based on gradient.
        
        Args:
            gradient: Gradient for parameter update
            learning_rate: Learning rate for update
        """
        old_mean = self.mean
        self.mean = np.clip(
            self.mean + learning_rate * gradient,
            self.min_value, self.max_value
        )
        
        # Adaptive uncertainty update
        self.std = max(0.01, self.std * 0.999)  # Slowly reduce uncertainty
        
        self.update_history.append({
            "timestamp": time.time(),
            "old_value": old_mean,
            "new_value": self.mean,
            "gradient": gradient,
            "learning_rate": learning_rate
        })
    
    def get_info(self) -> Dict[str, Any]:
        """Get parameter information.
        
        Returns:
            Parameter info dictionary
        """
        return {
            "name": self.name,
            "mean": self.mean,
            "std": self.std,
            "min_value": self.min_value,
            "max_value": self.max_value,
            "initial_value": self.initial_value,
            "total_updates": len(self.update_history),
            "relative_change": abs(self.mean - self.initial_value) / max(abs(self.initial_value), 1e-6)
        }


class MaterialPropertyLearner:
    """Learns material properties from real-world feedback."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize material property learner.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.materials = {}
        self._initialize_default_materials()
        self.learning_rate = self.config.get("learning_rate", 0.01)
        self.adaptation_history = []
    
    def _initialize_default_materials(self):
        """Initialize default material parameters."""
        default_materials = {
            "wood": {
                "density": ParameterDistribution(700, 400, 1000, 50, "wood_density"),
                "friction": ParameterDistribution(0.6, 0.2, 0.9, 0.1, "wood_friction"),
                "restitution": ParameterDistribution(0.1, 0.0, 0.3, 0.05, "wood_restitution"),
                "youngs_modulus": ParameterDistribution(1e10, 5e9, 2e10, 1e9, "wood_youngs_modulus")
            },
            "metal": {
                "density": ParameterDistribution(7850, 6000, 10000, 500, "metal_density"),
                "friction": ParameterDistribution(0.4, 0.1, 0.8, 0.1, "metal_friction"),
                "restitution": ParameterDistribution(0.05, 0.0, 0.2, 0.02, "metal_restitution"),
                "youngs_modulus": ParameterDistribution(2e11, 1e11, 3e11, 2e10, "metal_youngs_modulus")
            },
            "fabric": {
                "density": ParameterDistribution(100, 50, 200, 20, "fabric_density"),
                "stretch_stiffness": ParameterDistribution(0.1, 0.01, 0.5, 0.05, "fabric_stretch_stiffness"),
                "bend_stiffness": ParameterDistribution(0.01, 0.001, 0.1, 0.005, "fabric_bend_stiffness"),
                "friction": ParameterDistribution(0.7, 0.3, 1.0, 0.1, "fabric_friction")
            },
            "plastic": {
                "density": ParameterDistribution(1000, 800, 1500, 100, "plastic_density"),
                "friction": ParameterDistribution(0.3, 0.1, 0.6, 0.1, "plastic_friction"),
                "restitution": ParameterDistribution(0.4, 0.1, 0.8, 0.1, "plastic_restitution"),
                "youngs_modulus": ParameterDistribution(3e9, 1e9, 5e9, 5e8, "plastic_youngs_modulus")
            }
        }
        
        self.materials = default_materials
    
    def get_material_properties(self, material_name: str, sample: bool = True) -> Dict[str, float]:
        """Get material properties, either sampled or mean values.
        
        Args:
            material_name: Name of the material
            sample: Whether to sample from distributions or use mean values
            
        Returns:
            Dictionary of material properties
        """
        if material_name not in self.materials:
            logger.warning(f"Unknown material {material_name}, using default")
            material_name = "wood"  # Default fallback
        
        material = self.materials[material_name]
        properties = {}
        
        for prop_name, param_dist in material.items():
            if sample:
                properties[prop_name] = param_dist.sample()
            else:
                properties[prop_name] = param_dist.mean
        
        return properties
    
    def update_material_properties(
        self, 
        material_name: str, 
        property_gradients: Dict[str, float]
    ):
        """Update material properties based on gradients.
        
        Args:
            material_name: Name of the material to update
            property_gradients: Gradients for each property
        """
        if material_name not in self.materials:
            logger.warning(f"Cannot update unknown material {material_name}")
            return
        
        material = self.materials[material_name]
        
        for prop_name, gradient in property_gradients.items():
            if prop_name in material:
                material[prop_name].update(gradient, self.learning_rate)
                logger.debug(f"Updated {material_name}.{prop_name}: gradient={gradient:.6f}")
    
    def get_all_material_info(self) -> Dict[str, Dict[str, Any]]:
        """Get information about all materials.
        
        Returns:
            Dictionary of material information
        """
        info = {}
        for material_name, material in self.materials.items():
            info[material_name] = {}
            for prop_name, param_dist in material.items():
                info[material_name][prop_name] = param_dist.get_info()
        
        return info


class AdaptivePhysicsLearner:
    """Learn and adapt physics parameters for better sim-to-real transfer."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize adaptive physics learner.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.material_learner = MaterialPropertyLearner(config.get("material_learner", {}))
        
        # Global physics parameters
        self.global_params = {
            "gravity": ParameterDistribution(9.81, 8.0, 12.0, 0.5, "gravity"),
            "air_density": ParameterDistribution(1.225, 0.5, 2.0, 0.1, "air_density"),
            "timestep_multiplier": ParameterDistribution(1.0, 0.5, 2.0, 0.1, "timestep_multiplier")
        }
        
        # Contact model parameters
        self.contact_params = {
            "contact_stiffness": ParameterDistribution(1e6, 1e4, 1e8, 1e5, "contact_stiffness"),
            "contact_damping": ParameterDistribution(1000, 100, 10000, 500, "contact_damping"),
            "friction_cone_angle": ParameterDistribution(0.5, 0.1, 1.0, 0.1, "friction_cone_angle"),
            "restitution_threshold": ParameterDistribution(0.01, 0.001, 0.1, 0.005, "restitution_threshold")
        }
        
        self.adaptation_history = deque(maxlen=10000)
        self.real_world_feedback = []
        self.learning_rate = self.config.get("learning_rate", 0.01)
        self.gradient_estimator = GradientEstimator()
    
    def get_current_physics_config(self, sample: bool = True) -> Dict[str, Any]:
        """Get current physics configuration.
        
        Args:
            sample: Whether to sample from parameter distributions
            
        Returns:
            Physics configuration dictionary
        """
        config = {}
        
        # Global parameters
        config["global"] = {}
        for param_name, param_dist in self.global_params.items():
            config["global"][param_name] = param_dist.sample() if sample else param_dist.mean
        
        # Contact parameters
        config["contact"] = {}
        for param_name, param_dist in self.contact_params.items():
            config["contact"][param_name] = param_dist.sample() if sample else param_dist.mean
        
        # Material parameters
        config["materials"] = {}
        for material_name in self.material_learner.materials.keys():
            config["materials"][material_name] = self.material_learner.get_material_properties(
                material_name, sample
            )
        
        return config
    
    def adapt_physics_from_real_feedback(
        self, 
        real_world_outcomes: List[Dict[str, Any]], 
        sim_predictions: List[Dict[str, Any]],
        task_context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Continuously adapt physics parameters based on real-world feedback.
        
        Args:
            real_world_outcomes: List of real-world task outcomes
            sim_predictions: List of corresponding simulation predictions
            task_context: Context information about the task
            
        Returns:
            Updated physics parameters and adaptation statistics
        """
        logger.info(f"Adapting physics from {len(real_world_outcomes)} real-world samples")
        
        if len(real_world_outcomes) != len(sim_predictions):
            raise ValueError("Number of real outcomes must match number of sim predictions")
        
        # Compute physics discrepancy
        discrepancies = []
        for real_outcome, sim_prediction in zip(real_world_outcomes, sim_predictions):
            discrepancy = self.compute_physics_discrepancy(real_outcome, sim_prediction)
            discrepancies.append(discrepancy)
        
        # Estimate parameter gradients
        parameter_gradients = self.gradient_estimator.estimate_gradients(
            discrepancies, self.get_current_physics_config(sample=False)
        )
        
        # Update parameters
        adaptation_results = self._update_physics_parameters(parameter_gradients)
        
        # Record adaptation
        adaptation_record = {
            "timestamp": datetime.now().isoformat(),
            "num_samples": len(real_world_outcomes),
            "mean_discrepancy": np.mean([d["total_discrepancy"] for d in discrepancies]),
            "parameter_updates": parameter_gradients,
            "task_context": task_context or {},
            "adaptation_results": adaptation_results
        }
        
        self.adaptation_history.append(adaptation_record)
        self.real_world_feedback.extend(real_world_outcomes)
        
        logger.info(f"Physics adaptation completed. Mean discrepancy: {adaptation_record['mean_discrepancy']:.4f}")
        
        return {
            "updated_params": self.get_current_physics_config(sample=False),
            "adaptation_stats": adaptation_results,
            "discrepancy_reduction": self._calculate_discrepancy_reduction()
        }
    
    def compute_physics_discrepancy(
        self, 
        real_outcome: Dict[str, Any], 
        sim_prediction: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Compute discrepancy between real and simulated outcomes.
        
        Args:
            real_outcome: Real-world task outcome
            sim_prediction: Simulation prediction
            
        Returns:
            Discrepancy analysis
        """
        discrepancy = {
            "position_error": 0.0,
            "velocity_error": 0.0,
            "force_error": 0.0,
            "contact_error": 0.0,
            "trajectory_error": 0.0,
            "success_mismatch": 0.0,
            "total_discrepancy": 0.0
        }
        
        # Position discrepancy
        if "final_positions" in real_outcome and "final_positions" in sim_prediction:
            real_pos = np.array(real_outcome["final_positions"])
            sim_pos = np.array(sim_prediction["final_positions"])
            if real_pos.shape == sim_pos.shape:
                discrepancy["position_error"] = np.linalg.norm(real_pos - sim_pos)
        
        # Velocity discrepancy
        if "final_velocities" in real_outcome and "final_velocities" in sim_prediction:
            real_vel = np.array(real_outcome["final_velocities"])
            sim_vel = np.array(sim_prediction["final_velocities"])
            if real_vel.shape == sim_vel.shape:
                discrepancy["velocity_error"] = np.linalg.norm(real_vel - sim_vel)
        
        # Force discrepancy
        if "contact_forces" in real_outcome and "contact_forces" in sim_prediction:
            real_forces = np.array(real_outcome["contact_forces"])
            sim_forces = np.array(sim_prediction["contact_forces"])
            if real_forces.shape == sim_forces.shape:
                discrepancy["force_error"] = np.mean(np.abs(real_forces - sim_forces))
        
        # Contact timing discrepancy
        if "contact_times" in real_outcome and "contact_times" in sim_prediction:
            real_contacts = set(real_outcome["contact_times"])
            sim_contacts = set(sim_prediction["contact_times"])
            contact_overlap = len(real_contacts.intersection(sim_contacts))
            contact_union = len(real_contacts.union(sim_contacts))
            if contact_union > 0:
                discrepancy["contact_error"] = 1.0 - (contact_overlap / contact_union)
        
        # Trajectory discrepancy
        if "trajectory" in real_outcome and "trajectory" in sim_prediction:
            real_traj = np.array(real_outcome["trajectory"])
            sim_traj = np.array(sim_prediction["trajectory"])
            if real_traj.shape == sim_traj.shape:
                discrepancy["trajectory_error"] = np.mean(np.linalg.norm(real_traj - sim_traj, axis=-1))
        
        # Success mismatch
        real_success = real_outcome.get("success", False)
        sim_success = sim_prediction.get("success", False)
        discrepancy["success_mismatch"] = float(real_success != sim_success)
        
        # Calculate weighted total discrepancy
        weights = {
            "position_error": 0.25,
            "velocity_error": 0.15,
            "force_error": 0.20,
            "contact_error": 0.15,
            "trajectory_error": 0.15,
            "success_mismatch": 0.10
        }
        
        total = sum(weights[key] * discrepancy[key] for key in weights.keys())
        discrepancy["total_discrepancy"] = total
        
        return discrepancy
    
    def _update_physics_parameters(self, gradients: Dict[str, Any]) -> Dict[str, Any]:
        """Update physics parameters using computed gradients.
        
        Args:
            gradients: Parameter gradients
            
        Returns:
            Update statistics
        """
        update_stats = {
            "global_updates": 0,
            "contact_updates": 0,
            "material_updates": 0,
            "total_updates": 0
        }
        
        # Update global parameters
        if "global" in gradients:
            for param_name, gradient in gradients["global"].items():
                if param_name in self.global_params:
                    self.global_params[param_name].update(gradient, self.learning_rate)
                    update_stats["global_updates"] += 1
        
        # Update contact parameters
        if "contact" in gradients:
            for param_name, gradient in gradients["contact"].items():
                if param_name in self.contact_params:
                    self.contact_params[param_name].update(gradient, self.learning_rate)
                    update_stats["contact_updates"] += 1
        
        # Update material parameters
        if "materials" in gradients:
            for material_name, material_gradients in gradients["materials"].items():
                self.material_learner.update_material_properties(material_name, material_gradients)
                update_stats["material_updates"] += len(material_gradients)
        
        update_stats["total_updates"] = (
            update_stats["global_updates"] + 
            update_stats["contact_updates"] + 
            update_stats["material_updates"]
        )
        
        return update_stats
    
    def _calculate_discrepancy_reduction(self) -> float:
        """Calculate discrepancy reduction over recent adaptations.
        
        Returns:
            Discrepancy reduction percentage
        """
        if len(self.adaptation_history) < 2:
            return 0.0
        
        recent_records = list(self.adaptation_history)[-10:]  # Last 10 adaptations
        
        if len(recent_records) < 2:
            return 0.0
        
        initial_discrepancy = recent_records[0]["mean_discrepancy"]
        final_discrepancy = recent_records[-1]["mean_discrepancy"]
        
        if initial_discrepancy == 0:
            return 0.0
        
        reduction = (initial_discrepancy - final_discrepancy) / initial_discrepancy
        return max(0.0, reduction * 100.0)  # Return percentage
    
    def get_adaptation_statistics(self) -> Dict[str, Any]:
        """Get statistics about physics adaptation.
        
        Returns:
            Adaptation statistics
        """
        if not self.adaptation_history:
            return {"total_adaptations": 0}
        
        adaptations = list(self.adaptation_history)
        recent_adaptations = adaptations[-100:]  # Last 100
        
        stats = {
            "total_adaptations": len(adaptations),
            "total_real_samples": len(self.real_world_feedback),
            "recent_mean_discrepancy": np.mean([
                a["mean_discrepancy"] for a in recent_adaptations
            ]) if recent_adaptations else 0,
            "discrepancy_reduction": self._calculate_discrepancy_reduction(),
            "parameter_update_counts": self._get_parameter_update_counts(),
            "material_adaptation_stats": self.material_learner.get_all_material_info()
        }
        
        # Add convergence indicators
        if len(recent_adaptations) >= 10:
            recent_discrepancies = [a["mean_discrepancy"] for a in recent_adaptations[-10:]]
            stats["discrepancy_trend"] = np.polyfit(range(10), recent_discrepancies, 1)[0]
            stats["discrepancy_stability"] = np.std(recent_discrepancies)
        
        return stats
    
    def _get_parameter_update_counts(self) -> Dict[str, int]:
        """Get counts of parameter updates by category.
        
        Returns:
            Update counts by parameter category
        """
        counts = defaultdict(int)
        
        # Global parameters
        for param in self.global_params.values():
            counts["global"] += len(param.update_history)
        
        # Contact parameters
        for param in self.contact_params.values():
            counts["contact"] += len(param.update_history)
        
        # Material parameters
        for material in self.material_learner.materials.values():
            for param in material.values():
                counts["materials"] += len(param.update_history)
        
        return dict(counts)
    
    def save_adaptation_data(self, filepath: str):
        """Save adaptation data to file.
        
        Args:
            filepath: Output file path
        """
        adaptation_data = {
            "adaptation_history": list(self.adaptation_history),
            "current_parameters": self.get_current_physics_config(sample=False),
            "statistics": self.get_adaptation_statistics(),
            "material_info": self.material_learner.get_all_material_info(),
            "export_timestamp": datetime.now().isoformat()
        }
        
        # Convert numpy types for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, dict):
                return {key: convert_numpy(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(item) for item in obj]
            else:
                return obj
        
        serializable_data = convert_numpy(adaptation_data)
        
        with open(filepath, 'w') as f:
            json.dump(serializable_data, f, indent=2)
        
        logger.info(f"Adaptation data saved to {filepath}")
    
    def load_adaptation_data(self, filepath: str):
        """Load adaptation data from file.
        
        Args:
            filepath: Input file path
        """
        with open(filepath, 'r') as f:
            adaptation_data = json.load(f)
        
        # Restore adaptation history
        self.adaptation_history.extend(adaptation_data.get("adaptation_history", []))
        
        # Note: Parameter restoration would require more complex deserialization
        # of ParameterDistribution objects, which is omitted for brevity
        
        logger.info(f"Adaptation data loaded from {filepath}")


class GradientEstimator:
    """Estimates parameter gradients from physics discrepancies."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize gradient estimator.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.finite_difference_epsilon = self.config.get("finite_diff_eps", 1e-4)
    
    def estimate_gradients(
        self, 
        discrepancies: List[Dict[str, Any]], 
        current_params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Estimate parameter gradients from discrepancies.
        
        Args:
            discrepancies: List of physics discrepancies
            current_params: Current parameter values
            
        Returns:
            Estimated gradients for each parameter
        """
        if not discrepancies:
            return {}
        
        # Use average discrepancy as loss signal
        mean_discrepancy = np.mean([d["total_discrepancy"] for d in discrepancies])
        
        # Estimate gradients using finite differences (simplified)
        # In practice, this would use more sophisticated methods like
        # automatic differentiation or learned sensitivity analysis
        
        gradients = {
            "global": {},
            "contact": {},
            "materials": defaultdict(dict)
        }
        
        # Simplified gradient estimation based on discrepancy types
        # This is a heuristic approach - real implementation would be more sophisticated
        
        avg_discrepancy = {}
        for d in discrepancies:
            for key, value in d.items():
                if key not in avg_discrepancy:
                    avg_discrepancy[key] = []
                avg_discrepancy[key].append(value)
        
        for key in avg_discrepancy:
            avg_discrepancy[key] = np.mean(avg_discrepancy[key])
        
        # Global parameter gradients
        if avg_discrepancy.get("position_error", 0) > 0.1:
            gradients["global"]["gravity"] = -avg_discrepancy["position_error"] * 0.1
        
        # Contact parameter gradients
        if avg_discrepancy.get("contact_error", 0) > 0.1:
            gradients["contact"]["contact_stiffness"] = avg_discrepancy["contact_error"] * 1000
            gradients["contact"]["contact_damping"] = avg_discrepancy["contact_error"] * 100
        
        # Material parameter gradients (simplified)
        if avg_discrepancy.get("force_error", 0) > 0.1:
            for material in ["wood", "metal", "fabric", "plastic"]:
                if material not in gradients["materials"]:
                    gradients["materials"][material] = {}
                gradients["materials"][material]["friction"] = -avg_discrepancy["force_error"] * 0.01
        
        return gradients
