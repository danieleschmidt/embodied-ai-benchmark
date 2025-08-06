"""Advanced physics configuration and material properties."""

import numpy as np
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass, field
from enum import Enum


class PhysicsEngine(Enum):
    """Available physics engines."""
    BULLET = "bullet"
    FLEX = "flex"  
    MUJOCO = "mujoco"
    PHYSX = "physx"
    CUSTOM = "custom"


@dataclass
class MaterialProperties:
    """Physical material properties."""
    name: str
    density: float  # kg/m³
    friction_static: float = 0.6
    friction_dynamic: float = 0.4
    restitution: float = 0.1  # Bounciness (0-1)
    young_modulus: Optional[float] = None  # Pa (elasticity)
    poisson_ratio: Optional[float] = None  # Lateral strain ratio
    tensile_strength: Optional[float] = None  # Pa
    compressive_strength: Optional[float] = None  # Pa
    thermal_conductivity: Optional[float] = None  # W/m·K
    specific_heat: Optional[float] = None  # J/kg·K
    
    # Soft body properties
    stretch_stiffness: Optional[float] = None
    bend_stiffness: Optional[float] = None
    shear_stiffness: Optional[float] = None
    damping: float = 0.01
    
    # Visual properties
    color: tuple = field(default_factory=lambda: (0.7, 0.7, 0.7))
    texture: Optional[str] = None
    shininess: float = 0.5
    
    @classmethod
    def wood(cls, wood_type: str = "oak") -> 'MaterialProperties':
        """Create wood material properties."""
        wood_properties = {
            "oak": {"density": 700, "young_modulus": 11e9},
            "pine": {"density": 500, "young_modulus": 9e9},
            "birch": {"density": 600, "young_modulus": 14e9},
            "bamboo": {"density": 400, "young_modulus": 20e9}
        }
        
        props = wood_properties.get(wood_type, wood_properties["oak"])
        
        return cls(
            name=f"{wood_type}_wood",
            density=props["density"],
            friction_static=0.6,
            friction_dynamic=0.4,
            restitution=0.1,
            young_modulus=props["young_modulus"],
            poisson_ratio=0.35,
            tensile_strength=50e6,  # 50 MPa
            color=(0.6, 0.4, 0.2)
        )
    
    @classmethod  
    def metal(cls, metal_type: str = "steel") -> 'MaterialProperties':
        """Create metal material properties."""
        metal_properties = {
            "steel": {"density": 7850, "young_modulus": 200e9},
            "aluminum": {"density": 2700, "young_modulus": 69e9},
            "copper": {"density": 8960, "young_modulus": 110e9},
            "titanium": {"density": 4500, "young_modulus": 114e9}
        }
        
        props = metal_properties.get(metal_type, metal_properties["steel"])
        
        return cls(
            name=f"{metal_type}_metal",
            density=props["density"],
            friction_static=0.4,
            friction_dynamic=0.3,
            restitution=0.05,
            young_modulus=props["young_modulus"],
            poisson_ratio=0.3,
            tensile_strength=400e6,  # 400 MPa
            thermal_conductivity=50,
            color=(0.8, 0.8, 0.9)
        )
    
    @classmethod
    def fabric(cls, fabric_type: str = "cotton") -> 'MaterialProperties':
        """Create fabric material properties."""
        fabric_properties = {
            "cotton": {"density": 100, "stretch": 0.1},
            "silk": {"density": 80, "stretch": 0.15},
            "wool": {"density": 120, "stretch": 0.2},
            "denim": {"density": 150, "stretch": 0.05}
        }
        
        props = fabric_properties.get(fabric_type, fabric_properties["cotton"])
        
        return cls(
            name=f"{fabric_type}_fabric",
            density=props["density"],
            friction_static=0.7,
            friction_dynamic=0.6,
            restitution=0.02,
            stretch_stiffness=props["stretch"],
            bend_stiffness=0.01,
            shear_stiffness=0.05,
            damping=0.1,
            color=(0.5, 0.3, 0.7)
        )
    
    @classmethod
    def plastic(cls, plastic_type: str = "abs") -> 'MaterialProperties':
        """Create plastic material properties."""
        plastic_properties = {
            "abs": {"density": 1050, "young_modulus": 2.3e9},
            "pvc": {"density": 1380, "young_modulus": 3.5e9},
            "pe": {"density": 950, "young_modulus": 1.1e9},
            "pp": {"density": 900, "young_modulus": 1.5e9}
        }
        
        props = plastic_properties.get(plastic_type, plastic_properties["abs"])
        
        return cls(
            name=f"{plastic_type}_plastic",
            density=props["density"],
            friction_static=0.5,
            friction_dynamic=0.4,
            restitution=0.3,
            young_modulus=props["young_modulus"],
            poisson_ratio=0.35,
            tensile_strength=30e6,  # 30 MPa
            color=(0.2, 0.7, 0.9)
        )


class PhysicsConfig:
    """Configuration for physics simulation."""
    
    def __init__(self,
                 engine: PhysicsEngine = PhysicsEngine.BULLET,
                 gravity: Union[float, np.ndarray] = -9.81,
                 time_step: float = 0.01,
                 substeps: int = 10,
                 solver_iterations: int = 20,
                 enable_collisions: bool = True,
                 enable_friction: bool = True,
                 enable_soft_bodies: bool = False,
                 enable_fluids: bool = False):
        """Initialize physics configuration.
        
        Args:
            engine: Physics engine to use
            gravity: Gravity vector or magnitude
            time_step: Physics time step in seconds
            substeps: Number of substeps per frame
            solver_iterations: Constraint solver iterations
            enable_collisions: Enable collision detection
            enable_friction: Enable friction simulation
            enable_soft_bodies: Enable soft body dynamics
            enable_fluids: Enable fluid simulation
        """
        self.engine = engine
        self.gravity = np.array([0, gravity, 0]) if isinstance(gravity, (int, float)) else np.array(gravity)
        self.time_step = time_step
        self.substeps = substeps
        self.solver_iterations = solver_iterations
        self.enable_collisions = enable_collisions
        self.enable_friction = enable_friction
        self.enable_soft_bodies = enable_soft_bodies
        self.enable_fluids = enable_fluids
        
        # Material registry
        self.materials = {}
        self._register_default_materials()
        
        # Simulation parameters
        self.collision_margin = 0.001  # meters
        self.contact_breaking_threshold = 0.02
        self.deactivation_time = 2.0  # seconds
        self.sleep_threshold = 0.1  # m/s
        
        # Performance settings
        self.broadphase_type = "dynamic_aabb_tree"
        self.constraint_solver = "sequential_impulse"
        self.use_gpu_acceleration = False
        
        # Debugging and visualization
        self.debug_draw = False
        self.wireframe_mode = False
        self.show_contact_points = False
        self.show_constraint_limits = False
    
    def _register_default_materials(self):
        """Register default material types."""
        # Common materials
        self.materials["wood"] = MaterialProperties.wood()
        self.materials["steel"] = MaterialProperties.metal("steel")
        self.materials["aluminum"] = MaterialProperties.metal("aluminum")
        self.materials["plastic"] = MaterialProperties.plastic()
        self.materials["fabric"] = MaterialProperties.fabric()
        
        # Surface materials
        self.materials["rubber"] = MaterialProperties(
            name="rubber",
            density=1200,
            friction_static=1.2,
            friction_dynamic=1.0,
            restitution=0.8,
            color=(0.1, 0.1, 0.1)
        )
        
        self.materials["glass"] = MaterialProperties(
            name="glass",
            density=2500,
            friction_static=0.3,
            friction_dynamic=0.2,
            restitution=0.1,
            young_modulus=70e9,
            tensile_strength=50e6,
            color=(0.9, 0.9, 1.0)
        )
        
        self.materials["concrete"] = MaterialProperties(
            name="concrete",
            density=2400,
            friction_static=0.8,
            friction_dynamic=0.7,
            restitution=0.05,
            young_modulus=30e9,
            compressive_strength=30e6,
            color=(0.6, 0.6, 0.6)
        )
    
    def add_material(self, name: str, material: MaterialProperties):
        """Add custom material to registry.
        
        Args:
            name: Material name
            material: Material properties
        """
        self.materials[name] = material
    
    def get_material(self, name: str) -> Optional[MaterialProperties]:
        """Get material properties by name.
        
        Args:
            name: Material name
            
        Returns:
            Material properties or None if not found
        """
        return self.materials.get(name)
    
    def create_contact_model(self, 
                           material1: str, 
                           material2: str) -> Dict[str, float]:
        """Create contact model between two materials.
        
        Args:
            material1: First material name
            material2: Second material name
            
        Returns:
            Contact parameters dictionary
        """
        mat1 = self.get_material(material1)
        mat2 = self.get_material(material2)
        
        if not mat1 or not mat2:
            # Default contact parameters
            return {
                "friction": 0.5,
                "restitution": 0.1,
                "contact_damping": 0.1
            }
        
        # Combine material properties
        friction = np.sqrt(mat1.friction_static * mat2.friction_static)
        restitution = min(mat1.restitution, mat2.restitution)
        damping = (mat1.damping + mat2.damping) / 2
        
        return {
            "friction": friction,
            "restitution": restitution,
            "contact_damping": damping
        }
    
    def configure_for_precision_tasks(self):
        """Configure physics for high-precision manipulation tasks."""
        self.time_step = 0.001  # 1ms
        self.substeps = 20
        self.solver_iterations = 50
        self.collision_margin = 0.0005
        self.contact_breaking_threshold = 0.01
        self.deactivation_time = 0.5
    
    def configure_for_performance(self):
        """Configure physics for high-performance simulation."""
        self.time_step = 0.02  # 20ms  
        self.substeps = 5
        self.solver_iterations = 10
        self.collision_margin = 0.01
        self.use_gpu_acceleration = True
    
    def configure_for_soft_bodies(self):
        """Configure physics for soft body simulation."""
        self.enable_soft_bodies = True
        self.engine = PhysicsEngine.FLEX  # Better for soft bodies
        self.time_step = 0.005  # Smaller time step
        self.substeps = 4
        
        # Add soft body materials
        self.materials["cloth"] = MaterialProperties.fabric("cotton")
        self.materials["rope"] = MaterialProperties(
            name="rope",
            density=200,
            friction_static=0.8,
            friction_dynamic=0.7,
            stretch_stiffness=0.8,
            bend_stiffness=0.1,
            damping=0.05,
            color=(0.6, 0.4, 0.2)
        )
    
    def configure_for_fluids(self):
        """Configure physics for fluid simulation.""" 
        self.enable_fluids = True
        self.engine = PhysicsEngine.FLEX
        self.time_step = 0.002
        
        # Add fluid materials
        self.materials["water"] = MaterialProperties(
            name="water", 
            density=1000,
            friction_static=0.1,
            friction_dynamic=0.05,
            restitution=0.0,
            color=(0.2, 0.5, 1.0)
        )
        
        self.materials["oil"] = MaterialProperties(
            name="oil",
            density=800,
            friction_static=0.3,
            friction_dynamic=0.2,
            restitution=0.0,
            color=(0.8, 0.6, 0.2)
        )
    
    def get_engine_specific_config(self) -> Dict[str, Any]:
        """Get engine-specific configuration parameters.
        
        Returns:
            Engine-specific configuration dictionary
        """
        if self.engine == PhysicsEngine.BULLET:
            return {
                "collision_margin": self.collision_margin,
                "contact_breaking_threshold": self.contact_breaking_threshold,
                "deactivation_time": self.deactivation_time,
                "sleep_threshold": self.sleep_threshold,
                "broadphase": self.broadphase_type,
                "solver": self.constraint_solver
            }
        
        elif self.engine == PhysicsEngine.FLEX:
            return {
                "particle_radius": 0.005,
                "solid_rest_distance": 0.01,
                "fluid_rest_distance": 0.008,
                "num_iterations": self.solver_iterations,
                "relaxation_mode": "global",
                "collision_distance": self.collision_margin
            }
        
        elif self.engine == PhysicsEngine.MUJOCO:
            return {
                "timestep": self.time_step,
                "iterations": self.solver_iterations,
                "tolerance": 1e-8,
                "noslip_iterations": 0,
                "mpr_iterations": 50,
                "cone": "pyramidal"
            }
        
        elif self.engine == PhysicsEngine.PHYSX:
            return {
                "scene_flags": ["enable_ccd", "enable_stabilization"],
                "solver_type": "pgs",  # Projected Gauss-Seidel
                "bounce_threshold_velocity": 0.2,
                "friction_offset_threshold": 0.04,
                "ccd_max_separation": 0.04
            }
        
        else:  # Custom engine
            return {
                "gravity": self.gravity.tolist(),
                "time_step": self.time_step,
                "substeps": self.substeps,
                "solver_iterations": self.solver_iterations
            }
    
    def validate_configuration(self) -> List[str]:
        """Validate physics configuration and return warnings.
        
        Returns:
            List of validation warnings
        """
        warnings = []
        
        # Time step validation
        if self.time_step > 0.02:
            warnings.append("Large time step may cause instability")
        elif self.time_step < 0.0001:
            warnings.append("Very small time step may impact performance")
        
        # Substeps validation
        if self.substeps < 1:
            warnings.append("Substeps must be at least 1")
        elif self.substeps > 50:
            warnings.append("High substep count may impact performance")
        
        # Solver iterations validation
        if self.solver_iterations < 5:
            warnings.append("Low solver iterations may cause poor constraint solving")
        elif self.solver_iterations > 100:
            warnings.append("High solver iterations may impact performance")
        
        # Engine-specific validations
        if self.engine == PhysicsEngine.FLEX:
            if not self.enable_soft_bodies and not self.enable_fluids:
                warnings.append("FLEX engine is optimized for soft bodies/fluids")
        
        if self.enable_soft_bodies and self.engine == PhysicsEngine.BULLET:
            warnings.append("Bullet has limited soft body support, consider FLEX")
        
        if self.enable_fluids and self.engine != PhysicsEngine.FLEX:
            warnings.append("Fluid simulation requires FLEX engine")
        
        return warnings
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary.
        
        Returns:
            Configuration as dictionary
        """
        return {
            "engine": self.engine.value,
            "gravity": self.gravity.tolist(),
            "time_step": self.time_step,
            "substeps": self.substeps,
            "solver_iterations": self.solver_iterations,
            "enable_collisions": self.enable_collisions,
            "enable_friction": self.enable_friction,
            "enable_soft_bodies": self.enable_soft_bodies,
            "enable_fluids": self.enable_fluids,
            "collision_margin": self.collision_margin,
            "contact_breaking_threshold": self.contact_breaking_threshold,
            "deactivation_time": self.deactivation_time,
            "sleep_threshold": self.sleep_threshold,
            "materials": {name: {
                "density": mat.density,
                "friction_static": mat.friction_static,
                "friction_dynamic": mat.friction_dynamic,
                "restitution": mat.restitution,
                "color": mat.color
            } for name, mat in self.materials.items()},
            "debug_settings": {
                "debug_draw": self.debug_draw,
                "wireframe_mode": self.wireframe_mode,
                "show_contact_points": self.show_contact_points,
                "show_constraint_limits": self.show_constraint_limits
            }
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'PhysicsConfig':
        """Create configuration from dictionary.
        
        Args:
            config_dict: Configuration dictionary
            
        Returns:
            PhysicsConfig instance
        """
        config = cls()
        
        # Basic parameters
        config.engine = PhysicsEngine(config_dict.get("engine", "bullet"))
        config.gravity = np.array(config_dict.get("gravity", [0, -9.81, 0]))
        config.time_step = config_dict.get("time_step", 0.01)
        config.substeps = config_dict.get("substeps", 10)
        config.solver_iterations = config_dict.get("solver_iterations", 20)
        
        # Feature flags
        config.enable_collisions = config_dict.get("enable_collisions", True)
        config.enable_friction = config_dict.get("enable_friction", True)
        config.enable_soft_bodies = config_dict.get("enable_soft_bodies", False)
        config.enable_fluids = config_dict.get("enable_fluids", False)
        
        # Advanced parameters
        config.collision_margin = config_dict.get("collision_margin", 0.001)
        config.contact_breaking_threshold = config_dict.get("contact_breaking_threshold", 0.02)
        config.deactivation_time = config_dict.get("deactivation_time", 2.0)
        config.sleep_threshold = config_dict.get("sleep_threshold", 0.1)
        
        # Debug settings
        debug_settings = config_dict.get("debug_settings", {})
        config.debug_draw = debug_settings.get("debug_draw", False)
        config.wireframe_mode = debug_settings.get("wireframe_mode", False)
        config.show_contact_points = debug_settings.get("show_contact_points", False)
        config.show_constraint_limits = debug_settings.get("show_constraint_limits", False)
        
        return config