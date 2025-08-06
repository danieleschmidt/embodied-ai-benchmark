"""Advanced physics simulation module."""

from .physics_config import PhysicsConfig, MaterialProperties, PhysicsEngine
from .realistic_simulation import RealisticPhysicsSimulator, SoftBodySimulator

__all__ = [
    "PhysicsConfig", 
    "MaterialProperties", 
    "PhysicsEngine",
    "RealisticPhysicsSimulator",
    "SoftBodySimulator"
]