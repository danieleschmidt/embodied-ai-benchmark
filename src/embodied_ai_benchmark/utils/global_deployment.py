"""
Global Deployment and Multi-Region Support
"""

import asyncio
import json
import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import time
import hashlib
from pathlib import Path

logger = logging.getLogger(__name__)


class Region(Enum):
    """Supported deployment regions."""
    US_EAST_1 = "us-east-1"
    US_WEST_2 = "us-west-2"
    EU_WEST_1 = "eu-west-1"
    EU_CENTRAL_1 = "eu-central-1"
    ASIA_PACIFIC_1 = "ap-southeast-1"
    ASIA_PACIFIC_2 = "ap-northeast-1"


@dataclass
class RegionConfig:
    """Configuration for a specific region."""
    region: Region
    enabled: bool = True
    max_concurrent_tasks: int = 100
    data_residency_rules: Dict[str, str] = field(default_factory=dict)
    compliance_requirements: List[str] = field(default_factory=list)
    latency_targets: Dict[str, float] = field(default_factory=dict)
    failover_priority: int = 1
    
    
@dataclass
class GlobalMetrics:
    """Global deployment metrics."""
    timestamp: datetime
    region: Region
    active_tasks: int
    completed_tasks: int
    failed_tasks: int
    average_latency_ms: float
    cpu_utilization: float
    memory_utilization: float
    compliance_score: float
    

class GlobalDeploymentManager:
    """Manages global deployment across multiple regions."""
    
    def __init__(self, default_region: Region = Region.US_EAST_1):
        """Initialize global deployment manager.
        
        Args:
            default_region: Default region for deployments
        """
        self.default_region = default_region
        self.region_configs: Dict[Region, RegionConfig] = {}
        self.active_regions: List[Region] = []
        self.metrics_history: List[GlobalMetrics] = []
        
        # Load balancing
        self.current_loads: Dict[Region, int] = {}
        self.failover_chain: List[Region] = []
        
        # Data management
        self.data_locations: Dict[str, Region] = {}
        self.replication_policies: Dict[str, List[Region]] = {}
        
        self._setup_default_regions()
        
    def _setup_default_regions(self) -> None:
        """Setup default region configurations."""
        self.region_configs = {
            Region.US_EAST_1: RegionConfig(
                region=Region.US_EAST_1,
                compliance_requirements=["SOX", "NIST"],
                latency_targets={"api": 100.0, "processing": 1000.0},
                failover_priority=1
            ),
            Region.EU_WEST_1: RegionConfig(
                region=Region.EU_WEST_1,
                compliance_requirements=["GDPR", "ISO27001"],
                latency_targets={"api": 150.0, "processing": 1200.0},
                failover_priority=2,
                data_residency_rules={"user_data": "eu_only", "analytics": "global"}
            ),
            Region.ASIA_PACIFIC_1: RegionConfig(
                region=Region.ASIA_PACIFIC_1,
                compliance_requirements=["PDPA"],
                latency_targets={"api": 200.0, "processing": 1500.0},
                failover_priority=3
            )
        }
        
        self.active_regions = [Region.US_EAST_1]
        self.failover_chain = [Region.US_EAST_1, Region.EU_WEST_1, Region.ASIA_PACIFIC_1]
        
    def add_region(self, config: RegionConfig) -> None:
        """Add a new region configuration."""
        self.region_configs[config.region] = config
        if config.enabled and config.region not in self.active_regions:
            self.active_regions.append(config.region)
            
        # Update failover chain based on priority
        self.failover_chain = sorted(
            [r for r in self.region_configs.keys() if self.region_configs[r].enabled],
            key=lambda r: self.region_configs[r].failover_priority
        )
        
        logger.info(f"Added region {config.region.value} to deployment")
        
    def select_optimal_region(self, 
                            task_type: str,
                            data_requirements: Optional[Dict[str, str]] = None,
                            user_location: Optional[str] = None) -> Region:
        """Select the optimal region for a task.
        
        Args:
            task_type: Type of task to execute
            data_requirements: Data residency requirements
            user_location: User's geographic location
            
        Returns:
            Optimal region for the task
        """
        if not self.active_regions:
            return self.default_region
            
        candidate_regions = self.active_regions.copy()
        
        # Apply data residency filters
        if data_requirements:
            filtered_regions = []
            for region in candidate_regions:
                config = self.region_configs[region]
                if self._check_data_residency(region, data_requirements):
                    filtered_regions.append(region)
            candidate_regions = filtered_regions or candidate_regions
            
        # Apply load balancing
        if len(candidate_regions) > 1:
            # Find region with lowest current load
            loads = [(region, self.current_loads.get(region, 0)) for region in candidate_regions]
            loads.sort(key=lambda x: x[1])
            
            # Select from the least loaded regions (top 50%)
            cutoff = max(1, len(loads) // 2)
            best_regions = [region for region, _ in loads[:cutoff]]
            
            # If user location is specified, prefer nearby regions
            if user_location:
                region = self._select_by_proximity(best_regions, user_location)
            else:
                region = best_regions[0]
        else:
            region = candidate_regions[0] if candidate_regions else self.default_region
            
        # Update load tracking
        self.current_loads[region] = self.current_loads.get(region, 0) + 1
        
        return region
        
    def _check_data_residency(self, region: Region, requirements: Dict[str, str]) -> bool:
        """Check if region meets data residency requirements."""
        config = self.region_configs[region]
        
        for data_type, requirement in requirements.items():
            if requirement == "eu_only" and region not in [Region.EU_WEST_1, Region.EU_CENTRAL_1]:
                return False
            elif requirement == "us_only" and region not in [Region.US_EAST_1, Region.US_WEST_2]:
                return False
                
        return True
        
    def _select_by_proximity(self, regions: List[Region], user_location: str) -> Region:
        """Select region based on proximity to user."""
        # Simplified proximity mapping
        proximity_map = {
            "us": [Region.US_EAST_1, Region.US_WEST_2],
            "eu": [Region.EU_WEST_1, Region.EU_CENTRAL_1],
            "asia": [Region.ASIA_PACIFIC_1, Region.ASIA_PACIFIC_2]
        }
        
        user_region = user_location.lower()
        for location, preferred_regions in proximity_map.items():
            if location in user_region:
                for region in preferred_regions:
                    if region in regions:
                        return region
                        
        return regions[0]  # Fallback to first available
        
    def execute_with_failover(self, 
                            task_func: callable,
                            task_args: tuple = (),
                            task_kwargs: dict = None,
                            max_retries: int = 3) -> Any:
        """Execute a task with automatic failover between regions.
        
        Args:
            task_func: Function to execute
            task_args: Arguments for the function
            task_kwargs: Keyword arguments for the function
            max_retries: Maximum number of failover attempts
            
        Returns:
            Result of the task execution
        """
        task_kwargs = task_kwargs or {}
        
        for attempt in range(max_retries):
            if attempt < len(self.failover_chain):
                region = self.failover_chain[attempt]
            else:
                region = self.default_region
                
            try:
                logger.info(f"Executing task in region {region.value} (attempt {attempt + 1})")
                
                # Set region context for the task
                task_kwargs['region'] = region
                
                start_time = time.time()
                result = task_func(*task_args, **task_kwargs)
                execution_time = time.time() - start_time
                
                # Record successful execution
                self._record_execution_metrics(region, execution_time, success=True)
                
                return result
                
            except Exception as e:
                logger.warning(f"Task failed in region {region.value}: {e}")
                self._record_execution_metrics(region, 0, success=False)
                
                if attempt == max_retries - 1:
                    raise
                    
        raise RuntimeError("All regions failed to execute the task")
        
    def _record_execution_metrics(self, region: Region, execution_time: float, success: bool) -> None:
        """Record execution metrics for a region."""
        # Update current load
        if region in self.current_loads:
            self.current_loads[region] = max(0, self.current_loads[region] - 1)
            
        # Record metrics (simplified)
        metrics = GlobalMetrics(
            timestamp=datetime.now(),
            region=region,
            active_tasks=self.current_loads.get(region, 0),
            completed_tasks=1 if success else 0,
            failed_tasks=0 if success else 1,
            average_latency_ms=execution_time * 1000,
            cpu_utilization=0.0,  # Would be populated by monitoring
            memory_utilization=0.0,  # Would be populated by monitoring
            compliance_score=1.0 if success else 0.5
        )
        
        self.metrics_history.append(metrics)
        
        # Keep only recent metrics
        if len(self.metrics_history) > 1000:
            self.metrics_history = self.metrics_history[-1000:]
            
    def get_region_status(self) -> Dict[str, Any]:
        """Get status of all regions."""
        status = {}
        
        for region, config in self.region_configs.items():
            recent_metrics = [
                m for m in self.metrics_history[-100:]  # Last 100 entries
                if m.region == region
            ]
            
            if recent_metrics:
                avg_latency = sum(m.average_latency_ms for m in recent_metrics) / len(recent_metrics)
                success_rate = sum(1 for m in recent_metrics if m.failed_tasks == 0) / len(recent_metrics)
            else:
                avg_latency = 0.0
                success_rate = 1.0
                
            status[region.value] = {
                "enabled": config.enabled,
                "active": region in self.active_regions,
                "current_load": self.current_loads.get(region, 0),
                "average_latency_ms": avg_latency,
                "success_rate": success_rate,
                "compliance_requirements": config.compliance_requirements,
                "failover_priority": config.failover_priority
            }
            
        return status
        
    def setup_data_replication(self, data_id: str, primary_region: Region, 
                             replica_regions: List[Region]) -> None:
        """Setup data replication policy.
        
        Args:
            data_id: Unique identifier for the data
            primary_region: Primary storage region
            replica_regions: List of replica regions
        """
        self.data_locations[data_id] = primary_region
        self.replication_policies[data_id] = replica_regions
        
        logger.info(f"Setup replication for {data_id}: primary={primary_region.value}, "
                   f"replicas={[r.value for r in replica_regions]}")
                   
    def get_data_location(self, data_id: str, prefer_local: bool = True) -> Region:
        """Get the optimal location for accessing data.
        
        Args:
            data_id: Data identifier
            prefer_local: Whether to prefer local region access
            
        Returns:
            Optimal region for data access
        """
        if data_id not in self.data_locations:
            return self.default_region
            
        primary_region = self.data_locations[data_id]
        
        # If primary region is active and we don't prefer local, use primary
        if primary_region in self.active_regions and not prefer_local:
            return primary_region
            
        # Check for local replicas
        replicas = self.replication_policies.get(data_id, [])
        for region in self.active_regions:
            if region in replicas:
                return region
                
        # Fallback to primary if available
        if primary_region in self.active_regions:
            return primary_region
            
        return self.default_region


class MultiRegionLoadBalancer:
    """Advanced load balancer for multi-region deployments."""
    
    def __init__(self, deployment_manager: GlobalDeploymentManager):
        """Initialize load balancer.
        
        Args:
            deployment_manager: Global deployment manager instance
        """
        self.deployment_manager = deployment_manager
        self.request_history: List[Dict[str, Any]] = []
        self.circuit_breakers: Dict[Region, bool] = {}
        
    def route_request(self, 
                     request_id: str,
                     task_type: str,
                     priority: int = 1,
                     data_requirements: Optional[Dict[str, str]] = None) -> Region:
        """Route a request to the optimal region.
        
        Args:
            request_id: Unique request identifier
            task_type: Type of task
            priority: Request priority (1-10)
            data_requirements: Data residency requirements
            
        Returns:
            Selected region for the request
        """
        # Check circuit breakers
        available_regions = [
            region for region in self.deployment_manager.active_regions
            if not self.circuit_breakers.get(region, False)
        ]
        
        if not available_regions:
            # All regions are circuit broken, use default
            region = self.deployment_manager.default_region
        else:
            # Use deployment manager's selection logic
            region = self.deployment_manager.select_optimal_region(
                task_type=task_type,
                data_requirements=data_requirements
            )
            
            # Ensure selected region is available
            if region not in available_regions:
                region = available_regions[0]
                
        # Record request routing
        self.request_history.append({
            "timestamp": datetime.now().isoformat(),
            "request_id": request_id,
            "task_type": task_type,
            "priority": priority,
            "selected_region": region.value,
            "available_regions": [r.value for r in available_regions]
        })
        
        # Keep history manageable
        if len(self.request_history) > 10000:
            self.request_history = self.request_history[-10000:]
            
        return region
        
    def update_circuit_breaker(self, region: Region, is_healthy: bool) -> None:
        """Update circuit breaker status for a region.
        
        Args:
            region: Region to update
            is_healthy: Whether the region is healthy
        """
        self.circuit_breakers[region] = not is_healthy
        
        if not is_healthy:
            logger.warning(f"Circuit breaker opened for region {region.value}")
        else:
            logger.info(f"Circuit breaker closed for region {region.value}")
            
    def get_routing_statistics(self) -> Dict[str, Any]:
        """Get load balancing statistics."""
        if not self.request_history:
            return {"total_requests": 0}
            
        total_requests = len(self.request_history)
        
        # Count requests by region
        region_counts = {}
        for request in self.request_history:
            region = request["selected_region"]
            region_counts[region] = region_counts.get(region, 0) + 1
            
        # Calculate distribution percentages
        region_distribution = {
            region: (count / total_requests) * 100
            for region, count in region_counts.items()
        }
        
        return {
            "total_requests": total_requests,
            "region_distribution": region_distribution,
            "circuit_breaker_status": {
                region.value: is_broken 
                for region, is_broken in self.circuit_breakers.items()
            },
            "active_regions": [region.value for region in self.deployment_manager.active_regions]
        }


# Global instances
_global_deployment_manager: Optional[GlobalDeploymentManager] = None
_global_load_balancer: Optional[MultiRegionLoadBalancer] = None


def get_deployment_manager() -> GlobalDeploymentManager:
    """Get global deployment manager instance."""
    global _global_deployment_manager
    if _global_deployment_manager is None:
        _global_deployment_manager = GlobalDeploymentManager()
    return _global_deployment_manager


def get_load_balancer() -> MultiRegionLoadBalancer:
    """Get global load balancer instance."""
    global _global_load_balancer
    if _global_load_balancer is None:
        deployment_manager = get_deployment_manager()
        _global_load_balancer = MultiRegionLoadBalancer(deployment_manager)
    return _global_load_balancer


def execute_globally(task_func: callable, 
                    *args, 
                    region: Optional[Region] = None,
                    **kwargs) -> Any:
    """Execute a function with global deployment support.
    
    Args:
        task_func: Function to execute
        *args: Arguments for the function
        region: Specific region to use (optional)
        **kwargs: Keyword arguments for the function
        
    Returns:
        Result of the function execution
    """
    deployment_manager = get_deployment_manager()
    
    if region is None:
        region = deployment_manager.select_optimal_region("general")
        
    return deployment_manager.execute_with_failover(
        task_func, args, kwargs
    )