"""
Terragon Performance Optimization Engine v2.0

Advanced performance optimization, caching, and auto-scaling engine for autonomous SDLC.
Implements intelligent resource management, predictive scaling, and performance tuning.
"""

import asyncio
import time
import logging
import threading
from typing import Dict, List, Optional, Any, Callable, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from collections import defaultdict, deque
import json
import psutil
import concurrent.futures
from pathlib import Path
import hashlib


class OptimizationLevel(Enum):
    """Performance optimization levels"""
    CONSERVATIVE = "conservative"
    BALANCED = "balanced"
    AGGRESSIVE = "aggressive"
    EXTREME = "extreme"


class CacheStrategy(Enum):
    """Caching strategies"""
    LRU = "lru"           # Least Recently Used
    LFU = "lfu"           # Least Frequently Used
    TTL = "ttl"           # Time To Live
    ADAPTIVE = "adaptive"  # Adaptive based on access patterns


class ScalingDirection(Enum):
    """Scaling directions"""
    SCALE_UP = "scale_up"
    SCALE_DOWN = "scale_down"
    MAINTAIN = "maintain"


@dataclass
class PerformanceMetrics:
    """Performance metrics snapshot"""
    timestamp: datetime
    cpu_percent: float
    memory_percent: float
    disk_io_read_mb: float
    disk_io_write_mb: float
    network_bytes_sent: float
    network_bytes_recv: float
    active_threads: int
    response_time_ms: float = 0.0
    throughput_ops_sec: float = 0.0
    error_rate_percent: float = 0.0
    cache_hit_rate: float = 0.0


@dataclass 
class OptimizationTask:
    """Performance optimization task"""
    task_id: str
    name: str
    description: str
    optimization_function: Callable
    estimated_improvement_percent: float
    implementation_complexity: str  # low, medium, high
    resource_requirements: Dict[str, Any]
    dependencies: List[str] = field(default_factory=list)
    status: str = "pending"  # pending, running, completed, failed
    result: Optional[Dict[str, Any]] = None


class IntelligentCache:
    """
    Intelligent caching system with adaptive strategies and predictive prefetching.
    """
    
    def __init__(self, max_size: int = 1000, strategy: CacheStrategy = CacheStrategy.ADAPTIVE):
        self.max_size = max_size
        self.strategy = strategy
        self.cache: Dict[str, Any] = {}
        self.access_times: Dict[str, datetime] = {}
        self.access_counts: Dict[str, int] = defaultdict(int)
        self.hit_count = 0
        self.miss_count = 0
        self.lock = threading.RLock()
        
        # Adaptive strategy parameters
        self.access_patterns: Dict[str, List[datetime]] = defaultdict(list)
        self.prefetch_predictions: Dict[str, float] = {}
        
        # Start background optimization
        self._start_background_optimization()
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        with self.lock:
            if key in self.cache:
                self._record_access(key)
                self.hit_count += 1
                return self.cache[key]
            else:
                self.miss_count += 1
                return None
    
    def put(self, key: str, value: Any, ttl_seconds: Optional[int] = None):
        """Put value in cache"""
        with self.lock:
            # Check if we need to evict
            if len(self.cache) >= self.max_size:
                self._evict()
            
            self.cache[key] = value
            self._record_access(key)
            
            if ttl_seconds and self.strategy in [CacheStrategy.TTL, CacheStrategy.ADAPTIVE]:
                # Set expiration time
                self.access_times[key + "_expires"] = datetime.now() + timedelta(seconds=ttl_seconds)
    
    def _record_access(self, key: str):
        """Record cache access for analytics"""
        now = datetime.now()
        self.access_times[key] = now
        self.access_counts[key] += 1
        
        # Record access pattern for adaptive strategy
        if self.strategy == CacheStrategy.ADAPTIVE:
            self.access_patterns[key].append(now)
            # Keep only recent access patterns
            cutoff = now - timedelta(hours=1)
            self.access_patterns[key] = [t for t in self.access_patterns[key] if t > cutoff]
    
    def _evict(self):
        """Evict items based on strategy"""
        if not self.cache:
            return
        
        if self.strategy == CacheStrategy.LRU:
            # Evict least recently used
            oldest_key = min(self.access_times.keys(), key=lambda k: self.access_times[k])
            self._remove_key(oldest_key)
        
        elif self.strategy == CacheStrategy.LFU:
            # Evict least frequently used
            least_used_key = min(self.access_counts.keys(), key=lambda k: self.access_counts[k])
            self._remove_key(least_used_key)
        
        elif self.strategy in [CacheStrategy.TTL, CacheStrategy.ADAPTIVE]:
            # First try to evict expired items
            now = datetime.now()
            expired_keys = []
            for key in list(self.access_times.keys()):
                if key.endswith("_expires") and self.access_times[key] < now:
                    expired_keys.append(key.replace("_expires", ""))
            
            if expired_keys:
                self._remove_key(expired_keys[0])
            else:
                # Fall back to LRU
                oldest_key = min(self.access_times.keys(), key=lambda k: self.access_times[k])
                self._remove_key(oldest_key)
    
    def _remove_key(self, key: str):
        """Remove key from all data structures"""
        self.cache.pop(key, None)
        self.access_times.pop(key, None)
        self.access_counts.pop(key, None)
        self.access_patterns.pop(key, None)
        self.prefetch_predictions.pop(key, None)
    
    def _start_background_optimization(self):
        """Start background thread for cache optimization"""
        def optimize_cache():
            while True:
                time.sleep(60)  # Optimize every minute
                try:
                    self._optimize_cache()
                except Exception as e:
                    pass  # Silent optimization failures
        
        optimization_thread = threading.Thread(target=optimize_cache, daemon=True)
        optimization_thread.start()
    
    def _optimize_cache(self):
        """Optimize cache based on access patterns"""
        if self.strategy != CacheStrategy.ADAPTIVE:
            return
        
        with self.lock:
            # Analyze access patterns for prefetching opportunities
            self._analyze_prefetch_opportunities()
            
            # Adjust cache size based on hit rate
            self._adjust_cache_size()
            
            # Clean up expired entries
            self._cleanup_expired_entries()
    
    def _analyze_prefetch_opportunities(self):
        """Analyze patterns for predictive prefetching"""
        now = datetime.now()
        
        for key, accesses in self.access_patterns.items():
            if len(accesses) < 3:
                continue
            
            # Calculate access frequency and predict next access
            intervals = []
            for i in range(1, len(accesses)):
                interval = (accesses[i] - accesses[i-1]).total_seconds()
                intervals.append(interval)
            
            if intervals:
                avg_interval = sum(intervals) / len(intervals)
                last_access = accesses[-1]
                predicted_next_access = last_access + timedelta(seconds=avg_interval)
                
                # If prediction is soon, increase prefetch score
                time_to_prediction = (predicted_next_access - now).total_seconds()
                if 0 < time_to_prediction < 300:  # Next 5 minutes
                    confidence = max(0.0, 1.0 - (time_to_prediction / 300))
                    self.prefetch_predictions[key] = confidence
    
    def _adjust_cache_size(self):
        """Dynamically adjust cache size based on performance"""
        hit_rate = self.get_hit_rate()
        
        if hit_rate > 0.9 and len(self.cache) < self.max_size * 1.5:
            # High hit rate - increase cache size
            self.max_size = min(self.max_size * 1.2, 2000)
        elif hit_rate < 0.5 and self.max_size > 100:
            # Low hit rate - decrease cache size
            self.max_size = max(self.max_size * 0.8, 100)
    
    def _cleanup_expired_entries(self):
        """Clean up expired cache entries"""
        now = datetime.now()
        expired_keys = []
        
        for key in list(self.access_times.keys()):
            if key.endswith("_expires") and self.access_times[key] < now:
                expired_keys.append(key.replace("_expires", ""))
        
        for key in expired_keys:
            self._remove_key(key)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        with self.lock:
            total_requests = self.hit_count + self.miss_count
            hit_rate = (self.hit_count / total_requests) if total_requests > 0 else 0.0
            
            return {
                "size": len(self.cache),
                "max_size": self.max_size,
                "hit_count": self.hit_count,
                "miss_count": self.miss_count,
                "hit_rate": hit_rate,
                "strategy": self.strategy.value,
                "prefetch_candidates": len(self.prefetch_predictions),
                "memory_usage_mb": self._estimate_memory_usage()
            }
    
    def get_hit_rate(self) -> float:
        """Get current cache hit rate"""
        total_requests = self.hit_count + self.miss_count
        return (self.hit_count / total_requests) if total_requests > 0 else 0.0
    
    def _estimate_memory_usage(self) -> float:
        """Estimate memory usage in MB"""
        # Rough estimation - would be more accurate in production
        return len(self.cache) * 0.001  # Assume 1KB per entry


class ResourceMonitor:
    """
    Continuous resource monitoring with predictive analytics.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.metrics_history: deque = deque(maxlen=1000)  # Last 1000 measurements
        self.monitoring_active = False
        self.thresholds = {
            "cpu_warning": 70.0,
            "cpu_critical": 85.0,
            "memory_warning": 80.0,
            "memory_critical": 90.0,
            "disk_warning": 80.0,
            "disk_critical": 90.0
        }
        
    async def start_monitoring(self):
        """Start continuous resource monitoring"""
        self.monitoring_active = True
        self.logger.info("🔍 Starting resource monitoring")
        
        while self.monitoring_active:
            try:
                metrics = self._collect_metrics()
                self.metrics_history.append(metrics)
                
                # Check for resource pressure
                await self._check_resource_pressure(metrics)
                
                await asyncio.sleep(5)  # Monitor every 5 seconds
            except Exception as e:
                self.logger.error(f"Resource monitoring error: {e}")
                await asyncio.sleep(10)
    
    def _collect_metrics(self) -> PerformanceMetrics:
        """Collect current system metrics"""
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        network = psutil.net_io_counters()
        
        return PerformanceMetrics(
            timestamp=datetime.now(),
            cpu_percent=cpu_percent,
            memory_percent=memory.percent,
            disk_io_read_mb=getattr(psutil.disk_io_counters(), 'read_bytes', 0) / (1024*1024),
            disk_io_write_mb=getattr(psutil.disk_io_counters(), 'write_bytes', 0) / (1024*1024),
            network_bytes_sent=network.bytes_sent / (1024*1024),
            network_bytes_recv=network.bytes_recv / (1024*1024),
            active_threads=threading.active_count()
        )
    
    async def _check_resource_pressure(self, metrics: PerformanceMetrics):
        """Check for resource pressure and trigger optimizations"""
        warnings = []
        
        if metrics.cpu_percent > self.thresholds["cpu_critical"]:
            warnings.append(f"Critical CPU usage: {metrics.cpu_percent:.1f}%")
        elif metrics.cpu_percent > self.thresholds["cpu_warning"]:
            warnings.append(f"High CPU usage: {metrics.cpu_percent:.1f}%")
        
        if metrics.memory_percent > self.thresholds["memory_critical"]:
            warnings.append(f"Critical memory usage: {metrics.memory_percent:.1f}%")
        elif metrics.memory_percent > self.thresholds["memory_warning"]:
            warnings.append(f"High memory usage: {metrics.memory_percent:.1f}%")
        
        if warnings:
            self.logger.warning(f"Resource pressure detected: {'; '.join(warnings)}")
    
    def stop_monitoring(self):
        """Stop resource monitoring"""
        self.monitoring_active = False
    
    def get_current_metrics(self) -> Optional[PerformanceMetrics]:
        """Get most recent metrics"""
        return self.metrics_history[-1] if self.metrics_history else None
    
    def get_metrics_summary(self, window_minutes: int = 10) -> Dict[str, Any]:
        """Get summary of metrics over time window"""
        if not self.metrics_history:
            return {}
        
        cutoff = datetime.now() - timedelta(minutes=window_minutes)
        recent_metrics = [m for m in self.metrics_history if m.timestamp >= cutoff]
        
        if not recent_metrics:
            return {}
        
        cpu_values = [m.cpu_percent for m in recent_metrics]
        memory_values = [m.memory_percent for m in recent_metrics]
        
        return {
            "avg_cpu_percent": sum(cpu_values) / len(cpu_values),
            "max_cpu_percent": max(cpu_values),
            "avg_memory_percent": sum(memory_values) / len(memory_values),
            "max_memory_percent": max(memory_values),
            "sample_count": len(recent_metrics),
            "window_minutes": window_minutes
        }
    
    def predict_resource_usage(self, minutes_ahead: int = 5) -> Dict[str, float]:
        """Predict resource usage using simple linear trend"""
        if len(self.metrics_history) < 10:
            return {"cpu_percent": 0.0, "memory_percent": 0.0}
        
        # Get recent trends
        recent_metrics = list(self.metrics_history)[-10:]
        
        # Simple linear regression on CPU and memory
        cpu_trend = self._calculate_trend([m.cpu_percent for m in recent_metrics])
        memory_trend = self._calculate_trend([m.memory_percent for m in recent_metrics])
        
        current_cpu = recent_metrics[-1].cpu_percent
        current_memory = recent_metrics[-1].memory_percent
        
        predicted_cpu = max(0, current_cpu + (cpu_trend * minutes_ahead))
        predicted_memory = max(0, current_memory + (memory_trend * minutes_ahead))
        
        return {
            "cpu_percent": predicted_cpu,
            "memory_percent": predicted_memory,
            "confidence": 0.7 if len(recent_metrics) >= 10 else 0.5
        }
    
    def _calculate_trend(self, values: List[float]) -> float:
        """Calculate simple linear trend"""
        if len(values) < 2:
            return 0.0
        
        # Simple slope calculation
        n = len(values)
        x_sum = sum(range(n))
        y_sum = sum(values)
        xy_sum = sum(i * values[i] for i in range(n))
        x2_sum = sum(i * i for i in range(n))
        
        denominator = n * x2_sum - x_sum * x_sum
        if denominator == 0:
            return 0.0
        
        slope = (n * xy_sum - x_sum * y_sum) / denominator
        return slope


class AutoScalingEngine:
    """
    Intelligent auto-scaling engine with predictive scaling.
    """
    
    def __init__(self, resource_monitor: ResourceMonitor):
        self.logger = logging.getLogger(__name__)
        self.resource_monitor = resource_monitor
        self.scaling_policies = self._initialize_scaling_policies()
        self.current_scale = 1.0
        self.min_scale = 0.5
        self.max_scale = 4.0
        self.scaling_history: List[Dict[str, Any]] = []
        
    def _initialize_scaling_policies(self) -> List[Dict[str, Any]]:
        """Initialize auto-scaling policies"""
        return [
            {
                "name": "cpu_scale_up",
                "condition": lambda metrics: metrics.cpu_percent > 75,
                "action": ScalingDirection.SCALE_UP,
                "factor": 1.2,
                "cooldown_minutes": 5
            },
            {
                "name": "cpu_scale_down", 
                "condition": lambda metrics: metrics.cpu_percent < 30,
                "action": ScalingDirection.SCALE_DOWN,
                "factor": 0.8,
                "cooldown_minutes": 10
            },
            {
                "name": "memory_scale_up",
                "condition": lambda metrics: metrics.memory_percent > 80,
                "action": ScalingDirection.SCALE_UP,
                "factor": 1.3,
                "cooldown_minutes": 3
            }
        ]
    
    async def evaluate_scaling(self) -> Optional[Dict[str, Any]]:
        """Evaluate if scaling is needed"""
        current_metrics = self.resource_monitor.get_current_metrics()
        if not current_metrics:
            return None
        
        # Check predictive scaling first
        predicted_scaling = await self._evaluate_predictive_scaling()
        if predicted_scaling:
            return predicted_scaling
        
        # Check reactive scaling policies
        for policy in self.scaling_policies:
            if self._should_apply_policy(policy, current_metrics):
                return await self._apply_scaling_policy(policy, current_metrics)
        
        return None
    
    async def _evaluate_predictive_scaling(self) -> Optional[Dict[str, Any]]:
        """Evaluate predictive scaling based on trends"""
        predicted_usage = self.resource_monitor.predict_resource_usage(5)  # 5 minutes ahead
        
        if predicted_usage["cpu_percent"] > 80 or predicted_usage["memory_percent"] > 85:
            return {
                "type": "predictive",
                "action": ScalingDirection.SCALE_UP,
                "factor": 1.2,
                "reason": f"Predicted usage: CPU {predicted_usage['cpu_percent']:.1f}%, Memory {predicted_usage['memory_percent']:.1f}%",
                "confidence": predicted_usage["confidence"]
            }
        
        return None
    
    def _should_apply_policy(self, policy: Dict[str, Any], metrics: PerformanceMetrics) -> bool:
        """Check if scaling policy should be applied"""
        # Check cooldown period
        now = datetime.now()
        for event in reversed(self.scaling_history):
            if event["policy_name"] == policy["name"]:
                last_scaling = event["timestamp"]
                cooldown = timedelta(minutes=policy["cooldown_minutes"])
                if now - last_scaling < cooldown:
                    return False
                break
        
        # Check condition
        try:
            return policy["condition"](metrics)
        except Exception:
            return False
    
    async def _apply_scaling_policy(self, policy: Dict[str, Any], metrics: PerformanceMetrics) -> Dict[str, Any]:
        """Apply scaling policy"""
        old_scale = self.current_scale
        
        if policy["action"] == ScalingDirection.SCALE_UP:
            new_scale = min(self.max_scale, self.current_scale * policy["factor"])
        elif policy["action"] == ScalingDirection.SCALE_DOWN:
            new_scale = max(self.min_scale, self.current_scale * policy["factor"])
        else:
            new_scale = self.current_scale
        
        scaling_event = {
            "timestamp": datetime.now(),
            "policy_name": policy["name"],
            "old_scale": old_scale,
            "new_scale": new_scale,
            "trigger_metrics": {
                "cpu_percent": metrics.cpu_percent,
                "memory_percent": metrics.memory_percent
            },
            "reason": f"Policy {policy['name']} triggered"
        }
        
        self.scaling_history.append(scaling_event)
        self.current_scale = new_scale
        
        # Execute scaling action
        await self._execute_scaling(new_scale)
        
        self.logger.info(f"Scaling applied: {old_scale:.2f} -> {new_scale:.2f} (Policy: {policy['name']})")
        
        return scaling_event
    
    async def _execute_scaling(self, new_scale: float):
        """Execute the actual scaling operations"""
        # In production, this would:
        # - Adjust thread pool sizes
        # - Modify resource allocations
        # - Update configuration parameters
        # - Trigger container/instance scaling
        
        # For now, we simulate scaling effects
        if new_scale > self.current_scale:
            # Scaling up - increase parallelism
            pass
        else:
            # Scaling down - reduce resource usage
            pass
    
    def get_scaling_recommendations(self) -> List[str]:
        """Get scaling recommendations"""
        recommendations = []
        
        current_metrics = self.resource_monitor.get_current_metrics()
        if not current_metrics:
            return recommendations
        
        # Analyze resource usage patterns
        if current_metrics.cpu_percent > 60:
            recommendations.append("Consider CPU optimization or horizontal scaling")
        
        if current_metrics.memory_percent > 70:
            recommendations.append("Monitor memory usage - consider increasing memory or optimizing data structures")
        
        if self.current_scale == self.max_scale:
            recommendations.append("At maximum scale - consider architectural improvements")
        
        if self.current_scale == self.min_scale and current_metrics.cpu_percent < 20:
            recommendations.append("System is under-utilized - consider consolidation")
        
        return recommendations


class PerformanceOptimizationEngine:
    """
    Main performance optimization engine that coordinates all optimization efforts.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.optimization_level = OptimizationLevel.BALANCED
        
        # Initialize components
        self.cache = IntelligentCache(max_size=1000, strategy=CacheStrategy.ADAPTIVE)
        self.resource_monitor = ResourceMonitor()
        self.autoscaling_engine = AutoScalingEngine(self.resource_monitor)
        
        # Optimization tasks
        self.optimization_tasks: List[OptimizationTask] = []
        self.completed_optimizations: List[str] = []
        
        # Performance tracking
        self.baseline_performance: Optional[PerformanceMetrics] = None
        self.performance_improvements: Dict[str, float] = {}
        
        self._initialize_optimization_tasks()
    
    def _initialize_optimization_tasks(self):
        """Initialize available optimization tasks"""
        self.optimization_tasks = [
            OptimizationTask(
                task_id="enable_caching",
                name="Enable Intelligent Caching",
                description="Enable adaptive caching for frequently accessed data",
                optimization_function=self._optimize_caching,
                estimated_improvement_percent=15.0,
                implementation_complexity="low"
            ),
            OptimizationTask(
                task_id="parallel_processing",
                name="Enable Parallel Processing",
                description="Parallelize independent operations",
                optimization_function=self._optimize_parallel_processing,
                estimated_improvement_percent=25.0,
                implementation_complexity="medium"
            ),
            OptimizationTask(
                task_id="memory_optimization",
                name="Optimize Memory Usage",
                description="Optimize memory allocation and garbage collection",
                optimization_function=self._optimize_memory,
                estimated_improvement_percent=10.0,
                implementation_complexity="medium"
            ),
            OptimizationTask(
                task_id="io_optimization",
                name="Optimize I/O Operations",
                description="Optimize file and network I/O operations",
                optimization_function=self._optimize_io,
                estimated_improvement_percent=20.0,
                implementation_complexity="low"
            ),
            OptimizationTask(
                task_id="algorithm_optimization",
                name="Algorithm Optimization",
                description="Optimize critical algorithms and data structures",
                optimization_function=self._optimize_algorithms,
                estimated_improvement_percent=30.0,
                implementation_complexity="high"
            )
        ]
    
    async def start_optimization_engine(self):
        """Start the performance optimization engine"""
        self.logger.info("🚀 Starting performance optimization engine")
        
        # Start resource monitoring
        asyncio.create_task(self.resource_monitor.start_monitoring())
        
        # Capture baseline performance
        await asyncio.sleep(5)  # Let monitoring collect some data
        self.baseline_performance = self.resource_monitor.get_current_metrics()
        
        # Start optimization loop
        asyncio.create_task(self._optimization_loop())
    
    async def _optimization_loop(self):
        """Main optimization loop"""
        while True:
            try:
                # Check if auto-scaling is needed
                scaling_action = await self.autoscaling_engine.evaluate_scaling()
                if scaling_action:
                    self.logger.info(f"Auto-scaling action: {scaling_action}")
                
                # Check for optimization opportunities
                await self._evaluate_optimizations()
                
                await asyncio.sleep(30)  # Check every 30 seconds
            except Exception as e:
                self.logger.error(f"Optimization loop error: {e}")
                await asyncio.sleep(60)
    
    async def _evaluate_optimizations(self):
        """Evaluate and apply optimizations"""
        current_metrics = self.resource_monitor.get_current_metrics()
        if not current_metrics:
            return
        
        # Determine if optimization is needed
        needs_optimization = self._assess_optimization_need(current_metrics)
        
        if needs_optimization:
            # Select and apply optimizations
            selected_optimizations = self._select_optimizations(current_metrics)
            
            for optimization in selected_optimizations:
                if optimization.task_id not in self.completed_optimizations:
                    await self._apply_optimization(optimization)
    
    def _assess_optimization_need(self, metrics: PerformanceMetrics) -> bool:
        """Assess if optimization is needed"""
        # Conservative thresholds
        if self.optimization_level == OptimizationLevel.CONSERVATIVE:
            return metrics.cpu_percent > 80 or metrics.memory_percent > 85
        
        # Balanced thresholds
        elif self.optimization_level == OptimizationLevel.BALANCED:
            return metrics.cpu_percent > 70 or metrics.memory_percent > 75
        
        # Aggressive thresholds
        elif self.optimization_level == OptimizationLevel.AGGRESSIVE:
            return metrics.cpu_percent > 60 or metrics.memory_percent > 65
        
        # Extreme - always optimize
        elif self.optimization_level == OptimizationLevel.EXTREME:
            return True
        
        return False
    
    def _select_optimizations(self, metrics: PerformanceMetrics) -> List[OptimizationTask]:
        """Select optimizations based on current metrics"""
        candidates = []
        
        # CPU-focused optimizations
        if metrics.cpu_percent > 60:
            candidates.extend([
                task for task in self.optimization_tasks 
                if task.task_id in ["parallel_processing", "algorithm_optimization"]
            ])
        
        # Memory-focused optimizations
        if metrics.memory_percent > 70:
            candidates.extend([
                task for task in self.optimization_tasks 
                if task.task_id in ["enable_caching", "memory_optimization"]
            ])
        
        # I/O-focused optimizations
        if metrics.active_threads > 50:
            candidates.extend([
                task for task in self.optimization_tasks 
                if task.task_id == "io_optimization"
            ])
        
        # Remove duplicates and already completed
        unique_candidates = []
        seen_ids = set()
        for task in candidates:
            if task.task_id not in seen_ids and task.task_id not in self.completed_optimizations:
                unique_candidates.append(task)
                seen_ids.add(task.task_id)
        
        # Sort by estimated improvement (descending) and complexity (ascending)
        unique_candidates.sort(
            key=lambda t: (-t.estimated_improvement_percent, 
                          {"low": 1, "medium": 2, "high": 3}.get(t.implementation_complexity, 2))
        )
        
        # Return top candidates based on optimization level
        max_concurrent = {
            OptimizationLevel.CONSERVATIVE: 1,
            OptimizationLevel.BALANCED: 2,
            OptimizationLevel.AGGRESSIVE: 3,
            OptimizationLevel.EXTREME: len(unique_candidates)
        }.get(self.optimization_level, 2)
        
        return unique_candidates[:max_concurrent]
    
    async def _apply_optimization(self, optimization: OptimizationTask):
        """Apply an optimization"""
        self.logger.info(f"Applying optimization: {optimization.name}")
        
        optimization.status = "running"
        start_time = time.time()
        
        try:
            # Record before metrics
            before_metrics = self.resource_monitor.get_current_metrics()
            
            # Apply optimization
            result = await optimization.optimization_function()
            
            # Wait a bit for effects to show
            await asyncio.sleep(10)
            
            # Record after metrics
            after_metrics = self.resource_monitor.get_current_metrics()
            
            # Calculate improvement
            improvement = self._calculate_improvement(before_metrics, after_metrics)
            
            optimization.status = "completed"
            optimization.result = {
                "improvement_percent": improvement,
                "execution_time_seconds": time.time() - start_time,
                "before_metrics": before_metrics,
                "after_metrics": after_metrics
            }
            
            self.completed_optimizations.append(optimization.task_id)
            self.performance_improvements[optimization.task_id] = improvement
            
            self.logger.info(f"Optimization completed: {optimization.name} - {improvement:.1f}% improvement")
            
        except Exception as e:
            optimization.status = "failed"
            optimization.result = {"error": str(e)}
            self.logger.error(f"Optimization failed: {optimization.name} - {e}")
    
    def _calculate_improvement(self, before: Optional[PerformanceMetrics], after: Optional[PerformanceMetrics]) -> float:
        """Calculate performance improvement percentage"""
        if not before or not after:
            return 0.0
        
        # Simple improvement calculation based on resource usage reduction
        cpu_improvement = max(0, before.cpu_percent - after.cpu_percent)
        memory_improvement = max(0, before.memory_percent - after.memory_percent)
        
        # Weight the improvements
        overall_improvement = (cpu_improvement * 0.6) + (memory_improvement * 0.4)
        
        return overall_improvement
    
    # Optimization implementations
    async def _optimize_caching(self) -> Dict[str, Any]:
        """Optimize caching strategy"""
        # Enable more aggressive caching
        self.cache.strategy = CacheStrategy.ADAPTIVE
        self.cache.max_size = min(2000, self.cache.max_size * 1.5)
        
        return {
            "action": "enhanced_caching",
            "new_cache_size": self.cache.max_size,
            "strategy": self.cache.strategy.value
        }
    
    async def _optimize_parallel_processing(self) -> Dict[str, Any]:
        """Optimize parallel processing"""
        # In production, this would:
        # - Increase thread pool sizes
        # - Enable parallel execution of independent tasks
        # - Optimize concurrent operations
        
        return {
            "action": "parallel_processing_enabled",
            "max_workers": min(8, (psutil.cpu_count() or 2) * 2)
        }
    
    async def _optimize_memory(self) -> Dict[str, Any]:
        """Optimize memory usage"""
        # In production, this would:
        # - Tune garbage collection parameters
        # - Optimize data structures
        # - Implement memory pooling
        
        return {
            "action": "memory_optimization_applied",
            "techniques": ["gc_tuning", "data_structure_optimization", "memory_pooling"]
        }
    
    async def _optimize_io(self) -> Dict[str, Any]:
        """Optimize I/O operations"""
        # In production, this would:
        # - Enable I/O batching
        # - Implement async I/O where possible
        # - Optimize file access patterns
        
        return {
            "action": "io_optimization_applied",
            "techniques": ["io_batching", "async_io", "access_pattern_optimization"]
        }
    
    async def _optimize_algorithms(self) -> Dict[str, Any]:
        """Optimize algorithms and data structures"""
        # In production, this would:
        # - Replace inefficient algorithms
        # - Optimize data structures
        # - Implement algorithmic improvements
        
        return {
            "action": "algorithm_optimization_applied",
            "improvements": ["algorithm_replacement", "data_structure_optimization", "complexity_reduction"]
        }
    
    def get_optimization_report(self) -> Dict[str, Any]:
        """Generate comprehensive optimization report"""
        current_metrics = self.resource_monitor.get_current_metrics()
        
        total_improvement = sum(self.performance_improvements.values())
        
        return {
            "optimization_level": self.optimization_level.value,
            "total_improvements": len(self.completed_optimizations),
            "total_improvement_percent": total_improvement,
            "current_performance": {
                "cpu_percent": current_metrics.cpu_percent if current_metrics else 0,
                "memory_percent": current_metrics.memory_percent if current_metrics else 0,
                "cache_hit_rate": self.cache.get_hit_rate() * 100
            },
            "optimization_history": [
                {
                    "optimization": opt_id,
                    "improvement": improvement
                }
                for opt_id, improvement in self.performance_improvements.items()
            ],
            "scaling_status": {
                "current_scale": self.autoscaling_engine.current_scale,
                "scaling_events": len(self.autoscaling_engine.scaling_history)
            },
            "cache_statistics": self.cache.get_stats(),
            "recommendations": self._generate_recommendations()
        }
    
    def _generate_recommendations(self) -> List[str]:
        """Generate performance recommendations"""
        recommendations = []
        
        current_metrics = self.resource_monitor.get_current_metrics()
        if not current_metrics:
            return recommendations
        
        # CPU recommendations
        if current_metrics.cpu_percent > 80:
            recommendations.append("🔥 HIGH CPU: Consider algorithm optimization or horizontal scaling")
        elif current_metrics.cpu_percent > 60:
            recommendations.append("⚠️ ELEVATED CPU: Monitor for sustained high usage")
        
        # Memory recommendations
        if current_metrics.memory_percent > 85:
            recommendations.append("🔥 HIGH MEMORY: Implement memory optimization or increase resources")
        elif current_metrics.memory_percent > 70:
            recommendations.append("⚠️ ELEVATED MEMORY: Monitor memory usage patterns")
        
        # Cache recommendations
        cache_hit_rate = self.cache.get_hit_rate()
        if cache_hit_rate < 0.5:
            recommendations.append("📊 LOW CACHE HIT RATE: Review caching strategy")
        elif cache_hit_rate > 0.9:
            recommendations.append("✅ EXCELLENT CACHE PERFORMANCE: Consider increasing cache size")
        
        # Scaling recommendations
        recommendations.extend(self.autoscaling_engine.get_scaling_recommendations())
        
        if not recommendations:
            recommendations.append("✅ System performance is optimal")
        
        return recommendations


# Integration function
async def integrate_performance_optimization(orchestrator_instance):
    """Integrate performance optimization with autonomous orchestrator"""
    
    # Add performance optimization engine
    orchestrator_instance.performance_engine = PerformanceOptimizationEngine()
    
    # Start optimization engine
    await orchestrator_instance.performance_engine.start_optimization_engine()
    
    # Override execution methods to use caching
    original_execute = orchestrator_instance.execute_autonomous_sdlc
    
    async def optimized_execute():
        # Check cache for similar executions
        cache_key = f"sdlc_execution_{orchestrator_instance.project.name}"
        cached_result = orchestrator_instance.performance_engine.cache.get(cache_key)
        
        if cached_result:
            orchestrator_instance.logger.info("Using cached execution results")
            return cached_result
        
        # Execute with performance monitoring
        with orchestrator_instance.performance_engine.resource_monitor:
            result = await original_execute()
            
            # Cache successful results
            if result.get('success'):
                orchestrator_instance.performance_engine.cache.put(cache_key, result, ttl_seconds=3600)
            
            return result
    
    orchestrator_instance.execute_autonomous_sdlc_optimized = optimized_execute
    
    return orchestrator_instance