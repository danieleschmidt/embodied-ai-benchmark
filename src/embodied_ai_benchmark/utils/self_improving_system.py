"""
Self-Improving System with Machine Learning and Adaptive Optimization
"""

import json
import logging
import pickle
import threading
import time
from typing import Dict, List, Optional, Any, Callable, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
import numpy as np
from collections import deque, defaultdict
import statistics
import hashlib

# ML imports
try:
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.linear_model import LinearRegression
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False

logger = logging.getLogger(__name__)


class LearningDomain(Enum):
    """Domains for machine learning and optimization."""
    PERFORMANCE = "performance"
    RESOURCE_ALLOCATION = "resource_allocation"
    ERROR_PREDICTION = "error_prediction"
    USER_BEHAVIOR = "user_behavior"
    SYSTEM_OPTIMIZATION = "system_optimization"
    WORKLOAD_PREDICTION = "workload_prediction"


@dataclass
class LearningDataPoint:
    """Single data point for machine learning."""
    timestamp: datetime
    domain: LearningDomain
    features: Dict[str, float]
    target: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    

@dataclass
class OptimizationResult:
    """Result of an optimization attempt."""
    optimization_id: str
    timestamp: datetime
    domain: LearningDomain
    parameters_changed: Dict[str, Any]
    baseline_metric: float
    optimized_metric: float
    improvement_percentage: float
    successful: bool
    

class AdaptiveLearningEngine:
    """Machine learning engine for continuous system improvement."""
    
    def __init__(self, enable_ml: bool = None):
        """Initialize learning engine.
        
        Args:
            enable_ml: Whether to enable ML features (auto-detect if None)
        """
        self.enable_ml = enable_ml if enable_ml is not None else ML_AVAILABLE
        self.models: Dict[LearningDomain, Any] = {}
        self.scalers: Dict[LearningDomain, Any] = {}
        
        # Training data
        self.training_data: Dict[LearningDomain, List[LearningDataPoint]] = defaultdict(list)
        self.model_versions: Dict[LearningDomain, int] = defaultdict(int)
        
        # Performance tracking
        self.model_performance: Dict[LearningDomain, List[float]] = defaultdict(list)
        self.optimization_history: List[OptimizationResult] = []
        
        # Configuration
        self.min_training_samples = 100
        self.retrain_threshold = 1000  # Retrain after this many new samples
        self.model_accuracy_threshold = 0.7
        
        if not self.enable_ml:
            logger.warning("Machine learning features disabled - sklearn not available")
            
    def add_training_data(self, 
                         domain: LearningDomain,
                         features: Dict[str, float],
                         target: float,
                         metadata: Dict[str, Any] = None) -> None:
        """Add a training data point.
        
        Args:
            domain: Learning domain
            features: Feature values
            target: Target value to predict
            metadata: Additional metadata
        """
        data_point = LearningDataPoint(
            timestamp=datetime.now(),
            domain=domain,
            features=features,
            target=target,
            metadata=metadata or {}
        )
        
        self.training_data[domain].append(data_point)
        
        # Trigger retraining if we have enough data
        if (len(self.training_data[domain]) >= self.min_training_samples and
            len(self.training_data[domain]) % self.retrain_threshold == 0):
            self._retrain_model(domain)
            
    def predict(self, domain: LearningDomain, features: Dict[str, float]) -> Optional[float]:
        """Make a prediction using the trained model.
        
        Args:
            domain: Learning domain
            features: Feature values
            
        Returns:
            Predicted value or None if model not available
        """
        if not self.enable_ml or domain not in self.models:
            return None
            
        try:
            # Convert features to array
            feature_array = self._features_to_array(domain, features)
            
            # Scale features
            if domain in self.scalers:
                feature_array = self.scalers[domain].transform([feature_array])
            else:
                feature_array = [feature_array]
                
            # Make prediction
            prediction = self.models[domain].predict(feature_array)[0]
            return float(prediction)
            
        except Exception as e:
            logger.error(f"Prediction failed for domain {domain}: {e}")
            return None
            
    def get_feature_importance(self, domain: LearningDomain) -> Dict[str, float]:
        """Get feature importance scores for a domain.
        
        Args:
            domain: Learning domain
            
        Returns:
            Dictionary of feature names to importance scores
        """
        if not self.enable_ml or domain not in self.models:
            return {}
            
        model = self.models[domain]
        
        # Get feature names from training data
        if not self.training_data[domain]:
            return {}
            
        feature_names = list(self.training_data[domain][0].features.keys())
        
        try:
            if hasattr(model, 'feature_importances_'):
                importances = model.feature_importances_
                return dict(zip(feature_names, importances))
            else:
                # For linear models, use coefficient magnitudes
                if hasattr(model, 'coef_'):
                    importances = np.abs(model.coef_)
                    return dict(zip(feature_names, importances))
        except Exception as e:
            logger.error(f"Failed to get feature importance for {domain}: {e}")
            
        return {}
        
    def _retrain_model(self, domain: LearningDomain) -> None:
        """Retrain the model for a specific domain."""
        if not self.enable_ml:
            return
            
        data_points = self.training_data[domain]
        if len(data_points) < self.min_training_samples:
            return
            
        try:
            # Prepare training data
            features = []
            targets = []
            
            for dp in data_points:
                feature_array = self._features_to_array(domain, dp.features)
                features.append(feature_array)
                targets.append(dp.target)
                
            features = np.array(features)
            targets = np.array(targets)
            
            # Scale features
            scaler = StandardScaler()
            features_scaled = scaler.fit_transform(features)
            self.scalers[domain] = scaler
            
            # Choose model based on domain
            if domain in [LearningDomain.PERFORMANCE, LearningDomain.WORKLOAD_PREDICTION]:
                model = RandomForestRegressor(n_estimators=100, random_state=42)
            else:
                model = LinearRegression()
                
            # Train model
            model.fit(features_scaled, targets)
            self.models[domain] = model
            self.model_versions[domain] += 1
            
            # Evaluate model performance
            score = model.score(features_scaled, targets)
            self.model_performance[domain].append(score)
            
            logger.info(f"Retrained model for {domain.value} - Score: {score:.3f}")
            
        except Exception as e:
            logger.error(f"Model retraining failed for {domain}: {e}")
            
    def _features_to_array(self, domain: LearningDomain, features: Dict[str, float]) -> np.ndarray:
        """Convert feature dictionary to numpy array."""
        # Get consistent feature ordering from training data
        if self.training_data[domain]:
            reference_features = self.training_data[domain][0].features
            feature_names = sorted(reference_features.keys())
        else:
            feature_names = sorted(features.keys())
            
        return np.array([features.get(name, 0.0) for name in feature_names])
        
    def get_learning_statistics(self) -> Dict[str, Any]:
        """Get comprehensive learning statistics."""
        stats = {
            "ml_enabled": self.enable_ml,
            "domains": {},
            "optimization_history": len(self.optimization_history)
        }
        
        for domain in LearningDomain:
            domain_stats = {
                "training_samples": len(self.training_data[domain]),
                "model_trained": domain in self.models,
                "model_version": self.model_versions[domain],
                "latest_score": (self.model_performance[domain][-1] 
                               if self.model_performance[domain] else None)
            }
            
            if domain in self.models:
                domain_stats["feature_importance"] = self.get_feature_importance(domain)
                
            stats["domains"][domain.value] = domain_stats
            
        return stats


class SelfOptimizingSystem:
    """System that automatically optimizes its own parameters."""
    
    def __init__(self, learning_engine: AdaptiveLearningEngine):
        """Initialize self-optimizing system.
        
        Args:
            learning_engine: Learning engine instance
        """
        self.learning_engine = learning_engine
        self.optimization_parameters: Dict[str, Any] = {}
        self.parameter_bounds: Dict[str, Tuple[float, float]] = {}
        self.optimization_callbacks: Dict[str, Callable] = {}
        
        # Optimization state
        self.baseline_metrics: Dict[str, float] = {}
        self.current_experiments: Dict[str, Dict] = {}
        self.optimization_lock = threading.Lock()
        
        # Configuration
        self.optimization_interval = 3600  # 1 hour
        self.experiment_duration = 300  # 5 minutes
        self.improvement_threshold = 0.05  # 5% improvement required
        
        self._setup_default_parameters()
        
    def _setup_default_parameters(self) -> None:
        """Setup default optimization parameters."""
        self.optimization_parameters = {
            "batch_size": 32,
            "worker_threads": 4,
            "cache_size": 1000,
            "timeout_seconds": 30,
            "retry_attempts": 3,
            "memory_limit_mb": 2048
        }
        
        self.parameter_bounds = {
            "batch_size": (8, 128),
            "worker_threads": (1, 16),
            "cache_size": (100, 10000),
            "timeout_seconds": (10, 120),
            "retry_attempts": (1, 10),
            "memory_limit_mb": (512, 8192)
        }
        
    def register_optimization_callback(self, 
                                     parameter: str, 
                                     callback: Callable[[Any], None]) -> None:
        """Register a callback for parameter changes.
        
        Args:
            parameter: Parameter name
            callback: Function to call when parameter changes
        """
        self.optimization_callbacks[parameter] = callback
        
    def suggest_optimization(self, 
                           metric_name: str, 
                           current_value: float,
                           features: Dict[str, float]) -> Dict[str, Any]:
        """Suggest parameter optimizations based on current performance.
        
        Args:
            metric_name: Name of the metric to optimize
            current_value: Current metric value
            features: Current system features
            
        Returns:
            Optimization suggestions
        """
        suggestions = {}
        
        # Use ML predictions if available
        predicted_performance = self.learning_engine.predict(
            LearningDomain.PERFORMANCE, features
        )
        
        if predicted_performance is not None:
            # If predicted performance is worse than current, suggest optimizations
            if predicted_performance < current_value * 0.95:  # 5% degradation threshold
                suggestions = self._generate_ml_suggestions(features)
        else:
            # Fall back to heuristic optimizations
            suggestions = self._generate_heuristic_suggestions(metric_name, current_value, features)
            
        return suggestions
        
    def _generate_ml_suggestions(self, features: Dict[str, float]) -> Dict[str, Any]:
        """Generate optimization suggestions using ML feature importance."""
        suggestions = {}
        
        # Get feature importance for performance domain
        importance = self.learning_engine.get_feature_importance(LearningDomain.PERFORMANCE)
        
        # Sort features by importance
        sorted_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)
        
        # Suggest adjustments for top important features
        for feature_name, importance_score in sorted_features[:3]:  # Top 3 features
            if feature_name in self.optimization_parameters:
                current_value = self.optimization_parameters[feature_name]
                bounds = self.parameter_bounds.get(feature_name)
                
                if bounds:
                    min_val, max_val = bounds
                    
                    # Suggest increase or decrease based on heuristics
                    if importance_score > 0.1:  # High importance
                        if isinstance(current_value, (int, float)):
                            # Try increasing by 20%
                            new_value = min(max_val, current_value * 1.2)
                            suggestions[feature_name] = new_value
                            
        return suggestions
        
    def _generate_heuristic_suggestions(self, 
                                      metric_name: str, 
                                      current_value: float,
                                      features: Dict[str, float]) -> Dict[str, Any]:
        """Generate optimization suggestions using heuristics."""
        suggestions = {}
        
        # CPU utilization based optimizations
        cpu_percent = features.get("cpu_percent", 50)
        if cpu_percent > 80:
            # High CPU - reduce parallelism
            current_threads = self.optimization_parameters.get("worker_threads", 4)
            suggestions["worker_threads"] = max(1, current_threads - 1)
            
        elif cpu_percent < 30:
            # Low CPU - increase parallelism
            current_threads = self.optimization_parameters.get("worker_threads", 4)
            max_threads = self.parameter_bounds["worker_threads"][1]
            suggestions["worker_threads"] = min(max_threads, current_threads + 1)
            
        # Memory utilization based optimizations
        memory_percent = features.get("memory_percent", 50)
        if memory_percent > 85:
            # High memory - reduce cache size
            current_cache = self.optimization_parameters.get("cache_size", 1000)
            suggestions["cache_size"] = max(100, int(current_cache * 0.8))
            
        elif memory_percent < 40:
            # Low memory - increase cache size
            current_cache = self.optimization_parameters.get("cache_size", 1000)
            max_cache = self.parameter_bounds["cache_size"][1]
            suggestions["cache_size"] = min(max_cache, int(current_cache * 1.2))
            
        # Throughput based optimizations
        throughput = features.get("task_throughput", 10)
        if throughput < 5:
            # Low throughput - increase batch size
            current_batch = self.optimization_parameters.get("batch_size", 32)
            max_batch = self.parameter_bounds["batch_size"][1]
            suggestions["batch_size"] = min(max_batch, current_batch * 2)
            
        return suggestions
        
    def apply_optimization(self, parameter: str, new_value: Any) -> bool:
        """Apply an optimization to a parameter.
        
        Args:
            parameter: Parameter name
            new_value: New parameter value
            
        Returns:
            True if optimization was applied successfully
        """
        with self.optimization_lock:
            if parameter not in self.optimization_parameters:
                logger.error(f"Unknown optimization parameter: {parameter}")
                return False
                
            # Validate bounds
            if parameter in self.parameter_bounds:
                min_val, max_val = self.parameter_bounds[parameter]
                if not min_val <= new_value <= max_val:
                    logger.error(f"Parameter {parameter} value {new_value} outside bounds [{min_val}, {max_val}]")
                    return False
                    
            # Store old value for rollback
            old_value = self.optimization_parameters[parameter]
            
            try:
                # Apply the change
                self.optimization_parameters[parameter] = new_value
                
                # Call registered callback
                if parameter in self.optimization_callbacks:
                    self.optimization_callbacks[parameter](new_value)
                    
                logger.info(f"Applied optimization: {parameter} = {new_value} (was {old_value})")
                return True
                
            except Exception as e:
                # Rollback on error
                self.optimization_parameters[parameter] = old_value
                logger.error(f"Failed to apply optimization for {parameter}: {e}")
                return False
                
    def start_autonomous_optimization(self) -> None:
        """Start autonomous optimization in background thread."""
        def optimization_loop():
            while True:
                try:
                    self._run_optimization_cycle()
                    time.sleep(self.optimization_interval)
                except Exception as e:
                    logger.error(f"Error in optimization cycle: {e}")
                    time.sleep(60)  # Wait 1 minute before retrying
                    
        optimization_thread = threading.Thread(target=optimization_loop, daemon=True)
        optimization_thread.start()
        logger.info("Started autonomous optimization")
        
    def _run_optimization_cycle(self) -> None:
        """Run a single optimization cycle."""
        logger.info("Starting optimization cycle")
        
        # Collect current metrics (this would be implemented based on your monitoring system)
        current_metrics = self._collect_current_metrics()
        
        # Generate optimization suggestions
        suggestions = self.suggest_optimization(
            "overall_performance",
            current_metrics.get("performance_score", 0.5),
            current_metrics
        )
        
        if not suggestions:
            logger.info("No optimization suggestions generated")
            return
            
        # Test each suggestion
        for parameter, suggested_value in suggestions.items():
            self._test_optimization(parameter, suggested_value, current_metrics)
            
    def _collect_current_metrics(self) -> Dict[str, float]:
        """Collect current system metrics."""
        # This would interface with your monitoring system
        # For now, return mock data
        return {
            "performance_score": 0.75,
            "cpu_percent": 45.0,
            "memory_percent": 60.0,
            "task_throughput": 12.0,
            "error_rate": 2.5
        }
        
    def _test_optimization(self, 
                          parameter: str, 
                          suggested_value: Any,
                          baseline_metrics: Dict[str, float]) -> None:
        """Test an optimization suggestion."""
        experiment_id = f"{parameter}_{int(time.time())}"
        
        logger.info(f"Testing optimization: {parameter} = {suggested_value}")
        
        # Apply optimization
        if not self.apply_optimization(parameter, suggested_value):
            return
            
        # Wait for system to stabilize
        time.sleep(30)
        
        # Collect metrics during experiment
        experiment_metrics = []
        for _ in range(self.experiment_duration // 30):  # Collect every 30 seconds
            metrics = self._collect_current_metrics()
            experiment_metrics.append(metrics)
            time.sleep(30)
            
        # Calculate average performance during experiment
        avg_performance = statistics.mean(
            m.get("performance_score", 0) for m in experiment_metrics
        )
        
        baseline_performance = baseline_metrics.get("performance_score", 0)
        improvement = (avg_performance - baseline_performance) / baseline_performance
        
        # Decide whether to keep the optimization
        if improvement >= self.improvement_threshold:
            logger.info(f"Optimization successful: {parameter} = {suggested_value} "
                       f"(improvement: {improvement:.2%})")
            
            # Record successful optimization
            result = OptimizationResult(
                optimization_id=experiment_id,
                timestamp=datetime.now(),
                domain=LearningDomain.SYSTEM_OPTIMIZATION,
                parameters_changed={parameter: suggested_value},
                baseline_metric=baseline_performance,
                optimized_metric=avg_performance,
                improvement_percentage=improvement * 100,
                successful=True
            )
            
        else:
            logger.info(f"Optimization unsuccessful: {parameter} = {suggested_value} "
                       f"(improvement: {improvement:.2%})")
            
            # Rollback the optimization
            old_value = self._get_default_value(parameter)
            self.apply_optimization(parameter, old_value)
            
            result = OptimizationResult(
                optimization_id=experiment_id,
                timestamp=datetime.now(),
                domain=LearningDomain.SYSTEM_OPTIMIZATION,
                parameters_changed={parameter: suggested_value},
                baseline_metric=baseline_performance,
                optimized_metric=avg_performance,
                improvement_percentage=improvement * 100,
                successful=False
            )
            
        self.learning_engine.optimization_history.append(result)
        
        # Add training data for future learning
        features = baseline_metrics.copy()
        features[f"{parameter}_setting"] = suggested_value
        
        self.learning_engine.add_training_data(
            LearningDomain.SYSTEM_OPTIMIZATION,
            features,
            avg_performance,
            {"experiment_id": experiment_id, "parameter": parameter}
        )
        
    def _get_default_value(self, parameter: str) -> Any:
        """Get the default value for a parameter."""
        defaults = {
            "batch_size": 32,
            "worker_threads": 4,
            "cache_size": 1000,
            "timeout_seconds": 30,
            "retry_attempts": 3,
            "memory_limit_mb": 2048
        }
        return defaults.get(parameter, self.optimization_parameters.get(parameter))
        
    def get_optimization_status(self) -> Dict[str, Any]:
        """Get current optimization status."""
        recent_optimizations = [
            opt for opt in self.learning_engine.optimization_history
            if (datetime.now() - opt.timestamp).days < 7  # Last week
        ]
        
        successful_optimizations = [opt for opt in recent_optimizations if opt.successful]
        
        return {
            "current_parameters": self.optimization_parameters.copy(),
            "total_optimizations": len(self.learning_engine.optimization_history),
            "recent_optimizations": len(recent_optimizations),
            "successful_optimizations": len(successful_optimizations),
            "success_rate": (len(successful_optimizations) / max(1, len(recent_optimizations))) * 100,
            "learning_statistics": self.learning_engine.get_learning_statistics()
        }


# Global instances
_learning_engine: Optional[AdaptiveLearningEngine] = None
_self_optimizing_system: Optional[SelfOptimizingSystem] = None


def get_learning_engine() -> AdaptiveLearningEngine:
    """Get global learning engine instance."""
    global _learning_engine
    if _learning_engine is None:
        _learning_engine = AdaptiveLearningEngine()
    return _learning_engine


def get_self_optimizing_system() -> SelfOptimizingSystem:
    """Get global self-optimizing system instance."""
    global _self_optimizing_system
    if _self_optimizing_system is None:
        learning_engine = get_learning_engine()
        _self_optimizing_system = SelfOptimizingSystem(learning_engine)
    return _self_optimizing_system


def start_self_improvement() -> None:
    """Start the self-improvement system."""
    system = get_self_optimizing_system()
    system.start_autonomous_optimization()
    logger.info("Self-improvement system started")


def add_performance_data(features: Dict[str, float], performance: float) -> None:
    """Add performance data for learning."""
    engine = get_learning_engine()
    engine.add_training_data(LearningDomain.PERFORMANCE, features, performance)


def predict_performance(features: Dict[str, float]) -> Optional[float]:
    """Predict performance based on features."""
    engine = get_learning_engine()
    return engine.predict(LearningDomain.PERFORMANCE, features)