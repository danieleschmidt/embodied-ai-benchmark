"""Robust Validation Framework for Research Components."""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Any, Union, Callable
from dataclasses import dataclass, field
import time
import json
import traceback
import logging
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing
from functools import wraps
import hashlib
import pickle

from ..utils.logging_config import get_logger
from ..utils.error_handling import handle_errors, RetryableError, FatalError

logger = get_logger(__name__)


@dataclass
class ValidationConfig:
    """Configuration for robust validation framework."""
    max_retries: int = 3
    timeout_seconds: float = 300.0
    memory_limit_gb: float = 8.0
    gpu_memory_limit_gb: float = 4.0
    error_threshold: float = 0.1
    performance_threshold: float = 0.05
    statistical_significance: float = 0.05
    min_sample_size: int = 30
    validation_splits: int = 5
    enable_monitoring: bool = True
    save_checkpoints: bool = True
    checkpoint_dir: str = "validation_checkpoints"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class ValidationResult:
    """Result from validation process."""
    component_name: str
    test_name: str
    success: bool
    execution_time: float
    memory_usage_mb: float
    error_message: Optional[str] = None
    performance_metrics: Dict[str, float] = field(default_factory=dict)
    statistical_results: Dict[str, Any] = field(default_factory=dict)
    reproducibility_hash: Optional[str] = None
    checkpoint_path: Optional[str] = None


class ResourceMonitor:
    """Monitor system resources during validation."""
    
    def __init__(self, config: ValidationConfig):
        self.config = config
        self.start_time = None
        self.peak_memory = 0.0
        self.peak_gpu_memory = 0.0
        self.monitoring_active = False
        
    def __enter__(self):
        """Start monitoring resources."""
        self.start_time = time.time()
        self.peak_memory = self._get_memory_usage()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        self.monitoring_active = True
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Stop monitoring and check limits."""
        self.monitoring_active = False
        
        # Check memory limits
        if self.peak_memory > self.config.memory_limit_gb * 1024:
            logger.warning(f"Memory usage exceeded limit: {self.peak_memory:.1f}MB > {self.config.memory_limit_gb*1024:.1f}MB")
        
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.max_memory_allocated() / 1024**2
            if gpu_memory > self.config.gpu_memory_limit_gb * 1024:
                logger.warning(f"GPU memory usage exceeded limit: {gpu_memory:.1f}MB > {self.config.gpu_memory_limit_gb*1024:.1f}MB")
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        try:
            import psutil
            process = psutil.Process()
            memory_mb = process.memory_info().rss / 1024**2
            self.peak_memory = max(self.peak_memory, memory_mb)
            return memory_mb
        except ImportError:
            # Fallback if psutil not available
            return 0.0
    
    def get_current_usage(self) -> Dict[str, float]:
        """Get current resource usage."""
        usage = {
            'memory_mb': self._get_memory_usage(),
            'elapsed_time': time.time() - self.start_time if self.start_time else 0
        }
        
        if torch.cuda.is_available():
            usage['gpu_memory_mb'] = torch.cuda.memory_allocated() / 1024**2
            usage['gpu_memory_cached_mb'] = torch.cuda.memory_reserved() / 1024**2
        
        return usage
    
    def check_limits(self) -> bool:
        """Check if resource limits are exceeded."""
        usage = self.get_current_usage()
        
        if usage['memory_mb'] > self.config.memory_limit_gb * 1024:
            return False
        
        if usage['elapsed_time'] > self.config.timeout_seconds:
            return False
        
        if torch.cuda.is_available() and usage['gpu_memory_mb'] > self.config.gpu_memory_limit_gb * 1024:
            return False
        
        return True


def retry_on_failure(max_retries: int = 3, delay: float = 1.0):
    """Decorator for retrying failed operations."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except RetryableError as e:
                    last_exception = e
                    if attempt < max_retries:
                        logger.warning(f"Attempt {attempt + 1} failed, retrying in {delay}s: {e}")
                        time.sleep(delay * (2 ** attempt))  # Exponential backoff
                    else:
                        logger.error(f"Max retries exceeded for {func.__name__}")
                except FatalError as e:
                    logger.error(f"Fatal error in {func.__name__}: {e}")
                    raise
                except Exception as e:
                    last_exception = e
                    if attempt < max_retries:
                        logger.warning(f"Unexpected error in attempt {attempt + 1}, retrying: {e}")
                        time.sleep(delay)
                    else:
                        logger.error(f"Max retries exceeded for {func.__name__}")
            
            raise last_exception if last_exception else Exception(f"Failed after {max_retries} retries")
        
        return wrapper
    return decorator


class CheckpointManager:
    """Manage validation checkpoints for reproducibility."""
    
    def __init__(self, config: ValidationConfig):
        self.config = config
        self.checkpoint_dir = Path(config.checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True)
        
    def save_checkpoint(self, component_name: str, state: Dict[str, Any]) -> str:
        """Save validation checkpoint."""
        if not self.config.save_checkpoints:
            return ""
        
        timestamp = int(time.time())
        checkpoint_path = self.checkpoint_dir / f"{component_name}_{timestamp}.pkl"
        
        try:
            with open(checkpoint_path, 'wb') as f:
                pickle.dump(state, f)
            
            logger.info(f"Checkpoint saved: {checkpoint_path}")
            return str(checkpoint_path)
            
        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")
            return ""
    
    def load_checkpoint(self, checkpoint_path: str) -> Optional[Dict[str, Any]]:
        """Load validation checkpoint."""
        try:
            with open(checkpoint_path, 'rb') as f:
                state = pickle.load(f)
            
            logger.info(f"Checkpoint loaded: {checkpoint_path}")
            return state
            
        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
            return None
    
    def get_reproducibility_hash(self, inputs: Dict[str, Any]) -> str:
        """Generate reproducibility hash for inputs."""
        try:
            # Create deterministic hash of inputs
            serialized = json.dumps(inputs, sort_keys=True, default=str)
            return hashlib.sha256(serialized.encode()).hexdigest()[:16]
        except Exception:
            return "unknown"


class StatisticalValidator:
    """Statistical validation for research results."""
    
    def __init__(self, config: ValidationConfig):
        self.config = config
        
    def validate_statistical_significance(self, 
                                        baseline_results: List[float],
                                        novel_results: List[float],
                                        test_name: str = "t_test") -> Dict[str, Any]:
        """Validate statistical significance of results."""
        try:
            from scipy import stats
            
            # Ensure minimum sample size
            if len(baseline_results) < self.config.min_sample_size or len(novel_results) < self.config.min_sample_size:
                logger.warning(f"Sample size too small for reliable statistical test")
                return {
                    'significant': False,
                    'p_value': 1.0,
                    'effect_size': 0.0,
                    'warning': 'Insufficient sample size'
                }
            
            # Remove outliers
            baseline_clean = self._remove_outliers(baseline_results)
            novel_clean = self._remove_outliers(novel_results)
            
            # Choose appropriate test
            if test_name == "t_test":
                # Check normality first
                _, p_baseline = stats.shapiro(baseline_clean)
                _, p_novel = stats.shapiro(novel_clean)
                
                if p_baseline < 0.05 or p_novel < 0.05:
                    # Use Mann-Whitney U test for non-normal data
                    statistic, p_value = stats.mannwhitneyu(baseline_clean, novel_clean, alternative='two-sided')
                    test_used = "mann_whitney_u"
                else:
                    # Use t-test for normal data
                    statistic, p_value = stats.ttest_ind(baseline_clean, novel_clean)
                    test_used = "t_test"
            
            # Compute effect size (Cohen's d)
            effect_size = self._compute_cohens_d(baseline_clean, novel_clean)
            
            # Compute confidence interval
            confidence_interval = self._compute_confidence_interval(baseline_clean, novel_clean)
            
            return {
                'significant': p_value < self.config.statistical_significance,
                'p_value': p_value,
                'effect_size': effect_size,
                'test_statistic': statistic,
                'test_used': test_used,
                'confidence_interval': confidence_interval,
                'sample_sizes': (len(baseline_clean), len(novel_clean)),
                'baseline_mean': np.mean(baseline_clean),
                'novel_mean': np.mean(novel_clean),
                'improvement': (np.mean(novel_clean) - np.mean(baseline_clean)) / np.mean(baseline_clean) * 100
            }
            
        except Exception as e:
            logger.error(f"Statistical validation failed: {e}")
            return {
                'significant': False,
                'p_value': 1.0,
                'effect_size': 0.0,
                'error': str(e)
            }
    
    def _remove_outliers(self, data: List[float], method: str = "iqr") -> List[float]:
        """Remove outliers from data."""
        data_array = np.array(data)
        
        if method == "iqr":
            Q1 = np.percentile(data_array, 25)
            Q3 = np.percentile(data_array, 75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            mask = (data_array >= lower_bound) & (data_array <= upper_bound)
            return data_array[mask].tolist()
        
        elif method == "zscore":
            z_scores = np.abs((data_array - np.mean(data_array)) / np.std(data_array))
            mask = z_scores < 3  # 3 standard deviations
            return data_array[mask].tolist()
        
        return data
    
    def _compute_cohens_d(self, baseline: List[float], novel: List[float]) -> float:
        """Compute Cohen's d effect size."""
        n1, n2 = len(baseline), len(novel)
        s1, s2 = np.std(baseline, ddof=1), np.std(novel, ddof=1)
        
        # Pooled standard deviation
        pooled_std = np.sqrt(((n1 - 1) * s1**2 + (n2 - 1) * s2**2) / (n1 + n2 - 2))
        
        return (np.mean(novel) - np.mean(baseline)) / pooled_std
    
    def _compute_confidence_interval(self, baseline: List[float], novel: List[float], 
                                   confidence: float = 0.95) -> Tuple[float, float]:
        """Compute confidence interval for difference in means."""
        from scipy import stats
        
        diff = np.mean(novel) - np.mean(baseline)
        n1, n2 = len(baseline), len(novel)
        s1, s2 = np.std(baseline, ddof=1), np.std(novel, ddof=1)
        
        # Standard error of difference
        se_diff = np.sqrt(s1**2/n1 + s2**2/n2)
        
        # Degrees of freedom (Welch's formula)
        df = (s1**2/n1 + s2**2/n2)**2 / (s1**4/(n1**2*(n1-1)) + s2**4/(n2**2*(n2-1)))
        
        # Critical value
        alpha = 1 - confidence
        t_critical = stats.t.ppf(1 - alpha/2, df)
        
        # Confidence interval
        margin = t_critical * se_diff
        return (diff - margin, diff + margin)


class PerformanceProfiler:
    """Profile performance of research components."""
    
    def __init__(self, config: ValidationConfig):
        self.config = config
        self.profiles = {}
        
    def profile_component(self, component_name: str, 
                         component_func: Callable,
                         test_inputs: List[Any],
                         num_warmup: int = 5) -> Dict[str, float]:
        """Profile component performance."""
        logger.info(f"Profiling component: {component_name}")
        
        # Warmup runs
        for i in range(min(num_warmup, len(test_inputs))):
            try:
                component_func(test_inputs[i])
            except Exception as e:
                logger.warning(f"Warmup run {i} failed: {e}")
        
        # Actual profiling
        execution_times = []
        memory_usages = []
        gpu_times = []
        
        for test_input in test_inputs:
            with ResourceMonitor(self.config) as monitor:
                start_time = time.time()
                
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                    gpu_start = time.time()
                
                try:
                    result = component_func(test_input)
                    
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                        gpu_end = time.time()
                        gpu_times.append(gpu_end - gpu_start)
                    
                    end_time = time.time()
                    execution_times.append(end_time - start_time)
                    
                    usage = monitor.get_current_usage()
                    memory_usages.append(usage['memory_mb'])
                    
                except Exception as e:
                    logger.error(f"Profiling failed for input: {e}")
                    continue
        
        if not execution_times:
            return {'error': 'All profiling runs failed'}
        
        # Compute statistics
        profile_stats = {
            'mean_execution_time': np.mean(execution_times),
            'std_execution_time': np.std(execution_times),
            'median_execution_time': np.median(execution_times),
            'p95_execution_time': np.percentile(execution_times, 95),
            'p99_execution_time': np.percentile(execution_times, 99),
            'mean_memory_mb': np.mean(memory_usages),
            'peak_memory_mb': np.max(memory_usages),
            'throughput_ops_per_sec': len(execution_times) / sum(execution_times)
        }
        
        if gpu_times:
            profile_stats.update({
                'mean_gpu_time': np.mean(gpu_times),
                'gpu_utilization': np.mean(gpu_times) / np.mean(execution_times)
            })
        
        self.profiles[component_name] = profile_stats
        
        return profile_stats
    
    def compare_performance(self, baseline_name: str, novel_name: str) -> Dict[str, Any]:
        """Compare performance between baseline and novel components."""
        if baseline_name not in self.profiles or novel_name not in self.profiles:
            return {'error': 'Missing profile data for comparison'}
        
        baseline = self.profiles[baseline_name]
        novel = self.profiles[novel_name]
        
        comparison = {}
        
        # Compare key metrics
        for metric in ['mean_execution_time', 'mean_memory_mb', 'throughput_ops_per_sec']:
            if metric in baseline and metric in novel:
                baseline_val = baseline[metric]
                novel_val = novel[metric]
                
                if metric == 'mean_execution_time' or metric == 'mean_memory_mb':
                    # Lower is better
                    improvement = (baseline_val - novel_val) / baseline_val * 100
                else:
                    # Higher is better
                    improvement = (novel_val - baseline_val) / baseline_val * 100
                
                comparison[f'{metric}_improvement'] = improvement
                comparison[f'{metric}_speedup'] = baseline_val / novel_val if novel_val > 0 else float('inf')
        
        return comparison


class RobustValidationFramework:
    """Main framework for robust validation of research components."""
    
    def __init__(self, config: ValidationConfig):
        self.config = config
        self.resource_monitor = ResourceMonitor(config)
        self.checkpoint_manager = CheckpointManager(config)
        self.statistical_validator = StatisticalValidator(config)
        self.performance_profiler = PerformanceProfiler(config)
        self.validation_results = []
        
    @retry_on_failure(max_retries=3)
    def validate_component(self, 
                          component_name: str,
                          component_func: Callable,
                          test_suite: Dict[str, Callable],
                          baseline_func: Optional[Callable] = None) -> List[ValidationResult]:
        """Validate a research component comprehensively."""
        logger.info(f"Starting validation for component: {component_name}")
        
        results = []
        
        # Create checkpoint state
        checkpoint_state = {
            'component_name': component_name,
            'config': self.config.__dict__,
            'start_time': time.time()
        }
        
        try:
            # Run each test in the suite
            for test_name, test_func in test_suite.items():
                logger.info(f"Running test: {test_name}")
                
                result = self._run_single_test(
                    component_name, test_name, component_func, test_func, baseline_func
                )
                
                results.append(result)
                self.validation_results.append(result)
                
                # Check if error threshold exceeded
                error_rate = sum(1 for r in results if not r.success) / len(results)
                if error_rate > self.config.error_threshold:
                    logger.error(f"Error threshold exceeded: {error_rate:.2%} > {self.config.error_threshold:.2%}")
                    break
            
            # Save checkpoint
            checkpoint_state['results'] = [r.__dict__ for r in results]
            checkpoint_path = self.checkpoint_manager.save_checkpoint(component_name, checkpoint_state)
            
            # Update checkpoint paths
            for result in results:
                result.checkpoint_path = checkpoint_path
            
            logger.info(f"Validation completed for {component_name}: {len(results)} tests run")
            
        except Exception as e:
            logger.error(f"Validation failed for {component_name}: {e}")
            logger.error(traceback.format_exc())
            
            # Create failure result
            failure_result = ValidationResult(
                component_name=component_name,
                test_name="validation_framework",
                success=False,
                execution_time=0.0,
                memory_usage_mb=0.0,
                error_message=str(e)
            )
            results.append(failure_result)
        
        return results
    
    def _run_single_test(self, 
                        component_name: str,
                        test_name: str,
                        component_func: Callable,
                        test_func: Callable,
                        baseline_func: Optional[Callable] = None) -> ValidationResult:
        """Run a single test with full monitoring."""
        
        with ResourceMonitor(self.config) as monitor:
            start_time = time.time()
            
            try:
                # Generate test inputs
                test_inputs = test_func()
                
                # Run component function
                component_results = []
                baseline_results = []
                
                for test_input in test_inputs:
                    # Check resource limits
                    if not monitor.check_limits():
                        raise RetryableError("Resource limits exceeded during test")
                    
                    # Run component
                    component_result = component_func(test_input)
                    component_results.append(component_result)
                    
                    # Run baseline if provided
                    if baseline_func:
                        baseline_result = baseline_func(test_input)
                        baseline_results.append(baseline_result)
                
                # Compute performance metrics
                execution_time = time.time() - start_time
                final_usage = monitor.get_current_usage()
                
                # Statistical validation if baseline provided
                statistical_results = {}
                if baseline_results:
                    # Convert results to numerical values for comparison
                    component_values = self._extract_numeric_values(component_results)
                    baseline_values = self._extract_numeric_values(baseline_results)
                    
                    if component_values and baseline_values:
                        statistical_results = self.statistical_validator.validate_statistical_significance(
                            baseline_values, component_values
                        )
                
                # Performance profiling
                performance_metrics = self.performance_profiler.profile_component(
                    f"{component_name}_{test_name}", component_func, test_inputs[:10]  # Limit to 10 for profiling
                )
                
                # Generate reproducibility hash
                repro_inputs = {
                    'test_inputs': str(test_inputs)[:1000],  # Truncate for hash
                    'component_name': component_name,
                    'test_name': test_name
                }
                reproducibility_hash = self.checkpoint_manager.get_reproducibility_hash(repro_inputs)
                
                return ValidationResult(
                    component_name=component_name,
                    test_name=test_name,
                    success=True,
                    execution_time=execution_time,
                    memory_usage_mb=final_usage['memory_mb'],
                    performance_metrics=performance_metrics,
                    statistical_results=statistical_results,
                    reproducibility_hash=reproducibility_hash
                )
                
            except Exception as e:
                logger.error(f"Test {test_name} failed: {e}")
                
                return ValidationResult(
                    component_name=component_name,
                    test_name=test_name,
                    success=False,
                    execution_time=time.time() - start_time,
                    memory_usage_mb=monitor.get_current_usage()['memory_mb'],
                    error_message=str(e)
                )
    
    def _extract_numeric_values(self, results: List[Any]) -> List[float]:
        """Extract numeric values from results for statistical comparison."""
        numeric_values = []
        
        for result in results:
            if isinstance(result, (int, float)):
                numeric_values.append(float(result))
            elif isinstance(result, dict) and 'score' in result:
                numeric_values.append(float(result['score']))
            elif isinstance(result, dict) and 'accuracy' in result:
                numeric_values.append(float(result['accuracy']))
            elif isinstance(result, torch.Tensor):
                if result.numel() == 1:
                    numeric_values.append(result.item())
                else:
                    numeric_values.append(torch.mean(result).item())
            elif hasattr(result, '__len__') and len(result) > 0:
                # Try to extract first numeric element
                try:
                    numeric_values.append(float(result[0]))
                except (TypeError, ValueError, IndexError):
                    continue
        
        return numeric_values
    
    def validate_parallel(self, 
                         components: Dict[str, Tuple[Callable, Dict[str, Callable]]],
                         max_workers: Optional[int] = None) -> Dict[str, List[ValidationResult]]:
        """Validate multiple components in parallel."""
        if max_workers is None:
            max_workers = min(multiprocessing.cpu_count(), len(components))
        
        logger.info(f"Starting parallel validation with {max_workers} workers")
        
        all_results = {}
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all validation tasks
            future_to_component = {}
            
            for component_name, (component_func, test_suite) in components.items():
                future = executor.submit(
                    self.validate_component,
                    component_name,
                    component_func,
                    test_suite
                )
                future_to_component[future] = component_name
            
            # Collect results as they complete
            for future in as_completed(future_to_component):
                component_name = future_to_component[future]
                
                try:
                    results = future.result(timeout=self.config.timeout_seconds)
                    all_results[component_name] = results
                    logger.info(f"Validation completed for {component_name}")
                    
                except Exception as e:
                    logger.error(f"Parallel validation failed for {component_name}: {e}")
                    all_results[component_name] = [
                        ValidationResult(
                            component_name=component_name,
                            test_name="parallel_execution",
                            success=False,
                            execution_time=0.0,
                            memory_usage_mb=0.0,
                            error_message=str(e)
                        )
                    ]
        
        return all_results
    
    def generate_validation_report(self, results: Dict[str, List[ValidationResult]]) -> str:
        """Generate comprehensive validation report."""
        total_tests = sum(len(component_results) for component_results in results.values())
        successful_tests = sum(
            sum(1 for result in component_results if result.success)
            for component_results in results.values()
        )
        
        success_rate = successful_tests / max(total_tests, 1)
        
        report = f"""
# Robust Validation Report

## Summary
- **Total Components Validated**: {len(results)}
- **Total Tests Run**: {total_tests}
- **Successful Tests**: {successful_tests}
- **Success Rate**: {success_rate:.1%}

## Component Results
"""
        
        for component_name, component_results in results.items():
            component_success_rate = sum(1 for r in component_results if r.success) / max(len(component_results), 1)
            avg_execution_time = np.mean([r.execution_time for r in component_results])
            avg_memory_usage = np.mean([r.memory_usage_mb for r in component_results])
            
            report += f"""
### {component_name}
- **Success Rate**: {component_success_rate:.1%}
- **Average Execution Time**: {avg_execution_time:.3f}s
- **Average Memory Usage**: {avg_memory_usage:.1f}MB
"""
            
            # Statistical results summary
            statistical_results = [r.statistical_results for r in component_results if r.statistical_results]
            if statistical_results:
                significant_improvements = sum(1 for sr in statistical_results if sr.get('significant', False))
                avg_improvement = np.mean([sr.get('improvement', 0) for sr in statistical_results])
                
                report += f"""- **Statistical Significance**: {significant_improvements}/{len(statistical_results)} tests
- **Average Improvement**: {avg_improvement:.1f}%
"""
            
            # Failed tests
            failed_tests = [r for r in component_results if not r.success]
            if failed_tests:
                report += f"- **Failed Tests**: {len(failed_tests)}\n"
                for failed_test in failed_tests[:3]:  # Show first 3 failures
                    report += f"  - {failed_test.test_name}: {failed_test.error_message}\n"
        
        # Performance summary
        report += f"""
## Performance Summary
"""
        
        performance_data = []
        for component_results in results.values():
            for result in component_results:
                if result.performance_metrics:
                    performance_data.append(result.performance_metrics)
        
        if performance_data:
            avg_throughput = np.mean([p.get('throughput_ops_per_sec', 0) for p in performance_data])
            avg_memory = np.mean([p.get('mean_memory_mb', 0) for p in performance_data])
            
            report += f"""- **Average Throughput**: {avg_throughput:.1f} ops/sec
- **Average Memory Usage**: {avg_memory:.1f}MB
"""
        
        # Recommendations
        report += f"""
## Recommendations
"""
        
        if success_rate < 0.8:
            report += "- **Action Required**: Success rate below 80%. Review failed tests and improve error handling.\n"
        elif success_rate < 0.95:
            report += "- **Improvement Suggested**: Success rate could be improved. Consider additional error handling.\n"
        else:
            report += "- **Good**: High success rate achieved. Continue monitoring.\n"
        
        # Resource optimization
        high_memory_components = []
        for component_name, component_results in results.items():
            avg_memory = np.mean([r.memory_usage_mb for r in component_results])
            if avg_memory > 1000:  # > 1GB
                high_memory_components.append(component_name)
        
        if high_memory_components:
            report += f"- **Memory Optimization**: Consider optimizing memory usage for: {', '.join(high_memory_components)}\n"
        
        return report
    
    def save_results(self, results: Dict[str, List[ValidationResult]], output_path: str):
        """Save validation results to file."""
        try:
            # Convert results to serializable format
            serializable_results = {}
            
            for component_name, component_results in results.items():
                serializable_results[component_name] = [
                    {
                        'component_name': result.component_name,
                        'test_name': result.test_name,
                        'success': result.success,
                        'execution_time': result.execution_time,
                        'memory_usage_mb': result.memory_usage_mb,
                        'error_message': result.error_message,
                        'performance_metrics': result.performance_metrics,
                        'statistical_results': result.statistical_results,
                        'reproducibility_hash': result.reproducibility_hash,
                        'checkpoint_path': result.checkpoint_path
                    }
                    for result in component_results
                ]
            
            with open(output_path, 'w') as f:
                json.dump(serializable_results, f, indent=2)
            
            logger.info(f"Validation results saved to {output_path}")
            
        except Exception as e:
            logger.error(f"Failed to save validation results: {e}")


def create_robust_validation_framework(config: Optional[ValidationConfig] = None) -> RobustValidationFramework:
    """Factory function to create robust validation framework."""
    if config is None:
        config = ValidationConfig()
    
    framework = RobustValidationFramework(config)
    
    logger.info("Created Robust Validation Framework")
    logger.info(f"Max retries: {config.max_retries}, Timeout: {config.timeout_seconds}s")
    logger.info(f"Memory limit: {config.memory_limit_gb}GB, GPU limit: {config.gpu_memory_limit_gb}GB")
    
    return framework