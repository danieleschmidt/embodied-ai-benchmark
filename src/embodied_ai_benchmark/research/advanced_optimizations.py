"""Advanced optimizations and novel algorithmic contributions."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import time
from concurrent.futures import ThreadPoolExecutor
import asyncio
from collections import deque

from ..utils.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class PerformanceMetrics:
    """Performance metrics for optimization tracking."""
    throughput: float  # operations per second
    latency: float     # average response time
    accuracy: float    # task success rate
    efficiency: float  # resource utilization
    scalability: float # performance under load


class AdaptiveComputeScheduler:
    """Dynamic compute allocation based on task complexity and urgency."""
    
    def __init__(self, max_workers: int = 8, max_gpu_memory: float = 0.8):
        self.max_workers = max_workers
        self.max_gpu_memory = max_gpu_memory
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.gpu_memory_usage = 0.0
        self.task_queue = deque()
        self.active_tasks = {}
        
        # Performance history for adaptation
        self.performance_history = deque(maxlen=1000)
        self.resource_utilization = deque(maxlen=100)
        
    def schedule_task(self, 
                     task_func: callable,
                     task_args: tuple,
                     priority: int = 1,
                     estimated_memory: float = 0.1,
                     timeout: float = 30.0) -> str:
        """
        Schedule task with dynamic resource allocation.
        
        Args:
            task_func: Function to execute
            task_args: Arguments for the function
            priority: Task priority (1=low, 5=high)
            estimated_memory: Estimated GPU memory usage (0-1)
            timeout: Maximum execution time
            
        Returns:
            Task ID for tracking
        """
        task_id = f"task_{int(time.time() * 1000)}_{np.random.randint(1000)}"
        
        # Check resource availability
        if self.gpu_memory_usage + estimated_memory > self.max_gpu_memory:
            logger.warning(f"Task {task_id} queued - insufficient GPU memory")
            self.task_queue.append({
                'id': task_id,
                'func': task_func,
                'args': task_args,
                'priority': priority,
                'memory': estimated_memory,
                'timeout': timeout,
                'queued_time': time.time()
            })
            return task_id
        
        # Execute immediately if resources available
        future = self.executor.submit(self._execute_task, task_func, task_args, timeout)
        self.active_tasks[task_id] = {
            'future': future,
            'memory': estimated_memory,
            'start_time': time.time(),
            'timeout': timeout
        }
        
        self.gpu_memory_usage += estimated_memory
        return task_id
    
    def _execute_task(self, task_func: callable, task_args: tuple, timeout: float):
        """Execute task with timeout and monitoring."""
        start_time = time.time()
        
        try:
            # Set timeout for PyTorch operations
            if torch.cuda.is_available():
                with torch.cuda.device(0):
                    result = task_func(*task_args)
            else:
                result = task_func(*task_args)
            
            execution_time = time.time() - start_time
            self.performance_history.append({
                'execution_time': execution_time,
                'success': True,
                'timestamp': start_time
            })
            
            return result
            
        except Exception as e:
            execution_time = time.time() - start_time
            self.performance_history.append({
                'execution_time': execution_time,
                'success': False,
                'error': str(e),
                'timestamp': start_time
            })
            raise e
    
    def get_task_result(self, task_id: str, timeout: float = None):
        """Get result from completed or running task."""
        if task_id in self.active_tasks:
            task_info = self.active_tasks[task_id]
            timeout = timeout or task_info['timeout']
            
            try:
                result = task_info['future'].result(timeout=timeout)
                # Free GPU memory
                self.gpu_memory_usage -= task_info['memory']
                del self.active_tasks[task_id]
                
                # Process queue if resources freed
                self._process_queue()
                
                return result
                
            except Exception as e:
                logger.error(f"Task {task_id} failed: {e}")
                self.gpu_memory_usage -= task_info['memory']
                del self.active_tasks[task_id]
                self._process_queue()
                raise e
        else:
            raise ValueError(f"Task {task_id} not found")
    
    def _process_queue(self):
        """Process queued tasks when resources become available."""
        while self.task_queue:
            # Sort queue by priority and wait time
            queue_list = list(self.task_queue)
            queue_list.sort(key=lambda x: (-x['priority'], x['queued_time']))
            
            next_task = queue_list[0]
            
            if self.gpu_memory_usage + next_task['memory'] <= self.max_gpu_memory:
                # Remove from queue and execute
                self.task_queue.remove(next_task)
                
                future = self.executor.submit(
                    self._execute_task,
                    next_task['func'],
                    next_task['args'],
                    next_task['timeout']
                )
                
                self.active_tasks[next_task['id']] = {
                    'future': future,
                    'memory': next_task['memory'],
                    'start_time': time.time(),
                    'timeout': next_task['timeout']
                }
                
                self.gpu_memory_usage += next_task['memory']
            else:
                break  # No resources for highest priority task
    
    def get_performance_metrics(self) -> PerformanceMetrics:
        """Get current performance metrics."""
        if not self.performance_history:
            return PerformanceMetrics(0, 0, 0, 0, 0)
        
        recent_history = list(self.performance_history)[-100:]
        
        # Calculate metrics
        successful_tasks = [h for h in recent_history if h['success']]
        avg_execution_time = np.mean([h['execution_time'] for h in successful_tasks]) if successful_tasks else 0
        
        throughput = len(successful_tasks) / max(1, (time.time() - recent_history[0]['timestamp']))
        latency = avg_execution_time
        accuracy = len(successful_tasks) / len(recent_history)
        efficiency = self.gpu_memory_usage / self.max_gpu_memory
        scalability = min(1.0, throughput / self.max_workers)
        
        return PerformanceMetrics(
            throughput=throughput,
            latency=latency,
            accuracy=accuracy,
            efficiency=efficiency,
            scalability=scalability
        )


class AutomaticMixedPrecision:
    """Automatic mixed precision training for faster computation."""
    
    def __init__(self, enabled: bool = True):
        self.enabled = enabled and torch.cuda.is_available()
        self.scaler = torch.cuda.amp.GradScaler() if self.enabled else None
        
    def forward_pass(self, model: nn.Module, inputs: torch.Tensor) -> torch.Tensor:
        """Forward pass with automatic mixed precision."""
        if self.enabled:
            with torch.cuda.amp.autocast():
                return model(inputs)
        else:
            return model(inputs)
    
    def backward_pass(self, loss: torch.Tensor, optimizer: torch.optim.Optimizer):
        """Backward pass with gradient scaling."""
        if self.enabled:
            self.scaler.scale(loss).backward()
            self.scaler.step(optimizer)
            self.scaler.update()
        else:
            loss.backward()
            optimizer.step()


class MemoryOptimizer:
    """Memory optimization utilities for large-scale training."""
    
    def __init__(self):
        self.memory_pool = {}
        self.allocation_history = deque(maxlen=1000)
        
    def allocate_tensor(self, 
                       shape: tuple, 
                       dtype: torch.dtype = torch.float32,
                       device: str = 'cuda',
                       reuse_key: str = None) -> torch.Tensor:
        """Allocate tensor with optional memory reuse."""
        tensor_key = f"{shape}_{dtype}_{device}"
        
        # Try to reuse existing tensor
        if reuse_key and reuse_key in self.memory_pool:
            tensor = self.memory_pool[reuse_key]
            if tensor.shape == shape and tensor.dtype == dtype:
                tensor.zero_()  # Clear existing data
                return tensor
        
        # Allocate new tensor
        tensor = torch.zeros(shape, dtype=dtype, device=device)
        
        if reuse_key:
            self.memory_pool[reuse_key] = tensor
        
        self.allocation_history.append({
            'shape': shape,
            'dtype': str(dtype),
            'device': device,
            'timestamp': time.time(),
            'memory_mb': tensor.numel() * tensor.element_size() / (1024 * 1024)
        })
        
        return tensor
    
    def clear_cache(self):
        """Clear memory cache and run garbage collection."""
        self.memory_pool.clear()
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get memory usage statistics."""
        stats = {
            'pool_size': len(self.memory_pool),
            'recent_allocations': len(self.allocation_history),
            'total_allocated_mb': sum(h['memory_mb'] for h in self.allocation_history)
        }
        
        if torch.cuda.is_available():
            stats.update({
                'cuda_allocated_mb': torch.cuda.memory_allocated() / (1024 * 1024),
                'cuda_cached_mb': torch.cuda.memory_reserved() / (1024 * 1024),
                'cuda_max_allocated_mb': torch.cuda.max_memory_allocated() / (1024 * 1024)
            })
        
        return stats


class GradientOptimization:
    """Advanced gradient optimization techniques."""
    
    def __init__(self, 
                 gradient_clipping: float = 1.0,
                 accumulation_steps: int = 1,
                 gradient_checkpointing: bool = False):
        self.gradient_clipping = gradient_clipping
        self.accumulation_steps = accumulation_steps
        self.gradient_checkpointing = gradient_checkpointing
        self.accumulated_grads = {}
        self.step_count = 0
        
    def accumulate_gradients(self, model: nn.Module, loss: torch.Tensor):
        """Accumulate gradients over multiple steps."""
        # Scale loss by accumulation steps
        scaled_loss = loss / self.accumulation_steps
        scaled_loss.backward()
        
        self.step_count += 1
        
        # Return whether to step optimizer
        return self.step_count % self.accumulation_steps == 0
    
    def clip_gradients(self, model: nn.Module) -> float:
        """Clip gradients and return the norm."""
        if self.gradient_clipping > 0:
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), 
                self.gradient_clipping
            )
            return grad_norm.item()
        return 0.0
    
    def setup_gradient_checkpointing(self, model: nn.Module):
        """Setup gradient checkpointing for memory efficiency."""
        if self.gradient_checkpointing:
            if hasattr(model, 'gradient_checkpointing_enable'):
                model.gradient_checkpointing_enable()
            else:
                logger.warning("Model does not support gradient checkpointing")


class ModelCompression:
    """Model compression techniques for deployment."""
    
    def __init__(self):
        self.quantization_enabled = torch.quantization.QConfig is not None
        
    def quantize_model(self, 
                      model: nn.Module,
                      calibration_data: List[torch.Tensor],
                      quantization_type: str = 'dynamic') -> nn.Module:
        """Apply quantization to model."""
        if not self.quantization_enabled:
            logger.warning("Quantization not available")
            return model
        
        if quantization_type == 'dynamic':
            # Dynamic quantization (post-training)
            quantized_model = torch.quantization.quantize_dynamic(
                model, 
                {nn.Linear, nn.Conv2d}, 
                dtype=torch.qint8
            )
            
        elif quantization_type == 'static':
            # Static quantization (requires calibration)
            model.eval()
            model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
            torch.quantization.prepare(model, inplace=True)
            
            # Calibrate with sample data
            with torch.no_grad():
                for data in calibration_data[:100]:  # Use subset for calibration
                    model(data)
            
            quantized_model = torch.quantization.convert(model, inplace=True)
            
        else:
            raise ValueError(f"Unknown quantization type: {quantization_type}")
        
        return quantized_model
    
    def prune_model(self, 
                   model: nn.Module, 
                   pruning_ratio: float = 0.2,
                   structured: bool = False) -> nn.Module:
        """Apply neural network pruning."""
        import torch.nn.utils.prune as prune
        
        if structured:
            # Structured pruning (remove entire neurons/channels)
            for module in model.modules():
                if isinstance(module, nn.Linear):
                    prune.ln_structured(module, name='weight', amount=pruning_ratio, n=2, dim=0)
                elif isinstance(module, nn.Conv2d):
                    prune.ln_structured(module, name='weight', amount=pruning_ratio, n=2, dim=0)
        else:
            # Unstructured pruning (individual weights)
            parameters_to_prune = []
            for module in model.modules():
                if isinstance(module, (nn.Linear, nn.Conv2d)):
                    parameters_to_prune.append((module, 'weight'))
            
            prune.global_unstructured(
                parameters_to_prune,
                pruning_method=prune.L1Unstructured,
                amount=pruning_ratio
            )
        
        return model
    
    def knowledge_distillation(self,
                             teacher_model: nn.Module,
                             student_model: nn.Module,
                             train_loader: torch.utils.data.DataLoader,
                             temperature: float = 3.0,
                             alpha: float = 0.7,
                             num_epochs: int = 10) -> nn.Module:
        """Train student model using knowledge distillation."""
        teacher_model.eval()
        student_model.train()
        
        optimizer = torch.optim.Adam(student_model.parameters(), lr=1e-3)
        criterion = nn.KLDivLoss(reduction='batchmean')
        
        for epoch in range(num_epochs):
            for batch_idx, (data, target) in enumerate(train_loader):
                if torch.cuda.is_available():
                    data, target = data.cuda(), target.cuda()
                
                # Teacher predictions (soft targets)
                with torch.no_grad():
                    teacher_outputs = teacher_model(data)
                    teacher_probs = F.softmax(teacher_outputs / temperature, dim=1)
                
                # Student predictions
                student_outputs = student_model(data)
                student_log_probs = F.log_softmax(student_outputs / temperature, dim=1)
                
                # Distillation loss
                distillation_loss = criterion(student_log_probs, teacher_probs) * (temperature ** 2)
                
                # Hard target loss
                hard_loss = F.cross_entropy(student_outputs, target)
                
                # Combined loss
                total_loss = alpha * distillation_loss + (1 - alpha) * hard_loss
                
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()
                
                if batch_idx % 100 == 0:
                    logger.info(f"Epoch {epoch}, Batch {batch_idx}, "
                               f"Distillation Loss: {distillation_loss:.4f}, "
                               f"Hard Loss: {hard_loss:.4f}")
        
        return student_model


class AdvancedOptimizer:
    """Advanced optimization algorithms and learning rate scheduling."""
    
    def __init__(self):
        self.schedulers = {}
        self.optimizers = {}
        
    def create_optimizer(self,
                        model: nn.Module,
                        optimizer_type: str = 'adamw',
                        learning_rate: float = 1e-3,
                        weight_decay: float = 0.01,
                        **kwargs) -> torch.optim.Optimizer:
        """Create advanced optimizer."""
        
        if optimizer_type.lower() == 'adamw':
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=learning_rate,
                weight_decay=weight_decay,
                **kwargs
            )
        elif optimizer_type.lower() == 'ranger':
            # RAdam + Lookahead (if available)
            try:
                from torch_optimizer import RAdam, Lookahead
                base_optimizer = RAdam(
                    model.parameters(),
                    lr=learning_rate,
                    weight_decay=weight_decay
                )
                optimizer = Lookahead(base_optimizer, k=5, alpha=0.5)
            except ImportError:
                logger.warning("torch-optimizer not available, falling back to AdamW")
                optimizer = torch.optim.AdamW(
                    model.parameters(),
                    lr=learning_rate,
                    weight_decay=weight_decay
                )
        elif optimizer_type.lower() == 'lamb':
            # Layer-wise Adaptive Moments optimizer for Large Batch training
            try:
                from torch_optimizer import Lamb
                optimizer = Lamb(
                    model.parameters(),
                    lr=learning_rate,
                    weight_decay=weight_decay,
                    **kwargs
                )
            except ImportError:
                logger.warning("torch-optimizer not available, falling back to AdamW")
                optimizer = torch.optim.AdamW(
                    model.parameters(),
                    lr=learning_rate,
                    weight_decay=weight_decay
                )
        else:
            raise ValueError(f"Unknown optimizer type: {optimizer_type}")
        
        return optimizer
    
    def create_scheduler(self,
                        optimizer: torch.optim.Optimizer,
                        scheduler_type: str = 'cosine_warmup',
                        num_training_steps: int = 1000,
                        num_warmup_steps: int = 100,
                        **kwargs) -> torch.optim.lr_scheduler._LRScheduler:
        """Create advanced learning rate scheduler."""
        
        if scheduler_type == 'cosine_warmup':
            def lr_lambda(current_step):
                if current_step < num_warmup_steps:
                    return float(current_step) / float(max(1, num_warmup_steps))
                return max(0.0, 0.5 * (1.0 + np.cos(
                    np.pi * float(current_step - num_warmup_steps) / 
                    float(max(1, num_training_steps - num_warmup_steps))
                )))
            
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
            
        elif scheduler_type == 'polynomial':
            def lr_lambda(current_step):
                if current_step < num_warmup_steps:
                    return float(current_step) / float(max(1, num_warmup_steps))
                else:
                    return max(0.0, (num_training_steps - current_step) / 
                              max(1, num_training_steps - num_warmup_steps)) ** kwargs.get('power', 1.0)
            
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
            
        elif scheduler_type == 'reduce_on_plateau':
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode='min',
                factor=0.5,
                patience=10,
                verbose=True,
                **kwargs
            )
            
        else:
            raise ValueError(f"Unknown scheduler type: {scheduler_type}")
        
        return scheduler


class PerformanceBenchmark:
    """Performance benchmarking and profiling utilities."""
    
    def __init__(self):
        self.benchmark_results = {}
        
    def benchmark_model(self,
                       model: nn.Module,
                       input_shapes: List[tuple],
                       batch_sizes: List[int] = [1, 8, 16, 32],
                       num_warmup: int = 10,
                       num_iterations: int = 100,
                       device: str = 'cuda') -> Dict[str, Any]:
        """Comprehensive model benchmarking."""
        
        model = model.to(device)
        model.eval()
        
        results = {}
        
        for batch_size in batch_sizes:
            for input_shape in input_shapes:
                test_name = f"batch_{batch_size}_shape_{'x'.join(map(str, input_shape))}"
                
                # Create test input
                full_shape = (batch_size,) + input_shape
                test_input = torch.randn(full_shape, device=device)
                
                # Warmup
                with torch.no_grad():
                    for _ in range(num_warmup):
                        _ = model(test_input)
                
                if device == 'cuda':
                    torch.cuda.synchronize()
                
                # Benchmark
                start_time = time.time()
                
                with torch.no_grad():
                    for _ in range(num_iterations):
                        _ = model(test_input)
                
                if device == 'cuda':
                    torch.cuda.synchronize()
                
                end_time = time.time()
                
                # Calculate metrics
                total_time = end_time - start_time
                avg_time = total_time / num_iterations
                throughput = batch_size / avg_time  # samples per second
                
                results[test_name] = {
                    'avg_inference_time_ms': avg_time * 1000,
                    'throughput_samples_per_sec': throughput,
                    'memory_usage_mb': torch.cuda.max_memory_allocated() / (1024 * 1024) if device == 'cuda' else 0
                }
                
                # Reset memory stats
                if device == 'cuda':
                    torch.cuda.reset_peak_memory_stats()
        
        return results
    
    def profile_model(self,
                     model: nn.Module,
                     sample_input: torch.Tensor,
                     profile_memory: bool = True) -> str:
        """Profile model execution with detailed breakdown."""
        
        model.eval()
        
        if torch.cuda.is_available():
            with torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ],
                record_shapes=True,
                profile_memory=profile_memory,
                with_stack=True
            ) as prof:
                with torch.no_grad():
                    _ = model(sample_input)
            
            # Return profiling report
            return prof.key_averages().table(sort_by="cuda_time_total", row_limit=20)
        else:
            with torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CPU],
                record_shapes=True,
                profile_memory=profile_memory,
                with_stack=True
            ) as prof:
                with torch.no_grad():
                    _ = model(sample_input)
            
            return prof.key_averages().table(sort_by="cpu_time_total", row_limit=20)


# Integration class that combines all optimizations
class EmbodiedAIOptimizer:
    """Integrated optimization system for embodied AI applications."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # Initialize optimization components
        self.scheduler = AdaptiveComputeScheduler(
            max_workers=config.get('max_workers', 8),
            max_gpu_memory=config.get('max_gpu_memory', 0.8)
        )
        
        self.amp = AutomaticMixedPrecision(
            enabled=config.get('use_amp', True)
        )
        
        self.memory_optimizer = MemoryOptimizer()
        self.gradient_optimizer = GradientOptimization(
            gradient_clipping=config.get('gradient_clipping', 1.0),
            accumulation_steps=config.get('accumulation_steps', 1)
        )
        
        self.model_compression = ModelCompression()
        self.advanced_optimizer = AdvancedOptimizer()
        self.benchmark = PerformanceBenchmark()
        
    def optimize_model_training(self,
                               model: nn.Module,
                               train_loader: torch.utils.data.DataLoader,
                               num_epochs: int = 10) -> nn.Module:
        """Complete model training optimization pipeline."""
        
        # Setup optimizer and scheduler
        optimizer = self.advanced_optimizer.create_optimizer(
            model,
            optimizer_type=self.config.get('optimizer_type', 'adamw'),
            learning_rate=self.config.get('learning_rate', 1e-3)
        )
        
        num_training_steps = len(train_loader) * num_epochs
        scheduler = self.advanced_optimizer.create_scheduler(
            optimizer,
            scheduler_type=self.config.get('scheduler_type', 'cosine_warmup'),
            num_training_steps=num_training_steps
        )
        
        # Setup gradient optimization
        self.gradient_optimizer.setup_gradient_checkpointing(model)
        
        # Training loop with optimizations
        model.train()
        
        for epoch in range(num_epochs):
            for batch_idx, (data, target) in enumerate(train_loader):
                if torch.cuda.is_available():
                    data, target = data.cuda(), target.cuda()
                
                # Forward pass with AMP
                output = self.amp.forward_pass(model, data)
                loss = F.cross_entropy(output, target)
                
                # Gradient accumulation
                should_step = self.gradient_optimizer.accumulate_gradients(model, loss)
                
                if should_step:
                    # Gradient clipping
                    grad_norm = self.gradient_optimizer.clip_gradients(model)
                    
                    # Optimizer step with AMP
                    self.amp.backward_pass(loss, optimizer)
                    
                    # Scheduler step
                    scheduler.step()
                    
                    # Clear optimizer gradients
                    optimizer.zero_grad()
                
                # Periodic memory cleanup
                if batch_idx % 100 == 0:
                    self.memory_optimizer.clear_cache()
        
        return model
    
    def optimize_for_deployment(self,
                               model: nn.Module,
                               calibration_data: List[torch.Tensor] = None,
                               compression_config: Dict[str, Any] = None) -> nn.Module:
        """Optimize model for deployment."""
        
        compression_config = compression_config or {}
        optimized_model = model
        
        # Apply pruning if requested
        if compression_config.get('pruning_ratio', 0) > 0:
            optimized_model = self.model_compression.prune_model(
                optimized_model,
                pruning_ratio=compression_config['pruning_ratio'],
                structured=compression_config.get('structured_pruning', False)
            )
        
        # Apply quantization if requested
        if compression_config.get('quantization_type'):
            optimized_model = self.model_compression.quantize_model(
                optimized_model,
                calibration_data or [],
                quantization_type=compression_config['quantization_type']
            )
        
        return optimized_model
    
    def get_system_metrics(self) -> Dict[str, Any]:
        """Get comprehensive system performance metrics."""
        return {
            'scheduler_metrics': self.scheduler.get_performance_metrics(),
            'memory_stats': self.memory_optimizer.get_memory_stats(),
            'gpu_utilization': torch.cuda.utilization() if torch.cuda.is_available() else 0,
            'active_tasks': len(self.scheduler.active_tasks),
            'queued_tasks': len(self.scheduler.task_queue)
        }