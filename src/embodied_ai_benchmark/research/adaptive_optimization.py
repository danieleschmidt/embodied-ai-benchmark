"""Adaptive Optimization System for Research Components."""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingWarmRestarts
import time
import json
import os
from typing import Dict, List, Optional, Tuple, Any, Callable, Union
from dataclasses import dataclass, field
from pathlib import Path
from collections import defaultdict, deque
import threading
import multiprocessing
from concurrent.futures import ThreadPoolExecutor
import optuna
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel
import warnings
warnings.filterwarnings('ignore')

from ..utils.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class OptimizationConfig:
    """Configuration for adaptive optimization."""
    optimization_method: str = "bayesian"  # bayesian, evolutionary, gradient_based, hybrid
    max_iterations: int = 100
    population_size: int = 20
    early_stopping_patience: int = 10
    early_stopping_threshold: float = 0.01
    exploration_rate: float = 0.2
    exploitation_rate: float = 0.8
    learning_rate_adaptation: bool = True
    batch_size_adaptation: bool = True
    architecture_adaptation: bool = True
    memory_optimization: bool = True
    parallel_evaluation: bool = True
    max_parallel_jobs: int = 4
    cache_evaluations: bool = True
    save_checkpoints: bool = True
    checkpoint_dir: str = "optimization_checkpoints"


@dataclass
class ParameterSpace:
    """Defines parameter search space."""
    name: str
    param_type: str  # continuous, discrete, categorical
    bounds: Tuple[Any, Any]
    default_value: Any
    log_scale: bool = False
    importance: float = 1.0


@dataclass
class OptimizationResult:
    """Result from optimization process."""
    best_parameters: Dict[str, Any]
    best_score: float
    optimization_history: List[Dict[str, Any]]
    convergence_iteration: int
    total_iterations: int
    evaluation_time: float
    parameter_importance: Dict[str, float]
    confidence_intervals: Dict[str, Tuple[float, float]]


class BayesianOptimizer:
    """Bayesian optimization using Gaussian Process."""
    
    def __init__(self, config: OptimizationConfig):
        self.config = config
        self.parameter_spaces = {}
        self.evaluation_history = []
        self.gp_model = None
        self.acquisition_function = "expected_improvement"
        
    def add_parameter_space(self, param_space: ParameterSpace):
        """Add parameter space for optimization."""
        self.parameter_spaces[param_space.name] = param_space
        logger.info(f"Added parameter space: {param_space.name}")
    
    def optimize(self, objective_function: Callable, 
                initial_parameters: Optional[Dict[str, Any]] = None) -> OptimizationResult:
        """Run Bayesian optimization."""
        logger.info(f"Starting Bayesian optimization with {len(self.parameter_spaces)} parameters")
        
        # Initialize with default parameters or provided initial parameters
        if initial_parameters is None:
            initial_parameters = {
                name: space.default_value 
                for name, space in self.parameter_spaces.items()
            }
        
        # Evaluate initial point
        initial_score = objective_function(initial_parameters)
        self.evaluation_history.append({
            'iteration': 0,
            'parameters': initial_parameters.copy(),
            'score': initial_score,
            'timestamp': time.time()
        })
        
        best_score = initial_score
        best_parameters = initial_parameters.copy()
        
        # Initialize Gaussian Process
        self._initialize_gp()
        
        patience_counter = 0
        
        for iteration in range(1, self.config.max_iterations + 1):
            # Generate candidate parameters using acquisition function
            candidate_parameters = self._generate_candidate()
            
            # Evaluate candidate
            try:
                score = objective_function(candidate_parameters)
                
                self.evaluation_history.append({
                    'iteration': iteration,
                    'parameters': candidate_parameters.copy(),
                    'score': score,
                    'timestamp': time.time()
                })
                
                # Update best if improved
                if score > best_score:
                    improvement = score - best_score
                    best_score = score
                    best_parameters = candidate_parameters.copy()
                    patience_counter = 0
                    
                    logger.info(f"Iteration {iteration}: New best score {best_score:.4f} (improvement: {improvement:.4f})")
                else:
                    patience_counter += 1
                
                # Update Gaussian Process
                self._update_gp()
                
                # Early stopping
                if patience_counter >= self.config.early_stopping_patience:
                    logger.info(f"Early stopping at iteration {iteration}")
                    break
                    
            except Exception as e:
                logger.error(f"Evaluation failed at iteration {iteration}: {e}")
                continue
        
        # Compute parameter importance
        parameter_importance = self._compute_parameter_importance()
        
        # Compute confidence intervals
        confidence_intervals = self._compute_confidence_intervals(best_parameters)
        
        total_time = sum(h.get('evaluation_time', 0) for h in self.evaluation_history)
        
        return OptimizationResult(
            best_parameters=best_parameters,
            best_score=best_score,
            optimization_history=self.evaluation_history,
            convergence_iteration=len(self.evaluation_history) - patience_counter,
            total_iterations=len(self.evaluation_history),
            evaluation_time=total_time,
            parameter_importance=parameter_importance,
            confidence_intervals=confidence_intervals
        )
    
    def _initialize_gp(self):
        """Initialize Gaussian Process model."""
        # Create kernel
        kernel = ConstantKernel(1.0) * Matern(length_scale=1.0, nu=2.5) + WhiteKernel(noise_level=1e-5)
        
        self.gp_model = GaussianProcessRegressor(
            kernel=kernel,
            alpha=1e-6,
            normalize_y=True,
            n_restarts_optimizer=10
        )
    
    def _update_gp(self):
        """Update Gaussian Process with new data."""
        if len(self.evaluation_history) < 2:
            return
        
        # Prepare training data
        X = []
        y = []
        
        for entry in self.evaluation_history:
            param_vector = self._parameters_to_vector(entry['parameters'])
            X.append(param_vector)
            y.append(entry['score'])
        
        X = np.array(X)
        y = np.array(y)
        
        # Fit GP
        try:
            self.gp_model.fit(X, y)
        except Exception as e:
            logger.warning(f"GP fitting failed: {e}")
    
    def _generate_candidate(self) -> Dict[str, Any]:
        """Generate candidate parameters using acquisition function."""
        if len(self.evaluation_history) < 2:
            # Random sampling for initial points
            return self._random_sample()
        
        # Use acquisition function to select next point
        best_candidate = None
        best_acquisition_value = float('-inf')
        
        # Sample multiple candidates and select best according to acquisition function
        for _ in range(100):
            candidate = self._random_sample()
            acquisition_value = self._compute_acquisition(candidate)
            
            if acquisition_value > best_acquisition_value:
                best_acquisition_value = acquisition_value
                best_candidate = candidate
        
        return best_candidate if best_candidate else self._random_sample()
    
    def _random_sample(self) -> Dict[str, Any]:
        """Generate random sample from parameter space."""
        sample = {}
        
        for name, space in self.parameter_spaces.items():
            if space.param_type == "continuous":
                if space.log_scale:
                    log_low = np.log(space.bounds[0])
                    log_high = np.log(space.bounds[1])
                    sample[name] = np.exp(np.random.uniform(log_low, log_high))
                else:
                    sample[name] = np.random.uniform(space.bounds[0], space.bounds[1])
            
            elif space.param_type == "discrete":
                sample[name] = np.random.randint(space.bounds[0], space.bounds[1] + 1)
            
            elif space.param_type == "categorical":
                sample[name] = np.random.choice(space.bounds)
        
        return sample
    
    def _compute_acquisition(self, parameters: Dict[str, Any]) -> float:
        """Compute acquisition function value."""
        if self.gp_model is None:
            return np.random.random()
        
        param_vector = self._parameters_to_vector(parameters)
        
        try:
            # Predict mean and std
            mean, std = self.gp_model.predict(param_vector.reshape(1, -1), return_std=True)
            
            # Expected Improvement
            best_score = max(h['score'] for h in self.evaluation_history)
            
            if std[0] == 0:
                return 0.0
            
            z = (mean[0] - best_score) / std[0]
            
            # Expected improvement with exploration
            ei = (mean[0] - best_score) * self._normal_cdf(z) + std[0] * self._normal_pdf(z)
            
            return ei
            
        except Exception as e:
            logger.warning(f"Acquisition computation failed: {e}")
            return np.random.random()
    
    def _normal_cdf(self, x):
        """Standard normal CDF."""
        return 0.5 * (1 + np.tanh(x / np.sqrt(2)))
    
    def _normal_pdf(self, x):
        """Standard normal PDF."""
        return np.exp(-0.5 * x**2) / np.sqrt(2 * np.pi)
    
    def _parameters_to_vector(self, parameters: Dict[str, Any]) -> np.ndarray:
        """Convert parameters dict to vector."""
        vector = []
        
        for name, space in self.parameter_spaces.items():
            value = parameters.get(name, space.default_value)
            
            if space.param_type == "continuous":
                if space.log_scale:
                    normalized_value = (np.log(value) - np.log(space.bounds[0])) / (np.log(space.bounds[1]) - np.log(space.bounds[0]))
                else:
                    normalized_value = (value - space.bounds[0]) / (space.bounds[1] - space.bounds[0])
                vector.append(normalized_value)
            
            elif space.param_type == "discrete":
                normalized_value = (value - space.bounds[0]) / (space.bounds[1] - space.bounds[0])
                vector.append(normalized_value)
            
            elif space.param_type == "categorical":
                # One-hot encoding
                for option in space.bounds:
                    vector.append(1.0 if value == option else 0.0)
        
        return np.array(vector)
    
    def _compute_parameter_importance(self) -> Dict[str, float]:
        """Compute parameter importance using sensitivity analysis."""
        if len(self.evaluation_history) < 10:
            return {name: 1.0 for name in self.parameter_spaces.keys()}
        
        importance = {}
        
        for param_name in self.parameter_spaces.keys():
            # Compute correlation between parameter and score
            param_values = [h['parameters'].get(param_name, 0) for h in self.evaluation_history]
            scores = [h['score'] for h in self.evaluation_history]
            
            try:
                correlation = np.corrcoef(param_values, scores)[0, 1]
                importance[param_name] = abs(correlation) if not np.isnan(correlation) else 0.0
            except Exception:
                importance[param_name] = 0.0
        
        # Normalize importance scores
        total_importance = sum(importance.values())
        if total_importance > 0:
            importance = {k: v / total_importance for k, v in importance.items()}
        
        return importance
    
    def _compute_confidence_intervals(self, parameters: Dict[str, Any], confidence: float = 0.95) -> Dict[str, Tuple[float, float]]:
        """Compute confidence intervals for parameters."""
        confidence_intervals = {}
        
        for param_name, space in self.parameter_spaces.items():
            if space.param_type == "continuous":
                # Use GP uncertainty
                if self.gp_model is not None:
                    try:
                        param_vector = self._parameters_to_vector(parameters)
                        mean, std = self.gp_model.predict(param_vector.reshape(1, -1), return_std=True)
                        
                        # Approximate confidence interval
                        margin = 1.96 * std[0]  # 95% confidence
                        
                        current_value = parameters[param_name]
                        ci_lower = max(space.bounds[0], current_value - margin)
                        ci_upper = min(space.bounds[1], current_value + margin)
                        
                        confidence_intervals[param_name] = (ci_lower, ci_upper)
                    except Exception:
                        confidence_intervals[param_name] = (space.bounds[0], space.bounds[1])
                else:
                    confidence_intervals[param_name] = (space.bounds[0], space.bounds[1])
            else:
                confidence_intervals[param_name] = (parameters[param_name], parameters[param_name])
        
        return confidence_intervals


class EvolutionaryOptimizer:
    """Evolutionary optimization using genetic algorithms."""
    
    def __init__(self, config: OptimizationConfig):
        self.config = config
        self.parameter_spaces = {}
        self.population = []
        self.fitness_history = []
        
    def add_parameter_space(self, param_space: ParameterSpace):
        """Add parameter space for optimization."""
        self.parameter_spaces[param_space.name] = param_space
        
    def optimize(self, objective_function: Callable, 
                initial_parameters: Optional[Dict[str, Any]] = None) -> OptimizationResult:
        """Run evolutionary optimization."""
        logger.info(f"Starting evolutionary optimization with population size {self.config.population_size}")
        
        # Initialize population
        self._initialize_population(initial_parameters)
        
        best_score = float('-inf')
        best_parameters = None
        best_iteration = 0
        
        for generation in range(self.config.max_iterations):
            # Evaluate population
            fitness_scores = []
            
            for individual in self.population:
                try:
                    score = objective_function(individual)
                    fitness_scores.append(score)
                    
                    if score > best_score:
                        best_score = score
                        best_parameters = individual.copy()
                        best_iteration = generation
                        
                except Exception as e:
                    logger.error(f"Evaluation failed for individual: {e}")
                    fitness_scores.append(float('-inf'))
            
            self.fitness_history.append({
                'generation': generation,
                'best_score': max(fitness_scores),
                'avg_score': np.mean(fitness_scores),
                'population': [ind.copy() for ind in self.population],
                'fitness_scores': fitness_scores.copy()
            })
            
            logger.info(f"Generation {generation}: Best={max(fitness_scores):.4f}, Avg={np.mean(fitness_scores):.4f}")
            
            # Selection, crossover, and mutation
            if generation < self.config.max_iterations - 1:
                self._evolve_population(fitness_scores)
        
        # Compute parameter importance based on final population diversity
        parameter_importance = self._compute_parameter_importance()
        
        return OptimizationResult(
            best_parameters=best_parameters,
            best_score=best_score,
            optimization_history=self.fitness_history,
            convergence_iteration=best_iteration,
            total_iterations=len(self.fitness_history),
            evaluation_time=0.0,  # Not tracked in this implementation
            parameter_importance=parameter_importance,
            confidence_intervals={}
        )
    
    def _initialize_population(self, initial_parameters: Optional[Dict[str, Any]]):
        """Initialize population."""
        self.population = []
        
        # Add initial parameters if provided
        if initial_parameters:
            self.population.append(initial_parameters.copy())
        
        # Add default parameters
        default_params = {
            name: space.default_value 
            for name, space in self.parameter_spaces.items()
        }
        if default_params not in self.population:
            self.population.append(default_params)
        
        # Fill remaining population with random individuals
        while len(self.population) < self.config.population_size:
            individual = self._random_individual()
            self.population.append(individual)
    
    def _random_individual(self) -> Dict[str, Any]:
        """Generate random individual."""
        individual = {}
        
        for name, space in self.parameter_spaces.items():
            if space.param_type == "continuous":
                if space.log_scale:
                    log_low = np.log(space.bounds[0])
                    log_high = np.log(space.bounds[1])
                    individual[name] = np.exp(np.random.uniform(log_low, log_high))
                else:
                    individual[name] = np.random.uniform(space.bounds[0], space.bounds[1])
            
            elif space.param_type == "discrete":
                individual[name] = np.random.randint(space.bounds[0], space.bounds[1] + 1)
            
            elif space.param_type == "categorical":
                individual[name] = np.random.choice(space.bounds)
        
        return individual
    
    def _evolve_population(self, fitness_scores: List[float]):
        """Evolve population through selection, crossover, and mutation."""
        # Selection (tournament selection)
        new_population = []
        
        # Keep best individuals (elitism)
        elite_count = max(1, self.config.population_size // 10)
        elite_indices = np.argsort(fitness_scores)[-elite_count:]
        
        for idx in elite_indices:
            new_population.append(self.population[idx].copy())
        
        # Generate offspring through crossover and mutation
        while len(new_population) < self.config.population_size:
            # Tournament selection
            parent1 = self._tournament_selection(fitness_scores)
            parent2 = self._tournament_selection(fitness_scores)
            
            # Crossover
            child = self._crossover(parent1, parent2)
            
            # Mutation
            child = self._mutate(child)
            
            new_population.append(child)
        
        self.population = new_population
    
    def _tournament_selection(self, fitness_scores: List[float]) -> Dict[str, Any]:
        """Tournament selection."""
        tournament_size = 3
        tournament_indices = np.random.choice(len(self.population), tournament_size, replace=False)
        tournament_fitness = [fitness_scores[i] for i in tournament_indices]
        
        winner_idx = tournament_indices[np.argmax(tournament_fitness)]
        return self.population[winner_idx].copy()
    
    def _crossover(self, parent1: Dict[str, Any], parent2: Dict[str, Any]) -> Dict[str, Any]:
        """Uniform crossover."""
        child = {}
        
        for name in self.parameter_spaces.keys():
            if np.random.random() < 0.5:
                child[name] = parent1[name]
            else:
                child[name] = parent2[name]
        
        return child
    
    def _mutate(self, individual: Dict[str, Any]) -> Dict[str, Any]:
        """Gaussian mutation."""
        mutated = individual.copy()
        
        for name, space in self.parameter_spaces.items():
            if np.random.random() < 0.1:  # Mutation probability
                if space.param_type == "continuous":
                    # Gaussian mutation
                    current_value = mutated[name]
                    mutation_strength = (space.bounds[1] - space.bounds[0]) * 0.1
                    
                    new_value = current_value + np.random.normal(0, mutation_strength)
                    new_value = np.clip(new_value, space.bounds[0], space.bounds[1])
                    
                    mutated[name] = new_value
                
                elif space.param_type == "discrete":
                    mutated[name] = np.random.randint(space.bounds[0], space.bounds[1] + 1)
                
                elif space.param_type == "categorical":
                    mutated[name] = np.random.choice(space.bounds)
        
        return mutated
    
    def _compute_parameter_importance(self) -> Dict[str, float]:
        """Compute parameter importance based on population diversity."""
        if not self.fitness_history:
            return {name: 1.0 for name in self.parameter_spaces.keys()}
        
        # Use final generation data
        final_generation = self.fitness_history[-1]
        population = final_generation['population']
        fitness_scores = final_generation['fitness_scores']
        
        importance = {}
        
        for param_name in self.parameter_spaces.keys():
            param_values = [ind.get(param_name, 0) for ind in population]
            
            # Compute correlation with fitness
            try:
                correlation = np.corrcoef(param_values, fitness_scores)[0, 1]
                importance[param_name] = abs(correlation) if not np.isnan(correlation) else 0.0
            except Exception:
                importance[param_name] = 0.0
        
        # Normalize
        total_importance = sum(importance.values())
        if total_importance > 0:
            importance = {k: v / total_importance for k, v in importance.items()}
        
        return importance


class AdaptiveLearningRateScheduler:
    """Adaptive learning rate scheduler based on training dynamics."""
    
    def __init__(self, optimizer: optim.Optimizer, config: OptimizationConfig):
        self.optimizer = optimizer
        self.config = config
        self.loss_history = deque(maxlen=20)
        self.lr_history = []
        
        # Multiple scheduling strategies
        self.schedulers = {
            'plateau': ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5),
            'cosine': CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
        }
        
        self.current_strategy = 'plateau'
        self.strategy_performance = defaultdict(list)
        
    def step(self, loss: float, epoch: int):
        """Update learning rate based on current loss."""
        self.loss_history.append(loss)
        
        # Record current learning rate
        current_lr = self.optimizer.param_groups[0]['lr']
        self.lr_history.append(current_lr)
        
        # Apply current scheduling strategy
        if self.current_strategy == 'plateau':
            self.schedulers['plateau'].step(loss)
        elif self.current_strategy == 'cosine':
            self.schedulers['cosine'].step(epoch)
        
        # Adaptive strategy selection
        if len(self.loss_history) >= 10 and epoch % 10 == 0:
            self._adapt_strategy(epoch)
    
    def _adapt_strategy(self, epoch: int):
        """Adapt scheduling strategy based on recent performance."""
        if len(self.loss_history) < 10:
            return
        
        # Compute recent loss trend
        recent_losses = list(self.loss_history)[-10:]
        loss_trend = np.polyfit(range(len(recent_losses)), recent_losses, 1)[0]
        
        # Record performance for current strategy
        self.strategy_performance[self.current_strategy].append(loss_trend)
        
        # Switch strategy if current one is not improving
        if loss_trend > 0:  # Loss is increasing
            # Try different strategy
            strategies = list(self.schedulers.keys())
            current_idx = strategies.index(self.current_strategy)
            next_strategy = strategies[(current_idx + 1) % len(strategies)]
            
            logger.info(f"Switching learning rate strategy from {self.current_strategy} to {next_strategy}")
            self.current_strategy = next_strategy
    
    def get_lr_schedule_summary(self) -> Dict[str, Any]:
        """Get summary of learning rate schedule."""
        return {
            'current_strategy': self.current_strategy,
            'current_lr': self.optimizer.param_groups[0]['lr'],
            'lr_history': self.lr_history[-50:],  # Last 50 values
            'strategy_performance': dict(self.strategy_performance)
        }


class AdaptiveOptimizationSystem:
    """Main system for adaptive optimization of research components."""
    
    def __init__(self, config: OptimizationConfig):
        self.config = config
        self.optimizers = {
            'bayesian': BayesianOptimizer(config),
            'evolutionary': EvolutionaryOptimizer(config)
        }
        
        self.current_optimizer = self.optimizers[config.optimization_method]
        self.optimization_history = []
        self.cache = {} if config.cache_evaluations else None
        
        # Create checkpoint directory
        if config.save_checkpoints:
            Path(config.checkpoint_dir).mkdir(exist_ok=True)
    
    def add_parameter_space(self, param_space: ParameterSpace):
        """Add parameter space to all optimizers."""
        for optimizer in self.optimizers.values():
            optimizer.add_parameter_space(param_space)
    
    def optimize_component(self, 
                         component_creator: Callable,
                         evaluation_function: Callable,
                         initial_parameters: Optional[Dict[str, Any]] = None) -> OptimizationResult:
        """Optimize a research component."""
        logger.info(f"Starting optimization with {self.config.optimization_method} optimizer")
        
        def cached_objective(parameters: Dict[str, Any]) -> float:
            """Cached objective function."""
            
            # Check cache
            if self.cache is not None:
                param_key = self._parameters_to_key(parameters)
                if param_key in self.cache:
                    logger.debug(f"Cache hit for parameters: {param_key}")
                    return self.cache[param_key]
            
            # Create component with parameters
            try:
                component = component_creator(parameters)
                
                # Evaluate component
                score = evaluation_function(component, parameters)
                
                # Cache result
                if self.cache is not None:
                    self.cache[param_key] = score
                
                logger.debug(f"Evaluated parameters {parameters}: score = {score:.4f}")
                
                return score
                
            except Exception as e:
                logger.error(f"Evaluation failed for parameters {parameters}: {e}")
                return float('-inf')
        
        # Run optimization
        result = self.current_optimizer.optimize(cached_objective, initial_parameters)
        
        # Save optimization history
        self.optimization_history.append({
            'timestamp': time.time(),
            'method': self.config.optimization_method,
            'result': result,
            'cache_size': len(self.cache) if self.cache else 0
        })
        
        # Save checkpoint
        if self.config.save_checkpoints:
            self._save_checkpoint(result)
        
        logger.info(f"Optimization completed. Best score: {result.best_score:.4f}")
        
        return result
    
    def optimize_hyperparameters(self,
                                model_class: type,
                                train_function: Callable,
                                validation_function: Callable,
                                parameter_spaces: List[ParameterSpace]) -> OptimizationResult:
        """Optimize hyperparameters for model training."""
        
        # Add parameter spaces
        for param_space in parameter_spaces:
            self.add_parameter_space(param_space)
        
        def training_objective(parameters: Dict[str, Any]) -> float:
            """Training objective function."""
            try:
                # Create model with hyperparameters
                model = model_class(**parameters)
                
                # Train model
                trained_model = train_function(model, parameters)
                
                # Validate model
                validation_score = validation_function(trained_model, parameters)
                
                return validation_score
                
            except Exception as e:
                logger.error(f"Training failed for parameters {parameters}: {e}")
                return float('-inf')
        
        # Run optimization
        return self.current_optimizer.optimize(training_objective)
    
    def multi_objective_optimization(self,
                                   objective_functions: List[Callable],
                                   objective_weights: List[float],
                                   component_creator: Callable) -> OptimizationResult:
        """Multi-objective optimization with weighted objectives."""
        
        def combined_objective(parameters: Dict[str, Any]) -> float:
            """Combined objective function."""
            try:
                component = component_creator(parameters)
                
                # Evaluate all objectives
                objective_scores = []
                for obj_func in objective_functions:
                    score = obj_func(component, parameters)
                    objective_scores.append(score)
                
                # Weighted combination
                combined_score = sum(w * s for w, s in zip(objective_weights, objective_scores))
                
                return combined_score
                
            except Exception as e:
                logger.error(f"Multi-objective evaluation failed: {e}")
                return float('-inf')
        
        return self.current_optimizer.optimize(combined_objective)
    
    def adaptive_batch_size_optimization(self,
                                       model: nn.Module,
                                       train_loader: torch.utils.data.DataLoader,
                                       memory_limit_gb: float = 8.0) -> int:
        """Adaptively find optimal batch size based on memory constraints."""
        
        logger.info("Starting adaptive batch size optimization")
        
        # Binary search for maximum feasible batch size
        min_batch_size = 1
        max_batch_size = 1024
        optimal_batch_size = min_batch_size
        
        while min_batch_size <= max_batch_size:
            current_batch_size = (min_batch_size + max_batch_size) // 2
            
            try:
                # Test batch size
                test_successful = self._test_batch_size(
                    model, train_loader, current_batch_size, memory_limit_gb
                )
                
                if test_successful:
                    optimal_batch_size = current_batch_size
                    min_batch_size = current_batch_size + 1
                else:
                    max_batch_size = current_batch_size - 1
                    
            except Exception as e:
                logger.warning(f"Batch size {current_batch_size} failed: {e}")
                max_batch_size = current_batch_size - 1
        
        logger.info(f"Optimal batch size found: {optimal_batch_size}")
        
        return optimal_batch_size
    
    def _test_batch_size(self, model: nn.Module, train_loader: torch.utils.data.DataLoader,
                        batch_size: int, memory_limit_gb: float) -> bool:
        """Test if batch size is feasible within memory constraints."""
        
        # Create test batch
        device = next(model.parameters()).device
        
        try:
            # Get sample data
            sample_batch = next(iter(train_loader))
            
            if isinstance(sample_batch, (list, tuple)):
                # Replicate batch to desired size
                batch_multiplier = batch_size // len(sample_batch[0])
                if batch_multiplier > 1:
                    test_batch = []
                    for item in sample_batch:
                        if torch.is_tensor(item):
                            test_batch.append(item.repeat([batch_multiplier] + [1] * (item.dim() - 1)))
                        else:
                            test_batch.append(item)
                else:
                    test_batch = sample_batch
            else:
                test_batch = sample_batch
            
            # Move to device and test forward pass
            if torch.is_tensor(test_batch):
                test_batch = test_batch.to(device)
            elif isinstance(test_batch, (list, tuple)):
                test_batch = [t.to(device) if torch.is_tensor(t) else t for t in test_batch]
            
            # Forward pass
            with torch.no_grad():
                if isinstance(test_batch, (list, tuple)):
                    output = model(*test_batch)
                else:
                    output = model(test_batch)
            
            # Check memory usage
            if torch.cuda.is_available():
                memory_used_gb = torch.cuda.memory_allocated() / (1024**3)
                return memory_used_gb < memory_limit_gb
            else:
                return True  # Assume success for CPU
                
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                return False
            else:
                raise
        except Exception:
            return False
    
    def _parameters_to_key(self, parameters: Dict[str, Any]) -> str:
        """Convert parameters to cache key."""
        import hashlib
        param_str = json.dumps(parameters, sort_keys=True, default=str)
        return hashlib.md5(param_str.encode()).hexdigest()
    
    def _save_checkpoint(self, result: OptimizationResult):
        """Save optimization checkpoint."""
        checkpoint_path = Path(self.config.checkpoint_dir) / f"optimization_{int(time.time())}.json"
        
        try:
            checkpoint_data = {
                'config': self.config.__dict__,
                'result': {
                    'best_parameters': result.best_parameters,
                    'best_score': result.best_score,
                    'total_iterations': result.total_iterations,
                    'convergence_iteration': result.convergence_iteration,
                    'parameter_importance': result.parameter_importance
                },
                'timestamp': time.time()
            }
            
            with open(checkpoint_path, 'w') as f:
                json.dump(checkpoint_data, f, indent=2, default=str)
            
            logger.info(f"Checkpoint saved: {checkpoint_path}")
            
        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")
    
    def get_optimization_summary(self) -> Dict[str, Any]:
        """Get summary of all optimization runs."""
        if not self.optimization_history:
            return {}
        
        # Aggregate statistics
        all_scores = [h['result'].best_score for h in self.optimization_history]
        all_iterations = [h['result'].total_iterations for h in self.optimization_history]
        
        method_performance = defaultdict(list)
        for h in self.optimization_history:
            method_performance[h['method']].append(h['result'].best_score)
        
        return {
            'total_optimization_runs': len(self.optimization_history),
            'best_overall_score': max(all_scores) if all_scores else 0,
            'average_score': np.mean(all_scores) if all_scores else 0,
            'average_iterations': np.mean(all_iterations) if all_iterations else 0,
            'method_performance': {
                method: {
                    'count': len(scores),
                    'best_score': max(scores),
                    'average_score': np.mean(scores)
                }
                for method, scores in method_performance.items()
            },
            'cache_efficiency': len(self.cache) if self.cache else 0
        }


def create_adaptive_optimization_system(config: Optional[OptimizationConfig] = None) -> AdaptiveOptimizationSystem:
    """Factory function to create adaptive optimization system."""
    if config is None:
        config = OptimizationConfig()
    
    system = AdaptiveOptimizationSystem(config)
    
    logger.info(f"Created Adaptive Optimization System")
    logger.info(f"Method: {config.optimization_method}, Max iterations: {config.max_iterations}")
    logger.info(f"Population size: {config.population_size}, Parallel jobs: {config.max_parallel_jobs}")
    
    return system


# Example usage and predefined parameter spaces for common research components
def get_attention_fusion_parameter_spaces() -> List[ParameterSpace]:
    """Get parameter spaces for attention fusion optimization."""
    return [
        ParameterSpace("hidden_dim", "discrete", (128, 1024), 512),
        ParameterSpace("num_heads", "discrete", (4, 16), 8),
        ParameterSpace("dropout_rate", "continuous", (0.0, 0.5), 0.1),
        ParameterSpace("temperature", "continuous", (0.5, 2.0), 1.0),
        ParameterSpace("num_modalities", "discrete", (2, 6), 4)
    ]


def get_quantum_planning_parameter_spaces() -> List[ParameterSpace]:
    """Get parameter spaces for quantum planning optimization."""
    return [
        ParameterSpace("num_qubits", "discrete", (4, 12), 8),
        ParameterSpace("planning_horizon", "discrete", (10, 50), 20),
        ParameterSpace("coherence_time", "continuous", (50.0, 200.0), 100.0, log_scale=True),
        ParameterSpace("decoherence_rate", "continuous", (0.001, 0.1), 0.01, log_scale=True),
        ParameterSpace("superposition_depth", "discrete", (3, 10), 5)
    ]


def get_swarm_coordination_parameter_spaces() -> List[ParameterSpace]:
    """Get parameter spaces for swarm coordination optimization."""
    return [
        ParameterSpace("max_agents", "discrete", (5, 50), 20),
        ParameterSpace("communication_range", "continuous", (2.0, 10.0), 5.0),
        ParameterSpace("coordination_dim", "discrete", (32, 128), 64),
        ParameterSpace("emergence_layers", "discrete", (2, 8), 4),
        ParameterSpace("consensus_threshold", "continuous", (0.5, 0.95), 0.8)
    ]