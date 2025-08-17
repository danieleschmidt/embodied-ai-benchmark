"""Meta-Learning Algorithm with Model-Agnostic Meta-Learning Plus (MAML++) for Embodied AI.

Novel contributions:
1. Hierarchical task structure with meta-meta learning
2. Uncertainty-aware gradient optimization
3. Dynamic inner-loop adaptation
4. Cross-modal transfer capabilities
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, Categorical
import numpy as np
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
import time
from collections import defaultdict, deque
import math
from abc import ABC, abstractmethod

from ..utils.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class TaskMetadata:
    """Metadata for embodied AI tasks."""
    task_id: str
    difficulty_level: float  # 0.0 to 1.0
    sensory_modalities: List[str]  # ['vision', 'tactile', 'audio', etc.]
    action_space_dim: int
    observation_space_dim: int
    temporal_horizon: int
    object_types: List[str]
    physics_complexity: str  # 'simple', 'medium', 'complex'
    multi_agent: bool
    language_guided: bool


@dataclass
class AdaptationContext:
    """Context information for task adaptation."""
    support_demonstrations: List[Dict[str, Any]]
    task_metadata: TaskMetadata
    uncertainty_estimates: Dict[str, float]
    prior_task_similarity: float
    available_compute_budget: float
    real_time_constraints: bool


class UncertaintyAwareLinear(nn.Module):
    """Linear layer with learnable uncertainty estimation."""
    
    def __init__(self, in_features: int, out_features: int, uncertainty_method: str = "gaussian"):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.uncertainty_method = uncertainty_method
        
        # Main weights and biases
        self.weight_mu = nn.Parameter(torch.randn(out_features, in_features) * 0.1)
        self.bias_mu = nn.Parameter(torch.zeros(out_features))
        
        # Uncertainty parameters
        if uncertainty_method == "gaussian":
            self.weight_log_var = nn.Parameter(torch.ones(out_features, in_features) * -3.0)
            self.bias_log_var = nn.Parameter(torch.ones(out_features) * -3.0)
        elif uncertainty_method == "dropout":
            self.dropout_rate = nn.Parameter(torch.tensor(0.1))
        
    def forward(self, x: torch.Tensor, sample_uncertainty: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass with uncertainty estimation."""
        if self.uncertainty_method == "gaussian" and sample_uncertainty:
            # Sample weights from Gaussian distribution
            weight_std = torch.exp(0.5 * self.weight_log_var)
            bias_std = torch.exp(0.5 * self.bias_log_var)
            
            weight = self.weight_mu + weight_std * torch.randn_like(self.weight_mu)
            bias = self.bias_mu + bias_std * torch.randn_like(self.bias_mu)
            
            output = F.linear(x, weight, bias)
            
            # Estimate output uncertainty
            output_var = torch.sum((x.unsqueeze(-1) ** 2) * (weight_std ** 2).unsqueeze(0), dim=1) + bias_std ** 2
            uncertainty = torch.sqrt(output_var)
            
        elif self.uncertainty_method == "dropout":
            output = F.linear(x, self.weight_mu, self.bias_mu)
            if sample_uncertainty and self.training:
                output = F.dropout(output, p=torch.sigmoid(self.dropout_rate))
            uncertainty = torch.abs(output) * torch.sigmoid(self.dropout_rate)
        
        else:
            output = F.linear(x, self.weight_mu, self.bias_mu)
            uncertainty = torch.zeros_like(output)
        
        return output, uncertainty


class HierarchicalTaskEncoder(nn.Module):
    """Hierarchical encoder for multi-level task representations."""
    
    def __init__(self, 
                 observation_dim: int,
                 action_dim: int,
                 hidden_dim: int = 256,
                 num_hierarchy_levels: int = 3):
        super().__init__()
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.num_hierarchy_levels = num_hierarchy_levels
        
        # Multi-level encoders
        self.level_encoders = nn.ModuleList()
        self.level_decoders = nn.ModuleList()
        
        current_dim = observation_dim + action_dim
        for level in range(num_hierarchy_levels):
            # Encoder for this level
            encoder = nn.Sequential(
                UncertaintyAwareLinear(current_dim, hidden_dim),
                nn.ReLU(),
                UncertaintyAwareLinear(hidden_dim, hidden_dim // (2 ** level))
            )
            self.level_encoders.append(encoder)
            
            # Decoder for reconstruction
            decoder = nn.Sequential(
                UncertaintyAwareLinear(hidden_dim // (2 ** level), hidden_dim),
                nn.ReLU(),
                UncertaintyAwareLinear(hidden_dim, current_dim)
            )
            self.level_decoders.append(decoder)
            
            current_dim = hidden_dim // (2 ** level)
        
        # Cross-level attention
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=8,
            batch_first=True
        )
        
    def forward(self, 
                observations: torch.Tensor, 
                actions: torch.Tensor) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """Encode observations and actions hierarchically."""
        # Combine observations and actions
        input_data = torch.cat([observations, actions], dim=-1)
        
        level_representations = []
        level_uncertainties = []
        
        current_input = input_data
        for level in range(self.num_hierarchy_levels):
            encoding, uncertainty = self.level_encoders[level](current_input)
            level_representations.append(encoding)
            level_uncertainties.append(uncertainty)
            
            # Use encoding as input for next level
            current_input = encoding
        
        return level_representations, level_uncertainties


class DynamicInnerLoop(nn.Module):
    """Dynamic inner loop adaptation with learned step sizes and stopping criteria."""
    
    def __init__(self, 
                 model_parameters: int,
                 max_inner_steps: int = 10,
                 min_inner_steps: int = 1):
        super().__init__()
        self.model_parameters = model_parameters
        self.max_inner_steps = max_inner_steps
        self.min_inner_steps = min_inner_steps
        
        # Learned adaptation parameters
        self.step_size_predictor = nn.Sequential(
            nn.Linear(model_parameters + 64, 128),  # +64 for context features
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()  # Step size between 0 and 1
        )
        
        self.stopping_predictor = nn.Sequential(
            nn.Linear(model_parameters + 64, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()  # Probability of stopping
        )
        
        # Context encoder for adaptation context
        self.context_encoder = nn.Sequential(
            nn.Linear(100, 64),  # Assume 100-dim context vector
            nn.ReLU(),
            nn.Linear(64, 64)
        )
        
    def forward(self, 
                initial_params: torch.Tensor,
                gradients: torch.Tensor,
                context: torch.Tensor,
                loss_history: List[float]) -> Tuple[torch.Tensor, int]:
        """Perform dynamic inner loop adaptation."""
        current_params = initial_params.clone()
        encoded_context = self.context_encoder(context)
        
        for step in range(self.max_inner_steps):
            # Prepare input for predictors
            predictor_input = torch.cat([current_params.flatten(), encoded_context.flatten()])
            
            # Predict step size
            step_size = self.step_size_predictor(predictor_input)
            step_size = step_size * 0.1  # Scale to reasonable range
            
            # Apply gradient update
            current_params = current_params - step_size * gradients
            
            # Check stopping condition
            if step >= self.min_inner_steps:
                stop_prob = self.stopping_predictor(predictor_input)
                if torch.rand(1) < stop_prob:
                    break
        
        return current_params, step + 1


class CrossModalTransferModule(nn.Module):
    """Transfer learning across different sensory modalities."""
    
    def __init__(self, 
                 modality_configs: Dict[str, int],
                 shared_embedding_dim: int = 512):
        super().__init__()
        self.modality_configs = modality_configs
        self.shared_embedding_dim = shared_embedding_dim
        
        # Modality-specific encoders
        self.modality_encoders = nn.ModuleDict()
        for modality, input_dim in modality_configs.items():
            self.modality_encoders[modality] = nn.Sequential(
                nn.Linear(input_dim, shared_embedding_dim),
                nn.ReLU(),
                nn.Linear(shared_embedding_dim, shared_embedding_dim)
            )
        
        # Cross-modal attention
        self.cross_modal_attention = nn.MultiheadAttention(
            embed_dim=shared_embedding_dim,
            num_heads=8,
            batch_first=True
        )
        
        # Modality fusion
        self.fusion_network = nn.Sequential(
            nn.Linear(shared_embedding_dim * len(modality_configs), shared_embedding_dim),
            nn.ReLU(),
            nn.Linear(shared_embedding_dim, shared_embedding_dim)
        )
        
    def forward(self, modality_inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Fuse information across modalities."""
        # Encode each modality
        encoded_modalities = []
        for modality, input_data in modality_inputs.items():
            if modality in self.modality_encoders:
                encoded = self.modality_encoders[modality](input_data)
                encoded_modalities.append(encoded)
        
        if len(encoded_modalities) == 0:
            return torch.zeros(1, self.shared_embedding_dim)
        
        # Stack modalities for attention
        modality_stack = torch.stack(encoded_modalities, dim=1)  # [batch, num_modalities, embedding_dim]
        
        # Apply cross-modal attention
        attended_modalities, _ = self.cross_modal_attention(
            modality_stack, modality_stack, modality_stack
        )
        
        # Fuse modalities
        flattened = attended_modalities.flatten(start_dim=1)
        fused_representation = self.fusion_network(flattened)
        
        return fused_representation


class MetaLearningMAMLPlus:
    """Enhanced MAML with hierarchical learning, uncertainty, and cross-modal transfer."""
    
    def __init__(self,
                 observation_space_dim: int,
                 action_space_dim: int,
                 hidden_dim: int = 256,
                 num_hierarchy_levels: int = 3,
                 meta_lr: float = 0.001,
                 inner_lr: float = 0.01,
                 max_inner_steps: int = 10,
                 uncertainty_weight: float = 0.1,
                 modality_configs: Optional[Dict[str, int]] = None):
        """
        Initialize MAML++ for embodied AI.
        
        Args:
            observation_space_dim: Dimension of observation space
            action_space_dim: Dimension of action space
            hidden_dim: Hidden layer dimension
            num_hierarchy_levels: Number of hierarchical encoding levels
            meta_lr: Meta-learning rate
            inner_lr: Initial inner loop learning rate
            max_inner_steps: Maximum inner loop steps
            uncertainty_weight: Weight for uncertainty regularization
            modality_configs: Configuration for different sensory modalities
        """
        self.observation_space_dim = observation_space_dim
        self.action_space_dim = action_space_dim
        self.hidden_dim = hidden_dim
        self.num_hierarchy_levels = num_hierarchy_levels
        self.meta_lr = meta_lr
        self.inner_lr = inner_lr
        self.max_inner_steps = max_inner_steps
        self.uncertainty_weight = uncertainty_weight
        
        # Initialize components
        self.task_encoder = HierarchicalTaskEncoder(
            observation_space_dim, action_space_dim, hidden_dim, num_hierarchy_levels
        )
        
        # Main policy network
        self.policy_network = nn.Sequential(
            UncertaintyAwareLinear(observation_space_dim, hidden_dim),
            nn.ReLU(),
            UncertaintyAwareLinear(hidden_dim, hidden_dim),
            nn.ReLU(),
            UncertaintyAwareLinear(hidden_dim, action_space_dim)
        )
        
        # Value function for advantage estimation
        self.value_network = nn.Sequential(
            UncertaintyAwareLinear(observation_space_dim, hidden_dim),
            nn.ReLU(),
            UncertaintyAwareLinear(hidden_dim, hidden_dim),
            nn.ReLU(),
            UncertaintyAwareLinear(hidden_dim, 1)
        )
        
        # Dynamic inner loop
        param_count = sum(p.numel() for p in self.policy_network.parameters())
        self.dynamic_inner_loop = DynamicInnerLoop(param_count, max_inner_steps)
        
        # Cross-modal transfer
        if modality_configs:
            self.cross_modal_module = CrossModalTransferModule(modality_configs)
        else:
            self.cross_modal_module = None
        
        # Meta-optimizers
        self.meta_optimizer = torch.optim.Adam(
            list(self.policy_network.parameters()) + 
            list(self.value_network.parameters()) +
            list(self.task_encoder.parameters()) +
            list(self.dynamic_inner_loop.parameters()),
            lr=meta_lr
        )
        
        # Task history for meta-meta learning
        self.task_history = deque(maxlen=1000)
        self.adaptation_statistics = defaultdict(list)
        
    def encode_task(self, 
                   support_data: List[Tuple[torch.Tensor, torch.Tensor]],
                   task_metadata: TaskMetadata) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode task from support demonstrations."""
        if len(support_data) == 0:
            return torch.zeros(1, self.hidden_dim), torch.zeros(1, self.hidden_dim)
        
        # Extract observations and actions
        observations = torch.stack([obs for obs, _ in support_data])
        actions = torch.stack([act for _, act in support_data])
        
        # Hierarchical encoding
        level_representations, level_uncertainties = self.task_encoder(observations, actions)
        
        # Combine representations across levels
        combined_representation = torch.cat(level_representations, dim=-1)
        combined_uncertainty = torch.cat(level_uncertainties, dim=-1)
        
        # Pool across demonstrations
        task_embedding = combined_representation.mean(dim=0, keepdim=True)
        task_uncertainty = combined_uncertainty.mean(dim=0, keepdim=True)
        
        return task_embedding, task_uncertainty
    
    def adapt_to_task(self,
                     support_data: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
                     adaptation_context: AdaptationContext) -> Tuple[nn.Module, Dict[str, Any]]:
        """Adapt policy to new task using dynamic inner loop."""
        # Clone network for adaptation
        adapted_policy = type(self.policy_network)()
        adapted_policy.load_state_dict(self.policy_network.state_dict())
        
        # Encode task context
        task_obs_actions = [(obs, act) for obs, act, _ in support_data]
        task_embedding, task_uncertainty = self.encode_task(task_obs_actions, adaptation_context.task_metadata)
        
        # Prepare context for dynamic inner loop
        context_features = self._prepare_adaptation_context(adaptation_context, task_embedding)
        
        adaptation_metrics = {
            'inner_steps': 0,
            'loss_history': [],
            'uncertainty_reduction': 0.0,
            'adaptation_time': 0.0
        }
        
        start_time = time.time()
        
        # Dynamic inner loop adaptation
        loss_history = []
        for step in range(self.max_inner_steps):
            # Compute gradients
            total_loss = 0.0
            policy_gradients = torch.zeros_like(torch.cat([p.flatten() for p in adapted_policy.parameters()]))
            
            for obs, action, reward in support_data:
                # Forward pass
                predicted_action, action_uncertainty = adapted_policy[0](obs.unsqueeze(0))
                predicted_action, _ = adapted_policy[2](predicted_action)
                predicted_action, _ = adapted_policy[4](predicted_action)
                
                # Compute loss (simplified for demonstration)
                action_loss = F.mse_loss(predicted_action.squeeze(), action)
                uncertainty_loss = action_uncertainty.mean() * self.uncertainty_weight
                
                loss = action_loss + uncertainty_loss
                total_loss += loss.item()
                
                # Compute gradients
                loss.backward(retain_graph=True)
            
            loss_history.append(total_loss / len(support_data))
            adaptation_metrics['loss_history'] = loss_history
            
            # Apply dynamic inner loop update
            current_params = torch.cat([p.flatten() for p in adapted_policy.parameters()])
            current_gradients = torch.cat([p.grad.flatten() if p.grad is not None else torch.zeros_like(p.flatten()) 
                                         for p in adapted_policy.parameters()])
            
            updated_params, inner_steps = self.dynamic_inner_loop(
                current_params, current_gradients, context_features, loss_history
            )
            
            # Update policy parameters
            self._update_policy_params(adapted_policy, updated_params)
            
            adaptation_metrics['inner_steps'] = inner_steps
            
            # Early stopping based on loss improvement
            if len(loss_history) > 2 and abs(loss_history[-1] - loss_history[-2]) < 1e-6:
                break
        
        adaptation_metrics['adaptation_time'] = time.time() - start_time
        adaptation_metrics['final_loss'] = loss_history[-1] if loss_history else float('inf')
        
        # Store adaptation statistics
        self.adaptation_statistics['steps'].append(adaptation_metrics['inner_steps'])
        self.adaptation_statistics['time'].append(adaptation_metrics['adaptation_time'])
        self.adaptation_statistics['final_loss'].append(adaptation_metrics['final_loss'])
        
        return adapted_policy, adaptation_metrics
    
    def meta_update(self,
                   task_batch: List[Dict[str, Any]],
                   num_meta_epochs: int = 5) -> Dict[str, float]:
        """Perform meta-update across batch of tasks."""
        meta_losses = []
        meta_uncertainties = []
        
        for epoch in range(num_meta_epochs):
            total_meta_loss = 0.0
            total_uncertainty_loss = 0.0
            
            for task_data in task_batch:
                support_data = task_data['support']
                query_data = task_data['query']
                adaptation_context = task_data['context']
                
                # Adapt to task
                adapted_policy, adaptation_metrics = self.adapt_to_task(support_data, adaptation_context)
                
                # Evaluate on query set
                query_loss = 0.0
                query_uncertainty = 0.0
                
                for obs, action, reward in query_data:
                    # Forward pass with adapted policy
                    pred_action, action_uncertainty = adapted_policy[0](obs.unsqueeze(0))
                    pred_action, _ = adapted_policy[2](pred_action)
                    pred_action, _ = adapted_policy[4](pred_action)
                    
                    # Compute query loss
                    loss = F.mse_loss(pred_action.squeeze(), action)
                    uncertainty = action_uncertainty.mean()
                    
                    query_loss += loss
                    query_uncertainty += uncertainty
                
                query_loss /= len(query_data)
                query_uncertainty /= len(query_data)
                
                total_meta_loss += query_loss
                total_uncertainty_loss += query_uncertainty
            
            # Meta-gradient update
            meta_loss = total_meta_loss / len(task_batch)
            uncertainty_loss = total_uncertainty_loss / len(task_batch)
            
            total_loss = meta_loss + self.uncertainty_weight * uncertainty_loss
            
            self.meta_optimizer.zero_grad()
            total_loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                list(self.policy_network.parameters()) + 
                list(self.value_network.parameters()) +
                list(self.task_encoder.parameters()),
                max_norm=1.0
            )
            
            self.meta_optimizer.step()
            
            meta_losses.append(meta_loss.item())
            meta_uncertainties.append(uncertainty_loss.item())
        
        return {
            'meta_loss': np.mean(meta_losses),
            'uncertainty_loss': np.mean(meta_uncertainties),
            'adaptation_stats': {
                'avg_steps': np.mean(self.adaptation_statistics['steps'][-100:]),
                'avg_time': np.mean(self.adaptation_statistics['time'][-100:]),
                'avg_final_loss': np.mean(self.adaptation_statistics['final_loss'][-100:])
            }
        }
    
    def _prepare_adaptation_context(self, 
                                  adaptation_context: AdaptationContext,
                                  task_embedding: torch.Tensor) -> torch.Tensor:
        """Prepare context vector for dynamic inner loop."""
        # Combine various context information
        context_features = [
            task_embedding.flatten(),
            torch.tensor([adaptation_context.task_metadata.difficulty_level]),
            torch.tensor([len(adaptation_context.support_demonstrations)]),
            torch.tensor([adaptation_context.prior_task_similarity]),
            torch.tensor([adaptation_context.available_compute_budget]),
            torch.tensor([float(adaptation_context.real_time_constraints)])
        ]
        
        # Pad to fixed size (100 dimensions)
        context_vector = torch.cat(context_features)
        if context_vector.size(0) < 100:
            padding = torch.zeros(100 - context_vector.size(0))
            context_vector = torch.cat([context_vector, padding])
        elif context_vector.size(0) > 100:
            context_vector = context_vector[:100]
        
        return context_vector
    
    def _update_policy_params(self, policy: nn.Module, new_params: torch.Tensor):
        """Update policy parameters from flattened parameter vector."""
        param_idx = 0
        for param in policy.parameters():
            param_size = param.numel()
            new_param_data = new_params[param_idx:param_idx + param_size].view(param.shape)
            param.data.copy_(new_param_data)
            param_idx += param_size
    
    def evaluate_few_shot_performance(self,
                                    test_tasks: List[Dict[str, Any]],
                                    k_shot: int = 5) -> Dict[str, Any]:
        """Evaluate few-shot learning performance."""
        results = {
            'success_rates': [],
            'adaptation_times': [],
            'transfer_effectiveness': [],
            'uncertainty_calibration': []
        }
        
        for task_data in test_tasks:
            # Sample k-shot support set
            support_data = task_data['support'][:k_shot]
            query_data = task_data['query']
            adaptation_context = task_data['context']
            
            # Adapt to task
            adapted_policy, adaptation_metrics = self.adapt_to_task(support_data, adaptation_context)
            
            # Evaluate on query set
            correct_predictions = 0
            total_uncertainty = 0
            
            for obs, action, reward in query_data:
                with torch.no_grad():
                    pred_action, action_uncertainty = adapted_policy[0](obs.unsqueeze(0))
                    pred_action, _ = adapted_policy[2](pred_action)
                    pred_action, _ = adapted_policy[4](pred_action)
                
                # Check if prediction is close to target
                if torch.norm(pred_action.squeeze() - action) < 0.1:  # Threshold for "correct"
                    correct_predictions += 1
                
                total_uncertainty += action_uncertainty.mean().item()
            
            # Compute metrics
            success_rate = correct_predictions / len(query_data)
            avg_uncertainty = total_uncertainty / len(query_data)
            
            results['success_rates'].append(success_rate)
            results['adaptation_times'].append(adaptation_metrics['adaptation_time'])
            results['uncertainty_calibration'].append(avg_uncertainty)
            
            # Transfer effectiveness (simplified)
            baseline_performance = 0.2  # Random baseline
            transfer_effectiveness = (success_rate - baseline_performance) / (1.0 - baseline_performance)
            results['transfer_effectiveness'].append(max(0.0, transfer_effectiveness))
        
        # Aggregate results
        return {
            'avg_success_rate': np.mean(results['success_rates']),
            'std_success_rate': np.std(results['success_rates']),
            'avg_adaptation_time': np.mean(results['adaptation_times']),
            'avg_transfer_effectiveness': np.mean(results['transfer_effectiveness']),
            'avg_uncertainty': np.mean(results['uncertainty_calibration']),
            'few_shot_improvement': np.mean(results['success_rates']) - 0.2  # vs random baseline
        }
    
    def save_meta_learner(self, filepath: str):
        """Save meta-learner state."""
        state = {
            'policy_network': self.policy_network.state_dict(),
            'value_network': self.value_network.state_dict(),
            'task_encoder': self.task_encoder.state_dict(),
            'dynamic_inner_loop': self.dynamic_inner_loop.state_dict(),
            'meta_optimizer': self.meta_optimizer.state_dict(),
            'adaptation_statistics': dict(self.adaptation_statistics),
            'hyperparameters': {
                'observation_space_dim': self.observation_space_dim,
                'action_space_dim': self.action_space_dim,
                'hidden_dim': self.hidden_dim,
                'meta_lr': self.meta_lr,
                'inner_lr': self.inner_lr,
                'uncertainty_weight': self.uncertainty_weight
            }
        }
        
        if self.cross_modal_module:
            state['cross_modal_module'] = self.cross_modal_module.state_dict()
        
        torch.save(state, filepath)
        logger.info(f"Saved meta-learner to {filepath}")
    
    def load_meta_learner(self, filepath: str):
        """Load meta-learner state."""
        state = torch.load(filepath)
        
        self.policy_network.load_state_dict(state['policy_network'])
        self.value_network.load_state_dict(state['value_network'])
        self.task_encoder.load_state_dict(state['task_encoder'])
        self.dynamic_inner_loop.load_state_dict(state['dynamic_inner_loop'])
        self.meta_optimizer.load_state_dict(state['meta_optimizer'])
        self.adaptation_statistics = defaultdict(list, state['adaptation_statistics'])
        
        if 'cross_modal_module' in state and self.cross_modal_module:
            self.cross_modal_module.load_state_dict(state['cross_modal_module'])
        
        logger.info(f"Loaded meta-learner from {filepath}")
    
    def get_meta_learning_metrics(self) -> Dict[str, Any]:
        """Get comprehensive meta-learning metrics."""
        if len(self.adaptation_statistics['steps']) == 0:
            return {'status': 'no_data'}
        
        return {
            'adaptation_efficiency': {
                'avg_steps': np.mean(self.adaptation_statistics['steps']),
                'std_steps': np.std(self.adaptation_statistics['steps']),
                'min_steps': np.min(self.adaptation_statistics['steps']),
                'max_steps': np.max(self.adaptation_statistics['steps'])
            },
            'timing_performance': {
                'avg_adaptation_time': np.mean(self.adaptation_statistics['time']),
                'std_adaptation_time': np.std(self.adaptation_statistics['time']),
                'total_adaptation_time': np.sum(self.adaptation_statistics['time'])
            },
            'learning_progression': {
                'avg_final_loss': np.mean(self.adaptation_statistics['final_loss']),
                'loss_improvement_trend': np.polyfit(
                    range(len(self.adaptation_statistics['final_loss'])),
                    self.adaptation_statistics['final_loss'],
                    1
                )[0] if len(self.adaptation_statistics['final_loss']) > 1 else 0.0
            },
            'meta_learning_stability': {
                'step_variance': np.var(self.adaptation_statistics['steps']),
                'time_variance': np.var(self.adaptation_statistics['time']),
                'convergence_rate': len([loss for loss in self.adaptation_statistics['final_loss'] if loss < 0.1]) / len(self.adaptation_statistics['final_loss'])
            }
        }