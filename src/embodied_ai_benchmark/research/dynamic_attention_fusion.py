"""Dynamic Attention Fusion for Multi-Modal Embodied AI."""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import time
from concurrent.futures import ThreadPoolExecutor

from ..utils.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class AttentionConfig:
    """Configuration for dynamic attention fusion."""
    num_modalities: int = 4  # RGB, depth, tactile, proprioception
    hidden_dim: int = 512
    num_heads: int = 8
    dropout_rate: float = 0.1
    temperature: float = 1.0
    adaptive_threshold: float = 0.1
    cross_modal_layers: int = 3
    temporal_window: int = 16
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class DynamicCrossModalAttention(nn.Module):
    """Novel dynamic cross-modal attention mechanism."""
    
    def __init__(self, config: AttentionConfig):
        super().__init__()
        self.config = config
        self.hidden_dim = config.hidden_dim
        self.num_heads = config.num_heads
        self.head_dim = config.hidden_dim // config.num_heads
        
        # Multi-modal encoders
        self.modality_encoders = nn.ModuleDict({
            'rgb': nn.Sequential(
                nn.Linear(2048, config.hidden_dim),  # ResNet features
                nn.LayerNorm(config.hidden_dim),
                nn.ReLU()
            ),
            'depth': nn.Sequential(
                nn.Linear(1024, config.hidden_dim),
                nn.LayerNorm(config.hidden_dim),
                nn.ReLU()
            ),
            'tactile': nn.Sequential(
                nn.Linear(64, config.hidden_dim),
                nn.LayerNorm(config.hidden_dim),
                nn.ReLU()
            ),
            'proprioception': nn.Sequential(
                nn.Linear(32, config.hidden_dim),
                nn.LayerNorm(config.hidden_dim),
                nn.ReLU()
            )
        })
        
        # Dynamic attention weights
        self.attention_controller = nn.Sequential(
            nn.Linear(config.hidden_dim * config.num_modalities, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.num_modalities),
            nn.Softmax(dim=-1)
        )
        
        # Cross-modal transformers
        self.cross_modal_layers = nn.ModuleList([
            CrossModalTransformerLayer(config) for _ in range(config.cross_modal_layers)
        ])
        
        # Temporal fusion
        self.temporal_encoder = nn.LSTM(
            config.hidden_dim, config.hidden_dim, 
            batch_first=True, bidirectional=True
        )
        self.temporal_projection = nn.Linear(config.hidden_dim * 2, config.hidden_dim)
        
        # Adaptive threshold learning
        self.threshold_predictor = nn.Sequential(
            nn.Linear(config.hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
    def forward(self, modality_inputs: Dict[str, torch.Tensor], 
                temporal_history: Optional[List[Dict[str, torch.Tensor]]] = None) -> Dict[str, torch.Tensor]:
        """
        Forward pass with dynamic attention fusion.
        
        Args:
            modality_inputs: Dictionary of modality tensors
            temporal_history: Optional temporal context
            
        Returns:
            Fused representation and attention weights
        """
        batch_size = next(iter(modality_inputs.values())).size(0)
        device = next(iter(modality_inputs.values())).device
        
        # Encode each modality
        encoded_modalities = {}
        for modality, features in modality_inputs.items():
            if modality in self.modality_encoders:
                encoded_modalities[modality] = self.modality_encoders[modality](features)
        
        # Stack modalities for attention computation
        modality_stack = torch.stack(list(encoded_modalities.values()), dim=1)  # [B, M, H]
        
        # Compute dynamic attention weights
        concat_features = torch.cat(list(encoded_modalities.values()), dim=-1)
        attention_weights = self.attention_controller(concat_features)  # [B, M]
        
        # Apply cross-modal attention layers
        fused_representation = modality_stack
        layer_attentions = []
        
        for layer in self.cross_modal_layers:
            fused_representation, layer_attention = layer(fused_representation, attention_weights)
            layer_attentions.append(layer_attention)
        
        # Temporal fusion if history is provided
        if temporal_history:
            temporal_features = self._encode_temporal_context(
                fused_representation, temporal_history, encoded_modalities.keys()
            )
            fused_representation = temporal_features
        
        # Final weighted fusion
        weighted_modalities = fused_representation * attention_weights.unsqueeze(-1)
        final_representation = weighted_modalities.sum(dim=1)  # [B, H]
        
        # Adaptive thresholding
        confidence_threshold = self.threshold_predictor(final_representation)
        
        return {
            'fused_features': final_representation,
            'attention_weights': attention_weights,
            'layer_attentions': layer_attentions,
            'confidence_threshold': confidence_threshold,
            'modality_contributions': {
                mod: weighted_modalities[:, i, :] 
                for i, mod in enumerate(encoded_modalities.keys())
            }
        }
    
    def _encode_temporal_context(self, current_features: torch.Tensor, 
                                history: List[Dict[str, torch.Tensor]], 
                                modality_keys: List[str]) -> torch.Tensor:
        """Encode temporal context using LSTM."""
        # Create temporal sequence
        temporal_sequence = [current_features]
        
        for hist_step in history[-self.config.temporal_window+1:]:
            hist_encoded = []
            for modality in modality_keys:
                if modality in hist_step and modality in self.modality_encoders:
                    encoded = self.modality_encoders[modality](hist_step[modality])
                    hist_encoded.append(encoded)
            
            if hist_encoded:
                hist_features = torch.stack(hist_encoded, dim=1)
                temporal_sequence.append(hist_features)
        
        # Stack temporal sequence
        if len(temporal_sequence) > 1:
            temporal_tensor = torch.stack(temporal_sequence, dim=1)  # [B, T, M, H]
            batch_size, seq_len, num_mods, hidden_dim = temporal_tensor.shape
            
            # Flatten for LSTM
            temporal_flat = temporal_tensor.view(batch_size, seq_len, -1)
            
            # LSTM encoding
            lstm_out, _ = self.temporal_encoder(temporal_flat)
            temporal_encoded = self.temporal_projection(lstm_out[:, -1, :])  # Use last output
            
            # Reshape back to modality structure
            return temporal_encoded.view(batch_size, num_mods, hidden_dim)
        
        return current_features


class CrossModalTransformerLayer(nn.Module):
    """Cross-modal transformer layer with dynamic routing."""
    
    def __init__(self, config: AttentionConfig):
        super().__init__()
        self.config = config
        self.multihead_attn = nn.MultiheadAttention(
            config.hidden_dim, config.num_heads, 
            dropout=config.dropout_rate, batch_first=True
        )
        
        self.norm1 = nn.LayerNorm(config.hidden_dim)
        self.norm2 = nn.LayerNorm(config.hidden_dim)
        
        self.ffn = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(config.dropout_rate),
            nn.Linear(config.hidden_dim * 4, config.hidden_dim),
            nn.Dropout(config.dropout_rate)
        )
        
        # Dynamic routing mechanism
        self.routing_weights = nn.Parameter(torch.randn(config.num_modalities, config.num_modalities))
        
    def forward(self, x: torch.Tensor, attention_weights: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with dynamic routing.
        
        Args:
            x: Input tensor [B, M, H]
            attention_weights: Dynamic attention weights [B, M]
            
        Returns:
            Output tensor and attention matrix
        """
        batch_size, num_modalities, hidden_dim = x.shape
        
        # Dynamic routing based on attention weights
        routing_matrix = F.softmax(self.routing_weights, dim=-1)
        routed_attention = torch.matmul(attention_weights.unsqueeze(1), routing_matrix)
        
        # Self-attention with dynamic masking
        attn_mask = self._create_dynamic_mask(routed_attention)
        attn_output, attn_weights = self.multihead_attn(
            x, x, x, attn_mask=attn_mask
        )
        
        # Residual connection and normalization
        x = self.norm1(x + attn_output)
        
        # Feed-forward network
        ffn_output = self.ffn(x)
        x = self.norm2(x + ffn_output)
        
        return x, attn_weights
    
    def _create_dynamic_mask(self, attention_weights: torch.Tensor) -> torch.Tensor:
        """Create dynamic attention mask based on modality relevance."""
        batch_size, num_modalities = attention_weights.shape
        
        # Create mask that suppresses low-attention modalities
        threshold = self.config.adaptive_threshold
        mask = (attention_weights < threshold).unsqueeze(-1).expand(-1, -1, num_modalities)
        
        # Convert to attention mask format (additive)
        attn_mask = mask.float() * -1e9
        
        return attn_mask


class AdaptiveModalitySelector:
    """Adaptive modality selection based on task requirements."""
    
    def __init__(self, config: AttentionConfig):
        self.config = config
        self.modality_history = []
        self.performance_tracking = {}
        
    def select_modalities(self, available_modalities: List[str], 
                         task_context: Dict[str, Any],
                         performance_feedback: Optional[float] = None) -> List[str]:
        """
        Dynamically select optimal modalities for current task.
        
        Args:
            available_modalities: List of available modality types
            task_context: Context about current task
            performance_feedback: Performance from previous selection
            
        Returns:
            Selected modalities for optimal performance
        """
        # Update performance tracking
        if performance_feedback is not None and self.modality_history:
            last_selection = self.modality_history[-1]
            selection_key = tuple(sorted(last_selection))
            
            if selection_key not in self.performance_tracking:
                self.performance_tracking[selection_key] = []
            self.performance_tracking[selection_key].append(performance_feedback)
        
        # Task-based modality scoring
        modality_scores = self._score_modalities(available_modalities, task_context)
        
        # Performance-based adjustment
        adjusted_scores = self._adjust_for_performance(modality_scores)
        
        # Select top modalities
        sorted_modalities = sorted(adjusted_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Dynamic selection based on confidence
        selected = []
        total_score = sum(adjusted_scores.values())
        cumulative_score = 0
        
        for modality, score in sorted_modalities:
            cumulative_score += score
            selected.append(modality)
            
            # Stop when we have sufficient coverage
            if cumulative_score / total_score > 0.8 or len(selected) >= 3:
                break
        
        # Ensure minimum modalities
        if len(selected) < 2:
            selected = [mod for mod, _ in sorted_modalities[:2]]
        
        self.modality_history.append(selected)
        
        return selected
    
    def _score_modalities(self, modalities: List[str], task_context: Dict[str, Any]) -> Dict[str, float]:
        """Score modalities based on task context."""
        scores = {}
        
        # Base scores
        base_scores = {
            'rgb': 0.8,
            'depth': 0.7,
            'tactile': 0.6,
            'proprioception': 0.9
        }
        
        # Task-specific adjustments
        task_type = task_context.get('task_type', 'unknown')
        
        for modality in modalities:
            score = base_scores.get(modality, 0.5)
            
            # Task-specific boosts
            if task_type == 'manipulation':
                if modality in ['tactile', 'proprioception']:
                    score *= 1.3
            elif task_type == 'navigation':
                if modality in ['rgb', 'depth']:
                    score *= 1.2
            elif task_type == 'cooperative':
                if modality == 'rgb':  # For communication
                    score *= 1.1
            
            # Environmental adjustments
            lighting = task_context.get('lighting', 'normal')
            if lighting == 'dim' and modality == 'rgb':
                score *= 0.7
            elif lighting == 'dim' and modality == 'depth':
                score *= 1.2
            
            scores[modality] = score
        
        return scores
    
    def _adjust_for_performance(self, base_scores: Dict[str, float]) -> Dict[str, float]:
        """Adjust scores based on historical performance."""
        adjusted_scores = base_scores.copy()
        
        # Find best performing combinations
        if self.performance_tracking:
            best_combo = max(self.performance_tracking.items(), 
                           key=lambda x: np.mean(x[1]))
            best_modalities = best_combo[0]
            best_performance = np.mean(best_combo[1])
            
            # Boost modalities in best combinations
            for modality in best_modalities:
                if modality in adjusted_scores:
                    adjusted_scores[modality] *= (1 + best_performance * 0.2)
        
        return adjusted_scores


class PerformanceAnalyzer:
    """Analyze performance of dynamic attention fusion."""
    
    def __init__(self):
        self.metrics_history = []
        self.attention_patterns = []
        
    def analyze_attention_patterns(self, attention_outputs: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        """Analyze attention patterns for insights."""
        attention_weights = attention_outputs['attention_weights'].detach().cpu().numpy()
        layer_attentions = [la.detach().cpu().numpy() for la in attention_outputs['layer_attentions']]
        
        analysis = {
            'modality_dominance': self._analyze_modality_dominance(attention_weights),
            'attention_stability': self._analyze_attention_stability(attention_weights),
            'layer_evolution': self._analyze_layer_evolution(layer_attentions),
            'efficiency_metrics': self._compute_efficiency_metrics(attention_outputs)
        }
        
        self.attention_patterns.append(analysis)
        
        return analysis
    
    def _analyze_modality_dominance(self, attention_weights: np.ndarray) -> Dict[str, float]:
        """Analyze which modalities dominate attention."""
        mean_attention = np.mean(attention_weights, axis=0)
        std_attention = np.std(attention_weights, axis=0)
        
        modality_names = ['rgb', 'depth', 'tactile', 'proprioception']
        
        return {
            'dominance_order': [modality_names[i] for i in np.argsort(mean_attention)[::-1]],
            'dominance_scores': {modality_names[i]: mean_attention[i] for i in range(len(modality_names))},
            'stability_scores': {modality_names[i]: std_attention[i] for i in range(len(modality_names))},
            'entropy': -np.sum(mean_attention * np.log(mean_attention + 1e-8))
        }
    
    def _analyze_attention_stability(self, attention_weights: np.ndarray) -> Dict[str, float]:
        """Analyze stability of attention patterns."""
        return {
            'temporal_variance': np.var(attention_weights, axis=0).mean(),
            'attention_switching_rate': self._compute_switching_rate(attention_weights),
            'consistency_score': 1.0 - np.mean(np.std(attention_weights, axis=0))
        }
    
    def _analyze_layer_evolution(self, layer_attentions: List[np.ndarray]) -> Dict[str, Any]:
        """Analyze how attention evolves through layers."""
        if not layer_attentions:
            return {}
        
        # Compute attention evolution metrics
        layer_changes = []
        for i in range(1, len(layer_attentions)):
            change = np.mean(np.abs(layer_attentions[i] - layer_attentions[i-1]))
            layer_changes.append(change)
        
        return {
            'layer_changes': layer_changes,
            'refinement_trend': np.polyfit(range(len(layer_changes)), layer_changes, 1)[0],
            'final_layer_focus': np.max(layer_attentions[-1], axis=-1).mean() if layer_attentions else 0
        }
    
    def _compute_efficiency_metrics(self, attention_outputs: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Compute efficiency metrics for attention mechanism."""
        attention_weights = attention_outputs['attention_weights']
        
        return {
            'sparsity': torch.mean((attention_weights < 0.1).float()).item(),
            'entropy': -torch.sum(attention_weights * torch.log(attention_weights + 1e-8), dim=-1).mean().item(),
            'max_attention_ratio': torch.max(attention_weights, dim=-1)[0].mean().item(),
            'effective_modalities': torch.mean((attention_weights > 0.2).sum(dim=-1).float()).item()
        }
    
    def _compute_switching_rate(self, attention_weights: np.ndarray) -> float:
        """Compute rate of attention switching between timesteps."""
        if len(attention_weights) < 2:
            return 0.0
        
        switches = 0
        for i in range(1, len(attention_weights)):
            prev_max = np.argmax(attention_weights[i-1])
            curr_max = np.argmax(attention_weights[i])
            if prev_max != curr_max:
                switches += 1
        
        return switches / (len(attention_weights) - 1)
    
    def generate_performance_report(self) -> str:
        """Generate comprehensive performance report."""
        if not self.attention_patterns:
            return "No attention patterns recorded yet."
        
        recent_patterns = self.attention_patterns[-10:]  # Last 10 analyses
        
        # Aggregate statistics
        avg_entropy = np.mean([p['modality_dominance']['entropy'] for p in recent_patterns])
        avg_efficiency = np.mean([p['efficiency_metrics']['effective_modalities'] for p in recent_patterns])
        avg_stability = np.mean([p['attention_stability']['consistency_score'] for p in recent_patterns])
        
        report = f"""
Dynamic Attention Fusion Performance Report
==========================================

Attention Quality Metrics:
- Average Attention Entropy: {avg_entropy:.3f}
- Effective Modalities Used: {avg_efficiency:.2f}
- Attention Stability Score: {avg_stability:.3f}

Modality Usage Patterns:
"""
        
        # Most common dominant modality
        dominant_modalities = [p['modality_dominance']['dominance_order'][0] for p in recent_patterns]
        from collections import Counter
        modality_counts = Counter(dominant_modalities)
        
        for modality, count in modality_counts.most_common():
            percentage = count / len(recent_patterns) * 100
            report += f"- {modality}: {percentage:.1f}% of time dominant\n"
        
        return report


def create_dynamic_attention_fusion(config: Optional[AttentionConfig] = None) -> DynamicCrossModalAttention:
    """Factory function to create dynamic attention fusion model."""
    if config is None:
        config = AttentionConfig()
    
    model = DynamicCrossModalAttention(config)
    
    logger.info(f"Created Dynamic Attention Fusion model with {config.num_modalities} modalities")
    logger.info(f"Hidden dimension: {config.hidden_dim}, Heads: {config.num_heads}")
    
    return model


def benchmark_attention_fusion(model: DynamicCrossModalAttention, 
                              num_trials: int = 100,
                              batch_size: int = 32) -> Dict[str, float]:
    """Benchmark the attention fusion model."""
    logger.info(f"Benchmarking Dynamic Attention Fusion with {num_trials} trials")
    
    device = next(model.parameters()).device
    
    # Create synthetic test data
    test_inputs = {
        'rgb': torch.randn(batch_size, 2048, device=device),
        'depth': torch.randn(batch_size, 1024, device=device),
        'tactile': torch.randn(batch_size, 64, device=device),
        'proprioception': torch.randn(batch_size, 32, device=device)
    }
    
    # Benchmark performance
    model.eval()
    
    total_time = 0
    memory_usage = []
    attention_entropies = []
    
    with torch.no_grad():
        for trial in range(num_trials):
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
            
            start_time = time.time()
            
            outputs = model(test_inputs)
            
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            
            end_time = time.time()
            total_time += (end_time - start_time)
            
            # Track memory usage
            if torch.cuda.is_available():
                memory_usage.append(torch.cuda.max_memory_allocated() / 1024**2)  # MB
            
            # Track attention quality
            attention_weights = outputs['attention_weights']
            entropy = -torch.sum(attention_weights * torch.log(attention_weights + 1e-8), dim=-1).mean()
            attention_entropies.append(entropy.item())
    
    results = {
        'avg_inference_time': total_time / num_trials,
        'throughput_samples_per_sec': batch_size * num_trials / total_time,
        'avg_memory_mb': np.mean(memory_usage) if memory_usage else 0,
        'avg_attention_entropy': np.mean(attention_entropies),
        'attention_entropy_std': np.std(attention_entropies)
    }
    
    logger.info(f"Benchmark results: {results}")
    
    return results