"""Multi-Modal Sensory Fusion Framework with Cross-Modal Learning.

Novel contributions:
1. Attention-based cross-modal fusion with uncertainty quantification
2. Self-supervised cross-modal representation learning
3. Dynamic modality weighting based on reliability
4. Hierarchical multimodal reasoning with symbolic grounding
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import MultiheadAttention, TransformerEncoder, TransformerEncoderLayer
import numpy as np
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import time
import math
from collections import defaultdict, deque
from abc import ABC, abstractmethod
import cv2
from scipy.spatial.distance import cosine
from sklearn.decomposition import PCA

from ..utils.logging_config import get_logger

logger = get_logger(__name__)


class ModalityType(Enum):
    """Types of sensory modalities."""
    VISION_RGB = "vision_rgb"
    VISION_DEPTH = "vision_depth"
    VISION_SEMANTIC = "vision_semantic"
    AUDIO = "audio"
    TACTILE = "tactile"
    PROPRIOCEPTION = "proprioception"
    FORCE_TORQUE = "force_torque"
    LANGUAGE = "language"
    LIDAR = "lidar"
    IMU = "imu"


@dataclass
class ModalityData:
    """Container for sensory modality data."""
    modality_type: ModalityType
    data: torch.Tensor
    timestamp: float
    confidence: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    preprocessing_applied: List[str] = field(default_factory=list)
    
    
@dataclass
class FusionResult:
    """Result of multimodal fusion."""
    fused_representation: torch.Tensor
    modality_contributions: Dict[ModalityType, float]
    uncertainty_estimate: float
    cross_modal_alignments: Dict[Tuple[ModalityType, ModalityType], float]
    reasoning_trace: List[str]
    confidence_score: float


class ModalityEncoder(nn.Module):
    """Individual modality encoder with uncertainty estimation."""
    
    def __init__(self,
                 modality_type: ModalityType,
                 input_dim: int,
                 output_dim: int = 512,
                 use_attention: bool = True):
        super().__init__()
        self.modality_type = modality_type
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.use_attention = use_attention
        
        # Modality-specific preprocessing
        if modality_type in [ModalityType.VISION_RGB, ModalityType.VISION_DEPTH]:
            self.encoder = self._build_vision_encoder(input_dim, output_dim)
        elif modality_type == ModalityType.AUDIO:
            self.encoder = self._build_audio_encoder(input_dim, output_dim)
        elif modality_type in [ModalityType.TACTILE, ModalityType.FORCE_TORQUE]:
            self.encoder = self._build_tactile_encoder(input_dim, output_dim)
        elif modality_type == ModalityType.LANGUAGE:
            self.encoder = self._build_language_encoder(input_dim, output_dim)
        elif modality_type == ModalityType.PROPRIOCEPTION:
            self.encoder = self._build_proprioception_encoder(input_dim, output_dim)
        else:
            self.encoder = self._build_generic_encoder(input_dim, output_dim)
        
        # Uncertainty estimation
        self.uncertainty_head = nn.Sequential(
            nn.Linear(output_dim, output_dim // 2),
            nn.ReLU(),
            nn.Linear(output_dim // 2, 1),
            nn.Sigmoid()
        )
        
        # Self-attention for temporal modeling
        if use_attention:
            self.temporal_attention = MultiheadAttention(
                embed_dim=output_dim,
                num_heads=8,
                batch_first=True
            )
    
    def _build_vision_encoder(self, input_dim: int, output_dim: int) -> nn.Module:
        """Build CNN-based encoder for vision modalities."""
        # Assume input is flattened image patches or features
        return nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, output_dim)
        )
    
    def _build_audio_encoder(self, input_dim: int, output_dim: int) -> nn.Module:
        """Build encoder for audio modality."""
        return nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, output_dim)
        )
    
    def _build_tactile_encoder(self, input_dim: int, output_dim: int) -> nn.Module:
        """Build encoder for tactile/force modalities."""
        return nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )
    
    def _build_language_encoder(self, input_dim: int, output_dim: int) -> nn.Module:
        """Build transformer-based encoder for language."""
        encoder_layer = TransformerEncoderLayer(
            d_model=input_dim,
            nhead=8,
            dim_feedforward=input_dim * 2,
            batch_first=True
        )
        transformer = TransformerEncoder(encoder_layer, num_layers=3)
        
        return nn.Sequential(
            transformer,
            nn.Linear(input_dim, output_dim)
        )
    
    def _build_proprioception_encoder(self, input_dim: int, output_dim: int) -> nn.Module:
        """Build encoder for proprioception (joint angles, etc.)."""
        return nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )
    
    def _build_generic_encoder(self, input_dim: int, output_dim: int) -> nn.Module:
        """Build generic encoder for other modalities."""
        return nn.Sequential(
            nn.Linear(input_dim, min(512, input_dim * 2)),
            nn.ReLU(),
            nn.Linear(min(512, input_dim * 2), output_dim)
        )
    
    def forward(self, 
                x: torch.Tensor,
                temporal_context: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode modality data.
        
        Args:
            x: Input data [batch_size, input_dim] or [batch_size, seq_len, input_dim]
            temporal_context: Optional temporal context
            
        Returns:
            encoded: Encoded representation [batch_size, output_dim]
            uncertainty: Uncertainty estimate [batch_size, 1]
        """
        # Handle different input shapes
        if x.dim() == 3 and self.use_attention:
            # Temporal sequence input
            encoded, _ = self.temporal_attention(x, x, x)
            encoded = encoded.mean(dim=1)  # Average pooling
        else:
            # Single timestep or pre-pooled input
            if x.dim() == 3:
                x = x.mean(dim=1)  # Average pool temporal dimension
            encoded = self.encoder(x)
        
        # Estimate uncertainty
        uncertainty = self.uncertainty_head(encoded)
        
        return encoded, uncertainty


class CrossModalAttention(nn.Module):
    """Cross-modal attention mechanism for modality fusion."""
    
    def __init__(self,
                 feature_dim: int = 512,
                 num_heads: int = 8,
                 temperature: float = 1.0):
        super().__init__()
        self.feature_dim = feature_dim
        self.num_heads = num_heads
        self.temperature = temperature
        
        self.query_proj = nn.Linear(feature_dim, feature_dim)
        self.key_proj = nn.Linear(feature_dim, feature_dim)
        self.value_proj = nn.Linear(feature_dim, feature_dim)
        
        self.output_proj = nn.Linear(feature_dim, feature_dim)
        
        # Cross-modal alignment predictor
        self.alignment_predictor = nn.Sequential(
            nn.Linear(feature_dim * 2, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, 1),
            nn.Sigmoid()
        )
    
    def forward(self,
                query_features: torch.Tensor,
                key_features: torch.Tensor,
                value_features: torch.Tensor,
                uncertainty_weights: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Apply cross-modal attention.
        
        Args:
            query_features: Query modality features [batch_size, feature_dim]
            key_features: Key modality features [batch_size, feature_dim]
            value_features: Value modality features [batch_size, feature_dim]
            uncertainty_weights: Optional uncertainty weights [batch_size, 1]
            
        Returns:
            attended_features: Attended output [batch_size, feature_dim]
            attention_weights: Attention weights [batch_size, 1]
            alignment_score: Cross-modal alignment score [batch_size, 1]
        """
        batch_size = query_features.size(0)
        
        # Project to query, key, value
        q = self.query_proj(query_features)  # [batch_size, feature_dim]
        k = self.key_proj(key_features)      # [batch_size, feature_dim]
        v = self.value_proj(value_features)  # [batch_size, feature_dim]
        
        # Compute attention scores
        attention_scores = torch.sum(q * k, dim=-1, keepdim=True) / math.sqrt(self.feature_dim)
        attention_scores = attention_scores / self.temperature
        
        # Apply uncertainty weighting if provided
        if uncertainty_weights is not None:
            attention_scores = attention_scores * uncertainty_weights
        
        attention_weights = torch.softmax(attention_scores, dim=0)
        
        # Apply attention to values
        attended_features = attention_weights * v
        attended_features = self.output_proj(attended_features)
        
        # Compute cross-modal alignment
        concatenated_features = torch.cat([query_features, key_features], dim=-1)
        alignment_score = self.alignment_predictor(concatenated_features)
        
        return attended_features, attention_weights, alignment_score


class SelfSupervisedCrossModalLearner(nn.Module):
    """Self-supervised learning for cross-modal representations."""
    
    def __init__(self,
                 feature_dim: int = 512,
                 projection_dim: int = 128,
                 temperature: float = 0.07):
        super().__init__()
        self.feature_dim = feature_dim
        self.projection_dim = projection_dim
        self.temperature = temperature
        
        # Projection heads for contrastive learning
        self.projection_heads = nn.ModuleDict()
        
        # Cross-modal prediction networks
        self.cross_modal_predictors = nn.ModuleDict()
        
        # Reconstruction networks
        self.reconstruction_heads = nn.ModuleDict()
    
    def add_modality(self, modality_type: ModalityType, input_dim: int):
        """Add modality-specific projection and prediction heads."""
        modality_name = modality_type.value
        
        # Projection head for contrastive learning
        self.projection_heads[modality_name] = nn.Sequential(
            nn.Linear(self.feature_dim, self.feature_dim),
            nn.ReLU(),
            nn.Linear(self.feature_dim, self.projection_dim)
        )
        
        # Reconstruction head
        self.reconstruction_heads[modality_name] = nn.Sequential(
            nn.Linear(self.feature_dim, self.feature_dim),
            nn.ReLU(),
            nn.Linear(self.feature_dim, input_dim)
        )
    
    def add_cross_modal_predictor(self, 
                                 source_modality: ModalityType,
                                 target_modality: ModalityType):
        """Add cross-modal prediction network."""
        predictor_name = f"{source_modality.value}_to_{target_modality.value}"
        
        self.cross_modal_predictors[predictor_name] = nn.Sequential(
            nn.Linear(self.feature_dim, self.feature_dim),
            nn.ReLU(),
            nn.Linear(self.feature_dim, self.feature_dim // 2),
            nn.ReLU(),
            nn.Linear(self.feature_dim // 2, self.feature_dim)
        )
    
    def contrastive_loss(self,
                        features_a: torch.Tensor,
                        features_b: torch.Tensor,
                        modality_a: ModalityType,
                        modality_b: ModalityType) -> torch.Tensor:
        """Compute contrastive loss between two modalities."""
        # Project features
        proj_a = self.projection_heads[modality_a.value](features_a)
        proj_b = self.projection_heads[modality_b.value](features_b)
        
        # Normalize
        proj_a = F.normalize(proj_a, dim=-1)
        proj_b = F.normalize(proj_b, dim=-1)
        
        # Compute similarity matrix
        similarity_matrix = torch.matmul(proj_a, proj_b.T) / self.temperature
        
        # Labels (positive pairs are on diagonal)
        batch_size = proj_a.size(0)
        labels = torch.arange(batch_size).to(proj_a.device)
        
        # Cross-entropy loss
        loss_a = F.cross_entropy(similarity_matrix, labels)
        loss_b = F.cross_entropy(similarity_matrix.T, labels)
        
        return (loss_a + loss_b) / 2
    
    def cross_modal_prediction_loss(self,
                                   source_features: torch.Tensor,
                                   target_features: torch.Tensor,
                                   source_modality: ModalityType,
                                   target_modality: ModalityType) -> torch.Tensor:
        """Compute cross-modal prediction loss."""
        predictor_name = f"{source_modality.value}_to_{target_modality.value}"
        
        if predictor_name not in self.cross_modal_predictors:
            return torch.tensor(0.0, device=source_features.device)
        
        predicted_features = self.cross_modal_predictors[predictor_name](source_features)
        loss = F.mse_loss(predicted_features, target_features)
        
        return loss
    
    def reconstruction_loss(self,
                           features: torch.Tensor,
                           original_data: torch.Tensor,
                           modality: ModalityType) -> torch.Tensor:
        """Compute reconstruction loss."""
        reconstructed = self.reconstruction_heads[modality.value](features)
        loss = F.mse_loss(reconstructed, original_data)
        
        return loss


class DynamicModalityWeighting(nn.Module):
    """Dynamic weighting of modalities based on reliability and task relevance."""
    
    def __init__(self,
                 num_modalities: int,
                 feature_dim: int = 512,
                 context_dim: int = 64):
        super().__init__()
        self.num_modalities = num_modalities
        self.feature_dim = feature_dim
        self.context_dim = context_dim
        
        # Context encoder for task/environment information
        self.context_encoder = nn.Sequential(
            nn.Linear(context_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU()
        )
        
        # Modality reliability predictor
        self.reliability_predictor = nn.Sequential(
            nn.Linear(feature_dim + 64, 128),  # feature + context
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
        # Task relevance predictor
        self.relevance_predictor = nn.Sequential(
            nn.Linear(feature_dim + 64, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
        # Weight fusion network
        self.weight_fusion = nn.Sequential(
            nn.Linear(2, 16),  # reliability + relevance
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )
    
    def forward(self,
                modality_features: List[torch.Tensor],
                uncertainties: List[torch.Tensor],
                context: torch.Tensor) -> Tuple[torch.Tensor, List[float]]:
        """
        Compute dynamic weights for modalities.
        
        Args:
            modality_features: List of modality features [batch_size, feature_dim]
            uncertainties: List of uncertainty estimates [batch_size, 1]
            context: Task/environment context [batch_size, context_dim]
            
        Returns:
            weights: Modality weights [batch_size, num_modalities]
            weight_explanations: Per-modality weight explanations
        """
        batch_size = modality_features[0].size(0)
        
        # Encode context
        encoded_context = self.context_encoder(context)
        
        weights = []
        weight_explanations = []
        
        for i, (features, uncertainty) in enumerate(zip(modality_features, uncertainties)):
            # Combine features with context
            combined_input = torch.cat([features, encoded_context], dim=-1)
            
            # Predict reliability and relevance
            reliability = self.reliability_predictor(combined_input)
            relevance = self.relevance_predictor(combined_input)
            
            # Combine reliability and relevance
            weight_input = torch.cat([reliability, relevance], dim=-1)
            weight = self.weight_fusion(weight_input)
            
            # Adjust for uncertainty (lower uncertainty = higher weight)
            uncertainty_adjustment = 1.0 - uncertainty
            final_weight = weight * uncertainty_adjustment
            
            weights.append(final_weight)
            
            weight_explanations.append({
                'reliability': float(reliability.mean()),
                'relevance': float(relevance.mean()),
                'uncertainty_adjustment': float(uncertainty_adjustment.mean()),
                'final_weight': float(final_weight.mean())
            })
        
        # Normalize weights
        weights_tensor = torch.cat(weights, dim=-1)
        normalized_weights = F.softmax(weights_tensor, dim=-1)
        
        return normalized_weights, weight_explanations


class HierarchicalMultimodalReasoner(nn.Module):
    """Hierarchical reasoning with symbolic grounding."""
    
    def __init__(self,
                 feature_dim: int = 512,
                 reasoning_layers: int = 3,
                 symbolic_vocab_size: int = 1000):
        super().__init__()
        self.feature_dim = feature_dim
        self.reasoning_layers = reasoning_layers
        self.symbolic_vocab_size = symbolic_vocab_size
        
        # Hierarchical reasoning layers
        self.reasoning_layers_net = nn.ModuleList([
            nn.Sequential(
                nn.Linear(feature_dim, feature_dim),
                nn.ReLU(),
                nn.Linear(feature_dim, feature_dim),
                nn.LayerNorm(feature_dim)
            ) for _ in range(reasoning_layers)
        ])
        
        # Symbolic grounding network
        self.symbol_grounding = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(),
            nn.Linear(256, symbolic_vocab_size),
            nn.Softmax(dim=-1)
        )
        
        # Reasoning trace generator
        self.trace_generator = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 100)  # Simplified trace representation
        )
        
        # Symbol vocabulary (would be learned or predefined)
        self.symbol_vocabulary = [f"concept_{i}" for i in range(symbolic_vocab_size)]
    
    def forward(self,
                fused_features: torch.Tensor,
                reasoning_context: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, List[str], torch.Tensor]:
        """
        Perform hierarchical multimodal reasoning.
        
        Args:
            fused_features: Fused multimodal features [batch_size, feature_dim]
            reasoning_context: Optional reasoning context
            
        Returns:
            reasoned_features: Output after hierarchical reasoning
            reasoning_trace: Human-readable reasoning trace
            symbolic_grounding: Symbolic concept activations
        """
        current_features = fused_features
        reasoning_trace = []
        
        # Hierarchical reasoning
        for i, layer in enumerate(self.reasoning_layers_net):
            # Apply reasoning layer
            layer_output = layer(current_features)
            
            # Residual connection
            current_features = current_features + layer_output
            
            # Generate trace for this layer
            trace_vector = self.trace_generator(current_features)
            
            # Convert to reasoning step (simplified)
            reasoning_trace.append(f"Layer {i+1}: Processing multimodal integration")
        
        # Symbolic grounding
        symbolic_activations = self.symbol_grounding(current_features)
        
        # Extract top symbolic concepts
        top_k = 5
        top_symbols = torch.topk(symbolic_activations, top_k, dim=-1)
        
        for batch_idx in range(symbolic_activations.size(0)):
            batch_symbols = []
            for symbol_idx, activation in zip(top_symbols.indices[batch_idx], top_symbols.values[batch_idx]):
                symbol_name = self.symbol_vocabulary[symbol_idx.item()]
                batch_symbols.append(f"{symbol_name}({activation:.3f})")
            
            reasoning_trace.append(f"Symbolic grounding: {', '.join(batch_symbols)}")
        
        return current_features, reasoning_trace, symbolic_activations


class MultiModalSensoryFusion:
    """Main multimodal sensory fusion system."""
    
    def __init__(self,
                 modality_configs: Dict[ModalityType, int],
                 feature_dim: int = 512,
                 use_self_supervised: bool = True,
                 use_dynamic_weighting: bool = True,
                 use_hierarchical_reasoning: bool = True,
                 device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        """
        Initialize multimodal sensory fusion system.
        
        Args:
            modality_configs: Dictionary mapping modality types to input dimensions
            feature_dim: Common feature dimension for all modalities
            use_self_supervised: Whether to use self-supervised cross-modal learning
            use_dynamic_weighting: Whether to use dynamic modality weighting
            use_hierarchical_reasoning: Whether to use hierarchical reasoning
            device: Computation device
        """
        self.modality_configs = modality_configs
        self.feature_dim = feature_dim
        self.use_self_supervised = use_self_supervised
        self.use_dynamic_weighting = use_dynamic_weighting
        self.use_hierarchical_reasoning = use_hierarchical_reasoning
        self.device = device
        
        # Initialize modality encoders
        self.encoders = nn.ModuleDict()
        for modality, input_dim in modality_configs.items():
            self.encoders[modality.value] = ModalityEncoder(
                modality, input_dim, feature_dim
            ).to(device)
        
        # Cross-modal attention mechanisms
        self.cross_attention_modules = nn.ModuleDict()
        modality_list = list(modality_configs.keys())
        for i, modality_a in enumerate(modality_list):
            for modality_b in modality_list[i+1:]:
                key = f"{modality_a.value}_{modality_b.value}"
                self.cross_attention_modules[key] = CrossModalAttention(feature_dim).to(device)
        
        # Self-supervised learning
        if use_self_supervised:
            self.ssl_learner = SelfSupervisedCrossModalLearner(feature_dim).to(device)
            for modality, input_dim in modality_configs.items():
                self.ssl_learner.add_modality(modality, input_dim)
            
            # Add cross-modal predictors for all pairs
            for i, modality_a in enumerate(modality_list):
                for modality_b in modality_list:
                    if modality_a != modality_b:
                        self.ssl_learner.add_cross_modal_predictor(modality_a, modality_b)
        
        # Dynamic modality weighting
        if use_dynamic_weighting:
            self.dynamic_weighting = DynamicModalityWeighting(
                len(modality_configs), feature_dim
            ).to(device)
        
        # Hierarchical reasoning
        if use_hierarchical_reasoning:
            self.hierarchical_reasoner = HierarchicalMultimodalReasoner(feature_dim).to(device)
        
        # Fusion network
        self.fusion_network = nn.Sequential(
            nn.Linear(feature_dim * len(modality_configs), feature_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(feature_dim * 2, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, feature_dim)
        ).to(device)
        
        # Final output projections
        self.output_projections = nn.ModuleDict({
            'action': nn.Linear(feature_dim, 256),
            'state_estimation': nn.Linear(feature_dim, 128),
            'object_detection': nn.Linear(feature_dim, 512),
            'scene_understanding': nn.Linear(feature_dim, 256)
        }).to(device)
        
        # Performance tracking
        self.fusion_history = deque(maxlen=1000)
        self.modality_performance = defaultdict(list)
        self.cross_modal_alignments = defaultdict(list)
        
    def fuse_modalities(self,
                       modality_data: Dict[ModalityType, ModalityData],
                       task_context: Optional[torch.Tensor] = None,
                       return_detailed_analysis: bool = False) -> FusionResult:
        """
        Fuse multiple sensory modalities.
        
        Args:
            modality_data: Dictionary of modality data
            task_context: Optional task context for dynamic weighting
            return_detailed_analysis: Whether to return detailed analysis
            
        Returns:
            FusionResult with fused representation and analysis
        """
        start_time = time.time()
        
        # Encode each modality
        encoded_features = {}
        uncertainties = {}
        modality_contributions = {}
        
        for modality_type, data in modality_data.items():
            if modality_type.value in self.encoders:
                encoder = self.encoders[modality_type.value]
                features, uncertainty = encoder(data.data.to(self.device))
                
                encoded_features[modality_type] = features
                uncertainties[modality_type] = uncertainty
                
                # Track modality performance
                self.modality_performance[modality_type].append({
                    'timestamp': data.timestamp,
                    'confidence': data.confidence,
                    'uncertainty': float(uncertainty.mean()),
                    'feature_norm': float(torch.norm(features).item())
                })
        
        if not encoded_features:
            return FusionResult(
                fused_representation=torch.zeros(1, self.feature_dim).to(self.device),
                modality_contributions={},
                uncertainty_estimate=1.0,
                cross_modal_alignments={},
                reasoning_trace=[\"No modalities available for fusion\"],
                confidence_score=0.0
            )
        
        # Cross-modal attention and alignment
        cross_modal_alignments = {}
        attended_features = {}
        
        modality_list = list(encoded_features.keys())
        for i, modality_a in enumerate(modality_list):
            attended_features[modality_a] = encoded_features[modality_a]
            
            for modality_b in modality_list[i+1:]:
                key = f"{modality_a.value}_{modality_b.value}"
                if key in self.cross_attention_modules:
                    attention_module = self.cross_attention_modules[key]
                    
                    # Cross-modal attention
                    attended_a, attention_weights, alignment = attention_module(
                        encoded_features[modality_a],
                        encoded_features[modality_b],
                        encoded_features[modality_b],
                        uncertainties[modality_b]
                    )
                    
                    attended_features[modality_a] = attended_a
                    cross_modal_alignments[(modality_a, modality_b)] = float(alignment.mean())
                    
                    # Track alignment history
                    self.cross_modal_alignments[(modality_a, modality_b)].append(float(alignment.mean()))
        
        # Dynamic modality weighting
        if self.use_dynamic_weighting and task_context is not None:
            feature_list = [attended_features[mod] for mod in modality_list]
            uncertainty_list = [uncertainties[mod] for mod in modality_list]
            
            modality_weights, weight_explanations = self.dynamic_weighting(
                feature_list, uncertainty_list, task_context
            )
            
            # Apply weights
            weighted_features = []
            for i, modality in enumerate(modality_list):
                weight = modality_weights[0, i].unsqueeze(0).unsqueeze(1)
                weighted_feature = attended_features[modality] * weight
                weighted_features.append(weighted_feature)
                
                modality_contributions[modality] = float(modality_weights[0, i])
        else:
            weighted_features = list(attended_features.values())
            equal_weight = 1.0 / len(modality_list)
            modality_contributions = {mod: equal_weight for mod in modality_list}
        
        # Concatenate and fuse
        concatenated_features = torch.cat(weighted_features, dim=-1)
        fused_features = self.fusion_network(concatenated_features)
        
        # Hierarchical reasoning
        reasoning_trace = []
        if self.use_hierarchical_reasoning:
            reasoned_features, reasoning_steps, symbolic_grounding = self.hierarchical_reasoner(
                fused_features, task_context
            )
            fused_features = reasoned_features
            reasoning_trace.extend(reasoning_steps)
        
        # Estimate overall uncertainty
        uncertainty_values = [float(unc.mean()) for unc in uncertainties.values()]
        overall_uncertainty = np.mean(uncertainty_values) if uncertainty_values else 1.0
        
        # Compute confidence score
        confidence_factors = [
            1.0 - overall_uncertainty,  # Lower uncertainty = higher confidence
            np.mean(list(cross_modal_alignments.values())) if cross_modal_alignments else 0.5,
            len(encoded_features) / len(self.modality_configs)  # Modality coverage
        ]
        confidence_score = np.mean(confidence_factors)
        
        # Record fusion event
        fusion_time = time.time() - start_time
        self.fusion_history.append({
            'timestamp': time.time(),
            'modalities_used': [mod.value for mod in modality_list],
            'confidence_score': confidence_score,
            'uncertainty_estimate': overall_uncertainty,
            'fusion_time': fusion_time,
            'cross_modal_alignments': cross_modal_alignments
        })
        
        result = FusionResult(
            fused_representation=fused_features,
            modality_contributions=modality_contributions,
            uncertainty_estimate=overall_uncertainty,
            cross_modal_alignments=cross_modal_alignments,
            reasoning_trace=reasoning_trace,
            confidence_score=confidence_score
        )
        
        return result
    
    def train_self_supervised(self,
                             multimodal_batch: List[Dict[ModalityType, torch.Tensor]],
                             num_epochs: int = 10,
                             learning_rate: float = 0.001) -> Dict[str, float]:
        """Train self-supervised cross-modal learning."""
        if not self.use_self_supervised:
            logger.warning("Self-supervised learning not enabled")
            return {}
        
        optimizer = torch.optim.Adam(
            list(self.ssl_learner.parameters()) + 
            list(self.encoders.parameters()),
            lr=learning_rate
        )
        
        total_losses = defaultdict(list)
        
        for epoch in range(num_epochs):
            epoch_losses = defaultdict(float)
            
            for batch_data in multimodal_batch:
                optimizer.zero_grad()
                
                # Encode all modalities in batch
                encoded_batch = {}
                for modality, data in batch_data.items():
                    if modality.value in self.encoders:
                        features, _ = self.encoders[modality.value](data.to(self.device))
                        encoded_batch[modality] = features
                
                total_loss = 0.0
                
                # Contrastive loss between modality pairs
                modality_pairs = list(encoded_batch.keys())
                for i, mod_a in enumerate(modality_pairs):
                    for mod_b in modality_pairs[i+1:]:
                        contrastive_loss = self.ssl_learner.contrastive_loss(
                            encoded_batch[mod_a], encoded_batch[mod_b], mod_a, mod_b
                        )
                        total_loss += contrastive_loss
                        epoch_losses['contrastive'] += contrastive_loss.item()
                
                # Cross-modal prediction loss
                for source_mod in modality_pairs:
                    for target_mod in modality_pairs:
                        if source_mod != target_mod:
                            pred_loss = self.ssl_learner.cross_modal_prediction_loss(
                                encoded_batch[source_mod], encoded_batch[target_mod],
                                source_mod, target_mod
                            )
                            total_loss += pred_loss * 0.5  # Weight cross-modal prediction less
                            epoch_losses['cross_modal_prediction'] += pred_loss.item()
                
                # Reconstruction loss
                for modality, original_data in batch_data.items():
                    if modality in encoded_batch:
                        recon_loss = self.ssl_learner.reconstruction_loss(
                            encoded_batch[modality], original_data.to(self.device), modality
                        )
                        total_loss += recon_loss * 0.3  # Weight reconstruction less
                        epoch_losses['reconstruction'] += recon_loss.item()
                
                total_loss.backward()
                optimizer.step()
                
                epoch_losses['total'] += total_loss.item()
            
            # Average losses over batch
            for loss_type, loss_value in epoch_losses.items():
                epoch_losses[loss_type] = loss_value / len(multimodal_batch)
                total_losses[loss_type].append(epoch_losses[loss_type])
            
            if epoch % 2 == 0:
                logger.info(f"SSL Epoch {epoch}: total_loss={epoch_losses['total']:.4f}")
        
        # Return final average losses
        return {loss_type: np.mean(losses) for loss_type, losses in total_losses.items()}
    
    def get_modality_insights(self) -> Dict[str, Any]:
        """Get insights about modality performance and interactions."""
        insights = {
            'modality_statistics': {},
            'cross_modal_alignments': {},
            'fusion_performance': {},
            'reliability_analysis': {}
        }
        
        # Modality statistics
        for modality, performance_list in self.modality_performance.items():
            if performance_list:
                recent_performance = performance_list[-100:]  # Last 100 measurements
                insights['modality_statistics'][modality.value] = {
                    'avg_confidence': np.mean([p['confidence'] for p in recent_performance]),
                    'avg_uncertainty': np.mean([p['uncertainty'] for p in recent_performance]),
                    'feature_stability': 1.0 / (1.0 + np.std([p['feature_norm'] for p in recent_performance])),
                    'measurement_count': len(performance_list)
                }
        
        # Cross-modal alignment analysis
        for modality_pair, alignment_history in self.cross_modal_alignments.items():
            if alignment_history:
                recent_alignments = alignment_history[-50:]
                insights['cross_modal_alignments'][f"{modality_pair[0].value}_{modality_pair[1].value}"] = {
                    'avg_alignment': np.mean(recent_alignments),
                    'alignment_stability': 1.0 / (1.0 + np.std(recent_alignments)),
                    'alignment_trend': np.polyfit(range(len(recent_alignments)), recent_alignments, 1)[0]
                }
        
        # Fusion performance
        if self.fusion_history:
            recent_fusions = list(self.fusion_history)[-100:]
            insights['fusion_performance'] = {
                'avg_confidence_score': np.mean([f['confidence_score'] for f in recent_fusions]),
                'avg_uncertainty': np.mean([f['uncertainty_estimate'] for f in recent_fusions]),
                'avg_fusion_time': np.mean([f['fusion_time'] for f in recent_fusions]),
                'modality_usage_frequency': dict(Counter([
                    mod for fusion in recent_fusions for mod in fusion['modalities_used']
                ]))
            }
        
        return insights
    
    def adapt_to_modality_failure(self, failed_modality: ModalityType) -> Dict[str, Any]:
        """Adapt fusion when a modality fails or becomes unreliable."""
        logger.warning(f"Adapting to failure of modality: {failed_modality.value}")
        
        adaptation_strategies = {
            'compensation_strategies': [],
            'weight_adjustments': {},
            'alternative_modalities': []
        }
        
        # Identify compensatory modalities
        if failed_modality == ModalityType.VISION_RGB:
            compensation_strategies.append("Increase reliance on depth and tactile sensing")
            adaptation_strategies['alternative_modalities'] = [
                ModalityType.VISION_DEPTH, ModalityType.TACTILE, ModalityType.LIDAR
            ]
        elif failed_modality == ModalityType.TACTILE:
            compensation_strategies.append("Increase reliance on vision and force sensing")
            adaptation_strategies['alternative_modalities'] = [
                ModalityType.VISION_RGB, ModalityType.FORCE_TORQUE, ModalityType.AUDIO
            ]
        elif failed_modality == ModalityType.AUDIO:
            compensation_strategies.append("Increase reliance on visual and tactile cues")
            adaptation_strategies['alternative_modalities'] = [
                ModalityType.VISION_RGB, ModalityType.TACTILE, ModalityType.PROPRIOCEPTION
            ]
        
        # Adjust modality weights temporarily
        for alt_modality in adaptation_strategies['alternative_modalities']:
            if alt_modality in self.modality_configs:
                adaptation_strategies['weight_adjustments'][alt_modality.value] = 1.5  # Increase weight
        
        return adaptation_strategies
    
    def save_fusion_system(self, filepath: str):
        """Save multimodal fusion system state."""
        state = {
            'modality_configs': {mod.value: dim for mod, dim in self.modality_configs.items()},
            'encoders': {name: encoder.state_dict() for name, encoder in self.encoders.items()},
            'cross_attention_modules': {name: module.state_dict() for name, module in self.cross_attention_modules.items()},
            'fusion_network': self.fusion_network.state_dict(),
            'output_projections': {name: proj.state_dict() for name, proj in self.output_projections.items()},
            'config': {
                'feature_dim': self.feature_dim,
                'use_self_supervised': self.use_self_supervised,
                'use_dynamic_weighting': self.use_dynamic_weighting,
                'use_hierarchical_reasoning': self.use_hierarchical_reasoning
            },
            'performance_history': {
                'fusion_history': list(self.fusion_history),
                'modality_performance': dict(self.modality_performance),
                'cross_modal_alignments': dict(self.cross_modal_alignments)
            }
        }
        
        if self.use_self_supervised:
            state['ssl_learner'] = self.ssl_learner.state_dict()
        
        if self.use_dynamic_weighting:
            state['dynamic_weighting'] = self.dynamic_weighting.state_dict()
        
        if self.use_hierarchical_reasoning:
            state['hierarchical_reasoner'] = self.hierarchical_reasoner.state_dict()
        
        torch.save(state, filepath)
        logger.info(f"Saved multimodal fusion system to {filepath}")
    
    def load_fusion_system(self, filepath: str):
        """Load multimodal fusion system state."""
        state = torch.load(filepath, map_location=self.device)
        
        # Load encoders
        for name, state_dict in state['encoders'].items():
            if name in self.encoders:
                self.encoders[name].load_state_dict(state_dict)
        
        # Load other components
        if 'cross_attention_modules' in state:
            for name, state_dict in state['cross_attention_modules'].items():
                if name in self.cross_attention_modules:
                    self.cross_attention_modules[name].load_state_dict(state_dict)
        
        self.fusion_network.load_state_dict(state['fusion_network'])
        
        for name, state_dict in state['output_projections'].items():
            if name in self.output_projections:
                self.output_projections[name].load_state_dict(state_dict)
        
        if 'ssl_learner' in state and self.use_self_supervised:
            self.ssl_learner.load_state_dict(state['ssl_learner'])
        
        if 'dynamic_weighting' in state and self.use_dynamic_weighting:
            self.dynamic_weighting.load_state_dict(state['dynamic_weighting'])
        
        if 'hierarchical_reasoner' in state and self.use_hierarchical_reasoning:
            self.hierarchical_reasoner.load_state_dict(state['hierarchical_reasoner'])
        
        # Load performance history
        if 'performance_history' in state:
            self.fusion_history = deque(state['performance_history']['fusion_history'], maxlen=1000)
            self.modality_performance = defaultdict(list, state['performance_history']['modality_performance'])
            self.cross_modal_alignments = defaultdict(list, state['performance_history']['cross_modal_alignments'])
        
        logger.info(f"Loaded multimodal fusion system from {filepath}")