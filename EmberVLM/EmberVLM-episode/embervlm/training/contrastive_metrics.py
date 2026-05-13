"""
Enhanced metrics and visualization for Stage 1 contrastive learning.
Adds top-5 accuracy, similarity distribution analysis, and retrieval quality metrics.
"""

import torch
import torch.nn.functional as F
from typing import Dict, Tuple


def compute_enhanced_contrastive_metrics(
    image_features: torch.Tensor,
    text_features: torch.Tensor,
    temperature: float = 0.07
) -> Dict[str, float]:
    """
    Compute enhanced contrastive learning metrics including top-5 accuracy.
    
    Args:
        image_features: [B, D] normalized image features
        text_features: [B, D] normalized text features
        temperature: Temperature parameter for similarity scaling
        
    Returns:
        Dictionary with metrics: i2t_acc, t2i_acc, i2t_acc_top5, t2i_acc_top5,
                                 mean_similarity, std_similarity
    """
    batch_size = image_features.size(0)
    device = image_features.device
    
    # Compute similarity matrix [B, B]
    similarity = (image_features @ text_features.T) / temperature
    
    # Ground truth labels (diagonal)
    labels = torch.arange(batch_size, device=device)
    
    # === Image-to-Text Retrieval ===
    # Top-1 accuracy
    i2t_preds = similarity.argmax(dim=1)
    i2t_acc = (i2t_preds == labels).float().mean().item()
    
    # Top-5 accuracy
    i2t_top5 = similarity.topk(min(5, batch_size), dim=1)[1]
    i2t_acc_top5 = (i2t_top5 == labels.unsqueeze(1)).any(dim=1).float().mean().item()
    
    # Mean Reciprocal Rank (MRR)
    i2t_ranks = (similarity.argsort(dim=1, descending=True) == labels.unsqueeze(1)).nonzero()[:, 1] + 1
    i2t_mrr = (1.0 / i2t_ranks.float()).mean().item()
    
    # === Text-to-Image Retrieval ===
    similarity_t = similarity.T  # [B, B]
    
    # Top-1 accuracy
    t2i_preds = similarity_t.argmax(dim=1)
    t2i_acc = (t2i_preds == labels).float().mean().item()
    
    # Top-5 accuracy
    t2i_top5 = similarity_t.topk(min(5, batch_size), dim=1)[1]
    t2i_acc_top5 = (t2i_top5 == labels.unsqueeze(1)).any(dim=1).float().mean().item()
    
    # Mean Reciprocal Rank
    t2i_ranks = (similarity_t.argsort(dim=1, descending=True) == labels.unsqueeze(1)).nonzero()[:, 1] + 1
    t2i_mrr = (1.0 / t2i_ranks.float()).mean().item()
    
    # === Similarity Distribution Analysis ===
    # Diagonal (positive pairs)
    positive_sims = torch.diagonal(similarity)
    mean_pos_sim = positive_sims.mean().item()
    
    # Off-diagonal (negative pairs)
    mask = ~torch.eye(batch_size, dtype=torch.bool, device=device)
    negative_sims = similarity[mask]
    mean_neg_sim = negative_sims.mean().item()
    
    # Overall statistics
    mean_similarity = similarity.mean().item()
    std_similarity = similarity.std().item()
    
    # Separation: how well separated are positive and negative pairs
    separation = mean_pos_sim - mean_neg_sim
    
    return {
        # Top-1 accuracy
        'i2t_acc': i2t_acc,
        't2i_acc': t2i_acc,
        # Top-5 accuracy
        'i2t_acc_top5': i2t_acc_top5,
        't2i_acc_top5': t2i_acc_top5,
        # Mean Reciprocal Rank
        'i2t_mrr': i2t_mrr,
        't2i_mrr': t2i_mrr,
        # Similarity distribution
        'mean_similarity': mean_similarity,
        'std_similarity': std_similarity,
        'mean_pos_similarity': mean_pos_sim,
        'mean_neg_similarity': mean_neg_sim,
        'similarity_separation': separation,
    }


def log_contrastive_visualization_data(
    similarity_matrix: torch.Tensor,
    step: int,
    prefix: str = "train"
) -> Dict[str, any]:
    """
    Prepare data for visualization (histograms, heatmaps).
    
    Args:
        similarity_matrix: [B, B] similarity matrix
        step: Current training step
        prefix: "train" or "val"
        
    Returns:
        Dictionary with visualization data for WandB
    """
    batch_size = similarity_matrix.size(0)
    device = similarity_matrix.device
    
    # Extract positive and negative pairs
    mask = torch.eye(batch_size, dtype=torch.bool, device=device)
    positive_sims = similarity_matrix[mask].cpu().numpy()
    negative_sims = similarity_matrix[~mask].cpu().numpy()
    
    viz_data = {
        f'{prefix}/positive_similarities': positive_sims.tolist(),
        f'{prefix}/negative_similarities': negative_sims.tolist(),
        f'{prefix}/similarity_matrix': similarity_matrix.detach().cpu().numpy(),
    }
    
    return viz_data
