"""
Early stopping and best checkpoint tracking for training stages.
"""

import numpy as np
import torch
from pathlib import Path
from typing import Optional, Literal
import logging

logger = logging.getLogger(__name__)


class EarlyStopping:
    """
    Early stopping to halt training when validation metric stops improving.
    
    Args:
        patience: Number of epochs to wait for improvement before stopping
        mode: 'min' for metrics to minimize (loss), 'max' for metrics to maximize (accuracy)
        min_delta: Minimum change to qualify as improvement
        verbose: Whether to print messages
    """
    
    def __init__(
        self,
        patience: int = 3,
        mode: Literal['min', 'max'] = 'max',
        min_delta: float = 0.0,
        verbose: bool = True
    ):
        self.patience = patience
        self.mode = mode
        self.min_delta = min_delta
        self.verbose = verbose
        
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_epoch = 0
        
        # For tracking improvement
        self.score_history = []
        
        if mode == 'min':
            self.monitor_op = np.less
            self.best_score = np.inf
            self.delta = -min_delta
        elif mode == 'max':
            self.monitor_op = np.greater
            self.best_score = -np.inf
            self.delta = min_delta
        else:
            raise ValueError(f"mode must be 'min' or 'max', got {mode}")
    
    def __call__(self, score: float, epoch: int) -> bool:
        """
        Check if training should stop.
        
        Args:
            score: Current validation metric value
            epoch: Current epoch number
            
        Returns:
            True if model improved (should save checkpoint), False otherwise
        """
        self.score_history.append(score)
        
        # Check if this is an improvement
        if self.monitor_op(score - self.delta, self.best_score):
            if self.verbose:
                improvement = abs(score - self.best_score)
                logger.info(f"✓ Validation metric improved from {self.best_score:.4f} to {score:.4f} (+{improvement:.4f})")
            
            self.best_score = score
            self.best_epoch = epoch
            self.counter = 0
            return True  # Improvement - should save checkpoint
        else:
            self.counter += 1
            if self.verbose:
                logger.info(f"  No improvement for {self.counter}/{self.patience} epochs (best: {self.best_score:.4f} at epoch {self.best_epoch})")
            
            if self.counter >= self.patience:
                self.early_stop = True
                if self.verbose:
                    logger.info(f"⚠️  Early stopping triggered! No improvement for {self.patience} epochs.")
                    logger.info(f"   Best score: {self.best_score:.4f} at epoch {self.best_epoch}")
            
            return False  # No improvement
    
    def get_summary(self) -> dict:
        """Get summary of training progress."""
        return {
            'best_score': self.best_score,
            'best_epoch': self.best_epoch,
            'total_epochs': len(self.score_history),
            'stopped_early': self.early_stop,
            'score_history': self.score_history,
        }


class BestCheckpointTracker:
    """
    Track and save the best model checkpoint based on validation metrics.
    """
    
    def __init__(
        self,
        save_dir: str,
        metric_name: str,
        mode: Literal['min', 'max'] = 'max',
    ):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.metric_name = metric_name
        self.mode = mode
        
        if mode == 'min':
            self.best_score = np.inf
            self.is_better = lambda new, best: new < best
        else:
            self.best_score = -np.inf
            self.is_better = lambda new, best: new > best
        
        self.best_checkpoint_path = None
    
    def update(self, score: float, checkpoint_path: str) -> bool:
        """
        Update best checkpoint if score improved.
        
        Args:
            score: Current validation metric value
            checkpoint_path: Path to checkpoint to save
            
        Returns:
            True if this is the new best checkpoint
        """
        if self.is_better(score, self.best_score):
            self.best_score = score
            self.best_checkpoint_path = checkpoint_path
            
            # Create symlink to best checkpoint
            best_link = self.save_dir / 'best_checkpoint'
            if best_link.exists() or best_link.is_symlink():
                best_link.unlink()
            
            try:
                # For Windows, use a marker file instead of symlink
                with open(self.save_dir / 'best_checkpoint_path.txt', 'w') as f:
                    f.write(str(checkpoint_path))
                logger.info(f"💾 New best checkpoint: {checkpoint_path} ({self.metric_name}={score:.4f})")
            except Exception as e:
                logger.warning(f"Could not create best checkpoint marker: {e}")
            
            return True
        return False
    
    def load_best_checkpoint(self):
        """Return path to best checkpoint."""
        marker_file = self.save_dir / 'best_checkpoint_path.txt'
        if marker_file.exists():
            with open(marker_file, 'r') as f:
                return f.read().strip()
        return self.best_checkpoint_path


def compute_stage_metrics_summary(stage_name: str, metrics_history: list) -> dict:
    """
    Compute summary statistics for a training stage.
    
    Args:
        stage_name: Name of the stage (e.g., "Stage 1")
        metrics_history: List of metric dictionaries from each epoch
        
    Returns:
        Summary dictionary with improvement statistics
    """
    if not metrics_history:
        return {}
    
    summary = {
        'stage': stage_name,
        'total_epochs': len(metrics_history),
    }
    
    # Extract key metrics
    for key in metrics_history[0].keys():
        values = [m.get(key, np.nan) for m in metrics_history]
        if all(isinstance(v, (int, float)) for v in values):
            summary[f'{key}_initial'] = values[0]
            summary[f'{key}_final'] = values[-1]
            summary[f'{key}_best'] = max(values) if 'acc' in key or 'f1' in key else min(values)
            summary[f'{key}_improvement'] = values[-1] - values[0]
            
            # Percent improvement
            if values[0] != 0:
                summary[f'{key}_improvement_pct'] = ((values[-1] - values[0]) / abs(values[0])) * 100
    
    return summary


def log_stage_summary(stage_name: str, metrics_history: list):
    """
    Log a formatted summary of stage training progress.
    """
    summary = compute_stage_metrics_summary(stage_name, metrics_history)
    
    logger.info(f"\n{'='*60}")
    logger.info(f"  {stage_name} Training Summary")
    logger.info(f"{'='*60}")
    logger.info(f"Total Epochs: {summary.get('total_epochs', 0)}")
    logger.info("")
    
    # Log key improvements
    for key in sorted(summary.keys()):
        if key.endswith('_improvement') and not key.endswith('_improvement_pct'):
            metric_name = key.replace('_improvement', '')
            initial = summary.get(f'{metric_name}_initial', 0)
            final = summary.get(f'{metric_name}_final', 0)
            improvement = summary.get(key, 0)
            improvement_pct = summary.get(f'{key}_pct', 0)
            
            arrow = "↑" if improvement > 0 else "↓" if improvement < 0 else "→"
            logger.info(f"  {metric_name:30s}: {initial:8.4f} → {final:8.4f}  {arrow} {improvement:+.4f} ({improvement_pct:+.1f}%)")
    
    logger.info(f"{'='*60}\n")


def save_stage_metrics_json(
    stage_name: str,
    output_dir: str,
    metrics_history: list,
    step_metrics: Optional[dict] = None,
):
    """
    Save stage training metrics to a JSON file for cross-stage collation.

    Args:
        stage_name: e.g. 'stage1', 'stage2', 'stage3', 'stage4'
        output_dir: Stage output directory (e.g. outputs/trial_run/.../stage1)
        metrics_history: List of per-epoch val_metrics dicts
        step_metrics: Optional dict of per-step metric lists, e.g.
                      {'loss': [0.5, 0.4, ...], 'lr': [1e-4, ...], 'acc_i2t': [...]}
    """
    import json

    out_path = Path(output_dir) / 'training_metrics.json'
    data = {
        'stage': stage_name,
        'epoch_metrics': [],
        'step_metrics': step_metrics or {},
        'summary': compute_stage_metrics_summary(stage_name, metrics_history),
    }

    # Sanitize epoch metrics (convert numpy/tensor to float)
    for m in metrics_history:
        cleaned = {}
        for k, v in m.items():
            if isinstance(v, (int, float)):
                cleaned[k] = float(v)
            elif hasattr(v, 'item'):
                cleaned[k] = float(v.item())
        data['epoch_metrics'].append(cleaned)

    try:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        logger.info(f"✓ Saved {stage_name} metrics to {out_path}")
    except Exception as e:
        logger.warning(f"Failed to save {stage_name} metrics: {e}")
