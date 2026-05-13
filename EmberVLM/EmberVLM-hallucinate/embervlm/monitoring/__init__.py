"""
EmberVLM Monitoring Package

Provides comprehensive monitoring and visualization tools for training:
- W&B logging with advanced visualizations
- Carbon footprint tracking
- FLOPs counting
- Attention visualization
- Stage-specific visualizations
- Advanced 3D plots and publication-quality figures
- Shared paper-quality style utilities
- Paper-ready figure generators
- Efficiency / quantization figures
"""

from embervlm.monitoring.wandb_logger import WandbLogger, EnhancedWandbLogger
from embervlm.monitoring.carbon_tracker import CarbonTracker
from embervlm.monitoring.flops_counter import FLOPsCounter, count_model_flops
from embervlm.monitoring.attention_viz import AttentionVisualizer, analyze_model_attention
from embervlm.monitoring.visualization import TrainingVisualizer
from embervlm.monitoring.stage_visualizations import (
    Stage1Visualizer,
    Stage2Visualizer,
    Stage3Visualizer,
    Stage4Visualizer,
    CrossStageVisualizer,
    HallucinationVisualizer,
    CollatedVisualizer,
)
from embervlm.monitoring.advanced_visualizations import AdvancedVisualizer

# ── new modules ───────────────────────────────────────────────────────
from embervlm.monitoring.style import (
    set_paper_style,
    save_figure,
    fig_to_pil,
    tight_layout_with_padding,
    create_figure,
    robot_color,
    COLORS,
    PALETTE,
    ROBOT_COLORS,
    STAGE_COLORS,
    STAGE_LABELS,
)
from embervlm.monitoring.paper_figures import (
    plot_parameter_efficiency,
    plot_accuracy_efficiency_pareto,
    plot_multistage_training_overview,
    plot_embedding_alignment,
    plot_robot_precision_recall,
    plot_calibration_curve,
    plot_hallucination_histograms,
    plot_va_burst_timeline,
)
from embervlm.monitoring.efficiency_figures import (
    plot_quantization_impact,
    plot_device_latency_comparison,
)

__all__ = [
    # Loggers / trackers
    "WandbLogger",
    "EnhancedWandbLogger",
    "CarbonTracker",
    "FLOPsCounter",
    "count_model_flops",
    # Visualizers (class-based)
    "AttentionVisualizer",
    "analyze_model_attention",
    "TrainingVisualizer",
    "Stage1Visualizer",
    "Stage2Visualizer",
    "Stage3Visualizer",
    "Stage4Visualizer",
    "CrossStageVisualizer",
    "HallucinationVisualizer",
    "CollatedVisualizer",
    "AdvancedVisualizer",
    # Style utilities
    "set_paper_style",
    "save_figure",
    "fig_to_pil",
    "tight_layout_with_padding",
    "create_figure",
    "robot_color",
    "COLORS",
    "PALETTE",
    "ROBOT_COLORS",
    "STAGE_COLORS",
    "STAGE_LABELS",
    # Paper figure functions
    "plot_parameter_efficiency",
    "plot_accuracy_efficiency_pareto",
    "plot_multistage_training_overview",
    "plot_embedding_alignment",
    "plot_robot_precision_recall",
    "plot_calibration_curve",
    "plot_hallucination_histograms",
    "plot_va_burst_timeline",
    # Efficiency figures
    "plot_quantization_impact",
    "plot_device_latency_comparison",
]

