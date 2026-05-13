import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List

import numpy as np
import torch
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from embervlm.models import EmberVLM
from embervlm.models.vision_encoder import ImagePreprocessor

logger = logging.getLogger(__name__)


def generate_stage2_5_paper_figures(
    evaluation_summary_path: str,
    output_dir: str,
    openvlm_baselines_path: Optional[str] = None,
) -> Dict[str, str]:
    """Generate paper-ready figures from Stage 2.5 evaluation summary."""
    summary_path = Path(evaluation_summary_path)
    if not summary_path.exists():
        logger.warning(f"Stage 2.5 summary not found: {summary_path}")
        return {}

    with open(summary_path, 'r', encoding='utf-8') as f:
        summary = json.load(f)

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    figure_paths: Dict[str, str] = {}

    benches = summary.get('benchmarks', {})
    metric_map = {
        'Coherence': benches.get('coherence_overall', 0.0),
        'GQA': benches.get('gqa_vqa_accuracy', 0.0),
        'UniBench': benches.get('unibench_overall', 0.0),
        'SugarCrepe': benches.get('sugarcrepe_accuracy', 0.0),
        'Winoground': benches.get('winoground_accuracy', 0.0),
    }
    labels = [k for k, v in metric_map.items() if v is not None and v > 0]
    values = [metric_map[k] for k in labels]

    if labels:
        plt.figure(figsize=(10, 5))
        bars = plt.bar(labels, values)
        plt.ylim(0, 100)
        plt.ylabel('Accuracy (%)')
        plt.title('EmberVLM Stage 2.5 Evaluation Summary')
        plt.grid(axis='y', alpha=0.3)
        for bar, val in zip(bars, values):
            plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1, f"{val:.1f}%", ha='center', va='bottom')
        fig_path = out_dir / 'stage2_5_benchmark_bar.png'
        plt.tight_layout()
        plt.savefig(fig_path, dpi=200)
        plt.close()
        figure_paths['benchmark_bar'] = str(fig_path)

    categories = summary.get('coherence_results', {}).get('categories', {})
    if categories:
        cat_labels = list(categories.keys())
        cat_scores = []
        for key in cat_labels:
            passed = categories[key].get('passed', 0)
            total = max(categories[key].get('total', 1), 1)
            cat_scores.append((passed / total) * 100.0)

        heat = np.array(cat_scores, dtype=float).reshape(1, -1)
        plt.figure(figsize=(max(8, len(cat_labels) * 1.1), 2.6))
        im = plt.imshow(heat, cmap='YlGnBu', vmin=0, vmax=100, aspect='auto')
        plt.yticks([0], ['Score'])
        plt.xticks(np.arange(len(cat_labels)), cat_labels, rotation=30, ha='right')
        for i, s in enumerate(cat_scores):
            plt.text(i, 0, f"{s:.1f}", ha='center', va='center', color='black')
        plt.colorbar(im, label='Accuracy (%)')
        plt.title('Coherence Category Heatmap')
        fig_path = out_dir / 'stage2_5_category_heatmap.png'
        plt.tight_layout()
        plt.savefig(fig_path, dpi=220)
        plt.close()
        figure_paths['category_heatmap'] = str(fig_path)

    if openvlm_baselines_path:
        baseline_path = Path(openvlm_baselines_path)
        if baseline_path.exists():
            with open(baseline_path, 'r', encoding='utf-8') as f:
                baselines = json.load(f)

            comparison_metrics = ['gqa_vqa_accuracy', 'sugarcrepe_accuracy']
            nice_names = {'gqa_vqa_accuracy': 'GQA', 'sugarcrepe_accuracy': 'SugarCrepe'}
            available_metrics = [m for m in comparison_metrics if benches.get(m, 0.0) > 0]

            if available_metrics and isinstance(baselines, dict) and baselines:
                model_names = ['EmberVLM'] + list(baselines.keys())
                x = np.arange(len(available_metrics))
                width = 0.8 / len(model_names)

                plt.figure(figsize=(10, 5))
                ember_vals = [benches.get(m, 0.0) for m in available_metrics]
                plt.bar(x - 0.4 + width / 2, ember_vals, width, label='EmberVLM')

                for idx, (name, scores) in enumerate(baselines.items(), start=1):
                    vals = [float(scores.get(m, 0.0)) for m in available_metrics]
                    plt.bar(x - 0.4 + width / 2 + idx * width, vals, width, label=name)

                plt.xticks(x, [nice_names[m] for m in available_metrics])
                plt.ylabel('Accuracy (%)')
                plt.ylim(0, 100)
                plt.title('OpenVLM-style Comparison (Tiny VLM subset)')
                plt.legend(loc='best', fontsize=8)
                plt.grid(axis='y', alpha=0.3)
                fig_path = out_dir / 'stage2_5_openvlm_comparison.png'
                plt.tight_layout()
                plt.savefig(fig_path, dpi=220)
                plt.close()
                figure_paths['openvlm_comparison'] = str(fig_path)
        else:
            logger.warning(f"OpenVLM baseline file not found: {baseline_path}")

    return figure_paths


def generate_robot_topn_visualizations(
    model_path: str,
    image_dir: str,
    output_dir: str,
    top_n: int = 3,
    device: str = 'cuda',
) -> Dict[str, str]:
    """Generate robot top-N predictions and matrix heatmap from images."""
    model_path_p = Path(model_path)
    image_dir_p = Path(image_dir)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    images = sorted(list(image_dir_p.glob('*.jpg')) + list(image_dir_p.glob('*.png')) + list(image_dir_p.glob('*.jpeg')))
    images = [p for p in images if not p.name.startswith('sample-checks-')]
    if not images:
        logger.warning(f"No images found for robot top-N at: {image_dir_p}")
        return {}

    if device == 'cuda' and not torch.cuda.is_available():
        device = 'cpu'

    model = EmberVLM.from_pretrained(str(model_path_p)).to(device)
    model.eval()

    image_size = getattr(getattr(model, 'config', None), 'image_size', 224)
    preprocessor = ImagePreprocessor(image_size=image_size)

    rows: List[List[float]] = []
    image_names: List[str] = []
    details: List[Dict[str, Any]] = []

    with torch.no_grad():
        for path in images:
            pil = Image.open(path).convert('RGB')
            pixel = preprocessor(pil)
            if pixel.dim() == 3:
                pixel = pixel.unsqueeze(0)
            pixel = pixel.to(device=device, dtype=torch.float32)

            result = model.select_robots_topn(pixel_values=pixel, top_n=top_n, return_reasoning=True)
            all_scores = result.get('all_scores', {})
            robot_names = list(all_scores.keys())
            score_row = [float(all_scores[name]) for name in robot_names]
            rows.append(score_row)
            image_names.append(path.name)

            details.append({
                'image': path.name,
                'selected_robot': result.get('selected_robot'),
                'top_robots': result.get('top_robots', []),
                'reasoning_summary': result.get('reasoning_summary', ''),
                'all_scores': all_scores,
            })

    scores = np.array(rows, dtype=float)
    fig_path = out_dir / 'robot_topn_score_matrix.png'

    plt.figure(figsize=(max(8, len(robot_names) * 1.2), max(4, len(image_names) * 0.7)))
    im = plt.imshow(scores, cmap='viridis', aspect='auto')
    plt.xticks(np.arange(len(robot_names)), robot_names, rotation=25, ha='right')
    plt.yticks(np.arange(len(image_names)), image_names)
    plt.title(f'Robot Selection Score Matrix (Top-{top_n} enabled)')
    plt.xlabel('Robot')
    plt.ylabel('Image')
    for i in range(scores.shape[0]):
        top_idx = int(np.argmax(scores[i]))
        for j in range(scores.shape[1]):
            txt_color = 'white' if j == top_idx else 'black'
            plt.text(j, i, f"{scores[i, j]:.2f}", ha='center', va='center', fontsize=8, color=txt_color)
    plt.colorbar(im, label='Score')
    plt.tight_layout()
    plt.savefig(fig_path, dpi=220)
    plt.close()

    json_path = out_dir / 'robot_topn_predictions.json'
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(details, f, indent=2)

    return {
        'robot_topn_matrix': str(fig_path),
        'robot_topn_json': str(json_path),
    }
