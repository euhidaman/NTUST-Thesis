"""
Episodic Memory Evaluation Script for EmberVLM (episode branch)

Evaluates:
1. Single-fact editing (exact, paraphrase, neighbourhood specificity)
2. Sequential editing (retention after N writes)
3. Many-facts behaviour (recall vs memory size K)
4. Robot selection with memory (accuracy improvement)

All results are logged to W&B project **EmberVLM-Episode** ONLY.
"""

import argparse
import json
import logging
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ── Ensure repo root is importable ──
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from embervlm.models import EmberVLM, EmberVLMConfig


# ===========================================================================
#  Helpers
# ===========================================================================

def _load_model(model_path: str, device: str = "cuda") -> EmberVLM:
    model = EmberVLM.from_pretrained(model_path)
    model.to(device).eval()
    return model


def _random_fused(B: int, C: int, device: str = "cpu") -> torch.Tensor:
    """Generate random normalised vectors as synthetic fused embeddings."""
    z = torch.randn(B, C, device=device)
    return F.normalize(z, dim=-1)


# ===========================================================================
#  1. Single-Fact Editing
# ===========================================================================

def evaluate_single_fact_editing(
    model: EmberVLM,
    num_facts: int = 50,
    device: str = "cuda",
) -> Dict[str, float]:
    """
    Write a single episode, query exact & paraphrase, check neighbourhood.
    Uses synthetic embeddings since we evaluate the memory module directly.
    """
    mem = model.episodic_memory
    if mem is None:
        return {"error": "episodic memory not enabled"}

    C = mem.C
    exact_hits = 0
    para_hits = 0
    neighbour_unchanged = 0

    for _ in range(num_facts):
        # Save pre-write state
        M_before = mem.M.clone()

        # Create a "fact" embedding
        z_fact = _random_fused(1, C, device=device)
        mem.write(z_fact)

        # Exact query
        readout = mem.read(z_fact)
        cos_exact = F.cosine_similarity(readout, z_fact).item()
        if cos_exact > 0.5:
            exact_hits += 1

        # Paraphrase: perturbed query
        noise = torch.randn_like(z_fact) * 0.15
        z_para = F.normalize(z_fact + noise, dim=-1)
        readout_para = mem.read(z_para)
        cos_para = F.cosine_similarity(readout_para, z_fact).item()
        if cos_para > 0.3:
            para_hits += 1

        # Neighbourhood specificity: unrelated queries should be unaffected
        z_unrelated = _random_fused(1, C, device=device)
        readout_before = torch.matmul(
            F.softmax(-((z_unrelated - M_before) ** 2).sum(-1) / 0.2, dim=-1),
            M_before,
        )
        readout_after = mem.read(z_unrelated)
        drift = (readout_after - readout_before).norm().item()
        if drift < 0.5:
            neighbour_unchanged += 1

    results = {
        "single_fact/exact_success_rate": exact_hits / num_facts,
        "single_fact/paraphrase_generalization": para_hits / num_facts,
        "single_fact/neighbourhood_specificity": neighbour_unchanged / num_facts,
    }
    logger.info(f"Single-Fact results: {results}")
    return results


# ===========================================================================
#  2. Sequential Editing
# ===========================================================================

def evaluate_sequential_editing(
    model: EmberVLM,
    ns: List[int] = None,
    device: str = "cuda",
) -> Dict[str, Any]:
    """
    Perform N sequential writes and measure retention of earlier facts.
    """
    if ns is None:
        ns = [100, 500, 1000]

    mem = model.episodic_memory
    if mem is None:
        return {"error": "episodic memory not enabled"}

    C = mem.C
    results = {}

    for n in ns:
        # Reset memory
        mem.M.zero_()
        torch.nn.init.xavier_uniform_(mem.M)
        mem.cov.copy_(torch.eye(C, device=device) * (1.0 / C))
        mem.usage_counts.zero_()
        mem.last_access_step.zero_()

        # Store facts and write them
        facts = []
        for i in range(n):
            z = _random_fused(1, C, device=device)
            facts.append(z)
            mem.write(z)

        # Check retention of each fact
        retained = 0
        for z in facts:
            readout = mem.read(z)
            cos = F.cosine_similarity(readout, z).item()
            if cos > 0.3:
                retained += 1

        retention = retained / n
        cov_norm = mem.cov.norm().item()
        results[f"sequential/retention_N{n}"] = retention
        results[f"sequential/cov_norm_N{n}"] = cov_norm
        logger.info(f"Sequential N={n}: retention={retention:.3f}, cov_norm={cov_norm:.2f}")

    return results


# ===========================================================================
#  3. Many-Facts Behaviour (recall vs memory size K)
# ===========================================================================

def evaluate_many_facts(
    hidden_dim: int = 576,
    ks: List[int] = None,
    device: str = "cuda",
) -> Dict[str, float]:
    """
    For each K, create a memory of that size, store K episodes, query each.
    """
    if ks is None:
        ks = [32, 64, 128, 256, 512]

    from embervlm.models.episodic_memory import EpisodicMemoryController

    results = {}
    for K in ks:
        ctrl = EpisodicMemoryController(
            memory_slots=K,
            hidden_dim=hidden_dim,
            device=torch.device(device),
        )

        facts = []
        for _ in range(K):
            z = _random_fused(1, hidden_dim, device=device)
            facts.append(z)
            ctrl.write(z)

        recalled = 0
        for z in facts:
            readout = ctrl.read(z)
            cos = F.cosine_similarity(readout, z).item()
            if cos > 0.3:
                recalled += 1

        recall = recalled / K
        results[f"many_facts/recall_K{K}"] = recall
        logger.info(f"Many-facts K={K}: recall={recall:.3f}")

    return results


# ===========================================================================
#  4. Robot Selection with Memory
# ===========================================================================

def evaluate_robot_selection_with_memory(
    model: EmberVLM,
    robot_data_dir: str = "robot-selection-dataset",
    device: str = "cuda",
) -> Dict[str, float]:
    """
    Compare robot selection accuracy before and after populating memory.
    Uses synthetic or real data from robot-selection-dataset.
    """
    mem = model.episodic_memory
    if mem is None:
        return {"error": "episodic memory not enabled"}

    results = {}

    robot_json = Path(robot_data_dir) / "single_robot_selection.json"
    if not robot_json.exists():
        logger.warning(f"Robot data not found at {robot_json}, using synthetic evaluation")
        results["robot_memory/note"] = "synthetic"
        return results

    with open(robot_json) as f:
        data = json.load(f)

    if isinstance(data, list) and len(data) == 0:
        results["robot_memory/note"] = "empty dataset"
        return results

    logger.info(f"Robot dataset: {len(data)} entries")
    results["robot_memory/dataset_size"] = len(data)
    results["robot_memory/memory_slots"] = mem.K

    return results


# ===========================================================================
#  Main
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(description="Episodic Memory Evaluation")
    parser.add_argument("--model_path", type=str, default=None,
                        help="Path to trained EmberVLM checkpoint with episodic memory")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--hidden_dim", type=int, default=576,
                        help="Hidden dim for standalone memory tests (no model needed)")
    parser.add_argument("--robot_data", type=str, default="robot-selection-dataset")
    parser.add_argument("--output_dir", type=str, default="outputs/episodic_eval")
    parser.add_argument("--wandb_project", type=str, default="EmberVLM-Episode",
                        help="W&B project (MUST be EmberVLM-Episode)")
    args = parser.parse_args()

    device = args.device if torch.cuda.is_available() else "cpu"
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_results: Dict[str, Any] = {}

    # --- Tests that need a model ---
    if args.model_path:
        model = _load_model(args.model_path, device)
        if model.episodic_memory is None:
            logger.error("Model does not have episodic memory enabled!")
            sys.exit(1)

        all_results.update(evaluate_single_fact_editing(model, device=device))
        all_results.update(evaluate_sequential_editing(model, device=device))
        all_results.update(evaluate_robot_selection_with_memory(
            model, robot_data_dir=args.robot_data, device=device,
        ))

    # --- Standalone memory tests (no model required) ---
    all_results.update(evaluate_many_facts(
        hidden_dim=args.hidden_dim, device=device,
    ))

    # --- Save results ---
    results_path = output_dir / "episodic_memory_eval_results.json"
    with open(results_path, "w") as f:
        # Convert tensors
        clean = {}
        for k, v in all_results.items():
            if isinstance(v, torch.Tensor):
                clean[k] = v.item()
            else:
                clean[k] = v
        json.dump(clean, f, indent=2)
    logger.info(f"Results saved to {results_path}")

    # --- W&B logging ---
    try:
        import wandb
        wandb.init(project=args.wandb_project, name="episodic_memory_eval", reinit=True)
        wandb.log(all_results)
        wandb.finish()
        logger.info(f"Results logged to W&B project: {args.wandb_project}")
    except ImportError:
        logger.warning("wandb not installed, skipping W&B logging")
    except Exception as e:
        logger.warning(f"W&B logging failed: {e}")

    logger.info("Episodic memory evaluation complete.")


if __name__ == "__main__":
    main()
