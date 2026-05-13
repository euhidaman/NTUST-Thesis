"""
Visual Absence (VA) Neuron Activation Heatmap Analysis for EmberVLM.

Generates Figure 5-style heatmaps showing how FFN neurons in the language
model's middle layers respond differently when an object IS in the image
("present") versus when it is NOT ("absent").

Key insight: certain FFN neurons activate strongly for tokens that refer to
visually absent objects, forming a detectable hallucination signature.

This module:
  1.  Hooks into the MLP (FFN) sub-modules of selected decoder layers.
  2.  Runs forward passes with present/absent object queries on real images.
  3.  Captures per-layer, per-neuron activation magnitudes.
  4.  Generates:
        (a) A 2×2 heatmap grid (present/absent × two objects).
        (b) A cosine-similarity bar chart comparing absent-absent,
            present-present, and cross-status activation vectors.

Compatible with both trial and main training modes.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from PIL import Image as PILImage

logger = logging.getLogger(__name__)

# ── Style constants ──────────────────────────────────────────────────
CMAP_PRESENT = "YlOrBr"   # warm, sparse look  (low activation)
CMAP_ABSENT  = "GnBu"     # cool, dense look   (strong activation)
BAR_COLORS   = {"absent": "#EF5350", "present": "#66BB6A", "cross": "#42A5F5"}

DPI = 200
TOP_N_NEURONS = 100  # number of top-activation neurons to show on x-axis


# ── Probe definitions ────────────────────────────────────────────────

@dataclass
class VAProbe:
    """A single VA probe: one image plus present/absent object queries."""
    image: PILImage.Image
    image_name: str
    present_object: str          # object that IS in the image
    absent_object: str           # object that is NOT in the image
    question_template: str = "Is the {object} in the image?"

    def question_for(self, obj: str) -> str:
        return self.question_template.format(object=obj)


def build_default_probes(
    images: List[Tuple[str, PILImage.Image]],
) -> List[VAProbe]:
    """Build a set of VA probes from available evaluation images.

    Heuristic: we pair each image with a plausible *present* object
    (generic enough to be in any natural image) and an *absent* object
    (unlikely to appear in a random photo).
    """
    # Present/absent pairs — designed so *present* is likely true for
    # typical GQA/COCO photos and *absent* is clearly not there.
    pa_pairs = [
        # (present_object, absent_object, question_template)
        ("person", "elephant",
         "Is there a {object} in this image?"),
        ("sky", "submarine",
         "Can you see a {object} in this image?"),
        ("building", "penguin",
         "Is there a {object} visible in this image?"),
        ("tree", "helicopter",
         "Does this image contain a {object}?"),
        ("car", "giraffe",
         "Is there a {object} in the scene?"),
    ]

    probes: List[VAProbe] = []
    for i, (img_name, img) in enumerate(images):
        present, absent, tmpl = pa_pairs[i % len(pa_pairs)]
        probes.append(VAProbe(
            image=img,
            image_name=img_name,
            present_object=present,
            absent_object=absent,
            question_template=tmpl,
        ))
    return probes


# ── FFN hook infrastructure ──────────────────────────────────────────

class FFNActivationCapture:
    """Register forward hooks on LlamaMLP layers and store activations.

    After a forward pass, ``activations[layer_idx]`` holds the FFN output
    tensor ``[B, T, D_ffn]`` (detached, on CPU to save VRAM).
    """

    def __init__(self, layer_indices: List[int]):
        self.layer_indices = layer_indices
        self.activations: Dict[int, torch.Tensor] = {}
        self._handles: list = []

    # ── hook callback ────────────────────────────────────────────────
    def _make_hook(self, layer_idx: int):
        def hook_fn(_module, _input, output):
            # output: [B, T, D_ffn] from LlamaMLP.forward()
            with torch.no_grad():
                self.activations[layer_idx] = output.detach().cpu()
        return hook_fn

    def register(self, language_model: nn.Module) -> bool:
        """Attach hooks.  Returns True on success.

        Expected hierarchy (SmolLM / LlamaForCausalLM):
            language_model.model.model.layers[i].mlp
        """
        layers = None
        lm = language_model
        # Navigate: SmolLMBackbone → LlamaForCausalLM → LlamaModel → layers
        if hasattr(lm, "model") and hasattr(lm.model, "model") and hasattr(lm.model.model, "layers"):
            layers = lm.model.model.layers
        elif hasattr(lm, "model") and hasattr(lm.model, "layers"):
            layers = lm.model.layers
        elif hasattr(lm, "layers"):
            layers = lm.layers

        if layers is None:
            logger.warning("FFNActivationCapture: cannot find decoder layers.")
            return False

        for idx in self.layer_indices:
            if idx >= len(layers):
                logger.warning(f"FFNActivationCapture: layer {idx} out of range ({len(layers)} layers).")
                continue
            mlp = getattr(layers[idx], "mlp", None)
            if mlp is None:
                logger.warning(f"FFNActivationCapture: layer {idx} has no 'mlp'.")
                continue
            h = mlp.register_forward_hook(self._make_hook(idx))
            self._handles.append(h)

        logger.info(f"FFNActivationCapture: hooked {len(self._handles)} layers "
                    f"(indices {self.layer_indices[:5]}{'...' if len(self.layer_indices)>5 else ''})")
        return len(self._handles) > 0

    def reset(self):
        self.activations.clear()

    def remove(self):
        for h in self._handles:
            h.remove()
        self._handles.clear()
        self.activations.clear()


# ── Core analysis ────────────────────────────────────────────────────

@torch.no_grad()
def _run_single_probe(
    model,
    tokenizer,
    image: PILImage.Image,
    question: str,
    capture: FFNActivationCapture,
    device: torch.device,
    dtype: torch.dtype,
    preprocess_fn=None,
) -> Dict[int, torch.Tensor]:
    """Run one forward pass and return per-layer activation vectors.

    Returns
    -------
    dict  layer_idx → Tensor [D_ffn]  (mean over sequence positions)
    """
    capture.reset()

    # Tokenize
    prompt = f"<image>\nUser: {question}\nAssistant:"
    inputs = tokenizer(prompt, return_tensors="pt", padding=True,
                       truncation=True, max_length=256, add_special_tokens=True)
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)

    # Preprocess image
    pixel_values = None
    if preprocess_fn is not None:
        pixel_values = preprocess_fn(image)
    else:
        # Fallback: manual preprocessing
        img = image.convert("RGB").resize((224, 224), PILImage.LANCZOS)
        arr = np.array(img, dtype=np.float32) / 255.0
        t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std  = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        pixel_values = ((t - mean) / std).to(device=device, dtype=dtype)

    # Forward
    model.eval()
    _ = model(
        input_ids=input_ids,
        pixel_values=pixel_values,
        attention_mask=attention_mask,
        output_hidden_states=False,
    )

    # Collect: mean activation magnitude per neuron across sequence positions
    result: Dict[int, torch.Tensor] = {}
    for layer_idx, act in capture.activations.items():
        # act: [B, T, D_ffn]  →  [D_ffn]  (mean over B and T)
        result[layer_idx] = act.float().abs().mean(dim=(0, 1))  # [D_ffn]

    return result


def run_va_neuron_analysis(
    model,
    tokenizer,
    probes: List[VAProbe],
    layer_indices: Optional[List[int]] = None,
    top_n: int = TOP_N_NEURONS,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
    preprocess_fn=None,
) -> Dict[str, Any]:
    """Run complete VA neuron analysis across all probes.

    Parameters
    ----------
    model : EmberVLM
        The model (must be on *device* and in eval mode).
    tokenizer : PreTrainedTokenizer
    probes : list[VAProbe]
        Each probe has an image, present object, and absent object.
    layer_indices : list[int] | None
        Which decoder layers to monitor.  Defaults to middle layers.
    top_n : int
        Number of top-activation neurons to keep per layer.
    device, dtype : torch casting targets.
    preprocess_fn : callable | None
        ``image → pixel_values tensor``.  Uses fallback if None.

    Returns
    -------
    dict with keys:
        "probes"           – list of per-probe result dicts
        "cosine_absent"    – float, mean cosine sim among absent-absent pairs
        "cosine_present"   – float, mean cosine sim among present-present pairs
        "cosine_cross"     – float, mean cosine sim among cross-status pairs
        "layer_indices"    – the actual layer indices used
        "top_n"            – the top_n value
    """
    if device is None:
        device = next(model.parameters()).device

    # Default: monitor the middle third of the model
    if layer_indices is None:
        n_layers = 30  # SmolLM-135M
        try:
            lm = model.language_model
            if hasattr(lm, "model") and hasattr(lm.model, "model") and hasattr(lm.model.model, "layers"):
                n_layers = len(lm.model.model.layers)
            elif hasattr(lm, "model") and hasattr(lm.model, "layers"):
                n_layers = len(lm.model.layers)
        except Exception:
            pass
        # Middle third, every other layer (keeps hook count reasonable)
        start = n_layers // 3
        end = 2 * n_layers // 3
        layer_indices = list(range(start, end))

    capture = FFNActivationCapture(layer_indices)
    if not capture.register(model.language_model):
        logger.error("VA neuron analysis: failed to register hooks — aborting.")
        return {"probes": [], "error": "hook_registration_failed"}

    try:
        all_present_vecs: List[torch.Tensor] = []
        all_absent_vecs:  List[torch.Tensor] = []
        probe_results: List[Dict[str, Any]] = []

        for probe in probes:
            # --- present query ---
            q_present = probe.question_for(probe.present_object)
            acts_present = _run_single_probe(
                model, tokenizer, probe.image, q_present,
                capture, device, dtype, preprocess_fn,
            )

            # --- absent query ---
            q_absent = probe.question_for(probe.absent_object)
            acts_absent = _run_single_probe(
                model, tokenizer, probe.image, q_absent,
                capture, device, dtype, preprocess_fn,
            )

            # Build per-layer heatmap rows (top-N neurons per layer)
            heatmap_present: List[np.ndarray] = []
            heatmap_absent:  List[np.ndarray] = []
            used_layers: List[int] = []

            for l_idx in sorted(layer_indices):
                if l_idx not in acts_present or l_idx not in acts_absent:
                    continue
                vec_p = acts_present[l_idx]  # [D_ffn]
                vec_a = acts_absent[l_idx]

                # Select top-N based on ABSENT activations (VA neurons are
                # defined by their high response to absent tokens)
                combined = vec_p + vec_a
                _, topk_idx = combined.topk(min(top_n, combined.numel()))
                row_p = vec_p[topk_idx].numpy()
                row_a = vec_a[topk_idx].numpy()

                heatmap_present.append(row_p)
                heatmap_absent.append(row_a)
                used_layers.append(l_idx)

            # Flatten into single activation vector for cosine analysis
            if heatmap_present:
                vec_flat_p = np.concatenate(heatmap_present)
                vec_flat_a = np.concatenate(heatmap_absent)
                all_present_vecs.append(torch.from_numpy(vec_flat_p))
                all_absent_vecs.append(torch.from_numpy(vec_flat_a))

            probe_results.append({
                "image_name": probe.image_name,
                "present_object": probe.present_object,
                "absent_object": probe.absent_object,
                "question_present": q_present,
                "question_absent": q_absent,
                "heatmap_present": np.array(heatmap_present) if heatmap_present else None,
                "heatmap_absent": np.array(heatmap_absent) if heatmap_absent else None,
                "layers": used_layers,
            })

        # ── Cosine similarity statistics ────────────────────────────
        cos_absent = cos_present = cos_cross = 0.0

        def _pairwise_cosine(vecs: List[torch.Tensor]) -> float:
            if len(vecs) < 2:
                return 0.0
            sims = []
            for i in range(len(vecs)):
                for j in range(i + 1, len(vecs)):
                    c = torch.nn.functional.cosine_similarity(
                        vecs[i].unsqueeze(0), vecs[j].unsqueeze(0)).item()
                    sims.append(c)
            return float(np.mean(sims)) if sims else 0.0

        def _cross_cosine(vecs_a: List[torch.Tensor],
                          vecs_b: List[torch.Tensor]) -> float:
            if not vecs_a or not vecs_b:
                return 0.0
            sims = []
            for va in vecs_a:
                for vb in vecs_b:
                    c = torch.nn.functional.cosine_similarity(
                        va.unsqueeze(0), vb.unsqueeze(0)).item()
                    sims.append(c)
            return float(np.mean(sims)) if sims else 0.0

        cos_absent  = _pairwise_cosine(all_absent_vecs)
        cos_present = _pairwise_cosine(all_present_vecs)
        cos_cross   = _cross_cosine(all_absent_vecs, all_present_vecs)

        return {
            "probes": probe_results,
            "cosine_absent": cos_absent,
            "cosine_present": cos_present,
            "cosine_cross": cos_cross,
            "layer_indices": layer_indices,
            "top_n": top_n,
        }

    finally:
        capture.remove()


# ── Visualisation ────────────────────────────────────────────────────

def _normalize_heatmap(hm: np.ndarray) -> np.ndarray:
    """Row-wise min-max normalisation so each layer is 0-1."""
    out = np.zeros_like(hm, dtype=np.float32)
    for i in range(hm.shape[0]):
        row = hm[i]
        lo, hi = row.min(), row.max()
        if hi - lo > 1e-8:
            out[i] = (row - lo) / (hi - lo)
        else:
            out[i] = 0.0
    return out


def plot_va_neuron_heatmap(
    analysis_results: Dict[str, Any],
    output_dir: str = "./outputs",
    max_probes: int = 2,
    save: bool = True,
    step: Optional[int] = None,
) -> Tuple[Optional[plt.Figure], Optional[PILImage.Image]]:
    """Generate the VA neuron activation heatmap figure.

    Layout:
        Left 2×2: heatmaps for up to 2 probes (present/absent each).
        Right:    cosine-similarity bar chart.

    Parameters
    ----------
    analysis_results : dict
        Output of ``run_va_neuron_analysis()``.
    output_dir : str
        Directory to save the figure.
    max_probes : int
        Number of probes to visualise (uses first N with valid data).
    save : bool
        Whether to persist to disk.
    step : int | None
        Training step number. When provided the file is saved into a
        ``va_neuron_heatmaps/`` subfolder as
        ``va_neuron_activation_heatmap-step{step}.png``.

    Returns
    -------
    (fig, pil_image) or (None, None) on failure.
    """
    probes = [p for p in analysis_results.get("probes", [])
              if p.get("heatmap_present") is not None]
    if not probes:
        logger.warning("VA neuron heatmap: no valid probe data — skipping plot.")
        return None, None

    probes = probes[:max_probes]
    n_probes = len(probes)
    cos_a = analysis_results.get("cosine_absent", 0)
    cos_p = analysis_results.get("cosine_present", 0)
    cos_x = analysis_results.get("cosine_cross", 0)
    top_n = analysis_results.get("top_n", TOP_N_NEURONS)

    # ── Figure layout ────────────────────────────────────────────────
    #   Left: n_probes rows × 2 cols (present | absent)
    #   Right: 1 bar chart spanning full height
    fig = plt.figure(figsize=(18, 4 * n_probes + 1.5))
    fig.patch.set_facecolor("white")

    # GridSpec: n_probes rows, 5 cols  (4 for heatmaps, 1 for bar)
    gs = gridspec.GridSpec(n_probes, 5, figure=fig,
                           width_ratios=[2, 2, 0.15, 0.15, 1.8],
                           hspace=0.45, wspace=0.35)

    for row, probe in enumerate(probes):
        hm_p = _normalize_heatmap(probe["heatmap_present"])
        hm_a = _normalize_heatmap(probe["heatmap_absent"])
        layers = probe["layers"]
        present_obj = probe["present_object"]
        absent_obj  = probe["absent_object"]

        y_labels = [f"L{l}" for l in layers]

        # ── Present heatmap ──────────────────────────────────────
        ax_p = fig.add_subplot(gs[row, 0])
        im_p = ax_p.imshow(hm_p, aspect="auto", cmap=CMAP_PRESENT,
                           vmin=0, vmax=1, interpolation="nearest")
        ax_p.set_yticks(range(len(layers)))
        ax_p.set_yticklabels(y_labels, fontsize=8)
        ax_p.set_xlabel(f"Top-{top_n} FFN neurons", fontsize=9)
        ax_p.set_ylabel("Layers", fontsize=9)
        ax_p.set_title(f"{present_obj} (present)", fontsize=11,
                       fontweight="bold", color="#2E7D32")  # green

        # ── Absent heatmap ───────────────────────────────────────
        ax_a = fig.add_subplot(gs[row, 1])
        im_a = ax_a.imshow(hm_a, aspect="auto", cmap=CMAP_ABSENT,
                           vmin=0, vmax=1, interpolation="nearest")
        ax_a.set_yticks(range(len(layers)))
        ax_a.set_yticklabels(y_labels, fontsize=8)
        ax_a.set_xlabel(f"Top-{top_n} FFN neurons", fontsize=9)
        ax_a.set_title(f"{absent_obj} (absent)", fontsize=11,
                       fontweight="bold", color="#C62828")  # red

        # Colour bars (one per heatmap, narrow columns 2 & 3)
        cb_p = fig.colorbar(im_p, cax=fig.add_subplot(gs[row, 2]))
        cb_p.ax.tick_params(labelsize=7)
        cb_a = fig.colorbar(im_a, cax=fig.add_subplot(gs[row, 3]))
        cb_a.ax.tick_params(labelsize=7)

    # ── Cosine similarity bar chart (right panel) ────────────────
    ax_bar = fig.add_subplot(gs[:, 4])
    labels = ["absent\nvs absent", "present\nvs present", "present\nvs absent"]
    values = [cos_a, cos_p, cos_x]
    colors = [BAR_COLORS["absent"], BAR_COLORS["present"], BAR_COLORS["cross"]]

    bars = ax_bar.bar(labels, values, color=colors, edgecolor="white",
                      linewidth=1.5, width=0.55)
    for bar, v in zip(bars, values):
        ax_bar.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f"{v:.2f}", ha="center", va="bottom", fontsize=10,
                    fontweight="bold")
    ax_bar.set_ylabel("Cosine similarity", fontsize=11)
    ax_bar.set_ylim(0, max(max(values) * 1.25, 0.1))
    ax_bar.set_title("Activation similarity\nby visual status", fontsize=11,
                     fontweight="bold")
    ax_bar.grid(axis="y", alpha=0.3, linestyle="--")
    ax_bar.set_facecolor("#FAFAFA")
    ax_bar.tick_params(axis="x", labelsize=9)

    # ── Super-title ──────────────────────────────────────────────
    probe0 = probes[0]
    q_tmpl = probe0.get("question_present", "").split("?")[0]
    if not q_tmpl:
        q_tmpl = f"Is there a [{probe0['present_object']} / {probe0['absent_object']}]"
    step_label = f"   •   Step {step}" if step is not None else ""
    fig.suptitle(
        f"Activation Patterns of High $S^{{VA}}$ Neurons Across Varying Visual Contexts\n"
        f"Input: \"{q_tmpl}?\"   •   Image: {probe0['image_name'][:30]}{step_label}",
        fontsize=13, fontweight="bold", y=1.01,
    )

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    # ── Save ─────────────────────────────────────────────────────
    pil_img = None
    if save:
        out = Path(output_dir)
        if step is not None:
            out = out / "va_neuron_heatmaps"
            filename = f"va_neuron_activation_heatmap-step{step}.png"
        else:
            filename = "va_neuron_activation_heatmap.png"
        out.mkdir(parents=True, exist_ok=True)
        save_path = out / filename
        fig.savefig(str(save_path), bbox_inches="tight", dpi=DPI,
                    facecolor=fig.get_facecolor())
        logger.info(f"✓ Saved VA neuron heatmap: {save_path}")

    # Convert to PIL for W&B logging
    try:
        import io
        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight", dpi=DPI,
                    facecolor=fig.get_facecolor())
        buf.seek(0)
        pil_img = PILImage.open(buf).copy()
        buf.close()
    except Exception as e:
        logger.warning(f"VA neuron heatmap: PIL conversion failed: {e}")

    plt.close(fig)
    return fig, pil_img


# ── High-level entry point (called from Stage 2.5 evaluation) ───────

def generate_va_neuron_heatmap(
    model,
    tokenizer,
    images: List[Tuple[str, PILImage.Image]],
    output_dir: str,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
    preprocess_fn=None,
    step: Optional[int] = None,
) -> Optional[str]:
    """End-to-end: build probes → run analysis → generate figure.

    Parameters
    ----------
    model : EmberVLM (on device, eval mode)
    tokenizer : PreTrainedTokenizer
    images : list[(name, PIL.Image)]
        Evaluation images (at least 2 recommended).
    output_dir : str
        Where to save the heatmap PNG.
    device, dtype : model device / precision.
    preprocess_fn : callable | None
        Image → pixel_values.
    step : int | None
        Training step number for periodic tracking.

    Returns
    -------
    Path to saved figure, or None on failure.
    """
    if not images:
        logger.warning("VA neuron heatmap: no images supplied — skipping.")
        return None

    try:
        probes = build_default_probes(images[:3])  # use up to 3 images
        logger.info(f"VA neuron analysis: {len(probes)} probes, "
                    f"{len(images)} images")
        for p in probes:
            logger.info(f"  Probe: image={p.image_name[:25]}, "
                        f"present={p.present_object}, absent={p.absent_object}")

        results = run_va_neuron_analysis(
            model=model,
            tokenizer=tokenizer,
            probes=probes,
            device=device,
            dtype=dtype,
            preprocess_fn=preprocess_fn,
        )

        if results.get("error"):
            logger.warning(f"VA neuron analysis failed: {results['error']}")
            return None

        logger.info(f"VA neuron cosine similarities: "
                    f"absent-absent={results['cosine_absent']:.3f}, "
                    f"present-present={results['cosine_present']:.3f}, "
                    f"cross={results['cosine_cross']:.3f}")

        _, pil_img = plot_va_neuron_heatmap(
            results, output_dir=output_dir, max_probes=2, save=True,
            step=step,
        )

        # Log to W&B if available
        try:
            import wandb
            if wandb.run is not None and pil_img is not None:
                tag = "va_neuron_heatmap"
                if step is not None:
                    wandb.log({f"va_heatmap/step_{step}": wandb.Image(pil_img)},
                              step=step)
                else:
                    wandb.log({"stage2_5/va_neuron_heatmap": wandb.Image(pil_img)})
                logger.info("✓ VA neuron heatmap logged to W&B")
        except Exception:
            pass

        if step is not None:
            save_path = str(Path(output_dir) / "va_neuron_heatmaps"
                           / f"va_neuron_activation_heatmap-step{step}.png")
        else:
            save_path = str(Path(output_dir) / "va_neuron_activation_heatmap.png")
        return save_path if Path(save_path).exists() else None

    except Exception as e:
        logger.warning(f"VA neuron heatmap generation failed (non-fatal): {e}")
        import traceback
        traceback.print_exc()
        return None


# ── Periodic tracker for use inside training loops ───────────────────

class VAHeatmapTracker:
    """Lightweight wrapper that periodically generates VA neuron heatmaps
    during training.

    Usage inside a trainer::

        self.va_tracker = VAHeatmapTracker(
            output_dir=config.output_dir,
            interval=50,          # every 50 training steps
        )

        # inside the training loop, after the backward pass:
        self.va_tracker.maybe_generate(
            model=self.model,
            tokenizer=self.tokenizer,
            global_step=self.global_step,
            device=self.device,
        )

    The tracker lazily loads sample images on first invocation and caches
    them for subsequent calls.  It switches the model to eval mode
    temporarily, generates the heatmap, then restores the previous mode.
    """

    def __init__(
        self,
        output_dir: str,
        interval: int = 50,
        num_images: int = 3,
    ):
        self.output_dir = output_dir
        self.interval = max(1, interval)
        self.num_images = num_images
        self._images: Optional[List[Tuple[str, PILImage.Image]]] = None
        self._generated_steps: List[int] = []

    # ── lazy image loading ────────────────────────────────────────
    def _ensure_images(self) -> List[Tuple[str, PILImage.Image]]:
        """Load and cache sample images from unibench (or synthetic fallback)."""
        if self._images is not None:
            return self._images

        try:
            from embervlm.training.stage2_5_eval_lighteval import (
                get_sample_images_from_unibench,
            )
            self._images = get_sample_images_from_unibench(
                num_images=self.num_images,
            )
        except Exception:
            pass

        if not self._images:
            # Fallback: create a small synthetic image so the analysis can
            # still run (results will be less informative but the pipeline
            # won't break).
            logger.warning(
                "VAHeatmapTracker: unibench images unavailable — "
                "using synthetic 224×224 noise image as fallback."
            )
            import numpy as _np
            synth = PILImage.fromarray(
                _np.random.randint(0, 255, (224, 224, 3), dtype=_np.uint8)
            )
            self._images = [("synthetic_noise", synth)]

        return self._images

    # ── public API ────────────────────────────────────────────────
    def maybe_generate(
        self,
        model,
        tokenizer,
        global_step: int,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
        preprocess_fn=None,
        force: bool = False,
    ) -> Optional[str]:
        """Generate a heatmap if *global_step* is a multiple of *interval*.

        Parameters
        ----------
        model : EmberVLM (may be DDP-wrapped).
        tokenizer : PreTrainedTokenizer
        global_step : int
        device, dtype, preprocess_fn : forwarded to ``generate_va_neuron_heatmap``.
        force : bool
            Generate unconditionally, ignoring the interval check.

        Returns
        -------
        Path to the saved PNG, or ``None`` if skipped / failed.
        """
        if not force and (global_step % self.interval != 0 or global_step == 0):
            return None

        # Avoid duplicate generation for the same step
        if global_step in self._generated_steps:
            return None

        images = self._ensure_images()
        if not images:
            return None

        # Unwrap DDP if needed
        raw_model = model
        if hasattr(model, "module"):
            raw_model = model.module

        # Switch to eval mode, generate, restore
        was_training = raw_model.training
        raw_model.eval()
        try:
            with torch.no_grad():
                path = generate_va_neuron_heatmap(
                    model=raw_model,
                    tokenizer=tokenizer,
                    images=images,
                    output_dir=self.output_dir,
                    device=device,
                    dtype=dtype,
                    preprocess_fn=preprocess_fn,
                    step=global_step,
                )
        finally:
            if was_training:
                raw_model.train()

        if path:
            self._generated_steps.append(global_step)
            logger.info(
                f"[VAHeatmapTracker] step {global_step}: saved → {path}"
            )
        return path

    @property
    def generated_steps(self) -> List[int]:
        """Return the list of steps for which a heatmap was generated."""
        return list(self._generated_steps)
