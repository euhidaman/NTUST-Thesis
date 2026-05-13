# EmberVLM Visualization System

## Overview

EmberVLM produces visualizations at every stage of its 4-stage training pipeline, during the
Stage 2.5 hallucination evaluation, and as a final cross-stage collation after the full run
completes.  Every plot is saved as a PNG file (150-220 DPI) in the stage's own
`visualizations/` subfolder.  This document lists every figure that is actually generated,
where it lives on disk, and how to read it.


## Stage 1 -- Vision-Language Alignment

Output folder: `<output_dir>/stage1/visualizations/`

Plots are produced every 500 training steps and once more at the end of training with all
accumulated data.

### 1. Similarity Matrix  (`similarity_matrix_step{N}.png`)

Two side-by-side panels generated from the current batch of image and text embeddings.

- Left panel -- cosine similarity heatmap.  Rows are images, columns are texts.  Bright
  diagonal entries mean the model is pairing the right image with the right caption.  Off-
  diagonal brightness means confusion between pairs.
- Right panel -- softmax-normalised retrieval probabilities derived from the same matrix.  A
  sharper diagonal indicates more confident retrieval.
- The suptitle prints the batch retrieval accuracy (percentage of images whose highest-scoring
  text is the correct partner).

How to interpret:  A well-trained alignment stage shows a bright, narrow diagonal with a dark
off-diagonal.  If rows share bright cells across multiple columns the model is not
discriminating captions; if columns share bright cells across multiple rows the model is
returning the same caption for different images.

### 2. t-SNE Embedding Scatter  (`tsne_step{N}.png`)

A 2-D t-SNE projection of up to 100 image and text embeddings.

- Blue dots are image embeddings, orange dots are text embeddings.
- Thin lines connect matching image-text pairs.
- Short lines mean the pair is close in embedding space (well aligned).  Long crossing lines
  mean the model has not yet learned to bring matching pairs together.

How to interpret:  Early in training, blue and orange clusters tend to be separate and lines
are long.  As alignment improves the two point-clouds merge and lines shorten.  Overlapping
clusters with short lines indicate good cross-modal alignment.


## Stage 2 -- Instruction Tuning

Output folder: `<output_dir>/stage2/visualizations/`

Produced every 500 steps and at end of training.

### 3. Token Probability Distribution  (`token_probs_step{N}.png`)

Two panels derived from the model's logits and the ground-truth label tokens.

- Left panel -- histogram of the probability the model assigns to the correct next token
  across all positions.  The vertical dashed line marks the mean.  A distribution that piles
  up near 1.0 means the model is very confident and usually correct.
- Right panel -- bar chart of confidence categories ("Very Low" <0.1, "Low" 0.1-0.3,
  "Medium" 0.3-0.6, "High" 0.6-0.9, "Confident" >0.9).  Percentage annotations show the
  share in each bin.

How to interpret:  Early checkpoints show most mass in the low bins; as instruction tuning
progresses the distribution shifts right.  A tall "Confident" bar (>50 %) is a sign of good
language-modelling quality.  If "Very Low" remains large, the model is struggling with the
instruction format.


## Stage 2.5 -- Hallucination Evaluation

Output folder: `<output_dir>/stage2_5_evaluation/`  (some files in `visualizations/` subfolder,
some in `paper_figures/`)

Stage 2.5 is not a training stage.  It is a comprehensive evaluation run between Stage 2 and
Stage 3 that probes the model for coherence, hallucination, and benchmark performance.

### 4. Visual Results Grid  (`visual_results-trial.png` or `visual_results-main.png`)

A tall figure produced by `CoherenceChecker.save_visual_grid()`.  Each row is one evaluation
prompt; for vision prompts the test image is shown on the left, the model's generated response
on the right, colour-coded green (PASS) or red (FAIL).

How to interpret:  Skim for red rows.  If most failures cluster in vision categories
(scene description, object counting, spatial reasoning, colour identification) the vision
encoder or fusion module needs more training.  If text-only rows are failing, the language
backbone is the bottleneck.

### 5. Hallucination Dashboard  (`visualizations/hallucination_dashboard.png`)

A 2x3 dashboard produced by `HallucinationVisualizer`:

- [Top-left] Category pass/fail -- Stacked horizontal bars.  Each bar is one evaluation
  category.  Green length is the number of passed tests, red length is failed.  Categories
  fully green are mastered; categories with long red segments need attention.
- [Top-centre] Text vs Vision comparison -- Two grouped bars showing hallucination rate for
  text-only prompts versus vision-language prompts.  If the vision bar is much taller the
  model hallucinates more when images are involved, pointing to weak visual grounding.
- [Top-right] Failure type pie -- Breaks down all failures into types: Hallucination (model
  invents facts), Repetition (output loops), Too Short (under 3 words), Gibberish
  (incoherent), Code Output (programming artefacts), Other.  A large Hallucination slice is
  the primary concern; Repetition or Gibberish indicate generation pathologies.
- [Bottom-left] Response length box plots -- Side-by-side boxes for passed vs failed
  responses.  If failed responses are systematically shorter the model may be under-
  generating; if longer, it may be padding with hallucinated content.
- [Bottom-centre] Category heatmap -- Single-row heatmap of per-category hallucination rate
  on a red-to-green scale.  Dark red cells are the categories where the model hallucinates
  most.
- [Bottom-right] Quality gauge -- Semi-circular gauge from 0-100 showing the overall
  coherence score.  Green zone is >= 70 %, yellow 40-70 %, red <40 %.

### 6. Paper Figures  (`paper_figures/`)

Publication-ready charts generated by `generate_stage2_5_paper_figures()`:

- `stage2_5_benchmark_bar.png` -- Bar chart of all benchmark scores (Coherence, GQA,
  UniBench, SugarCrepe, Winoground).  Each bar is annotated with its percentage.  Missing or
  zero benchmarks are excluded automatically.
- `stage2_5_category_heatmap.png` -- Annotated heatmap showing per-category pass rate from
  the coherence evaluation on a YlGnBu colour scale.
- `stage2_5_openvlm_comparison.png` (only if a baselines JSON file is provided) -- Grouped
  bar chart comparing EmberVLM against other tiny VLMs on GQA and SugarCrepe.


## Stage 3 -- Robot Selection

Output folder: `<output_dir>/stage3/visualizations/`

Produced every 500 steps and at end of training.

### 7. Confusion Matrix  (`confusion_matrix_step{N}.png`)

Two side-by-side 5x5 matrices for the five robot types (Drone, Underwater, Humanoid, Wheeled,
Legged).

- Left -- raw counts.  Cell (i, j) is "how many times did the model predict robot j when the
  true answer was robot i".
- Right -- row-normalised (each row sums to 1.0).  Diagonal cells are per-class recall.

The suptitle shows overall accuracy.

How to interpret:  Strong diagonal = correct predictions.  Bright off-diagonal cells reveal
systematic confusions (e.g. Drone confused with Legged suggests the model struggles to
distinguish aerial from ground ambulatory scenarios).  Compare early vs late steps to confirm
that confusions resolve.

### 8. Confidence Calibration  (`calibration_step{N}.png`)

Two panels built from the model's softmax confidences and correctness labels.

- Left -- reliability diagram.  Bars show accuracy in each confidence bin; the dashed
  diagonal is perfect calibration.  Bars above the line mean the model is under-confident;
  bars below mean over-confident.  ECE (Expected Calibration Error) is annotated.
- Right -- histogram of confidence values with a mean marker.

How to interpret:  A well-calibrated model has bars hugging the diagonal.  If the confidence
histogram is bunched near 1.0 but accuracy bars are below the diagonal, the model is
dangerously over-confident.  ECE < 0.05 is considered good calibration.


## Stage 4 -- Advanced Reasoning (Chain-of-Thought)

Output folder: `<output_dir>/stage4/visualizations/`

Stage 4 has two training phases (Phase 1: frozen backbone, Phase 2: joint fine-tuning).
Periodic plots are generated every 500 steps; end-of-training plots use all accumulated data.

### 9. Phase Comparison  (`phase_comparison_step{N}.png`)

A 2x2 figure comparing Phase 1 vs Phase 2 training dynamics:

- [Top-left] Loss curves for both phases on the same axes, with a vertical dashed line at
  the Phase 1 -> Phase 2 transition.
- [Top-right] Robot selection accuracy by phase.
- [Bottom-left] Consistency loss by phase.
- [Bottom-right] Grouped bar chart summarising final values of key metrics for each phase.

How to interpret:  Phase 2 should show a brief loss spike (the backbone unfreezes) followed
by a drop below Phase 1's final loss.  If Phase 2 accuracy does not exceed Phase 1, joint
fine-tuning may be overfitting or the learning rate is too aggressive.

### 10. Reasoning Quality Metrics  (`reasoning_quality_step{N}.png`)

A 2x2 dashboard derived from per-sample evaluation of generated reasoning chains:

- [Top-left] Coherence score histogram (0-1 scale).  Coherence measures whether the reasoning
  chain is logically structured and relevant to the task.
- [Top-right] Consistency score histogram (0-1 scale).  Consistency measures whether the
  reasoning chain and the final robot selection agree.
- [Bottom-left] Reasoning step count bar chart -- how many "think" steps the model produces.
- [Bottom-right] Scatter of step count vs coherence, with a linear trend line and Pearson
  correlation annotation.

How to interpret:  High coherence (mode > 0.7) and high consistency (mode > 0.8) indicate
strong reasoning.  If step counts cluster at exactly 1 the model may not be using chain-of-
thought properly.  A positive correlation between steps and coherence suggests the model
benefits from longer reasoning; a negative correlation suggests it degrades with verbosity.

Requires at least 2 evaluation samples to generate; trial runs with very few steps will
produce an empty folder.

### 11. Chain-of-Thought Comparison  (`cot_comparison_step{N}.png`)

A grid figure comparing model performance with and without chain-of-thought prompting:

- Each row is one evaluation sample.
- Left column: the task prompt and the model's direct prediction (no reasoning).
- Right column: the task prompt, the model's generated reasoning chain, and its final
  prediction.
- Cells are colour-coded green (correct) or red (incorrect).

How to interpret:  If CoT columns are greener than direct columns, chain-of-thought is
helping.  If both are roughly the same, the reasoning module may not be contributing.  If CoT
columns are redder, the model's reasoning degrades its accuracy.

Requires at least 1 evaluation sample with both with-CoT and without-CoT results.


## End-of-Pipeline: Collated Visuals

Output folder: `<output_dir>/collate_visuals/`

After all four stages and Stage 2.5 finish, `train_all.py` loads the per-stage
`training_metrics.json` files and the Stage 2.5 `evaluation_summary.json` to produce five
cross-stage summary figures.

### 12. Training Timeline  (`training_timeline.png`)

A tall 3-row figure showing the entire training run as a continuous timeline:

- Row 1 -- Loss curve.  A single continuous line across all stages, with each stage's
  region shaded in a different colour (blue=Stage 1, purple=Stage 2, orange=Stage 2.5,
  green=Stage 3, red=Stage 4).  White dashed vertical lines mark stage boundaries.  Loss
  should decrease within each stage; modest jumps at boundaries are normal due to new data.
- Row 2 -- Key accuracy metric per stage.  Plots whatever primary metric each stage tracks
  (e.g. retrieval accuracy for Stage 1, robot accuracy for Stage 3).
- Row 3 -- Learning rate schedule.  Shows the LR trajectory including warmup ramps and
  cosine decay within each stage.

How to interpret:  Look for consistent downward loss trends within each stage.  A stage
whose loss plateaus early may need more data or a higher learning rate.  Large loss jumps at a
boundary that never recover indicate catastrophic forgetting.

### 13. Capability Radar  (`capability_radar.png`)

A polar radar chart with five axes representing model capabilities:

- Vision Alignment -- derived from Stage 1 final loss.
- Language Quality -- derived from Stage 2 final loss.
- Hallucination Control -- derived from Stage 2.5 coherence score.
- Robot Selection -- derived from Stage 3 final robot accuracy.
- Reasoning Quality -- derived from Stage 4 final loss.

A progressive polygon is drawn for each stage that has completed, using a lighter fill for
earlier stages.  The polygon should grow outward as more stages are trained.

How to interpret:  A balanced pentagon is ideal.  A collapsed axis reveals the weakest
capability.  If the polygon does not grow significantly after a stage, that stage may not
have contributed enough learning.

### 14. Hallucination Journey  (`hallucination_journey.png`)

Two side-by-side panels focused on hallucination behaviour across the pipeline:

- Left panel -- Proxy indicators across stages.  Since direct hallucination measurement only
  happens at Stage 2.5, other stages use proxy metrics: loss ratio (lower is better),
  perplexity, vision failure rate, selection error, incoherence rate.  These are displayed as
  a grouped bar chart with a trend line overlaid.  A downward trend suggests the model is
  becoming less hallucination-prone.
- Right panel -- Stage 2.5 per-category coherence breakdown.  Each category is shown as a
  stacked bar (pass + fail), annotated with text (pencil emoji) or vision (eye emoji)
  markers.

How to interpret:  The left panel tracks hallucination risk longitudinally; the right panel
gives the definitive per-category snapshot.  Categories with high fail counts in the right
panel are where the model hallucinates most and should be prioritised for data augmentation
or targeted fine-tuning.

### 15. Learning Dynamics Heatmap  (`learning_dynamics_heatmap.png`)

A normalised heatmap summarising all tracked metrics across the full training run:

- Rows are metric names (loss, accuracy, learning rate, etc. from all stages).
- Columns are evenly-spaced time bins spanning the entire run.
- Cell colour uses a red-yellow-green scale (RdYlGn): green = good, red = bad.
- White vertical lines mark stage boundaries.

How to interpret:  Scan each row for colour transitions.  A row that turns green over time
shows improvement.  A row that stays red throughout is a persistent weakness.  Sudden colour
shifts at a stage boundary indicate the new stage's impact on that metric.

### 16. Achievement Summary  (`achievement_summary.png`)

A dark-themed 2x3 dashboard card suitable for presentations:

- Four cells (one per training stage) each contain a mini sparkline of loss and accuracy,
  plus a colour-coded improvement badge showing percentage improvement from start to finish of
  that stage.
- One cell shows Stage 2.5 evaluation results as a compact score card.
- One cell shows model metadata (architecture, parameter count, training config).

How to interpret:  This is intended as a one-glance summary.  Green badges indicate positive
improvement; red badges indicate regression.  The sparklines let you visually confirm the
trend.


## End-of-Pipeline: Robot Top-N Score Matrix

Output folder: `<output_dir>/figures/`

### 17. Robot Top-N Score Matrix  (`robot_topn_score_matrix.png`)

A heatmap generated by running the final model over a set of sample images.

- Rows are sample images, columns are the five robot types.
- Each cell shows the model's score for that robot, with the highest-scoring robot
  highlighted in bold white text.

How to interpret:  Scores should be clearly differentiated for each image -- one robot scores
much higher than the rest.  If scores are nearly uniform across robots the model is guessing.
The JSON companion file `robot_topn_predictions.json` has the full reasoning text.


## Directory Structure

```
<output_dir>/
  stage1/
    visualizations/
      similarity_matrix_step{N}.png
      tsne_step{N}.png
    training_metrics.json
  stage2/
    visualizations/
      token_probs_step{N}.png
    training_metrics.json
  stage2_5_evaluation/
    visual_results-trial.png  (or visual_results-main.png)
    evaluation_summary.json
    visualizations/
      hallucination_dashboard.png
    paper_figures/
      stage2_5_benchmark_bar.png
      stage2_5_category_heatmap.png
      stage2_5_openvlm_comparison.png   (conditional)
  stage3/
    visualizations/
      confusion_matrix_step{N}.png
      calibration_step{N}.png
    training_metrics.json
  stage4/
    visualizations/
      phase_comparison_step{N}.png
      reasoning_quality_step{N}.png
      cot_comparison_step{N}.png
    training_metrics.json
  collate_visuals/
    training_timeline.png
    capability_radar.png
    hallucination_journey.png
    learning_dynamics_heatmap.png
    achievement_summary.png
  figures/
    robot_topn_score_matrix.png
    robot_topn_predictions.json
```


## Dependencies

All visualizations use matplotlib and numpy.  Stage 1 t-SNE requires scikit-learn.  No
other visualisation-specific libraries are needed (seaborn, plotly, etc. are not required).

```
matplotlib >= 3.7.0
numpy >= 1.24.0
scikit-learn >= 1.3.0   (for t-SNE only)
pillow >= 9.5.0
```

