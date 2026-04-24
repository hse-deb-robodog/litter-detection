# AutoResearcher — Litter Segmentation

## Mission

You are an autonomous ML research agent. Your goal is to **maximise `val_iou`**
(intersection-over-union on the validation set) for a pixel-wise litter
segmentation model trained on the TACO dataset.

Every experiment runs for a fixed **TIME_LIMIT** seconds so results are
directly comparable. After each run, examine what worked, form a hypothesis,
modify `auto-research/train.py`, and run the next experiment.

---

## Rules

1. **Only edit `auto-research/train.py`.** Never modify
   `auto-research/prepare.py`, `auto-research/program.md`, or
   `auto-research/analysis.ipynb`.
2. **Do not change `TIME_LIMIT`** (default 1200 s / 20 min per run) unless the
   human instructs you to. Consistent time budgets make experiments comparable.
3. Every experiment must be a distinct MLflow run with a descriptive
   `--run-name` that captures what changed (e.g. `deeper-encoder`,
   `focal-loss`, `resnet34-backbone`).
4. Always read the latest `val_iou` from the MLflow run before deciding on the
   next change.
5. One change at a time — isolate variables so you know what caused improvement.

---

## Setup (first time only)

```bash
# Install dependencies
uv sync

# Download and preprocess the TACO dataset (~10 min, one-time)
uv run python auto-research/prepare.py

# Launch MLflow UI (optional, for human review)
uv run mlflow ui
```

---

## Running an experiment

```bash
uv run python auto-research/train.py --run-name <descriptive-name> [--time-limit SECONDS]
```

The script prints per-epoch metrics and logs everything to MLflow. The best
checkpoint is saved under `models/checkpoints/best_model_<model>.pth`.

---

## What to optimise in `auto-research/train.py`

Everything between the `# ── Hyperparameters` and bottom of the file is yours
to change. Ideas in rough priority order:

### Architecture
- **Encoder depth**: add/remove stages in `ENCODER_CHANNELS`
- **Backbone**: replace the custom encoder with a pretrained ResNet/EfficientNet
  using `torchvision.models` feature extractors (freeze early layers)
- **Attention**: add CBAM, SE blocks, or a lightweight transformer decoder
- **Decoder**: try FPN-style multi-scale fusion instead of plain U-Net skip
  connections

### Loss
- Current: BCE + Dice (equal weight)
- Try: Focal loss (helps extreme class imbalance), Lovász-Softmax,
  weighted combos of the above

### Optimiser & schedule
- Current: AdamW + OneCycleLR
- Try: SGD + cosine annealing, Lion optimiser

### Regularisation
- Dropout rate, stochastic depth, mixup on images/masks

### Data augmentation
- Current: RandomResizedCrop, flips, ColorJitter, GaussNoise
- Try: GridDistortion, ElasticTransform, CutMix, copy-paste augmentation

### Batch & resolution
- `BATCH_SIZE`, `CROP_SIZE` — larger crops give more context but fewer
  gradient updates per second

---

## Metric interpretation

| `val_iou` range | Interpretation                            |
|-----------------|-------------------------------------------|
| < 0.20          | Model barely segments anything            |
| 0.20 – 0.45     | Learning something, room for improvement  |
| 0.45 – 0.65     | Solid baseline                            |
| > 0.65          | Strong result for this dataset/time budget|

---

## Agent loop

```
1. Read the last run's val_iou from MLflow (or stdout)
2. Hypothesise one change likely to improve it
3. Edit `auto-research/train.py` — one logical change
4. Run: `uv run python auto-research/train.py --run-name <name>`
5. Compare val_iou with previous best
6. If improved: keep change, go to step 2
   If worse:     revert change, go to step 2 with different hypothesis
7. After 8–12 experiments, write a short summary of findings to `auto-research/findings.md`
```

---

## Current best

> Update this section after each experiment.

| Run name | val_iou | Notes |
|----------|---------|-------|
| baseline | 0.1845  | U-Net [32,64,128,256], BCE+Dice, AdamW+OneCycleLR, 17 epochs in 20 min |
| deeper-encoder-64-128-256-512 | 0.2698 | U-Net [64,128,256,512], BCE+Dice, AdamW+OneCycleLR, 25M params, 6 epochs in 20 min |
| focal-dice-loss | 0.2243 | Focal(gamma=2)+Dice loss, reverted — worse |
| lower-lr-1e-4 | 0.1790 | LR=1e-4 too slow for time budget, reverted |
| larger-crop-448 | 0.2127 | CROP=448, batch=6 — fewer steps hurts, reverted |
| se-attention-blocks | 0.2339 | SE attention on each ConvBlock — slower per epoch, reverted |
| enhanced-augmentation-grid-elastic | 0.2531 | GridDistortion+ElasticTransform added — close but below best, reverted |
| resnet34-pretrained-backbone | 0.5119 | ResNet34 ImageNet pretrained encoder, 23 epochs — massive +89% improvement |
| resnet34-enhanced-augmentation | 0.5465 | + GridDistortion+ElasticTransform augmentation — further improvement |
| resnet34-aug-pct-start-0p15 | 0.5347 | pct_start=0.15 (was 0.05) — worse, reverted |
| resnet34-aug-tversky-loss | 0.5362 | Tversky loss (alpha=0.3 beta=0.7) — worse than BCE+Dice, reverted |
| resnet34-aug-lr-5e-4 | 0.5872 | LR=5e-4 (was 3e-4) — improved |
| resnet34-aug-lr-8e-4 | 0.5936 | LR=8e-4 — marginal further improvement |
| resnet34-label-smoothing-0p05 | 0.5951 | Label smoothing 0.05 on BCE — slight improvement |
| resnet34-label-smooth-0p02 | 0.6016 | Label smoothing 0.02 (reduced) — new best |
| resnet34-label-smooth-0p01 | 0.6125 | Label smoothing 0.01 — even better |
| resnet34-smooth01-no-vflip | 0.6241 | Remove VerticalFlip augmentation — new best |
| efficientnetb4-backbone | 0.5596 | EfficientNet-B4, 30 min budget, 15 epochs — ~114s/epoch vs 52s for ResNet34, still converging at end, reverted |
| resnet34-90min | **0.6738** | ResNet34UNet, 90 min budget, 108 epochs — best ResNet34 result |
| efficientnetb4-90min | 0.6355 | EfficientNet-B4, 90 min budget, 49 epochs — ~110s/epoch, slower convergence |
| b4-crop-320 | 0.5796 | EfficientNet-B4, crop=320, LR=8e-4, 33 epochs, 45 min |
| b4-crop320-lr-6e-4 | 0.5501 | EfficientNet-B4, crop=320, LR=6e-4, 33 epochs, 45 min — worse |
| b4-crop-288 | 0.5807 | EfficientNet-B4, crop=288, LR=8e-4, 38 epochs, 45 min |
| b4-crop320-pct-0p1 | 0.5595 | EfficientNet-B4, crop=320, pct_start=0.1, 33 epochs, 45 min — worse |
| b4-frozen-backbone-5ep | 0.5575 | EfficientNet-B4, crop=320, frozen backbone 5ep then unfreeze, 36 epochs, 45 min — worse |
| b4-crop320-lr-1e-3 | **0.6165** | EfficientNet-B4, crop=320, LR=1e-3, 33 epochs, 45 min — best B4 in loop 4 |
| b4l2-lr-1p2e-3 | 0.6029 | LR=1.2e-3 — slightly too high, worse |
| b4l2-lr-1p5e-3 | 0.6166 | LR=1.5e-3 — matches baseline, not a clear win |
| b4l2-smooth-0p005 | **0.6323** | label_smooth=0.005 (halved) — new best B4 at 45 min |
| b4l2-no-gaussnoise | 0.6078 | no GaussNoise — worse, reverted |
| b4l2-lighter-aug | 0.5939 | no GaussNoise + no GridDistortion — worse, reverted |
| b4l2-crop-352 | 0.5650 | CROP=352 — fewer epochs, worse, reverted |
