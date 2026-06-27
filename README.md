# Eikonal Corner-Sharpness Control Dataset for FNO Discretization Invariance

Control dataset isolating **one variable**: the reflex-corner angle θ of a single wedge/notch obstacle in a 2D eikonal problem. **Hypothesis tested:** plain FNO2d maintains discretization invariance as θ sharpens (gradient kinks worsen), or error grows systematically with kink severity at zero-shot (θ, resolution) test cells.

## PDE

\[
|\nabla V(x)| = c(x), \quad x \in \Omega \setminus \mathcal{G}, \qquad V(x) = 0,\; x \in \mathcal{G}
\]

Single-speed eikonal with \(c(x) = 1\) in free space. Obstacles are excluded regions (infinite cost). Ground truth is computed with [`scikit-fmm`](https://pythonhosted.org/scikit-fmm/) (`skfmm.travel_time`).

## Dataset structure

| Split | Resolution | Organization |
|-------|------------|--------------|
| Train | 64×64 only | All 7 θ values mixed (~900 samples/θ → ~6300 total; 15% stratified val split) |
| Test (zero-shot) | 64, 128, 256, 512 | `data/test/theta_{θ}/res_{R}/samples.npz`, 50 samples/cell |

Test geometries use **new** random position/rotation/scale (never reused from training).

θ sweep: **{180, 150, 120, 90, 60, 30, 10}** degrees (180° = flat wall control).

## Quick start

```bash
cd control_dataset
pip install -r requirements.txt

# 1. Validate kink severity (run before training)
python validate_kink_severity.py

# 2. Generate dataset (full: ~30 min–2 hr depending on hardware)
python generate_dataset.py

# 2b. Pre-training verification (resolution scaling, θ balance, train/test separation)
python verify_dataset.py

# Quick smoke test (small data):
python generate_dataset.py --quick

# 3. Train FNO once on combined 64×64 training set
python train.py

# 4. Evaluate zero-shot on all (θ, resolution) buckets
python evaluate.py
```

## Files

| File | Purpose |
|------|---------|
| `geometry.py` | Wedge polygon construction, rasterization, FMM solver |
| `generate_dataset.py` | Dataset generation with train/test splits |
| `verify_dataset.py` | Pre-training checks (resolution, θ scale, train/test separation) |
| `kink_utils.py` | Near-kink region masks for kink-weighted loss |
| `validate_kink_severity.py` | Sanity check: gradient jump vs θ |
| `model_fno.py` | Standard FNO2d (Li et al. 2021 style) |
| `train.py` | Training loop, kink-weighted relative L2 loss |
| `evaluate.py` | (θ, resolution) error grid + core plot + qualitative 512² examples |

## Outputs

- `data/manifest.json` — dataset metadata
- `results/validation/kink_severity_vs_theta.png` — control-variable validation
- `results/evaluation/error_vs_kink_severity.png` — **core deliverable figure**
- `results/evaluation/qualitative_512/` — predicted vs GT heatmaps at 512²

## Model

Plain **FNO2d** only (no PNO/DAFNO/obstacle encoding). Input: binary occupancy + (x, y) grid channels. Output: value function V̂(x). Trained **once** on all θ mixed at 64×64.

Loss: relative L2, \(\| \hat V - V \|_2 / \| V \|_2\).

## Reproducibility

- Fixed seed (`--seed 42`) for dataset generation
- Each sample stores full polygon vertices, θ, pose, scale, and goal
- Occupancy and FMM solutions are regenerated independently at each resolution (no upsampling)
- RTX A2000 4 GB

## Citation

Li et al., "Fourier Neural Operator for Parametric Partial Differential Equations", ICLR 2021.

## Key findings

Reported model: `checkpoints/fno_best_val_kink.pt` (λ_kink=100, best validation kink loss at epoch 8).

- **Whole-domain relative L2** is flat (~0.48–0.50) across θ and test resolution (64–512); this metric is insensitive to the near-kink region (<1% of free-space pixels).
- **Near-kink relative L2** jumps once any kink exists (θ<180°) to ~0.40–0.52 and does not track corner sharpness within that band.
- **Lambda sweep (100, 175, 350, 700):** best validation kink loss bottoms at epochs 7–8 (~0.447–0.449) for all weights; final val_kink ~0.57–0.59 at epoch 200 — plateau is weight-independent.
- **Extended run (500 max epochs, LR×0.97/10 epochs, early stop patience 40):** early stopping at epoch 48; best val_kink still 0.4468 at epoch 8; no later improvement below epoch-20 val_kink. Free-space prediction std remains ~24–31% of ground-truth std (modest gain vs baseline ~16–22%, still far from full amplitude).

Full tables: `results/evaluation/summary.txt`, `results/evaluation/error_grid.csv`.
