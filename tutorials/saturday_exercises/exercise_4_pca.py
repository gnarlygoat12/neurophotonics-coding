import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from pathlib import Path

data_dir = Path("/projectnb2/npcr25/projects/two_photon/Ex1_jRGECO1a_ResonantScanning/processed/TSeries-03042024-run02-054")
out_dir = Path(__file__).parent

FS = 15
ALPHA = 0.7
N_PLOT = 50  # how many components to show in the scree/cumulative plots

F = np.load(data_dir / "F.npy")
Fneu = np.load(data_dir / "Fneu.npy")
iscell = np.load(data_dir / "iscell.npy")

good_cells = iscell[:, 0] == 1
F_good = F[good_cells, :]
Fneu_good = Fneu[good_cells, :]
n_good, n_frames = F_good.shape

F_corrected = F_good - ALPHA * Fneu_good


def zscore(x, axis=-1):
    return (x - x.mean(axis=axis, keepdims=True)) / x.std(axis=axis, keepdims=True)


F_z = zscore(F_corrected, axis=1)  # per-cell z-score across time

# Fit on (frames x cells) -- one data point per frame, one feature per cell.
pca = PCA(n_components=n_good)
pca_scores = pca.fit_transform(F_z.T)  # (n_frames, n_good)
explained = pca.explained_variance_ratio_
cumulative = np.cumsum(explained)

n_80 = int(np.searchsorted(cumulative, 0.80) + 1)
n_90 = int(np.searchsorted(cumulative, 0.90) + 1)

print(f"PC1 alone explains {explained[0] * 100:.1f}% of total variance")
print(f"PC1+PC2+PC3: {explained[:3].sum() * 100:.1f}%")
print(f"Top 10 PCs combined: {explained[:10].sum() * 100:.1f}%")
print(f"Components needed for 80% cumulative variance: {n_80}")
print(f"Components needed for 90% cumulative variance: {n_90}")

pc1_loadings = pca.components_[0]
top5_idx = np.argsort(np.abs(pc1_loadings))[::-1][:5]
low5_idx = np.argsort(np.abs(pc1_loadings))[:5]

# --- Plot 1: scree, cumulative variance, PC1-vs-PC2 trajectory ---
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

axes[0].plot(np.arange(1, N_PLOT + 1), explained[:N_PLOT] * 100, "o-", color="tab:blue", markersize=3)
axes[0].set_xlabel("Principal component")
axes[0].set_ylabel("Variance explained (%)")
axes[0].set_title("Scree plot")

axes[1].plot(np.arange(1, N_PLOT + 1), cumulative[:N_PLOT] * 100, "o-", color="tab:green", markersize=3)
axes[1].axhline(80, color="gray", linestyle="--", linewidth=1, label="80%")
axes[1].axhline(90, color="gray", linestyle=":", linewidth=1, label="90%")
axes[1].set_xlabel("Number of components")
axes[1].set_ylabel("Cumulative variance (%)")
axes[1].set_title(f"Cumulative variance (first {N_PLOT} PCs)")
axes[1].legend(fontsize=8)

sc = axes[2].scatter(pca_scores[:, 0], pca_scores[:, 1], c=np.arange(n_frames), cmap="viridis", s=3)
axes[2].set_xlabel("PC1 score")
axes[2].set_ylabel("PC2 score")
axes[2].set_title("Population trajectory (PC1 vs PC2)")
fig.colorbar(sc, ax=axes[2], label="frame")

fig.tight_layout()
fig.savefig(out_dir / "exercise4_pca_variance.png", dpi=150)

# --- Plot 2: top-5 |PC1 loading| cells' raw traces vs PC1 score over time ---
t = np.arange(n_frames) / FS
fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True, gridspec_kw={"height_ratios": [2, 1]})

offset = 0
for idx in top5_idx:
    axes[0].plot(t, F_z[idx] + offset, linewidth=0.8, label=f"cell {idx} (loading={pc1_loadings[idx]:.2f})")
    offset += 6
axes[0].set_ylabel("z-scored F (offset)")
axes[0].set_title("Top 5 |PC1 loading| cells")
axes[0].legend(fontsize=7, loc="upper right")

axes[1].plot(t, pca_scores[:, 0], color="black", linewidth=0.8)
axes[1].set_xlabel("Time (s)")
axes[1].set_ylabel("PC1 score")

fig.tight_layout()
fig.savefig(out_dir / "exercise4_pca_pc1_vs_cells.png", dpi=150)

# --- Plot 3: contrast with low-|loading| cells, per the README's suggested check ---
fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True, gridspec_kw={"height_ratios": [2, 1]})
offset = 0
for idx in low5_idx:
    axes[0].plot(t, F_z[idx] + offset, linewidth=0.8, label=f"cell {idx} (loading={pc1_loadings[idx]:.2f})")
    offset += 6
axes[0].set_ylabel("z-scored F (offset)")
axes[0].set_title("Bottom 5 |PC1 loading| cells (near-zero loading)")
axes[0].legend(fontsize=7, loc="upper right")

axes[1].plot(t, pca_scores[:, 0], color="black", linewidth=0.8)
axes[1].set_xlabel("Time (s)")
axes[1].set_ylabel("PC1 score")

fig.tight_layout()
fig.savefig(out_dir / "exercise4_pca_pc1_vs_lowloading_cells.png", dpi=150)

print("Saved plots to tutorials/saturday_exercises/")
