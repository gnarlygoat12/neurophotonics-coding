import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

data_dir = Path("/projectnb2/npcr25/projects/two_photon/Ex1_jRGECO1a_ResonantScanning/processed/TSeries-03042024-run02-054")
out_dir = Path(__file__).parent

F = np.load(data_dir / "F.npy")
Fneu = np.load(data_dir / "Fneu.npy")
iscell = np.load(data_dir / "iscell.npy")

good_cells = iscell[:, 0] == 1
F_good = F[good_cells, :]
Fneu_good = Fneu[good_cells, :]
n_good = F_good.shape[0]

ALPHA = 0.7
F_corrected = F_good - ALPHA * Fneu_good


def per_cell_corr(a, b):
    corrs = np.empty(a.shape[0])
    for i in range(a.shape[0]):
        corrs[i] = np.corrcoef(a[i], b[i])[0, 1]
    return corrs


corr_before = per_cell_corr(F_good, Fneu_good)
corr_after = per_cell_corr(F_corrected, Fneu_good)

mean_before = corr_before.mean()
mean_after = corr_after.mean()
pct_reduction = 100 * (abs(mean_before) - abs(mean_after)) / abs(mean_before)

print(f"Mean per-cell correlation (F vs Fneu):")
print(f"  Before correction: {mean_before:.2f}")
print(f"  After correction:  {mean_after:.2f}")
print(f"  Reduction: {pct_reduction:.1f}%")

# --- Plot 1: example cells, raw vs corrected traces ---
example_idx = [0, n_good // 2, n_good - 1]


def zscore(x):
    return (x - x.mean()) / x.std()


n_plot_frames = 100 * 15  # 100 s window at 15 Hz, for readability

fig, axes = plt.subplots(len(example_idx), 2, figsize=(12, 8), sharex=True)
for row, idx in enumerate(example_idx):
    t = np.arange(n_plot_frames) / 15

    ax = axes[row, 0]
    ax.plot(t, zscore(F_good[idx, :n_plot_frames]), color="tab:blue", label="F (raw)")
    ax.plot(t, zscore(Fneu_good[idx, :n_plot_frames]), color="tab:red", linestyle="--", label="Fneu")
    ax.set_ylabel(f"cell {idx}\n(z-scored)")
    if row == 0:
        ax.set_title("Raw F vs Fneu")
        ax.legend(fontsize=8)

    ax = axes[row, 1]
    ax.plot(t, zscore(F_corrected[idx, :n_plot_frames]), color="tab:green", label="F (corrected)")
    ax.plot(t, zscore(Fneu_good[idx, :n_plot_frames]), color="tab:red", linestyle="--", label="Fneu")
    if row == 0:
        ax.set_title("Corrected F vs Fneu")
        ax.legend(fontsize=8)

axes[-1, 0].set_xlabel("Time (s)")
axes[-1, 1].set_xlabel("Time (s)")
fig.tight_layout()
fig.savefig(out_dir / "exercise1_neuropil_correction.png", dpi=150)

# --- Plot 2: per-cell correlation before vs after ---
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

order = np.argsort(corr_before)
ax = axes[0]
ax.plot(corr_before[order], "o-", color="tab:blue", markersize=3, label="before")
ax.plot(corr_after[order], "o-", color="tab:green", markersize=3, label="after")
ax.set_xlabel("Cell (sorted by raw correlation)")
ax.set_ylabel("Correlation (F vs Fneu)")
ax.legend()
ax.set_title("Per-cell correlation, before vs after")

ax = axes[1]
lims = [min(corr_before.min(), corr_after.min()) - 0.05, corr_before.max() + 0.05]
ax.plot(lims, lims, "k--", linewidth=1)
ax.scatter(corr_before, corr_after, s=10, color="tab:purple")
ax.set_xlabel("Correlation before")
ax.set_ylabel("Correlation after")
ax.set_xlim(lims)
ax.set_ylim(lims)
ax.set_title("Before vs after (each point = one cell)")

fig.tight_layout()
fig.savefig(out_dir / "exercise1_per_cell_correlation.png", dpi=150)

print("Saved plots to tutorials/friday_exercises/")
