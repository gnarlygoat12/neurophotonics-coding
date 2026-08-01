import warnings
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import convolution_matrix
from sklearn.linear_model import Lasso
from pathlib import Path

warnings.filterwarnings("ignore")  # NaN correlation for the rare all-zero fit is handled explicitly below

data_dir = Path("/Users/kluglab/Downloads/TSeries-03042024-run02-054")
out_dir = Path(__file__).parent

FS = 15
N_FRAMES = 2000  # ~2.2 min subset, matches README's tractability note
NEUCOEFF = 0.7
ALPHA = 1.0  # much larger than the synthetic-data alpha; real F is on a bigger absolute scale
N_CELLS = 30

F = np.load(data_dir / "F.npy")
Fneu = np.load(data_dir / "Fneu.npy")
iscell = np.load(data_dir / "iscell.npy")
spks_real = np.load("/Users/kluglab/Downloads/spks-2.npy")  # the *real* Suite2p inference, not the zeroed copy in data_dir

good_idx = np.where(iscell[:, 0] == 1)[0][:N_CELLS]
Fc = (F - NEUCOEFF * Fneu)[good_idx, :N_FRAMES]
spks_sub = spks_real[good_idx, :N_FRAMES]


def autocovariance(x, maxlag):
    x = x - x.mean()
    return np.array([np.mean(x[: len(x) - lag] * x[lag:]) for lag in range(1, maxlag + 1)])


def estimate_tau(x, maxlag=10):
    acov = autocovariance(x, maxlag)
    mask = acov > 0
    lags = np.arange(1, maxlag + 1)[mask]
    vals = acov[mask]
    if len(lags) < 3:
        return np.nan
    slope, _ = np.polyfit(lags, np.log(vals), 1)
    tau = -1 / (FS * slope)
    return float(np.clip(tau, 0.05, 3.0))  # clip against occasional noisy single-cell fits


taus = np.array([estimate_tau(Fc[i]) for i in range(N_CELLS)])
print(f"Per-cell tau: min={taus.min():.2f}s max={taus.max():.2f}s median={np.median(taus):.2f}s")

s_hats = np.zeros_like(Fc)
correlations = np.full(N_CELLS, np.nan)
for i in range(N_CELLS):
    tau = taus[i]
    n_kernel = int(5 * tau * FS) + 1
    h = np.exp(-np.arange(n_kernel) / FS / tau)
    H = convolution_matrix(h, N_FRAMES, mode="full")[:N_FRAMES, :]

    trace = Fc[i]
    target = trace - trace.min()
    lasso = Lasso(alpha=ALPHA, positive=True, fit_intercept=False, max_iter=5000)
    lasso.fit(H, target)
    s_hats[i] = lasso.coef_

    if lasso.coef_.std() > 0:
        correlations[i] = np.corrcoef(lasso.coef_, spks_sub[i])[0, 1]

valid = ~np.isnan(correlations)
print(f"Correlation with Suite2p spks: mean={correlations[valid].mean():.2f} "
      f"median={np.median(correlations[valid]):.2f} "
      f"n>0.5={np.sum(correlations[valid] > 0.5)}/{N_CELLS}")

# --- Plot 1: tau distribution + example autocovariance curves ---
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
axes[0].hist(taus, bins=12, color="tab:blue", alpha=0.7)
axes[0].axvline(1.0, color="red", linestyle="--", label="Suite2p fixed τ=1.0s")
axes[0].set_xlabel("Estimated τ (s)")
axes[0].set_ylabel("Number of cells")
axes[0].set_title("Per-cell τ distribution")
axes[0].legend()

example_cells = [np.argmin(taus), np.argsort(taus)[len(taus) // 2], np.argmax(taus)]
for ci in example_cells:
    acov = autocovariance(Fc[ci], 10)
    lags = np.arange(1, 11) / FS
    axes[1].plot(lags, acov / acov[0], "o-", label=f"cell {ci} (τ={taus[ci]:.2f}s)")
axes[1].set_xlabel("Lag (s)")
axes[1].set_ylabel("Normalized autocovariance")
axes[1].set_title("Example autocovariance decay")
axes[1].legend(fontsize=8)
fig.tight_layout()
fig.savefig(out_dir / "exercise2_kernel_estimation.png", dpi=150)

# --- Plot 2: low/medium/high activity example comparisons ---
activity = spks_sub.sum(axis=1)
order = np.argsort(activity)
low_idx, med_idx, high_idx = order[len(order) // 6], order[len(order) // 2], order[-len(order) // 6]

t_axis = np.arange(N_FRAMES) / FS


def zscore(x):
    return (x - x.mean()) / x.std()


for label, ci in [("low", low_idx), ("medium", med_idx), ("high", high_idx)]:
    fig, axes = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
    axes[0].plot(t_axis, Fc[ci], color="tab:blue")
    axes[0].set_ylabel("Corrected F")
    axes[0].set_title(f"Cell {good_idx[ci]} ({label} activity), corr r={correlations[ci]:.2f}")

    axes[1].plot(t_axis, zscore(spks_sub[ci]), color="tab:red", label="Suite2p (OASIS)")
    axes[1].plot(t_axis, zscore(s_hats[ci]), color="tab:green", label="This exercise (Lasso)")
    axes[1].set_ylabel("z-scored\nspike amplitude")
    axes[1].set_xlabel("Time (s)")
    axes[1].legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(out_dir / f"exercise2_real_comparison_{label}.png", dpi=150)

print("Saved plots to tutorials/friday_exercises/")
