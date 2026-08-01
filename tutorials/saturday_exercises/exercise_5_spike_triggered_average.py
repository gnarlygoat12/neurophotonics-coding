import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from pathlib import Path

data_dir = Path("/projectnb2/npcr25/projects/two_photon/Ex1_jRGECO1a_ResonantScanning/processed/TSeries-03042024-run02-054")
spks_dir = Path("/projectnb2/npcr25/projects/two_photon/Ex1_jRGECO1a_ResonantScanning/Suite2P-inferred-spikes/TSeries-03042024-run02-054")
out_dir = Path(__file__).parent

FS = 15
ALPHA = 0.7
REFRACTORY_S = 0.15  # matches Exercise 2's peak-finding convention
PEAK_HEIGHT_FRAC = 0.15
PRE_S, POST_S = 2.0, 2.0  # window around each trigger

F = np.load(data_dir / "F.npy")
Fneu = np.load(data_dir / "Fneu.npy")
iscell = np.load(data_dir / "iscell.npy")
spks_real = np.load(spks_dir / "spks.npy")  # the real Suite2p inference, not the zeroed copy next to F.npy

good_idx = np.where(iscell[:, 0] == 1)[0]
F_good = (F - ALPHA * Fneu)[good_idx, :]
spks_good = spks_real[good_idx, :]
n_good, n_frames = F_good.shape


def zscore(x, axis=-1):
    return (x - x.mean(axis=axis, keepdims=True)) / x.std(axis=axis, keepdims=True)


F_z = zscore(F_good, axis=1)


def get_peaks(trace):
    peaks, _ = find_peaks(trace, height=trace.max() * PEAK_HEIGHT_FRAC, distance=int(REFRACTORY_S * FS))
    return peaks


all_peaks = [get_peaks(spks_good[i]) for i in range(n_good)]
n_peaks = np.array([len(p) for p in all_peaks])

ref_local = int(np.argmax(n_peaks))  # most active good cell -> reference
ref_cell_id = good_idx[ref_local]
trigger_frames = all_peaks[ref_local]

pre_frames = int(PRE_S * FS)
post_frames = int(POST_S * FS)
valid_triggers = trigger_frames[(trigger_frames >= pre_frames) & (trigger_frames < n_frames - post_frames)]

print(f"Reference cell: suite2p index {ref_cell_id} (local good-cell idx {ref_local})")
print(f"Valid trigger events: {len(valid_triggers)}")

window_len = pre_frames + post_frames + 1
lags = (np.arange(window_len) - pre_frames) / FS

sta = np.zeros((n_good, window_len))
sem = np.zeros((n_good, window_len))
for i in range(n_good):
    windows = np.stack([F_z[i, t - pre_frames:t + post_frames + 1] for t in valid_triggers])
    sta[i] = windows.mean(axis=0)
    sem[i] = windows.std(axis=0, ddof=1) / np.sqrt(windows.shape[0])

pre_peak = np.max(np.abs(sta[:, :pre_frames]), axis=1)
post_peak = np.max(np.abs(sta[:, pre_frames:]), axis=1)

others = np.delete(np.arange(n_good), ref_local)
top_others = others[np.argsort(post_peak[others])[::-1][:4]]
n_exceed = int(np.sum(post_peak[others] > pre_peak[others]))

print(f"Reference cell's own peak response: {post_peak[ref_local]:.2f}")
print(f"Best other cells' peak response: {np.round(post_peak[top_others], 2)}")
print(f"{n_exceed}/{len(others)} other cells have post-trigger peak > their own pre-trigger peak")

# --- Plot 1: population overview ---
fig = plt.figure(figsize=(14, 10))
gs = fig.add_gridspec(2, 2)

ax0 = fig.add_subplot(gs[0, 0])
order = np.argsort(post_peak)
im = ax0.imshow(sta[order], aspect="auto", extent=[lags[0], lags[-1], 0, n_good],
                cmap="RdBu_r", vmin=-1, vmax=1, origin="lower")
ax0.set_xlabel("Lag (s)")
ax0.set_ylabel("Cell (sorted by peak response)")
ax0.set_title("Every good cell's triggered average")
fig.colorbar(im, ax=ax0, label="z-scored F")

ax1 = fig.add_subplot(gs[0, 1])
ax1.plot(lags, sta[ref_local], color="black", linewidth=2, label=f"reference (cell {ref_cell_id})")
for ci in top_others:
    ax1.plot(lags, sta[ci], linewidth=1, label=f"cell {good_idx[ci]}")
ax1.axvline(0, color="gray", linestyle=":", linewidth=1)
ax1.set_xlabel("Lag (s)")
ax1.set_ylabel("z-scored F")
ax1.set_title("Reference vs. best 'other' cells")
ax1.legend(fontsize=8)

ax2 = fig.add_subplot(gs[1, 0])
ax2.hist(post_peak[others], bins=20, color="tab:blue", alpha=0.7)
ax2.axvline(post_peak[ref_local], color="red", linestyle="--", label="reference cell")
ax2.set_xlabel("Peak triggered-average response")
ax2.set_ylabel("Number of cells")
ax2.set_title("Distribution of peak response across population")
ax2.legend(fontsize=8)

ax3 = fig.add_subplot(gs[1, 1])
ax3.scatter(pre_peak[others], post_peak[others], s=10, color="tab:blue", label="other cells")
ax3.scatter(pre_peak[ref_local], post_peak[ref_local], s=60, color="red", label="reference cell")
lims = [0, max(pre_peak.max(), post_peak.max()) * 1.05]
ax3.plot(lims, lims, "k--", linewidth=1)
ax3.set_xlabel("Pre-trigger peak")
ax3.set_ylabel("Post-trigger peak")
ax3.set_xlim(lims)
ax3.set_ylim(lims)
ax3.set_title("Post- vs. pre-trigger peak (per cell)")
ax3.legend(fontsize=8)

fig.tight_layout()
fig.savefig(out_dir / "exercise5_triggered_average.png", dpi=150)

# --- Plot 2: convergence with increasing number of events ---
event_counts = [n for n in [10, 30, 100, len(valid_triggers)] if n <= len(valid_triggers)]
fig, ax = plt.subplots(figsize=(8, 5))
for n in event_counts:
    windows = np.stack([F_z[ref_local, t - pre_frames:t + post_frames + 1] for t in valid_triggers[:n]])
    ax.plot(lags, windows.mean(axis=0), label=f"n={n}")
ax.axvline(0, color="gray", linestyle=":", linewidth=1)
ax.set_xlabel("Lag (s)")
ax.set_ylabel("z-scored F")
ax.set_title(f"Reference cell (cell {ref_cell_id}) STA convergence with more events")
ax.legend()
fig.tight_layout()
fig.savefig(out_dir / "exercise5_convergence.png", dpi=150)

print("Saved plots to tutorials/saturday_exercises/")
