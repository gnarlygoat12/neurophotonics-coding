import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

data_dir = Path("/Users/kluglab/Downloads/TSeries-03042024-run02-054")

F = np.load(data_dir / "F.npy")             # (125, 4535) - raw fluorescence
Fneu = np.load(data_dir / "Fneu.npy")       # (125, 4535) - neuropil fluorescence
iscell = np.load(data_dir / "iscell.npy")   # (125, 2) - cell quality scores
stat = np.load(data_dir / "stat.npy", allow_pickle=True)  # (125,) - ROI locations
ops = np.load(data_dir / "ops.npy", allow_pickle=True).item()  # dict - run config + meanImg

good_cells = iscell[:, 0] == 1
F_good = F[good_cells, :]

print(f"Total cells: {F.shape[0]}")
print(f"Good cells: {F_good.shape[0]} ({100 * F_good.shape[0] / F.shape[0]:.1f}%)")
print(f"Frames: {F.shape[1]} ({F.shape[1] / 15 / 60:.1f} minutes at 15 Hz)")

# Raw fluorescence traces for a few cells, first 100 seconds
fs = 15  # Hz
n_frames_100s = int(100 * fs)
good_idx = np.where(good_cells)[0][:5]

fig, ax = plt.subplots(figsize=(10, 5))
t = np.arange(n_frames_100s) / fs
for offset, idx in enumerate(good_idx):
    trace = F[idx, :n_frames_100s]
    ax.plot(t, trace - trace.min() + offset * 2000, label=f"cell {idx}")
ax.set_xlabel("Time (s)")
ax.set_ylabel("Fluorescence (offset for display)")
ax.set_title("Raw fluorescence traces (first 100 s)")
ax.legend()
fig.tight_layout()
fig.savefig(Path(__file__).parent / "step0_raw_traces.png", dpi=150)

# Mean image with detected cell locations
fig, ax = plt.subplots(figsize=(7, 7))
ax.imshow(ops["meanImg"], cmap="gray")
for cell_idx in np.where(good_cells)[0]:
    y, x = stat[cell_idx]["med"]
    ax.plot(x, y, "c.", markersize=4)
ax.set_title("Mean image with detected cells (cyan)")
ax.axis("off")
fig.tight_layout()
fig.savefig(Path(__file__).parent / "step0_mean_image_with_rois.png", dpi=150)

print("Saved plots to tutorials/friday_exercises/")
