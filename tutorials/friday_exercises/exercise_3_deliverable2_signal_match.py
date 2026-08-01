import time
import numpy as np
import matplotlib.pyplot as plt
import tifffile
from scipy import ndimage
from pathlib import Path

data_dir = Path("/Users/kluglab/Downloads/TSeries-03042024-run02-054")
out_dir = Path(__file__).parent
movie_path = data_dir / "TSeries-03042024-run02-054_Ch2_registered.tif"

F = np.load(data_dir / "F.npy")
state = np.load(out_dir / "exercise3_deliverable1_state.npz")
tuned_label_ids = state["tuned_label_ids"]
match_pairs = state["match_pairs"]  # (di, ti) pairs: detection index -> Suite2p ROI index
labeled = state["labeled"]

n_matches = len(match_pairs)
print(f"Matched pairs from Deliverable 1: {n_matches}")

matched_labels = tuned_label_ids[match_pairs[:, 0]]  # connected-component label id per match
suite2p_idx = match_pairs[:, 1]  # index into F / stat for the matched Suite2p ROI

# Pixel counts per matched region, for computing per-frame means from per-frame sums
region_npix = ndimage.sum(np.ones_like(labeled), labeled, index=matched_labels)

# --- Extract pixel-averaged trace per matched detection, frame by frame ---
n_frames = F.shape[1]
detection_traces = np.zeros((n_matches, n_frames))

t0 = time.time()
with tifffile.TiffFile(movie_path) as tif:
    assert len(tif.pages) == n_frames
    for frame_idx, page in enumerate(tif.pages):
        frame = page.asarray()
        sums = ndimage.sum(frame, labeled, index=matched_labels)
        detection_traces[:, frame_idx] = sums / region_npix
        if frame_idx % 1000 == 0:
            print(f"  frame {frame_idx}/{n_frames} ({time.time() - t0:.0f}s elapsed)")
print(f"Finished reading movie in {time.time() - t0:.0f}s")

# --- Correlate against Suite2p's own F for the matched cell ---
correlations = np.array([
    np.corrcoef(detection_traces[i], F[suite2p_idx[i]])[0, 1]
    for i in range(n_matches)
])

print(f"Across {n_matches} matched pairs:")
print(f"  mean correlation:   {correlations.mean():.2f}")
print(f"  median correlation: {np.median(correlations):.2f}")
print(f"  range: [{correlations.min():.2f}, {correlations.max():.2f}]")
print(f"  >0.5: {np.sum(correlations > 0.5)}/{n_matches} ({np.mean(correlations > 0.5):.0%})")
print(f"  <0.2: {np.sum(correlations < 0.2)}/{n_matches} ({np.mean(correlations < 0.2):.0%})")

np.savez(
    out_dir / "exercise3_deliverable2_state.npz",
    detection_traces=detection_traces,
    correlations=correlations,
    suite2p_idx=suite2p_idx,
)

# --- Plot best and worst matches: traces only (image overlay in a second figure) ---
order = np.argsort(correlations)
worst = order[:2]
best = order[-2:]

fig, axes = plt.subplots(4, 1, figsize=(11, 9), sharex=False)
t_axis = np.arange(n_frames) / 15


def zscore(x):
    return (x - x.mean()) / x.std()


for row, idx in enumerate(list(best[::-1]) + list(worst)):
    ax = axes[row]
    ax.plot(t_axis, zscore(F[suite2p_idx[idx]]), color="tab:red", label="Suite2p F")
    ax.plot(t_axis, zscore(detection_traces[idx]), color="tab:blue", label="This detection's pixels")
    tag = "good match" if row < 2 else "poor match"
    ax.set_title(f"{tag}: r={correlations[idx]:.2f}")
    if row == 0:
        ax.legend(fontsize=8)
axes[-1].set_xlabel("Time (s)")
fig.tight_layout()
fig.savefig(out_dir / "exercise3_match_quality_examples.png", dpi=150)
print("Saved plots to tutorials/friday_exercises/")
