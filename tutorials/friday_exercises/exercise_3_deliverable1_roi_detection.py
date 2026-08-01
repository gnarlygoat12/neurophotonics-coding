import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
from pathlib import Path

data_dir = Path("/Users/kluglab/Downloads/TSeries-03042024-run02-054")
out_dir = Path(__file__).parent

ops = np.load(data_dir / "ops.npy", allow_pickle=True).item()
stat = np.load(data_dir / "stat.npy", allow_pickle=True)

mean_img = ops["meanImg"]
true_centers = np.array([s["med"] for s in stat])  # (y, x) per Suite2p convention
true_npix = np.array([s["npix"] for s in stat])
print(f"Ground truth: {len(stat)} ROIs, size range {true_npix.min()}-{true_npix.max()} px")

MATCH_DIST = 30  # pixels (~12 um)


def detect(mean_img, percentile, size_range, sigma=0.7):
    smoothed = ndimage.gaussian_filter(mean_img, sigma=sigma)
    threshold = np.percentile(smoothed, percentile)
    binary = smoothed > threshold
    labeled, n = ndimage.label(binary)
    centers = []
    label_ids = []
    for region_id in range(1, n + 1):
        mask = labeled == region_id
        npix = mask.sum()
        if size_range[0] <= npix <= size_range[1]:
            ys, xs = np.where(mask)
            centers.append((ys.mean(), xs.mean()))
            label_ids.append(region_id)
    return smoothed, labeled, np.array(centers) if centers else np.empty((0, 2)), np.array(label_ids, dtype=int)


def one_to_one_match(det_centers, true_centers, max_dist):
    pairs = []
    for di, dc in enumerate(det_centers):
        for ti, tc in enumerate(true_centers):
            d = np.linalg.norm(dc - tc)
            if d <= max_dist:
                pairs.append((d, di, ti))
    pairs.sort(key=lambda x: x[0])

    matched_det, matched_true = set(), set()
    match_pairs = []
    for d, di, ti in pairs:
        if di in matched_det or ti in matched_true:
            continue
        matched_det.add(di)
        matched_true.add(ti)
        match_pairs.append((di, ti))

    tp = len(match_pairs)
    fp = len(det_centers) - tp
    fn = len(true_centers) - tp
    sensitivity = tp / (tp + fn) if (tp + fn) else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    return match_pairs, matched_det, matched_true, sensitivity, precision


# --- Loose detector: 80th percentile, bare size>10 floor ---
_, _, loose_centers, _ = detect(mean_img, percentile=80, size_range=(10, np.inf))
_, _, _, loose_sens, loose_prec = one_to_one_match(loose_centers, true_centers, MATCH_DIST)
print(f"Loose (80th pct, size>10): n_detections={len(loose_centers)} "
      f"sensitivity={loose_sens:.1%} precision={loose_prec:.1%}")

# --- Tuned detector: 99th percentile, size matching real cell dimensions ---
smoothed, labeled, tuned_centers, tuned_label_ids = detect(mean_img, percentile=99, size_range=(20, 1500))
match_pairs, matched_det, matched_true, tuned_sens, tuned_prec = one_to_one_match(
    tuned_centers, true_centers, MATCH_DIST
)
print(f"Tuned (99th pct, size 20-1500): n_detections={len(tuned_centers)} "
      f"sensitivity={tuned_sens:.1%} precision={tuned_prec:.1%}")

# --- Where do the errors cluster? Check left/right brightness split ---
h, w = mean_img.shape
left_mean = mean_img[:, : w // 2].mean()
right_mean = mean_img[:, w // 2 :].mean()
print(f"Left-half mean brightness: {left_mean:.1f}, right-half: {right_mean:.1f}")

fp_idx = [i for i in range(len(tuned_centers)) if i not in matched_det]
fn_idx = [i for i in range(len(true_centers)) if i not in matched_true]
fp_on_right = np.mean([tuned_centers[i, 1] > w / 2 for i in fp_idx]) if fp_idx else float("nan")
print(f"False positives on the brighter half: {fp_on_right:.1%} ({len(fp_idx)} total FPs)")

if fn_idx:
    fn_brightness = [mean_img[int(true_centers[i, 0]), int(true_centers[i, 1])] for i in fn_idx]
    tp_true_idx = [ti for _, ti in match_pairs]
    tp_brightness = [mean_img[int(true_centers[i, 0]), int(true_centers[i, 1])] for i in tp_true_idx]
    print(f"Mean brightness at missed-cell centers: {np.mean(fn_brightness):.1f} "
          f"vs found-cell centers: {np.mean(tp_brightness):.1f}")

# --- Plot ---
fig, axes = plt.subplots(1, 3, figsize=(16, 5.5))
axes[0].imshow(mean_img, cmap="gray")
axes[0].set_title("Raw mean image")
axes[0].axis("off")

axes[1].imshow(smoothed > np.percentile(smoothed, 99), cmap="gray")
axes[1].set_title("Smoothed threshold mask (tuned, 99th pct)")
axes[1].axis("off")

axes[2].imshow(mean_img, cmap="gray")
for ti in range(len(true_centers)):
    color = "lime" if ti in matched_true else "magenta"
    y, x = true_centers[ti]
    axes[2].plot(x, y, "o", color=color, markersize=4, markeredgewidth=0)
for di in fp_idx:
    y, x = tuned_centers[di]
    axes[2].plot(x, y, "x", color="red", markersize=6)
axes[2].axvline(w / 2, color="yellow", linestyle="--", linewidth=1)
axes[2].set_title("Green=matched, Magenta=missed, Red=false positive")
axes[2].axis("off")

fig.tight_layout()
fig.savefig(out_dir / "exercise3_roi_detection_results.png", dpi=150)
print("Saved plot to tutorials/friday_exercises/exercise3_roi_detection_results.png")

# Save tuned-detector state for Deliverable 2
np.savez(
    out_dir / "exercise3_deliverable1_state.npz",
    tuned_centers=tuned_centers,
    tuned_label_ids=tuned_label_ids,
    match_pairs=np.array(match_pairs),
    labeled=labeled,
)
