import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import convolution_matrix
from scipy.signal import find_peaks
from sklearn.linear_model import Lasso
from pathlib import Path

out_dir = Path(__file__).parent
rng = np.random.default_rng(0)

FS = 15  # Hz, matches the real dataset's frame rate
TAU = 1.0  # s, calcium decay timescale
N_FRAMES = 2000  # ~2.2 min, matches the real-data subset used in Deliverable 2
DURATION = N_FRAMES / FS
RATE_HZ = 1.0
REFRACTORY_S = 0.15
SPIKE_AMP = 8.0  # arbitrary units per spike
SNR_TARGET = 3.0

# --- 1. Kernel ---
n_kernel = int(5 * TAU * FS) + 1
t_kernel = np.arange(n_kernel) / FS
h = np.exp(-t_kernel / TAU)

# --- 2. Poisson spike train with refractory period ---
spike_times = []
t = 0.0
while t < DURATION:
    t += rng.exponential(1.0 / RATE_HZ)
    if t < DURATION:
        if not spike_times or (t - spike_times[-1]) >= REFRACTORY_S:
            spike_times.append(t)

true_frames = np.unique(np.round(np.array(spike_times) * FS).astype(int))
true_frames = true_frames[true_frames < N_FRAMES]
spikes = np.zeros(N_FRAMES)
spikes[true_frames] = SPIKE_AMP

# --- 3. Convolve causally to make synthetic fluorescence ---
fluor_clean = np.convolve(spikes, h, mode="full")[:N_FRAMES]

# --- 4. Noise: shot + Gaussian, scaled to hit SNR ~ 3 (signal std / noise std) ---
SHOT_COEFF = 0.3  # shot noise alone (coeff=1) already exceeds the SNR-3 budget
shot = rng.normal(0, SHOT_COEFF * np.sqrt(np.clip(fluor_clean, 1e-6, None)))
target_noise_std = fluor_clean.std() / SNR_TARGET
gauss_var = max(target_noise_std ** 2 - shot.var(), 0.0)
gauss = rng.normal(0, np.sqrt(gauss_var), N_FRAMES)
noise = shot + gauss
achieved_snr = fluor_clean.std() / noise.std()
fluor_noisy = fluor_clean + noise

print(f"True spikes: {len(true_frames)}")
print(f"Achieved SNR (signal std / noise std): {achieved_snr:.2f}")

# --- 5. Toeplitz convolution matrix (causal) ---
H = convolution_matrix(h, N_FRAMES, mode="full")[:N_FRAMES, :]

# --- 6. Baseline subtract ---
target = fluor_noisy - fluor_noisy.min()

# --- 7. Lasso deconvolution ---
# fit_intercept=False: the model has no offset term (Hs already starts near 0
# for an all-zero spike train), and centering here would conflict with the
# positive=True constraint and blow up numerically.
ALPHA = 0.065
lasso = Lasso(alpha=ALPHA, positive=True, fit_intercept=False, max_iter=5000)
lasso.fit(H, target)
s_hat = lasso.coef_
print(f"Nonzero recovered coefficients: {(s_hat > 0).sum()}")

# --- 8. Peak finding ---
peaks, _ = find_peaks(s_hat, height=s_hat.max() * 0.15, distance=int(REFRACTORY_S * FS))
print(f"Recovered peaks: {len(peaks)}")

# --- 9. One-to-one matching within a tolerance window, then score ---
TOLERANCE = 3  # frames (~200 ms)
pairs = []
for pi in peaks:
    for ti in true_frames:
        d = abs(int(pi) - int(ti))
        if d <= TOLERANCE:
            pairs.append((d, pi, ti))
pairs.sort(key=lambda x: x[0])

matched_true = set()
matched_pred = set()
tp = 0
for d, pi, ti in pairs:
    if pi in matched_pred or ti in matched_true:
        continue
    matched_pred.add(pi)
    matched_true.add(ti)
    tp += 1

fp = len(peaks) - tp
fn = len(true_frames) - tp
sensitivity = tp / (tp + fn) if (tp + fn) else 0.0
precision = tp / (tp + fp) if (tp + fp) else 0.0
f1 = 2 * precision * sensitivity / (precision + sensitivity) if (precision + sensitivity) else 0.0

print(f"TP={tp} FP={fp} FN={fn}")
print(f"Sensitivity: {sensitivity:.2f}")
print(f"Precision:   {precision:.2f}")
print(f"F1:          {f1:.2f}")

# --- Plot ---
t_axis = np.arange(N_FRAMES) / FS
fig, axes = plt.subplots(4, 1, figsize=(12, 9), sharex=True)

axes[0].plot(t_axis, fluor_noisy, color="gray", linewidth=0.8, label="noisy")
axes[0].plot(t_axis, fluor_clean, color="black", linewidth=1, label="noiseless")
axes[0].set_ylabel("Fluorescence")
axes[0].set_title(f"Synthetic fluorescence (SNR ≈ {achieved_snr:.1f})")
axes[0].legend(fontsize=8)

axes[1].vlines(true_frames / FS, 0, spikes[true_frames], color="black")
axes[1].set_ylabel("True spikes")

axes[2].plot(t_axis, s_hat, color="tab:green")
axes[2].plot(peaks / FS, s_hat[peaks], "rx")
axes[2].set_ylabel("Recovered s")

axes[3].plot(t_axis, s_hat, color="tab:green")
for ti in true_frames:
    axes[3].axvline(ti / FS, color="black", linestyle="--", linewidth=0.5, alpha=0.5)
axes[3].set_ylabel("Recovered\n+ true (dashed)")
axes[3].set_xlabel("Time (s)")

fig.tight_layout()
fig.savefig(out_dir / "exercise2_synthetic_validation.png", dpi=150)
print("Saved plot to tutorials/friday_exercises/exercise2_synthetic_validation.png")
