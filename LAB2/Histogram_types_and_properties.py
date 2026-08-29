import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt

# ---- Histogram Slicing parameters ----
SLICE_A, SLICE_B = 100, 180
SLICE_HIGHLIGHT = 255


# ------------------------------------------------------------------ #
# 1. Histogram Stretching
# ------------------------------------------------------------------ #
def full_scale_stretch(gray_img):
    min_val, max_val = float(gray_img.min()), float(gray_img.max())
    if max_val == min_val:
        return gray_img.copy()
    stretched = (gray_img.astype(np.float64) - min_val) * (255.0 / (max_val - min_val))
    return np.clip(stretched, 0, 255).astype(np.uint8)


# ------------------------------------------------------------------ #
# 2. Histogram Slicing
# ------------------------------------------------------------------ #
def histogram_slicing(gray_img, a=SLICE_A, b=SLICE_B, highlight=SLICE_HIGHLIGHT, keep_bg=True):
    out = gray_img.copy() if keep_bg else np.zeros_like(gray_img)
    mask = (gray_img >= a) & (gray_img <= b)
    out[mask] = highlight
    return out


# ------------------------------------------------------------------ #
# 3. Histogram Specification
# ------------------------------------------------------------------ #
def compute_cdf(hist):
    cdf = np.cumsum(hist).astype(np.float64)
    cdf /= cdf[-1] if cdf[-1] != 0 else 1
    return cdf


def synthetic_gaussian_histogram(mean=128, std=40, size=256):
    x = np.arange(size)
    return np.exp(-0.5 * ((x - mean) / std) ** 2)


def histogram_specification(gray_img, reference_hist):
    src_hist = cv2.calcHist([gray_img], [0], None, [256], [0, 256]).flatten()
    src_cdf = compute_cdf(src_hist)
    ref_cdf = compute_cdf(reference_hist)

    mapping = np.zeros(256, dtype=np.uint8)
    ref_idx = 0
    for src_idx in range(256):
        while ref_idx < 255 and ref_cdf[ref_idx] < src_cdf[src_idx]:
            ref_idx += 1
        mapping[src_idx] = ref_idx
    return mapping[gray_img]


def plot_row(axes_row, img, title):
    ax_img, ax_hist = axes_row
    ax_img.imshow(img, cmap="gray")
    ax_img.set_title(title)
    ax_img.axis("off")
    hist = cv2.calcHist([img], [0], None, [256], [0, 256]).flatten()
    ax_hist.bar(range(256), hist, width=1.0, color="steelblue")
    ax_hist.set_title(f"{title} - Histogram")
    ax_hist.set_xlim([0, 256])


def main():
    if len(sys.argv) > 1:
        path = sys.argv[1]
    else:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk(); root.withdraw()
        path = filedialog.askopenfilename(title="Select an image")
        if not path:
            print("No file selected.")
            return

    ref_path = sys.argv[2] if len(sys.argv) > 2 else None

    img = cv2.imread(path)
    if img is None:
        print(f"Could not read image at {path}")
        return
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    stretched = full_scale_stretch(gray)
    sliced = histogram_slicing(gray)

    if ref_path:
        ref_img = cv2.imread(ref_path)
        if ref_img is not None:
            reference_hist = cv2.calcHist(
                [cv2.cvtColor(ref_img, cv2.COLOR_BGR2GRAY)], [0], None, [256], [0, 256]
            ).flatten()
        else:
            reference_hist = synthetic_gaussian_histogram()
    else:
        reference_hist = synthetic_gaussian_histogram()

    specified = histogram_specification(gray, reference_hist)

    fig, axes = plt.subplots(4, 2, figsize=(11, 16))
    plot_row(axes[0], gray, "Original")
    plot_row(axes[1], stretched, "Histogram Stretching")
    plot_row(axes[2], sliced, f"Histogram Slicing [{SLICE_A}-{SLICE_B}]")
    plot_row(axes[3], specified, "Histogram Specification")

    fig.suptitle("Histogram Types & Properties (Combined)", fontsize=15, fontweight="bold")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()