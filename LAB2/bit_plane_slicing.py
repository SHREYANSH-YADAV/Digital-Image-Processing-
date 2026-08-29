import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt


def get_bit_planes(gray_img):
    planes = []
    for bit in range(8):
        plane = np.bitwise_and(gray_img, 1 << bit)
        plane = np.where(plane > 0, 255, 0).astype(np.uint8)
        planes.append(plane)
    return planes


def reconstruct_from_planes(planes, num_planes_used):
    recon = np.zeros_like(planes[0], dtype=np.uint16)
    for bit in range(8 - num_planes_used, 8):
        bit_val = (planes[bit] // 255).astype(np.uint16)
        recon += bit_val * (1 << bit)
    return recon.astype(np.uint8)


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

    img = cv2.imread(path)
    if img is None:
        print(f"Could not read image at {path}")
        return
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    planes = get_bit_planes(gray)

    fig, axes = plt.subplots(2, 5, figsize=(18, 7))
    axes = axes.flatten()

    axes[0].imshow(gray, cmap="gray")
    axes[0].set_title("Original Grayscale")
    axes[0].axis("off")

    for bit in range(8):
        ax = axes[bit + 1]
        ax.imshow(planes[bit], cmap="gray")
        ax.set_title(f"Bit Plane {bit} ({'MSB' if bit == 7 else 'LSB' if bit == 0 else ''})")
        ax.axis("off")

    recon = reconstruct_from_planes(planes, 4)
    axes[9].imshow(recon, cmap="gray")
    axes[9].set_title("Reconstructed\n(top 4 bit-planes only)")
    axes[9].axis("off")

    fig.suptitle("Bit Plane Slicing", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()