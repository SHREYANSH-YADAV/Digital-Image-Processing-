import cv2
import numpy as np
import pywt
import matplotlib.pyplot as plt
from tkinter import Tk, filedialog

# Select an image
root = Tk()
root.withdraw()
path = filedialog.askopenfilename(
    title="Select Image",
    filetypes=[("Images", "*.jpg *.png *.jpeg *.bmp")]
)
root.destroy()

if not path:
    print("No image selected!")
    exit()

# Read image in grayscale
img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

if img is None:
    print("Unable to read image!")
    exit()

# Make dimensions even
img = img[:img.shape[0] // 2 * 2, :img.shape[1] // 2 * 2]

# Convert to floating point
img = img.astype(float)

# Haar Wavelet Transform
LL, (LH, HL, HH) = pywt.dwt2(img, 'haar')

# 1. IMAGE COMPRESSION
# Remove small coefficients
coeffs = [LL, (LH.copy(), HL.copy(), HH.copy())]

for band in coeffs[1]:
    band[np.abs(band) < 20] = 0

compressed = pywt.idwt2(tuple(coeffs), 'haar')
compressed = np.clip(compressed, 0, 255).astype(np.uint8)

# 2. IMAGE DENOISING
LH_d = pywt.threshold(LH, 20, mode='soft')
HL_d = pywt.threshold(HL, 20, mode='soft')
HH_d = pywt.threshold(HH, 20, mode='soft')

denoised = pywt.idwt2((LL, (LH_d, HL_d, HH_d)), 'haar')
denoised = np.clip(denoised, 0, 255).astype(np.uint8)

# 3. EDGE DETECTION AND FEATURE EXTRACTION
edges = np.sqrt(LH**2 + HL**2 + HH**2)
edges = cv2.normalize(edges, None, 0, 255, cv2.NORM_MINMAX)
edges = edges.astype(np.uint8)

features = np.abs(pywt.idwt2(
    (np.zeros_like(LL), (LH, HL, HH)), 'haar'
))
features = cv2.normalize(features, None, 0, 255, cv2.NORM_MINMAX)
features = features.astype(np.uint8)

# 4. IMAGE FUSION
# Fuse original image with its smoothed version
smooth = cv2.GaussianBlur(img.astype(np.uint8), (5, 5), 1)
LL2, (LH2, HL2, HH2) = pywt.dwt2(smooth.astype(float), 'haar')

LL_f = (LL + LL2) / 2
LH_f = np.where(np.abs(LH) > np.abs(LH2), LH, LH2)
HL_f = np.where(np.abs(HL) > np.abs(HL2), HL, HL2)
HH_f = np.where(np.abs(HH) > np.abs(HH2), HH, HH2)

fused = pywt.idwt2((LL_f, (LH_f, HL_f, HH_f)), 'haar')
fused = np.clip(fused, 0, 255).astype(np.uint8)

# Display wavelet sub-bands
plt.figure(figsize=(10, 7))

plt.subplot(2, 2, 1)
plt.imshow(LL, cmap='gray')
plt.title("LL - Approximation")

plt.subplot(2, 2, 2)
plt.imshow(LH, cmap='gray')
plt.title("LH - Detail")

plt.subplot(2, 2, 3)
plt.imshow(HL, cmap='gray')
plt.title("HL - Detail")

plt.subplot(2, 2, 4)
plt.imshow(HH, cmap='gray')
plt.title("HH - Diagonal Detail")

plt.tight_layout()
plt.show()

# Display all results
plt.figure(figsize=(12, 8))

images = [
    (img, "Original Image"),
    (compressed, "Image Compression"),
    (denoised, "Image Denoising"),
    (edges, "Edge Detection"),
    (features, "Feature Extraction"),
    (fused, "Image Fusion")
]

for i, (image, title) in enumerate(images):
    plt.subplot(2, 3, i + 1)
    plt.imshow(image, cmap='gray')
    plt.title(title)
    plt.axis('off')

plt.tight_layout()
plt.show()

print("All four wavelet operations completed!")

