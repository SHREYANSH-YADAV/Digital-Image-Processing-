import cv2
import numpy as np
import matplotlib.pyplot as plt
from tkinter import Tk, filedialog
import os

# Step 1: Select the input image
root = Tk()
root.withdraw()

image_path = filedialog.askopenfilename(
    title="Select an Image",
    filetypes=[
        ("Image Files", "*.jpg *.jpeg *.png *.bmp *.tif *.tiff")
    ]
)

root.destroy()

if not image_path:
    print("No image selected.")
    exit()

# Step 2: Read the image
image = cv2.imread(image_path)

if image is None:
    print("Error: Unable to load the image.")
    exit()

# Step 3: Convert the image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Step 4: Apply Gaussian Blur to reduce noise
blurred = cv2.GaussianBlur(
    gray,
    (5, 5),
    1.4
)

# Step 5: Apply Canny Edge Detection
edges = cv2.Canny(
    blurred,
    threshold1=50,
    threshold2=150
)

# Step 6: Save the edge-detected image
folder = os.path.dirname(image_path)
filename = os.path.splitext(os.path.basename(image_path))[0]

output_path = os.path.join(
    folder,
    filename + "_edges.png"
)

cv2.imwrite(output_path, edges)

# Step 7: Display the results
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title("Original Image")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.imshow(gray, cmap="gray")
plt.title("Grayscale Image")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.imshow(edges, cmap="gray")
plt.title("Canny Edge Detection")
plt.axis("off")

plt.tight_layout()
plt.show()

print("Edge detection completed successfully!")
print("Output image saved at:", output_path)

