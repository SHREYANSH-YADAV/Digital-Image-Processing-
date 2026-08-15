from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

image = Image.open("Leena.png").convert("RGB")

img = np.array(image)

gray = image.convert("L")

red = img[:, :, 0]
green = img[:, :, 1]
blue = img[:, :, 2]

# 3. Display all images

plt.figure(figsize=(12, 8))

plt.subplot(2, 3, 1)
plt.imshow(image)
plt.title("Original Image")
plt.axis("off")

plt.subplot(2, 3, 2)
plt.imshow(gray, cmap="gray")
plt.title("Grayscale")
plt.axis("off")

plt.subplot(2, 3, 3)
plt.imshow(red, cmap="Reds")
plt.title("Red Layer")
plt.axis("off")

plt.subplot(2, 3, 4)
plt.imshow(green, cmap="Greens")
plt.title("Green Layer")
plt.axis("off")

plt.subplot(2, 3, 5)
plt.imshow(blue, cmap="Blues")
plt.title("Blue Layer")
plt.axis("off")

plt.tight_layout()
plt.show() 

gray.save("grayscale.png")

Image.fromarray(red).save("red_layer.png")
Image.fromarray(green).save("green_layer.png")
Image.fromarray(blue).save("blue_layer.png")