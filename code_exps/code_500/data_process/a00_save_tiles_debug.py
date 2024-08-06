import numpy as np
import matplotlib.pyplot as plt

file = "C:/University of Limerick/AI_ML/code_exps/code_500/input/train_256_36/00a7fb880dc12c5de82df39b30533da9.npz"
images = np.load(file)["arr_0"]

# Check the shape of the images
print("Original shape of images:", images.shape)

# Since images have the shape (1, 1536, 1536, 3), no need to transpose
# Normalize the images
img = images[0] / 255.0

# Display the image
plt.imshow(1.0 - img)
plt.show()