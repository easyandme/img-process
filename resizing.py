import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Define the image name
inputimage = 'static/img/1.tif'
# Load the imagef
img = Image.open(inputimage)

img = np.asarray(img)
img2 = img
for i in range(3):
    img2 = np.concatenate((img2, np.rot90(img, i+1)), axis=1)
# Prepare figure window
plt.figure()
# Show image on the window
plt.imshow(img2, cmap='gray')
# Show the window
plt.show()