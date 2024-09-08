from PIL import Image
import numpy as np
from matplotlib import pyplot as plt

# image must be cropped to pixel area only
IMAGE_PATH = '../stuff/robot_4_crop.jpg'

OUTPUT_PATH = '../stuff/robot_4_background.png'

X_PIXELS = 30
Y_PIXELS = 31

X_OUT = 32
Y_OUT = 32

start_pixel_value = 255

# filter off = 300 (no pixel higher)
filter_value = 300


new_image = np.ones((Y_OUT, X_OUT, 3), dtype='uint8')*start_pixel_value


img = Image.open(IMAGE_PATH)
img = np.asarray(img)
print(img)
print(img.shape)

pixels_per_square_y = img.shape[0]/Y_PIXELS
pixels_per_square_x = img.shape[1]/X_PIXELS

print(pixels_per_square_x)
print(pixels_per_square_y)

init_y = int((Y_OUT - Y_PIXELS)/2)
init_x = int((X_OUT - X_PIXELS)/2)

y = int(pixels_per_square_y/4)
for i in range(init_y, init_y + Y_PIXELS):
    x = int(pixels_per_square_x/4)
    for j in range(init_x, init_x + X_PIXELS):

        pixel = img[y, x]

        # if np.mean(pixel) > filter_value:
        #     new_image[i, j, :] = start_pixel_value
        # else:
        new_image[i, j] = pixel[:3]
        
        x = int(x + pixels_per_square_x)
    y = int(y + pixels_per_square_y)

print(new_image)


plt.imshow(new_image)
plt.show()

new_image = Image.fromarray(new_image)
new_image.save(OUTPUT_PATH)

print('\n\nSAVED TO ', OUTPUT_PATH)