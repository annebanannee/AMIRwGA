# import necessary libraries
import os
import cv2
import numpy as np

# ===================================================================================================================
# apply colormap on an image
def apply_color_map(image, color_map=cv2.COLORMAP_JET):
    # make sure it's an 8-bit image
    if image.dtype != np.uint8:
        image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    # apply farbmapping
    colored_image = cv2.applyColorMap(image, color_map)

    return colored_image

# ===================================================================================================================
# apply colormap and manipulate alpha chanel
def apply_color_map1(image, color_map):
    threshold = 150
    # convert image to grayscale
    if len(image.shape) == 3:
        intensity = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        intensity = image

    # apply colormap
    colored_img = cv2.applyColorMap(intensity, color_map)

    # create empty image with alpha chanel (BGRA) = chanel for transparency
    colored_img_with_alpha = cv2.cvtColor(colored_img, cv2.COLOR_BGR2BGRA)

    # iterate over every pixel
    for i in range(intensity.shape[0]):
        for j in range(intensity.shape[1]):
            if intensity[i, j] <= threshold:
                # set transparency (alpha) to 150 for pixel under threshold; make pixel darker than 50 visible
                colored_img_with_alpha[i, j, 3] = 75
            else:
                # set transparency to 0 for pixel over threshold; make pixel lighter than 50 invisible
                colored_img_with_alpha[i, j, 3] = 0

    return colored_img_with_alpha

# -------------------------------------------------------------------------------------------------------------------
def overlay_images_with_alpha(ct_image, pet_image_with_alpha):
    # split pet image in colorchanel and alphachanel
    b, g, r, alpha = cv2.split(pet_image_with_alpha)

    # norm alphachanel to [0, 1]
    alpha = alpha.astype(float) / 255.0
    alpha_inv = 1.0 - alpha

    # overlay ct and pet images based on alphachanel
    for c in range(0, 3):
        ct_image[:, :, c] = (alpha_inv * ct_image[:, :, c] + alpha * pet_image_with_alpha[:, :, c])

    return ct_image


