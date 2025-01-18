## import necessary libraries
import cv2
import numpy as np

# ===================================================================================================================
## preprocessing methods
# negative transformation
def negative_transformation(image):
    negative_image = cv2.bitwise_not(image)
    return negative_image

# -------------------------------------------------------------------------------------------------------------------
# gauß-filter
def gaussian_blur_filter(image, ksize=(5, 5)):
    blurred_image = cv2.GaussianBlur(image, ksize, 0)
    return blurred_image

# -------------------------------------------------------------------------------------------------------------------
# median-filter
def median_filter(image, ksize):
    if ksize % 2 == 0:
        raise ValueError("kernel size must be odd.")
    filtered_image = cv2.medianBlur(image, ksize)
    return filtered_image

# -------------------------------------------------------------------------------------------------------------------
# contrast-optimization
def clahe_contrast_optimization(image):
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    clahe_image = clahe.apply(image)
    return clahe_image

# -------------------------------------------------------------------------------------------------------------------
# normalization
def normalize_image(image):
    min_val = np.min(image)
    max_val = np.max(image)
    normalized_image = (image - min_val) / (max_val - min_val) * 255
    return normalized_image.astype(np.uint8)

# -------------------------------------------------------------------------------------------------------------------
# resampling (only applied on pet later)
def resampling(image, size=(512, 512)):
    return cv2.resize(image, size, interpolation=cv2.INTER_NEAREST)

# -------------------------------------------------------------------------------------------------------------------
# crop
def crop_image(image, crop_rect):
    x, y, w, h = crop_rect
    cropped_image = image[y:y+h, x:x+w]
    return cropped_image

# -------------------------------------------------------------------------------------------------------------------
def bilateral_filter(image):
    image = cv2.bilateralFilter(image, d=9, sigmaColor=75, sigmaSpace=75)
    return image

# -------------------------------------------------------------------------------------------------------------------
def histogram_equalization(image):
    image = cv2.equalizeHist(image)
    return image

# -------------------------------------------------------------------------------------------------------------------
def add_padding(image, is_pet_image=False):
    padded_image = cv2.copyMakeBorder(image, 100, 100, 100, 100, cv2.BORDER_CONSTANT, value=255)
    return padded_image

# ===================================================================================================================
## apply preprocessing methods in a method

def preprocess_all_images(image, is_pet_image=False):
    if is_pet_image:
        image = resampling(image)
    image = crop_image(image, (80, 40, 350, 310))
    return image


def process_image(image, is_pet_image=False):
    #image = bilateral_filter(image)
    #image = histogram_equalization(image)
    image = gaussian_blur_filter(image)
    image = clahe_contrast_optimization(image)
    image = median_filter(image, 5)
    #image = negative_transformation(image)
    image = normalize_image(image)
    return image


def remove_padding(image, padding_size=225):
    # Get the dimensions of the image
    height, width = image.shape[:2]
    #print(f'bildhöhe: {height} und -breite: {width}')
    # Remove the padding by slicing the image to its original size
    unpadded_image = image[padding_size:height-padding_size, padding_size:width-padding_size]

    return unpadded_image


