## import necessary libraries
import numpy as np
import cv2

# ===================================================================================================================
## auxiliary methods

# normalized histogram/probability for every intensity value of an image
def P_I(_image):
    # density=True normalizes the histogram so the sum of all bins is 1 in total
    # this is basically the probability of every intensity-value
    hist, _ = np.histogram(_image.flatten(), bins=256, range=(0, 256), density=True)
    return hist

# entropy H of an image I
# use one-dimensional histogram-calculation
def H_I(_image):
    p = P_I(_image)
    # constant to avoid log(0)
    H = -np.sum(p * np.log2(p + 1e-10))
    return H

# -------------------------------------------------------------------------------------------------------------------
# normalized histogram/probability for the intensity values of two images
def P_I_J(ct_image, pet_image):
    hist_2d, _, _ = np.histogram2d(ct_image.flatten(), pet_image.flatten(), bins=256,
                                   range=[[0,256], [0,256]], density=True)
    return hist_2d

# entropy H of two images I & J
# use two-dimensional histogram-calculation
def H_I_J(ct_image, pet_image):
    p = P_I_J(ct_image, pet_image)
    H = -np.sum(p * np.log2(p + 1e-10))
    return H

# -------------------------------------------------------------------------------------------------------------------
# mean-intensity of an image
def mu(_image):
    mu = np.mean(_image.flatten())
    return mu

# standard deviation of the pixel intensities
def sigma(_image):
    sigma = np.std(_image.flatten())
    return sigma

# ===================================================================================================================
## methods / fitness functions

# mutual information
# to be maximized
def MI(ct_image, pet_image):
    MI = H_I(ct_image) + H_I(pet_image) - H_I_J(ct_image, pet_image)
    #print(f"mutual information: {MI}")
    return MI

# -------------------------------------------------------------------------------------------------------------------
# normalized mutual information
# to be maximized
def NMI(ct_image, pet_image):
    NMI = 2 * MI(ct_image, pet_image)/(H_I(ct_image) + H_I(pet_image))
    #print(f"normalized mutual information: {NMI}")
    return NMI

# -------------------------------------------------------------------------------------------------------------------
# sum of squared differences
# to be minimized
def SSD(ct_image, pet_image):
    # make sure the images have the same size
    if ct_image.shape == pet_image.shape:
        ct_image = ct_image.astype(np.uint8)
        pet_image = pet_image.astype(np.uint8)

        SD = np.square(ct_image - pet_image)
        SSD = np.sum(SD)
        #print(f"sum of squared differences: {SSD}")
        return SSD
    else:
        raise ValueError("the images are not having the same size.")

# -------------------------------------------------------------------------------------------------------------------
# normalized cross correlation
# to be maximized
def NCC(ct_image, pet_image):
    # make sure the images have the same size
    if ct_image.shape == pet_image.shape:

        # calculate the mean intensity of the images
        mu_I, mu_J = mu(ct_image), mu(pet_image)
        # calculate the standard deviation of the intensities
        sigma_I, sigma_J = sigma(ct_image), sigma(pet_image)

        NCC = np.sum(((ct_image - mu_I) * (pet_image - mu_J))/(ct_image.size * sigma_I * sigma_J))

        return NCC
    else:
        raise ValueError("the images are not having the same size.")

# ===================================================================================================================
## calculate fitness values over a whole dict and keep them as reference values
# calculate the NCC over a whole dict
def calculate_NCC(reference_dict):
    # sort the dict items (here: from 1 to 21)
    sorted_dict = dict(sorted(reference_dict.items(), key=lambda item: int(item[0][1:])))
    # create emty dict to save fitness values
    reference_NCC_values = {}

    for patient_folder in sorted_dict:
        # define images
        images = reference_dict[patient_folder]
        # divide into ct an pet images
        ct_image = images['ct']
        pet_image = images['pet']

        # make sure the images have the same size
        if ct_image.shape == pet_image.shape:
            # calculate the normalized cross correlation
            mu_I, mu_J = mu(ct_image), mu(pet_image)
            sigma_I, sigma_J = sigma(ct_image), sigma(pet_image)
            NCC = np.sum(((ct_image - mu_I) * (pet_image - mu_J)) / (ct_image.size * sigma_I * sigma_J))

            # save NCC value in the dict for the corresponding patient
            reference_NCC_values[patient_folder] = NCC
            print(f"normalized cross correlation: {NCC}")

    return reference_NCC_values

# -------------------------------------------------------------------------------------------------------------------
# calcualte MI over a whole dict
def calculate_MI(reference_dict):
    # sort the dict items (here: from 1 to 21)
    sorted_dict = dict(sorted(reference_dict.items(), key=lambda item: int(item[0][1:])))
    # create emty dict to save fitness values
    reference_MI_values = {}

    for patient_folder in sorted_dict:
        # define images
        images = reference_dict[patient_folder]
        # divide into ct and pet images
        ct_image = images['ct']
        pet_image = images['pet']

        # calculate the mutual information
        MI = H_I(ct_image) + H_I(pet_image) - H_I_J(ct_image, pet_image)

        # save MI value in the dict for the corresponding patient
        reference_MI_values[patient_folder] = MI
        print(f"MI für {patient_folder}: {MI}")

    return reference_MI_values


#def calculate_MI(image_dict):
#    sorted_dict = dict(sorted(image_dict.items(), key=lambda item: int(item[0][1:])))
#    for patient_folder in sorted_dict:
#        # define images
#        images = image_dict[patient_folder]
#        # divide into ct an pet images
#        ct_image = images['ct']
#        pet_image = images['pet']#
#
#        MI = H_I(ct_image) + H_I(pet_image) - H_I_J(ct_image, pet_image)
#        print(MI)
#    return MI


# same as NCC but with printed NCC
#def NCC_(ct_image, pet_image):
#    if ct_image.shape == pet_image.shape:
#        mu_I, mu_J = mu(ct_image), mu(pet_image)
#        sigma_I, sigma_J = sigma(ct_image), sigma(pet_image)
#        NCC = np.sum(((ct_image - mu_I) * (pet_image - mu_J))/(ct_image.size * sigma_I * sigma_J))
#        print(f"normalized cross correlation: {NCC}")
#        return NCC
#    else:
#        raise ValueError("the images are not having the same size.")
