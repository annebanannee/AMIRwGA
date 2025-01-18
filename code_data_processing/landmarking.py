## import necessary libraries
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
import numpy as np

# ------------------------------------------------------------------------------------------------
## define directories
# preprocessed directory: where the data is taken from
# landmarked directory: where the processed data is saved
PREPROCESSED_DIRECTORY = '/Users/anne/PycharmProjects/genetic_algorithm/data/preprocessed_data'
LANDMARKED_DIRECTORY = '/Users/anne/PycharmProjects/genetic_algorithm/data/data_landmarked/landmarked'

# create list for the landmarks
landmarks = []

# ------------------------------------------------------------------------------------------------
## method
# method to draw a point/landmark in an image through clicking in the image
def onclick(event):
    global x, y
    # define coordinates of the landmark/point
    x, y = event.xdata, event.ydata
    # add the selected coordinates  for the landmarks to the list
    landmarks.append((int(x), int(y)))
    # draw a red point
    plt.scatter(x, y, c='red', s=50)
    plt.draw()

# ------------------------------------------------------------------------------------------------
## guarantee that programm runs over all directories, subdirectories and images
# if there's no directory for the landmarked data, create one
if not os.path.exists(LANDMARKED_DIRECTORY):
    os.makedirs(LANDMARKED_DIRECTORY)

# iterate over patients/subdirectories
for patient_folder in os.listdir(PREPROCESSED_DIRECTORY):
    patient_path = os.path.join(PREPROCESSED_DIRECTORY, patient_folder)

    # ignore non-directories such as .DS_Store
    if not os.path.isdir(patient_path):
        continue

    # create new path for the landmarked patient data
    landmarked_patient_path = os.path.join(LANDMARKED_DIRECTORY, patient_folder)

    # create data_landmarked subdirectory if not already existing
    if not os.path.exists(landmarked_patient_path):
        os.makedirs(landmarked_patient_path)

    # filter image files only .png, .jpg and .jpeg accepted
    images = [f for f in os.listdir(patient_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

# ----------------------------------------------------------------------------------------------
## apply on all images
    # run over all images
    for image_name in images:
        # define the path where the images are saved later
        img_path = os.path.join(patient_path, image_name)
        img = mpimg.imread(img_path)

        if img.dtype != np.uint8:
            img = (img * 255).astype(np.uint8)

        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

        # create plots for the images
        fig, ax = plt.subplots()
        ax.imshow(img, cmap='gray')
        # title of the plot
        plt.title(f'{patient_folder} - {image_name}: click to set landmarks')

        # clear the landmark list, so they can be selected new for the next image
        landmarks.clear()
        #
        cid = fig.canvas.mpl_connect('button_press_event', onclick)

        # show the image
        plt.show()

        # take the coordinates from the landmarks-list
        for x, y in landmarks:
            # define the size, color, etc. of the points/landmarks that are drawn in the plot
            img = cv2.circle(img, (x, y), radius=5, color=(255, 0, 0), thickness=-1)

        # define path where the image is saved
        output_path = os.path.join(landmarked_patient_path, image_name)
        # save landmarked image
        cv2.imwrite(output_path, img)

        # if everything worked for a single image, print message
        print(f"landmarks set and image saved: {output_path}")
# if everything worked for all patients, print message
print("process finished.")

