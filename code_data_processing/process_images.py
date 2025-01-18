## import necessary libraries
import os
import cv2

# ===================================================================================================================
def create_output_dirs(base_dir, patient_folder):
    patient_dir = os.path.join(base_dir, patient_folder)
    if not os.path.exists(patient_dir):
        os.makedirs(patient_dir)
    return patient_dir

# -------------------------------------------------------------------------------------------------------------------
def save_images(patient_dir, image, image_type):
    os.makedirs(patient_dir, exist_ok=True)

    if image_type == 'ct_image.png':
        filename = "ct_image.png"
    elif image_type == 'pet_image.png':
        filename = "pet_image.png"
    else:
        raise ValueError("image_type muss entweder 'ct' oder 'pet' sein.")

    output_path = os.path.join(patient_dir, filename)

    # debug
    print(f"Saving image to: {output_path}")
    if not cv2.imwrite(output_path, image):
        print(f"Failed to save the image to {output_path}. Please check the path and permissions.")

# -------------------------------------------------------------------------------------------------------------------
def save_registered_images(patient_dir, transformed_image, generation_count):
    output_path = os.path.join(patient_dir, f'registered_image_gen_{generation_count}.png')
    cv2.imwrite(output_path, transformed_image)


# ===================================================================================================================
def load_images_GRAYSCALE(dir_path):
    images_dict = {}
    for patient_folder in os.listdir(dir_path):
        patient_path = os.path.join(dir_path, patient_folder)
        if not os.path.isdir(patient_path) or patient_folder.startswith('.'):
            continue

        expected_images = ['ct_image.png', 'pet_image.png']
        images = {img: os.path.join(patient_path, img) for img in expected_images}

        if not all(os.path.exists(img_path) for img_path in images.values()):
            print(f"error: missing 'ct' or 'pet' image in patient folder {patient_folder}.")
            continue

        try:
            ct_image = cv2.imread(images['ct_image.png'], cv2.IMREAD_GRAYSCALE)
            pet_image = cv2.imread(images['pet_image.png'], cv2.IMREAD_GRAYSCALE)

            if ct_image is None or pet_image is None:
                raise ValueError(f"error reading 'ct' or 'pet' image in {patient_folder}.")

            images_dict[patient_folder] = {'ct': ct_image, 'pet': pet_image}

        except Exception as e:
            print(f"error reading images in patient folder {patient_folder}: {e}")

    print("successfully loaded all images.")
    print(50 * "_")
    return images_dict

# -------------------------------------------------------------------------------------------------------------------
def load_images(dir_path):
    images_dict = {}
    for patient_folder in os.listdir(dir_path):
        patient_path = os.path.join(dir_path, patient_folder)
        if not os.path.isdir(patient_path) or patient_folder.startswith('.'):
            continue

        expected_images = ['ct_image.png', 'pet_image.png']
        images = {img: os.path.join(patient_path, img) for img in expected_images}

        if not all(os.path.exists(img_path) for img_path in images.values()):
            print(f"error: missing 'ct' or 'pet' image in patient folder {patient_folder}.")
            continue

        try:
            ct_image = cv2.imread(images['ct_image.png'], cv2.IMREAD_COLOR)
            pet_image = cv2.imread(images['pet_image.png'], cv2.IMREAD_COLOR)

            if ct_image is None or pet_image is None:
                raise ValueError(f"error reading 'ct' or 'pet' image in {patient_folder}.")

            images_dict[patient_folder] = {'ct': ct_image, 'pet': pet_image}

        except Exception as e:
            print(f"error reading images in patient folder {patient_folder}: {e}")

    print("successfully loaded all images.")
    print(50 * "_")
    return images_dict

