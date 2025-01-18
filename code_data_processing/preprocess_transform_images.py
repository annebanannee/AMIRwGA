## import necessary methods
from algorithm_code._3_initialize_population import random_transformation
from algorithm_code._1_coding_decoding import apply_transformations
from .process_images import load_images, save_images, create_output_dirs, load_images_GRAYSCALE

# ===================================================================================================================
def preprocess_images(src_dir, dest_dir, preprocess_method):
    images_dict = load_images_GRAYSCALE(src_dir)
    for patient_folder, images in images_dict.items():
        patient_path = create_output_dirs(dest_dir, patient_folder)
        pet_image = images['pet']
        ct_image = images['ct']

        try:
            preprocessed_pet_image = preprocess_method(pet_image, is_pet_image=True)
            save_images(patient_path, preprocessed_pet_image, 'pet_image.png')

            preprocessed_ct_image = preprocess_method(ct_image, is_pet_image=False)
            save_images(patient_path, preprocessed_ct_image, 'ct_image.png')

            print(f"images for {patient_folder} preprocessed and saved.")
        except Exception as e:
            print(f"error preprocessing images in patient folder {patient_folder}: {e}")
            continue

    print("successfully preprocessed and saved all images.")
    print(50 * "_")

# ===================================================================================================================
def transform_images(src_dir, dest_dir, tl, tu, sl, su, rl, ru, shl, shu):
    images_dict = load_images_GRAYSCALE(src_dir)
    for patient_folder, images in images_dict.items():
        patient_path = create_output_dirs(dest_dir, patient_folder)
        pet_image = images['pet']
        ct_image = images['ct']

        try:
            transformation_matrix, tx, ty, sx, sy, a, shx, shy = random_transformation(tl, tu, sl, su, rl, ru, shl, shu)

            print(f"matrix for {patient_folder}:{transformation_matrix}")
            print(f"transformation parameter for {patient_folder}: {tx, ty, sx, sy, a, shx, shy}")
            transformed_image = apply_transformations(pet_image, transformation_matrix)
            save_images(patient_path, transformed_image, 'pet_image.png')
            save_images(patient_path, ct_image, 'ct_image.png')

            print(f"pet image for {patient_folder} transfomred and saved .")
        except Exception as e:
            print(f"error processing images in patient folder {patient_folder}: {e}")
            continue

    print("successfully transformed and saved all images.")
    print(50 * "_")

