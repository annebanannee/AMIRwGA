## import necessary methods
from algorithm_code._2_fitness_functions import NCC, MI
from code_data_processing.process_images import load_images
from code_data_processing.preprocess_transform_images import transform_images, preprocess_images
from code_data_processing.overlay_images import overlay_images
from code_data_processing.preprocessing_methods import preprocess_all_images, process_image, add_padding
from test_preprocessing import test_fitness
# ===================================================================================================================
# define directories
RAW_DIRECTORY = '/Users/anne/PycharmProjects/genetic_algorithm/data/raw_data'
PRE_PREPROCESSED_DIRECTORY = '/Users/anne/PycharmProjects/genetic_algorithm/data/pre_preprocessed_data'
PREPROCESSED_DIRECTORY = '/Users/anne/PycharmProjects/genetic_algorithm/data/preprocessed_data'
PADDED_100_PREPROCESSED = '/Users/anne/PycharmProjects/genetic_algorithm/data/padded_100_preprocessed'
PADDED_225_PREPROCESSED = '/Users/anne/PycharmProjects/genetic_algorithm/data/padded_225_preprocessed'
OVERLAY_BEFORE = '/Users/anne/PycharmProjects/genetic_algorithm/data/overlay_before'

SLIGHTLY_TRANSFORMED = '/Users/anne/PycharmProjects/genetic_algorithm/data_transformed/slightly_new'
MORE_SLIGHTLY_TRANSFORMED = '/Users/anne/PycharmProjects/genetic_algorithm/data_transformed/padded_100_more_slightly'
TRANSFORMED = '/Users/anne/PycharmProjects/genetic_algorithm/data_transformed/transformed'
PADDED_TRANSFORMED = '/Users/anne/PycharmProjects/genetic_algorithm/data_transformed/padded_225_transformed'

LANDMARKED = '/Users/anne/PycharmProjects/genetic_algorithm/data_landmarked/landmarked'
LANDMARKED_TRANSFORMED = '/Users/anne/PycharmProjects/genetic_algorithm/data_landmarked/landmarked_transformed_data'
# ===================================================================================================================
## apply (pre-)processing methods
# -------------------------------------------------------------------------------------------------------------------
# load
#images_dict = load_images(PRE_PREPROCESSED_DIRECTORY)
#NCC_dict = load_images(NCC_DATA)
#NMI_dict = load_images(NMI_DATA)
#MI_dict = load_images(MI_DATA)
#SSD_dict = load_images(SSD_DATA)

# -------------------------------------------------------------------------------------------------------------------
# preprocess
#preprocess_images(PRE_PREPROCESSED_DIRECTORY, '/Users/anne/PycharmProjects/genetic_algorithm/data/preprocessed_black', process_image)
#preprocess_images(PREPROCESSED_DIRECTORY, '/Users/anne/PycharmProjects/genetic_algorithm/data/padded_100', add_padding)


# -------------------------------------------------------------------------------------------------------------------
# test fitness
#fitness_functions = [NCC, NMI, MI, SSD]
#optimization_directions = [True, True, True, False]
#test_fitness(images_dict, fitness_functions, NCC_preprocess_image, optimization_directions)
#test_fitness(images_dict, fitness_functions, NMI_preprocess_image, optimization_directions)
#test_fitness(images_dict, fitness_functions, MI_preprocess_image, optimization_directions)
#test_fitness(images_dict, fitness_functions, SSD_preprocess_image, optimization_directions)
#test_fitness(images_dict, fitness_functions, non_specific_preprocess_images, optimization_directions)

# transform
#transform_images('/Users/anne/PycharmProjects/genetic_algorithm/data/padded_100',
#                 '/Users/anne/PycharmProjects/genetic_algorithm/data_transformed/more_slightly_100',
#                 45, 45, 0.95, 1.15, -15, 15, -0.02, 0.02)

transform_images(PREPROCESSED_DIRECTORY, SLIGHTLY_TRANSFORMED, -20, 20, 0.99, 1.05, -5, 5, -0.005, 0.005)

#preprocess_images(PADDED_TRANSFORMED, PADDED_TRANSFORMED, cut)

# for the application of the genetic algorithm on selected images
#ct_image_path = '/Users/anne/PycharmProjects/genetic_algorithm/data/preprocessed_data/p1/ct_image.png'
#pet_image_path = '/Users/anne/PycharmProjects/genetic_algorithm/data/preprocessed_data/p1/pet_image.png'

# ===================================================================================================================
## read the images
#ct_image = cv2.imread(ct_image_path)
#pet_image = cv2.imread(pet_image_path)
# -------------------------------------------------------------------------------------------------------------------
# overlay
#overlay_images(PREPROCESSED_DIRECTORY, OVERLAY_BEFORE)
#process_image(ct_image)
#process_image(pet_image)