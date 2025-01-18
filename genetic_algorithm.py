## import necessary libraries and methods
from code_data_processing.preprocess_transform_images import load_images

from code_visualization.visualization_methods import plot_patient_fitness_generations, \
    plot_total_average_fitness_generations, plot_total_average_variance_generations, \
    plot_patient_variance_generations, plot_mutation_crossover_rate_and_num_selected, plot_best_fitness_per_run
from code_visualization.image_visualization import overlay_images_with_alpha, apply_color_map1

from algorithm_code._2_fitness_functions import MI, calculate_MI
from algorithm_code._3_initialize_population import select_transformation_parameter
from algorithm_code._4_selection_methods import tournament_selection
from algorithm_code._6_crossover_mutation import uniform_crossover, two_point_crossover, single_point_crossover
from algorithm_code.apply_algorithm import process_images_and_run_algorithm

from algorithm_code._8_genetic_algorithm import genetic_algorithm2
import cv2
import numpy as np
import matplotlib.pyplot as plt
# ===================================================================================================================
## define paths
# reference directories
PREPROCESSED_DIRECTORY = '/Users/anne/PycharmProjects/genetic_algorithm/data/preprocessed_data'
PADDED_100_PREPROCESSED = '/Users/anne/PycharmProjects/genetic_algorithm/data/padded_100'
PADDED_225_PREPROCESSED = '/Users/anne/PycharmProjects/genetic_algorithm/data/padded_225'

# -------------------------------------------------------------------------------------------------------------------
# image directories (with transformed pet images)
SLIGHTLY_TRANSFORMED = '/Users/anne/PycharmProjects/genetic_algorithm/data_transformed/slightly_new'
MORE_SLIGHTLY_TRANSFORMED = '/Users/anne/PycharmProjects/genetic_algorithm/data_transformed/more_slightly_new'
TRANSFORMED = '/Users/anne/PycharmProjects/genetic_algorithm/data_transformed/transformed_225'

# -------------------------------------------------------------------------------------------------------------------
# output directories with colored relevant (breast) structure only
BREAST_NO_TRANSFORM_OUTPUT = '/Users/anne/PycharmProjects/genetic_algorithm/data_output/breast_no_transform'
BREAST_SLIGHTLY_OUTPUT = '/Users/anne/PycharmProjects/genetic_algorithm/data_output/breast_slightly_output'
BREAST_MORE_SLIGHTLY_OUTPUT = '/Users/anne/PycharmProjects/genetic_algorithm/data_output/torso_more_slightly_output'

# output directories with whole colored torso
TORSO_NO_TRANSFORM_OUTPUT = '/Users/anne/PycharmProjects/genetic_algorithm/data_output/torso_no_transform'
TORSO_SLIGHTLY_OUTPUT = '/Users/anne/PycharmProjects/genetic_algorithm/data_output/torso_slightly_output'
TORSO_MORE_SLIGHTLY_OUTPUT = '/Users/anne/PycharmProjects/genetic_algorithm/data_output/breast_more_slightly_output'
TORSO_TRANSFORMED_OUTPUT = '/Users/anne/PycharmProjects/genetic_algorithm/data_output/torso_transformed_output'

# ===================================================================================================================
## declare variables/parameters for GA
maximizing_fitness_function = MI
selection_method = tournament_selection
crossover_method = uniform_crossover
minimize = False
population_size = 150
initial_mutation_rate = 0.05
final_mutation_rate = 0.001
crossover_rate = 0.9
final_crossover_rate = 0.3
number_selected_individuals = 25
final_number_selected = 25
number_generations = 100

# ===================================================================================================================
## load directories
reference_dict = load_images(PREPROCESSED_DIRECTORY)
#reference_dict = load_images(PADDED_100_PREPROCESSED)
#reference_dict = load_images(PADDED_225_PREPROCESSED)

reference_dict = calculate_MI(reference_dict)

# -------------------------------------------------------------------------------------------------------------------
image_dict = load_images(SLIGHTLY_TRANSFORMED)
#image_dict = load_images(MORE_SLIGHTLY_TRANSFORMED)
#image_dict = load_images('/Users/anne/PycharmProjects/genetic_algorithm/data_transformed/more_slightly_new')
#image_dict = load_images(SLIGHTLY_TRANSFORMED)

# ===================================================================================================================
## declare output to save the overlaid images in
#output = BREAST_SLIGHTLY_OUTPUT
#output = BREAST_NO_TRANSFORM_OUTPUT
output = '/Users/anne/PycharmProjects/genetic_algorithm/data_output/more_only_transformed_pet'
#output = '/Users/anne/PycharmProjects/genetic_algorithm/data_output/slightly_only_transformed_pet'

# ===================================================================================================================
## final algorithm for the whole dict
variance_per_image_pair, \
average_fitness_all_generations, \
average_variance_all_generations, \
mutation_rates, crossover_rates, \
    count_selected_individuals = process_images_and_run_algorithm(reference_dict, image_dict, output,
                                                                  maximizing_fitness_function, population_size,
                                                                  number_selected_individuals,
                                                                  final_number_selected, number_generations,
                                                                  crossover_rate, final_crossover_rate,
                                                                  initial_mutation_rate, final_mutation_rate,
                                                                  selection_method, crossover_method)

# ===================================================================================================================
## visualize

#fitness score evolution per patient: line for every patient
plot_patient_fitness_generations(average_fitness_all_generations)
# total fitness score evolution: line for all patients (average)
plot_total_average_fitness_generations(average_fitness_all_generations)

plot_patient_variance_generations(variance_per_image_pair)
plot_total_average_variance_generations(average_variance_all_generations)

plot_mutation_crossover_rate_and_num_selected(number_generations, mutation_rates, crossover_rates)


# ===================================================================================================================
#reference_ct_image_path = '/Users/anne/PycharmProjects/genetic_algorithm/data/preprocessed_data/p1/ct_image.png'
#reference_pet_image_path = '/Users/anne/PycharmProjects/genetic_algorithm/data/preprocessed_data/p1/pet_image.png'
#ct_image_path = '/Users/anne/PycharmProjects/genetic_algorithm/data_transformed/slightly_new/p1/ct_image.png'
#pet_image_path = '/Users/anne/PycharmProjects/genetic_algorithm/data_transformed/slightly_new/p1/pet_image.png'

#reference_ct_image = cv2.imread(reference_ct_image_path)
#reference_pet_image =  cv2.imread(reference_pet_image_path)
#ct_image = cv2.imread(ct_image_path)
#pet_image = cv2.imread(pet_image_path)

#ct_image_copied = ct_image.copy()
#pet_image_copied = pet_image.copy()

#reference_fitness_value = MI(reference_ct_image, reference_pet_image)
#initial_fitness = MI(ct_image, pet_image)

#fitness_percentage = (initial_fitness / reference_fitness_value) * 100
#print(f"initial fitness percentage for: {fitness_percentage}%")

#pet_image_ = apply_color_map1(pet_image_copied, color_map=cv2.COLORMAP_HSV)

#overlaid_image_before = overlay_images_with_alpha(ct_image_copied, pet_image_)

# depending on the fitness percentage select the transformation parameter (slightly, more slightly, transformed)
#tl, tu, sl, su, rl, ru, shl, shu = select_transformation_parameter(fitness_percentage)

#final_best_parameter_per_run = []
#best_fitness_per_run = []
#for i in range(20):
#    print(f"\n### Run {i + 1} ###")
## final algorithm for selected image pairs
#    population_matrix, gray_coded_population, overlaid_images, \
#                average_fitness_per_generation, variances_per_generation, \
#                mutation_rates, crossover_rates, count_selected_individuals, \
#                transformed_pet_image, best_fitness, final_best_parameter = genetic_algorithm2(reference_fitness_value, tl, tu, sl, su, rl, ru, shl, shu,
#                                                          overlaid_image_before, ct_image, pet_image,
#                                                              maximizing_fitness_function, population_size,
#                                                          number_selected_individuals, final_number_selected,
#                                                          number_generations, crossover_rate, final_crossover_rate, initial_mutation_rate,
#                                                          final_mutation_rate, selection_method, crossover_method, minimize)

#    final_best_parameter_per_run.append(final_best_parameter)
#    best_fitness_per_run.append(best_fitness)

# Berechne Mittelwert und Varianz über die Läufe hinweg für jede Generation
#average_best_fitness = np.mean(best_fitness_per_run, axis=0)

#plot_best_fitness_per_run(best_fitness_per_run, final_best_parameter_per_run, average_best_fitness)
# Plotten der Diagramme
#runs = list(range(1, len(final_best_parameter_per_run) + 1))

#plt.figure(figsize=(13, 8))

#plt.plot(runs, best_fitness_per_run, label='best fitness per run', marker='o', color='b')
#plt.axhline(average_best_fitness, color='r', linestyle='--', label=f'average best fitness over runs: {average_best_fitness:.2f}')
#plt.xlabel('run')
#plt.ylabel('best fitness')
#plot_mutation_crossover_rate_and_num_selected(number_generations, mutation_rates, crossover_rates)
#plt.tight_layout()
#plt.show()

#tx_values = [run[0] for run in final_best_parameter_per_run]
#ty_values = [run[1] for run in final_best_parameter_per_run]
#sx_values = [run[2] for run in final_best_parameter_per_run]
#sy_values = [run[3] for run in final_best_parameter_per_run]
#r_values = [run[4] for run in final_best_parameter_per_run]
#shx_values = [run[5] for run in final_best_parameter_per_run]
#shy_values = [run[6] for run in final_best_parameter_per_run]

#fig, axs = plt.subplots(4, 1, figsize=(13, 8))

# 1. Translation (tx, ty)
#axs[0].plot(runs, tx_values, label="tx", color="blue")
#axs[0].set_ylabel("tx", color="blue")
#axs[0].tick_params(axis='y', labelcolor="blue")

#ax_twin0 = axs[0].twinx()
#ax_twin0.plot(runs, ty_values, label="ty", color="red")
#ax_twin0.set_ylabel("ty", color="red")
#ax_twin0.tick_params(axis='y', labelcolor="red")

#axs[0].set_title('best translation parameters (tx and ty) over runs')

# 2. Scaling (sx, sy)
#axs[1].plot(runs, sx_values, label="sx", color="green")
#axs[1].set_ylabel("sx", color="green")
#axs[1].tick_params(axis='y', labelcolor="green")

#ax_twin1 = axs[1].twinx()
#ax_twin1.plot(runs, sy_values, label="sy", color="orange")
#ax_twin1.set_ylabel("sy", color="orange")
#ax_twin1.tick_params(axis='y', labelcolor="orange")

#axs[1].set_title('best scaling parameters (sx and sy) over runs')

# 3. Rotation (r)
#axs[2].plot(runs, r_values, label="r", color="purple")
#axs[2].set_ylabel("r", color="purple")
#axs[2].tick_params(axis='y', labelcolor="purple")

#axs[2].set_title('best rotation parameter (r) over runs')

# 4. Shearing (shx, shy)
#axs[3].plot(runs, shx_values, label="shx", color="brown")
#axs[3].set_ylabel("shx", color="brown")
#axs[3].tick_params(axis='y', labelcolor="brown")

#ax_twin3 = axs[3].twinx()
#ax_twin3.plot(runs, shy_values, label="shy", color="cyan")
#ax_twin3.set_ylabel("shy", color="cyan")
#ax_twin3.tick_params(axis='y', labelcolor="cyan")

#axs[3].set_title('best shearing parameters (shx and shy) over runs')

# Set x labels
#for ax in axs:
#    ax.set_xlabel("run")

#plt.tight_layout()
#plt.show()