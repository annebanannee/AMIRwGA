from algorithm_code._2_fitness_functions import NCC, MI
from algorithm_code._4_selection_methods import tournament_selection, rankbased_selection
from algorithm_code._8_genetic_algorithm import genetic_algorithm
#from algorithm_code._6_crossover_mutation import single_point_crossover, two_point_crossover, uniform_crossover
from code_data_processing.process_images import load_images

DATA = '/Users/anne/PycharmProjects/genetic_algorithm/data/preprocessed_data'
OUTPUT = '/Users/anne/PycharmProjects/genetic_algorithm/data_output/output_images'


population_size = 50
NCC_crossover_rate = 0.7
MI_crossover_rate = 0.9
number_selected_individuals = population_size // 5
number_generations = 50
image_dict = load_images(DATA)

def tune_mutation(image_dict, fitness_function, selection_method, crossover_method, crossover_rate):
    sorted_dict = dict(sorted(image_dict.items(), key=lambda item: int(item[0][1:])))

    mutation_rates = [0.001, 0.005, 0.01]
    result = {}

    best_combination = None
    best_score = float('-inf')

    for mutation_rate in mutation_rates:
        fitness_scores = []

        for patient_folder in sorted_dict:
            # Define images
            images = image_dict[patient_folder]
            # Divide into ct and pet images
            ct_image = images['ct']
            pet_image = images['pet']

            print(f"Testing mutation rate: {mutation_rate} for patient {patient_folder}")

            # Run the genetic algorithm
            _, _, _, average_fitness_per_generation = genetic_algorithm(
                ct_image, pet_image, fitness_function, population_size,
                number_selected_individuals, number_generations, crossover_rate, mutation_rate,
                selection_method, crossover_method, minimize=False)

            final_fitness_value = average_fitness_per_generation[-1]

            if isinstance(final_fitness_value, (int, float)):
                fitness_scores.append(final_fitness_value)
                print(f"Fitness value for mutation rate {mutation_rate}: {final_fitness_value}")
            else:
                print(f"Unexpected data type in av_fitness_per_gen: {type(final_fitness_value)}")

        if fitness_scores:
            # Calculate the average fitness score across all patients for the current mutation rate
            av_score = sum(fitness_scores) / len(fitness_scores)
            result[mutation_rate] = av_score

            # Check if the current mutation rate is the best one so far
            if av_score > best_score:
                best_score = av_score
                best_combination = mutation_rate

    print(f"Best mutation rate for {fitness_function.__name__} with {selection_method.__name__}, {crossover_method} "
          f"and a crossover rate of {crossover_rate}: {best_combination} with an average score of {best_score}")



tune_mutation(image_dict, NCC, tournament_selection, 'uniform', NCC_crossover_rate)
tune_mutation(image_dict, MI, rankbased_selection, 'two_point', MI_crossover_rate)
