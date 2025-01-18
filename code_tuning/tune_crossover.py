from algorithm_code._2_fitness_functions import NCC, NMI, MI
from algorithm_code._4_selection_methods import tournament_selection, rankbased_selection, roulette_selection, SUS_selection
from algorithm_code._8_genetic_algorithm import genetic_algorithm
#from algorithm_code._6_crossover_mutation import single_point_crossover, two_point_crossover, uniform_crossover
from code_data_processing.process_images import load_images

DATA = '/Users/anne/PycharmProjects/genetic_algorithm/data/preprocessed_data'
OUTPUT = '/Users/anne/PycharmProjects/genetic_algorithm/data_output/output_images'


population_size = 50
mutation_rate = 0.01
number_selected_individuals = population_size // 5
number_generations = 50
image_dict = load_images(DATA)

def tune_crossover(image_dict, fitness_function, selection_method):
    sorted_dict = dict(sorted(image_dict.items(), key=lambda item: int(item[0][1:])))

    crossover_methods = ['single_point', 'two_point', 'uniform']
    crossover_rates = [0.3, 0.5, 0.7, 0.9]

    result = {}
    # average_fitness_all_generations = [[] for _ in range(number_generations)]
    for crossover_method in crossover_methods:
        best_combination = None
        best_score = float('-inf')
        print(f"testing fitness function: {crossover_method}")

    # iterate over folders of the sorted dict
        for patient_folder in sorted_dict:
            # define images
            images = image_dict[patient_folder]
            # divide into ct an pet images
            ct_image = images['ct']
            pet_image = images['pet']

            for crossover_rate in crossover_rates:
                print(
                    f"testing combination: {crossover_method} with {crossover_rate} for patient {patient_folder}")
                fitness_scores = []

                _, _, _, average_fitness_per_generation = genetic_algorithm(
                    ct_image, pet_image, fitness_function, population_size,
                    number_selected_individuals, number_generations, crossover_rate, mutation_rate,
                    selection_method, crossover_method, minimize=False)

                final_fitness_value = average_fitness_per_generation[-1]

                if isinstance(final_fitness_value, (int, float)):
                    fitness_scores.append(final_fitness_value)
                    print(
                        f"fitness value for {crossover_method} with crossover rate {crossover_rate}: {final_fitness_value}")
                else:
                    print(f"unexpected data type in av_fitness_per_gen: {type(final_fitness_value)}")

                if fitness_scores:
                    av_score = sum(fitness_scores) / len(fitness_scores)
                    result[(crossover_method, crossover_rate)] = av_score

                    if av_score > best_score:
                        best_score = av_score
                        best_combination = (crossover_method, crossover_rate)

        print(
            f"best crossover method and rate for {fitness_function.__name__} with {selection_method.__name__}: {best_combination[0]} with rate {best_combination[1]} and an average score of {best_score}")


tune_crossover(image_dict, NCC, tournament_selection)
tune_crossover(image_dict, MI, rankbased_selection)



