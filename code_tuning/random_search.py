import numpy as np
import random
import matplotlib.pyplot as plt
import pandas as pd
from algorithm_code._2_fitness_functions import NCC
from algorithm_code._4_selection_methods import tournament_selection, rankbased_selection, SUS_selection, \
    roulette_selection
from algorithm_code.apply_algorithm import process_images_and_run_algorithm_for_tuning
from code_data_processing.process_images import load_images
import openpyxl
from tqdm import tqdm
import logging

param_ranges = {
    'population_size': (20, 70),
    'number_selected_individuals': (5, 40),
    'number_generations': (40, 100),
    'crossover_rate': (0.3, 0.9),
    'mutation_rate': (0.01, 0.1),
    'selection_method': [tournament_selection, rankbased_selection, roulette_selection, SUS_selection],
    'crossover_method': ['uniform', 'two_point', 'single_point'],
}


def generate_random_parameters():
    population_size = random.randint(20, 70)
    number_selected_individuals = random.randint(5, 40)
    number_generations = random.randint(40, 100)
    crossover_rate = random.uniform(0.3, 0.9)
    mutation_rate = random.uniform(0.01, 0.1)
    selection_method = random.choice([tournament_selection, rankbased_selection, roulette_selection, SUS_selection])
    crossover_method = random.choice(['uniform', 'two_point', 'single_point'])

    return population_size, number_selected_individuals, number_generations, \
        crossover_rate, mutation_rate, selection_method, crossover_method

def calculate_variance_of_parameters(population_matrix):
    # Assume population_matrix is a list of transformation matrices
    # Stack all transformation matrices into a 3D array
    population_array = np.array(population_matrix)

    # Calculate variance along the axis corresponding to the population
    # (i.e., across all individuals for each parameter)
    variance_per_parameter = np.var(population_array, axis=0)

    return variance_per_parameter


def plot_results(average_fitness_all_generations, average_variance_all_generations):
    num_generations = len(average_fitness_all_generations)

    plt.figure(figsize=(14, 6))
    for i, fitness_list in enumerate(average_fitness_all_generations):
        plt.plot(fitness_list, label=f'generation {i + 1}')
    plt.xlabel('generations')
    plt.ylabel('average fitness')
    plt.title('average fitness over generations')
    plt.legend()
    plt.show()

    plt.figure(figsize=(14, 6))
    for i, variance_list in enumerate(average_variance_all_generations):
        plt.plot(variance_list, label=f'generation {i + 1}')
    plt.xlabel('generations')
    plt.ylabel('average variance')
    plt.title('average variance over generations')
    plt.legend()
    plt.show()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def save_results_to_excel(all_runs, filename='results.xlsx'):
    with pd.ExcelWriter(filename) as writer:
        for idx, run in enumerate(all_runs):
            # extract parameter and results
            best_params = run['best_params']
            average_fitness_all_generations = run['average_fitness']
            average_variance_all_generations = run['average_variance']

            # create DataFrames for fitness scores and variances
            results_df = pd.DataFrame({
                'generation': [f'generation {i + 1}' for i in range(len(average_fitness_all_generations))],
                'average fitness': average_fitness_all_generations,
                'variance': average_variance_all_generations
            })
            params_df = pd.DataFrame([best_params])

            sheet_name = f'run {idx + 1}'
            params_df.to_excel(writer, sheet_name=sheet_name, startrow=0, index=False)
            results_df.to_excel(writer, sheet_name=sheet_name, startrow=len(params_df) + 2, index=False)

    print(f"results saved to {filename}")

def random_search(image_dict, maximizing_fitness_function, num_iterations):
    best_score = float('-inf')
    best_params = None
    best_results = None
    all_runs = []

    for i in tqdm(range(num_iterations), desc="Random Search Progress"):
        params = generate_random_parameters()
        logging.info(f"Iteration {i + 1}/{num_iterations} - Testing parameters: {params}")

        average_fitness_all_generations, average_variance_all_generations = process_images_and_run_algorithm_for_tuning(
            image_dict, maximizing_fitness_function, *params)

        last_gen_fitness = average_fitness_all_generations[-1] if average_fitness_all_generations else []
        avg_last_gen_fitness = np.mean(last_gen_fitness) if last_gen_fitness else 0

        if avg_last_gen_fitness > best_score:
            best_score = avg_last_gen_fitness
            best_params = params
            best_results = (average_fitness_all_generations, average_variance_all_generations)

        all_runs.append({
            'best_params': params,
            'average_fitness': average_fitness_all_generations,
            'average_variance': average_variance_all_generations
        })

        save_results_to_excel(all_runs, filename=f'random_search_results.xlsx')
        #plot_results(best_results[0], best_results[1])

    logging.info(f"Best parameters: {best_params}")
    logging.info(f"Best score: {best_score}")
    return best_params, best_results


DATA = '/Users/anne/PycharmProjects/genetic_algorithm/data/preprocessed_data'
image_dict = load_images(DATA)
maximizing_fitness_function = NCC
num_iterations = 100

best_params, best_results = random_search(image_dict, maximizing_fitness_function, num_iterations)
