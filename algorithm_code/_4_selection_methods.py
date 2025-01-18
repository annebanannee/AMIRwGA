## import necessary libraries
import random
import numpy as np
import math

# ===================================================================================================================
## fitnessproportional selection methods
# roulette selection
def roulette_selection_(population_matrix, population_coded, fitness_values, number_selected_individuals):
    total_fitness = sum(fitness_values)
    selection_p = [fitness / total_fitness for fitness in fitness_values]

    selected_indices = random.choices(range(len(population_matrix)), weights=selection_p, k=number_selected_individuals)


    selected_individuals_matrix = [population_matrix[i] for i in selected_indices]
    selected_individuals_bitstrings = [population_coded[i] for i in selected_indices]
    selected_fitness_values = [fitness_values[i] for i in selected_indices]

    # debug if necessary
    #for i in selected_indices:
        #print(f"Selected individual index: {i}, Fitness score: {fitness_scores[i]}")

    #print(f"selected individuals with roulette selection (matrices): {selected_individuals_matrix}")
    #print(f"as gray coded bitstrings: {selected_individuals_bitstrings}")

    return selected_fitness_values, selected_individuals_matrix, selected_individuals_bitstrings

def roulette_selection(population_matrix, population_coded, parameters, fitness_values, number_selected_individuals):
    total_fitness = sum(fitness_values)
    selection_p = [fitness / total_fitness for fitness in fitness_values]

    selected_indices = random.choices(range(len(population_matrix)), weights=selection_p, k=number_selected_individuals)

    selected_individuals_matrix = [population_matrix[i] for i in selected_indices]
    selected_individuals_bitstrings = [population_coded[i] for i in selected_indices]
    selected_fitness_values = [fitness_values[i] for i in selected_indices]
    selected_parameters = [parameters[i] for i in selected_indices]

    return selected_fitness_values, selected_individuals_matrix, selected_individuals_bitstrings, selected_parameters

# -------------------------------------------------------------------------------------------------------------------
# stochastic universal sampling
def SUS_selection_(population_matrix, gray_coded_population, fitness_values, number_selected_individuals):
    total_fitness = sum(fitness_values)
    # distance
    d = total_fitness / number_selected_individuals
    starting_point = random.uniform(0, d)
    points = [starting_point + i * d for i in range(number_selected_individuals)]

    selected_individuals = []
    sum_fitness = 0
    i = 0

    for point in points:
        while sum_fitness < point:
            sum_fitness += fitness_values[i]
            i += 1
        selected_individuals.append(i-1)

    selected_individuals = list(set(selected_individuals))
    selected_individuals_matrix = [population_matrix[i] for i in selected_individuals]
    selected_individuals_bitstrings = [gray_coded_population[i] for i in selected_individuals]
    selected_fitness_values = [fitness_values[i] for i in selected_individuals]

    # debug if necessary
    #for i in selected_individuals:
        #print(f"SUS selected individual index: {i}, Fitness score: {fitness_scores[i]}")

    #print(f"selected individuals with SUS-selection (matrices): {selected_individuals_matrix}")
    #print(f"as gray coded bitstrings: {selected_individuals_bitstrings}")

    return selected_fitness_values, selected_individuals_matrix, selected_individuals_bitstrings

def SUS_selection(population_matrix, gray_coded_population, parameters, fitness_values, number_selected_individuals):
    total_fitness = sum(fitness_values)
    d = total_fitness / number_selected_individuals
    starting_point = random.uniform(0, d)
    points = [starting_point + i * d for i in range(number_selected_individuals)]

    selected_individuals = []
    sum_fitness = 0
    i = 0

    for point in points:
        while sum_fitness < point:
            sum_fitness += fitness_values[i]
            i += 1
        selected_individuals.append(i-1)

    selected_individuals = list(set(selected_individuals))
    selected_individuals_matrix = [population_matrix[i] for i in selected_individuals]
    selected_individuals_bitstrings = [gray_coded_population[i] for i in selected_individuals]
    selected_fitness_values = [fitness_values[i] for i in selected_individuals]
    selected_parameters = [parameters[i] for i in selected_individuals]

    return selected_fitness_values, selected_individuals_matrix, selected_individuals_bitstrings, selected_parameters

# ===================================================================================================================
## rankbased selection methods
def rankbased_selection_(population_matrix, gray_coded_population, fitness_values, number_selected_individuals):
    # sort fitness values and populations
    sorted_fitness_values = np.argsort(fitness_values)
    sorted_population = [population_matrix[i] for i in sorted_fitness_values]
    sorted_coded_population = [gray_coded_population[i] for i in sorted_fitness_values]

    # calcualte the ranks for the selection probability
    ranks = np.arange(1, len(sorted_population) + 1)
    selection_p = ranks / ranks.sum()

    # make sure that 'number_selected_individuals' is not bigger than the population size
    number_selected_individuals = min(number_selected_individuals, len(sorted_population))

    # choose individuals based on their rank-probabilities
    chosen_indices = np.random.choice(len(sorted_population), size=number_selected_individuals,
                                      p=selection_p, replace=False)
    selected_individuals_matrix = [sorted_population[i] for i in chosen_indices]
    selected_individuals_bitstrings = [sorted_coded_population[i] for i in chosen_indices]
    selected_fitness_values = [sorted_fitness_values[i] for i in chosen_indices]

    # debug if necessary
    #print(f"selected individuals with rank-based selection (matrices): {selected_individuals_matrix}")
    #print(f"as gray coded bitstrings: {selected_individuals_bitstrings}")

    return selected_fitness_values, selected_individuals_matrix, selected_individuals_bitstrings

def rankbased_selection(population_matrix, gray_coded_population, parameters, fitness_values, number_selected_individuals):
    sorted_fitness_values = np.argsort(fitness_values)
    sorted_population = [population_matrix[i] for i in sorted_fitness_values]
    sorted_coded_population = [gray_coded_population[i] for i in sorted_fitness_values]
    sorted_parameters = [parameters[i] for i in sorted_fitness_values]

    ranks = np.arange(1, len(sorted_population) + 1)
    selection_p = ranks / ranks.sum()

    number_selected_individuals = min(number_selected_individuals, len(sorted_population))

    chosen_indices = np.random.choice(len(sorted_population), size=number_selected_individuals,
                                      p=selection_p, replace=False)

    selected_individuals_matrix = [sorted_population[i] for i in chosen_indices]
    selected_individuals_bitstrings = [sorted_coded_population[i] for i in chosen_indices]
    selected_fitness_values = [sorted_fitness_values[i] for i in chosen_indices]
    selected_parameters = [sorted_parameters[i] for i in chosen_indices]

    return selected_fitness_values, selected_individuals_matrix, selected_individuals_bitstrings, selected_parameters

# ===================================================================================================================
## tournament selection method
def tournament_selection(population_matrix, gray_coded_population, parameters, fitness_values,
                         number_selected_individuals, tournament_size=3):
    selected_individuals = []
    selected_fitness_values = []
    selected_parameters = []
    remaining_individuals = list(range(len(population_matrix)))

    for i in range(number_selected_individuals):
        current_tournament_size = min(tournament_size, len(remaining_individuals))

        # If no individuals remain, stop the process
        if current_tournament_size <= 0:
            break

        # Randomly sample individuals for the tournament
        tournament = random.sample(remaining_individuals, current_tournament_size)
        winner = max(tournament, key=lambda i: fitness_values[i])

        selected_individuals.append(winner)
        selected_fitness_values.append(fitness_values[winner])

        # Get the corresponding parameters for the winner
        selected_parameters.append(parameters[winner])

        remaining_individuals.remove(winner)

    # Create selected individuals matrix and gray coded bitstrings
    selected_individuals_matrix = [population_matrix[i] for i in selected_individuals]
    selected_individuals_bitstrings = [gray_coded_population[i] for i in selected_individuals]

    # Optional debugging output
    # for i in selected_individuals:
    # print(f"Tournament selected individual index: {i}, Fitness score: {fitness_values[i]}")

    # Return selected fitness values, individuals matrix, bitstrings, and parameters
    return selected_fitness_values, selected_individuals_matrix, selected_individuals_bitstrings, selected_parameters

# ===================================================================================================================
# adjust the selection size (= number selected individuals) within the generations
def adjust_selection_size(initial_size, final_size, current_generation, total_generations):
    # parameter that controls the decrease of the curve (the higher the value, the slower the decrease in the beginning)
    k = 5

    # prevent division with 0
    if current_generation == 0:
        return initial_size

    # reduction portion based oon the actual generation with e-function
    normalized_generation = current_generation / total_generations
    reduction_ratio = 1 - math.exp(-k * normalized_generation)

    # calculate new size that reduces slower firstly and shrinks faster with more generations
    new_size = final_size + (initial_size - final_size) * (1 - reduction_ratio)

    # make sure the new size is not smaller than the final size
    return int(max(new_size, final_size))

