## import necessary libraries and methods
from ._1_coding_decoding import apply_transformations
import numpy as np

# ===================================================================================================================
def test_fitness_and_select(ct_image, pet_image, maximizing_fitness_function, population_matrix, gray_coded_population,
                            parameter, number_selected_individuals, selection_method, minimize=False):
    ## test fitness
    fitness_values = []
    count = 0
    # iterate over populations to calculate the fitness for every transformation
    for individual in population_matrix:
        count += 1
        # apply transformation matrix aka individual on the pet image with the 'apply_transformation' method
        transformed_pet_image = apply_transformations(pet_image, individual)
        # calcualte fitness with the fitness function that is transfered as a parameter in the method
        fitness_value = maximizing_fitness_function(ct_image, transformed_pet_image)
        # if there's a minimizing fitness function flip the sign
        if minimize:
            fitness_value = -fitness_value
        # add fitness value to the list
        fitness_values.append(fitness_value)
        # debug if necessary
        #print(f"all fitness values for a population: {fitness_values}")

    ## select
    # differentiate between the selection types and apply the corresponding selection method
    # divide into individuals as matrices and as bitstrings for further processing
    best_fitness_values, best_individuals_matrix, \
        best_coded_individuals, best_parameters = selection_method(population_matrix, gray_coded_population, parameter,
                                                                   fitness_values, number_selected_individuals)
    # debug if necessary
    #print(f"Best individuals (matrix): {best_individuals_matrix}")
    #print(f"Best coded individuals (bitstrings): {best_coded_individuals}")
    #print(f"Best fitness values: {best_fitness_values}")
    best_parameters = [parameter[i] for i in range(len(population_matrix))
                       if any(np.array_equal(population_matrix[i], best_ind) for best_ind in best_individuals_matrix)]

    ## sort the selected individuals and return
    # combine fitness values, individuals (as matrices and bitsrtrings) into one list for sorting
    combined = list(zip(best_fitness_values, best_individuals_matrix, best_coded_individuals, best_parameters))
    # debug if necessary
    #print(f"combined list before sorting: {combined}")
    # sort based on fitness values (descending order by default, unless minimizing)
    if combined:
        combined.sort(key=lambda x: x[0], reverse=not minimize)
        # unpack the sorted fitness values and individuals
        sorted_fitness_values, sorted_individuals_matrix, sorted_coded_individuals, sorted_parameter = zip(*combined)
        return sorted_fitness_values, list(sorted_individuals_matrix), list(sorted_coded_individuals), \
            list(sorted_parameter)
    else:
        print("no individuals to sort")
        return [], [], [], []
