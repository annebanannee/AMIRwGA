## import necessary libraries
import numpy as np
import random
from ._1_coding_decoding import transformation_matrix, individual_to_bitstring, binary_to_gray, gray_to_binary, matrix_to_bitstring

# ===================================================================================================================
# select transformation parameter depending on the fitnesspercentage of the images in relation to reference images
def select_transformation_parameter(fitness_percentage):
    if fitness_percentage > 90:
        # transformation ranges for 'not' transformed data
        tl, tu = -20, 20
        sl, su = 0.99, 1.05
        rl, ru = -3, 3
        shl, shu = -0.001, 0.001

    elif fitness_percentage > 60:
        # transformation ranges for slightly transformed data
        tl, tu = -50, 50
        sl, su = 0.95, 1.10
        rl, ru = -7, 7
        shl, shu = -0.007, 0.007

    elif 30 < fitness_percentage < 60:
        # transformation ranges for more slightly transformed data
        tl, tu = -100, 100
        sl, su = 0.90, 1.15
        rl, ru = -25, 25
        shl, shu = -0.025, 0.025

    else:
        # transformation ranges for transformed data
        tl, tu = -120, 120
        sl, su = 0.5, 1.50
        rl, ru = -45, 45
        shl, shu = -0.07, 0.07

    return tl, tu, sl, su, rl, ru, shl, shu

# ===================================================================================================================
# initialize random transformations within the ranges to create a population
def random_transformation(tl, tu, sl, su, rl, ru, shl, shu):
    tx, ty = random.randint(tl, tu), random.randint(tl, tu)
    sx, sy = random.uniform(sl, su), random.uniform(sl, su)
    a = random.uniform(rl, ru)
    shx, shy = random.uniform(shl, shu), random.uniform(shl, shu)
    return transformation_matrix(tx, ty, sx, sy, a, shx, shy), tx, ty, sx, sy, a, shx, shy



# -------------------------------------------------------------------------------------------------------------------
## initialize population as binary-, gray-coded-, decoded-bitstring and matrix
def initialize_population(population_size, tl, tu, sl, su, rl, ru, shl, shu):
    # four lists to differentiate between the different populations
    population = []
    population_as_bitstring = []
    population_as_gray_coded_bitstring = []

    parameter = []
    # decoded just to check if the coding worked properly
    #population_decoded = []

    # create a population which is made of individuals; as many as the population_size instructs
    for individual in range(population_size):
        transformation, tx, ty, sx, sy, a, shx, shy = random_transformation(tl, tu, sl, su, rl, ru, shl, shu)
        binary_bitstring = matrix_to_bitstring(transformation, tl, tu, sl, su, rl, ru, shl, shu)
        gray_coded_bitstring = binary_to_gray(binary_bitstring)
        #decoded_bitstring = gray_to_binary(gray_coded_bitstring)

        # create the different populations thourough adding the individuals (transformations) the the lists
        population.append(transformation)
        population_as_bitstring.append(binary_bitstring)
        population_as_gray_coded_bitstring.append(gray_coded_bitstring)
        #population_decoded.append(decoded_bitstring)
        parameter.append((tx, ty, sx, sy, a, shx, shy))
    # debug if necessary
    #print(f"population as matrices:{population}")
    #print(f"population as gray-code-bitstrings:{population_as_gray_coded_bitstring}")
    return population, population_as_bitstring, population_as_gray_coded_bitstring, parameter #, population_decoded
