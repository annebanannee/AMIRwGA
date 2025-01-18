## import necessary library
import random
import numpy as np

# ===================================================================================================================
## crossover methods (parents only as bitstrings)
# single point
def single_point_crossover(parent1, parent2):
    # choose a random point which is in the range of the bitstring length (parent-length)
    point = random.randint(1, len(parent1) - 1)

    # child1: take the values from bitstring start till the point from p1 and combine with the values
    # from the point till bitstring end from p2
    child1 = parent1[:point] + parent2[point:]
    # child2: take the values from bitstring start till the point from p2 and combine with the values
    # from the point till bitstring end from p1
    child2 = parent2[:point] + parent1[point:]
    return child1, child2

# -------------------------------------------------------------------------------------------------------------------
# two point
def two_point_crossover(parent1, parent2):
    if len(parent1) != len(parent2):
        raise ValueError("Parents must have the same length.")

    length = len(parent1)
    point1 = random.randint(1, length - 1)
    point2 = random.randint(1, length - 1)

    if point1 > point2:
        point1, point2 = point2, point1

    child1 = parent1[:point1] + parent2[point1:point2] + parent1[point2:]
    child2 = parent2[:point1] + parent1[point1:point2] + parent2[point2:]

    return child1, child2

# -------------------------------------------------------------------------------------------------------------------
# uniform
def uniform_crossover(parent1, parent2, crossover_rate=0.5):
    if len(parent1) != len(parent2):
        raise ValueError("Parents must have the same length.")

    length = len(parent1)
    child1 = []
    child2 = []

    for i in range(length):
        if random.random() < crossover_rate:

            child1.append(parent2[i])
            child2.append(parent1[i])
        else:
            child1.append(parent1[i])
            child2.append(parent2[i])

    return ''.join(child1), ''.join(child2)

# ===================================================================================================================
## mutation methods
# mutation in GA
def mutation(individual, mutation_rate):
    # new individual is the given individual
    new_individual = list(individual)
    # iterate over the individuals length (=bitstring length) every single number
    for i in range(len(new_individual)):
        # if the mutation_rate is higher than a random value
        if random.random() < mutation_rate:
            # the bit of the bitstring is flipped
            new_individual[i] = '0' if new_individual[i] == '1' else '1'
    return ''.join(new_individual)

# -------------------------------------------------------------------------------------------------------------------
# mutation in AR
# different kind of mutation used in the adaptive radiation method
def mutation_(individual, mutation_rate):
    individual = np.array(individual)
    new_individual = individual.copy()

    for i in range(new_individual.size):
        if random.random() < mutation_rate:
            new_individual.flat[i] = 1 - new_individual.flat[i]

    return new_individual

# ===================================================================================================================
## adjustion methods
# mutation
def adjust_mutation_rate(current_generation, max_generations, initial_rate=0.1, final_rate=0.01):
    decay_factor = (final_rate - initial_rate) / max_generations
    new_rate = initial_rate + decay_factor * current_generation
    return max(final_rate, new_rate)

# -------------------------------------------------------------------------------------------------------------------
# crossover
def adjust_crossover_rate(generation, number_generations, initial_crossover_rate, final_crossover_rate):
    #
    return initial_crossover_rate - ((initial_crossover_rate - final_crossover_rate) * (generation / number_generations))
