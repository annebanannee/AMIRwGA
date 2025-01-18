#import scikit as skopt
from tqdm import tqdm
import logging
from concurrent.futures import ProcessPoolExecutor
from skopt import gp_minimize
from skopt.space import Real, Integer, Categorical
from algorithm_code._4_selection_methods import rankbased_selection, roulette_selection, tournament_selection, SUS_selection
from algorithm_code._8_genetic_algorithm import genetic_algorithm_for_tuning
from algorithm_code._2_fitness_functions import NCC, MI
from algorithm_code._6_crossover_mutation import single_point_crossover, two_point_crossover, uniform_crossover
from code_data_processing.process_images import load_images

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

DATA = '/Users/anne/PycharmProjects/genetic_algorithm/data/preprocessed_data'
image_dict = load_images(DATA)
tl, tu = -20, 20
sl, su = 0.99, 1.05
rl, ru = -5, 5
shl, shu = -0.005, 0.005

def objective(params):
    population_size, generations, crossover_rate, mutation_rate, selection_method, crossover_method, num_selected = params
    num_selected = min(num_selected, population_size)

    total_fitness = 0
    count = 0

    for patient_folder in image_dict:
        images = image_dict[patient_folder]
        ct_image = images['ct']
        pet_image = images['pet']
        _, _, _, average_fitness_per_generation, variances_per_generation = genetic_algorithm_for_tuning(tl, tu, sl, su, rl, ru, shl, shu,
            ct_image, pet_image, MI, population_size,
            num_selected, generations, crossover_rate, mutation_rate,
            selection_method, crossover_method, minimize=False)

        final_fitness_value = average_fitness_per_generation[-1]

        total_fitness += final_fitness_value
        count += 1

    average_fitness = total_fitness / count if count > 0 else 0
    logging.info(f"evaluating parameters: {params} -- objective function result: {-average_fitness}")
    return float(-average_fitness)

def parallel_objective(params):
    return objective(params)


search_space = [
    Integer(20, 50, name='population_size'),
    Integer(50, 100, name='generations'),
    Categorical([0.7, 0.8, 0.9], name='crossover_rate'),
    Categorical([0.001, 0.002, 0.005, 0.007], name='mutation_rate'),
    Categorical([rankbased_selection, tournament_selection, SUS_selection], name='selection_method'),
    Categorical([single_point_crossover, two_point_crossover, uniform_crossover], name='crossover_method'),
    Integer(10, 40, name='num_selected')]



def tqdm_gp_minimize(func, dimensions, n_calls, **kwargs):
    with tqdm(total=n_calls) as pbar:
        def callback(res):
            pbar.update(1)
        kwargs['callback'] = callback
        return gp_minimize(func, dimensions, n_calls=n_calls, **kwargs)

with ProcessPoolExecutor() as executor:
    result = tqdm_gp_minimize(parallel_objective, search_space, n_calls=20, random_state=0, n_jobs=-1)


logging.info(f"best parameter: {result.x}")
logging.info(f"bester score: {-result.fun}")
