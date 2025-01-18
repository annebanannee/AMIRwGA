from algorithm_code._2_fitness_functions import NCC, NMI, MI
from algorithm_code._6_crossover_mutation import two_point_crossover, uniform_crossover, single_point_crossover
from algorithm_code._4_selection_methods import tournament_selection, rankbased_selection, roulette_selection, SUS_selection
from algorithm_code._8_genetic_algorithm import genetic_algorithm_for_tuning
from code_data_processing.process_images import load_images
from algorithm_code.apply_algorithm import process_images_and_run_algorithm_for_tuning

DATA = '/Users/anne/PycharmProjects/genetic_algorithm/data/preprocessed_data'
#OUTPUT = '/Users/anne/PycharmProjects/genetic_algorithm/data_output/output_images'


image_dict = load_images(DATA)


def parameter_tuning(image_dict, fitness_function, runs):
    selection_methods = [rankbased_selection, tournament_selection, SUS_selection]
    crossover_methods = [uniform_crossover, single_point_crossover, two_point_crossover]
    mutation_rates = [0.001, 0.003, 0.005, 0.007, 0.01]
    crossover_rates = [0.1, 0.3, 0.5, 0.7]

    best_results = []

    # Parameter für genetischen Algorithmus
    tl, tu = -20, 20
    sl, su = 0.99, 1.05
    rl, ru = -3, 3
    shl, shu = -0.001, 0.001
    population_size = 50
    number_selected_individuals = population_size // 5
    number_generations = 50

    # Beste Fitness initialisieren
    best_overall_fitness = float('inf')
    best_overall_params = None

    for selection_method in selection_methods:
        for crossover_method in crossover_methods:
            for mutation_rate in mutation_rates:
                for crossover_rate in crossover_rates:
                    print(f'Evaluating: {selection_method.__name__}, {crossover_method.__name__}, '
                          f'Mutation rate: {mutation_rate}, Crossover rate: {crossover_rate}')

                    avg_fitness_runs = []
                    for run in range(runs):
                        print(f"  Run {run + 1}/{runs}")

                        # Hier wird der genetische Algorithmus aufgerufen
                        average_fitness_all_generations, _ = process_images_and_run_algorithm_for_tuning(
                            tl, tu, sl, su, rl, ru, shl, shu, image_dict, fitness_function, population_size,
                            number_selected_individuals, number_generations, crossover_rate, mutation_rate,
                            selection_method, crossover_method, minimize=False
                        )

                        # Fitness nach den Generationen berechnen
                        best_fitness = average_fitness_all_generations[-1]  # Letzte Generation = beste Fitness
                        avg_fitness_runs.append(best_fitness)
                        print(f"    Fitness after generation {number_generations}: {best_fitness}")

                    # Durchschnittliche Fitness über alle Runs
                    mean_fitness = sum(avg_fitness_runs) / runs
                    print(f"  Average fitness across {runs} runs: {mean_fitness}")

                    # Überprüfe, ob diese Fitness besser ist als die bisher beste
                    if mean_fitness < best_overall_fitness:
                        best_overall_fitness = mean_fitness
                        best_overall_params = {
                            'selection_method': selection_method.__name__,
                            'crossover_method': crossover_method.__name__,
                            'mutation_rate': mutation_rate,
                            'crossover_rate': crossover_rate
                        }

                        print(
                            f"  New best overall fitness: {best_overall_fitness} with parameters: {best_overall_params}")

                    # Speichere die besten Ergebnisse
                    best_results.append({
                        'params': {
                            'selection_method': selection_method.__name__,
                            'crossover_method': crossover_method.__name__,
                            'mutation_rate': mutation_rate,
                            'crossover_rate': crossover_rate
                        },
                        'mean_fitness': mean_fitness
                    })

    print(f"\nBest overall fitness: {best_overall_fitness} with parameters: {best_overall_params}")

    return best_results, best_overall_fitness, best_overall_params


# Beispiel-Aufruf mit 1 Run:
best_results = parameter_tuning(image_dict, MI, 1)