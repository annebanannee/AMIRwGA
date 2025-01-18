# import necessary libraries
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import cv2

# ===================================================================================================================
# create plot with the average fitness (all individuals) per generation with reference value
def plot_fitness_over_generations(average_fitness_per_generation, reference_fitness_value):
    generations = list(range(1, len(average_fitness_per_generation) + 1))

    plt.figure(figsize=(10, 6))

    plt.axhline(y=reference_fitness_value, color='r', linestyle='--', label='fitness reference (100%')
    plt.plot(generations, average_fitness_per_generation, marker='o', linestyle='-', color='b')

    plt.title("average fitness over generations", fontsize=14)
    plt.xlabel("generation", fontsize=12)
    plt.ylabel("average fitness", fontsize=12)
    plt.grid(True)
    plt.show()

# -------------------------------------------------------------------------------------------------------------------
# create plot with the best fitness in comparison to the average fitness per generation
def plot_av_and_best_fitness_per_gen(average_fitness_per_generation, best_fitness_per_generation):
    generations = list(range(1, len(average_fitness_per_generation) + 1))

    plt.figure(figsize=(10, 6))

    plt.plot(generations, average_fitness_per_generation, label='average_fitness', color='b')
    plt.plot(generations, best_fitness_per_generation, label='best_fitness', color='r')

    plt.title("best (red)) and average (blue) fitness per generation")
    plt.xlabel("generation")
    plt.ylabel("best fitness")

    plt.grid(True)
    plt.show()

# ===================================================================================================================
# create a lineplot for every patients fitness over the generations
def plot_patient_fitness_generations(average_fitness_all_generations):
    # create empty dataframe
    data = {
        "generation": [],
        "fitness": [],
        "patient": []
    }
    # fill datafram thourough iterating over the generations and patients
    for generation, fitness_list in enumerate(average_fitness_all_generations):
        for patient_idx, fitness in enumerate(fitness_list):
            data["generation"].append(generation + 1)
            data["fitness"].append(fitness)
            data["patient"].append(f"patient {patient_idx + 1}")

    df = pd.DataFrame(data)

    # create plot
    plt.figure(figsize=(12, 6))
    sns.lineplot(x="generation", y="fitness", hue="patient", data=df, palette="tab20", marker="o")
    plt.title("fitness score evolution of all patients over the generations")
    plt.xlabel("generation")
    plt.ylabel("average fitness")
    plt.legend(loc='upper right', bbox_to_anchor=(1.15, 1))
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# -------------------------------------------------------------------------------------------------------------------
# create a lineplot of the average fitness values of all patients over the generations
def plot_total_average_fitness_generations(average_fitness_all_generations):
    # number of patients
    num_patients = len(average_fitness_all_generations[0])

    # calcualte average fitness per generation
    average_fitness_per_generation = [
        sum(fitness_list) / num_patients for fitness_list in average_fitness_all_generations
]

    # create dataframe
    df = pd.DataFrame({
        "generation": range(1, len(average_fitness_per_generation) + 1),
        "average fitness": average_fitness_per_generation
    })

    # create plot
    plt.figure(figsize=(10, 6))
    sns.lineplot(x="generation", y="average fitness", data=df, marker="o")
    plt.title("average fitness of all patients over the generations")
    plt.xlabel("generation")
    plt.ylabel("average fitness")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# ===================================================================================================================
# create a lineplot for every patients variance over the generations
def plot_patient_variance_generations(variance_per_image_pair):
    # create empty dataframe
    data = {
        "generation": [],
        "variance": [],
        "patient": []
    }

    # fill dataframe by iterating over the generations and patients
    for patient, variances in variance_per_image_pair.items():
        for generation, variance in enumerate(variances):
            data["generation"].append(generation + 1)
            data["variance"].append(np.mean(variance))  # Assuming variance is a list per generation
            data["patient"].append(patient)

    df = pd.DataFrame(data)
    plt.figure(figsize=(12, 6))
    sns.lineplot(x="generation", y="variance", hue="patient", data=df, palette="tab20", marker="o")
    plt.title("variance evolution of transformation parameters for each patient over the generations")
    plt.xlabel("generation")
    plt.ylabel("average variance")
    plt.legend(loc='upper right', bbox_to_anchor=(1.15, 1))
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# -------------------------------------------------------------------------------------------------------------------
# create a lineplot of the average variance of all patients over the generations
def plot_total_average_variance_generations(average_variance_all_generations):
    # calculate the average variance per generation across all patients
    overall_avg_variance_per_generation = [np.mean(variance) for variance in average_variance_all_generations]

    # create dataframe
    df = pd.DataFrame({
        "generation": range(1, len(overall_avg_variance_per_generation) + 1),
        "average variance": overall_avg_variance_per_generation
    })

    # create plot
    plt.figure(figsize=(10, 6))
    sns.lineplot(x="generation", y="average variance", data=df, marker="o")
    plt.title("average variance of transformation parameters across all patients over the generations")
    plt.xlabel("generation")
    plt.ylabel("average variance")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# ===================================================================================================================
# plot the evolution of the mutation rate, the crossover rate and the number of selected individuals
def plot_mutation_crossover_rate_and_num_selected(number_generations, mutation_rates, crossover_rates):
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # first y-axis for the mutation rate (blue)
    ax1.set_xlabel('generation')
    ax1.set_ylabel('mutation rate', color='b')
    ax1.plot(range(1, number_generations + 1), mutation_rates, marker='o', linestyle='-', color='b')
    ax1.tick_params(axis='y', labelcolor='b')

    # 2nd y-axis for the crossover rate (green)
    # divide 2nd axis
    ax2 = ax1.twinx()
    ax2.set_ylabel('crossover rate', color='g')
    ax2.plot(range(1, number_generations + 1), crossover_rates, marker='s', linestyle='-', color='g')
    ax2.tick_params(axis='y', labelcolor='g')

    plt.title('evolution of mutation rate and crossover rate over the generations')
    ax1.grid(True)
    fig.tight_layout()

    plt.show()

# ===================================================================================================================
def plot_parameters_over_generations(parameters):
    # extract parameter
    generations = list(range(1, len(parameters) + 1))

    # initialise parameter
    tx_values = [gen[0] for gen in parameters]
    ty_values = [gen[1] for gen in parameters]
    sx_values = [gen[2] for gen in parameters]
    sy_values = [gen[3] for gen in parameters]
    r_values = [gen[4] for gen in parameters]
    shx_values = [gen[5] for gen in parameters]
    shy_values = [gen[6] for gen in parameters]

    # create subplot with one col and 4 rows
    fig, axs = plt.subplots(4, 1, figsize=(13, 8))

    # 1. Translation (tx, ty)
    axs[0].plot(generations, tx_values, label="tx", color="blue")
    axs[0].set_ylabel("tx", color="blue")
    axs[0].tick_params(axis='y', labelcolor="blue")

    ax_twin0 = axs[0].twinx()
    ax_twin0.plot(generations, ty_values, label="ty", color="red")
    ax_twin0.set_ylabel("ty", color="red")
    ax_twin0.tick_params(axis='y', labelcolor="red")

    axs[0].set_title('best translation parameters (tx and ty) over generations')

    # 2. Scaling (sx, sy)
    axs[1].plot(generations, sx_values, label="sx", color="green")
    axs[1].set_ylabel("sx", color="green")
    axs[1].tick_params(axis='y', labelcolor="green")

    ax_twin1 = axs[1].twinx()
    ax_twin1.plot(generations, sy_values, label="sy", color="orange")
    ax_twin1.set_ylabel("sy", color="orange")
    ax_twin1.tick_params(axis='y', labelcolor="orange")

    axs[1].set_title('best scaling parameters (sx and sy) over generations')

    # 3. Rotation (r)
    axs[2].plot(generations, r_values, label="r", color="purple")
    axs[2].set_ylabel("r", color="purple")
    axs[2].tick_params(axis='y', labelcolor="purple")

    axs[2].set_title('best rotation parameter (r) over generations')

    # 4. Shearing (shx, shy)
    axs[3].plot(generations, shx_values, label="shx", color="brown")
    axs[3].set_ylabel("shx", color="brown")
    axs[3].tick_params(axis='y', labelcolor="brown")

    ax_twin3 = axs[3].twinx()
    ax_twin3.plot(generations, shy_values, label="shy", color="cyan")
    ax_twin3.set_ylabel("shy", color="cyan")
    ax_twin3.tick_params(axis='y', labelcolor="cyan")

    axs[3].set_title('best shearing parameters (shx and shy) over generations')

    # Set x labels
    for ax in axs:
        ax.set_xlabel("generation")

    plt.tight_layout()
    plt.show()


# ===================================================================================================================
# create subplots where overlaid images of first and last gen as well as graphs for the fitness value are shown
def show_overlay_and_graphs(overlaid_image_first_gen, overlaid_image_last_gen, average_fitness_per_generation,
                                    best_fitness_per_generation, reference_fitness_value):
    generations = list(range(1, len(average_fitness_per_generation) + 1))

    fig, axes = plt.subplots(2, 2, figsize=(13, 8))

    overlaid_image_first_gen = cv2.cvtColor(overlaid_image_first_gen, cv2.COLOR_BGR2RGB)
    axes[0, 0].imshow(overlaid_image_first_gen)
    axes[0, 0].set_title("overlaid image first generation")

    overlaid_image_last_gen = cv2.cvtColor(overlaid_image_last_gen, cv2.COLOR_BGR2RGB)
    axes[0, 1].imshow(overlaid_image_last_gen)
    axes[0, 1].set_title("overlaid image last generation")

    axes[1, 0].plot(generations, average_fitness_per_generation, label='average_fitness', color='b')
    axes[1, 0].plot(generations, best_fitness_per_generation, label='best_fitness', color='r')
    axes[1, 0].set_title("best fitness (red) and average fitness (blue) per generation")

    axes[1, 1].axhline(y=reference_fitness_value, color='r', linestyle='--', label='fitness reference (100%')
    axes[1, 1].plot(generations, average_fitness_per_generation, marker='o', linestyle='-', color='b')
    axes[1, 1].set_title("average fitness over generations with reference value (red)")

    plt.tight_layout()
    plt.show()

def plot_best_fitness_per_run(best_fitness_per_run, final_best_parameter_per_run, average_best_fitness):
    runs = list(range(1, len(final_best_parameter_per_run) + 1))

    plt.figure(figsize=(10, 6))

    plt.plot(runs, best_fitness_per_run, label='best fitness per run', marker='o', color='b')
    plt.axhline(average_best_fitness, color='r', linestyle='--', label=f'average best fitness over runs: {average_best_fitness:.2f}')
    plt.xlabel('run')
    plt.ylabel('best fitness')

    plt.tight_layout()
    plt.show()