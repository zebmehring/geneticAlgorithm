#!/usr/bin/env python3

# Please feel free to modify any part of the code or add any functions, or modify the argument of the given functions. But please keep the name of the given functions

# Please feel free to import any libraries you need.

# You are required to finish the genetic_algorithm function, and you may need to complete crossover, mutate and select.

import random
import argparse
import matplotlib.pyplot as plt

# Source: https://stackoverflow.com/questions/12116685/how-can-i-require-my-python-scripts-argument-to-be-a-float-between-0-0-1-0-usin
def probability(x):
    x = float(x)
    if x < 0.0 or x > 1.0:
        raise argparse.ArgumentTypeError("%r not in range [0.0, 1.0]"%(x,))
    return x

def rank_selection(population, fitnesses, pool_size):
    parent_population = []
    ranks = []
    fitnesses = sorted(fitnesses, key=lambda t: t[0])
    for rank in range(len(population)):
        for repeat in range(rank + 1):
            ranks.append(population[fitnesses[rank][1]])
    for parent in range(pool_size):
        r = random.randint(0, len(ranks) - 1)
        parent_population.append(ranks[r])
    return parent_population


def roulette_selection(population, fitnesses, pool_size):
    parent_population = []
    wheel = []
    fitnesses = sorted(fitnesses, key=lambda t: t[0])
    total_fitness = sum(map(lambda t: t[0], fitnesses))
    for parent in range(len(population)):
        for repeat in range(fitnesses[parent][0] // total_fitness):
            wheel.append(population[fitnesses[parent][1]])
    for parent in range(pool_size):
        r = random.randint(0, len(wheel) - 1)
        parent_population.append(wheel[r])
    return parent_population


def greedy_selection(population, fitnesses, pool_size):
    parent_population = []
    fitnesses = sorted(fitnesses, key=lambda t: t[0])
    fitnesses.reverse()
    for parent in range(pool_size):
        parent_population.append(population[fitnesses[parent][1]])
    return parent_population


def select(population, fitnesses, pool_size, selection_strategy):
    return selection_strategy(population, fitnesses, pool_size)


def single_point_crossover(parent1, parent2, crossover_probability):
    if random.random() > crossover_probability:
        return (parent1, parent2)
    r = random.randint(0, min(len(parent1), len(parent2)))
    return (parent1[:r] + parent2[r:], parent2[:r] + parent1[r:])


def two_point_crossover(parent1, parent2, crossover_probability):
    if random.random() > crossover_probability:
        return (parent1, parent2)
    r1 = random.randint(0, min(len(parent1), len(parent2)))
    r2 = random.randint(r1, min(len(parent1), len(parent2)))
    return (parent1[:r1] + parent2[r1:r2] + parent1[r2:], parent2[:r1] + parent1[r1:r2] + parent2[r2:])


def uniform_crossover(parent1, parent2, crossover_probability):
    if random.random() <= 0.5:
        current_parent = parent1
        _current_parent = parent2
    else:
        current_parent = parent2
        _current_parent = parent1
    child1 = ""
    child2 = ""
    for gene in range(min(len(parent1), len(parent2))):
        if random.random() <= crossover_probability:
            if current_parent == parent1:
                current_parent = parent2
                _current_parent = parent1
            else:
                current_parent = parent1
                _current_parent = parent2
        child1 += current_parent[gene]
        child2 += _current_parent[gene]
    return (child1, child2)


def crossover(parent_population, crossover_probability, crossover_strategy):
    child_population = []
    for parent1 in parent_population:
        for parent2 in parent_population:
            if parent1 != parent2:
                child1, child2 = crossover_strategy(parent1, parent2, crossover_probability)
                child_population.append(child1)
                child_population.append(child2)
    return child_population


def mutate(individual, mutation_probability):
    mutated = ""
    for index, gene in enumerate(individual):
        if random.random() <= mutation_probability:
            if index % 3:
                mutated += str(random.randint(0, 9))
            else:
                mutated += str(random.randint(1, 4))
        else:
            mutated += gene
    return mutated


def generational_replacement(old_population, child_population, fitnesses, *args):
    new_population = []
    for chromosome in range(len(old_population)):
        r = random.randint(0, len(child_population) - 1)
        new_population.append(child_population[r])
        # child_population.pop(r)
    return new_population


def overlapping_replacement(old_population, child_population, fitnesses, culling_factor):
    new_population = []
    for chromosome in range(culling_factor):
        min_fitness = min(fitnesses, key=lambda t: t[0])
        old_population.pop(min_fitness[1])
        fitnesses.pop(fitnesses.index(min_fitness))
    for chromosome in old_population:
        new_population.append(chromosome)
    for chromosome in range(culling_factor):
        r = random.randint(0, len(child_population) - 1)
        new_population.append(child_population[r])
        # child_population.pop(r)
    return new_population


def elitist_replacement(old_population, child_population, fitnesses, elitism_factor):
    new_population = []
    for chromosome in range(elitism_factor):
        max_fitness = max(fitnesses, key=lambda t: t[0])
        new_population.append(old_population[max_fitness[1]])
        fitnesses.pop(fitnesses.index(max_fitness))
    for chromosome in range(elitism_factor, len(old_population)):
        r = random.randint(0, len(child_population) - 1)
        new_population.append(child_population[r])
        # child_population.pop(r)
    return new_population


def random_replacement(old_population, child_population, fitnesses, *args):
    new_population = []
    for i in range(len(old_population)):
        if random.random() <= 0.5:
            new_population.append(old_population[i])
        else:
            r = random.randint(0, len(child_population) - 1)
            new_population.append(child_population[r])
            # child_population.pop(r)
    return new_population


def replace(old_population, child_population, fitnesses, replacement_strategy, replacement_factor):
    return replacement_strategy(old_population, child_population, fitnesses, replacement_factor)


def converges(population):
    for individual in range(len(population)):
        if population[individual] != population[0]:
            return False
    return True


def genetic_algorithm(population, food_map_file_name, generations, selection_strategy, pool_size, crossover_probability, crossover_strategy, mutation_probability, replacement_strategy, replacement_factor):
    stats = []
    food_map, map_size = get_map(food_map_file_name)
    convergence_stats = []

    for generation in range(generations):
        # check for stagnation
        if converges(population): 
            return max_fitness, max_individual, max_trial, stats, population
        else:
            convergence_stats.append((len(set(population)), len(set(population)) / len(population)))

        # step 2: evaluate the fitness of each chromosome
        fitnesses = []
        trials = []
        for index, individual in enumerate(population):
            trial, fitness = ant_simulator(food_map, map_size, individual)
            fitnesses.append((fitness, index))
            trials.append(trial)
        stats.append((max(fitnesses, key=lambda t: t[0])[0], min(fitnesses, key=lambda t: t[0])[0], sum(map(lambda t: t[0], fitnesses)) / len(population)))
        max_fitness, max_index = max(fitnesses, key=lambda t: t[0])
        max_individual = population[max_index]
        max_trial = trials[max_index]

        # step 3: create a new population
        parent_population = select(population, fitnesses, pool_size, selection_strategy)
        child_population = crossover(parent_population, crossover_probability, crossover_strategy)
        for child in child_population:
            child = mutate(child, mutation_probability)

        # step 4: merge populations into a new poulation
        population = replace(population, child_population, fitnesses, replacement_strategy, replacement_factor)

    return max_fitness, max_individual, max_trial, stats, population, convergence_stats


def initialize_population(num_population):
    population = []
    for i in range(num_population):
        individual = ""
        for g in range(30):
            if g % 3:
                individual += str(random.randint(0, 9))
            else:
                individual += str(random.randint(1, 4))
        population.append(individual)
    return population


def ant_simulator(food_map, map_size, ant_genes):
    """
    parameters:
        food_map: a list of list of strings, representing the map of the environment with food
            "1": there is a food at the position
            "0": there is no food at the position
            (0, 0) position: the top left corner of the map
            (x, y) position: x is the row, and y is the column
        map_size: a list of int, the dimension of the map. It is in the format of [row, column]
        ant_genes: a string with length 30. It encodes the ant's genes, for more information, please refer to the handout.

    return:
        trial: a list of list of strings, representing the trials
            1: there is food at that position, and the spot was not visited by the ant
            0: there is no food at that position, and the spot was not visited by the ant
            empty: the spot has been visited by the ant

    It takes in the food_map and its dimension of the map and the ant's gene information, and return the trial in the map
    """

    step_time = 200

    trial = []
    for i in food_map:
        line = []
        for j in i:
            line.append(j)
        trial.append(line)

    position_x, position_y = 0, 0
    # face down, left, up, right
    orientation = [(1, 0), (0, -1), (-1, 0), (0, 1)]
    fitness = 0
    state = 0
    orientation_state = 3
    gene_list = [ant_genes[i: i + 3] for i in range(0, len(ant_genes), 3)]

    for i in range(step_time):
        if trial[position_x][position_y] == "1":
            fitness += 1
        trial[position_x][position_y] = " "

        sensor_x = (
            position_x + orientation[orientation_state][0]) % map_size[0]
        sensor_y = (
            position_y + orientation[orientation_state][1]) % map_size[1]
        sensor_result = trial[sensor_x][sensor_y]

        if sensor_result == "1":
            state = int(gene_list[state][2])
        else:
            state = int(gene_list[state][1])

        action = gene_list[state][0]

        if action == "1":  # move forward
            position_x = (
                position_x + orientation[orientation_state][0]) % map_size[0]
            position_y = (
                position_y + orientation[orientation_state][1]) % map_size[1]
        elif action == "2":  # turn right
            orientation_state = (orientation_state + 1) % 4
        elif action == "3":  # turn left
            orientation_state = (orientation_state - 1) % 4
        elif action == "4":  # do nothing
            pass
        else:
            raise Exception("invalid action number!")

    return trial, fitness


def get_map(file_name):
    """
    parameters:
        file_name: a string, the name of the file which stored the map. The first line of the map is the dimension (row, column), the rest is the map
            1: there is a food at the position
            0: there is no food at the position

    return:
        food_map: a list of list of strings, representing the map of the environment with food
            "1": there is a food at the position
            "0": there is no food at the position
            (0, 0) position: the top left corner of the map
            (x, y) position: x is the row, and y is the column
        map_size: a list of int, the dimension of the map. It is in the format of [row, column]

    It takes in the file_name of the map, and return the food_map and the dimension map_size
    """
    food_map = []
    map_file = open(file_name, "r")
    first_line = True
    map_size = []

    for line in map_file:
        line = line.strip()
        if first_line:
            first_line = False
            map_size = line.split()
            continue
        if line:
            food_map.append(line.split())

    map_file.close()
    return food_map, [int(i) for i in map_size]


def display_trials(trials, target_file):
    """
    parameters:
        trials: a list of list of strings, representing the trials
            1: there is food at that position, and the spot was not visited by the ant
            0: there is no food at that position, and the spot was not visited by the ant
            empty: the spot has been visited by the ant
        taret_file: a string, the name the target_file to be saved

    It takes in the trials, and target_file, and saved the trials in the target_file. You can open the target_file to take a look at the ant's trial.
    """
    trial_file = open(target_file, "w")
    for line in trials:
        trial_file.write(" ".join(line))
        trial_file.write("\n")
    trial_file.close()


def display_population(population, target_file, food_map_file_name):
    food_map, map_size = get_map(food_map_file_name)
    population_file = open(target_file, "w")
    for individual in population:
        fitness = ant_simulator(food_map, map_size, individual)[1]
        population_file.write("individual: %s" % (individual))
        population_file.write("\t|\t")
        population_file.write("fitness: %d" % (fitness))
        population_file.write("\n")
    population_file.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a genetic algorithm")
    parser.add_argument("-m", "--map", type=str, 
                        help="specify the file from which to read in the map", default="muir.txt")
    parser.add_argument("-p", "--population-size", type=int, 
                        help="specify the population size", default=10)
    parser.add_argument("-g", "--generations", type=int,
                        help="specify the number of generations", default=40)
    parser.add_argument("-ps", "--pool-size", type=int,
                        help="specify the number of parents to generate", default=10)
    parser.add_argument("-s", "--selection-strategy", type=str, choices=["rank", "roulette", "greedy"],
                        help="specify the selection strategy to use", default="rank")
    parser.add_argument("-cp", "--crossover-probability", type=probability,
                        help="specify the probability of crossover for each parent pair", default=0.1)
    parser.add_argument("-c", "--crossover-strategy", type=str, choices=["single", "dual", "uniform"],
                        help="specify the crossover strategy to use", default="uniform")
    parser.add_argument("-mp", "--mutation-probability", type=probability,
                        help="specify the probability of mutation for each child", default=0.05)
    parser.add_argument("-r", "--replacement-strategy", type=str, choices=["generational", "overlapping", "elitist", "random"],
                        help="specify the replacement strategy to use", default="elitist")
    parser.add_argument("-rf", "--replacement-factor", type=int,
                        help="specify the number of individuals to replace/keep for overlapping/elitist selection", default=3)
    args = parser.parse_args()

    food_map_file_name = args.map
    population_size = args.population_size
    generations = args.generations
    pool_size = args.pool_size
    crossover_probability = args.crossover_probability
    mutation_probability = args.mutation_probability
    replacement_factor = args.replacement_factor

    if args.selection_strategy == "rank":
        selection_strategy = rank_selection
    elif args.selection_strategy == "roulette":
        selection_strategy = roulette_selection
    elif args.selection_strategy == "greedy":
        selection_strategy = greedy_selection

    if args.crossover_strategy == "single":
        crossover_strategy = single_point_crossover
    elif args.crossover_strategy == "dual":
        crossover_strategy = two_point_crossover
    elif args.crossover_strategy == "uniform":
        crossover_strategy = uniform_crossover

    if args.replacement_strategy == "generational":
        replacement_strategy = generational_replacement
    elif args.replacement_strategy == "overlapping":
        replacement_strategy = overlapping_replacement
    elif args.replacement_strategy == "elitist":
        replacement_strategy = elitist_replacement
    elif args.replacement_strategy == "random":
        replacement_strategy = random_replacement

    if population_size < 0:
        raise Exception("population_size (%r) must be a positive number" % (population_size))

    if generations < 0:
        raise Exception("generations (%r) must be a positive number" % (generations))

    if pool_size > population_size:
        raise Exception("pool size (%r) cannot exceed population_size (%r)" % (pool_size, population_size))

    if (selection_strategy == elitist_replacement or selection_strategy == overlapping_replacement) and replacement_factor > population_size:
        raise Exception("replacement_factor (%r) cannot exceed population_size (%r)" % (replacement_factor, population_size))

    population = initialize_population(population_size)
    display_population(population, "initial_population.txt", food_map_file_name)

    max_fitness, max_individual, max_trial, stats, population, convergence_stats = genetic_algorithm(population, food_map_file_name, generations, selection_strategy, pool_size, crossover_probability, crossover_strategy, mutation_probability, replacement_strategy, replacement_factor)

    display_trials(max_trial, "max_trial.txt")
    display_population(population, "final_population.txt", food_map_file_name)

    plt.figure(1)
    plt.plot([i for i in range(len(stats))], [i[0] for i in stats], marker = "o", color = "green", label = "most fit individual")
    plt.plot([i for i in range(len(stats))], [i[1] for i in stats], marker = "o", color = "red", label = "least fit individual")
    plt.plot([i for i in range(len(stats))], [i[2] for i in stats], marker = "o", color = "blue", label = "average fitness")
    plt.xlabel("generation")
    plt.xlim((0, generations))
    plt.ylim((0, max(i[0] for i in stats) + 10))
    plt.ylabel("fitness")
    plt.legend()
    plt.savefig("fitness of each generation.png")

    plt.figure(2)
    muir_fitness = []
    santafe_fitness = []
    muir_food_map, muir_map_size = get_map("muir.txt")
    santafe_food_map, santafe_map_size = get_map("santafe.txt")
    for individual in population:
        trial, individual_muir_fitness = ant_simulator(muir_food_map, muir_map_size, individual)
        trial, individual_santafe_fitness = ant_simulator(santafe_food_map, santafe_map_size, individual)
        muir_fitness.append(individual_muir_fitness)
        santafe_fitness.append(individual_santafe_fitness)

    plt.plot([i for i in range(len(muir_fitness))], muir_fitness, marker = "o", color = "blue", label = "muir", linestyle = "None")
    plt.plot([i for i in range(len(santafe_fitness))], santafe_fitness, marker = "o", color = "green", label = "santa fe", linestyle = "None")
    plt.xlabel("individuals in the last generation")
    plt.xlim((0, population_size))
    plt.ylim((0, max(muir_fitness + santafe_fitness) + 10))
    plt.ylabel("fitness")
    plt.legend()
    plt.savefig("fitness of last generation on muir and santa fe map.png")

    plt.figure(3)
    plt.plot([i for i in range(generations)], [i[0] for i in convergence_stats], marker = "o", color = "blue")
    plt.xlabel("generation")
    plt.xlim((0, generations))
    plt.ylim((0, max(i[0] for i in convergence_stats) + 10))
    plt.ylabel("unique individuals")
    plt.savefig("unique individuals.png")

    plt.figure(3)
    plt.plot([i for i in range(generations)], [i[1] for i in convergence_stats], marker = "o", color = "green")
    plt.xlabel("generation")
    plt.xlim((0, generations))
    plt.ylim((0, 1))
    plt.ylabel("convergence ratio")
    plt.savefig("convergence ratios.png")

    # Example of how to use get_map, ant_simulator and display trials function
    """
    food_map, map_size = get_map("muir.txt")
    ant_genes = "335149249494173115455311387263"
    trial, fitness = ant_simulator(food_map, map_size, ant_genes)
    display_trials(trial, "trial.txt")
    """
