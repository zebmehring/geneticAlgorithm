import matplotlib.pyplot as plt
import os
import re


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


if __name__ == "__main__":
    current_file = open("../final_population.txt", "r")
    population = []
    population_size = 1000
    for line in current_file:
        m = re.search(r"^individual: (\d+)\s+\|\s+fitness: (\d+)", line)
        individual = m.group(1)
        population.append(individual)
    current_file.close()
    plt.figure(1)
    muir_fitness = []
    santafe_fitness = []
    muir_food_map, muir_map_size = get_map("muir.txt")
    santafe_food_map, santafe_map_size = get_map("santafe.txt")
    for individual in population:
        trial, individual_muir_fitness = ant_simulator(
            muir_food_map, muir_map_size, individual)
        trial, individual_santafe_fitness = ant_simulator(
            santafe_food_map, santafe_map_size, individual)
        muir_fitness.append(individual_muir_fitness)
        santafe_fitness.append(individual_santafe_fitness)
    plt.plot([i for i in range(len(muir_fitness))], muir_fitness,
             marker="o", color="blue", label="muir")
    plt.plot([i for i in range(len(santafe_fitness))], santafe_fitness,
             marker="o", color="green", label="santa fe")
    plt.xlabel("individual in the last generation")
    plt.xlim((0, population_size))
    plt.ylim((0, max(muir_fitness + santafe_fitness) + 10))
    plt.ylabel("fitness")
    plt.legend()
    plt.savefig("muir_vs_santa_fe.png")

    print("average muir fitness: %f" % (sum(muir_fitness) / len(muir_fitness)))
    print("average santa fe fitness: %f" %
          (sum(santafe_fitness) / len(santafe_fitness)))
