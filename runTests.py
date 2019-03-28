import argparse
import os
import subprocess
import datetime
import re
import glob


# Source: https://stackoverflow.com/questions/12116685/how-can-i-require-my-python-scripts-argument-to-be-a-float-between-0-0-1-0-usin
def probability(x):
    x = float(x)
    if x < 0.0 or x > 1.0:
        raise argparse.ArgumentTypeError("%r not in range [0.0, 1.0]" % (x,))
    return x


def run_tests(batch, n, program, args):
    os.mkdir(os.path.join("runs", batch))
    for i in range(n):
        command = []
        command.append("python3")
        command.append(program)
        for arg in args:
            command.append(arg)
        command.append("-o")
        command.append(os.path.join("runs", batch, str(i)))
        os.mkdir(os.path.join("runs", batch, str(i)))
        subprocess.call(command)


def parse_results(batch, n):
    best_run, best_fitness = (-1, -1)
    for i in range(n):
        current_run = os.path.join(os.path.join(
            "runs", batch, str(i), "final_population.txt"))
        if not os.path.exists(current_run):
            continue
        current_file = open(current_run, "r")
        max_fitness = -1
        for line in current_file:
            m = re.search(r"^individual: \d+\s+\|\s+fitness: (\d+)", line)
            fitness = int(m.group(1))
            if fitness > max_fitness:
                max_fitness = fitness
        current_file.close()
        if max_fitness > best_fitness:
            best_fitness = max_fitness
            best_run = i
    return best_run, best_fitness


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", type=int, default=100)
    parser.add_argument("--parse", default=False, action='store_true')
    parser.add_argument("--clean", default=False, action='store_true')
    parser.add_argument("-m", "--map", type=str, default="muir.txt")
    parser.add_argument("-p", "--population-size", type=int, default=10)
    parser.add_argument("-g", "--generations", type=int, default=40)
    parser.add_argument("-ps", "--pool-size", type=int, default=10)
    parser.add_argument("-s", "--selection-strategy", type=str,
                        choices=["rank", "roulette", "greedy"], default="rank")
    parser.add_argument("-cp", "--crossover-probability",
                        type=probability, default=0.1)
    parser.add_argument("-c", "--crossover-strategy", type=str,
                        choices=["single", "dual", "uniform"], default="uniform")
    parser.add_argument("-mp", "--mutation-probability",
                        type=probability, default=0.05)
    parser.add_argument("-r", "--replacement-strategy", type=str, choices=[
                        "generational", "overlapping", "elitist", "random"], default="elitist")
    parser.add_argument("-rf", "--replacement-factor", type=int, default=3)
    args = parser.parse_args()

    batch = str(datetime.datetime.now()).replace(' ', '|')

    arg_list = []
    arg_list.append("-m")
    arg_list.append(str(args.map))
    arg_list.append("-p")
    arg_list.append(str(args.population_size))
    arg_list.append("-g")
    arg_list.append(str(args.generations))
    arg_list.append("-ps")
    arg_list.append(str(args.pool_size))
    arg_list.append("-s")
    arg_list.append(str(args.selection_strategy))
    arg_list.append("-cp")
    arg_list.append(str(args.crossover_probability))
    arg_list.append("-c")
    arg_list.append(str(args.crossover_strategy))
    arg_list.append("-mp")
    arg_list.append(str(args.mutation_probability))
    arg_list.append("-r")
    arg_list.append(str(args.replacement_strategy))
    arg_list.append("-rf")
    arg_list.append(str(args.replacement_factor))

    run_tests(batch, args.n, "geneticAlgorithm.py", arg_list)

    output = open(os.path.join("runs", batch, "_parameters"), "w")
    output.write(" ".join(arg_list))
    output.close()

    if args.parse:
        max_run, max_fitness = parse_results(batch, args.n)
        output = open(os.path.join("runs", batch, "_result"), "w")
        output.write("maximum fitness %d on run %d" % (max_fitness, max_run))
        output.close()
        if args.clean:
            subprocess.call(["rm", os.path.join("runs", batch, "_result")])
            for i in range(args.n):
                if i != max_run:
                    command = ["rm", "-rf"]
                    command.append(os.path.join("runs", batch, str(i)))
                    subprocess.call(command)
                else:
                    for file in glob.glob(os.path.join("runs", batch, str(i), "*")):
                        command = ["mv", file, os.path.join(
                            "runs", batch, file.split('/')[-1])]
                        subprocess.call(command)
                    command = ["rmdir", os.path.join("runs", batch, str(i))]
                    subprocess.call(command)
