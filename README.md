## Ant Simulator - A Genetic Algorithm
Implementation of a genetic algorithm to evolve ants to consume food on various maps. Created as a part of a course on Artificial Intelligence at Yale University.

The `ant_simulator`, `get_map`, and `display_trial` methods were written by course staff, along with a skeleton for the `__main__` which I modified heavily. All other additions are my own unless explicitly indicated, and were based on my understanding of concepts discussed in the course.

### Overview
The program evolves 'ants', represented by 30-digit strings, to move around on a 2D array of 1s and 0s read in from a file. These strings represent 'genomes' which dictate how the ant moves about the space. One move is made per timestep according to the genome. The map itself is a sort of maze: a 1 is meant to indicate the presence of food, while a 0 indicates its abscense. An ant automatically consumes food when it moves on top of the corresponding position, and food is not ever replenished. The goal of the algorithm is to evolve ants which consume the most food in a fixed amonut of time.

According to the assignment specification (and the behavior defined in `ant_simulator`), the 30-digit genomes are grouped into triplets indexed by position (the first three characters are the 0th triplet, the fourth through sixth are the 1st, etc.). These collectively define a finite state machine 'controller' which dictates the behavior of the ant. 

At each timestep, an ant executes two actions based on the value of the current triplet (arbitrarily initialized to the 0th). First, the ant takes some action (either moving forward or rotating) based on the value of the first triplet. Next, it transitions to a new triplet based on the presence or absence of food at the current map position. Specifically, the behavior is (quoted from the assignment specification):

| Index in triplet | Range | Meaning |
|---|---|---|
| 0 | [1,4] | The action that the ant takes upon entering this state, where<br> - 1 = move forward one cell<br> - 2 = rotate clockwise ninety degrees without changing cells<br> - 3 = rotate counter-clockwise ninety degrees without changing cells<br> - 4 = do nothing  |
| 1 | [0,9] | If the ant is in this state and the sensor value is false (there is no food in the square ahead of it), then the ant will transition to the state with the unique identifier indicated by this digit. |
| 2 | [0,9] | If the ant is in this state and the sensor value is true (there is food in the square ahead of it), then the ant will transition to the state with the unique identifier indicated by this digit. |

Ants evolve via a genetic algorithm parameterized by values specified on the command line. For information on how to indicate them, simply run `python3 geneticAlgorithm.py [-h|--help]`.

### Genetic Algorithm
My genetic algorithm operates according the following procedure. For each generation:
1. Compute the fitness of each individual in the generation.
1. Create a new population.
    1. Select a pool of parents using the specified selection strategy.
    1. Create a pool of children by (with some probability) performing crossover on all pairs of parents.
    1. Mutate each child in the child pool (with some probability).
1. Create the next generation by merging the child pool into the population according to the replacement
strategy.

#### Fitness Computation
The fitness of each ant is computed as the amount of food it eats after 200 time steps on the given map. This is done using the provided `ant_simulator` method.

#### Selection
A pool of a specified number of parents (which we’ll denote as _p_ in subsequent sections) is selected from the
population according to the specified selection strategy. The implemented strategies are as follows:
* **Random** — Individuals are chosen randomly from the population.
* **Greedy** — The population is sorted by fitness and the _p_ most-fit individuals are chosen as parents.
* **Roulette** — An empty list of genes (the “wheel”) is created. Each individual in the population is added to the list a number of times equal to the quotient of their fitness and the sum total fitness of all individuals in the generation (rounded down). A random index is chosen from the list _p_ times to select the parent generation.
* **Rank** — An empty list of genes is created. The population is sorted in increasing order of fitness, and each individual in the population is added to the list a number of times equal to their index (plus one) in the sorted population. A random index is chosen from the list _p_ times to select the parent generation.

#### Crossover
For each possible pairing of parents from the parent pool generated previously (a total of _p<sup>2</sup>_ pairings), a pair of children is generated with a specified crossover probability according to the specified crossover strategy. If this probability threshold is not reached, the parents are returned as children instead. The resultant two genomes are added to the pool of children. If crossover occurs, the possible strategies are:
* **Single-Point** — A random index (between 0 and 29, inclusive) is chosen. The first child is a copy of the first parent’s genes up to the index, and the second parent’s genes from the index onwards. The second child is the inverse (second parent’s genes up to the index, first parent’s genes from the index onwards).
* **Two-Point** — Two random indices (between 0 and 29, inclusive) are chosen. The first child is a copy of the first parent’s genes up to the first index, the second parent’s genes from the first index to the second index, and the first parent’s genes from the second index onwards. The second child is the inverse.
* **Uniform** — Uniform crossover with a random (50-50) distribution. For each gene in the genome, a random source parent is chosen. One child gets the first parent’s gene, the other gets the second parent’s gene. This is repeated for all genes in the genome.

#### Mutation
Each child is mutated gene-by-gene. For every gene in the genome, a random replacement is generated with some specified mutation probability (there are _2p<sup>2</sup>_ children in total). The range of possible values for the generated gene depends on the index. If the probability threshold is not reached, the original gene is used.

#### Replacement
The existing population is merged with the pool of children which results after the crossover and mutation steps, according to a specified replacement strategy. For purposes of the discussion below, the population is of size _n_. The implemented replacement strategies are:
* **Random** — _n_ individuals are chosen at random from either the original population or the child pool, each with equal probability.
* **Generational** — _n_ individuals are chosen at random from the child pool.
* **Elitist** — The population is sorted in order of decreasing fitness, and a specified number (_e_) of the most-fit individuals are added to an empty list. A further _n − e_ individuals are chosen at random from the child pool and are added to the list. The new size-_n_ list is returned as the new population.
* **Overlapping** — Syntactic sugar for elitism with a very high _e_. The population is sorted in order of increasing fitness, and a specified number (_o_) of the least-fit individuals are removed from the population. _o_ individuals are chosen at random from the child pool and are added to the list as replacements, which is then returned as the new population.

For more information about how to specify these parameters, run `python3 geneticAlgorithm.py [-h|--help]`.

### Results
After some experimentation, I determined a set of fairly good parameters for the Muir trail map for short-timescale simulations:

| Paremeter | Value |
|---|---|
| Population size | 1000 |
| Number of generations | 200 |
| Parent pool size | 50 |
| Selection strategy | Rank selection |
| Crossover probability | 10% |
| Crossover strategy | Uniform crossover |
| Mutation probability | 5% |
| Replacement strategy | Generational replacement with elitism |
| Elitism factor | 5 |

Other parameters may yield better results, and further experimentation is certainly warranted.

### Analysis
These parameters were chosen after experimenting with various approaches. As one would expect, a larger population allowed for more consistent high-fitness individuals, since it gives the algorithm a bigger “space” of genes to work with. 1000 was selected as a good compromise between size and the incurred hit to runtime. Maximum fitness also tended to correlate with the number of generations, as a larger number of generations gives more time for good solutions to be found. 200 was chosen as a decent compromise between performance and runtime. The size of the parent pool, crossover probability, mutation probability, and the degree of elitism were chosen based on suggestions offered in lecture, and they produced good results, so they were kept.

The selection strategies were chosen based on the comments offered in lecture and on experimentation. Rank selection biases the parent pool towards the fittest parents, allowing for a high quality breeding pool without letting disproportionately high fitness scores completely dominate (as with roulette selection). Uniform crossover allows for more varied children than other strategies. Elitism ensures that the best solutions are never lost, while generational replacement lets the population evolve towards new solutions quickly. In 100 runs, the given parameters were found to give the best average maximum fitness (83) compared with variations of the parameters above.
