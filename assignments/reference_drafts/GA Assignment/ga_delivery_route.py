"""
GA Assignment - Genetic Algorithm for Delivery Route Optimisation
(REFERENCE DRAFT - rewrite and understand before submitting)

A van starts at depot A, visits B, C, D, E, F and G exactly once, and returns to A.
The genetic algorithm searches for the visiting order with the shortest total distance.

Chromosome : an ordering of the six delivery stops, e.g. ["B", "E", "D", "C", "F", "G"]
Route      : A + chromosome + A  ->  A-B-E-D-C-F-G-A
Fitness    : total route distance in km (smaller is better)

Run:  python ga_delivery_route.py
"""
import random
from itertools import permutations

# ---------------------------------------------------------------------------
# Dataset: symmetric distance matrix in kilometres (from the assignment sheet)
# ---------------------------------------------------------------------------
LOCATIONS = ["A", "B", "C", "D", "E", "F", "G"]
DISTANCE_MATRIX = [
    # A  B  C  D  E  F  G
    [0, 4, 8, 6, 7, 9, 5],  # A (depot)
    [4, 0, 5, 7, 3, 8, 6],  # B
    [8, 5, 0, 4, 6, 3, 7],  # C
    [6, 7, 4, 0, 5, 6, 8],  # D
    [7, 3, 6, 5, 0, 4, 6],  # E
    [9, 8, 3, 6, 4, 0, 5],  # F
    [5, 6, 7, 8, 6, 5, 0],  # G
]
# Look-up table so we can write DISTANCE["A"]["B"] instead of matrix indices.
DISTANCE = {a: {b: DISTANCE_MATRIX[i][j] for j, b in enumerate(LOCATIONS)}
            for i, a in enumerate(LOCATIONS)}

DEPOT = "A"
STOPS = ["B", "C", "D", "E", "F", "G"]

# ---------------------------------------------------------------------------
# GA settings
# ---------------------------------------------------------------------------
SEED = 42                # fixed seed -> the same result every run (reproducible)
POPULATION_SIZE = 30
GENERATIONS = 60         # the assignment asks for at least 50
TOURNAMENT_SIZE = 3
CROSSOVER_RATE = 0.9
MUTATION_RATE = 0.2
ELITE_COUNT = 2


# ---------------------------------------------------------------------------
# Fitness
# ---------------------------------------------------------------------------
def full_route(chromosome):
    """Add the depot at the start and the end: [B, E, ...] -> [A, B, E, ..., A]."""
    return [DEPOT] + list(chromosome) + [DEPOT]


def route_distance(chromosome):
    """Total km of the route, including the return journey to the depot."""
    route = full_route(chromosome)
    return sum(DISTANCE[route[i]][route[i + 1]] for i in range(len(route) - 1))


# ---------------------------------------------------------------------------
# GA operators
# ---------------------------------------------------------------------------
def create_population(size, rng):
    """Random valid routes: each one is a shuffled copy of the six stops."""
    population = []
    for _ in range(size):
        chromosome = STOPS[:]
        rng.shuffle(chromosome)
        population.append(chromosome)
    return population


def tournament_selection(population, rng):
    """Pick TOURNAMENT_SIZE random routes and return the shortest one as a parent."""
    contestants = rng.sample(population, TOURNAMENT_SIZE)
    return min(contestants, key=route_distance)


def order_crossover(parent1, parent2, rng):
    """Order crossover (OX): always gives a valid route with no repeated or missing stop.

    1. Copy a random slice of parent1 into the child at the same positions.
    2. Fill the empty positions, left to right, with the remaining stops in the
       order they appear in parent2.
    Example: parent1 = B C D E F G, slice = positions 2-3 (D E)
             parent2 = G F E D C B  -> remaining in parent2 order: G F C B
             child   = G F D E C B
    """
    size = len(parent1)
    start, end = sorted(rng.sample(range(size), 2))
    child = [None] * size
    child[start:end + 1] = parent1[start:end + 1]

    remaining = [stop for stop in parent2 if stop not in child]
    for i in range(size):
        if child[i] is None:
            child[i] = remaining.pop(0)
    return child


def swap_mutation(chromosome, rng):
    """With probability MUTATION_RATE, swap two stops. The route stays valid."""
    chromosome = chromosome[:]
    if rng.random() < MUTATION_RATE:
        i, j = rng.sample(range(len(chromosome)), 2)
        chromosome[i], chromosome[j] = chromosome[j], chromosome[i]
    return chromosome


# ---------------------------------------------------------------------------
# The GA loop
# ---------------------------------------------------------------------------
def run_ga(seed=SEED, verbose=True):
    rng = random.Random(seed)
    population = create_population(POPULATION_SIZE, rng)
    best = min(population, key=route_distance)

    for generation in range(1, GENERATIONS + 1):
        # Elitism: the best routes go into the next generation unchanged.
        ranked = sorted(population, key=route_distance)
        next_population = [route[:] for route in ranked[:ELITE_COUNT]]

        # Fill the rest of the new population with children.
        while len(next_population) < POPULATION_SIZE:
            parent1 = tournament_selection(population, rng)
            parent2 = tournament_selection(population, rng)
            if rng.random() < CROSSOVER_RATE:
                child = order_crossover(parent1, parent2, rng)
            else:
                child = parent1[:]
            next_population.append(swap_mutation(child, rng))

        population = next_population
        generation_best = min(population, key=route_distance)
        if route_distance(generation_best) < route_distance(best):
            best = generation_best

        if verbose and (generation == 1 or generation % 10 == 0):
            print(f"Generation {generation:3d}: best distance so far = {route_distance(best)} km")

    return best


def brute_force_best():
    """Check every one of the 6! = 720 routes (only possible because the problem is tiny)."""
    return min(route_distance(p) for p in permutations(STOPS))


def main():
    print("Genetic Algorithm - delivery route optimisation")
    print(f"Population {POPULATION_SIZE}, generations {GENERATIONS}, seed {SEED}, "
          f"tournament {TOURNAMENT_SIZE}, crossover {CROSSOVER_RATE}, "
          f"mutation {MUTATION_RATE}, elites {ELITE_COUNT}\n")

    best = run_ga()
    print("\nBest route    :", " -> ".join(full_route(best)))
    print("Total distance:", route_distance(best), "km")

    # Extra checks for the evaluation (Part D)
    print("\nCheck: shortest possible distance (all 720 routes tried):", brute_force_best(), "km")
    print("Seed sensitivity (same settings, seeds 1-10):")
    for seed in range(1, 11):
        result = run_ga(seed, verbose=False)
        print(f"  seed {seed:2d}: {route_distance(result)} km  {' -> '.join(full_route(result))}")


if __name__ == "__main__":
    main()

