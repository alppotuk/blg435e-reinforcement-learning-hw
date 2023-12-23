import random
from DistanceDict import DistanceDict

class TSPGA: # short for Travelling Salesmen Problem Genetic Algorithm
    def __init__(self, cities, population_size=50, crossover_probability=0.8, mutation_probability=0.8, elitism_rate=0.1, advanced_functions= False):
        self.cities = cities
        self.population_size = population_size
        self.crossover_probability = crossover_probability
        self.mutation_probability = mutation_probability
        self.elitism_rate = elitism_rate
        self.advanced_functions = advanced_functions
        self.distance_dict = DistanceDict(cities).distance_dict 

    def generate_random_solution(self): # returns a random generated solution
        return random.sample(self.cities, len(self.cities))

    def calculate_total_distance(self, solution): # returns total distance score of given solution 
        total_distance = 0
        for i in range(len(solution) - 1):
            total_distance += self.distance_dict[solution[i]][solution[i + 1]]
        total_distance += self.distance_dict[solution[-1]][solution[0]]  # complete the loop
        return total_distance

    def GA(self, generations=100): # Genetic Algorithm
        population = [self.generate_random_solution() for _ in range(self.population_size)]

        for generation in range(generations):
            print(f"Generations done: {generation}/{generations}", end="\r", flush=True)
            population.sort(key = lambda  x: self.calculate_fitness(x), reverse=True) # sort solutions based on their fitness scores (descending)

            # select parents for crossover based on their fitness scores
            # onyl bests will mate to produce offsprings (survival of the fittest)
            parents = population[:self.population_size//2] 
            child_n = int(self.population_size * (1 - self.elitism_rate) )
            new_generation = population[:(self.population_size- child_n)] # initiate new generation
            for i in range(child_n): 
                parent1, parent2 = random.sample(parents, 2)
                crossover_factor = random.random()
                mutation_factor = random.random()
                child = parent1
                if crossover_factor < self.crossover_probability: 
                    child = self.advanced_crossover(parent1, parent2)  if self.advanced_functions else  self.crossover(parent1, parent2)
                if mutation_factor < self.mutation_probability:
                    child = self.advanced_mutate(child) if self.advanced_functions else self.mutate(child)
                new_generation.append(child)
            population = new_generation # replace new generation for loop
            best_solution = max(population, key=self.calculate_fitness) # max fitness score -> min total distance 

        best_solution = max(population, key=self.calculate_fitness) # get best solution after generations  
        print(f"best solution found after {generation + 1} generations : {' '.join(str(city.index) for city in best_solution)} with total distance of {self.calculate_total_distance(best_solution)}")
        return best_solution
    
    def calculate_fitness(self, solution): # fitness (of a solution) = 1 / total distance => maxing fitness score -> minimizing total distance
        return 1 / self.calculate_total_distance(solution)

    def crossover(self, parent1, parent2): # one point crossover
        # random crossover function 
        crossover_point = random.randint(1, len(parent1) - 1)
        child = parent1[:crossover_point] + [city for city in parent2 if city not in parent1[:crossover_point]]
        return child 
    
    def advanced_crossover(self, parent1, parent2): # two point crossover
        crossover_point_1 , crossover_point_2 = sorted(random.sample(range(1, len(parent1) - 1), 2))
        segment1 = parent1[:crossover_point_1]
        segment3 = parent1[crossover_point_2:]
        segment2 = [city for city in (set(parent2) - set(segment1 + segment3))]
        child = segment1 + segment2 + segment3
        return child

    def mutate(self, solution): # mutation with two random mutation points (exchanging)
        mutation_point1, mutation_point2 = random.sample(range(len(solution)), 2)
        solution[mutation_point1], solution[mutation_point2] = solution[mutation_point2], solution[mutation_point1]
        return solution
    
    def advanced_mutate(self, solution): # mutates 20% of the solution
        num_cities_to_shuffle = max(1, int(len(solution) * 0.2)) 
        indices_to_shuffle = random.sample(range(len(solution)), num_cities_to_shuffle)
        shuffled_subset = [solution[i] for i in indices_to_shuffle]
        random.shuffle(shuffled_subset)
        # replace the old values with the shuffled ones 
        for i, index in enumerate(indices_to_shuffle):
            solution[index] = shuffled_subset[i]        
        return solution

    def run_algorithm(self, algorithm_version,  generations=100): # sets parameters based on selected mode and runs the algorithm
        print(f"GA started with mode: {algorithm_version.capitalize()}")

        if algorithm_version == "basic": # no elitism no advanced functions
            self.elitism_rate = 0 
            solution = self.GA(generations= generations)
            return (solution,  self.calculate_total_distance(solution))
        elif algorithm_version == "elitism": # yes elitism no advanced functions
            self.elitism_rate = 0.1 
            solution = self.GA(generations=generations)
            return (solution,  self.calculate_total_distance(solution))
        elif algorithm_version == "advanced": # more elitism yes advanced functions
            self.advanced_functions = True
            self.elitism_rate = 0.15
            solution = self.GA(generations=generations)
            return (solution,  self.calculate_total_distance(solution))
        else:
            raise ValueError("Avaible algorithm options: 'basic', 'elitism', 'advanced'")