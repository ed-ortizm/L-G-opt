#!/usr/bin/env python3
from ga import *
population = np.random.random((5,2))
fitness = fitness(population)
#print(population,fitness(population))
#print(population,mating_pool(population,fitness(population),2))
parents = mating_pool(population,fitness,5)
offsprings = crossover(parents)
#print(parents,'\n\n',offsprings)
offsprings_mutated = mutation(offsprings)
print('\n',offsprings,'\n\n', offsprings_mutated)
diff = offsprings - offsprings_mutated
print('\n', diff)
