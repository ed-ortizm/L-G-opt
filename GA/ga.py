#!/usr/bin/env python3
import numpy as np
from math import sin, pi
import sys
# Plotting
import matplotlib
import matplotlib.pyplot as plt

## From charbonneau1995: GAs in astronomy and astrophysics

# A top-level view of a genetic algorithm is as follows: given a
# target phenotype and a tolerance criterion,
# 1. Construct a random initial population and evaluate the
# fitness of its members.
# 2. Construct a new population by breeding selected individuals
# from the old population.
# 3. Evaluate the fitness of each member of the new population.
# 4. Replace the old population by the new population.
# 5. Test convergence; unless fittest phenotype matches target
# phenotype within tolerance, goto step 2.

## https://github.com/ahmedfgad/GeneticAlgorithmPython/blob/master/ga.py

def fitness(population,nn=1):
    # Fitness value of my individuals
    n= nn
    fitness = np.zeros(population[:,0].size)
    i = 0
    for x in population:
        f = (16*x[0]*(1-x[0])*x[1]*(1-x[1])*sin(n*pi*x[0])*sin(n*pi*x[1]))**2
        fitness[i] = f
        i = i+1
    return fitness
population = np.random.random((5,2))
fitness = fitness(population)
#print(population,fitness(population))
def mating_pool(population, fitness, n_parents):
    parents = np.zeros((n_parents,2))
    for parent in range(n_parents):
        max_fit_idx = np.where(fitness==np.max(fitness))
        # out: (array([val]),)
        max_fit_idx = max_fit_idx[0][0]
        parents[parent,:] = population[max_fit_idx,:]
        # Now we delete that fitness value to make sure we go for the next one
        # during the next iteration
        fitness[max_fit_idx] = -1.
    return parents
#print(population,mating_pool(population,fitness(population),2))
parents = mating_pool(population,fitness,5)
def crossover(parents):
    n_offsprings = parents[:,0].size
    offsprings = np.zeros((n_offsprings*2,2))
    for i in range(n_offsprings):
        # Indexes for the mates
        p1_idx = i% parents.shape[0]
        p2_idx = (i+1)% parents.shape[0]
        p1_x = str(parents[p1_idx][0])[2:10]
        p1_y = str(parents[p1_idx][1])[2:10]
        p1_xy = p1_x + p1_y
        p2_x = str(parents[p2_idx][0])[2:10]
        p2_y = str(parents[p2_idx][1])[2:10]
        p2_xy = p2_x + p2_y
        # I could generate 2 and take the most fit, like one child policy :O
        offsp_1 = p1_xy[0:3] + p2_xy[3:]
        offsp_1_x = float('0.' + offsp_1[:8])
        offsp_1_y = float('0.' + offsp_1[8:])

        offsp_2 = p1_xy[3:] + p2_xy[:3]
        offsp_2_x = float('0.' + offsp_2[:8])
        offsp_2_y = float('0.' + offsp_2[8:])

        offsprings[2*i][0] = offsp_1_x
        offsprings[2*i][1] = offsp_1_y
        offsprings[2*i+1][0] = offsp_2_x
        offsprings[2*i+1][1] = offsp_2_y
    return offsprings
offsprings = crossover(parents)

print(parents,'\n\n',offsprings)
def mutation(offspring_crossover, num_mutations=1):
    pass
