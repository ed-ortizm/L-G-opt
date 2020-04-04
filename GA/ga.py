#!/usr/bin/env python3
import numpy as np
from math import sin, pi
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
def mating_pool(population, fitness, n_parents):
    # Return the individuals sorted from the fittest to the les fittet
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
def crossover(parents):
    n_offsprings = parents[:,0].size
    offsprings = np.zeros((n_offsprings*2+2,2))
    for i in range(n_offsprings+1):
        # Indexes for the mates
        p1_idx = i% parents.shape[0]
        p2_idx = (i+1)% parents.shape[0]
        p1_x = str(parents[p1_idx][0])[2:10]
        p1_y = str(parents[p1_idx][1])[2:10]
        p1_xy = p1_x + p1_y
        p2_x = str(parents[p2_idx][0])[2:10]
        p2_y = str(parents[p2_idx][1])[2:10]
        p2_xy = p2_x + p2_y
        # Offspring 1
        offsp_1 = p1_xy[0:3] + p2_xy[3:]
        offsp_1_x = float('0.' + offsp_1[:8])
        offsp_1_y = float('0.' + offsp_1[8:])
        # Offspring 2
        offsp_2 = p1_xy[3:] + p2_xy[:3]
        offsp_2_x = float('0.' + offsp_2[:8])
        offsp_2_y = float('0.' + offsp_2[8:])
        # Collecting offsprings
        offsprings[2*i][0] = offsp_1_x
        offsprings[2*i][1] = offsp_1_y
        offsprings[2*i+1][0] = offsp_2_x
        offsprings[2*i+1][1] = offsp_2_y
    return offsprings
def mutation(offsprings, num_mutations=1,p_mut=0.01):
    offsprings_mutated = np.zeros(offsprings.shape)
    i = 0
    for offsp in offsprings:
        x = str(offsp[0])[2:]
        if len(x)!=8:
            x = x + '0'
        y = str(offsp[1])[2:]
        if len(y)!=8:
            y = y + '0'
        xy = x+y
        for mutation in range(num_mutations):
            if np.random.random() < p_mut:
                idx = np.random.randint(0,15)
                if idx == 0:
                    xy = str(np.random.randint(0,9)) + xy[1:]
                elif idx == 15:
                    xy = xy[:idx] + str(np.random.randint(0,9))
                else:
                    xy = xy[0:idx] + str(np.random.randint(0,9)) + xy[idx+1:]
        offsprings_mutated[i][0] = float('0.' + xy[:8])
        offsprings_mutated[i][1] = float('0.' + xy[8:])
        i = i+1
    return offsprings_mutated
