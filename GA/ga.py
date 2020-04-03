#!/usr/bin/env python3
import numpy as np
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

def breeding(population, fitness, n_parents):
    pass
def crossover(parents,offpring_size):
    pass
def mutation(offspring_crossover, num_mutations=1)
