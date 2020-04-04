#!/usr/bin/env python3
from ga import *
tolerance = 0.01
n_gens = 100
#### From charbonneau1995: GAs in astronomy and astrophysics

## 1. Construct a random initial population and evaluate the fitness of it.
n_points = 100
population = np.random.random((n_points,2))
fitnesses = fitness(population)
# Now I sort the points according to their fitness measure. Then, I chose app
# the 25% of the fittest individuals to do the breeding
n_parents = int(0.25 * n_points)
parents = mating_pool(population,fitnesses,n_parents)
# it may happend that I already got the fittet individual for my tolerance.
# Therefore I'm gonna test that
fittest = fitness(parents)
e = abs(1-fittest[0])
if  e < tolerance:
    print('We did it! It took ' + str(0) + ' generations.')
    print(parents[0])
for n_gen in range(n_gens):
    ## 2. Construct a new population by breeding selected individuals from the old
    # population.
    # I breed the individuals
    offsprings = crossover(parents)
    # I allow for any possible mutations
    offsprings = mutation(offsprings, num_mutations=1,p_mut=0.01)

    ## 3. Evaluate the fitness of each member of the new population.
    # The new population consist of paremts and offsprings
    popuplation = np.concatenate((parents,offsprings))
    fitnesses = fitness(population)

    ## 4. Replace the old population by the new population.
    # I keep the number of the initial population to be similar to the first one
    parents = mating_pool(population,fitnesses,n_parents)

    ## 5. Test convergence
    fittest = fitness(parents)
    e = abs(1-fittest[0])
    if  e < tolerance:
        print('We did it! It took ' + str(n_gen) + ' generations.')
        print(parents[0])
        break
    else:
        print(n_gen,'\t',parents[0])

#unless fittest phenotype matches target
# phenotype within tolerance, goto step 2.
