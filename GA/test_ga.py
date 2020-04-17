#!/usr/bin/env python3
from ga import *
nn= 21
tolerance = 0.05
n_gens = 5_000
plot = F_plt(nn)
#### From charbonneau1995: GAs in astronomy and astrophysics

## 1. Construct a random initial population and evaluate the fitness of it.
n_parents = 10
population = np.random.random((n_parents,2))
print('Initial population: ', population.shape[0])
fitnesses = fitness(population,nn=nn)
# Now I sort the points according to their fitness measure. Then, I chose app
# the 25% of the fittest individuals to do the breeding
#n_parents = int(0.25 * n_points)
#parents = mating_pool(population,fitnesses,n_parents)
# better obviate this, otherwise population is too homogeneous since the
# beginning and then convergence is compromised (report)
parents = population
# it may happend that I already got the fittest individual for my tolerance.
# Therefore I'm gonna test that
fittest = fitness(parents,nn=nn)
e = abs(1-fittest[0])
if  e < tolerance:
    print('We did it! It took ' + str(0) + ' generations.')
    print(parents[0])
for n_gen in range(n_gens):
    print('gen ', n_gen)
    #print('number of parents: ',parents.shape[0])
    ## 2. Construct a new population by breeding selected individuals from the old
    # population.
    # I breed the individuals
    offsprings = crossover(parents)
    # I allow for any possible mutations
    offsprings = mutation(offsprings, num_mutations=1,p_mut=0.01)
    # I gather all the individuals
    new_pop = np.zeros((parents.shape[0]+offsprings.shape[0],2))
    ## 3. Evaluate the fitness of each member of the new population.
    # The new population consist of paremts and offsprings
    new_pop = np.concatenate((parents,offsprings))
    fitnesses = fitness(new_pop,nn=nn)
    ## 4. Replace the old population by the new population.
    # I keep the number of the initial population to be similar to the first one
    parents = mating_pool(new_pop,fitnesses,n_parents)
    if n_gen in np.arange(0,1100,100): plot.plt2(parents)
    ## 5. Test convergence
    fittest = fitness(parents,nn=nn)
    e = abs(1-fittest[0])
    #unless fittest phenotype matches target
    # phenotype within tolerance, goto step 2.
    if  e < tolerance:
        print('We did it! It took ' + str(n_gen) + ' generations.')
        print(parents[0],fittest[0])
        break
