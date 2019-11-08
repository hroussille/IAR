import cma
import gym, gym_fastsim
from deap import *
import numpy as np
from fixed_structure_nn_numpy import SimpleNeuralControllerNumpy
from scipy.spatial import KDTree

from deap import algorithms
from deap import base
from deap import benchmarks
from deap import creator
from deap import tools

import array
import random
import operator
import math

from plot import *

from scoop import futures

from novelty_search import *

MIN_VALUE = -1
MAX_VALUE = 1
MIN_STRATEGY = -10
MAX_STRATEGY = 10

def eval_nn(genotype, nbstep=2000, dump=False, render=False, name=""):
    nn=SimpleNeuralControllerNumpy(5,2,2,10)
    nn.set_parameters(genotype)
    observation = env.reset()
    old_pos=None
    total_dist=0
    fit = 0

    if (dump):
        f=open("traj"+name+".log","w")
    for t in range(nbstep):
        if render:
            env.render()
        action=nn.predict(observation)
        observation, reward, done, info = env.step(action)
        pos=info["robot_pos"][:2]
        if(dump):
            f.write(" ".join(map(str,pos))+"\n")
        if (old_pos is not None):
            d=math.sqrt((pos[0]-old_pos[0])**2+(pos[1]-old_pos[1])**2)
            total_dist+=d
        old_pos=list(pos)
        if(done):
            break
    if (dump):
        f.close()
    dist_obj=info["dist_obj"]
    #print("End of eval, total_dist=%f"%(total_dist))

    if done:
        fit = 1

    #return (math.sqrt((pos[0]-env.goalPos[0])**2+(pos[1]-env.goalPos[1])**2), pos)
    return (fit, pos)

nn=SimpleNeuralControllerNumpy(5,2,2,10)
center=nn.get_parameters()

## Il vous est recommandé de gérer les différentes variantes avec cette variable. Les 3 valeurs possibles seront:
## "FIT+NS": expérience multiobjectif avec la fitness et la nouveauté (NSGA-2)
## "NS": nouveauté seule
## "FIT": fitness seule
## pour les variantes avec un seul objectif, vous pourrez, au choix, utiliser CMA-ES ou NSGA-2 avec un seul objectif,
## il vous est cependant recommandé d'utiliser NSGA-2 car cela limitera la différence entre les variantes et cela 
##vous fera gagner du temps pour la suite.
variant="FIT"

# votre code contiendra donc des tests comme suit pour gérer la différence entre ces variantes:
if (variant=="FIT+NS"):
    pass ## A remplacer par les instructions appropriées
elif (variant=="FIT"):
    pass ## A remplacer par les instructions appropriées
elif (variant=="NS"):
    pass ## A remplacer par les instructions appropriées
else:
    print("Variante inconnue: "+variant)

def eval_fit(x):
    return eval_nn(x)

def eval_fit_ns(x):
    return eval_nn(x)

def eval_ns(x):
    return eval_nn(x)

if (variant=="FIT+NS"):
    creator.create("FitnessMin", base.Fitness, weights=(1.0, 1.0))
elif (variant=="FIT"):
    creator.create("FitnessMin", base.Fitness, weights=(1.0,))
elif (variant=="NS"):
    creator.create("FitnessMin", base.Fitness, weights=(1.0,))
else:
    print("Variante inconnue: "+variant)


creator.create("Individual", array.array, typecode="d", fitness=creator.FitnessMin, strategy=None)
creator.create("Strategy", array.array, typecode="d")

# Génération d'un individu avec une distribution uniforme dans les bornes indiquées
def generateES(icls, scls, size, imin, imax, smin, smax):
    ind = icls(random.uniform(imin, imax) for _ in range(size))
    ind.strategy = scls(random.uniform(smin, smax) for _ in range(size))
    return ind

# Fonction utilisée pour mettre une borne inférieure à la mutation
def checkStrategy(minstrategy):
    def decorator(func):
        def wrappper(*args, **kargs):
            children = func(*args, **kargs)
            for child in children:
                for i, s in enumerate(child.strategy):
                    if s < minstrategy:
                        child.strategy[i] = minstrategy
            return children
        return wrappper
    return decorator

def launch_nsga2(mu=100, lambda_=200, cxpb=0.6, mutpb=0.3, ngen=200, verbose=False):

    random.seed()

    toolbox = base.Toolbox()
    toolbox.register("individual", generateES, creator.Individual, creator.Strategy, len(center), MIN_VALUE, MAX_VALUE, MIN_STRATEGY, MAX_STRATEGY)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("mate", tools.cxESBlend, alpha=0.1)
    toolbox.register("mutate", tools.mutESLogNormal, c=1.0, indpb=0.03)
    toolbox.register("select", tools.selNSGA2)

    if (variant=="FIT+NS"):
         toolbox.register("evaluate", eval_fit_ns)
    elif (variant=="FIT"):
         toolbox.register("evaluate", eval_fit)
    elif (variant=="NS"):
         toolbox.register("evaluate", eval_ns)

    # Application de la borne minimale
    toolbox.decorate("mate", checkStrategy(MIN_STRATEGY))
    toolbox.decorate("mutate", checkStrategy(MIN_STRATEGY))

    population = toolbox.population(n=mu)
    paretofront = tools.ParetoFront()

    fbd=open("bd.log","w")

    # Evaluate the individuals with an invalid fitness
    invalid_ind = [ind for ind in population if not ind.fitness.valid]
    fitnesses_bds = toolbox.map(toolbox.evaluate, invalid_ind)

    for ind, (fit, bd) in zip(invalid_ind, fitnesses_bds):

        print("Fit: "+str(fit)) 
        print("BD: "+str(bd))

        ## A compléter: initialiser ind.fitness.values de façon appropriée, selon la variante  

        fbd.write(" ".join(map(str,[bd]))+"\n")
        fbd.flush()

        ind.fit = fit
        ind.bd = bd

    if paretofront is not None:
        paretofront.update(population)

    #print("Pareto Front: "+str(paretofront))

    k=15
    add_strategy="random"
    lambdaNov=6
    archive = None

    for ind in population:
        ind.novelty = 0
        #print("Fit=%f Nov=%f"%(ind.fit, ind.novelty))

    ## A completer: mettre a jour l'archive et le calcul de nouveauté (qui sera dans ind.novelty)
    archive = updateNovelty([], population, None)
    #print(archive)


    for ind in population:
        if (variant=="FIT+NS"):
            ind.fitness.values = [ind.fit, ind.novelty]
        elif (variant=="FIT"):
            ind.fitness.values = [ind.fit]
        elif (variant=="NS"):
            ind.fitness.values = [ind.novelty]

    population = toolbox.select(population, len(population))

    # On garde trace de l'individu le plus proche de la sortie
    indexmin, valuemin = min(enumerate([i.fit for i in population]), key=operator.itemgetter(1))

    # Begin the generational process
    for gen in range(1, ngen + 1):
        if (gen%10==0):
            print("+",end="", flush=True)
        else:
            print(".",end="", flush=True)

        ## A completer: générer les 'offspring' et les évaluer

        offspring = tools.selTournamentDCD(population, len(population))
        offspring = [toolbox.clone(ind) for ind in offspring]

        for ind1, ind2 in zip(offspring[::2], offspring[1::2]):
            if random.random() <= cxpb:
                toolbox.mate(ind1, ind2)

            if random.random() <= mutpb:
                toolbox.mutate(ind1)

            if random.random() <= mutpb:
                toolbox.mutate(ind2)
            del ind1.fitness.values, ind2.fitness.values

          # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses_bds = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, (fit, bd) in zip(invalid_ind, fitnesses_bds):
                ind.fit = fit
                ind.bd = bd

        pq=population+offspring

        ## A completer: mettre à jour l'archive et calculer les nouveautés

        archive = updateNovelty(population, offspring, archive)

        for ind in pq:
             if (variant=="FIT+NS"):
                 ind.fitness.values = [ind.fit, ind.novelty]
             elif (variant=="FIT"):
                 ind.fitness.values = [ind.fit]
             elif (variant=="NS"):
                 ind.fitness.values = [ind.novelty]

        # Select the next generation population
        population = toolbox.select(pq, mu)

        # Update the hall of fame with the generated individuals
        if paretofront is not None:
            paretofront.update(population)

        indexmin, newvaluemin = min(enumerate([i.fit for i in pq]), key=operator.itemgetter(1))

        if (newvaluemin<valuemin):
            valuemin=newvaluemin
            print("Gen "+str(gen)+", new min ! min fit="+str(valuemin)+" index="+str(indexmin))
            print("Novelty : ", pq[indexmin].novelty)
            eval_nn(pq[indexmin],True,"_gen%04d"%(gen))
    fbd.close()
    return population, None, paretofront

env = gym.make('FastsimSimpleNavigation-v0')

if (__name__ == "__main__"):

    pop, logbook, paretofront= launch_nsga2(ngen=200)

    #plot_pareto_front(paretofront, "Final pareto front")

    # evaluation finale des individus sur le front de pareto final (pour sauver les trajectoires associées)
    for i,p in enumerate(paretofront):
        print("Visualizing indiv "+str(i)+", fit="+str(p.fitness.values))
        eval_nn(p,True,"_last_pareto_front_ind_%d"%(i))

    env.close()
