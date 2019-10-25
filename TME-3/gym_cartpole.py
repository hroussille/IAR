import cma
import gym
import numpy as np
from fixed_structure_nn_numpy import SimpleNeuralControllerNumpy

_env = None

def eval_nn(genotype, render=False, verbose=False):
    total_reward=0
    nn=SimpleNeuralControllerNumpy(4,1,2,5)
    nn.set_parameters(genotype)
    observation = _env.reset()
    for t in range(1000):
        if render:
            _env.render()
        action=nn.predict(observation)
        if action>0:
            action=1
        else:
            action=0
        observation, reward, done, info = _env.step(action)
        total_reward+=reward
        if done:
            if verbose:
                print("Episode finished after %d timesteps"%(t+1))
            break
    return -total_reward

### A completer pour optimiser les parametres du reseau de neurones avec CMA-ES ###


def fit_cartpole():
    global _env
    hist = []

    _env = gym.make('CartPole-v1')
    _env._max_episode_steps = 500
    nn=SimpleNeuralControllerNumpy(4,1,2,5)
    nn.init_random_params()

    es = cma.CMAEvolutionStrategy(nn.get_parameters(), 0.2)

    for _ in range(800):
        solutions = es.ask()
        es.tell(solutions, [eval_nn(x) for x in solutions])
        hist.append(-es.result.fbest)

    _env.close()

    return hist, -es.result.fbest, es.result.xbest

