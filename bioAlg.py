import random

from deap import creator
from deap import tools
import gym

import algelitism
from telega import PredictModel
from deap import base, algorithms
import matplotlib.pyplot as plt
import time

env = gym.make('CartPole-v1')

network = PredictModel()

LENGTH_CHROM = network.get_total_weights()
LOW = -1.0
MAX = 1.0
ETA = 20


POP_SIZE = 20
P_CROSS = 0.9
P_MUT = 0.1
MAX_GEN = 70
HALL_OF_FAME = 2


hof = tools.HallOfFame(HALL_OF_FAME)

creator.create('FitnessMax', base.Fitness, weights=(1.0,))
creator.create('Individual', list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()
toolbox.register('randomWeights', random.uniform, LOW, MAX)
toolbox.register('individualCreator', tools.initRepeat, creator.Individual, toolbox.randomWeights, LENGTH_CHROM)
toolbox.register('popCreator', tools.initRepeat, list, toolbox.individualCreator)


population = toolbox.popCreator(n=POP_SIZE)


def get_score(individual):
    print(individual)
    network.set_weights(individual)

    observation = env.reset()
    action_counter = 0
    total_reward = 0

    flag_done = False

    while not flag_done and action_counter < 500:
        action_counter += 1
        reshape = observation.reshape(1, -1)
        print(reshape)
        action = int(network.predict(reshape))

        observation, reward, done, info = env.step(action)
        total_reward += reward

    return total_reward


toolbox.register('evaluate', get_score)
toolbox.register('select', tools.selTournament, tournsize=2)
toolbox.register('mate', tools.cxSimulatedBinaryBounded, low=LOW, up=MAX, eta=ETA)
toolbox.register('mutate', tools.mutPolynomialBounded,
                 low=LOW, up=MAX, eta=ETA, indpb=1.0/LENGTH_CHROM)


population, logbook = algelitism.eaSimpleElitism(population, toolbox,
                                        cxpb=P_CROSS,
                                        mutpb=P_MUT,
                                        ngen=MAX_GEN,
                                        halloffame=hof,
                                        verbose=True)

best = hof.items[0]
print(best)


observation = env.reset()
action = int(network.predict(observation.reshape(1, -1)))

while True:
    env.render()
    observation, reward, done, info = env.step(action)

    if done:
        break

    time.sleep(0.03)
    action = int(network.predict(observation.reshape(1, -1)))

env.close()





