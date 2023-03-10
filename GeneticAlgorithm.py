from abc import ABC, abstractmethod
from copy import deepcopy
import random
import numpy as np

ROULET_SELOECTION_PARAMETER = 1200

class Solver(ABC):
    """A solver. It may be initialized with some hyperparameters."""

    @abstractmethod
    def get_parameters(self):
        """Returns a dictionary of hyperparameters"""
        ...

    @abstractmethod
    def solve(self, problem, pop0, *args, **kwargs):
        """
        A method that solves the given problem for given initial solutions.
        It may accept or require additional parameters.
        Returns the solution and may return additional info.
        """
        ...


class GeneticAlgorithmClass(Solver):

    def __init__(self, parameters=dict):
         self.parameters = parameters

    def get_parameters(self) -> dict:
        return self.parameters


    def cross(self, firstElement = list, secondElement = list):
        firstCopied, secondCopied = list(deepcopy(firstElement)), list(deepcopy(secondElement))
        crossingPoint = int(random.randint(1, len(firstCopied) - 2))
        firstChild, secondChild = firstCopied[:crossingPoint] + secondCopied[crossingPoint:], secondCopied[:crossingPoint] + firstCopied[crossingPoint:]
        return firstChild, secondChild

    def crossPopulation(self, population, crossingProbability: float = 0.7):
        crossedPop = list(deepcopy(population))
        for i in range(0, len(population) - 1):
            if random.uniform(0, 1) < crossingProbability:
                crossedPop[i], crossedPop[i+1] = self.cross(population[i], population[i+1])
        return crossedPop


    def rouletteSelection(self, population):
        copiedPop = deepcopy(population)
        weights = []
        ratingFunc = self.get_parameters()["ratingFunction"]
        for element in copiedPop:
            value = ratingFunc(element)
            weights.append(value + ROULET_SELOECTION_PARAMETER)
        sumOfWeights = sum(weights)
        probabilities = [weight/sumOfWeights for weight in weights]
        tempPopulationIndexes = np.random.choice(range(0, len(copiedPop)), len(copiedPop), p=probabilities)
        tempPopulation = []
        for index in tempPopulationIndexes:
            tempPopulation.append(copiedPop[index])
        return tempPopulation


    def mutation(self, population, propability: float = 0.05):
        mutatedPop = deepcopy(population)
        for i in range(0, len(population)):
            for j in range(0, len(population[i])):
                mutationProbability = random.uniform(0, 1)
                if mutationProbability < propability:
                    if mutatedPop[i][j] == 1:
                        mutatedPop[i][j] = 0
                    elif mutatedPop[i][j] == 0:
                        mutatedPop[i][j] = 1
                    else: raise ValueError("Given switch is incorrect")
        return mutatedPop

    def findBest(self, population):
        copiedPop = list(deepcopy(population))
        values = []
        ratingFunc = self.get_parameters()["ratingFunction"]
        for i in range(0, len(copiedPop)):
            values.append(ratingFunc(copiedPop[i]))
        bestVal = max(values)
        bestIndex = values.index(bestVal)
        bestFlight = copiedPop[bestIndex]
        return bestFlight, bestVal

    def solve(self, q_func, pop0):
        t = 0
        population = pop0
        self.parameters["ratingFunction"] = q_func
        best, bestValue = self.findBest(population)
        while t <= self.get_parameters()["maxIteration"]:
            tempPopulation = self.rouletteSelection(population)
            crossedPopulation = self.crossPopulation(tempPopulation, self.get_parameters()["crossingProbability"])
            mutatedPopulation = self.mutation(crossedPopulation, self.get_parameters()["mutationProbability"])
            bestTemp, bestValueTemp = self.findBest(mutatedPopulation)
            if bestValueTemp > bestValue:
                best, bestValue = list(deepcopy(bestTemp)), bestValueTemp
            population = mutatedPopulation
            t += 1
        return best, bestValue

