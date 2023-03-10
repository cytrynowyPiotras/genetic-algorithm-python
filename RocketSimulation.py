import GeneticAlgorithm
import random
from statistics import stdev

LANDING_GAIN = 2000
CRASH_PANALTY = -1000

class Rocket:
    def __init__(self,
                fuel,
                height = 200,
                vel = 0,
                mass = 200,
                gravityAcceleration = 0.09,
                engineForce = 45,
                accWeightLoss = 1,
                landingHeight = 2,
                landingVel = 2
                ):
        self.landingHeight = landingHeight
        self.landingVel = landingVel
        self.fuel = fuel
        self.height = height
        self.vel = vel
        self.mass = mass + fuel
        self.gravityAcceleration = gravityAcceleration
        self.engineForce = engineForce
        self.accWeightLoss = accWeightLoss

    def landed(self):
        return self.height < self.landingHeight and self.height > 0 and abs(self.vel) < self.landingVel

    def crashed(self):
        return self.height < 0

    def VechicleOn(self):
        self.vel += self.engineForce / self.mass - self.gravityAcceleration
        self.mass -= self.accWeightLoss
        self.height += self.vel

    def VechicleOff(self):
        self.vel -= self.gravityAcceleration
        self.height += self.vel

def rate_flight(onOff = list):
    fuel = onOff.count(1)
    rocket = Rocket(fuel)
    for switch in onOff:
        if switch == 1:
            rocket.VechicleOn()
        elif switch == 0:
            rocket.VechicleOff()
        else:
            raise ValueError("Switch has incorrect data")
        if rocket.crashed():
            return CRASH_PANALTY - fuel
        elif rocket.landed():
            return LANDING_GAIN - fuel
    return - fuel


def populationGenerator(populationSize: int = 100):
    population = [ [ random.choice([0, 1]) for i in range(0, 200)] for j in range(0, populationSize)]
    return population

def average(result:list):
    suma = sum(result)
    return suma / len(result)

def findBestParameters():
    print("Looking for best parameters:")
    parameters = {
        "ratingFunction": rate_flight,
        "populationSize": 50,
        "mutationProbability": 0,
        "crossingProbability": 0,
        "maxIteration": 50,
    }
    mutations = [0.005, 0.010, 0.015, 0.020, 0.025]
    crossings = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    bestAvg = CRASH_PANALTY
    bestMutProb = 0
    bestCrossProb= 0
    for mutation in mutations:
        for cross in  crossings:
            result = []
            for i in range(0, 10):
                parameters["mutationProbability"] = mutation
                parameters["crossingProbability"] = cross
                population = populationGenerator(parameters["populationSize"])
                solver = GeneticAlgorithm.GeneticAlgorithmClass(parameters)
                best, bestVal = solver.solve(rate_flight,population )
                result.append(bestVal)
            Average = average(result)
            deviation = stdev(result)
            bestResult = max(result)
            if Average > bestAvg:
                bestCrossProb = cross
                bestMutProb = mutation
                bestAvg = Average
            print(f"mutationProb: {mutation}, crossingProb: {cross}, average: {Average}, deviation: {deviation}, best: {bestResult}")
    return bestCrossProb, bestMutProb


def main():
    crossProb, mutProb = findBestParameters()
    parameters = {
        "populationSize": 200,
        "mutationProbability": mutProb,
        "crossingProbability": crossProb,
        "maxIteration": 300,
    }
    startingPop = populationGenerator(parameters["populationSize"])
    solver = GeneticAlgorithm.GeneticAlgorithmClass(parameters)
    result, resultVal = solver.solve(rate_flight, startingPop)
    print(f"Best result: {resultVal}, crossing prob: {crossProb}, mutation prob: {mutProb}")


if __name__ == "__main__":
    main()