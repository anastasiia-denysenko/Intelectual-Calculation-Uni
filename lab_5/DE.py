import numpy as np
from torch import nn
import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import copy
import random

class DifferentialEvolution:
    def __init__(self, func, bounds, pop_size=10, max_generations=5, F=0.8, CR=0.7):
        self.func = func
        self.bounds = bounds
        self.pop_size = pop_size
        self.max_generations = max_generations
        self.F = F
        self.CR = CR
        self.history_params = []
        self.history_scores = []

    def optimize(self):
        dim = len(self.bounds)
        pop = [np.array([np.random.uniform(low, high) for (low, high) in self.bounds]) for _ in range(self.pop_size)]
        fitness = [self.func(ind) for ind in pop]

        for gen in range(self.max_generations):
            print(f"GENERATION {gen + 1}/{self.max_generations}", '\n')
            for i in range(self.pop_size):
                a, b, c = np.array(pop)[np.random.choice(self.pop_size, 3, replace=False)]
                mutant = np.clip(a + self.F * (b - c), [b[0] for b in self.bounds], [b[1] for b in self.bounds])

                trial = np.array([
                    mutant[j] if random.random() < self.CR else pop[i][j]
                    for j in range(dim)
                ])
                trial_fitness = self.func(trial)

                self.history_params.append(trial)
                self.history_scores.append(trial_fitness)

                if trial_fitness < fitness[i]:
                    pop[i] = trial
                    fitness[i] = trial_fitness

        best_idx = np.argmin(fitness)
        return pop[best_idx], fitness[best_idx]