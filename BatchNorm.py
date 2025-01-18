import numpy as np
import matplotlib.pyplot as plt
import random

class BasicBatchNorm:
    def computed_mean(self, x):
        return np.mean(x, axis=0)
    def computed_standard_deviation(self, x):
        return np.std(x, axis=0)
    def normalize(self, x, mean, std):
        return (x - mean) / std
    def forward(self, x):
        self.mean = self.computed_mean(x)
        self.std = self.computed_standard_deviation(x)
        return self.normalize(x, self.mean, self.std)

class BatchNorm:
    def __init__(self, epsilon):
        self.epsilon = epsilon
    def computed_mean(self, x):
        return np.mean(x, axis=0)
    def computed_variance(self, x):
        return np.var(x, axis=0)
    def normalize(self, x, mean, variance):
        return (x - mean) / (variance+self.epsilon)**0.5
    def forward(self, x):
        self.mean = self.computed_mean(x)
        self.std = self.computed_variance(x)**0.5
        return self.normalize(x, self.mean, self.std)

class BatchNormLearnable:
    def __init__(self, epsilon, beta = 0, gamma = 0):
        self.epsilon = epsilon
        self.beta = random.random()
        self.gamma =  random.random()
    
    def computed_mean(self, x):
        return np.mean(x, axis=0)
    def computed_variance(self, x):
        return np.var(x, axis=0)
    def normalize(self, x, mean, variance):
        return (x - mean) / (variance+self.epsilon)**0.5
    def scale_shift(self, x):
        return self.gamma * x + self.beta
    def forward(self, x):
        self.mean = self.computed_mean(x)
        self.std = self.computed_variance(x)**0.5
        normalized = self.normalize(x, self.mean, self.std)
        return self.scale_shift(normalized)
    def backward(self, x):
        

arr = np.arange(12).reshape(4, 3)
# plt.plot(arr)
normalised = BatchNorm().forward(arr)
# plt.plot(normalised)

print(f"Mean : {np.mean(arr.flattn())}, Std : {np.std(arr.flatten())}")
print(f"Mean : {np.mean(normalised.flattn())}, Std : {np.std(normalised.flatten())}")