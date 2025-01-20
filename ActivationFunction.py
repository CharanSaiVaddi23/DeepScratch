import math

class Activations:
    def __init__(self, a, l):
        self.Alpha = a
        self.Lambda = l
    def ReLU(self, x):
        return max(0,x)
    def ReLU_derivative(self, x):
            return 1 if x>=0 else 0
    
    def sigmoid(self, x):
        return 1/(1+math.exp(-x))
    def sigmoid_derivative(self, x):
        sig = self.sigmoid(x)
        return sig * (1 - sig)
            
    def tanh(self, x):
        return (math.exp(x) - math.exp(-x))/(math.exp(x) + math.exp(-x))
    def tanh_derivate(self, x):
        t = self.tanh(x)
        return 1 - t**2
    
    def LeakyReLU(self, x):
        return max(self.Alpha * x, x)
    def LeakyReLU_derivative(self, x):
        return 1 if x>=0 return self.Alpha
    
    def swish(self, x):
        return x * self.sigmoid(x)
    def swish_derivative(self, x):
        s = self.sigmoid(x)
        return s * (1-s) + x
        
    def ELU(self, x):
        return max(self.Alpha * (math.exp(x) - 1), x)
    def ELU_derivative(self, x):
        return 1 if x>0 else self.Alpha * math.exp(x)
    
    def SELU(self, x):
        return max(self.Lambda * self.Alpha * (math.exp(x) - 1), self.Lambda * x)
    def SELU_derivative(self, x):
        return self.Lambda if x>=0 else self.Lambda * self.Alpha * math.exp(x)
    
    def GELU(self, x):
        return 0.5 * x * (1 + erf(x / sqrt(2)))
    def GELU_derivative(self, x):
        return 0.5 * (erf(x / sqrt(2)) + 0.797885 * exp(- (x**2)/2) * x + 1)