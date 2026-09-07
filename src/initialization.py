import numpy as np

class Initializer:
    def initialize(self,n_inputs,n_outputs): raise NotImplementedError
    def __call__(self, n_inputs, n_outputs):
        return self.initialize(n_inputs,n_outputs)

class Xavier(Initializer):
    def initialize(self,n_inputs,n_outputs):
        return np.random.randn(n_outputs,n_inputs) * np.sqrt(1 / n_inputs)

class HE(Initializer):
    def initialize(self,n_inputs,n_outputs):
        return np.random.randn(n_outputs,n_inputs) * np.sqrt(2 / n_inputs)

class Uniform(Initializer):
    def initialize(self,n_inputs,n_outputs):
        return np.random.uniform(-1,1,(n_outputs, n_inputs))