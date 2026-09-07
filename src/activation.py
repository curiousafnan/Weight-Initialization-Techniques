import numpy as np

class Activation:
    def forward(self, z):
        raise NotImplementedError

    def derivative(self, z):
        raise NotImplementedError

    def __call__(self, z):
        return self.forward(z)


class Sigmoid(Activation):
    def forward(self, z):
        return 1 / (1 + np.exp(-z))

    def derivative(self, z):
        s = self.forward(z)
        return s * (1 - s)


class Relu(Activation):
    def forward(self, z):
        return np.maximum(0, z)

    def derivative(self, z):
        return (z > 0).astype(float)


class TanH(Activation):
    def forward(self, z):
        return np.tanh(z)

    def derivative(self, z):
        t = np.tanh(z)
        return 1 - t**2


class ELU(Activation):
    def __init__(self, alpha=1.0):
        self.alpha = alpha

    def forward(self, z):
        return np.where(z > 0, z, self.alpha * (np.exp(z) - 1))

    def derivative(self, z):
        return np.where(z > 0, 1, self.alpha * np.exp(z))


class SeLU(Activation):
    def __init__(self, alpha=1.6732632423543772, scale=1.0507009873554805):
        self.alpha = alpha
        self.scale = scale

    def forward(self, z):
        return self.scale * np.where(z > 0,z,self.alpha * (np.exp(z) - 1))

    def derivative(self, z):
        return self.scale * np.where(z > 0,1,self.alpha * np.exp(z))


class GeLU(Activation):
    def forward(self, z):
        return 0.5 * z * (1 + np.tanh(np.sqrt(2 / np.pi) *(z + 0.044715 * z**3)))

    def derivative(self, z):
        c = np.sqrt(2 / np.pi)
        inner = c * (z + 0.044715 * z**3)
        tanh_inner = np.tanh(inner)
        inner_derivative = c * (1 + 3 * 0.044715 * z**2)
        return (0.5 * (1 + tanh_inner)+ 0.5 * z *(1 - tanh_inner**2) *inner_derivative)

if __name__ == "__main__":
    z = np.array([-2., -1., 0., 1., 2.])
    for act in [Sigmoid(), Relu(), TanH(), ELU(), SeLU(), GeLU()]:
        print(act.__class__.__name__, act.derivative(z))