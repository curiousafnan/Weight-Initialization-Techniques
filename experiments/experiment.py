from src.mlp import *
from src.activation import *
from src.gradient_flow import *
from src.initialization import *

def build_deep_network(n_layers, activation,initializer, loss_fn,neurons_per_layer, input_size=8,epochs=100):
    """Build an n-layer network with given activation, return the ANN"""
    layers = [Layer(no_perceptrons=neurons,initializer=initializer,activation=activation,lr=0.01) for neurons in neurons_per_layer[:-1]]
    layers.append(Layer(no_perceptrons=neurons_per_layer[-1],initializer=initializer,activation=Sigmoid(),lr=0.01))
    network = ANN(input_size,epochs,loss_fn,*layers)
    network.compile()
    return network


def measure_activation_spread(ann: ANN, X):
    """
    Run ONE forward pass only (no backward pass needed here).
    At each layer, after computing the activation output,
    record np.std(layer_output) — the spread of that layer's activations.
    
    Return: list of stds, one per layer, ordered layer 1 -> layer N.
    
    Hint: you'll need each Layer to expose its stored activation output
    after forward() runs — check what you already store as 
    self._last_activation in Layer.forward().
    """
    history= []
    y = X[0]
    for layer in ann.layers:
        y=layer.forward(y)
        history.append(np.std(layer._last_activation))
    return history


def measure_gradient_magnitudes(ann:ANN, X, y, loss_fn:Loss):
    """
    Run one forward + backward pass.
    Return a list of gradient magnitudes, one per layer,
    ordered from layer 1 (input side) to layer N (output side).
    """
    y_pred = ann.forward(X[0])
    dL_dA = ann.loss(y[0],y_pred)
    ann.backprop(dL_dA,weights_update=False)
    magnitudes = [np.mean(np.abs(layer.dL_dW)) for layer in ann.layers]
    return magnitudes


def run_initialization_experiment(activation, good_initializer, bad_initializer,input_size,X,y,n_layers=10):
    """
    Build TWO networks with the SAME activation but DIFFERENT initializers
    (e.g. Sigmoid+Xavier vs Sigmoid+Uniform-with-large-scale).
    Run measure_activation_spread on both.
    Run measure_gradient_magnitudes on both.
    
    Return: a dict comparing the two, e.g.
        {"good": {"spread": [...], "gradients": [...]},
         "bad":  {"spread": [...], "gradients": [...]}}
    """
    init = {"good":good_initializer,"bad":bad_initializer}
    result = {}
    for key,value in init.items():
        network = build_deep_network(n_layers=10,activation=activation,initializer=value,loss_fn=MSE(),neurons_per_layer=[32,16,8,4,1],input_size=input_size)
        spread = measure_activation_spread(network,X)
        gradient = measure_gradient_magnitudes(network,X,y,loss_fn=MSE())
        result[key]={"spread":spread,"gradient":gradient}
    return result

if __name__ == "__main__":

    np.random.seed(42)
    X = np.random.randn(10, 8)
    y = np.random.randint(0, 2, size=(10, 1))
    experiments = [
        {
            "name": "Sigmoid + Xavier",
            "activation": Sigmoid(),
            "initializer": Xavier()
        },
        {
            "name": "Tanh + Xavier",
            "activation": TanH(),
            "initializer": Xavier()
        },
        {
            "name": "ReLU + He",
            "activation": Relu(),
            "initializer": HE()
        }
    ]
    results = {}

    for experiment in experiments:

        print("\n" + "=" * 60)
        print(experiment["name"])
        print("=" * 60)

        network = build_deep_network(
            n_layers=10,
            activation=experiment["activation"],
            initializer=experiment["initializer"],
            loss_fn=MSE(),
            neurons_per_layer=[
                32, 32, 32, 32, 32,
                32, 32, 32, 32, 1
            ],
            input_size=8
        )

        spread = measure_activation_spread(
            network,
            X
        )

        gradients = measure_gradient_magnitudes(
            network,
            X,
            y,
            loss_fn=MSE()
        )

        results[experiment["name"]] = {
            "spread": spread,
            "gradients": gradients
        }
        print("\nActivation Spread:")

        for layer, value in enumerate(spread, start=1):
            print(f"Layer {layer:2d}: {value:.6f}")

        # --------------------------------
        # Print gradient magnitude
        # --------------------------------
        print("\nGradient Magnitudes:")

        for layer, value in enumerate(gradients, start=1):
            print(f"Layer {layer:2d}: {value:.10f}")