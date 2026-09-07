# Day 30 — Weight Initialization Strategies

## Key Insight
Xavier/He initialization controls the variance of a layer's weighted sum — but it cannot change an activation function's own derivative ceiling. Sigmoid (max derivative 0.25) decays catastrophically across depth even with perfect initialization; TanH (max derivative 1.0) barely decays at all, using the exact same initialization math.

## Quick Start
```python
from src.initialization import Xavier, He, Uniform
from src.mlp import Layer, ANN, Sigmoid, TanH, Relu, MSE

layer = Layer(no_perceptrons=16, activation=Relu(), initializer=He(), input=30, lr=0.01)
```

## The Math
```
Var(sum of n terms) = n × Var(each term)
→ scale weights by c where c² = 1/n  →  c = 1/√n        (Xavier)
→ ReLU halves variance each layer    →  c = √(2/n)       (He)
→ bias needs no randomness — weights alone break symmetry → zero-init
```

## Experiment Results — 10-layer network, gradient magnitude (layer 1 → layer 10)
| Setup | Layer 1 | Layer 10 | Decay Factor |
|---|---|---|---|
| Sigmoid + Xavier | 0.0000000513 | 0.1020612887 | ~2,000,000x |
| TanH + Xavier | 0.0122416772 | 0.0352097983 | ~3x |
| ReLU + He | 0.0094195189 | 0.1396603385 | ~15x |

## Why Sigmoid Still Decays Despite Xavier
```
sigmoid'(0) = 0.25   (maximum possible)
tanh'(0)    = 1.00   (maximum possible)
```
Xavier scales the INPUT to the activation correctly. It cannot raise the activation's own derivative ceiling. That ceiling compounds multiplicatively across depth regardless of initialization quality.

## Tests
```
python -m pytest tests/test.py
```

## Tomorrow — Day 31
PyTorch. Depth, activation, and initialization are three separate, interacting levers — this week's mental checklist for debugging why a network won't train, now applied inside a framework that automates the mechanics but not the reasoning.