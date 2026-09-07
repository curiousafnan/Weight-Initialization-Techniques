# Approach — Day 30: Weight Initialization Strategies

## The Math

### Why Scale by Layer Width
If each weight `w_i` has variance `v` and each input `x_i` has variance ≈1, then `Var(w_i·x_i) ≈ v`. Summing `n` such independent terms:
```
Var(z) = Var(w1x1 + w2x2 + ... + wnxn) = n × v
```
Without correction, a layer with 1000 inputs produces `z` with ~1000x the variance of a layer with 1 input — purely from summing more terms, unrelated to anything learned.

### Deriving the Scaling Factor
To keep `Var(z) ≈ 1` regardless of `n`, scale each weight by a constant `c` such that:
```
Var(c·w) = c² × Var(w) = c² × 1 = 1/n
c² = 1/n
c = 1/√n
```
This is Xavier/Glorot initialization: `W ~ randn() × 1/√n_inputs`.

### He Initialization — Compensating for ReLU
ReLU zeroes out roughly half of all pre-activation values (everything negative), cutting the variance passed forward by roughly half at each layer. He initialization pre-compensates by doubling the initial variance:
```
c = √(2/n)
```
so that after ReLU's variance-halving effect, the resulting variance lands back near the target.

### Why Bias Initializes to Zero
Weights break symmetry between neurons because each neuron's weight row is an independent random draw — different neurons see different weighted combinations of the same input, even before any bias is added. Bias is a single scalar added uniformly across all inputs; it does not need randomness to preserve neuron-to-neuron differentiation, and zero-init avoids injecting extra variance into `z`.

## Design Decisions

### Initializer Class Hierarchy
Followed the same pattern as `Activation` and `Loss` from Days 28-29 — an abstract `Initializer` base class with `__call__`, and `Xavier`, `He`, `Uniform` subclasses each implementing `initialize(n_inputs, n_outputs)`. Kept the pattern consistent for composability in experiments.

### Two Measurement Metrics
- `measure_activation_spread`: forward-pass-only, `std` of each layer's output — measures whether initialization keeps the *signal* well-scaled
- `measure_gradient_magnitudes`: forward+backward, `|dL/dW|` per layer (reused from Day 29) — measures whether initialization keeps the *gradient* well-scaled

### No Weight-Copying Between Good/Bad Initializers
Unlike Day 29 (where weights were copied across activations to isolate the activation as the only variable), today the initializer itself IS the variable under test — each network uses its own naturally-initialized weights via `compile()`.

## Key Finding — Initialization Helps But Doesn't Erase Activation Ceilings
Xavier initialization is mathematically identical whether paired with Sigmoid or TanH — same `1/√n` scaling. Yet Sigmoid+Xavier still showed ~2 million-times gradient decay across 10 layers, while TanH+Xavier showed only ~3x decay. The difference: `sigmoid'(0) = 0.25` (maximum), `tanh'(0) = 1.0` (maximum). Xavier controls the variance of `z` going INTO the activation, but cannot change the activation's own derivative ceiling. Sigmoid's low ceiling compounds multiplicatively across depth regardless of how well-scaled its inputs are.