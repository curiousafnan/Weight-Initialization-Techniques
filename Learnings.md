# Learnings — Day 30: Weight Initialization Strategies

## Key Concepts Learned

### Two Things Kill Learning Before It Starts
Symmetry and scale. Symmetry: if all neurons in a layer are identical (e.g. all-positive or all-equal weights), they compute identical gradients and learn identical things forever — the network never differentiates its neurons. Scale: if activations are too large, they explode; too small, they vanish. Good initialization breaks symmetry (via independent random weight draws) and keeps signal scale stable through every layer (via `1/√n` or `√(2/n)` scaling).

### Variance Compounds When Summing Random Terms
Summing `n` independent random variables, each with variance `v`, gives a total variance of `n×v`. This is why a layer's weighted sum `z` grows uncontrollably with more inputs unless each individual weight's variance is scaled down proportionally.

### Xavier vs He — Why the Factor of 2
Xavier (`1/√n`) keeps `Var(z) ≈ 1` regardless of layer width. He (`√(2/n)`) adds an extra factor of 2 specifically because ReLU zeroes out roughly half of all values — pre-compensating for that variance loss so the effective post-ReLU variance still lands near the target.

### Bias Doesn't Need Random Initialization
Weights break symmetry between neurons because each neuron has an independently-drawn weight row — different neurons already compute different `z` values from the same input, even with identical zero bias. Bias can safely initialize to zero; it doesn't need to carry the symmetry-breaking burden that weights do.

### Initialization Helps But Doesn't Erase an Activation's Own Ceiling
The most important empirical finding of the day: Sigmoid+Xavier and TanH+Xavier use the exact same initialization math, yet Sigmoid+Xavier still decayed by ~2 million times across 10 layers while TanH+Xavier decayed by only ~3x. Reason: sigmoid'(0) = 0.25 is the maximum possible sigmoid derivative, while tanh'(0) = 1.0 is TanH's maximum. Xavier controls the variance of z going into the activation, but it cannot change the activation function's own derivative ceiling — that ceiling compounds multiplicatively with depth regardless of how well-scaled the inputs are. This means: for deep networks, activation choice and initialization choice are not substitutes for each other — both need to be right.

### Interpreting Suspicious Results Correctly
Every experiment showed "Layer 10: 0.000000" for activation spread. This looked like a bug at first, but the output layer has exactly 1 neuron, and np.std() of a single value is always 0 by definition — not a bug, just an artifact of measuring spread on a single-neuron layer.

## Bugs Caught

### Missing Parentheses on Dictionary .items
Wrote `for key, value in init.items:` instead of `init.items()` — referenced the method itself instead of calling it, which would raise a TypeError trying to unpack a non-iterable method object. Fixed by adding the parentheses.

## Carrying Forward to Day 31 (PyTorch)
Depth, activation choice, and initialization choice are three separate levers that interact — proven directly this week by holding one constant while varying another and watching outcomes diverge wildly. When a PyTorch model trains poorly, this gives a concrete mental checklist beyond "add more layers" or "lower the learning rate": check whether the activation's derivative ceiling is appropriate for the depth, and whether initialization is paired correctly with the chosen activation (He for ReLU-family, Xavier for Sigmoid/TanH-family).