# Problem Statement — Day 30: Weight Initialization Strategies

## Problem
Implement Xavier/Glorot and He weight initialization from first principles, and empirically test whether proper initialization mitigates the vanishing gradient problem observed on Day 29 — and whether its effectiveness depends on which activation function it's paired with.

## Core Questions
- Why does the number of inputs to a layer affect the ideal scale of its initial weights?
- Why does He initialization need double the variance of Xavier?
- Does correct initialization fully solve vanishing gradients, or only partially — and does the answer depend on the activation function?

## Requirements
- Derive Xavier (`1/√n`) and He (`√(2/n)`) scaling from variance-of-sums first principles
- Implement `Initializer` base class with `Xavier`, `He`, `Uniform` subclasses
- Integrate into `Layer` — weights from initializer, bias initialized to zero
- Measure activation spread (`std` per layer) and gradient magnitude (`|dL/dW|` per layer) across a 10-layer network for: Sigmoid+Xavier, TanH+Xavier, ReLU+He
- Explain any divergence between activations under identical initialization

## Success Criteria
- Correct derivation connecting variance-of-sum (`n×v`) to the `1/√n` scaling factor
- Working `Initializer` class hierarchy integrated cleanly into `Layer`
- Empirical result distinguishing Sigmoid+Xavier's residual decay from TanH+Xavier's near-flat gradient flow, explained via each activation's derivative ceiling