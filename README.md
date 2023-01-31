# Convex-2D-GPU-Collision-Detection

Currently only fast GPU collision detection between rectangles is implemented. But it can easily be extended to handle arbitrary convex 2D shapes. Additionally it contains a Monte Carlo Sampling function that computes collision probabilities between two rectangles, one with fixed variance, one with a provided variance. In SAT.py you see the basic application of how to use the separating axis therorem to determine if two rectangles (or convex shapes) collide. The advantage of this approach is that it has a fixed runtime and therefore suits well for GPU implementation.

Compile using:
```nvcc SAT.cu -o SAT```

Run using:
```./SAT```