# Evolving Ising Thermally Diffusive Models

This repository contains code and resources for simulating and evolving Thermally Diffusing Ising models using genetic algorithms. The Ising model is a mathematical model used in statistical mechanics to describe ferromagnetism in materials.

The particular Ising model simulated here consists of a 2D grid of spins and a local temperature field that diffuses along the connectivity of the spins which can change with evolution.

The **fitness** of the model can be evaluated on anything, but in this example we focus on maximizing the temperature across a temperature gradient.
The grid is initialized with a high temperature on one side and a low temperature on the other, and the goal is to evolve the connectivity of the spins to maximize the average temperature across the grid.

Future experiments will explore:
- Maximizing entropy production 
- Minimizing energy consumption while maximizing temperature diffusion
- Something resembling an engine
- Something encouraging mixing
- Something encouraging insulation