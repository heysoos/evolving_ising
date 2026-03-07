"""
evolving_ising
==============
Thermally Diffusing Ising model optimised with CMA-ES.

The package is split into focused submodules so that physics, optimisation,
experiment orchestration, and visualisation can be imported independently:

Submodules
----------
model       IsingModel          — 2D spin grid with checkerboard Metropolis
diffusion   TemperatureDiffuser — graph-based heat diffusion over J_nk bonds
optim       SeparableCMAES      — diagonal-covariance CMA-ES (no evosax dependency)
runner      EvoConfig, EvoRunner — high-level optimisation loop (glue layer)
objectives  PhysicsSetup, EXPERIMENTS, make_eval_fn — fitness function factory
experiment  ExperimentResult, Checkpointer, run_experiment — orchestration & I/O
viz         plot_*, generate_report — visualisation (no JAX dependency)

Quick start
-----------
    from evolving_ising import IsingModel, TemperatureDiffuser, SeparableCMAES

    ising    = IsingModel((32, 32), neighborhood="von_neumann", boundary="periodic_lr")
    diffuser = TemperatureDiffuser(alpha=0.35, conductance_mode="abs")
    es       = SeparableCMAES(dim=D, pop_size=32, sigma_init=0.5)

For the full experiment pipeline see run_experiments.py and
evolving_ising.objectives / evolving_ising.experiment.
"""

from .model     import IsingModel
from .diffusion import TemperatureDiffuser
from .optim     import CMAState, SeparableCMAES
from .runner    import EvoConfig, EvoRunner

__all__ = [
    "IsingModel",
    "TemperatureDiffuser",
    "CMAState",
    "SeparableCMAES",
    "EvoConfig",
    "EvoRunner",
]