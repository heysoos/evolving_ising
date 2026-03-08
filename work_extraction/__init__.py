"""Work extraction experiment framework for the Evolving Ising project."""

from .thermodynamics import (
    CycleAccumulator,
    temperature_schedule,
    temperature_schedule_np,
    run_cycle_with_accounting,
    run_multiple_cycles,
    compute_remodel_work,
)
from .controller import LocalController, LocalMagnetisationTracker
from .budgets import (
    BaseBudget,
    NoBudget,
    BondBudget,
    NeighbourhoodBudget,
    DiffusingBudget,
)
from .optimiser import WorkExtractionES, evaluate_fitness
from .train import DEFAULT_CONFIG, ExperimentResult, make_budget, run_experiment

__all__ = [
    'CycleAccumulator',
    'temperature_schedule',
    'temperature_schedule_np',
    'run_cycle_with_accounting',
    'run_multiple_cycles',
    'compute_remodel_work',
    'LocalController',
    'LocalMagnetisationTracker',
    'BaseBudget',
    'NoBudget',
    'BondBudget',
    'NeighbourhoodBudget',
    'DiffusingBudget',
    'WorkExtractionES',
    'evaluate_fitness',
    'DEFAULT_CONFIG',
    'ExperimentResult',
    'make_budget',
    'run_experiment',
]