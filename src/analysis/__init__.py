"""
Package analysis
Esporta le funzioni principali per calcolare statistiche e generare grafici.
"""

from .plot_convergence import run_convergence_analysis
from .plot_ablation import run_ablation_analysis, plot_ablation_study
from .plot_tuning import run_tuning_analysis
from .plot_scalability import run_scalability_analysis
from .stats import compute_summary_stats

__all__ = [
    "run_convergence_analysis",
    "run_ablation_analysis",
    "plot_ablation_study",
    "run_tuning_analysis",
    "run_scalability_analysis",
    "compute_summary_stats",
]
