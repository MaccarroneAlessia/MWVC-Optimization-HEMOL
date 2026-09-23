"""
plot_convergence.py — Generazione del grafico di convergenza per MMAS-MWVCP (3 classi).

Esegue 10 run indipendenti con seed differenti su tre istanze rappresentative per classe:
  - SPI: vc_20_120_05.txt
  - MPI: vc_200_750_05.txt
  - LPI: vc_800_10000.txt

Calcola l'ottimo globale con ILP (o Lower Bound LP) e traccia la curva di convergenza media
con banda di variabilità (±1 Std Dev) per ciascuna classe su 3 subplot.

Scrive i log di esecuzione in results/log/plot_convergence.log.

Uso da riga di comando:
    python -m src.analysis.plot_convergence
    oppure
    python src/analysis/plot_convergence.py
"""

import sys
from pathlib import Path
from typing import List, Tuple, Union
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Fix path resolution
_HERE = Path(__file__).resolve().parent
PROJECT_ROOT = _HERE.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.core.graph import Graph
from src.core.mmas import MMAS_Solver
from src.core.exact_solver import solve_exact_ilp, compute_lower_bound
from src.utils.logger import setup_logger

CSV_DIR = PROJECT_ROOT / "results" / "csv"
PLOTS_DIR = PROJECT_ROOT / "results" / "plots"
LOG_DIR = PROJECT_ROOT / "results" / "log"

DEFAULT_PLOT_PATH = PLOTS_DIR / "convergence_3_classes.png"
ALT_PLOT_PATH = PLOTS_DIR / "convergence_band.png"

logger = setup_logger("CONVERGENCE", "plot_convergence.log")

DEFAULT_TARGETS: List[Tuple[Path, str, int]] = [
    (PROJECT_ROOT / "wvcp-instances" / "vc_20_120_05.txt", "SPI (vc_20_120_05)", 20000),
    (PROJECT_ROOT / "wvcp-instances" / "vc_200_750_05.txt", "MPI (vc_200_750_05)", 20000),
    (PROJECT_ROOT / "wvcp-instances" / "vc_800_10000.txt", "LPI (vc_800_10000)", 20000),
]


def run_convergence_analysis(
    target_instances: List[Tuple[Union[Path, str], str, int]] = None,
    output_path: Union[Path, str] = None,
    n_runs: int = 10,
    budget: int = 20000,
    force: bool = False,
    instance_path: Union[Path, str] = None,
) -> Path:
    """
    Esegue l'analisi di convergenza su multi-seed per le istanze specificate e salva il grafico a 3 subplot.
    Se il file di output esiste e force=False, restituisce il percorso senza ricalcolare.
    """
    if output_path is None:
        output_path = DEFAULT_PLOT_PATH
    else:
        output_path = Path(output_path)

    if output_path.exists() and not force:
        msg = f"[INFO] Grafico convergenza già esistente: {output_path}"
        logger.info(msg)
        return output_path

    output_path.parent.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    # Gestione fallback target_instances
    if target_instances is None:
        if instance_path is not None:
            ipath = Path(instance_path)
            target_instances = [(ipath, f"Istanza ({ipath.stem})", budget)]
        else:
            target_instances = DEFAULT_TARGETS

    logger.info(f"[INFO] Avvio analisi di convergenza su {len(target_instances)} istanze ({n_runs} run per istanza)...")

    sns.set_theme(style="whitegrid", palette="muted")
    n_plots = len(target_instances)
    fig, axes = plt.subplots(1, n_plots, figsize=(6 * n_plots, 5))
    if n_plots == 1:
        axes = [axes]

    for idx, (fpath, title, inst_budget) in enumerate(target_instances):
        ax = axes[idx]
        fpath = Path(fpath)
        logger.info(f"[INFO] [{idx+1}/{n_plots}] Elaborazione istanza: {fpath.name} ({title})...")

        if not fpath.exists():
            logger.warning(f"[WARNING] Istanza non trovata: {fpath}, skipping...")
            continue

        graph = Graph.load_from_file(fpath)
        logger.info(f"  -> Grafo caricato: |V|={graph.n}, |E|={len(graph.edges)}")

        # Calcolo riferimento (ILP per SPI / grafi piccoli, Lower Bound altrimenti)
        ref_val = None
        ref_label = ""
        if "20_" in fpath.name or "25_" in fpath.name:
            logger.info("  -> Calcolo Ottimo ILP (time limit 30s)...")
            try:
                ref_val, _ = solve_exact_ilp(graph, time_limit_s=30)
                ref_label = f"OPT ILP ({ref_val:.0f})"
                logger.info(f"  -> Soluzione ILP trovata: {ref_val:.0f}")
            except Exception as e:
                ref_val = compute_lower_bound(graph)
                ref_label = f"Lower Bound ({ref_val:.0f})"
                logger.info(f"  -> Solver ILP non disponibile. Lower Bound LP: {ref_val:.0f} ({e})")
        else:
            logger.info("  -> Calcolo Lower Bound LP...")
            ref_val = compute_lower_bound(graph)
            ref_label = f"Lower Bound ({ref_val:.0f})"
            logger.info(f"  -> Lower Bound LP: {ref_val:.0f}")

        n_ants = max(5, graph.n // 10)
        all_runs_logs = []

        logger.info(f"  -> Avvio {n_runs} run MMAS (n_ants={n_ants}, max_fe={inst_budget})...")
        for seed_idx, seed in enumerate(range(42, 42 + n_runs), start=1):
            solver = MMAS_Solver(
                graph,
                n_ants=n_ants,
                alpha=1.0,
                beta=2.0,
                rho=0.10,
                max_fe=inst_budget,
                seed=seed,
            )
            res = solver.solve(verbose=False)
            log = res["evaluator"].convergence_log  # lista di (fe, best_w)
            all_runs_logs.append(log)
            logger.info(f"     [Run {seed_idx}/{n_runs}] seed={seed} -> best_weight={res['best_weight']:.0f} (fe_at_best={res['fe_at_best']})")

        fe_grid = np.linspace(10, inst_budget, 200)
        interpolated_weights = []

        for log in all_runs_logs:
            fes, weights = zip(*log)
            interp_w = np.interp(fe_grid, fes, weights)
            interpolated_weights.append(interp_w)

        interpolated_weights = np.array(interpolated_weights)
        mean_curve = np.mean(interpolated_weights, axis=0)
        std_curve = np.std(interpolated_weights, axis=0)
        logger.info(f"  [OK] Risultato {fpath.name}: Peso finale medio = {mean_curve[-1]:.1f} ± {std_curve[-1]:.1f}")

        ax.plot(fe_grid, mean_curve, color="#1f77b4", lw=2, label=f"MMAS Media ({n_runs} run)")
        ax.fill_between(
            fe_grid,
            mean_curve - std_curve,
            mean_curve + std_curve,
            color="#1f77b4",
            alpha=0.25,
            label="± 1 Std Dev",
        )

        if ref_val is not None:
            ax.axhline(ref_val, color="red", linestyle="--", lw=1.8, label=ref_label)

        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.set_xlabel("Function Evaluations (FE)", fontsize=11)
        ax.set_ylabel("Peso Vertex Cover", fontsize=11)
        ax.legend(fontsize=10, loc="upper right")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)

    # Salva anche nel percorso alternativo se richiesto dai notebook legacy
    if output_path != ALT_PLOT_PATH:
        plt.savefig(ALT_PLOT_PATH, dpi=300)

    plt.close()
    logger.info(f"[SUCCESSO] Grafico convergenza salvato in: {output_path} (e {ALT_PLOT_PATH})")
    return output_path


def main():
    run_convergence_analysis(force=True)


if __name__ == "__main__":
    main()
