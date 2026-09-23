"""
plot_convergence_runs.py — Grafico di Convergenza Colorato per Singola Run sulle 3 Classi di Problema

Genera un grafico a 3 subplot (SPI, MPI, LPI) mostrando:
  - 10 run individuali con colori vivaci e distinti per ciascun seed
  - Curva media di convergenza (linea spessa evidenziata)
  - Riferimento teorico (Ottimo ILP per SPI, Lower Bound LP per MPI ed LPI)
"""

import sys
import os
import time
from pathlib import Path
from typing import List, Tuple, Dict, Any
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from concurrent.futures import ProcessPoolExecutor, as_completed

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

_HERE = Path(__file__).resolve().parent
PROJECT_ROOT = _HERE.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.core.graph import Graph
from src.core.mmas import MMAS_Solver
from src.core.exact_solver import solve_exact_ilp, compute_lower_bound
from src.utils.logger import setup_logger

PLOTS_DIR = PROJECT_ROOT / "results" / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)
OUT_PLOT = PLOTS_DIR / "convergence_runs_3_classes.png"
ALT_PLOT = PLOTS_DIR / "convergence_band.png"

logger = setup_logger("CONV_RUNS")

# Impostazioni di stile per grafici accademici vivaci
plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 9,
    "figure.titlesize": 14,
    "figure.autolayout": True,
})

TARGET_INSTANCES = [
    {"class": "SPI", "path": PROJECT_ROOT / "wvcp-instances" / "vc_20_120.txt", "title": "SPI: vc_20_120 (|V|=20, |E|=120)", "budget": 20000},
    {"class": "MPI", "path": PROJECT_ROOT / "wvcp-instances" / "vc_200_750.txt", "title": "MPI: vc_200_750 (|V|=200, |E|=750)", "budget": 20000},
    {"class": "LPI", "path": PROJECT_ROOT / "wvcp-instances" / "vc_800_10000.txt", "title": "LPI: vc_800_10000 (|V|=800, |E|=10000)", "budget": 20000},
]

N_RUNS = 10
SEEDS = [42 + i * 101 for i in range(N_RUNS)]


def run_single_seed_task(args):
    inst_info, seed, seed_idx = args
    fpath = inst_info["path"]
    graph = Graph.load_from_file(str(fpath))
    
    n_ants = max(5, graph.n // 10)
    solver = MMAS_Solver(
        graph,
        n_ants=n_ants,
        alpha=1.0,
        beta=2.0,
        rho=0.10,
        max_fe=inst_info["budget"],
        seed=seed,
    )
    res = solver.solve(verbose=False)
    log = res["evaluator"].convergence_log  # list of (fe, best_w)
    
    return {
        "class": inst_info["class"],
        "seed_idx": seed_idx,
        "seed": seed,
        "best_weight": float(res["best_weight"]),
        "fe_at_best": res["fe_at_best"],
        "log": log,
    }


def generate_colorful_convergence_plots():
    logger.info("=" * 70)
    logger.info("GENERAZIONE GRAFICO CONVERGENZA COLORATO PER SINGOLE RUN (3 CLASSI)")
    logger.info("=" * 70)
    
    # 1. Calcolo o recupero reference values (OPT / LB)
    ref_values = {}
    for inst in TARGET_INSTANCES:
        fpath = inst["path"]
        if not fpath.exists():
            # Fallback path if missing trailing .txt or suffix
            alt_p = Path(str(fpath) + ".txt") if not str(fpath).endswith(".txt") else Path(str(fpath).replace(".txt", "_05.txt"))
            if alt_p.exists():
                inst["path"] = alt_p
                fpath = alt_p
                
        graph = Graph.load_from_file(str(fpath))
        if inst["class"] == "SPI":
            try:
                opt, _ = solve_exact_ilp(graph, time_limit_s=30)
                ref_values["SPI"] = (float(opt), f"OPT ILP ({opt:.0f})")
            except Exception:
                lb = compute_lower_bound(graph)
                ref_values["SPI"] = (float(lb), f"Lower Bound ({lb:.0f})")
        else:
            lb = compute_lower_bound(graph)
            ref_values[inst["class"]] = (float(lb), f"Lower Bound LP ({lb:.0f})")
            
    # 2. Parallel execution of all 30 runs (3 classes x 10 runs)
    tasks = []
    for inst in TARGET_INSTANCES:
        for seed_idx, seed in enumerate(SEEDS, 1):
            tasks.append((inst, seed, seed_idx))
            
    runs_by_class = {"SPI": [], "MPI": [], "LPI": []}
    
    max_workers = min(os.cpu_count() or 4, len(tasks))
    logger.info(f"Esecuzione parallelizzata di {len(tasks)} run su {max_workers} worker CPU...")
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(run_single_seed_task, t): t for t in tasks}
        for future in as_completed(futures):
            res = future.result()
            runs_by_class[res["class"]].append(res)
            
    # Sort runs by seed_idx for deterministic color mapping
    for c in runs_by_class:
        runs_by_class[c].sort(key=lambda x: x["seed_idx"])
        
    # 3. Plotting figure with 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.5))
    
    # Vibrant distinct palette for individual 10 runs
    palette = sns.color_palette("tab10", N_RUNS)
    
    for idx, inst in enumerate(TARGET_INSTANCES):
        ax = axes[idx]
        cls = inst["class"]
        runs = runs_by_class[cls]
        budget = inst["budget"]
        
        fe_grid = np.linspace(1, budget, 250)
        interpolated_runs = []
        
        # Plot individual run trajectories with vibrant colors
        for r in runs:
            seed_idx = r["seed_idx"]
            color = palette[seed_idx - 1]
            fes, weights = zip(*r["log"])
            interp_w = np.interp(fe_grid, fes, weights)
            interpolated_runs.append(interp_w)
            
            ax.plot(
                fe_grid,
                interp_w,
                color=color,
                linewidth=1.2,
                alpha=0.75,
                label=f"Run #{seed_idx} (s={r['seed']})"
            )

        # Plot mean trajectory with thick black/dark-navy line
        mean_curve = np.mean(interpolated_runs, axis=0)
        ax.plot(
            fe_grid,
            mean_curve,
            color="#000000",
            linewidth=3.0,
            linestyle="-",
            label="Media MMAS (10 Run)"
        )
        
        # Reference horizontal line (OPT / LB)
        if cls in ref_values:
            ref_val, ref_label = ref_values[cls]
            ax.axhline(
                ref_val,
                color="#d62728",
                linestyle="--",
                linewidth=2.0,
                label=ref_label
            )

        ax.set_title(inst["title"], fontsize=12, fontweight="bold")
        ax.set_xlabel("Valutazioni della Soluzione (FE)", fontsize=11)
        ax.set_ylabel("Peso Vertex Cover (w)", fontsize=11)
        ax.grid(True, linestyle="--", alpha=0.5)
        
        # Legend with 2 columns to fit nicely
        ax.legend(fontsize=8, loc="upper right", frameon=True, facecolor="white", edgecolor="gray", ncol=2)

    plt.suptitle("Analisi di Convergenza Colorata per Run Singole sulle 3 Classi di Problema (SPI, MPI, LPI)", fontsize=14, fontweight="bold")
    plt.tight_layout()
    
    plt.savefig(OUT_PLOT, dpi=300, bbox_inches="tight")
    plt.savefig(ALT_PLOT, dpi=300, bbox_inches="tight")
    plt.close()
    
    logger.info(f"Grafico salvato in: {OUT_PLOT}")
    logger.info(f"Grafico alternativo salvato in: {ALT_PLOT}")
    print(f"\nGRAFICO CONVERGENZA COLORATO SALVATO IN:\n  1. {OUT_PLOT}\n  2. {ALT_PLOT}")


if __name__ == "__main__":
    generate_colorful_convergence_plots()
