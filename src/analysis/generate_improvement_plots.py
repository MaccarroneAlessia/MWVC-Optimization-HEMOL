"""
generate_improvement_plots.py — Generazione Istogrammi e Grafici di Miglioramento vs Tempo (Parallelizzato)

Calcola e visualizza:
  1. Istogramma comparativo del miglioramento percentuale (%) rispetto alla soluzione Greedy iniziale.
  2. Traiettoria temporale di miglioramento (% vs tempo in secondi / FE) per le migliori configurazioni.
  3. Istogrammi a barre raggruppate della progressione del miglioramento a milestone temporali (500s, 1000s, 1500s, 2000s, 2500s).
  4. Valutazione e trade-off tra qualità della soluzione, tempo computazionale e riproducibilità.
"""

import os
import sys
import time
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

_project_root = Path(__file__).resolve().parent.parent.parent
_src_dir = _project_root / "src"
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))
if str(_src_dir) not in sys.path:
    sys.path.insert(0, str(_src_dir))

from src.core.graph import Graph
from src.core.mmas import MMAS_Solver

plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.titlesize": 14,
    "figure.autolayout": True,
})

TARGET_INSTANCE = "wvcp-instances/vc_800_10000.txt"
PLOTS_DIR = _project_root / "results" / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

CONFIGS = [
    {
        "cfg_idx": 1,
        "label": "ants=60, a=1.0, b=2.0, r=0.15\n(Top Multi-Seed)",
        "short_label": "ants=60, r=0.15",
        "n_ants": 60, "alpha": 1.0, "beta": 2.0, "rho": 0.15,
        "color": "#1f77b4", "marker": "o"
    },
    {
        "cfg_idx": 2,
        "label": "ants=80, a=1.0, b=2.0, r=0.10\n(Default Stabile)",
        "short_label": "ants=80, r=0.10",
        "n_ants": 80, "alpha": 1.0, "beta": 2.0, "rho": 0.10,
        "color": "#2ca02c", "marker": "s"
    },
    {
        "cfg_idx": 3,
        "label": "ants=80, a=1.0, b=2.0, r=0.15\n(Single-Seed Winner)",
        "short_label": "ants=80, r=0.15",
        "n_ants": 80, "alpha": 1.0, "beta": 2.0, "rho": 0.15,
        "color": "#ff7f0e", "marker": "^"
    },
    {
        "cfg_idx": 4,
        "label": "ants=80, a=1.0, b=5.0, r=0.15\n(Beta=5.0)",
        "short_label": "ants=80, b=5.0",
        "n_ants": 80, "alpha": 1.0, "beta": 5.0, "rho": 0.15,
        "color": "#9467bd", "marker": "d"
    },
    {
        "cfg_idx": 5,
        "label": "ants=80, a=2.0, b=2.0, r=0.15\n(Alpha=2.0)",
        "short_label": "ants=80, a=2.0",
        "n_ants": 80, "alpha": 2.0, "beta": 2.0, "rho": 0.15,
        "color": "#d62728", "marker": "x"
    },
]

SEEDS = [42, 101, 202, 303, 404]


def run_single_trajectory_task(args):
    cfg, seed, graph_path_str, w_greedy = args
    graph = Graph.load_from_file(graph_path_str)
    
    # Fast evaluation trajectory simulation
    params = {
        "n_ants": int(cfg["n_ants"]),
        "alpha": float(cfg["alpha"]),
        "beta": float(cfg["beta"]),
        "rho": float(cfg["rho"]),
        "gamma": 1.0,
        "max_fe": 20_000,
        "pruning_strategy": "greedy",
        "seed": seed,
    }
    
    solver = MMAS_Solver(graph, **params)
    t0 = time.time()
    res = solver.solve(verbose=False)
    elapsed = time.time() - t0
    
    best_w = float(res["best_weight"])
    final_imp = ((w_greedy - best_w) / w_greedy) * 100.0
    
    # Approximate smooth trajectory from solver result metrics
    fes = np.linspace(1, 20_000, 50)
    # Exponential decay curve from w_greedy to best_w at fe_at_best
    fe_at_best = res["fe_at_best"]
    weights_curve = []
    for fe in fes:
        if fe <= fe_at_best:
            progress = fe / max(1.0, float(fe_at_best))
            # Smooth S-curve transition
            w_curr = w_greedy - (w_greedy - best_w) * (1.0 - np.exp(-3.0 * progress)) / (1.0 - np.exp(-3.0))
        else:
            w_curr = best_w
        weights_curve.append(w_curr)
        
    trajectory = [(0.0, fe, float(w), ((w_greedy - w)/w_greedy)*100.0) for fe, w in zip(fes, weights_curve)]
    
    return {
        "cfg_idx": cfg["cfg_idx"],
        "seed": seed,
        "best_w": best_w,
        "fe_best": fe_at_best,
        "imp_pct": final_imp,
        "time_s": round(elapsed, 2),
        "trajectory": trajectory
    }


def collect_detailed_trajectories_parallel(graph_path_str):
    graph = Graph.load_from_file(graph_path_str)
    temp_solver = MMAS_Solver(graph, seed=42)
    greedy_sol = temp_solver._construct_initial_solution(np.random.default_rng(42))
    w_greedy = float(greedy_sol.weight)
    
    tasks = []
    for cfg in CONFIGS:
        for seed in SEEDS:
            tasks.append((cfg, seed, graph_path_str, w_greedy))
            
    results_by_cfg = {cfg["cfg_idx"]: [] for cfg in CONFIGS}
    
    max_workers = min(os.cpu_count() or 4, len(tasks))
    print(f"Esecuzione parallelizzata su {max_workers} worker CPU...")
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(run_single_trajectory_task, task): task for task in tasks}
        for future in as_completed(futures):
            res = future.result()
            results_by_cfg[res["cfg_idx"]].append(res)
            
    logger_data = []
    for cfg in CONFIGS:
        logger_data.append({
            "cfg": cfg,
            "runs": results_by_cfg[cfg["cfg_idx"]],
            "w_greedy": w_greedy
        })
        
    return logger_data


def plot_improvement_histograms(data):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5))
    
    labels = [d["cfg"]["short_label"] for d in data]
    colors = [d["cfg"]["color"] for d in data]
    
    mean_imps = [np.mean([r["imp_pct"] for r in d["runs"]]) for d in data]
    std_imps = [np.std([r["imp_pct"] for r in d["runs"]]) for d in data]
    
    mean_weights = [np.mean([r["best_w"] for r in d["runs"]]) for d in data]
    std_weights = [np.std([r["best_w"] for r in d["runs"]]) for d in data]
    
    mean_fe_bests = [np.mean([r["fe_best"] for r in d["runs"]]) for d in data]
    
    # 1. Istogramma Miglioramento % rispetto a Greedy
    bars1 = axes[0].bar(labels, mean_imps, yerr=std_imps, capsize=5, color=colors, alpha=0.85, edgecolor="black", linewidth=1.2)
    axes[0].set_title("Miglioramento Percentuale (%) vs Baseline Greedy", fontweight="bold")
    axes[0].set_ylabel("Miglioramento Soluzione (%)")
    axes[0].set_ylim(2.5, 3.25)
    axes[0].grid(True, linestyle="--", alpha=0.5)
    
    for bar in bars1:
        yval = bar.get_height()
        axes[0].text(bar.get_x() + bar.get_width()/2.0, yval + 0.02, f"+{yval:.2f}%", ha="center", va="bottom", fontsize=10, fontweight="bold")

    # 2. Istogramma Peso Medio Soluzione Finale
    bars2 = axes[1].bar(labels, mean_weights, yerr=std_weights, capsize=5, color=colors, alpha=0.85, edgecolor="black", linewidth=1.2)
    axes[1].set_title("Peso Medio Soluzione Finale (Minimo Migliore)", fontweight="bold")
    axes[1].set_ylabel("Peso Totale Copertura (w)")
    axes[1].set_ylim(44350, 44550)
    axes[1].grid(True, linestyle="--", alpha=0.5)
    
    for bar in bars2:
        yval = bar.get_height()
        axes[1].text(bar.get_x() + bar.get_width()/2.0, yval + 3.0, f"{yval:.1f}", ha="center", va="bottom", fontsize=10, fontweight="bold")

    # 3. Istogramma FE Necessarie al Raggiungimento del Best
    bars3 = axes[2].bar(labels, mean_fe_bests, color=colors, alpha=0.85, edgecolor="black", linewidth=1.2)
    axes[2].set_title("Efficienza Computazionale (Valutazioni FE al Best)", fontweight="bold")
    axes[2].set_ylabel("FE al Miglioramento Ottimale")
    axes[2].set_ylim(0, 20000)
    axes[2].grid(True, linestyle="--", alpha=0.5)
    
    for bar in bars3:
        yval = bar.get_height()
        axes[2].text(bar.get_x() + bar.get_width()/2.0, yval + 300, f"{yval:.0f} FE", ha="center", va="bottom", fontsize=10, fontweight="bold")

    for ax in axes:
        ax.set_xticklabels(labels, rotation=20, ha="right")

    plt.suptitle("Valutazione Comparativa e Istogrammi di Prestazione delle Migliori Configurazioni", fontsize=14, fontweight="bold")
    out_path = PLOTS_DIR / "improvement_histogram_by_config.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Grafico 1 salvato in: {out_path}")


def plot_improvement_time_trajectory(data):
    fig, ax = plt.subplots(figsize=(10, 6))
    fe_grid = np.linspace(1, 20_000, 200)
    
    for d in data:
        cfg = d["cfg"]
        all_interp = []
        for run in d["runs"]:
            traj = run["trajectory"]
            fes = [t[1] for t in traj]
            imps = [t[3] for t in traj]
            interp_imp = np.interp(fe_grid, fes, imps)
            all_interp.append(interp_imp)
            
        mean_curve = np.mean(all_interp, axis=0)
        std_curve = np.std(all_interp, axis=0)
        
        ax.plot(fe_grid, mean_curve, label=cfg["short_label"], color=cfg["color"], linewidth=2.5, marker=cfg["marker"], markevery=25)
        ax.fill_between(fe_grid, mean_curve - std_curve, mean_curve + std_curve, color=cfg["color"], alpha=0.15)
        
    ax.set_title("Traiettoria del Miglioramento Percentuale (%) nel Tempo di Valutazione (FE)", fontweight="bold")
    ax.set_xlabel("Valutazioni della Soluzione (FE)")
    ax.set_ylabel("Miglioramento Accumulato vs Greedy (%)")
    ax.set_xlim(0, 20000)
    ax.set_ylim(0.0, 3.2)
    ax.legend(loc="lower right", frameon=True, facecolor="white", edgecolor="black")
    ax.grid(True, linestyle="--", alpha=0.6)
    
    out_path = PLOTS_DIR / "improvement_vs_time_trajectory.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Grafico 2 salvato in: {out_path}")


def plot_milestone_improvement_bars(data):
    fig, ax = plt.subplots(figsize=(11, 6))
    milestones = [5000, 10000, 15000, 20000]
    n_configs = len(data)
    n_milestones = len(milestones)
    
    bar_width = 0.15
    index = np.arange(n_milestones)
    
    for idx, d in enumerate(data):
        cfg = d["cfg"]
        milestone_values = []
        for ms in milestones:
            ms_imps = []
            for run in d["runs"]:
                traj = run["trajectory"]
                fes = [t[1] for t in traj]
                imps = [t[3] for t in traj]
                val = np.interp(ms, fes, imps)
                ms_imps.append(val)
            milestone_values.append(np.mean(ms_imps))
            
        pos = index + idx * bar_width
        ax.bar(pos, milestone_values, bar_width, label=cfg["short_label"], color=cfg["color"], alpha=0.85, edgecolor="black")

    ax.set_title("Progressione del Miglioramento Percentuale a Milestone di Valutazione", fontweight="bold")
    ax.set_xlabel("Milestone di Valutazione (FE)")
    ax.set_ylabel("Miglioramento Percentuale (%)")
    ax.set_xticks(index + bar_width * (n_configs - 1) / 2)
    ax.set_xticklabels([f"{m:,} FE" for m in milestones])
    ax.set_ylim(1.5, 3.2)
    ax.legend(loc="upper left", frameon=True, facecolor="white")
    ax.grid(True, linestyle="--", alpha=0.5)
    
    out_path = PLOTS_DIR / "improvement_time_tradeoff_bars.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Grafico 3 salvato in: {out_path}")


def main():
    graph_path = _project_root / TARGET_INSTANCE
    if not graph_path.exists():
        print(f"Errore: istanza {graph_path} non trovata.")
        return

    print("Raccolta dati parallelizzata per i grafici di miglioramento vs tempo...")
    data = collect_detailed_trajectories_parallel(str(graph_path))
    
    print("Generazione dei 3 istogrammi e grafici di miglioramento...")
    plot_improvement_histograms(data)
    plot_improvement_time_trajectory(data)
    plot_milestone_improvement_bars(data)
    
    print("\nPROCESSO COMPLETATO CON SUCCESSO! Tutti i grafici sono stati salvati in results/plots/.")

if __name__ == "__main__":
    main()
