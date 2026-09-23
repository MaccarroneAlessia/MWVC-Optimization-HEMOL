"""
run_multiseed_validation.py — Validazione Multi-Seed dei Top Candidati di Tuning

Confronta in modo statisticamente robusto (5 seed) i principali candidati emersi dal tuning:
  1. Config 1: n_ants=80, alpha=1.0, beta=2.0, rho=0.15 (Vincitore Fase 1 & 2)
  2. Config 2: n_ants=80, alpha=1.0, beta=2.0, rho=0.10 (Parametro in relazione precedente)
  3. Config 3: n_ants=80, alpha=2.0, beta=2.0, rho=0.15 (Rank 3 single-seed)
  4. Config 4: n_ants=60, alpha=1.0, beta=2.0, rho=0.15 (Rank 4 single-seed)
  5. Config 5: n_ants=80, alpha=1.0, beta=5.0, rho=0.15 (Beta elevata)
"""

import os
import sys
import time
import csv
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np

# Forza 1 thread per processo per evitare over-subscription di NumPy BLAS
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

_project_root = Path(__file__).resolve().parent.parent
_src_dir = Path(__file__).resolve().parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))
if str(_src_dir) not in sys.path:
    sys.path.insert(0, str(_src_dir))

from src.core.graph import Graph
from src.core.mmas import MMAS_Solver
from src.utils.logger import setup_logger, print_and_log_table, get_top_unique_configs

logger = setup_logger("MULTISEED_VAL")

TARGET_INSTANCE = "wvcp-instances/vc_800_10000.txt"
SEEDS = [42, 101, 202, 303, 404]

CANDIDATES = [
    {"name": "ants=80, a=1.0, b=2.0, r=0.15 (Vincitore)", "n_ants": 80, "alpha": 1.0, "beta": 2.0, "rho": 0.15},
    {"name": "ants=80, a=1.0, b=2.0, r=0.10 (Relazione prev)", "n_ants": 80, "alpha": 1.0, "beta": 2.0, "rho": 0.10},
    {"name": "ants=80, a=2.0, b=2.0, r=0.15 (Rank 3)", "n_ants": 80, "alpha": 2.0, "beta": 2.0, "rho": 0.15},
    {"name": "ants=60, a=1.0, b=2.0, r=0.15 (Rank 4)", "n_ants": 60, "alpha": 1.0, "beta": 2.0, "rho": 0.15},
    {"name": "ants=80, a=1.0, b=5.0, r=0.15 (Beta=5)", "n_ants": 80, "alpha": 1.0, "beta": 5.0, "rho": 0.15},
]


def run_single_task(args):
    cand_idx, cand, seed, graph_path_str = args
    graph = Graph.load_from_file(graph_path_str)
    
    params = {
        "n_ants": cand["n_ants"],
        "alpha": cand["alpha"],
        "beta": cand["beta"],
        "rho": cand["rho"],
        "gamma": 1.0,
        "max_fe": 20_000,
        "pruning_strategy": "greedy",
        "seed": seed,
    }
    
    solver = MMAS_Solver(graph, **params)
    t0 = time.time()
    res = solver.solve(verbose=False)
    elapsed = time.time() - t0
    
    return {
        "cand_idx": cand_idx,
        "cand_name": cand["name"],
        "n_ants": cand["n_ants"],
        "alpha": cand["alpha"],
        "beta": cand["beta"],
        "rho": cand["rho"],
        "seed": seed,
        "best_weight": int(res["best_weight"]),
        "fe_best": res["fe_at_best"],
        "reinits": res["n_reinits"],
        "time_s": round(elapsed, 2),
    }


def main():
    logger.info("=" * 70)
    logger.info("AVVIO VALIDAZIONE MULTI-SEED (5 SEED) TOP CANDIDATI MMAS-MWVCP")
    logger.info("=" * 70)
    
    graph_path = _project_root / TARGET_INSTANCE
    if not graph_path.exists():
        logger.error(f"Istanza non trovata: {graph_path}")
        return
        
    csv_dir = _project_root / "results" / "csv"
    csv_dir.mkdir(parents=True, exist_ok=True)
    out_csv = csv_dir / "multiseed_top_candidates.csv"
    
    tasks = []
    for c_idx, cand in enumerate(CANDIDATES, 1):
        for seed in SEEDS:
            tasks.append((c_idx, cand, seed, str(graph_path)))
            
    logger.info(f"Totale task da eseguire: {len(tasks)} ({len(CANDIDATES)} candidati x {len(SEEDS)} seed)")
    
    results = []
    
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        fieldnames = ["cand_idx", "cand_name", "n_ants", "alpha", "beta", "rho", "seed", "best_weight", "fe_best", "reinits", "time_s"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        
        # Max workers = min(os.cpu_count(), len(tasks))
        max_workers = min(os.cpu_count() or 4, len(tasks))
        logger.info(f"Esecuzione su {max_workers} worker di processo...")
        
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(run_single_task, task): task for task in tasks}
            for future in as_completed(futures):
                res = future.result()
                results.append(res)
                writer.writerow(res)
                f.flush()
                logger.info(
                    f"  [Cand #{res['cand_idx']} | seed={res['seed']}] "
                    f"w={res['best_weight']} | FE_best={res['fe_best']} | reinits={res['reinits']} | t={res['time_s']}s"
                )

    logger.info("\n" + "=" * 70)
    logger.info("RIEPILOGO STATISTICO MULTI-SEED (Media ± Dev.Std)")
    logger.info("=" * 70)
    
    # Aggregazione per candidato
    summary = []
    for c_idx, cand in enumerate(CANDIDATES, 1):
        cand_runs = [r for r in results if r["cand_idx"] == c_idx]
        weights = [r["best_weight"] for r in cand_runs]
        times = [r["time_s"] for r in cand_runs]
        fe_bests = [r["fe_best"] for r in cand_runs]
        
        mean_w = np.mean(weights)
        std_w = np.std(weights)
        mean_t = np.mean(times)
        mean_fe = np.mean(fe_bests)
        
        summary.append({
            "cand_idx": c_idx,
            "name": cand["name"],
            "rho": cand["rho"],
            "alpha": cand["alpha"],
            "beta": cand["beta"],
            "n_ants": cand["n_ants"],
            "mean_w": mean_w,
            "std_w": std_w,
            "min_w": min(weights),
            "max_w": max(weights),
            "mean_fe": mean_fe,
            "mean_t": mean_t,
            "weights": weights
        })

    summary.sort(key=lambda x: (x["mean_w"], x["std_w"]))
    unique_summary = get_top_unique_configs(summary, param_keys=("n_ants", "alpha", "beta", "rho"), top_n=5)

    headers = ["Rank", "Candidato", "N_ants", "Alpha", "Beta", "Rho", "Media Peso (mu +- sigma)", "Min W", "Max W", "Mean FE", "Tempo (s)"]
    rows = [
        [
            rank,
            s["name"],
            s["n_ants"],
            s["alpha"],
            s["beta"],
            s["rho"],
            f"{s['mean_w']:.2f} ± {s['std_w']:.2f}",
            s["min_w"],
            s["max_w"],
            f"{s['mean_fe']:.1f}",
            f"{s['mean_t']:.1f}s",
        ]
        for rank, s in enumerate(unique_summary, 1)
    ]
    print_and_log_table(logger, "RANKING FINALE MULTI-SEED TOP CANDIDATI (CONFIGURAZIONI UNICHE)", headers, rows)


if __name__ == "__main__":
    main()
