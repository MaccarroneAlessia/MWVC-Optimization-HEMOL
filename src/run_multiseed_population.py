"""
run_multiseed_population.py — Validazione Multi-Seed Dedicata per la Dimensione Colonia (N_ants)

Valuta in modo esaustivo su 5 seed (42, 101, 202, 303, 404) l'impatto di N_ants in {40, 60, 80, 120, 160}
sia per il default ufficiale rho=0.10 sia per rho=0.15.
"""

import os
import sys
import time
import csv
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np

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

logger = setup_logger("POP_MULTISEED")

TARGET_INSTANCE = "wvcp-instances/vc_800_10000.txt"
SEEDS = [42, 101, 202, 303, 404]
POP_SIZES = [40, 60, 80, 120, 160]


def run_pop_task(args):
    n_ants, rho, seed, graph_path_str = args
    graph = Graph.load_from_file(graph_path_str)
    
    params = {
        "n_ants": n_ants,
        "alpha": 1.0,
        "beta": 2.0,
        "rho": rho,
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
        "n_ants": n_ants,
        "rho": rho,
        "seed": seed,
        "best_weight": int(res["best_weight"]),
        "fe_best": res["fe_at_best"],
        "reinits": res["n_reinits"],
        "time_s": round(elapsed, 2),
    }


def main():
    logger.info("=" * 75)
    logger.info("AVVIO VALIDAZIONE MULTI-SEED DEDICATA N_ANTS IN {40, 60, 80, 120, 160}")
    logger.info("=" * 75)
    
    graph_path = _project_root / TARGET_INSTANCE
    if not graph_path.exists():
        logger.error(f"Istanza non trovata: {graph_path}")
        return
        
    csv_dir = _project_root / "results" / "csv"
    csv_dir.mkdir(parents=True, exist_ok=True)
    out_csv = csv_dir / "multiseed_population_results.csv"
    
    existing_results = []
    completed_keys = set()
    
    if out_csv.exists():
        with open(out_csv, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    time_s = float(row["time_s"])
                    n_ants = int(row["n_ants"])
                    rho = float(row["rho"])
                    seed = int(row["seed"])
                    
                    # Filtra esecuzioni anomale causate da sospensione del sistema (> 3600s)
                    if time_s <= 3600.0:
                        res = {
                            "n_ants": n_ants,
                            "rho": rho,
                            "seed": seed,
                            "best_weight": int(row["best_weight"]),
                            "fe_best": int(row["fe_best"]),
                            "reinits": int(row["reinits"]),
                            "time_s": time_s,
                        }
                        existing_results.append(res)
                        completed_keys.add((n_ants, rho, seed))
                    else:
                        logger.warning(
                            f"Rilevata esecuzione anomala nel CSV per [ants={n_ants} | rho={rho} | seed={seed}] (t={time_s}s > 3600s). Verrà rieseguita."
                        )
                except (ValueError, KeyError):
                    continue
                    
    logger.info(f"Trovate {len(existing_results)} esecuzioni valide già completate nel CSV.")
    
    all_tasks = []
    tasks_to_run = []
    for rho in [0.10, 0.15]:
        for n_ants in POP_SIZES:
            for seed in SEEDS:
                all_tasks.append((n_ants, rho, seed, str(graph_path)))
                if (n_ants, rho, seed) not in completed_keys:
                    tasks_to_run.append((n_ants, rho, seed, str(graph_path)))
                    
    logger.info(f"Totale task previsti: {len(all_tasks)}. Task rimanenti da eseguire: {len(tasks_to_run)}.")
    
    new_results = []
    if tasks_to_run:
        max_workers = min(os.cpu_count() or 4, len(tasks_to_run))
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(run_pop_task, task): task for task in tasks_to_run}
            for future in as_completed(futures):
                res = future.result()
                # Verifica sicurezza sul tempo plausibile per singola istanza LPI (max 3600s)
                if res["time_s"] > 3600.0:
                    logger.warning(f"Tempo non plausibile per task [ants={res['n_ants']} | rho={res['rho']} | seed={res['seed']}]: {res['time_s']}s > 3600s. Riesecuzione...")
                    task_args = (res["n_ants"], res["rho"], res["seed"], str(graph_path))
                    res = run_pop_task(task_args)
                new_results.append(res)
                logger.info(
                    f"  [ants={res['n_ants']:<3d} | rho={res['rho']} | seed={res['seed']}] "
                    f"w={res['best_weight']} | FE_best={res['fe_best']} | reinits={res['reinits']} | t={res['time_s']}s"
                )

    combined_results = existing_results + new_results
    
    # Salva il CSV aggiornato ed ordinato
    fieldnames = ["n_ants", "rho", "seed", "best_weight", "fe_best", "reinits", "time_s"]
    combined_results.sort(key=lambda x: (x["rho"], x["n_ants"], x["seed"]))
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(combined_results)

    results = combined_results

    for rho in [0.10, 0.15]:
        sub_summary = []
        for n_ants in POP_SIZES:
            runs = [r for r in results if r["n_ants"] == n_ants and abs(r["rho"] - rho) < 1e-5]
            if not runs:
                continue
            weights = [r["best_weight"] for r in runs]
            fe_bests = [r["fe_best"] for r in runs]
            
            mean_w = np.mean(weights)
            std_w = np.std(weights)
            mean_fe = np.mean(fe_bests)
            
            sub_summary.append({
                "n_ants": n_ants,
                "rho": rho,
                "mean_w": mean_w,
                "std_w": std_w,
                "min_w": min(weights),
                "max_w": max(weights),
                "mean_fe": mean_fe,
            })
            
        unique_sub = get_top_unique_configs(sub_summary, param_keys=("n_ants", "rho"), top_n=5)
        headers = ["Rank", "N_ants", "Rho", "Media Peso (mu +- sigma)", "Min W", "Max W", "Mean FE Best"]
        rows = [
            [
                rank,
                s["n_ants"],
                s["rho"],
                f"{s['mean_w']:.2f} ± {s['std_w']:.2f}",
                s["min_w"],
                s["max_w"],
                f"{s['mean_fe']:.1f}"
            ]
            for rank, s in enumerate(unique_sub, 1)
        ]
        print_and_log_table(logger, f"RISULTATI MULTI-SEED DIMENSIONE COLONIA PER rho = {rho}", headers, rows)

if __name__ == "__main__":
    main()
