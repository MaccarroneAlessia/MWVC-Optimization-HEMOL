"""
run_ablation_switch.py — Esperimento di Ablation: Switch Fisso vs Ciclo-Dinamico (su rho=0.10 e rho=0.15)

Confronta la strategia di transizione Exploration->Exploitation proposta:
  1. cycle: Transizione ciclo-dinamica sincronizzata con la stagnazione (Strategy 1 - Proposta)
  2. fixed_budget: Transizione a quota fissa del budget totale (75% FE_max - ACO Tradizionale)
  3. always: Sempre Global-Best (Intensificazione pura)
  4. never: Sempre Iteration-Best (Esplorazione pura)
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

logger = setup_logger("ABLATION_SWITCH")

TARGET_INSTANCE = "wvcp-instances/vc_800_10000.txt"
SEEDS = [42, 101, 202, 303, 404]

SWITCH_MODES = [
    {"mode": "cycle", "description": "Ciclo-Dinamica (Sincronizzata col Reset - Proposta)"},
    {"mode": "fixed_budget", "description": "Switch Fisso (75% FE_max - Tradizionale)"},
    {"mode": "never", "description": "Solo Iteration-Best (Esplorazione pura)"},
    {"mode": "always", "description": "Solo Global-Best (Intensificazione pura)"},
]


def run_ablation_task(args):
    mode_info, rho, seed, graph_path_str = args
    graph = Graph.load_from_file(graph_path_str)
    
    params = {
        "n_ants": 80,
        "alpha": 1.0,
        "beta": 2.0,
        "rho": rho,
        "gamma": 1.0,
        "max_fe": 20_000,
        "gb_switch_ratio": 0.75,
        "gb_switch_mode": mode_info["mode"],
        "pruning_strategy": "greedy",
        "seed": seed,
    }
    
    solver = MMAS_Solver(graph, **params)
    t0 = time.time()
    res = solver.solve(verbose=False)
    elapsed = time.time() - t0
    
    return {
        "mode": mode_info["mode"],
        "description": mode_info["description"],
        "rho": rho,
        "seed": seed,
        "best_weight": int(res["best_weight"]),
        "fe_best": res["fe_at_best"],
        "reinits": res["n_reinits"],
        "time_s": round(elapsed, 2),
    }


def main():
    logger.info("=" * 75)
    logger.info("AVVIO ABLATION STUDY: SWITCH FISSO VS CICLO-DINAMICO (rho=0.10 e rho=0.15)")
    logger.info("=" * 75)
    
    graph_path = _project_root / TARGET_INSTANCE
    if not graph_path.exists():
        logger.error(f"Istanza non trovata: {graph_path}")
        return
        
    csv_dir = _project_root / "results" / "csv"
    csv_dir.mkdir(parents=True, exist_ok=True)
    out_csv = csv_dir / "ablation_switch_results.csv"
    
    existing_results = []
    completed_keys = set()
    if out_csv.exists():
        with open(out_csv, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    res = {
                        "mode": row["mode"],
                        "description": row["description"],
                        "rho": float(row["rho"]),
                        "seed": int(row["seed"]),
                        "best_weight": int(row["best_weight"]),
                        "fe_best": float(row["fe_best"]),
                        "reinits": int(row["reinits"]),
                        "time_s": float(row["time_s"]),
                    }
                    existing_results.append(res)
                    completed_keys.add((row["mode"], float(row["rho"]), int(row["seed"])))
                except (ValueError, KeyError):
                    continue
                    
    tasks = []
    for rho in [0.10, 0.15]:
        for mode_info in SWITCH_MODES:
            for seed in SEEDS:
                if (mode_info["mode"], rho, seed) not in completed_keys:
                    tasks.append((mode_info, rho, seed, str(graph_path)))
                
    logger.info(f"Trovate {len(existing_results)} esecuzioni ablation valide nel CSV.")
    logger.info(f"Task rimanenti da eseguire: {len(tasks)}")
    
    new_results = []
    if tasks:
        with open(out_csv, "a", newline="", encoding="utf-8") as f:
            fieldnames = ["mode", "description", "rho", "seed", "best_weight", "fe_best", "reinits", "time_s"]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if not existing_results:
                writer.writeheader()
            
            max_workers = min(os.cpu_count() or 4, len(tasks))
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                futures = {executor.submit(run_ablation_task, task): task for task in tasks}
                for future in as_completed(futures):
                    res = future.result()
                    new_results.append(res)
                    writer.writerow(res)
                    f.flush()
                    logger.info(
                        f"  [rho={res['rho']} | mode={res['mode']:<12s} | seed={res['seed']}] "
                        f"w={res['best_weight']} | FE_best={res['fe_best']} | reinits={res['reinits']} | t={res['time_s']}s"
                    )

    results = existing_results + new_results

    for rho in [0.10, 0.15]:
        summary = []
        for mode_info in SWITCH_MODES:
            mode = mode_info["mode"]
            runs = [r for r in results if r["mode"] == mode and abs(r["rho"] - rho) < 1e-5]
            if not runs:
                continue
            weights = [r["best_weight"] for r in runs]
            
            mean_w = float(np.mean(weights))
            std_w = float(np.std(weights))
            
            summary.append({
                "mode": mode,
                "description": mode_info["description"],
                "rho": rho,
                "mean_weight": mean_w,
                "std_weight": std_w,
            })
            
        summary.sort(key=lambda x: (x["mean_weight"], x["std_weight"]))
        unique_summary = get_top_unique_configs(summary, param_keys=("mode", "rho"), top_n=5)
        headers = ["Rank", "Modalità Switch", "Rho", "Descrizione", "Media Peso (mu +- sigma)"]
        rows = [
            [
                rank,
                s["mode"],
                s["rho"],
                s["description"],
                f"{s['mean_weight']:.2f} ± {s['std_weight']:.2f}"
            ]
            for rank, s in enumerate(unique_summary, 1)
        ]
        print_and_log_table(logger, f"RANKING ABLATION SWITCH PER rho = {rho} (ORDINATO PER MEDIA PESO)", headers, rows)

if __name__ == "__main__":
    main()
