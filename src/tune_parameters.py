"""
tune_parameters.py — Parameter Tuning Robusto a 2 Fasi con Validazione Multi-Seed e Parallelizzazione

Caratteristiche:
  1. Fase 1: Esplorazione esaustiva su alpha/beta/rho (inclusi rho=0.15 e rho=0.20 per testare la saturazione del trend).
  2. Filtro Multi-Seed: I Top 3 candidati della Fase 1 vengono valutati su 5 seed differenti (seed 42, 101, 202, 303, 404)
     per calcolare peso medio e deviazione standard (mu +- sigma) eliminando il rumore stocastico.
  3. Fase 2: Validazione della dimensione colonia n_ants in [40, 60, 80, 120, 160] con il vincitore multi-seed.
  4. Parallelizzazione Multi-Core: Utilizza ProcessPoolExecutor per accelerare drammaticamente le esecuzioni.
"""

import sys
import time
import csv
import itertools
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

# Risoluzione dinamica dei percorsi
_project_root = Path(__file__).resolve().parent.parent
_src_dir = Path(__file__).resolve().parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))
if str(_src_dir) not in sys.path:
    sys.path.insert(0, str(_src_dir))

try:
    from src.core.graph import Graph
    from src.core.mmas import MMAS_Solver
    from src.utils.logger import setup_logger, print_and_log_table, get_top_unique_configs
except ImportError:
    from core.graph import Graph
    from core.mmas import MMAS_Solver
    from utils.logger import setup_logger, print_and_log_table, get_top_unique_configs

logger = setup_logger("TUNING")

# --- GRIGLIE DEI PARAMETRI ---

# FASE 1: Esplorazione estesa su alpha, beta, rho (inclusi rho=0.15 e rho=0.20)
PARAM_GRID_FASE1 = {
    "n_ants": [80],                     # Formula adattiva max(5, |V|/10) per LPI n=800
    "alpha": [1.0, 2.0],                # Test sia alpha=1.0 che alpha=2.0
    "beta": [2.0, 5.0],                 # Euristica Chvátal pesata e spinta
    "rho": [0.02, 0.10, 0.15, 0.20],    # Test esteso per verificare saturazione di rho
}

# Impostazioni fisse
FIXED_PARAMS = {
    "gamma": 1.0,
    "max_fe": 20_000,
    "pruning_strategy": "greedy",
}

# Seed per la validazione multi-seed
MULTI_SEEDS = [42, 101, 202, 303, 404]

# Istanza di tuning (LPI: vc_800_10000.txt)
TARGET_INSTANCE = "wvcp-instances/vc_800_10000.txt"


def evaluate_config_task(args):
    """Funzione di supporto eseguita in parallelo dai processi worker."""
    config, graph_path_str, fixed_params, seed, idx_str, phase_name = args
    graph = Graph.load_from_file(graph_path_str)
    
    params = {**config, **fixed_params, "seed": seed}
    solver = MMAS_Solver(graph, **params)
    
    t0 = time.time()
    res = solver.solve(verbose=False)
    elapsed = time.time() - t0

    return {
        "config_id": idx_str,
        "phase": phase_name,
        "seed": seed,
        "n_ants": config["n_ants"],
        "alpha": config["alpha"],
        "beta": config["beta"],
        "rho": config["rho"],
        "best_weight": int(res["best_weight"]),
        "fe_best": res["fe_at_best"],
        "reinits": res["n_reinits"],
        "time_s": round(elapsed, 2),
    }


def generate_combinations(grid):
    """Genera tutte le combinazioni di parametri da una griglia."""
    keys = list(grid.keys())
    values = list(grid.values())
    for combination in itertools.product(*values):
        yield dict(zip(keys, combination))


def run_phase_parallel(phase_name, graph_path_str, combinations, csv_writer, results_list, global_idx_start=1, seed=42):
    """Esegue un blocco di test in parallelo su più processi."""
    logger.info(f"\n>>> AVVIO {phase_name} ({len(combinations)} combinazioni in parallelo) <<<")
    
    tasks = []
    for idx, config in enumerate(combinations, global_idx_start):
        task_arg = (config, graph_path_str, FIXED_PARAMS, seed, f"{idx}", phase_name)
        tasks.append(task_arg)

    phase_results = []
    
    # Esecuzione in parallelo sul pool di processi CPU disponibili
    with ProcessPoolExecutor() as executor:
        futures = {executor.submit(evaluate_config_task, task): task for task in tasks}
        for future in as_completed(futures):
            res_row = future.result()
            phase_results.append(res_row)
            results_list.append(res_row)
            csv_writer.writerow(res_row)
            logger.info(
                f"  [{res_row['config_id']:>2s}] {phase_name} -> ants={res_row['n_ants']}, "
                f"a={res_row['alpha']}, b={res_row['beta']}, r={res_row['rho']} | "
                f"w={res_row['best_weight']} | fe_best={res_row['fe_best']:5d} | t={res_row['time_s']:.2f}s"
            )

    phase_results.sort(key=lambda x: (x["best_weight"], x["fe_best"]))
    return phase_results


def run_multi_seed_validation(top_candidates, graph_path_str, csv_writer, results_list):
    """Valuta i top candidati su più seed per eliminare il rumore stocastico."""
    logger.info(f"\n>>> AVVIO VALIDAZIONE MULTI-SEED sui Top {len(top_candidates)} Candidati su {len(MULTI_SEEDS)} seed <<<")
    
    validated_winners = []
    
    tasks = []
    for cand_idx, cand in enumerate(top_candidates, 1):
        config = {
            "n_ants": cand["n_ants"],
            "alpha": cand["alpha"],
            "beta": cand["beta"],
            "rho": cand["rho"],
        }
        for s in MULTI_SEEDS:
            task_arg = (config, graph_path_str, FIXED_PARAMS, s, f"MS-{cand_idx}", "MULTI-SEED")
            tasks.append((cand_idx, task_arg))

    # Esecuzione parallela
    ms_runs_by_cand = {c_idx: [] for c_idx in range(1, len(top_candidates) + 1)}
    
    with ProcessPoolExecutor() as executor:
        futures = {executor.submit(evaluate_config_task, t[1]): t[0] for t in tasks}
        for future in as_completed(futures):
            c_idx = futures[future]
            res_row = future.result()
            ms_runs_by_cand[c_idx].append(res_row)
            results_list.append(res_row)
            csv_writer.writerow(res_row)

    # Aggregazione statistica per ciascun candidato
    logger.info("\n=== RISULTATI VALIDAZIONE MULTI-SEED ===")
    for cand_idx, cand in enumerate(top_candidates, 1):
        runs = ms_runs_by_cand[cand_idx]
        weights = [r["best_weight"] for r in runs]
        fe_bests = [r["fe_best"] for r in runs]
        
        import numpy as np
        mean_w = np.mean(weights)
        std_w = np.std(weights)
        mean_fe = np.mean(fe_bests)
        
        validated_entry = {
            "config": cand,
            "mean_weight": mean_w,
            "std_weight": std_w,
            "mean_fe_best": mean_fe,
            "weights": weights
        }
        validated_winners.append(validated_entry)
        
        logger.info(
            f"  Candidato #{cand_idx} (a={cand['alpha']}, b={cand['beta']}, r={cand['rho']}): "
            f"w_medio = {mean_w:.2f} ± {std_w:.2f} | min={min(weights)} | max={max(weights)} | FE_medio = {mean_fe:.1f}"
        )

    # Selezione del vero vincitore robusto (minor peso medio, minor std)
    validated_winners.sort(key=lambda x: (x["mean_weight"], x["std_weight"], x["mean_fe_best"]))
    robust_winner = validated_winners[0]["config"]
    
    logger.info(f"\n>>> VINCITORE MULTI-SEED ROBUSTO: a={robust_winner['alpha']}, b={robust_winner['beta']}, r={robust_winner['rho']} <<<")
    return robust_winner


def main():
    logger.info("=" * 65)
    logger.info("PARAMETER TUNING MULTI-SEED E PARALLELO PER MMAS-MWVCP")
    logger.info("=" * 65)

    graph_path = _project_root / TARGET_INSTANCE
    if not graph_path.exists():
        graph_path = Path(TARGET_INSTANCE)
    if not graph_path.exists():
        graph_path = _src_dir / TARGET_INSTANCE
    if not graph_path.exists():
        logger.error(f"Impossibile trovare l'istanza: {TARGET_INSTANCE}")
        return

    graph_path_str = str(graph_path)
    logger.info(f"Istanza target: {graph_path.name}")

    csv_dir = _project_root / "results" / "csv"
    csv_dir.mkdir(parents=True, exist_ok=True)
    csv_path = csv_dir / "tuning_results.csv"

    all_results = []

    with open(csv_path, "w", newline="", encoding="utf-8") as csvfile:
        fieldnames = ["config_id", "phase", "seed", "n_ants", "alpha", "beta", "rho", "best_weight", "fe_best", "reinits", "time_s"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        # --- FASE 1: Esplorazione Esaustiva ---
        combos_fase1 = list(generate_combinations(PARAM_GRID_FASE1))
        fase1_sorted = run_phase_parallel("FASE 1", graph_path_str, combos_fase1, writer, all_results, global_idx_start=1, seed=42)
        csvfile.flush()

        # Estrazione dei Top 3 Candidati per la validazione multi-seed
        # Rimuove eventuali duplicati di parametri se presenti
        top_candidates = []
        seen = set()
        for res in fase1_sorted:
            key = (res["alpha"], res["beta"], res["rho"])
            if key not in seen:
                seen.add(key)
                top_candidates.append(res)
            if len(top_candidates) >= 3:
                break

        # --- VALIDAZIONE MULTI-SEED sui Top 3 ---
        robust_winner = run_multi_seed_validation(top_candidates, graph_path_str, writer, all_results)
        csvfile.flush()

        # --- FASE 2: Validazione n_ants ---
        PARAM_GRID_FASE2 = {
            "n_ants": [40, 60, 80, 120, 160],
            "alpha": [robust_winner["alpha"]],
            "beta": [robust_winner["beta"]],
            "rho": [robust_winner["rho"]],
        }
        combos_fase2 = list(generate_combinations(PARAM_GRID_FASE2))
        run_phase_parallel("FASE 2 (n_ants)", graph_path_str, combos_fase2, writer, all_results, global_idx_start=len(combos_fase1) + 1, seed=42)
        csvfile.flush()

    top_5_unique = get_top_unique_configs(all_results, param_keys=("n_ants", "alpha", "beta", "rho"), top_n=5)
    headers = ["Rank", "Fase", "Seed", "N_ants", "Alpha", "Beta", "Rho", "Best Weight", "FE at Best", "Time (s)"]
    rows = [
        [i, res["phase"], res["seed"], res["n_ants"], res["alpha"], res["beta"], res["rho"], res["best_weight"], res["fe_best"], res["time_s"]]
        for i, res in enumerate(top_5_unique, 1)
    ]
    print_and_log_table(logger, "TOP 5 CONFIGURAZIONI UNICHE GLOBALI DA TUNING", headers, rows)


if __name__ == "__main__":
    main()
