"""
main.py — Esecuzione completa del benchmark MMAS.

Schema di esecuzione (da consegna):
  - SPI e MPI: 10 istanze distinte per configurazione, 1 run ciascuna (seed fisso 42).
  - LPI:       1 istanza per configurazione, 10 run indipendenti con seed variabili.

Output:
  results/csv/<prefix>.csv  — un file per ogni configurazione (n, m)
  Colonne: run, instance, seed, best_weight, cover_size,
           fe_at_best, fe_used, fe_budget_pct, iterations,
           n_reinits, stopped_early, time_seconds
"""

import sys
import os
import csv
import time
from pathlib import Path
from collections import defaultdict

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)

from src.core.graph import Graph
from src.core.mmas import MMAS_Solver
from src.utils.logger import setup_logger
from src.core.exact_solver import compute_lower_bound, solve_exact_ilp, quality_gap

logger = setup_logger("MAIN")

INSTANCE_CLASSES = {
    "SPI": {(20, 60): "vc_20_60", (20, 120): "vc_20_120", (25, 150): "vc_25_150"},
    "MPI": {
        (100, 500): "vc_100_500",
        (100, 2000): "vc_100_2000",
        (200, 750): "vc_200_750",
        (200, 3000): "vc_200_3000",
    },
    "LPI": {(800, 10000): "vc_800_10000"},
}

CSV_HEADER = [
    "run",
    "instance",
    "seed",
    "best_weight",
    "lower_bound",
    "gap_pct_lb",
    "opt_weight",
    "gap_pct_opt",
    "cover_size",
    "fe_at_best",
    "fe_used",
    "fe_budget_pct",
    "iterations",
    "n_reinits",
    "stopped_early",
    "time_seconds",
]

SOLVER_PARAMS = {
    "alpha": 1.0,
    "beta": 2.0,
    "gamma": 1.0,
    "rho": 0.10,
    "max_fe": 20_000,
    #"stagnation_limit": 50,
    #"max_reinits": 5,
    # stagnation_limit non specificato → calcolato adativamente in mmas.py
    "pruning_strategy": "greedy",
}


def discover_instances(instances_dir: Path) -> dict:
    """Raggruppa i file .txt per prefisso (es. 'vc_20_60')."""
    groups = defaultdict(list)
    for f in sorted(instances_dir.glob("*.txt")):
        name = f.stem
        parts = name.split("_")
        if len(parts) == 4:
            prefix = "_".join(parts[:3])
        elif len(parts) == 3:
            prefix = name
        else:
            continue
        groups[prefix].append(str(f))
    return dict(groups)

def run_solver(file_path: str, seed: int) -> tuple:
    """Carica il grafo, esegue una run e restituisce (graph, result_dict, elapsed_seconds)."""
    graph = Graph.load_from_file(file_path)
    
    # Formiche Adattive: proporzionali alla dimensione del grafo (minimo 5)
    n_ants = max(5, graph.n // 10)
    
    params = {**SOLVER_PARAMS, "n_ants": n_ants, "seed": seed}
    solver = MMAS_Solver(graph, **params)
    t0 = time.time()
    res = solver.solve(verbose=False)
    elapsed = time.time() - t0
    return graph, res, elapsed


def get_max_allowed_time(n_nodes: int) -> float:
    """Restituisce la soglia di tempo massima plausibile in secondi basata sui nodi."""
    if n_nodes <= 30:
        return 120.0     # 2 minuti max per SPI (n <= 25)
    elif n_nodes <= 300:
        return 900.0     # 15 minuti max per MPI (n <= 200)
    else:
        return 3600.0    # 1 ora max per LPI (n = 800)


def run_solver_safe(file_path: str, seed: int, max_retries: int = 3) -> tuple:
    """
    Esegue run_solver controllando che il tempo impiegato sia plausibile.
    Se supera la soglia (es. 1h per LPI, 15m per MPI, 2m per SPI a causa di standby/sleep del PC),
    stampa un errore esplicito e riesegue l'istanza automaticamente.
    """
    for attempt in range(1, max_retries + 1):
        graph, res, elapsed = run_solver(file_path, seed)
        max_allowed = get_max_allowed_time(graph.n)

        if elapsed <= max_allowed:
            return graph, res, elapsed

        fname = Path(file_path).name
        logger.error(
            f"[ERRORE TEMPO NON PLAUSIBILE] Istanza '{fname}' (seed {seed}): "
            f"tempo misurato {elapsed:.2f}s > limite {max_allowed:.0f}s (probabile standby del PC). "
            f"Riesecuzione automatica in corso... (tentativo {attempt}/{max_retries})"
        )

    logger.warning(
        f"[WARN] Istanza '{Path(file_path).name}' ha superato il limite dopo {max_retries} tentativi. "
        f"Si procede con il tempo misurato ({elapsed:.2f}s)."
    )
    return graph, res, elapsed


def build_csv_row(
    run_idx: int,
    file_path: str,
    seed: int,
    res: dict,
    elapsed: float,
    lb: float,
    opt: float | None,
) -> dict:
    """Costruisce un dizionario-riga con tutte le metriche, inclusi lower bound e gap."""
    fe_used = res["fe_used"]
    fe_at_best = res["fe_at_best"]
    fe_budget_pct = round((fe_at_best / fe_used * 100) if fe_used > 0 else 0.0, 2)
    bw = res["best_weight"]
    return {
        "run": run_idx,
        "instance": Path(file_path).name,
        "seed": seed,
        "best_weight": f"{bw:.0f}",
        "lower_bound": f"{lb:.0f}",
        "gap_pct_lb": quality_gap(bw, lb),
        "opt_weight": f"{opt:.0f}" if opt is not None else "N/A",
        "gap_pct_opt": quality_gap(bw, opt) if opt is not None else "N/A",
        "cover_size": res["cover_size"],
        "fe_at_best": fe_at_best,
        "fe_used": fe_used,
        "fe_budget_pct": fe_budget_pct,
        "iterations": res["iterations"],
        "n_reinits": res["n_reinits"],
        "stopped_early": int(res["stopped_by_reinit"]),
        "time_seconds": f"{elapsed:.3f}",
    }


def main():
    instances_dir = Path(project_root) / "wvcp-instances"
    csv_dir = Path(project_root) / "results" / "csv"
    csv_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("Benchmark MMAS-MWVCP  ->  output CSV per configurazione")
    logger.info("=" * 60)
    logger.info("LEGENDA OUTPUT:")
    logger.info("  w       = best_weight (peso minimo trovato)")
    logger.info("  lb      = lower bound (soglia minima teorica calcolata col matching)")
    logger.info("  opt     = ottimo globale esatto calcolato con ILP (solo SPI)")
    logger.info("  gap     = distanza % tra il risultato 'w' e 'lb' o 'opt' (0% = perfetto)")
    logger.info("  fe_best = valutazione in cui è stato trovato il best_weight")
    logger.info("  fe_tot  = valutazioni (budget) totali consumate")
    logger.info("  reinits = numero di reset (stagnation recovery) effettuati")
    logger.info("  t       = tempo impiegato per la singola istanza")
    logger.info("=" * 60)

    groups = discover_instances(instances_dir)

    for class_name, class_instances in INSTANCE_CLASSES.items():
        logger.info(f"\n[{class_name}] Elaborazione istanze in corso...")
        is_lpi = class_name == "LPI"

        for (n, m), prefix in class_instances.items():
            if prefix not in groups:
                logger.warning(f"  [SKIP] Prefisso '{prefix}' non trovato in {instances_dir}")
                continue

            files = groups[prefix]
            csv_path = csv_dir / f"{prefix}.csv"

            with open(csv_path, "w", newline="", encoding="utf-8") as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=CSV_HEADER)
                writer.writeheader()

                # SPI ha n piccolo: calcoliamo la soluzione ottima esatta via ILP.
                # MPI e LPI: solo il lower bound via matching (ILP sarebbe troppo lento).
                is_spi = (class_name == "SPI")

                if is_lpi:
                    # LPI: 10 run indipendenti sulla stessa istanza con seed variabili.
                    # Calcoliamo LB una sola volta (è deterministica sul grafo).
                    file_path = files[0]
                    first_graph, first_res, first_elapsed = run_solver_safe(file_path, seeds := [run_idx * 42 + 7 for run_idx in range(1, 11)][0])
                    lb = compute_lower_bound(first_graph)
                    logger.info(f"  {prefix} - Lower Bound (matching) = {lb:.0f}")

                    # Scrivi la prima run già eseguita
                    row = build_csv_row(1, file_path, seeds, first_res, first_elapsed, lb, None)
                    writer.writerow(row)
                    logger.info(
                        f"  {prefix} - Run  1/10 | w={first_res['best_weight']:.0f} | "
                        f"lb={lb:.0f} (gap={quality_gap(first_res['best_weight'], lb):.1f}%) | "
                        f"opt=N/A (gap=N/A) | "
                        f"fe_best={first_res['fe_at_best']} | fe_tot={first_res['fe_used']} | "
                        f"reinits={first_res['n_reinits']} | t={first_elapsed:.2f}s"
                    )

                    for run_idx in range(2, 11):
                        seed = run_idx * 42 + 7
                        _, res, elapsed = run_solver_safe(file_path, seed)
                        row = build_csv_row(run_idx, file_path, seed, res, elapsed, lb, None)
                        writer.writerow(row)
                        logger.info(
                            f"  {prefix} - Run {run_idx:2d}/10 | w={res['best_weight']:.0f} | "
                            f"lb={lb:.0f} (gap={quality_gap(res['best_weight'], lb):.1f}%) | "
                            f"opt=N/A (gap=N/A) | "
                            f"fe_best={res['fe_at_best']} | fe_tot={res['fe_used']} | "
                            f"reinits={res['n_reinits']} | t={elapsed:.2f}s"
                        )
                else:
                    # SPI/MPI: 1 run per ciascuna delle 10 istanze distinte (seed fisso).
                    for run_idx, file_path in enumerate(files, start=1):
                        seed = 42
                        graph, res, elapsed = run_solver_safe(file_path, seed)

                        # Lower bound per tutti; OPT esatto solo per SPI
                        lb = compute_lower_bound(graph)
                        if is_spi:
                            ilp_result = solve_exact_ilp(graph, time_limit_s=60.0)
                            opt = ilp_result[0] if ilp_result is not None else None
                        else:
                            opt = None

                        row = build_csv_row(run_idx, file_path, seed, res, elapsed, lb, opt)
                        writer.writerow(row)

                        gap_lb_str = f"{quality_gap(res['best_weight'], lb):.1f}%"
                        gap_opt_str = f"{quality_gap(res['best_weight'], opt):.1f}%" if opt else "N/A"
                        opt_str = f"{opt:.0f}" if opt is not None else "N/A"
                        logger.info(
                            f"  {prefix} - Istanza {run_idx:2d}/{len(files)} | "
                            f"w={res['best_weight']:.0f} | "
                            f"lb={lb:.0f} (gap={gap_lb_str}) | "
                            f"opt={opt_str} (gap={gap_opt_str}) | "
                            f"fe_best={res['fe_at_best']} | fe_tot={res['fe_used']} | "
                            f"reinits={res['n_reinits']} | t={elapsed:.2f}s"
                        )

            logger.info(f"  -> Salvato: {csv_path}")

    logger.info(f"\n[SUCCESSO] Tutti i CSV salvati in: {csv_dir}")
    #logger.info("Eseguire 'python src/aggregate_results.py' per il riassunto per classe.")


if __name__ == "__main__":
    main()
