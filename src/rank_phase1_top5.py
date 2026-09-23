"""
rank_phase1_top5.py — Selezione e Ranking Top 5 della Fase 1 e Multi-Seed

Legge i risultati della Fase 1 e della Validazione Multi-Seed, aggrega i dati per ogni
combinazione unica di iperparametri (alpha, beta, rho, n_ants) e stila le classifiche Top 5
ordinate rigorosamente per:
  1. Peso Minimo / Medio (miglior valore di Vertex Cover)
  2. Deviazione Standard (minore variabilità stocastica)
  3. FE medi a Best (velocità di convergenza)
  4. Tempo di Esecuzione (efficienza computazionale)
"""

import sys
import csv
from pathlib import Path
import numpy as np

_project_root = Path(__file__).resolve().parent.parent
_src_dir = Path(__file__).resolve().parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))
if str(_src_dir) not in sys.path:
    sys.path.insert(0, str(_src_dir))

from src.utils.logger import setup_logger, print_and_log_table, get_top_unique_configs

logger = setup_logger("RANK_FASE1")


def load_csv_records(csv_path: Path):
    if not csv_path.exists():
        return []
        
    records = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                records.append({
                    "n_ants": int(row["n_ants"]),
                    "alpha": float(row["alpha"]),
                    "beta": float(row["beta"]),
                    "rho": float(row["rho"]),
                    "seed": int(row["seed"]),
                    "best_weight": int(row["best_weight"]),
                    "fe_best": float(row["fe_best"]),
                    "reinits": int(row.get("reinits", 0)),
                    "time_s": float(row["time_s"]),
                    "phase": row.get("phase", "FASE 1"),
                })
            except (ValueError, KeyError):
                continue
    return records


def aggregate_by_config(records):
    grouped = {}
    for r in records:
        key = (r["alpha"], r["beta"], r["rho"], r["n_ants"])
        if key not in grouped:
            grouped[key] = []
        grouped[key].append(r)
        
    summary = []
    for (alpha, beta, rho, n_ants), runs in grouped.items():
        weights = [r["best_weight"] for r in runs]
        times = [r["time_s"] for r in runs]
        fe_bests = [r["fe_best"] for r in runs]
        
        mean_w = float(np.mean(weights))
        std_w = float(np.std(weights)) if len(weights) > 1 else 0.0
        min_w = int(np.min(weights))
        max_w = int(np.max(weights))
        mean_t = float(np.mean(times))
        mean_fe = float(np.mean(fe_bests))
        
        summary.append({
            "alpha": alpha,
            "beta": beta,
            "rho": rho,
            "n_ants": n_ants,
            "mean_weight": mean_w,
            "std_weight": std_w,
            "min_weight": min_w,
            "max_weight": max_w,
            "mean_time_s": mean_t,
            "mean_fe": mean_fe,
            "n_runs": len(runs),
        })
        
    # Ordinamento multi-criterio: 
    # 1. Peso Medio / Minimo (crescente)
    # 2. Deviazione Standard (crescente)
    # 3. FE medio (crescente)
    # 4. Tempo medio (crescente)
    summary.sort(key=lambda x: (x["mean_weight"], x["min_weight"], x["std_weight"], x["mean_fe"], x["mean_time_s"]))
    return summary


def save_ranking_csv(out_csv: Path, summary_list):
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["rank", "alpha", "beta", "rho", "n_ants", "min_weight", "mean_weight", "std_weight", "max_weight", "mean_fe", "mean_time_s", "n_runs"]
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for rank, s in enumerate(summary_list, 1):
            row_dict = {"rank": rank}
            row_dict.update(s)
            writer.writerow(row_dict)


def main():
    logger.info("=" * 75)
    logger.info("ELEZIONE E RANKING TOP 5 CONFIGURAZIONI (FASE 1 & MULTI-SEED)")
    logger.info("=" * 75)
    
    csv_tuning = _project_root / "results" / "csv" / "tuning_results.csv"
    csv_multiseed = _project_root / "results" / "csv" / "multiseed_top_candidates.csv"
    
    # ---------------------------------------------------------
    # 1. ELEZIONE TOP 5 FASE 1 (GRID SEARCH RUNS)
    # ---------------------------------------------------------
    records_p1 = load_csv_records(csv_tuning)
    p1_only = [r for r in records_p1 if "FASE 1" in r["phase"].upper()] or records_p1
    
    if p1_only:
        summary_p1 = aggregate_by_config(p1_only)
        top5_p1 = get_top_unique_configs(summary_p1, param_keys=("alpha", "beta", "rho", "n_ants"), top_n=5)
        
        headers_p1 = [
            "Rank", "Alpha", "Beta", "Rho", "N_ants", 
            "Min Peso", "Media Peso (mu +- sigma)", "Mean FE", "Tempo Medio (s)"
        ]
        rows_p1 = [
            [
                rank,
                s["alpha"],
                s["beta"],
                s["rho"],
                s["n_ants"],
                s["min_weight"],
                f"{s['mean_weight']:.2f} ± {s['std_weight']:.2f}",
                f"{s['mean_fe']:.1f}",
                f"{s['mean_time_s']:.2f}s"
            ]
            for rank, s in enumerate(top5_p1, 1)
        ]
        print_and_log_table(logger, "CLASSIFICA TOP 5 CONFIGURAZIONI (FASE 1 - GRID SEARCH)", headers_p1, rows_p1)
        save_ranking_csv(_project_root / "results" / "csv" / "top5_phase1_ranking.csv", top5_p1)

    # ---------------------------------------------------------
    # 2. ELEZIONE TOP 5 MULTI-SEED (ROBUSTEZZA SU 5 SEED)
    # ---------------------------------------------------------
    records_ms = load_csv_records(csv_multiseed)
    if records_ms:
        summary_ms = aggregate_by_config(records_ms)
        top5_ms = get_top_unique_configs(summary_ms, param_keys=("alpha", "beta", "rho", "n_ants"), top_n=5)
        
        headers_ms = [
            "Rank", "Alpha", "Beta", "Rho", "N_ants", 
            "Media Peso (mu +- sigma)", "Min W", "Max W", "Mean FE", "Tempo Medio (s)"
        ]
        rows_ms = [
            [
                rank,
                s["alpha"],
                s["beta"],
                s["rho"],
                s["n_ants"],
                f"{s['mean_weight']:.2f} ± {s['std_weight']:.2f}",
                s["min_weight"],
                s["max_weight"],
                f"{s['mean_fe']:.1f}",
                f"{s['mean_time_s']:.2f}s"
            ]
            for rank, s in enumerate(top5_ms, 1)
        ]
        print_and_log_table(logger, "CLASSIFICA TOP 5 CONFIGURAZIONI (MULTI-SEED ROBUSTEZZA - 5 SEEDS)", headers_ms, rows_ms)
        save_ranking_csv(_project_root / "results" / "csv" / "top5_multiseed_ranking.csv", top5_ms)


if __name__ == "__main__":
    main()

