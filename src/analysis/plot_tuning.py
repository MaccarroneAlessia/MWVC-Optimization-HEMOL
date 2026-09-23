"""
plot_tuning.py — Generazione Heatmap per Tuning dei Parametri MMAS.

Carica i risultati della Grid Search da CSV (results/csv/grid_search_static.csv o simili)
oppure esegue una mini grid search se non presenti, e genera la Heatmap alpha vs beta/rho.

Scrive i log di esecuzione in results/log/plot_tuning.log.

Uso:
    python -m src.analysis.plot_tuning
    oppure
    python src/analysis/plot_tuning.py
"""

import sys
from pathlib import Path
import pandas as pd
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
from src.utils.logger import setup_logger

CSV_DIR = PROJECT_ROOT / "results" / "csv"
PLOTS_DIR = PROJECT_ROOT / "results" / "plots"
LOG_DIR = PROJECT_ROOT / "results" / "log"
DEFAULT_HEATMAP_PATH = PLOTS_DIR / "tuning_heatmap_lpi.png"

logger = setup_logger("TUNING", "plot_tuning.log")


def run_tuning_analysis(
    output_path: Path | str = None,
    force: bool = False,
) -> Path:
    """
    Genera la heatmap del tuning parametri e la salva nel percorso specificato.
    Se il file di output esiste e force=False, restituisce il percorso senza ricalcolare.
    """
    if output_path is None:
        output_path = DEFAULT_HEATMAP_PATH
    else:
        output_path = Path(output_path)

    if output_path.exists() and not force:
        msg = f"[INFO] Heatmap tuning già esistente: {output_path}"
        logger.info(msg)
        return output_path

    output_path.parent.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    csv_paths = [
        CSV_DIR / "grid_search_static.csv",
        CSV_DIR / "grid_search_report.csv",
        CSV_DIR / "tuning_results.csv",
    ]

    df_tune = None
    for p in csv_paths:
        if p.exists():
            logger.info(f"[INFO] Caricamento dati tuning da: {p}")
            df_tune = pd.read_csv(p)
            break

    if df_tune is None:
        logger.info("[INFO] Nessun CSV di tuning trovato. Avvio Grid Search per il tuning...")
        instance_path = PROJECT_ROOT / "wvcp-instances" / "vc_800_10000.txt"
        if not instance_path.exists():
            err_msg = f"Istanza non trovata per tuning: {instance_path}"
            logger.error(err_msg)
            raise FileNotFoundError(err_msg)

        logger.info(f"[INFO] Caricamento istanza per tuning: {instance_path.name}...")
        g_tune = Graph.load_from_file(instance_path)
        n_ants = max(5, g_tune.n // 10)
        alphas = [0.5, 1.0, 2.0]
        betas = [1.0, 2.0, 3.0]
        rhos = [0.05, 0.10, 0.20]
        total_combs = len(alphas) * len(betas) * len(rhos)

        records = []
        comb_count = 0
        for a in alphas:
            for b in betas:
                for r in rhos:
                    comb_count += 1
                    logger.info(f"  [Tuning {comb_count}/{total_combs}] Test alpha={a}, beta={b}, rho={r} (5 run)...")
                    weights = []
                    for seed in range(42, 47):
                        solver = MMAS_Solver(
                            g_tune, alpha=a, beta=b, rho=r, n_ants=n_ants, max_fe=5000, seed=seed
                        )
                        res = solver.solve(verbose=False)
                        weights.append(res["best_weight"])
                    mean_w = np.mean(weights)
                    logger.info(f"     -> Media peso: {mean_w:.1f}")
                    records.append({
                        "alpha": a, "beta": b, "rho": r,
                        "w": mean_w, "best_weight": mean_w,
                        "Class": "LPI" if "800_" in instance_path.name else "MPI",
                    })

        df_tune = pd.DataFrame(records)
        save_csv = CSV_DIR / "grid_search_static.csv"
        save_csv.parent.mkdir(parents=True, exist_ok=True)
        df_tune.to_csv(save_csv, index=False)
        logger.info(f"[OK] Grid Search salvata in: {save_csv}")

    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(7, 5))

    # Identifica la colonna valore
    val_col = None
    for col in ["w", "best_weight", "mean_weight", "mean_w"]:
        if col in df_tune.columns:
            val_col = col
            break

    if val_col is None:
        err_msg = f"Nessuna colonna valore trovata in {df_tune.columns.tolist()}"
        logger.error(err_msg)
        raise KeyError(err_msg)

    # Prepara la matrice pivot per la heatmap
    if "Class" in df_tune.columns and "LPI" in df_tune["Class"].values:
        sub = df_tune[df_tune["Class"] == "LPI"].copy()
        sub["alpha_beta"] = sub.apply(lambda row: f"α={row['alpha']}, β={row['beta']}", axis=1)
        pivot = sub.pivot_table(index="alpha_beta", columns="rho", values=val_col)
        sns.heatmap(pivot, annot=True, fmt=".0f", cmap="viridis_r", linewidths=0.5)
        plt.title("Heatmap dei Parametri (Classe LPI)", fontsize=13, fontweight="bold")
    else:
        pivot = df_tune.groupby(["alpha", "beta"])[val_col].mean().unstack()
        sns.heatmap(pivot, annot=True, fmt=".1f", cmap="YlGnBu_r", cbar_kws={"label": "Peso Medio"})
        plt.title("Heatmap Grid Search: Alpha vs Beta", fontsize=13, fontweight="bold")
        plt.xlabel("Beta (Euristica Chvátal)", fontsize=11)
        plt.ylabel("Alpha (Feromone)", fontsize=11)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    alt_path = output_path.parent / "tuning_heatmap.png"
    if alt_path != output_path:
        plt.savefig(alt_path, dpi=300)
    plt.close()
    logger.info(f"[SUCCESSO] Heatmap tuning salvata in: {output_path} e {alt_path}")
    return output_path


def main():
    run_tuning_analysis(force=True)


if __name__ == "__main__":
    main()
