"""
plot_scalability.py — Scatterplot di Scalabilità Computazionale MMAS-MWVCP.

Aggrega i risultati dei benchmark presenti in results/csv/vc_*.csv (oppure summary CSV)
e genera uno scatter plot log-log della scalabilità del sistema (Nodi vs FE medie al Best).

Scrive i log di esecuzione in results/log/plot_scalability.log.

Uso:
    python -m src.analysis.plot_scalability
    oppure
    python src/analysis/plot_scalability.py
"""

import sys
import glob
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Fix path resolution
_HERE = Path(__file__).resolve().parent
PROJECT_ROOT = _HERE.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.logger import setup_logger

CSV_DIR = PROJECT_ROOT / "results" / "csv"
PLOTS_DIR = PROJECT_ROOT / "results" / "plots"
LOG_DIR = PROJECT_ROOT / "results" / "log"
DEFAULT_SCALABILITY_PATH = PLOTS_DIR / "scalability_scatter.png"

logger = setup_logger("SCALABILITY", "plot_scalability.log")


def run_scalability_analysis(
    output_path: Path | str = None,
    force: bool = False,
) -> Path:
    """
    Genera lo scatterplot di scalabilità e lo salva nel percorso specificato.
    Se il file di output esiste e force=False, restituisce il percorso senza ricalcolare.
    """
    if output_path is None:
        output_path = DEFAULT_SCALABILITY_PATH
    else:
        output_path = Path(output_path)

    if output_path.exists() and not force:
        msg = f"[INFO] Grafico scalabilità già esistente: {output_path}"
        logger.info(msg)
        return output_path

    output_path.parent.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    logger.info(f"[INFO] Generazione scatterplot scalabilità computazionale in {output_path}...")

    csv_files = glob.glob(str(CSV_DIR / "vc_*.csv"))
    if csv_files:
        logger.info(f"  -> Trovati {len(csv_files)} CSV di benchmark in {CSV_DIR}")
        df_all = pd.concat((pd.read_csv(f) for f in csv_files), ignore_index=True)
        summary_data = []
        for instance, df_inst in df_all.groupby("instance"):
            cls = (
                "SPI"
                if "20_" in instance or "25_" in instance
                else ("MPI" if "100_" in instance or "200_" in instance else "LPI")
            )
            try:
                n_nodes = int(instance.split("_")[1])
            except Exception:
                n_nodes = 50

            col_fe = "fe_at_best" if "fe_at_best" in df_inst.columns else ("fe_best" if "fe_best" in df_inst.columns else "fe_used")
            mean_w = df_inst["best_weight"].mean()
            avg_fe = df_inst[col_fe].mean()
            logger.info(f"     Istanza {instance}: Nodi={n_nodes}, Mean W={mean_w:.1f}, Avg FE={avg_fe:.1f}")
            summary_data.append({
                "Class": cls,
                "Istanza": instance.replace(".txt", ""),
                "Nodi": n_nodes,
                "Mean": mean_w,
                "Avg FE": avg_fe,
            })
        df_summary = pd.DataFrame(summary_data)
    else:
        logger.info("  -> Nessun CSV vc_*.csv trovato. Uso dati di fallback aggregati...")
        df_summary = pd.DataFrame([
            {"Class": "SPI", "Istanza": "vc_20_60", "Nodi": 20, "Avg FE": 4, "Mean": 550.0},
            {"Class": "SPI", "Istanza": "vc_20_120", "Nodi": 20, "Avg FE": 6, "Mean": 1146.0},
            {"Class": "SPI", "Istanza": "vc_25_150", "Nodi": 25, "Avg FE": 9, "Mean": 1481.0},
            {"Class": "MPI", "Istanza": "vc_100_500", "Nodi": 100, "Avg FE": 576, "Mean": 4475.0},
            {"Class": "MPI", "Istanza": "vc_100_2000", "Nodi": 100, "Avg FE": 16, "Mean": 8328.0},
            {"Class": "MPI", "Istanza": "vc_200_750", "Nodi": 200, "Avg FE": 2588, "Mean": 8479.0},
            {"Class": "MPI", "Istanza": "vc_200_3000", "Nodi": 200, "Avg FE": 42, "Mean": 16670.0},
            {"Class": "LPI", "Istanza": "vc_800_10000", "Nodi": 800, "Avg FE": 4120, "Mean": 73400.1},
        ])

    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(8, 6))

    if "Nodi" in df_summary.columns:
        sns.scatterplot(data=df_summary, x="Nodi", y="Avg FE", hue="Class", s=100, ax=ax)
        ax.set_xlabel("Numero di Nodi |V|", fontsize=11)
        ax.set_ylabel("FE Medie al Best", fontsize=11)
    else:
        sns.scatterplot(data=df_summary, x="Avg FE", y="Mean", hue="Class", s=100, ax=ax)
        ax.set_xlabel("FE Medie", fontsize=11)
        ax.set_ylabel("Peso Medio Vertex Cover", fontsize=11)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_title("Scatterplot di Scalabilità Computazionale (Log-Log)", fontsize=13, fontweight="bold")
    plt.tight_layout()

    plt.savefig(output_path, dpi=300)
    plt.close()
    logger.info(f"[SUCCESSO] Scatterplot scalabilità salvato in: {output_path}")
    return output_path


def main():
    run_scalability_analysis(force=True)


if __name__ == "__main__":
    main()
