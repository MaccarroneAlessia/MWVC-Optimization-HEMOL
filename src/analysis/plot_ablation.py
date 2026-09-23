"""
plot_ablation.py — Grafici di analisi ed Ablation Study per MMAS-MWVCP.

Fornisce:
  1. run_ablation_analysis: esegue l'Ablation Study sulle varianti dell'algoritmo (No Recovery, No Pruning, No Pheromones, No Heuristic)
     e genera il barplot delle performance medie salvando in results/plots/ablation_study.png.
  2. plot_ablation_study: legge i CSV reali da results/csv/ e produce boxplot, scatter plot e distribuzione budget.

Scrive i log di esecuzione in results/log/plot_ablation.log.

Uso:
    python -m src.analysis.plot_ablation
    oppure
    python src/analysis/plot_ablation.py
"""

import csv
import sys
from pathlib import Path
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
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
DEFAULT_ABLATION_PLOT = PLOTS_DIR / "ablation_study_lpi.png"

logger = setup_logger("ABLATION", "plot_ablation.log")

PREFIX_TO_CLASS = {
    "vc_20_60": "SPI", "vc_20_120": "SPI", "vc_25_150": "SPI",
    "vc_100_500": "MPI", "vc_100_2000": "MPI",
    "vc_200_750": "MPI", "vc_200_3000": "MPI",
    "vc_800_10000": "LPI",
}

CLASS_COLORS = {"SPI": "#43A047", "MPI": "#1E88E5", "LPI": "#FB8C00"}
CLASS_ORDER = ["SPI", "MPI", "LPI"]


def run_ablation_analysis(
    instance_path: Path | str = None,
    output_path: Path | str = None,
    n_runs: int = 10,
    budget: int = 20000,
    force: bool = False,
) -> Path:
    """
    Esegue lo studio di ablazione sulle varianti MMAS e salva il barplot.
    Se il file di output esiste e force=False, restituisce il percorso senza ricalcolare.
    """
    if output_path is None:
        output_path = DEFAULT_ABLATION_PLOT
    else:
        output_path = Path(output_path)

    if output_path.exists() and not force:
        msg = f"[INFO] Grafico ablazione già esistente: {output_path}"
        logger.info(msg)
        return output_path

    if instance_path is None:
        instance_path = PROJECT_ROOT / "wvcp-instances" / "vc_800_10000.txt"
    else:
        instance_path = Path(instance_path)

    if not instance_path.exists():
        err_msg = f"Istanza non trovata: {instance_path}"
        logger.error(err_msg)
        raise FileNotFoundError(err_msg)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    msg_start = f"[INFO] Avvio Ablation Study su {instance_path.name} ({n_runs} seed per variante)..."
    logger.info(msg_start)

    graph = Graph.load_from_file(instance_path)
    n_ants = max(5, graph.n // 10)
    logger.info(f"  -> Grafo caricato: |V|={graph.n}, |E|={len(graph.edges)}, n_ants={n_ants}")

    variants = [
        {"name": "MMAS Completo", "params": {"alpha": 1.0, "beta": 2.0, "rho": 0.10, "max_reinits": None, "pruning_strategy": "greedy"}},
        {"name": "Senza Recovery", "params": {"alpha": 1.0, "beta": 2.0, "rho": 0.10, "max_reinits": 0, "pruning_strategy": "greedy"}},
        {"name": "Senza Pruning", "params": {"alpha": 1.0, "beta": 2.0, "rho": 0.10, "max_reinits": None, "pruning_strategy": None}},
        {"name": "No Feromone (α=0)", "params": {"alpha": 0.0, "beta": 2.0, "rho": 0.10, "max_reinits": None, "pruning_strategy": "greedy"}},
        {"name": "No Euristica (β=0)", "params": {"alpha": 1.0, "beta": 0.0, "rho": 0.10, "max_reinits": None, "pruning_strategy": "greedy"}},
    ]

    ablation_results = []
    for var_idx, var in enumerate(variants, start=1):
        logger.info(f"[INFO] [{var_idx}/{len(variants)}] Valutazione variante: '{var['name']}'...")
        weights = []
        for seed_idx, seed in enumerate(range(42, 42 + n_runs), start=1):
            params: dict = dict(var["params"])
            s = MMAS_Solver(graph, n_ants=n_ants, max_fe=budget, seed=seed, **params)
            res = s.solve(verbose=False)
            weights.append(res["best_weight"])
            logger.info(f"     [Run {seed_idx}/{n_runs}] seed={seed} -> best_weight={res['best_weight']:.0f}")
        
        mean_w = np.mean(weights)
        std_w = np.std(weights)
        logger.info(f"  [OK] Variante '{var['name']}' completata: {mean_w:.1f} ± {std_w:.1f}")

        ablation_results.append({
            "Variante": var["name"],
            "Peso_Medio": mean_w,
            "Peso_Std": std_w,
        })

    import pandas as pd
    df_ablation = pd.DataFrame(ablation_results)

    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(10, 6))
    bars = sns.barplot(data=df_ablation, x="Variante", y="Peso_Medio", color="steelblue", alpha=0.85)
    plt.errorbar(
        x=range(len(df_ablation)),
        y=df_ablation["Peso_Medio"],
        yerr=df_ablation["Peso_Std"],
        fmt="none",
        c="black",
        capsize=5,
    )
    plt.xticks(rotation=20, ha="right", fontsize=11)
    plt.title(f"Utilità dei Componenti (Ablation Study) su {instance_path.name}", fontsize=14, fontweight="bold")
    plt.ylabel("Peso Medio Finale (10 run)", fontsize=12)

    # Aggiungi etichette valore sulle barre
    for patch in bars.patches:
        height = patch.get_height()
        if not np.isnan(height) and height > 0:
            plt.text(
                patch.get_x() + patch.get_width() / 2.0,
                height + (max(df_ablation["Peso_Std"]) * 0.1 if max(df_ablation["Peso_Std"]) > 0 else 5),
                f"{height:.1f}",
                ha="center",
                va="bottom",
                fontsize=9,
                fontweight="bold",
            )

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    alt_path = output_path.parent / "ablation_study.png"
    if alt_path != output_path:
        plt.savefig(alt_path, dpi=300)
    plt.close()
    logger.info(f"[SUCCESSO] Grafico Ablation Study salvato in: {output_path} e {alt_path}")
    return output_path


def load_data() -> tuple[list[dict], dict[str, list[dict]]]:
    if not CSV_DIR.exists():
        logger.error(f"[ERRORE] Cartella CSV non trovata: {CSV_DIR}")
        return [], {}

    all_rows = []
    by_config = {}
    for f in sorted(CSV_DIR.glob("*.csv")):
        prefix = f.stem
        if prefix not in PREFIX_TO_CLASS:
            continue
        with open(f, newline="", encoding="utf-8") as fh:
            rows = []
            for r in csv.DictReader(fh):
                r["config"] = prefix
                r["class"] = PREFIX_TO_CLASS[prefix]
                r["best_weight"] = float(r["best_weight"])
                r["fe_at_best"] = int(r["fe_at_best"])
                r["fe_budget_pct"] = float(r["fe_budget_pct"])
                rows.append(r)
            by_config[prefix] = rows
            all_rows.extend(rows)
    return all_rows, by_config


def plot_boxplot_by_config(by_config: dict, save_dir: Path):
    configs_per_class = defaultdict(list)
    for prefix, rows in sorted(by_config.items()):
        cls = PREFIX_TO_CLASS[prefix]
        configs_per_class[cls].append((prefix, [r["best_weight"] for r in rows]))

    n_classes = len([c for c in CLASS_ORDER if c in configs_per_class])
    if n_classes == 0:
        return
    fig, axes = plt.subplots(1, n_classes, figsize=(5 * n_classes, 5), sharey=False)
    if n_classes == 1:
        axes = [axes]

    plt.style.use("seaborn-v0_8-whitegrid")

    for ax, cls in zip(axes, [c for c in CLASS_ORDER if c in configs_per_class]):
        entries = configs_per_class[cls]
        labels = [e[0] for e in entries]
        data = [e[1] for e in entries]

        try:
            bp = ax.boxplot(data, tick_labels=labels, patch_artist=True,
                            medianprops={"color": "white", "linewidth": 2},
                            flierprops={"marker": "o", "markersize": 4,
                                        "markerfacecolor": CLASS_COLORS[cls], "alpha": 0.6})
        except TypeError:
            bp = ax.boxplot(data, labels=labels, patch_artist=True,
                            medianprops={"color": "white", "linewidth": 2},
                            flierprops={"marker": "o", "markersize": 4,
                                        "markerfacecolor": CLASS_COLORS[cls], "alpha": 0.6})

        for patch in bp["boxes"]:
            patch.set_facecolor(CLASS_COLORS[cls])
            patch.set_alpha(0.75)
        ax.set_title(f"Classe {cls}", fontsize=12, fontweight="bold")
        ax.set_xlabel("Configurazione", fontsize=10)
        ax.set_ylabel("Best Weight" if ax == axes[0] else "", fontsize=10)
        ax.tick_params(axis="x", rotation=20, labelsize=8)

    fig.suptitle("Distribuzione Best Weight per Configurazione e Classe",
                 fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    path = save_dir / "boxplot_best_weight.png"
    plt.savefig(path, dpi=200, bbox_inches="tight")
    logger.info(f"  [OK] {path}")
    plt.close()


def plot_fe_vs_weight(all_rows: list[dict], save_dir: Path):
    if not all_rows:
        return
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(9, 5))

    for cls in CLASS_ORDER:
        pts = [r for r in all_rows if r["class"] == cls]
        if pts:
            ax.scatter(
                [r["fe_at_best"] for r in pts],
                [r["best_weight"] for r in pts],
                label=cls, color=CLASS_COLORS[cls],
                alpha=0.75, edgecolors="white", linewidth=0.4, s=55,
            )

    ax.set_xlabel("FE al momento della miglior soluzione trovata", fontsize=11)
    ax.set_ylabel("Best Weight", fontsize=11)
    ax.set_title("Efficienza di Convergenza — FE al Best vs Best Weight",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=10, frameon=True)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    plt.tight_layout()
    path = save_dir / "fe_vs_weight.png"
    plt.savefig(path, dpi=200, bbox_inches="tight")
    logger.info(f"  [OK] {path}")
    plt.close()


def plot_budget_distribution(all_rows: list[dict], save_dir: Path):
    if not all_rows:
        return
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(10, 4))

    bins = np.linspace(0, 100, 21)
    for cls in CLASS_ORDER:
        pcts = [r["fe_budget_pct"] for r in all_rows if r["class"] == cls]
        if pcts:
            ax.hist(pcts, bins=bins, alpha=0.55, label=cls,
                    color=CLASS_COLORS[cls], edgecolor="white", linewidth=0.5)

    ax.set_xlabel("% Budget FE consumato prima di trovare il best", fontsize=11)
    ax.set_ylabel("Frequenza (run)", fontsize=11)
    ax.set_title("Posizione della Soluzione Ottimale nel Budget di FE",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=10)
    ax.set_xlim(0, 100)
    plt.tight_layout()
    path = save_dir / "budget_distribution.png"
    plt.savefig(path, dpi=200, bbox_inches="tight")
    logger.info(f"  [OK] {path}")
    plt.close()


def plot_ablation_study(save_dir: Path = PLOTS_DIR):
    save_dir.mkdir(parents=True, exist_ok=True)
    all_rows, by_config = load_data()

    if all_rows:
        logger.info(f"[INFO] Lettura dati da: {CSV_DIR}")
        logger.info(f"[INFO] {len(all_rows)} run caricate da {len(by_config)} configurazioni.")
        logger.info(f"[INFO] Salvataggio grafici in: {save_dir}")
        plot_boxplot_by_config(by_config, save_dir)
        plot_fe_vs_weight(all_rows, save_dir)
        plot_budget_distribution(all_rows, save_dir)

    run_ablation_analysis(output_path=save_dir / "ablation_study_mpi.png", force=True)
    logger.info("[SUCCESSO] Grafici di ablazione completati.")


def main():
    plot_ablation_study()


if __name__ == "__main__":
    main()
