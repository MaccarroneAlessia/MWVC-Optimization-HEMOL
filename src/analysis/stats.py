"""
stats.py — Statistiche aggregate dei risultati del benchmark MMAS-MWVCP.

Legge i CSV prodotti da src/main.py (in results/csv/) e calcola per ogni
configurazione e per ogni classe (SPI, MPI, LPI):

  Metriche di qualità:
    - mean_best_weight, std_best_weight, min_best_weight, max_best_weight

  Metriche di efficienza:
    - mean_fe_at_best, std_fe_at_best    → FE necessarie per trovare il best
    - mean_fe_budget_pct                 → % del budget consumato prima del best

  Metriche di convergenza:
    - mean_reinits                       → frequenza di stagnazione
    - pct_stopped_early                  → run fermate da max_reinits

  Test statistici (richiede scipy):
    - Test di Wilcoxon signed-rank tra best_weight delle classi
      (SPI vs MPI, MPI vs LPI, SPI vs LPI)

Output:
  - results/stats_report.csv
  - Tabella riepilogativa a console

Uso:
    python src/analysis/stats.py
"""

import csv
import sys
from pathlib import Path
from statistics import mean, stdev

_HERE = Path(__file__).resolve().parent
PROJECT_ROOT = _HERE.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.logger import setup_logger, print_and_log_table

logger = setup_logger("STATS", "stats.log")

CSV_DIR = PROJECT_ROOT / "results" / "csv"
OUTPUT_PATH = PROJECT_ROOT / "results" / "stats_report.csv"

# Mapping prefisso → classe
PREFIX_TO_CLASS = {
    "vc_20_60": "SPI", "vc_20_120": "SPI", "vc_25_150": "SPI",
    "vc_100_500": "MPI", "vc_100_2000": "MPI",
    "vc_200_750": "MPI", "vc_200_3000": "MPI",
    "vc_800_10000": "LPI",
}

PREFIX_META = {
    "vc_20_60":    (20,  60),  "vc_20_120":   (20, 120),
    "vc_25_150":   (25, 150),  "vc_100_500":  (100, 500),
    "vc_100_2000": (100, 2000),"vc_200_750":  (200, 750),
    "vc_200_3000": (200, 3000),"vc_800_10000":(800, 10000),
}

REPORT_HEADER = [
    "level", "class", "config", "n", "m", "n_runs",
    "mean_best_weight", "std_best_weight", "min_best_weight", "max_best_weight",
    "mean_fe_at_best", "std_fe_at_best", "mean_fe_budget_pct",
    "mean_reinits", "pct_stopped_early",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_std(values: list) -> float:
    return round(stdev(values), 4) if len(values) > 1 else 0.0


def load_csv(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def compute_stats(rows: list[dict], level: str, cls: str, config: str, n, m) -> dict:
    weights   = [float(r["best_weight"])   for r in rows]
    fe_bests  = [float(r["fe_at_best"])    for r in rows]
    fe_pcts   = [float(r["fe_budget_pct"]) for r in rows]
    reinits   = [float(r["n_reinits"])     for r in rows]
    stopped   = [int(r["stopped_early"])   for r in rows]
    return {
        "level":            level,
        "class":            cls,
        "config":           config,
        "n":                n,
        "m":                m,
        "n_runs":           len(rows),
        "mean_best_weight": round(mean(weights), 2),
        "std_best_weight":  _safe_std(weights),
        "min_best_weight":  round(min(weights), 0),
        "max_best_weight":  round(max(weights), 0),
        "mean_fe_at_best":  round(mean(fe_bests), 1),
        "std_fe_at_best":   _safe_std(fe_bests),
        "mean_fe_budget_pct": round(mean(fe_pcts), 2),
        "mean_reinits":     round(mean(reinits), 2),
        "pct_stopped_early":round(sum(stopped) / len(stopped) * 100, 1),
    }


def wilcoxon_test(a: list[float], b: list[float], label_a: str, label_b: str) -> str:
    """Esegue il test di Wilcoxon signed-rank se scipy è disponibile."""
    try:
        from scipy.stats import wilcoxon
        if len(a) < 2 or len(b) < 2:
            return f"  {label_a} vs {label_b}: campioni insufficienti"
        n = min(len(a), len(b))
        stat, p = wilcoxon(a[:n], b[:n], alternative="two-sided")
        sig = "✓ SIGNIFICATIVO (p<0.05)" if p < 0.05 else "✗ non significativo"
        return f"  {label_a} vs {label_b}: W={stat:.2f}, p={p:.4f}  →  {sig}"
    except ImportError:
        return "  [SKIP] scipy non installato — pip install scipy"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def compute_summary_stats(csv_dir: Path = CSV_DIR, output_path: Path = OUTPUT_PATH):
    if not csv_dir.exists():
        print(f"[ERRORE] Cartella CSV non trovata: {csv_dir}")
        print("  Esegui prima: python src/main.py")
        sys.exit(1)

    # --- Caricamento dati ---
    config_rows: dict[str, list[dict]] = {}
    for f in sorted(csv_dir.glob("*.csv")):
        prefix = f.stem
        if prefix not in PREFIX_TO_CLASS:
            continue
        rows = load_csv(f)
        if rows:
            config_rows[prefix] = rows

    if not config_rows:
        print("[ERRORE] Nessun CSV valido in", csv_dir)
        sys.exit(1)

    # --- Statistiche per configurazione ---
    config_stats = []
    class_rows_all: dict[str, list[dict]] = {"SPI": [], "MPI": [], "LPI": []}

    for prefix, rows in sorted(config_rows.items()):
        cls = PREFIX_TO_CLASS[prefix]
        n, m = PREFIX_META[prefix]
        config_stats.append(compute_stats(rows, "config", cls, prefix, n, m))
        class_rows_all[cls].extend(rows)

    # --- Statistiche per classe ---
    class_stats = []
    for cls in ["SPI", "MPI", "LPI"]:
        rows = class_rows_all[cls]
        if rows:
            class_stats.append(compute_stats(rows, "class", cls, cls, "-", "-"))

    all_stats = config_stats + class_stats

    # --- Salvataggio CSV ---
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=REPORT_HEADER)
        writer.writeheader()
        writer.writerows(all_stats)
    print(f"[OK] Report salvato in: {output_path}\n")

    # --- Stampa e Logging Tabella ---
    CLASS_ORDER = ["SPI", "MPI", "LPI"]
    headers = [
        "Lvl", "Config", "N", "M", "Runs",
        "Mean W", "Std W", "Min W", "Max W",
        "MeanFEbest", "StdFEbest", "MeanFE%",
        "MeanReinit", "Stopped%"
    ]
    rows = []
    for cls in CLASS_ORDER:
        cfg_rows = [s for s in config_stats if s["class"] == cls]
        for s in cfg_rows:
            rows.append([
                s["level"], s["config"], s["n"], s["m"], s["n_runs"],
                s["mean_best_weight"], s["std_best_weight"],
                s["min_best_weight"], s["max_best_weight"],
                s["mean_fe_at_best"], s["std_fe_at_best"],
                s["mean_fe_budget_pct"],
                s["mean_reinits"], s["pct_stopped_early"]
            ])
        cls_s = next((s for s in class_stats if s["class"] == cls), None)
        if cls_s:
            rows.append([
                "CLASS", f"[{cls}] TOT", "-", "-", cls_s["n_runs"],
                cls_s["mean_best_weight"], cls_s["std_best_weight"],
                cls_s["min_best_weight"], cls_s["max_best_weight"],
                cls_s["mean_fe_at_best"], cls_s["std_fe_at_best"],
                cls_s["mean_fe_budget_pct"],
                cls_s["mean_reinits"], cls_s["pct_stopped_early"]
            ])

    print_and_log_table(logger, "STATISTICHE AGGREGATE E STATS REPORT BENCHMARK", headers, rows)

    # --- Test di Wilcoxon tra classi ---
    logger.info("\n[Test di Wilcoxon tra classi — best_weight]")
    def cls_weights(cls):
        return [float(r["best_weight"]) for r in class_rows_all[cls]]
    
    t1 = wilcoxon_test(cls_weights("SPI"), cls_weights("MPI"), "SPI", "MPI")
    t2 = wilcoxon_test(cls_weights("MPI"), cls_weights("LPI"), "MPI", "LPI")
    t3 = wilcoxon_test(cls_weights("SPI"), cls_weights("LPI"), "SPI", "LPI")
    
    logger.info(t1)
    logger.info(t2)
    logger.info(t3)


if __name__ == "__main__":
    compute_summary_stats()
