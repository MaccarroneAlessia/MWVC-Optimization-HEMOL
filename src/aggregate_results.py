"""
aggregate_results.py — Aggregazione dei risultati per configurazione e per classe.

Legge tutti i CSV prodotti da src/main.py in results/csv/ e produce:
  - results/csv/summary_by_config.csv   → una riga per ogni configurazione (n, m)
  - results/csv/summary_by_class.csv    → una riga per classe (SPI, MPI, LPI)
  - stampa a console entrambe le tabelle

    python src/aggregate_results.py

Metriche prodotte:
  - mean_best_weight / std / min / max  → qualità delle soluzioni
  - mean_fe_at_best / std               → efficienza di convergenza
  - mean_fe_budget_pct                  → posizione relativa del best nel budget
  - mean_cover_size                     → dimensione media del cover
  - mean_reinits                        → frequenza di stagnazione
  - mean_time_s / std_time_s            → costo computazionale
  - pct_stopped_early                   → % di run fermate per max_reinits
"""

import csv
import sys
from pathlib import Path
from statistics import mean, stdev

_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from src.utils.logger import setup_logger, print_and_log_table

logger = setup_logger("AGGREGATE", "aggregate_results.log")

# ---------------------------------------------------------------------------
# Mapping prefisso → classe (deve essere allineato con src/main.py)
# ---------------------------------------------------------------------------
PREFIX_TO_CLASS = {
    "vc_20_60": "SPI",
    "vc_20_120": "SPI",
    "vc_25_150": "SPI",
    "vc_100_500": "MPI",
    "vc_100_2000": "MPI",
    "vc_200_750": "MPI",
    "vc_200_3000": "MPI",
    "vc_800_10000": "LPI",
}

PREFIX_META = {
    "vc_20_60":     (20, 60),
    "vc_20_120":    (20, 120),
    "vc_25_150":    (25, 150),
    "vc_100_500":   (100, 500),
    "vc_100_2000":  (100, 2000),
    "vc_200_750":   (200, 750),
    "vc_200_3000":  (200, 3000),
    "vc_800_10000": (800, 10000),
}

CLASS_ORDER = ["SPI", "MPI", "LPI"]

# Colonne condivise tra entrambe le tabelle
SHARED_FIELDS = [
    "n_runs",
    "mean_best_weight",
    "std_best_weight",
    "min_best_weight",
    "max_best_weight",
    "mean_fe_at_best",
    "std_fe_at_best",
    "mean_fe_budget_pct",
    "mean_cover_size",
    "mean_reinits",
    "mean_time_s",
    "std_time_s",
    "pct_stopped_early",
]

CONFIG_HEADER = ["class", "config", "n", "m"] + SHARED_FIELDS
CLASS_HEADER = ["class"] + SHARED_FIELDS


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_csv(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def safe_stdev(values: list) -> float:
    return round(stdev(values), 4) if len(values) > 1 else 0.0


def aggregate_rows(rows: list[dict]) -> dict:
    """Calcola tutte le metriche aggregate da una lista di righe CSV."""
    valid_rows = [r for r in rows if r.get("best_weight") is not None and r["best_weight"] != ""]
    best_weights  = [float(r["best_weight"])  for r in valid_rows]
    fe_at_bests   = [int(r["fe_at_best"])     for r in valid_rows]
    fe_budget_pct = [float(r["fe_budget_pct"]) for r in valid_rows]
    cover_sizes   = [int(r["cover_size"])     for r in valid_rows]
    reinits       = [int(r["n_reinits"])      for r in valid_rows]
    times         = [float(r["time_seconds"]) for r in valid_rows]
    stopped       = [int(r["stopped_early"])  for r in valid_rows]

    return {
        "n_runs":             len(rows),
        "mean_best_weight":   round(mean(best_weights), 2),
        "std_best_weight":    safe_stdev(best_weights),
        "min_best_weight":    round(min(best_weights), 0),
        "max_best_weight":    round(max(best_weights), 0),
        "mean_fe_at_best":    round(mean(fe_at_bests), 1),
        "std_fe_at_best":     safe_stdev([float(x) for x in fe_at_bests]),
        "mean_fe_budget_pct": round(mean(fe_budget_pct), 2),
        "mean_cover_size":    round(mean(cover_sizes), 2),
        "mean_reinits":       round(mean(reinits), 2),
        "mean_time_s":        round(mean(times), 3),
        "std_time_s":         safe_stdev(times),
        "pct_stopped_early":  round(sum(stopped) / len(stopped) * 100, 1),
    }


# ---------------------------------------------------------------------------
# Stampa a console
# ---------------------------------------------------------------------------

COL_W = [8, 14, 6, 7, 7, 16, 14, 16, 14, 14, 14, 16, 14, 12, 11, 11, 17]
HEADERS_LABELS = [
    "Classe", "Config", "N", "M", "Runs",
    "Mean W", "Std W", "Min W", "Max W",
    "MeanFEbest", "StdFEbest", "MeanFE%", "MeanCover",
    "MeanReinit", "MeanTime", "StdTime", "Stopped%",
]
SEP = "-" * (sum(COL_W) + 3 * (len(COL_W) - 1))


def fmt_row(vals: list) -> str:
    return " | ".join(str(v).ljust(w) for v, w in zip(vals, COL_W))


def print_config_row(r: dict) -> None:
    vals = [
        r["class"], r["config"], r["n"], r["m"], r["n_runs"],
        r["mean_best_weight"], r["std_best_weight"],
        r["min_best_weight"], r["max_best_weight"],
        r["mean_fe_at_best"], r["std_fe_at_best"],
        r["mean_fe_budget_pct"], r["mean_cover_size"],
        r["mean_reinits"], r["mean_time_s"], r["std_time_s"],
        r["pct_stopped_early"],
    ]
    print(fmt_row(vals))


def print_class_row(cls: str, r: dict) -> None:
    # La tabella per classe non ha "config/n/m", li sostituiamo con "-"
    vals = [
        cls, "-", "-", "-", r["n_runs"],
        r["mean_best_weight"], r["std_best_weight"],
        r["min_best_weight"], r["max_best_weight"],
        r["mean_fe_at_best"], r["std_fe_at_best"],
        r["mean_fe_budget_pct"], r["mean_cover_size"],
        r["mean_reinits"], r["mean_time_s"], r["std_time_s"],
        r["pct_stopped_early"],
    ]
    print(fmt_row(vals))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    project_root = Path(__file__).resolve().parent.parent
    csv_dir = project_root / "results" / "csv"
    config_summary_path = project_root / "results" / "summary_by_config.csv"
    class_summary_path  = project_root / "results" / "summary_by_class.csv"

    if not csv_dir.exists():
        print(f"[ERRORE] Cartella CSV non trovata: {csv_dir}")
        print("  Esegui prima 'python src/main.py'.")
        sys.exit(1)

    # Caricamento raw data
    config_data: dict[str, list[dict]] = {}
    for csv_file in sorted(csv_dir.glob("*.csv")):
        prefix = csv_file.stem
        if prefix not in PREFIX_TO_CLASS:
            print(f"[WARN] Prefisso sconosciuto, skip: {prefix}")
            continue
        rows = load_csv(csv_file)
        if not rows:
            print(f"[WARN] CSV vuoto, skip: {csv_file.name}")
            continue
        config_data[prefix] = rows

    if not config_data:
        print("[ERRORE] Nessun CSV valido trovato.")
        sys.exit(1)

    # -----------------------------------------------------------------------
    # TABELLA 1: una riga per configurazione (n, m)
    # -----------------------------------------------------------------------
    config_rows = []
    class_raw: dict[str, list[dict]] = {c: [] for c in CLASS_ORDER}

    for prefix, rows in sorted(config_data.items()):
        cls  = PREFIX_TO_CLASS[prefix]
        n, m = PREFIX_META[prefix]
        agg  = aggregate_rows(rows)
        config_rows.append({"class": cls, "config": prefix, "n": n, "m": m, **agg})
        class_raw[cls].extend(rows)  # accumula per la tabella per classe

    # Salva CSV per configurazione
    with open(config_summary_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CONFIG_HEADER)
        writer.writeheader()
        writer.writerows(config_rows)
    print(f"[OK] Summary per configurazione: {config_summary_path}")

    # -----------------------------------------------------------------------
    # TABELLA 2: una riga per classe (SPI, MPI, LPI)
    # -----------------------------------------------------------------------
    class_rows = []
    for cls in CLASS_ORDER:
        if not class_raw[cls]:
            continue
        agg = aggregate_rows(class_raw[cls])
        class_rows.append({"class": cls, **agg})

    # Salva CSV per classe
    with open(class_summary_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CLASS_HEADER)
        writer.writeheader()
        writer.writerows(class_rows)
    print(f"[OK] Summary per classe:         {class_summary_path}\n")

    # -----------------------------------------------------------------------
    # Stampa a console: Tabella 1 — per configurazione
    # -----------------------------------------------------------------------
    # Tabella 1: Per configurazione
    table1_rows = []
    for cls in CLASS_ORDER:
        cls_entries = [r for r in config_rows if r["class"] == cls]
        for r in cls_entries:
            table1_rows.append([
                r["class"], r["config"], r["n"], r["m"], r["n_runs"],
                r["mean_best_weight"], r["std_best_weight"],
                r["min_best_weight"], r["max_best_weight"],
                r["mean_fe_at_best"], r["std_fe_at_best"],
                r["mean_fe_budget_pct"], r["mean_cover_size"],
                r["mean_reinits"], r["mean_time_s"], r["std_time_s"],
                r["pct_stopped_early"],
            ])
    print_and_log_table(logger, "TABELLA 1 — Risultati per Configurazione (n, m)", HEADERS_LABELS, table1_rows)

    # Tabella 2: Per classe
    table2_rows = []
    for r in class_rows:
        table2_rows.append([
            r["class"], "-", "-", "-", r["n_runs"],
            r["mean_best_weight"], r["std_best_weight"],
            r["min_best_weight"], r["max_best_weight"],
            r["mean_fe_at_best"], r["std_fe_at_best"],
            r["mean_fe_budget_pct"], r["mean_cover_size"],
            r["mean_reinits"], r["mean_time_s"], r["std_time_s"],
            r["pct_stopped_early"],
        ])
    print_and_log_table(logger, "TABELLA 2 — Risultati Aggregati per Classe (SPI / MPI / LPI)", HEADERS_LABELS, table2_rows)


if __name__ == "__main__":
    main()
