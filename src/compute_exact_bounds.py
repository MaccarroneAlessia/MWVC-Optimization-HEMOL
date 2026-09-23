"""
compute_exact_bounds.py — Riferimenti esatti per la valutazione di MMAS.

Per ogni istanza in wvcp-instances/ calcola:
  - lp_bound      : valore ottimo del rilassamento lineare (x_v in [0,1]), scipy.optimize.linprog (HiGHS)
  - lp_frac_half  : frazione di vertici con x_v = 1/2 nella soluzione LP (half-integrality)
  - ilp_weight    : miglior soluzione intera trovata da scipy.optimize.milp (HiGHS)
  - ilp_status    : "optimal" se l'ottimo è dimostrato, "time_limit" se il limite di tempo è scaduto
  - ilp_dual_bound: lower bound dimostrato dal branch-and-bound (= ilp_weight se optimal)
  - ilp_mip_gap   : gap relativo residuo del solver
  - ilp_time_s    : tempo di calcolo

Output: results/csv/exact_bounds.csv

Uso:
    python src/compute_exact_bounds.py                 # SPI + MPI, limite 120 s per istanza
    python src/compute_exact_bounds.py --time-limit 300
    python src/compute_exact_bounds.py --include-lpi   # prova anche LPI (non chiude: fornisce solo il dual bound)
"""

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, linprog, milp
from scipy.sparse import csc_matrix

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.core.graph import Graph  # noqa: E402

HEADER = ["instance", "lp_bound", "lp_frac_half", "ilp_weight", "ilp_status",
          "ilp_dual_bound", "ilp_mip_gap", "ilp_time_s"]


def constraint_matrix(graph: Graph) -> csc_matrix:
    rows, cols = [], []
    for r, (u, v) in enumerate(graph.edges):
        rows += [r, r]
        cols += [u, v]
    return csc_matrix((np.ones(len(rows)), (rows, cols)), shape=(len(graph.edges), graph.n))


def solve_instance(path: Path, time_limit: float, do_ilp: bool) -> dict:
    g = Graph.load_from_file(str(path))
    A = constraint_matrix(g)
    w = g.weights.astype(float)

    lp = linprog(w, A_ub=-A, b_ub=-np.ones(A.shape[0]), bounds=(0, 1), method="highs")
    row = {
        "instance": path.name,
        "lp_bound": round(float(lp.fun), 2),
        "lp_frac_half": round(float(np.mean(np.isclose(lp.x, 0.5))), 4),
        "ilp_weight": "", "ilp_status": "not_run", "ilp_dual_bound": "", "ilp_mip_gap": "", "ilp_time_s": "",
    }
    if not do_ilp:
        return row

    t0 = time.time()
    res = milp(c=w, constraints=LinearConstraint(A, lb=1.0, ub=np.inf),
               integrality=np.ones(g.n), bounds=Bounds(0, 1),
               options={"time_limit": time_limit, "disp": False})
    elapsed = time.time() - t0
    if res.x is not None:
        x = np.round(res.x).astype(bool)
        row["ilp_weight"] = int(round(float(w[x].sum())))
    row["ilp_status"] = "optimal" if res.status == 0 else ("time_limit" if res.status == 1 else f"status_{res.status}")
    row["ilp_dual_bound"] = round(float(getattr(res, "mip_dual_bound", np.nan)), 2)
    row["ilp_mip_gap"] = round(float(getattr(res, "mip_gap", np.nan)), 6)
    row["ilp_time_s"] = round(elapsed, 1)
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--time-limit", type=float, default=120.0)
    ap.add_argument("--include-lpi", action="store_true")
    args = ap.parse_args()

    inst_dir = PROJECT_ROOT / "wvcp-instances"
    out = PROJECT_ROOT / "results" / "csv" / "exact_bounds.csv"
    out.parent.mkdir(parents=True, exist_ok=True)

    with open(out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=HEADER)
        writer.writeheader()
        for p in sorted(inst_dir.glob("*.txt")):
            is_lpi = p.stem.startswith("vc_800_")
            row = solve_instance(p, args.time_limit, do_ilp=(not is_lpi) or args.include_lpi)
            writer.writerow(row)
            f.flush()
            print(f"{row['instance']:22s} LP={row['lp_bound']:>9} ILP={row['ilp_weight']!s:>7} "
                  f"({row['ilp_status']}, {row['ilp_time_s']} s)")
    print(f"\nSalvato: {out}")


if __name__ == "__main__":
    main()
