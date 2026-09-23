"""
exact_solver.py — Calcolo della soluzione ottima e del lower bound per MWVCP.

Per valutare la qualità dell'algoritmo MMAS -> confronto peso
della soluzione trovata con un riferimento teorico

  1. compute_lower_bound(graph): lower bound veloce via Maximum Weighted Matching.
     Funziona su tutte le classi (SPI, MPI, LPI) in O(m log m).
     -> Per ogni arco del matching massimale, almeno un endpoint deve stare nel vertex cover, 
        quindi LB ≥ Σ min(w_u, w_v) sugli archi selezionati.

  2. solve_lp_relaxation(graph): lower bound più preciso via LP Relaxation.
     Stessa formulazione dell'ILP ma senza il vincolo di integrità (x_v ∈ [0,1] invece di x_v ∈ {0,1}). 
     Si risolve in tempo polinomiale con scipy.linprog
     e restituisce un valore che soddisfa sempre OPT/2 ≤ LP ≤ OPT (soluzione ottima).
     ichiede obbligatoriamente la libreria esterna scipy (scipy.optimize.linprog e le matrici sparse scipy.sparse)

  3. solve_exact_ilp(graph): soluzione ottima tramite ILP (Integer Linear Program).
     Usabile SOLO per istanze piccole (n ≤ 50, quindi SPI). Richiede scipy >= 1.7.
     Formulazione:
       min  Σ w_v · x_v       (x_v ∈ {0,1})
       s.t. x_u + x_v ≥ 1    per ogni arco (u,v)

  4. quality_gap(algo_weight, reference): gap percentuale rispetto al reference.

Riepilogo per classe:
  SPI  → compute_lower_bound + solve_lp_relaxation + solve_exact_ilp  (ottimo esatto)
  MPI  → compute_lower_bound + solve_lp_relaxation                    (bound teorici)
  LPI  → compute_lower_bound + solve_lp_relaxation                    (bound teorici)
"""

import numpy as np
from .graph import Graph


def compute_lower_bound(graph: Graph) -> float:
    """
    Lower bound per MWVCP tramite Maximal Weighted Matching greedy.

    Costruisce un matching greedy (archi ordinati per min-peso decrescente)
    e somma il min-peso degli endpoint per ogni arco selezionato.
    Ogni vertex cover valido deve coprire almeno un endpoint per ogni arco
    del matching → la somma è un lower bound garantito per OPT.

    Complessità: O(m log m).
    """
    if graph.m == 0:
        return 0.0

    # Ordina gli archi per min(w_u, w_v) decrescente: matching più "pesante" prima.
    edges_sorted = sorted(
        graph.edges,
        key=lambda e: min(graph.weights[e[0]], graph.weights[e[1]]),
        reverse=True,
    )

    matched = np.zeros(graph.n, dtype=bool)
    lb = 0.0
    for u, v in edges_sorted:
        if not matched[u] and not matched[v]:
            lb += min(graph.weights[u], graph.weights[v])
            matched[u] = True
            matched[v] = True

    return lb


def solve_exact_ilp(graph: Graph, time_limit_s: float = 30.0):
    """
    Soluzione ottima di MWVCP via ILP con scipy.optimize.milp.

    Usabile praticamente solo per n ≤ 50 (SPI). Per istanze più grandi
    il branch-and-bound del solver esplode esponenzialmente.

    Formulazione:
      min  w^T x
      s.t. x_u + x_v >= 1  per ogni arco (u,v)
           x_v ∈ {0,1}

    Restituisce:
      (optimal_weight, in_cover_mask)  se trovata soluzione ottima
      None                             se scipy non disponibile o timeout

    Raises:
      RuntimeError se il grafo non ha soluzioni feasible (impossibile).
    """
    try:
        from scipy.optimize import milp, LinearConstraint, Bounds
    except ImportError:
        return None  # scipy non installato

    n = graph.n
    m = len(graph.edges)
    if m == 0:
        return 0.0, np.zeros(n, dtype=bool)

    # Coefficienti della funzione obiettivo
    c = graph.weights.astype(float)

    # Matrice dei vincoli: una riga per arco, x_u + x_v >= 1
    row_idx, col_idx = [], []
    for row, (u, v) in enumerate(graph.edges):
        row_idx += [row, row]
        col_idx += [u, v]
    data = np.ones(len(row_idx))

    from scipy.sparse import csc_matrix
    A = csc_matrix((data, (row_idx, col_idx)), shape=(m, n))

    constraints = LinearConstraint(A, lb=1.0, ub=np.inf)
    bounds = Bounds(lb=0.0, ub=1.0)
    integrality = np.ones(n)  # tutte le variabili intere

    result = milp(
        c=c,
        constraints=constraints,
        integrality=integrality,
        bounds=bounds,
        options={"time_limit": time_limit_s, "disp": False},
    )

    if result.status not in (0, 1):
        return None  # infeasible o timeout senza soluzione

    x = np.round(result.x).astype(bool)
    optimal_weight = float(graph.weights[x].sum())
    return optimal_weight, x


def quality_gap(algo_weight: float, reference: float) -> float:
    """
    Gap percentuale tra la soluzione algoritmica e un reference (LB o OPT).

    gap > 0  → l'algoritmo è sopra il reference (peggiore, atteso)
    gap = 0  → l'algoritmo ha raggiunto esattamente il reference
    gap < 0  → impossibile (solo se reference è un LB e non un OPT)

    Formula: gap = (algo - ref) / ref * 100
    """
    if reference <= 0:
        return float("nan")
    return round((algo_weight - reference) / reference * 100, 2)
