"""
ant.py — La formica artificiale per il MWVCP (versione ottimizzata -> operazioni numpy vettorizzate).

Ogni formica costruisce una copertura dei vertici valida selezionando
probabilisticamente i nodi in base a feromone (tau_v) ed euristica (eta_v),
dove eta_v = d_S(v)^gamma / w(v) (euristica di Chvatal).
"""

import numpy as np
from .graph import Graph
from .solution import Solution
from .pheromone import PheromoneManager
from src.utils.logger import setup_logger

logger = setup_logger()
from .pruning import pruning_greedy


class Ant:
    """
    Rappresenta un agente. Riceve il grafo e un costruttore di soluzioni.
    """

    def __init__(self, graph: Graph):
        self.graph = graph
        self.solution: Solution = Solution(graph)

    def construct_solution(
        self,
        pheromone: PheromoneManager,
        # valori default -> mmas_solver riceve i parametri da main.py
        # regolano la regola di transizione con cui una formixa seleziona il prossimo vertice v da inserire nel vertex cover 
        alpha: float = 1.0, # Peso della Memoria Feromonica
        beta: float = 2.0, # Peso dell'euristica / desirabilità locale
        gamma: float = 1.0, # Parametro di scaling dell'euristica 
        rng: np.random.Generator = None # Generatore di numeri casuali
    ) -> Solution:
        """
        Costruisce una soluzione da zero. Si ferma quando TUTTI gli archi 
        sono coperti (`uncovered_count == 0`).
        """
        if rng is None:
            rng = np.random.default_rng()

        g = self.graph
        n = g.n

        # Reset della soluzione
        self.solution = Solution(g)
        sol = self.solution

        # Pre-calcolo 1 / peso
        # È una costante -> non ha senso ricalcolarla nel ciclo while
        # 1e-10 per non dividere MAI per zero, anche se i pesi sono >= 1.
        inv_weights = 1.0 / np.maximum(g.weights, 1e-10)

        # array booleano locale 1 se è stato scelto l'arco indice i
        in_cover = np.zeros(n, dtype=bool)

        # finché ci sono archi scoperti nel grafo...
        while sol.uncovered_count > 0:
            
            # Invece di controllare i nodi uno ad uno:
            # - Prendo i gradi scoperti correnti
            # - La maschera dei candidati sono i nodi NON nel cover CON grado residuo > 0
            # Nodi con grado residuo 0 sono inutili, non coprirebbero nulla di nuovo
            uncov_deg = sol.node_uncov_deg.astype(np.float64)
            candidate_mask = (~in_cover) & (uncov_deg > 0)

            if not np.any(candidate_mask):
                logger.error("ERRORE: non ci sono candidati validi!!!!!--")
                break 

            # EURISTICA eta: Grado residuo diviso per il peso
            # Se gamma = 1 è la classica euristica di Chvátal, altrimenti la si eleva
            if gamma == 1.0:
                eta = uncov_deg * inv_weights
            else:
                eta = (uncov_deg ** gamma) * inv_weights

            # feromoni dalla mappa
            tau = pheromone.tau

            # calcolo probabilità (Regola di Transizione)
            # P_i = (tau^alpha) * (eta^beta)

            # se alpha o beta sono 1, salto la potenza
            if alpha == 1.0 and beta == 1.0:
                scores = tau * eta
            elif alpha == 1.0:
                scores = tau * (eta ** beta)
            elif beta == 1.0:
                scores = (tau ** alpha) * eta
            else:
                scores = (tau ** alpha) * (eta ** beta)

            # Annullo le probabilità dei nodi che non sono candidati validi
            scores[~candidate_mask] = 0.0

            # Normalizzazione -> somma delle probabilità  1.0
            total = scores.sum()
            if total <= 0:
                # Se per assurdo va a 0 (underflow)
                # tutti i candidati alla pari
                scores[candidate_mask] = 1.0
                total = scores.sum()

            scores /= total

            # Selezione roulette-wheel -> Scelgo un nodo rispettando le probabilità
            # La np.random.choice prende in input n (array da 0 a n-1)
            chosen_node = rng.choice(n, p=scores)

            # Infilo il vincitore nella soluzione
            # Aggiorna i gradi residui per il prossimo giro
            sol.add_vertex(chosen_node)
            in_cover[chosen_node] = True

        return sol

    def apply_pruning(self) -> Solution:
        """
        Daemon action (pruning.py)
        Le formiche sono stocastiche potrebbero mettere dentro anche nodi non necessari e quindi ridondanti.
        Questo metodo serve ad eliminarli prima di valutare la soluzione.
        """
        return pruning_greedy(self.solution)
