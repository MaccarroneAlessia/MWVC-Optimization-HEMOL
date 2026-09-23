"""
pruning.py — Daemon Action per la potatura/prunning delle soluzioni MWVCP.

Fornisce strategie deterministiche per eliminare vertici ridondanti 
dalle soluzioni generate stocasticamente dalle formiche. 
Un vertice è definito ridondante se la sua rimozione non altera la validità 
del Vertex Cover (ovvero, tutti i suoi archi incidenti risultano già coperti 
da altri vertici del cover).

Poiché il pruning esplora lo spazio della singola soluzione in modo
deterministico locale senza invocare la Funzione Obiettivo, la sua esecuzione
non consuma Function Evaluations (FE).
"""

import numpy as np
from .solution import Solution


def pruning_greedy(solution: Solution) -> Solution:
    """
    Approccio greedy base: ordina i vertici del cover in ordine decrescente 
    rispetto al loro peso e tenta la rimozione sequenziale. 
    L'obiettivo è minimizzare rapidamente il costo totale eliminando
    per primi i vertici più onerosi.
    """
    g = solution.graph
    
    # Filtra i nodi correntemente inclusi nella soluzione.
    cover_nodes = np.where(solution.in_cover)[0]
    
    # Esegue un ordinamento decrescente sfruttando l'inversione di segno.
    sorted_indices = np.argsort(-g.weights[cover_nodes])
    sorted_nodes = cover_nodes[sorted_indices]

    for v in sorted_nodes:
        # Verifica che il nodo non sia stato già scartato nei passaggi precedenti.
        if not solution.in_cover[v]:
            continue
            
        # Rimuove il vertice se risulta ridondante.
        if solution.is_redundant(v):
            solution.remove_vertex(v)

    return solution


def pruning_advanced(solution: Solution) -> Solution:
    """
    Approccio avanzato: ordina i vertici massimizzando il rapporto Peso/Grado.
    La priorità di rimozione è data ai vertici molto costosi che contribuiscono
    poco alla connettività. Il processo è iterativo fino a convergenza locale.
    """
    g = solution.graph
    improved = True
    while improved:
        improved = False
        cover_nodes = np.where(solution.in_cover)[0]
        
        # Evita le divisioni per zero garantendo stabilità numerica nei calcoli dei rapporti.
        degrees_safe = np.maximum(g.degrees[cover_nodes], 1).astype(np.float64)
        scores = g.weights[cover_nodes] / degrees_safe
        
        sorted_indices = np.argsort(-scores)
        sorted_nodes = cover_nodes[sorted_indices]

        for v in sorted_nodes:
            if not solution.in_cover[v]:
                continue
            if solution.is_redundant(v):
                solution.remove_vertex(v)
                improved = True

    return solution


def pruning_rcost(solution: Solution, gamma: float = 1.0) -> Solution:
    """
    Approccio basato su rcost (Bouamama et al.): calcola per ogni vertice
    un valore rcost(v) = d(v)^gamma / w(v). 
    I vertici con rcost inferiore offrono il peggior bilancio grado/peso 
    e vengono esaminati per primi (ordinamento crescente).
    """
    g = solution.graph
    improved = True
    while improved:
        improved = False
        cover_nodes = np.where(solution.in_cover)[0]
        
        weights_safe = np.maximum(g.weights[cover_nodes], 1e-10)
        rcost = (g.degrees[cover_nodes].astype(np.float64) ** gamma) / weights_safe
        
        # Ordinamento crescente per eliminare prima i vertici con rapporto di copertura peggiore.
        sorted_indices = np.argsort(rcost)  
        sorted_nodes = cover_nodes[sorted_indices]

        for v in sorted_nodes:
            if not solution.in_cover[v]:
                continue
            if solution.is_redundant(v):
                solution.remove_vertex(v)
                improved = True

    return solution
