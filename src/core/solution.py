"""
solution.py — Classe Solution per il Vertex Cover (versione ottimizzata numpy).

Una soluzione S e' un sottoinsieme di nodi V tale che ogni arco (u, v) in E
ha almeno un endpoint in S. Il costo w(S) = sum_{v in S} w(v).

Questa versione usa array numpy booleani per il cover e operazioni
vettorizzate per l'aggiornamento incrementale.
"""

import numpy as np
from typing import Set
from .graph import Graph


class Solution:
    """
    Rappresentazione ultra-efficiente di un Vertex Cover.
    Invece di ricalcolare quanti archi sono coperti da zero ogni volta, 
    aggiorno i contatori in modo "incrementale" solo guardando i vicini!
    """

    # __slots__ mi permette di risparmiare memoria RAM evitando che Python 
    # crei un dizionario interno per ogni singolo oggetto Solution. 
    # (Quando hai decine di formiche che creano migliaia di soluzioni, si sente!)
    __slots__ = ['graph', 'in_cover', 'weight', 'node_uncov_deg',
                 'uncovered_count', 'cover_size']

    def __init__(self, graph: Graph):
        self.graph = graph
        
        # Inizialmente nessun nodo fa parte del cover (tutto False)
        self.in_cover = np.zeros(graph.n, dtype=bool)
        
        # Il peso totale iniziale ovviamente è 0
        self.weight: float = 0.0
        
        # IMPORTANTE: node_uncov_deg tiene traccia del grado "scoperto" di ogni nodo.
        # Ovvero: quanti archi *non ancora coperti* toccano questo nodo?
        # All'inizio è semplicemente il grado totale del nodo
        self.node_uncov_deg = np.array(graph.degrees, dtype=np.int32)
        
        # Archi non coperti iniziali = tutti gli archi del grafo
        self.uncovered_count: int = graph.m  
        self.cover_size: int = 0

    @property
    def cover(self) -> Set[int]:
        """
        Funzione di comodo per compatibilità: mi restituisce il classico 
        Set di Python guardando dove l'array booleano è True.
        """
        return set(np.where(self.in_cover)[0])

    def add_vertex(self, v: int) -> None:
        """
        Aggiunge il nodo v al cover. Aggiorna incrementalmente
        il conteggio degli archi coperti in O(deg(v)).
        """
        if self.in_cover[v]:
            return # È già nel cover, skip

        # Lo aggiungo e aggiorno pesi e contatori base
        self.in_cover[v] = True
        self.cover_size += 1
        self.weight += self.graph.weights[v]

        # Per ogni nodo 'u' vicino a 'v'...
        neighbors = self.graph.adj_list[v]
        for u in neighbors:
            if not self.in_cover[u]:
                # L'arco (v, u) passa da scoperto a coperto
                self.uncovered_count -= 1
            
            # Dal momento che l'arco tra v e u è ora coperto, 
            # il grado "scoperto" del vicino 'u' diminuisce di 1.
            self.node_uncov_deg[u] -= 1
            
        # Visto che 'v' è nel cover, TUTTI i suoi archi sono coperti, 
        # quindi il suo grado scoperto precipita a 0.
        self.node_uncov_deg[v] = 0

    def remove_vertex(self, v: int) -> None:
        """
        Rimuove il nodo v dal cover. Aggiorna incrementalmente
        il conteggio degli archi coperti in O(deg(v)).
        Per la Daemon Action (il Pruning)
        """
        if not self.in_cover[v]:
            return

        # Lo rimuovo e scalo i pesi
        self.in_cover[v] = False
        self.cover_size -= 1
        self.weight -= self.graph.weights[v]

        uncov_increase = 0
        uncov_deg_v = 0
        
        neighbors = self.graph.adj_list[v]
        for u in neighbors:
            # Visto che ho tolto v, l'arco (v, u) per forza di cose 
            # aumenta il grado scoperto di u.
            self.node_uncov_deg[u] += 1
            
            # Ma l'arco (v,u) è rimasto TOTALMENTE scoperto? 
            # Sì, se e solo se neanche 'u' fa parte del cover!
            if not self.in_cover[u]:
                # L'arco (v, u) torna scoperto
                uncov_increase += 1
                uncov_deg_v += 1

        # Il grado scoperto di v = numero di vicini fuori dal cover
        self.node_uncov_deg[v] = uncov_deg_v
        self.uncovered_count += uncov_increase

    def is_valid(self) -> bool:
        """
        Verifica se una soluzione è valida (cioè è un Vertex Cover vero).
        Restituisce True solo se NON restano archi scoperti.
        """
        return self.uncovered_count == 0

    def is_redundant(self, v: int) -> bool:
        """
        Verifica se un nodo 'v' è ridondante: 
        Un nodo 'v' è ridondante se TUTTI i suoi vicini sono a loro volta nel cover.
        -> se lo rimuovo, non scopro nessun arco (perché ci pensano 
        già i suoi vicini a tenere coperti gli archi). 
        -> per il pruning
        """
        if not self.in_cover[v]:
            return False
            
        for u in self.graph.adj_list[v]:
            if not self.in_cover[u]:
                return False
        return True

    def copy(self) -> "Solution":
        """
        Copia veloce per salvare la Global Best o la Iteration Best senza
        incasinare i puntatori in memoria di Python.
        """
        new_sol = Solution.__new__(Solution)
        new_sol.graph = self.graph
        new_sol.in_cover = self.in_cover.copy()
        new_sol.weight = self.weight
        new_sol.node_uncov_deg = self.node_uncov_deg.copy()
        new_sol.uncovered_count = self.uncovered_count
        new_sol.cover_size = self.cover_size
        return new_sol

    def get_cover_vertices(self) -> list:
        """
        Stampa per comodità (i vertici nei file originali spesso partono da 1, 
        non da 0 come fa Python).
        """
        return sorted(int(v) + 1 for v in np.where(self.in_cover)[0])

    def __repr__(self) -> str:
        valid_str = "OK" if self.is_valid() else f"NO ({self.uncovered_count} archi scoperti)"
        return f"Solution(|S|={self.cover_size}, w={self.weight:.0f}, valid={valid_str})"

    def __lt__(self, other: "Solution") -> bool:
        """
        Permette di usare il < (minore) tra due soluzioni: vince chi pesa meno!
        """
        return self.weight < other.weight
