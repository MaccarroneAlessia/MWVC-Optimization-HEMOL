"""
graph.py — Parsing e struttura dati del grafo pesato per il MWVCP.

Questo script serve a leggere i file .txt (le "istanze" di test) e a 
costruire la nostra struttura dati Grafo.
Formato dei file DIMACS (per non dimenticare):
  Riga 1: n (numero di nodi)
  Riga 2: w_0 w_1 ... w_n (i pesi di ogni nodo)
  Righe 3 in poi: matrice di adiacenza (chi è collegato a chi) con 0 e 1.

I pesi e la matrice vengono salvati direttamente come array NumPy
(al contrario delle liste di Python quando i grafi sono diventati 
enormi (800 nodi) il codice si piantava. NumPy in memoria è contiguo e 
veloce.)
"""

import numpy as np
from pathlib import Path
from typing import List

class Graph:
    """
    Rappresenta il Grafo. 
    L'obiettivo del problema è scegliere un set di nodi (il "cover") in modo 
    da "coprire" tutti gli archi spendendo il minor 'peso' possibile.

    Attributi:
        n (int):              Numero di nodi.
        m (int):              Numero di archi.
        weights (np.ndarray): Vettore dei pesi dei nodi, shape (n,).
        adj_matrix (np.ndarray): Matrice di adiacenza booleana, shape (n, n).
        adj_list (list[list[int]]): Lista di adiacenza per accesso rapido ai vicini.
        degrees (np.ndarray): Vettore dei gradi dei nodi, shape (n,).
        edges (list[tuple[int, int]]): Lista di archi come coppie (u, v) con u < v.
    """

    def __init__(self, n: int, weights: np.ndarray, adj_matrix: np.ndarray):
        self.n = n
        
        # float64 per i pesi
        # dividere il grado per il peso (d/w) evita divisioni intere strane.
        self.weights = weights.astype(np.float64)
        
        # La matrice è fatta di soli 0 e 1
        self.adj_matrix = adj_matrix.astype(bool)

        # Costruisci lista di adiacenza e lista archi
        self.adj_list: List[List[int]] = [[] for _ in range(n)]
        self.edges: List[tuple] = []
        
        # Costruisco la lista iterando sulla metà superiore della matrice 
        # (il grafo è non orientato e simmetrico, evita doppioni).
        for i in range(n):
            for j in range(i + 1, n):
                if self.adj_matrix[i, j]:
                    self.adj_list[i].append(j)
                    self.adj_list[j].append(i)
                    self.edges.append((i, j))

        self.m = len(self.edges)
        
        # Gradi dei nodi pre-calcolati
        # Serviranno per l'euristica iniziale
        self.degrees = np.array([len(self.adj_list[v]) for v in range(n)], dtype=np.int32)

    @classmethod
    def load_from_file(cls, filepath: str) -> "Graph":
        """
        Carica un'istanza MWVCP dal formato standard del benchmark.
        
        Args:
            filepath: Percorso al file .txt dell'istanza.
            
        Returns:
            Oggetto Graph inizializzato.
        """
        path = Path(filepath)
        with open(path, "r") as f:
            lines = f.readlines()

        # Riga 1: numero di nodi
        n = int(lines[0].strip())

        # Leggo la seconda riga e splitto i pesi. Metto tutto in array NumPy
        weights = np.array([int(x) for x in lines[1].split()], dtype=np.float64)
        assert len(weights) == n, f"Errore: attesi {n} pesi, trovati {len(weights)}!"

        # Righe 3..n+2: matrice di adiacenza
        # Inizializzo a False e poi metto a True dove leggo 1
        adj_matrix = np.zeros((n, n), dtype=bool)
        for i in range(n):
            row_values = [int(x) for x in lines[2 + i].split()]
            assert len(row_values) == n, f"ERRORE Riga {i}: attesi {n} valori, trovati {len(row_values)}"
            for j in range(n):
                if row_values[j] == 1:
                    adj_matrix[i, j] = True

        return cls(n, weights, adj_matrix)

    def get_instance_info(self) -> str:
        """Restituisce una stringa descrittiva dell'istanza."""
        density = (2 * self.m) / (self.n * (self.n - 1)) if self.n > 1 else 0
        return (
            f"Grafo: n={self.n}, m={self.m}, "
            f"densità={density:.4f}, "
            f"peso_totale={self.weights.sum():.0f}, "
            f"peso_medio={self.weights.mean():.1f}, "
            f"grado_medio={self.degrees.mean():.1f}"
        )

    def __repr__(self) -> str:
        return f"Graph(n={self.n}, m={self.m})"
