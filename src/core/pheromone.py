"""
pheromone.py — Gestione della scia di feromone per il MMAS.

Al contrario del classico TSP dove il feromone viene depositato sugli archi,
per il problema MWVCP il feromone (tau_v) è associato ai singoli vertici (a soluzione è un sottoinsieme S ⊆ V)
e rappresenta la desiderabilità storica di includere un determinato nodo v nel Vertex Cover.

Il sistema segue l'algoritmo Min-Max Ant System (MMAS): la traccia di feromone
è confinata matematicamente nell'intervallo [tau_min, tau_max]. Questa scelta
previene il rischio di stagnazione prematura garantendo un livello minimo di
esplorazione (tau_min) ed evitando un'intensificazione eccessiva (tau_max).
"""

import numpy as np


class PheromoneManager:
    """
    Gestisce i livelli di feromone e l'aggiornamento secondo le regole MMAS.
    """

    def __init__(self, n: int, rho: float = 0.02, initial_best_weight: float = 1000.0):
        self.n = n
        
        # rho è il tasso di evaporazione del feromone. Valori bassi (es. 0.02)
        # favoriscono l'esplorazione prolungata dato il budget ristretto (20.000 FE).
        self.rho = rho

        # I limiti tau_max e tau_min sono calcolati dinamicamente 
        # in base alle euristiche standard di Stützle e Hoos.
        self.tau_max = 1.0 / (rho * initial_best_weight)
        self.tau_min = self.tau_max / (2.0 * n)

        # Inizializzazione ottimistica: assegnare tau_max a tutti i nodi 
        # promuove un'esplorazione iniziale intensa sull'intero grafo.
        self.tau = np.full(n, self.tau_max, dtype=np.float64)

    def evaporate(self) -> None:
        """
        Applica il tasso di evaporazione a tutti i vertici mediante operazione vettoriale.
        """
        self.tau *= (1.0 - self.rho)

    def deposit(self, in_cover: np.ndarray, best_weight: float) -> None:
        """
        Deposita nuovo feromone sui nodi appartenenti alla soluzione fornita 
        (generalmente l'Iteration-Best o la Global-Best).
        """
        if best_weight <= 0:
            return
            
        # La quantità di feromone depositata è inversamente proporzionale 
        # al costo della soluzione.
        delta = 1.0 / best_weight
        
        # Incremento vettorizzato del feromone sfruttando la maschera booleana.
        self.tau[in_cover] += delta

    def clamp(self) -> None:
        """
        Applica il clamping (np.clip) a tutti i valori del feromone, 
        forzandoli nei limiti [tau_min, tau_max] per impedire la stagnazione.
        """
        np.clip(self.tau, self.tau_min, self.tau_max, out=self.tau)

    def update(self, in_cover: np.ndarray, best_weight: float) -> None:
        """
        Esegue l'intero ciclo di aggiornamento del feromone in sequenza 
        (evaporazione, deposito, clamping).
        """
        self.evaporate()
        self.deposit(in_cover, best_weight)
        self.clamp()

    def update_bounds(self, global_best_weight: float) -> None:
        """
        Ricalcola i limiti quando viene trovato un nuovo ottimo globale 
        e riapplica il clamping ai valori esistenti.
        """
        if global_best_weight <= 0:
            return
        self.tau_max = 1.0 / (self.rho * global_best_weight)
        self.tau_min = self.tau_max / (2.0 * self.n)
        # Dopo aver aggiornato i bounds, clamp nuovamente
        self.clamp()

    def reinitialize(self) -> None:
        """
        Re-inizializzazione del feromone al valore massimo. 
        Attivata quando l'algoritmo rileva una fase di stagnazione prolungata.
        """
        self.tau[:] = self.tau_max

    def get_convergence_factor(self) -> float:
        """
        Calcola il Convergence Factor, utile per monitorare empiricamente 
        l'avvicinamento dell'algoritmo alla stagnazione.
        (1.0 = massima stagnazione, 0.0 = massima esplorazione).
        """
        if self.tau_max <= self.tau_min:
            return 1.0
        range_tau = self.tau_max - self.tau_min
        # Per ogni nodo, calcola la distanza relativa dal centro
        mid = (self.tau_max + self.tau_min) / 2.0
        distances = np.abs(self.tau - mid) / (range_tau / 2.0)
        return float(np.mean(distances))

    def __repr__(self) -> str:
        return (
            f"PheromoneManager(n={self.n}, rho={self.rho}, "
            f"tau=[{self.tau_min:.6f}, {self.tau_max:.6f}])"
        )
