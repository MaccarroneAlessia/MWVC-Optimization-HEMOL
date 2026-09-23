"""
evaluator.py — Monitoraggio delle Function Evaluations (FE) -> BUDGET

L'Evaluator controlla il budget computazionale, fissato a 20.000 FE da consegna. 
- Ogni costruzione di una soluzione valida da parte di una formica (incluso il pruning) consuma una FE. 
La classe si occupa inoltre di tracciare la convergenza (convergence_log) 
per facilitare le analisi nello studio di ablazione.
"""

from typing import List, Tuple

class Evaluator:
    """
    Oggetto che tiene traccia di quante soluzioni abbiamo valutato.
    Viene condiviso tra i moduli per garantire che il limite non venga mai superato.
    """

    def __init__(self, max_fe: int = 20_000):
        # Il budget massimo fissato è 20.000 valutazioni
        self.max_fe = max_fe
        self.fe_count: int = 0
        
        # Inizializzazione del peso ottimo trovato
        self.global_best_weight: float = float("inf")
        
        # Tracciamento storico per l'analisi della convergenza e i plot finali
        self.convergence_log: List[Tuple[int, float]] = []
        
        # Registra l'istante esatto (in FE) in cui è stato aggiornato l'ottimo globale
        self.fe_at_best: int = 0

    def register_evaluation(self, weight: float) -> None:
        """
        Registra una nuova valutazione, aggiornando contatori e storico.
        """
        self.fe_count += 1

        # Aggiornamento dell'ottimo globale se si trova una soluzione strettamente migliore
        if weight < self.global_best_weight:
            self.global_best_weight = weight
            self.fe_at_best = self.fe_count

        # Traccia l'evoluzione della soluzione per le metriche
        self.convergence_log.append((self.fe_count, self.global_best_weight))

    def is_budget_exhausted(self) -> bool:
        """Restituisce True se il budget di valutazioni è stato esaurito."""
        return self.fe_count >= self.max_fe

    def remaining_fe(self) -> int:
        """Restituisce il numero di FE rimanenti."""
        return max(0, self.max_fe - self.fe_count)

    def reset(self) -> None:
        """Ripristina lo stato per consentire run multiple o nuovi esperimenti."""
        self.fe_count = 0
        self.global_best_weight = float("inf")
        self.convergence_log = []
        self.fe_at_best = 0

    def get_progress(self) -> str:
        """Genera una stringa riassuntiva del progresso dell'algoritmo."""
        pct = (self.fe_count / self.max_fe) * 100
        return (
            f"FE: {self.fe_count}/{self.max_fe} ({pct:.1f}%) | "
            f"Best: {self.global_best_weight:.0f} (a FE={self.fe_at_best})"
        )

    def __repr__(self) -> str:
        return f"Evaluator(fe={self.fe_count}/{self.max_fe}, best={self.global_best_weight:.0f})"
