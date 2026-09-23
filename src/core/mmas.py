"""
mmas.py — L'Orchestratore dell'algoritmo Min-Max Ant System (MMAS).

Implementa il ciclo principale della metaeuristica ACO. L'esecuzione è governata 
dal budget massimo di valutazioni (FE) per rispettare i requisiti rigorosi 
degli studi sperimentali.

Il codice integra due concetti algoritmici essenziali:
1. Alternanza (Exploration/Exploitation): l'aggiornamento dei feromoni utilizza
   la Iteration-Best nella prima fase (es. 75% del tempo) per esplorare lo spazio, 
   passando poi alla Global-Best per intensificare la ricerca attorno all'ottimo.
2. Stagnation Recovery adattivo: la soglia di stagnazione non è fissa ma si calibra
   automaticamente sul budget e sulla dimensione del problema. Istanze piccole (SPI)
   entrano in stagnazione ogni poche iterazioni (max reiter ad es 10-15); istanze grandi (LPI) hanno una finestra
   più lunga proporzionale alle iterazioni totali disponibili. In questo modo il budget
   di 20.000 FE viene sfruttato integralmente indipendentemente dalla classe.
"""

import time
import numpy as np
from typing import Optional, Dict, Any
from .graph import Graph
from .solution import Solution
from .pheromone import PheromoneManager
from .ant import Ant
from .evaluator import Evaluator
from .pruning import pruning_greedy, pruning_advanced, pruning_rcost
from src.utils.logger import setup_logger

logger = setup_logger()


class MMAS_Solver:
    def __init__(
        self,
        graph: Graph,
        n_ants: int = 10,
        alpha: float = 1.0,
        beta: float = 2.0,
        gamma: float = 1.0,
        rho: float = 0.10,
        max_fe: int = 20_000,
        # stagnation_limit: int = 50,
        stagnation_limit: Optional[int] = None,
        gb_switch_ratio: float = 0.75,
        gb_switch_mode: str = "cycle",  # "cycle", "fixed_budget", "always", "never"
        pruning_strategy: str = "greedy",
        max_reinits: Optional[int] = None,
        seed: Optional[int] = None,
    ):
        # Inizializza l'orchestrazione memorizzando i parametri operativi.
        self.graph = graph
        self.n_ants = n_ants
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.rho = rho
        self.max_fe = max_fe

        # stagnation_limit adattivo: se non specificato esplicitamente, viene
        # calcolato in solve() in base al budget e alla dimensione del problema.
        self._stagnation_limit_override = stagnation_limit

        # Soglia temporale per la transizione tra Exploration e Exploitation.
        self.gb_switch_ratio = gb_switch_ratio
        self.gb_switch_mode = gb_switch_mode

        self.pruning_strategy = pruning_strategy
        self.seed = seed

        # Mappatura dinamica della strategia di pruning configurata.
        if pruning_strategy is None or (isinstance(pruning_strategy, str) and pruning_strategy.lower() == "none"):
            self._pruning_fn = None
        else:
            self._pruning_fn = {
                "greedy": pruning_greedy,
                "advanced": pruning_advanced,
                "rcost": pruning_rcost,
            }[pruning_strategy]

    def _construct_initial_solution(self, rng: np.random.Generator) -> Solution:
        """
        Costruisce una soluzione iniziale deterministica (Greedy) necessaria
        per calibrare i limiti iniziali del feromone tau_max e tau_min.
        Seleziona iterativamente i vertici che massimizzano il rapporto grado/peso 
        sugli archi ancora scoperti, e sottopone infine il risultato al pruning.
        """
        sol = Solution(self.graph)
        weights = self.graph.weights

        while sol.uncovered_count > 0:
            uncov_deg = sol.node_uncov_deg.astype(np.float64)
            mask = (~sol.in_cover) & (uncov_deg > 0)
            if not np.any(mask):
                break
                
            # Calcola il rapporto euristico assegnando priorità nulla ai nodi non candidabili.
            ratios = np.where(mask, uncov_deg / np.maximum(weights, 1e-10), -1.0)
            best_v = int(np.argmax(ratios))
            sol.add_vertex(best_v)

        # Applica pruning alla soluzione greedy
        if self._pruning_fn is not None:
            self._pruning_fn(sol)
        return sol

    def solve(self, verbose: bool = False) -> Dict[str, Any]:
        """
        Esegue il ciclo principale dell'algoritmo fino ad esaurimento del budget FE.
        """
        start_time = time.time()
        rng = np.random.default_rng(self.seed)
        evaluator = Evaluator(max_fe=self.max_fe)

        # Inizializzazione della soluzione greedy per impostare il baseline.
        greedy_sol = self._construct_initial_solution(rng)
        evaluator.register_evaluation(greedy_sol.weight)

        # Conserva la Global-Best (l'ottimo globale non è soggetto ad alcun reset).
        global_best = greedy_sol.copy()

        if verbose:
            logger.info(f"[INIT] Soluzione greedy iniziale: {greedy_sol}")

        # Inizializzazione del Pheromone Manager.
        pheromone = PheromoneManager(
            n=self.graph.n,
            rho=self.rho,
            initial_best_weight=global_best.weight,
        )

        iteration = 0
        no_improve_count = 0
        n_reinits = 0
        stopped_by_reinit = False

        # Stima delle iterazioni totali basata sul budget e sulla grandezza della colonia.
        estimated_total_iters = self.max_fe // self.n_ants

        # --- Parametri adattativi: stagnation_limit e max_reinits ---
        # stagnation_limit × max_reinits ≈ budget_totale_iterazioni
        # In questo modo il numero di reinit è calibrato sul problema:
        # - SPI (n=20): poche iters → finestra breve, più reinit frequenti
        # - LPI (n=800): tante iters → finestra lunga, reinit meno frequenti
        # Il prodotto garantisce che il budget FE sia sfruttato integralmente.
        if self._stagnation_limit_override is not None:
            stagnation_limit = self._stagnation_limit_override
        else:
            stagnation_limit = max(20, min(200, estimated_total_iters // 10))

        # max_reinits = quante fasi di stagnation ci stanno nel budget totale.
        # Con stagnation_limit × max_reinits ≈ estimated_total_iters,
        # l'algoritmo esaurisce il budget esattamente dopo max_reinits reset.
        max_reinits = max(3, estimated_total_iters // stagnation_limit)

        if verbose:
            logger.info(
                f"[CONFIG] stagnation_limit={stagnation_limit} | "
                f"max_reinits={max_reinits} | "
                f"iter_stimate={estimated_total_iters} "
                f"(prodotto={stagnation_limit * max_reinits} ≈ {estimated_total_iters})"
            )

        # Ciclo principale ACO.
        while not evaluator.is_budget_exhausted():
            iteration += 1
            iteration_best: Optional[Solution] = None

            # Costruzione stocastica delle soluzioni da parte dell'intera colonia.
            for ant_idx in range(self.n_ants):
                if evaluator.is_budget_exhausted():
                    break

                # Costruzione probabilistica
                ant = Ant(self.graph)
                sol = ant.construct_solution(
                    pheromone=pheromone,
                    alpha=self.alpha,
                    beta=self.beta,
                    gamma=self.gamma,
                    rng=rng,
                )

                # Fase di Daemon Action: potatura locale dei nodi ridondanti.
                if self._pruning_fn is not None:
                    self._pruning_fn(sol)

                # Valutazione effettiva della soluzione (consuma esattamente 1 FE).
                assert sol.is_valid(), f"Errore: Formica {ant_idx} sol non valida!"
                evaluator.register_evaluation(sol.weight)

                # Aggiorna l'Iteration-Best se applicabile.
                if iteration_best is None or sol.weight < iteration_best.weight:
                    iteration_best = sol.copy()

                # Aggiorna la Global-Best e ricalcola i limiti dei feromoni per rispecchiare il nuovo ottimo.
                if sol.weight < global_best.weight:
                    global_best = sol.copy()
                    no_improve_count = 0 # Azzera il contatore di stagnazione.
                    
                    pheromone.update_bounds(global_best.weight)

            if iteration_best is None:
                break

            # --- Transizione Esplorazione / Intensificazione ---
            if self.gb_switch_mode == "cycle":
                progress = no_improve_count / max(1, stagnation_limit)
                use_global_best = progress >= self.gb_switch_ratio
            elif self.gb_switch_mode == "fixed_budget":
                progress = evaluator.fe_count / float(self.max_fe)
                use_global_best = progress >= self.gb_switch_ratio
            elif self.gb_switch_mode == "always":
                use_global_best = True
            elif self.gb_switch_mode == "never":
                use_global_best = False
            else:
                use_global_best = (no_improve_count / max(1, stagnation_limit)) >= self.gb_switch_ratio

            if use_global_best:
                deposit_cover = global_best.in_cover
                deposit_weight = global_best.weight
            else:
                deposit_cover = iteration_best.in_cover
                deposit_weight = iteration_best.weight

            # Delega al Pheromone Manager l'aggiornamento dei livelli di feromone.
            pheromone.update(deposit_cover, deposit_weight)

            # Incrementa contatore di stagnazione
            no_improve_count += 1

            # --- Meccanismo di Stagnation Recovery ---
            # Quando non si migliora per `stagnation_limit` iterazioni consecutive,
            # si resetta il feromone a tau_max per ri-esplorare. La global_best
            # viene sempre conservata. Dopo `max_reinits` reset il ciclo termina:
            # entrambi i parametri sono calibrati per coprire tutto il budget.
            if no_improve_count >= stagnation_limit:
                pheromone.reinitialize()
                no_improve_count = 0
                n_reinits += 1
                if verbose:
                    logger.info(
                        f"  [REINIT #{n_reinits}/{max_reinits}] iter={iteration} | "
                        f"Best conservato: {global_best.weight:.0f}"
                    )
                if n_reinits >= max_reinits:
                    stopped_by_reinit = True
                    break

            if verbose and iteration % 100 == 0:
                cf = pheromone.get_convergence_factor()
                logger.info(
                    f"  [ITER {iteration:4d}] {evaluator.get_progress()} | "
                    f"ib={iteration_best.weight:.0f} | cf={cf:.3f}"
                )

        if verbose:
            logger.info(f"\n[DONE] {evaluator.get_progress()}")
            logger.info(f"  Miglior soluzione trovata: {global_best}")

        elapsed_time = time.time() - start_time
        return {
            "best_solution": global_best,
            "best_weight": global_best.weight,
            "cover_size": int(global_best.cover_size),
            "evaluator": evaluator,
            "iterations": iteration,
            "fe_used": evaluator.fe_count,
            "fe_at_best": evaluator.fe_at_best,
            "n_reinits": n_reinits,
            "stopped_by_reinit": stopped_by_reinit,
            "time_s": elapsed_time,
        }
