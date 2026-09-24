# MWVC-Optimization-HEMOL

Sviluppo, tuning ed analisi sperimentale di un algoritmo metaeuristico avanzato basato su **Max-Min Ant System (MMAS)** per la risoluzione del **Minimum Weight Vertex Cover Problem (MWVCP)**.

---

## 📌 Struttura del Progetto

```
MWVC-Optimization-HEMOL/
├── src/
│   ├── core/                        # Moduli principali dell'algoritmo MMAS
│   │   ├── graph.py                 # Caricamento grafi DIMACS e rappresentazione NumPy
│   │   ├── solution.py              # Gestione della copertura booleana dei vertici
│   │   ├── ant.py                   # Costruzione stocastica vettorizzata (roulette-wheel)
│   │   ├── pruning.py               # Daemon Action deterministica (Pruning Greedy)
│   │   ├── pheromone.py             # Tracce di feromone nodali tau(v), clamping e reset
│   │   ├── evaluator.py             # Monitoraggio del budget di 20.000 FE
│   │   ├── mmas.py                  # Orchestratore del ciclo ACO e Stagnazione Ciclo-Dinamica
│   │   └── exact_solver.py          # Formulazione ILP (scipy.milp) e Rilassamento LP (scipy.linprog)
│   │
│   ├── compute_exact_bounds.py      # Script per il calcolo dell'ottimo ILP e Rilassamento LP su tutte le istanze
│   ├── main.py                      # Esecuzione del benchmark ufficiale (SPI, MPI, LPI)
│   ├── tune_parameters.py           # Tuning in due fasi (Grid Search & Multi-seed)
│   ├── run_multiseed_validation.py  # Validazione multi-seed delle migliori configurazioni
│   ├── run_multiseed_population.py  # Sensibilità della dimensione della colonia (N_ants)
│   ├── run_ablation_switch.py       # Studio di ablazione sulle modalità di switch (Iteration-Best / Global-Best)
│   ├── rank_phase1_top5.py          # Ranking ed elezione delle top 5 configurazioni
│   ├── aggregate_results.py         # Aggregazione dati CSV e riepilogo per classe
│   └── analysis/                    # Moduli di analisi e generazione grafici
│       ├── stats.py                 # Calcolo metriche di sintesi (media, std, FE)
│       ├── plot_convergence.py      # Generazione curve di convergenza su 3 classi
│       ├── plot_convergence_runs.py # Traiettorie di convergenza sulle 10 run LPI
│       ├── plot_ablation.py         # Grafici dello studio di ablazione componenti
│       ├── plot_tuning.py           # Generazione mappe termiche 2D e sensibilità N_ants
│       ├── plot_scalability.py      # Scatterplot log-log di scalabilità temporale
│       └── generate_improvement_plots.py # Plot traiettorie e trade-off miglioramento vs tempo
│
├── doc/
│   └── report.ipynb                 # Notebook Jupyter interattivo per la riproduzione di tabelle e grafici
│
├── progetto/
│   ├── RelazioneHeuristic_ACO_MWVC_FINALE.pdf  # Documento PDF finale della relazione di progetto
│   ├── figure/                      # Figure ed asset grafici
│   ├── slide/                       # Materiale di presentazione
│   └── results/                     # Backup ed esportazioni dei risultati
├── wvcp-instances/                  # Istanze di benchmark (.txt) per le tre classi SPI, MPI, LPI
└── results/                         # Output sperimentali
    ├── csv/                         # Risultati in formato CSV (incluso exact_bounds.csv)
    └── plots/                       # Figure in formato vettoriale e PNG
```

---

## ⚙️ Modulo `src/compute_exact_bounds.py`

Il file [`src/compute_exact_bounds.py`](file:///c:/Users/macca/Desktop/università/magistrale/MWVC-Optimization-HEMOL/src/compute_exact_bounds.py) è lo script dedicato alla risoluzione esatta ed al calcolo dei bound teorici di riferimento su tutte le 90 istanze del benchmark:

- **Istanze SPI ($n \le 25$)**: Calcola la soluzione ottima globale esatta $\text{OPT}_{\text{ILP}}$ tramite **Programmazione Lineare Intera (ILP)** risolta con solutore CBC/HiGHS (`scipy.optimize.milp`).
- **Istanze MPI ed LPI ($n \ge 100$)**: Calcola il **Rilassamento Lineare continuo (LP Relaxation)** $\text{LB}_{\text{LP}}$ tramite `scipy.optimize.linprog` (con vincoli sparsi `csc_matrix`) ed esegue la ricerca dell'ottimo ILP entro un time-limit prefissato.
- **Output**: Genera il file [`results/csv/exact_bounds.csv`](file:///c:/Users/macca/Desktop/università/magistrale/MWVC-Optimization-HEMOL/results/csv/exact_bounds.csv), contenente per ogni istanza l'ottimo ILP, il rilassamento LP, lo stato di ottimalità ed il tempo di calcolo.

Esecuzione:
```bash
python src/compute_exact_bounds.py
```

---

## 🚀 Guida all'Esecuzione

### 1. Esecuzione del Benchmark Principale MMAS
```bash
python src/main.py
```

### 2. Calcolo dei Bound Esatti ed LP
```bash
python src/compute_exact_bounds.py
```

### 3. Riproduzione Analisi e Report
Aprire ed eseguire il notebook Jupyter:
```bash
jupyter notebook doc/report.ipynb
```

---

## 📊 Risorse e Presentazione

* **Presentazione PowerPoint Online:** [presentazione_mwvc.pptx](https://1drv.ms/p/c/410bdb10d49381ae/IQDVp5k09o03Ralw5Zt-uMjwASutuYBK_iU3V3djnMWqMdU?e=mbLhZ0)
* **Relazione PDF Finale:** [`progetto/RelazioneHeuristic_ACO_MWVC_FINALE.pdf`](file:///c:/Users/macca/Desktop/università/magistrale/MWVC-Optimization-HEMOL/progetto/RelazioneHeuristic_ACO_MWVC_FINALE.pdf)
