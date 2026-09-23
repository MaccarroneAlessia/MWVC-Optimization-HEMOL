# Core: Il Motore Matematico dell'Algoritmo

Questa cartella contiene il cuore del solver basato su **MMAS (Max-Min Ant System)** per la risoluzione del **MWVCP (Minimum Weight Vertex Cover Problem)**.

Tutto il codice è stato ottimizzato sfruttando `numpy` per eliminare i lenti cicli *for* di Python, passando da strutture dati classiche (come i `Set`) ad array booleani e operazioni matriciali vettorizzate.

## L'Algoritmo ACO

Il nostro algoritmo si sviluppa in cicli iterativi fino all'esaurimento del budget di 20.000 **Function Evaluations (FE)**:

1. **Costruzione Stocastica**: Una colonia di formiche costruisce *S*, un vertex cover valido, scegliendo i nodi in modo probabilistico. La probabilità di scegliere un nodo $v$ dipende dalla traccia di feromone $\tau_v$ e dall'euristica dinamica di Chvátal $\eta_v = d_{\text{uncov}}(v)^\gamma / w_v$, dove $d_{\text{uncov}}(v)$ è il grado residuo non coperto.

2. **Pruning Greedy (Daemon Action)**: Le soluzioni generate vengono potate tramite una procedura deterministica che elimina i nodi ridondanti (vertici la cui rimozione non lascia alcun arco scoperto), ordinandoli per peso decrescente per eliminare prioritariamente i nodi più costosi.

3. **Deposito Selettivo del Feromone**: Solo la soluzione guida $S_{\text{guide}}$ ha il permesso di depositare nuovo feromone. La scelta tra *Iteration-Best* ($S_{\text{ib}}$) e *Global-Best* ($S_{\text{gb}}$) è governata dal parametro `gb_switch_mode`, che supporta 4 modalità:
   - `cycle` (**Ciclo-Dinamica**, proposta): alterna in base all'indice di stagnazione $\phi_{\text{cycle}} = k_{\text{stg}} / Stg_{\text{limit}}$. Se $\phi < 0.75$ deposita su $S_{\text{ib}}$ (esplorazione); se $\phi \ge 0.75$ passa a $S_{\text{gb}}$ (intensificazione).
   - `fixed_budget`: passa a $S_{\text{gb}}$ al raggiungimento del 75% del budget totale FE.
   - `never`: deposita sempre su $S_{\text{ib}}$ (esplorazione pura).
   - `always`: deposita sempre su $S_{\text{gb}}$ (intensificazione pura).

4. **Evaporazione e Clamping MAX-MIN**: I feromoni evaporano con tasso $\rho$ e vengono mantenuti strettamente tra $[\tau_{\min}, \tau_{\max}]$ per prevenire la stagnazione precoce. I limiti vengono **ricalcolati dinamicamente** ogni volta che viene trovato un nuovo ottimo globale: $\tau_{\max} = 1/(\rho \cdot w(S_{\text{gb}}))$, $\tau_{\min} = \tau_{\max} / (2|V|)$.

5. **Stagnation Recovery Adattivo**: Se la colonia non migliora per `stagnation_limit` iterazioni consecutive, tutti i feromoni vengono resettati a $\tau_{\max}$ per ri-esplorare nuovi bacini di attrazione. Sia `stagnation_limit` sia il numero massimo di reset (`max_reinits`) sono **calibrati automaticamente** in base al budget e alla dimensione del problema, garantendo che il budget FE venga sfruttato integralmente indipendentemente dalla classe di istanza (SPI, MPI, LPI).

## File Sorgente

| File | Descrizione |
| :--- | :--- |
| `graph.py` | Caricamento in memoria dei grafi dalle istanze DIMACS. Mantiene le liste di adiacenza e gli array NumPy di pesi e gradi per l'accesso rapido. |
| `solution.py` | Gestisce una singola copertura dei vertici tramite maschera booleana NumPy (`in_cover`). Aggiungere o rimuovere nodi aggiorna in $O(\Delta)$ il conteggio degli archi scoperti senza ricalcolare il grafo intero. |
| `evaluator.py` | Monitora rigorosamente quante valutazioni della funzione obiettivo (FE) sono state consumate, fermando l'esecuzione a 20.000. Traccia inoltre la convergenza storica (`convergence_log`) e l'istante esatto del best (`fe_at_best`). |
| `pheromone.py` | Gestisce i livelli di feromone $\tau_v$ *sui singoli vertici*. Implementa evaporazione, deposito vettorizzato, clamping $[\tau_{\min}, \tau_{\max}]$, aggiornamento dinamico dei limiti (`update_bounds`) e re-inizializzazione per stagnation recovery. Calcola inoltre il *Convergence Factor* per il monitoraggio empirico. |
| `ant.py` | La mente probabilistica della formica. Combina $\tau_v^\alpha$ ed $\eta_v^\beta$ (euristica di Chvátal) tramite `np.random.choice` per costruire soluzioni ammissibili con selezione *roulette-wheel* vettorizzata. |
| `pruning.py` | La *Daemon Action* deterministica. Analizza la soluzione della formica ed elimina i nodi più pesanti e ridondanti, producendo una copertura minimale di peso contenuto. Fornisce 3 strategie: `pruning_greedy`, `pruning_advanced` e `pruning_rcost`. |
| `mmas.py` | Il direttore d'orchestra. Inizializza la colonia, gestisce l'alternanza tra *Iteration-Best* e *Global-Best* secondo le 4 modalità di switch, implementa lo stagnation recovery adattivo con reset feromonico, e coordina l'intero ciclo ACO fino all'esaurimento del budget. |
| `exact_solver.py` | Solutore esatto e bound teorici. Fornisce: (1) Lower Bound via Maximum Weighted Matching, (2) LP Relaxation via `scipy.linprog`, (3) Soluzione ILP esatta via `scipy.optimize.milp` (solo per istanze SPI con $n \le 50$). |
| `../compute_exact_bounds.py` | Script di livello super-modulo per il calcolo sistematico di $\text{OPT}_{\text{ILP}}$ e $\text{LB}_{\text{LP}}$ su tutte le 90 istanze e salvataggio su `results/csv/exact_bounds.csv`. |
| `logger.py` | Alias di compatibilità che re-esporta `setup_logger` da `src.utils.logger`. |

## Parametri, Iperparametri e Configurazione Finale

### Configurazione Eletta

Tutti i valori sono stati selezionati mediante Grid Search a 16 combinazioni, validazione multi-seed su 5 seed indipendenti e studio di sensibilità sulla dimensione della colonia, tutti sull'istanza LPI (`vc_800_10000`, $|V|=800$, $|E|=10000$):

$$\alpha = 1.0, \quad \beta = 2.0, \quad \rho = 0.10, \quad N_{\text{ants}} = 80, \quad \gamma_{\text{switch}} = 0.75$$

Risultato multi-seed sulla configurazione eletta: $\mu = 44450.60 \pm 23.44$, convergenza media in sole $4136$ FE.

---

### Effetto di Ogni Parametro (dati da CSV su istanza LPI)

#### $\alpha$ — Peso Feromonico (Memoria di Colonia)

Regola quanto le formiche si affidano alla memoria storica (feromone accumulato dalle iterazioni precedenti).

| $\alpha$ | Effetto osservato su LPI | Dato di riferimento |
| :---: | :--- | :--- |
| **1.0** ✓ | Equilibrio ottimale tra memoria e guida euristica | $\mu = 44450.60, \sigma = 23.44$ (multi-seed) |
| 2.0 | La colonia diventa iper-conservativa, si blocca sui primi minimi locali trovati | $\mu = 44495.60, \sigma = 64.65$ (multi-seed): varianza quasi **3× maggiore** |
| 0.0 | La memoria viene ignorata, le formiche usano solo l'euristica greedy | Testato nell'ablation: $\mu = 11520.0$ su MPI — funziona ma perde la guida evolutiva |

$\alpha$ troppo alto → convergenza prematura e instabilità; $\alpha = 0$ → greedy puro senza apprendimento.

---

#### $\beta$ — Peso Euristico (Desiderabilità di Chvátal)

Regola quanto le formiche si affidano all'euristica locale $\eta_v = d_{\text{uncov}}(v) / w_v$ (rapporto archi scoperti / peso del nodo).

| $\beta$ | Effetto osservato su LPI | Dato di riferimento |
| :---: | :--- | :--- |
| **2.0** ✓ | Equilibrio tra guida locale e diversificazione stocastica | $\mu = 44450.60, \sigma = 23.44$ |
| 5.0 | Le formiche diventano quasi greedy: convergenza iniziale rapida ($t \approx 1522$s) ma diversificazione ridotta | $\mu = 44463.80, \sigma = 20.76$ (multi-seed): peso medio peggiore di $+13.2$ |
| 0.0 | Le formiche sono "cieche": scelgono i nodi quasi a caso, producendo coperture pesanti | Testato nell'ablation: degrado significativo ($\mu = 11529.0$, $\sigma = 12.4$ su MPI) |

$\beta$ alto → greedy miope, trova velocemente ma si blocca presto; $\beta = 0$ → nessuna guida euristica, convergenza lentissima.

---

#### $\rho$ — Tasso di Evaporazione

Regola la velocità con cui la colonia "dimentica" le soluzioni passate. Valori bassi conservano la memoria a lungo; valori alti la cancellano rapidamente.

| $\rho$ | Effetto osservato su LPI | Dato di riferimento |
| :---: | :--- | :--- |
| 0.02 | Il feromone si accumula e satura, congelando l'algoritmo su soluzioni subottimali | $w = 44563$ (seed=42, Fase 1): **peggior risultato** della griglia per $\alpha=1, \beta=2$ |
| **0.10** ✓ | Oblio graduale: le tracce promettenti sopravvivono abbastanza da essere rafforzate | $\mu = 44450.60, \sigma = 23.44$: **minima varianza** e convergenza rapidissima ($4136$ FE) |
| 0.15 | Ancora competitivo ma con convergenza più lenta e varianza maggiore | $\mu = 44454.20, \sigma = 31.76$: sigma $+35\%$ e FE medio $13672$ (3.3× più lento) |
| 0.20 | Cancella le tracce troppo presto, le formiche non consolidano le soluzioni migliori | $w = 44463$ (seed=42, Fase 1) |

$\rho$ troppo basso → stagnazione per saturazione feromonica; $\rho$ troppo alto → l'informazione si perde prima di essere utile.

---

#### $N_{\text{ants}}$ — Dimensione della Colonia

Regola quante formiche costruiscono soluzioni in parallelo per ogni ciclo. Con budget fisso a $20000$ FE, più formiche per ciclo = meno cicli totali di aggiornamento feromonico.

| $N_{\text{ants}}$ | $\mu$ | $\sigma$ | FE medio al best | Note |
| :---: | :---: | :---: | :---: | :--- |
| 40 | 44451.80 | 29.06 | 9837.2 | Molti cicli ma scarsa diversità per ciclo |
| 60 | 44448.80 | 30.91 | 7794.8 | Quasi equivalente a 80 ($\Delta\mu = 1.8$) |
| **80** ✓ | **44450.60** | **23.44** | **4136.0** | **Minima varianza, convergenza più rapida** |
| 120 | 44490.80 | 34.33 | 6056.4 | Lieve degrado: meno cicli di apprendimento |
| 160 | 44525.60 | 140.18 | 7487.6 | Varianza altissima: pochi cicli MMAS, sottocampionamento |

colonie piccole → convergenza lenta per scarsa diversità intra-ciclo; colonie grandi → troppo pochi cicli MMAS per raffinare il feromone.

---

#### `gb_switch_mode` — Strategia di Alternanza Esplorazione/Intensificazione

Determina come viene scelta la soluzione guida $S_{\text{guide}}$ per il deposito feromonico (dati multi-seed su LPI, $\rho = 0.10$):

| Modalità | $\mu$ | $\sigma$ | Comportamento |
| :--- | :---: | :---: | :--- |
| `fixed_budget` | **44435.40** | **18.82** | Switch fisso al 75% del budget — migliore a $\rho = 0.10$ |
| `cycle` | 44450.60 | 23.44 | Ciclo-Dinamica sincronizzata col reset — equivale a `never` a $\rho = 0.10$ |
| `never` | 44450.60 | 23.44 | Sola $S_{\text{ib}}$ (esplorazione pura) — identica a `cycle` a $\rho = 0.10$ |
| `always` | 44592.80 | 115.72 | Sola $S_{\text{gb}}$ (intensificazione pura) — **nettamente peggiore** ($+142$ peso, $\sigma$ 5× maggiore) |

> **Nota**: A $\rho = 0.10$, `cycle` e `never` producono risultati identici seed per seed perché la soglia $\phi_{\text{cycle}} \ge 0.75$ non viene raggiunta prima che scatti il reset per stagnazione. Le due modalità si differenziano a $\rho = 0.15$, dove `cycle` ($\sigma = 31.76$) risulta più stabile di `never` ($\sigma = 44.64$).
