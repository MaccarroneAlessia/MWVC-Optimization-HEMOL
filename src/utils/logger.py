"""
logger.py — Sistema di logging globale e utility di formattazione tabelle per il progetto.

Inizializza un logger configurato per scrivere sia su console (INFO) 
che su un file persistente in results/log/ (DEBUG) per consentire
un'analisi post-esecuzione dettagliata e la tracciabilità delle anomalie.
"""

import logging
from pathlib import Path
from typing import List, Dict, Any, Sequence

def setup_logger(name="HEMOL", log_file="esecuzione.log"):
    """
    Configura e restituisce l'istanza del logger di progetto.
    """
    project_root = Path(__file__).resolve().parent.parent.parent
    log_dir = project_root / "results" / "log"
    log_dir.mkdir(parents=True, exist_ok=True)
    
    log_path = log_dir / log_file
    
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    
    if not logger.handlers:
        file_handler = logging.FileHandler(log_path, mode='a', encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)
        
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        formatter = logging.Formatter('%(asctime)s | %(levelname)-7s | %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        
    return logger


def print_and_log_table(logger: logging.Logger, title: str, headers: List[str], rows: List[List[Any]], col_widths: Sequence[int] = None) -> None:
    """
    Formatta una tabella ASCII leggibile, la stampa a console e la invia contestualmente al logger.info.
    """
    if not rows:
        return

    if col_widths is None:
        col_widths = [
            max(len(str(h)), max((len(str(r[i])) for r in rows), default=0)) + 2
            for i, h in enumerate(headers)
        ]
    
    header_str = " | ".join(str(h).ljust(w) for h, w in zip(headers, col_widths))
    sep_str = "-" * len(header_str)
    
    table_lines = [
        "",
        "=" * len(header_str),
        title,
        "=" * len(header_str),
        sep_str,
        header_str,
        sep_str,
    ]
    for r in rows:
        row_str = " | ".join(str(v).ljust(w) for v, w in zip(r, col_widths))
        table_lines.append(row_str)
    table_lines.append(sep_str)
    table_lines.append("")
    
    full_text = "\n".join(table_lines)
    logger.info(full_text)


def get_top_unique_configs(
    results: List[Dict[str, Any]], 
    param_keys: Sequence[str] = ("n_ants", "alpha", "beta", "rho"), 
    top_n: int = 5,
    sort_key_fn=None
) -> List[Dict[str, Any]]:
    """
    Seleziona le prime top_n configurazioni UNICHE (distinte per la combinazione di iperparametri).
    Garantisce che ogni entry visualizzata nella Top N differisca da tutte le altre per almeno un parametro.
    """
    if sort_key_fn is None:
        def default_sort(x):
            weight = x.get("mean_weight", x.get("mean_w", x.get("best_weight", x.get("weight", float("inf")))))
            std = x.get("std_weight", x.get("std_w", 0.0))
            fe = x.get("mean_fe_best", x.get("mean_fe", x.get("fe_best", float("inf"))))
            return (weight, std, fe)
        sort_key_fn = default_sort

    sorted_results = sorted(results, key=sort_key_fn)
    unique_entries = []
    seen_params = set()

    for res in sorted_results:
        # Costruisce la tupla identificativa della configurazione
        param_tuple = tuple(res[k] for k in param_keys if k in res)
        if param_tuple not in seen_params:
            seen_params.add(param_tuple)
            unique_entries.append(res)
            if len(unique_entries) >= top_n:
                break

    return unique_entries
