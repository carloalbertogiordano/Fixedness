"""
candidate_filter.py — Phase 2 senza Z3.

Sostituisce il loop blocking-clause con:
  1. OracleIndex  — numpy arrays per range query O(n/64) vectorizzato
  2. PartitionCache — cache candidati base per partizione Mondrian
  3. ac3()        — Arc Consistency 3 per FD ciclici
  4. filter_candidates() — pipeline completa per un record
"""

from __future__ import annotations
from collections import deque
from typing import Dict, List, Set, Tuple, Optional
import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# OracleIndex
# ─────────────────────────────────────────────────────────────────────────────

class OracleIndex:
    """
    Pre-computa numpy arrays per ogni attributo QI dell'oracle.
    Costruita una volta per run, passata a ogni worker.
    """
    def __init__(self, df_oracle, db, col_mapping):
        qi_cols     = col_mapping.get('_qi_cols', [])
        bk_col      = col_mapping.get('_bk_col')
        bk_sa       = col_mapping.get('_bk_sa_logical')

        self.df_oracle = df_oracle
        self.n       = len(df_oracle)
        self.qi_arrays: Dict[int, np.ndarray] = {}  # attr_id → array of domain indices
        self.bk_array: Optional[np.ndarray]   = None
        self.bk_attr_id: Optional[int]        = None

        # QI arrays
        for qi in qi_cols:
            attr_id = col_mapping.get(qi)
            if not isinstance(attr_id, int):
                continue
            if qi not in df_oracle.columns:
                continue
            domain    = db.attributes[attr_id].domain
            val_to_idx = {v: i for i, v in enumerate(domain)}
            col_vals   = df_oracle[qi].astype(str).tolist()
            self.qi_arrays[attr_id] = np.array(
                [val_to_idx.get(v, -1) for v in col_vals], dtype=np.int32
            )

        # BK array (domain indices of the BK SA column)
        if bk_col and bk_sa and bk_col in df_oracle.columns:
            bk_attr_id = col_mapping.get(bk_sa)
            if isinstance(bk_attr_id, int):
                self.bk_attr_id = bk_attr_id
                domain     = db.attributes[bk_attr_id].domain
                val_to_idx = {v: i for i, v in enumerate(domain)}
                col_vals   = df_oracle[bk_col].astype(str).tolist()
                # -1 = UNKNOWN
                self.bk_array = np.array(
                    [val_to_idx.get(v, -1) for v in col_vals], dtype=np.int32
                )


# ─────────────────────────────────────────────────────────────────────────────
# PartitionCache
# ─────────────────────────────────────────────────────────────────────────────

class PartitionCache:
    """
    Cache del pre-filtro QI per partizione Mondrian.
    Tutti i record nella stessa partizione hanno le stesse range QI → stessi
    candidati base → calcoliamo una volta sola.
    """
    def __init__(self):
        self._cache: Dict[tuple, np.ndarray] = {}

    def get_base_candidates(self, qi_key: tuple, compute_fn) -> np.ndarray:
        if qi_key not in self._cache:
            self._cache[qi_key] = compute_fn()
        return self._cache[qi_key]


# ─────────────────────────────────────────────────────────────────────────────
# AC-3
# ─────────────────────────────────────────────────────────────────────────────

def ac3(domains: Dict[int, Set[int]], constraints) -> bool:
    """
    Arc Consistency 3 su domini discreti.
    constraints: list of Constraint con .attributes=[det_id, dep_id] e .mapping.
    Modifica domains in-place.
    Ritorna False se un dominio diventa vuoto (inconsistente).
    """
    if not constraints:
        return True

    # Indice: det_attr → lista constraint con quel det
    det_index: Dict[int, list] = {}
    for c in constraints:
        if len(c.attributes) >= 2 and c.mapping:
            det_index.setdefault(c.attributes[0], []).append(c)

    queue = deque(constraints)

    while queue:
        c = queue.popleft()
        if len(c.attributes) < 2 or not c.mapping:
            continue

        det_id, dep_id = c.attributes[0], c.attributes[1]
        if det_id not in domains or dep_id not in domains:
            continue

        # Calcola dep valori supportati da almeno un det valore
        allowed_dep: Set[int] = set()
        for det_val in domains[det_id]:
            allowed_dep |= c.mapping.get(det_val, set())

        new_dep = domains[dep_id] & allowed_dep
        if new_dep == domains[dep_id]:
            continue  # nessun cambiamento

        if not new_dep:
            return False  # dominio vuoto → incoerente

        domains[dep_id] = new_dep
        # Re-accoda constraint che hanno dep_id come determinante
        for c2 in det_index.get(dep_id, []):
            queue.append(c2)

    return True


# ─────────────────────────────────────────────────────────────────────────────
# filter_candidates — pipeline principale
# ─────────────────────────────────────────────────────────────────────────────

def filter_candidates(
    r_id: int,
    knowledge_map: Dict[Tuple[int, int], Tuple[int, int]],
    oracle_index: OracleIndex,
    db,
    col_mapping: dict,
    partition_cache: PartitionCache,
    true_link_idx: int,
) -> Tuple[Set[int], int]:
    """
    Ritorna (set di oracle indices compatibili con record r_id, n_candidates).

    Pipeline:
      1. Pre-filtro QI vectorizzato (con cache partizione)
      2. Filtro BK per entry specifiche
      3. AC-3 su SA domains per ogni candidato superstite
    """
    qi_cols    = col_mapping.get('_qi_cols', [])
    all_sa_ids = [
        col_mapping[name] for name in col_mapping
        if isinstance(col_mapping[name], int)
        and name not in qi_cols
        and name not in ('Identity_Link',)
        and name not in ('_qi_cols', '_bk_col', '_bk_sa_logical')
    ]
    fd_constraints = [
        c for c in db.constraints
        if c.type in ('functional_dependency', 'fd_restriction') and c.mapping
    ]

    # ── Step 1: pre-filtro QI vectorizzato ──────────────────────
    # Chiave cache: tuple ordinata di (attr_id, v_min, v_max) per i QI
    qi_key = tuple(sorted(
        (attr_id, knowledge_map[(r_id, attr_id)][0], knowledge_map[(r_id, attr_id)][1])
        for attr_id in oracle_index.qi_arrays
        if (r_id, attr_id) in knowledge_map
    ))

    def _compute_base():
        mask = np.ones(oracle_index.n, dtype=bool)
        for attr_id, arr in oracle_index.qi_arrays.items():
            km = knowledge_map.get((r_id, attr_id))
            if km is None:
                continue
            v_min, v_max = km
            mask &= (arr >= v_min) & (arr <= v_max) & (arr >= 0)
        return np.where(mask)[0]

    base_idxs = partition_cache.get_base_candidates(qi_key, _compute_base)

    # ── Step 2: filtro BK (record-specifico, non cacheable) ─────
    candidates = list(base_idxs)
    if oracle_index.bk_array is not None and oracle_index.bk_attr_id is not None:
        bk_attr_id = oracle_index.bk_attr_id
        km_bk      = knowledge_map.get((r_id, bk_attr_id))
        if km_bk is not None:
            v_min, v_max = km_bk
            bk_arr = oracle_index.bk_array
            # UNKNOWN (-1) non viene filtrato; valori fuori range eliminati
            candidates = [
                j for j in candidates
                if bk_arr[j] == -1 or (v_min <= bk_arr[j] <= v_max)
            ]

    # ── Step 3: AC-3 su SA domains per ogni candidato ───────────
    compatible = set()
    for j in candidates:
        if _check_sa_consistency(j, r_id, knowledge_map, oracle_index,
                                  all_sa_ids, fd_constraints, db):
            compatible.add(j)

    # Includi true_link se non già presente (ground truth sempre feasible)
    compatible.add(true_link_idx)

    return compatible, len(compatible)


def _check_sa_consistency(
    j: int,
    r_id: int,
    knowledge_map: Dict,
    oracle_index: OracleIndex,
    sa_attr_ids: List[int],
    fd_constraints: list,
    db,
) -> bool:
    """
    Verifica che esista un'assegnazione SA consistente con:
      - knowledge_map ranges per il record r_id
      - BK value per oracle entry j (se noto)
      - FD constraints (via AC-3)
    """
    # Costruisce domini SA dal knowledge_map
    domains: Dict[int, Set[int]] = {}
    for attr_id in sa_attr_ids:
        km = knowledge_map.get((r_id, attr_id))
        if km is None:
            d_size = len(db.attributes[attr_id].domain)
            domains[attr_id] = set(range(d_size))
        else:
            v_min, v_max = km
            domains[attr_id] = set(range(v_min, v_max + 1))

    # Fissa BK se noto per questa entry oracle
    if oracle_index.bk_array is not None and oracle_index.bk_attr_id is not None:
        bk_idx = int(oracle_index.bk_array[j])
        if bk_idx >= 0:  # non UNKNOWN
            bk_attr_id = oracle_index.bk_attr_id
            if bk_attr_id in domains:
                # BK fissa il valore SA esattamente
                if bk_idx not in domains[bk_attr_id]:
                    return False  # BK fuori range km → impossibile
                domains[bk_attr_id] = {bk_idx}

    # AC-3: propaga FD constraints fino a fixpoint
    return ac3(domains, fd_constraints)
