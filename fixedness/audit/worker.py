import z3
import time
import resource
from typing import Dict, Tuple, Set

from fixedness.audit.candidate_filter import OracleIndex, PartitionCache, filter_candidates

# ── Shared worker state (set once per process via init_worker) ───────────────
_db            = None
_knowledge_map = None
_oracle_index  = None
_col_mapping   = None
_part_cache    = None

# ── Base Z3 solver cache (oracle + FD constraints, rebuilt when oracle changes)
_base_solver         = None
_base_solver_identity = None
_base_solver_attrs   = None
_base_solver_oracle_id = None   # id() sentinella per rilevare cambio oracle


def init_worker(db, df_oracle, col_mapping, knowledge_map):
    """Pool initializer: runs once per worker process, avoids re-pickling large objects."""
    global _db, _knowledge_map, _oracle_index, _col_mapping, _part_cache, _base_solver
    _db            = db
    _knowledge_map = knowledge_map
    _col_mapping   = col_mapping
    _oracle_index  = OracleIndex(df_oracle, db, col_mapping)
    _part_cache    = PartitionCache()
    _base_solver   = None   # forza rebuild al prossimo record


def _build_base_solver():
    """
    Costruisce il solver Z3 base una volta per processo/configurazione.
    Contiene: vincoli oracle (identity → QI/BK conditions) + FD constraints.
    Usa variabili fisse (non dipendono da r_id) per poter essere riusato con push/pop.
    """
    global _base_solver, _base_solver_identity, _base_solver_attrs, _base_solver_oracle_id

    db           = _db
    col_mapping  = _col_mapping
    oracle_index = _oracle_index

    solver       = z3.Solver()
    identity_var = z3.Int("identity")
    solver.add(identity_var >= 0, identity_var < oracle_index.n)

    clinical_vars = {}
    for attr_name, attr_id in col_mapping.items():
        if not isinstance(attr_id, int): continue
        if attr_name == 'Identity_Link': continue
        v = z3.Int(f"a{attr_id}")
        clinical_vars[attr_id] = v
        d_size = len(db.attributes[attr_id].domain)
        solver.add(v >= 0, v < d_size)

    qi_cols       = col_mapping.get('_qi_cols', [])
    bk_col        = col_mapping.get('_bk_col',        'Known_Diagnosis')
    bk_sa_logical = col_mapping.get('_bk_sa_logical', 'Diagnosis')

    for j, ora_row in oracle_index.df_oracle.iterrows():
        match_conditions = []
        valid_match      = True
        for qi in qi_cols:
            if qi not in col_mapping: continue
            qi_attr_id = col_mapping[qi]
            try:
                ora_val_idx = db.attributes[qi_attr_id].domain.index(str(ora_row[qi]))
                match_conditions.append(clinical_vars[qi_attr_id] == ora_val_idx)
            except KeyError:
                pass
            except ValueError:
                valid_match = False; break
        if valid_match and bk_col in ora_row.index and ora_row[bk_col] != 'UNKNOWN':
            bk_attr_id = col_mapping.get(bk_sa_logical)
            if bk_attr_id is not None:
                try:
                    bk_val_idx = db.attributes[bk_attr_id].domain.index(str(ora_row[bk_col]))
                    match_conditions.append(clinical_vars[bk_attr_id] == bk_val_idx)
                except (ValueError, KeyError):
                    valid_match = False
        if valid_match:
            if match_conditions:
                solver.add(z3.Implies(identity_var == j, z3.And(*match_conditions)))
        else:
            solver.add(identity_var != j)

    for constraint in db.constraints:
        if constraint.type != 'functional_dependency': continue
        det_a, dep_a = constraint.attributes[0], constraint.attributes[1]
        if det_a not in clinical_vars or dep_a not in clinical_vars: continue
        for det_idx, dep_set in constraint.mapping.items():
            for dep_idx in dep_set:
                solver.add(z3.Implies(
                    clinical_vars[det_a] == det_idx,
                    clinical_vars[dep_a] == dep_idx
                ))

    _base_solver           = solver
    _base_solver_identity  = identity_var
    _base_solver_attrs     = clinical_vars
    _base_solver_oracle_id = id(oracle_index)


def smt_linkage_worker(args):
    """
    Per ogni record r_id:
      Phase 1 — Z3 UNSAT proof  → fixedness = 1.0 (prova formale)
      Phase 2 — candidate_filter → n_c esatto, sponginess senza cap

    Ottimizzazione: oracle + FD constraints costruiti una volta (base solver),
    riusati con push/pop per ogni record (solo knowledge_map varia).
    """
    r_id, true_link_idx, timeout_ms, mem_limit_gb = args
    knowledge_map = _knowledge_map
    col_mapping   = _col_mapping
    oracle_index  = _oracle_index
    part_cache    = _part_cache

    if mem_limit_gb > 0:
        limit_bytes = int(mem_limit_gb * 1024 * 1024 * 1024)
        resource.setrlimit(resource.RLIMIT_AS, (limit_bytes, limit_bytes))

    # ── Phase 1: Z3 UNSAT check (base solver condiviso) ─────────────────────
    if id(oracle_index) != _base_solver_oracle_id:
        _build_base_solver()

    solver        = _base_solver
    identity_var  = _base_solver_identity
    clinical_vars = _base_solver_attrs

    if timeout_ms > 0:
        solver.set("timeout", timeout_ms)

    solver.push()   # livello 1: knowledge_map per questo record
    for (rc, ac), (v_min, v_max) in knowledge_map.items():
        if rc == r_id and ac in clinical_vars:
            solver.add(clinical_vars[ac] >= v_min, clinical_vars[ac] <= v_max)

    solver.push()   # livello 2: esclude il vero link (cerca controesempio)
    solver.add(identity_var != true_link_idx)

    # Snapshot statistiche pre-solve per diff incrementale
    stats_pre  = solver.statistics()
    pre_keys   = stats_pre.keys()
    def _pre(name):
        return stats_pre[pre_keys.index(name)][1] if name in pre_keys else 0

    t0       = time.perf_counter()
    res      = solver.check()
    z3_ms    = (time.perf_counter() - t0) * 1000.0

    stats_post = solver.statistics()
    post_keys  = stats_post.keys()
    def _post(name):
        return stats_post[post_keys.index(name)][1] if name in post_keys else 0

    conflicts    = _post('conflicts')    - _pre('conflicts')
    decisions    = _post('decisions')    - _pre('decisions')
    propagations = _post('propagations') - _pre('propagations')
    smt_fixed = (res == z3.unsat)

    solver.pop()   # rimuove vincolo identity != true_link_idx
    solver.pop()   # rimuove knowledge_map di questo record

    # ── Phase 2: candidate_filter (AC-3 + numpy, nessun cap) ────────────────
    compatible, n_c = filter_candidates(
        r_id            = r_id,
        knowledge_map   = knowledge_map,
        oracle_index    = oracle_index,
        db              = _db,
        col_mapping     = col_mapping,
        partition_cache = part_cache,
        true_link_idx   = true_link_idx,
    )

    ac3_fixed = (n_c == 1)

    if smt_fixed and ac3_fixed:
        fixedness  = 1.0
        promotion  = 'both'
    elif smt_fixed:
        fixedness  = 1.0
        promotion  = 'smt'
    elif ac3_fixed:
        fixedness  = 1.0
        promotion  = 'ac3'
    else:
        fixedness  = 0.0
        promotion  = 'none'

    sponginess = 1.0 / n_c  # P_guess: prob. of correct re-identification (n_c >= 1 guaranteed)

    return r_id, {
        'fixedness':         fixedness,
        'sponginess':        sponginess,
        'status':            str(res),
        'candidates':        n_c,
        'solve_ms':          round(z3_ms, 2),
        'conflicts':         conflicts,
        'decisions':         decisions,
        'propagations':      propagations,
        'promotion_source':  promotion,
    }
