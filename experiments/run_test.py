"""
run_test.py — smoke test sequenziale, nessun ProcessPool, output verboso.
Esegui da project root: python run_test.py
"""
import sys
import os
import time
import yaml
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from fixedness.core.loader import load_attack_scenario
from fixedness.anonymizers import get_anonymization_map
import fixedness.audit.worker as _wmod
from fixedness.audit.worker import smt_linkage_worker
from fixedness.audit.candidate_filter import OracleIndex, PartitionCache
from fixedness.sat.translator import SATTranslator, compute_rho_analytical

CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'config.yaml')


def _cap_oracle(df_ora, db, id_attr_id, n_records, cap):
    """Ritorna df_ora ridotto a `cap` righe mantenendo tutti i true link."""
    true_idxs = set()
    for r in range(n_records):
        true_idxs.add(db.get_ground_truth(r, id_attr_id))
    # Garantisce che i true link siano inclusi, aggiunge rumore fino a cap
    noise = [i for i in range(len(df_ora)) if i not in true_idxs][:cap - len(true_idxs)]
    keep  = sorted(true_idxs) + noise
    return df_ora.iloc[keep].reset_index(drop=True), {old: new for new, old in enumerate(keep)}


def main():
    with open(CONFIG_PATH, 'r') as f:
        config = yaml.safe_load(f)

    smoke   = config.get('sweep', {}).get('smoke_test', {})
    RECORD_LIMIT = smoke.get('n_records', 5)
    ORACLE_CAP   = smoke.get('oracle_cap', 200)
    SAT_TIMEOUT  = config.get('system', {}).get('sat_timeout_sec', 20)

    config['experiment']['real']['limit'] = RECORD_LIMIT
    config['system']['ram_per_core_gb']   = 0  # disabilita setrlimit (modalità sequenziale)

    print("=" * 60)
    print(f"SukokuToDbRisk — Smoke Test ({RECORD_LIMIT} records, oracle cap={ORACLE_CAP})")
    print("=" * 60)

    # ── 1. Load scenario ────────────────────────────────────────────
    print("\n[1] Loading attack scenario...")
    db, df_med, df_ora, col_mapping = load_attack_scenario(config)
    print(f"    Medical records : {len(db.records)}")
    print(f"    Oracle size     : {len(df_ora)}")
    print(f"    Attributes      : {len(db.attributes)}")
    print(f"    FD constraints  : {sum(1 for c in db.constraints if c.type == 'functional_dependency')}")

    # ── 2. Reindex oracle (cap) ─────────────────────────────────────
    id_attr_id = col_mapping['Identity_Link']
    df_ora_small, idx_remap = _cap_oracle(df_ora, db, id_attr_id, len(db.records), ORACLE_CAP)
    print(f"\n[2] Oracle capped: {len(df_ora_small)} rows (true links guaranteed)")

    # Remap true_link_idx in db records to new oracle indices
    for rec in db.records:
        old_idx = rec.values.get(id_attr_id)
        if old_idx in idx_remap:
            rec.values[id_attr_id] = idx_remap[old_idx]

    # Remap identity_domain in db.attributes to match capped oracle
    db.attributes[id_attr_id].domain = [
        db.attributes[id_attr_id].domain[old] for old in sorted(idx_remap, key=idx_remap.get)
    ]

    # ── 3. Phase transition metric ──────────────────────────────────
    print("\n[3] SAT phase transition metric (analytical)...")
    rho_info = compute_rho_analytical(db)
    print(
        f"    N={rho_info['n_records']}, M={rho_info['n_attributes']}, "
        f"max_domain={rho_info['max_domain']}, "
        f"clauses={rho_info['total_clauses_3sat']}, "
        f"base={rho_info['base_vars']}, aux={rho_info['aux_vars']}, "
        f"total_vars={rho_info['total_vars']}, rho={rho_info['rho']:.4f}"
    )

    # ── 4. Mondrian ─────────────────────────────────────────────────
    k = config['experiment']['anonymization']['k']
    print(f"\n[4] Mondrian k={k}...")
    knowledge_map = get_anonymization_map(db, df_med, config)
    print(f"    Knowledge map entries: {len(knowledge_map)}")

    # ── 5. Per-record linkage audit ─────────────────────────────────
    timeout_ms = SAT_TIMEOUT * 1000

    _wmod._db            = db
    _wmod._knowledge_map = knowledge_map
    _wmod._col_mapping   = col_mapping
    _wmod._base_solver   = None
    _wmod._oracle_index  = OracleIndex(df_ora_small, db, col_mapping)
    _wmod._part_cache    = PartitionCache()

    print(f"\n[5] Building Z3 base solver (oracle={len(df_ora_small)} rows)...", end='', flush=True)
    t_build = time.perf_counter()
    _wmod._build_base_solver()
    build_ms = (time.perf_counter() - t_build) * 1000
    print(f" {build_ms:.0f}ms")

    n_rec = len(db.records)
    print(f"\n[6] Linkage audit ({n_rec} records, sequential)...")
    hdr = f"    {'Rec':>4}  {'True CF':>16}  {'Fix':>5}  {'Spon':>5}  {'Cands':>5}  {'Z3ms':>6}  {'Conf':>5}  {'Dec':>5}  Status"
    print(hdr)
    print("    " + "─" * (len(hdr) - 4))

    fixed_count = 0
    t_audit_start = time.perf_counter()
    for r in range(n_rec):
        true_link_idx = db.get_ground_truth(r, id_attr_id)
        t_rec = time.perf_counter()
        r_id, res = smt_linkage_worker((r, true_link_idx, timeout_ms, 0))
        wall_ms = (time.perf_counter() - t_rec) * 1000

        cf_str = db.attributes[id_attr_id].domain[true_link_idx][:16]
        fixed_count += int(res['fixedness'] == 1.0)

        elapsed_total = time.perf_counter() - t_audit_start
        eta = elapsed_total / (r + 1) * (n_rec - r - 1)
        eta_s = f"{eta:.0f}s" if eta < 60 else f"{int(eta//60)}m{int(eta%60):02d}s"
        progress = f"[{r+1}/{n_rec} | ETA {eta_s}]" if r < n_rec - 1 else f"[{r+1}/{n_rec} | done]"

        print(f"    {r_id:>4}  {cf_str:>16}  {res['fixedness']:>5.2f}  {res['sponginess']:>5.2f}"
              f"  {res['candidates']:>5}  {res['solve_ms']:>6.1f}  {res['conflicts']:>5}  {res['decisions']:>5}"
              f"  {res['status']}  {progress}")

    total_ms = (time.perf_counter() - t_audit_start) * 1000
    print(f"\n    Total audit: {total_ms:.0f}ms  ({total_ms/n_rec:.1f}ms/rec avg)")
    print("\n" + "=" * 60)
    print(f"Fixed: {fixed_count}/{n_rec}  ({fixed_count/n_rec:.1%})")
    print("Smoke test complete.")


if __name__ == "__main__":
    main()
