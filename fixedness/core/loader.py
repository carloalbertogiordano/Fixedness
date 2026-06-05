"""
loader.py — generic attack scenario loader.

Reads any ScenarioConfig (from a scenario YAML) and builds the Database
with attributes, records, and FD constraints — no schema-specific code here.

Returns:
  db         — Database with attributes, records, FD constraints
  df_med     — DataFrame (target table, logical column names)
  df_ora     — DataFrame (oracle table, logical column names + BK column)
  col_to_id  — dict logical_name→attr_id + metadata keys (_qi_cols, _bk_col, _bk_sa_logical)
"""

import os
import sqlite3
import numpy as np
import pandas as pd

from fixedness.core.models import Database
from fixedness.core.scenario import ScenarioConfig, FDSpec, load_scenario, scenario_path


def _default_db_path():
    base = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    return os.path.join(base, 'database', 'rich_test.db')


def _build_fd_mapping(fd: FDSpec, db: Database, det_id: int, dep_id: int) -> dict:
    """Build AC-3 mapping {det_val_idx → set of allowed dep_val_idx} from an FDSpec."""
    det_domain = db.attributes[det_id].domain
    dep_domain = db.attributes[dep_id].domain
    dep_idx    = {v: i for i, v in enumerate(dep_domain)}

    if fd.type == 'exact':
        mapping: dict = {}
        for rec in db.records:
            dv = rec.values.get(det_id)
            rv = rec.values.get(dep_id)
            if dv is not None and rv is not None:
                mapping.setdefault(dv, set()).add(rv)
        return mapping

    if fd.type == 'restriction':
        mapping = {}
        for d_idx, d_val in enumerate(det_domain):
            allowed = fd.map.get(str(d_val), [])
            mapping[d_idx] = {dep_idx[v] for v in allowed if v in dep_idx}
        return mapping

    if fd.type == 'numeric_to_categorical':
        # det is numeric; dep is categorical
        # det >= threshold → dep ∈ high_vals; else dep ∉ high_vals
        high_dep = {dep_idx[v] for v in fd.high_vals if v in dep_idx}
        low_dep  = set(range(len(dep_domain))) - high_dep
        mapping  = {}
        for d_idx, d_val in enumerate(det_domain):
            try:
                num = float(d_val)
            except (ValueError, TypeError):
                mapping[d_idx] = set(range(len(dep_domain)))
                continue
            mapping[d_idx] = high_dep if num >= fd.threshold else low_dep
        return mapping

    if fd.type == 'categorical_to_numeric':
        # det is categorical; dep is numeric
        # det ∈ high_det_vals → dep >= threshold; else dep < threshold
        high_dep_idxs: set = set()
        low_dep_idxs:  set = set()
        for d_idx, d_val in enumerate(dep_domain):
            try:
                num = float(d_val)
            except (ValueError, TypeError):
                high_dep_idxs.add(d_idx)
                low_dep_idxs.add(d_idx)
                continue
            (high_dep_idxs if num >= fd.threshold else low_dep_idxs).add(d_idx)

        high_det = set(fd.high_det_vals)
        mapping  = {}
        for d_idx, d_val in enumerate(det_domain):
            mapping[d_idx] = high_dep_idxs if d_val in high_det else low_dep_idxs
        return mapping

    raise ValueError(f"Unknown FD type: {fd.type!r}")


def load_attack_scenario(config, scenario: ScenarioConfig = None):
    """
    Load and build the attack scenario.

    If scenario is None, reads config['experiment']['scenario'] (default: 'rich_medical')
    to select the scenario YAML from fixedness/scenarios/.
    """
    if scenario is None:
        sc_name  = config['experiment'].get('scenario', 'rich_medical')
        scenario = load_scenario(scenario_path(sc_name))

    anon_cfg      = config['experiment']['anonymization']
    qi_logical    = anon_cfg['quasi_identifiers']
    sa_logical    = anon_cfg['sensitive_attributes']
    leak_ratio    = anon_cfg['background_knowledge_frac']
    limit         = config['experiment']['real'].get('limit', 0)
    strategy      = config['experiment']['real'].get('sampling_strategy', 'random')
    seed          = config.get('_seed', 42)
    db_path       = config['experiment'].get('db_path', _default_db_path())
    if not os.path.isabs(db_path) and not os.path.exists(db_path):
        db_path = _default_db_path()
    bk_sa_logical = anon_cfg.get('bk_sa', 'Diagnosis')
    bk_col        = f"Known_{bk_sa_logical}"

    # Inverse maps: logical name → DB column name
    logical_to_target = {log: db for db, log in scenario.target_col_map.items()}
    logical_to_oracle = {log: db for db, log in scenario.oracle_col_map.items()}

    # ── 1. Load from SQLite ──────────────────────────────────────
    conn = sqlite3.connect(db_path)
    if limit > 0 and strategy == 'first':
        df_target = pd.read_sql(
            f"SELECT * FROM {scenario.target_table} LIMIT {limit}", conn)
    else:
        df_target = pd.read_sql(f"SELECT * FROM {scenario.target_table}", conn)
    df_oracle_raw = pd.read_sql(f"SELECT * FROM {scenario.oracle_table}", conn)
    conn.close()

    if limit > 0 and strategy == 'random':
        df_target = df_target.sample(
            n=min(limit, len(df_target)), random_state=seed).reset_index(drop=True)

    df_target     = df_target.reset_index(drop=True)
    df_oracle_raw = df_oracle_raw.reset_index(drop=True)

    # ── 2. df_med (target table, logical column names) ──────────
    all_logical   = qi_logical + sa_logical
    rename_target = {db_col: log
                     for db_col, log in scenario.target_col_map.items()
                     if log in all_logical}
    df_med = df_target.rename(columns=rename_target).reset_index(drop=True)

    # ── 3. df_ora (oracle table, logical column names + BK) ─────
    rename_oracle = {db_col: log
                     for db_col, log in scenario.oracle_col_map.items()
                     if log in qi_logical}
    df_ora = df_oracle_raw.rename(columns=rename_oracle).copy().reset_index(drop=True)

    bk_sa_db_col = logical_to_target.get(bk_sa_logical, bk_sa_logical.lower())
    pid_to_sa    = dict(zip(
        df_target[scenario.join_key].tolist(),
        df_target[bk_sa_db_col].astype(str).tolist(),
    ))
    rng = np.random.RandomState(seed)
    df_ora[bk_col] = [
        pid_to_sa[pid] if pid in pid_to_sa and rng.rand() < leak_ratio else 'UNKNOWN'
        for pid in df_oracle_raw[scenario.join_key]
    ]

    # ── 4. Build Database ────────────────────────────────────────
    db        = Database()
    col_to_id = {}

    for logical in all_logical:
        db_col = logical_to_target.get(logical, logical.lower())
        vals   = set(df_target[db_col].astype(str).unique())
        if logical in qi_logical:
            ora_db_col = logical_to_oracle.get(logical)
            if ora_db_col and ora_db_col in df_oracle_raw.columns:
                vals |= set(df_oracle_raw[ora_db_col].astype(str).unique())
        col_to_id[logical] = db.add_attribute(logical, sorted(vals))

    identity_domain          = df_oracle_raw[scenario.identity_col].astype(str).tolist()
    identity_attr_id         = db.add_attribute('Identity_Link', identity_domain)
    col_to_id['Identity_Link'] = identity_attr_id

    # ── 5. Add Records ───────────────────────────────────────────
    id_lookup = dict(zip(
        df_oracle_raw[scenario.join_key].tolist(),
        df_oracle_raw[scenario.identity_col].astype(str).tolist(),
    ))
    for _, row in df_target.iterrows():
        values = {}
        for logical in all_logical:
            db_col  = logical_to_target.get(logical, logical.lower())
            attr_id = col_to_id[logical]
            domain  = db.attributes[attr_id].domain
            val_str = str(row[db_col])
            try:
                values[attr_id] = domain.index(val_str)
            except ValueError:
                values[attr_id] = 0

        identity_val = id_lookup.get(row[scenario.join_key], 'UNKNOWN')
        try:
            values[identity_attr_id] = identity_domain.index(identity_val)
        except ValueError:
            values[identity_attr_id] = 0

        db.add_record(values)

    # ── 6. FD constraints from ScenarioConfig ───────────────────
    for fd_spec in scenario.functional_dependencies:
        if fd_spec.det not in col_to_id or fd_spec.dep not in col_to_id:
            continue
        det_id  = col_to_id[fd_spec.det]
        dep_id  = col_to_id[fd_spec.dep]
        mapping = _build_fd_mapping(fd_spec, db, det_id, dep_id)
        c_type  = 'functional_dependency' if fd_spec.type == 'exact' else 'fd_restriction'
        db.add_constraint(c_type, [det_id, dep_id], mapping=mapping)

    # ── 7. Metadata for solver ───────────────────────────────────
    col_to_id['_qi_cols']       = qi_logical
    col_to_id['_bk_col']        = bk_col
    col_to_id['_bk_sa_logical'] = bk_sa_logical

    return db, df_med, df_ora, col_to_id
