"""
loader.py — carica lo scenario di attacco da rich_test.db (SQLite).

Ritorna:
  db         — Database con attributi, record, vincoli FD (con mapping AC-3)
  df_med     — DataFrame medico con colonne logiche
  df_ora     — DataFrame oracle con colonne logiche + BK column
  col_to_id  — dict logical_name→attr_id + chiavi speciali
"""

import os
import sqlite3
import numpy as np
import pandas as pd
from fixedness.core.models import Database

PAT_LOGICAL = {
    'age':               'Age',
    'sex':               'Sesso',
    'city':              'City',
    'zip':               'Zip',
    'region':            'Region',
    'job':               'Job',
    'bmi':               'BMI',
    'smoking_status':    'Fumatore',
    'physical_activity': 'Attivita',
    'diagnosis':         'Diagnosis',
    'treatment':         'Treatment',
    'drug_class':        'DrugClass',
    'insurance_tier':    'InsuranceTier',
    'hba1c':             'HbA1c',
    'ldl':               'LDL',
    'glicemia_digiuno':  'Glicemia_Digiuno',
    'pressione_sistolica': 'Pressione',
    'risk_score':        'RiskScore',
}
LOGICAL_PAT = {v: k for k, v in PAT_LOGICAL.items()}

PER_LOGICAL = {
    'sex':    'Sesso',
    'city':   'City',
    'zip':    'Zip',
    'region': 'Region',
    'job':    'Job',
}

# FD esatte (forward) — mapping derivato dai record ground-truth
FD_EXACT_PAIRS = [
    ('Diagnosis',     'Treatment'),
    ('Diagnosis',     'DrugClass'),
    ('Diagnosis',     'InsuranceTier'),
    ('DrugClass',     'InsuranceTier'),
    ('Zip',           'City'),
    ('Zip',           'Region'),
]

# Restrizione InsuranceTier → DrugClass (ciclo con DrugClass→InsuranceTier)
_TIER_TO_DRUGS = {
    'Base':     {'Nessuno', 'Antivirali_FANS'},
    'Standard': {'Nessuno', 'Antivirali_FANS', 'Antiipertensivi'},
    'Premium':  {'Nessuno', 'Antivirali_FANS', 'Antiipertensivi',
                 'Anticoagulanti', 'Ipoglicemizzanti'},
}

# Soglie cliniche per FD bidirezionale HbA1c ↔ Diagnosis
_HBA1C_THRESH = 6.5   # >= 6.5 → Diabete; < 6.5 → not Diabete

# Soglie Pressione ↔ Diagnosi ad alta pressione
_PRESS_THRESH = 130   # >= 130 → Ipertensione o Cardiopatia
_PRESS_HIGH_DIAGS = {'Ipertensione', 'Cardiopatia'}


def _default_db_path():
    # Risale due livelli da fixedness/core/ → tests/fixedness_test/database/
    base = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    return os.path.join(base, 'database', 'rich_test.db')


def _build_exact_fd_mapping(db, det_id, dep_id):
    """Deriva mapping {det_val_idx → {dep_val_idx}} dai record ground-truth."""
    mapping = {}
    for rec in db.records:
        dv = rec.values.get(det_id)
        rv = rec.values.get(dep_id)
        if dv is not None and rv is not None:
            mapping.setdefault(dv, set()).add(rv)
    return mapping


def _build_tier_to_drug_mapping(db, tier_id, drug_id):
    """Restrizione InsuranceTier → allowed DrugClass (from _TIER_TO_DRUGS)."""
    tier_domain = db.attributes[tier_id].domain
    drug_domain = db.attributes[drug_id].domain
    drug_idx = {v: i for i, v in enumerate(drug_domain)}
    mapping = {}
    for t_idx, t_val in enumerate(tier_domain):
        allowed = _TIER_TO_DRUGS.get(t_val, set())
        mapping[t_idx] = {drug_idx[d] for d in allowed if d in drug_idx}
    return mapping


def _build_hba1c_to_diag_mapping(db, hba1c_id, diag_id):
    """Range FD: HbA1c value → allowed Diagnosis set (threshold-based)."""
    hba1c_domain = db.attributes[hba1c_id].domain
    diag_domain  = db.attributes[diag_id].domain
    diag_idx     = {v: i for i, v in enumerate(diag_domain)}

    diabete_idx  = diag_idx.get('Diabete')
    non_diab_idx = {i for v, i in diag_idx.items() if v != 'Diabete'}

    mapping = {}
    for h_idx, h_val in enumerate(hba1c_domain):
        try:
            h_float = float(h_val)
        except (ValueError, TypeError):
            mapping[h_idx] = set(range(len(diag_domain)))
            continue
        if h_float >= _HBA1C_THRESH:
            mapping[h_idx] = {diabete_idx} if diabete_idx is not None else set()
        else:
            mapping[h_idx] = non_diab_idx
    return mapping


def _build_diag_to_hba1c_mapping(db, diag_id, hba1c_id):
    """Range FD: Diagnosis → allowed HbA1c indices (threshold-based)."""
    diag_domain  = db.attributes[diag_id].domain
    hba1c_domain = db.attributes[hba1c_id].domain

    high_hba1c = set()
    low_hba1c  = set()
    for h_idx, h_val in enumerate(hba1c_domain):
        try:
            h_float = float(h_val)
        except (ValueError, TypeError):
            high_hba1c.add(h_idx); low_hba1c.add(h_idx)
            continue
        if h_float >= _HBA1C_THRESH:
            high_hba1c.add(h_idx)
        else:
            low_hba1c.add(h_idx)

    mapping = {}
    for d_idx, d_val in enumerate(diag_domain):
        mapping[d_idx] = high_hba1c if d_val == 'Diabete' else low_hba1c
    return mapping


def _build_press_to_diag_mapping(db, press_id, diag_id):
    """Range FD: Pressione ≥ _PRESS_THRESH → {Ipertensione, Cardiopatia}."""
    press_domain = db.attributes[press_id].domain
    diag_domain  = db.attributes[diag_id].domain
    diag_idx     = {v: i for i, v in enumerate(diag_domain)}

    high_diag = {diag_idx[d] for d in _PRESS_HIGH_DIAGS if d in diag_idx}
    low_diag  = {i for v, i in diag_idx.items() if v not in _PRESS_HIGH_DIAGS}

    mapping = {}
    for p_idx, p_val in enumerate(press_domain):
        try:
            p_int = int(float(p_val))
        except (ValueError, TypeError):
            mapping[p_idx] = set(range(len(diag_domain)))
            continue
        mapping[p_idx] = high_diag if p_int >= _PRESS_THRESH else low_diag
    return mapping


def _build_diag_to_press_mapping(db, diag_id, press_id):
    """Range FD: Diagnosis → allowed Pressione indices."""
    diag_domain  = db.attributes[diag_id].domain
    press_domain = db.attributes[press_id].domain

    high_press = set()
    low_press  = set()
    for p_idx, p_val in enumerate(press_domain):
        try:
            p_int = int(float(p_val))
        except (ValueError, TypeError):
            high_press.add(p_idx); low_press.add(p_idx)
            continue
        if p_int >= _PRESS_THRESH:
            high_press.add(p_idx)
        else:
            low_press.add(p_idx)

    mapping = {}
    for d_idx, d_val in enumerate(diag_domain):
        mapping[d_idx] = high_press if d_val in _PRESS_HIGH_DIAGS else low_press
    return mapping


def load_attack_scenario(config):
    anon_cfg      = config['experiment']['anonymization']
    qi_logical    = anon_cfg['quasi_identifiers']
    sa_logical    = anon_cfg['sensitive_attributes']
    leak_ratio    = anon_cfg['background_knowledge_frac']
    limit    = config['experiment']['real'].get('limit', 0)
    strategy = config['experiment']['real'].get('sampling_strategy', 'random')
    seed     = config.get('_seed', 42)
    db_path  = config['experiment'].get('db_path', _default_db_path())
    if not os.path.isabs(db_path) and not os.path.exists(db_path):
        db_path = _default_db_path()
    bk_sa_logical = anon_cfg.get('bk_sa', 'Diagnosis')
    bk_col        = f"Known_{bk_sa_logical}"

    # ── 1. Carica da SQLite ──────────────────────────────────────
    conn   = sqlite3.connect(db_path)
    if limit > 0 and strategy == 'first':
        df_pat = pd.read_sql(f"SELECT * FROM patients LIMIT {limit}", conn)
    else:
        df_pat = pd.read_sql("SELECT * FROM patients", conn)
    df_per = pd.read_sql("SELECT * FROM persons", conn)
    conn.close()

    if limit > 0 and strategy == 'random':
        df_pat = df_pat.sample(n=min(limit, len(df_pat)),
                               random_state=seed).reset_index(drop=True)

    df_pat = df_pat.reset_index(drop=True)
    df_per = df_per.reset_index(drop=True)

    # ── 2. df_med ────────────────────────────────────────────────
    all_logical    = qi_logical + sa_logical
    rename_pat_fwd = {db_col: log for db_col, log in PAT_LOGICAL.items()
                      if log in all_logical}
    df_med = df_pat.rename(columns=rename_pat_fwd).reset_index(drop=True)

    # ── 3. df_ora ────────────────────────────────────────────────
    rename_per_fwd = {db_col: log for db_col, log in PER_LOGICAL.items()
                      if log in qi_logical}
    df_ora = df_per.rename(columns=rename_per_fwd).copy().reset_index(drop=True)

    bk_sa_db_col = LOGICAL_PAT.get(bk_sa_logical, bk_sa_logical.lower())
    pid_to_sa    = dict(zip(df_pat['person_id'].tolist(),
                            df_pat[bk_sa_db_col].astype(str).tolist()))
    rng = np.random.RandomState(config.get('_seed', 42))
    known_vals = [
        pid_to_sa[pid] if pid in pid_to_sa and rng.rand() < leak_ratio else 'UNKNOWN'
        for pid in df_per['person_id']
    ]
    df_ora[bk_col] = known_vals

    # ── 4. Costruisce Database ───────────────────────────────────
    db        = Database()
    col_to_id = {}

    for logical in all_logical:
        db_col = LOGICAL_PAT.get(logical, logical.lower())
        vals   = set(df_pat[db_col].astype(str).unique())
        if db_col in df_per.columns:
            vals |= set(df_per[db_col].astype(str).unique())
        col_to_id[logical] = db.add_attribute(logical, sorted(vals))

    identity_domain            = df_per['tax_code'].astype(str).tolist()
    identity_attr_id           = db.add_attribute('Identity_Link', identity_domain)
    col_to_id['Identity_Link'] = identity_attr_id

    # ── 5. Aggiunge Record ───────────────────────────────────────
    tc_lookup = dict(zip(df_per['person_id'].tolist(),
                         df_per['tax_code'].astype(str).tolist()))
    for _, row in df_pat.iterrows():
        values = {}
        for logical in all_logical:
            db_col  = LOGICAL_PAT.get(logical, logical.lower())
            attr_id = col_to_id[logical]
            domain  = db.attributes[attr_id].domain
            val_str = str(row[db_col])
            try:
                values[attr_id] = domain.index(val_str)
            except ValueError:
                values[attr_id] = 0
        cf = tc_lookup.get(int(row['person_id']), 'UNKNOWN')
        try:
            values[identity_attr_id] = identity_domain.index(cf)
        except ValueError:
            values[identity_attr_id] = 0
        db.add_record(values)

    # ── 6. Vincoli FD con mapping per AC-3 ──────────────────────
    # 6a. FD esatte forward (derivate dai record)
    for det, dep in FD_EXACT_PAIRS:
        if det not in col_to_id or dep not in col_to_id:
            continue
        det_id = col_to_id[det]
        dep_id = col_to_id[dep]
        mapping = _build_exact_fd_mapping(db, det_id, dep_id)
        db.add_constraint('functional_dependency', [det_id, dep_id], mapping=mapping)

    # 6b. Restrizione InsuranceTier → DrugClass (ciclo con DrugClass→InsuranceTier)
    if 'InsuranceTier' in col_to_id and 'DrugClass' in col_to_id:
        tier_id = col_to_id['InsuranceTier']
        drug_id = col_to_id['DrugClass']
        mapping = _build_tier_to_drug_mapping(db, tier_id, drug_id)
        db.add_constraint('fd_restriction', [tier_id, drug_id], mapping=mapping)

    # 6c. HbA1c ↔ Diagnosis (ciclo bidirezionale threshold-based)
    if 'HbA1c' in col_to_id and 'Diagnosis' in col_to_id:
        hba1c_id = col_to_id['HbA1c']
        diag_id  = col_to_id['Diagnosis']
        db.add_constraint('fd_restriction', [hba1c_id, diag_id],
                          mapping=_build_hba1c_to_diag_mapping(db, hba1c_id, diag_id))
        db.add_constraint('fd_restriction', [diag_id, hba1c_id],
                          mapping=_build_diag_to_hba1c_mapping(db, diag_id, hba1c_id))

    # 6d. Pressione ↔ Diagnosis (ciclo bidirezionale threshold-based)
    if 'Pressione' in col_to_id and 'Diagnosis' in col_to_id:
        press_id = col_to_id['Pressione']
        diag_id  = col_to_id['Diagnosis']
        db.add_constraint('fd_restriction', [press_id, diag_id],
                          mapping=_build_press_to_diag_mapping(db, press_id, diag_id))
        db.add_constraint('fd_restriction', [diag_id, press_id],
                          mapping=_build_diag_to_press_mapping(db, diag_id, press_id))

    # ── 7. Metadati per il solver ────────────────────────────────
    col_to_id['_qi_cols']       = qi_logical
    col_to_id['_bk_col']        = bk_col
    col_to_id['_bk_sa_logical'] = bk_sa_logical

    return db, df_med, df_ora, col_to_id
