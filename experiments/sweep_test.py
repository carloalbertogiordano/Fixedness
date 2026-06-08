"""
sweep_test.py — parameter sweep one-at-a-time.
Varia: k, bk_frac, oracle_noise, n_patients.
oracle_noise = righe oracle extra oltre ai true links (noise = identità non-pazienti).
Esegui da project root: python sweep_test.py
"""
import sys, os, copy, csv, time, yaml
import pandas as pd
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from fixedness.core.loader import load_attack_scenario
from fixedness.anonymizers import get_anonymization_map
import fixedness.audit.worker as _wmod
from fixedness.audit.worker import smt_linkage_worker
from fixedness.audit.candidate_filter import OracleIndex, PartitionCache

CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'config.yaml')


def _load_sweep_cfg(base_cfg):
    sw       = base_cfg.get('sweep', {})
    defaults = sw.get('defaults', {})
    DEFAULTS = dict(k=5, oracle_noise=50, n_patients=50, bk_frac=0.9,
                    method='mondrian_k', l=2, t=0.2, epsilon=1.0, sigma=3)
    DEFAULTS.update(defaults)
    sweeps_raw = sw.get('basic', [])
    SWEEPS = [(entry['param'], entry['values']) for entry in sweeps_raw]
    SAT_TIMEOUT = base_cfg.get('system', {}).get('sat_timeout_sec', 20)
    return DEFAULTS, SWEEPS, SAT_TIMEOUT


# ── helpers ────────────────────────────────────────────────────────────────

def build_configs(DEFAULTS, SWEEPS):
    seen, confs = set(), []
    for sp, vals in SWEEPS:
        for v in vals:
            p   = dict(DEFAULTS, **{sp: v})
            key = tuple(sorted(p.items()))
            if key in seen: continue
            seen.add(key)
            confs.append((sp, p))
    return confs


def make_oracle(df_ora, true_idxs, noise):
    """Ritorna (df_small, remap: old→new).  true_idxs sempre inclusi."""
    noise_rows = [i for i in range(len(df_ora)) if i not in true_idxs][:noise]
    keep       = sorted(true_idxs) + noise_rows
    remap      = {old: new for new, old in enumerate(keep)}
    return df_ora.iloc[keep].reset_index(drop=True), remap


def run_config(params, base_cfg, sat_timeout):
    cfg = copy.deepcopy(base_cfg)
    cfg['experiment']['real']['limit']                              = params['n_patients']
    cfg['experiment']['anonymization']['k']                         = params['k']
    cfg['experiment']['anonymization']['method']                    = params['method']
    cfg['experiment']['anonymization']['l']                         = params['l']
    cfg['experiment']['anonymization']['t']                         = params['t']
    cfg['experiment']['anonymization']['epsilon']                   = params['epsilon']
    cfg['experiment']['anonymization']['sigma']                     = params['sigma']
    cfg['experiment']['anonymization']['background_knowledge_frac'] = params['bk_frac']
    cfg['system']['sat_timeout_sec']  = sat_timeout
    cfg['system']['ram_per_core_gb']  = 0

    db, df_med, df_ora, col_mapping = load_attack_scenario(cfg)
    id_attr_id = col_mapping['Identity_Link']
    n_rec      = len(db.records)
    true_idxs  = {db.get_ground_truth(r, id_attr_id) for r in range(n_rec)}

    noise     = min(params['oracle_noise'], len(df_ora) - len(true_idxs))
    df_small, remap = make_oracle(df_ora, true_idxs, noise)
    knowledge_map   = get_anonymization_map(db, df_med, cfg)
    timeout_ms      = sat_timeout * 1000

    _wmod._db            = db
    _wmod._knowledge_map = knowledge_map
    _wmod._col_mapping   = col_mapping
    _wmod._oracle_index  = OracleIndex(df_small, db, col_mapping)
    _wmod._part_cache    = PartitionCache()

    fix_list, spon_list, cand_list = [], [], []
    for r in range(n_rec):
        orig = db.get_ground_truth(r, id_attr_id)
        new  = remap[orig]
        _, res = smt_linkage_worker((r, new, timeout_ms, 0))

        fix_list.append(res['fixedness'])
        spon_list.append(res['sponginess'])
        cand_list.append(res['candidates'])

    return {
        'n_records':        n_rec,
        'actual_oracle':    len(df_small),
        'n_fixed':          sum(1 for f in fix_list if f == 1.0),
        'mean_fixedness':   round(sum(fix_list)  / n_rec, 4),
        'mean_sponginess':  round(sum(spon_list) / n_rec, 4),
        'mean_candidates':  round(sum(cand_list) / n_rec, 2),
    }


# ── main ───────────────────────────────────────────────────────────────────

def main():
    with open(CONFIG_PATH) as f:
        base_cfg = yaml.safe_load(f)

    DEFAULTS, SWEEPS, SAT_TIMEOUT = _load_sweep_cfg(base_cfg)
    configs  = build_configs(DEFAULTS, SWEEPS)
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", f"sweep_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    os.makedirs(out_dir, exist_ok=True)
    csv_path = f"{out_dir}/sweep_results.csv"

    fields = ['sweep_param','method','k','bk_frac','oracle_noise','n_patients',
              'n_records','actual_oracle','n_fixed',
              'mean_fixedness','mean_sponginess','mean_candidates','elapsed_s']

    print(f"\nSweep: {len(configs)} configs  |  timeout={SAT_TIMEOUT}s/record  |  AC-3 exact candidate count")
    print(f"Output: {csv_path}\n")
    hdr = f"{'Sweep':<13} {'method':<22} {'k':>3} {'bk':>5} {'noise':>6} {'npat':>5}  {'fixed':>6}  {'fix':>6}  {'spon':>6}  {'cands':>6}  {'t(s)':>6}"
    print(hdr)
    print("─" * len(hdr))

    rows = []
    for sweep_param, params in configs:
        t0 = time.time()
        try:
            agg     = run_config(params, base_cfg, SAT_TIMEOUT)
            elapsed = time.time() - t0
            status  = f"{agg['n_fixed']}/{agg['n_records']}"
            row = {
                'sweep_param':     sweep_param,
                'method':          params['method'],
                'k':               params['k'],
                'bk_frac':         params['bk_frac'],
                'oracle_noise':    params['oracle_noise'],
                'n_patients':      params['n_patients'],
                'n_records':       agg['n_records'],
                'actual_oracle':   agg['actual_oracle'],
                'n_fixed':         agg['n_fixed'],
                'mean_fixedness':  agg['mean_fixedness'],
                'mean_sponginess': agg['mean_sponginess'],
                'mean_candidates': agg['mean_candidates'],
                'elapsed_s':       round(elapsed, 1),
            }
        except Exception as e:
            elapsed = time.time() - t0
            status  = f"ERR"
            row = {f: params.get(f, '?') for f in fields}
            row.update(sweep_param=sweep_param, n_records='?', actual_oracle='?',
                       n_fixed='?', mean_fixedness='ERR', mean_sponginess='ERR',
                       mean_candidates='ERR', elapsed_s=round(elapsed,1))
            print(f"  ERROR config {params}: {e}")

        rows.append(row)
        fix  = row['mean_fixedness']
        spon = row['mean_sponginess']
        cand = row['mean_candidates']
        print(f"{sweep_param:<13} {params['method']:<22} {params['k']:>3} {params['bk_frac']:>5.1f} "
              f"{params['oracle_noise']:>6} {params['n_patients']:>5}  "
              f"{status:>6}  {fix!s:>6}  {spon!s:>6}  {cand!s:>6}  {elapsed:.1f}s")

    with open(csv_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    # ── summary ────────────────────────────────────────────────────
    print(f"\nSaved → {csv_path}")
    print("\n── Summary by sweep parameter ────────────────────────────────────")
    df = pd.DataFrame([r for r in rows if r['mean_fixedness'] != 'ERR'])
    df['mean_fixedness']  = df['mean_fixedness'].astype(float)
    df['mean_sponginess'] = df['mean_sponginess'].astype(float)
    df['mean_candidates'] = df['mean_candidates'].astype(float)
    for param in ['k', 'bk_frac', 'oracle_noise', 'n_patients', 'method']:
        sub = df[df['sweep_param'] == param][[param,'mean_fixedness','mean_sponginess','mean_candidates']]
        if sub.empty: continue
        print(f"\n  {param}:")
        print(sub.sort_values(param).to_string(index=False))


if __name__ == '__main__':
    main()
