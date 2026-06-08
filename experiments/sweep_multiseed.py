"""
sweep_multiseed.py — sweep con N semi per variance + solver stats.
Config: sweep.multiseed (supporta campo opzionale fixed: {param: val})
Output: tests/fixedness_test/results/multiseed_YYYYMMDD_HHMMSS/multiseed_results.csv
Esegui da project root: python experiments/sweep_multiseed.py
"""
import sys, os, csv, yaml, pandas as pd
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from parallel_runner import compute_n_workers, run_parallel

CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'config.yaml')

FIELDS = ['sweep_param', 'param_value', 'method', 'k', 'bk_frac', 'oracle_noise', 'n_patients',
          'n_records', 'oracle_size', 'n_fixed_mean',
          'mean_fixedness', 'std_fixedness', 'mean_sponginess', 'std_sponginess',
          'mean_candidates', 'std_candidates', 'mean_conflicts', 'std_conflicts',
          'mean_decisions', 'std_decisions', 'mean_propagations', 'std_propagations',
          'mean_solve_ms', 'std_solve_ms']


def main():
    with open(CONFIG_PATH) as f:
        base_cfg = yaml.safe_load(f)

    sw   = base_cfg.get('sweep', {})
    defs = dict(k=5, oracle_noise=50, n_patients=50, bk_frac=0.9,
                method='mondrian_k', l=2, t=0.2, epsilon=1.0, sigma=3)
    defs.update(sw.get('defaults', {}))

    SEEDS   = sw.get('seeds', [42, 123, 777])
    SAT_T   = base_cfg.get('system', {}).get('sat_timeout_sec', 20)
    N_W     = compute_n_workers(base_cfg)

    seen, configs, params_table = set(), [], {}
    for entry in sw.get('multiseed', []):
        sp    = entry['param']
        fixed = entry.get('fixed', {})
        for v in entry['values']:
            p   = dict(defs, **fixed, **{sp: v})
            key = (sp,) + tuple(sorted(p.items()))
            if key in seen:
                continue
            seen.add(key)
            label = f"{sp}={v}"
            if fixed:
                label += ',' + ','.join(f"{k}={v2}" for k, v2 in sorted(fixed.items()))
            configs.append((label, p))
            params_table[label] = (sp, v, p)

    print(f"\nMulti-seed sweep: {len(configs)} configs × {len(SEEDS)} seeds")

    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", f"multiseed_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    os.makedirs(out_dir, exist_ok=True)
    csv_path = f"{out_dir}/multiseed_results.csv"
    print(f"Output: {csv_path}\n")

    rows = []

    def on_done(label, agg):
        sp, v, p = params_table[label]
        rows.append(dict(
            sweep_param=sp, param_value=v, method=p['method'], k=p['k'],
            bk_frac=p['bk_frac'], oracle_noise=p['oracle_noise'], n_patients=p['n_patients'],
            **{f: agg.get(f, '') for f in FIELDS[7:]},
        ))

    run_parallel(configs, SEEDS, base_cfg, SAT_T, N_W, on_config_done=on_done)

    with open(csv_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    print(f"\nSaved → {csv_path}")


if __name__ == '__main__':
    main()
