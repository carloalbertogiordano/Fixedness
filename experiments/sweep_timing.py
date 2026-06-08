"""
sweep_timing.py — misura runtime solver al variare di k, oracle_noise, n_patients.
Output: tests/fixedness_test/results/timing_YYYYMMDD_HHMMSS/timing_results.csv
Esegui da project root: python experiments/sweep_timing.py
"""
import sys, os, csv, yaml, pandas as pd
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from parallel_runner import compute_n_workers, run_parallel

CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'config.yaml')

FIELDS = ['sweep_param', 'param_value', 'k', 'oracle_noise', 'n_patients',
          'n_records', 'oracle_size', 'mean_build_ms',
          'mean_solve_ms', 'std_solve_ms', 'mean_median_ms', 'mean_p95_ms', 'mean_max_ms']


def main():
    with open(CONFIG_PATH) as f:
        base_cfg = yaml.safe_load(f)

    sw   = base_cfg.get('sweep', {})
    defs = dict(k=5, oracle_noise=50, n_patients=50, bk_frac=0.9,
                method='mondrian_k', l=2, t=0.2, epsilon=1.0, sigma=3)
    defs.update(sw.get('defaults', {}))

    SEEDS = sw.get('seeds', [42, 123, 777])
    SAT_T = base_cfg.get('system', {}).get('sat_timeout_sec', 20)
    N_W   = compute_n_workers(base_cfg)

    seen, configs, params_table = set(), [], {}
    for entry in sw.get('timing', []):
        sp = entry['param']
        for v in entry['values']:
            p   = dict(defs, **{sp: v})
            key = tuple(sorted(p.items()))
            if key in seen:
                continue
            seen.add(key)
            label = f"{sp}={v}"
            configs.append((label, p))
            params_table[label] = (sp, v, p)

    print(f"\nTiming sweep: {len(configs)} configs × {len(SEEDS)} seeds")

    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", f"timing_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    os.makedirs(out_dir, exist_ok=True)
    csv_path = f"{out_dir}/timing_results.csv"
    print(f"Output: {csv_path}\n")

    rows = []

    def on_done(label, agg):
        sp, v, p = params_table[label]
        rows.append(dict(
            sweep_param=sp, param_value=v,
            k=p['k'], oracle_noise=p['oracle_noise'], n_patients=p['n_patients'],
            n_records=agg['n_records'], oracle_size=agg['oracle_size'],
            mean_build_ms=agg['mean_build_ms'],
            mean_solve_ms=agg['mean_solve_ms'], std_solve_ms=agg['std_solve_ms'],
            mean_median_ms=agg['mean_median_ms'],
            mean_p95_ms=agg['mean_p95_ms'],
            mean_max_ms=agg['mean_max_ms'],
        ))

    run_parallel(configs, SEEDS, base_cfg, SAT_T, N_W, on_config_done=on_done)

    with open(csv_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    print(f"\nSaved → {csv_path}")

    df = pd.DataFrame(rows)
    if df.empty:
        return
    for col in ('mean_solve_ms', 'mean_median_ms', 'mean_p95_ms', 'mean_max_ms'):
        df[col] = df[col].astype(float)

    for sp in ['k', 'oracle_noise', 'n_patients']:
        sub = df[df['sweep_param'] == sp][
            ['param_value', 'n_records', 'oracle_size',
             'mean_build_ms', 'mean_solve_ms', 'mean_p95_ms', 'mean_max_ms']
        ]
        if sub.empty:
            continue
        print(f"\n── {sp} ──")
        print(sub.sort_values('param_value').to_string(index=False))


if __name__ == '__main__':
    main()
