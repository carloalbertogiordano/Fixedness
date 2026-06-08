"""
sweep_crossproduct.py — griglia completa k × method con N semi.
Output: tests/fixedness_test/results/crossproduct_YYYYMMDD_HHMMSS/
  crossproduct_results.csv
  matrix_{fixedness,sponginess,conflicts,solve_ms}.txt
Esegui da project root: python experiments/sweep_crossproduct.py
"""
import sys, os, csv, yaml, pandas as pd
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from parallel_runner import compute_n_workers, run_parallel

CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'config.yaml')

FIELDS = ['k', 'method', 'n_records', 'oracle_size', 'n_fixed_mean',
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

    cp       = sw.get('crossproduct', {})
    K_VALUES = cp.get('k_values',  [2, 5, 10, 25, 50])
    METHODS  = cp.get('methods',   ['mondrian_k'])
    SEEDS    = sw.get('seeds',     [42, 123, 777])
    SAT_T    = base_cfg.get('system', {}).get('sat_timeout_sec', 20)
    N_W      = compute_n_workers(base_cfg)

    configs      = []
    params_table = {}
    for k in K_VALUES:
        for method in METHODS:
            label = f"k={k},{method}"
            params = dict(defs, k=k, method=method)
            configs.append((label, params))
            params_table[label] = (k, method)

    print(f"\nCross-product: {len(K_VALUES)}k × {len(METHODS)}methods = {len(configs)} configs")

    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", f"crossproduct_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    os.makedirs(out_dir, exist_ok=True)
    csv_path = f"{out_dir}/crossproduct_results.csv"
    print(f"Output: {csv_path}\n")

    rows = []

    def on_done(label, agg):
        k, method = params_table[label]
        rows.append(dict(k=k, method=method, **{f: agg.get(f, '') for f in FIELDS[2:]}))

    run_parallel(configs, SEEDS, base_cfg, SAT_T, N_W, on_config_done=on_done)
    rows.sort(key=lambda r: (r['k'], r['method']))

    with open(csv_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    print(f"\nSaved → {csv_path}")

    df = pd.DataFrame(rows)
    if df.empty:
        return
    for col in ('mean_fixedness', 'mean_sponginess', 'mean_conflicts', 'mean_solve_ms'):
        df[col] = df[col].astype(float)

    for metric, label in [('mean_fixedness', 'Fixedness'), ('mean_sponginess', 'Sponginess'),
                           ('mean_conflicts', 'Conflicts'), ('mean_solve_ms', 'SolveMS')]:
        mat  = df.pivot(index='k', columns='method', values=metric)
        path = f"{out_dir}/matrix_{metric.replace('mean_', '')}.txt"
        with open(path, 'w') as mf:
            mf.write(f"── {label} ({len(SEEDS)} seeds) ──\n{mat.to_string()}\n")
        print(f"\n── {label} ──\n{mat.to_string()}")


if __name__ == '__main__':
    main()
