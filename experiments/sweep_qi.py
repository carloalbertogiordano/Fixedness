"""
sweep_qi.py — varia il set di quasi-identifiers × k con N semi.
Config: sweep.qi_sets, sweep.qi.{k_values, method}
Output: tests/fixedness_test/results/qi_YYYYMMDD_HHMMSS/qi_results.csv
Esegui da project root: python experiments/sweep_qi.py
"""
import sys, os, csv, yaml, pandas as pd
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from parallel_runner import compute_n_workers, run_parallel

CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'config.yaml')

FIELDS = ['qi_name', 'n_qi', 'qi_list', 'k', 'n_records', 'oracle_size', 'n_fixed_mean',
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

    qi_cfg   = sw.get('qi', {})
    K_VALUES = qi_cfg.get('k_values', [2, 5, 10, 25, 50])
    METHOD   = qi_cfg.get('method',   'mondrian_k')
    QI_SETS  = sw.get('qi_sets', [])
    SEEDS    = sw.get('seeds', [42, 123, 777])
    SAT_T    = base_cfg.get('system', {}).get('sat_timeout_sec', 20)
    N_W      = compute_n_workers(base_cfg)

    if not QI_SETS:
        print("ERROR: nessun qi_sets in config.yaml sweep.qi_sets")
        return

    configs      = []
    params_table = {}
    for qi_entry in QI_SETS:
        qi_name = qi_entry['name']
        qi_list = qi_entry['qi']
        for k in K_VALUES:
            label  = f"{qi_name},k={k}"
            params = dict(defs, k=k, method=METHOD, qi_list=qi_list)
            configs.append((label, params))
            params_table[label] = (qi_name, qi_list, k)

    total = len(QI_SETS) * len(K_VALUES)
    print(f"\nQI sweep: {len(QI_SETS)} QI sets × {len(K_VALUES)} k = {total} configs")

    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", f"qi_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    os.makedirs(out_dir, exist_ok=True)
    csv_path = f"{out_dir}/qi_results.csv"
    print(f"Output: {csv_path}\n")

    rows = []

    def on_done(label, agg):
        qi_name, qi_list, k = params_table[label]
        rows.append(dict(
            qi_name=qi_name, n_qi=len(qi_list), qi_list='+'.join(qi_list), k=k,
            **{f: agg.get(f, '') for f in FIELDS[4:]},
        ))

    run_parallel(configs, SEEDS, base_cfg, SAT_T, N_W, on_config_done=on_done)
    rows.sort(key=lambda r: (r['n_qi'], r['k']))

    with open(csv_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    print(f"\nSaved → {csv_path}")

    df = pd.DataFrame(rows)
    if df.empty:
        return
    df['mean_fixedness']  = df['mean_fixedness'].astype(float)
    df['mean_sponginess'] = df['mean_sponginess'].astype(float)

    for metric, lbl in [('mean_fixedness', 'Fixedness'), ('mean_sponginess', 'Sponginess')]:
        pivot = df.pivot_table(index='n_qi', columns='k', values=metric, aggfunc='mean')
        print(f"\n── {lbl}: n_qi × k ──\n{pivot.to_string()}")


if __name__ == '__main__':
    main()
