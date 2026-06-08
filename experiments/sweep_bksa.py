"""
sweep_bksa.py — varia il sensitive attribute usato come background knowledge × k.
Config: sweep.bksa.{bk_sa_values, k_values, method}
Output: tests/fixedness_test/results/bksa_YYYYMMDD_HHMMSS/bksa_results.csv
Esegui da project root: python experiments/sweep_bksa.py
"""
import sys, os, csv, yaml, pandas as pd
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from parallel_runner import compute_n_workers, run_parallel

CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'config.yaml')

FIELDS = ['bk_sa', 'k', 'n_records', 'oracle_size', 'n_fixed_mean',
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

    bksa_cfg   = sw.get('bksa', {})
    BK_SA_VALS = bksa_cfg.get('bk_sa_values', ['Diagnosis'])
    K_VALUES   = bksa_cfg.get('k_values',     [2, 5, 10, 25, 50])
    METHOD     = bksa_cfg.get('method',        'mondrian_k')
    SEEDS      = sw.get('seeds', [42, 123, 777])
    SAT_T      = base_cfg.get('system', {}).get('sat_timeout_sec', 20)
    N_W        = compute_n_workers(base_cfg)

    configs      = []
    params_table = {}
    for bk_sa in BK_SA_VALS:
        for k in K_VALUES:
            label  = f"{bk_sa},k={k}"
            params = dict(defs, k=k, method=METHOD, bk_sa=bk_sa)
            configs.append((label, params))
            params_table[label] = (bk_sa, k)

    total = len(BK_SA_VALS) * len(K_VALUES)
    print(f"\nBK-SA sweep: {len(BK_SA_VALS)} SA × {len(K_VALUES)} k = {total} configs")

    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", f"bksa_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    os.makedirs(out_dir, exist_ok=True)
    csv_path = f"{out_dir}/bksa_results.csv"
    print(f"Output: {csv_path}\n")

    rows = []

    def on_done(label, agg):
        bk_sa, k = params_table[label]
        rows.append(dict(bk_sa=bk_sa, k=k,
                         **{f: agg.get(f, '') for f in FIELDS[2:]}))

    run_parallel(configs, SEEDS, base_cfg, SAT_T, N_W, on_config_done=on_done)
    rows.sort(key=lambda r: (r['bk_sa'], r['k']))

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
        pivot = df.pivot(index='bk_sa', columns='k', values=metric)
        print(f"\n── {lbl}: bk_sa × k ──\n{pivot.to_string()}")


if __name__ == '__main__':
    main()
