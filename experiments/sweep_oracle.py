"""
sweep_oracle.py — analisi sensibilità oracolo: noise×method, bkfrac×method, griglia 2D.
Config: sweep.oracle
Output: tests/fixedness_test/results/oracle_YYYYMMDD_HHMMSS/oracle_results.csv
Esegui da project root: python experiments/sweep_oracle.py
"""
import sys, os, csv, yaml
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from parallel_runner import compute_n_workers, run_parallel

CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'config.yaml')

FIELDS = ['sweep_type', 'method', 'oracle_noise', 'bk_frac', 'k',
          'n_records', 'oracle_size', 'n_fixed_mean',
          'mean_fixedness', 'std_fixedness', 'mean_sponginess', 'std_sponginess',
          'mean_candidates', 'std_candidates', 'mean_conflicts', 'std_conflicts',
          'mean_decisions', 'std_decisions', 'mean_propagations', 'std_propagations',
          'mean_solve_ms', 'std_solve_ms']

_STAT_FIELDS = FIELDS[5:]  # n_records onward


def main():
    with open(CONFIG_PATH) as f:
        base_cfg = yaml.safe_load(f)

    sw   = base_cfg.get('sweep', {})
    defs = dict(k=5, oracle_noise=50, n_patients=50, bk_frac=0.9,
                method='mondrian_k', l=2, t=0.2, epsilon=1.0, sigma=3)
    defs.update(sw.get('defaults', {}))

    ora_cfg     = sw.get('oracle', {})
    NOISE_VALS  = ora_cfg.get('oracle_noise_values', [0, 10, 25, 50, 100, 300, 1000])
    BKFRAC_VALS = ora_cfg.get('bk_frac_values',     [0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0])
    METHODS     = ora_cfg.get('methods', [
        'no_anonymization', 'mondrian_k', 'l_diversity', 't_closeness',
        'microaggregation', 'laplace_dp', 'local_dp',
        'suppression', 'noise_addition', 'randomized_response',
    ])
    GRID_METHOD = ora_cfg.get('grid_method', 'mondrian_k')
    SEEDS       = sw.get('seeds', [42, 123, 777])
    SAT_T       = base_cfg.get('system', {}).get('sat_timeout_sec', 20)
    N_W         = compute_n_workers(base_cfg)

    configs, params_table = [], {}

    # 1. oracle_noise × method
    for method in METHODS:
        for noise in NOISE_VALS:
            label = f"noise={noise},method={method}"
            configs.append((label, dict(defs, method=method, oracle_noise=noise)))
            params_table[label] = dict(sweep_type='noise_vs_method', method=method,
                                       oracle_noise=noise, bk_frac=defs['bk_frac'],
                                       k=defs['k'])

    # 2. bk_frac × method
    for method in METHODS:
        for bk in BKFRAC_VALS:
            label = f"bkfrac={bk},method={method}"
            configs.append((label, dict(defs, method=method, bk_frac=bk)))
            params_table[label] = dict(sweep_type='bkfrac_vs_method', method=method,
                                       oracle_noise=defs['oracle_noise'], bk_frac=bk,
                                       k=defs['k'])

    # 3. griglia 2D oracle_noise × bk_frac (grid_method fisso)
    for noise in NOISE_VALS:
        for bk in BKFRAC_VALS:
            label = f"grid,noise={noise},bkfrac={bk}"
            configs.append((label, dict(defs, method=GRID_METHOD,
                                        oracle_noise=noise, bk_frac=bk)))
            params_table[label] = dict(sweep_type='grid', method=GRID_METHOD,
                                       oracle_noise=noise, bk_frac=bk, k=defs['k'])

    n_noise = len(METHODS) * len(NOISE_VALS)
    n_bk    = len(METHODS) * len(BKFRAC_VALS)
    n_grid  = len(NOISE_VALS) * len(BKFRAC_VALS)
    print(f"\nOracle sweep: {len(configs)} configs × {len(SEEDS)} seeds")
    print(f"  noise × method : {len(METHODS)} × {len(NOISE_VALS)} = {n_noise}")
    print(f"  bkfrac × method: {len(METHODS)} × {len(BKFRAC_VALS)} = {n_bk}")
    print(f"  grid ({GRID_METHOD}): {len(NOISE_VALS)} × {len(BKFRAC_VALS)} = {n_grid}")

    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", f"oracle_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    os.makedirs(out_dir, exist_ok=True)
    csv_path = f"{out_dir}/oracle_results.csv"
    print(f"Output: {csv_path}\n")

    rows = []

    def on_done(label, agg):
        meta = params_table[label]
        rows.append(dict(
            sweep_type=meta['sweep_type'], method=meta['method'],
            oracle_noise=meta['oracle_noise'], bk_frac=meta['bk_frac'], k=meta['k'],
            **{f: agg.get(f, '') for f in _STAT_FIELDS},
        ))

    run_parallel(configs, SEEDS, base_cfg, SAT_T, N_W, on_config_done=on_done)

    with open(csv_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    print(f"\nSaved → {csv_path}")


if __name__ == '__main__':
    main()
