"""
sweep_rho.py — Phase transition rho vs N (analytical) + effective rho (post-anonymization).
Output: tests/fixedness_test/results/rho_YYYYMMDD_HHMMSS/rho_results.csv
Esegui da project root: python experiments/sweep_rho.py
"""
import sys, os, csv, yaml
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from fixedness.core.loader import load_attack_scenario
from fixedness.sat.translator import compute_rho_analytical, compute_effective_rho
from fixedness.anonymizers import get_anonymization_map

CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'config.yaml')

FIELDS = ['n_patients', 'method', 'k', 'n_attributes', 'max_domain',
          'base_vars', 'aux_vars', 'total_vars', 'total_clauses_3sat', 'rho', 'effective_rho', 'retention_ratio']


def main():
    with open(CONFIG_PATH) as f:
        config = yaml.safe_load(f)

    sw   = config.get('sweep', {})
    defs = dict(k=5, oracle_noise=50, n_patients=50, bk_frac=0.9, method='mondrian_k',
                l=2, t=0.2, epsilon=1.0, sigma=3)
    defs.update(sw.get('defaults', {}))

    method = defs['method']
    k      = defs['k']

    n_values = next(
        (e['values'] for e in sw.get('multiseed', []) if e['param'] == 'n_patients'),
        [5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000]
    )

    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", f"rho_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    os.makedirs(out_dir, exist_ok=True)
    csv_path = f"{out_dir}/rho_results.csv"

    print(f"\nRho sweep: {len(n_values)} N values | method={method} k={k}")
    print(f"Output: {csv_path}\n")

    # Load full DB once so domains reflect full vocabulary
    full_cfg = yaml.safe_load(open(CONFIG_PATH))
    full_cfg['experiment']['real']['limit'] = 0
    full_cfg['_seed'] = 42
    print("Loading full database...")
    db_full, df_med_full, _, _ = load_attack_scenario(full_cfg)
    print(f"  {len(db_full.records)} records, {len(db_full.attributes)} attributes\n")

    rows = []
    for n in n_values:
        r = compute_rho_analytical(db_full, n_records=n)

        # Effective rho: need actual anonymization on n records
        anon_cfg = yaml.safe_load(open(CONFIG_PATH))
        anon_cfg['experiment']['real']['limit'] = n
        anon_cfg['experiment']['anonymization']['method'] = method
        anon_cfg['experiment']['anonymization']['k'] = k
        anon_cfg['_seed'] = 42
        db_n, df_med_n, _, _ = load_attack_scenario(anon_cfg)

        knowledge_map = get_anonymization_map(db_n, df_med_n, anon_cfg)
        # rho_info for n records (correct domain sizes)
        rho_n = compute_rho_analytical(db_n)
        eff_rho = compute_effective_rho(db_n, knowledge_map, rho_n)
        retention = eff_rho / r['rho'] if r['rho'] > 0 else 0.0

        row = {
            'n_patients':         n,
            'method':             method,
            'k':                  k,
            'n_attributes':       r['n_attributes'],
            'max_domain':         r['max_domain'],
            'base_vars':          r['base_vars'],
            'aux_vars':           r['aux_vars'],
            'total_vars':         r['total_vars'],
            'total_clauses_3sat': r['total_clauses_3sat'],
            'rho':                round(r['rho'], 6),
            'effective_rho':      round(eff_rho, 6),
            'retention_ratio':    round(retention, 6),
        }
        rows.append(row)
        print(f"  N={n:5d}  rho={r['rho']:.4f}  eff_rho={eff_rho:.4f}  retention={retention*100:.1f}%")

    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n{len(rows)} rows → {csv_path}")


if __name__ == '__main__':
    main()
