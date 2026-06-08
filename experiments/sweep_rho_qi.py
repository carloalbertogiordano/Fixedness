"""
sweep_rho_qi.py — Phase transition rho per ogni QI set × N (analitico).
Config: sweep.qi_sets, sweep.multiseed[n_patients].values
Output: tests/fixedness_test/results/rho_qi_YYYYMMDD_HHMMSS/rho_qi_results.csv
Esegui da project root: python experiments/sweep_rho_qi.py
"""
import sys, os, csv, yaml, copy
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from fixedness.core.loader import load_attack_scenario
from fixedness.sat.translator import compute_rho_analytical

CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'config.yaml')

FIELDS = ['qi_name', 'n_qi', 'qi_list',
          'n_patients', 'n_attributes', 'max_domain',
          'base_vars', 'aux_vars', 'total_vars', 'total_clauses_3sat', 'rho']


def main():
    with open(CONFIG_PATH) as f:
        base_cfg = yaml.safe_load(f)

    sw      = base_cfg.get('sweep', {})
    qi_sets = sw.get('qi_sets', [])
    if not qi_sets:
        print("ERROR: nessun qi_sets in config.yaml sweep.qi_sets")
        return

    n_values = next(
        (e['values'] for e in sw.get('multiseed', []) if e['param'] == 'n_patients'),
        [5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000]
    )

    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", f"rho_qi_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    os.makedirs(out_dir, exist_ok=True)
    csv_path = f"{out_dir}/rho_qi_results.csv"

    print(f"\nRho×QI sweep: {len(qi_sets)} QI sets × {len(n_values)} N values")
    print(f"Output: {csv_path}\n")

    rows = []

    for qi_entry in qi_sets:
        qi_name = qi_entry['name']
        qi_list = qi_entry['qi']

        # Build config for this QI set, full N to get complete domains
        cfg = copy.deepcopy(base_cfg)
        cfg['experiment']['real']['limit'] = 0
        cfg['experiment']['anonymization']['quasi_identifiers'] = qi_list
        cfg['_seed'] = 42

        print(f"Loading DB for QI set '{qi_name}' {qi_list} ...")
        db, _, _, _ = load_attack_scenario(cfg)

        attr_info = {a.name: len(a.domain)
                     for a_id, a in db.attributes.items()
                     if a.name != 'Identity_Link'}
        print(f"  M={len(attr_info)} clinical attrs: " +
              ", ".join(f"{n}({d})" for n, d in attr_info.items()))

        for n in n_values:
            r = compute_rho_analytical(db, n_records=n)
            rows.append({
                'qi_name':            qi_name,
                'n_qi':               len(qi_list),
                'qi_list':            '+'.join(qi_list),
                'n_patients':         n,
                'n_attributes':       r['n_attributes'],
                'max_domain':         r['max_domain'],
                'base_vars':          r['base_vars'],
                'aux_vars':           r['aux_vars'],
                'total_vars':         r['total_vars'],
                'total_clauses_3sat': r['total_clauses_3sat'],
                'rho':                round(r['rho'], 6),
            })

        # Summary at N=50 (benchmark default)
        r50 = next(row for row in rows
                   if row['qi_name'] == qi_name and row['n_patients'] == 50)
        print(f"  N=50  rho={r50['rho']:.4f}  M={r50['n_attributes']}  "
              f"max_domain={r50['max_domain']}\n")

    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    # Console summary table
    print(f"{'QI set':<22} {'N':>6} {'M':>4} {'D_max':>6} {'rho':>8}")
    print("-" * 52)
    for row in rows:
        if row['n_patients'] in (5, 20, 50, 200, 5000):
            print(f"{row['qi_name']:<22} {row['n_patients']:>6} "
                  f"{row['n_attributes']:>4} {row['max_domain']:>6} {row['rho']:>8.4f}")

    print(f"\n{len(rows)} rows → {csv_path}")


if __name__ == '__main__':
    main()
