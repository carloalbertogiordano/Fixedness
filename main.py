import sys
import os
import yaml
import multiprocessing
from datetime import datetime
from tqdm import tqdm

from fixedness.core.loader import load_attack_scenario
from fixedness.anonymizers import get_anonymization_map
import fixedness.audit.worker as _wmod
from fixedness.audit.worker import smt_linkage_worker
from fixedness.audit.candidate_filter import OracleIndex, PartitionCache
from fixedness.sat.translator import SATTranslator, compute_rho_analytical


def run_deanon_audit():
    print("--- Identity De-anonymization Audit (Linkage Attack) ---")

    _here = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(_here, 'config.yaml'), 'r') as f:
        config = yaml.safe_load(f)

    n_cores = config['system']['max_cores']
    if n_cores <= 0:
        try:
            with open('/proc/meminfo') as _mf:
                avail_kb = next(int(l.split()[1]) for l in _mf if l.startswith('MemAvailable'))
            avail_gb = avail_kb / (1024 ** 2)
            ram_per_core = max(1.0, config['system'].get('ram_per_core_gb', 2.0))
            n_cores = min(max(1, multiprocessing.cpu_count() - 1),
                          max(1, int(avail_gb / ram_per_core)))
        except Exception:
            n_cores = min(4, max(1, multiprocessing.cpu_count() - 1))
    else:
        n_cores = max(1, n_cores)

    # 1. Caricamento scenario
    db, df_med, df_ora, col_mapping = load_attack_scenario(config)

    # 2. Phase transition metric (analytical — no clause list generated, safe for any N)
    rho_info = compute_rho_analytical(db)
    print(
        f"[SAT] 3-SAT (analytical): N={rho_info['n_records']}, "
        f"M={rho_info['n_attributes']}, max_domain={rho_info['max_domain']}, "
        f"{rho_info['total_clauses_3sat']} clauses, "
        f"{rho_info['base_vars']} base + {rho_info['aux_vars']} aux = "
        f"{rho_info['total_vars']} vars, rho={rho_info['rho']:.4f}"
    )

    # 3. Anonimizzazione
    print(f"Applying Mondrian (k={config['experiment']['anonymization']['k']})...")
    knowledge_map = get_anonymization_map(db, df_med, config)

    # 4. Setta globals nel padre — figli ereditano via fork COW, zero pickle
    _wmod._db            = db
    _wmod._knowledge_map = knowledge_map
    _wmod._col_mapping   = col_mapping
    _wmod._oracle_index  = OracleIndex(df_ora, db, col_mapping)
    _wmod._part_cache    = PartitionCache()

    res_dir  = os.path.join(_here, 'results', datetime.now().strftime('%Y%m%d_%H%M%S'))
    os.makedirs(res_dir, exist_ok=True)
    csv_path = os.path.join(res_dir, 'full_audit.csv')

    timeout_ms   = config['system']['sat_timeout_sec'] * 1000
    mem_limit_gb = config['system']['ram_per_core_gb']

    id_attr_id = col_mapping['Identity_Link']
    tasks = [
        (r, db.get_ground_truth(r, id_attr_id), timeout_ms, mem_limit_gb)
        for r in range(len(db.records))
    ]

    print(f"Auditing {len(tasks)} identities ({n_cores} workers, oracle={len(df_ora)} rows)...")

    with open(csv_path, 'w') as f:
        f.write("record,identity,fixedness,sponginess,candidates,status,solve_ms,promotion_source\n")

    # Pool senza initializer — fork eredita i globals dal padre
    with multiprocessing.Pool(processes=n_cores) as pool:
        with tqdm(total=len(tasks), desc="Linkage Audit") as pbar:
            for r_id, res in pool.imap_unordered(smt_linkage_worker, tasks, chunksize=1):
                real_identity = db.attributes[id_attr_id].domain[
                    db.get_ground_truth(r_id, id_attr_id)
                ]
                row = (f"{r_id},{real_identity},"
                       f"{res['fixedness']},{res['sponginess']},"
                       f"{res['candidates']},{res['status']},{res['solve_ms']},"
                       f"{res['promotion_source']}\n")
                with open(csv_path, 'a') as f:
                    f.write(row)
                    f.flush()
                pbar.update(1)

    print(f"\nAudit completed. Results: {csv_path}")


if __name__ == "__main__":
    run_deanon_audit()
