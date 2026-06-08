"""
Parallel execution engine for fixedness sweep experiments.

Each job = one (config, seed) → all records for that run.
Workers are separate processes (Z3 process-local, no shared state).
Log messages route through a queue to avoid stdout interleaving.

Config keys used from base_cfg:
  system.max_cores       — hard cap on parallel workers
  system.ram_per_core_gb — if > 0, caps workers by available RAM / this value
"""
import sys, os, copy, time, statistics, random, threading
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
from collections import defaultdict

_FTEST = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'tests', 'fixedness_test')
sys.path.insert(0, _FTEST)

METRIC_NAMES = ['fixedness', 'sponginess', 'candidates', 'conflicts',
                'decisions', 'propagations', 'solve_ms']

# ── Worker-process globals ────────────────────────────────────────────────────

_log_q = None

def _init_worker(q):
    global _log_q
    _log_q = q

def _wlog(msg):
    (_log_q.put if _log_q is not None else print)(msg)


# ── Job function (runs in subprocess) ────────────────────────────────────────

def _job_worker(args):
    label, params, seed, base_cfg, sat_timeout = args

    sys.path.insert(0, _FTEST)
    from fixedness.core.loader import load_attack_scenario
    from fixedness.anonymizers import get_anonymization_map
    import fixedness.audit.worker as _wmod
    from fixedness.audit.worker import smt_linkage_worker
    from fixedness.audit.candidate_filter import OracleIndex, PartitionCache
    from fixedness.sat.translator import compute_rho_analytical, compute_effective_rho

    cfg  = copy.deepcopy(base_cfg)
    anon = cfg['experiment']['anonymization']
    cfg['experiment']['real']['limit'] = params['n_patients']
    anon['k']                          = params['k']
    anon['method']                     = params['method']
    anon['l']                          = params['l']
    anon['t']                          = params['t']
    anon['epsilon']                    = params['epsilon']
    anon['sigma']                      = params['sigma']
    anon['background_knowledge_frac']  = params['bk_frac']
    if 'bk_sa'   in params: anon['bk_sa']             = params['bk_sa']
    if 'qi_list' in params: anon['quasi_identifiers'] = params['qi_list']
    cfg['system']['sat_timeout_sec'] = sat_timeout
    cfg['system']['ram_per_core_gb'] = 0
    cfg['_seed'] = seed

    db, df_med, df_ora, col_mapping = load_attack_scenario(cfg)
    id_attr_id = col_mapping['Identity_Link']
    n_rec      = len(db.records)
    true_idxs  = {db.get_ground_truth(r, id_attr_id) for r in range(n_rec)}

    noise  = min(params['oracle_noise'], len(df_ora) - len(true_idxs))
    rng    = random.Random(seed)
    others = [i for i in range(len(df_ora)) if i not in true_idxs]
    extra  = rng.sample(others, min(noise, len(others)))
    keep   = sorted(true_idxs) + sorted(extra)
    remap  = {old: new for new, old in enumerate(keep)}
    df_small = df_ora.iloc[keep].reset_index(drop=True)

    rho_info = compute_rho_analytical(db)
    knowledge_map = get_anonymization_map(db, df_med, cfg)
    eff_rho = compute_effective_rho(db, knowledge_map, rho_info)
    timeout_ms    = sat_timeout * 1000

    _wmod._db            = db
    _wmod._knowledge_map = knowledge_map
    _wmod._col_mapping   = col_mapping
    _wmod._base_solver   = None
    _wmod._oracle_index  = OracleIndex(df_small, db, col_mapping)
    _wmod._part_cache    = PartitionCache()

    t_build = time.perf_counter()
    _wmod._build_base_solver()
    build_ms = round((time.perf_counter() - t_build) * 1000)
    _wlog(f"  [{label}|s={seed}] oracle={len(df_small)} build={build_ms}ms {n_rec}rec ...")

    t0      = time.perf_counter()
    metrics = {m: [] for m in METRIC_NAMES}
    for r in range(n_rec):
        orig = db.get_ground_truth(r, id_attr_id)
        _, res = smt_linkage_worker((r, remap[orig], timeout_ms, 0))
        for m in METRIC_NAMES:
            metrics[m].append(res.get(m, 0))

    rec_ms = round((time.perf_counter() - t0) * 1000)
    fix_n  = sum(1 for f in metrics['fixedness'] if f == 1.0)
    mean_s = sum(metrics['sponginess']) / n_rec if n_rec else 0
    _wlog(f"  [{label}|s={seed}] {rec_ms}ms fix={fix_n}/{n_rec} spon={mean_s:.3f}")

    st = sorted(metrics['solve_ms'])
    return label, seed, {
        'n_rec':      n_rec,
        'oracle_sz':  len(df_small),
        'build_ms':   build_ms,
        'median_ms':  round(statistics.median(st), 2)                            if st else 0,
        'p95_ms':     round(st[max(0, int(0.95 * len(st)) - 1)], 2)             if st else 0,
        'max_ms':     round(max(st), 2)                                          if st else 0,
        'metrics':    metrics,
        'rho':        rho_info['rho'],
        'eff_rho':    eff_rho,
    }


# ── Aggregation ───────────────────────────────────────────────────────────────

def aggregate(seed_results):
    """Aggregate metrics across seeds. seed_results: list of per-seed result dicts."""
    n_rec     = seed_results[0]['n_rec']
    oracle_sz = seed_results[0]['oracle_sz']
    agg       = {
        'n_records': n_rec, 
        'oracle_size': oracle_sz,
        'rho': round(statistics.mean(r['rho'] for r in seed_results), 4),
        'effective_rho': round(statistics.mean(r['eff_rho'] for r in seed_results), 4)
    }

    for m in METRIC_NAMES:
        vals = [sum(r['metrics'][m]) / n_rec for r in seed_results]
        agg[f'mean_{m}'] = round(statistics.mean(vals), 4)
        agg[f'std_{m}']  = round(statistics.stdev(vals) if len(vals) > 1 else 0.0, 4)

    agg['n_fixed_mean'] = round(statistics.mean(
        [sum(1 for f in r['metrics']['fixedness'] if f == 1.0) / n_rec
         for r in seed_results]), 4)

    for key in ('build_ms', 'median_ms', 'p95_ms', 'max_ms'):
        agg[f'mean_{key}'] = round(statistics.mean(r[key] for r in seed_results), 2)

    return agg


# ── Worker count ──────────────────────────────────────────────────────────────

def compute_n_workers(cfg):
    max_cores = cfg.get('system', {}).get('max_cores', 4)
    ram_gb    = cfg.get('system', {}).get('ram_per_core_gb', 0)
    if ram_gb > 0:
        try:
            import psutil
            avail  = psutil.virtual_memory().available / (1024 ** 3)
            by_ram = max(1, int(avail / ram_gb))
        except ImportError:
            by_ram = max_cores
    else:
        by_ram = max_cores
    return max(1, min(max_cores, by_ram))


# ── Main parallel runner ──────────────────────────────────────────────────────

def run_parallel(configs, seeds, base_cfg, sat_timeout, n_workers, on_config_done=None):
    """
    configs:         list of (label, params_dict)
    seeds:           list of int
    on_config_done:  callable(label, agg) — called when all seeds for a config finish
    Returns:         {label: agg_dict}  — absent if all seeds for that label errored
    """
    n_jobs = len(configs) * len(seeds)
    print(f"  {len(configs)} configs × {len(seeds)} seeds = {n_jobs} jobs | workers={n_workers}")

    log_q = multiprocessing.Queue()

    def _writer():
        while True:
            msg = log_q.get()
            if msg is None:
                break
            print(msg, flush=True)

    writer = threading.Thread(target=_writer, daemon=True)
    writer.start()

    pending = defaultdict(list)
    results = {}
    n_done  = 0
    t0_all  = time.time()

    with ProcessPoolExecutor(max_workers=n_workers,
                             initializer=_init_worker,
                             initargs=(log_q,)) as exe:
        futures = {
            exe.submit(_job_worker, (label, params, seed, base_cfg, sat_timeout)): (label, seed)
            for label, params in configs
            for seed in seeds
        }

        for f in as_completed(futures):
            label, seed = futures[f]
            try:
                _, _, res = f.result()
                pending[label].append(res)
            except Exception as e:
                log_q.put(f"  ✗ [{label}|s={seed}]: {e}")
                pending[label].append(None)

            n_done += 1
            ela   = time.time() - t0_all
            eta   = ela / n_done * (n_jobs - n_done) if n_done < n_jobs else 0
            eta_s = f"{int(eta//60)}m{int(eta%60):02d}s" if eta >= 60 else f"{eta:.1f}s"
            log_q.put(f"  [{n_done}/{n_jobs} | {ela:.0f}s | ETA {eta_s}]")

            if len(pending[label]) == len(seeds):
                good = [r for r in pending[label] if r is not None]
                if good:
                    agg = aggregate(good)
                    results[label] = agg
                    if on_config_done:
                        on_config_done(label, agg)
                    log_q.put(
                        f"  ✓ {label}  fix={agg['mean_fixedness']:.3f}±{agg['std_fixedness']:.3f}"
                        f"  spon={agg['mean_sponginess']:.3f}  Z3={agg['mean_solve_ms']:.2f}ms")

    log_q.put(None)
    writer.join()
    return results
