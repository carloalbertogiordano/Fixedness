# Fixedness — Privacy Audit Framework

A metric-based framework for quantifying re-identification risk in anonymized databases. Given a medical database and a public oracle, it simulates a linkage attack using SMT solving (Z3) and AC-3 constraint propagation, then reports two per-record metrics:

- **Fixedness** — probability that a record maps to a unique identity in the oracle (0 = safe, 1 = deterministic leak)
- **Sponginess** — fraction of oracle candidates eliminated by the anonymized data (higher = less ambiguity for the attacker)

---

## Repository structure

```
fixedness/
├── core/
│   ├── loader.py          # DB loader, attack scenario setup
│   └── models.py          # Core data structures (Database, Attribute, Record)
├── sat/
│   └── translator.py      # LIA encoding for Z3, analytical rho computation
├── audit/
│   ├── worker.py          # Per-record SMT+AC-3 worker (multiprocessing)
│   └── candidate_filter.py # OracleIndex, PartitionCache, AC-3 pruning
└── anonymizers/
    ├── base.py            # Abstract anonymizer
    ├── factory.py         # Method dispatch
    ├── syntactic.py       # Mondrian k-anonymity, l-diversity, t-closeness
    ├── formal.py          # Laplace DP, local DP
    └── perturbative.py    # Noise addition, microaggregation, randomized response
main.py                    # Entry point — full linkage audit
config.yaml                # All parameters (method, k, QI set, sweep config)
scripts/
└── build_rich_db.py       # Regenerate the 8-table synthetic DB + CSVs from scratch
database/                  # Pre-generated CSVs (synthetic, SEED=42)
```

---

## Threat model

The attacker holds:
- An **anonymized** version of the medical database (patients table with generalized QI values)
- A **public oracle** (persons table + insurance, consumer, social side-tables)
- Partial **background knowledge**: a known sensitive attribute value for a fraction of records (`background_knowledge_frac`)

The framework encodes the anonymization constraints as a Linear Integer Arithmetic (LIA) formula and queries Z3. Per-record candidate sets are then pruned with AC-3 on the oracle. The result is a fixedness/sponginess score for each record.

---

## Quick start

### Requirements

```
Python 3.11+
z3-solver
pandas
numpy
pyyaml
tqdm
```

Install into a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
pip install z3-solver pandas numpy pyyaml tqdm
```

### Run the audit

```bash
python main.py
```

Results are written to `results/<timestamp>/full_audit.csv` with columns:
`record, identity, fixedness, sponginess, candidates, status, solve_ms, promotion_source`

`promotion_source` indicates which phase determined fixedness=1: `smt` (Z3 UNSAT proof), `ac3` (single oracle candidate), `both`, or `none`.

### Configuration

Edit `config.yaml`:

| Key | Description |
|-----|-------------|
| `experiment.anonymization.method` | `mondrian_k`, `l_diversity`, `t_closeness`, `laplace_dp`, `local_dp`, `noise_addition`, `microaggregation`, `randomized_response`, `suppression`, `no_anonymization` |
| `experiment.anonymization.k` | k-anonymity parameter |
| `experiment.anonymization.epsilon` | DP privacy budget |
| `experiment.anonymization.quasi_identifiers` | QI columns exposed to the attacker |
| `experiment.anonymization.sensitive_attributes` | SA columns being protected |
| `experiment.anonymization.background_knowledge_frac` | Fraction of records for which attacker knows `bk_sa` |
| `experiment.real.limit` | Max records to audit (0 = all) |
| `system.max_cores` | Worker processes (-1 auto from available RAM) |
| `system.sat_timeout_sec` | Z3 timeout per record (0 = unlimited) |

---

## Synthetic database

All data in `database/` is **fully synthetic** — no real individuals are represented.

- Names, tax codes, addresses, phone numbers, and emails are generated algorithmically from fixed lookup tables using `numpy`/`random` with SEED=42. While the generated codice fiscale strings follow the Italian format, they are not validated against any real registry and carry no information about real people.
- Consumer spending scores are sampled from the [Mall Customer Segmentation dataset](https://www.kaggle.com/datasets/vjchoudhary7/customer-segmentation-tutorial-in-python) (200 rows, public domain, no PII — only age, income, and spending score). No rows from that dataset appear in the output CSVs; it is used solely to parameterize the spending-score distribution.
- Medical, clinical, and social data are generated from epidemiological models and have no relation to any real patient.

The pre-generated CSVs encode a realistic medical scenario (SEED=42):

| Table | Role | Rows |
|-------|------|------|
| `patients.csv` | Medical DB — target of the attack | 5 000 |
| `persons.csv` | Oracle — public demographic registry | 7 500 (5 000 matched + 2 500 noise) |
| `medical_visits.csv` | Visit records (FK → patients) | variable |
| `prescriptions.csv` | Drug prescriptions (FK → patients) | variable |
| `lab_results.csv` | Lab tests (FK → patients) | variable |
| `insurance_policies.csv` | Insurance (FK → persons) | variable |
| `consumer_profiles.csv` | Consumer data (FK → persons) | variable |
| `social_links.csv` | Social graph — Facebook + Twitter (FK → persons) | variable |

Deterministic functional dependencies are embedded in the schema (e.g., `diagnosis → treatment`, `zip → city`, `policy_type → coverage_level`) to make the SMT encoding non-trivial.

To regenerate the DB from scratch:

```bash
python scripts/build_rich_db.py
```

---

## Metrics — formal definition

Let *R* be an anonymized record, *O* the oracle, *C(R)* the set of oracle candidates consistent with *R*.

**Fixedness**:
$$\phi(R) = \begin{cases} 1 & \text{if } |C(R)| = 1 \\ 0 & \text{if } |C(R)| = 0 \text{ (SAT conflict)} \\ 0 & \text{if UNSAT} \end{cases}$$

**Sponginess**:
$$\sigma(R) = 1 - \frac{|C(R)|}{|O|}$$

High sponginess + fixedness = 1 → deterministic re-identification.  
Low sponginess (many candidates) → attacker is lost in the oracle noise.

---

## Attack scenarios

Scenarios are declared in `fixedness/scenarios/` as YAML files. Each scenario specifies the target and oracle tables, the join key (hidden ground truth), and the functional dependencies registered with the solver. To switch scenario, set `experiment.scenario` in `config.yaml`.

| Scenario | File | FDs registered | Purpose |
|----------|------|---------------|---------|
| Full threat model | `rich_medical.yaml` | 11 (all known FDs) | Standard audit |
| Schema incompleteness | `rich_medical_blind_fd.yaml` | 7 (HbA1c↔Diagnosis and Pressione↔Diagnosis omitted) | Adversarial: solver blind to clinical threshold FDs that a domain-expert attacker can exploit; compare fixedness scores against full scenario to quantify the underestimation gap |

---

## Complexity / phase transition

The SAT encoding uses a 3-SAT reduction to estimate the phase-transition parameter ρ = M/N (clauses per variable). The analytical rho is printed at startup and measures how far the instance is from the satisfiability threshold (~4.267 for random 3-SAT).
