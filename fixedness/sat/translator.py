from typing import List, Tuple, Dict, Optional, Set
from fixedness.core.models import Database, Constraint


def compute_rho_analytical(
    db: Database,
    n_records: Optional[int] = None,
    exclude_attr_names: Optional[Set[str]] = None,
) -> dict:
    """
    Computes the 3-SAT clause-to-variable ratio rho analytically.

    Mirrors the sparse encoding used by SATTranslator (base_vars = N*M*max_domain)
    and the linear-chain Tseitin reduction, but counts clauses and variables
    without generating any list — safe for any N or domain size.

    Returns a dict with n_records, n_attributes, max_domain, base_vars, aux_vars,
    total_vars, total_clauses_3sat, rho.
    """
    if exclude_attr_names is None:
        exclude_attr_names = {'Identity_Link'}

    N = n_records if n_records is not None else len(db.records)
    N = max(1, N)

    clinical = [(a_id, attr) for a_id, attr in db.attributes.items()
                if attr.name not in exclude_attr_names]
    M = len(clinical)
    if M == 0:
        return {'n_records': N, 'n_attributes': 0, 'max_domain': 0,
                'base_vars': 0, 'aux_vars': 0, 'total_vars': 0,
                'total_clauses_3sat': 0, 'rho': 0.0}

    domain_sizes: Dict[int, int] = {a_id: len(attr.domain) for a_id, attr in clinical}
    max_d = max(domain_sizes.values())
    attr_id_set: Set[int] = set(domain_sizes)

    base_vars = N * M * max_d
    aux_vars = 0
    clauses_3sat = 0

    # ── At-least-one (N clauses per attribute, each of length d_a) ──────────
    for a_id, _ in clinical:
        d_a = domain_sizes[a_id]
        if d_a <= 3:
            clauses_3sat += N             # already ternary or shorter
        else:
            clauses_3sat += N * (d_a - 2)  # linear chain: d_a-2 ternary clauses
            aux_vars     += N * (d_a - 3)  #               d_a-3 aux vars

    # ── At-most-one (pairwise binary, already ≤3 literals, no expansion) ────
    for a_id, _ in clinical:
        d_a = domain_sizes[a_id]
        clauses_3sat += N * (d_a * (d_a - 1) // 2)

    # ── Structural constraints ────────────────────────────────────────────────
    n_pairs = N * (N - 1) // 2
    for cond in db.constraints:
        if not all(a in attr_id_set for a in cond.attributes):
            continue

        if cond.type == 'unique':
            a_id = cond.attributes[0]
            d_a  = domain_sizes[a_id]
            # binary clauses per pair per value — stay in 3-SAT
            clauses_3sat += n_pairs * d_a

        elif cond.type == 'functional_dependency':
            a1, a2   = cond.attributes[0], cond.attributes[1]
            d_a1, d_a2 = domain_sizes[a1], domain_sizes[a2]
            # 2 original 4-literal clauses per (r1,r2,v,k) triple
            n_orig        = n_pairs * d_a1 * d_a2 * 2
            # each 4-literal → 2 ternary + 1 aux (linear chain)
            clauses_3sat += n_orig * 2
            aux_vars      += n_orig

        elif cond.type == 'check':
            a_id       = cond.attributes[0]
            d_a        = domain_sizes[a_id]
            op         = cond.params.get('op')
            thresh     = cond.params.get('val', 0)
            n_forbidden = sum(
                1 for v in range(d_a)
                if (op == '<'  and not v < thresh) or
                   (op == '>'  and not v > thresh) or
                   (op == '==' and v != thresh)
            )
            # unit clauses, stay as-is
            clauses_3sat += N * n_forbidden

    total_vars = base_vars + aux_vars
    rho = clauses_3sat / max(total_vars, 1)

    return {
        'n_records':        N,
        'n_attributes':     M,
        'max_domain':       max_d,
        'base_vars':        base_vars,
        'aux_vars':         aux_vars,
        'total_vars':       total_vars,
        'total_clauses_3sat': clauses_3sat,
        'rho':              rho,
    }

class SATTranslator:
    def __init__(self, db: Database):
        self.db = db
        self.num_records = len(db.records)
        self.num_attributes = len(db.attributes)
        self.domain_sizes = {a_id: len(attr.domain) for a_id, attr in db.attributes.items()}
        self.max_domain = max(self.domain_sizes.values())
        
        # Base variables: (r, a, v)
        self.base_vars_limit = self.num_records * self.num_attributes * self.max_domain
        self.next_aux_var = self.base_vars_limit + 1

    def get_var(self, r_id: int, a_id: int, v_idx: int) -> int:
        return (r_id * self.num_attributes * self.max_domain) + (a_id * self.max_domain) + v_idx + 1

    def generate_clauses(self) -> List[List[int]]:
        clauses = []
        
        # 1. Domain: Each cell must have exactly one value
        for r in range(self.num_records):
            for a in range(self.num_attributes):
                d_size = self.domain_sizes[a]
                # At least one
                clauses.append([self.get_var(r, a, v) for v in range(d_size)])
                # At most one
                for v1 in range(d_size):
                    for v2 in range(v1 + 1, d_size):
                        clauses.append([-self.get_var(r, a, v1), -self.get_var(r, a, v2)])

        # 2. Constraints
        for cond in self.db.constraints:
            if cond.type == 'unique':
                a_id = cond.attributes[0]
                for v in range(self.domain_sizes[a_id]):
                    for r1 in range(self.num_records):
                        for r2 in range(r1 + 1, self.num_records):
                            clauses.append([-self.get_var(r1, a_id, v), -self.get_var(r2, a_id, v)])
            
            elif cond.type == 'functional_dependency':
                a1, a2 = cond.attributes
                for r1 in range(self.num_records):
                    for r2 in range(r1 + 1, self.num_records):
                        for v in range(self.domain_sizes[a1]):
                            for k in range(self.domain_sizes[a2]):
                                clauses.append([-self.get_var(r1, a1, v), -self.get_var(r2, a1, v), -self.get_var(r1, a2, k), self.get_var(r2, a2, k)])
                                clauses.append([-self.get_var(r1, a1, v), -self.get_var(r2, a1, v), self.get_var(r1, a2, k), -self.get_var(r2, a2, k)])
            
            elif cond.type == 'check':
                a_id = cond.attributes[0]
                op = cond.params.get('op')
                threshold_idx = cond.params.get('val') # Assume val is index in domain
                
                for r in range(self.num_records):
                    for v in range(self.domain_sizes[a_id]):
                        # Map logical op to forbidden value indices
                        forbidden = False
                        if op == '<':
                            if not (v < threshold_idx): forbidden = True
                        elif op == '>':
                            if not (v > threshold_idx): forbidden = True
                        elif op == '==':
                            if not (v == threshold_idx): forbidden = True
                            
                        if forbidden:
                            # If value violates check, it cannot be True
                            clauses.append([-self.get_var(r, a_id, v)])
        return clauses

    def get_base_vars(self) -> List[int]:
        """Returns all base variable IDs (r, a, v)."""
        base_vars = []
        for r in range(self.num_records):
            for a in range(self.num_attributes):
                for v in range(self.domain_sizes[a]):
                    base_vars.append(self.get_var(r, a, v))
        return base_vars

    def to_3sat(self, clauses: List[List[int]]) -> List[List[int]]:
        three_sat = []
        aux = self.next_aux_var
        for c in clauses:
            if len(c) <= 3:
                three_sat.append(c)
            else:
                l = len(c)
                a_vars = list(range(aux, aux + l - 3))
                aux += (l - 3)
                three_sat.append([c[0], c[1], a_vars[0]])
                for i in range(l - 4):
                    three_sat.append([-a_vars[i], c[i+2], a_vars[i+1]])
                three_sat.append([-a_vars[-1], c[-2], c[-1]])
        self.next_aux_var = aux
        return three_sat
