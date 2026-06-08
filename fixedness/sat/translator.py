from typing import List, Tuple, Dict, Optional, Set
from fixedness.core.models import Database, Constraint


def compute_rho_analytical(
    db: Database,
    n_records: Optional[int] = None,
    exclude_attr_names: Optional[Set[str]] = None,
) -> dict:
    """
    Computes the 3-SAT clause-to-variable ratio rho analytically.

    Returns a dict with breakdown of clauses (amo, alon, structural)
    to allow effective_rho computation later.
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
                'total_clauses_3sat': 0, 'rho': 0.0,
                'clauses_amo_total': 0, 'clauses_structural': 0}

    domain_sizes: Dict[int, int] = {a_id: len(attr.domain) for a_id, attr in clinical}
    max_d = max(domain_sizes.values())
    attr_id_set: Set[int] = set(domain_sizes)

    base_vars = N * M * max_d
    aux_vars = 0
    
    clauses_alon = 0
    clauses_amo  = 0
    clauses_struct = 0

    # ── At-least-one (N clauses per attribute, each of length d_a) ──────────
    for a_id, _ in clinical:
        d_a = domain_sizes[a_id]
        if d_a <= 3:
            clauses_alon += N
        else:
            clauses_alon += N * (d_a - 2)
            aux_vars     += N * (d_a - 3)

    # ── At-most-one (pairwise binary) ───────────────────────────────────────
    for a_id, _ in clinical:
        d_a = domain_sizes[a_id]
        clauses_amo += N * (d_a * (d_a - 1) // 2)

    # ── Structural constraints ────────────────────────────────────────────────
    n_pairs = N * (N - 1) // 2
    for cond in db.constraints:
        if not all(a in attr_id_set for a in cond.attributes):
            continue

        if cond.type == 'unique':
            a_id = cond.attributes[0]
            d_a  = domain_sizes[a_id]
            clauses_struct += n_pairs * d_a

        elif cond.type == 'functional_dependency':
            a1, a2   = cond.attributes[0], cond.attributes[1]
            d_a1, d_a2 = domain_sizes[a1], domain_sizes[a2]
            n_orig        = n_pairs * d_a1 * d_a2 * 2
            clauses_struct += n_orig * 2
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
            clauses_struct += N * n_forbidden

    total_clauses_3sat = clauses_alon + clauses_amo + clauses_struct
    total_vars = base_vars + aux_vars
    rho = total_clauses_3sat / max(total_vars, 1)

    return {
        'n_records':        N,
        'n_attributes':     M,
        'max_domain':       max_d,
        'base_vars':        base_vars,
        'aux_vars':         aux_vars,
        'total_vars':       total_vars,
        'total_clauses_3sat': total_clauses_3sat,
        'rho':              rho,
        'clauses_amo_total': clauses_amo,
        'clauses_structural': clauses_alon + clauses_struct,
        'domain_sizes':     domain_sizes
    }


def compute_effective_rho(
    db: Database,
    knowledge_map: Dict[Tuple[int, int], Tuple[int, int]],
    rho_info: dict
) -> float:
    """
    Computes effective rho by weighting clauses based on knowledge_map activity.
    
    w(r,a) = 1 - (v_max - v_min) / (D_a - 1)
    effective_rho = (sum(w(r,a) * C(d_a, 2)) + clauses_fd + clauses_bk) / total_vars
    """
    total_vars = rho_info['total_vars']
    if total_vars == 0:
        return 0.0
        
    domain_sizes = rho_info['domain_sizes']
    
    clausole_amo_effettive = 0.0
    clausole_bk = 0
    
    # We iterate over all records and attributes that were included in rho_info
    # (clinical attributes, excluding Identity_Link)
    attr_ids = set(domain_sizes.keys())
    n_records = rho_info['n_records']
    
    for r in range(n_records):
        for a_id in attr_ids:
            d_a = domain_sizes[a_id]
            v_min, v_max = knowledge_map.get((r, a_id), (0, d_a - 1))
            
            # 1. Weight w(r,a)
            if d_a > 1:
                w = 1.0 - (v_max - v_min) / (d_a - 1)
            else:
                w = 1.0 # fixed by domain
                
            # 2. Weighted AMO
            c_amo_cell = d_a * (d_a - 1) // 2
            clausole_amo_effettive += c_amo_cell * w
            
            # 3. BK clauses (unit clauses for exact knowledge)
            if v_min == v_max:
                clausole_bk += 1

    effective_rho = (clausole_amo_effettive + rho_info['clauses_structural'] + clausole_bk) / total_vars
    return effective_rho


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
