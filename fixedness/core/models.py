from dataclasses import dataclass, field
from typing import List, Any, Dict, Set, Tuple
import json

@dataclass
class Attribute:
    name: str
    domain: List[Any]
    id: int

@dataclass
class Constraint:
    type: str  # 'unique', 'functional_dependency', 'fd_restriction', 'check'
    attributes: List[int]  # [det_attr_id, dep_attr_id] for FDs
    params: Dict[str, Any] = field(default_factory=dict)
    # mapping: det_val_idx → set of allowed dep_val_idx (used by AC-3)
    # exact FD:   {0: {3}}          (singleton)
    # restriction: {0: {1,2,3}}     (one-to-many)
    # range FD:   {h: set_of_diag}  (threshold-based)
    mapping: Dict[int, Set[int]] = field(default_factory=dict)

@dataclass
class Record:
    id: int
    values: Dict[int, int] # attr_id -> value_index

class Database:
    def __init__(self):
        self.attributes: Dict[int, Attribute] = {}
        self.records: List[Record] = []
        self.constraints: List[Constraint] = []
        
    def add_attribute(self, name: str, domain: List[Any]):
        attr_id = len(self.attributes)
        self.attributes[attr_id] = Attribute(name, domain, attr_id)
        return attr_id
    
    def add_record(self, values: Dict[int, int]):
        record_id = len(self.records)
        self.records.append(Record(record_id, values))
        return record_id

    def add_constraint(self, constraint_type: str, attributes: List[int],
                       params=None, mapping=None):
        self.constraints.append(
            Constraint(constraint_type, attributes, params or {}, mapping or {})
        )

    def get_ground_truth(self, r_id, a_id):
        return self.records[r_id].values[a_id]
