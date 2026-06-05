from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional
import os
import yaml


@dataclass
class FDSpec:
    det: str
    dep: str
    type: str                             # exact | restriction | numeric_to_categorical | categorical_to_numeric
    threshold: Optional[float] = None     # numeric_to_categorical / categorical_to_numeric
    high_vals: List[str] = field(default_factory=list)      # dep values when det >= threshold
    high_det_vals: List[str] = field(default_factory=list)  # det values that imply dep >= threshold
    map: Dict[str, List[str]] = field(default_factory=dict) # restriction: det_val → [dep_vals]


@dataclass
class ScenarioConfig:
    target_table: str
    oracle_table: str
    join_key: str          # FK in target that links to oracle (hidden ground truth)
    identity_col: str      # column in oracle used as the published identity label
    target_col_map: Dict[str, str]          # DB column → logical name (target table)
    oracle_col_map: Dict[str, str]          # DB column → logical name (oracle table)
    functional_dependencies: List[FDSpec]


def load_scenario(path: str) -> ScenarioConfig:
    with open(path, 'r') as f:
        raw = yaml.safe_load(f)

    fds = []
    for fd_raw in raw.get('functional_dependencies', []):
        fds.append(FDSpec(
            det=fd_raw['det'],
            dep=fd_raw['dep'],
            type=fd_raw['type'],
            threshold=fd_raw.get('threshold'),
            high_vals=fd_raw.get('high_vals', []),
            high_det_vals=fd_raw.get('high_det_vals', []),
            map=fd_raw.get('map', {}),
        ))

    return ScenarioConfig(
        target_table=raw['target_table'],
        oracle_table=raw['oracle_table'],
        join_key=raw['join_key'],
        identity_col=raw['identity_col'],
        target_col_map=raw.get('target_col_map', {}),
        oracle_col_map=raw.get('oracle_col_map', {}),
        functional_dependencies=fds,
    )


def scenario_path(name: str) -> str:
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base, 'scenarios', f'{name}.yaml')
