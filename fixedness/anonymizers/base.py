from abc import ABC, abstractmethod
from typing import Dict, List, Tuple
import pandas as pd
from fixedness.core.models import Database

KnowledgeMap = Dict[Tuple[int, int], Tuple[int, int]]


class Anonymizer(ABC):
    @abstractmethod
    def build_knowledge_map(self, db: Database, df_med: pd.DataFrame,
                            config: dict) -> KnowledgeMap:
        ...

    # ── shared helpers ────────────────────────────────────────────────

    def _attr_name_to_id(self, db: Database) -> Dict[str, int]:
        return {attr.name: aid for aid, attr in db.attributes.items()}

    def _qi_names(self, config: dict) -> List[str]:
        return config['experiment']['anonymization']['quasi_identifiers']

    def _sa_names(self, config: dict) -> List[str]:
        return config['experiment']['anonymization'].get('sensitive_attributes', [])

    def _full_range_map(self, db: Database) -> KnowledgeMap:
        km = {}
        for r in range(len(db.records)):
            for aid, attr in db.attributes.items():
                km[(r, aid)] = (0, len(attr.domain) - 1)
        return km

    def _set_exact(self, km: KnowledgeMap, db: Database,
                   attr_ids: List[int], records=None) -> None:
        rng = range(len(db.records)) if records is None else records
        for r in rng:
            for aid in attr_ids:
                v = db.records[r].values.get(aid)
                if v is not None:
                    km[(r, aid)] = (v, v)

    def _set_partition_range(self, km: KnowledgeMap, db: Database,
                             partition: List[int], attr_ids: List[int]) -> None:
        for aid in attr_ids:
            vals = [db.records[i].values[aid] for i in partition
                    if aid in db.records[i].values]
            if not vals:
                continue
            rng = (min(vals), max(vals))
            for r in partition:
                km[(r, aid)] = rng


def apply_mondrian(df: pd.DataFrame, k: int,
                   qi_cols: List[str]) -> List[List[int]]:
    """Core Mondrian recursive median split. Returns list of index-lists."""
    partitions: List[List[int]] = []

    def spans(indices):
        return {col: len(df.iloc[indices][col].unique()) for col in qi_cols}

    def split(indices):
        if len(indices) < 2 * k:
            partitions.append(indices)
            return
        sp       = spans(indices)
        best_col = max(sp, key=sp.get)
        if sp[best_col] == 1:          # all same value — can't split
            partitions.append(indices)
            return
        ordered = df.iloc[indices][best_col].sort_values().index.tolist()
        mid     = len(ordered) // 2
        lhs, rhs = ordered[:mid], ordered[mid:]
        if len(lhs) >= k and len(rhs) >= k:
            split(lhs)
            split(rhs)
        else:
            partitions.append(indices)

    split(list(range(len(df))))
    return partitions
