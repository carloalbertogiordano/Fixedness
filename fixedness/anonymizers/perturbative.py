import numpy as np
from .base import Anonymizer, KnowledgeMap
from fixedness.core.models import Database
import pandas as pd


class NoiseAddition(Anonymizer):
    """
    Aggiunge rumore gaussiano agli indici di dominio dei QI.
    Range pubblicato: (v_noisy ± 2σ) clampato al dominio.
    SA: valore esatto.
    """
    def build_knowledge_map(self, db, df_med, config):
        sigma  = config['experiment']['anonymization'].get('sigma', 3)
        qi     = self._qi_names(config)
        sa     = self._sa_names(config)
        a2id   = self._attr_name_to_id(db)
        n_rec  = len(db.records)
        km     = self._full_range_map(db)
        rng    = np.random.RandomState(config.get('_seed', 42))
        delta  = max(1, int(round(2 * sigma)))

        qi_ids = [a2id[q] for q in qi if q in a2id]
        sa_ids = [a2id[s] for s in sa if s in a2id]

        for r in range(n_rec):
            for aid in qi_ids:
                v      = db.records[r].values.get(aid, 0)
                d_max  = len(db.attributes[aid].domain) - 1
                noise  = int(round(rng.normal(0, sigma)))
                v_n    = max(0, min(d_max, v + noise))
                km[(r, aid)] = (max(0, v_n - delta), min(d_max, v_n + delta))
            self._set_exact(km, db, sa_ids, records=[r])
        return km


class MicroAggregation(Anonymizer):
    """
    Raggruppa record in cluster di ≥ k ordinati sul primo QI.
    QI: centroide del cluster (indice medio arrotondato).
    SA: valore esatto.
    """
    def build_knowledge_map(self, db, df_med, config):
        k      = config['experiment']['anonymization']['k']
        qi     = self._qi_names(config)
        sa     = self._sa_names(config)
        a2id   = self._attr_name_to_id(db)
        n_rec  = len(db.records)
        km     = self._full_range_map(db)

        qi_ids = [a2id[q] for q in qi if q in a2id]
        sa_ids = [a2id[s] for s in sa if s in a2id]

        if not qi_ids:
            return km

        # Ordina sul primo QI, partiziona in blocchi di k
        sort_aid = qi_ids[0]
        order    = sorted(range(n_rec), key=lambda r: db.records[r].values.get(sort_aid, 0))
        groups   = [order[i:i + k] for i in range(0, n_rec, k)]
        if len(groups) > 1 and len(groups[-1]) < k:
            groups[-2] += groups.pop()

        for grp in groups:
            self._set_partition_range(km, db, grp, qi_ids)
            self._set_exact(km, db, sa_ids, records=grp)
        return km


class RandomizedResponse(Anonymizer):
    """
    Modello pessimistico: QI completamente randomizzati → full range.
    SA: valore esatto.
    L'attaccante non può dedurre nulla dai QI perturbati.
    """
    def build_knowledge_map(self, db, df_med, config):
        sa     = self._sa_names(config)
        a2id   = self._attr_name_to_id(db)
        km     = self._full_range_map(db)

        sa_ids = [a2id[s] for s in sa if s in a2id]
        self._set_exact(km, db, sa_ids)
        # QI: restano full-range (pessimistico)
        return km
