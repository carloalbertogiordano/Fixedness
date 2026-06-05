import numpy as np
from .base import Anonymizer, KnowledgeMap
from fixedness.core.models import Database
import pandas as pd


class LaplaceDP(Anonymizer):
    """
    Central Differential Privacy (Laplace mechanism) sui QI.
    Noise ~ Laplace(0, sensitivity/ε) aggiunto all'indice di dominio.
    Range pubblicato: v_noisy ± ⌈3/ε⌉ (copertura ~99% della distribuzione).
    SA: valore esatto (DP applicata solo ai QI).
    """
    def build_knowledge_map(self, db, df_med, config):
        epsilon = max(1e-6, config['experiment']['anonymization'].get('epsilon', 1.0))
        qi      = self._qi_names(config)
        sa      = self._sa_names(config)
        a2id    = self._attr_name_to_id(db)
        n_rec   = len(db.records)
        km      = self._full_range_map(db)
        rng     = np.random.RandomState(config.get('_seed', 42))

        scale  = 1.0 / epsilon          # sensitivity = 1 nello spazio degli indici
        delta  = int(np.ceil(3 * scale))

        qi_ids = [a2id[q] for q in qi if q in a2id]
        sa_ids = [a2id[s] for s in sa if s in a2id]

        for r in range(n_rec):
            for aid in qi_ids:
                v      = db.records[r].values.get(aid, 0)
                d_max  = len(db.attributes[aid].domain) - 1
                noise  = int(round(rng.laplace(0, scale)))
                v_n    = max(0, min(d_max, v + noise))
                km[(r, aid)] = (max(0, v_n - delta), min(d_max, v_n + delta))
            self._set_exact(km, db, sa_ids, records=[r])
        return km


class LocalDP(Anonymizer):
    """
    Local Differential Privacy: ogni record perturbato indipendentemente prima
    della raccolta (randomized response). Scala del rumore = d_size / (2·ε),
    molto più alta del caso centrale (1/ε). Per ε piccolo il delta supera
    l'intero dominio → full range; per ε grande il range si restringe.
    SA: valore esatto.
    """
    def build_knowledge_map(self, db, df_med, config):
        epsilon = max(1e-6, config['experiment']['anonymization'].get('epsilon', 1.0))
        qi      = self._qi_names(config)
        sa      = self._sa_names(config)
        a2id    = self._attr_name_to_id(db)
        n_rec   = len(db.records)
        km      = self._full_range_map(db)
        rng     = np.random.RandomState(config.get('_seed', 42))

        qi_ids = [a2id[q] for q in qi if q in a2id]
        sa_ids = [a2id[s] for s in sa if s in a2id]

        for r in range(n_rec):
            for aid in qi_ids:
                v      = db.records[r].values.get(aid, 0)
                d_max  = len(db.attributes[aid].domain) - 1
                d_size = d_max + 1
                # LDP noise scale is d_size times larger than central DP
                scale  = d_size / (2.0 * epsilon)
                delta  = int(np.ceil(3 * scale))
                if delta >= d_max:
                    continue  # full range already set — epsilon too small
                noise  = int(round(rng.laplace(0, scale)))
                v_n    = max(0, min(d_max, v + noise))
                km[(r, aid)] = (max(0, v_n - delta), min(d_max, v_n + delta))
            self._set_exact(km, db, sa_ids, records=[r])
        return km
