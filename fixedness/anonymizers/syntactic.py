from collections import Counter
from typing import List
from .base import Anonymizer, KnowledgeMap, apply_mondrian
from fixedness.core.models import Database
import pandas as pd


class NoAnonymization(Anonymizer):
    """Baseline: tutti i valori esatti, nessuna protezione."""
    def build_knowledge_map(self, db, df_med, config):
        km    = self._full_range_map(db)
        a2id  = self._attr_name_to_id(db)
        all_names = self._qi_names(config) + self._sa_names(config)
        ids   = [a2id[n] for n in all_names if n in a2id]
        self._set_exact(km, db, ids)
        return km


class Suppression(Anonymizer):
    """
    QI combinations con frequenza < k vengono soppresse (full range).
    QI e SA dei record non soppressi rimangono esatti.
    """
    def build_knowledge_map(self, db, df_med, config):
        k     = config['experiment']['anonymization']['k']
        qi    = self._qi_names(config)
        sa    = self._sa_names(config)
        a2id  = self._attr_name_to_id(db)
        n_rec = len(db.records)
        km    = self._full_range_map(db)

        qi_ids = [a2id[q] for q in qi if q in a2id]
        sa_ids = [a2id[s] for s in sa if s in a2id]

        combo_count = Counter(
            tuple(db.records[r].values.get(aid) for aid in qi_ids)
            for r in range(n_rec)
        )
        for r in range(n_rec):
            combo = tuple(db.records[r].values.get(aid) for aid in qi_ids)
            if combo_count[combo] >= k:
                self._set_exact(km, db, qi_ids + sa_ids, records=[r])
            # else: record soppresso → QI e SA restano full-range
        return km


class MondrianK(Anonymizer):
    """
    Mondrian k-anonymity standard.
    QI: range della partizione. SA: valore esatto (pubblicato as-is).
    """
    def build_knowledge_map(self, db, df_med, config):
        k     = config['experiment']['anonymization']['k']
        qi    = self._qi_names(config)
        sa    = self._sa_names(config)
        a2id  = self._attr_name_to_id(db)
        km    = self._full_range_map(db)

        partitions = apply_mondrian(df_med, k, qi)
        qi_ids = [a2id[q] for q in qi if q in a2id]
        sa_ids = [a2id[s] for s in sa if s in a2id]

        for part in partitions:
            self._set_partition_range(km, db, part, qi_ids)
            self._set_exact(km, db, sa_ids, records=part)
        return km


class LDiversity(Anonymizer):
    """
    l-diversity: Mondrian + ogni partizione ha ≥ l valori distinti del SA primario.
    SA knowledge: range della partizione (non esatto — diversità garantita).
    """
    def build_knowledge_map(self, db, df_med, config):
        k     = config['experiment']['anonymization']['k']
        l     = config['experiment']['anonymization'].get('l', 2)
        qi    = self._qi_names(config)
        sa    = self._sa_names(config)
        a2id  = self._attr_name_to_id(db)
        km    = self._full_range_map(db)

        sa_id_primary = a2id.get(sa[0]) if sa else None
        partitions    = apply_mondrian(df_med, k, qi)
        if sa_id_primary is not None and l > 1:
            partitions = _enforce_l_diversity(partitions, db, sa_id_primary, l)

        qi_ids = [a2id[q] for q in qi if q in a2id]
        sa_ids = [a2id[s] for s in sa if s in a2id]

        for part in partitions:
            self._set_partition_range(km, db, part, qi_ids)
            # SA: range della partizione (attaccante sa solo che il valore è nel range)
            self._set_partition_range(km, db, part, sa_ids)
        return km


class TCloseness(Anonymizer):
    """
    t-closeness: Mondrian + distribuzione SA di ogni partizione entro distanza t
    dalla distribuzione globale (total variation distance).
    SA knowledge: full range — la distribuzione mimics quella globale, nulla si deduce.
    """
    def build_knowledge_map(self, db, df_med, config):
        k     = config['experiment']['anonymization']['k']
        t     = config['experiment']['anonymization'].get('t', 0.2)
        qi    = self._qi_names(config)
        sa    = self._sa_names(config)
        a2id  = self._attr_name_to_id(db)
        n_rec = len(db.records)
        km    = self._full_range_map(db)

        sa_id_primary = a2id.get(sa[0]) if sa else None
        partitions    = apply_mondrian(df_med, k, qi)

        if sa_id_primary is not None:
            all_sa  = [db.records[r].values.get(sa_id_primary) for r in range(n_rec)]
            total   = len(all_sa)
            g_freq  = {v: c / total for v, c in Counter(all_sa).items()}
            partitions = _enforce_t_closeness(partitions, db, sa_id_primary, t, g_freq)

        qi_ids = [a2id[q] for q in qi if q in a2id]
        sa_ids = [a2id[s] for s in sa if s in a2id]

        for part in partitions:
            self._set_partition_range(km, db, part, qi_ids)
            # SA: full range — t-closeness garantisce distribuzione simile a globale
            for aid in sa_ids:
                for r in part:
                    km[(r, aid)] = (0, len(db.attributes[aid].domain) - 1)
        return km


# ── helper functions ───────────────────────────────────────────────────────

def _enforce_l_diversity(partitions, db, sa_id, l):
    """Post-processing: merge partizioni che violano l-diversity."""
    changed = True
    while changed:
        changed = False
        merged  = set()
        new_parts = []
        for i, part in enumerate(partitions):
            if i in merged:
                continue
            sa_vals = {db.records[r].values.get(sa_id) for r in part}
            if len(sa_vals) >= l:
                new_parts.append(part)
                merged.add(i)
                continue
            # Cerca il partner che massimizza i SA distinti combinati
            best = None
            for j, other in enumerate(partitions):
                if j == i or j in merged:
                    continue
                combined_sa = sa_vals | {db.records[r].values.get(sa_id) for r in other}
                if best is None or len(combined_sa) > best[0]:
                    best = (len(combined_sa), j, part + other)
            if best is not None:
                new_parts.append(best[2])
                merged.add(i)
                merged.add(best[1])
                changed = True
            else:
                new_parts.append(part)
                merged.add(i)
        partitions = new_parts
    return partitions


def _tv_distance(p_freq, g_freq):
    """Total variation distance = 0.5 * L1."""
    keys = set(p_freq) | set(g_freq)
    return 0.5 * sum(abs(p_freq.get(k, 0) - g_freq.get(k, 0)) for k in keys)


def _enforce_t_closeness(partitions, db, sa_id, t, g_freq):
    """Post-processing: merge partizioni con TV distance > t dalla distribuzione globale."""
    changed = True
    while changed:
        changed = False
        merged  = set()
        new_parts = []
        for i, part in enumerate(partitions):
            if i in merged:
                continue
            sa_vals = [db.records[r].values.get(sa_id) for r in part]
            p_freq  = {v: c / len(sa_vals) for v, c in Counter(sa_vals).items()}
            if _tv_distance(p_freq, g_freq) <= t:
                new_parts.append(part)
                merged.add(i)
                continue
            # Cerca merge che porta più vicino a g_freq
            best = None
            for j, other in enumerate(partitions):
                if j == i or j in merged:
                    continue
                comb    = part + other
                c_sa    = [db.records[r].values.get(sa_id) for r in comb]
                c_freq  = {v: c / len(c_sa) for v, c in Counter(c_sa).items()}
                dist    = _tv_distance(c_freq, g_freq)
                if best is None or dist < best[0]:
                    best = (dist, j, comb)
            if best is not None:
                new_parts.append(best[2])
                merged.add(i)
                merged.add(best[1])
                changed = True
            else:
                new_parts.append(part)
                merged.add(i)
        partitions = new_parts
    return partitions
