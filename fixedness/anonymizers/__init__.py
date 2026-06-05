from .factory import get_anonymizer, available_methods
from .base import KnowledgeMap


def get_anonymization_map(db, df, config):
    method = config['experiment']['anonymization'].get('method', 'mondrian_k')
    return get_anonymizer(method).build_knowledge_map(db, df, config)
