import sys
from abc import abstractmethod
from collections import defaultdict

from tqdm import tqdm


class Model:
    # fo_lang_code::zh,jp,po,ko
    def __init__(self, fo_lang_code=None):
        self.name = ""
        self.fo_lang_code = fo_lang_code

    @abstractmethod
    def build_model(self, docs):
        pass

    @abstractmethod
    def get_doc_similarity(self, doc1_tokens, doc2_tokens):
        pass

    def get_model_name(self):
        return self.name

    def generate_links_with_scores(self, sources: dict, targets: dict) -> dict[str:list[tuple]]:
        links_with_scores = defaultdict(list)
        tokens = dict()
        for source_id in sources:
            source_content = sources[source_id]
            tokens[source_id] = source_content.split()
        for target_id in targets:
            target_content = targets[target_id]
            tokens[target_id] = target_content.split()

        for source_id in tqdm(sources, file=sys.stdout, colour="#646cff"):
            for target_id in targets:
                links_with_scores[source_id].append((
                    source_id, target_id,
                    self.get_doc_similarity(tokens[source_id], tokens[target_id])
                ))
        return links_with_scores
