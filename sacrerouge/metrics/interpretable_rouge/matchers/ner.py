from overrides import overrides
from spacy.tokens import Token
from typing import List, Set, Tuple

from sacrerouge.common.stats import calculate_pr_f1
from sacrerouge.data import MetricsDict
from sacrerouge.metrics.interpretable_rouge.matchers import Matcher


@Matcher.register('ner')
class NERMatcher(Matcher):
    def __init__(self, name: str, tag_sets: List[List[str]]) -> None:
        super().__init__(name, content_type='topic')
        self.tag_sets = [set(tags) for tags in tag_sets]

    def is_match(self, token1: Token, token2: Token) -> bool:
        for tags in self.tag_sets:
            if token1.ent_type_ in tags and token2.ent_type_ in tags:
                return True
        return False

    def is_candidate(self, token: Token) -> bool:
        for tags in self.tag_sets:
            if token.ent_type_ in tags:
                return True
        return False
