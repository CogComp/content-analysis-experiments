from overrides import overrides
from spacy.tokens import Token
from typing import List, Tuple

from sacrerouge.common.stats import calculate_pr_f1
from sacrerouge.data import MetricsDict
from sacrerouge.metrics.interpretable_rouge.matchers import Matcher
from sacrerouge.metrics.interpretable_rouge.util import calculate_maximum_matching


@Matcher.register('pos')
class POSMatcher(Matcher):
    def __init__(self, name: str, tags: List[str]) -> None:
        super().__init__(name, content_type='topic')
        self.tags = set(tags)

    def is_match(self, token1: Token, token2: Token) -> bool:
        return token1.pos_ in self.tags and token2.pos_ in self.tags

    def is_candidate(self, token: Token) -> bool:
        return token.pos_ in self.tags
