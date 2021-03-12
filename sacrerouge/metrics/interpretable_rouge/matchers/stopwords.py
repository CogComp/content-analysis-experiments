import os
from overrides import overrides
from spacy.tokens import Token
from typing import List, Set, Tuple

from sacrerouge.common import DATA_ROOT
from sacrerouge.common.stats import calculate_pr_f1
from sacrerouge.data import MetricsDict
from sacrerouge.metrics.interpretable_rouge.matchers import Matcher
from sacrerouge.metrics.interpretable_rouge.util import calculate_maximum_matching


@Matcher.register('stopwords')
class StopwordMatcher(Matcher):
    def __init__(self, name: str, data_dir: str = f'{DATA_ROOT}/metrics/ROUGE-1.5.5/data'):
        super().__init__(name, content_type='stopwords')
        self.stopwords = self._load_stopwords(data_dir)

    def _load_stopwords(self, root: str) -> Set[str]:
        file_path = os.path.join(root, 'smart_common_words.txt')
        return set(open(file_path, 'r').read().splitlines())

    def is_match(self, token1: Token, token2: Token) -> bool:
        return token1.text.lower() in self.stopwords and token2.text.lower() in self.stopwords

    def is_candidate(self, token: Token) -> bool:
        return token.text.lower() in self.stopwords
