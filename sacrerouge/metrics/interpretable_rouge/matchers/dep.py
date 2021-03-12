from overrides import overrides
from spacy.tokens import Token
from typing import List, Dict

from sacrerouge.common.stats import calculate_pr_f1
from sacrerouge.data import MetricsDict
from sacrerouge.metrics.interpretable_rouge.matchers import Matcher, TupleMatcher
from sacrerouge.metrics.interpretable_rouge.util import calculate_maximum_matching


@Matcher.register('dependency')
class DependencyMatcher(Matcher):
    def __init__(self, name: str, relations: List[str]) -> None:
        super().__init__(name, content_type='topic')
        self.relations = set(relations)

    def is_match(self, token1: Token, token2: Token) -> bool:
        return token1.dep_ in self.relations and token2.dep_ in self.relations

    def is_candidate(self, token: Token) -> bool:
        return token.dep_ in self.relations


@Matcher.register('dependency-verb-relations')
class DependencyVerbRelationsMatcher(TupleMatcher):
    def __init__(self, name: str, relations: List[str]) -> None:
        super().__init__(name)
        self.relations = set(relations)

    def _get_children(self, tokens: List[Token], head: Token) -> List[Token]:
        children = []
        for token in tokens:
            if token.head == head:
                children.append(token)
        return children

    def get_tuples(self, tokens: List[Token]) -> List[Dict[str, int]]:
        tuples = []
        for token in tokens:
            if token.pos_ == 'VERB':
                children = self._get_children(tokens, token)

                # Filter to only children with tags we care about
                children = list(filter(lambda child: child.dep_ in self.relations, children))

                # If the remaining children have the same set of relations that we're
                # looking for, we keep the match. Otherwise, I don't know what to do
                if len(self.relations) == len(set(child.dep_ for child in children)):
                    match = {'VERB': token._.index}
                    for child in children:
                        match[child.dep_] = child._.index
                    tuples.append(match)

        return tuples