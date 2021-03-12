from spacy.tokens import Token

from sacrerouge.metrics.interpretable_rouge.matchers import Matcher


@Matcher.register('phrasal')
class PhrasalMatcher(Matcher):
    def __init__(self, name: str) -> None:
        # For now, we can only do NP chunks
        super().__init__(name, content_type='topic')

    def is_match(self, token1: Token, token2: Token) -> bool:
        return token1._.is_np is True and token2._.is_np is True

    def is_candidate(self, token: Token) -> bool:
        return token._.is_np is True
