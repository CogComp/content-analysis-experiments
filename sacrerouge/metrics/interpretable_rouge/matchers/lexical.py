from spacy.tokens import Token

from sacrerouge.metrics.interpretable_rouge.matchers import Matcher


@Matcher.register('lexical')
class LexicalMatcher(Matcher):
    def __init__(self, name: str) -> None:
        super().__init__(name, content_type='misc')

    def is_match(self, token1: Token, token2: Token) -> bool:
        return token1._.matching_text == token2._.matching_text

    def is_candidate(self, token: Token) -> bool:
        return token._.is_matchable
