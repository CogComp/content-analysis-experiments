from spacy.tokens import Token

from sacrerouge.metrics.interpretable_rouge.matchers import Matcher


@Matcher.register('special-tokens')
class SpecialTokensMatcher(Matcher):
    def __init__(self, name: str) -> None:
        super().__init__(name, content_type='misc')

    def is_match(self, token1: Token, token2: Token) -> bool:
        for token in [token1, token2]:
            if token.text in ['<s>', '</s>']:
                return True
        return False

    def is_candidate(self, token: Token) -> bool:
        return token.text in ['<s>', '</s>']
