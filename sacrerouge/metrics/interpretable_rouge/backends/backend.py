from collections import defaultdict
from spacy.tokens import Token
from typing import Dict, List, Tuple

from sacrerouge.common import Registrable
from sacrerouge.data import MetricsDict


class Backend(Registrable):
    def get_matches_list(self,
                         summary: List[str],
                         references: List[List[str]]) -> Tuple[List[Token], List[Token], List[Tuple[int, int, float]]]:
        raise NotImplementedError
