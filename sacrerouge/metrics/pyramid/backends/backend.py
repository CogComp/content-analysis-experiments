from collections import defaultdict
from spacy.tokens import Token
from typing import Dict, List, Set, Tuple

from sacrerouge.common import Registrable
from sacrerouge.data import MetricsDict


class Backend(Registrable):
    def _get_scu_to_indices(self, index_to_scus: List[Set[int]]) -> Dict[int, Set[int]]:
        scu_to_indices = defaultdict(set)
        for i, scus in enumerate(index_to_scus):
            for scu in scus:
                scu_to_indices[scu].add(i)
        return scu_to_indices

    def _get_matches(self,
                     summary_indices: Set[int],
                     reference_indices: Set[int],
                     all_matches: List[Tuple[int, int, float]]) -> List[Tuple[int, int, float]]:
        matches = []
        for i, j, weight in all_matches:
            if i in summary_indices and j in reference_indices:
                matches.append((i, j, weight))
        return matches

    def _sum_scu_token_weight(self,
                              index_to_scus: List[Set[int]],
                              weights: List[float]) -> float:
        total = 0
        for i, scus in enumerate(index_to_scus):
            if len(scus) > 0:
                total += weights[i]
        return total

    def _get_indices_complement(self, num_tokens: int, indices: List[int]) -> List[int]:
        indices = set(indices)
        complement = []
        for i in range(num_tokens):
            if i not in indices:
                complement.append(i)
        return complement

    def _get_non_scu_indices(self, index_to_scus: List[Set[int]]) -> List[int]:
        indices = []
        for i, scus in enumerate(index_to_scus):
            if len(scus) == 0:
                indices.append(i)
        return indices
