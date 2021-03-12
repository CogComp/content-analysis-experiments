import os
import spacy
from collections import defaultdict
from spacy.tokens import Token
from nltk.stem import PorterStemmer
from tqdm import tqdm
from typing import Dict, List, Set, Tuple

from sacrerouge.common import DATA_ROOT
from sacrerouge.data.types import SummaryType
from sacrerouge.metrics.interpretable_rouge.backends import Backend
from sacrerouge.metrics.interpretable_rouge.util import calculate_maximum_matching


@Backend.register('rouge-backend')
class RougeBackend(Backend):
    def __init__(self) -> None:
        pass

    def _group_by_token(self, tokens: List[Token]) -> Dict[str, List[int]]:
        groups = defaultdict(list)
        for token in tokens:
            if token._.is_matchable:
                groups[token._.matching_text].append(token._.index)
        return groups

    def _get_rouge_matches(self,
                           summary_tokens: List[Token],
                           reference_tokens: List[Token]) -> List[Tuple[int, int, float]]:
        summary_tokens = self._group_by_token(summary_tokens)
        reference_tokens = self._group_by_token(reference_tokens)

        matches = []
        for token in summary_tokens.keys():
            if token in reference_tokens:
                summary_indices = summary_tokens[token]
                reference_indices = reference_tokens[token]
                for i in summary_indices:
                    for j in reference_indices:
                        matches.append((i, j, 1.0))
        return matches

    def _get_matches_list(self,
                          summary_tokens: List[Token],
                          reference_tokens_list: List[List[Token]]):
        return [self._get_rouge_matches(summary_tokens, reference_tokens) for reference_tokens in reference_tokens_list]

    def _get_summary_weights(self, tokens: List[Token]) -> List[float]:
        weights = []
        for token in tokens:
            if token._.is_matchable:
                weights.append(1.0)
            else:
                weights.append(0.0)
        return weights

    def get_matches_list(self,
                         summary_tokens_lists: List[List[Token]],
                         reference_tokens_lists: List[List[Token]]):
        # Compute all of the ROUGE matches
        output_summary_tokens_list = []
        output_reference_tokens_lists = []
        output_matches_lists = []
        output_precision_weights_list = []
        output_recall_weights_lists = []

        for summary_tokens_list, reference_tokens_list in zip(summary_tokens_lists, reference_tokens_lists):
            for summary_tokens in summary_tokens_list:
                output_summary_tokens_list.append(summary_tokens)
                output_reference_tokens_lists.append(reference_tokens_list)
                output_matches_lists.append(self._get_matches_list(summary_tokens, reference_tokens_list))
                output_precision_weights_list.append(self._get_summary_weights(summary_tokens))
                output_recall_weights_lists.append([self._get_summary_weights(reference_tokens) for reference_tokens in reference_tokens_list])
        return output_summary_tokens_list, output_reference_tokens_lists, output_reference_tokens_lists, \
               output_matches_lists, output_matches_lists, \
               output_precision_weights_list, output_recall_weights_lists

    def get_total_weight(self, matches: List[Tuple[int, int, float]]) -> float:
        return calculate_maximum_matching(matches)
