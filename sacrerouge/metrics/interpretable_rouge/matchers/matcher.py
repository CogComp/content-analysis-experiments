from collections import defaultdict
from spacy.tokens import Token
from typing import Dict, List, Tuple

from sacrerouge.common import Registrable
from sacrerouge.data import MetricsDict
from sacrerouge.metrics.interpretable_rouge.backends import Backend, BertScoreBackend, RougeBackend
from sacrerouge.metrics.interpretable_rouge.util import calculate_maximum_matching


class Matcher(Registrable):
    def __init__(self, name: str, content_type: str):
        self.name = name
        self.content_type = content_type

    def select_matches(self,
                       summary_tokens: List[Token],
                       reference_tokens: List[Token],
                       matches: List[Tuple[int, int]],
                       weights: List[float],
                       metric: str,
                       backend: Backend) -> Tuple[List[Tuple[int, int]], MetricsDict]:
        common_matches = []
        for i, j, weight in matches:
            summary_token = summary_tokens[i]
            reference_token = reference_tokens[j]
            if self.is_match(summary_token, reference_token):
                common_matches.append((i, j, weight))

        norm_weight = 0
        if metric == 'precision':
            tokens = summary_tokens
        else:
            tokens = reference_tokens
        for i, token in enumerate(tokens):
            if self.is_candidate(token):
                norm_weight += weights[i]

        matching_weight = backend.get_total_weight(common_matches)
        metrics = MetricsDict({
            self.name: {
                f'{metric}_weight': matching_weight,
                f'{metric}_norm_weight': norm_weight,
            }
        })

        return common_matches, metrics

    def is_match(self, token1: Token, token2: Token) -> bool:
        raise NotImplementedError

    def is_candidate(self, token: Token) -> bool:
        raise NotImplementedError

    def finalize(self,
                 metrics_list: List[MetricsDict],
                 total_num_matches: int,
                 metric: str) -> MetricsDict:
        metrics = sum(metrics_list)

        matched_weight = metrics[self.name][f'{metric}_weight']
        norm_weight = metrics[self.name][f'{metric}_norm_weight']
        measure = 0.0
        if norm_weight != 0.0:
            measure = matched_weight / norm_weight * 100
        metrics[self.name][metric] = measure

        coverage = matched_weight / total_num_matches * 100 if total_num_matches > 0 else 0
        metrics[self.name][f'{metric}_coverage'] = coverage

        return metrics

    def _get_cross_product(self,
                           summary_indices: List[int],
                           reference_indices: List[int]) -> List[Tuple[int, int]]:
        matches = []
        for i in summary_indices:
            for j in reference_indices:
                matches.append((i, j))
        return matches

    def _group_indices_by_token(self, tokens: List[Token], indices: List[int]) -> Dict[str, List[int]]:
        groups = defaultdict(list)
        for index in indices:
            groups[tokens[index]._.matching_text].append(index)
        return groups


class TupleMatcher(Matcher):
    def __init__(self, name: str):
        super().__init__(name, content_type='semantic')

    def get_tuples(self, tokens: List[Token]) -> List[Dict[str, int]]:
        raise NotImplementedError

    def _select_matches_bert(self,
                             summary_tuples: List[Dict[str, int]],
                             reference_tuples: List[Dict[str, int]],
                             matches: List[Tuple[int, int, float]]) -> float:
        edge_to_weight = {(i, j): weight for i, j, weight in matches}
        matched_set = set()
        for summary_tuple in summary_tuples:
            for reference_tuple in reference_tuples:
                assert len(summary_tuple) == len(reference_tuple)
                this_matches = set()
                matched = True
                for key, i in summary_tuple.items():
                    j = reference_tuple[key]
                    if (i, j) in edge_to_weight:
                        weight = edge_to_weight[(i, j)]
                        this_matches.add((i, j, weight))
                    else:
                        matched = False
                        break

                if matched:
                    matched_set |= this_matches

        total_weight = sum(weight for _, _, weight in matched_set)
        common_matches = list(matched_set)
        return common_matches, total_weight

    def _select_matches_rouge(self,
                              summary_tuples: List[Tuple[int]],
                              reference_tuples: List[Tuple[int]],
                              matches: List[Tuple[int, int, float]]) -> float:
        edge_to_weight = {(i, j): weight for i, j, weight in matches}
        meta_matches = []
        meta_indices_to_matches = defaultdict(list)
        for s_i, summary_tuple in enumerate(summary_tuples):
            for r_j, reference_tuple in enumerate(reference_tuples):
                assert len(summary_tuple) == len(reference_tuple)
                # See if these each component of these two tuples can be aligned
                matched = True
                tuple_matches = []
                for key, i in summary_tuple.items():
                    j = reference_tuple[key]
                    if (i, j) not in edge_to_weight:
                        matched = False
                        break
                    else:
                        tuple_matches.append((i, j, edge_to_weight[(i, j)]))

                # Add the tuples to the meta matching. We don't allow for tuples to be matched twice, although
                # this is theoretically possible if the components of a tuple are reused. I don't think this is
                # a big enough deal to cause any significant impact on the results
                if matched:
                    # The tuple match explains len(summary_tuple) edges
                    meta_matches.append((s_i, r_j, len(summary_tuple)))
                    meta_indices_to_matches[(s_i, r_j)] = tuple_matches

        # Compute the matching on the meta graph. This is the total number of edges explained by the tuple matches
        meta_weight, matching = calculate_maximum_matching(meta_matches, return_matching=True)

        # Convert the matching back into the set of alignments that it explains
        common_matches = []
        for s_i, r_j in matching:
            common_matches.extend(meta_indices_to_matches[(s_i, r_j)])

        return common_matches, meta_weight

    def select_matches(self,
                       summary_tokens: List[Token],
                       reference_tokens: List[Token],
                       matches: List[Tuple[int, int]],
                       weights: List[float],
                       metric: str,
                       backend: Backend) -> Tuple[List[Tuple[int, int]], MetricsDict]:
        summary_tuples = self.get_tuples(summary_tokens)
        reference_tuples = self.get_tuples(reference_tokens)

        if isinstance(backend, BertScoreBackend):
            common_matches, total_weight = self._select_matches_bert(summary_tuples, reference_tuples, matches)
        else:
            common_matches, total_weight = self._select_matches_rouge(summary_tuples, reference_tuples, matches)

        if metric == 'precision':
            tuples = summary_tuples
        else:
            tuples = reference_tuples

        norm_weight = 0
        norm_indices = set()
        for tup in tuples:
            for i in tup.values():
                norm_indices.add(i)
        for i in norm_indices:
            norm_weight += weights[i]

        return common_matches, MetricsDict({
            self.name: {
                f'{metric}_weight': total_weight,
                f'{metric}_norm_weight': norm_weight,
            }
        })