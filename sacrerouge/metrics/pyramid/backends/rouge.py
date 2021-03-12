import os
import re
from collections import defaultdict
from nltk.stem import PorterStemmer
from typing import Dict, List, Set, Tuple

from sacrerouge.common import DATA_ROOT
from sacrerouge.data import MetricsDict
from sacrerouge.metrics.pyramid.backends import Backend
from sacrerouge.metrics.interpretable_rouge.util import calculate_maximum_matching


@Backend.register('rouge-pyramid-backend')
class RougePyramidBackend(Backend):
    _non_alphanumeric_regex = re.compile('[^A-Za-z0-9]')

    def __init__(self,
                 use_porter_stemmer: bool = True,
                 remove_stopwords: bool = False,
                 rouge_data_dir: str = f'{DATA_ROOT}/metrics/ROUGE-1.5.5/data') -> None:
        self.use_porter_stemmer = use_porter_stemmer
        self.remove_stopwords = remove_stopwords
        self.stemmer = PorterStemmer(PorterStemmer.ORIGINAL_ALGORITHM)
        self.stemmer_exceptions = self._load_stemmer_exceptions(rouge_data_dir)
        self.stopwords = self._load_stopwords(rouge_data_dir)

    def _load_stemmer_exceptions(self, root: str) -> Dict[str, str]:
        exceptions = {}
        for filename in ['adj.exc', 'adv.exc', 'noun.exc', 'verb.exc']:
            file_path = os.path.join(root, 'WordNet-2.0-Exceptions', filename)
            with open(file_path, 'r') as f:
                for line in f:
                    # I think there is a bug in the original perl script
                    # to construct the exceptions database. Some of the lines
                    # have more than 2 words on them, but the script only
                    # maps the first to the second, ignoring the third.
                    columns = line.strip().split()
                    exceptions[columns[0]] = columns[1]
        return exceptions

    def _load_stopwords(self, root: str) -> Set[str]:
        file_path = os.path.join(root, 'smart_common_words.txt')
        return set(open(file_path, 'r').read().splitlines())

    def _group_by_token(self, tokens: List[str]) -> Dict[str, List[int]]:
        groups = defaultdict(list)
        for i, token in enumerate(tokens):
            if token:
                groups[token].append(i)
        return groups

    def _get_rouge_matches(self,
                           summary_tokens: List[str],
                           reference_tokens: List[str]) -> List[Tuple[int, int, float]]:
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
                          summary_tokens: List[str],
                          reference_tokens_list: List[List[str]]):
        return [self._get_rouge_matches(summary_tokens, reference_tokens) for reference_tokens in reference_tokens_list]

    def _get_summary_weights(self, tokens: List[str]) -> List[float]:
        return [1.0] * len(tokens)

    def tokenize(self, summary: str, scu_to_offsets: Dict[int, List[Tuple[int, int]]]):
        summary = summary.lower()

        tokens = []
        index_to_scus = []
        for match in re.finditer(r'\S+', summary):
            token = match.group(0)
            offset = match.start()
            assert summary[offset:offset + len(token)] == token, (summary, offset, token)

            # Even though this token may decompose into smaller tokens
            # (e.g. "it's" -> "it s"), we will still map it to the same offset
            # as a simplifying assumption
            for subtoken in RougePyramidBackend._non_alphanumeric_regex.sub(' ', token).split():
                if self.remove_stopwords and subtoken in self.stopwords:
                    continue
                if self.use_porter_stemmer and len(subtoken) > 3:
                    if subtoken in self.stemmer_exceptions:
                        subtoken = self.stemmer_exceptions[subtoken]
                    else:
                        subtoken = self.stemmer.stem(subtoken)

                # Find all of the SCUs that overlap with this subtoken
                scus = set()
                for scu_id, offsets_list in scu_to_offsets.items():
                    for start, end in offsets_list:
                        if start <= offset and offset < end:
                            scus.add(scu_id)
                            break
                tokens.append(subtoken)
                index_to_scus.append(scus)

        return tokens, index_to_scus

    def get_matches_list(self,
                         summary_tokens_list: List[List[str]],
                         reference_tokens_lists: List[List[List[str]]],
                         summary_index_to_scus_list,
                         reference_index_to_scus_lists):
        # Compute all of the ROUGE matches
        output_matches_lists = []
        output_precision_weights_list = []
        output_recall_weights_lists = []

        for summary_tokens, reference_tokens_list in zip(summary_tokens_list, reference_tokens_lists):
            output_matches_lists.append(self._get_matches_list(summary_tokens, reference_tokens_list))
            output_precision_weights_list.append(self._get_summary_weights(summary_tokens))
            output_recall_weights_lists.append([self._get_summary_weights(reference_tokens) for reference_tokens in reference_tokens_list])

        return summary_index_to_scus_list, \
               reference_index_to_scus_lists, reference_index_to_scus_lists, \
               output_matches_lists, output_matches_lists, \
               output_precision_weights_list, output_recall_weights_lists

    def get_total_weight(self, matches: List[Tuple[int, int, float]]) -> float:
        return calculate_maximum_matching(matches)

    def calculate_standard_metric(self,
                                  summary_index_to_scus: List[Set[int]],
                                  reference_index_to_scus: List[Set[int]],
                                  summary_weights: List[float],
                                  reference_weights: List[float],
                                  matches: List[Tuple[int, int, float]]) -> MetricsDict:
        # This is the standard ROUGE calculation except the SCU-based matches are
        # given priority over non-SCU matches to maximize the percentage of the
        # ROUGE score the SCU matches contribute.
        summary_scu_to_indices = self._get_scu_to_indices(summary_index_to_scus)
        reference_scu_to_indices = self._get_scu_to_indices(reference_index_to_scus)

        all_matches = []
        for scu in summary_scu_to_indices.keys():
            summary_indices = summary_scu_to_indices[scu]
            reference_indices = reference_scu_to_indices[scu]
            scu_matches = self._get_matches(summary_indices, reference_indices, matches)
            all_matches.extend(scu_matches)
        num_scu_matches, matching = calculate_maximum_matching(all_matches, return_matching=True)

        # Mark which tokens were matched and therefore no long eligible
        summary_matches = [False] * len(summary_index_to_scus)
        references_matches = [False] * len(reference_index_to_scus)
        for i, j in matching:
            summary_matches[i] = True
            references_matches[j] = True

        summary_indices = [i for i in range(len(summary_index_to_scus)) if not summary_matches[i]]
        reference_indices = [i for i in range(len(reference_index_to_scus)) if not references_matches[i]]
        non_scus_matches = self._get_matches(summary_indices, reference_indices, matches)
        num_non_scu_matches = calculate_maximum_matching(non_scus_matches)

        intersection = num_scu_matches + num_non_scu_matches
        return MetricsDict({
            'weight': intersection,
            'summary_weight': sum(summary_weights),
            'reference_weight': sum(reference_weights),
            'scu_weight': num_scu_matches,
            'non_scu_weight': num_non_scu_matches,
        })

    def calculate_scu_metric(self,
                             summary_index_to_scus: List[Set[int]],
                             reference_index_to_scus: List[Set[int]],
                             summary_weights: List[float],
                             reference_weights: List[float],
                             matches: List[Tuple[int, int, float]]) -> MetricsDict:
        summary_scu_to_indices = self._get_scu_to_indices(summary_index_to_scus)
        reference_scu_to_indices = self._get_scu_to_indices(reference_index_to_scus)

        all_matches = []
        for scu in summary_scu_to_indices.keys():
            summary_indices = summary_scu_to_indices[scu]
            reference_indices = reference_scu_to_indices[scu]
            scu_matches = self._get_matches(summary_indices, reference_indices, matches)
            all_matches.extend(scu_matches)

        intersection = calculate_maximum_matching(all_matches)
        return MetricsDict({
            'weight': intersection,
            'summary_weight': self._sum_scu_token_weight(summary_index_to_scus, summary_weights),
            'reference_weight': self._sum_scu_token_weight(reference_index_to_scus, reference_weights)
        })

    def calculate_non_scu_metric(self,
                                 summary_index_to_scus: List[Set[int]],
                                 reference_index_to_scus: List[Set[int]],
                                 summary_weights: List[float],
                                 reference_weights: List[float],
                                 matches: List[Tuple[int, int, float]]) -> MetricsDict:
        summary_scu_to_indices = self._get_scu_to_indices(summary_index_to_scus)
        reference_scu_to_indices = self._get_scu_to_indices(reference_index_to_scus)

        all_matches = []
        # For each SCU, we have to match the summary SCU tokens to any
        # reference token NOT in that SCU and vice versa.
        for scu in summary_scu_to_indices:
            summary_indices = summary_scu_to_indices[scu]
            reference_indices = reference_scu_to_indices[scu]

            summary_complement = self._get_indices_complement(len(summary_index_to_scus), summary_indices)
            reference_complement = self._get_indices_complement(len(reference_index_to_scus), reference_indices)

            this_matches = self._get_matches(summary_indices, reference_complement, matches)
            all_matches.extend(this_matches)

            this_matches = self._get_matches(summary_complement, reference_indices, matches)
            all_matches.extend(this_matches)

        # Then we have to match any token not part of any SCU in the summary
        # to any token not part of any SCU in the reference.
        summary_indices = self._get_non_scu_indices(summary_index_to_scus)
        reference_indices = self._get_non_scu_indices(reference_index_to_scus)
        this_matches = self._get_matches(summary_indices, reference_indices, matches)
        all_matches.extend(this_matches)

        intersection = calculate_maximum_matching(all_matches)
        return MetricsDict({
            'weight': intersection,
            'summary_weight': sum(summary_weights),
            'reference_weight': sum(reference_weights)
        })