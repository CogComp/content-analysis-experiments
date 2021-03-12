import os
import re
from collections import defaultdict
from nltk.stem import PorterStemmer
from typing import Dict, List, Set, Tuple

from sacrerouge.common import DATA_ROOT
from sacrerouge.data import MetricsDict, Pyramid, PyramidAnnotation
from sacrerouge.data.fields import PyramidField, PyramidAnnotationField
from sacrerouge.data.jackknifers import PyramidJackknifer
from sacrerouge.data.types import SummaryType
from sacrerouge.metrics import Metric
from sacrerouge.metrics.interpretable_rouge.util import calculate_maximum_matching


@Metric.register('pyramid-rouge')
class PyramidRouge(Metric):
    _non_alphanumeric_regex = re.compile('[^A-Za-z0-9]')

    def __init__(self,
                 use_porter_stemmer: bool = True,
                 remove_stopwords: bool = False,
                 rouge_data_dir: str = f'{DATA_ROOT}/metrics/ROUGE-1.5.5/data') -> None:
        super().__init__(['pyramid'], jackknifer=PyramidJackknifer())
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

    def _get_summary_scu_to_offsets(self, annotation: PyramidAnnotation) -> Dict[int, List[Tuple[int, int]]]:
        scu_to_offsets = defaultdict(list)
        for scu in annotation.scus:
            for contributor in scu.contributors:
                for part in contributor.parts:
                    scu_to_offsets[scu.scu_id].append((part.start, part.end))
        return scu_to_offsets

    def _get_reference_scu_to_offsets(self, pyramid: Pyramid, index: int) -> Dict[int, List[Tuple[int, int]]]:
        scu_to_offsets = defaultdict(list)
        for scu in pyramid.scus:
            for contributor in scu.contributors:
                if contributor.summary_index == index:
                    for part in contributor.parts:
                        scu_to_offsets[scu.scu_id].append((part.start, part.end))
        return scu_to_offsets

    def _filter_scu_to_offsets(self,
                               scu_to_offsets: Dict[int, List[Tuple[int, int]]],
                               valid_scus: Set[int]) -> Dict[int, List[Tuple[int, int]]]:
        return {scu: offsets for scu, offsets in scu_to_offsets.items() if scu in valid_scus}

    def _get_scu_intersection(self, annotation: PyramidAnnotation, pyramid: Pyramid, index: int) -> Set[int]:
        annotation_scus = annotation.get_scu_id_set()
        reference_scus = pyramid.get_scu_id_set(index)
        return annotation_scus & reference_scus

    def _tokenize(self, summary: str, scu_to_offsets: Dict[int, List[Tuple[int, int]]]):
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
            for subtoken in PyramidRouge._non_alphanumeric_regex.sub(' ', token).split():
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

    def _compute_standard_rouge(self,
                                summary_tokens: List[str],
                                summary_index_to_scus: List[Set[int]],
                                reference_tokens: List[str],
                                reference_index_to_scus: List[Set[int]]) -> MetricsDict():
        # This is the standard ROUGE calculation except the SCU-based matches are
        # given priority over non-SCU matches to maximize the percentage of the
        # ROUGE score the SCU matches contribute.
        summary_scu_to_indices = self._get_scu_to_indices(summary_index_to_scus)
        reference_scu_to_indices = self._get_scu_to_indices(reference_index_to_scus)

        all_matches = []
        for scu in summary_scu_to_indices.keys():
            summary_indices = summary_scu_to_indices[scu]
            reference_indices = reference_scu_to_indices[scu]
            matches = self._get_matches(summary_tokens, summary_indices,
                                        reference_tokens, reference_indices)
            all_matches.extend(matches)
        num_scu_matches, matching = calculate_maximum_matching(all_matches, return_matching=True)

        # Mark which tokens were matched and therefore no long eligible
        summary_matches = [False] * len(summary_tokens)
        references_matches = [False] * len(reference_tokens)
        for i, j in matching:
            summary_matches[i] = True
            references_matches[j] = True

        summary_indices = [i for i in range(len(summary_tokens)) if not summary_matches[i]]
        reference_indices = [i for i in range(len(reference_tokens)) if not references_matches[i]]
        matches = self._get_matches(summary_tokens, summary_indices,
                                    reference_tokens, reference_indices)
        num_non_scu_matches = calculate_maximum_matching(matches)

        intersection = num_scu_matches + num_non_scu_matches
        m = MetricsDict({
            'intersection': intersection,
            'num_summary_tokens': len(summary_tokens),
            'num_reference_tokens': len(reference_tokens),
            'num_scu_matches': num_scu_matches,
            'num_non_scu_matches': num_non_scu_matches,
        })
        return m

    def _get_scu_to_indices(self, index_to_scus: List[Set[int]]) -> Dict[int, List[int]]:
        scu_to_indices = defaultdict(list)
        for i, scus in enumerate(index_to_scus):
            for scu in scus:
                scu_to_indices[scu].append(i)
        return scu_to_indices

    def _get_matches(self,
                     summary_tokens: List[str],
                     summary_indices: List[int],
                     reference_tokens: List[str],
                     reference_indices: List[int]) -> List[Tuple[int, int]]:
        matches = []
        for i in summary_indices:
            for j in reference_indices:
                if summary_tokens[i] == reference_tokens[j]:
                    matches.append((i, j, 1.0))
        return matches

    def _count_tokens_with_scus(self, index_to_scus: List[Set[int]]) -> int:
        return sum([1 for scus in index_to_scus if len(scus) > 0])

    def _compute_scu_rouge(self,
                           summary_tokens: List[str],
                           summary_index_to_scus: List[Set[int]],
                           reference_tokens: List[str],
                           reference_index_to_scus: List[Set[int]]) -> MetricsDict:
        summary_scu_to_indices = self._get_scu_to_indices(summary_index_to_scus)
        reference_scu_to_indices = self._get_scu_to_indices(reference_index_to_scus)

        all_matches = []
        for scu in summary_scu_to_indices.keys():
            summary_indices = summary_scu_to_indices[scu]
            reference_indices = reference_scu_to_indices[scu]
            matches = self._get_matches(summary_tokens, summary_indices,
                                        reference_tokens, reference_indices)
            all_matches.extend(matches)

        intersection = calculate_maximum_matching(all_matches)
        return MetricsDict({
            'intersection': intersection,
            'num_summary_tokens': self._count_tokens_with_scus(summary_index_to_scus),
            'num_reference_tokens': self._count_tokens_with_scus(reference_index_to_scus)
        })

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

    def _compute_non_scu_rouge(self,
                               summary_tokens: List[str],
                               summary_index_to_scus: List[Set[int]],
                               reference_tokens: List[str],
                               reference_index_to_scus: List[Set[int]]) -> MetricsDict:
        summary_scu_to_indices = self._get_scu_to_indices(summary_index_to_scus)
        reference_scu_to_indices = self._get_scu_to_indices(reference_index_to_scus)

        all_matches = []
        # For each SCU, we have to match the summary SCU tokens to any
        # reference token NOT in that SCU and vice versa.
        for scu in summary_scu_to_indices:
            summary_indices = summary_scu_to_indices[scu]
            reference_indices = reference_scu_to_indices[scu]

            summary_complement = self._get_indices_complement(len(summary_tokens), summary_indices)
            reference_complement = self._get_indices_complement(len(reference_tokens), reference_indices)

            matches = self._get_matches(summary_tokens, summary_indices,
                                        reference_tokens, reference_complement)
            all_matches.extend(matches)

            matches = self._get_matches(summary_tokens, summary_complement,
                                        reference_tokens, reference_indices)
            all_matches.extend(matches)

        # Then we have to match any token not part of any SCU in the summary
        # to any token not part of any SCU in the reference.
        summary_indices = self._get_non_scu_indices(summary_index_to_scus)
        reference_indices = self._get_non_scu_indices(reference_index_to_scus)
        matches = self._get_matches(summary_tokens, summary_indices,
                                    reference_tokens, reference_indices)
        all_matches.extend(matches)

        intersection = calculate_maximum_matching(all_matches)
        return MetricsDict({
            'intersection': intersection,
            'num_summary_tokens': len(summary_tokens),
            'num_reference_tokens': len(reference_tokens)
        })

    def _add_pr(self, metrics: MetricsDict) -> None:
        intersection = metrics['intersection']
        num_summary_tokens = metrics['num_summary_tokens']
        num_reference_tokens = metrics['num_reference_tokens']

        precision = 0.0
        if num_summary_tokens != 0.0:
            precision = intersection / num_summary_tokens * 100
        recall = 0.0
        if num_reference_tokens != 0.0:
            recall = intersection / num_reference_tokens * 100
        if precision + recall == 0:
            f1 = 0.0
        else:
            f1 = 2 * (precision * recall) / (precision + recall)

        metrics['precision'] = precision
        metrics['recall'] = recall
        metrics['f1'] = f1

    def _run(self,
             summary: SummaryType,
             annotation: PyramidAnnotation,
             pyramid: Pyramid) -> MetricsDict:
        summary_all_scus_to_offsets = self._get_summary_scu_to_offsets(annotation)

        standard_counts = MetricsDict({'intersection': 0, 'num_summary_tokens': 0, 'num_reference_tokens': 0, 'num_scu_matches': 0, 'num_non_scu_matches': 0})
        scu_counts = MetricsDict({'intersection': 0, 'num_summary_tokens': 0, 'num_reference_tokens': 0})
        non_scu_counts = MetricsDict({'intersection': 0, 'num_summary_tokens': 0, 'num_reference_tokens': 0})

        total_common_scus = 0
        for i, reference in enumerate(pyramid.summaries):
            reference_all_scus_to_offsets = self._get_reference_scu_to_offsets(pyramid, i)
            valid_scus = self._get_scu_intersection(annotation, pyramid, i)
            total_common_scus += len(valid_scus)

            # Take only the SCUs which are common between the summary and reference
            summary_scus_to_offsets = self._filter_scu_to_offsets(summary_all_scus_to_offsets, valid_scus)
            reference_scus_to_offsets = self._filter_scu_to_offsets(reference_all_scus_to_offsets, valid_scus)

            # Tokenize each
            summary_tokens, summary_index_to_scus = self._tokenize(annotation.summary, summary_scus_to_offsets)
            reference_tokens, reference_index_to_scus = self._tokenize(reference, reference_scus_to_offsets)

            # Compute ROUGE
            standard_counts += self._compute_standard_rouge(summary_tokens, summary_index_to_scus,
                                                            reference_tokens, reference_index_to_scus)

            scu_counts += self._compute_scu_rouge(summary_tokens, summary_index_to_scus,
                                                  reference_tokens, reference_index_to_scus)

            non_scu_counts += self._compute_non_scu_rouge(summary_tokens, summary_index_to_scus,
                                                          reference_tokens, reference_index_to_scus)

        avg_common_scus = total_common_scus / len(pyramid.summaries)

        self._add_pr(standard_counts)
        self._add_pr(scu_counts)
        self._add_pr(non_scu_counts)
        return MetricsDict({
            'common_scus': avg_common_scus,
            'standard-rouge': standard_counts,
            'scu-rouge': scu_counts,
            'non-scu-rouge': non_scu_counts,
        })

    def score_multi_all(self,
                        annotations_list: List[List[PyramidAnnotationField]],
                        pyramid_list: List[PyramidField]) -> List[List[MetricsDict]]:
        # Just take the data, not the fields
        summaries_list = [[field.summary for field in fields] for fields in annotations_list]
        annotations_list = [[field.annotation for field in fields] for fields in annotations_list]
        pyramid_list = [field.pyramid for field in pyramid_list]

        metrics_lists = []
        from tqdm import tqdm
        for summaries, annotations, pyramid in tqdm(zip(summaries_list, annotations_list, pyramid_list), total=len(annotations_list)):
            metrics_lists.append([])
            for summary, annotation in zip(summaries, annotations):
                metrics_lists[-1].append(self._run(summary, annotation, pyramid))
        return metrics_lists
