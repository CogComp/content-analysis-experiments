import re
from collections import defaultdict
from typing import Dict, List, Set, Tuple

from sacrerouge.data import MetricsDict, Pyramid, PyramidAnnotation
from sacrerouge.data.fields import PyramidField, PyramidAnnotationField
from sacrerouge.data.jackknifers import PyramidJackknifer
from sacrerouge.metrics import Metric
from sacrerouge.metrics.pyramid.backends import Backend


@Metric.register('pyramid-comparison')
class PyramidComparison(Metric):
    _non_alphanumeric_regex = re.compile('[^A-Za-z0-9]')

    def __init__(self, name: str, backend: Backend) -> None:
        super().__init__(['pyramid'], jackknifer=PyramidJackknifer())
        self.name = name
        self.backend = Backend.from_params(backend)

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

    def _count_tokens_with_scus(self, index_to_scus: List[Set[int]]) -> int:
        return sum([1 for scus in index_to_scus if len(scus) > 0])

    def _get_scu_intersection(self,
                              summary_index_to_scus: List[Set[int]],
                              reference_index_to_scus: List[Set[int]]) -> Set[int]:
        summary_scus = set()
        reference_scus = set()
        for scus in summary_index_to_scus:
            summary_scus |= scus
        for scus in reference_index_to_scus:
            reference_scus |= scus
        return summary_scus & reference_scus

    def _filter_index_to_scus(self, index_to_scus: List[Set[int]], valid_scus: Set[int]) -> List[Set[int]]:
        filtered = []
        for scus in index_to_scus:
            filtered.append(scus & valid_scus)
        return filtered

    def _add_pr(self, metrics: MetricsDict) -> None:
        intersection = metrics['weight']
        num_summary_tokens = metrics['summary_weight']
        num_reference_tokens = metrics['reference_weight']

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
             summary_index_to_scus: List[Set[int]],
             reference_index_to_scus_list: List[List[Set[int]]],
             matches_list: List[List[Tuple[int, int, float]]],
             precision_weights: List[float],
             recall_weights_list: List[List[float]]):

        standard_counts = MetricsDict({'weight': 0, 'summary_weight': 0, 'reference_weight': 0, 'scu_weight': 0, 'non_scu_weight': 0})
        scu_counts = MetricsDict({'weight': 0, 'summary_weight': 0, 'reference_weight': 0})
        non_scu_counts = MetricsDict({'weight': 0, 'summary_weight': 0, 'reference_weight': 0})

        for matches, reference_index_to_scus, recall_weights in zip(matches_list, reference_index_to_scus_list, recall_weights_list):
            # Filter the SCUs to just those which the summary and reference have in common
            valid_scus = self._get_scu_intersection(summary_index_to_scus, reference_index_to_scus)
            this_summary_index_to_scus = self._filter_index_to_scus(summary_index_to_scus, valid_scus)
            this_reference_index_to_scus = self._filter_index_to_scus(reference_index_to_scus, valid_scus)

            standard_counts += self.backend.calculate_standard_metric(this_summary_index_to_scus, this_reference_index_to_scus, precision_weights, recall_weights, matches)
            scu_counts += self.backend.calculate_scu_metric(this_summary_index_to_scus, this_reference_index_to_scus, precision_weights, recall_weights, matches)
            non_scu_counts += self.backend.calculate_non_scu_metric(this_summary_index_to_scus, this_reference_index_to_scus, precision_weights, recall_weights, matches)

        self._add_pr(standard_counts)
        self._add_pr(scu_counts)
        self._add_pr(non_scu_counts)
        return MetricsDict({
            f'{self.name}-standard': standard_counts,
            f'{self.name}-scu': scu_counts,
            f'{self.name}-non-scu': non_scu_counts
        })

    def score_multi_all(self,
                        annotations_list: List[List[PyramidAnnotationField]],
                        pyramid_list: List[PyramidField]) -> List[List[MetricsDict]]:
        # Just take the data, not the fields
        summaries_list = [[field.summary for field in fields] for fields in annotations_list]
        annotations_list = [[field.annotation for field in fields] for fields in annotations_list]
        pyramid_list = [field.pyramid for field in pyramid_list]

        # Prepare BertScore input
        input_summaries = []
        input_references_list = []
        input_summary_index_to_scus_list = []
        input_reference_index_to_scus_lists = []

        for summaries, annotations, pyramid in zip(summaries_list, annotations_list, pyramid_list):
            references = []
            reference_index_to_scus_list = []
            for i, reference in enumerate(pyramid.summaries):
                reference_scus_to_offsets = self._get_reference_scu_to_offsets(pyramid, i)
                reference_tokens, reference_index_to_scus = self.backend.tokenize(reference, reference_scus_to_offsets)
                references.append(reference_tokens)
                reference_index_to_scus_list.append(reference_index_to_scus)

            for annotation in annotations:
                summary_scus_to_offsets = self._get_summary_scu_to_offsets(annotation)
                summary_tokens, summary_index_to_scus = self.backend.tokenize(annotation.summary, summary_scus_to_offsets)

                input_summaries.append(summary_tokens)
                input_references_list.append(references)
                input_summary_index_to_scus_list.append(summary_index_to_scus)
                input_reference_index_to_scus_lists.append(reference_index_to_scus_list)

        summary_index_to_scus_list, precision_reference_index_to_scus_lists, recall_reference_index_to_scus_lists, \
                precision_matches_lists, recall_matches_lists, precision_weights_list, recall_weights_lists = \
            self.backend.get_matches_list(input_summaries, input_references_list,
                                          input_summary_index_to_scus_list, input_reference_index_to_scus_lists)

        index = 0
        results_lists = []
        for annotations in annotations_list:
            results_lists.append([])
            for _ in annotations:
                results_lists[-1].append(self._run(summary_index_to_scus_list[index],
                                                   recall_reference_index_to_scus_lists[index],
                                                   recall_matches_lists[index],
                                                   precision_weights_list[index],
                                                   recall_weights_lists[index]))
                index += 1
        return results_lists
