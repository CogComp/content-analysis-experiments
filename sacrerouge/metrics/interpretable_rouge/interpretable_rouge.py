import os
import spacy
from collections import defaultdict
from spacy.tokens import Token
from nltk.stem import PorterStemmer
from tqdm import tqdm
from typing import Dict, List, Set, Tuple

from sacrerouge.common import DATA_ROOT
from sacrerouge.data import MetricsDict
from sacrerouge.data.fields import ReferencesField, SummaryField
from sacrerouge.data.jackknifers import ReferencesJackknifer
from sacrerouge.data.types import SummaryType
from sacrerouge.metrics import Metric
from sacrerouge.metrics.interpretable_rouge.backends import Backend
from sacrerouge.metrics.interpretable_rouge.matchers import Matcher

Token.set_extension('matching_text', default=None)
Token.set_extension('is_matchable', default=None)
Token.set_extension('index', default=None)
Token.set_extension('is_np', default=None)


@Metric.register('interpretable-rouge')
class InterpretableRouge(Metric):
    def __init__(self,
                 name: str,
                 backend: Backend,
                 matchers: List[Matcher],
                 rouge_data_dir: str = f'{DATA_ROOT}/metrics/ROUGE-1.5.5/data',
                 remove_stopwords: bool = False,
                 use_porter_stemmer: bool = True,
                 pretokenized_text: bool = False) -> None:
        super().__init__(['references'], jackknifer=ReferencesJackknifer())
        self.name = name
        self.backend = Backend.from_params(backend)
        self.matchers = [Matcher.from_params(params) for params in matchers]
        self.remove_stopwords = remove_stopwords
        self.use_porter_stemmer = use_porter_stemmer
        self.pretokenized_text = pretokenized_text

        self.stemmer = PorterStemmer(PorterStemmer.ORIGINAL_ALGORITHM)
        self.stemmer_exceptions = self._load_stemmer_exceptions(rouge_data_dir)
        self.stopwords = self._load_stopwords(rouge_data_dir)
        self.nlp = spacy.load('en_core_web_sm')

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

    def _preprocess_summary(self, summary: SummaryType) -> List[Token]:
        if isinstance(summary, str):
            summary = [summary]

        tokens = []
        for sentence in summary:
            if self.pretokenized_text:
                doc = self.nlp.tokenizer.tokens_from_list(sentence.split())
                self.nlp.tagger(doc)
                self.nlp.parser(doc)
                self.nlp.entity(doc)
            else:
                doc = self.nlp(sentence)

            for token in doc:
                text = token.text.lower()
                # There are some whitespace only tokens. I don't know where they come from
                if len(text.strip()) == 0:
                    continue

                is_matchable = True
                if self.remove_stopwords and text in self.stopwords:
                    is_matchable = False
                elif self.use_porter_stemmer and len(text) > 3:
                    if text in self.stemmer_exceptions:
                        text = self.stemmer_exceptions[text]
                    else:
                        text = self.stemmer.stem(text)

                token._.matching_text = text
                token._.is_matchable = is_matchable and not token.is_punct
                token._.index = len(tokens)
                tokens.append(token)

            for chunk in doc.noun_chunks:
                for token in chunk:
                    token._.is_np = True

        return tokens

    def _preprocess_all(self, summaries_list: List[List[str]]):
        tokens_lists = []
        for summaries in tqdm(summaries_list, desc='Preprocessing'):
            tokens_lists.append([])
            for summary in summaries:
                tokens_lists[-1].append(self._preprocess_summary(summary))
        return tokens_lists

    def _count_matchable_tokens(self, tokens: List[Token]) -> int:
        return sum([1 for token in tokens if token._.is_matchable])

    def _run_metric(self,
                    summary_tokens: List[Token],
                    reference_tokens_list: List[List[Token]],
                    matches_list: List[List[Tuple[int, int, float]]],
                    token_weights_list: List[List[float]],
                    metric: str):
        total_weight = 0
        total_normalization_weight = 0

        total_matches = 0
        content_type_to_total_matches = defaultdict(float)
        matcher_metrics = [[] for _ in self.matchers]
        for reference_tokens, matches, weights in zip(reference_tokens_list, matches_list, token_weights_list):
            total_weight += self.backend.get_total_weight(matches)
            total_normalization_weight += sum(weights)

            all_matches = []
            content_type_to_matches = defaultdict(list)
            for i, matcher in enumerate(self.matchers):
                category_matches, metrics = matcher.select_matches(summary_tokens, reference_tokens, matches, weights, metric, self.backend)
                content_type_to_matches[matcher.content_type].extend(category_matches)
                all_matches.extend(category_matches)
                matcher_metrics[i].append(metrics)

            total_matches += self.backend.get_total_weight(all_matches)
            for content_type, content_matches in content_type_to_matches.items():
                content_type_to_total_matches[content_type] += self.backend.get_total_weight(content_matches)

        # Compute the aggregated metrics for each matcher
        metrics = MetricsDict()
        for matcher, metrics_list in zip(self.matchers, matcher_metrics):
            metrics.update(matcher.finalize(metrics_list, total_weight, metric))

        # Add the standard rouge score
        measure = 0.0
        if total_normalization_weight != 0.0:
            measure = total_weight / total_normalization_weight * 100
        metrics[self.name] = {
            f'{metric}_total_weight': total_weight,
            f'{metric}_total_norm_weight': total_normalization_weight,
            metric: measure
        }

        # Compute the metric for just the edges that the categories selected
        measure = 0.0
        if total_normalization_weight != 0.0:
            measure = total_matches / total_normalization_weight * 100

        coverage = 0.0
        if total_weight != 0.0:
            coverage = total_matches / total_weight * 100

        # Calculate each content type coverage
        for content_type, content_total_matches in content_type_to_total_matches.items():
            content_coverage = 0.0
            if total_weight != 0.0:
                content_coverage = content_total_matches / total_weight * 100
            metrics[f'{metric}_content-coverages'][content_type] = content_coverage

        metrics[f'interpretable-{self.name}'] = {
            f'{metric}_total_weight': total_matches,
            f'{metric}_total_norm_weight': total_normalization_weight,
            metric: measure,
            f'{metric}_coverage': coverage
        }

        return metrics

    def _combine_metrics(self, recall_metrics: MetricsDict, precision_metrics: MetricsDict) -> MetricsDict:
        combined = MetricsDict()
        combined.update(recall_metrics)
        combined.update(precision_metrics)

        for key in combined.keys():
            if 'precision' and 'recall' in combined[key]:
                precision = combined[key]['precision']
                recall = combined[key]['recall']
                f1 = 0.0
                if precision + recall != 0.0:
                    f1 = 2 * (precision * recall) / (precision + recall)
                combined[key]['f1'] = f1

        return combined

    def _run(self,
             summary_tokens: List[Token],
             precision_reference_tokens_list: List[Token],
             recall_reference_tokens_list: List[Token],
             precision_matches_list: List[List[Tuple[int, int, float]]],
             recall_matches_list: List[List[Tuple[int, int, float]]],
             precision_weights_list: List[float],
             recall_weights_list: List[List[float]]) -> MetricsDict:

        recall_metrics = self._run_metric(summary_tokens,
                                          recall_reference_tokens_list,
                                          recall_matches_list,
                                          recall_weights_list,
                                          'recall')

        num_references = len(precision_reference_tokens_list)
        precision_weights_list = [precision_weights_list] * num_references
        precision_metrics = self._run_metric(summary_tokens,
                                             precision_reference_tokens_list,
                                             precision_matches_list,
                                             precision_weights_list,
                                             'precision')

        output_metrics = self._combine_metrics(recall_metrics, precision_metrics)
        return output_metrics

    def score_multi_all(self,
                        summaries_list: List[List[SummaryField]],
                        references_list: List[ReferencesField]) -> List[List[MetricsDict]]:
        # Just take the summaries themselves, not the fields
        summaries_list = [[field.summary for field in fields] for fields in summaries_list]
        references_list = [field.references for field in references_list]

        # Preprocess the summaries
        summary_tokens_lists = self._preprocess_all(summaries_list)
        reference_tokens_lists = self._preprocess_all(references_list)

        input_summary_tokens_list, input_reference_precision_tokens_lists, input_reference_recall_tokens_lists, \
                precision_matches_list, recall_matches_lists, \
                precision_weights_list, recall_weights_lists = \
            self.backend.get_matches_list(summary_tokens_lists, reference_tokens_lists)

        index = 0
        results_list = []
        for summaries in summaries_list:
            results_list.append([])
            for summary in summaries:
                results_list[-1].append(self._run(input_summary_tokens_list[index],
                                                  input_reference_precision_tokens_lists[index],
                                                  input_reference_recall_tokens_lists[index],
                                                  precision_matches_list[index],
                                                  recall_matches_lists[index],
                                                  precision_weights_list[index],
                                                  recall_weights_lists[index]))
                index += 1

        return results_list