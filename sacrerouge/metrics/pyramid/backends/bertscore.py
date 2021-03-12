from bert_score import create_idf_dict, score
from spacy.lang.en import English
from spacy.tokens import Token
from typing import Dict, List, Set, Tuple

from sacrerouge.data import MetricsDict
from sacrerouge.metrics.pyramid.backends import Backend


@Backend.register('bertscore-pyramid-backend')
class BertScorePyramidBackend(Backend):
    def __init__(self, idf: bool = False):
        self.idf = idf
        nlp = English()
        self.tokenizer = nlp.Defaults.create_tokenizer(nlp)

    def tokenize(self, summary: str, scu_to_offsets: Dict[int, List[Tuple[int, int]]]):
        tokens = []
        index_to_scus = []
        for token in self.tokenizer(summary):
            # There are some whitespace only tokens. I don't know where they come from
            if len(str(token).strip()) == 0:
                continue
            tokens.append(str(token))

            scus = set()
            for scu_id, offsets_list in scu_to_offsets.items():
                for start, end in offsets_list:
                    if start <= token.idx and token.idx < end:
                        scus.add(scu_id)
                        break
            index_to_scus.append(scus)
        return tokens, index_to_scus

    def _get_summary_strings(self, tokens: List[Token]) -> str:
        return ' '.join(list(map(lambda token: str(token), tokens)))

    def _get_dummy_token(self, text: str):
        doc = self.tokenizer.tokens_from_list([text])
        return doc[0]

    def get_matches_list(self,
                         summary_tokens_list: List[str],
                         reference_tokens_lists: List[List[str]],
                         summary_index_to_scus_list,
                         reference_index_to_scus_lists):
        summaries_list = [self._get_summary_strings(summary_tokens) for summary_tokens in summary_tokens_list]
        references_list = [[self._get_summary_strings(reference_tokens) for reference_tokens in reference_tokens_list] for reference_tokens_list in reference_tokens_lists]

        if self.idf:
            unique_references = list(set(reference for references in references_list for reference in references))
            idf = create_idf_dict(unique_references, lang='en')
        else:
            idf = False

        input_candidates = []
        input_references = []
        empty_inputs = set()
        for i, (summary, references) in enumerate(zip(summaries_list, references_list)):
            if len(summary) == 0:
                empty_inputs.add(i)
            else:
                input_candidates.append(summary)
                input_references.append(references)

        # Score the summaries
        precisions, recalls, f1s, precision_alignments_list, recall_alignments_list, precision_indices, recall_indices, precision_weights, recall_weights \
            = score(input_candidates, input_references,
                    lang='en',
                    idf=idf,
                    return_alignments=True)

        output_summary_index_to_scus_list = []
        output_precision_reference_index_to_scu_lists = []
        output_recall_reference_index_to_scu_lists = []
        output_precision_matches_lists = []
        output_recall_matches_lists = []
        output_precision_weights_list = []
        output_recall_weights_lists = []

        bos = self._get_dummy_token('<s>')
        eos = self._get_dummy_token('</s>')

        index = 0
        for i in range(len(summaries_list)):
            summary_tokens = summary_tokens_list[i]
            reference_tokens_list = reference_tokens_lists[i]
            summary_index_to_scus = summary_index_to_scus_list[i]
            reference_index_to_scus_list = reference_index_to_scus_lists[i]

            if i in empty_inputs:
                output_precision_reference_index_to_scu_lists.append([])
                output_recall_reference_index_to_scu_lists.append([])
                output_precision_matches_lists.append([])
                output_recall_matches_lists.append([])
                output_precision_weights_list.append([])
                output_recall_weights_lists.append([])
            else:
                precision_index = precision_indices[index]
                recall_index = recall_indices[index]

                p_weights = precision_weights[index]
                if len(p_weights) < len(summary_tokens) + 2:
                    print(f'Warning: Truncating summary from {len(summary_tokens)} to {len(p_weights) - 2} tokens')
                    summary_index_to_scus = summary_index_to_scus[:len(p_weights) - 2]

                # The BertScore code adds <s> and </s> tokens to the start and end of the sequence. Some
                # tokens are aligned to these special tokens. We also add empty sets for the tokens' mapping to SCUs. All
                # of the other tokens will shift accordingly
                output_summary_index_to_scus_list.append([set()] + summary_index_to_scus + [set()])

                r_weights = recall_weights[index]
                recall_reference_tokens = reference_tokens_list[recall_index]
                reference_index_to_scus = reference_index_to_scus_list[recall_index]
                if len(r_weights) < len(recall_reference_tokens) + 2:
                    print(f'Warning: Truncating reference from {len(recall_reference_tokens)} to {len(r_weights) - 2} tokens')
                    reference_index_to_scus = reference_index_to_scus[:len(r_weights) - 2]

                output_precision_reference_index_to_scu_lists.append([[set()] + reference_index_to_scus_lists[i][precision_index] + [set()]])
                output_recall_reference_index_to_scu_lists.append([[set()] + reference_index_to_scus + [set()]])

                output_precision_matches_lists.append([precision_alignments_list[index]])
                output_recall_matches_lists.append([recall_alignments_list[index]])

                output_precision_weights_list.append(p_weights)
                output_recall_weights_lists.append([r_weights])

                index += 1

        return output_summary_index_to_scus_list, \
               output_precision_reference_index_to_scu_lists, output_recall_reference_index_to_scu_lists, \
               output_precision_matches_lists, output_recall_matches_lists, \
               output_precision_weights_list, output_recall_weights_lists

    def get_total_weight(self, matches: List[Tuple[int, int, float]]) -> float:
        unique_matches = set(matches)
        score = sum(weight for _, _, weight in unique_matches)
        return score

    def calculate_standard_metric(self,
                                  summary_index_to_scus: List[Set[int]],
                                  reference_index_to_scus: List[Set[int]],
                                  summary_weights: List[float],
                                  reference_weights: List[float],
                                  matches: List[Tuple[int, int, float]]) -> MetricsDict:
        total_weight = sum(w for _, _, w in matches)
        scu_weight = 0
        non_scu_weight = 0

        for i, j, weight in matches:
            if len(summary_index_to_scus[i] & reference_index_to_scus[j]) > 0:
                scu_weight += weight
            else:
                non_scu_weight += weight

        return MetricsDict({
            'weight': total_weight,
            'summary_weight': sum(summary_weights),
            'reference_weight': sum(reference_weights),
            'scu_weight': scu_weight,
            'non_scu_weight': non_scu_weight,
        })

    def calculate_scu_metric(self,
                             summary_index_to_scus: List[Set[int]],
                             reference_index_to_scus: List[Set[int]],
                             summary_weights: List[float],
                             reference_weights: List[float],
                             matches: List[Tuple[int, int, float]]) -> MetricsDict:
        total_weight = 0
        for i, j, weight in matches:
            if len(summary_index_to_scus[i] & reference_index_to_scus[j]) > 0:
                total_weight += weight
        return MetricsDict({
            'weight': total_weight,
            'summary_weight': self._sum_scu_token_weight(summary_index_to_scus, summary_weights),
            'reference_weight': self._sum_scu_token_weight(reference_index_to_scus, reference_weights)
        })

    def calculate_non_scu_metric(self,
                                 summary_index_to_scus: List[Set[int]],
                                 reference_index_to_scus: List[Set[int]],
                                 summary_weights: List[float],
                                 reference_weights: List[float],
                                 matches: List[Tuple[int, int, float]]) -> MetricsDict:
        total_weight = 0
        for i, j, weight in matches:
            if len(summary_index_to_scus[i] & reference_index_to_scus[j]) == 0:
                total_weight += weight
        return MetricsDict({
            'weight': total_weight,
            'summary_weight': sum(summary_weights),
            'reference_weight': sum(reference_weights)
        })