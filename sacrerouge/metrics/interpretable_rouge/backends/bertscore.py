import spacy
from bert_score import create_idf_dict, score
from spacy.tokens import Token
from typing import List, Tuple

from sacrerouge.metrics.interpretable_rouge.backends import Backend


@Backend.register('bertscore-backend')
class BertScoreBackend(Backend):
    def __init__(self, idf: bool):
        self.idf = idf
        self.nlp = spacy.load('en_core_web_sm')

    def _get_summary_strings(self, tokens: List[Token]) -> str:
        return ' '.join(list(map(lambda token: str(token), tokens)))

    def _get_dummy_token(self, text: str):
        doc = self.nlp.tokenizer.tokens_from_list([text])
        return doc[0]

    def get_matches_list(self,
                         summary_tokens_lists: List[List[Token]],
                         reference_tokens_lists: List[List[Token]]):
        summaries_list = [[self._get_summary_strings(summary_tokens) for summary_tokens in summary_tokens_list] for summary_tokens_list in summary_tokens_lists]
        references_list = [[self._get_summary_strings(reference_tokens) for reference_tokens in reference_tokens_list] for reference_tokens_list in reference_tokens_lists]

        if self.idf:
            unique_references = list(set(reference for references in references_list for reference in references))
            idf = create_idf_dict(unique_references, lang='en')
        else:
            idf = False

        input_candidates = []
        input_references = []
        empty_inputs = set()
        for i, (summaries, references) in enumerate(zip(summaries_list, references_list)):
            for j, summary in enumerate(summaries):
                if len(summary) == 0:
                    empty_inputs.add((i, j))
                else:
                    input_candidates.append(summary)
                    input_references.append(references)

        # Score the summaries
        precisions, recalls, f1s, precision_alignments_list, recall_alignments_list, precision_indices, recall_indices, precision_weights, recall_weights \
            = score(input_candidates, input_references,
                    lang='en',
                    idf=idf,
                    return_alignments=True)

        bos = self._get_dummy_token('<s>')
        eos = self._get_dummy_token('</s>')

        index = 0
        output_summary_tokens_list = []
        output_reference_precision_tokens_lists = []
        output_reference_recall_tokens_lists = []
        output_matches_precision_lists = []
        output_matches_recall_lists = []
        output_precision_weights_list = []
        output_recall_weights_lists = []
        for i, (summary_tokens_list, reference_tokens_list) in enumerate(zip(summary_tokens_lists, reference_tokens_lists)):
            for j, summary_tokens in enumerate(summary_tokens_list):
                if (i, j) in empty_inputs:
                    output_summary_tokens_list.append([])
                    output_reference_precision_tokens_lists.append([[]])
                    output_reference_recall_tokens_lists.append([[]])
                    output_matches_precision_lists.append([[]])
                    output_matches_recall_lists.append([[]])
                    output_precision_weights_list.append([])
                    output_recall_weights_lists.append([[]])
                else:
                    # The BertScore tokenizer only allows for 512 subword tokens (including <s> and </s>).
                    # This potentially means we need to truncate input summary
                    p_weight = precision_weights[index]
                    if len(p_weight) < len(summary_tokens) + 2:
                        print(f'Warning: Truncating summary from {len(summary_tokens)} to {len(p_weight) - 2} tokens')
                        summary_tokens = summary_tokens[:len(p_weight) - 2]

                    # The BertScore code adds <s> and </s> tokens to the start and end of the sequence. Some
                    # tokens are aligned to these special tokens. To be faithful to the original code, we allow for this
                    # by adding dummy tokens to the summaries.
                    output_summary_tokens_list.append([bos] + summary_tokens + [eos])

                    precision_index = precision_indices[index]
                    recall_index = recall_indices[index]

                    r_weight = recall_weights[index]
                    recall_reference_tokens = reference_tokens_list[recall_index]
                    if len(r_weight) < len(recall_reference_tokens) + 2:
                        print(f'Warning: Truncating reference from {len(recall_reference_tokens)} to {len(r_weight) - 2} tokens')
                        recall_reference_tokens = recall_reference_tokens[:len(r_weight) - 2]

                    output_reference_precision_tokens_lists.append([[bos] + reference_tokens_list[precision_index] + [eos]])
                    output_reference_recall_tokens_lists.append([[bos] + recall_reference_tokens + [eos]])

                    output_matches_precision_lists.append([precision_alignments_list[index]])
                    output_matches_recall_lists.append([recall_alignments_list[index]])

                    output_precision_weights_list.append(p_weight)
                    output_recall_weights_lists.append([r_weight])

                    index += 1

        return output_summary_tokens_list, output_reference_precision_tokens_lists, output_reference_recall_tokens_lists, \
               output_matches_precision_lists, output_matches_recall_lists, \
               output_precision_weights_list, output_recall_weights_lists

    def get_total_weight(self, matches: List[Tuple[int, int, float]]) -> float:
        unique_matches = set(matches)
        score = sum(weight for _, _, weight in unique_matches)
        return score
