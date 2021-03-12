import argparse
import numpy as np
import os
import sys
from collections import defaultdict
from typing import Dict, List

sys.path.append('.')

from sacrerouge.data import Metrics, MetricsDict
from sacrerouge.io import JsonlReader


def load_metrics(scores_jsonl: str, summarizer_type: str) -> List[MetricsDict]:
    metrics_list = []
    with JsonlReader(scores_jsonl, Metrics) as f:
        for metrics in f:
            if summarizer_type == 'all' or metrics.summarizer_type == summarizer_type:
                metrics_list.append(metrics.metrics)
    return metrics_list


def calculate_coverages(metrics_list: List[MetricsDict], jackknifing: bool) -> Dict[str, float]:
    category_to_coverage_list = defaultdict(list)
    for metrics in metrics_list:
        for name in metrics.keys():
            if name in ['rouge', 'bertscore']:
                continue

            if (name.endswith('_jk') and jackknifing) or (not name.endswith('_jk') and not jackknifing):
                if 'recall_coverage' in metrics[name]:
                    coverage = metrics[name]['recall_coverage']
                    if jackknifing:
                        name = name[:-3]
                    if name.startswith('interpretable-'):
                        # This is all of the categories together
                        name = 'all'
                    category_to_coverage_list[name].append(coverage)

    category_to_coverage = {}
    for category, coverage_list in category_to_coverage_list.items():
        category_to_coverage[category] = np.mean(coverage_list)
    return category_to_coverage


def write_coverages_table(category_to_coverage: Dict[str, float], output_file: str) -> None:
    all_coverage = category_to_coverage['all']
    del category_to_coverage['all']

    # Sort them by coverage, descending
    coverages = list(category_to_coverage.items())
    coverages.sort(key=lambda t: -t[1])

    dirname = os.path.dirname(output_file)
    if dirname:
        os.makedirs(dirname, exist_ok=True)
    with open(output_file, 'w') as out:
        out.write('\\begin{tabular}{cc}\n')
        out.write('Category & Coverage \\\\\n')
        out.write('\\midrule\n')
        out.write(f'all & {all_coverage:.1f}\\% \\\\\n')
        out.write('\\midrule\n')
        for category, coverage in coverages:
            out.write(f'{category}, {coverage:.1f}\\% \\\\\n')
        out.write('\\end{tabular}\n')


def calculate_content_coverage(metrics_list: List[MetricsDict], jackknifing: bool) -> Dict[str, float]:
    content_to_coverage_list = defaultdict(list)
    for metrics in metrics_list:
        key = 'recall_content-coverages_jk' if jackknifing else 'recall_content-coverages'
        for name, coverage in metrics[key].items():
            content_to_coverage_list[name].append(coverage)
    return {name: np.mean(coverages) for name, coverages in content_to_coverage_list.items()}


def write_content_table(content_coverages, output_file: str) -> None:
    dirname = os.path.dirname(output_file)
    if dirname:
        os.makedirs(dirname, exist_ok=True)
    with open(output_file, 'w') as out:
        for name, coverage in content_coverages.items():
            out.write(f'{name}\t{coverage}\n')


def main(args):
    metrics_list = load_metrics(args.scores_jsonl, args.summarizer_type)
    jackknifing = args.summarizer_type != 'peer'
    category_to_coverage = calculate_coverages(metrics_list, jackknifing)
    write_coverages_table(category_to_coverage, args.output_tex)

    content_coverages = calculate_content_coverage(metrics_list, jackknifing)
    write_content_table(content_coverages, args.content_output_tex)


if __name__ == '__main__':
    argp = argparse.ArgumentParser()
    argp.add_argument('scores_jsonl')
    argp.add_argument('summarizer_type')
    argp.add_argument('output_tex')
    argp.add_argument('content_output_tex')
    args = argp.parse_args()
    main(args)
