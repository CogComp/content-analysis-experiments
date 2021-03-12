import argparse
import json
import math
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
from typing import List

sys.path.append('.')

from sacrerouge.data import Metrics, MetricsDict
from sacrerouge.io import JsonlReader

plt.rcParams.update({'font.size': 22})


def load_metrics(scores_jsonl: str, summarizer_type: str) -> List[MetricsDict]:
    metrics_list = []
    with JsonlReader(scores_jsonl, Metrics) as f:
        for metrics in f:
            if summarizer_type == 'all' or metrics.summarizer_type == summarizer_type:
                metrics_list.append(metrics.metrics)
    return metrics_list


def get_name(metric: str) -> str:
    if metric == 'rouge':
        return 'ROUGE'
    elif metric == 'bertscore':
        return 'BERTScore'
    raise Exception(metric)


def calculate_scu_contributions(name: str, metrics_list: List[MetricsDict], jackknifing: bool) -> List[float]:
    contributions = []
    for metrics in metrics_list:
        metric = f'{name}-standard_jk' if jackknifing else f'{name}-standard'
        num_scu_matches = metrics[metric]['scu_weight']
        num_non_scu_matches = metrics[metric]['non_scu_weight']
        contribution = num_scu_matches / (num_scu_matches + num_non_scu_matches) * 100
        contributions.append(contribution)
    return contributions


def calculate_scu_coverages(name: str, metrics_list: List[MetricsDict], jackknifing: bool) -> List[float]:
    coverages = []
    for metrics in metrics_list:
        standard_metric = f'{name}-standard_jk' if jackknifing else f'{name}-standard'
        scu_metric = f'{name}-scu_jk' if jackknifing else f'{name}-scu'
        # The number of tokens is the sum over all of the references, so it
        # will be a multiple of 3 (if jackknifing) or 4 (otherwise) of the actual
        # number of tokens. This does not matter because both counts are multiplied
        # by the same factor, so it will cancel out
        num_tokens = metrics[standard_metric]['summary_weight']
        num_scu_tokens = metrics[scu_metric]['summary_weight']
        coverage = num_scu_tokens / num_tokens * 100
        coverages.append(coverage)
    return coverages


def plot_contributions(contributions: List[float], metric: str, output_png: str) -> None:
    name = get_name(metric)
    num_bins = math.ceil(max(contributions) / 5)
    counts, bins = np.histogram(contributions, bins=num_bins)
    counts = counts / sum(counts) * 100
    plt.figure(figsize=(10, 5))
    plt.hist(bins[:-1], bins, weights=counts)
    # plt.hist(contributions, num_bins, density=True)
    plt.xlabel(f'Proportion of {name} explained by SCU matches')
    plt.ylabel('Percent of summaries')
    plt.yticks(list(range(5, 21, 5)))
    plt.ylim(0, 20)
    plt.xlim(0, 65)
    plt.gca().set_xticklabels(['{:.0f}%'.format(x) for x in plt.gca().get_xticks()])
    plt.gca().set_yticklabels(['{:.0f}%'.format(x) for x in plt.gca().get_yticks()])
    print(np.mean(contributions))
    plt.axvline(np.mean(contributions), color='r', linewidth=4)
    plt.tight_layout()

    dirname = os.path.dirname(output_png)
    if dirname:
        os.makedirs(dirname, exist_ok=True)
    plt.savefig(output_png)


def plot_coverages(coverages: List[float], output_png: str) -> None:
    num_bins = math.ceil(max(coverages))
    plt.figure()
    plt.hist(coverages, num_bins, density=True)
    plt.xlabel('Percent of tokens belonging to an SCU')
    plt.ylabel('Percent of summaries')
    plt.gca().set_xticklabels(['{:.0f}%'.format(x) for x in plt.gca().get_xticks()])
    plt.gca().set_yticklabels(['{:.0f}%'.format(x * 100) for x in plt.gca().get_yticks()])
    plt.axvline(np.mean(coverages), color='r')

    dirname = os.path.dirname(output_png)
    if dirname:
        os.makedirs(dirname, exist_ok=True)
    plt.savefig(output_png)


def save_stats(stats, output_file: str) -> None:
    dirname = os.path.dirname(output_file)
    if dirname:
        os.makedirs(dirname, exist_ok=True)
    with open(output_file, 'w') as out:
        out.write(json.dumps(stats, indent=2))


def main(args):
    metrics_list = load_metrics(args.scores_jsonl, args.summarizer_type)

    jackknifing = args.summarizer_type != 'peer'
    contributions = calculate_scu_contributions(args.metric, metrics_list, jackknifing)
    coverages = calculate_scu_coverages(args.metric, metrics_list, jackknifing)

    plot_contributions(contributions, args.metric, args.contribution_png)
    plot_coverages(coverages, args.coverage_png)

    stats = {
        'contribution': {
            'mean': np.mean(contributions),
            'std': np.std(contributions)
        },
        'coverage': {
            'mean': np.mean(coverages),
            'std': np.std(coverages)
        }
    }
    save_stats(stats, args.summary_json)


if __name__ == '__main__':
    argp = argparse.ArgumentParser()
    argp.add_argument('metric')
    argp.add_argument('scores_jsonl')
    argp.add_argument('summarizer_type')
    argp.add_argument('contribution_png')
    argp.add_argument('coverage_png')
    argp.add_argument('summary_json')
    args = argp.parse_args()
    main(args)
