import argparse
import json
import os
import sys

sys.path.append('.')

from sacrerouge.data import MetricsDict


def round_metrics(metrics: MetricsDict, num_digits: int):
    for key, value in metrics.items():
        if isinstance(value, float):
            metrics[key] = round(value, num_digits)
        else:
            round_metrics(value, num_digits)


def calculate_relative_difference(metrics, difference):
    rel_difference = {}
    for key in metrics.keys():
        value = metrics[key]
        diff = difference[key]
        if isinstance(value, float):
            if value == 0.0:
                rel_difference[key] = 0.0
            else:
                rel_difference[key] = diff / value * 100
        else:
            rel_difference[key] = calculate_relative_difference(value, diff)
    return rel_difference


def write_table(metrics1: MetricsDict,
                metrics2: MetricsDict,
                difference: MetricsDict,
                rel_difference: MetricsDict,
                output_path: str) -> None:
    # Put all of the data into tuples and sort by the relative difference
    data = []
    for key in rel_difference.keys():
        if 'f1' in rel_difference[key]:
            value1 = metrics1[key]['f1']
            value2 = metrics2[key]['f1']
            data.append((key, value1, value2, difference[key]['f1'], rel_difference[key]['f1']))
    data.sort(key=lambda t: -t[4])

    # Prepare the lines for writing
    lines = []
    for category, value1, value2, diff, rel_diff in data:
        lines.append(' & '.join([category, f'{value1:.1f}', f'{value2:.1f}', f'{diff:.1f}', f'{rel_diff:.1f}']) + ' \\\\')

    dirname = os.path.dirname(output_path)
    if dirname:
        os.makedirs(dirname, exist_ok=True)
    with open(output_path, 'w') as out:
        out.write('\n'.join(lines))


def main(args):
    metrics1 = MetricsDict(json.loads(open(args.metrics_json1, 'r').read())['metrics'])
    metrics2 = MetricsDict(json.loads(open(args.metrics_json2, 'r').read())['metrics'])

    # If you calculate the differences based on the true values and then do rounding, the results
    # look a little weird in the table because the rounded values no longer make sense.
    # For example (26.61 - 21.95 = 4.66; The table would show 26.6 - 22.0 = 4.7). Therefore, we
    # first round the numbers, then calculate the differences. This shouldn't have any major
    # impact on the results, but it will avoid any confusion from the reader.
    round_metrics(metrics1, 1)
    round_metrics(metrics2, 1)

    difference = metrics2 - metrics1
    rel_difference = calculate_relative_difference(metrics1, difference)

    write_table(metrics1, metrics2, difference, rel_difference, args.output_tex)


if __name__ == '__main__':
    argp = argparse.ArgumentParser()
    argp.add_argument('metrics_json1')
    argp.add_argument('metrics_json2')
    argp.add_argument('output_tex')
    args = argp.parse_args()
    main(args)
