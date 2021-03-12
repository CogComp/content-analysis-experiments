from typing import Tuple


def calculate_pr_f1(reference_total: int, summary_total: int, intersection: int) -> Tuple[float, float, float]:
    precision = 0.0
    if summary_total != 0.0:
        precision = intersection / summary_total * 100
    recall = 0.0
    if reference_total != 0.0:
        recall = intersection / reference_total * 100
    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * (precision * recall) / (precision + recall)
    return precision, recall, f1
