"""
Evaluation utilities for measuring reviewer performance.
"""
from src.review_agents.reviewer import review
from src.utils.matching import find_best_match, match_findings


def evaluate_reviewers(prs: list, custom_prompt: str = None) -> dict:
    """
    Evaluate both diff-only and context-aware reviewers across all PRs.

    Args:
        prs: List of PR dictionaries

    Returns:
        Dictionary with evaluation metrics for both approaches
    """
    diff_only_results = []
    context_aware_results = []

    print(f"Evaluating {len(prs)} PRs...")

    for i, pr in enumerate(prs, 1):
        print(f"  Processing PR {i}/{len(prs)}: {pr['id']}")

        # Review with both approaches - ONLY difference is context and task info!
        diff_only_review = review(pr, context=None, task_context=None, custom_prompt=custom_prompt)
        context_aware_review = review(pr, context=pr.get('context', ''), task_context=pr.get('task_context', ''), custom_prompt=custom_prompt)

        # Evaluate findings
        diff_only_results.append(evaluate_single_review(diff_only_review, pr))
        context_aware_results.append(evaluate_single_review(context_aware_review, pr))

    # Calculate aggregate metrics
    diff_only_metrics = calculate_metrics(diff_only_results)
    context_aware_metrics = calculate_metrics(context_aware_results)

    return {
        'diff_only': diff_only_metrics,
        'context_aware': context_aware_metrics,
        'detailed_results': {
            'diff_only': diff_only_results,
            'context_aware': context_aware_results
        }
    }


def evaluate_single_review(review: dict, pr: dict) -> dict:
    """
    Evaluate a single review against expected issues.

    Uses the unified matching logic from src.utils.matching.

    Args:
        review: Review results with findings
        pr: PR dictionary with expected_issues

    Returns:
        Dictionary with true_positives, false_positives, false_negatives
    """
    found_issues = [f['issue'] for f in review.get('findings', [])]
    expected_issues = pr['expected_issues']

    # Calculate matches using unified matching logic
    matched_expected = set()
    matched_found = set()

    for i, expected in enumerate(expected_issues):
        # Find best match among found issues
        match_idx, match_score = find_best_match(expected, found_issues, threshold=0.3)

        if match_idx is not None and match_idx not in matched_found:
            matched_expected.add(i)
            matched_found.add(match_idx)

    true_positives = len(matched_expected)
    false_negatives = len(expected_issues) - true_positives
    false_positives = len(found_issues) - len(matched_found)

    return {
        'pr_id': pr['id'],
        'true_positives': true_positives,
        'false_positives': false_positives,
        'false_negatives': false_negatives,
        'expected_count': len(expected_issues),
        'found_count': len(found_issues)
    }


def calculate_metrics(results: list) -> dict:
    """
    Calculate aggregate precision, recall, and F1 score.

    Args:
        results: List of evaluation results

    Returns:
        Dictionary with precision, recall, f1 scores
    """
    total_tp = sum(r['true_positives'] for r in results)
    total_fp = sum(r['false_positives'] for r in results)
    total_fn = sum(r['false_negatives'] for r in results)

    # Calculate metrics
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'true_positives': total_tp,
        'false_positives': total_fp,
        'false_negatives': total_fn
    }


def calculate_metrics_from_results(results: list) -> dict:
    """
    Calculate metrics from a list of review results (Notebook 2 format).

    Args:
        results: List of dicts with 'review', 'expected' keys

    Returns:
        Dictionary with precision, recall, f1 scores
    """
    total_tp = 0
    total_fp = 0
    total_fn = 0

    for result in results:
        findings = result['review']['findings']
        expected = result['expected']

        matched_indices = match_findings(findings, expected)

        tp = len(matched_indices)
        fp = len(findings) - tp
        fn = len(expected) - tp

        total_tp += tp
        total_fp += fp
        total_fn += fn

    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return {'precision': precision, 'recall': recall, 'f1': f1}
