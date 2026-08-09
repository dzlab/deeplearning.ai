"""
Shared evaluation helpers for notebooks.
Extracted from Lab 2 to avoid code duplication in Lab 3.
"""
import pandas as pd
from src.utils.matching import match_findings


def calculate_metrics(results):
    """
    Calculate precision, recall, and F1 scores from evaluation results.

    Args:
        results: List of dicts with 'review' and 'expected' keys

    Returns:
        Dict with 'precision', 'recall', 'f1' keys

    Example:
        >>> results = [
        ...     {'review': {'findings': [...]}, 'expected': ['issue1', 'issue2']},
        ...     ...
        ... ]
        >>> metrics = calculate_metrics(results)
        >>> print(f"F1: {metrics['f1']:.2%}")
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

    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'tp': total_tp,
        'fp': total_fp,
        'fn': total_fn
    }


def evaluate_agent_on_prs(prs, review_fn, context_fn=None):
    """
    Run an agent on all PRs and return results.

    Args:
        prs: List of PR dictionaries
        review_fn: Function that takes (pr, context) and returns review results
        context_fn: Optional function that takes (pr) and returns context string.
                   If None, review_fn should handle context internally.

    Returns:
        List of results with 'pr_id', 'review', 'expected' keys

    Example:
        >>> def my_reviewer(pr, context):
        ...     return review(pr, context=context)
        >>> results = evaluate_agent_on_prs(all_prs, my_reviewer, lambda pr: retriever.retrieve_for_pr(pr))
    """
    results = []

    for pr in prs:
        if context_fn:
            context = context_fn(pr)
            review_result = review_fn(pr, context)
        else:
            review_result = review_fn(pr)

        results.append({
            'pr_id': pr['id'],
            'review': review_result,
            'expected': pr['expected_issues']
        })

    return results


def evaluate_general_and_ensemble(prs: list, retriever) -> tuple:
    """
    Run the general reviewer and the specialized ensemble across all PRs,
    sharing the same retrieved context so the comparison is apples-to-apples.

    Args:
        prs: List of PR dictionaries
        retriever: ContextRetriever used to fetch context for each PR

    Returns:
        Tuple of (results_general, results_ensemble), each a list of dicts
        with 'pr_id', 'review', 'expected' keys
    """
    from src.review_agents.reviewer import review
    from src.review_agents.specialized import review_ensemble

    results_general, results_ensemble = [], []

    for i, pr in enumerate(prs, 1):
        print(f"  Processing PR {i}/{len(prs)}: {pr['id']}")
        context = retriever.retrieve_for_pr(pr, n_results=15)
        task_context = pr.get('task_context', '')

        general_review = review(pr, context=context, task_context=task_context)
        ensemble_review = review_ensemble(pr, context=context, task_context=task_context)

        results_general.append({'pr_id': pr['id'], 'review': general_review, 'expected': pr['expected_issues']})
        results_ensemble.append({'pr_id': pr['id'], 'review': ensemble_review, 'expected': pr['expected_issues']})

    return results_general, results_ensemble


def build_efficiency_table(metrics_general: dict, metrics_ensemble: dict, num_prs: int) -> pd.DataFrame:
    """
    Build a cost-vs-quality comparison table.

    Cost is a *simulated* proxy based on relative LLM call count (General = 1
    call/PR, Ensemble = 2 calls/PR for the Security + Pattern agents), not a
    real dollar figure - both approaches use the same model (gpt-4o-mini) on
    similarly sized prompts, so call count is a reasonable stand-in for spend.
    "Quality per $" = F1 score / relative cost multiplier.

    Args:
        metrics_general: Metrics dict for the general agent
        metrics_ensemble: Metrics dict for the ensemble agent
        num_prs: Number of PRs evaluated

    Returns:
        DataFrame comparing API calls, relative cost, F1, and quality per $
    """
    return pd.DataFrame([
        {
            'Approach': 'General',
            'API Calls': num_prs,
            'Relative Cost': '1x',
            'F1 Score': f"{metrics_general['f1']:.1%}",
            'Quality per $': f"{metrics_general['f1']:.3f}"
        },
        {
            'Approach': 'Ensemble',
            'API Calls': num_prs * 2,
            'Relative Cost': '2x',
            'F1 Score': f"{metrics_ensemble['f1']:.1%}",
            'Quality per $': f"{metrics_ensemble['f1'] / 2:.3f}"
        }
    ])


def _pr_f1(found_count: int, total_findings: int, expected_count: int) -> float:
    """Per-PR F1 from recall (found/expected) and precision (found/total findings)."""
    precision = found_count / total_findings if total_findings > 0 else 0
    recall = found_count / expected_count if expected_count > 0 else 0
    return 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0


def build_per_pr_table(all_prs: list, results_general: list, results_ensemble: list) -> tuple:
    """
    Compare General vs Ensemble per PR using F1, not raw recall.

    Raw "issues found" alone favors whichever approach reports more findings
    and can't detect false positives - two agents can both find 2/2 expected
    issues while one also reports several false alarms the other doesn't. F1
    (which factors in each approach's total finding count, i.e. precision)
    is what actually determines review quality per PR.

    Args:
        all_prs: List of PR dictionaries
        results_general: General agent results (list of dicts with 'review')
        results_ensemble: Ensemble results (list of dicts with 'review')

    Returns:
        Tuple of (per_pr_df, summary_dict). summary_dict has 'ensemble_wins',
        'general_wins', 'ties', 'ensemble_perfect', 'total'.
    """
    rows = []
    ensemble_wins = general_wins = ties = ensemble_perfect = 0

    for pr, general_result, ensemble_result in zip(all_prs, results_general, results_ensemble):
        expected = pr['expected_issues']
        expected_count = len(expected)

        general_findings = general_result['review']['findings']
        ensemble_findings = ensemble_result['review']['findings']
        general_found = len(match_findings(general_findings, expected))
        ensemble_found = len(match_findings(ensemble_findings, expected))

        general_f1 = _pr_f1(general_found, len(general_findings), expected_count)
        ensemble_f1 = _pr_f1(ensemble_found, len(ensemble_findings), expected_count)

        if ensemble_f1 > general_f1 + 1e-9:
            winner = 'Ensemble'
            ensemble_wins += 1
        elif general_f1 > ensemble_f1 + 1e-9:
            winner = 'General'
            general_wins += 1
        else:
            winner = 'Tie'
            ties += 1

        if ensemble_found == expected_count:
            ensemble_perfect += 1

        rows.append({
            'PR ID': pr['id'],
            'Expected': expected_count,
            'General': f"{general_found}/{expected_count} found, {len(general_findings)} total",
            'Ensemble': f"{ensemble_found}/{expected_count} found, {len(ensemble_findings)} total",
            'Winner': winner
        })

    summary = {
        'ensemble_wins': ensemble_wins,
        'general_wins': general_wins,
        'ties': ties,
        'ensemble_perfect': ensemble_perfect,
        'total': len(all_prs)
    }

    return pd.DataFrame(rows), summary


def print_per_pr_summary(summary: dict):
    """Print the win/tie breakdown produced by build_per_pr_table()."""
    total = summary['total']
    print(f"\nSummary:")
    print(f"  Ensemble wins:  {summary['ensemble_wins']}/{total} PRs")
    print(f"  General wins:   {summary['general_wins']}/{total} PRs")
    print(f"  Ties:           {summary['ties']}/{total} PRs")
    print(f"  Ensemble perfect catches: {summary['ensemble_perfect']}/{total} PRs (100% recall)")


def build_context_per_pr_table(all_prs: list, results_no_context: list, results_selective: list, results_full: list) -> tuple:
    """
    Break down No Context vs Selective vs Full context per PR by issues found.

    Unlike build_per_pr_table (which picks a per-PR winner via F1), this table's
    job is to make finding *counts* visible - the selective-context story is
    about matching (or beating) full context's recall while reporting far fewer
    total findings, which is what drives its precision advantage.

    Args:
        all_prs: List of PR dictionaries
        results_no_context: No-context results (list of dicts with 'review')
        results_selective: Selective-context results (list of dicts with 'review')
        results_full: Full-context results (list of dicts with 'review')

    Returns:
        Tuple of (per_pr_df, summary_dict). summary_dict has 'total_prs' and
        the total findings reported across all PRs for each approach.
    """
    rows = []
    no_context_total = selective_total = full_total = 0

    for pr, nc_result, sel_result, full_result in zip(all_prs, results_no_context, results_selective, results_full):
        expected = pr['expected_issues']
        expected_count = len(expected)

        nc_findings = nc_result['review']['findings']
        sel_findings = sel_result['review']['findings']
        full_findings = full_result['review']['findings']

        nc_found = len(match_findings(nc_findings, expected))
        sel_found = len(match_findings(sel_findings, expected))
        full_found = len(match_findings(full_findings, expected))

        no_context_total += len(nc_findings)
        selective_total += len(sel_findings)
        full_total += len(full_findings)

        rows.append({
            'PR ID': pr['id'],
            'Expected': expected_count,
            'No Context': f"{nc_found}/{expected_count} found, {len(nc_findings)} total",
            'Selective': f"{sel_found}/{expected_count} found, {len(sel_findings)} total",
            'Full Context': f"{full_found}/{expected_count} found, {len(full_findings)} total",
        })

    summary = {
        'total_prs': len(all_prs),
        'no_context_total_findings': no_context_total,
        'selective_total_findings': selective_total,
        'full_total_findings': full_total,
    }

    return pd.DataFrame(rows), summary


def print_context_per_pr_summary(summary: dict):
    """Print the total-findings breakdown produced by build_context_per_pr_table()."""
    total = summary['total_prs']
    print(f"\nTotal Findings Reported (across {total} PRs):")
    print(f"  No Context:   {summary['no_context_total_findings']}")
    print(f"  Selective:    {summary['selective_total_findings']}")
    print(f"  Full Context: {summary['full_total_findings']}")
    print("\nFewer, more relevant findings -> higher precision without sacrificing recall.")
