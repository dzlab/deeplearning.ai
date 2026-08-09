"""
Analysis utilities for understanding reviewer performance patterns.
"""
import pandas as pd
from IPython.display import display
from src.data.sample_prs import load_all_prs


def analyze_findings(results: dict) -> dict:
    """
    Analyze which issues were caught by each approach.

    Args:
        results: Dictionary with evaluation results

    Returns:
        Dictionary with analysis of caught and missed issues
    """
    all_prs = load_all_prs()

    diff_only_results = results['detailed_results']['diff_only']
    context_aware_results = results['detailed_results']['context_aware']

    # Track which PRs had issues caught
    context_only_issues = []
    missed_by_both = []

    for i, pr in enumerate(all_prs):
        diff_tp = diff_only_results[i]['true_positives']
        context_tp = context_aware_results[i]['true_positives']
        expected = len(pr['expected_issues'])

        # Issues caught only by context-aware
        if context_tp > diff_tp:
            context_only_issues.extend([
                f"{pr['id']}: {issue}"
                for issue in pr['expected_issues'][:context_tp - diff_tp]
            ])

        # Issues missed by both
        if context_tp < expected:
            missed_count = expected - context_tp
            missed_by_both.extend([
                f"{pr['id']}: {issue}"
                for issue in pr['expected_issues'][:missed_count]
            ])

    # Create summary table
    summary_data = []
    for i, pr in enumerate(all_prs):
        diff_found = diff_only_results[i]['true_positives']
        context_found = context_aware_results[i]['true_positives']

        # Determine improvement indicator
        if context_found > diff_found:
            improvement = '↑ Better'
        elif context_found == diff_found:
            improvement = '= Same'
        else:
            improvement = '↓ Worse'

        summary_data.append({
            'PR ID': pr['id'],
            'Expected Issues': diff_only_results[i]['expected_count'],
            'Diff-Only Found': diff_found,
            'Context-Aware Found': context_found,
            'Result': improvement
        })

    summary_table = pd.DataFrame(summary_data)

    return {
        'context_only': context_only_issues[:5],  # Limit to 5 examples
        'missed_by_both': missed_by_both[:5],      # Limit to 5 examples
        'table': summary_table
    }


def display_findings_summary(findings_summary: dict):
    """
    Print the context-only / missed-by-both issue lists and show the per-PR table
    produced by analyze_findings().
    """
    context_only_text = "\n".join(f"  • {issue}" for issue in findings_summary['context_only']) or '  (none)'
    missed_by_both_text = "\n".join(f"  • {issue}" for issue in findings_summary['missed_by_both']) or '  (none)'

    print(
        f"Issues caught ONLY with context:\n{context_only_text}\n\n"
        f"Issues missed by BOTH approaches:\n{missed_by_both_text}\n\n"
        "Per-PR Summary:"
    )
    display(findings_summary['table'])
