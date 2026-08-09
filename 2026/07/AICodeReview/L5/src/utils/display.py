"""
Display utilities for creating comparison tables and formatted output.
"""
import pandas as pd
from IPython.display import HTML
from src.utils.matching import issues_match, find_best_match


def create_comparison_table(diff_only_review: dict, context_aware_review: dict, expected_issues: list) -> pd.DataFrame:
    """
    Create a side-by-side comparison table showing which issues were found.

    Uses the same matching logic as evaluation for consistency.

    Args:
        diff_only_review: Results from diff-only reviewer
        context_aware_review: Results from context-aware reviewer
        expected_issues: List of expected issues to find

    Returns:
        Styled pandas DataFrame
    """
    # Extract found issues as strings
    diff_only_issues = [f['issue'] for f in diff_only_review.get('findings', [])]
    context_aware_issues = [f['issue'] for f in context_aware_review.get('findings', [])]

    rows = []
    for expected in expected_issues:
        # Use unified matching logic
        diff_only_found = any(issues_match(expected, found) for found in diff_only_issues)
        context_found = any(issues_match(expected, found) for found in context_aware_issues)

        rows.append({
            'Expected Issue': expected,
            'Diff-Only': '✓' if diff_only_found else '✗',
            'Context-Aware': '✓' if context_found else '✗'
        })

    df = pd.DataFrame(rows)

    # Style the dataframe
    def color_cells(val):
        if val == '✓':
            return 'color: green; font-weight: bold'
        elif val == '✗':
            return 'color: red; font-weight: bold'
        return ''

    styled = df.style.applymap(color_cells, subset=['Diff-Only', 'Context-Aware'])

    return styled


def create_detailed_matching_table(review_results: dict, expected_issues: list) -> pd.DataFrame:
    """
    Create a detailed table showing which expected issues were found.

    Args:
        review_results: Results from reviewer with 'findings' list
        expected_issues: List of expected issues to find

    Returns:
        Styled pandas DataFrame showing matches
    """
    found_issues = [f['issue'] for f in review_results.get('findings', [])]

    rows = []
    matched_found_indices = set()

    for expected in expected_issues:
        # Find best match
        match_idx, match_score = find_best_match(expected, found_issues, threshold=0.3)

        if match_idx is not None and match_idx not in matched_found_indices:
            # Found a match
            matched_found_indices.add(match_idx)
            rows.append({
                'Status': '✓',
                'Expected Issue': expected,
                'Reported Matching Issue': found_issues[match_idx]
            })
        else:
            # No match found
            rows.append({
                'Status': '✗',
                'Expected Issue': expected,
                'Reported Matching Issue': ''
            })

    df = pd.DataFrame(rows)

    # Style the dataframe
    def color_status(val):
        if val == '✓':
            return 'color: green; font-weight: bold; font-size: 16px'
        elif val == '✗':
            return 'color: red; font-weight: bold; font-size: 16px'
        return ''

    def color_row(row):
        if row['Status'] == '✓':
            return ['background-color: #e8f5e9'] * len(row)
        elif row['Status'] == '✗':
            return ['background-color: #ffebee'] * len(row)
        return [''] * len(row)

    styled = df.style.applymap(color_status, subset=['Status']).apply(color_row, axis=1)

    return styled


def format_findings(findings: list) -> str:
    """Format findings as a readable string."""
    if not findings:
        return "No issues found"

    output = []
    for i, finding in enumerate(findings, 1):
        severity = finding['severity'].upper()
        issue = finding['issue']
        output.append(f"{i}. [{severity}] {issue}")

    return "\n".join(output)


def display_metrics_table(no_context_metrics: dict, selective_metrics: dict, full_context_metrics: dict):
    """
    Display a comparison table of metrics for three approaches.

    Args:
        no_context_metrics: Metrics for no context approach
        selective_metrics: Metrics for selective context approach
        full_context_metrics: Metrics for full context approach
    """
    from IPython.display import display

    metrics_df = pd.DataFrame([
        {
            'Approach': 'No Context',
            'Precision': f"{no_context_metrics['precision']:.2%}",
            'Recall': f"{no_context_metrics['recall']:.2%}",
            'F1 Score': f"{no_context_metrics['f1']:.2%}"
        },
        {
            'Approach': 'Full Context (All Chunks)',
            'Precision': f"{full_context_metrics['precision']:.2%}",
            'Recall': f"{full_context_metrics['recall']:.2%}",
            'F1 Score': f"{full_context_metrics['f1']:.2%}"
        },
        {
            'Approach': 'Selective Context (Top 15)',
            'Precision': f"{selective_metrics['precision']:.2%}",
            'Recall': f"{selective_metrics['recall']:.2%}",
            'F1 Score': f"{selective_metrics['f1']:.2%}"
        }
    ])

    display(metrics_df)

    # Print key insight
    selective_vs_full_f1 = selective_metrics['f1'] - full_context_metrics['f1']
    selective_vs_full_recall = selective_metrics['recall'] - full_context_metrics['recall']

    print(f"\nKey Insight:")
    print(f"  F1:     Selective vs Full = {selective_vs_full_f1:+.1%}")
    print(f"  Recall: Selective vs Full = {selective_vs_full_recall:+.1%}")

    if selective_vs_full_f1 > 0.02:
        print("\n✓ Selective context OUTPERFORMS full context!")
        print("  → Focused retrieval beats dumping all context")
    elif selective_vs_full_f1 < -0.02:
        print("\n⚠️  Full context performs slightly better")
    else:
        print("\n≈ Both approaches perform similarly")


def build_metrics_dataframe(results: dict) -> pd.DataFrame:
    """
    Build a Diff-Only vs Context-Aware precision/recall/F1 comparison table.

    Args:
        results: Dict with 'diff_only' and 'context_aware' metrics sub-dicts
                 (each with 'precision', 'recall', 'f1' keys)

    Returns:
        DataFrame with one row per approach
    """
    return pd.DataFrame([
        {
            'Approach': 'Diff-Only',
            'Precision': f"{results['diff_only']['precision']:.2%}",
            'Recall': f"{results['diff_only']['recall']:.2%}",
            'F1 Score': f"{results['diff_only']['f1']:.2%}"
        },
        {
            'Approach': 'Context-Aware',
            'Precision': f"{results['context_aware']['precision']:.2%}",
            'Recall': f"{results['context_aware']['recall']:.2%}",
            'F1 Score': f"{results['context_aware']['f1']:.2%}"
        }
    ])


def print_metrics_summary(results: dict):
    """Print the precision/recall/F1 improvement of context-aware over diff-only."""
    precision_improvement = results['context_aware']['precision'] - results['diff_only']['precision']
    recall_improvement = results['context_aware']['recall'] - results['diff_only']['recall']
    f1_improvement = results['context_aware']['f1'] - results['diff_only']['f1']

    print(
        f"\nContext-Aware Improvements:\n"
        f"  Precision: +{precision_improvement:.1%}\n"
        f"  Recall:    +{recall_improvement:.1%}\n"
        f"  F1 Score:  +{f1_improvement:.1%}"
    )
