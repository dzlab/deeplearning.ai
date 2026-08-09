"""
Plotting utilities for visualizing evaluation results.
"""
import plotly.graph_objects as go


def create_metrics_chart(results: dict) -> go.Figure:
    """
    Create a grouped bar chart comparing metrics between approaches.

    Args:
        results: Dictionary with 'diff_only' and 'context_aware' metrics

    Returns:
        Plotly figure object
    """
    diff_only = results['diff_only']
    context_aware = results['context_aware']

    # Create grouped bar chart
    fig = go.Figure()

    # Add diff-only bars
    fig.add_trace(go.Bar(
        name='Diff-Only',
        x=['Precision', 'Recall', 'F1 Score'],
        y=[diff_only['precision'], diff_only['recall'], diff_only['f1']],
        marker_color='#ff7f0e',
        text=[f"{v:.1%}" for v in [diff_only['precision'], diff_only['recall'], diff_only['f1']]],
        textposition='auto',
    ))

    # Add context-aware bars
    fig.add_trace(go.Bar(
        name='Context-Aware',
        x=['Precision', 'Recall', 'F1 Score'],
        y=[context_aware['precision'], context_aware['recall'], context_aware['f1']],
        marker_color='#2ca02c',
        text=[f"{v:.1%}" for v in [context_aware['precision'], context_aware['recall'], context_aware['f1']]],
        textposition='auto',
    ))

    # Update layout
    fig.update_layout(
        title='Review Performance: Diff-Only vs Context-Aware',
        xaxis_title='Metric',
        yaxis_title='Score',
        yaxis=dict(range=[0, 1], tickformat='.0%'),
        barmode='group',
        template='plotly_white',
        height=500,
        font=dict(size=14),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )

    return fig


def plot_metrics_comparison(no_context_metrics: dict, selective_metrics: dict, full_context_metrics: dict):
    """
    Create and display a grouped bar chart comparing three approaches.

    Args:
        no_context_metrics: Metrics for no context approach
        selective_metrics: Metrics for selective context approach
        full_context_metrics: Metrics for full context approach
    """
    fig = go.Figure()

    metrics = ['Precision', 'Recall', 'F1 Score']
    no_ctx_vals = [no_context_metrics['precision'], no_context_metrics['recall'], no_context_metrics['f1']]
    full_ctx_vals = [full_context_metrics['precision'], full_context_metrics['recall'], full_context_metrics['f1']]
    selective_vals = [selective_metrics['precision'], selective_metrics['recall'], selective_metrics['f1']]

    fig.add_trace(go.Bar(name='No Context', x=metrics, y=no_ctx_vals, marker_color='#94a3b8'))
    fig.add_trace(go.Bar(name='Full Context (All Chunks)', x=metrics, y=full_ctx_vals, marker_color='#ef4444'))
    fig.add_trace(go.Bar(name='Selective Context (Top 15)', x=metrics, y=selective_vals, marker_color='#10b981'))

    fig.update_layout(
        title='Context Quality Comparison: Full vs Selective',
        xaxis_title='Metric',
        yaxis_title='Score',
        barmode='group',
        yaxis=dict(range=[0, 1]),
        height=500
    )

    fig.show()


def plot_context_efficiency(context_sizes: dict, f1_scores: dict, labels: list):
    """
    Create and display a scatter plot showing context size vs quality trade-off.

    Args:
        context_sizes: Dict mapping approach names to context sizes (in characters)
        f1_scores: Dict mapping approach names to F1 scores
        labels: List of approach names in order ['No Context', 'Selective', 'Full']
    """
    fig = go.Figure()

    # Define styling for each approach
    colors = {'No Context': '#94a3b8', 'Selective': '#ef4444', 'Full': '#3b82f6'}
    symbols = {'No Context': 'x', 'Selective': 'circle', 'Full': 'circle'}
    sizes = {'No Context': 16, 'Selective': 16, 'Full': 16}

    # Add data points
    for label in labels:
        size_kb = context_sizes[label] / 1000
        f1 = f1_scores[label]
        marker_size = sizes[label]

        fig.add_trace(go.Scatter(
            x=[size_kb], y=[f1],
            mode='markers+text',
            name=label,
            marker=dict(
                size=marker_size,
                color=colors[label],
                symbol=symbols[label],
                line=dict(width=2, color='white')
            ),
            text=[label],
            textposition='top center',
            textfont=dict(size=11)
        ))

    # Calculate dynamic axis range
    f1_values = list(f1_scores.values())
    y_min = max(0, min(f1_values) - 0.1)
    y_max = min(1, max(f1_values) + 0.1)

    # Calculate x-axis range with some padding
    size_values_kb = [context_sizes[label] / 1000 for label in labels]
    x_max = max(size_values_kb) * 1.1

    fig.update_layout(
        title='Context Efficiency: Quality vs. Cost Trade-off',
        xaxis_title='Context Size (KB) — Lower is Better →',
        yaxis_title='F1 Score — Higher is Better ↑',
        yaxis=dict(
            range=[y_min, y_max],
            gridcolor='rgba(128, 128, 128, 0.2)',
            showgrid=True
        ),
        xaxis=dict(
            range=[-1, x_max],
            gridcolor='rgba(128, 128, 128, 0.2)',
            showgrid=True
        ),
        showlegend=True,
        legend=dict(
            orientation="v",
            yanchor="top",
            y=0.95,
            xanchor="right",
            x=0.95
        ),
        height=500,
        font=dict(size=12),
        template='plotly_white'
    )

    fig.show()

    # Calculate and display efficiency metrics
    selective_size = context_sizes['Selective'] / 1000
    full_size = context_sizes['Full'] / 1000
    selective_f1 = f1_scores['Selective']
    full_f1 = f1_scores['Full']

    selective_efficiency = selective_f1 / max(selective_size, 0.01)
    full_efficiency = full_f1 / max(full_size, 0.01)

    print("\nContext Efficiency (F1 per KB):")
    print(f"  Selective: {selective_efficiency:.3f} F1/KB  ← {selective_efficiency / full_efficiency:.1f}x more efficient")
    print(f"  Full:      {full_efficiency:.3f} F1/KB")
    print(f"\nSelective achieves {selective_f1 / max(full_f1, 0.01):.1%} performance with {selective_size / full_size:.1%} of the context size")
