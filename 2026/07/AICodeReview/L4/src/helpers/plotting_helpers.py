"""
Plotting helpers for Notebook 3 (Specialized Agents).
Keeps Plotly figure construction out of the notebook so cells read as comparisons, not chart code.
"""
import plotly.graph_objects as go


def plot_general_vs_ensemble(metric_names: list, general_values: list, ensemble_values: list):
    """
    Show a grouped bar chart comparing General vs Ensemble across metrics.

    Args:
        metric_names: Labels for the x-axis (e.g. ['Precision', 'Recall', 'F1 Score'])
        general_values: General agent's score for each metric
        ensemble_values: Ensemble agent's score for each metric
    """
    fig = go.Figure()

    fig.add_trace(go.Bar(
        name='General (Lab 2)',
        x=metric_names,
        y=general_values,
        marker_color='#94a3b8',
        text=[f'{v:.1%}' for v in general_values],
        textposition='outside'
    ))

    fig.add_trace(go.Bar(
        name='Ensemble',
        x=metric_names,
        y=ensemble_values,
        marker_color='#10b981',
        text=[f'{v:.1%}' for v in ensemble_values],
        textposition='outside'
    ))

    fig.update_layout(
        title='General Agent vs Ensemble Performance',
        xaxis_title='Metric',
        yaxis_title='Score',
        barmode='group',
        yaxis=dict(range=[0, 1.1]),
        height=500
    )

    fig.show()


def plot_precision_recall(approaches: list):
    """
    Show a precision-recall scatter plot comparing multiple approaches.

    Args:
        approaches: List of dicts with 'name', 'metrics' ({'precision', 'recall', ...}),
                    'color', and 'size' keys
    """
    fig = go.Figure()

    for approach in approaches:
        metrics = approach['metrics']
        fig.add_trace(go.Scatter(
            x=[metrics['recall']],
            y=[metrics['precision']],
            mode='markers+text',
            name=approach['name'],
            marker=dict(size=approach['size'], color=approach['color']),
            text=[approach['name']],
            textposition='top center'
        ))

    fig.add_shape(
        type="line",
        x0=0, y0=0, x1=1, y1=1,
        line=dict(color="lightgray", dash="dash", width=1)
    )

    fig.update_layout(
        title='Precision vs Recall: General vs Ensemble',
        xaxis_title='Recall',
        yaxis_title='Precision',
        xaxis=dict(range=[0, 1.05]),
        yaxis=dict(range=[0, 1.15]),
        showlegend=False,
        height=500
    )

    fig.show()


def print_approach_metrics(name: str, metrics: dict):
    """Print a one-line Precision/Recall/F1 summary for an approach."""
    print(f"  {name}: P={metrics['precision']:.1%}, R={metrics['recall']:.1%}, F1={metrics['f1']:.1%}")
