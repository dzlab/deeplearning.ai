"""
A simple logit processor that tracks probabilities for both structured and unstructured generation.

For each token generated, we store:
- The raw logits the model would assign naturally
- The filtered logits after applying structural constraints
- A mapping from vocabulary indices to token strings
"""
from typing import TYPE_CHECKING, Optional, Union, List, Literal, Dict, Any

import numpy as np
import torch
from numpy.typing import NDArray
import matplotlib.pyplot as plt

from outlines.processors.base_logits_processor import OutlinesLogitsProcessor, Array

if TYPE_CHECKING:
    from outlines.generate import Generator

# Try importing pandas, but don't fail if not available
try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    pd = Any  # For type hints when pandas is not available

class LogitTrackingProcessor(OutlinesLogitsProcessor):
    """Tracks logits for both structured and unstructured token generation.
    
    For each position in the sequence, stores:
    - unstructured_logits: Raw logits from the model
    - structured_logits: Logits after applying constraints
    - vocab_tokens: Mapping from vocab indices to token strings
    
    Each logit matrix has:
    - Columns: One for each position in the generated sequence
    - Rows: One for each token in the vocabulary
    
    Attributes
    ----------
    processor : Optional[OutlinesLogitsProcessor]
        The processor that applies structural constraints
    unstructured_logits : List[NDArray]
        Raw logits from the model for each position
    structured_logits : List[NDArray]
        Logits after applying constraints for each position
    vocab_tokens : Optional[List[str]]
        Mapping from vocabulary indices to token strings
    chosen_tokens : List[int]
        Track actual chosen token IDs during generation
    """
    
    def __init__(self, processor=None):
        """Initialize the tracking processor.
        
        Parameters
        ----------
        processor : Optional[OutlinesLogitsProcessor]
            The processor that applies structural constraints.
            If None, only tracks raw logits.
        """
        self.processor = processor
        self.unstructured_logits = []  # List of logit arrays, one per position
        self.structured_logits = []    # List of logit arrays, one per position
        self.vocab_tokens = None      # Will store the vocabulary mapping
        self.chosen_tokens = []       # Track actual chosen tokens during generation
        
    def process_logits(self, input_ids: Array, logits: Array) -> Array:
        """Process logits and store them.
        
        This method:
        1. Stores the raw logits from the model
        2. Applies any structural constraints if a processor exists
        3. Stores the constrained logits
        4. Tracks the chosen token ID
        
        Parameters
        ----------
        input_ids : Array
            The input token ids for each sequence in the batch
        logits : Array
            The original logits to process, shape (batch_size, vocab_size)
            
        Returns
        -------
        Array
            The processed logits, shape (batch_size, vocab_size)
            
        Notes
        -----
        - For unconstrained generation (no processor), structured = unstructured
        - Token IDs are tracked from input_ids to ensure we capture the actual choices
        """
        # Always store the raw logits as unstructured
        self.unstructured_logits.append(logits[0].detach().cpu().numpy().copy())
        
        # Store the actual chosen token ID if available
        if len(input_ids[0]) > 0:
            self.chosen_tokens.append(input_ids[0][-1].item())
        
        # Apply structural constraints if we have a processor
        if self.processor is not None:
            processed = self.processor.process_logits(input_ids, logits)
            self.structured_logits.append(processed[0].detach().cpu().numpy().copy())
            return processed
            
        # For unconstrained generation, structured = unstructured
        self.structured_logits.append(logits[0].detach().cpu().numpy().copy())
        return logits
            
    def get_probabilities(self, as_matrix: bool = False) -> Dict[str, Union[List[NDArray], NDArray]]:
        """Get probability distributions computed from stored logits.
        
        Parameters
        ----------
        as_matrix : bool
            If True, convert probability lists to matrices.
            Each matrix will have shape (vocab_size, n_positions)
        
        Returns
        -------
        Dict[str, Union[List[NDArray], NDArray]]
            Contains:
            - unstructured: Raw probability distributions
            - structured: Probability distributions after constraints
            Each can be either a list of arrays or a single matrix
        """
        # Convert logits to probabilities
        unstructured_probs = [
            torch.softmax(torch.tensor(logits), dim=-1).numpy()
            for logits in self.unstructured_logits
        ]
        structured_probs = [
            torch.softmax(torch.tensor(logits), dim=-1).numpy()
            for logits in self.structured_logits
        ]
        
        if as_matrix:
            # Stack arrays into matrices
            unstructured = np.column_stack(unstructured_probs)
            structured = np.column_stack(structured_probs)
        else:
            # Return as lists
            unstructured = unstructured_probs
            structured = structured_probs
            
        return {
            'unstructured': unstructured,
            'structured': structured
        }

    def get_logits(self, as_matrix: bool = False) -> Dict[str, Union[List[NDArray], NDArray]]:
        """Get the stored logit values.
        
        Parameters
        ----------
        as_matrix : bool
            If True, convert logit lists to matrices.
            Each matrix will have shape (vocab_size, n_positions)
        
        Returns
        -------
        Dict[str, Union[List[NDArray], NDArray]]
            Contains:
            - unstructured: Raw logit values
            - structured: Logit values after constraints
            Each can be either a list of arrays or a single matrix
        """
        if as_matrix:
            unstructured = np.column_stack(self.unstructured_logits)
            structured = np.column_stack(self.structured_logits)
        else:
            unstructured = self.unstructured_logits
            structured = self.structured_logits
            
        return {
            'unstructured': unstructured,
            'structured': structured
        }
        
    def get_top_tokens(
        self,
        k: int = 10,
        positions: Optional[Union[int, List[int]]] = None,
        include_logits: bool = True
    ) -> List[Dict[str, Any]]:
        """Get the top k tokens at specified positions with their probabilities and logits.
        
        Parameters
        ----------
        k : int, optional
            Number of top tokens to return, by default 10
        positions : Union[int, List[int]], optional
            Position(s) to analyze. Can be a single position or list of positions.
            By default analyzes all positions.
        include_logits : bool, optional
            Whether to include raw logit values in addition to probabilities
            
        Returns
        -------
        List[Dict[str, Any]]
            List of dictionaries, one per position, containing:
            - position: Position in sequence
            - text_so_far: Text generated up to this position
            - tokens: List of top k token dictionaries, each containing:
                - token: The token string
                - natural_prob: Unconstrained probability
                - constrained_prob: Probability after constraints
                - natural_logit: Raw logit value (if include_logits=True)
                - constrained_logit: Constrained logit value (if include_logits=True)
                - is_chosen: Whether this token was actually chosen
        """
        # Convert single position to list
        if positions is None:
            positions = list(range(len(self.structured_logits)))
        elif isinstance(positions, int):
            positions = [positions]
            
        # Get probabilities and logits
        probs = self.get_probabilities()
        logits = self.get_logits() if include_logits else None
        
        # Get vocab mapping
        vocab = self.get_vocab_mapping()
        
        results = []
        for pos in positions:
            if pos >= len(self.unstructured_logits):
                continue
                
            # Get text generated so far
            text_so_far = self.sequence(pos)
            
            # Get values for this position
            u_probs = probs['unstructured'][pos]
            s_probs = probs['structured'][pos]
            
            if include_logits:
                u_logits = logits['unstructured'][pos]
                s_logits = logits['structured'][pos]
            
            # Get top k indices by maximum probability
            top_indices = np.argsort(np.maximum(u_probs, s_probs))[-k:][::-1]
            
            # Get the actual next token for comparison
            next_token = self.sequence(pos + 1)[len(text_so_far):] if pos < len(self.structured_logits)-1 else ""
            
            # Build token info list
            tokens = []
            for idx in top_indices:
                token = vocab[idx]
                token_info = {
                    'token': token,
                    'natural_prob': float(u_probs[idx]),
                    'constrained_prob': float(s_probs[idx]),
                    'is_chosen': token == next_token
                }
                
                if include_logits:
                    token_info.update({
                        'natural_logit': float(u_logits[idx]),
                        'constrained_logit': float(s_logits[idx])
                    })
                    
                tokens.append(token_info)
            
            results.append({
                'position': pos,
                'text_so_far': text_so_far,
                'tokens': tokens
            })
            
        return results

    def get_vocab_mapping(self) -> List[str]:
        """Get the mapping from vocabulary indices to token strings.
        
        Returns
        -------
        List[str]
            List of token strings, where index matches vocabulary index
        
        Raises
        ------
        AttributeError
            If no tokenizer is available
        """
        if not hasattr(self, 'tokenizer'):
            raise AttributeError("No tokenizer available for mapping tokens")
            
        if self.vocab_tokens is None:
            # Create the mapping if we haven't yet
            self.vocab_tokens = [
                self.processor.tokenizer.decode([i])[0]
                for i in range(len(self.unstructured_logits[0]))
            ]
            
        return self.vocab_tokens
        
    def clear(self):
        """Clear all stored logits."""
        self.unstructured_logits = []
        self.structured_logits = []
        self.chosen_tokens = []

    def to_dataframe(
        self,
        show: Literal["probs", "logits"] = "probs",
        top_k: Optional[int] = None,
        min_value: Optional[float] = None
    ) -> "pd.DataFrame":
        """Convert tracking data to a pandas DataFrame for analysis.
        
        Parameters
        ----------
        show : Literal["probs", "logits"], optional
            Whether to show probabilities or logit values, by default "probs"
        top_k : Optional[int], optional
            If provided, only include the top k tokens at each position
            (based on maximum of structured/unstructured values)
        min_value : Optional[float], optional
            If provided, only include tokens with values >= min_value
            in either structured or unstructured distribution
            
        Returns
        -------
        pd.DataFrame
            DataFrame with columns:
            - position: Token position in sequence
            - token: String representation of token
            - natural: Raw model values (probs/logits)
            - constrained: Values after constraints
            
        Examples
        --------
        >>> # Get probability data for top 10 tokens
        >>> df = processor.to_dataframe(show="probs", top_k=10)
        >>> df.sort_values("natural", ascending=False).head()
        >>>
        >>> # Get logit data above threshold
        >>> df = processor.to_dataframe(show="logits", min_value=-5)
        >>> df.query("position == 0").nlargest(5, "natural")
        >>>
        >>> # Get all tokens with probability > 1%
        >>> df = processor.to_dataframe(show="probs", min_value=0.01)
            
        Raises
        ------
        ImportError
            If pandas is not installed
        """
        if not PANDAS_AVAILABLE:
            raise ImportError(
                "pandas is required for DataFrame support. "
                "Please install it with: pip install pandas"
            )
            
        # Get values based on show parameter
        if show == "probs":
            values = self.get_probabilities()
        else:
            values = self.get_logits()
            
        # Get vocab mapping
        vocab = self.get_vocab_mapping()
        
        # Create lists to store data
        rows = []
        
        # Process each position
        for pos in range(len(self.unstructured_logits)):
            u_vals = values['unstructured'][pos]
            s_vals = values['structured'][pos]
            
            # Get indices to include based on filters
            if top_k is not None or min_value is not None:
                # Get maximum value between structured/unstructured for sorting
                max_vals = np.maximum(u_vals, s_vals)
                
                if top_k is not None and min_value is not None:
                    # Both filters: get top k among values >= min_value
                    valid_indices = np.where(max_vals >= min_value)[0]
                    if len(valid_indices) > top_k:
                        valid_indices = valid_indices[np.argsort(max_vals[valid_indices])[-top_k:]]
                elif top_k is not None:
                    # Just top k: get indices of k largest values
                    valid_indices = np.argsort(max_vals)[-top_k:]
                else:  # min_value is not None
                    # Just threshold: get indices of all values >= min_value
                    valid_indices = np.where(max_vals >= min_value)[0]
            else:
                # No filters: include all tokens
                valid_indices = range(len(vocab))
            
            # Add rows for valid indices
            for idx in valid_indices:
                rows.append({
                    'position': pos,
                    'token': vocab[idx],
                    'natural': u_vals[idx],
                    'constrained': s_vals[idx]
                })
        
        return pd.DataFrame(rows)

    def sequence(self, pos: Optional[int] = None) -> str:
        """Get the sequence of tokens generated up to a position.
        
        Parameters
        ----------
        pos : Optional[int], optional
            Position to reconstruct up to (exclusive).
            If None, returns the entire sequence.
            
        Returns
        -------
        str
            The concatenated string of chosen tokens
            
        Raises
        ------
        AttributeError
            If no tokenizer is available for decoding
        """
        if not self.chosen_tokens:
            return ""
            
        if not hasattr(self, 'tokenizer'):
            raise AttributeError("No tokenizer available for decoding sequence")
            
        # Get the tokenizer
        if hasattr(self.processor, 'tokenizer'):
            tokenizer = self.processor.tokenizer
        else:
            tokenizer = self.tokenizer
            
        # Get tokens up to the specified position
        end_pos = len(self.chosen_tokens) if pos is None else pos
        tokens_to_decode = self.chosen_tokens[:end_pos]
        
        # Decode the sequence
        return "".join(tokenizer.decode(tokens_to_decode))


def track_logits(generator: "Generator") -> "Generator":
    """Add probability tracking to any generator.
    
    This is a convenience function that wraps a generator's logits processor
    with a LogitTrackingProcessor, enabling analysis of token probabilities
    during generation.
    
    Parameters
    ----------
    generator : Generator
        The generator to add tracking to
        
    Returns
    -------
    Generator
        The same generator with tracking enabled
        
    Examples
    --------
    >>> # Track probabilities for unconstrained text generation
    >>> generator = generate.text(model)
    >>> generator = track_logits(generator)
    >>>
    >>> # Track probabilities for JSON generation
    >>> generator = generate.json(model, schema)
    >>> generator = track_logits(generator)
    """
    # If there's no logits_processor, throw an error. Logit tracking
    # is currently only supported for structured generators.
    if generator.logits_processor is None:
        raise ValueError("Logit tracking is not supported for this generator")

    # Create tracking processor, wrapping any existing processor
    tracking = LogitTrackingProcessor(generator.logits_processor)

    # Add tokenizer for token mapping
    if hasattr(generator.logits_processor, 'tokenizer'):
        tracking.tokenizer = generator.logits_processor.tokenizer
    
    # Set as the generator's processor
    generator.logits_processor = tracking
    
    return generator

# This function applies a simple chat template to the prompt
def template(model, prompt: str, system_prompt: str = "You are a helpful assistant, responding in JSON.") -> str:
    return model.tokenizer.tokenizer.apply_chat_template(
        [{"role": "system", "content": system_prompt}, {"role": "user", "content": prompt}],
        tokenize=False,
        add_bos=True,
        add_generation_prompt=True,
    )

def plot_token_distributions(tracking_processor, k=10, positions=None, prefix=""):
    """Plot token probability distributions before and after applying constraints.
    
    Creates a horizontal bar chart showing:
    - Blue bars: What tokens the model would naturally choose
    - Orange bars: What tokens are allowed by structural constraints
    
    Parameters
    ----------
    tracking_processor : LogitTrackingProcessor
        The processor containing tracked probabilities
    k : int, optional
        Number of top tokens to show in each plot, by default 10
    positions : List[int], optional
        Which positions to plot. If None, plots all positions.
    prefix : str, optional
        Prefix for the output filename
        
    Notes
    -----
    - Bar height indicates probability (how likely the model thinks each token is)
    - Tokens are sorted by maximum probability across both distributions
    - Only probabilities > 1% show their exact values
    - Grid lines help compare probabilities between tokens
    """
    # Get probability matrices and vocab mapping
    probs = tracking_processor.get_probabilities(as_matrix=True)
    vocab = tracking_processor.get_vocab_mapping()
    
    # Determine positions to plot
    if positions is None:
        positions = list(range(probs['unstructured'].shape[1]))
    n_positions = len(positions)
    
    # Create plot
    fig, axes = plt.subplots(1, n_positions)
    if n_positions == 1:
        axes = [axes]
    
    for idx, pos in enumerate(positions):
        # Get probabilities for this position
        unstructured = probs['unstructured'][:, pos]
        structured = probs['structured'][:, pos]
        
        # Get top k tokens by maximum probability
        top_indices = np.argsort(np.maximum(unstructured, structured))[-k:]
        
        # Create bar positions
        y = np.arange(len(top_indices))
        height = 0.35
        
        # Plot bars
        axes[idx].barh(y - height/2, unstructured[top_indices], height, 
                      label='Unconstrained', alpha=0.7, color='skyblue')
        axes[idx].barh(y + height/2, structured[top_indices], height,
                      label='Constrained', alpha=0.7, color='orange')
        
        # Customize plot
        axes[idx].set_title('Next token probability')
        axes[idx].set_yticks(y)
        axes[idx].set_yticklabels([vocab[i] for i in top_indices])
        axes[idx].set_xlabel('Probability')
        axes[idx].tick_params(axis='both', labelsize=16)  # This changes tick label sizes
        
        # Add legend
        axes[idx].legend(loc='lower right', bbox_to_anchor=(1, 1.1))
        axes[idx].grid(True, alpha=0.3)
        
        # Add probability values
        for i, (v1, v2) in enumerate(zip(unstructured[top_indices], structured[top_indices])):
            if v1 > 0.01:  # Only show probabilities > 1%
                axes[idx].text(v1 + 0.01, i - height/2, f'{v1:.1%}', va='center')
            if v2 > 0.01:
                axes[idx].text(v2 + 0.01, i + height/2, f'{v2:.1%}', va='center')
    
    plt.tight_layout()
    # plt.savefig(f"{prefix}token_distributions.png", dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()


def plot_heatmap(tracking_processor, k=50, positions=None, prefix="", show_both=True, kind="logits", show_tokens=True):
    """Plot a heatmap of token probabilities across sequence positions.
    
    Creates a heatmap visualization showing how token probabilities evolve
    across different positions in the sequence. Optionally shows both
    natural and constrained probabilities side by side.
    
    Parameters
    ----------
    tracking_processor : LogitTrackingProcessor
        The processor containing tracked probabilities
    k : int, optional
        Number of top tokens to include in the heatmap, by default 50
    positions : List[int], optional
        Which positions to plot. If None, plots all positions.
    prefix : str, optional
        Prefix for the output filename
    show_both : bool, optional
        If True, shows both natural and constrained probabilities side by side.
        If False, only shows natural probabilities.
    kind : str, optional
        Whether to plot logits or probabilities, by default "logits"
    show_tokens : bool, optional
        Whether to show the token strings on the y-axis, by default True
        
    Notes
    -----
    - Brighter colors indicate higher probabilities
    - Y-axis shows token strings
    - X-axis shows position in sequence
    - Near-zero probabilities are masked out (shown in gray)
    - For constrained generation, blocked tokens appear masked
    """
    # Get probability matrices and vocab mapping
    if kind == "logits":
        things = tracking_processor.get_logits(as_matrix=True)
        # For logits, mask out very negative values
        threshold = -1e9  # Logits below this are effectively zero probability
    else:
        things = tracking_processor.get_probabilities(as_matrix=True)
        # For probabilities, mask out near-zero values
        threshold = 0.001  # Probabilities below 0.1% are masked
    
    vocab = tracking_processor.get_vocab_mapping()
    
    # Determine positions to plot
    if positions is None:
        positions = list(range(things['unstructured'].shape[1]))
    
    # Get indices of top k tokens (by maximum probability across all positions)
    max_probs = np.maximum(
        things['unstructured'].max(axis=1),
        things['structured'].max(axis=1)
    )
    top_indices = np.argsort(max_probs)[-k:]
    
    # Create masked arrays for better visualization
    def mask_array(arr):
        if kind == "logits":
            return np.ma.masked_where(arr < threshold, arr)
        else:
            return np.ma.masked_where(arr < threshold, arr)
    
    unstructured_masked = mask_array(things['unstructured'][top_indices][:, positions])
    structured_masked = mask_array(things['structured'][top_indices][:, positions])

    unstructured_masked, structured_masked = [(x - x.mean(0)) / x.std(0) for x in (unstructured_masked, structured_masked)]

    # Create figure
    if show_both:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 8))
        fig.suptitle(f'Token {kind.capitalize()} Evolution', fontsize=16, y=1.05)
    else:
        fig, ax1 = plt.subplots(1, 1, figsize=(8, 8))
    
    # Plot natural probabilities with masked array
    im1 = ax1.imshow(
        unstructured_masked,
        aspect='auto',
        cmap='viridis',
    )
    ax1.set_title(f'Natural Token {kind.capitalize()}')
    ax1.set_xlabel('Position in Sequence')
    ax1.set_ylabel('Token')
    if show_tokens:
        ax1.set_yticks(range(len(top_indices)))
        ax1.set_yticklabels([vocab[i] for i in top_indices])
    plt.colorbar(im1, ax=ax1, label=f'{kind.capitalize()}')
    
    # Plot constrained probabilities if requested
    if show_both:
        im2 = ax2.imshow(
            structured_masked,
            aspect='auto',
            cmap='viridis',
        )
        ax2.set_title(f'Constrained Token {kind.capitalize()}')
        ax2.set_xlabel('Position in Sequence')
        ax2.set_yticks([])  # Hide y-ticks since they're the same as ax1
        plt.colorbar(im2, ax=ax2, label=f'{kind.capitalize()}')
    
    plt.tight_layout()
    # plt.savefig(f"{prefix}{kind}_heatmap.png", dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()

