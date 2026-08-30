"""
2026.5.1
2026.5.2
4.57.6
0.24.0
__UNSLOTH_VERSIONING__
"""

# Unsloth auto generated code
# Copyright 2023-present Daniel Han-Chen, Michael Han-Chen & the Unsloth team. All rights reserved.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.


import os
import sys
import torch
import importlib.util
import math
if importlib.util.find_spec("unsloth_studio") is None:
    UNSLOTH_STUDIO_ENABLED = False
else:
    UNSLOTH_STUDIO_ENABLED = os.environ.get("UNSLOTH_STUDIO_DISABLED", "0") == "0"
pass
from typing import Any, List, Optional, Tuple, Union, Dict, Set, Callable
import math

UNSLOTH_ENABLE_LOGGING = os.environ.get("UNSLOTH_ENABLE_LOGGING", "0") == "1"
UNSLOTH_ENABLE_CCE = os.environ.get("UNSLOTH_ENABLE_CCE", "1") == "1"
UNSLOTH_COMPILE_DISABLE = os.environ.get("UNSLOTH_COMPILE_DISABLE", "0") in ("1", "partial",)
UNSLOTH_COMPILE_LOCATION = os.environ.get("UNSLOTH_COMPILE_LOCATION", "unsloth_compiled_cache")
if UNSLOTH_COMPILE_LOCATION not in sys.path:
    sys.path.insert(0, UNSLOTH_COMPILE_LOCATION)

import logging
logger_compiler = logging.getLogger(__name__)
if UNSLOTH_ENABLE_LOGGING:
    logger_compiler.setLevel(logging.DEBUG)

global INFERENCE_RUNS
INFERENCE_RUNS = 0

try:
    import torch._dynamo.eval_frame as torch_dynamo_eval_frame
    torch_dynamo_eval_frame._stance.stance
    torch_compiler_set_stance = torch.compiler.set_stance
except:
    torch_dynamo_eval_frame = None
    torch_compiler_set_stance = None
pass

from unsloth_zoo import DEVICE_TYPE_TORCH, DEVICE_COUNT


from unsloth_zoo.loss_utils import (
    fused_linear_cross_entropy,
    unsloth_fused_ce_loss,
)

scaled_dot_product_attention = torch.nn.functional.scaled_dot_product_attention
@torch.compiler.disable(recursive = False)
def disable_compile_scaled_dot_product_attention(*args, **kwargs):
    return scaled_dot_product_attention(*args, **kwargs)
pass


from transformers.modeling_flash_attention_utils import is_flash_attn_available

if is_flash_attn_available():
    try:
        from transformers.modeling_flash_attention_utils import flash_attn_supports_top_left_mask
    except:
        flash_attn_supports_top_left_mask = None
    try:
        from transformers.modeling_flash_attention_utils import _flash_attention_forward
    except:
        _flash_attention_forward = None
    try:
        from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
    except:
        FlashAttentionKwargs = None
    try:
        from transformers.modeling_flash_attention_utils import flash_attn_varlen_func
    except:
        flash_attn_varlen_func = None
else:
    flash_attn_supports_top_left_mask = None
    _flash_attention_forward = None
    FlashAttentionKwargs = None
    flash_attn_varlen_func = None
pass


torch_compile_options = {'epilogue_fusion': True, 'max_autotune': False, 'shape_padding': True, 'trace.enabled': False, 'triton.cudagraphs': False, 'debug': False, 'dce': True, 'memory_planning': True, 'coordinate_descent_tuning': False, 'trace.graph_diagram': False, 'compile_threads': 30, 'group_fusion': True, 'disable_progress': True, 'verbose_progress': False, 'triton.multi_kernel': 0, 'triton.use_block_ptr': False, 'triton.enable_persistent_tma_matmul': True, 'triton.autotune_at_compile_time': False, 'triton.cooperative_reductions': False, 'cuda.compile_opt_level': '-O2', 'cuda.enable_cuda_lto': True, 'combo_kernels': False, 'benchmark_combo_kernel': True, 'combo_kernel_foreach_dynamic_shapes': True}

from torch.nn import CrossEntropyLoss

@torch.compile(fullgraph = True, dynamic = True, options = torch_compile_options)
def normal_cross_entropy_loss(self, hidden_states, labels):
    logits = self.lm_head(hidden_states)
    logits = logits.float()
    # Shift so that tokens < n predict n
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    # Flatten the tokens
    loss_fct = CrossEntropyLoss()
    shift_logits = shift_logits.view(-1, self.config.vocab_size)
    shift_labels = shift_labels.view(-1)
    # Enable model parallelism
    shift_labels = shift_labels.to(shift_logits.device)
    loss = loss_fct(shift_logits, shift_labels)
    return loss, logits
pass

# We need an empty logits flag to warn people logits will not be returned anymore unless asked ie
# os.environ['UNSLOTH_RETURN_LOGITS'] = '1'
LOGITS_ERROR_STRING = \
    "Unsloth: Logits are empty from 2024.11 onwards. To get raw logits again, please "\
    'set the environment variable `UNSLOTH_RETURN_LOGITS` to `"1" BEFORE starting to train ie before `trainer.train()`. For example:\n'\
    "```\nimport os\n"\
    "os.environ['UNSLOTH_RETURN_LOGITS'] = '1'\n"\
    "trainer.train()\n```\n"\
    "No need to restart your console - just add `os.environ['UNSLOTH_RETURN_LOGITS'] = '1'` before trainer.train() and re-run the cell!"

def raise_logits_error(*args, **kwargs): raise NotImplementedError(LOGITS_ERROR_STRING)
def return_none(*args, **kwargs): return None
class EmptyLogits:
    def __init__(self): return
    def raise_getattr_error(self, attr): return return_none if attr == "to" else raise_logits_error
    __getitem__ = raise_logits_error
    __getattr__ = raise_getattr_error
    def __repr__(self): return LOGITS_ERROR_STRING
    def __str__ (self): return LOGITS_ERROR_STRING
pass
EMPTY_LOGITS = EmptyLogits()
functions = dir(torch.Tensor)
for j, function in enumerate(functions):
    if function.startswith("__") and function.endswith("__"):
        exec(f"def raise_{j}(*args, **kwargs): print('{function}')", globals(), locals())
        try: exec(f"EMPTY_LOGITS.{function} = raise_{j}", globals(), locals())
        except: continue
pass


def mask_attention_mask_out(labels = None, attention_mask = None):
    if labels is not None and attention_mask is not None:
        attention_mask = attention_mask.to(device = labels.device)
        labels[attention_mask == 0] = -100
    return labels
pass


from torch import Tensor
import torch
import torch.nn as nn
from torch.nn import functional as F
from unsloth_zoo.temporary_patches.common import torch_compile
from typing import Any, List, Optional, Tuple, Union, Dict, Set, Callable
from transformers.models.gemma3n.modeling_gemma3n import (F, copy, math, Callable, Sequence, Optional, Union, torch, nn, ACT2FN, Cache, DynamicCache, GenerationMixin, create_causal_mask, create_sliding_window_causal_mask, FlashAttentionKwargs, BaseModelOutputWithPast, ModelOutput, CausalLMOutputWithPast, ROPE_INIT_FUNCTIONS, dynamic_rope_update, ALL_ATTENTION_FUNCTIONS, PreTrainedModel, Unpack, TransformersKwargs, can_return_tuple, deprecate_kwarg, Gemma3nAudioConfig, Gemma3nConfig, Gemma3nTextConfig, Gemma3nVisionConfig, logger, __name__, Gemma3nModel, Gemma3nCausalLMOutputWithPast, Gemma3nTextScaledWordEmbedding, Gemma3nTextDecoderLayer, Gemma3nPreTrainedModel, Gemma3nTextModel, Gemma3nForCausalLM, Gemma3nForConditionalGeneration, Gemma3nAudioSSCPConvBlock, Gemma3nAudioConformerLightConv1d, create_causal_mask, create_sliding_window_causal_mask)

@torch.compiler.disable(recursive = False)
def Gemma3nRMSNorm_forward(self, x: torch.Tensor) -> torch.Tensor:
    # Llama does x.to(float16) * w whilst Gemma2 is (x * w).to(float16)
    # See https://github.com/huggingface/transformers/pull/29402
    output = self._norm(x.float()) * self.weight.float()
    return output.type_as(x)

class Gemma3nRMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6, with_scale: bool = True):
        super().__init__()
        self.eps = eps
        self.with_scale = with_scale

        if self.with_scale:
            self.weight = nn.Parameter(torch.ones(dim))
        else:
            self.register_buffer("weight", torch.tensor(1.0), persistent=False)

    def _norm(self, x):
        return x / torch.sqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return Gemma3nRMSNorm_forward(self, x=x)

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.eps}"


@torch.compiler.disable(recursive = False)
def Gemma3nAudioRelativePositionEmbedding_forward(self, queries: torch.Tensor, keys: torch.Tensor) -> torch.Tensor:
    # queries: [B, U, W, N, H] (batch, num_query_blocks, query_block_size, num_heads, head_dim)
    # keys:    [B, U, C, N, H] (batch, num_query_blocks, key_context_size, num_heads, head_dim)
    # C = W + L + R (key_context_size)
    # F_span = L + R + 1 (max_span + 1)

    batch_size, num_query_blocks, query_block_size, num_heads, head_dim = queries.shape
    _, _, key_context_size, _, _ = keys.shape

    # Relative positions for sinusoidal embeddings: [L, L-1, ..., -R]
    # Length is L+R+1 = self.max_span + 1
    pos_indices = torch.arange(self.max_backward, -self.max_forward - 1, -1, device=queries.device).unsqueeze(
        0
    )  # Shape [1, F_span]

    max_span_plus_1 = pos_indices.shape[1]  # F_span

    sin_emb_timing_signal = self._get_timing_signal_1d_pos(
        pos_indices, dtype=queries.dtype
    )  # Shape [1, F_span, self.channels]

    # Project sinusoidal embeddings: [1, F_span, self.channels] -> [1, F_span, N*H]
    projected_sin_emb = self.pos_proj(sin_emb_timing_signal)
    # Reshape to [1, F_span, N, H] then squeeze to [F_span, N, H]
    sin_emb = projected_sin_emb.reshape(1, max_span_plus_1, self.num_heads, self.head_dim).squeeze(
        0
    )  # Shape [F, N, H]

    # term_ac: Query-Key content interaction
    # queries: [B, U, W, N, H] -> permute to [B, N, U, W, H] for matmul
    # keys:    [B, U, C, N, H] -> permute to [B, N, U, H, C] for matmul
    queries_p = queries.permute(0, 3, 1, 2, 4)  # [B, N, U, W, H]
    keys_p_t = keys.permute(0, 3, 1, 4, 2)  # [B, N, U, H, C]
    term_ac = torch.matmul(queries_p, keys_p_t)  # [B, N, U, W, C]

    # term_bd: Query-Position interaction
    # Original einsum: term_bd_unshifed = torch.einsum('buwnh,fnh->bnuwf', queries, sin_emb)
    # queries shape: [B, U, W, N, H]
    # sin_emb shape: [F, N, H]
    # Target output shape: [B, N, U, W, F]

    # Permute queries to [B, N, U, W, H] for easier broadcasting with sin_emb
    q_permuted = queries.permute(0, 3, 1, 2, 4)

    # Permute sin_emb to [N, H, F] to prepare for matmul
    # sin_emb original is [F, N, H]
    s_permuted = sin_emb.permute(1, 2, 0)  # Shape: [N, H, F]

    # Reshape queries for matmul: [B, N, U*W, H]
    q_reshaped = q_permuted.reshape(batch_size, num_heads, num_query_blocks * query_block_size, head_dim)

    # Perform matmul: [B, N, U*W, H] @ [N, H, F]
    # s_permuted ([N, H, F]) will be broadcast to [B, N, H, F]
    # Result: [B, N, U*W, F]
    term_bd_unshifed_matmul = torch.matmul(q_reshaped, s_permuted)

    # Reshape to target [B, N, U, W, F]
    term_bd_unshifed = term_bd_unshifed_matmul.reshape(
        batch_size,
        num_heads,
        num_query_blocks,
        query_block_size,
        max_span_plus_1,
    )

    # Apply relative shift to term_bd_unshifed
    term_bd_shifted = self._relative_shift(
        term_bd_unshifed,
        batch_size,
        num_heads,
        num_query_blocks,
        query_block_size,
        key_context_size,
        max_span_plus_1,
    )  # Shape [B, N, U, W, C]

    return term_ac + term_bd_shifted

class Gemma3nAudioRelativePositionEmbedding(nn.Module):
    def __init__(self, config: Gemma3nAudioConfig):
        super().__init__()
        self.config = config

        self.num_heads = self.config.conf_num_attention_heads
        self.channels = self.config.hidden_size
        self.head_dim = self.channels // self.num_heads
        self.max_backward = max(0, self.config.conf_attention_context_left - 1)
        self.max_forward = self.config.conf_attention_context_right

        self.pos_proj = nn.Linear(self.channels, self.num_heads * self.head_dim, bias=False)

        min_timescale = 1.0
        max_timescale = 1.0e4
        num_timescales = self.channels // 2
        log_timescale_increment = math.log(float(max_timescale) / float(min_timescale)) / max(num_timescales - 1, 1)
        inv_timescales = min_timescale * torch.exp(torch.arange(num_timescales) * -log_timescale_increment)
        self.register_buffer(
            "inv_timescales",
            inv_timescales.float().unsqueeze(0).unsqueeze(0),
            persistent=False,
        )

    def _get_timing_signal_1d_pos(self, position: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        position = position.float().unsqueeze(-1)
        scaled_time = position * self.inv_timescales.to(device=position.device, dtype=torch.float32)
        timing_signal = torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], dim=-1)
        return timing_signal.type(dtype)

    def _relative_shift(
        self,
        term_bd_before_shift: torch.Tensor,
        batch_size: int,
        num_heads: int,
        num_query_blocks: int,
        query_block_size: int,
        key_context_size: int,
        max_span_plus_1: int,
    ) -> torch.Tensor:
        """Performs the relative shift.

        Args:
          term_bd_before_shift: Tensor of shape [B, N, U, W, F_span]. batch_size
            (B), num_heads (N), num_query_blocks (U), query_block_size (W),
            key_context_size (C = W+L+R), max_span_plus_1 (F_span = L+R+1).

        Returns:
          Tensor of shape [B, N, U, W, C].
        """
        # term_bd_before_shift shape: [B, N, U, W, F_span]
        # Target shape after shift:  [B, N, U, W, C]

        # Padding amount for the last dimension (F_span) to become (C + 1)
        # C = key_context_size
        # F_span = max_span_plus_1
        pad_amount_last_dim = (key_context_size + 1) - max_span_plus_1

        # PyTorch F.pad expects (pad_left, pad_right, pad_top, pad_bottom ...)
        # We only pad the last dimension on the right.
        padding_tuple = (0, pad_amount_last_dim)

        term_bd_padded = nn.functional.pad(term_bd_before_shift, padding_tuple)
        # Shape after pad: [B, N, U, W, C+1]

        # Reshape for slicing (emulating JAX's behavior)
        # [B, N, U, W * (C+1)]
        term_bd_reshaped = term_bd_padded.reshape(
            (
                batch_size,
                num_heads,
                num_query_blocks,
                query_block_size * (key_context_size + 1),
            )
        )

        # Slice to effective [B, N, U, W * C]
        term_bd_sliced = term_bd_reshaped[:, :, :, : query_block_size * key_context_size]

        # Reshape back to [B, N, U, W, C]
        term_bd_shifted = term_bd_sliced.reshape(
            (
                batch_size,
                num_heads,
                num_query_blocks,
                query_block_size,
                key_context_size,
            )
        )
        return term_bd_shifted

    def forward(self, queries: torch.Tensor, keys: torch.Tensor) -> torch.Tensor:
        return Gemma3nAudioRelativePositionEmbedding_forward(self, queries=queries, keys=keys)


@torch.compiler.disable(recursive = False)
def Gemma3nAudioAttention_forward(self, hidden_states: torch.Tensor, mask: torch.BoolTensor) -> torch.Tensor:
    # sl.Dense uses jax.numpy.einsum("...a,abcd->...bcd") and jax.numpy.select()
    qkv_shape = (*hidden_states.shape[:-1], self.num_heads, self.head_dim)
    query_states = self.q_proj(hidden_states).reshape(qkv_shape).contiguous()
    key_states = self.k_proj(hidden_states).reshape(qkv_shape).contiguous()
    value_states = self.v_proj(hidden_states).reshape(qkv_shape).contiguous()

    per_dim_scale_sp = torch.nn.functional.softplus(self.per_dim_scale)

    broadcast_shape = (1, 1, 1, self.head_dim)
    per_dim_scale_sp_broadcast = per_dim_scale_sp.view(broadcast_shape)
    query_states = query_states * self.q_scale * per_dim_scale_sp_broadcast

    batch_size, q_time = query_states.shape[:2]

    query_blocks = self._convert_to_block(query_states)
    key_blocks = self._extract_block_context(key_states)
    value_blocks = self._extract_block_context(value_states)
    num_query_blocks = query_blocks.shape[1]

    # 1. Create a mask indicating originally valid positions.
    original_valid_mask = ~mask  # True for valid, False for padded

    # 2. Extract blocks from this validity mask.
    extracted_valid_mask_blocks = self._extract_block_context(original_valid_mask)

    # If subframe_factor was used in _extract_block_context for a [B, T] input mask,
    # the shape might be [B, U, C/SF, SF]. Reshape to [B, U, C].
    # batch_size and num_query_blocks are known from query_blocks.
    # self.context_size is C.
    if (
        extracted_valid_mask_blocks.ndim == 4
        and extracted_valid_mask_blocks.shape[2] * extracted_valid_mask_blocks.shape[3] == self.context_size
    ):
        extracted_valid_mask_blocks = extracted_valid_mask_blocks.reshape(
            batch_size, num_query_blocks, self.context_size
        )
    # After potential reshape, ensure it's [B, U, C] if it was from a [B,T] mask.
    # This assertion might be too strict if _extract_block_context handles higher-rank inputs differently,
    # but for the mask case, this should hold.
    if extracted_valid_mask_blocks.shape != (
        batch_size,
        num_query_blocks,
        self.context_size,
    ):
        raise ValueError(
            "Shape of extracted_valid_mask_blocks"
            f" {extracted_valid_mask_blocks.shape} is not ({batch_size},"
            f" {num_query_blocks}, {self.context_size}) after potential reshape."
        )

    # 3. Expand dimensions for broadcasting with logits and causal mask.
    # Target shape for broadcasting with logits [B,N,U,W,C]
    # extracted_valid_mask_blocks to [B, 1, U, 1, C]
    condition_from_input_validity = extracted_valid_mask_blocks.unsqueeze(1).unsqueeze(-2)

    # self.local_causal_valid_mask is [W, C], True where allowed by local window.
    # Expand to [1, 1, 1, W, C]
    condition_from_causality = self.local_causal_valid_mask.unsqueeze(0).unsqueeze(0).unsqueeze(0)

    # 4. Combine the two conditions.
    # final_condition will be True where a key is *both* originally valid *and* causally accessible.
    # Broadcasts to [B, 1, U, W, C]
    final_condition_for_where = torch.logical_and(
        condition_from_input_validity,
        condition_from_causality.to(condition_from_input_validity.device),  # Ensure same device
    )

    # Embed queries and keys
    logits = self.relative_position_embedding(query_blocks, key_blocks)

    # Apply attention logit softcap
    # Ensure softcap is on the same device as logits
    softcap_val = self.softcap.to(logits.device)
    logits = logits / softcap_val
    logits = torch.tanh(logits)
    logits = logits * softcap_val

    # Apply the combined mask.
    # final_condition_for_where will broadcast with logits [B,N,U,W,C]
    logits = torch.where(final_condition_for_where, logits, torch.finfo(logits.dtype).min)
    probabilities = torch.nn.functional.softmax(logits, dim=-1, dtype = torch.float32).to(logits.dtype).to(logits.dtype).to(dtype=value_blocks.dtype)

    # context_vectors is adapted from jax.numpy.einsum("BNuwc,BucNH->BuwNH", ...)
    b_dim, n_dim, u_dim, w_dim, c_dim = probabilities.shape
    h_dim = value_blocks.shape[-1]
    prob_bun = probabilities.permute(0, 2, 1, 3, 4).reshape(-1, w_dim, c_dim)
    v_bun = value_blocks.permute(0, 1, 3, 2, 4).reshape(-1, c_dim, h_dim)
    result_bmm = torch.bmm(prob_bun, v_bun)
    context_vectors = result_bmm.reshape(b_dim, u_dim, n_dim, w_dim, h_dim).permute(0, 1, 3, 2, 4)
    context_vectors = context_vectors.reshape(
        (
            batch_size,
            num_query_blocks * self.chunk_size,
            self.num_heads,
            self.head_dim,
        )
    )
    context_vectors = context_vectors[:, :q_time]

    return context_vectors

class Gemma3nAudioAttention(nn.Module):
    def __init__(self, config: Gemma3nAudioConfig):
        super().__init__()
        self.config = config

        self.num_heads = self.config.conf_num_attention_heads
        self.hidden_size = self.config.hidden_size
        self.head_dim = self.hidden_size // self.num_heads

        self.chunk_size = self.config.conf_attention_chunk_size
        self.max_future_horizon = self.config.conf_attention_context_right
        self.max_past_horizon = max(0, self.config.conf_attention_context_left - 1)
        self.attention_logits_soft_cap = self.config.conf_attention_logit_cap
        self.context_size = self.chunk_size + self.max_past_horizon + self.max_future_horizon

        self.relative_position_embedding = Gemma3nAudioRelativePositionEmbedding(config)
        self.per_dim_scale = nn.Parameter(torch.zeros((self.head_dim,)))

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)

        q_scale = self.head_dim**-0.5
        r_softplus_0 = 1.0 / torch.nn.functional.softplus(torch.tensor(0.0))
        self.register_buffer("q_scale", (q_scale * r_softplus_0).clone().detach(), persistent=False)

        lower_causal_mask = torch.tril(
            torch.ones((self.context_size, self.chunk_size), dtype=torch.bool),
            diagonal=0,
        ).T
        upper_causal_mask = torch.tril(
            torch.ones((self.chunk_size, self.context_size), dtype=torch.bool),
            diagonal=self.max_past_horizon + self.max_future_horizon,
        )
        local_causal_valid_mask = torch.ones((self.chunk_size, self.context_size), dtype=torch.bool)
        local_causal_valid_mask = local_causal_valid_mask * lower_causal_mask * upper_causal_mask
        self.register_buffer("local_causal_valid_mask", local_causal_valid_mask, persistent=False)

        self.register_buffer(
            "softcap",
            torch.tensor(self.attention_logits_soft_cap).float(),
            persistent=False,
        )

    def _pad_dim1(self, x: torch.Tensor, pad_left: int, pad_right: int) -> torch.Tensor:
        batch, _, *tail_shape = x.shape
        left = x.new_zeros((batch, pad_left, *tail_shape))
        right = x.new_zeros((batch, pad_right, *tail_shape))
        x = torch.cat([left, x, right], dim=1)
        return x

    def _convert_to_block(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Turns a sequence to non overlapping blocks.

        Args:
            hidden_states: a tensor of [batch, time, ...].

        Returns:
            A tensor of [batch, num_blocks, block_size, ...], with necessary
            paddings,
            where output[:, i, ...] are x[:, i*block_size:(i+1)*block_size, ...].
        """
        shape = hidden_states.shape
        b, t = shape[:2]
        num_blocks = (t + self.chunk_size - 1) // self.chunk_size

        if (padding_len := num_blocks * self.chunk_size - t) > 0:
            hidden_states = self._pad_dim1(hidden_states, 0, padding_len)

        permute_dims = (b, num_blocks, self.chunk_size) + shape[2:]
        hidden_states = hidden_states.reshape(permute_dims).contiguous()
        return hidden_states

    def _extract_block_context(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Extracts temporal context for every block.

        Args:
            hidden_states: a tensor of [batch, time, ...].

        Returns:
            A tensor of [batch, num_blocks, context_size, ...], with necessary
            paddings,
            where context_size = block_size + left_context + right_context,
            and output[:, i, ...] are x[:, start-left_context:end+right_context,
            ...],
            start = i * block_size, end = (i + 1) * block_size.
        """
        pad_left = self.max_past_horizon
        # The JAX equivalent padding for signal.frame with pad_mode='valid' is
        # (left_context, right_context + block_size - 1) on the time dimension.
        # PyTorch's _pad_dim1 applies padding symmetrically if only one value is given,
        # or (pad_dim_start, pad_dim_end) if two are given.
        # Our _pad_dim1(x, pad_left, pad_right) pads dim -2 (time for [B,T,N,H])
        # or dim 1 (time for [B,T]).
        # The current pad_right calculation matches the JAX effective padding.
        pad_right = self.max_future_horizon + self.chunk_size - 1
        hidden_states = self._pad_dim1(hidden_states, pad_left, pad_right)

        frame_len = self.context_size
        frame_step = self.chunk_size

        # Directly use unfold without the subframe_factor logic
        # x.unfold(dimension, size, step)
        # dimension=1 (time dimension, assuming x is [B, T_padded, ...])
        # size=frame_len (context_size)
        # step=frame_step (chunk_size)
        x_unfolded = hidden_states.unfold(dimension=1, size=frame_len, step=frame_step)

        # If x was [B, T_padded], x_unfolded is [B, num_blocks, frame_len]
        # If x was [B, T_padded, N, H], x_unfolded is [B, num_blocks, N, H, frame_len]
        # We want to match JAX's typical output for such operations which might be
        # [B, num_blocks, frame_len, N, H] if N, H are present.
        # The relative_position_embedding expects keys as [B, U, C, N, H].
        # If x_unfolded is [B, U, N, H, C(frame_len)], we need to move C.
        if hidden_states.ndim > 2 and x_unfolded.ndim > 3:  # Check if inner dimensions (like N, H) exist
            # Current shape after unfold for [B, T_pad, N, H] is [B, U, N, H, C]
            # Target shape for keys in RPE: [B, U, C, N, H]
            x_unfolded = torch.movedim(x_unfolded, source=-1, destination=2)

        return x_unfolded.contiguous()

    def forward(self, hidden_states: torch.Tensor, mask: torch.BoolTensor) -> torch.Tensor:
        return Gemma3nAudioAttention_forward(self, hidden_states=hidden_states, mask=mask)


@torch.compiler.disable(recursive = False)
def Gemma3nAudioCumulativeGroupNorm_forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
    """Applies cumulative group norm, optionally using a mask.

    Args:
      hidden_states: Input tensor, shape [B, T, *feature_dims, C].

    Returns:
      Normalized tensor with the same shape as x.
    """
    expected_input_suffix = self.feature_dims + (self.num_channels,)
    if hidden_states.shape[2:] != expected_input_suffix:
        raise ValueError(
            f"Input tensor shape suffix {hidden_states.shape[2:]} does not match expected"
            f" suffix (feature_dims + num_channels) {expected_input_suffix}"
        )

    input_dtype = hidden_states.dtype
    # Calculations are performed in float32 for numerical stability.
    calc_dtype = torch.float32
    x_calc = hidden_states.to(calc_dtype)

    # Prepare a broadcastable mask (`mask_calc`).
    # If no mask is provided, treat all elements as valid
    # (mask_calc is all ones).
    # Otherwise, expand the [B, T] mask to [B, T, 1, ..., 1] for broadcasting.
    mask_calc = torch.ones_like(x_calc, dtype=calc_dtype)

    # Cumulative Statistics Calculation
    # 1. Sum of values over reduction axes at each time step.
    sum_values_at_t = torch.sum(x_calc, dim=self.reduction_axes, keepdim=True)
    # 2. Cumulative sum of values over time.
    cum_sum_values = torch.cumsum(sum_values_at_t, dim=1)

    # 3. Count of valid elements in the normalization group at each time step.
    #    (A "group" here consists of all features at a given Batch, Time).
    elements_in_group_at_t = torch.sum(mask_calc, dim=self.reduction_axes, keepdim=True)
    # 4. Cumulative count of valid elements over time.
    cum_count_elements = torch.cumsum(elements_in_group_at_t, dim=1)
    # Avoid division by zero if all preceding elements were masked.
    safe_cum_count_elements = torch.clamp(cum_count_elements, min=1.0)

    # 5. Cumulative mean.
    cum_mean = cum_sum_values / safe_cum_count_elements

    # 6. Sum of squared differences from the cumulative mean.
    #    Only sum for valid elements: (x_calc - cum_mean)^2 * mask_calc.
    #    Using x_calc here for the difference, as cum_mean already accounts for masking.
    squared_diff_from_mean = (x_calc - cum_mean).pow(2)
    sum_sq_diff_at_t = torch.sum(squared_diff_from_mean, dim=self.reduction_axes, keepdim=True)

    # 7. Cumulative sum of squared differences over time.
    cum_sum_sq_diff = torch.cumsum(sum_sq_diff_at_t, dim=1)

    # 8. Cumulative variance.
    cum_variance = cum_sum_sq_diff / safe_cum_count_elements

    # Normalize the input using the calculated cumulative statistics:
    # (x - E[x]) / sqrt(Var[x] + eps)
    normalized_x = (x_calc - cum_mean) * torch.rsqrt(cum_variance + self.eps)

    # Apply affine transformation (scale and bias) if enabled.
    # Scale and bias are applied per-channel (last dimension).
    scale = self.weight.to(calc_dtype)
    # Reshape for broadcasting: [C] -> [1, ..., 1, C]
    scale_view_shape = [1] * (hidden_states.dim() - 1) + [self.num_channels]
    normalized_x = normalized_x * scale.view(scale_view_shape)

    # Zero out outputs for time steps that were originally masked (where mask_calc is 0).
    # This ensures padded/invalid positions in the input result in zero output.
    final_output = normalized_x * mask_calc

    return final_output.to(input_dtype)

class Gemma3nAudioCumulativeGroupNorm(nn.Module):
    """Applies Group Normalization cumulatively over the time dimension.

    This layer normalizes the input by calculating the mean and variance
    cumulatively over the time dimension (dim 1). The statistics are computed
    over all feature dimensions (specified by `feature_dims` and `num_channels`)
    for elements marked as valid by the optional `mask`.

    If a `mask` is provided (True for valid, False for invalid/padded),
    invalid time steps do not contribute to the statistics calculation, and
    their corresponding output values are zeroed out.

    Scale and bias, if enabled, are applied per-channel (last dimension).
    This behavior is similar to JAX's `GroupNormalization` with `num_groups=1`
    and `cumulative=True`.
    """

    def __init__(
        self,
        num_channels: int,  # Number of channels (size of the last dimension)
        feature_dims: Sequence[int],  # Sizes of non-channel feature dimensions, e.g., (H, W) for input [B,T,H,W,C]
        eps: float = 1e-3,
    ):
        super().__init__()
        self.num_channels = num_channels
        self.feature_dims = tuple(feature_dims)
        self.eps = eps

        # Scale parameter depends only on the channel dimension
        self.weight = nn.Parameter(torch.ones(num_channels))

        # Axes for normalization: all dimensions except Batch (0) and Time (1).
        # For input [B, T, *feature_dims, C], these are dims from 2 onwards.
        self.reduction_axes = tuple(range(2, 2 + len(self.feature_dims) + 1))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return Gemma3nAudioCumulativeGroupNorm_forward(self, hidden_states=hidden_states)


@torch.compiler.disable(recursive = False)
def Gemma3nAudioSubSampleConvProjection_forward(self, audio_encodings: torch.Tensor) -> torch.Tensor:
    # audio_encodings is [B, T, F_in]
    # Reshape to [B, 1, T, F_in] (Batch, Channels=1, Height=Time, Width=F_in)
    audio_encodings_reshaped = audio_encodings.unsqueeze(1)
    x = self.conv_0(audio_encodings_reshaped)
    x = self.conv_1(x)
    # x from conv_1 is [B, C_out_1, T_out_1, F_out_1]
    b, c_out, t_out, f_out = x.shape
    # Permute to [B, T_out_1, F_out_1, C_out_1] then flatten F_out_1 and C_out_1
    x_permuted = x.permute(0, 2, 3, 1).contiguous()
    output_flattened = x_permuted.view(b, t_out, f_out * c_out)
    output = self.input_proj_linear(output_flattened)
    return output

class Gemma3nAudioSubSampleConvProjection(nn.Module):
    def __init__(self, config: Gemma3nAudioConfig):
        super().__init__()
        self.config = config

        current_f_for_block_input = config.input_feat_size  # Start with original feature dim
        calculated_block_padding = []
        calculated_f_out_dims = []  # Tracking frequency dimension output sizes

        for i in range(2):  # Assuming 2 conv layers as per sscp_conv_... arrays
            kernel_h, kernel_w = config.sscp_conv_kernel_size[i]
            stride_h, stride_w = config.sscp_conv_stride_size[i]

            # Padding for Time (Height for Conv2d) - REVERSE_CAUSAL like
            # JAX 'reverse_causal' padding is (0, kernel_size - 1)
            pad_t_top = 0
            pad_t_bottom = kernel_h - 1

            # Frequency Padding (Width for Conv2d)
            # Based on JAX effective padding (1,1) for F_in=10, K_w=3, S_w=2
            # and the successful test configuration.
            # If kernel/stride/input_freq for frequency changes, this might need re-evaluation
            # to match generic JAX 'SAME' behavior if it differs.
            pad_f_left = 1
            pad_f_right = 1

            manual_padding_tuple = (
                pad_f_left,
                pad_f_right,
                pad_t_top,
                pad_t_bottom,
            )
            calculated_block_padding.append(manual_padding_tuple)

            # Calculate output frequency dimension after this convolution
            # This uses the actual padding applied and kernel/stride.
            f_in_padded = current_f_for_block_input + pad_f_left + pad_f_right
            f_out_after_conv = (f_in_padded - kernel_w) // stride_w + 1  # Assuming dilation_w = 1
            calculated_f_out_dims.append(f_out_after_conv)
            current_f_for_block_input = f_out_after_conv

        self.conv_0 = Gemma3nAudioSSCPConvBlock(
            idx=0,
            input_freq_dim=config.input_feat_size,  # Pass original feature dim
            config=config,
            manual_padding=calculated_block_padding[0],
        )
        self.conv_1 = Gemma3nAudioSSCPConvBlock(
            idx=1,
            input_freq_dim=calculated_f_out_dims[0],  # Output freq dim from conv_0
            config=config,
            manual_padding=calculated_block_padding[1],
        )
        final_c_out = config.sscp_conv_channel_size[-1]
        final_f_out = calculated_f_out_dims[-1]  # Final frequency dimension
        self.input_proj_in_features = final_c_out * final_f_out
        self.input_proj_linear = nn.Linear(self.input_proj_in_features, self.config.hidden_size, bias=False)

    def forward(self, audio_encodings: torch.Tensor) -> torch.Tensor:
        return Gemma3nAudioSubSampleConvProjection_forward(self, audio_encodings=audio_encodings)


@torch.compiler.disable(recursive = False)
def Gemma3nAudioConformerAttention_forward(self, audio_encodings: torch.Tensor, audio_mel_mask: torch.BoolTensor) -> torch.Tensor:
    audio_encodings_input_to_attn = audio_encodings
    audio_encodings = torch.clamp(audio_encodings, -self.gradient_clipping, self.gradient_clipping)
    audio_encodings_norm = self.pre_attn_norm(audio_encodings)
    # Output of self.attn is [B, T, NumHeads, HeadDim]
    audio_encodings_attn_out = self.attn(audio_encodings_norm, audio_mel_mask)

    # Reshape from [B, T, NumHeads, HeadDim] to [B, T, NumHeads * HeadDim]
    # NumHeads * HeadDim = hidden_size
    b, t, num_heads, head_dim = audio_encodings_attn_out.shape
    audio_encodings_reshaped = audio_encodings_attn_out.reshape(b, t, num_heads * head_dim)

    audio_encodings = self.post(audio_encodings_reshaped)
    audio_encodings = torch.clamp(audio_encodings, -self.gradient_clipping, self.gradient_clipping)
    return audio_encodings_input_to_attn + self.post_norm(audio_encodings)

class Gemma3nAudioConformerAttention(nn.Module):
    def __init__(self, config: Gemma3nAudioConfig):
        super().__init__()
        self.config = config
        self.post_in_features = self.config.hidden_size
        self.register_buffer("gradient_clipping", torch.tensor(self.config.gradient_clipping), persistent=False)
        self.pre_attn_norm = Gemma3nRMSNorm(self.config.hidden_size)
        self.attn = Gemma3nAudioAttention(config)
        self.post = nn.Linear(self.post_in_features, self.config.hidden_size, bias=False)
        self.post_norm = Gemma3nRMSNorm(self.config.hidden_size)

    def forward(self, audio_encodings: torch.Tensor, audio_mel_mask: torch.BoolTensor) -> torch.Tensor:
        return Gemma3nAudioConformerAttention_forward(self, audio_encodings=audio_encodings, audio_mel_mask=audio_mel_mask)


@torch.compiler.disable(recursive = False)
def Gemma3nAudioConformerFeedForward_forward(self, audio_encodings: torch.Tensor) -> torch.Tensor:
    residual = audio_encodings
    audio_encodings = torch.clamp(audio_encodings, -self.gradient_clipping, self.gradient_clipping)
    audio_encodings = self.pre_layer_norm(audio_encodings)
    audio_encodings: torch.Tensor = self.ffw_layer_1(audio_encodings)
    audio_encodings = nn.functional.silu(audio_encodings)
    audio_encodings: torch.Tensor = self.ffw_layer_2(audio_encodings)
    audio_encodings = torch.clamp(audio_encodings, -self.gradient_clipping, self.gradient_clipping)
    audio_encodings = self.post_layer_norm(audio_encodings)
    return residual + (audio_encodings * self.post_layer_scale)

class Gemma3nAudioConformerFeedForward(nn.Module):
    def __init__(self, config: Gemma3nAudioConfig):
        super().__init__()
        self.config = config

        self.register_buffer("gradient_clipping", torch.tensor(self.config.gradient_clipping), persistent=False)

        self.pre_layer_norm = Gemma3nRMSNorm(self.config.hidden_size)
        self.ffw_layer_1 = nn.Linear(self.config.hidden_size, self.config.hidden_size * 4, bias=False)
        self.ffw_layer_2 = nn.Linear(self.config.hidden_size * 4, self.config.hidden_size, bias=False)
        self.post_layer_norm = Gemma3nRMSNorm(self.config.hidden_size)
        self.post_layer_scale = torch.tensor(self.config.conf_residual_weight)

    def forward(self, audio_encodings: torch.Tensor) -> torch.Tensor:
        return Gemma3nAudioConformerFeedForward_forward(self, audio_encodings=audio_encodings)


@torch.compiler.disable(recursive = False)
def Gemma3nAudioConformerBlock_forward(self, audio_encodings: torch.Tensor, audio_mel_mask: torch.BoolTensor) -> torch.Tensor:
    audio_encodings = self.ffw_layer_start(audio_encodings)
    audio_encodings = self.attention(audio_encodings, audio_mel_mask)
    validity_mask_for_lconv = ~audio_mel_mask  # True for valid
    audio_encodings_for_lconv_input = audio_encodings * validity_mask_for_lconv.unsqueeze(-1).to(
        audio_encodings.dtype
    )
    audio_encodings = self.lconv1d(audio_encodings_for_lconv_input)

    audio_encodings = self.ffw_layer_end(audio_encodings)
    audio_encodings = torch.clamp(audio_encodings, -self.gradient_clipping, self.gradient_clipping)
    output = self.norm(audio_encodings)
    return output

class Gemma3nAudioConformerBlock(nn.Module):
    def __init__(self, config: Gemma3nAudioConfig):
        super().__init__()
        self.config = config

        self.ffw_layer_start = Gemma3nAudioConformerFeedForward(self.config)
        self.attention = Gemma3nAudioConformerAttention(self.config)
        self.lconv1d = Gemma3nAudioConformerLightConv1d(self.config)
        self.ffw_layer_end = Gemma3nAudioConformerFeedForward(self.config)
        self.register_buffer("gradient_clipping", torch.tensor(self.config.gradient_clipping), persistent=False)
        self.norm = Gemma3nRMSNorm(self.config.hidden_size)

    def forward(self, audio_encodings: torch.Tensor, audio_mel_mask: torch.BoolTensor) -> torch.Tensor:
        return Gemma3nAudioConformerBlock_forward(self, audio_encodings=audio_encodings, audio_mel_mask=audio_mel_mask)


@torch.compiler.disable(recursive = False)
def Gemma3nTextLaurelBlock_forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
    laurel_hidden_states: torch.Tensor = self.linear_left(hidden_states)
    laurel_hidden_states: torch.Tensor = self.linear_right(laurel_hidden_states)
    normed_laurel_hidden_states = self.post_laurel_norm(laurel_hidden_states)
    return hidden_states + normed_laurel_hidden_states

class Gemma3nTextLaurelBlock(nn.Module):
    """Learned Augmented Residual Layer"""

    def __init__(self, config: Gemma3nTextConfig):
        super().__init__()
        self.config = config

        self.linear_left = nn.Linear(self.config.hidden_size, self.config.laurel_rank, bias=False)
        self.linear_right = nn.Linear(self.config.laurel_rank, self.config.hidden_size, bias=False)
        self.post_laurel_norm = Gemma3nRMSNorm(self.config.hidden_size, eps=self.config.rms_norm_eps)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return Gemma3nTextLaurelBlock_forward(self, hidden_states=hidden_states)


@torch.compiler.disable(recursive = False)
def Gemma3nTextMLP_forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
    gate_proj = self.gate_proj(hidden_states)
    if self.activation_sparsity > 0.0:
        gate_proj = self._gaussian_topk(gate_proj)
    activations = self.act_fn(gate_proj)
    up_proj = self.up_proj(hidden_states)
    down_proj = self.down_proj(activations * up_proj)
    return down_proj

class Gemma3nTextMLP(nn.Module):
    def __init__(self, config: Gemma3nTextConfig, layer_idx: int = 0):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size[layer_idx]
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_activation]
        self.activation_sparsity = config.activation_sparsity_pattern[layer_idx]

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return Gemma3nTextMLP_forward(self, hidden_states=hidden_states)

    def _gaussian_topk(self, inputs: torch.Tensor) -> torch.Tensor:
        target_sparsity_tensor = torch.tensor(self.activation_sparsity, dtype=torch.float32, device=inputs.device)
        # normal_dist and std_multiplier are adapted from jax.scipy.stats.norm.ppf().
        #
        # References:
        #   *   https://docs.jax.dev/en/latest/_autosummary/jax.scipy.stats.norm.ppf.html
        #   *   https://pytorch.org/docs/stable/distributions.html#torch.distributions.normal.Normal
        #   *   https://pytorch.org/docs/stable/distributions.html#torch.distributions.transformed_distribution.TransformedDistribution.icdf
        normal_dist = torch.distributions.normal.Normal(0, 1)
        std_multiplier: torch.Tensor = normal_dist.icdf(target_sparsity_tensor)
        std_multiplier = std_multiplier.type(inputs.dtype)
        inputs_mean = torch.mean(inputs, dim=-1, keepdim=True)
        inputs_std = torch.std(inputs, dim=-1, keepdim=True, unbiased=False)
        cutoff_x = inputs_mean + inputs_std * std_multiplier
        return nn.functional.relu(inputs - cutoff_x)


@torch.compiler.disable(recursive = False)
def Gemma3nTextAltUp_forward(self, corrected: torch.Tensor) -> torch.Tensor:
    """
    This is only defined as the `forward` so that accelerate hooks can move correctly `correct_output_scale`
    (which is a nn.Parameter, not a Module) between devices when offloading. It is otherwise only used in
    `scale_corrected_output`
    """
    return (corrected.type_as(self.correct_output_scale) * self.correct_output_scale).type_as(corrected)

class Gemma3nTextAltUp(nn.Module):
    """Alternating Updates (AltUp)

    The AltUp module wraps transformer layers. The `predict` step modifies the
    input to the transformer layer, and the `correct` step propagates the output
    of the transformer layer to the sparsely updated dimensions.

    See more in the research paper:

    https://proceedings.neurips.cc/paper_files/paper/2023/file/f2059277ac6ce66e7e5543001afa8bb5-Paper-Conference.pdf
    """

    def __init__(self, config: Gemma3nTextConfig):
        super().__init__()
        self.config = config
        self.correct_output_scale = nn.Parameter(torch.zeros(self.config.hidden_size))
        self.correction_coefs = nn.Linear(self.config.altup_num_inputs, self.config.altup_num_inputs, bias=False)
        self.prediction_coefs = nn.Linear(self.config.altup_num_inputs, self.config.altup_num_inputs**2, bias=False)
        self.modality_router = nn.Linear(self.config.hidden_size, self.config.altup_num_inputs, bias=False)
        self.router_norm = Gemma3nRMSNorm(self.config.hidden_size, eps=self.config.rms_norm_eps)
        self.register_buffer("router_input_scale", torch.tensor(self.config.hidden_size**-1.0), persistent=False)

    def compute_router_modalities(self, x: torch.Tensor) -> torch.Tensor:
        router_inputs = self.router_norm(x) * self.router_input_scale
        routed = self.modality_router(router_inputs)
        return torch.tanh(routed.float()).type_as(x)

    def predict(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Predicts the output of a layer using a trainable map.

        Args:
            hidden_states: A 4D tensor of shape `[num_altup_inputs, batch_size, num_tokens, hidden_size]` derived by
                stacking the input embeddings and preprocessing the last `num_altup_inputs - 1` matrices.

        Returns:
            A 4D tensor of shape `[num_altup_inputs, batch_size, num_tokens, hidden_size]` containing the predictions.
        """
        modalities = self.compute_router_modalities(hidden_states[self.config.altup_active_idx])

        if self.training and self.config.altup_coef_clip is not None:
            self.prediction_coefs.weight.data.clamp_(-self.config.altup_coef_clip, self.config.altup_coef_clip)

        # Project and then transpose all 2D matrices contained so that mulmat gives the correct result
        all_coefs: torch.Tensor = (
            self.prediction_coefs(modalities)
            .reshape(*modalities.shape[:-1], self.config.altup_num_inputs, self.config.altup_num_inputs)
            .permute(0, 1, 3, 2)
        )

        # permute hidden_states to [batch_size, num_tokens, hidden_size, altup_num_inputs]
        predictions = torch.matmul(hidden_states.permute(1, 2, 3, 0), all_coefs)
        predictions = predictions.permute(3, 0, 1, 2)  # undo the permute
        predictions += hidden_states  # add the original input
        return predictions.contiguous().type_as(hidden_states)

    def correct(self, predictions: torch.Tensor, activated: torch.Tensor) -> torch.Tensor:
        """Corrects the predictions relative to the

        Args:
            predictions: A 4D tensor of shape `[num_altup_inputs, batch_size, num_tokens, hidden_size]` derived by
                stacking the input embeddings and preprocessing the last `num_altup_inputs - 1` matrices.
            activated: A 3D tensor of shape `[batch_size, num_tokens, hidden_size]` containing the activated inputs.

        Returns:
            A 4D tensor of shape `[num_altup_inputs, batch_size, num_tokens, hidden_size]` correcting the original
                predictions relative to the activated input embeddings.
        """
        modalities = self.compute_router_modalities(activated)
        innovation = activated - predictions[self.config.altup_active_idx]  # (batch, num_tokens, hidden_size)
        innovation = innovation.repeat(self.config.altup_num_inputs, 1, 1, 1)  # Repeat on dim0 to match predictions

        if self.config.altup_coef_clip is not None:
            self.correction_coefs.weight.data.clamp_(-self.config.altup_coef_clip, self.config.altup_coef_clip)

        # all_coefs adapted from jax.numpy.einsum("...p,pi->...i", ...)
        # Permute to (altup_num_inputs, batch_size, num_tokens) as the last dim is a scalar applied to each altup input
        # and expand on dim1 for broadcastability
        all_coefs: torch.Tensor = self.correction_coefs(modalities) + 1.0
        all_coefs = all_coefs.permute(2, 0, 1).unsqueeze(-1)

        corrected = torch.mul(innovation, all_coefs)
        corrected += predictions  # add the original input
        return corrected.contiguous().type_as(activated)

    def forward(self, corrected: torch.Tensor) -> torch.Tensor:
        return Gemma3nTextAltUp_forward(self, corrected=corrected)

    def scale_corrected_output(self, corrected: torch.Tensor) -> torch.Tensor:
        """Scales the provided 3D tensor of shape [batch_size, num_tokens, hidden_size]."""
        return self.forward(corrected)


@torch.compiler.disable(recursive = False)
@torch.no_grad()
@dynamic_rope_update  # power user: used with advanced RoPE types (e.g. dynamic rope)
def Gemma3nTextRotaryEmbedding_forward(self, x, position_ids):
    inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1).to(x.device)
    position_ids_expanded = position_ids[:, None, :].float()

    device_type = x.device.type if isinstance(x.device.type, str) and x.device.type != "mps" else "cpu"
    with torch.autocast(device_type=device_type, enabled=False):  # Force float32
        freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos() * self.attention_scaling
        sin = emb.sin() * self.attention_scaling

    return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)

class Gemma3nTextRotaryEmbedding(nn.Module):
    inv_freq: torch.Tensor  # fix linting for `register_buffer`

    def __init__(self, config: Gemma3nTextConfig, device=None):
        super().__init__()
        # BC: "rope_type" was originally "type"
        if hasattr(config, "rope_scaling") and isinstance(config.rope_scaling, dict):
            self.rope_type = config.rope_scaling.get("rope_type", config.rope_scaling.get("type"))
        else:
            self.rope_type = "default"
        self.max_seq_len_cached = config.max_position_embeddings
        self.original_max_seq_len = config.max_position_embeddings

        self.config = config
        self.rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type]

        inv_freq, self.attention_scaling = self.rope_init_fn(self.config, device)
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.original_inv_freq = self.inv_freq


    def forward(self, x, position_ids):
        return Gemma3nTextRotaryEmbedding_forward(self, x=x, position_ids=position_ids)


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    dropout: float = 0.0,
    scaling: Optional[float] = None,
    softcap: Optional[float] = None,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
    if scaling is None:
        scaling = module.head_dim**-0.5

    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)

    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling

    if softcap is not None:
        attn_weights = attn_weights / softcap
        attn_weights = torch.tanh(attn_weights)
        attn_weights = attn_weights * softcap
    if attention_mask is not None:  # no matter the length, we just slice it
        causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
        attn_weights = attn_weights + causal_mask

    # upcast attention to fp32
    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype = torch.float32).to(attn_weights.dtype).to(query.dtype)
    attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()
    return attn_output, attn_weights


def apply_rotary_pos_emb(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    position_ids: Optional[torch.Tensor] = None,
    unsqueeze_dim: int = 1,
):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        x (`torch.Tensor`): The tensor to embed.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`, *optional*):
            Deprecated and unused.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    return (x * cos) + (rotate_half(x) * sin)


@torch.compiler.disable(recursive = False)
@deprecate_kwarg("past_key_value", new_name="past_key_values", version="4.58")
def Gemma3nTextAttention_forward(
    self,
    hidden_states: torch.Tensor,
    position_embeddings: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    past_key_values: Optional[Cache] = None,
    cache_position: Optional[torch.LongTensor] = None,
    **kwargs: Unpack[FlashAttentionKwargs],
) -> tuple[torch.Tensor, Optional[torch.Tensor], Optional[tuple[torch.Tensor]]]:
    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, self.config.head_dim)

    cos, sin = position_embeddings

    query_states = self.q_proj(hidden_states).view(hidden_shape)
    query_states = self.q_norm(query_states)
    query_states = apply_rotary_pos_emb(query_states, cos, sin, unsqueeze_dim=2)
    query_states = query_states.transpose(1, 2)

    # For layers with shared KV (from kv sharing point onwards), we reuse the same keys/values states as the last non-sharing layer
    if self.is_kv_shared_layer and past_key_values is not None:
        key_states, value_states = past_key_values.shared_layers[self.kv_shared_layer_index]
        # Device of past layer may be different from current one
        key_states = key_states.to(query_states.device)
        value_states = value_states.to(query_states.device)
    else:
        key_states = self.k_proj(hidden_states).view(hidden_shape)
        key_states = self.k_norm(key_states)
        key_states = apply_rotary_pos_emb(key_states, cos, sin, unsqueeze_dim=2)
        key_states = key_states.transpose(1, 2)

        value_states = self.v_proj(hidden_states).view(hidden_shape)
        value_states = self.v_norm(value_states)
        value_states = value_states.transpose(1, 2)

    if past_key_values is not None:
        # sin and cos are specific to RoPE models; cache_position needed for the static cache
        cache_kwargs = {
            "sin": sin,
            "cos": cos,
            "cache_position": cache_position,
            "sliding_window": self.sliding_window,
        }
        if not self.is_kv_shared_layer:
            key_states, value_states = past_key_values.update(
                key_states, value_states, self.layer_idx, cache_kwargs
            )
        if self.store_full_length_kv:
            if not hasattr(past_key_values, "shared_layers"):
                past_key_values.shared_layers = {}
            past_key_values.shared_layers[self.layer_idx] = key_states, value_states

    attention_interface: Callable = eager_attention_forward
    if self.config._attn_implementation != "eager":
        attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

    attn_output, attn_weights = attention_interface(
        self,
        query_states,
        key_states,
        value_states,
        attention_mask,
        dropout=self.attention_dropout if self.training else 0.0,
        scaling=1.0,
        sliding_window=self.sliding_window,
        **kwargs,
    )

    attn_output = attn_output.reshape(*input_shape, -1).contiguous()
    attn_output = self.o_proj(attn_output)
    return attn_output, attn_weights

class Gemma3nTextAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: Gemma3nTextConfig, layer_idx: int):
        super().__init__()
        self.is_sliding = config.layer_types[layer_idx] == "sliding_attention"
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.attention_dropout = self.config.attention_dropout
        self.is_causal = True

        self.q_proj = nn.Linear(
            config.hidden_size, config.num_attention_heads * self.head_dim, bias=config.attention_bias
        )
        self.k_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.v_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.o_proj = nn.Linear(
            config.num_attention_heads * self.head_dim, config.hidden_size, bias=config.attention_bias
        )
        self.sliding_window = config.sliding_window if self.is_sliding else None

        self.q_norm = Gemma3nRMSNorm(dim=config.head_dim, eps=config.rms_norm_eps)
        self.k_norm = Gemma3nRMSNorm(dim=config.head_dim, eps=config.rms_norm_eps)
        self.v_norm = Gemma3nRMSNorm(dim=config.head_dim, eps=config.rms_norm_eps, with_scale=False)

        first_kv_shared_layer_idx = self.config.num_hidden_layers - self.config.num_kv_shared_layers
        self.is_kv_shared_layer = layer_idx >= first_kv_shared_layer_idx > 0
        prev_layers = config.layer_types[:first_kv_shared_layer_idx]
        if self.is_kv_shared_layer:
            # For shared layers, find the last non-shared layer of the same type before sharing starts
            self.kv_shared_layer_index = len(prev_layers) - 1 - prev_layers[::-1].index(config.layer_types[layer_idx])
            self.store_full_length_kv = False
        else:
            self.kv_shared_layer_index = None
            # For non-shared layers, store full-length kv if this is the last non-shared layer of its type
            self.store_full_length_kv = layer_idx == len(prev_layers) - 1 - prev_layers[::-1].index(
                config.layer_types[layer_idx]
            )


    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        past_key_values: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple[torch.Tensor, Optional[torch.Tensor], Optional[tuple[torch.Tensor]]]:
        return Gemma3nTextAttention_forward(self, hidden_states=hidden_states, position_embeddings=position_embeddings, attention_mask=attention_mask, past_key_values=past_key_values, cache_position=cache_position, **kwargs)


@torch.compiler.disable(recursive = False)
@can_return_tuple
def Gemma3nTextModel_forward(
    self,
    input_ids: Optional[torch.LongTensor] = None,
    per_layer_inputs: Optional[torch.Tensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[Cache] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    cache_position: Optional[torch.LongTensor] = None,
    **kwargs: Unpack[TransformersKwargs],
) -> BaseModelOutputWithPast:
    r"""
    per_layer_inputs (torch.Tensor, *optional*, defaults to None):
        Pre-computed per-layer embeddings. If None, they are derived from input_ids if provided.
    """
    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )
    use_cache = use_cache if use_cache is not None else self.config.use_cache

    if (input_ids is None) ^ (inputs_embeds is not None):
        raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

    if self.gradient_checkpointing and self.training and use_cache:
        logger.warning_once(
            "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`."
        )
        use_cache = False

    if input_ids is not None:
        inputs_embeds = self.embed_tokens(input_ids)
        per_layer_inputs = self.get_per_layer_inputs(input_ids)

    per_layer_inputs = self.project_per_layer_inputs(inputs_embeds, per_layer_inputs)

    if use_cache and past_key_values is None and not self.training:
        past_key_values = DynamicCache(config=self.config)

    if cache_position is None:
        past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
        cache_position = torch.arange(
            past_seen_tokens,
            past_seen_tokens + inputs_embeds.shape[1],
            device=inputs_embeds.device,
        )

    if position_ids is None:
        position_ids = cache_position.unsqueeze(0)

    # It may already have been prepared by e.g. `generate`
    if not isinstance(causal_mask_mapping := attention_mask, dict):
        # Prepare mask arguments
        mask_kwargs = {
            "config": self.config,
            "input_embeds": inputs_embeds,
            "attention_mask": attention_mask,
            "cache_position": cache_position,
            "past_key_values": past_key_values,
            "position_ids": position_ids,
        }
        # Create the masks
        causal_mask_mapping = {
            "full_attention": create_causal_mask(**mask_kwargs),
            "sliding_attention": create_sliding_window_causal_mask(**mask_kwargs),
        }

    # embed positions
    hidden_states_0 = inputs_embeds

    # Initialize RoPE embeddings
    position_embeddings_global = self.rotary_emb(hidden_states_0, position_ids)
    position_embeddings_local = self.rotary_emb_local(hidden_states_0, position_ids)

    # Expand hidden_states to support per-layer inputs
    target_magnitude = (torch.mean(((hidden_states_0).to(torch.float32)**2), dim=-1, keepdim=True)**(0.5)).to((hidden_states_0).dtype)
    epsilon_tensor = torch.tensor(1e-5)

    temp_hidden_states = [hidden_states_0]
    for i in range(1, self.config.altup_num_inputs):
        # altup_proj adapted from jax.numpy.einsum("btp,pd->btd", ...)
        altup_proj = self.altup_projections[i - 1](hidden_states_0)
        current_hidden_state = altup_proj.to(dtype=hidden_states_0.dtype, device=target_magnitude.device)
        new_magnitude = torch.mean((current_hidden_state).to(torch.float32)**2, dim=-1, keepdim=True)
        new_magnitude = torch.sqrt(torch.maximum((new_magnitude).to(torch.float32), epsilon_tensor.to(target_magnitude.device)).to((current_hidden_state).dtype))
        current_hidden_state = current_hidden_state * target_magnitude / new_magnitude
        temp_hidden_states.append(current_hidden_state)

    hidden_states = torch.stack(temp_hidden_states, dim=0)  # [num_altup_inputs, batch, seq_len, hidden_size]

    # decoder layers
    all_hidden_states = () if output_hidden_states else None
    all_self_attns = () if output_attentions else None

    for decoder_layer in self.layers[: self.config.num_hidden_layers]:
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        causal_mask = causal_mask_mapping[decoder_layer.attention_type]
        per_layer_input = per_layer_inputs[:, :, decoder_layer.layer_idx, :]

        layer_outputs = decoder_layer(
            hidden_states,
            position_embeddings_global,
            position_embeddings_local,
            per_layer_input,
            attention_mask=causal_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            **kwargs,
        )

        hidden_states = layer_outputs[0]

        if output_attentions:
            all_self_attns += (layer_outputs[1],)

    # add hidden states from the last decoder layer (but before reprojecting to stay consistent with layer output)
    if output_hidden_states:
        all_hidden_states += (hidden_states,)

    # Per-layer inputs to single output
    target_magnitude = (torch.mean(((hidden_states[0]).to(torch.float32)**2), dim=-1, keepdim=True)**(0.5)).to((hidden_states[0]).dtype)
    temp_hidden_states = [hidden_states[0]]
    for i in range(1, self.config.altup_num_inputs):
        # altup_unembed_projections adapted from jax.numpy.einsum("btp,pd->btd", ...)
        altup_unemb_proj: torch.Tensor = self.altup_unembed_projections[i - 1](hidden_states[i])
        current_hidden_state = altup_unemb_proj.to(dtype=hidden_states_0.dtype, device=target_magnitude.device)
        new_magnitude = torch.mean((current_hidden_state).to(torch.float32)**2, dim=-1, keepdim=True)
        new_magnitude = torch.sqrt(torch.maximum((new_magnitude).to(torch.float32), epsilon_tensor.to(target_magnitude.device)).to((current_hidden_state).dtype))
        current_hidden_state = current_hidden_state * target_magnitude / new_magnitude
        temp_hidden_states.append(current_hidden_state)

    hidden_states = torch.stack(temp_hidden_states)
    hidden_states = torch.mean(hidden_states, dim=0)
    hidden_states = self.norm(hidden_states)

    return BaseModelOutputWithPast(
        last_hidden_state=hidden_states,
        past_key_values=past_key_values,
        hidden_states=all_hidden_states,
        attentions=all_self_attns,
    )

class Gemma3nTextModel(Gemma3nPreTrainedModel):
    config: Gemma3nTextConfig

    def __init__(self, config: Gemma3nTextConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        # Gemma3n downcasts the below to bfloat16, causing sqrt(3072)=55.4256 to become 55.5. See https://github.com/huggingface/transformers/pull/29402
        self.embed_tokens = Gemma3nTextScaledWordEmbedding(
            config.vocab_size, config.hidden_size, self.padding_idx, embed_scale=self.config.hidden_size**0.5
        )
        self.layers = nn.ModuleList(
            [Gemma3nTextDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )

        self.norm = Gemma3nRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Gemma3nTextRotaryEmbedding(config=config)
        self.gradient_checkpointing = False

        # TODO (raushan): Fix this after RoPE refactor. For now we hack it by
        # reassigning thetas when we want to create a local RoPE layer. Config
        # defaults should hold values for global RoPE.
        config = copy.deepcopy(config)
        config.rope_theta = config.rope_local_base_freq
        config.rope_scaling = {"rope_type": "default"}
        self.rotary_emb_local = Gemma3nTextRotaryEmbedding(config=config)

        self.hidden_size = config.hidden_size
        self.hidden_size_per_layer_input = config.hidden_size_per_layer_input

        self.embed_tokens_per_layer = Gemma3nTextScaledWordEmbedding(
            config.vocab_size_per_layer_input,
            config.num_hidden_layers * config.hidden_size_per_layer_input,
            self.padding_idx,
            embed_scale=config.hidden_size_per_layer_input**0.5,
        )

        self.per_layer_model_projection = nn.Linear(
            self.hidden_size,
            config.num_hidden_layers * config.hidden_size_per_layer_input,
            bias=False,
        )

        self.per_layer_projection_norm = Gemma3nRMSNorm(config.hidden_size_per_layer_input, eps=config.rms_norm_eps)

        self.altup_projections = nn.ModuleList(
            [nn.Linear(self.hidden_size, self.hidden_size, bias=False) for _ in range(1, self.config.altup_num_inputs)]
        )

        self.altup_unembed_projections = nn.ModuleList(
            [nn.Linear(self.hidden_size, self.hidden_size, bias=False) for _ in range(1, self.config.altup_num_inputs)]
        )

        self.register_buffer("per_layer_projection_scale", torch.tensor(self.hidden_size**-0.5), persistent=False)
        self.register_buffer("per_layer_input_scale", torch.rsqrt(torch.tensor(2.0)), persistent=False)

        # Initialize weights and apply final processing
        self.post_init()


    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        per_layer_inputs: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> BaseModelOutputWithPast:
        return Gemma3nTextModel_forward(self, input_ids=input_ids, per_layer_inputs=per_layer_inputs, attention_mask=attention_mask, position_ids=position_ids, past_key_values=past_key_values, inputs_embeds=inputs_embeds, use_cache=use_cache, output_attentions=output_attentions, output_hidden_states=output_hidden_states, cache_position=cache_position, **kwargs)

    def get_per_layer_inputs(self, input_ids: torch.LongTensor) -> torch.Tensor:
        return self.embed_tokens_per_layer(input_ids).reshape(
            *input_ids.shape,
            self.config.num_hidden_layers,
            self.hidden_size_per_layer_input,
        )

    def project_per_layer_inputs(
        self,
        inputs_embeds: torch.Tensor,
        per_layer_inputs: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        per_layer_projection: torch.Tensor = self.per_layer_model_projection(inputs_embeds)
        per_layer_projection = per_layer_projection * self.per_layer_projection_scale.to(
            dtype=inputs_embeds.dtype, device=per_layer_projection.device
        )
        per_layer_projection = per_layer_projection.reshape(
            *inputs_embeds.shape[:-1],
            self.config.num_hidden_layers,
            self.hidden_size_per_layer_input,
        )
        per_layer_projection = self.per_layer_projection_norm(per_layer_projection)

        if per_layer_inputs is None:
            return per_layer_projection

        if per_layer_projection.shape != per_layer_inputs.shape:
            # per-layer inputs are sometimes padded with zeros, slice the relevant embeddings.
            per_layer_inputs = per_layer_inputs[..., : self.config.num_hidden_layers, :]

        return (per_layer_projection + per_layer_inputs) * self.per_layer_input_scale.to(
            dtype=inputs_embeds.dtype, device=per_layer_projection.device
        )


@torch.compiler.disable(recursive = False)
@can_return_tuple
def Gemma3nForCausalLM_forward(
    self,
    input_ids: Optional[torch.LongTensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[Cache] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    labels: Optional[torch.LongTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    cache_position: Optional[torch.LongTensor] = None,
    logits_to_keep: Union[int, torch.Tensor] = 0,
    **kwargs,
) -> CausalLMOutputWithPast:
    r"""
    Example:

    ```python
    >>> from transformers import AutoTokenizer, Gemma3nForCausalLM

    >>> model = Gemma3nForCausalLM.from_pretrained("google/gemma-2-9b")
    >>> tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-9b")

    >>> prompt = "What is your favorite condiment?"
    >>> inputs = tokenizer(prompt, return_tensors="pt")

    >>> # Generate
    >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
    >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    "What is your favorite condiment?"
    ```"""

    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )
    # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
    outputs: BaseModelOutputWithPast = self.model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        use_cache=use_cache,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        cache_position=cache_position,
        **kwargs,
    )

    hidden_states = outputs.last_hidden_state
    # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
    slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
    logits = self.lm_head(hidden_states[:, slice_indices, :]) if os.environ.get('UNSLOTH_RETURN_LOGITS', '0') == '1' else EMPTY_LOGITS
    loss = None
    NOT_RETURN_LOGITS = os.environ.get('UNSLOTH_RETURN_LOGITS', '0') == '0'
    RETURN_HIDDEN_STATES = os.environ.get("UNSLOTH_RETURN_HIDDEN_STATES", "0") == "1"
    
    n_items = None
    if (kwargs) != () and type(kwargs) is dict:
        n_items = (kwargs).get("num_items_in_batch", None)
        if n_items is None: n_items = (kwargs).get("n_items", None)
    if n_items is None:
        all_locals = locals()
        if 'loss_kwargs' in all_locals:
            __kwargs = all_locals['loss_kwargs']
            if type(__kwargs) is dict:
                n_items = __kwargs.get("num_items_in_batch", None)
                if n_items is None: n_items = __kwargs.get("n_items", None)
        if n_items is None and 'kwargs' in all_locals:
            __kwargs = all_locals['kwargs']
            if type(__kwargs) is dict:
                n_items = __kwargs.get("num_items_in_batch", None)
                if n_items is None: n_items = __kwargs.get("n_items", None)
        if n_items is None:
            all_locals = all_locals.values()
            for __kwargs in all_locals:
                if type(__kwargs) is dict:
                    n_items = __kwargs.get("num_items_in_batch", None)
                    if n_items is None: n_items = __kwargs.get("n_items", None)
                    break
    pass
    
    requires_grad_ = self.lm_head.weight.requires_grad
    requires_grad_ = requires_grad_ or self.lm_head.weight.dtype == torch.float32
    
    if RETURN_HIDDEN_STATES:
        logits = hidden_states[:, slice_indices, :]
    elif labels is None:
        
    
        # Set compiler stance to fail on recompiles for inference
        global INFERENCE_RUNS
        if torch_dynamo_eval_frame is not None:
            old_stance = torch_dynamo_eval_frame._stance.stance
        else:
            old_stance = None
        if old_stance is not None and INFERENCE_RUNS == 1:
            # Skip guards and return to eager -> we still need guards!
            torch_compiler_set_stance(stance = "eager_on_recompile", skip_guard_eval_unsafe = False)
            if UNSLOTH_ENABLE_LOGGING:
                logger_compiler.info(
                    f"Unsloth: Removing compiler guards after 1 inference run. " \
                    f"DYNAMO_STANCE.stance = {torch_dynamo_eval_frame._stance.stance} " \
                    f"DYNAMO_STANCE.skip_guard_eval_unsafe = {torch_dynamo_eval_frame._stance.skip_guard_eval_unsafe}"
                )
        elif old_stance == "eager_on_recompile":
            pass
        elif old_stance == "default" and INFERENCE_RUNS > 1:
            # Reset compiler stance
            torch_compiler_set_stance(stance = "default", skip_guard_eval_unsafe = False)
            if UNSLOTH_ENABLE_LOGGING:
                logger_compiler.info(
                    f"Unsloth: Reseting guards. " \
                    f"DYNAMO_STANCE.stance = {torch_dynamo_eval_frame._stance.stance} " \
                    f"DYNAMO_STANCE.skip_guard_eval_unsafe = {torch_dynamo_eval_frame._stance.skip_guard_eval_unsafe}"
                )
            INFERENCE_RUNS = 0
        INFERENCE_RUNS += 1
    
        logits = self.lm_head(hidden_states[:, slice_indices, :])
    elif (() == () and () == ()) and (UNSLOTH_ENABLE_CCE) and NOT_RETURN_LOGITS and self.loss_function.__name__.endswith("ForCausalLMLoss") and labels is not None and not requires_grad_:
        loss = fused_linear_cross_entropy(
            hidden_states      = hidden_states[:, slice_indices, :],
            lm_weight          = self.lm_head.weight,
            labels             = labels.to(self.lm_head.weight.device),
            num_items_in_batch = n_items,
            logit_softcapping  = None if (self.config.final_logit_softcapping) == () else (self.config.final_logit_softcapping),
        )
    elif self.loss_function.__name__.endswith("ForCausalLMLoss") and labels is not None:
        lm_head_weight = self.lm_head.weight
        lm_head_bias   = getattr(self.lm_head, "bias", None)
    
        # ========= NEW fused =========
        _hidden_states = hidden_states[:, slice_indices, :]
        torch._dynamo.mark_dynamic(_hidden_states, 1)
        torch._dynamo.mark_dynamic(labels, 1)
        loss = unsloth_fused_ce_loss(
            trainer              = None,
            hidden_states        = _hidden_states,
            lm_head_weight       = lm_head_weight,
            lm_head_bias         = lm_head_bias,
            labels               = labels,
            mask                 = None,
            n_items              = n_items,
            scaling              = getattr(self, "accelerator_scaler", None),
            target_gb            = None,
            torch_compile        = not UNSLOTH_COMPILE_DISABLE,
            logit_scale_multiply = () if () != () else 0,
            logit_scale_divide   = () if () != () else 0,
            logit_softcapping    = (self.config.final_logit_softcapping) if (self.config.final_logit_softcapping) != () else 0,
        )
    else:
        logits = self.lm_head(hidden_states[:, slice_indices, :])
        if () != ():
            logits = logits * ()
        if () != ():
            logits = logits / ()
        if (self.config.final_logit_softcapping) not in (None, (),):
            logits = logits / (self.config.final_logit_softcapping)
            logits = torch.tanh(logits)
            logits = logits * (self.config.final_logit_softcapping)
        loss = self.loss_function(logits, labels.to(self.lm_head.weight.device), vocab_size=self.vocab_size, **kwargs)


    return CausalLMOutputWithPast(
        loss=loss,
        logits=logits,
        past_key_values=outputs.past_key_values,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
    )

class Gemma3nForCausalLM(Gemma3nPreTrainedModel, GenerationMixin):
    _tied_weights_keys = ["lm_head.weight"]
    _tp_plan = {"lm_head": "colwise_rep"}
    _pp_plan = {"lm_head": (["hidden_states"], ["logits"])}
    config: Gemma3nTextConfig
    base_model_prefix = "model"
    _checkpoint_conversion_mapping = {"model.language_model": "model"}

    def __init__(self, config: Gemma3nTextConfig):
        super().__init__(config)
        self.model = Gemma3nTextModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()


    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        return Gemma3nForCausalLM_forward(self, input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids, past_key_values=past_key_values, inputs_embeds=inputs_embeds, labels=labels, use_cache=use_cache, output_attentions=output_attentions, output_hidden_states=output_hidden_states, cache_position=cache_position, logits_to_keep=logits_to_keep, **kwargs)


@torch.compiler.disable(recursive = False)
def Gemma3nMultimodalEmbedder_forward(
    self,
    input_ids: Optional[torch.LongTensor] = None,
    inputs_embeds: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Embeds token ids or soft tokens for multimodal content into language model space.

    Args:
        input_ids: A torch.LongTensor containing the token ids to embed. Values should be in the range
            `[vocab_offset, vocab_offset + vocab_size)`.
        inputs_embeds: A torch.Tensor containing the soft tokens to embed.

    Returns:
        A torch.Tensor of embeddings with  shape `[batch_size, seq_len, self.config.text_config.hidden_size]`.
    """
    if (input_ids is None) ^ (inputs_embeds is not None):
        raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

    if inputs_embeds is not None:
        emb_norm = self.soft_embedding_norm(inputs_embeds)
    else:
        hard_emb = self.embedding((input_ids - self.vocab_offset).clamp_(0))
        emb_norm = self.hard_embedding_norm(hard_emb)

    emb_norm_proj = self.embedding_projection(emb_norm)
    return self.embedding_post_projection_norm(emb_norm_proj)

class Gemma3nMultimodalEmbedder(nn.Module):
    """Embeds token ids or soft tokens for multimodal content into language model space."""

    def __init__(
        self,
        multimodal_config: Union[Gemma3nAudioConfig, Gemma3nVisionConfig],
        text_config: Gemma3nTextConfig,
    ):
        super().__init__()

        self.multimodal_hidden_size = multimodal_config.hidden_size
        self.eps = multimodal_config.rms_norm_eps
        self.vocab_offset = multimodal_config.vocab_offset
        self.vocab_size = multimodal_config.vocab_size
        self.text_hidden_size = text_config.hidden_size

        self.embedding = nn.Embedding(self.vocab_size, self.multimodal_hidden_size)
        self.hard_embedding_norm = Gemma3nRMSNorm(self.multimodal_hidden_size, eps=self.eps)
        self.soft_embedding_norm = Gemma3nRMSNorm(self.multimodal_hidden_size, eps=self.eps)
        self.embedding_projection = nn.Linear(self.multimodal_hidden_size, self.text_hidden_size, bias=False)
        self.embedding_post_projection_norm = Gemma3nRMSNorm(self.text_hidden_size, eps=self.eps, with_scale=False)

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return Gemma3nMultimodalEmbedder_forward(self, input_ids=input_ids, inputs_embeds=inputs_embeds)


@torch.compiler.disable(recursive = False)
@can_return_tuple
def Gemma3nForConditionalGeneration_forward(
    self,
    input_ids: Optional[torch.LongTensor] = None,  # text inputs
    pixel_values: Optional[torch.FloatTensor] = None,  # vision inputs
    input_features: Optional[torch.FloatTensor] = None,  # audio inputs
    attention_mask: Optional[torch.Tensor] = None,
    input_features_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[Cache] = None,
    token_type_ids: Optional[torch.LongTensor] = None,
    cache_position: Optional[torch.LongTensor] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    labels: Optional[torch.LongTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    logits_to_keep: Union[int, torch.Tensor] = 0,
    **lm_kwargs,
) -> Gemma3nCausalLMOutputWithPast:
    r"""
    input_features_mask (torch.Tensor, *optional*, defaults to None):
        The attention mask for the input audio.
    labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
        Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
        config.text_config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are
        ignored (masked), the loss is only computed for the tokens with labels in
        `[0, ..., config.text_config.vocab_size]`.

    Example:

    ```python
    >>> from PIL import Image
    >>> import requests
    >>> from transformers import AutoProcessor, Gemma3ForConditionalGeneration

    >>> model = Gemma3ForConditionalGeneration.from_pretrained("google/gemma-3-4b-it")
    >>> processor = AutoProcessor.from_pretrained("google/gemma-3-4b-it")

    >>> messages = [
    ...     {
    ...         "role": "system",
    ...         "content": [
    ...             {"type": "text", "text": "You are a helpful assistant."}
    ...         ]
    ...     },
    ...     {
    ...         "role": "user", "content": [
    ...             {"type": "image", "url": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg"},
    ...             {"type": "text", "text": "Where is the cat standing?"},
    ...         ]
    ...     },
    ... ]

    >>> inputs = processor.apply_chat_template(
    ...     messages,
    ...     tokenizer=True,
    ...     return_dict=True,
    ...     return_tensors="pt",
    ...     add_generation_prompt=True
    ... )
    >>> # Generate
    >>> generate_ids = model.generate(**inputs)
    >>> processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    "user\nYou are a helpful assistant.\n\n\n\n\n\nWhere is the cat standing?\nmodel\nBased on the image, the cat is standing in a snowy area, likely outdoors. It appears to"
    ```
    """
    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )

    outputs = self.model(
        input_ids=input_ids,
        pixel_values=pixel_values,
        input_features=input_features,
        attention_mask=attention_mask,
        input_features_mask=input_features_mask,
        position_ids=position_ids,
        past_key_values=past_key_values,
        token_type_ids=token_type_ids,
        cache_position=cache_position,
        inputs_embeds=inputs_embeds,
        labels=mask_attention_mask_out(labels = labels, attention_mask = attention_mask),
        use_cache=use_cache,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=True,
        **lm_kwargs,
    )

    hidden_states = outputs.last_hidden_state
    # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
    slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
    logits = self.lm_head(hidden_states[:, slice_indices, :]) if os.environ.get('UNSLOTH_RETURN_LOGITS', '0') == '1' else EMPTY_LOGITS
    loss = None
    NOT_RETURN_LOGITS = os.environ.get('UNSLOTH_RETURN_LOGITS', '0') == '0'
    RETURN_HIDDEN_STATES = os.environ.get("UNSLOTH_RETURN_HIDDEN_STATES", "0") == "1"
    
    all_locals = locals()
    n_items = None
    if 'loss_kwargs' in all_locals:
        __kwargs = all_locals['loss_kwargs']
        if type(__kwargs) is dict:
            n_items = __kwargs.get("num_items_in_batch", None)
            if n_items is None: n_items = __kwargs.get("n_items", None)
    if n_items is None and 'kwargs' in all_locals:
        __kwargs = all_locals['kwargs']
        if type(__kwargs) is dict:
            n_items = __kwargs.get("num_items_in_batch", None)
            if n_items is None: n_items = __kwargs.get("n_items", None)
    if n_items is None:
        all_locals = all_locals.values()
        for __kwargs in all_locals:
            if type(__kwargs) is dict:
                n_items = __kwargs.get("num_items_in_batch", None)
                if n_items is None: n_items = __kwargs.get("n_items", None)
                break
    pass
    
    requires_grad_ = self.lm_head.weight.requires_grad
    requires_grad_ = requires_grad_ or self.lm_head.weight.dtype == torch.float32
    
    if RETURN_HIDDEN_STATES:
        logits = hidden_states[:, slice_indices, :]
    elif labels is None:
        
    
        # Set compiler stance to fail on recompiles for inference
        global INFERENCE_RUNS
        if torch_dynamo_eval_frame is not None:
            old_stance = torch_dynamo_eval_frame._stance.stance
        else:
            old_stance = None
        if old_stance is not None and INFERENCE_RUNS == 1:
            # Skip guards and return to eager -> we still need guards!
            torch_compiler_set_stance(stance = "eager_on_recompile", skip_guard_eval_unsafe = False)
            if UNSLOTH_ENABLE_LOGGING:
                logger_compiler.info(
                    f"Unsloth: Removing compiler guards after 1 inference run. " \
                    f"DYNAMO_STANCE.stance = {torch_dynamo_eval_frame._stance.stance} " \
                    f"DYNAMO_STANCE.skip_guard_eval_unsafe = {torch_dynamo_eval_frame._stance.skip_guard_eval_unsafe}"
                )
        elif old_stance == "eager_on_recompile":
            pass
        elif old_stance == "default" and INFERENCE_RUNS > 1:
            # Reset compiler stance
            torch_compiler_set_stance(stance = "default", skip_guard_eval_unsafe = False)
            if UNSLOTH_ENABLE_LOGGING:
                logger_compiler.info(
                    f"Unsloth: Reseting guards. " \
                    f"DYNAMO_STANCE.stance = {torch_dynamo_eval_frame._stance.stance} " \
                    f"DYNAMO_STANCE.skip_guard_eval_unsafe = {torch_dynamo_eval_frame._stance.skip_guard_eval_unsafe}"
                )
            INFERENCE_RUNS = 0
        INFERENCE_RUNS += 1
    
        logits = self.lm_head(hidden_states[:, slice_indices, :])
    else:
        lm_head_weight = self.lm_head.weight
        lm_head_bias   = getattr(self.lm_head, "bias", None)
    
        # ========= NEW fused =========
        _hidden_states = hidden_states[:, slice_indices, :]
        torch._dynamo.mark_dynamic(_hidden_states, 1)
        torch._dynamo.mark_dynamic(labels, 1)
        if attention_mask is not None:
            torch._dynamo.mark_dynamic(attention_mask, 1)
        loss = unsloth_fused_ce_loss(
            trainer              = None,
            hidden_states        = _hidden_states,
            lm_head_weight       = lm_head_weight,
            lm_head_bias         = lm_head_bias,
            labels               = labels,
            mask                 = attention_mask,
            n_items              = n_items,
            scaling              = getattr(self, "accelerator_scaler", None),
            target_gb            = None,
            torch_compile        = not UNSLOTH_COMPILE_DISABLE,
            logit_scale_multiply = () if () != () else 0,
            logit_scale_divide   = () if () != () else 0,
            logit_softcapping    = (self.config.get_text_config().final_logit_softcapping) if (self.config.get_text_config().final_logit_softcapping) != () else 0,
        )


    return Gemma3nCausalLMOutputWithPast(
        loss=loss,
        logits=logits,
        past_key_values=outputs.past_key_values,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
        image_hidden_states=outputs.image_hidden_states,
        audio_hidden_states=outputs.audio_hidden_states,
    )

class Gemma3nForConditionalGeneration(Gemma3nPreTrainedModel, GenerationMixin):
    _checkpoint_conversion_mapping = {}
    _tied_weights_keys = ["lm_head.weight"]
    base_model_prefix = "model"

    def __init__(self, config: Gemma3nConfig):
        super().__init__(config)
        self.model = Gemma3nModel(config)
        self.lm_head = nn.Linear(config.text_config.hidden_size, config.text_config.vocab_size, bias=False)
        self.post_init()

    def get_input_embeddings(self):
        return self.model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.model.set_input_embeddings(value)

    def set_decoder(self, decoder):
        self.model.set_decoder(decoder)

    def get_decoder(self):
        return self.model.get_decoder()

    def get_image_features(self, pixel_values):
        return self.model.get_image_features(pixel_values)

    # Make modules available through conditional class for BC
    @property
    def language_model(self):
        return self.model.language_model

    @property
    def vision_tower(self):
        return self.model.vision_tower

    @property
    def multi_modal_projector(self):
        raise AttributeError("Use embed_vision instead of multi_modal_projector.")


    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,  # text inputs
        pixel_values: Optional[torch.FloatTensor] = None,  # vision inputs
        input_features: Optional[torch.FloatTensor] = None,  # audio inputs
        attention_mask: Optional[torch.Tensor] = None,
        input_features_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **lm_kwargs,
    ) -> Gemma3nCausalLMOutputWithPast:
        return Gemma3nForConditionalGeneration_forward(self, input_ids=input_ids, pixel_values=pixel_values, input_features=input_features, attention_mask=attention_mask, input_features_mask=input_features_mask, position_ids=position_ids, past_key_values=past_key_values, token_type_ids=token_type_ids, cache_position=cache_position, inputs_embeds=inputs_embeds, labels=labels, use_cache=use_cache, output_attentions=output_attentions, output_hidden_states=output_hidden_states, logits_to_keep=logits_to_keep, **lm_kwargs)

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        inputs_embeds=None,
        cache_position=None,
        position_ids=None,
        pixel_values=None,
        input_features=None,
        attention_mask=None,
        input_features_mask=None,
        token_type_ids=None,
        use_cache=True,
        logits_to_keep=None,
        labels=None,
        **kwargs,
    ):
        # Overwritten -- custom `position_ids` and `pixel_values` handling
        model_inputs = super().prepare_inputs_for_generation(
            input_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            cache_position=cache_position,
            use_cache=use_cache,
            logits_to_keep=logits_to_keep,
            token_type_ids=token_type_ids,
            **kwargs,
        )

        # If we're in cached decoding stage, multimodal inputs should be None because input ids do not contain special
        # tokens anymore. Otherwise multimodal inputs should be passed to model.
        # NOTE: use_cache=False always needs pixel_values, input_features, and input_features_mask
        if cache_position[0] == 0:
            model_inputs["pixel_values"] = pixel_values
            model_inputs["input_features"] = input_features
            model_inputs["input_features_mask"] = input_features_mask

        return model_inputs

    @property
    def audio_tower(self):
        return self.model.audio_tower


if hasattr(logger, "addFilter"):
    import logging
    class HideLoggingMessage(logging.Filter):
        def __init__(self, text): self.text = text
        def filter(self, x): return not (self.text in x.getMessage())
    pass
    logger.addFilter(HideLoggingMessage("`use_cache=True`"))

