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

from torch import Tensor
import torch
import torch.nn as nn
from torch.nn import functional as F
from unsloth_zoo.temporary_patches.common import torch_compile
from typing import Any, List, Optional, Tuple, Union, Dict, Set, Callable
from trl.trainer.online_dpo_trainer import (Any, AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer, BasePairwiseJudge, BaseTrainer, Callable, DPODataCollatorWithPadding, DataCollator, DataLoader, Dataset, EvalPrediction, F, FSDP, GenerationConfig, IterableDataset, MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING_NAMES, OnlineDPOConfig, OnlineDPOTrainer, OptimizerNames, Optional, Path, PeftConfig, PreTrainedModel, PreTrainedTokenizerBase, ProcessorMixin, RewardFunc, SIMPLE_CHAT_TEMPLATE, Trainer, TrainerCallback, Union, VLLMClient, apply_chat_template, broadcast_object_list, create_reference_model, disable_dropout_in_model, empty_cache, ensure_master_addr_port, gather_object, is_conversational, is_flash_attn_2_available, is_peft_model, is_vllm_available, jinja2, logger, logging, maybe_apply_chat_template, nn, nullcontext, os, pad, prepare_deepspeed, prepare_fsdp, profiling_context, re, seed_worker, textwrap, torch, truncate_right, unwrap_model_for_generation, version, warnings, wraps, AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer, BasePairwiseJudge, Callable, DPODataCollatorWithPadding, DataCollator, Dataset, EvalPrediction, F, GenerationConfig, IterableDataset, MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING_NAMES, OnlineDPOConfig, Optional, PeftConfig, PreTrainedModel, PreTrainedTokenizerBase, ProcessorMixin, RewardFunc, Trainer, TrainerCallback, Union, VLLMClient, create_reference_model, disable_dropout_in_model, ensure_master_addr_port, is_vllm_available, logger, nn, os, pad, prepare_deepspeed, prepare_fsdp, re, torch, version, warnings, F, apply_chat_template, is_conversational, re, F, FSDP, is_peft_model, nn, nullcontext, os, re, version, F, Optional, PreTrainedModel, Trainer, logger, os, re, torch, F, FSDP, nn, os, re, F, FSDP, nn, re, torch)


import os
import math
import logging
from typing import *
from dataclasses import dataclass, field
from packaging.version import Version
import torch
import numpy as np
from contextlib import nullcontext
from torch.nn import functional as F
import inspect
from transformers import DataCollatorForSeq2Seq, DataCollatorForLanguageModeling as TransformersDataCollatorForLanguageModeling
from transformers.training_args import ParallelMode
from unsloth_zoo.device_type import DEVICE_TYPE, device_synchronize

# Wrap trainer with padding to right and enable training mode
import functools
from types import MethodType
try:
    from unsloth_zoo.gradient_checkpointing import reset_unsloth_gradient_checkpointing_buffers
except:
    def reset_unsloth_gradient_checkpointing_buffers(): pass
def prepare_for_training_mode(f):
    @functools.wraps(f)
    def wrapper(self, *args, **kwargs):
        # Finish the previous W&B run if this is a subsequent train() call.
        # We do this at the START of train() (not the end) so that
        # evaluate() / log() still work after train() completes.
        # HF's WandbCallback.setup() will call wandb.init() for the new run.
        # See: https://github.com/unslothai/unsloth/issues/3954
        if getattr(self, '_unsloth_training_completed', False):
            try:
                import wandb
                if wandb.run is not None:
                    wandb.finish()
                    # Reset HF's WandbCallback so it calls wandb.init() for the new run
                    for cb in self.callback_handler.callbacks:
                        if type(cb).__name__ == 'WandbCallback':
                            cb._initialized = False
                            break
            except:
                pass
        # Enable training mode
        _was_training = None
        # Get gradient checkpointing setting from training arguments
        use_gc = getattr(self.args, 'gradient_checkpointing', True)
        if hasattr(self, 'model') and hasattr(self.model, "training"):
            _was_training = self.model.training
        if hasattr(self, 'model') and hasattr(self.model, "for_training"):
            self.model.for_training(use_gradient_checkpointing=use_gc)
        output = f(self, *args, **kwargs)
        # Restore previous mode when possible
        if hasattr(self, 'model') and hasattr(self.model, "for_inference"):
            if _was_training is False:
                self.model.for_inference()
            elif _was_training is True and hasattr(self.model, "for_training"):
                self.model.for_training(use_gradient_checkpointing=use_gc)
        # Reset gradient checkpointing buffers to free memory while staying ready for next run
        try:
            reset_unsloth_gradient_checkpointing_buffers()
        except:
            pass
        # Mark that training completed so the next train() call can
        # finish this W&B run before starting a new one
        self._unsloth_training_completed = True
        return output
    return wrapper
pass

torch_compile_options = {
    "epilogue_fusion"   : True,
    "max_autotune"      : False,
    "shape_padding"     : True,
    "trace.enabled"     : False,
    "triton.cudagraphs" : False,
}

@torch.compile(dynamic = True, fullgraph = True, options = torch_compile_options,)
def chunked_hidden_states_selective_log_softmax(
    hidden_states: torch.Tensor,
    lm_head: torch.Tensor,
    index: torch.Tensor,
    chunks: int = 4,
    logit_scale_multiply: float = 0.0,
    logit_scale_divide: float = 0.0,
    logit_softcapping: float = 0.0,
    temperature: float = 1.0,
) -> torch.Tensor:
    # All Unsloth Zoo code licensed under AGPL3
    flat_hidden_states = hidden_states.reshape(-1, hidden_states.shape[-1])
    flat_index = index.reshape(-1)

    chunked_hidden_states = torch.chunk(flat_hidden_states, chunks=chunks, dim=0)
    chunked_index = torch.chunk(flat_index, chunks=chunks, dim=0)

    all_per_token_logps = []

    for chunk_hidden_states, chunk_index in zip(chunked_hidden_states, chunked_index):
        chunk_logits = chunk_hidden_states.to(lm_head.dtype) @ lm_head.t()

        if logit_scale_multiply != 0.0:
            chunk_logits = chunk_logits * logit_scale_multiply
        if logit_scale_divide != 0.0:
            chunk_logits = chunk_logits / logit_scale_divide
        if logit_softcapping != 0.0:
            chunk_logits = logit_softcapping * torch.tanh(chunk_logits / logit_softcapping)

        chunk_logits = chunk_logits.to(torch.float32)

        if temperature != 1.0:
            chunk_logits = chunk_logits / temperature

        selected_logits = torch.gather(chunk_logits, dim=-1, index=chunk_index.unsqueeze(-1)).squeeze(-1)
        logsumexp_values = torch.logsumexp(chunk_logits, dim=-1)
        per_token_logps = selected_logits - logsumexp_values
        all_per_token_logps.append(per_token_logps)

    all_per_token_logps = torch.concat(all_per_token_logps)

    all_per_token_logps = all_per_token_logps.reshape((hidden_states.shape[0], hidden_states.shape[1]))
    return all_per_token_logps

@torch.compile(dynamic = True, fullgraph = True, options = torch_compile_options,)
def chunked_selective_log_softmax(logits, index, temperature: float = 1.0):
    # Split into 4 chunks only
    chunked_logits = torch.chunk(logits.reshape(-1, logits.shape[-1]), chunks = 4, dim = 0)
    chunked_index  = torch.chunk(index.reshape(-1), chunks = 4, dim = 0)
    all_per_token_logps = []
    # Below loop does the same as selective_log_softmax(chunk_logits, chunk_index)
    for chunk_logits, chunk_index in zip(chunked_logits, chunked_index):
        chunk_logits = chunk_logits.to(torch.float32)
        if temperature != 1.0:
            chunk_logits = chunk_logits / temperature
        selected_logits = torch.gather(chunk_logits, dim = -1, index = chunk_index.unsqueeze(-1)).squeeze(-1)
        logsumexp_values = torch.logsumexp(chunk_logits, dim = -1)
        per_token_logps = selected_logits - logsumexp_values
        all_per_token_logps.append(per_token_logps)
    pass
    all_per_token_logps = torch.concat(all_per_token_logps)
    all_per_token_logps = all_per_token_logps.reshape((logits.shape[0], logits.shape[1]))
    return all_per_token_logps

def calculate_pad_tokens_in_prompt(
    input_ids: torch.Tensor,
    logits_to_keep: int,
    pad_token_id: int
) -> torch.Tensor:
    """
    Given prompt tensor, it returns all the left padded tokens in that sequence. so [pad, pad, pad, cat] = 3 tokens
    """
    if logits_to_keep >= input_ids.shape[1]:
        raise ValueError("logits_to_keep must be smaller than the sequence length.")

    prompt_section = input_ids[:, :-logits_to_keep]

    padding_mask = (prompt_section == pad_token_id)

    pad_token_counts = padding_mask.sum(dim=1)

    return pad_token_counts

def create_completion_attention_mask(
    completion_input_ids: torch.Tensor,
    left_pad_tokens_per_prompt: torch.Tensor,
    max_left_pad: int,
    pad_token_id: int
) -> torch.Tensor:
    """
    Given that we have a sequence, [p,p,p,c,c,c,pad,pad,pad]

    Where p are extra prompt tokens we got from slicing the torch tensor, c is completion tokens
    and pad are pad tokens, this function would make a completion mask that would 0 out the pad
    and p tokens. so in this example [0,0,0,1,1,1,0,0,0]
    """
    batch_size, completion_len = completion_input_ids.shape
    device = completion_input_ids.device

    num_tokens_to_mask = max_left_pad - left_pad_tokens_per_prompt

    indices = torch.arange(completion_len, device=device).unsqueeze(0)
    shift_mask = indices >= num_tokens_to_mask.unsqueeze(1)

    non_padding_mask = (completion_input_ids != pad_token_id)

    final_mask = shift_mask & non_padding_mask

    return final_mask

def left_pack_padding(tensor: torch.Tensor, pad_id: int) -> torch.Tensor:
    """
    Moves all padding tokens in each sequence of a batch to the right.
    """
    mask = (tensor != pad_id)
    # Must do stable=True since binary mark is unordered
    sorted_indices = torch.argsort(mask, dim=1, descending=True, stable=True)
    packed_tensor = torch.gather(tensor, 1, sorted_indices)
    return packed_tensor

def align_logprobs_with_mask(
    logprob_tensor: torch.Tensor,
    attention_mask: torch.Tensor,
    pad_value: float = 0.0
) -> torch.Tensor:
    """
    Aligns a log probability tensor with a given attention mask.
    """

    device = logprob_tensor.device
    batch_size, logprob_seq_len = logprob_tensor.shape
    mask_seq_len = attention_mask.shape[1]

    padded_logprobs = torch.full(
        attention_mask.shape,
        fill_value=pad_value,
        dtype=logprob_tensor.dtype,
        device=device
    )

    left_pad_counts = torch.argmax(attention_mask, dim=1)

    cols = torch.arange(logprob_seq_len, device=device)
    dest_indices = left_pad_counts.unsqueeze(1) + cols

    # Create destination row indices
    # Shape: [batch_size, logprob_seq_len]
    row_indices = torch.arange(batch_size, device=device).unsqueeze(1).expand_as(dest_indices)

    # --- 4. Filter out-of-bounds indices and perform assignment ---
    # Create a mask to identify only the indices that are within the bounds
    # of the target tensor's sequence length.
    valid_mask = dest_indices < mask_seq_len

    # Use this mask to select only the valid row indices, column indices,
    # and the corresponding values from the logprob tensor.
    # This flattens the selected elements into 1D tensors.
    valid_rows = row_indices[valid_mask]
    valid_cols = dest_indices[valid_mask]
    valid_vals = logprob_tensor[valid_mask]

    # Place the valid values into their correct positions in the padded tensor
    # using a single, efficient advanced indexing operation.
    padded_logprobs[valid_rows, valid_cols] = valid_vals

    return padded_logprobs

def autotune_batch_and_chunks(
    total_input_rows,
    seq_len,
    hidden_size,
    vocab_size,
    dtype_bytes=16,
    multiplier=None
):
    if multiplier is None:
        final_m = max(4, seq_len // 4096)
    else:
        final_m = multiplier

    if torch.cuda.is_available():
        free_bytes, _ = torch.cuda.mem_get_info()
        limit_gb = (free_bytes / (1024**3))*.80
    elif hasattr(torch, "xpu") and torch.xpu.is_available():
        # For XPU: estimate free memory from total - reserved
        total_mem = torch.xpu.get_device_properties(0).total_memory
        reserved_mem = torch.xpu.memory_reserved()
        free_bytes = total_mem - reserved_mem
        limit_gb = (free_bytes / (1024**3)) * 0.80
    else:
        # Fallback: assume 8GB available
        limit_gb = 8.0

    bytes_to_gb = 1024**3

    b_vals = torch.arange(total_input_rows, 0, -1, device='cpu', dtype=torch.float32)

    hidden_gb = (b_vals * seq_len * hidden_size * dtype_bytes) / bytes_to_gb

    base_logits = ((b_vals/total_input_rows) * b_vals * seq_len * vocab_size * dtype_bytes) / bytes_to_gb
    logits_gb = base_logits / final_m

    total_mem_gb = hidden_gb + logits_gb

    valid_mask = total_mem_gb <= limit_gb
    valid_indices = torch.nonzero(valid_mask, as_tuple=False)

    if valid_indices.shape[0] == 0:
        #This means your GPU will OOM
        return 4, final_m

    best_idx = valid_indices[0].item()
    final_b = int(b_vals[best_idx].item())

    return final_b, final_m

def sanitize_logprob(logprob):
    """Local port of trl.scripts.vllm_serve.sanitize_logprob.
    Filters NaN logprobs from vLLM outputs."""
    value = logprob.logprob
    if math.isnan(value):
        logging.getLogger(__name__).warning(
            f"Generated NaN logprob, token logprob '{logprob}' will be ignored"
        )
        return None
    return value
def vLLMSamplingParams(**kwargs):
    from vllm import SamplingParams

    sampling_params = SamplingParams(**kwargs)
    sampling_params._set_kwargs = kwargs
    return sampling_params
@dataclass
class UnslothOnlineDPOConfig(OnlineDPOConfig):
    """
    
    Configuration class for the [`OnlineDPOTrainer`].

    This class includes only the parameters that are specific to Online DPO training. For a full list of training
    arguments, please refer to the [`~transformers.TrainingArguments`] documentation. Note that default values in this
    class may differ from those in [`~transformers.TrainingArguments`].

    Using [`~transformers.HfArgumentParser`] we can turn this class into
    [argparse](https://docs.python.org/3/library/argparse#module-argparse) arguments that can be specified on the
    command line.

    Parameters:
        reward_model_path (`str`, *optional*):
            Path to the reward model. Either `judge` or `reward_model_path` must be set, but not both.
        judge (`str`, *optional*):
            Name of the judge to use. Either `judge` or `reward_model_path` must be set, but not both.
        max_new_tokens (`int`, *optional*, defaults to `64`):
            Maximum number of tokens to generate per completion.
        max_length (`int`, *optional*, defaults to `256`):
            Maximum total length of the sequence (prompt + completion) used to compute log probabilities. If the
            sequence exceeds this limit, the leftmost tokens will be truncated to preserve as much of the completion as
            possible.
        temperature (`float`, *optional*, defaults to `0.9`):
            Temperature for sampling. The higher the temperature, the more random the completions.
        missing_eos_penalty (`float`, *optional*):
            Penalty applied to the score when the model fails to generate an EOS token. This is useful to encourage to
            generate completions shorter than the maximum length (`max_new_tokens`). The penalty must be a positive
            value. This parameter only works when using `reward_funcs` and not when using `judge`.
        beta (`float` or `list[float]`, *optional*, defaults to `0.1`):
            Parameter controlling the deviation from the reference model. Higher β means less deviation from the
            reference model. For the IPO loss (`loss_type="ipo"`), β is the regularization parameter denoted by τ in
            the [paper](https://huggingface.co/papers/2310.12036). If a list of floats is provided then the β is
            selected for each new epoch and the last β is used for the rest of the epochs.
        loss_type (`str`, *optional*, defaults to `"sigmoid"`):
            Type of loss to use. Possible values are:

                - `"sigmoid"`: sigmoid loss from the original [DPO](https://huggingface.co/papers/2305.18290) paper.
                - `"ipo"`: IPO loss from the [IPO](https://huggingface.co/papers/2310.12036) paper.

        dataset_num_proc (`int`, *optional*):
            Number of processes to use for processing the dataset.

            <Deprecated version="0.22.0">

            This parameter is deprecated and will be removed in version 0.25.0. Since OnlineDPO does not involve
            dataset preparation, you can safely remove it.

            </Deprecated>

        disable_dropout (`bool`, *optional*, defaults to `True`):
            Whether to disable dropout in the model and reference model.

        > Parameters that control generation

        top_p (`float`, *optional*, defaults to `1.0`):
            Float that controls the cumulative probability of the top tokens to consider. Must be in (0, 1]. Set to
            `1.0` to consider all tokens.
        top_k (`int`, *optional*):
            Number of highest probability vocabulary tokens to keep for top-k-filtering. If `None`, top-k-filtering is
            disabled and all tokens are considered.
        min_p (`float`, *optional*):
            Minimum token probability, which will be scaled by the probability of the most likely token. It must be a
            value between `0.0` and `1.0`. Typical values are in the `0.01-0.2` range.
        repetition_penalty (`float`, *optional*, defaults to `1.0`):
            Float that penalizes new tokens based on whether they appear in the prompt and the generated text so far.
            Values > `1.0` encourage the model to use new tokens, while values < `1.0` encourage the model to repeat
            tokens.
        use_transformers_paged (`bool`, *optional*, defaults to `False`):
            Whether to use the `transformers` paged implementation for generation. If set to `True`, the `transformers`
            paged implementation will be used for generation instead of the default padded implementation. This
            parameter is only effective when `use_vllm` is set to `False`.
        cache_implementation (`str`, *optional*):
            Implementation of the cache method for faster generation when `use_vllm` is set to `False`.
        generation_kwargs (`dict[str, Any]`, *optional*):
            Additional keyword arguments to pass to [`~transformers.GenerationConfig`] (if using transformers) or
            `SamplingParams` (if using vLLM) when sampling completions. This can be used to further customize the
            generation behavior, such as setting `suppress_tokens`, `num_beams`, etc. If it contains keys that conflict
            with the other generation parameters (like `min_p`, `top_p`, etc.), they will override them.

        > Parameters that control generation acceleration powered by vLLM

        use_vllm (`bool`, *optional*, defaults to `False`):
            Whether to use vLLM for generating completions. If set to `True`, the trainer will use vLLM for generation
            instead of the default model.generate(). Requires `vllm` to be installed.
        vllm_model_impl (`str`, *optional*, defaults to `"vllm"`):
            Model implementation to use for vLLM. Must be one of `"transformers"` or `"vllm"`. `"transformers"`: Use
            the `transformers` backend for model implementation. `"vllm"`: Use the `vllm` library for model
            implementation.
        vllm_mode (`str`, *optional*, defaults to `"server"`):
            Mode to use for vLLM integration when `use_vllm` is set to `True`. Must be one of `"server"` or
            `"colocate"`.

            - `"server"`: The trainer will send generation requests to a separate vLLM server. Make sure a TRL vLLM
              server is running (start with `trl vllm-serve`).
            - `"colocate"`: vLLM will run in the same process and share the training GPUs. This avoids the need for a
              separate server but may cause resource contention with training.
        vllm_guided_decoding_regex (`str`, *optional*):
            Regex for vLLM guided decoding. If `None` (default), guided decoding is disabled.

        > Parameters that control the vLLM server (only used when `vllm_mode` is `"server"`)

        vllm_server_base_url (`str`, *optional*):
            Base URL for the vLLM server (e.g., `"http://localhost:8000"`). If provided, `vllm_server_host` and
            `vllm_server_port` are ignored.
        vllm_server_host (`str`, *optional*, defaults to `"0.0.0.0"`):
            Host of the vLLM server to connect to. Ignored if `vllm_server_base_url` is provided.
        vllm_server_port (`int`, *optional*, defaults to `8000`):
            Port of the vLLM server to connect to. Ignored if `vllm_server_base_url` is provided.
        vllm_server_timeout (`float`, *optional*, defaults to `240.0`):
            Total timeout duration in seconds to wait for the vLLM server to be up. If the server is not up after the
            timeout, a `ConnectionError` is raised.

        > Parameters that control colocated vLLM execution (only used when `vllm_mode` is `"colocate"`)

        vllm_gpu_memory_utilization (`float`, *optional*, defaults to `0.55`):
            Control the GPU memory utilization for vLLM. This setting only applies when `vllm_mode` is set to
            `"colocate"`. If you are using `vllm_mode="server"`, this parameter must be passed separately when
            launching the vLLM server via the `--vllm_gpu_memory_utilization` flag.
        vllm_tensor_parallel_size (`int`, *optional*, defaults to `1`):
            Control the tensor parallel size for vLLM. This setting only applies when `vllm_mode` is set to
            `"colocate"`. If you are using `vllm_mode="server"`, this parameter must be passed separately when
            launching the vLLM server via the `--vllm_tensor_parallel_size` flag.

        > Other parameters

        ds3_gather_for_generation (`bool`, *optional*, defaults to `True`):
            This setting applies to DeepSpeed ZeRO-3. If enabled, the policy model weights are gathered for generation,
            improving generation speed. However, disabling this option allows training models that exceed the VRAM
            capacity of a single GPU, albeit at the cost of slower generation. Disabling this option is not compatible
            with vLLM generation.
        model_init_kwargs (`dict[str, Any]`, *optional*):
            Keyword arguments to pass to `AutoModelForCausalLM.from_pretrained` when instantiating the model from a
            string.
    
    """
    vllm_sampling_params: Optional[Any] = field(
        default = None,
        metadata = {'help': 'vLLM SamplingParams'},
    )
    unsloth_num_chunks : Optional[int] = field(
        default = -1,
        metadata = {'help': 'Chunk size to reduce memory usage. -1 is most efficient.'},
    )
    unsloth_logit_chunk_multiplier : Optional[int] = field(
            default = None,
            metadata = {'help': 'Multiplier for chunked logit computations.'},
        )
    unsloth_grpo_mini_batch : Optional[int] = field(
        default = None,
        metadata = {'help': 'Mini batch size for GRPO hidden state accumulation. Default is None unless user defines it.'},
    )
    max_seq_length : Optional[int] = field(
        default = None,
        metadata = {'help': 'Maximum sequence length to truncate to.'},
    )
    def __init__(
        self,
        output_dir = None,
        overwrite_output_dir = None,
        do_train = False,
        do_eval = False,
        do_predict = False,
        eval_strategy = 'no',
        prediction_loss_only = False,
        per_device_train_batch_size = 4,
        per_device_eval_batch_size = 4,
        per_gpu_train_batch_size = None,
        per_gpu_eval_batch_size = None,
        gradient_accumulation_steps = 2,
        eval_accumulation_steps = 2,
        eval_delay = 0,
        torch_empty_cache_steps = 250,
        learning_rate = 5e-05,
        weight_decay = 0.01,
        adam_beta1 = 0.9,
        adam_beta2 = 0.999,
        adam_epsilon = 1e-08,
        max_grad_norm = 1.0,
        num_train_epochs = 3.0,
        max_steps = -1,
        lr_scheduler_type = 'linear',
        lr_scheduler_kwargs = None,
        warmup_ratio = 0.1,
        warmup_steps = 0,
        log_level = 'passive',
        log_level_replica = 'warning',
        log_on_each_node = True,
        logging_dir = None,
        logging_strategy = 'steps',
        logging_first_step = False,
        logging_steps = 1,
        logging_nan_inf_filter = False,
        save_strategy = 'steps',
        save_steps = 500,
        save_total_limit = None,
        save_safetensors = True,
        save_on_each_node = False,
        save_only_model = False,
        restore_callback_states_from_checkpoint = False,
        no_cuda = False,
        use_cpu = False,
        use_mps_device = False,
        seed = 3407,
        data_seed = 3407,
        jit_mode_eval = False,
        bf16 = False,
        fp16 = False,
        fp16_opt_level = 'O1',
        half_precision_backend = 'auto',
        bf16_full_eval = False,
        fp16_full_eval = False,
        tf32 = None,
        local_rank = -1,
        ddp_backend = None,
        tpu_num_cores = None,
        tpu_metrics_debug = False,
        debug = '',
        dataloader_drop_last = False,
        eval_steps = None,
        dataloader_num_workers = 0,
        dataloader_prefetch_factor = None,
        past_index = -1,
        run_name = None,
        disable_tqdm = None,
        remove_unused_columns = True,
        label_names = None,
        load_best_model_at_end = False,
        metric_for_best_model = None,
        greater_is_better = None,
        ignore_data_skip = False,
        fsdp = None,
        fsdp_min_num_params = 0,
        fsdp_config = None,
        fsdp_transformer_layer_cls_to_wrap = None,
        accelerator_config = None,
        parallelism_config = None,
        deepspeed = None,
        label_smoothing_factor = 0.0,
        optim = 'adamw_8bit',
        optim_args = None,
        adafactor = False,
        group_by_length = False,
        length_column_name = 'length',
        report_to = 'none',
        project = 'huggingface',
        trackio_space_id = 'trackio',
        ddp_find_unused_parameters = None,
        ddp_bucket_cap_mb = None,
        ddp_broadcast_buffers = None,
        dataloader_pin_memory = True,
        dataloader_persistent_workers = False,
        skip_memory_metrics = True,
        use_legacy_prediction_loop = False,
        push_to_hub = False,
        resume_from_checkpoint = None,
        hub_model_id = None,
        hub_strategy = 'every_save',
        hub_token = None,
        hub_private_repo = None,
        hub_always_push = False,
        hub_revision = None,
        gradient_checkpointing = True,
        gradient_checkpointing_kwargs = None,
        include_inputs_for_metrics = False,
        eval_do_concat_batches = True,
        fp16_backend = 'auto',
        push_to_hub_model_id = None,
        push_to_hub_organization = None,
        push_to_hub_token = None,
        mp_parameters = '',
        auto_find_batch_size = False,
        full_determinism = False,
        torchdynamo = None,
        ray_scope = 'last',
        ddp_timeout = 1800,
        torch_compile = False,
        torch_compile_backend = None,
        torch_compile_mode = None,
        include_tokens_per_second = False,
        include_num_input_tokens_seen = False,
        neftune_noise_alpha = None,
        optim_target_modules = None,
        batch_eval_metrics = False,
        eval_on_start = False,
        use_liger_kernel = False,
        liger_kernel_config = None,
        eval_use_gather_object = False,
        average_tokens_across_devices = True,
        reward_model_path = None,
        judge = None,
        max_new_tokens = 64,
        max_length = 512,
        temperature = 0.9,
        top_p = 1.0,
        top_k = None,
        min_p = None,
        repetition_penalty = 1.0,
        generation_kwargs = {},
        use_transformers_paged = False,
        cache_implementation = None,
        missing_eos_penalty = None,
        loss_type = 'sigmoid',
        disable_dropout = True,
        use_vllm = False,
        vllm_model_impl = 'vllm',
        vllm_guided_decoding_regex = None,
        vllm_gpu_memory_utilization = 0.55,
        vllm_mode = 'colocate',
        vllm_server_base_url = None,
        vllm_server_host = '0.0.0.0',
        vllm_server_port = 8000,
        vllm_server_timeout = 240.0,
        vllm_tensor_parallel_size = 1,
        ds3_gather_for_generation = True,
        model_init_kwargs = None,
        reward_weights = None,
        dataset_num_proc = None,
        gpu_memory_utilization = None,
        vllm_sampling_params = None,
        unsloth_num_chunks = -1,
        unsloth_logit_chunk_multiplier = None,
        unsloth_grpo_mini_batch = None,
        max_seq_length = None,
        **kwargs,
    ):
        if learning_rate < 1e-7: print(f'Unsloth: Your learning rate of `{learning_rate}` is too small and less than 1e-7! Consider increasing it, otherwise gradient updates will be close to 0!')
        if learning_rate > 1: print(f'Unsloth: Your learning rate of `{learning_rate}` is way too larger > 1! Consider decreasing it to 1e-1, otherwise gradient updates will explode!')
        if num_train_epochs is None:
            num_train_epochs = 3.0  # Default to 3 epochs if None, max_steps will override
        if output_dir is None and save_strategy == 'steps' and save_steps == 500:
            output_dir = 'unsloth_training_checkpoints'
            save_strategy = 'no'
        import multiprocessing as _mp
        if dataset_num_proc is None:
            if _mp.get_start_method() != 'fork':
                dataset_num_proc = None
            else:
                import psutil
                dataset_num_proc = min(max((psutil.cpu_count() or 1)+4, 2), 64)
                memory_gb_left = psutil.virtual_memory().available / (1024**3)
                if memory_gb_left <= 2: dataset_num_proc = 1
                else: dataset_num_proc = min(dataset_num_proc, int(memory_gb_left))
        if temperature <= 0:
            raise ValueError('Unsloth: Please set a positive non-zero temperature since your results will be wrong.')
        elif temperature >= 10:
            raise ValueError('Unsloth: Please set a positive non-zero temperature less than 10, since sampling will be quite erratic.')
        
        
        super().__init__(
            output_dir = output_dir,
            overwrite_output_dir = overwrite_output_dir,
            do_train = do_train,
            do_eval = do_eval,
            do_predict = do_predict,
            eval_strategy = eval_strategy,
            prediction_loss_only = prediction_loss_only,
            per_device_train_batch_size = per_device_train_batch_size,
            per_device_eval_batch_size = per_device_eval_batch_size,
            per_gpu_train_batch_size = per_gpu_train_batch_size,
            per_gpu_eval_batch_size = per_gpu_eval_batch_size,
            gradient_accumulation_steps = gradient_accumulation_steps,
            eval_accumulation_steps = eval_accumulation_steps,
            eval_delay = eval_delay,
            torch_empty_cache_steps = torch_empty_cache_steps,
            learning_rate = learning_rate,
            weight_decay = weight_decay,
            adam_beta1 = adam_beta1,
            adam_beta2 = adam_beta2,
            adam_epsilon = adam_epsilon,
            max_grad_norm = max_grad_norm,
            num_train_epochs = num_train_epochs,
            max_steps = max_steps,
            lr_scheduler_type = lr_scheduler_type,
            lr_scheduler_kwargs = lr_scheduler_kwargs,
            warmup_ratio = warmup_ratio,
            warmup_steps = warmup_steps,
            log_level = log_level,
            log_level_replica = log_level_replica,
            log_on_each_node = log_on_each_node,
            logging_dir = logging_dir,
            logging_strategy = logging_strategy,
            logging_first_step = logging_first_step,
            logging_steps = logging_steps,
            logging_nan_inf_filter = logging_nan_inf_filter,
            save_strategy = save_strategy,
            save_steps = save_steps,
            save_total_limit = save_total_limit,
            save_safetensors = save_safetensors,
            save_on_each_node = save_on_each_node,
            save_only_model = save_only_model,
            restore_callback_states_from_checkpoint = restore_callback_states_from_checkpoint,
            no_cuda = no_cuda,
            use_cpu = use_cpu,
            use_mps_device = use_mps_device,
            seed = seed,
            data_seed = data_seed,
            jit_mode_eval = jit_mode_eval,
            bf16 = bf16,
            fp16 = fp16,
            fp16_opt_level = fp16_opt_level,
            half_precision_backend = half_precision_backend,
            bf16_full_eval = bf16_full_eval,
            fp16_full_eval = fp16_full_eval,
            tf32 = tf32,
            local_rank = local_rank,
            ddp_backend = ddp_backend,
            tpu_num_cores = tpu_num_cores,
            tpu_metrics_debug = tpu_metrics_debug,
            debug = debug,
            dataloader_drop_last = dataloader_drop_last,
            eval_steps = eval_steps,
            dataloader_num_workers = dataloader_num_workers,
            dataloader_prefetch_factor = dataloader_prefetch_factor,
            past_index = past_index,
            run_name = run_name,
            disable_tqdm = disable_tqdm,
            remove_unused_columns = remove_unused_columns,
            label_names = label_names,
            load_best_model_at_end = load_best_model_at_end,
            metric_for_best_model = metric_for_best_model,
            greater_is_better = greater_is_better,
            ignore_data_skip = ignore_data_skip,
            fsdp = fsdp,
            fsdp_min_num_params = fsdp_min_num_params,
            fsdp_config = fsdp_config,
            fsdp_transformer_layer_cls_to_wrap = fsdp_transformer_layer_cls_to_wrap,
            accelerator_config = accelerator_config,
            parallelism_config = parallelism_config,
            deepspeed = deepspeed,
            label_smoothing_factor = label_smoothing_factor,
            optim = optim,
            optim_args = optim_args,
            adafactor = adafactor,
            group_by_length = group_by_length,
            length_column_name = length_column_name,
            report_to = report_to,
            project = project,
            trackio_space_id = trackio_space_id,
            ddp_find_unused_parameters = ddp_find_unused_parameters,
            ddp_bucket_cap_mb = ddp_bucket_cap_mb,
            ddp_broadcast_buffers = ddp_broadcast_buffers,
            dataloader_pin_memory = dataloader_pin_memory,
            dataloader_persistent_workers = dataloader_persistent_workers,
            skip_memory_metrics = skip_memory_metrics,
            use_legacy_prediction_loop = use_legacy_prediction_loop,
            push_to_hub = push_to_hub,
            resume_from_checkpoint = resume_from_checkpoint,
            hub_model_id = hub_model_id,
            hub_strategy = hub_strategy,
            hub_token = hub_token,
            hub_private_repo = hub_private_repo,
            hub_always_push = hub_always_push,
            hub_revision = hub_revision,
            gradient_checkpointing = gradient_checkpointing,
            gradient_checkpointing_kwargs = gradient_checkpointing_kwargs,
            include_inputs_for_metrics = include_inputs_for_metrics,
            eval_do_concat_batches = eval_do_concat_batches,
            fp16_backend = fp16_backend,
            push_to_hub_model_id = push_to_hub_model_id,
            push_to_hub_organization = push_to_hub_organization,
            push_to_hub_token = push_to_hub_token,
            mp_parameters = mp_parameters,
            auto_find_batch_size = auto_find_batch_size,
            full_determinism = full_determinism,
            torchdynamo = torchdynamo,
            ray_scope = ray_scope,
            ddp_timeout = ddp_timeout,
            torch_compile = torch_compile,
            torch_compile_backend = torch_compile_backend,
            torch_compile_mode = torch_compile_mode,
            include_tokens_per_second = include_tokens_per_second,
            include_num_input_tokens_seen = include_num_input_tokens_seen,
            neftune_noise_alpha = neftune_noise_alpha,
            optim_target_modules = optim_target_modules,
            batch_eval_metrics = batch_eval_metrics,
            eval_on_start = eval_on_start,
            use_liger_kernel = use_liger_kernel,
            liger_kernel_config = liger_kernel_config,
            eval_use_gather_object = eval_use_gather_object,
            average_tokens_across_devices = average_tokens_across_devices,
            reward_model_path = reward_model_path,
            judge = judge,
            max_new_tokens = max_new_tokens,
            max_length = max_length,
            temperature = temperature,
            top_p = top_p,
            top_k = top_k,
            min_p = min_p,
            repetition_penalty = repetition_penalty,
            generation_kwargs = generation_kwargs,
            use_transformers_paged = use_transformers_paged,
            cache_implementation = cache_implementation,
            missing_eos_penalty = missing_eos_penalty,
            loss_type = loss_type,
            disable_dropout = disable_dropout,
            use_vllm = use_vllm,
            vllm_model_impl = vllm_model_impl,
            vllm_guided_decoding_regex = vllm_guided_decoding_regex,
            vllm_gpu_memory_utilization = vllm_gpu_memory_utilization,
            vllm_mode = vllm_mode,
            vllm_server_base_url = vllm_server_base_url,
            vllm_server_host = vllm_server_host,
            vllm_server_port = vllm_server_port,
            vllm_server_timeout = vllm_server_timeout,
            vllm_tensor_parallel_size = vllm_tensor_parallel_size,
            ds3_gather_for_generation = ds3_gather_for_generation,
            model_init_kwargs = model_init_kwargs,
            reward_weights = reward_weights,
            dataset_num_proc = dataset_num_proc,
            gpu_memory_utilization = gpu_memory_utilization,**kwargs)
        self.vllm_sampling_params = vllm_sampling_params
        self.unsloth_num_chunks = unsloth_num_chunks
        if unsloth_grpo_mini_batch is not None:
            if self.generation_batch_size >= unsloth_grpo_mini_batch:
                self.unsloth_grpo_mini_batch = unsloth_grpo_mini_batch
            else:
                raise ValueError(
                    f"Unsloth GRPO mini batch size needs to be less than or equal to the effective generation batch size, "
                    f"which is self.per_device_train_batch_size * gradient_accumulation_steps."
                )
        self.unsloth_logit_chunk_multiplier = unsloth_logit_chunk_multiplier
        self.max_seq_length = max_seq_length

pass

class _UnslothOnlineDPOTrainer(BaseTrainer):
    r""""""

    _tag_names = ["trl", "online-dpo"]
    _name = "Online DPO"
    _paper = {
        "title": "Direct Language Model Alignment from Online AI Feedback",
        "id": "2402.04792",
        # docstyle-ignore
        "citation": textwrap.dedent("""\
            @article{guo2024direct,
                title        = {{Direct Language Model Alignment from Online AI Feedback}},
                author       = {Shangmin Guo and Biao Zhang and Tianlin Liu and Tianqi Liu and Misha Khalman and Felipe Llinares and Alexandre Ram{\'{e}} and Thomas Mesnard and Yao Zhao and Bilal Piot and Johan Ferret and Mathieu Blondel},
                year         = 2024,
                eprint       = {arXiv:2402.04792}
            }"""),
    }

    def __init__(
        self,
        model: Union[PreTrainedModel, nn.Module, str],
        ref_model: Union[PreTrainedModel, nn.Module, None] = None,
        reward_funcs: Optional[Union[RewardFunc, list[RewardFunc]]] = None,
        judge: Optional[BasePairwiseJudge] = None,
        args: Optional[OnlineDPOConfig] = None,
        data_collator: Optional[DataCollator] = None,
        train_dataset: Optional[Union[Dataset, IterableDataset]] = None,
        eval_dataset: Optional[Union[Dataset, IterableDataset, dict[str, Union[Dataset, IterableDataset]]]] = None,
        processing_class: Optional[Union[PreTrainedTokenizerBase, ProcessorMixin]] = None,
        reward_processing_classes: Optional[Union[PreTrainedTokenizerBase, list[PreTrainedTokenizerBase]]] = None,
        peft_config: Optional["PeftConfig"] = None,
        compute_metrics: Optional[Callable[[EvalPrediction], dict]] = None,
        callbacks: Optional[list[TrainerCallback]] = None,
        optimizers: tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (None, None),
        preprocess_logits_for_metrics: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
        # Deprecated parameters
        reward_model: Optional[Union[PreTrainedModel, nn.Module]] = None,
        reward_processing_class: Optional[PreTrainedTokenizerBase] = None,
    ) -> None:

        if hasattr(model, 'vllm_engine') and hasattr(args, 'use_vllm'):
            if (getattr(args, 'use_vllm', False) == False):
                args.use_vllm = True
        if not os.environ.get("TRL_EXPERIMENTAL_SILENCE"):
            warnings.warn(
                "This trainer will soon be moved to trl.experimental and is a candidate for removal. If you rely on "
                "it and want it to remain, please share your comments here: "
                "https://github.com/huggingface/trl/issues/4223. Silence this warning by setting environment variable "
                "TRL_EXPERIMENTAL_SILENCE=1."
            )
        if ref_model is model:
            raise ValueError(
                "`model` and `ref_model` cannot be the same object. If you want `ref_model` to be the "
                "same as `model`, either omit the `ref_model` argument or pass `None`."
            )

        self.ref_model = ref_model

        # Handle deprecated parameters for backward compatibility
        if reward_model is not None:
            warnings.warn(
                "The `reward_model` parameter is deprecated and will be removed in version 0.25.0. "
                "Please use `reward_funcs` instead. For example, change `reward_model=model` to `reward_funcs=model`.",
            )
            # Convert old reward_model to new reward_funcs format
            if reward_funcs is None:
                reward_funcs = reward_model
            else:
                warnings.warn(
                    "Both `reward_model` and `reward_funcs` are provided. Using `reward_funcs` and ignoring "
                    "`reward_model`.",
                )

        if reward_processing_class is not None:
            warnings.warn(
                "The `reward_processing_class` parameter is deprecated and will be removed in version 0.25.0. "
                "Please use `reward_processing_classes` instead. For example, change "
                "`reward_processing_class=tokenizer` to `reward_processing_classes=tokenizer`.",
            )
            # Convert old reward_processing_class to new reward_processing_classes format
            if reward_processing_classes is None:
                reward_processing_classes = reward_processing_class
            else:
                warnings.warn(
                    "Both `reward_processing_class` and `reward_processing_classes` are provided. Using "
                    "`reward_processing_classes` and ignoring `reward_processing_class`.",
                )

        # Validate reward configuration - must have exactly one of: judge, or reward_funcs
        reward_configs = sum(x is not None for x in [judge, reward_funcs])
        if reward_configs == 0:
            raise ValueError("One of `judge` or `reward_funcs` must be provided.")
        elif reward_configs > 1:
            if judge is not None:
                logger.warning(
                    "Both `judge` and `reward_funcs` are provided. Using `judge` and ignoring `reward_funcs`.",
                    UserWarning,
                )
                reward_funcs = None
        self.judge = judge

        # Handle reward_funcs
        if reward_funcs is not None:
            if not isinstance(reward_funcs, list):
                reward_funcs = [reward_funcs]
            self.reward_func_names = []

            # Process reward functions [convert strings to models, collect names]
            model_init_kwargs = args.model_init_kwargs or {}
            for i, reward_func in enumerate(reward_funcs):
                if isinstance(reward_func, str):
                    # Load model from string path
                    reward_funcs[i] = AutoModelForSequenceClassification.from_pretrained(
                        reward_func, num_labels=1, **model_init_kwargs
                    )
                if isinstance(reward_funcs[i], nn.Module):
                    self.reward_func_names.append(reward_funcs[i].config._name_or_path.split("/")[-1])
                else:
                    self.reward_func_names.append(reward_funcs[i].__name__)
            self.reward_funcs = reward_funcs

            # Handle reward processing classes for reward_funcs
            if reward_processing_classes is None:
                reward_processing_classes = [None] * len(reward_funcs)
            elif not isinstance(reward_processing_classes, list):
                reward_processing_classes = [reward_processing_classes]
            else:
                if len(reward_processing_classes) != len(reward_funcs):
                    raise ValueError(
                        "The number of reward processing classes must match the number of reward functions."
                    )

            self.reward_processing_classes = []
            for reward_processing_class_i, reward_func in zip(reward_processing_classes, reward_funcs):
                if isinstance(reward_func, PreTrainedModel):
                    if reward_processing_class_i is None:
                        reward_processing_class_i = AutoTokenizer.from_pretrained(reward_func.config._name_or_path)
                    if reward_processing_class_i.pad_token_id is None:
                        reward_processing_class_i.pad_token = reward_processing_class_i.eos_token
                    # Set pad token ID on reward model config
                    reward_func.config.pad_token_id = reward_processing_class_i.pad_token_id
                self.reward_processing_classes.append(reward_processing_class_i)
        else:
            self.reward_funcs = None
            self.reward_func_names = []
            self.reward_processing_classes = []

        # Handle reward_weights
        if reward_funcs is not None:
            if args.reward_weights is not None:
                if len(args.reward_weights) != len(self.reward_funcs):
                    raise ValueError(
                        f"Number of reward weights ({len(args.reward_weights)}) must match number of reward "
                        f"functions ({len(self.reward_funcs)})"
                    )
                self.reward_weights = torch.tensor(args.reward_weights, dtype=torch.float32)
            else:
                self.reward_weights = torch.ones(len(self.reward_funcs), dtype=torch.float32)
        else:
            self.reward_weights = None

        if args.missing_eos_penalty is not None and reward_funcs is None and judge is None:
            # Check if this is the old reward_model case
            if reward_model is not None:
                logger.warning(
                    "The `missing_eos_penalty` parameter is deprecated when used with the deprecated `reward_model` parameter. "
                    "Please use `reward_funcs` instead of `reward_model` to continue using this feature.",
                    FutureWarning,
                    stacklevel=2,
                )
            else:
                raise ValueError("`missing_eos_penalty` is only supported when `reward_funcs` is provided.")

        if args is None:
            raise ValueError("`args` must be provided.")

        # Check that the processing_class is provided
        if processing_class is None:
            raise ValueError("`processing_class` must be provided.")

        model_init_kwargs = args.model_init_kwargs or {}
        if isinstance(model, str):
            model_id = model

            # Handle dtype in model_init_kwargs
            dtype = model_init_kwargs.get("dtype")
            if isinstance(dtype, torch.dtype) or dtype == "auto" or dtype is None:
                pass
            elif isinstance(dtype, str):
                dtype = getattr(torch, dtype)
                model_init_kwargs["dtype"] = dtype
            else:
                raise ValueError(
                    "Invalid `dtype` passed to `OnlineDPOConfig`. Expected either 'auto' or a string "
                    f"representing a `torch.dtype` (e.g., 'float32'), but got {dtype}."
                )

            model = AutoModelForCausalLM.from_pretrained(model_id, **model_init_kwargs)
        else:
            if args.model_init_kwargs is not None:
                raise ValueError(
                    "You passed `model_init_kwargs` to the `OnlineDPOConfig`, but your model is already instantiated. "
                    "This argument can only be used when the `model` argument is a string."
                )
        self.is_encoder_decoder = model.config.is_encoder_decoder
        self.is_vision_model = model.config.model_type in MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING_NAMES.keys()

        if False:
            pass

        # Enable gradient checkpointing if requested
        if args.gradient_checkpointing:
            model = self._enable_gradient_checkpointing(model, args)

        # Disable dropout in the model and reference model
        if args.disable_dropout:
            disable_dropout_in_model(model)
            if self.ref_model is not None:
                disable_dropout_in_model(self.ref_model)

        # Handle the ref_model
        # Usually, the user wants the ref model to be the initial version of the model. When using PEFT, it's easy to
        # get the ref model, as it's just the model with a disabled adapter. When not using PEFT, we need to create
        # the ref model from the model by copying it and disable the gradients and set it in evaluation mode.
        if ref_model is None:  # No ref model provided, the most common case
            if False:
                self.ref_model = create_reference_model(model)  # copy, disable gradients, set eval mode
            else:
                self.ref_model = None  # we don't need a ref model here, we can just disable the adapter.
        else:  # rare case, the user provided a ref model
            self.ref_model = ref_model
            self.ref_model.eval()

        # Disable the gradient and set the reward model in eval mode
        if reward_funcs is not None:
            for reward_func in reward_funcs:
                if isinstance(reward_func, PreTrainedModel):
                    reward_func.eval()

        self.max_length = args.max_length

        self.stats = {
            "objective/kl": [],
            "objective/entropy": [],
            "objective/non_score_reward": [],
            "rewards/chosen": [],
            "rewards/rejected": [],
            "rewards/accuracies": [],
            "rewards/margins": [],
            "logps/chosen": [],
            "logps/rejected": [],
            "val/contain_eos_token": [],
            "beta": [],
        }
        if self.reward_funcs is not None:
            self.stats["objective/rlhf_reward"] = []
            self.stats["objective/scores_margin"] = []
            self.stats["objective/scores"] = []

        # Store generation parameters for later use
        self.use_vllm = args.use_vllm
        self.num_generations = 2  # Generate 2 completions per prompt for Online DPO
        self.temperature = args.temperature
        self.top_p = args.top_p
        self.top_k = args.top_k
        self.min_p = args.min_p
        self.repetition_penalty = args.repetition_penalty
        self.use_transformers_paged = args.use_transformers_paged
        self.vllm_mode = args.vllm_mode if args.use_vllm else None
        self.vllm_gpu_memory_utilization = args.vllm_gpu_memory_utilization
        self.vllm_tensor_parallel_size = args.vllm_tensor_parallel_size
        self.vllm_model_impl = args.vllm_model_impl

        # Handle pad token for processors or tokenizers
        if isinstance(processing_class, ProcessorMixin):
            tokenizer = processing_class.tokenizer
        elif isinstance(processing_class, PreTrainedTokenizerBase):
            tokenizer = processing_class
        else:
            raise TypeError("The `processing_class` must be either a `PreTrainedTokenizerBase` or a `ProcessorMixin`")

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        self.pad_token = tokenizer.pad_token
        self.pad_token_id = tokenizer.pad_token_id
        self.eos_token_id = tokenizer.eos_token_id

        # Vision tokens for VLM support
        self.image_token_id = getattr(processing_class, "image_token_id", None)
        self.vision_start_token_id = getattr(processing_class, "vision_start_token_id", None)
        self.vision_end_token_id = getattr(processing_class, "vision_end_token_id", None)
        # Get the image token string for token collapsing
        self.image_token = None
        if self.image_token_id is not None:
            self.image_token = tokenizer.decode([self.image_token_id])

        # Define the collator if not provided
        if data_collator is None:
            data_collator = DPODataCollatorWithPadding(pad_token_id=self.pad_token_id)

        # The trainer estimates the number of FLOPs [floating-point operations] using the number of elements in the
        # input tensor associated with the key "input_ids". However, in Online DPO, the sampled data does not include
        # the "input_ids" key. As a result, the trainer issues the warning: "Could not estimate the number of tokens
        # of the input, floating-point operations will not be computed." To suppress this warning, we set the
        # "estimate_tokens" key in the model's "warnings_issued" dictionary to True. This acts as a flag to indicate
        # that the warning has already been issued.
        model.warnings_issued["estimate_tokens"] = True

        super().__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            compute_metrics=compute_metrics,
            callbacks=callbacks,
            optimizers=optimizers,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        )

        # Add tags for models that have been loaded with the correct transformers version
        if hasattr(self.model, "add_model_tags"):
            self.model.add_model_tags(self._tag_names)

        self._beta = args.beta

        # Set up generation configuration and vLLM after super[].__init__
        if self.use_vllm:
            if not is_vllm_available():
                raise ImportError(
                    "vLLM is not available and `use_vllm` is set to True. Please install vLLM with "
                    "`pip install trl[vllm]` to use it."
                )

            if self.vllm_mode == "server":
                if self.accelerator.is_main_process:
                    if args.vllm_server_base_url is not None:
                        base_url = args.vllm_server_base_url
                    else:
                        base_url = f"http://{args.vllm_server_host}:{args.vllm_server_port}"
                    self.vllm_client = VLLMClient(base_url=base_url, connection_timeout=args.vllm_server_timeout)
                    self.vllm_client.init_communicator(device=torch.cuda.current_device())
                else:
                    self.vllm_client = None
            elif self.vllm_mode == "colocate":
                vllm_kwargs = {
                    "model": model.name_or_path,
                    "tensor_parallel_size": self.vllm_tensor_parallel_size,
                    "gpu_memory_utilization": self.vllm_gpu_memory_utilization,
                    "model_impl": self.vllm_model_impl,
                    "max_num_seqs": self.args.per_device_train_batch_size * self.vllm_tensor_parallel_size,
                    "max_model_len": args.max_length + args.max_new_tokens,
                    "distributed_executor_backend": "external_launcher",
                    "seed": self.accelerator.process_index // self.vllm_tensor_parallel_size,
                    "max_num_batched_tokens": 4096,
                }
                os.environ["RANK"] = str(self.accelerator.process_index)
                os.environ["LOCAL_RANK"] = str(self.accelerator.local_process_index)
                os.environ["WORLD_SIZE"] = str(self.accelerator.num_processes)
                ensure_master_addr_port()

                self.llm = model.vllm_engine
            else:
                raise ValueError(f"vllm_mode must be either 'server' or 'colocate', got '{self.vllm_mode}'.")
            self.guided_decoding_regex = args.vllm_guided_decoding_regex
            self._last_loaded_step = -1
            generation_params = {
                "n": 2,
                "repetition_penalty": self.repetition_penalty,
                "temperature": self.temperature,
                "top_p": self.top_p,
                "top_k": -1 if self.top_k is None else self.top_k,
                "min_p": 0.0 if self.min_p is None else self.min_p,
                "max_tokens": args.max_new_tokens,
                "detokenize": False,
            }
            if args.generation_kwargs is not None:
                generation_params.update(args.generation_kwargs)
            if self.guided_decoding_regex:
                generation_params["guided_decoding"] = GuidedDecodingParams(regex=self.guided_decoding_regex)
            self.generation_config = SamplingParams(**generation_params)
            self.accelerator.wait_for_everyone()
        else:
            # Set up transformers generation config
            generation_kwargs = {
                "max_new_tokens": args.max_new_tokens,
                "do_sample": True,
                "pad_token_id": self.pad_token_id,
                "bos_token_id": tokenizer.bos_token_id,
                "eos_token_id": self.eos_token_id,
                "temperature": self.temperature,
                "top_k": self.top_k,
                "top_p": self.top_p,
                "repetition_penalty": self.repetition_penalty,
                "use_cache": True if not self.args.gradient_checkpointing else False,
            }
            # Add min_p if supported
            if self.min_p is not None:
                generation_kwargs["min_p"] = self.min_p
            if args.generation_kwargs is not None:
                generation_kwargs.update(args.generation_kwargs)
            # Remove None values
            generation_kwargs = {k: v for k, v in generation_kwargs.items() if v is not None}
            self.generation_config = GenerationConfig(**generation_kwargs)

        if self.ref_model is not None:
            if self.is_deepspeed_enabled:
                self.ref_model = prepare_deepspeed(self.ref_model, self.accelerator)
            elif self.is_fsdp_enabled:
                self.ref_model = prepare_fsdp(self.ref_model, self.accelerator)
            else:
                self.ref_model = self.accelerator.prepare_model(self.ref_model, evaluation_mode=True)
        if self.reward_funcs is not None:
            for i, reward_func in enumerate(self.reward_funcs):
                if isinstance(reward_func, PreTrainedModel):
                    if self.is_deepspeed_enabled:
                        self.reward_funcs[i] = prepare_deepspeed(reward_func, self.accelerator)
                    else:
                        # set device placement to True to make `prepare_model` move `reward_func` to device when using fsdp
                        self.reward_funcs[i] = self.accelerator.prepare_model(
                            reward_func, evaluation_mode=True, device_placement=True
                        )

    @property
    def beta(self):
        if isinstance(self._beta, list):
            epoch = self.state.epoch
            return self._beta[epoch] if epoch < len(self._beta) else self._beta[-1]
        else:
            return self._beta

    @staticmethod
    def tokenize_row(feature, is_encoder_decoder: bool, tokenizer: PreTrainedTokenizerBase) -> dict[str, Any]:
        """Tokenize a single row from a DPO specific dataset."""
        if not is_encoder_decoder:
            batch = tokenizer(feature["prompt"], add_special_tokens=False)
            # Add BOS token to head of prompt. Avoid adding if it's already there
            if tokenizer.bos_token_id is not None:
                prompt_len_input_ids = len(batch["input_ids"])
                if prompt_len_input_ids == 0 or tokenizer.bos_token_id != batch["input_ids"][0]:
                    batch["input_ids"] = [tokenizer.bos_token_id] + batch["input_ids"]
                    batch["attention_mask"] = [1] + batch["attention_mask"]
        else:
            batch = tokenizer(feature["prompt"], add_special_tokens=True)
        batch = {f"prompt_{key}": value for key, value in batch.items()}
        return batch

    # Same as Trainer.get_train_dataloader but skip the "remove_unused_columns".
    @wraps(Trainer.get_train_dataloader)
    def get_train_dataloader(self) -> DataLoader:
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        train_dataset = self.train_dataset
        data_collator = self.data_collator
        dataloader_params = {
            "batch_size": self._train_batch_size,
            "collate_fn": data_collator,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
            "persistent_workers": self.args.dataloader_persistent_workers,
        }

        if not isinstance(train_dataset, torch.utils.data.IterableDataset):
            dataloader_params["sampler"] = self._get_train_sampler()
            dataloader_params["drop_last"] = self.args.dataloader_drop_last
            dataloader_params["worker_init_fn"] = seed_worker
            dataloader_params["prefetch_factor"] = self.args.dataloader_prefetch_factor

        return self.accelerator.prepare(DataLoader(train_dataset, **dataloader_params))

    # Same as Trainer.get_eval_dataloader but skip the "remove_unused_columns".
    @wraps(Trainer.get_eval_dataloader)
    def get_eval_dataloader(self, eval_dataset: Optional[Union[str, Dataset]] = None) -> DataLoader:
        if eval_dataset is None and self.eval_dataset is None:
            raise ValueError("Trainer: evaluation requires an eval_dataset.")

        # If we have persistent workers, don't do a fork bomb especially as eval datasets
        # don't change during training
        dataloader_key = eval_dataset if isinstance(eval_dataset, str) else "eval"
        if (
            hasattr(self, "_eval_dataloaders")
            and dataloader_key in self._eval_dataloaders
            and self.args.dataloader_persistent_workers
        ):
            return self.accelerator.prepare(self._eval_dataloaders[dataloader_key])

        eval_dataset = (
            self.eval_dataset[eval_dataset]
            if isinstance(eval_dataset, str)
            else eval_dataset
            if eval_dataset is not None
            else self.eval_dataset
        )
        data_collator = self.data_collator

        dataloader_params = {
            "batch_size": self.args.eval_batch_size,
            "collate_fn": data_collator,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
            "persistent_workers": self.args.dataloader_persistent_workers,
        }

        if not isinstance(eval_dataset, torch.utils.data.IterableDataset):
            dataloader_params["sampler"] = self._get_eval_sampler(eval_dataset)
            dataloader_params["drop_last"] = self.args.dataloader_drop_last
            dataloader_params["prefetch_factor"] = self.args.dataloader_prefetch_factor

        # accelerator.free_memory() will destroy the references, so
        # we need to store the non-prepared version
        eval_dataloader = DataLoader(eval_dataset, **dataloader_params)
        if self.args.dataloader_persistent_workers:
            if hasattr(self, "_eval_dataloaders"):
                self._eval_dataloaders[dataloader_key] = eval_dataloader
            else:
                self._eval_dataloaders = {dataloader_key: eval_dataloader}

        return self.accelerator.prepare(eval_dataloader)

    def _enable_gradient_checkpointing(self, model: PreTrainedModel, args: OnlineDPOConfig) -> PreTrainedModel:
        """Enables gradient checkpointing for the model."""
        # Ensure use_cache is disabled
        model.config.use_cache = False

        # Enable gradient checkpointing on the base model for PEFT
        if is_peft_model(model):
            model.base_model.gradient_checkpointing_enable()
        # Enable gradient checkpointing for non-PEFT models
        else:
            model.gradient_checkpointing_enable()

        gradient_checkpointing_kwargs = args.gradient_checkpointing_kwargs or {}
        use_reentrant = (
            "use_reentrant" not in gradient_checkpointing_kwargs or gradient_checkpointing_kwargs["use_reentrant"]
        )

        if use_reentrant:
            model.enable_input_require_grads()

        return model

    def _generate_vllm(self, prompts, images=None):
        eos_token_id = self.eos_token_id
        pad_token_id = self.pad_token_id

        # Generate completion_ids and prompt_ids based on mode
        if self.vllm_mode == "server":
            completion_ids, prompt_ids = self._generate_vllm_server(prompts, images)
        elif self.vllm_mode == "colocate":
            completion_ids, prompt_ids = self._generate_vllm_colocate(prompts, images)

        # Shared padding, masking, and tensor conversion logic
        max_prompt_length = max(len(ids) for ids in prompt_ids)
        prompt_mask = [[0] * (max_prompt_length - len(ids)) + [1] * len(ids) for ids in prompt_ids]
        prompt_ids = [[pad_token_id] * (max_prompt_length - len(ids)) + ids for ids in prompt_ids]
        max_tokens = self.generation_config.max_tokens
        completion_mask = [[1] * len(ids) + [0] * (max_tokens - len(ids)) for ids in completion_ids]
        completion_ids = [
            ids + [eos_token_id] if ids[-1] != eos_token_id and len(ids) < max_tokens else ids
            for ids in completion_ids
        ]
        completion_ids = [ids + [pad_token_id] * (max_tokens - len(ids)) for ids in completion_ids]

        # Convert to tensors
        prompt_ids = torch.tensor(prompt_ids, device=self.accelerator.device)
        prompt_mask = torch.tensor(prompt_mask, device=self.accelerator.device)
        completion_ids = torch.tensor(completion_ids, device=self.accelerator.device)
        completion_mask = torch.tensor(completion_mask, device=self.accelerator.device)

        return prompt_ids, prompt_mask, completion_ids, completion_mask

    def _generate_vllm_server(self, prompts, images=None):
        """Generate completions using vLLM server mode"""
        has_images = images is not None

        # Update vLLM server weights if needed
        if hasattr(self, "_last_loaded_step") and self.state.global_step != self._last_loaded_step:
            self._move_model_to_vllm()
            self._last_loaded_step = self.state.global_step
        elif not hasattr(self, "_last_loaded_step"):
            self._move_model_to_vllm()
            self._last_loaded_step = self.state.global_step

        # Apply chat template if conversational
        if is_conversational({"prompt": prompts[0]}):
            prompts_text = [apply_chat_template({"prompt": p}, self.processing_class)["prompt"] for p in prompts]
        else:
            prompts_text = prompts
        # Gather all prompts to main process
        all_prompts = gather_object(prompts_text)
        if has_images:
            all_images = gather_object(images)

        if self.accelerator.is_main_process:
            # Since 'prompts' contains 'num_generations' duplicates, we first take unique prompts, and generate
            # num_generations outputs for each one. This is faster than generating outputs for each duplicate
            # prompt individually.
            ordered_set_of_prompts = all_prompts[:: self.num_generations]
            if has_images:
                ordered_set_of_images = all_images[:: self.num_generations]
            else:
                ordered_set_of_images = None
            completion_ids = self.vllm_client.generate(
                prompts=ordered_set_of_prompts,
                images=ordered_set_of_images,
                n=self.num_generations,
                repetition_penalty=self.repetition_penalty,
                temperature=self.temperature,
                top_p=self.top_p,
                top_k=-1 if self.top_k is None else self.top_k,
                min_p=0.0 if self.min_p is None else self.min_p,
                max_tokens=self.generation_config.max_tokens,
                guided_decoding_regex=self.guided_decoding_regex if hasattr(self, "guided_decoding_regex") else None,
                generation_kwargs=self.args.generation_kwargs,
            )
            # Flatten: each prompt generates 2 completions
            completion_ids = [[comp_id] for prompt_completions in completion_ids for comp_id in prompt_completions]
        else:
            completion_ids = [None] * (len(all_prompts) * 2)

        # Broadcast completions to all processes
        completion_ids = broadcast_object_list(completion_ids, from_process=0)

        # Each process takes its slice
        process_slice = slice(
            self.accelerator.process_index * len(prompts) * 2,
            (self.accelerator.process_index + 1) * len(prompts) * 2,
        )
        completion_ids = completion_ids[process_slice]

        # Create prompt_ids by tokenizing locally
        prompt_inputs = self.processing_class(
            text=prompts_text,
            return_tensors="pt",
            padding=True,
            padding_side="left",
            add_special_tokens=False,
        )
        prompt_ids = []
        for prompt_tokens in prompt_inputs["input_ids"]:
            prompt_ids.extend([prompt_tokens.tolist(), prompt_tokens.tolist()])  # 2 copies for 2 completions
        return completion_ids, prompt_ids

    def _generate_vllm_colocate(self, prompts, images=None):
        """Generate completions using vLLM colocate mode"""
        # Update model weights if needed - only after gradient accumulation completes
        if self.state.global_step != self._last_loaded_step:
            self._move_model_to_vllm()
            self._last_loaded_step = self.state.global_step

        # Apply chat template if conversational
        if is_conversational({"prompt": prompts[0]}):
            prompts_text = [apply_chat_template({"prompt": p}, self.processing_class)["prompt"] for p in prompts]
        else:
            prompts_text = prompts

        # Prepare vLLM inputs with images if available
        if images is not None:
            vllm_inputs = []
            for prompt, image in zip(prompts_text, images):
                if image is not None:
                    vllm_inputs.append({"prompt": prompt, "multi_modal_data": {"image": image}})
                else:
                    vllm_inputs.append(prompt)
        else:
            vllm_inputs = prompts_text

        outputs = self.llm.generate(vllm_inputs, self.generation_config, use_tqdm=False, lora_request = self.model.load_lora('online_dpo_trainer_lora_model', load_tensors = True))

        completion_ids = [list(output.outputs[i].token_ids) for i in range(2) for output in outputs]
        prompt_ids = [list(output.prompt_token_ids) for _ in range(2) for output in outputs]

        return completion_ids, prompt_ids

    def _move_model_to_vllm(self):
        """Synchronize model weights to vLLM server with support for PEFT, DeepSpeed, and FSDP"""
        # For DeepSpeed ZeRO-3 and FSDP, we need to gather all parameters before operations
        deepspeed_plugin = self.accelerator.state.deepspeed_plugin
        zero_stage_3 = deepspeed_plugin is not None and deepspeed_plugin.zero_stage == 3
        if zero_stage_3:
            import deepspeed

            gather_if_zero3 = deepspeed.zero.GatheredParameters
        else:
            gather_if_zero3 = nullcontext

        if is_peft_model(self.model):
            # With PEFT and FSDP/DeepSpeed ZeRO Stage 3, we must gather the full model at once before merging, as
            # merging adapters in a sharded manner is not supported.
            # TODO: does this work with FSDP?
            with gather_if_zero3(list(self.model.parameters())):
                self.model.merge_adapter()

                # Update vLLM weights while parameters are gathered
                if self.is_fsdp_enabled:  # note if using FSDP, gather_if_zero3 is nullcontext
                    # Update vLLM weights while parameters are gathered
                    # For PEFT with FSDP we need to use the memory efficient post-order traversal
                    fsdp_plugin = getattr(self.accelerator.state, "fsdp_plugin", None)
                    fsdp_version = getattr(fsdp_plugin, "fsdp_version", 1) if fsdp_plugin else 1
                    if fsdp_version == 1:
                        # use memory-efficient post-order traversal for FSDP
                        self._sync_fsdp1_params_to_vllm(self.model)
                    elif fsdp_version == 2:
                        self._sync_fsdp2_params_to_vllm(self.model)
                else:
                    # DeepSpeed ZeRO-3 with PEFT
                    for name, param in self.model.named_parameters():
                        # When using PEFT, we need to recover the original parameter name and discard some parameters
                        name = name.removeprefix("base_model.model.").replace(".base_layer", "")
                        if self.model.prefix in name:
                            continue
                        # When module to save, remove its prefix and discard the original module
                        if "original_module" in name:
                            continue
                        name = self._fix_param_name_to_vllm(name, extra_prefixes=["modules_to_save.default."])

                        if self.vllm_mode == "server" and self.accelerator.is_main_process:
                            self.vllm_client.update_named_param(name, param.data)
                        elif self.vllm_mode == "colocate":

                            pass

                            pass
                # Unmerge adapters while parameters are still gathered
                self.model.unmerge_adapter()
                # Parameters will automatically be repartitioned when exiting the context
        else:
            # For non-PEFT models, simply gather (if needed) and update each parameter individually.
            if self.is_fsdp_enabled:
                fsdp_plugin = getattr(self.accelerator.state, "fsdp_plugin", None)
                fsdp_version = getattr(fsdp_plugin, "fsdp_version", 1) if fsdp_plugin else 1
                if fsdp_version == 1:
                    self._sync_fsdp1_params_to_vllm(self.model)  # use memory-efficient post-order traversal for FSDP
                elif fsdp_version == 2:
                    self._sync_fsdp2_params_to_vllm(self.model)
            else:
                for name, param in self.model.named_parameters():
                    name = self._fix_param_name_to_vllm(name)
                    with gather_if_zero3([param]):
                        if self.vllm_mode == "server" and self.accelerator.is_main_process:
                            self.vllm_client.update_named_param(name, param.data)
                        elif self.vllm_mode == "colocate":

                            pass

                            pass

        # Reset cache on vLLM
        if self.vllm_mode == "server" and self.accelerator.is_main_process:
            self.vllm_client.reset_prefix_cache()
        elif self.vllm_mode == "colocate":
            self.llm.reset_prefix_cache()

    def _sync_fsdp1_params_to_vllm(self, module: nn.Module, prefix: str = "", visited=None):
        """Memory-efficient post-order traversal of FSDP modules to extract full parameters and sync with vLLM."""
        # For FSDP1, we need to recurse into children and also use summon_full_params
        if visited is None:
            visited = set()
        for child_name, child_module in module.named_children():
            child_prefix = f"{prefix}.{child_name}" if prefix else child_name
            self._sync_fsdp1_params_to_vllm(
                child_module, prefix=child_prefix, visited=visited
            )  # recurse into the child

        if isinstance(module, FSDP):
            with FSDP.summon_full_params(module, recurse=False, writeback=False):
                for param_name, param in module.named_parameters():
                    full_name = f"{prefix}.{param_name}" if prefix else param_name
                    full_name = self._fix_param_name_to_vllm(full_name, extra_prefixes=["_fsdp_wrapped_module."])

                    if full_name in visited:
                        continue  # skip FSDP subtrees already traversed
                    visited.add(full_name)

                    if self.vllm_mode == "server" and self.accelerator.is_main_process:
                        self.vllm_client.update_named_param(full_name, param.data)
                    elif self.vllm_mode == "colocate":

                        pass

                        pass

    def _sync_fsdp2_params_to_vllm(self, module: nn.Module):
        # For FSDP2, module already covers all parameters, so no need for recursion
        for name, param in module.items():
            if param.is_cpu:
                param = param.to(torch.device("cuda"))
            param = param.full_tensor()

            if self.vllm_mode == "server" and self.accelerator.is_main_process:
                self.vllm_client.update_named_param(name, param)
            elif self.vllm_mode == "colocate":

                pass

                pass

    def _fix_param_name_to_vllm(self, name, extra_prefixes: Optional[list[str]] = None):
        """Clean parameter names for vLLM compatibility"""
        extra_prefixes = extra_prefixes or []
        prefixes = ["_checkpoint_wrapped_module."] + extra_prefixes
        for prefix in prefixes:
            name = name.replace(prefix, "")
        return name

    def process_vision_row(
        self, features: dict[str, Union[list, torch.Tensor]], processing_class=None
    ) -> dict[str, list[int]]:
        """
        Process a vision row for VLM models (adapted from DPO trainer)
        """
        processor = processing_class or self.processing_class
        processed_features = processor(images=[features["image"]], text=features["prompt"], add_special_tokens=False)

        prompt_input_ids = processed_features["input_ids"][0]

        # Create the output dict with required fields
        output = {
            "prompt_input_ids": prompt_input_ids,
            "prompt_attention_mask": processed_features["attention_mask"][0],
        }

        # Add vision-specific fields
        if "pixel_values" in processed_features:
            output["pixel_values"] = processed_features["pixel_values"][0]
        if "pixel_attention_mask" in processed_features:
            output["pixel_attention_mask"] = processed_features["pixel_attention_mask"][0]
        if "image_sizes" in processed_features:
            output["image_sizes"] = processed_features["image_sizes"][0]

        return output

    def _generate(self, model, prompts, images=None):
        """Generate completions using the model"""
        device = next(model.parameters()).device
        eos_token_id = self.eos_token_id
        pad_token_id = self.pad_token_id

        # Apply chat template and tokenize the input
        inputs = [{"prompt": prompt} for prompt in prompts]

        # Add images if provided (VLM support)
        if images is not None:
            for i, image in enumerate(images):
                inputs[i]["image"] = image

        # Apply chat template to get text prompts
        prompts_text = [maybe_apply_chat_template(x, self.processing_class)["prompt"] for x in inputs]

        # Handle image token collapsing/removal
        # The chat template sometimes inserts a single image token into the prompt text. However, when this text is
        # later tokenized, the single image token string is expanded into multiple image token IDs, depending on the
        # image size. We need to handle this properly.
        if self.image_token is not None and images is not None:
            escaped_img_token = re.escape(self.image_token)
            # Search for the image token in the chat template
            if hasattr(self.processing_class, "chat_template") and self.processing_class.chat_template:
                if re.search(escaped_img_token, self.processing_class.chat_template):
                    # Collapse repeated image tokens back into a single token
                    prompts_text = [
                        re.sub(rf"({escaped_img_token})+", self.image_token, text) for text in prompts_text
                    ]
                else:
                    # If the chat template doesn't use the image token, remove all instances
                    if self.vision_end_token_id is not None:
                        escaped_eoi_token = re.escape(
                            self.processing_class.tokenizer.decode([self.vision_end_token_id])
                        )
                        prompts_text = [
                            re.sub(rf"({escaped_img_token})+{escaped_eoi_token}", "", text) for text in prompts_text
                        ]
                    else:
                        # If vision_end_token_id is None, just remove the image tokens
                        prompts_text = [re.sub(rf"({escaped_img_token})+", "", text) for text in prompts_text]

        # Prepare kwargs for processing class
        kwargs = {}
        if images is not None:
            kwargs = {"images": [[img] for img in images]}

        # Process inputs using the processing class (handles both VLM and LLM)
        prompt_inputs = self.processing_class(
            text=prompts_text,
            return_tensors="pt",
            padding=True,
            padding_side="left",
            add_special_tokens=False,
            **kwargs,
        )

        prompt_inputs = {k: v.to(device) for k, v in prompt_inputs.items()}
        # Convert vision inputs to model's dtype for proper computation
        if "pixel_values" in prompt_inputs:
            # Handle DataParallel wrapped models
            model_dtype = getattr(model, "dtype", None)
            if model_dtype is None and hasattr(model, "module"):
                model_dtype = model.module.dtype
            if model_dtype is not None:
                prompt_inputs["pixel_values"] = prompt_inputs["pixel_values"].to(model_dtype)

        # Sample 2 completions per prompt of size `max_new_tokens` from the model
        prompt_ids = prompt_inputs["input_ids"].repeat(2, 1)
        prompt_mask = prompt_inputs["attention_mask"].repeat(2, 1)

        # Prepare vision inputs if available
        vision_generation_kwargs = {}
        if self.is_vision_model and images is not None:
            if "pixel_values" in prompt_inputs:
                vision_generation_kwargs["pixel_values"] = prompt_inputs["pixel_values"].repeat(2, 1, 1, 1)
            if "pixel_attention_mask" in prompt_inputs:
                vision_generation_kwargs["pixel_attention_mask"] = prompt_inputs["pixel_attention_mask"].repeat(2, 1)
            if "image_sizes" in prompt_inputs:
                vision_generation_kwargs["image_sizes"] = prompt_inputs["image_sizes"].repeat(2, 1)
            if "image_grid_thw" in prompt_inputs:
                vision_generation_kwargs["image_grid_thw"] = prompt_inputs["image_grid_thw"].repeat(2, 1)

        if self.use_transformers_paged:
            previous_attn = self.model_wrapped.config._attn_implementation

            if is_flash_attn_2_available():
                self.model_wrapped.config._attn_implementation = "paged_attention"
            else:
                self.model_wrapped.config._attn_implementation = "sdpa_paged"
            with (
                profiling_context(self, "transformers.generate_batch"),
                unwrap_model_for_generation(
                    model, self.accelerator, gather_deepspeed3_params=self.args.ds3_gather_for_generation
                ) as unwrapped_model,
                torch.no_grad(),
                FSDP.summon_full_params(self.model_wrapped, recurse=False) if self.is_fsdp_enabled else nullcontext(),
            ):
                # Cast to the appropriate dtype based on training configuration
                if self.args.bf16:
                    unwrapped_model.to(torch.bfloat16)
                elif self.args.fp16:
                    unwrapped_model.to(torch.float16)
                with torch.inference_mode():
                    all_outputs = unwrapped_model.generate_batch(
                        prompt_ids.tolist(),
                        generation_config=self.generation_config,
                        progress_bar=False,
                    )
                    unwrapped_model.train()  # restore training mode, as generate_batch forces eval mode
            completion_ids = [output.generated_tokens for output in all_outputs.values()]
            completion_ids = [torch.tensor(ids, device=device) for ids in completion_ids]
            completion_ids = pad(completion_ids, padding_value=self.pad_token_id, padding_side="right")
            prompt_completion_ids = torch.cat([prompt_ids, completion_ids], dim=1)
            # Restore the original attention implementation, training mode
            self.model_wrapped.config._attn_implementation = previous_attn

            # Extract completion_ids and create completion_mask
            prompt_length = prompt_ids.size(1)
            completion_ids = prompt_completion_ids[:, prompt_length:]
            completion_ids, completion_mask = truncate_right(completion_ids, eos_token_id, pad_token_id)

            return prompt_ids, prompt_mask, completion_ids, completion_mask
        else:
            # Regular generation path
            with (
                profiling_context(self, "transformers.generate"),
                unwrap_model_for_generation(
                    model, self.accelerator, gather_deepspeed3_params=self.args.ds3_gather_for_generation
                ) as unwrapped_model,
                torch.no_grad(),
                FSDP.summon_full_params(self.model_wrapped, recurse=False) if self.is_fsdp_enabled else nullcontext(),
            ):
                # Setup cache implementation if specified
                if self.args.cache_implementation is not None:
                    unwrapped_model.generation_config.cache_implementation = self.args.cache_implementation

                # Standard generation
                output = unwrapped_model.generate(
                    input_ids=prompt_ids,
                    attention_mask=prompt_mask,
                    generation_config=self.generation_config,
                    **vision_generation_kwargs,
                )

            completion_ids = output[:, prompt_ids.size(1) :]
            completion_ids, completion_mask = truncate_right(completion_ids, eos_token_id, pad_token_id)

            return prompt_ids, prompt_mask, completion_ids, completion_mask

    def _calculate_rewards_from_functions(self, prompts, completions, completion_ids_list, **reward_kwargs):
        """
        Calculate rewards using reward functions
        """
        device = self.accelerator.device
        rewards_per_func = torch.zeros(len(prompts), len(self.reward_funcs), device=device)

        # Add trainer state to reward kwargs for dynamic reward shaping
        reward_kwargs["trainer_state"] = self.state

        for i, (reward_func, reward_processing_class) in enumerate(
            zip(self.reward_funcs, self.reward_processing_classes)
        ):
            if isinstance(reward_func, nn.Module):  # Model-based reward function
                # Handle conversational vs text input
                if is_conversational({"prompt": prompts[0]}):
                    messages = [{"messages": p + c} for p, c in zip(prompts, completions)]
                    texts = [apply_chat_template(x, reward_processing_class)["text"] for x in messages]
                else:
                    texts = [p + c for p, c in zip(prompts, completions)]

                # Tokenize and get reward scores
                reward_inputs = reward_processing_class(
                    text=texts, return_tensors="pt", padding=True, padding_side="right", add_special_tokens=False
                )
                reward_inputs = {k: v.to(device) for k, v in reward_inputs.items()}

                with torch.inference_mode():
                    rewards_per_func[:, i] = reward_func(**reward_inputs).logits[:, 0]  # Shape (B*G,)
            else:
                # Custom reward function
                output_reward_func = reward_func(
                    prompts=prompts, completions=completions, completion_ids=completion_ids_list, **reward_kwargs
                )
                # Convert None values to NaN
                output_reward_func = [reward if reward is not None else torch.nan for reward in output_reward_func]
                rewards_per_func[:, i] = torch.tensor(output_reward_func, dtype=torch.float32, device=device)

        # Weight and sum across all reward functions
        if self.reward_weights is not None:
            total_rewards = (rewards_per_func * self.reward_weights.to(device).unsqueeze(0)).nansum(dim=1)
        else:
            total_rewards = rewards_per_func.nansum(dim=1)

        return total_rewards

    def _forward(self, model, prompt_ids, prompt_mask, completion_ids, completion_mask, vision_inputs=None):
        # Get the number of tokens to truncate from prompt
        num_tokens_to_truncate = max(prompt_ids.size(1) + completion_ids.size(1) - self.max_length, 0)

        # Truncate left to avoid oom
        prompt_ids = prompt_ids[:, num_tokens_to_truncate:]
        prompt_mask = prompt_mask[:, num_tokens_to_truncate:]

        # Concat the prompt and completion
        prompt_completion_ids = torch.cat((prompt_ids, completion_ids), dim=1)
        prompt_completion_mask = torch.cat((prompt_mask, completion_mask), dim=1)

        # Prepare model kwargs with vision inputs if available
        model_kwargs = {"attention_mask": prompt_completion_mask}
        if vision_inputs is not None:
            if "pixel_values" in vision_inputs:
                model_kwargs["pixel_values"] = vision_inputs["pixel_values"]
            if "pixel_attention_mask" in vision_inputs:
                model_kwargs["pixel_attention_mask"] = vision_inputs["pixel_attention_mask"]
            if "image_sizes" in vision_inputs:
                model_kwargs["image_sizes"] = vision_inputs["image_sizes"]
            if "image_grid_thw" in vision_inputs:
                model_kwargs["image_grid_thw"] = vision_inputs["image_grid_thw"]

        # Get the logprobs of the completions from the model
        output = model(prompt_completion_ids, **model_kwargs)

        # There is 1 offset, because the model predicts the next token
        prompt_len = prompt_ids.size(1)
        start_idx = prompt_len - 1 if prompt_len > 0 else 0
        # Only slice off the last logit when we have a prompt, otherwise we need all logits
        end_idx = -1 if prompt_len > 0 else None
        logits = output.logits[:, start_idx:end_idx]

        # Take the completion tokens logprob
        logprobs = torch.take_along_dim(logits.log_softmax(dim=-1), completion_ids.unsqueeze(-1), dim=2).squeeze(-1)
        return logprobs

    def training_step(
        self, model: nn.Module, inputs: dict[str, Union[torch.Tensor, Any]], num_items_in_batch: Optional[int] = None
    ) -> torch.Tensor:
        model.train()

        prompts = inputs["prompt"]
        batch_size = len(prompts)

        # Handle images for VLM support
        has_images = "image" in inputs
        images = None
        if has_images:
            images = inputs["image"]
            # Convert conversational prompts to include image tokens
            for prompt in prompts:
                if isinstance(prompt, list):
                    for message in prompt:
                        if not isinstance(message, dict):
                            continue
                        content = message.get("content")
                        role = message.get("role")
                        if isinstance(content, str):
                            if role == "user":
                                message["content"] = [{"type": "image"}, {"type": "text", "text": content}]
                            elif role == "system":
                                message["content"] = [{"type": "text", "text": content}]

        if self.args.use_vllm:
            prompt_ids, prompt_mask, completion_ids, completion_mask = self._generate_vllm(prompts, images)
        else:
            prompt_ids, prompt_mask, completion_ids, completion_mask = self._generate(model, prompts, images)

        contain_eos_token = torch.any(completion_ids == self.eos_token_id, dim=-1)

        # Extract vision inputs if available for VLM support
        vision_inputs = None
        if has_images and self.is_vision_model and not self.args.use_vllm:
            # For vision models with transformers generation, we need to prepare vision inputs
            # Process the images to get vision inputs that can be passed through the forward pass
            vision_inputs = {}
            kwargs = {"images": [[img] for img in images]}
            processed = self.processing_class(
                text=[""] * len(images),  # Dummy text for vision processing
                return_tensors="pt",
                **kwargs,
            )
            # Handle DataParallel wrapped models
            model_device = getattr(model, "device", None)
            model_dtype = getattr(model, "dtype", None)
            if model_device is None and hasattr(model, "module"):
                model_device = model.module.device
                model_dtype = model.module.dtype
            # Move vision tensors to device and convert to model dtype
            # Need to duplicate for 2 completions per prompt
            if "pixel_values" in processed:
                vision_inputs["pixel_values"] = (
                    processed["pixel_values"].to(model_device, dtype=model_dtype).repeat(2, 1, 1, 1)
                )
            if "pixel_attention_mask" in processed:
                vision_inputs["pixel_attention_mask"] = processed["pixel_attention_mask"].to(model_device).repeat(2, 1)
            if "image_sizes" in processed:
                vision_inputs["image_sizes"] = processed["image_sizes"].to(model_device).repeat(2, 1)
            if "image_grid_thw" in processed:
                vision_inputs["image_grid_thw"] = processed["image_grid_thw"].to(model_device).repeat(2, 1)

        logprobs = self._forward(model, prompt_ids, prompt_mask, completion_ids, completion_mask, vision_inputs)
        with torch.no_grad():
            if self.ref_model is not None:
                ref_logprobs = self._forward(
                    self.ref_model, prompt_ids, prompt_mask, completion_ids, completion_mask, vision_inputs
                )
            else:  # peft case: we just need to disable the adapter
                with self.model.disable_adapter():
                    ref_logprobs = self._forward(
                        self.model, prompt_ids, prompt_mask, completion_ids, completion_mask, vision_inputs
                    )

        # Decode the completions, and format them if the input is conversational
        device = logprobs.device
        completions = self.processing_class.batch_decode(completion_ids, skip_special_tokens=True)
        if is_conversational({"prompt": prompts[0]}):
            completions = [[{"role": "assistant", "content": completion}] for completion in completions]

        # Get the reward from reward functions, judge, or deprecated reward_model
        if self.reward_funcs is not None:
            # First create completion_ids_list for custom reward functions
            completion_ids_list = [completion_ids[i].tolist() for i in range(completion_ids.shape[0])]

            # Extract additional fields from inputs for reward functions
            reward_kwargs = {}
            keys = [key for key in inputs if key not in ["prompt"]]
            for key in keys:
                if isinstance(inputs[key], (list, tuple)):
                    # Repeat input fields to match number of completions (2 per prompt)
                    reward_kwargs[key] = inputs[key] * 2
                else:
                    reward_kwargs[key] = inputs[key]

            # Calculate rewards using reward functions
            rewards = self._calculate_rewards_from_functions(
                prompts=2 * prompts, completions=completions, completion_ids_list=completion_ids_list, **reward_kwargs
            )

            # Apply missing EOS penalty if configured
            if self.args.missing_eos_penalty is not None:
                rewards[~contain_eos_token] -= self.args.missing_eos_penalty

            # Split rewards into chosen/rejected pairs
            first_half, second_half = rewards.split(batch_size)
            mask = first_half >= second_half
        elif self.judge is not None:
            # Once formatted, conversational data may contain special tokens (such as <|im_start|>) that are not
            # directly understandable by the judge and could alter its judgment. To avoid this and make the judge
            # independent of the model's chat template, we use the raw conversation data, and apply our own chat
            # template to it.
            if is_conversational({"prompt": prompts[0]}):
                environment = jinja2.Environment()
                template = environment.from_string(SIMPLE_CHAT_TEMPLATE)
                prompts = [template.render(messages=prompt) for prompt in prompts]
                completions = [template.render(messages=completion) for completion in completions]

            ranks_of_first_completion = self.judge.judge(
                prompts, list(zip(completions[:batch_size], completions[batch_size:]))
            )

            # convert ranks to a True/False mask:
            # when rank == 0, it means the first completion is the best
            # when rank == 1, it means the second completion is the best
            mask = torch.tensor([rank == 0 for rank in ranks_of_first_completion], device=device)

        batch_range = torch.arange(batch_size, device=device)
        chosen_indices = batch_range + (~mask * batch_size)
        rejected_indices = batch_range + (mask * batch_size)

        # Build tensor so that the first half is the chosen examples and the second half the rejected examples
        cr_indices = torch.cat((chosen_indices, rejected_indices), dim=0)  # cr = chosen and rejected
        cr_logprobs = logprobs[cr_indices]
        cr_ref_logprobs = ref_logprobs[cr_indices]

        # mask out the padding tokens
        padding_mask = ~completion_mask.bool()
        cr_padding_mask = padding_mask[cr_indices]

        cr_logprobs_sum = (cr_logprobs * ~cr_padding_mask).sum(1)
        cr_ref_logprobs_sum = (cr_ref_logprobs * ~cr_padding_mask).sum(1)

        # Split the chosen and rejected examples
        chosen_logprobs_sum, rejected_logprobs_sum = torch.split(cr_logprobs_sum, batch_size)
        chosen_ref_logprobs_sum, rejected_ref_logprobs_sum = torch.split(cr_ref_logprobs_sum, batch_size)
        pi_logratios = chosen_logprobs_sum - rejected_logprobs_sum
        ref_logratios = chosen_ref_logprobs_sum - rejected_ref_logprobs_sum

        logits = pi_logratios - ref_logratios

        if self.args.loss_type == "sigmoid":
            losses = -F.logsigmoid(self.beta * logits)
        elif self.args.loss_type == "ipo":
            losses = (logits - 1 / (2 * self.beta)) ** 2
        else:
            raise NotImplementedError(f"invalid loss type {self.loss_type}")

        loss = losses.mean()

        # Log everything
        if self.reward_funcs is not None:
            # When using reward_funcs, we have rewards instead of scores
            scores_margin = rewards[chosen_indices] - rewards[rejected_indices]
            self.stats["objective/scores_margin"].append(
                self.accelerator.gather_for_metrics(scores_margin.mean()).mean().item()
            )
            self.stats["objective/scores"].append(self.accelerator.gather_for_metrics(rewards.mean()).mean().item())
        self.stats["val/contain_eos_token"].append(contain_eos_token.float().mean().item())
        self.stats["logps/chosen"].append(self.accelerator.gather_for_metrics(chosen_logprobs_sum).mean().item())
        self.stats["logps/rejected"].append(self.accelerator.gather_for_metrics(rejected_logprobs_sum).mean().item())

        kl = logprobs - ref_logprobs
        mean_kl = kl.sum(1).mean()
        self.stats["objective/kl"].append(self.accelerator.gather_for_metrics(mean_kl).mean().item())
        non_score_reward = (-self.beta * kl).sum(1)
        mean_non_score_reward = non_score_reward.mean()
        self.stats["objective/non_score_reward"].append(
            self.accelerator.gather_for_metrics(mean_non_score_reward).mean().item()
        )
        if self.reward_funcs is not None:
            # Calculate RLHF reward by combining rewards with non_score_reward
            rlhf_reward = rewards + non_score_reward
            self.stats["objective/rlhf_reward"].append(self.accelerator.gather_for_metrics(rlhf_reward).mean().item())

        mean_entropy = -logprobs.sum(1).mean()
        self.stats["objective/entropy"].append(self.accelerator.gather_for_metrics(mean_entropy).mean().item())
        chosen_rewards = self.beta * (chosen_logprobs_sum - chosen_ref_logprobs_sum)
        gathered_chosen_rewards = self.accelerator.gather_for_metrics(chosen_rewards)
        self.stats["rewards/chosen"].append(gathered_chosen_rewards.mean().item())
        rejected_rewards = self.beta * (rejected_logprobs_sum - rejected_ref_logprobs_sum)
        gathered_rejected_rewards = self.accelerator.gather_for_metrics(rejected_rewards)
        self.stats["rewards/rejected"].append(gathered_rejected_rewards.mean().item())
        margin = gathered_chosen_rewards - gathered_rejected_rewards
        self.stats["rewards/margins"].append(margin.mean().item())
        accuracy = margin > 0
        self.stats["rewards/accuracies"].append(accuracy.float().mean().item())
        self.stats["beta"].append(self.beta)

        if (
            self.args.torch_empty_cache_steps is not None
            and self.state.global_step % self.args.torch_empty_cache_steps == 0
        ):
            empty_cache()

        kwargs = {}

        # For LOMO optimizers you need to explicitly use the learning rate
        if self.args.optim in [OptimizerNames.LOMO, OptimizerNames.ADALOMO]:
            kwargs["learning_rate"] = self._get_learning_rate()

        if self.args.n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu parallel training

        self.accelerator.backward(loss, **kwargs)

        return loss.detach() / self.args.gradient_accumulation_steps

    # Same as Trainer._maybe_log_save_evaluate but log our metrics
    def _maybe_log_save_evaluate(
        self, tr_loss, grad_norm, model, trial, epoch, ignore_keys_for_eval, start_time, learning_rate=None
    ):
        if self.control.should_log and self.state.global_step > self._globalstep_last_logged:
            logs: dict[str, float] = {}

            # all_gather + mean() to get average loss over all processes
            tr_loss_scalar = self._nested_gather(tr_loss).mean().item()

            # reset tr_loss to zero
            tr_loss -= tr_loss

            logs["loss"] = round(tr_loss_scalar / (self.state.global_step - self._globalstep_last_logged), 4)
            if grad_norm is not None:
                logs["grad_norm"] = grad_norm.detach().item() if isinstance(grad_norm, torch.Tensor) else grad_norm
            if learning_rate is not None:
                logs["learning_rate"] = learning_rate
            else:
                logs["learning_rate"] = self._get_learning_rate()

            # Add our metrics
            for key, val in self.stats.items():
                logs[key] = sum(val) / len(val)
            self.stats = {key: [] for key in self.stats}  # reset stats

            self._total_loss_scalar += tr_loss_scalar
            self._globalstep_last_logged = self.state.global_step
            self.store_flos()
            self.log(logs, start_time)

        metrics = None
        if self.control.should_evaluate:
            metrics = self._evaluate(trial, ignore_keys_for_eval)
            is_new_best_metric = self._determine_best_metric(metrics=metrics, trial=trial)

            if self.args.save_strategy == "best":
                self.control.should_save = is_new_best_metric

        if self.control.should_save:
            self._save_checkpoint(model, trial)
            self.control = self.callback_handler.on_save(self.args, self.state, self.control)

    # Ensure the model card is saved along with the checkpoint
    def _save_checkpoint(self, model, trial):
        if self.args.hub_model_id is None:
            model_name = Path(self.args.output_dir).name
        else:
            model_name = self.args.hub_model_id.split("/")[-1]
        self.create_model_card(model_name=model_name)
        super()._save_checkpoint(model, trial)
class UnslothOnlineDPOTrainer(_UnslothOnlineDPOTrainer):
    """
    
    Initialize OnlineDPOTrainer.

    Args:
        model (`Union[str, nn.Module, PreTrainedModel]`):
            Model to be trained. Can be either:

            - A string, being the *model id* of a pretrained model hosted inside a model repo on huggingface.co, or a
              path to a *directory* containing model weights saved using
              [`~transformers.PreTrainedModel.save_pretrained`], e.g., `'./my_model_directory/'`. The model is loaded
              using [`~transformers.AutoModelForCausalLM.from_pretrained`] with the keyword arguments in
              `args.model_init_kwargs`.
            - A [`~transformers.PreTrainedModel`] object. Only causal language models are supported.
        ref_model ([`~transformers.PreTrainedModel`] or `torch.nn.Module` or `None`):
            The reference model to use for training. If None is specified, the reference model will be created from the
            model.
        judge ([`BasePairwiseJudge`]):
            The judge to use for pairwise comparison of model completions.
        reward_funcs (`Union[RewardFunc, list[RewardFunc]]`, *optional*):
            Reward functions to be used for computing the rewards. To compute the rewards, we call all the reward
            functions with the prompts and completions and sum the rewards. Can be either:

            - A single reward function: Can be a string (path to model), a [`~transformers.PreTrainedModel`], or a
              custom callable function.
            - A list of reward functions: Must all be of compatible types.

            Note: Only one of `judge`, or `reward_funcs` should be provided.
        args ([`OnlineDPOConfig`]):
            The online DPO config arguments to use for training.
        data_collator ([`~transformers.DataCollator`]):
            The data collator to use for training. If None is specified, the default data collator
            ([`DPODataCollatorWithPadding`]) will be used which will pad the sequences to the maximum length of the
            sequences in the batch, given a dataset of paired sequences.
        train_dataset ([`~datasets.Dataset`] or [`~datasets.IterableDataset`]):
            The dataset to use for training.
        eval_dataset ([`~datasets.Dataset`], [`~datasets.IterableDataset`] or `dict[str, Union[Dataset, IterableDataset]]`):
            The dataset to use for evaluation.
        processing_class ([`~transformers.PreTrainedTokenizerBase`] or [`~transformers.ProcessorMixin`], *optional*):
            Processing class used to process the data. If provided, will be used to automatically process the inputs
            for the model, and it will be saved along the model to make it easier to rerun an interrupted training or
            reuse the fine-tuned model.
        reward_processing_classes ([`~transformers.PreTrainedTokenizerBase`] or `list[PreTrainedTokenizerBase]`, *optional*):
            Processing classes corresponding to the reward functions specified in `reward_funcs`. Can be either:

            - A single processing class: Used when `reward_funcs` contains only one reward function.
            - A list of processing classes: Must match the order and length of the reward functions in `reward_funcs`.

            If set to `None`, the tokenizer for each model-based reward function is automatically loaded using
            [`~transformers.AutoTokenizer.from_pretrained`].
        peft_config ([`~peft.PeftConfig`], *optional*):
            PEFT configuration used to wrap the model. If `None`, the model is not wrapped.
        compute_metrics (`Callable[[EvalPrediction], dict]`, *optional*):
            The function to use to compute the metrics. Must take a `EvalPrediction` and return a dictionary string to
            metric values.
        callbacks (`list[transformers.TrainerCallback]`):
            The callbacks to use for training.
        optimizers (`tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR]`):
            The optimizer and scheduler to use for training.
        preprocess_logits_for_metrics (`Callable[[torch.Tensor, torch.Tensor], torch.Tensor]`):
            The function to use to preprocess the logits before computing the metrics.

        reward_model:

            <Deprecated version="0.22.0">

            This parameter is deprecated and will be removed in version 0.25.0. Use `reward_funcs` instead.

            </Deprecated>
    
    """
    def __init__(
        self,
        model,
        ref_model = None,
        reward_funcs = None,
        judge = None,
        args = None,
        data_collator = None,
        train_dataset = None,
        eval_dataset = None,
        processing_class = None,
        reward_processing_classes = None,
        peft_config = None,
        compute_metrics = None,
        callbacks = None,
        preprocess_logits_for_metrics = None,
        reward_model = None,
        reward_processing_class = None,
        **kwargs
    ):
        if args is None: args = UnslothOnlineDPOConfig()
        use_bf16 = getattr(args, 'bf16', False)
        if type(use_bf16) is not bool: use_bf16 = False
        use_fp16 = getattr(args, 'fp16', False)
        if type(use_fp16) is not bool: use_fp16 = False
        force_float32 = False
        full_finetuning = os.environ.get('UNSLOTH_ENABLE_FULL_FINETUNING', '0') == '1'
        if not full_finetuning and (os.environ.get('UNSLOTH_FORCE_FLOAT32', '0') == '1'):
            print('Unsloth: Switching to float32 training since model cannot work with float16')
            force_float32 = True
        mixed_precision_dtype = os.environ.get('UNSLOTH_MIXED_PRECISION', 'float32')
        dtype = getattr(model.config, 'dtype', None) or getattr(model.config, 'torch_dtype', None)
        if dtype is None: dtype = model.get_input_embeddings().weight.dtype
        from unsloth_zoo.utils import _get_dtype
        dtype = _get_dtype(dtype)
        float16 = dtype == torch.float16
        if not force_float32 and (float16 and use_bf16): raise TypeError('Unsloth: Model is in float16 precision but you want to use bfloat16 precision. Set fp16 to `True` and bf16 to `False`')
        if not force_float32 and (not float16 and use_fp16): raise TypeError('Unsloth: Model is in bfloat16 precision but you want to use float16 precision. Set fp16 to `False` and bf16 to `True`')
        if force_float32:
            # Forced float32 training
            args.fp16 = False
            args.bf16 = False
            os.environ['ACCELERATE_MIXED_PRECISION'] = 'no'
            if hasattr(args, 'mixed_precision'): args.mixed_precision = 'no'
            # args.mixed_precision is a new argument which needs to be set now
        elif (not use_bf16 and not use_fp16) and mixed_precision_dtype == 'float32':
            # Mixed precision training
            args.fp16 = float16
            args.bf16 = not float16
            os.environ['ACCELERATE_MIXED_PRECISION'] = 'fp16' if float16 else 'bf16'
            if hasattr(args, 'mixed_precision'): args.mixed_precision = 'fp16' if float16 else 'bf16'
            # args.mixed_precision is a new argument which needs to be set now
        elif mixed_precision_dtype == 'bfloat16':
            # Both False since bfloat16 full finetuning doesn't do any autocasting.
            args.fp16 = False
            args.bf16 = False
            os.environ['ACCELERATE_MIXED_PRECISION'] = 'no'
            if hasattr(args, 'mixed_precision'): args.mixed_precision = 'no'
            # args.mixed_precision is a new argument which needs to be set now
        
        if getattr(args, 'eval_dataset', None) is not None and getattr(args, 'eval_strategy', 'no') == 'no':
            args.eval_strategy = 'steps'
            if getattr(args, 'eval_steps', None) is None: args.eval_steps = 0.1
        ga_steps = getattr(args, 'gradient_accumulation_steps', None)
        if ga_steps is not None and ga_steps > 1:
            from transformers import __version__ as transformers_version
            if Version(transformers_version) <= Version('4.45.2'):
                print('**** Unsloth: Please use our fixed gradient_accumulation_steps by updating transformers, TRL and Unsloth!\n'
                      '`pip install --upgrade --no-cache-dir --force-reinstall --no-deps unsloth transformers trl unsloth_zoo`')
        if getattr(args, 'eval_strategy', 'no') != 'no':
            eval_bsz = getattr(args, 'per_device_eval_batch_size', 8)
            if eval_bsz == 8 and args.per_device_train_batch_size < eval_bsz: args.per_device_eval_batch_size = args.per_device_train_batch_size
            if getattr(args, 'eval_accumulation_steps', None) is None and ga_steps is not None: args.eval_accumulation_steps = ga_steps
        fp16_full_eval = getattr(args, 'fp16_full_eval', False)
        if type(fp16_full_eval) is not bool: fp16_full_eval = False
        bf16_full_eval = getattr(args, 'bf16_full_eval', False)
        if type(bf16_full_eval) is not bool: bf16_full_eval = False
        if args.fp16 and bf16_full_eval: args.bf16_full_eval = False; args.fp16_full_eval = True
        if args.bf16 and fp16_full_eval: args.bf16_full_eval = True; args.fp16_full_eval = False
        if force_float32:
            args.bf16_full_eval = False
            args.fp16_full_eval = False
        elif os.environ.get('UNSLOTH_MIXED_PRECISION', 'float32') == 'bfloat16':
            args.bf16_full_eval = True
            args.fp16_full_eval = False
        elif not bf16_full_eval and not fp16_full_eval:
            args.bf16_full_eval = args.bf16
            args.fp16_full_eval = args.fp16
        _output_logits = False
        if locals().get('compute_metrics', None) is not None: _output_logits = True
        if locals().get('preprocess_logits_for_metrics', None) is not None: _output_logits = True
        if _output_logits:
            os.environ['UNSLOTH_RETURN_LOGITS'] = '1'
        if model is not None:
            _warnings_issued = getattr(model, 'warnings_issued', None)
            if _warnings_issued is None:
                model.warnings_issued = {}
            elif not isinstance(_warnings_issued, dict):
                try:
                    model.warnings_issued = dict(_warnings_issued)
                except Exception:
                    model.warnings_issued = {}
        if 'max_seq_length' not in locals() and not hasattr(args, 'max_seq_length'):
            pass
        else:
            model_max_seq_length = getattr(model, 'max_seq_length', None)
            args_max_seq_length  = getattr(args,  'max_seq_length', None)
            if args_max_seq_length is None and model_max_seq_length is not None:
                max_seq_length = model.max_seq_length
                if hasattr(args, 'max_seq_length'): args.max_seq_length = max_seq_length
            elif args_max_seq_length is not None and model_max_seq_length is not None:
                if args_max_seq_length > model_max_seq_length:
                    print('Unsloth: You set `max_seq_length` as ' + str(args_max_seq_length) + ' but '
                           'the maximum the model supports is ' + str(model_max_seq_length) + '. We shall reduce it.')
                    args.max_seq_length = model_max_seq_length
        if model is not None and hasattr(model, 'for_training'):
            model.for_training(use_gradient_checkpointing=getattr(args, 'gradient_checkpointing', True))
        if 'tokenizer' in locals() and hasattr(tokenizer, 'padding_side'): tokenizer.padding_side = 'right'
        if 'processing_class' in locals():
            if hasattr(processing_class, 'padding_side'): processing_class.padding_side = 'right'
            if hasattr(processing_class, 'tokenizer') and hasattr(processing_class.tokenizer, 'padding_side'): processing_class.tokenizer.padding_side = 'right'
        __tokenizer = processing_class if 'processing_class' in locals() else tokenizer
        from unsloth_zoo.vision_utils import UnslothVisionDataCollator
        if not isinstance(data_collator, UnslothVisionDataCollator):
            if isinstance(data_collator, DataCollatorForSeq2Seq) and 'labels' not in train_dataset.column_names:
                data_collator = TransformersDataCollatorForLanguageModeling(
                    __tokenizer,
                    mlm = False,
                    mlm_probability = 0.0,
                    pad_to_multiple_of = getattr(args, 'pad_to_multiple_of', None),
                )
            elif isinstance(data_collator, TransformersDataCollatorForLanguageModeling) and 'labels' in train_dataset.column_names:
                data_collator = DataCollatorForSeq2Seq(
                    __tokenizer,
                    pad_to_multiple_of = getattr(args, 'pad_to_multiple_of', None),
                )
        else:
            if hasattr(args, 'remove_unused_columns'): args.remove_unused_columns = False
            if hasattr(args, 'dataset_text_field'): args.dataset_text_field = ''
            if hasattr(args, 'dataset_kwargs'): args.dataset_kwargs = {'skip_prepare_dataset': True}
        if not isinstance(data_collator, UnslothVisionDataCollator):
            if not hasattr(__tokenizer, 'pad') and hasattr(__tokenizer, 'tokenizer'):
                if isinstance(data_collator, DataCollatorForSeq2Seq):
                    data_collator = DataCollatorForSeq2Seq(
                        __tokenizer.tokenizer,
                        pad_to_multiple_of = getattr(args, 'pad_to_multiple_of', None),
                    )
                else:
                    data_collator = TransformersDataCollatorForLanguageModeling(
                        __tokenizer.tokenizer,
                        mlm = False,
                        mlm_probability = 0.0,
                        pad_to_multiple_of = getattr(args, 'pad_to_multiple_of', None),
                    )
        other_metrics = []
        
        from unsloth_zoo.logging_utils import PatchRLStatistics
        PatchRLStatistics('online_dpo_trainer', other_metrics)
        
        # [TODO] Fix up DataParallel multiplying batch sizes
        # [TODO] DDP works, but DP seems to not work? [TODO]
        if getattr(args, "parallel_mode", None) == ParallelMode.NOT_DISTRIBUTED and args.n_gpu > 1:
            if getattr(args, "_n_gpu", 1) != 1:
                args._n_gpu = 1
        if "model" in locals() and hasattr(model, "for_training"):
            model.for_training(use_gradient_checkpointing=getattr(args, 'gradient_checkpointing', True))
        super().__init__(
            model = model,
            ref_model = ref_model,
            reward_funcs = reward_funcs,
            judge = judge,
            args = args,
            data_collator = data_collator,
            train_dataset = train_dataset,
            eval_dataset = eval_dataset,
            processing_class = processing_class,
            reward_processing_classes = reward_processing_classes,
            peft_config = peft_config,
            compute_metrics = compute_metrics,
            callbacks = callbacks,
            preprocess_logits_for_metrics = preprocess_logits_for_metrics,
            reward_model = reward_model,
            reward_processing_class = reward_processing_class,**kwargs)
        if "model" in locals() and hasattr(model, "for_inference"):
            model.for_inference()
        if hasattr(self, 'neftune_hook_handle'):
            self.neftune_hook_handle.remove()
            if hasattr(self, 'neftune_hook_handle'): del self.neftune_hook_handle
        if getattr(args, 'neftune_noise_alpha', None) is not None:
            model.get_input_embeddings().neftune_noise_alpha = self.neftune_noise_alpha
        pass
        if hasattr(self, 'accelerator'):
            scaler = self.accelerator.scaler
            current_model = model
            while hasattr(current_model, 'model'):
                current_model.accelerator_scaler = scaler
                current_model = current_model.model
            current_model.accelerator_scaler = scaler
        pass
        if hasattr(self, 'train'):
            self.train = MethodType(prepare_for_training_mode(self.__class__.train), self)
        pass
        if hasattr(self, 'llm') and self.llm is not None and hasattr(self.llm, 'get_tokenizer'):
            _vllm_tok = self.llm.get_tokenizer()
            _pc = getattr(self, 'processing_class', None) or getattr(self, 'tokenizer', None)
            if _vllm_tok is not None and _pc is not None and getattr(_pc, 'chat_template', None) is not None and getattr(_vllm_tok, 'chat_template', None) is None:
                _vllm_tok.chat_template = _pc.chat_template
        pass
        
pass


if hasattr(logger, "addFilter"):
    import logging
    class HideLoggingMessage(logging.Filter):
        def __init__(self, text): self.text = text
        def filter(self, x): return not (self.text in x.getMessage())
    pass
    logger.addFilter(HideLoggingMessage("`use_cache=True`"))

