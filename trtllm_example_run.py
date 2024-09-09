# SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import ast
import csv
import os
from pathlib import Path

import numpy as np
import torch
from trtllm_example_utils import (DEFAULT_HF_MODEL_DIRS, DEFAULT_PROMPT_TEMPLATES,
                   add_common_args, load_tokenizer, prepare_enc_dec_inputs,
                   read_model_name, supports_inflight_batching,
                   throttle_generator)

import tensorrt_llm
import tensorrt_llm.profiler
from tensorrt_llm.logger import logger
from tensorrt_llm.runtime import PYTHON_BINDINGS, ModelRunner
from tensorrt_llm import LLM

if PYTHON_BINDINGS:
    from tensorrt_llm.runtime import ModelRunnerCpp


def main(args, BATCH, INPUT_LENGTH, OUTPUT_LENGTH):
    runtime_rank = tensorrt_llm.mpi_rank()
    logger.set_level(args.log_level)

    # different handling if encoder-decoder models
    is_enc_dec = {'encoder', 'decoder'}.issubset({
        name
        for name in os.listdir(args.engine_dir)
        if os.path.isdir(os.path.join(args.engine_dir, name))
    })
    if is_enc_dec:
        logger.warning(
            "This path is an encoder-decoder model. Using different handling.")
        assert not args.use_py_session, "Encoder-decoder models don't have a unified python runtime, please use its own examples/enc_dec/run.py instead."

    model_name, model_version = read_model_name(
        args.engine_dir if not is_enc_dec else os.path.
        join(args.engine_dir, 'encoder'))

    if args.tokenizer_dir is None and model_name in DEFAULT_HF_MODEL_DIRS:
        logger.warning(
            "tokenizer_dir is not specified. Try to infer from model_name, but this may be incorrect."
        )
        args.tokenizer_dir = DEFAULT_HF_MODEL_DIRS[model_name]

    tokenizer, pad_id, end_id = load_tokenizer(
        tokenizer_dir=args.tokenizer_dir,
        vocab_file=args.vocab_file,
        model_name=model_name,
        model_version=model_version,
        tokenizer_type=args.tokenizer_type,
    )

    if args.end_id:
        end_id = args.end_id

    prompt_template = None
    if args.use_prompt_template and model_name in DEFAULT_PROMPT_TEMPLATES:
        prompt_template = DEFAULT_PROMPT_TEMPLATES[model_name]

    batch_input_ids = [[666] * INPUT_LENGTH] * BATCH
    batch_input_ids = [
        torch.tensor(x, dtype=torch.int32) for x in batch_input_ids
    ]

    stop_words_list = None
    if args.stop_words:
        stop_words_list = tensorrt_llm.runtime.decode_words_list(
            args.stop_words, tokenizer)
    if model_version == 'glm4':  # add default stop token ids for GLM-4
        glm4_stop_ids = [[151329], [151336], [151338]]
        if stop_words_list is None:
            stop_words_list = [glm4_stop_ids] * len(batch_input_ids)
        else:
            for req_stop_words_list in stop_words_list:
                req_stop_words_list.extend(glm4_stop_ids)

    bad_words_list = None
    if args.bad_words:
        bad_words_list = tensorrt_llm.runtime.decode_words_list(
            args.bad_words, tokenizer)

    if is_enc_dec:
        encoder_input_ids, encoder_input_features, encoder_output_lengths, decoder_input_ids = prepare_enc_dec_inputs(
            batch_input_ids, model_name, args.engine_dir,
            args.multimodal_input_file)

    input_lengths = [x.size(0) for x in decoder_input_ids
                     ] if is_enc_dec else [x.size(0) for x in batch_input_ids]

    encoder_input_lengths = [
        x.size(0) for x in (encoder_input_features or encoder_input_ids)
    ] if is_enc_dec else None

    if not args.use_py_session and not supports_inflight_batching(
            os.path.join(args.engine_dir, "decoder") if is_enc_dec else args.
            engine_dir):
        logger.warning(
            "The given engine does not support in-flight batching, fallback to python session"
        )
        args.use_py_session = True

    if not PYTHON_BINDINGS and not args.use_py_session:
        logger.warning(
            "Python bindings of C++ session is unavailable, fallback to Python session."
        )
        args.use_py_session = True
    if args.debug_mode and not args.use_py_session:
        logger.warning(
            "Debug mode is not supported in C++ session for now, fallback to Python session."
        )
        args.use_py_session = True
    if args.return_all_generated_tokens and args.use_py_session:
        raise ValueError(
            "Returning all the generated tokens at each step is not supported in the Python session, use C++ session instead."
        )
    if (not args.return_all_generated_tokens) and args.streaming and (
            args.num_beams > 1):
        logger.warning(
            "Setting return_all_generated_tokens to True since streaming AND beam search are done simultaneously. "
            "Returning the full beams at each streaming step is needed because beam search + streaming can change previous outputs. "
            "WARNING: using this option may increase network usage significantly (quadratically w.r.t output length)."
        )
        args.return_all_generated_tokens = True
    runner_cls = ModelRunner if args.use_py_session else ModelRunnerCpp
    runner_kwargs = dict(
        engine_dir=args.engine_dir,
        lora_dir=args.lora_dir,
        rank=runtime_rank,
        debug_mode=args.debug_mode,
        lora_ckpt_source=args.lora_ckpt_source,
        gpu_weights_percent=args.gpu_weights_percent,
        max_output_len=OUTPUT_LENGTH,
    )
    if not args.use_py_session:
        runner_kwargs.update(is_enc_dec=is_enc_dec)
    if args.medusa_choices is not None:
        args.medusa_choices = ast.literal_eval(args.medusa_choices)
        assert args.temperature == 1.0, "Medusa should use temperature == 1.0"
        assert args.num_beams == 1, "Medusa should use num_beams == 1"
        runner_kwargs.update(medusa_choices=args.medusa_choices)
    if args.lookahead_config is not None:
        args.lookahead_config = ast.literal_eval(args.lookahead_config)
        assert len(
            args.lookahead_config
        ) == 3, "Lookahead needs [max_window_size, max_ngram_size, max_verification_set_size]"
        runner_kwargs.update(lookahead_config=args.lookahead_config)
    if not args.use_py_session:
        runner_kwargs.update(
            max_batch_size=len(batch_input_ids),
            max_input_len=max(
                encoder_input_lengths if is_enc_dec else input_lengths),
            max_beam_width=args.num_beams,
            max_attention_window_size=args.max_attention_window_size,
            sink_token_length=args.sink_token_length,
            max_tokens_in_paged_kv_cache=args.max_tokens_in_paged_kv_cache,
            kv_cache_enable_block_reuse=args.kv_cache_enable_block_reuse,
            kv_cache_free_gpu_memory_fraction=args.
            kv_cache_free_gpu_memory_fraction,
            enable_chunked_context=args.enable_chunked_context,
            multi_block_mode=args.multi_block_mode)
    runner_kwargs.update(
        enable_context_fmha_fp32_acc=args.enable_context_fmha_fp32_acc)
    runner = runner_cls.from_dir(**runner_kwargs)

    with torch.no_grad():
        outputs = runner.generate(
            batch_input_ids=decoder_input_ids
            if is_enc_dec else batch_input_ids,
            encoder_input_ids=encoder_input_ids if is_enc_dec else None,
            encoder_input_features=encoder_input_features
            if is_enc_dec else None,
            encoder_output_lengths=encoder_output_lengths
            if is_enc_dec else None,
            max_new_tokens=OUTPUT_LENGTH,
            max_attention_window_size=args.max_attention_window_size,
            sink_token_length=args.sink_token_length,
            end_id=end_id,
            pad_id=pad_id,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            num_beams=args.num_beams,
            length_penalty=args.length_penalty,
            early_stopping=args.early_stopping,
            repetition_penalty=args.repetition_penalty,
            presence_penalty=args.presence_penalty,
            frequency_penalty=args.frequency_penalty,
            stop_words_list=stop_words_list,
            bad_words_list=bad_words_list,
            output_cum_log_probs=(None != None),
            output_log_probs=(None != None),
            random_seed=args.random_seed,
            lora_uids=args.lora_task_uids,
            prompt_table=args.prompt_table_path,
            prompt_tasks=args.prompt_tasks,
            streaming=args.streaming,
            output_sequence_lengths=True,
            no_repeat_ngram_size=args.no_repeat_ngram_size,
            return_dict=True,
            medusa_choices=args.medusa_choices,
            return_all_generated_tokens=args.return_all_generated_tokens)
        torch.cuda.synchronize()
        if runtime_rank == 0:
            assert int(outputs['output_ids'].shape[0]) == BATCH
            assert int(outputs['output_ids'].shape[1]) == 1
            assert int(outputs['output_ids'].shape[2]) == INPUT_LENGTH + OUTPUT_LENGTH
            return outputs


if __name__ == '__main__':
    trt_engine_dir = "/mnt/public/trt_models/Meta-Llama-3-8B-Instruct-engine"
    # trt_engine_dir = "/mnt/public/trt_models/Meta-Llama-3-8B-Instruct-1gpu-engine"
    hf_model_dir = "/mnt/public/Meta-Llama-3-8B-Instruct"

    parser = argparse.ArgumentParser()
    parser = add_common_args(parser)
    args = parser.parse_args(args=["--end_id", "-100", "--engine_dir", trt_engine_dir, "--tokenizer_dir", hf_model_dir])

    outputs = main(args, 3, 128, 128)
