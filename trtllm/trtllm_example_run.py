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

from profile_config import GPU_ID_LIST, GPU_NUM, PARALLEL_NAME, BACKEND, MODEL_NICKNAME, WARMUP, TESTFREQ, PROFILE_CFG, TRT_ENGINE_DIR, HF_MODEL_DIR, DATA_TYPE, GPU_NAME, PROFILE_RESULT_DIR
assert BACKEND == "trtllm"
## Import hyper params ##

import os
import time
GPU_PROFILE_STATE = {
    "gpu_id_list": GPU_ID_LIST, # PROFILE GPU ID
    "start_time": time.time(),
    "running": None,
    "flag": "", # 用来打标记的
}


import argparse
import ast
import os

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
import threading

if PYTHON_BINDINGS:
    from tensorrt_llm.runtime import ModelRunnerCpp

from utils import utils

def main(args, profile_cfg_list):
    runtime_rank = tensorrt_llm.mpi_rank()
    logger.set_level(args.log_level)
    torch.cuda.empty_cache()
    
    model_name, model_version = read_model_name(args.engine_dir)

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


    stop_words_list = None
    if args.stop_words:
        stop_words_list = tensorrt_llm.runtime.decode_words_list(
            args.stop_words, tokenizer)
    if model_version == 'glm4':  # add default stop token ids for GLM-4
        raise ValueError
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

    np_profile_cfg_list = np.array(profile_cfg_list)
    profile_max_batch_size = max(np_profile_cfg_list[:, 0])
    profile_max_input_len = max(np_profile_cfg_list[:, 1])
    profile_max_output_len = max(np_profile_cfg_list[:, 2])
    if not args.use_py_session and not supports_inflight_batching(args.engine_dir):
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
        max_output_len=profile_max_output_len,
    )
    if not args.use_py_session:
        runner_kwargs.update(is_enc_dec=False)
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
            max_batch_size=profile_max_batch_size,
            max_input_len=profile_max_input_len,
            max_beam_width=args.num_beams,
            max_attention_window_size=args.max_attention_window_size,
            sink_token_length=args.sink_token_length,
            max_tokens_in_paged_kv_cache=args.max_tokens_in_paged_kv_cache,
            kv_cache_enable_block_reuse=args.kv_cache_enable_block_reuse,
            kv_cache_free_gpu_memory_fraction=args.kv_cache_free_gpu_memory_fraction,
            enable_chunked_context=args.enable_chunked_context,
            multi_block_mode=args.multi_block_mode)
    runner_kwargs.update(
        enable_context_fmha_fp32_acc=args.enable_context_fmha_fp32_acc)
    runner = runner_cls.from_dir(**runner_kwargs)

    with torch.no_grad():
        for exp_id, (BATCH, INPUT_LENGTH, OUTPUT_LENGTH) in enumerate(profile_cfg_list):

            runner.max_batch_size = BATCH
            runner.max_input_len = INPUT_LENGTH
            runner.max_seq_len = INPUT_LENGTH + OUTPUT_LENGTH

            batch_input_ids = [[666] * INPUT_LENGTH] * BATCH
            batch_input_ids = [torch.tensor(x, dtype=torch.int32) for x in batch_input_ids]

            torch.cuda.synchronize()
            if runtime_rank == 0:
                GPU_PROFILE_STATE["running"] = True
                gpu_profile_data = {
                    "time_list": list(), 
                    "power_list": list(), 
                    "flag_list": list(), 
                }
                # 启动一个线程来监控GPU功耗
                monitor_thread = threading.Thread(target=utils.monitor_gpu_power, args=(GPU_PROFILE_STATE, gpu_profile_data))
                monitor_thread.start()
            
                st = torch.cuda.Event(enable_timing=True)
                ed = torch.cuda.Event(enable_timing=True)

            if runtime_rank == 0:
                GPU_PROFILE_STATE["flag"] = "warmup"

            for _ in range(WARMUP):
                outputs = runner.generate(
                    batch_input_ids=batch_input_ids,
                    encoder_input_ids=None,
                    encoder_input_features=None,
                    encoder_output_lengths=None,
                    max_new_tokens=1, # just prefill
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
                GPU_PROFILE_STATE["flag"] = ""

            torch.cuda.synchronize()
            if runtime_rank == 0:
                print("Start prefill, ", time.strftime('%Y.%m.%d-%H:%M:%S', time.localtime(time.time())))
                st.synchronize()
                st.record()
                GPU_PROFILE_STATE["flag"] = "prefill"
            for _ in range(TESTFREQ):
                outputs = runner.generate(
                    batch_input_ids=batch_input_ids,
                    encoder_input_ids=None,
                    encoder_input_features=None,
                    encoder_output_lengths=None,
                    max_new_tokens=1, # just prefill
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
                ed.record()
                ed.synchronize()
                GPU_PROFILE_STATE["flag"] = ""
                prefill_latency = st.elapsed_time(ed) / TESTFREQ

            torch.cuda.synchronize()
            if runtime_rank == 0:
                print("Start decode, ", time.strftime('%Y.%m.%d-%H:%M:%S', time.localtime(time.time())))
                st.synchronize()
                st.record()
                GPU_PROFILE_STATE["flag"] = "decode"

            for _ in range(TESTFREQ):
                outputs = runner.generate(
                    batch_input_ids=batch_input_ids,
                    encoder_input_ids=None,
                    encoder_input_features=None,
                    encoder_output_lengths=None,
                    max_new_tokens=OUTPUT_LENGTH, # prefill+decode
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
                ed.record()
                ed.synchronize()
                GPU_PROFILE_STATE["flag"] = ""
                print("End decode, ", time.strftime('%Y.%m.%d-%H:%M:%S', time.localtime(time.time())))
                total_latency = st.elapsed_time(ed) / TESTFREQ

                decode_latency = total_latency - prefill_latency

                # 停止监控线程
                GPU_PROFILE_STATE["running"] = False
                monitor_thread.join()

                assert int(outputs['output_ids'].shape[0]) == BATCH
                assert int(outputs['output_ids'].shape[1]) == 1
                assert int(outputs['output_ids'].shape[2]) == INPUT_LENGTH + OUTPUT_LENGTH, f"{outputs['output_ids'].shape, INPUT_LENGTH + OUTPUT_LENGTH}"

                # 计算平均功耗，注意被打上decode flag的阶段，既包含prefill，又包含decode。因此当output_length=1时，测出其实就是prefill的功耗
                avg_gpu_power, stderr_gpu_power = utils.get_gpu_avg_power(gpu_profile_data, "decode")

                decode_token_output_latency = decode_latency / OUTPUT_LENGTH
                decode_tokens_per_second = (1000 / decode_token_output_latency) * BATCH

                total_token_output_latency = total_latency / (OUTPUT_LENGTH + 1)
                total_tokens_per_second = (1000 / total_token_output_latency) * BATCH

                exp_result = {
                    "experiment_timestamp": time.strftime('%Y.%m.%d-%H:%M:%S', time.localtime(time.time())),
                    "model_nickname": MODEL_NICKNAME,
                    "backend": BACKEND,
                    "gpu_name": GPU_NAME,
                    "gpu_num": GPU_NUM,
                    "parallel": PARALLEL_NAME,
                    "data_type": DATA_TYPE,
                    "batch": BATCH,
                    "input_length": INPUT_LENGTH,
                    "output_length": OUTPUT_LENGTH,
                    "prefill_latency(ms)": prefill_latency,
                    "decode_latency(ms)": decode_latency if OUTPUT_LENGTH > 1 else "-", # output_length=1时，decode_latency无意义
                    # "total_latency(ms)": total_latency,
                    # "total_throughput(token/s)": total_tokens_per_second,
                    "decode_throughput(tokens/s)": decode_tokens_per_second if OUTPUT_LENGTH > 1 else "-", # output_length=1时，decode_latency无意义
                    "gpu_power_avg(W)": avg_gpu_power,
                    "gpu_power_std(W)": stderr_gpu_power,
                }
                print(str(exp_result).replace("{", "----------------------------------------------------\n").replace("}", "\n----------------------------------------------------").replace(", ", "\n"))

                result_str = ""
                if exp_id == 0:
                    title_str = ""
                    for title_key in exp_result.keys():
                        title_str += title_key + ", "
                    title_str = title_str.rstrip(", ")
                    title_str += "\n"
                    result_str += title_str

                for key, value in exp_result.items():
                    if isinstance(value, float):
                        result_str += "{:.3f}".format(value).ljust(len(key)) + ", "
                    else:
                        result_str += str(value).ljust(len(key)) + ", "
                result_str = result_str.rstrip(", ")
                result_str += "\n"

                with open(PROFILE_RESULT_DIR, "a") as f:
                    f.write(result_str)
            else:
                pass



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser = add_common_args(parser)
    args = parser.parse_args(args=["--end_id", "-100", "--engine_dir", TRT_ENGINE_DIR, "--tokenizer_dir", HF_MODEL_DIR])

    main(args, PROFILE_CFG)
