
from profile_config import GPU_ID_LIST, GPU_NUM, PARALLEL_NAME, BACKEND, MODEL_NICKNAME, WARMUP, TESTFREQ, PROFILE_CFG, TRT_ENGINE_DIR, HF_MODEL_DIR, DATA_TYPE, GPU_NAME, PROFILE_RESULT_DIR
assert BACKEND in ("hf", "vllm")
## Import hyper params ##

import os
import time
GPU_PROFILE_STATE = {
    "gpu_id_list": GPU_ID_LIST, # PROFILE GPU ID
    "start_time": time.time(),
    "running": None,
    "flag": "", # 用来打标记的
}
os.environ["CUDA_VISIBLE_DEVICES"] = str(GPU_ID_LIST).replace("[", "").replace("]", "").replace(" ", "")
print(f"CUDA_VISIBLE_DEVICES = {os.environ['CUDA_VISIBLE_DEVICES']}")

import torch
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer, AutoModelForCausalLM
import threading
from utils import utils


def profile(model_nickname, model, backend, batch, input_length, output_length):
    GPU_PROFILE_STATE["running"] = True
    gpu_profile_data = {
        "time_list": list(), 
        "power_list": list(), 
        "flag_list": list(), 
    }
    # 启动一个线程来监控GPU功耗
    monitor_thread = threading.Thread(target=utils.monitor_gpu_power, args=(GPU_PROFILE_STATE, gpu_profile_data))
    monitor_thread.start()


    torch.cuda.empty_cache()

    device = torch.device("cuda")
    
    st = torch.cuda.Event(enable_timing=True)
    ed = torch.cuda.Event(enable_timing=True)

    torch.cuda.empty_cache()
    torch.cuda.nvtx.range_push("{} batch {} input_length {} output_length {}".format(model_nickname, batch, input_length, output_length))
    # prepare inputs
    input_ids = [[666] * input_length] * batch
    input_ids_t = torch.tensor(input_ids)

    # params
    if backend == "vllm":
        ignore_eos = True
        sampling_params_query = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=1, ignore_eos=ignore_eos)
        sampling_params_total = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=output_length+1, ignore_eos=ignore_eos)

    # warm up
    GPU_PROFILE_STATE["flag"] = "warmup"
    for _ in range(WARMUP):
        if backend == "hf":
            logits = model.generate(input_ids_t, num_beams=1, max_length=input_length+1, use_cache=True)
        elif backend == "vllm":
            outputs = model.generate(prompt_token_ids=input_ids, sampling_params=sampling_params_query, use_tqdm=False)
        else:
            raise ValueError
    GPU_PROFILE_STATE["flag"] = ""

    # for output in outputs:
    #     prompt = output.prompt
    #     generated_text = output.outputs[0].text
    #     print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")

    st.synchronize()
    st.record()
    GPU_PROFILE_STATE["flag"] = "prefill"
    for _ in range(TESTFREQ):
        if backend == "hf":
            logits = model.generate(input_ids_t, num_beams=1, max_length=input_length+1, use_cache=True)
        elif backend == "vllm":
            outputs = model.generate(prompt_token_ids=input_ids, sampling_params=sampling_params_query, use_tqdm=False)
        else:
            raise ValueError
    ed.record()
    ed.synchronize()
    GPU_PROFILE_STATE["flag"] = ""
    prefill_latency = st.elapsed_time(ed) / TESTFREQ

    st.synchronize()
    st.record()
    GPU_PROFILE_STATE["flag"] = "decode"
    for _ in range(TESTFREQ):
        if backend == "hf":
            logits = model.generate(input_ids_t, num_beams=1, max_length=input_length+output_length+1, use_cache=True)
        elif backend == "vllm":
            outputs = model.generate(prompt_token_ids=input_ids, sampling_params=sampling_params_total, use_tqdm=False) # model is llm
        else:
            raise ValueError
    ed.record()
    ed.synchronize()
    GPU_PROFILE_STATE["flag"] = ""
    total_latency = st.elapsed_time(ed) / TESTFREQ
    torch.cuda.nvtx.range_pop()

    decode_latency = total_latency - prefill_latency

    # 停止监控线程
    GPU_PROFILE_STATE["running"] = False
    monitor_thread.join()

    # 计算平均功耗，注意被打上decode flag的阶段，既包含prefill，又包含decode。因此当output_length=1时，测出其实就是prefill的功耗
    avg_gpu_power, stderr_gpu_power = utils.get_gpu_avg_power(gpu_profile_data, "decode")

    return prefill_latency, decode_latency, total_latency, float(avg_gpu_power), float(stderr_gpu_power)


if __name__ == "__main__":

    # tokenizer = AutoTokenizer.from_pretrained(HF_MODEL_DIR, trust_remote_code = True)
    if BACKEND == "hf":
        model = AutoModelForCausalLM.from_pretrained(HF_MODEL_DIR, trust_remote_code = True, resume_download = True, device_map="auto").half()
    elif BACKEND == "vllm":
        model = LLM(model=HF_MODEL_DIR, tensor_parallel_size=len(GPU_ID_LIST))
    else:
        raise ValueError

    for exp_id, (BATCH, INPUT_LENGTH, OUTPUT_LENGTH) in enumerate(PROFILE_CFG):
        prefill_latency, decode_latency, total_latency, avg_gpu_power, stderr_gpu_power = profile(MODEL_NICKNAME, model, BACKEND, BATCH, INPUT_LENGTH, OUTPUT_LENGTH)

        decode_token_output_latency = decode_latency / OUTPUT_LENGTH
        decode_tokens_per_second = (1000 / decode_token_output_latency) * BATCH

        total_token_output_latency = total_latency / (OUTPUT_LENGTH + 1)
        total_tokens_per_second = (1000 / total_token_output_latency) * BATCH

        utils.write_profile_result(
            out_csv_dir = PROFILE_RESULT_DIR, 
            write_title_flag = exp_id == 0, 
            model_nickname = MODEL_NICKNAME, 
            backend = BACKEND, 
            gpu_name = GPU_NAME, 
            gpu_num = GPU_NUM, 
            parallel = PARALLEL_NAME, 
            data_type = DATA_TYPE, 
            batch = BATCH, 
            input_length = INPUT_LENGTH, 
            output_length = OUTPUT_LENGTH, 
            prefill_latency = prefill_latency,
            decode_latency = decode_latency, 
            decode_throughput = decode_tokens_per_second, 
            avg_gpu_power = avg_gpu_power,
        )

    
    """
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        on_trace_ready=torch.profiler.tensorboard_trace_handler("../profile_output/"),
    ) as profiler:
        # main("/share/datasets/public_models/Llama-2-13b-hf/", "hf")
        main("/share/datasets/public_models/Llama-2-13b-hf/", "vllm")
    """
