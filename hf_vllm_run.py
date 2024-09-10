GPU_ID_LIST = [0, 1, 2, 3, 4, 5, 6, 7]
MODEL_NICKNAME = "llama3_70b"
BACKEND = "vllm"
PROFILE_CFG = [
    # # batch, input_length, output_length
    # (1, 32, 8),

    (1, 128, 8192),
    (1, 128, 16384),
    (1, 128, 32768),
    (1, 128, 65536),
    (2, 128, 8192),
    (2, 128, 16384),
    (2, 128, 32768),
    (2, 128, 65536),
    (4, 128, 8192),
    (4, 128, 16384),
    (4, 128, 32768),
    (4, 128, 65536),
    (8, 128, 8192),
    (8, 128, 16384),
    (8, 128, 32768),
    (8, 128, 65536),
    # (16, 128, 128), # OOM

    (1, 256, 8192),
    (1, 256, 16384),
    (1, 256, 32768),
    (1, 256, 65536),
    (2, 256, 8192),
    (2, 256, 16384),
    (2, 256, 32768),
    (2, 256, 65536),
    (4, 256, 8192),
    (4, 256, 16384),
    (4, 256, 32768),
    (4, 256, 65536),
    # (8, 256, 128), # OOM

    (1, 512, 8192),
    (1, 512, 16384),
    (1, 512, 32768),
    (1, 512, 65536),
    (2, 512, 8192),
    (2, 512, 16384),
    (2, 512, 32768),
    (2, 512, 65536),
    # (4, 512, 128), # OOM

    (1, 1024, 8192),
    (1, 1024, 16384),
    (1, 1024, 32768),
    (1, 1024, 65536),
    # (2, 1024, 128), # OOM
]

    # (1, 128, 128),
    # (1, 128, 256),
    # (1, 128, 512),
    # (1, 128, 1024),
    # (1, 128, 2048),
    # (1, 128, 4096),
    # (2, 128, 128),
    # (2, 128, 256),
    # (2, 128, 512),
    # (2, 128, 1024),
    # (2, 128, 2048),
    # (2, 128, 4096),
    # (4, 128, 128),
    # (4, 128, 256),
    # (4, 128, 512),
    # (4, 128, 1024),
    # (4, 128, 2048),
    # (4, 128, 4096),
    # (8, 128, 128),
    # (8, 128, 256),
    # (8, 128, 512),
    # (8, 128, 1024),
    # (8, 128, 2048),
    # (8, 128, 4096),
    # (16, 128, 128), # OOM

    # (1, 256, 128),
    # (1, 256, 256),
    # (1, 256, 512),
    # (1, 256, 1024)
    # (1, 256, 2048),
    # (1, 256, 4096),
    # (2, 256, 128),
    # (2, 256, 256),
    # (2, 256, 512),
    # (2, 256, 1024),
    # (2, 256, 2048),
    # (2, 256, 4096),
    # (4, 256, 128),
    # (4, 256, 256),
    # (4, 256, 512),
    # (4, 256, 1024),
    # (4, 256, 2048),
    # (4, 256, 4096),
    # (8, 256, 128), # OOM

    # (1, 512, 128),
    # (1, 512, 256),
    # (1, 512, 512),
    # (1, 512, 1024),
    # (1, 512, 2048),
    # (1, 512, 4096),
    # (2, 512, 128),
    # (2, 512, 256),
    # (2, 512, 512),
    # (2, 512, 1024),
    # (2, 512, 2048),
    # (2, 512, 4096),
    # (4, 512, 128), # OOM

    # (1, 1024, 128),
    # (1, 1024, 256),
    # (1, 1024, 512),
    # (1, 1024, 1024),
    # (1, 1024, 2048),
    # (1, 1024, 4096),
    # (2, 1024, 128), # OOM

WARMUP, TESTFREQ = 4, 10

## Hyper Param Above ##


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
import subprocess
import threading
import numpy as np


# 定义一个函数来获取当前的GPU功耗
def get_gpu_power():
    result = subprocess.run(["nvidia-smi", "--query-gpu=power.draw", "--format=csv,noheader,nounits"], stdout=subprocess.PIPE)
    power_for_all_gpu = result.stdout.decode("utf-8").strip().split("\n")
    assert len(power_for_all_gpu) == 8 # total GPU number
    total_gpu_power = 0.
    for gpu_id in GPU_PROFILE_STATE["gpu_id_list"]:
        total_gpu_power += float(power_for_all_gpu[gpu_id])
    return total_gpu_power


# 定义一个函数来测量并记录GPU功耗
def monitor_gpu_power(gpu_profile_state, gpu_profile_data):
    while gpu_profile_state["running"]:
        power = get_gpu_power()
        gpu_profile_data["time_list"].append(time.time() - gpu_profile_state["start_time"])
        gpu_profile_data["power_list"].append(power)
        gpu_profile_data["flag_list"].append(gpu_profile_state["flag"])
        # time.sleep(gpu_profile_interval)


def get_decode_avg_power(gpu_profile_data):
    first_decode_time = None
    last_decode_time = None
    for t, p, f in zip(gpu_profile_data["time_list"], gpu_profile_data["power_list"], gpu_profile_data["flag_list"]):
        if f == "decode":
            if first_decode_time is None:
                first_decode_time = t
            last_decode_time = t

    # print(gpu_profile_data)
    # 从40%处开始统计到90%，避免一开始的功耗波动
    start_record_time = first_decode_time + (last_decode_time - first_decode_time) * 0.40
    end_record_time = first_decode_time + (last_decode_time - first_decode_time) * 0.90
    record_power_list = list()
    for t, p, f in zip(gpu_profile_data["time_list"], gpu_profile_data["power_list"], gpu_profile_data["flag_list"]):
        if f == "decode" and t >= start_record_time and t <= end_record_time:
            record_power_list.append(p)
    record_power_list = np.array(record_power_list)
    avg_decode_power = np.mean(record_power_list)
    stderr_decode_power = np.std(record_power_list)
    return avg_decode_power, stderr_decode_power



def profile(model_nickname, model, backend, batch, input_length, output_length):
    GPU_PROFILE_STATE["running"] = True
    gpu_profile_data = {
        "time_list": list(), 
        "power_list": list(), 
        "flag_list": list(), 
    }
    # 启动一个线程来监控GPU功耗
    monitor_thread = threading.Thread(target=monitor_gpu_power, args=(GPU_PROFILE_STATE, gpu_profile_data))
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

    # 计算平均功耗
    avg_decode_power, stderr_decode_power = get_decode_avg_power(gpu_profile_data)

    return prefill_latency, decode_latency, total_latency, float(avg_decode_power), float(stderr_decode_power)


if __name__ == "__main__":

    if MODEL_NICKNAME == "llama3_8b":
        model_dir = "/mnt/public/Meta-Llama-3-8B-Instruct"
    elif MODEL_NICKNAME == "llama3_70b":
        model_dir = "/mnt/public/Meta-Llama-3-70B-Instruct-hf"
    else:
        raise ValueError
    
    # tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code = True)
    if BACKEND == "hf":
        model = AutoModelForCausalLM.from_pretrained(model_dir, trust_remote_code = True, resume_download = True, device_map="auto").half()
    elif BACKEND == "vllm":
        model = LLM(model=model_dir, tensor_parallel_size=len(GPU_ID_LIST))
    else:
        raise ValueError

    for exp_id, (batch, input_length, output_length) in enumerate(PROFILE_CFG):
        prefill_latency, decode_latency, total_latency, avg_decode_power, stderr_decode_power = profile(MODEL_NICKNAME, model, BACKEND, batch, input_length, output_length)

        decode_token_output_latency = decode_latency / output_length
        decode_tokens_per_second = (1000 / decode_token_output_latency) * batch

        total_token_output_latency = total_latency / (output_length + 1)
        total_tokens_per_second = (1000 / total_token_output_latency) * batch

        exp_result = {
            "experiment_timestamp": time.strftime('%Y.%m.%d-%H:%M:%S', time.localtime(time.time())),
            "model_nickname": MODEL_NICKNAME,
            "backend": BACKEND,
            "gpu_num": len(GPU_ID_LIST),
            "batch": batch,
            "input_length": input_length,
            "output_length": output_length,
            "prefill_latency(ms)": prefill_latency,
            "decode_latency(ms)": decode_latency,
            "total_latency(ms)": total_latency,
            "total_throughput(token/s)": total_tokens_per_second,
            "decode_throughput(tokens/s)": decode_tokens_per_second,
            "decode_power_avg(W)": avg_decode_power,
            "decode_power_stderr(W)": stderr_decode_power,
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

        with open(f"profile_result.csv", "a") as f:
            f.write(result_str)

    
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
