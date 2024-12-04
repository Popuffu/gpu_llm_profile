import subprocess
import time
import numpy as np


# 获取当前指定ID的GPU功耗
def get_gpu_power(gpu_id_list):
    gpu_id_list_str = str(gpu_id_list).replace("[", "").replace("]", "").replace(" ", "")
    result = subprocess.run(["nvidia-smi", "--query-gpu=power.draw", "--format=csv,noheader,nounits", "-i", gpu_id_list_str], stdout=subprocess.PIPE)
    power_for_all_gpu = result.stdout.decode("utf-8").strip().split("\n")
    assert len(power_for_all_gpu) == len(gpu_id_list)
    total_gpu_power = 0.
    for gpu_id in gpu_id_list:
        total_gpu_power += float(power_for_all_gpu[gpu_id])
    return total_gpu_power


# 定义一个函数来测量并记录GPU功耗
def monitor_gpu_power(gpu_profile_state, gpu_profile_data):
    while gpu_profile_state["running"]:
        power = get_gpu_power(gpu_profile_state["gpu_id_list"])
        gpu_profile_data["time_list"].append(time.time() - gpu_profile_state["start_time"])
        gpu_profile_data["power_list"].append(power)
        gpu_profile_data["flag_list"].append(gpu_profile_state["flag"])
        # time.sleep(gpu_profile_interval)


def get_gpu_avg_power(gpu_profile_data, flag):
    first_flag_time = None
    last_flag_time = None
    for t, p, f in zip(gpu_profile_data["time_list"], gpu_profile_data["power_list"], gpu_profile_data["flag_list"]):
        if f == flag:
            if first_flag_time is None:
                first_flag_time = t
            last_flag_time = t

    # print(gpu_profile_data)
    # 从40%处开始统计到90%，避免一开始的功耗波动
    start_record_time = first_flag_time + (last_flag_time - first_flag_time) * 0.40
    end_record_time = first_flag_time + (last_flag_time - first_flag_time) * 0.90
    record_power_list = list()
    for t, p, f in zip(gpu_profile_data["time_list"], gpu_profile_data["power_list"], gpu_profile_data["flag_list"]):
        if f == flag and t >= start_record_time and t <= end_record_time:
            record_power_list.append(p)
    record_power_list = np.array(record_power_list)
    avg_flag_power = np.mean(record_power_list)
    stderr_flag_power = np.std(record_power_list)
    return avg_flag_power, stderr_flag_power


def write_profile_result(out_csv_dir, write_title_flag, model_nickname, backend, gpu_name, gpu_num, parallel, data_type, batch, input_length, output_length, prefill_latency, decode_latency, decode_throughput, avg_gpu_power):
    exp_result = {
        "Experiment_timestamp": time.strftime('%Y.%m.%d-%H:%M:%S', time.localtime(time.time())),
        "Model_nickname": model_nickname,
        "Backend": backend,
        "GPU_name": gpu_name,
        "GPU_num": gpu_num,
        "Parallel": parallel,
        "Model_data_type": data_type,
        "Batch": batch,
        "In_token": input_length,
        "Out_token": output_length,
        "T_prefill(ms)": prefill_latency,
        "T_decode(ms)": decode_latency if output_length > 1 else "-", # output_length=1时，decode_latency无意义
        "THT_decode(token/s)": decode_throughput if output_length > 1 else "-", # output_length=1时，decode_latency无意义
        "GPU_power(W)": avg_gpu_power,
    }
    print(str(exp_result).replace("{", "----------------------------------------------------\n").replace("}", "\n----------------------------------------------------").replace(", ", "\n"))

    result_str = ""
    if write_title_flag:
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

    with open(out_csv_dir, "a") as f:
        f.write(result_str)
