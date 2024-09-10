import subprocess
import time
import numpy as np

# 定义一个函数来获取当前的GPU功耗
def get_gpu_power(gpu_id_list):
    result = subprocess.run(["nvidia-smi", "--query-gpu=power.draw", "--format=csv,noheader,nounits"], stdout=subprocess.PIPE)
    power_for_all_gpu = result.stdout.decode("utf-8").strip().split("\n")
    assert len(power_for_all_gpu) == 8 # total GPU number
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
