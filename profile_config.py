
GPU_ID_LIST = [0, 1, 2, 3, 4, 5, 6, 7]
BACKEND = "trtllm"
MODEL_NICKNAME = "llama3.1_70b"
DATA_TYPE = "FP16"

GPU_NUM = len(GPU_ID_LIST)
TP_SIZE = GPU_NUM
PP_SIZE = 1

WARMUP = 1
TESTFREQ = 4

PROFILE_CFG = [(1, 1, 1024),]

assert BACKEND in ("hf", "vllm", "trtllm")
assert DATA_TYPE in ("FP16", "FP8", "W4A16KV8G128", "W8A8smooth") # FP16默认TP8

PARALLEL_NAME = f"TP{TP_SIZE}PP{PP_SIZE}"

# # example: [(batch, input_length, output_length), ...]
# for output_length in [1024, 1024+256]:#[128, 256, 512, 1024]:
#     batch_list = [256]
#     for batch in batch_list:
#         PROFILE_CFG.append((batch, 1, output_length))

for output_length in [2, 1024, 1024+256]:#[128, 256, 512, 1024]:
    batch_list = [256]
    for batch in batch_list:
        PROFILE_CFG.append((batch, 1, output_length))

# for output_length in [128,256,384,512,640,768,1024,1152,1280,1536,2048,]:
#     batch_list = [256]
#     for batch in batch_list: # 
#         PROFILE_CFG.append((batch, 1, output_length))


# 获取当前指定ID的GPU型号
def get_gpu_short_name(gpu_id_list):
    import subprocess
    import re
    gpu_id_list_str = str(gpu_id_list).replace("[", "").replace("]", "").replace(" ", "")
    result = subprocess.run(["nvidia-smi", "--query-gpu=gpu_name", "--format=csv,noheader", "-i", gpu_id_list_str], stdout=subprocess.PIPE)
    name_for_all_gpu = result.stdout.decode("utf-8").strip().split("\n")
    assert len(name_for_all_gpu) == len(gpu_id_list)
    assert all(x == name_for_all_gpu[0] for x in name_for_all_gpu) # 这些ID的GPU型号应该一致
    # Use regex to match relevant patterns, shorten full gpu names
    match = re.match(r'NVIDIA\s*(A\d{3,4}|H\d{3,4}|GeForce\s*RTX\s*\d{3,4})', name_for_all_gpu[0])
    assert match
    gpu_name = match.group(1).replace('GeForce RTX ', 'RTX').replace(" ", "")
    return gpu_name

GPU_NAME = get_gpu_short_name(GPU_ID_LIST) # 根据nvidia-smi自动获取GPU型号一致
assert GPU_NAME in ("RTX3090", "RTX4090", "A100", "A800", "H100", "H800")


if MODEL_NICKNAME == "llama3_8b":
    MODEL_NAME = "Meta-Llama-3-8B-Instruct"
elif MODEL_NICKNAME == "llama3.1_70b":
    MODEL_NAME = "Meta-Llama-3.1-70B-Instruct"
else:
    raise ValueError

PROFILE_RESULT_DIR = f"{MODEL_NICKNAME}_{GPU_NAME}_fccm_profile.csv"

TRTLLM_EXAMPLE_CODE_DIR = "/mnt/public/yangxinhao/TensorRT-LLM/examples"
PYTHON_CODE_DIR = "/mnt/public/yangxinhao/gpu_llm_profile"
HF_MODEL_DIR = f"/mnt/resource/public_models/{MODEL_NAME}"
TRT_BASE_DIR = f"/mnt/volume/yangxinhao/{MODEL_NAME}-{len(GPU_ID_LIST)}x{GPU_NAME}-{DATA_TYPE}-{PARALLEL_NAME}"

TRT_CKPT_DIR = f"{TRT_BASE_DIR}-ckpt"
TRT_ENGINE_DIR = f"{TRT_BASE_DIR}-engine"

if __name__ == "__main__":
    # DO NOT CHANGE THIS PRINT!
    # for the bash script to directly call this file to get the BACKEND and GPU_NUM variable!
    print(f"{BACKEND} {GPU_NUM} {GPU_NAME} {DATA_TYPE} {MODEL_NAME} {HF_MODEL_DIR} {TRT_CKPT_DIR} {TRT_ENGINE_DIR} {TP_SIZE} {PP_SIZE} {TRTLLM_EXAMPLE_CODE_DIR} {PYTHON_CODE_DIR}")
    