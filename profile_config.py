
GPU_ID_LIST = [0, 1, 2, 3, 4, 5, 6, 7]
BACKEND = "trtllm"
assert BACKEND in ("hf", "vllm", "trtllm")
MODEL_NICKNAME = "llama3_70b"

WARMUP = 0
TESTFREQ = 1

PROFILE_CFG = list()

PROFILE_CFG = [(1, 1, 128),]
# example: [(batch, input_length, output_length), ...]
for batch in [8, 16, 32, 64]: # 
    for input_length in [1]:
        if batch == 8:
            out_list = [65536]
        else:
            out_list = [1024, 2048, 4096, 8192, 16384, 32768, 65536]
        for output_length in out_list:
            PROFILE_CFG.append((batch, input_length, output_length))

# PROFILE_CFG = [
#     (1, 32, 32),
#     (1, 32, 64),
# ]

if MODEL_NICKNAME == "llama3_8b":
    TRT_ENGINE_DIR = "/mnt/public/trt_models/Meta-Llama-3-8B-Instruct-8gpu-engine"
    # TRT_ENGINE_DIR = "/mnt/public/trt_models/Meta-Llama-3-8B-Instruct-1gpu-engine"
    HF_MODEL_DIR = "/mnt/public/Meta-Llama-3-8B-Instruct"
elif MODEL_NICKNAME == "llama3_70b":
    assert len(GPU_ID_LIST) == 8
    TRT_ENGINE_DIR = "/mnt/public/trt_models/Meta-Llama-3-70B-Instruct-hf-8gpu-new-engine"
    HF_MODEL_DIR = "/mnt/public/Meta-Llama-3-70B-Instruct-hf"
else:
    raise ValueError


if __name__ == "__main__":
    # DO NOT CHANGE THIS PRINT!
    # for the bash script to directly call this file to get the BACKEND and GPU_NUM variable!
    print(f"{BACKEND} {len(GPU_ID_LIST)}")
