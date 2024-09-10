
GPU_ID_LIST = [0, 1, 2, 3, 4, 5, 6, 7]
BACKEND = "trtllm"
assert BACKEND in ("hf", "vllm", "trtllm")
MODEL_NICKNAME = "llama3_70b"

WARMUP = 4
TESTFREQ = 10

PROFILE_CFG = list()
# example: [(batch, input_length, output_length), ...]
for batch in [1]:
    for input_length in [1]:
        for output_length_k in [0.25, 0.5, 1, 2, 4, 8, 16]:#, 32, 64, 128, 256]:
            PROFILE_CFG.append((batch, input_length, int(output_length_k * 1024)))

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
    TRT_ENGINE_DIR = "/mnt/public/trt_models/Meta-Llama-3-70B-Instruct-hf-8gpu-engine"
    HF_MODEL_DIR = "/mnt/public/Meta-Llama-3-70B-Instruct-hf"
else:
    raise ValueError


if __name__ == "__main__":
    # DO NOT CHANGE THIS PRINT!
    # for the bash script to directly call this file to get the BACKEND and GPU_NUM variable!
    print(f"{BACKEND} {len(GPU_ID_LIST)}")
