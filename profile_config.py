
GPU_ID_LIST = [0, 1, 2, 3, 4, 5, 6, 7]
BACKEND = "trtllm"
MODEL_NICKNAME = "llama3_70b"
DATA_TYPE = "FP16"
GPU_NAME = "RTX4090"

WARMUP = 1
TESTFREQ = 4

PROFILE_CFG = [(1, 1, 128),]

assert BACKEND in ("hf", "vllm", "trtllm")
assert DATA_TYPE in ("FP16", "W4A16KV8")

# example: [(batch, input_length, output_length), ...]
for batch in [1, 2, 4, 8, 16, 32, 64, 128, 256]: # 
    for input_length in [1]:
        for output_length in [1024]:
            PROFILE_CFG.append((batch, input_length, output_length))

if MODEL_NICKNAME == "llama3_8b":
    MODEL_NAME = "Meta-Llama-3-8B-Instruct"
    assert len(GPU_ID_LIST) in [1, 2, 4, 8]
    if DATA_TYPE == "FP16":
        TRT_ENGINE_DIR = f"/mnt/public/trt_models/{MODEL_NAME}-{len(GPU_ID_LIST)}gpu-fp16-engine"
    elif DATA_TYPE == "W4A16KV8":
        TRT_ENGINE_DIR = f"/mnt/public/trt_models/{MODEL_NAME}-{len(GPU_ID_LIST)}gpu-awq-w4a16kv8-engine"
    else:
        raise ValueError
    HF_MODEL_DIR = f"/mnt/public/{MODEL_NAME}"
elif MODEL_NICKNAME == "llama3_70b":
    MODEL_NAME = "Meta-Llama-3-70B-Instruct-hf"
    assert len(GPU_ID_LIST) == 8
    if DATA_TYPE == "FP16":
        TRT_ENGINE_DIR = f"/mnt/public/trt_models/{MODEL_NAME}-{len(GPU_ID_LIST)}gpu-fp16-engine"
    elif DATA_TYPE == "W4A16KV8":
        TRT_ENGINE_DIR = f"/mnt/public/trt_models/{MODEL_NAME}-{len(GPU_ID_LIST)}gpu-awq-w4a16kv8-engine"
    else:
        raise ValueError
    HF_MODEL_DIR = f"/mnt/public/{MODEL_NAME}"
else:
    raise ValueError


if __name__ == "__main__":
    # DO NOT CHANGE THIS PRINT!
    # for the bash script to directly call this file to get the BACKEND and GPU_NUM variable!
    print(f"{BACKEND} {len(GPU_ID_LIST)} {DATA_TYPE} {MODEL_NAME}")
