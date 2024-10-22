
GPU_ID_LIST = [0, 1, 2, 3, 4, 5, 6, 7]
BACKEND = "trtllm"
MODEL_NICKNAME = "llama3_70b"
DATA_TYPE = "FP16"
GPU_NAME = "RTX4090"

WARMUP = 1
TESTFREQ = 1

PROFILE_CFG = [(1, 1, 512),]

assert BACKEND in ("hf", "vllm", "trtllm")
assert DATA_TYPE in ("FP16", "FP16-TP4PP2", "FP16-TP2PP4", "FP16-TP1PP8", "W4A16KV8G128", "W8A8") # FP16默认TP8

# example: [(batch, input_length, output_length), ...]
for output_length in [1024+2048]:
    batch_list = [1]
    for i in range(2, 129, 2):
        batch_list.append(i)
            
    for batch in batch_list: # 
        for input_length in [1]:
            PROFILE_CFG.append((batch, input_length, output_length))


if MODEL_NICKNAME == "llama3_8b":
    MODEL_NAME = "Meta-Llama-3-8B-Instruct"
    assert len(GPU_ID_LIST) in [1, 2, 4, 8]
    if DATA_TYPE == "FP16":
        TRT_ENGINE_DIR = f"/mnt/public/trt_models/{MODEL_NAME}-{len(GPU_ID_LIST)}gpu-fp16-engine"
    elif DATA_TYPE == "W4A16KV8G128":
        TRT_ENGINE_DIR = f"/mnt/public/trt_models/{MODEL_NAME}-{len(GPU_ID_LIST)}gpu-awq-w4a16kv8-g128-engine"
    else:
        raise ValueError
    HF_MODEL_DIR = f"/mnt/public/{MODEL_NAME}"
elif MODEL_NICKNAME == "llama3_70b":
    MODEL_NAME = "Meta-Llama-3-70B-Instruct-hf"
    if GPU_NAME == "RTX4090":
        assert len(GPU_ID_LIST) == 8
    if DATA_TYPE == "FP16":
        TRT_ENGINE_DIR = f"/mnt/public/trt_models/{MODEL_NAME}-{len(GPU_ID_LIST)}gpu-fp16-engine" # 默认全使用TP
    elif DATA_TYPE == "FP16-TP4PP2":
        TRT_ENGINE_DIR = f"/mnt/public/trt_models/{MODEL_NAME}-{len(GPU_ID_LIST)}gpu-tp4-pp2-fp16-engine"
    elif DATA_TYPE == "FP16-TP2PP4":
        TRT_ENGINE_DIR = f"/mnt/public/trt_models/{MODEL_NAME}-{len(GPU_ID_LIST)}gpu-tp2-pp4-fp16-engine"
    elif DATA_TYPE == "FP16-TP1PP8":
        TRT_ENGINE_DIR = f"/mnt/public/trt_models/{MODEL_NAME}-{len(GPU_ID_LIST)}gpu-tp1-pp8-fp16-engine"
    elif DATA_TYPE == "W4A16KV8G128":
        TRT_ENGINE_DIR = f"/mnt/public/trt_models/{MODEL_NAME}-{len(GPU_ID_LIST)}gpu-awq-w4a16kv8-g128-engine"
    elif DATA_TYPE == "W8A8":
        TRT_ENGINE_DIR = f"/mnt/public/trt_models/{MODEL_NAME}-{len(GPU_ID_LIST)}gpu-awq-w4a16kv8-g128-engine"
    else:
        raise ValueError
    HF_MODEL_DIR = f"/mnt/public/{MODEL_NAME}"
else:
    raise ValueError


if __name__ == "__main__":
    # DO NOT CHANGE THIS PRINT!
    # for the bash script to directly call this file to get the BACKEND and GPU_NUM variable!
    print(f"{BACKEND} {len(GPU_ID_LIST)} {DATA_TYPE} {MODEL_NAME}")
