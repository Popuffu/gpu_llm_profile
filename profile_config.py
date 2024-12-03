
GPU_ID_LIST = [0]#, 1, 2, 3, 4, 5, 6, 7]
BACKEND = "trtllm"
MODEL_NICKNAME = "llama3.1_70b"
DATA_TYPE = "W4A16KV8G128"
GPU_NAME = "A100"

GPU_NUM = len(GPU_ID_LIST)
TP_SIZE = GPU_NUM
PP_SIZE = 1

WARMUP = 1
TESTFREQ = 1

PROFILE_CFG = [(1, 1, 1024),]

assert BACKEND in ("hf", "vllm", "trtllm")
assert DATA_TYPE in ("FP16", "FP8", "W4A16KV8G128", "W8A8smooth") # FP16默认TP8

PARALLEL_NAME = f"TP{TP_SIZE}PP{PP_SIZE}"

# example: [(batch, input_length, output_length), ...]
for output_length in [1024, 1024+4096]:#[1024, 1024+4096]:#[1024, 1024+4096]:
    batch_list = []
    for i in range(128, 2048, 64):#range(2, 129, 2):
        batch_list.append(i)
            
    for batch in batch_list: # 
        for input_length in [1]:
            PROFILE_CFG.append((batch, input_length, output_length))


if MODEL_NICKNAME == "llama3_8b":
    MODEL_NAME = "Meta-Llama-3-8B-Instruct"
elif MODEL_NICKNAME == "llama3.1_70b":
    MODEL_NAME = "Meta-Llama-3.1-70B-Instruct"
else:
    raise ValueError

PROFILE_RESULT_DIR = f"{MODEL_NICKNAME}_{GPU_NAME}_profile.csv"

TRTLLM_EXAMPLE_CODE_DIR = "/mnt/public/yangxinhao/TensorRT-LLM/examples"
PYTHON_CODE_DIR = "/mnt/public/yangxinhao/gpu_llm_profile"
HF_MODEL_DIR = f"/mnt/datasets/public_models/{MODEL_NAME}"
TRT_BASE_DIR = f"{PYTHON_CODE_DIR}/trt_models/{MODEL_NAME}-{len(GPU_ID_LIST)}x{GPU_NAME}-{DATA_TYPE}-{PARALLEL_NAME}"

TRT_CKPT_DIR = f"{TRT_BASE_DIR}-ckpt"
TRT_ENGINE_DIR = f"{TRT_BASE_DIR}-engine"

if __name__ == "__main__":
    # DO NOT CHANGE THIS PRINT!
    # for the bash script to directly call this file to get the BACKEND and GPU_NUM variable!
    print(f"{BACKEND} {GPU_NUM} {DATA_TYPE} {MODEL_NAME} {HF_MODEL_DIR} {TRT_CKPT_DIR} {TRT_ENGINE_DIR} {TP_SIZE} {PP_SIZE} {TRTLLM_EXAMPLE_CODE_DIR} {PYTHON_CODE_DIR}")
    