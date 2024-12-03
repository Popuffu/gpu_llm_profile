#!/bin/bash

# Exit immediately if any command exits with a non-zero status
set -e

# Get both params from the Python script
read BACKEND GPU_NUM DATA_TYPE MODEL_NAME HF_MODEL_DIR TRT_CKPT_DIR TRT_ENGINE_DIR TP_SIZE PP_SIZE TRTLLM_EXAMPLE_CODE_DIR PYTHON_CODE_DIR <<< $(python profile_config.py)

echo "-------------------------------------"
echo "Backend: $BACKEND"
echo "GPU_NUM: $GPU_NUM (TP: $TP_SIZE, PP: $PP_SIZE)"
echo "DATA_TYPE: $DATA_TYPE"
echo "MODEL_NAME: $MODEL_NAME"
echo "HF_MODEL_DIR: $HF_MODEL_DIR"
echo "TRT_CKPT_DIR: $TRT_CKPT_DIR"
echo "TRT_ENGINE_DIR: $TRT_ENGINE_DIR"
echo "TRTLLM_EXAMPLE_CODE_DIR: $TRTLLM_EXAMPLE_CODE_DIR"
echo "PYTHON_CODE_DIR: $PYTHON_CODE_DIR"
echo "-------------------------------------"

cd $TRTLLM_EXAMPLE_CODE_DIR/llama
echo Start to generate ckpt

workers=8

if [ "$DATA_TYPE" == "FP16" ]; then
    python3 convert_checkpoint.py \
        --model_dir $HF_MODEL_DIR \
        --output_dir $TRT_CKPT_DIR \
        --dtype float16 \
        --tp_size $TP_SIZE \
        --pp_size $PP_SIZE \
        --workers $workers

elif [ "$DATA_TYPE" == "FP8" ]; then
    python3 ../quantization/quantize.py \
        --model_dir $HF_MODEL_DIR \
        --output_dir $TRT_CKPT_DIR \
        --dtype float16 \
        --qformat fp8 \
        --kv_cache_dtype fp8 \
        --calib_size 512 \
        --tp_size $TP_SIZE \
        --pp_size $PP_SIZE

elif [ "$DATA_TYPE" == "W4A16KV8G128" ]; then
    python ../quantization/quantize.py \
        --model_dir $HF_MODEL_DIR \
        --output_dir $TRT_CKPT_DIR \
        --dtype float16 \
        --qformat int4_awq \
        --awq_block_size 128 \
        --kv_cache_dtype int8 \
        --calib_size 32 \
        --tp_size $TP_SIZE \
        --pp_size $PP_SIZE

elif [ "$DATA_TYPE" == "W8A8smooth" ]; then # 没跑通，报一个tensor在CPU和GPU不同位置的错
    python3 convert_checkpoint.py \
        --model_dir $HF_MODEL_DIR \
        --output_dir $TRT_CKPT_DIR \
        --dtype float16 \
        --smoothquant 0.5 \
        --per_token \
        --per_channel \
        --tp_size $TP_SIZE \
        --pp_size $PP_SIZE

else
    echo "Unknown data_type: $DATA_TYPE"
    exit 1
fi

echo Start to generate engine

if [ "$DATA_TYPE" == "FP8" ]; then
    trtllm-build \
        --checkpoint_dir $TRT_CKPT_DIR \
        --output_dir $TRT_ENGINE_DIR \
        --gemm_plugin auto \
        --workers $workers
        # --use_fp8_context_fmha enable

else
    trtllm-build \
        --checkpoint_dir $TRT_CKPT_DIR \
        --output_dir $TRT_ENGINE_DIR \
        --gemm_plugin auto \
        --workers $workers
fi

# Clear ckpt intermediate file
echo Clear generated ckpt
rm -r $TRT_CKPT_DIR
