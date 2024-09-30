#!/bin/bash

# Get both params from the Python script
read BACKEND GPU_NUM DATA_TYPE MODEL_NAME <<< $(python profile_config.py)
echo "Backend: $BACKEND"
echo "GPU_NUM: $GPU_NUM"
echo "DATA_TYPE: $DATA_TYPE"
echo "MODEL_NAME: $MODEL_NAME"

hf_model_dir=/mnt/public/$MODEL_NAME
group_size=128

cd /mnt/public/TensorRT-LLM/examples/llama
echo Start to generate ckpt

if [ "$DATA_TYPE" == "FP16" ]; then
    cpkt_dir=/mnt/public/trt_models/$MODEL_NAME-$GPU_NUM"gpu-fp16-ckpt"
    engine_dir=/mnt/public/trt_models/$MODEL_NAME-$GPU_NUM"gpu-fp16-engine"
    python3 convert_checkpoint.py \
        --model_dir $hf_model_dir \
        --output_dir $cpkt_dir \
        --dtype float16 \
        --tp_size $GPU_NUM \
        --workers $GPU_NUM

elif [ "$DATA_TYPE" == "W4A16KV8G64" ]; then
    cpkt_dir=/mnt/public/trt_models/$MODEL_NAME-$GPU_NUM"gpu-awq-w4a16kv8-g"$group_size"-ckpt"
    engine_dir=/mnt/public/trt_models/$MODEL_NAME-$GPU_NUM"gpu-awq-w4a16kv8-g"$group_size"-engine"
    python ../quantization/quantize.py \
        --model_dir $hf_model_dir \
        --output_dir $cpkt_dir \
        --dtype float16 \
        --qformat int4_awq \
        --awq_block_size $group_size \
        --kv_cache_dtype int8 \
        --calib_size 32 \
        --tp_size $GPU_NUM

else
    echo "Unknown data_type: $DATA_TYPE"
    exit 1
fi

echo Start to generate engine
trtllm-build \
    --checkpoint_dir $cpkt_dir \
    --output_dir $engine_dir \
    --gemm_plugin auto \
    --workers $GPU_NUM
    # --max_batch_size 64 --max_input_len 1048576 --max_seq_len 1048576 

# Clear ckpt intermediate file
rm -r $cpkt_dir
