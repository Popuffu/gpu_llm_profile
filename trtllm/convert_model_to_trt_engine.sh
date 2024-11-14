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

elif [ "$DATA_TYPE" == "FP16-TP4PP2" ]; then
    cpkt_dir=/mnt/public/trt_models/$MODEL_NAME-$GPU_NUM"gpu-tp4-pp2-fp16-ckpt"
    engine_dir=/mnt/public/trt_models/$MODEL_NAME-$GPU_NUM"gpu-tp4-pp2-fp16-engine"
    python3 convert_checkpoint.py \
        --model_dir $hf_model_dir \
        --output_dir $cpkt_dir \
        --dtype float16 \
        --tp_size 4 \
        --pp_size 2 \
        --workers 1

elif [ "$DATA_TYPE" == "FP16-TP2PP4" ]; then
    cpkt_dir=/mnt/public/trt_models/$MODEL_NAME-$GPU_NUM"gpu-tp2-pp4-fp16-ckpt"
    engine_dir=/mnt/public/trt_models/$MODEL_NAME-$GPU_NUM"gpu-tp2-pp4-fp16-engine"
    python3 convert_checkpoint.py \
        --model_dir $hf_model_dir \
        --output_dir $cpkt_dir \
        --dtype float16 \
        --tp_size 2 \
        --pp_size 4 \
        --workers 1

elif [ "$DATA_TYPE" == "FP16-TP1PP8" ]; then
    cpkt_dir=/mnt/public/trt_models/$MODEL_NAME-$GPU_NUM"gpu-tp1-pp8-fp16-ckpt"
    engine_dir=/mnt/public/trt_models/$MODEL_NAME-$GPU_NUM"gpu-tp1-pp8-fp16-engine"
    python3 convert_checkpoint.py \
        --model_dir $hf_model_dir \
        --output_dir $cpkt_dir \
        --dtype float16 \
        --tp_size 1 \
        --pp_size 8 \
        --workers 1

elif [ "$DATA_TYPE" == "FP8-TP8PP1" ]; then
    cpkt_dir=/mnt/public/trt_models/$MODEL_NAME-$GPU_NUM"gpu-tp8-pp1-fp8-ckpt_fmha"
    engine_dir=/mnt/public/trt_models/$MODEL_NAME-$GPU_NUM"gpu-tp8-pp1-fp8-engine_fmha"
    python3 ../quantization/quantize.py \
        --model_dir $hf_model_dir \
        --output_dir $cpkt_dir \
        --dtype float16 \
        --qformat fp8 \
        --kv_cache_dtype fp8 \
        --calib_size 512 \
        --tp_size 8 \
        --pp_size 1

elif [ "$DATA_TYPE" == "W4A16KV8G128" ]; then
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

elif [ "$DATA_TYPE" == "W8A8" ]; then # 没跑通，报一个tensor在CPU和GPU不同位置的错
    cpkt_dir=/mnt/public/trt_models/$MODEL_NAME-$GPU_NUM"gpu-awq-w8a8-ckpt"
    engine_dir=/mnt/public/trt_models/$MODEL_NAME-$GPU_NUM"gpu-awq-w8a8-engine"
    python3 convert_checkpoint.py --model_dir $hf_model_dir \
                                --output_dir $cpkt_dir \
                                --dtype float16 \
                                --smoothquant 0.5 \
                                --per_token \
                                --per_channel

else
    echo "Unknown data_type: $DATA_TYPE"
    exit 1
fi

echo Start to generate engine

if [ "$DATA_TYPE" == "FP8-TP8PP1" ]; then
    trtllm-build \
        --checkpoint_dir $cpkt_dir \
        --output_dir $engine_dir \
        --gemm_plugin auto \
        --workers $GPU_NUM \
        --use_fp8_context_fmha enable

else
    trtllm-build \
        --checkpoint_dir $cpkt_dir \
        --output_dir $engine_dir \
        --gemm_plugin auto \
        --workers $GPU_NUM
fi

# Clear ckpt intermediate file
rm -r $cpkt_dir
