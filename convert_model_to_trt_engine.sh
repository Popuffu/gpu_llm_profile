model_name=Meta-Llama-3-8B-Instruct
hf_model_dir=/mnt/public/$model_name
cpkt_dir=/mnt/public/trt_models/$model_name-ckpt
engine_dir=/mnt/public/trt_models/$model_name-engine
tp_size=8
dtype=float16

cd /mnt/public/gpu_llm_profile/TensorRT-LLM/examples/llama
python3 convert_checkpoint.py --model_dir $hf_model_dir --output_dir $cpkt_dir --dtype $dtype --tp_size $tp_size
trtllm-build --checkpoint_dir $cpkt_dir --output_dir $engine_dir --gemm_plugin auto
