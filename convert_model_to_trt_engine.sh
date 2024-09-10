model_name=Meta-Llama-3-70B-Instruct-hf
tp_size=8
hf_model_dir=/mnt/public/$model_name
cpkt_dir=/mnt/public/trt_models/$model_name-$tp_size"gpu-ckpt"
engine_dir=/mnt/public/trt_models/$model_name-$tp_size"gpu-engine"
dtype=float16

cd /mnt/public/TensorRT-LLM/examples/llama
echo Start to generate ckpt
python3 convert_checkpoint.py --model_dir $hf_model_dir --output_dir $cpkt_dir --dtype $dtype --tp_size $tp_size
echo Start to generate engine
trtllm-build --checkpoint_dir $cpkt_dir --output_dir $engine_dir --gemm_plugin auto
