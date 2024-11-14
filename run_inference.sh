engine_dir=/mnt/public/trt_models/Meta-Llama-3-70B-Instruct-hf-8gpu-tp8-pp1-fp8-engine
hf_model_dir=/mnt/public/Meta-Llama-3-70B-Instruct-hf

mpirun -n 8 --allow-run-as-root \
    python ./run.py \
    --max_output_len 128 \
    --max_input_length 256 \
    --engine_dir $engine_dir \
    --tokenizer_dir $hf_model_dir
