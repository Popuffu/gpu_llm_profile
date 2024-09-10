#!/bin/bash

# Get both BACKEND and GPU_NUM from the Python script
read BACKEND GPU_NUM <<< $(python profile_config.py)
echo "Backend: $BACKEND"
echo "GPU_NUM: $GPU_NUM"

if [ "$BACKEND" == "trtllm" ]; then
    # 1. update the max_seq_len in trtllm engine config.json
    PYTHONPATH=/mnt/public/gpu_llm_profile:$PYTHONPATH python trtllm/update_trtllm_engine_config.py
    # 2. run the trtllm engine to profile
    PYTHONPATH=/mnt/public/gpu_llm_profile:$PYTHONPATH mpirun -n $GPU_NUM --allow-run-as-root python trtllm/trtllm_example_run.py

elif [ "$BACKEND" == "hf" ] || [ "$BACKEND" == "vllm" ]; then
    PYTHONPATH=/mnt/public/gpu_llm_profile:$PYTHONPATH python hf_vllm/hf_vllm_run.py

else
    echo "Unknown backend: $BACKEND"
    exit 1
fi
