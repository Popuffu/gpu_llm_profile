#!/bin/bash

# Exit immediately if any command exits with a non-zero status
set -e

# Get both params from the Python script
read BACKEND GPU_NUM GPU_NAME DATA_TYPE MODEL_NAME HF_MODEL_DIR TRT_CKPT_DIR TRT_ENGINE_DIR TP_SIZE PP_SIZE TRTLLM_EXAMPLE_CODE_DIR PYTHON_CODE_DIR <<< $(python profile_config.py)

echo "-------------------------------------"
echo "Backend: $BACKEND"
echo "GPU: $GPU_NUM * $GPU_NAME (TP: $TP_SIZE, PP: $PP_SIZE)"
echo "DATA_TYPE: $DATA_TYPE"
echo "MODEL_NAME: $MODEL_NAME"
echo "HF_MODEL_DIR: $HF_MODEL_DIR"
echo "TRT_CKPT_DIR: $TRT_CKPT_DIR"
echo "TRT_ENGINE_DIR: $TRT_ENGINE_DIR"
echo "TRTLLM_EXAMPLE_CODE_DIR: $TRTLLM_EXAMPLE_CODE_DIR"
echo "PYTHON_CODE_DIR: $PYTHON_CODE_DIR"
echo "-------------------------------------"

if [ "$BACKEND" == "trtllm" ]; then
    # 1. update the max_seq_len in trtllm engine config.json
    PYTHONPATH=$PYTHON_CODE_DIR:$PYTHONPATH python trtllm/update_trtllm_engine_config.py
    # 2. run the trtllm engine to profile
    PYTHONPATH=$PYTHON_CODE_DIR:$PYTHONPATH mpirun -n $GPU_NUM --allow-run-as-root python trtllm/trtllm_example_run.py

elif [ "$BACKEND" == "hf" ] || [ "$BACKEND" == "vllm" ]; then
    PYTHONPATH=$PYTHON_CODE_DIR:$PYTHONPATH python hf_vllm/hf_vllm_run.py

else
    echo "Unknown backend: $BACKEND"
    exit 1
fi
