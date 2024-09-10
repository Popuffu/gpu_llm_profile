## Introduction

This project is designed to provide a profiling tool for large language model (LLM) inference on GPUs.

## Features

- Different input and output token length
- Prefill and Decode latency breakdown
- GPU power consumption statistics
- Multi-GPU profiling support

## Framework Support

- HuggingFace
- vLLM
- TensorRT-LLM

## Getting Started

1. Install the required libraries: vllm, tensorrt-llm, etc.
2. Change the profiling params and local model path in `profile_config.py`
3. If you want to use the trtllm, you must generate the tensorrt-llm engine first by running `trtllm/convert_model_to_trt_engine.sh`. The intermediate tensorrt-llm ckpt can be removed after you successfully get the engine.
4. Run `run_profile.sh`
5. Get your profiling results in `profile_results.csv`

## Version of key libraries during development (for reference only)
- tensorrt-llm:   0.13.0.dev2024090300
- tensorrt:       10.3.0
- torch:          2.4.0
- transformers:   4.44.2
- mpi4py:         4.0.0