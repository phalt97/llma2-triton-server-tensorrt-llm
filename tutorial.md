# Triton Server with TensorRT-LLM

## Requirements

    1. Docker & Nvidia Docker
    2. NVIDIA GPU min 8GB VRAM
    3. Ubuntu 20.04/Ubuntu 22.04

## Step 1: Strart docker container and download llama2 model

    ```
    nvidia-docker run -d -it --name trtllm -v $PWD:/workspace --shm-size=16G --network=host nvcr.io/nvidia/tritonserver:23.10-trtllm-python-py3 /bin/bash
    ```

    Run this command to access container
    ```
    docker exec -it trtllm bash
    ```

    Download model after login to your huggingface hub

    ```
    python download.py
    ```
    model path: /root/.cache/huggingface/hub/models--huggyllama--llama-7b/snapshots/8416d3fefb0cb3ff5775a7b13c1692d10ff1aa16

    ```
    export MODEL_PATH=<model path>
    ```
## Step 2: Clone tensorrtllm_backend and tens tensorRT-LLM repository:

    ```
    https://github.com/triton-inference-server/tensorrtllm_backend.git

    cd tensorrtllm_backend/

    git clone https://github.com/NVIDIA/TensorRT-LLM.git
    ```

## Step 3: Install TensorRT-LLM 

    ```
    pip install git+https://github.com/NVIDIA/TensorRT-LLM.git

    mkdir /usr/local/lib/python3.10/dist-packages/tensorrt_llm/libs/

    cp /opt/tritonserver/backends/tensorrtllm/* /usr/local/lib/python3.10/dist-packages/tensorrt_llm/libs/

    ```

## Step 4: Build TensorRT engine

    ```
    cd tensorrt_llm/examples/llama
    ```

    ```
    # Build a single-GPU float16 engine from HF weights.
    # use_gpt_attention_plugin is necessary in LLaMA.
    # Try use_gemm_plugin to prevent accuracy issue.
    # It is recommend to use --remove_input_padding along with --use_gpt_attention_plugin for better performance

    # Build the LLaMA 7B model using a single GPU and FP16.
    python build.py --model_dir $MODEL_PATH \
                    --dtype float16 \
                    --remove_input_padding \
                    --use_gpt_attention_plugin float16 \
                    --enable_context_fmha \
                    --use_gemm_plugin float16 \
                    --output_dir llama/7B/trt_engines/fp16/1-gpu/

    # Build the LLaMA 7B model using a single GPU and apply INT8 weight-only quantization.
    python build.py --model_dir $MODEL_PATH \
                    --dtype float16 \
                    --remove_input_padding \
                    --use_gpt_attention_plugin float16 \
                    --enable_context_fmha \
                    --use_gemm_plugin float16 \
                    --use_weight_only \
                    --output_dir llama/7B/trt_engines/weight_only/1-gpu/

    # Build LLaMA 7B using 2-way tensor parallelism.
    python build.py --model_dir $MODEL_PATH \
                    --dtype float16 \
                    --remove_input_padding \
                    --use_gpt_attention_plugin float16 \
                    --enable_context_fmha \
                    --use_gemm_plugin float16 \
                    --output_dir llama/7B/trt_engines/fp16/2-gpu/ \
                    --world_size 2 \
                    --tp_size 2
    
    # Build the LLaMA 13B model using a single GPU and BF16.
    python build.py --model_dir $MODEL_PATH \
                    --dtype bfloat16 \
                    --remove_input_padding \
                    --use_gpt_attention_plugin bfloat16 \
                    --enable_context_fmha \
                    --use_gemm_plugin bfloat16 \
                    --output_dir llama/13B/trt_engines/bf16/1-gpu/

    # Build the LLaMA 13B model using a single GPU and apply INT8 weight-only quantization.
    python build.py --model_dir $MODEL_PATH \
                    --dtype float16 \
                    --remove_input_padding \
                    --use_gpt_attention_plugin float16 \
                    --enable_context_fmha \
                    --use_gemm_plugin float16 \
                    --use_weight_only \
                    --output_dir llama/13B/trt_engines/weight_only/1-gpu/
    ```

## Step 5: Create the model repository

    ```
    # Create the model repository that will be used by the Triton server
    cd tensorrtllm_backend

    mkdir model-repository

    # Copy the example models to the model repository
    cp -r all_models/inflight_batcher_llm/* model-repository/

    # Copy the TRT engine to triton_model_repo/tensorrt_llm/1/
    cp tensorrt_llm/examples/llama/llama/13B/trt_engines/weight_only/1-gpu/* model-repository/tensorrt_llm/1
    ```

## Step 6: Run Server

    ```
    mpirun --allow-run-as-root  -n 1 /opt/tritonserver/bin/tritonserver --model-repository=/workspace/tensorrtllm_backend/model-repository --disable-auto-complete-config --backend-config=python,shm-region-prefix-name=prefix0_ :
    ```