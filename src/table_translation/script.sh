CUDA_VISIBLE_DEVICES=0,1 python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen3-235B-A22B-Instruct-2507 \
    --trust-remote-code \
    --gpu-memory-utilization 0.9 \
    --max-model-len 2096

export HF_HOME=/data/asca/MMTQA/.cache
CUDA_VISIBLE_DEVICES=0,1 \
vllm serve Qwen/Qwen3-32B \
    --tensor-parallel-size 2 \
    --gpu-memory-utilization 0.8