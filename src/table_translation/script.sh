python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-3B-Instruct \
    --trust-remote-code \
    --gpu-memory-utilization 0.30 \
    --max-model-len 2096