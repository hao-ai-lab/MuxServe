## Cost File

The `llama.json` contains the profiled cost of LLaMA models on A100 GPU (80GB). The format of `llama.json` is:
```
model_name: {
    "num_gpus": {
        "percentage of SMs": {
            "batch size": {
                "sequence length": {
                    "prefill": 0.0,
                    "decoding": 0.0,
                }
            }
        }
    }
}
```

For `prefill`, the input tokens is `batch size`$\times$`sequence length`. For `decoding`, the KV cache is `batch size`$\times$`sequence length`, while the input tokens is `batch size`. All the cost is time in milliseconds.
