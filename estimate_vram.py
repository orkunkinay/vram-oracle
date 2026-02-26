import argparse
import json
from transformers import AutoConfig

def estimate_vram(model_name: str, context_length: int, batch_size: int = 1, dtype: str = "float16"):
    dtype_bytes_map = {"float16": 2, "bfloat16": 2, "float32": 4}
    if dtype not in dtype_bytes_map:
        raise ValueError("dtype must be float16, bfloat16, or float32")

    bytes_per_param = dtype_bytes_map[dtype]
    bytes_per_kv = dtype_bytes_map[dtype]

    config = AutoConfig.from_pretrained(model_name)

    hidden_size = config.hidden_size
    num_layers = config.num_hidden_layers
    num_heads = config.num_attention_heads
    num_kv_heads = getattr(config, "num_key_value_heads", num_heads)
    head_dim = hidden_size // num_heads

    # Prefer exact count if present; else rough rule of thumb
    if hasattr(config, "num_parameters") and config.num_parameters is not None:
        total_params = int(config.num_parameters)
    else:
        total_params = int(12 * hidden_size * hidden_size * num_layers)

    weight_memory = total_params * bytes_per_param

    kv_per_token_per_layer = 2 * num_kv_heads * head_dim * bytes_per_kv
    kv_total = batch_size * context_length * num_layers * kv_per_token_per_layer

    def to_gb(x): return x / (1024**3)

    return {
        "model": model_name,
        "context_length": context_length,
        "batch_size": batch_size,
        "dtype": dtype,
        "estimated_parameters": total_params,
        "weights_GB": to_gb(weight_memory),
        "kv_cache_GB": to_gb(kv_total),
        "total_estimated_GB": to_gb(weight_memory + kv_total),
    }

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_name", required=True)
    p.add_argument("--context_length", type=int, required=True)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default="float16")
    args = p.parse_args()

    out = estimate_vram(args.model_name, args.context_length, args.batch_size, args.dtype)
    print(json.dumps(out, indent=2))

    with open("result.json", "w") as f:
        json.dump(out, f, indent=2)

if __name__ == "__main__":
    main()
