# vram-oracle

### Estimate LLM VRAM usage before deployment

`vram-oracle` is a lightweight tool that estimates GPU memory requirements for decoder-only Hugging Face models.

It computes:

* Model weight memory
* KV cache memory
* Total estimated VRAM
* Effect of context length
* Effect of batch size
* Effect of dtype (fp16 / bf16 / fp32)

Designed for:

* LLM infra engineers
* vLLM users
* Researchers
* Long-context model experimentation
* Hardware planning & deployment decisions

---

## Why This Exists

Large language models fail silently when VRAM is insufficient.

Before launching inference jobs, scaling context windows, or experimenting with speculative decoding, you need to answer:

> Will this fit in memory?

`vram-oracle` gives you a fast answer — without loading model weights.

---

## What It Estimates

For a decoder-only transformer:

Total VRAM ≈

```
Model Weights + KV Cache
```

Where:

### Weights

```
parameters × bytes_per_param
```

### KV Cache

```
batch_size × context_length × num_layers × 
(2 × num_kv_heads × head_dim × bytes_per_dtype)
```

It reads architecture parameters directly from Hugging Face configs.

---

## Run via GitHub Actions (Phone-Friendly)

This repo includes a manual GitHub Action workflow.

You can:

1. Open the repo on your phone
2. Go to **Actions**
3. Click **Estimate VRAM**
4. Enter:

   * `model_name`
   * `context_length`
   * `batch_size`
   * `dtype`
5. Click **Run workflow**

Results will appear in:

* The job summary
* `result.json` artifact

No local setup required.

---

## Local Usage

```bash
pip install transformers huggingface_hub

python estimate_vram.py \
  --model_name nvidia/Llama-3.1-Nemotron-Nano-4B-v1.1 \
  --context_length 128000 \
  --batch_size 1 \
  --dtype float16
```

Example output:

```json
{
  "model": "nvidia/Llama-3.1-Nemotron-Nano-4B-v1.1",
  "context_length": 128000,
  "batch_size": 1,
  "dtype": "float16",
  "estimated_parameters": 4000000000,
  "weights_GB": 7.45,
  "kv_cache_GB": 9.32,
  "total_estimated_GB": 16.77
}
```

---

## Use Cases

### 1️⃣ Long-Context Planning

How much VRAM does 128k context require?

### 2️⃣ Speculative Decoding Research

How does deeper speculation (longer decode horizon) affect KV memory?

### 3️⃣ Hardware Comparison

Will this fit on:

* 24GB RTX 4090?
* 48GB A6000?
* 80GB A100?

### 4️⃣ vLLM Deployment Planning

Estimate memory budget before serving.

---

## 📌 Notes

* This tool assumes decoder-only architectures.
* Uses config-based parameter estimates (no weight download required).
* Does not include:

  * Activation memory
  * CUDA graph overhead
  * Fragmentation
  * Optimizer states (training)

For inference planning, estimates are typically close to real-world usage.

---

## Philosophy

Infrastructure research should be predictive.

Before benchmarking.
Before OOM errors.
Before deployment.

We estimate first.

---

## Contributions

Open to:

* Multi-model batch estimation
* Tensor parallel support
* Quantization-aware estimates
* vLLM compatibility flags
* Speculative decoding memory modeling
* Visualization tools

---

## License

MIT
