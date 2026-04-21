# HuggingFace Models — Agentic & Long Context Reasoning

> Curated list of best models per supported architecture for agentic capabilities and long context reasoning.

---

## 🟦 llama / mllama

| Model | Parameters | Context | Agentic | Long Context | Tool Use | License | HuggingFace ID | Notes |
|---|---|---|---|---|---|---|---|---|
| Meta-Llama-4-Scout-Instruct | ~109B MoE | 128K | ★★★★★ | ★★★★★ | Yes | Llama 4 Community | meta-llama/Llama-4-Scout-Instruct | Latest Llama 4; top agentic performance |
| Meta-Llama-3.1-70B-Instruct | 70B | 128K | ★★★★☆ | ★★★★☆ | Yes | Llama 3 Community | meta-llama/Meta-Llama-3.1-70B-Instruct | Strong agentic & RAG; broad ecosystem |
| Meta-Llama-3.1-8B-Instruct | 8B | 128K | ★★★☆☆ | ★★★☆☆ | Yes | Llama 3 Community | meta-llama/Meta-Llama-3.1-8B-Instruct | Lightweight; good for edge agentic tasks |
| Meta-Llama-3.2-11B-Vision-Instruct | 11B | 128K | ★★★☆☆ | ★★★☆☆ | Yes | Llama 3 Community | meta-llama/Llama-3.2-11B-Vision-Instruct | Multimodal llama; tool use support |

---

## 🟩 mistral / mixtral

| Model | Parameters | Context | Agentic | Long Context | Tool Use | License | HuggingFace ID | Notes |
|---|---|---|---|---|---|---|---|---|
| Devstral-Small-2505 | 24B | 128K | ★★★★★ | ★★★★☆ | Yes | Apache 2.0 | mistralai/Devstral-Small-2505 | Purpose-built agentic coding model |
| Mixtral-8x22B-Instruct-v0.1 | ~141B MoE | 64K | ★★★★★ | ★★★★☆ | Yes | Apache 2.0 | mistralai/Mixtral-8x22B-Instruct-v0.1 | Best Mixtral; near-frontier reasoning |
| Mistral-Nemo-Instruct-2407 | 12B | 128K | ★★★★☆ | ★★★★☆ | Yes | Apache 2.0 | mistralai/Mistral-Nemo-Instruct-2407 | 128K context; strong reasoning & agents |
| Mixtral-8x7B-Instruct-v0.1 | ~47B MoE | 32K | ★★★★☆ | ★★★☆☆ | Yes | Apache 2.0 | mistralai/Mixtral-8x7B-Instruct-v0.1 | Fast MoE; great tool use |
| Mistral-7B-Instruct-v0.3 | 7B | 32K | ★★★☆☆ | ★★★☆☆ | Yes | Apache 2.0 | mistralai/Mistral-7B-Instruct-v0.3 | Fast; great function calling baseline |

---

## 🟨 qwen2 / qwen2_vl / qwen2_5_vl

| Model | Parameters | Context | Agentic | Long Context | Tool Use | License | HuggingFace ID | Notes |
|---|---|---|---|---|---|---|---|---|
| Qwen2.5-VL-72B-Instruct | 72B | 128K | ★★★★★ | ★★★★★ | Yes | Qwen License 2.0 | Qwen/Qwen2.5-VL-72B-Instruct | Best multimodal Qwen; 128K context |
| Qwen2-72B-Instruct | 72B | 128K | ★★★★★ | ★★★★★ | Yes | Qwen License 2.0 | Qwen/Qwen2-72B-Instruct | Top long-context & agentic; multilingual |
| Qwen2.5-32B-Instruct | 32B | 128K | ★★★★★ | ★★★★★ | Yes | Apache 2.0 | Qwen/Qwen2.5-32B-Instruct | Strong coding & reasoning; 24GB GPU friendly |
| Qwen2.5-Coder-32B-Instruct | 32B | 128K | ★★★★★ | ★★★★☆ | Yes | Apache 2.0 | Qwen/Qwen2.5-Coder-32B-Instruct | #1 coding agentic model; multi-file reasoning |
| Qwen2-VL-72B-Instruct | 72B | 32K | ★★★★☆ | ★★★★☆ | Yes | Qwen License 2.0 | Qwen/Qwen2-VL-72B-Instruct | Multimodal; strong agent + vision tasks |
| Qwen2-7B-Instruct | 7B | 128K | ★★★★☆ | ★★★★☆ | Yes | Apache 2.0 | Qwen/Qwen2-7B-Instruct | Efficient; excellent for agentic pipelines |

---

## 🟥 gpt_bigcode

| Model | Parameters | Context | Agentic | Long Context | Tool Use | License | HuggingFace ID | Notes |
|---|---|---|---|---|---|---|---|---|
| starcoder2-15b-instruct-v0.1 | 15B | 16K | ★★★☆☆ | ★★★☆☆ | Yes | Apache 2.0 | bigcode/starcoder2-15b-instruct-v0.1 | Best coding base for StarCoder arch |

---

## 🟪 gpt_oss

| Model | Parameters | Context | Agentic | Long Context | Tool Use | License | HuggingFace ID | Notes |
|---|---|---|---|---|---|---|---|---|
| GPT-OSS-120B | 120B | 128K | ★★★★★ | ★★★★★ | Yes | Apache 2.0 | openai/gpt-oss-120b | Flagship open-weight; frontier agentic reasoning |
| GPT-OSS-20B | 20B | 128K | ★★★★★ | ★★★★★ | Yes | Apache 2.0 | openai/gpt-oss-20b | OpenAI open-weight; excellent tool calling & agents |

---

## 🟧 gemma3_text

| Model | Parameters | Context | Agentic | Long Context | Tool Use | License | HuggingFace ID | Notes |
|---|---|---|---|---|---|---|---|---|
| gemma-3-27b-it | 27B | 128K | ★★★★★ | ★★★★★ | Yes | Apache 2.0 | google/gemma-3-27b-it | Best Gemma; tool calling + long context |
| gemma-3-12b-it | 12B | 128K | ★★★★☆ | ★★★★☆ | Yes | Apache 2.0 | google/gemma-3-12b-it | Great 16GB GPU option; solid agents |
| gemma-3-4b-it | 4B | 128K | ★★★☆☆ | ★★★☆☆ | Yes | Apache 2.0 | google/gemma-3-4b-it | Edge-friendly; 128K context |

---

## Rating Guide

| Rating | Meaning |
|---|---|
| ★★★★★ | Excellent — frontier-level for this category |
| ★★★★☆ | Strong — production-ready, minor trade-offs |
| ★★★☆☆ | Good — capable but not specialized |

---

*Sources: Hugging Face, HuggingFace Blog, BentoML, Shakudo, ML Journey — April 2026*
