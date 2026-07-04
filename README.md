# EduTutor — Curriculum Q&A with Fine-tuned LLM

Fine-tuned Mistral-7B-Instruct on 9,205 curriculum Q&A pairs using QLoRA.
Served via TGI with AWQ 4-bit quantization. End-to-end LLM engineering project
covering fine-tuning, quantization, inference, evaluation, and deployment.

## Live Demo

🎯 [Try EduTutor](https://edututor-sm.streamlit.app)

## Results

| Model | ROUGE-L | Exact Match | Avg Latency | Size |
|---|---|---|---|---|
| Base Mistral-7B | 0.3346 | 1.5% | 4125ms | 14.5GB |
| Fine-tuned LoRA | 0.3404 | 1.5% | 3807ms | 14.5GB |
| Fine-tuned AWQ 4-bit | 0.3249 | 0.5% | 705ms | 4.1GB |

AWQ quantization reduced model size by 71% and inference latency by 83%
with minimal quality loss — the primary production value of quantization.

> ROUGE-L scores reflect generative model behavior. Semantically correct
> answers score lower than extractive models due to wording variation.

## Architecture

```
[Streamlit UI] → [FastAPI + Weave] → [TGI Server] → [Mistral-7B AWQ]
```

## Stack

| Component | Technology |
|---|---|
| Base model | Mistral-7B-Instruct-v0.2 |
| Fine-tuning | QLoRA (r=8, NF4, 1 epoch) |
| Quantization | AWQ 4-bit (71% size reduction) |
| Inference | TGI on HuggingFace Spaces (A10G) |
| API layer | FastAPI + Weave observability |
| UI | Streamlit |
| Experiment tracking | Weights & Biases |

## Training Details

- **Dataset:** SciQ — 9,205 science Q&A pairs with explanations
- **Method:** QLoRA — 4-bit base model + LoRA adapters (r=8, alpha=16)
- **Trainable params:** 13.6M out of 7.2B (0.36%)
- **Training:** 1 epoch, 1151 steps, ~8 hours on Kaggle T4
- **Final eval loss:** 1.29 | Token accuracy: 68.6%

## Links

| Resource | Link |
|---|---|
| Live Demo | [edututor-sm.streamlit.app](https://edututor-sm.streamlit.app) |
| FastAPI | [edututor-api Space](https://huggingface.co/spaces/Shuvam-Maity/edututor-api) |
| TGI Server | [edututor-tgi Space](https://huggingface.co/spaces/Shuvam-Maity/edututor-tgi) |
| AWQ Model | [Shuvam-Maity/edututor-mistral-awq](https://huggingface.co/Shuvam-Maity/edututor-mistral-awq) |
| LoRA Adapters | [Shuvam-Maity/edututor-mistral-lora](https://huggingface.co/Shuvam-Maity/edututor-mistral-lora) |
| Dataset | [Shuvam-Maity/edututor-sciq-instruct](https://huggingface.co/datasets/Shuvam-Maity/edututor-sciq-instruct) |
| W&B Training | [wandb.ai run](https://wandb.ai/models-st-xavier-s-college/huggingface) |

## Project Structure

```
EduTutor/
├── streamlit_app.py      ← Streamlit UI
├── requirements.txt      ← UI dependencies
└── README.md
```

Training and serving code lives in the HuggingFace Spaces:
- `edututor-api` Space — FastAPI inference layer
- `edututor-tgi` Space — TGI serving

## Production Notes

TGI is currently running on A10G GPU on HuggingFace Spaces.
For cost optimization the Space is paused when not in use.

For always-on production deployment this would run on dedicated
GPU infrastructure with TGI's continuous batching and
PagedAttention for efficient memory management.

## Author

**Shuvam Maity**
M.Sc. Data Science, St. Xavier's College Kolkata

[LinkedIn](https://linkedin.com/in/shuvam-maity) ·
[HuggingFace](https://huggingface.co/Shuvam-Maity) ·
[GitHub](https://github.com/Shuvam-Maity)
