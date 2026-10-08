import gradio as gr
import spaces
import torch
import weave
import time
import os
from transformers import AutoModelForCausalLM, AutoTokenizer

weave.init("models-st-xavier-s-college/edututor-production")

MODEL_ID  = "Shuvam-Maity/edututor-mistral-awq"
HF_TOKEN  = os.environ.get("HF_TOKEN")

# Load tokenizer at startup (no GPU needed)
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, token=HF_TOKEN)
print("Tokenizer loaded.")

# Model loaded lazily inside GPU context
model = None

@spaces.GPU
@weave.op()
def generate(question: str) -> str:
    global model

    if model is None:
        print("Loading model on GPU...")
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            token=HF_TOKEN,
            dtype=torch.float16,
            device_map="cuda"
        )
        model.eval()
        print("Model loaded.")

    if not question.strip():
        return "Please enter a question."

    prompt  = f"<s>[INST] {question} [/INST]"
    inputs  = tokenizer(prompt, return_tensors="pt").to("cuda")

    start = time.time()
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=200,
            temperature=0.7,
            do_sample=True
        )
    latency = round((time.time() - start) * 1000)

    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    answer = answer.split("[/INST]")[-1].strip()

    return f"{answer}\n\n⏱ Latency: {latency}ms"

with gr.Blocks() as demo:
    gr.Markdown("# 📚 EduTutor — Curriculum Q&A")
    gr.Markdown("Fine-tuned Mistral-7B (AWQ 4-bit) on SciQ dataset")
    question = gr.Textbox(label="Question", placeholder="What is osmosis?")
    output   = gr.Textbox(label="Response")
    btn      = gr.Button("Get Answer", variant="primary")
    btn.click(fn=generate, inputs=question, outputs=output)

demo.launch()