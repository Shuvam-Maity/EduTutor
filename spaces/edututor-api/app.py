from fastapi import FastAPI
from pydantic import BaseModel
import weave
import time
import requests

app = FastAPI(title="EduTutor API")

weave.init("models-st-xavier-s-college/edututor-production")

TGI_URL = "https://shuvam-maity-edututor-tgi.hf.space/generate"

class QuestionRequest(BaseModel):
    question: str
    max_tokens: int = 200

@app.get("/")
def root():
    return {"message": "EduTutor API is running"}

@app.get("/health")
def health():
    return {"status": "ok"}

@weave.op()
@app.post("/generate")
async def generate(request: QuestionRequest):
    start = time.time()

    response = requests.post(
        TGI_URL,
        headers={"Content-Type": "application/json"},
        json={
            "inputs": f"<s>[INST] {request.question} [/INST]",
            "parameters": {
                "max_new_tokens": request.max_tokens,
                "temperature"   : 0.7
            }
        },
        timeout=60
    )

    result  = response.json()
    answer  = result.get("generated_text", str(result))
    latency = round((time.time() - start) * 1000)

    return {
        "answer"    : answer,
        "latency_ms": latency,
        "model"     : "Shuvam-Maity/edututor-mistral-awq"
    }