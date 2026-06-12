from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM
from fastapi.middleware.cors import CORSMiddleware
import torch

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model_path = "./tinyllama-coding-model"

print("Loading tokenizer...")

try:
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        use_fast=False,
        trust_remote_code=True
    )

    print("Loading model...")

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=torch.float32,
        device_map="auto"
    )

    print("Model loaded successfully!")

except Exception as e:
    print(f"Error loading model: {e}")
    raise


class Question(BaseModel):
    prompt: str


@app.post("/ask")
def ask_ai(data: Question):

    text = f"""
You are a professional coding assistant.
Give clear and correct coding answers.

Instruction: {data.prompt}

Answer:
"""

    inputs = tokenizer(text, return_tensors="pt")

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=350,
            temperature=0.2,
            top_p=0.9,
            do_sample=True,
            repetition_penalty=1.15,
            no_repeat_ngram_size=3,
            pad_token_id=tokenizer.eos_token_id
        )

    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    answer = result.replace(text, "").strip()

    return {"response": answer}