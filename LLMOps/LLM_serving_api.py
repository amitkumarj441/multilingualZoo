import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig

from typing import Any, List

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI()

class ItemList(BaseModel):
    choices: List[Any]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_key = "llama/Llama-2-13b-hf" # here place you model name

tokenizer = AutoTokenizer.from_pretrained(model_key)
model = AutoModelForCausalLM.from_pretrained(
    model_key,
    torch_dtype=torch.float16,
    device_map="auto",
)

print("Loading " + model_key + "...")
print("gpu_count", torch.cuda.device_count())

async def inference(instruction: str, gen_kwargs: dict):
    generated_text = ""
    gen_in = tokenizer(instruction, return_tensors="pt").input_ids.cuda()

    generation_config = GenerationConfig(**gen_kwargs)
    with torch.no_grad():
        gen_ids = model.generate(
            gen_in,
            generation_config=generation_config,
        )
        generated_text = tokenizer.batch_decode(gen_ids, skip_special_tokens=True)[
            0
        ]

        text_without_prompt = generated_text[len(instruction) :]

    response = text_without_prompt

    return response


@app.post("/generate")
async def generate(
    instruction: str = None,
    max_new_tokens: int = 768,
    use_cache: bool = True,
    num_return_sequences: int = 1,
    do_sample: bool = True,
    repetition_penalty: float = 1.1,
    temperature: float = 0.5,
    top_k: int = 50,
    top_p: float = 1.0,
    early_stopping: bool = True,
):
    if not instruction:
        raise HTTPException(
            status_code=400, detail="Please Provide a valid text message"
        )

    gen_kwargs = {
        "top_p": top_p,
        "max_new_tokens": max_new_tokens,
        "use_cache": use_cache,
        "do_sample": do_sample,
        "num_return_sequences": num_return_sequences,
        "repetition_penalty": repetition_penalty,
        "temperature": temperature,
        "top_k": top_k,
        "early_stopping": early_stopping,
        "pad_token_id": tokenizer.eos_token_id,
    }

    generation = await inference(
        instruction=instruction,
        gen_kwargs=gen_kwargs,
    )

    return {
        "generation": generation,
    }
