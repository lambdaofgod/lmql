import uuid
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

app = FastAPI()


class QueueRequest(BaseModel):
    action: str
    input_ids: List[List[int]]
    attention_mask: Optional[List[List[int]]] = None
    model_identifier: str
    client_id: str
    sample_id: int


class QueueResponse(BaseModel):
    sample_id: int


class ResultResponse(BaseModel):
    client_id: str
    sample_id: int
    next_token_logits: List[List[float]]


model_cache = {}
tokenizer_cache = {}
results = {}


def get_model_and_tokenizer(model_identifier):
    if model_identifier not in model_cache:
        model_cache[model_identifier] = AutoModelForCausalLM.from_pretrained(
            model_identifier
        )
        tokenizer_cache[model_identifier] = AutoTokenizer.from_pretrained(
            model_identifier
        )

    return model_cache[model_identifier], tokenizer_cache[model_identifier]


@app.post("/queue", response_model=QueueResponse)
async def queue(request: QueueRequest):
    model, tokenizer = get_model_and_tokenizer(request.model_identifier)
    input_ids = request.input_ids
    attention_mask = request.attention_mask

    if attention_mask is None:
        attention_mask = [[1] * len(input_ids[0])]

    input_ids_tensor = torch.tensor(input_ids, dtype=torch.long)
    attention_mask_tensor = torch.tensor(attention_mask, dtype=torch.long)

    with torch.no_grad():
        output = model(input_ids=input_ids_tensor, attention_mask=attention_mask_tensor)

    next_token_logits = output.logits[:, -1].tolist()
    results[request.client_id] = {
        "sample_id": request.sample_id,
        "next_token_logits": next_token_logits,
    }

    return QueueResponse(sample_id=request.sample_id)


@app.get("/results/{client_id}", response_model=List[ResultResponse])
async def get_results(client_id: str):
    if client_id in results:
        response = [
            ResultResponse(
                client_id=client_id,
                sample_id=results[client_id]["sample_id"],
                next_token_logits=results[client_id]["next_token_logits"],
            )
        ]
        del results[client_id]
        return response
    else:
        return []
