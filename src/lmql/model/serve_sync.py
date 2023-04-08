import uuid
from fastapi import FastAPI
from pydantic import BaseModel, Field
from typing import List, Optional
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from tokenizers import Tokenizer
from pathlib import Path
import rwkv.model


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


strategy = "cuda fp16"


class ModelCache(BaseModel):
    model_cache: dict = Field(default_factory=dict)
    tokenizer_cache: dict = Field(default_factory=dict)
    results: dict = Field(default_factory=dict)

    def load_model_from_identifier(self, model_identifier):
        if "rwkv" in model_identifier:
            model_path = Path(model_identifier)
            model = rwkv.model.RWKV(model=str(model_path), strategy=strategy)

            maybe_tokenizer_paths = list(model_path.parent.glob("*.json"))
            assert len(maybe_tokenizer_paths) == 1
            tokenizer_path = maybe_tokenizer_paths[0]
            tokenizer = Tokenizer.from_file(tokenizer_path)
            return model, tokenizer
        else:
            model = AutoModelForCausalLM.from_pretrained(model_identifier)
            tokenizer = AutoTokenizer.from_pretrained(model_identifier)
        return model, tokenizer

    def get_model_and_tokenizer(self, model_identifier):
        if model_identifier not in self.model_cache:
            model, tokenizer = self.load_model_from_identifier(model_identifier)
            self.model_cache[model_identifier] = model
            self.tokenizer_cache[model_identifier] = self.tokenizer_cache

        return (
            self.model_cache[model_identifier],
            self.tokenizer_cache[model_identifier],
        )


def get_app():
    app = FastAPI()

    app.state.model_cache = ModelCache()

    @app.post("/queue", response_model=QueueResponse)
    async def queue(request: QueueRequest):
        model, tokenizer = app.state.model_cache.get_model_and_tokenizer(
            request.model_identifier
        )
        input_ids = request.input_ids
        attention_mask = request.attention_mask

        if attention_mask is None:
            attention_mask = [[1] * len(input_ids[0])]

        input_ids_tensor = torch.tensor(input_ids, dtype=torch.long)
        attention_mask_tensor = torch.tensor(attention_mask, dtype=torch.long)

        with torch.no_grad():
            output = model(
                input_ids=input_ids_tensor, attention_mask=attention_mask_tensor
            )

        next_token_logits = output.logits[:, -1].tolist()
        app.state.model_cache.results[request.client_id] = {
            "sample_id": request.sample_id,
            "next_token_logits": next_token_logits,
        }

        return QueueResponse(sample_id=request.sample_id)

    @app.get("/results/{client_id}", response_model=List[ResultResponse])
    async def get_results(client_id: str):
        if client_id in app.state.model_cache.results:
            response = [
                ResultResponse(
                    client_id=client_id,
                    sample_id=app.state.model_cache.results[client_id]["sample_id"],
                    next_token_logits=app.state.model_cache.results[client_id][
                        "next_token_logits"
                    ],
                )
            ]
            del app.state.model_cache.results[client_id]
            return response
        else:
            return []

    return app
