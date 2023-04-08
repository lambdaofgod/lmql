from typing import List, Optional
from fastapi import FastAPI
from pydantic import BaseModel, BaseSettings, Field


class QueuePayloadForward(BaseModel):
    action: str = "forward"
    input_ids: List[List[int]]
    attention_mask: Optional[List[List[int]]]
    model_identifier: str
    client_id: str
    sample_id: int


class QueuePayloadTokenize(BaseModel):
    action: str = "tokenize"
    text: str
    client_id: str
    sample_id: int


class QueuePayloadDetokenize(BaseModel):
    action: str = "detokenize"
    input_ids: List[int]
    client_id: str
    sample_id: int


class QueuePayload(BaseModel):
    forward: Optional[QueuePayloadForward]
    tokenize: Optional[QueuePayloadTokenize]
    detokenize: Optional[QueuePayloadDetokenize]


class QueueResults(BaseModel):
    sample_id: int


class ResultsResponse(BaseModel):
    client_id: str
    sample_id: int
    next_token_logits: List[List[float]]


class AppConfig(BaseModel):
    model_descriptor: str
    port: int = Field(default=8080)
    host: str = Field(default="localhost")
    cuda: bool = Field(default=False)
    dtype: bool = Field(default="none")
    num_tokenizer_processes: int = Field(default=2)
    tokenizer_descriptor: str = Field(default=None)
    cache: Optional[str] = Field(default=None)
