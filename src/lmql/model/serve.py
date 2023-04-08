"""
Serves a transformers model as LMQL inference API.
"""

from dataclasses import dataclass, field
from collections import defaultdict

import json
from http.server import BaseHTTPRequestHandler, HTTPServer
from multiprocessing import Queue as MPQueue
from queue import Empty
from queue import Queue
import multiprocessing
from typing import Dict
import requests
import asyncio
import sys
import atexit
import argparse
import time
import os
import subprocess

from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch


@dataclass
class InferenceServerState:
    model_identifier: str
    tokenizer_descriptor: str
    dtype: str

    queue: Queue
    tokenize_queue: Queue
    all_results_queue: Queue

    sample_count: int = 0
    client_results_queues: Dict[str, Queue] = field(default_factory=dict)

    exit: bool = False


class TokenizerProcessor:
    def __init__(self, state: InferenceServerState):
        self.model_identifier = state.tokenizer_descriptor
        self.queue = state.tokenize_queue
        self.state = state

    def shutdown(self):
        self.state.exit = True

    def tokenize(self, tokenizer, sample_id, client_id, item):
        text = item["text"]

        if text == "<EOS>":
            input_ids = [tokenizer.eos_token_id]
        elif text == "<BOS>":
            input_ids = [tokenizer.bos_token_id]
        else:
            input_ids = tokenizer(text)["input_ids"]

        self.state.all_results_queue.put(
            {"sample_id": sample_id, "client_id": client_id, "input_ids": input_ids}
        )

    def detokenize(self, tokenizer, sample_id, client_id, item):
        input_ids = item["input_ids"]

        text = tokenizer.decode(input_ids)
        self.state.all_results_queue.put(
            {"sample_id": sample_id, "client_id": client_id, "text": text}
        )

    def run(self, index):
        # load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(self.model_identifier)
        print("Tokenizer #{} {} ready!".format(index, self.model_identifier))

        while not self.state.exit:
            item = self.queue.get()
            if item is None:
                time.sleep(0.1)
                continue

            sample_id = item["sample_id"]
            client_id = item["client_id"]
            action = item["action"]

            if action == "tokenize":
                self.tokenize(tokenizer, sample_id, client_id, item)
            elif action == "detokenize":
                self.detokenize(tokenizer, sample_id, client_id, item)
            else:
                print("error: unknown TokenizerProcessor action {}".format(action))

        print("Tokenizer #{} shut down.".format(index))

    def run_in_parallel(self, n=2):
        atexit.register(self.shutdown)

        workers = []

        for i in range(n):
            p = multiprocessing.Process(target=self.run, args=(i,))
            p.start()
            workers.append(p)

        return workers


class ModelProcessor:
    def __init__(
        self, state: InferenceServerState, cuda: bool = False, cache: str = None
    ):
        self.model_identifier = state.model_identifier
        self.queue = state.queue
        self.state = state
        self.cuda = cuda

        self.cache = None
        if cache is not None:
            from rocksdict import Rdict

            self.cache = Rdict(cache)

        self.request_count = 0
        self.requests_cached = 0
        self.last_report = time.time()
        self.last_request_count = 0

        try:
            self.nvidia_logging = (
                subprocess.Popen(["nvidia-smi"], stdout=subprocess.PIPE).wait() == 0
            )
        except:
            self.nvidia_logging = False

    def shutdown(self):
        self.state.exit = True

    def __del__(self):
        if self.cache is not None:
            self.cache.close()

    def print_stats(self):
        if self.nvidia_logging:
            visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", None)
            cmds = ["nvidia-smi"]
            if visible_devices is not None:
                cmds.append("-i={}".format(visible_devices))
            cmds += [
                "--query-gpu=name,memory.used,memory.total,utilization.gpu",
                "--format=csv,noheader",
            ]
            output = [
                l.split(", ")
                for l in subprocess.check_output(cmds).decode("utf-8").split("\n")
                if l.strip() != ""
            ]
            gpu_usage = [
                "GPU {} {}, util {}".format(i, row[1] + "/" + row[2], row[3])
                for i, row in enumerate(output)
            ]
        else:
            gpu_usage = ["GPU monitoring not available on non-CUDA systems"]

        print(" " * 100, end="\r")
        # fancy unicode based terminal spinner
        terminal_spinner_chars = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
        throughput = (self.request_count - self.last_request_count) / (
            time.time() - self.last_report
        )
        self.last_report = time.time()
        self.last_request_count = self.request_count

        # format throughput to two decimal places
        print(
            "{} {:.2f} calls/s, Requests Served: {}, Queue: {} [{}]".format(
                terminal_spinner_chars[
                    self.request_count % len(terminal_spinner_chars)
                ],
                throughput,
                self.request_count,
                self.state.queue.qsize(),
                ", ".join(gpu_usage),
            ),
            end="\r",
        )

    def run(self):
        dtype = self.state.dtype
        if dtype == "float16":
            dtype = torch.float16
        else:
            dtype = None

        # load model
        if not self.cuda:
            print("Loading {} (CPU)".format(self.model_identifier))
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_identifier, torch_dtype=dtype, resume_download=True
            )
        else:
            print("Loading {} (Multi-GPU)".format(self.model_identifier))
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_identifier,
                torch_dtype=dtype,
                resume_download=True,
                device_map="auto",
            )
        self.model.eval()

        print("Ready!".format(self.model_identifier))

        while not self.state.exit:
            self.print_stats()
            # wait for self.queue to have an item
            try:
                item = self.queue.get(timeout=1.0)
            except Empty:
                continue
            except KeyboardInterrupt:
                break

            if item is None:
                time.sleep(0.1)
                continue

            self.request_count += 1

            device = "cuda" if self.cuda else "cpu"

            sample_id = item["sample_id"]
            client_id = item["client_id"]
            input_ids = torch.tensor(item["input_ids"], dtype=torch.long).to(device)
            attention_mask = item.get("attention_mask", None)

            if attention_mask is None:
                attention_mask = torch.ones_like(input_ids).to(device)
            else:
                attention_mask = torch.tensor(attention_mask, dtype=torch.long).to(
                    device
                )

            if self.cache is not None:
                key = (
                    "IDs:"
                    + str(input_ids.tolist())
                    + " MASK:"
                    + str(attention_mask.tolist())
                )
                if key in self.cache:
                    self.requests_cached += 1
                    self.state.all_results_queue.put(
                        {
                            "client_id": client_id,
                            "sample_id": sample_id,
                            "next_token_logits": self.cache[key],
                        }
                    )
                    continue

            res = self.model.forward(input_ids=input_ids, attention_mask=attention_mask)

            if input_ids.ndimension() == 2:
                next_token_logits = res.logits[:, -1]
            else:
                next_token_logits = res.logits[-1]

            if self.cache is not None:
                key = (
                    "IDs:"
                    + str(input_ids.tolist())
                    + " MASK:"
                    + str(attention_mask.tolist())
                )
                self.cache[key] = next_token_logits.tolist()

            self.state.all_results_queue.put(
                {
                    "client_id": client_id,
                    "sample_id": sample_id,
                    "next_token_logits": next_token_logits.detach().tolist(),
                }
            )

        print("Processor shut down")

    def oom_reloading_run(self):
        while True:
            try:
                self.run()
                return
            except RuntimeError as e:
                if "CUDA out of memory" in str(e):
                    print("Crashed due to OOM, reloading model.")
                    continue
                else:
                    import traceback

                    traceback.print_exc()
                    print("Crashed with", e, "reloading model...")
                    continue

    def run_in_parallel(self):
        atexit.register(self.shutdown)

        p = multiprocessing.Process(target=self.oom_reloading_run)
        p.start()
        return p


from typing import List, Optional
from fastapi import FastAPI
from pydantic import BaseModel, BaseSettings, Field
from serving.types import *


def get_app(app_config: AppConfig):

    app = FastAPI()

    @app.post("/queue")
    async def queue(payload: QueuePayload):
        if payload.forward is not None:
            # process forward action
            input_ids = payload.forward.input_ids
            attention_mask = payload.forward.attention_mask
            model_identifier = payload.forward.model_identifier
            client_id = payload.forward.client_id
            sample_id = payload.forward.sample_id

            # TODO: implement model processing and queueing of input_ids and attention_mask

            return QueueResults(sample_id=sample_id)
        elif payload.tokenize is not None:
            # process tokenize action
            text = payload.tokenize.text
            client_id = payload.tokenize.client_id
            sample_id = payload.tokenize.sample_id

            # TODO: implement tokenization and queueing of text

            return QueueResults(sample_id=sample_id)
        elif payload.detokenize is not None:
            # process detokenize action
            input_ids = payload.detokenize.input_ids
            client_id = payload.detokenize.client_id
            sample_id = payload.detokenize.sample_id

            # TODO: implement detokenization and queueing of input_ids

            return QueueResults(sample_id=sample_id)

    @app.get("/results/{client_id}", response_model=List[ResultsResponse])
    async def results(client_id: str):
        # TODO: implement retrieval of results for specified client_id
        return []

    @app.on_event("startup")
    def start():
        manager = multiprocessing.Manager()
        app.state = InferenceServerState(
            app_config.model_descriptor,
            app_config.tokenizer_descriptor,
            app_config.dtype,
            queue=manager.Queue(),
            tokenize_queue=manager.Queue(),
            all_results_queue=manager.Queue(),
        )

        # prepare configuration
        if app_config.tokenizer_descriptor is None:
            tokenizer_descriptor = app_config.model_descriptor
        # run model in separate process
        app.state.processor = ModelProcessor(
            app.state, cuda=app_config.cuda, cache=app_config.cache
        )
        app.state.processor.run_in_parallel()

        # run tokenizers in separate process
        app.state.tokenizer_processor = TokenizerProcessor(app.state)
        app.state.tokenizer_processor.run_in_parallel(
            n=app_config.num_tokenizer_processes
        )

    @app.on_event("shutdown")
    def shutdown():
        app.state.processor.shutdown()
        app.state.tokenizer_processor.shutdown()

    return app


import uvicorn


def main(model_name="gpt2-medium"):
    config = AppConfig(model_descriptor=model_name)
    app = get_app(config)
    print(config)
    return app
    # uvicorn.run(app, host=config.host, port=config.port)


import fire


if __name__ == "__main__":
    fire.Fire(main)
