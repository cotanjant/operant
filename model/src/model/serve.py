import argparse
import json
import time
import uuid
from typing import List, Optional, Union, Dict, Any
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
import uvicorn

from .gpu_engine import GPUEngine, DEFAULT_SYSTEM_PROMPT

app = FastAPI(title="Operant GPU Inference Engine", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

engine: Optional[GPUEngine] = None

class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    model: Optional[str] = "operant"
    messages: List[Message]
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 0.9
    max_tokens: Optional[int] = 512
    stream: Optional[bool] = False
    stop: Optional[Union[str, List[str]]] = None

@app.get("/health")
@app.get("/")
def health():
    return {
        "status": "ok",
        "service": "operant-gpu-inference",
        "device": str(engine.device if engine else "uninitialized"),
    }

@app.get("/v1/models")
def list_models():
    model_name = engine.model_name if engine else "operant"
    return {
        "object": "list",
        "data": [
            {
                "id": model_name,
                "object": "model",
                "created": int(time.time()),
                "owned_by": "operant",
            }
        ],
    }

@app.post("/v1/chat/completions")
def chat_completions(req: ChatRequest):
    if engine is None:
        raise HTTPException(status_code=500, detail="GPU Engine not initialized")

    messages_dicts = [{"role": m.role, "content": m.content} for m in req.messages]
    if not any(m.role == "system" for m in req.messages):
        messages_dicts.insert(0, {"role": "system", "content": DEFAULT_SYSTEM_PROMPT})
    prompt = engine.format_chat_prompt(messages_dicts)

    stop_list = []
    if isinstance(req.stop, str):
        stop_list = [req.stop]
    elif isinstance(req.stop, list):
        stop_list = req.stop

    completion_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
    created_time = int(time.time())
    model_name = req.model or engine.model_name

    if req.stream:
        def stream_generator():
            # Initial role chunk
            first_chunk = {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": created_time,
                "model": model_name,
                "choices": [
                    {
                        "index": 0,
                        "delta": {"role": "assistant"},
                        "finish_reason": None,
                    }
                ],
            }
            yield f"data: {json.dumps(first_chunk)}\n\n"

            for token in engine.generate_stream(
                prompt=prompt,
                temperature=req.temperature or 0.7,
                top_p=req.top_p or 0.9,
                max_tokens=req.max_tokens or 512,
                stop=stop_list,
            ):
                chunk = {
                    "id": completion_id,
                    "object": "chat.completion.chunk",
                    "created": created_time,
                    "model": model_name,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"content": token},
                            "finish_reason": None,
                        }
                    ],
                }
                yield f"data: {json.dumps(chunk)}\n\n"

            final_chunk = {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": created_time,
                "model": model_name,
                "choices": [
                    {
                        "index": 0,
                        "delta": {},
                        "finish_reason": "stop",
                    }
                ],
            }
            yield f"data: {json.dumps(final_chunk)}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(stream_generator(), media_type="text/event-stream")

    # Non-streaming
    res = engine.generate(
        prompt=prompt,
        temperature=req.temperature or 0.7,
        top_p=req.top_p or 0.9,
        max_tokens=req.max_tokens or 512,
        stop=stop_list,
    )

    return {
        "id": completion_id,
        "object": "chat.completion",
        "created": created_time,
        "model": model_name,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": res["text"],
                },
                "finish_reason": res["finish_reason"],
            }
        ],
        "usage": {
            "prompt_tokens": res["prompt_tokens"],
            "completion_tokens": res["completion_tokens"],
            "total_tokens": res["prompt_tokens"] + res["completion_tokens"],
        },
    }

def main():
    global engine
    parser = argparse.ArgumentParser(description="Operant GPU Inference Server")
    parser.add_argument("--model-file", "-m", default="out", help="Path to model directory")
    parser.add_argument("--host", default="0.0.0.0", help="Host address to bind")
    parser.add_argument("--port", "-p", type=int, default=8000, help="Port to bind")
    parser.add_argument("--device", "-d", default=None, help="Device to use (e.g. cuda:0, mps, cpu)")
    args = parser.parse_args()

    engine = GPUEngine(model_path=args.model_file, device=args.device)
    print(f"Starting Operant GPU OpenAI API on http://{args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port)

if __name__ == "__main__":
    main()
