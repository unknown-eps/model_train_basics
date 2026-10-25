"""A small HTTP mock of token-by-token generation, not a real language model."""

import asyncio
import random

from fastapi import FastAPI
from pydantic import BaseModel, Field

MAX_TOKEN_ID = 32_000  # Inclusive upper bound.
RANDOM_SEED = 42
TOKEN_DELAY_SECONDS = 0.01

app = FastAPI(title="Mock vLLM")


class GenerationRequest(BaseModel):
    id: int = Field(ge=0, le=2**31 - 1)
    prompt: str


class GenerationResponse(BaseModel):
    id: int
    tokens: list[int]


class MockVLLM:
    async def generate(self, request: GenerationRequest) -> GenerationResponse:
        # Stable per-ID seeds make results reproducible regardless of scheduling.
        rng = random.Random(RANDOM_SEED + request.id)
        num_tokens = rng.randint(2048, 4096)
        vocabulary_size = MAX_TOKEN_ID + 1
        previous_token = sum(request.prompt.encode("utf-8")) % vocabulary_size
        tokens = []

        for _ in range(num_tokens):
            # Simulate each decoding step without blocking other HTTP requests.
            await asyncio.sleep(TOKEN_DELAY_SECONDS)
            offset = rng.randrange(vocabulary_size)
            token = (previous_token + offset) % vocabulary_size
            tokens.append(token)
            previous_token = token

        return GenerationResponse(id=request.id, tokens=tokens)


model = MockVLLM()


@app.post("/generate", response_model=GenerationResponse)
async def generate(request: GenerationRequest) -> GenerationResponse:
    return await model.generate(request)
