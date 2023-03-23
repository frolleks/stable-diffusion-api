import torch
from torch import autocast
from diffusers import StableDiffusionPipeline
from fastapi import FastAPI
from fastapi.responses import Response
from pydantic import BaseModel


class TextGen(BaseModel):
    # You may use other models, the default is "CompVis/stable-diffusion-v1-4". Other models include "waifu" for now.
    model: str | None = "default"
    prompt: str


app = FastAPI()


@app.get("/")
def root():
    return {"message": "API is working"}


@app.post(
    "/generate",
    responses={
        200: {
            "content": {"image/png": {}}
        }
    },
    response_class=Response,
)
def generate(body: TextGen):
    if body.model == "default":
        model_name = "CompVis/stable-diffusion-v1-4"
    elif body.model == "waifu":
        model_name = "hakurei/waifu-diffusion"

    pipe = StableDiffusionPipeline.from_pretrained(
        model_name,
        torch_dtype=torch.float16
    ).to('cuda')
    prompt = body.prompt
    with autocast("cuda"):
        image = pipe(prompt, guidance_scale=6)["sample"][0]

    return Response(content=image, media_type="image/png")
