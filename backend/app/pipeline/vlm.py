"""Shared OpenAI structured-output plumbing.

Started as private helpers inside stage 3 (label), then the reconstruction
verification stage needed the same call — construct a client with an explicit
key, encode an image, ask for structured JSON — and duplicating it would have
been a second implementation of "ask the VLM a question" to keep in step for no
reason. Factored out once two stages needed it, not before.
"""

import base64

from pydantic import BaseModel

from app.pipeline.base import PipelineContext

__all__ = ["client", "image_part", "parse"]


def client(ctx: PipelineContext, stage: str):
    """An OpenAI client holding the key explicitly; see `app.config.Settings.require`."""
    from openai import OpenAI

    return OpenAI(api_key=ctx.settings.require("openai_api_key", stage))


def image_part(data: bytes, media_type: str = "image/png") -> dict:
    encoded = base64.b64encode(data).decode()
    return {"type": "input_image", "image_url": f"data:{media_type};base64,{encoded}"}


def parse(
    ctx: PipelineContext, stage: str, images: list[bytes], prompt: str, schema: type[BaseModel]
):
    """One structured-output call over one or more images plus a text prompt.

    Images are listed in the content array in the order given; the prompt is
    responsible for saying which is which when there is more than one — "the
    first image is the original photo, the second is ...".
    """
    response = client(ctx, stage).responses.parse(
        model=ctx.settings.openai_model,
        input=[
            {
                "role": "user",
                "content": [image_part(img) for img in images]
                + [{"type": "input_text", "text": prompt}],
            }
        ],
        text_format=schema,
    )
    parsed = response.output_parsed
    if parsed is None:
        raise RuntimeError(f"{stage}: model returned no parsable output")
    return parsed
