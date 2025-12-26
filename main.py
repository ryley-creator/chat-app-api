from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from huggingface_hub import InferenceClient
from dotenv import load_dotenv
import io
import os

load_dotenv()

app = FastAPI()

api_key = os.environ.get("api_key")

client = InferenceClient(
    provider="fal-ai",
    api_key=api_key,
)

class ImageRequest(BaseModel):
    prompt: str

@app.post("/generate-image")
def generate_image(req: ImageRequest):
    image = client.text_to_image(
        req.prompt,
        model="Tongyi-MAI/Z-Image-Turbo",
        width=1024,
        height=1024,
    )

    img_bytes = io.BytesIO()
    image.save(img_bytes, format="PNG")
    img_bytes.seek(0)

    return StreamingResponse(img_bytes, media_type="image/png")