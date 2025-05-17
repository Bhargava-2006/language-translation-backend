from fastapi import FastAPI, Request
from pydantic import BaseModel
from transformers import pipeline

app = FastAPI()

# Use Hugging Face inference API directly (no large model loaded into memory)
translator = pipeline("translation", model="facebook/mbart-large-50-many-to-many-mmt")

class TextRequest(BaseModel):
    text: str

@app.post("/translate")
async def translate_text(req: TextRequest):
    result = translator(req.text, src_lang="en_XX", tgt_lang="te_IN")
    return {"translation": result[0]['translation_text']}
