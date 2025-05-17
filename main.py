from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from fastapi.middleware.cors import CORSMiddleware
import nltk

nltk.download('punkt')
from nltk.tokenize import sent_tokenize

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class TranslationRequest(BaseModel):
    text: str

@app.on_event("startup")
def load_models():
    global model, tokenizer
    model_name = "aryaumesh/english-to-telugu"
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

@app.post("/translate/")
async def get_translation(request: TranslationRequest):
    text = request.text.strip()
    lines = text.split('\n')
    translated_lines = []

    for line in lines:
        line = line.strip()
        if not line:
            translated_lines.append("")
            continue

        sentences = sent_tokenize(line)
        translated_sentences = []

        for sentence in sentences:
            if not sentence.endswith(('.', '?', '!', ':', ';')):
                sentence += '.'

            inputs = tokenizer(sentence, return_tensors="pt")
            translated = model.generate(**inputs)
            translated_text = tokenizer.decode(translated[0], skip_special_tokens=True)
            translated_sentences.append(translated_text)

        translated_lines.append(" ".join(translated_sentences))

    return {
        "original_text": request.text,
        "translated_text": "\n".join(translated_lines)
    }


