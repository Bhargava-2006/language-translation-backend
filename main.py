from fastapi import FastAPI
from pydantic import BaseModel
from transformers import MBartForConditionalGeneration, MBart50TokenizerFast
from fastapi.middleware.cors import CORSMiddleware
import nltk.data

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add the nltk data path to find punkt tokenizer data
nltk.data.path.append("/root/nltk_data")  # Adjust path if needed

# Define request structure
class TranslationRequest(BaseModel):
    text: str
    use_pretrained: bool

@app.on_event("startup")
def load_models():
    global mbart_model, mbart_tokenizer

    pretrained_model_name = "aryaumesh/english-to-telugu"
    mbart_model = MBartForConditionalGeneration.from_pretrained(pretrained_model_name)
    mbart_tokenizer = MBart50TokenizerFast.from_pretrained(pretrained_model_name)

from nltk.tokenize import sent_tokenize

@app.post("/translate/")
async def get_translation(request: TranslationRequest):
    text = request.text.strip()
    
    # Step 1: Split input into lines to preserve line breaks
    lines = text.split('\n')

    translated_lines = []

    for line in lines:
        line = line.strip()
        if not line:
            translated_lines.append("")  # Preserve empty lines
            continue

        # Step 2: Split line into sentences
        sentences = sent_tokenize(line)
        translated_sentences = []

        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence.endswith(('.', '?', '!', ':', ';')):
                sentence += '.'

            inputs = mbart_tokenizer(sentence, return_tensors="pt")
            translated = mbart_model.generate(**inputs)
            translated_text = mbart_tokenizer.decode(translated[0], skip_special_tokens=True)

            translated_sentences.append(translated_text)

        # Step 3: Reconstruct translated line
        translated_line = " ".join(translated_sentences)
        translated_lines.append(translated_line)

    # Step 4: Rejoin translated lines with newline characters
    full_translation = "\n".join(translated_lines)

    return {
        "original_text": request.text,
        "translated_text": full_translation
    }
